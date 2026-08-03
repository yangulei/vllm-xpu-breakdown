# SPDX-License-Identifier: Apache-2.0
"""Rebuilding a profile from uploaded trace files.

The mirror of the live profiler: the same reconstruction, the same two-pass
merge, but the traces come from a browser rather than from a run on this host.
That is what lets a profile captured on a machine with the GPU be analysed on
one without.
"""
from __future__ import annotations

import logging
import os
import traceback
from typing import Any

from breakdown import runs
from breakdown.model_info import fetch_model_config, summarize_config
from breakdown.profiling import runstate
from breakdown.profiling.runstate import save_state
from breakdown.profiling.traces import (
    _build_result_from_traces, _merge_two_pass_result, _parse_trace_filename)

logger = logging.getLogger("vllm_xpu_breakdown")

def save_uploads(files: list[tuple[str, Any]]) -> list[tuple[str, dict]]:
    """Persist uploaded traces under ``output/traces``.

    Each file keeps its original (descriptive) name, because that name is how
    the configuration is recovered - the download endpoint writes
    ``..._{mode}[_prefill|_decode]_ctx{C}_in{S}_out{gen}_bs{B}_tp{TP}...``
    precisely so an upload does not have to be re-described by hand.

    ``files`` is ``[(filename, save_fn), ...]``; ``save_fn(dest)`` writes it.
    """
    from werkzeug.utils import secure_filename
    trace_dir = os.path.abspath("output/traces")
    os.makedirs(trace_dir, exist_ok=True)
    saved: list[tuple[str, dict]] = []
    for filename, save in files:
        orig = os.path.basename(filename or "")
        name = secure_filename(filename) or "uploaded_trace.json"
        dest = os.path.join(trace_dir, name)
        save(dest)
        saved.append((dest, _parse_trace_filename(orig)))
    return saved


def build_from_uploads(saved: list[tuple[str, dict]], form) -> tuple[bool, str]:
    """Reconstruct a profile from uploaded traces. ``(ok, error)``.

    Mirrors the live profiler so a **download -> upload round-trip** rebuilds
    the same graph on a machine with no accelerator: a ``_prefill`` + ``_decode``
    pair rebuilds *both* phases (each with its own batch/query size, spliced by
    :func:`_merge_two_pass_result`), and an untagged upload rebuilds a single
    run from its rank-0 file.
    """
    model_id = (form.get("model_id") or "").strip()

    # Recover the profiled configuration from the descriptive download
    # filenames, falling back to explicit form fields (form always wins).
    metas = [m for _, m in saved if m]

    def _from_names(key: str, default=None):
        for m in metas:
            v = m.get(key)
            if v is not None and v != "":
                return v
        return default

    mode = form.get("mode") or _from_names("mode") or "eager"
    tp_size = int(form.get("tensor_parallel_size") or form.get("tp_size")
                  or _from_names("tp") or 1)
    quant_form = form.get("quantization")
    quantization = (quant_form if quant_form not in (None, "", "auto", "none")
                    else _from_names("quantization"))
    if quantization in ("", "auto", "none"):
        quantization = None
    query_len = form.get("query_len") or _from_names("query_len")
    query_len = int(query_len) if query_len else None
    context_len = form.get("context_len") or _from_names("context_len")
    context_len = int(context_len) if context_len else None

    # Split the uploads by pass tag. A prefill file + a decode file form a
    # two-pass pair (each may itself carry extra rank files); anything without a
    # tag is a plain single run (rank-0 first).
    pre = [(p, m) for p, m in saved if m.get("pass") == "prefill"]
    dec = [(p, m) for p, m in saved if m.get("pass") == "decode"]
    untagged = [(p, m) for p, m in saved if not m.get("pass")]
    two_pass = bool(pre) and bool(dec)

    if two_pass:
        pf_batch = (pre[0][1].get("batch_size")
                    or int(form.get("prefill_batch_size") or 1))
        dc_batch = (dec[0][1].get("batch_size")
                    or int(form.get("decode_batch_size")
                            or form.get("batch_size") or 1))
        batch_size = dc_batch
    else:
        batch_size = int(form.get("batch_size")
                         or _from_names("batch_size") or 1)
        pf_batch = dc_batch = batch_size

    # Fetch model config for shape symbols / summary (best-effort).
    try:
        summary = summarize_config(fetch_model_config(model_id)) if model_id else {}
    except Exception:
        summary = {}

    actual_layers = form.get("actual_layers") or summary.get("num_layers")
    actual_layers = int(actual_layers) if actual_layers else None
    profiled_layers = (form.get("num_profile_layers")
                       or _from_names("profiled_layers") or actual_layers)
    profiled_layers = int(profiled_layers) if profiled_layers else None
    layer_scale = (
        actual_layers / profiled_layers
        if actual_layers and profiled_layers else 1.0
    )

    with runstate._profile_lock:
        runstate._profile_state.clear()
        runstate._profile_state.update({
            "status": "running",
            "result": None,
            "error": None,
            "model_id": model_id,
            "settings": {
                "mode": mode,
                "batch_size": batch_size,
                "prefill_batch_size": pf_batch if two_pass else None,
                "decode_batch_size": dc_batch if two_pass else None,
                "tp_size": tp_size,
                "quantization": quantization,
                "query_len": query_len,
                "context_len": context_len,
                "uploaded": True,
            },
            "run_id": runs.new_run_id((model_id or "upload").split("/")[-1]),
        })

    def _build(rank_files: list[str], bsz: int, qlen: int | None) -> dict:
        return _build_result_from_traces(
            rank_files[:tp_size] if len(rank_files) >= tp_size else rank_files,
            model_id=model_id,
            summary=summary,
            tp_size=tp_size,
            batch_size=bsz,
            mode=mode,
            quantization=quantization,
            profiled_layers=profiled_layers,
            actual_layers=actual_layers,
            layer_scale=layer_scale,
            query_len=qlen,
            context_len=context_len,
        )

    try:
        if two_pass:
            # Mirror the live two-pass build: the prefill pass supplies the
            # prefill tree (S = query_len), the decode pass is the steady-state
            # base (B = decode_batch); ``_merge_two_pass_result`` splices them.
            res_pre = _build([p for p, _ in pre], pf_batch, query_len or None)
            res_dec = _build([p for p, _ in dec], dc_batch,
                             1 if query_len else None)
            result = _merge_two_pass_result(res_pre, res_dec, pf_batch, dc_batch)
        else:
            group = untagged or dec or pre
            # A lone decode-tagged pass computes 1 new token/seq (query forced
            # to 1); otherwise use the recovered query length.
            qlen = (1 if (not untagged and dec and query_len)
                    else (query_len or None))
            result = _build([p for p, _ in group], batch_size, qlen)

        result["query_len"] = query_len or None
        result["context_len"] = context_len or None
        result["context_len_aligned"] = context_len or None
        with runstate._profile_lock:
            runstate._profile_state["status"] = "done"
            runstate._profile_state["result"] = result
            runstate._profile_state["error"] = None
    except Exception:
        err = traceback.format_exc()
        with runstate._profile_lock:
            runstate._profile_state["status"] = "error"
            runstate._profile_state["error"] = err
        save_state()
        return False, err
    save_state()
    return True, ""
