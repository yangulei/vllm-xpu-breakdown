#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
vLLM Ops/Kernels Breakdown — Web Application.

Interactive web UI for profiling vLLM inference on Intel XPU and visualizing
the op dispatch breakdown.

Usage:
    python app.py [--port 8080] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import functools
import io
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

# Force spawn for multiprocessing so vLLM's EngineCore doesn't hit
# "Cannot re-initialize CUDA in forked subprocess".
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakdown.analyzer import (
    AnalyzedOp,
    analyze_ops,
    dtype_size,
    estimate_flops,
    estimate_memory,
)
from breakdown.classifier import Backend, classify_op
from breakdown.graph_from_trace import build_graph_from_trace
from breakdown.trace_parser import _detect_device_via_torch
from breakdown.model_info import (
    fetch_model_config,
    get_dim_symbols,
    min_profile_layers,
    summarize_config,
)
from breakdown.registry import ALL_VLLM_XPU_OPS
from breakdown import shape_matrix, shape_matrix_xlsx
from breakdown.bench import (
    devices as bench_devices,
    history as bench_history,
    rank as bench_rank,
    reports as bench_reports,
    resolve as bench_resolve,
    runner as bench_runner,
    spec as bench_spec,
    store as bench_store,
)

app = Flask(__name__, static_folder="static")


# Cached at import time — won't change during server lifetime.
_DEVICE = _detect_device_via_torch() or "xpu"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vllm_xpu_breakdown")

# ---- Config Cache ----
# Persists successfully loaded model configs to disk so they appear as suggestions.

_CONFIG_CACHE_DIR = Path(__file__).parent / "output" / "config_cache"
_CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_config_cache_lock = threading.Lock()


def _cache_key(model_id: str) -> str:
    """Convert model_id to a safe filename."""
    return model_id.replace("/", "__")


def _save_config_cache(model_id: str, config: dict[str, Any]) -> None:
    """Persist config.json to disk cache."""
    key = _cache_key(model_id)
    path = _CONFIG_CACHE_DIR / f"{key}.json"
    with _config_cache_lock:
        path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")


def _load_cached_model_ids() -> list[str]:
    """Return list of model IDs that have been successfully cached."""
    ids: list[str] = []
    with _config_cache_lock:
        for p in sorted(_CONFIG_CACHE_DIR.glob("*.json")):
            ids.append(p.stem.replace("__", "/"))
    return ids


def _load_cached_config(model_id: str) -> dict[str, Any] | None:
    """Load a cached config from disk, or None if not cached."""
    key = _cache_key(model_id)
    path = _CONFIG_CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# Global state for async profiling
_profile_state = {
    "status": "idle",   # idle | running | done | error
    "result": None,
    "error": None,
    "model_id": None,
    "settings": None,
}
_profile_lock = threading.Lock()


# ---- Static file serving ----

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---- Model API ----

@app.route("/api/cached-models")
def get_cached_models():
    """Return list of model IDs whose config.json has been cached locally."""
    return jsonify({"ok": True, "models": _load_cached_model_ids()})


@app.route("/api/model/<path:model_id>")
def get_model_config(model_id: str):
    """Fetch config.json from HuggingFace (or cache) and return summary."""
    try:
        # Try cache first
        config = _load_cached_config(model_id)
        if config is None:
            config = fetch_model_config(model_id)
        # Cache on success
        _save_config_cache(model_id, config)
        summary = summarize_config(config)
        return jsonify({
            "ok": True,
            "config": config,
            "summary": summary,
            "min_profile_layers": min_profile_layers(summary),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ---- Profile API ----


def _set_num_hidden_layers(hf_config, n: int):
    """Set the decoder layer count where it actually lives in the HF config.

    Module-level (picklable) so it can be passed as a ``hf_overrides`` callable
    to vLLM, which pickles the config when spawning the EngineCore subprocess.
    Some multimodal models (e.g. MiniMax-M3) nest ``num_hidden_layers`` under
    ``text_config``; a top-level override is ignored there.
    """
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
        text_cfg.num_hidden_layers = n
    else:
        hf_config.num_hidden_layers = n
    return hf_config


def _build_layer_override(
    profiled_layers: int,
    quantization: str | None,
    layers_under_text_config: bool,
):
    """Build the ``hf_overrides`` value for reduced-layer profiling.

    Normally we return a callable that sets ``num_hidden_layers`` where it
    actually lives (top level, or nested under ``text_config`` for multimodal
    models like MiniMax-M3). vLLM applies callables in place, preserving the
    rest of the config.

    When ``quantization`` is requested, vLLM's ``get_quant_config`` rejects a
    callable ("hf_overrides must be a dict ...") because it reads the quant
    config out of ``hf_overrides``. In that case we return a **dict** override
    targeting the right key instead — vLLM applies nested ``text_config`` dicts
    recursively.
    """
    if quantization:
        if layers_under_text_config:
            return {"text_config": {"num_hidden_layers": profiled_layers}}
        return {"num_hidden_layers": profiled_layers}
    # Module-level partial so it pickles for the spawned EngineCore subprocess.
    return functools.partial(_set_num_hidden_layers, n=profiled_layers)


def _make_token_ids(n: int, vocab_size: int, seed: int) -> list[int]:
    """Deterministically build ``n`` valid, non-special token ids.

    Ids are drawn from a safe interior range of the vocabulary (avoiding the
    low ids that are typically special/control tokens) using a cheap hash of the
    position and ``seed``. Two calls with the same ``(n, vocab_size, seed)``
    yield the identical sequence, which is what lets the prefix-cache warm pass
    and the profiled pass share an exact-match context prefix.
    """
    if n <= 0:
        return []
    lo = 256
    hi = max(lo + 1, vocab_size - 256)
    span = hi - lo
    return [lo + ((i * 2654435761 + seed * 40503 + 12345) % span) for i in range(n)]


def _get_block_size(llm, default: int = 16) -> int:
    """Read the KV-cache block size from a constructed vLLM engine (robust)."""
    engine = getattr(llm, "llm_engine", None)
    for attr in ("vllm_config", "engine_config", "model_config"):
        cfg = getattr(engine, attr, None)
        cc = getattr(cfg, "cache_config", None)
        bs = getattr(cc, "block_size", None)
        if bs:
            return int(bs)
    cc = getattr(engine, "cache_config", None)
    bs = getattr(cc, "block_size", None)
    if bs:
        return int(bs)
    return default


def _get_vocab_size(llm, summary: dict, default: int = 32000) -> int:
    """Best-effort tokenizer vocabulary size for synthetic-prompt generation."""
    try:
        tok = llm.get_tokenizer()
        vs = getattr(tok, "vocab_size", None) or len(tok)
        if vs and vs > 512:
            return int(vs)
    except Exception:
        pass
    vs = summary.get("vocab_size")
    return int(vs) if vs and vs > 512 else default


def _collect_node_names(node: dict | None) -> set[str]:
    """Set of all display ``name`` values in a graph tree (for diagnostics)."""
    if not node:
        return set()
    out = {node.get("name", "")}
    for c in node.get("children", []):
        out |= _collect_node_names(c)
    return out


def _collect_module_types(node: dict | None) -> set[str]:
    """Set of all ``module_type`` (class) values in a graph tree."""
    if not node:
        return set()
    out = {node.get("module_type", "")}
    for c in node.get("children", []):
        out |= _collect_module_types(c)
    return out


# Descriptive trace filename produced by the download endpoint, e.g.
#   vllm_trace_MiniMax-M3_XPU_eager_decode_ctx2048_in1536_out8_bs32_tp4_6layers.json.gz
# The stable ``_ctx…_in…_out…_bs…_tp…[_quant]_…layers`` tail (plus the optional
# ``_prefill``/``_decode`` pass tag and the ``_device_mode`` before it) lets the
# upload endpoint recover the full profiled configuration, making a
# download -> upload reconstruction a lossless round-trip.
_TRACE_NAME_RE = re.compile(
    r"_(?P<device>XPU|CUDA|GPU|CPU)_(?P<mode>eager|compile)"
    r"(?:_(?P<pass>prefill|decode))?"
    r"_ctx(?P<ctx>\d+)_in(?P<qin>\d+)_out(?P<gen>[A-Za-z0-9]+)"
    r"_bs(?P<bs>\d+)_tp(?P<tp>\d+)"
    r"(?:_(?P<quant>[A-Za-z0-9]+))?_(?P<layers>[A-Za-z0-9]+)layers",
    re.IGNORECASE,
)


# Rank marker in a raw vLLM per-rank trace filename, e.g.
#   dp0_pp0_tp0_dcp0_ep0_rank0.<id>.pt.trace.json.gz
# The tensor-parallel rank is encoded as ``rank<N>`` (preferred) or ``tp<N>``.
_RANK_NAME_RE = re.compile(r"rank(?P<rank>\d+)", re.IGNORECASE)
_TP_RANK_NAME_RE = re.compile(r"(?:^|[_/])tp(?P<rank>\d+)", re.IGNORECASE)


def _trace_rank(path: str) -> int | None:
    """Extract the tensor-parallel rank index from a raw trace filename.

    vLLM writes one trace per rank named ``…_tp<N>_…_rank<N>.<id>.pt.trace.json.gz``.
    Returns the rank as an int (``rank<N>`` preferred, ``tp<N>`` fallback), or
    ``None`` when no rank marker is present (e.g. a merged/descriptive name).
    """
    name = os.path.basename(path or "")
    m = _RANK_NAME_RE.search(name) or _TP_RANK_NAME_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group("rank"))
    except (TypeError, ValueError):
        return None


def _rank0_first(rank_files: list[str]) -> list[str]:
    """Reorder trace files so the tensor-parallel **rank-0** file comes first.

    Multi-rank traces arrive sorted by mtime (whichever rank flushed last), so
    ``rank_files[0]`` is not necessarily rank 0. The rank-1..N allreduce (and
    other collectives) can idle much longer than rank 0 waiting to synchronize,
    which inflates their device time; rank 0 is the representative worker, so
    the OP breakdown, reconstructed graph and downloadable trace are all built
    from it. This lifts the file whose name encodes ``rank0``/``tp0`` to the
    front (stable order otherwise). If no file carries a rank marker, the list
    is returned unchanged.
    """
    if len(rank_files) <= 1:
        return list(rank_files)
    ranked = [(f, _trace_rank(f)) for f in rank_files]
    if all(r is None for _, r in ranked):
        return list(rank_files)
    # rank-0 first; unknown ranks sorted last, otherwise stable by rank index.
    return [f for f, _ in sorted(
        ranked,
        key=lambda fr: (fr[1] is None, fr[1] if fr[1] is not None else 0),
    )]


def _parse_trace_filename(name: str) -> dict:
    """Recover profiling config from a download-endpoint trace filename.

    Returns a dict with keys ``pass`` (``"prefill"``/``"decode"``/``None``),
    ``mode``, ``device``, ``context_len``, ``query_len``, ``gen``,
    ``batch_size``, ``tp``, ``quantization`` and ``profiled_layers`` (``None``
    when the name encodes ``all`` layers). An unrecognized name yields ``{}``.
    """
    m = _TRACE_NAME_RE.search(name or "")
    if not m:
        return {}
    g = m.groupdict()

    def _int(v: str | None) -> int | None:
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    quant = g.get("quant")
    if quant and quant.lower() in ("none", "auto"):
        quant = None
    return {
        "device": (g.get("device") or "").upper(),
        "mode": (g.get("mode") or "").lower(),
        "pass": (g.get("pass") or "").lower() or None,
        "context_len": _int(g.get("ctx")),
        "query_len": _int(g.get("qin")),
        "gen": _int(g.get("gen")),
        "batch_size": _int(g.get("bs")),
        "tp": _int(g.get("tp")),
        "quantization": quant,
        "profiled_layers": _int(g.get("layers")),  # "all" -> None
    }


def _build_result_from_traces(
    rank_files: list[str],
    *,
    model_id: str,
    summary: dict,
    dim_symbols: dict,
    tp_size: int,
    batch_size: int,
    mode: str = "eager",
    max_model_len: int | None = None,
    max_tokens: int | None = None,
    quantization: str | None = None,
    profiled_layers: int | None = None,
    actual_layers: int | None = None,
    layer_scale: float = 1.0,
    trace_file: str | None = None,
    ref_module_tree: dict | None = None,
    query_len: int | None = None,
    context_len: int | None = None,
) -> dict:
    """Parse one or more trace files and build the profile result dict.

    Shared by the live profiler (``_run_profile``) and the trace-upload
    endpoint so both paths reconstruct the model graph and op breakdown the
    same way. With TP>1, vLLM writes one trace per rank; the ranks 1..N idle
    longer on collectives (their allreduce device time is inflated by the wait
    to synchronize with rank 0), so **rank 0 is always used** as the
    representative worker for the op breakdown, the reconstructed graph and the
    downloadable trace. ``_rank0_first`` lifts the ``rank0``/``tp0`` file to the
    front regardless of the mtime order the files arrive in; the other ranks are
    ignored.
    """
    from breakdown.trace_parser import parse_trace_file

    rank_files = _rank0_first(rank_files)
    op_dicts = parse_trace_file(rank_files[0])

    if not op_dicts:
        raise RuntimeError(
            f"No ops found in trace file {rank_files[0]}. "
            "The trace may not contain any captured events."
        )

    analyzed = analyze_ops(
        op_dicts,
        dim_symbols=dim_symbols,
        batch_size=batch_size,
        seq_len=None,
        model_dtype=summary.get("dtype", "bfloat16"),
        num_layers=summary.get("num_layers"),
    )

    backend_totals: dict[str, dict] = {}
    total_dev = sum(o.device_time_us for o in analyzed)
    for b in Backend:
        ops = [o for o in analyzed if o.backend == b.value]
        dev = sum(o.device_time_us for o in ops)
        backend_totals[b.value] = {
            "device_time_us": dev,
            "pct": round(dev / total_dev * 100, 1) if total_dev > 0 else 0,
            "num_ops": len(ops),
            "num_calls": sum(o.call_count for o in ops),
        }

    profile_result = {
        "model_id": model_id,
        "mode": mode,
        "batch_size": batch_size,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "tp_size": tp_size,
        "quantization": quantization,
        "summary": summary,
        "total_device_time_us": total_dev,
        "total_cpu_time_us": sum(o.cpu_time_us for o in analyzed),
        "backends": backend_totals,
        "ops": [o.to_dict() for o in analyzed],
        "profiled_layers": profiled_layers,
        "actual_layers": actual_layers,
        "layer_scale": layer_scale,
        "trace_file": trace_file if trace_file is not None else rank_files[0],
    }

    # Reconstruct the model graph directly from the profiler trace.
    try:
        graph = build_graph_from_trace(
            rank_files[0],
            summary=summary,
            tp_size=tp_size,
            batch_size=batch_size,
            quantization=quantization,
            ref_module_tree=ref_module_tree,
            query_len=query_len,
            context_len=context_len,
        )
        graph["profiled_layers"] = profiled_layers
        graph["actual_layers"] = actual_layers
        graph["layer_scale"] = layer_scale
        profile_result["graph"] = graph
        if ref_module_tree:
            names = _collect_node_names(graph.get("prefill")
                                        or graph.get("decode"))
            recovered = names & {"q_norm", "k_norm", "input_layernorm",
                                 "post_attention_layernorm"}
            if recovered:
                logger.info("Module-name recovery applied to graph: %s",
                            ", ".join(sorted(recovered)))
            else:
                logger.warning(
                    "Module-name recovery: reference tree was available but no "
                    "attribute names landed on the graph (structural alignment "
                    "found no match). Trace module classes seen: %s",
                    ", ".join(sorted(_collect_module_types(
                        graph.get("prefill") or graph.get("decode")))[:20]),
                )
    except Exception:
        logger.warning("Graph reconstruction failed", exc_info=True)

    return profile_result


def _merge_two_pass_result(pre: dict, dec: dict,
                           prefill_bs: int, decode_bs: int) -> dict:
    """Splice a prefill-batch pass and a decode-batch pass into one result.

    Real serving decouples the phases: prefill typically runs ~1 sequence at a
    time while decode batches many concurrent sequences. A single
    ``llm.generate`` call cannot express that (it prefills and decodes the same
    batch), so we profile two passes and merge them here:

    - ``pre`` — full result from a pass run at ``prefill_bs`` (its **prefill**
      phase is the faithful one; ``S`` = query_len).
    - ``dec`` — full result from a pass run at ``decode_bs`` (its **decode**
      phase is faithful; ``B`` = decode_bs).

    The merged result keeps the decode pass as the base (its op breakdown
    reflects the steady-state, throughput-bound decode batch) and overlays the
    prefill pass's prefill graph tree, so the reconstructed graph shows
    prefill@``prefill_bs`` together with decode@``decode_bs``.
    """
    result = dict(dec)
    result["batch_size"] = decode_bs
    result["prefill_batch_size"] = prefill_bs
    result["decode_batch_size"] = decode_bs
    result["two_pass"] = True
    # Retain BOTH passes' trace files so the trace-download endpoint can serve
    # either phase. ``trace_file`` (inherited from the decode pass via
    # ``dict(dec)``) stays the default so existing clients are unaffected.
    result["prefill_trace_file"] = pre.get("trace_file")
    result["decode_trace_file"] = dec.get("trace_file")

    gpre = pre.get("graph") or {}
    gdec = dec.get("graph") or {}
    graph = dict(gdec)
    graph["prefill"] = gpre.get("prefill")
    # Symbols: the decode pass supplies ``B`` (decode batch); the prefill pass
    # supplies the prefill token dims ``S`` / ``S+C`` / ``C``.
    sym = dict(gdec.get("symbols") or {})
    presym = gpre.get("symbols") or {}
    for k in ("S", "S+C", "C"):
        if k in presym:
            sym[k] = presym[k]
    graph["symbols"] = sym
    result["graph"] = graph
    return result


def _run_profile(model_id: str, mode: str, max_model_len: int,
                 batch_size: int, max_tokens: int, prompt: str,
                 num_profile_layers: int | None = None,
                 tp_size: int = 1,
                 quantization: str | None = None,
                 gpu_memory_utilization: float | None = None,
                 query_len: int | None = None,
                 context_len: int | None = None,
                 prefill_batch_size: int | None = None,
                 decode_batch_size: int | None = None):
    """Run profiling in a background thread using vLLM's native profiler.

    On XPU hardware, vLLM automatically selects XPUWorker which uses
    the correct profiler activities (["CPU", "XPU"]).

    Args:
        num_profile_layers: If set (e.g. 1), override the model to load only
            this many layers. Timing is then scaled by actual_layers/profiled
            for the full model estimate. Enables profiling models too large
            for the GPU.
        tp_size: tensor parallel size (default 1). With TP>1, vLLM creates
            one trace file per rank; we parse all and aggregate timing.
        quantization: quantization method (e.g. "fp8", "gptq", "awq").
            Passed as --quantization to vLLM.
        gpu_memory_utilization: fraction of device memory vLLM may use. Lower
            it (e.g. 0.8) when vLLM's init footprint leaves too little headroom
            for the default (0.92) on small-VRAM cards. None keeps vLLM default.
        query_len: number of *new* prompt tokens the profiled prefill computes
            (the "Query Len"). Drives the prefill token dimension ``S``.
        context_len: number of prior context tokens the query attends to (the
            "Context Len"). When >0, those tokens are pre-computed in an
            un-profiled warm pass and served from the prefix cache (APC) during
            the profiled run, so the profiled prefill computes only ``query_len``
            new tokens while attention still reads the full ``context_len+query_len``
            KV. Rounded down to a KV block boundary so the whole context caches.
        prefill_batch_size: number of concurrent sequences for the **prefill**
            phase (typically 1 in real serving). When it differs from
            ``decode_batch_size`` the run is profiled in two passes — a prefill
            pass at this batch and a decode pass at ``decode_batch_size`` — and
            the two phase graphs are merged. ``None`` falls back to
            ``batch_size`` (single pass, legacy behaviour).
        decode_batch_size: number of concurrent sequences for the **decode**
            phase (often 32/64/128). See ``prefill_batch_size``. ``None`` falls
            back to ``batch_size``.
    """
    global _profile_state
    try:
        from vllm import LLM, SamplingParams, TokensPrompt

        # Fetch model config for analysis
        try:
            config = fetch_model_config(model_id)
            summary = summarize_config(config)
            dim_symbols = get_dim_symbols(summary)
        except Exception:
            config = {}
            summary = {}
            dim_symbols = {}

        actual_layers = summary.get("num_layers") or 1
        if num_profile_layers == "min":
            # Auto-calculate minimum layers needed
            profiled_layers = min_profile_layers(summary)
        elif num_profile_layers:
            profiled_layers = int(num_profile_layers)
        else:
            profiled_layers = actual_layers
        layer_scale = actual_layers / profiled_layers

        trace_dir = os.path.abspath("output/traces")
        os.makedirs(trace_dir, exist_ok=True)

        engine_kwargs: dict = {
            "model": model_id,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tp_size,
            "profiler_config": {
                "profiler": "torch",
                "torch_profiler_dir": trace_dir,
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_stack": True,
                "torch_profiler_with_flops": True,
                "torch_profiler_use_gzip": True,
            },
        }

        # Always use dummy weights for profiling — timing doesn't depend on
        # weight values, and dummy avoids KeyError when layers are reduced.
        engine_kwargs["load_format"] = "dummy"

        # Optionally cap device memory usage (leaves headroom for vLLM's init
        # footprint on small-VRAM cards; None keeps vLLM's default).
        if gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

        # Vision-language models: disable multimodal memory profiling so the run
        # captures the language-model ops on a text prompt. This avoids vLLM's
        # dummy image/video profiling path through the vision tower (which the
        # static graph already covers) and keeps the profile focused on the LLM.
        if summary.get("vit_hidden_size"):
            engine_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

        # Sparse-attention models (e.g. MiniMax-M3) select fixed-size KV blocks
        # via the lightning indexer, so the KV-cache block size must match the
        # sparse block size; otherwise vLLM cannot reconcile a common kernel
        # block size across the sparse/full attention backends.
        sparse_block = summary.get("sparse_block_size")
        if sparse_block:
            engine_kwargs["block_size"] = int(sparse_block)

        # Normalize the query/context sizing knobs. ``query_len`` sets the
        # number of new prompt tokens the profiled prefill computes; when
        # ``context_len`` > 0 we serve that many prior tokens from the prefix
        # cache so the profiled prefill sees ``S = query_len`` new tokens
        # attending to a ``context_len``-token KV context.
        query_len = int(query_len) if query_len else 0
        context_len = int(context_len) if context_len else 0
        use_token_prompts = query_len > 0
        # The context length actually served from the prefix cache, floored to a
        # whole number of KV blocks (set below once the block size is known). The
        # graph reconstruction symbolizes this value as ``C`` so attention KV
        # dims read ``C`` / ``S+C`` instead of a bare number.
        profiled_context_len = 0

        # Enable Automatic Prefix Caching so the context prefix computed in the
        # warm pass is reused (not recomputed) during the profiled run.
        if context_len > 0:
            engine_kwargs["enable_prefix_caching"] = True

        # Quantization method
        if quantization:
            engine_kwargs["quantization"] = quantization

        # Override layer count for reduced-layer profiling.
        if profiled_layers < actual_layers:
            # Some multimodal models (e.g. MiniMax-M3) nest the decoder layer
            # count under ``text_config``. A top-level ``num_hidden_layers``
            # override is silently ignored there, so the full model is built
            # and exhausts device memory (UR_RESULT_ERROR_DEVICE_LOST). The
            # helper returns a callable normally, but a dict when quantization
            # is set (vLLM's ``get_quant_config`` requires a dict override).
            engine_kwargs["hf_overrides"] = _build_layer_override(
                profiled_layers,
                quantization,
                bool(summary.get("layers_under_text_config")),
            )

        # Set compile / eager mode
        if mode == "compile":
            os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"
            engine_kwargs["enforce_eager"] = False
        else:
            os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)
            engine_kwargs["enforce_eager"] = True

        # Resolve the per-phase batch sizes. When prefill and decode batches
        # differ we profile two passes (real serving prefills ~1 sequence while
        # decoding many); otherwise a single pass reproduces legacy behaviour.
        pf_batch = int(prefill_batch_size) if prefill_batch_size else int(batch_size)
        dc_batch = int(decode_batch_size) if decode_batch_size else int(batch_size)
        two_pass = pf_batch != dc_batch
        max_batch = max(pf_batch, dc_batch)

        # Pin the scheduler so every decode step runs the *full* requested batch
        # (``B = decode_batch``). Left to its defaults, vLLM's continuous-batching
        # scheduler caps per-iteration concurrency (by ``max_num_seqs`` and by how
        # many sequences' KV fits in cache) and runs an oversized batch in
        # *partial-batch waves* — e.g. a batch of 32 dispatched as 29 + 3. Each
        # wave has a different row count, so its ops neither symbolize to ``B``
        # nor merge with the full-batch ops, surfacing as duplicated ``29``/``3``
        # nodes in the reconstructed decode graph. ``max_num_seqs = max_batch``
        # forces the scheduler to admit the whole batch in one iteration;
        # ``max_num_batched_tokens`` is sized to also admit a whole batch's
        # prefill tokens in a single step (prefill pass: ``pf_batch × query_len``;
        # decode pass: ``dc_batch`` single-token prefills) so a full-shape step is
        # never chunked. If the batch's KV cannot fit device memory, raise
        # ``gpu_memory_utilization`` or lower Context/Batch rather than letting the
        # run silently split.
        engine_kwargs["max_num_seqs"] = max_batch
        _prefill_step_tokens = pf_batch * max(int(query_len), 1)
        engine_kwargs["max_num_batched_tokens"] = max(
            _prefill_step_tokens, max_batch, 2048)

        llm = LLM(**engine_kwargs)

        if use_token_prompts:
            block_size = _get_block_size(llm)
            vocab_size = _get_vocab_size(llm, summary)
            # Round the context down to a whole number of KV blocks so the entire
            # context prefix is cacheable (a trailing partial block would be
            # recomputed and shift the profiled prefill token count).
            ctx_aligned = (context_len // block_size) * block_size
            profiled_context_len = ctx_aligned
            ctx_ids = _make_token_ids(ctx_aligned, vocab_size, seed=0)
        else:
            ctx_aligned = 0
            ctx_ids = []

        def _list_trace_files() -> set:
            return {os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
                    if f.endswith(".json") or f.endswith(".json.gz")}

        def _install_span_hooks() -> bool:
            """Install capture-time module-name span hooks in the worker(s).

            Registers forward hooks that emit ``record_function(
            "module::<qname>::<Cls>")`` spans around every module's forward, so
            the trace carries real attribute names (``q_norm``/``k_norm``,
            ``self_attn``, ...) and ``build_graph_from_trace`` reconstructs the
            tree with exact names — no reference-tree overlay needed (research
            R1). Best-effort: on any failure the run proceeds and naming falls
            back to the ``module_naming`` overlay.
            """
            try:
                from breakdown.module_hooks import install_module_span_hooks_on
                llm.apply_model(install_module_span_hooks_on)
                return True
            except Exception:
                logger.warning("module span hooks: install failed; module names "
                               "will fall back to the name overlay",
                               exc_info=True)
                return False

        def _remove_span_hooks(installed: bool) -> None:
            if not installed:
                return
            try:
                from breakdown.module_hooks import remove_module_span_hooks_on
                llm.apply_model(remove_module_span_hooks_on)
            except Exception:
                logger.warning("module span hooks: remove failed", exc_info=True)

        def _profiled_pass(pass_batch: int, pass_query_len: int,
                           pass_max_tokens: int):
            """Warm + run one profiled generate at the given batch/query size.

            ``pass_max_tokens`` is the number of tokens to generate this pass:
            the decode pass uses the full decode budget (so decode steps are
            captured), while the prefill pass uses **1** — it only needs the
            single prefill step (``S`` = ``query_len``), and generating extra
            decode tokens would only bloat the trace and slow the run.

            ``ignore_eos`` keeps every sequence alive for the full budget so the
            profiled decode step reflects the requested batch (a sequence hitting
            EOS early would shrink the observed decode concurrency ``B``).

            Returns ``(rank_files, cache_hit_note)``: the trace file(s) newly
            written by this pass (newest first, capped to ``tp_size``) and an
            optional prefix-cache-miss note.
            """
            note = None
            pass_sampling = SamplingParams(max_tokens=pass_max_tokens,
                                           ignore_eos=True)
            if use_token_prompts:
                def _full_prompt(query_seed: int) -> "TokensPrompt":
                    q = _make_token_ids(pass_query_len, vocab_size, seed=query_seed)
                    return TokensPrompt(prompt_token_ids=ctx_ids + q)

                # Distinct query per batch item (shared context prefix) so every
                # sequence genuinely prefills its own tokens instead of
                # cache-hitting a sibling; profiled seeds differ from warmup
                # seeds so the profiled queries are never served from cache.
                def _batch(base_seed: int) -> list["TokensPrompt"]:
                    return [_full_prompt(base_seed + b) for b in range(pass_batch)]

                # Warm the shared prefix cache once (un-profiled) so the profiled
                # run reads the context from cache instead of recomputing it.
                if ctx_ids:
                    llm.generate(
                        [TokensPrompt(prompt_token_ids=ctx_ids)],
                        SamplingParams(max_tokens=1), use_tqdm=False,
                    )
                for w in range(2):
                    llm.generate(_batch(1000 * (w + 1)), pass_sampling,
                                 use_tqdm=False)
                profiled_prompts = _batch(900000)

                before = _list_trace_files()
                _spans = _install_span_hooks()
                llm.start_profile()
                outputs = llm.generate(profiled_prompts, pass_sampling,
                                       use_tqdm=False)
                llm.stop_profile()
                _remove_span_hooks(_spans)

                # Verify the context was served from cache. A miss means the
                # profiled prefill recomputed the whole context (S = context +
                # query), so record a note rather than silently misreporting.
                if context_len > 0 and outputs:
                    cached = getattr(outputs[0], "num_cached_tokens", None)
                    if cached is not None and cached < ctx_aligned:
                        note = (
                            f"Prefix cache hit only {cached}/{ctx_aligned} "
                            "context tokens; profiled prefill may include "
                            "context recompute."
                        )
            else:
                # --- Legacy text-prompt path (Query/Context Len unspecified) ---
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                conversations = [conversation] * pass_batch
                prompts = [prompt] * pass_batch

                # First warmup pass also detects chat-template support.
                try:
                    llm.chat(conversations, pass_sampling, use_tqdm=False)
                    use_chat = True
                except Exception:
                    llm.generate(prompts, pass_sampling, use_tqdm=False)
                    use_chat = False

                def _run_inference():
                    if use_chat:
                        llm.chat(conversations, pass_sampling, use_tqdm=False)
                    else:
                        llm.generate(prompts, pass_sampling, use_tqdm=False)

                # Warmup primes Triton JIT / autotuning so the profiled trace
                # reflects steady-state timing. Detection pass above is warmup
                # #1; run 2 more for 3 total.
                for _ in range(2):
                    _run_inference()

                before = _list_trace_files()
                _spans = _install_span_hooks()
                llm.start_profile()
                _run_inference()
                llm.stop_profile()
                _remove_span_hooks(_spans)

            # torch's profiler writes the trace on stop_profile; wait briefly for
            # the new file(s) to appear, then return them (newest first).
            new_files: list[str] = []
            for _ in range(20):
                new_files = sorted(_list_trace_files() - before,
                                   key=os.path.getmtime, reverse=True)
                if len(new_files) >= tp_size:
                    break
                time.sleep(0.5)
            if not new_files:
                new_files = sorted(_list_trace_files(),
                                   key=os.path.getmtime, reverse=True)
            return new_files[:tp_size], note

        # --- Run the profiled pass(es) ---
        if two_pass:
            # Prefill pass: batch = prefill_batch, real query_len, generate only
            # 1 token so the trace holds exactly the prefill step (S=query_len)
            # — we keep only its prefill phase. Decode pass: batch = decode_batch,
            # query_len forced to 1 so decode is 1 new token/seq (matches real
            # decode and avoids OOM from prefilling decode_batch x query_len
            # tokens), generating the full decode budget — we keep its decode
            # phase.
            pre_files, note_pre = _profiled_pass(pf_batch, query_len,
                                                 pass_max_tokens=1)
            dec_query = 1 if use_token_prompts else 0
            dec_files, note_dec = _profiled_pass(dc_batch, dec_query,
                                                 pass_max_tokens=max_tokens)
            cache_hit_note = note_pre or note_dec
        else:
            # Single pass yields both phases from one run, so it needs the full
            # decode budget.
            single_files, cache_hit_note = _profiled_pass(
                dc_batch, query_len, pass_max_tokens=max_tokens)

        # Module attribute names (q_norm/k_norm, ...) come primarily from the
        # capture-time ``module::`` spans installed above, which the
        # reconstruction reads directly from the trace. We still capture a
        # ``named_modules()`` reference tree from the live model as a *fallback*
        # overlay: ``build_graph_from_trace`` uses it only if the trace lacks
        # those spans (e.g. hook install failed, or an older trace).
        ref_module_tree = None
        try:
            from breakdown.module_naming import ref_tree_from_llm
            ref_module_tree = ref_tree_from_llm(llm)
        except Exception:
            logger.warning("ref_tree_from_llm raised; fallback name overlay "
                           "disabled", exc_info=True)
            ref_module_tree = None

        if ref_module_tree:
            logger.info(
                "Module-name overlay (fallback): got reference tree from live "
                "model (root=%s, %d top-level children)",
                ref_module_tree.get("cls"),
                len(ref_module_tree.get("children", [])),
            )
        else:
            logger.warning(
                "Module-name overlay (fallback): could NOT read module names "
                "from the live model (ref_tree_from_llm returned None). If the "
                "capture-time spans also failed, q_norm/k_norm and other "
                "attribute names will fall back to class heuristics."
            )
            # Fallback: the offline meta-device path. Heavy (re-instantiates on
            # meta) but lets naming work when the live-model traversal fails.
            if config:
                try:
                    from breakdown.module_naming import ref_tree_from_config
                    ref_module_tree = ref_tree_from_config(
                        config,
                        dtype=summary.get("dtype", "bfloat16"),
                        model_id=model_id,
                    )
                    if ref_module_tree:
                        logger.info(
                            "Module-name overlay (fallback): recovered names via "
                            "meta-device fallback."
                        )
                except Exception:
                    logger.warning("ref_tree_from_config fallback failed",
                                   exc_info=True)
                    ref_module_tree = None

        # --- Parse trace files & build the result ---
        # With TP>1, vLLM produces one trace file per rank; each pass returns all
        # of its rank files (mtime order) and ``_build_result_from_traces`` picks
        # the rank-0 file via ``_rank0_first``.
        def _build(files: list[str], bsz: int, qlen: int | None) -> dict:
            if not files:
                raise RuntimeError(
                    f"No trace files found in {trace_dir}. "
                    "Profiling may have failed in the worker process."
                )
            return _build_result_from_traces(
                files,
                model_id=model_id,
                summary=summary,
                dim_symbols=dim_symbols,
                tp_size=tp_size,
                batch_size=bsz,
                mode=mode,
                max_model_len=max_model_len,
                max_tokens=max_tokens,
                quantization=quantization,
                profiled_layers=profiled_layers,
                actual_layers=actual_layers,
                layer_scale=layer_scale,
                ref_module_tree=ref_module_tree,
                query_len=qlen,
                context_len=profiled_context_len or None,
            )

        if two_pass:
            res_pre = _build(pre_files, pf_batch, query_len or None)
            res_dec = _build(dec_files, dc_batch,
                             1 if use_token_prompts else None)
            profile_result = _merge_two_pass_result(
                res_pre, res_dec, pf_batch, dc_batch)
        else:
            profile_result = _build(single_files, dc_batch, query_len or None)

        profile_result["query_len"] = query_len or None
        profile_result["context_len"] = context_len or None
        profile_result["context_len_aligned"] = profiled_context_len or None
        if cache_hit_note:
            profile_result["cache_hit_note"] = cache_hit_note

        with _profile_lock:
            _profile_state["status"] = "done"
            _profile_state["result"] = profile_result
            _profile_state["error"] = None

    except Exception as e:
        with _profile_lock:
            _profile_state["status"] = "error"
            _profile_state["error"] = traceback.format_exc()
    finally:
        os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


@app.route("/api/profile", methods=["POST"])
def start_profile():
    """Start a profiling run. Non-blocking — poll /api/profile/status."""
    global _profile_state

    with _profile_lock:
        if _profile_state["status"] == "running":
            return jsonify({"ok": False, "error": "Profiling already in progress"}), 409

    data = request.json or {}
    model_id = data.get("model_id", "")
    if not model_id:
        return jsonify({"ok": False, "error": "model_id is required"}), 400

    mode = data.get("mode", "eager")
    max_model_len = data.get("max_model_len", 4096)
    batch_size = data.get("batch_size", 1)
    max_tokens = data.get("max_tokens", 128)
    prompt = data.get("prompt", "Write a short essay about AI.")
    num_profile_layers = data.get("num_profile_layers")  # None = all layers
    tp_size = data.get("tensor_parallel_size", 1)
    quantization = data.get("quantization")  # None = no quantization
    gpu_memory_utilization = data.get("gpu_memory_utilization")  # None = vLLM default
    query_len = data.get("query_len")  # new prefill tokens (None = legacy prompt)
    context_len = data.get("context_len")  # cached prefix tokens (None/0 = none)
    # Per-phase batch sizes. When they differ, profiling runs two passes
    # (prefill@prefill_batch_size + decode@decode_batch_size) and merges them.
    # Absent → fall back to the single ``batch_size`` (legacy single pass).
    prefill_batch_size = data.get("prefill_batch_size")
    decode_batch_size = data.get("decode_batch_size")

    # The engine must fit the whole sequence it will ever see: cached context +
    # new query tokens + the decode tokens we generate. The frontend sizes
    # max_model_len from Query+Context; bump it to also cover the decode budget.
    if query_len:
        needed = int(query_len) + int(context_len or 0) + int(max_tokens) + 16
        if needed > int(max_model_len):
            max_model_len = needed

    with _profile_lock:
        _profile_state = {
            "status": "running",
            "result": None,
            "error": None,
            "model_id": model_id,
            "settings": {
                "mode": mode,
                "batch_size": batch_size,
                "prefill_batch_size": prefill_batch_size,
                "decode_batch_size": decode_batch_size,
                "max_model_len": max_model_len,
                "tp_size": tp_size,
                "quantization": quantization,
                "query_len": query_len,
                "context_len": context_len,
            },
        }

    thread = threading.Thread(
        target=_run_profile,
        args=(model_id, mode, max_model_len, batch_size, max_tokens, prompt,
              num_profile_layers, tp_size, quantization, gpu_memory_utilization,
              query_len, context_len, prefill_batch_size, decode_batch_size),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "status": "running"})


@app.route("/api/profile/upload", methods=["POST"])
def upload_profile():
    """Reconstruct the model graph and op breakdown from uploaded trace(s).

    Accepts a multipart form with one or more ``trace`` files (torch profiler
    Chrome traces, ``.json`` or ``.json.gz``). The upload path mirrors the live
    profiler so a **download -> upload round-trip** reconstructs the same graph
    (and drives the Shape Matrix export) on a machine **without an XPU/GPU**:

      - **Two-pass pair** — upload the separate ``…_prefill_…`` and
        ``…_decode_…`` files the download endpoint produces for a two-pass run
        (distinct prefill/decode batch sizes). Each pass's phase graph is built
        with that pass's batch/query size and merged via ``_merge_two_pass_result``
        (prefill tree + decode base), exactly like a live two-pass run — so the
        reconstructed graph has **both** phases, not decode only.
      - **Single trace** — one file (optionally several rank files for TP>1,
        in any order) reconstructs a single run; the **rank-0** file is used and
        the other ranks are ignored (rank 0 is the representative worker — see
        ``_rank0_first``).

    The descriptive download filenames are self-describing
    (``…_{mode}[_prefill|_decode]_ctx{C}_in{S}_out{gen}_bs{B}_tp{TP}[_{quant}]_{N}layers``),
    so mode/TP/quant/query/context/batch and the prefill|decode split are
    recovered from the names via ``_parse_trace_filename`` — no need to re-enter
    them. Explicit form fields still override, and legacy names without the
    encoded tail fall back to form fields:

      - ``model_id``: HF id used to fetch config for shape symbols / summary
      - ``tensor_parallel_size`` / ``tp_size``: ranks represented by the uploads
      - ``batch_size`` / ``prefill_batch_size`` / ``decode_batch_size``
      - ``quantization``, ``mode``, ``query_len``, ``context_len``
      - ``num_profile_layers`` / ``actual_layers``: for reduced-layer scaling

    Parsing is fast, so this runs synchronously and stores the result in the
    shared profile state so ``/api/profile/result`` and ``/api/profile/trace``
    work exactly as they do for a live profiling run.
    """
    global _profile_state

    with _profile_lock:
        if _profile_state["status"] == "running":
            return jsonify({"ok": False, "error": "Profiling already in progress"}), 409

    files = request.files.getlist("trace")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"ok": False, "error": "No trace file uploaded"}), 400

    form = request.form
    model_id = (form.get("model_id") or "").strip()

    # Persist uploads under output/traces so the trace-download endpoint works,
    # keeping each file's original (descriptive) name for config recovery.
    from werkzeug.utils import secure_filename
    trace_dir = os.path.abspath("output/traces")
    os.makedirs(trace_dir, exist_ok=True)
    saved: list[tuple[str, dict]] = []  # (saved_path, parsed-filename meta)
    for f in files:
        orig = os.path.basename(f.filename or "")
        name = secure_filename(f.filename) or "uploaded_trace.json"
        dest = os.path.join(trace_dir, name)
        f.save(dest)
        saved.append((dest, _parse_trace_filename(orig)))

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
        dim_symbols = get_dim_symbols(summary) if summary else {}
    except Exception:
        summary = {}
        dim_symbols = {}

    actual_layers = form.get("actual_layers") or summary.get("num_layers")
    actual_layers = int(actual_layers) if actual_layers else None
    profiled_layers = (form.get("num_profile_layers")
                       or _from_names("profiled_layers") or actual_layers)
    profiled_layers = int(profiled_layers) if profiled_layers else None
    layer_scale = (
        actual_layers / profiled_layers
        if actual_layers and profiled_layers else 1.0
    )

    with _profile_lock:
        _profile_state = {
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
        }

    # Reconstruct real module attribute names by instantiating the model on
    # ``meta`` (no weights). Heavy + network-dependent, and only used as a
    # *fallback* overlay — traces profiled by this tool already carry
    # capture-time ``module::`` name spans, so ``build_graph_from_trace``
    # ignores this tree when those spans are present.
    ref_module_tree = None
    if model_id:
        try:
            from breakdown.module_naming import ref_tree_from_config
            ref_module_tree = ref_tree_from_config(
                fetch_model_config(model_id),
                dtype=summary.get("dtype", "bfloat16"),
                model_id=model_id,
            )
        except Exception:
            ref_module_tree = None

    def _build(rank_files: list[str], bsz: int, qlen: int | None) -> dict:
        return _build_result_from_traces(
            rank_files[:tp_size] if len(rank_files) >= tp_size else rank_files,
            model_id=model_id,
            summary=summary,
            dim_symbols=dim_symbols,
            tp_size=tp_size,
            batch_size=bsz,
            mode=mode,
            quantization=quantization,
            profiled_layers=profiled_layers,
            actual_layers=actual_layers,
            layer_scale=layer_scale,
            ref_module_tree=ref_module_tree,
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
        with _profile_lock:
            _profile_state["status"] = "done"
            _profile_state["result"] = result
            _profile_state["error"] = None
    except Exception:
        err = traceback.format_exc()
        with _profile_lock:
            _profile_state["status"] = "error"
            _profile_state["error"] = err
        return jsonify({"ok": False, "error": err}), 500

    return jsonify({"ok": True, "status": "done"})


@app.route("/api/profile/status")
def profile_status():
    """Check profiling status."""
    with _profile_lock:
        return jsonify({
            "status": _profile_state["status"],
            "model_id": _profile_state["model_id"],
            "settings": _profile_state.get("settings"),
            "error": _profile_state["error"],
        })


@app.route("/api/profile/result")
def profile_result():
    """Get profiling results (only available when status=done)."""
    with _profile_lock:
        if _profile_state["status"] != "done":
            return jsonify({
                "ok": False,
                "status": _profile_state["status"],
                "error": _profile_state.get("error"),
            }), 202
        result = _profile_state["result"]
    # Don't expose internal trace_file path(s); indicate availability instead.
    _internal = {"trace_file", "prefill_trace_file", "decode_trace_file"}
    client_result = {k: v for k, v in result.items() if k not in _internal}
    client_result["has_trace"] = bool(result.get("trace_file"))
    client_result["has_prefill_trace"] = bool(result.get("prefill_trace_file"))
    client_result["has_decode_trace"] = bool(result.get("decode_trace_file"))
    return jsonify({"ok": True, "data": client_result})


@app.route("/api/profile/trace")
def download_trace():
    """Download a profiled trace file with a descriptive filename.

    Two-pass runs (separate prefill/decode batch sizes) write a separate trace
    per pass. Use ``?pass=prefill`` or ``?pass=decode`` to pick one; without it
    the default (decode, i.e. the merged base) is served for backward compat.
    """
    with _profile_lock:
        if _profile_state["status"] != "done" or not _profile_state.get("result"):
            return jsonify({"ok": False, "error": "No profile result available"}), 404
        result = _profile_state["result"]

    which = (request.args.get("pass") or "").strip().lower()
    if which == "prefill":
        trace_path = result.get("prefill_trace_file")
        pass_tag = "_prefill"
        pass_bs = result.get("prefill_batch_size")
        pass_gen = 1  # prefill pass generates a single token
    elif which == "decode":
        trace_path = result.get("decode_trace_file") or result.get("trace_file")
        pass_tag = "_decode"
        pass_bs = result.get("decode_batch_size", result.get("batch_size", 1))
        pass_gen = result.get("max_tokens", "")
    else:
        trace_path = result.get("trace_file")
        # Tag the filename only when the run actually has distinct passes.
        pass_tag = "_decode" if result.get("two_pass") else ""
        pass_bs = result.get("decode_batch_size", result.get("batch_size", 1))
        pass_gen = result.get("max_tokens", "")

    if not trace_path or not os.path.isfile(trace_path):
        return jsonify({"ok": False, "error": "Trace file not found"}), 404

    # Build a descriptive filename encoding the profiled configuration:
    #   vllm_trace_{model}_{mode}[_prefill|_decode]_ctx{context}_in{query}_out{gen}_bs{bs}_tp{tp}_{n}layers.json.gz
    # where "ctx" is the block-aligned prefix-cache context the prefill attends
    # to, "in" is the query length (new prefill tokens, S), "out" is the number
    # of generated decode tokens, and "bs" is the pass's batch. The model id is
    # reduced to its final path component (org prefix dropped).
    model_short = result["model_id"].split("/")[-1]
    mode = result.get("mode", "eager")
    bs = pass_bs if pass_bs is not None else result.get("batch_size", 1)
    ctx = result.get("context_len_aligned") or result.get("context_len") or 0
    qin = result.get("query_len") or 0
    gen = pass_gen
    tp = result.get("tp_size", 1) or 1
    quant = result.get("quantization")
    layers = result.get("profiled_layers", "all")
    device = _DEVICE.upper()
    ext = ".json.gz" if trace_path.endswith(".gz") else ".json"
    quant_part = f"_{quant}" if quant else ""
    download_name = (
        f"vllm_trace_{model_short}_{device}_{mode}{pass_tag}_ctx{ctx}_in{qin}"
        f"_out{gen}_bs{bs}_tp{tp}{quant_part}_{layers}layers{ext}"
    )

    return send_file(
        trace_path,
        mimetype="application/gzip" if ext == ".json.gz" else "application/json",
        as_attachment=True,
        download_name=download_name,
    )


from breakdown.shape_derive import (  # noqa: F401  (re-exported for tests)
    _MAX_MATRIX_ROWS,
    _WEIGHT_ROLES,
    _bytes_to_dtype,
    _config_symbols,
    _flatten_graph_nodes,
    _format_op_shape_with_dtypes,
    _partially_resolve_dim,
    _profile_op_memory,
    _resolve_shape_ints,
    _validate_derived_shapes,
)


def _norm_quant(q: object) -> str | None:
    """Normalize a quantization selection: "", "auto", "none" → None."""
    if not q or str(q).lower() in ("auto", "none"):
        return None
    return str(q).lower()


def _profile_template_for(model_id: str, quantization: object = None
                          ) -> tuple[dict, dict | None, str | None]:
    """The latest completed profile graph, validated against a request.

    Returns ``(template, profile_settings, error)``; ``error`` is a
    user-facing message when the state cannot serve this model/quantization.
    Shared by the Shape Matrix export and the ``/api/perf/*`` pipeline.
    """
    with _profile_lock:
        state_status = _profile_state["status"]
        state_model = _profile_state.get("model_id")
        state_result = _profile_state.get("result")
        profile_settings = _profile_state.get("settings")
    if state_status != "done" or not state_result:
        return {}, None, (
            "The Shape Matrix is derived from a profiling run, but no "
            "completed run is available. Run a profile first.")
    template = state_result.get("graph")
    if not template or not (template.get("prefill") or template.get("decode")):
        return {}, None, ("The latest profile has no reconstructed graph to "
                          "derive shapes from.")
    if state_model and state_model != model_id:
        return {}, None, (f"Latest profile is for '{state_model}', not "
                          f"'{model_id}'. Profile that model or switch the "
                          "model ID.")

    # The derived shapes/dtypes/memory are only valid for the quantization the
    # run actually used, so the requested quantization must match the profiled
    # one.
    requested_quant = _norm_quant(quantization)
    profiled_quant = _norm_quant(
        (profile_settings or {}).get("quantization")
        if profile_settings else
        template.get("config", {}).get("quantization")
    )
    if requested_quant != profiled_quant:
        return {}, None, (
            f"Latest profile used quantization '{profiled_quant or 'none'}', "
            f"not '{requested_quant or 'none'}'. Re-profile with the requested "
            "quantization or change the selection.")
    return template, profile_settings, None

@app.route("/api/export/shape-matrix", methods=["POST"])
def export_shape_matrix():
    """Export op shapes/dtypes across configurations, grounded in a profiling run.

    Produces a flat Excel table where each row is one
    (Phase, SeqLen, CtxLen, BatchSize, TP, Op) combination.
    Columns for Phase/SeqLen/CtxLen/BatchSize/TP enable Excel filtering
    to select any desired configuration subset.

    Prefill configs: sweep seq_lens × context_lens × batch_sizes × tp_sizes
      (assumes chunked prefill; seq_len = chunk size)
    Decode configs: seq_len fixed to 1, sweep context_lens × batch_sizes × tp_sizes

    The op set + real shapes come from the latest completed profiling run for
    this model (its reconstructed graph is used as a symbolic template): each
    config re-resolves S/B/C/TP and recomputes Memory/FLOPs from the resolved
    shapes (using the recorded per-tensor dtypes). The op set is fixed at the
    profiled config (TP collectives, MoE routing, chunked-prefill splits), so the
    caller profiles at each TP it needs; S/B/C are parametric from one profile.
    The frontend ensures a matching profile exists (reusing the latest run or
    launching a fresh one) before calling this endpoint.

    The rows themselves are built by :mod:`breakdown.shape_matrix`; this
    endpoint only validates the request and serializes. ``/api/perf/*`` consumes
    the same rows in-process, without going through Excel.
    """
    data = request.json or {}

    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"ok": False, "error": "No model_id specified"}), 400

    sweep = {k: data.get(k, v) for k, v in shape_matrix.DEFAULT_SWEEP.items()}

    # Validate inputs
    for key in ("prefill_seq_lens", "tp_sizes", "decode_ctx_lens",
                "decode_batch_sizes"):
        if not isinstance(sweep[key], list) or not sweep[key]:
            return jsonify({"ok": False,
                            "error": f"{key} must be a non-empty list"}), 400

    configs = shape_matrix.build_configs(**sweep)
    if not configs:
        return jsonify({"ok": False,
                        "error": "No configurations generated."}), 400

    template, profile_settings, err = _profile_template_for(model_id,
                                                            data.get("quantization"))
    if err:
        return jsonify({"ok": False, "error": err}), 400

    estimated_rows = shape_matrix.estimate_row_count(template, configs)
    if estimated_rows > shape_matrix.MAX_MATRIX_ROWS:
        return jsonify({
            "ok": False,
            "error": f"Too many rows ({estimated_rows}). Max is "
                     f"{shape_matrix.MAX_MATRIX_ROWS}. "
                     "Reduce seq_lens, batch_sizes, ctx_lens, or tp_sizes."
        }), 400

    rows = shape_matrix.build_rows(template, configs)
    info_rows = shape_matrix.build_info_rows(model_id, template,
                                             profile_settings)
    payload = shape_matrix_xlsx.write_workbook(
        rows, info_rows, shape_matrix_xlsx.sheet_name_for(model_id))

    model_name = model_id.replace("/", "_")
    # Tag with the quantization method, or the model's concrete activation dtype
    # (e.g. bf16/fp16) when the run is unquantized — never a bare "none".
    pcfg = template.get("config", {})
    quant_tag = pcfg.get("quantization") or _bytes_to_dtype(
        pcfg.get("dtype_bytes", 2)
    )
    filename = f"vllm_xpu_shape_matrix_{model_name}_{quant_tag}.xlsx"

    return Response(
        payload,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



# ---------------------------------------------------------------------------
# Benchmark pipeline (/api/bench/*): profile -> replay -> ranked targets
#
# The web layer only wraps breakdown.bench; every stage also runs headless via
# ``python -m breakdown.bench``. Benchmark runs are long, so they use the same
# async job + lock pattern as profiling.
# ---------------------------------------------------------------------------
_bench_state: dict[str, Any] = {
    "status": "idle",     # idle | running | done | error
    "run_id": None,
    "error": None,
    "ops": [],            # per-op progress
    "result": None,
}
_bench_lock = threading.Lock()


@app.route("/api/bench/plan", methods=["POST"])
def bench_plan():
    """Sweep the profiled graph into replay cases for a new benchmark run.

    The op set and the operands come from the profile, so this endpoint needs a
    completed profiling run for the requested model/quantization - the same
    precondition the Shape Matrix export has.
    """
    data = request.json or {}
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"ok": False, "error": "No model_id specified"}), 400

    template, _settings, err = _profile_template_for(model_id,
                                                     data.get("quantization"))
    if err:
        return jsonify({"ok": False, "error": err}), 400

    sweep = {k: data.get(k, v) for k, v in shape_matrix.DEFAULT_SWEEP.items()}
    tp_sizes = sweep.get("tp_sizes") or [1]
    configs = shape_matrix.build_configs(**sweep)
    if shape_matrix.estimate_row_count(template, configs) > \
            shape_matrix.MAX_MATRIX_ROWS:
        return jsonify({"ok": False, "error": "Sweep too large - reduce it"}), 400

    rows = shape_matrix.build_rows(template, configs)
    device = data.get("device") or bench_devices.detect_device()
    cases, coverage = bench_spec.build_cases(rows, device=device)

    run_id = data.get("run_id") or bench_store.make_run_id(
        model_id, int(tp_sizes[0]), device)
    paths = bench_store.run_paths(run_id).ensure()
    bench_runner.write_cases(cases, paths.cases)

    # Classify before benchmarking, so the plan can say what will *not* be
    # measured - and why - instead of the run silently omitting it.
    status: dict[str, Any] = {}
    for case in cases:
        if case.op not in status:
            st, detail = bench_resolve.classify(case.op, case.args)
            status[case.op] = {"status": st, "detail": detail,
                               "backend": case.backend}
    by_status: dict[str, list] = {}
    for op, st in status.items():
        by_status.setdefault(st["status"], []).append(op)
    coverage.update({"op_status": status, "ops_by_status": by_status,
                     "device": device, "sweep": configs})
    bench_store.write_json(paths.plan, coverage)
    bench_store.RunMeta(
        run_id=run_id, model_id=model_id, device=device, tp=int(tp_sizes[0]),
        device_name=bench_devices.device_name(device),
        sku=bench_devices.sku_for_device(bench_devices.device_name(device)),
        smoke=bool(data.get("smoke")),
        sweep={**sweep, "configs": len(configs)}).write(paths)

    return jsonify({"ok": True, "run_id": run_id, "rows": len(rows),
                    "cases": len(cases), "coverage": coverage})


def _run_bench(run_id: str, device: str, budget: float,
               ops: list[str] | None) -> None:
    paths = bench_store.run_paths(run_id)
    try:
        cases = [bench_spec.BenchCase.from_dict(c)
                 for c in json.load(open(paths.cases))]

        def progress(op_result):
            with _bench_lock:
                _bench_state["ops"].append(asdict(op_result))

        result = bench_runner.run(cases, paths, device, budget=budget, ops=ops,
                                  on_op=progress)
        with _bench_lock:
            _bench_state["status"] = "done"
            _bench_state["result"] = result.to_dict()
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.exception("bench run failed")
        with _bench_lock:
            _bench_state["status"] = "error"
            _bench_state["error"] = str(exc)


@app.route("/api/bench/run", methods=["POST"])
def bench_run():
    """Replay a run's cases. Non-blocking - poll /api/bench/status."""
    data = request.json or {}
    run_id = data.get("run_id")
    if not run_id:
        return jsonify({"ok": False, "error": "run_id is required"}), 400
    paths = bench_store.run_paths(run_id)
    if not os.path.isfile(paths.cases):
        return jsonify({"ok": False,
                        "error": f"no cases for run {run_id}"}), 400
    with _bench_lock:
        if _bench_state["status"] == "running":
            return jsonify({"ok": False,
                            "error": "A benchmark run is already in progress"}), 409
        _bench_state.update({"status": "running", "run_id": run_id,
                             "error": None, "ops": [], "result": None})

    meta = bench_store.read_meta(paths)
    thread = threading.Thread(
        target=_run_bench,
        args=(run_id, data.get("device") or meta.get("device")
              or bench_devices.detect_device(),
              float(data.get("budget", 0.5)), data.get("ops")),
        daemon=True)
    thread.start()
    return jsonify({"ok": True, "status": "running", "run_id": run_id})


@app.route("/api/bench/status")
def bench_status():
    with _bench_lock:
        return jsonify({"ok": True, **dict(_bench_state)})


@app.route("/api/bench/runs")
def bench_runs():
    return jsonify({"ok": True, "runs": bench_store.list_runs()})


@app.route("/api/bench/results")
def bench_results():
    """A run's measured cases, enriched with utilization and the traced time."""
    run_id = request.args.get("run_id")
    runs = bench_store.list_runs()
    run_id = run_id or (runs[0]["run_id"] if runs else None)
    if not run_id:
        return jsonify({"ok": False, "error": "no benchmark runs yet"}), 404
    paths = bench_store.run_paths(run_id)
    meta = bench_store.read_meta(paths)
    records = bench_store.read_results(paths.results)
    peaks = bench_devices.peaks(meta.get("sku") or bench_devices.DEFAULT_SKU)
    rich = bench_reports.enrich(records, peaks)
    return jsonify({"ok": True, "run_id": run_id,
                    "summary": bench_reports.summarize(rich),
                    "coverage": bench_reports.coverage(rich),
                    "records": rich})


@app.route("/api/bench/targets")
def bench_targets():
    """The ranked optimization targets of a run (recomputed with ?refresh=1)."""
    runs = bench_store.list_runs()
    run_id = request.args.get("run_id") or (runs[0]["run_id"] if runs else None)
    if not run_id:
        return jsonify({"ok": False, "error": "no benchmark runs yet"}), 404
    paths = bench_store.run_paths(run_id)
    meta = bench_store.read_meta(paths)
    if request.args.get("refresh") not in ("1", "true") and \
            os.path.isfile(paths.targets):
        with open(paths.targets) as fh:
            return jsonify({"ok": True, "run_id": run_id,
                            "targets": json.load(fh)})

    records = bench_store.read_results(paths.results)
    if not records:
        return jsonify({"ok": False,
                        "error": f"run {run_id} has no benchmark results"}), 400
    rc = bench_rank.RankConfig(
        target_util=float(request.args.get(
            "target_util", bench_rank.DEFAULT_TARGET_UTIL)),
        tp=meta.get("tp"), top=int(request.args.get("top", 0)),
        run_id=run_id,
        provenance={"run_id": run_id, "commits": meta.get("commits") or {}})
    try:
        doc = bench_rank.rank(records, rc)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    bench_store.write_json(paths.targets, doc)
    try:
        conn = bench_history.connect(
            bench_history.db_path(bench_store.bench_root()))
        bench_history.ingest(conn, meta or {"run_id": run_id}, records, doc)
    except Exception as exc:  # noqa: BLE001 - history is best-effort
        logger.warning("bench history ingest failed: %s", exc)
    return jsonify({"ok": True, "run_id": run_id, "targets": doc})


@app.route("/api/bench/report")
def bench_report():
    """Download a run's report workbook (built on demand)."""
    run_id = request.args.get("run_id")
    if not run_id:
        return jsonify({"ok": False, "error": "run_id is required"}), 400
    paths = bench_store.run_paths(run_id)
    records = bench_store.read_results(paths.results)
    if not records:
        return jsonify({"ok": False, "error": "run has no results"}), 400
    meta = bench_store.read_meta(paths)
    targets = None
    if os.path.isfile(paths.targets):
        with open(paths.targets) as fh:
            targets = json.load(fh)
    bench_reports.write_workbook(
        records, paths.report,
        bench_devices.peaks(meta.get("sku") or bench_devices.DEFAULT_SKU),
        targets)
    return send_file(paths.report, as_attachment=True,
                     download_name=f"{run_id}_bench.xlsx")


@app.route("/api/bench/history")
def bench_history_api():
    """Runs in the history db, or a per-shape diff of two runs."""
    conn = bench_history.connect(
        bench_history.db_path(bench_store.bench_root()))
    base, new = request.args.get("base"), request.args.get("new")
    if base and new:
        return jsonify({"ok": True, "base": base, "new": new,
                        "changes": bench_history.compare(
                            conn, base, new,
                            float(request.args.get("threshold", 0.10)))})
    return jsonify({"ok": True, "runs": bench_history.runs(conn)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Breakdown Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Starting vLLM Breakdown at http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=args.debug)
