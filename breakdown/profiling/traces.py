# SPDX-License-Identifier: Apache-2.0
"""From trace files on disk to one reconstructed result.

Three things live here, and they are all about *which bytes to read and what
they mean*: choosing rank 0's trace out of a tensor-parallel run's N, encoding
and decoding the descriptive filename a download is served under, and merging
a two-pass (separate prefill/decode batch) run into a single result.

The filename builder and its parser are deliberately adjacent: together they
make download-then-upload a lossless round trip, so a change to one that is
not mirrored in the other silently breaks re-importing your own trace.
"""
from __future__ import annotations

import logging
import os
import re

from breakdown.op_breakdown import backend_totals, summarize_ops
from breakdown.trace import build_graph_from_trace

logger = logging.getLogger("vllm_xpu_breakdown")

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
_RANK_NAME_RE = re.compile(r"rank[-_]?(?P<rank>\d+)", re.IGNORECASE)


_TP_RANK_NAME_RE = re.compile(r"(?:^|[_/])tp(?P<rank>\d+)", re.IGNORECASE)


def _trace_rank(path: str) -> int | None:
    """Extract the tensor-parallel rank index from a raw trace filename.

    vLLM writes one trace per rank. Two naming forms occur in the wild:
    ``…_tp<N>_…_rank<N>.<id>.pt.trace.json.gz`` and
    ``<id>-rank-<N>.<id>.pt.trace.json.gz`` — hence the optional ``[-_]``
    separator. Returns the rank as an int (``rank<N>`` preferred, ``tp<N>``
    fallback), or ``None`` when no rank marker is present (e.g. a merged or
    descriptive name).
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

def trace_download_name(result: dict, which: str, device: str,
                        trace_path: str) -> str:
    """The descriptive filename a downloaded trace is served under.

    Every profiled parameter is in the name, which is what makes
    download -> upload a lossless round trip: :func:`_parse_trace_filename` is
    the exact inverse, and the two must move together. They now sit together
    too -- the builder used to be thirty lines inside the Flask route, where a
    change to it would not obviously break the parser.

    ``ctx`` is the block-aligned prefix-cache context the prefill attends to,
    ``in`` the query length (new prefill tokens, S), ``out`` the number of
    generated decode tokens, ``bs`` the pass's batch. The model id is reduced
    to its final path component.
    """
    if which == "prefill":
        pass_tag, pass_bs, gen = "_prefill", result.get("prefill_batch_size"), 1
    elif which == "decode":
        pass_tag = "_decode"
        pass_bs = result.get("decode_batch_size", result.get("batch_size", 1))
        gen = result.get("max_tokens", "")
    else:
        # Tag the filename only when the run really has distinct passes.
        pass_tag = "_decode" if result.get("two_pass") else ""
        pass_bs = result.get("decode_batch_size", result.get("batch_size", 1))
        gen = result.get("max_tokens", "")

    bs = pass_bs if pass_bs is not None else result.get("batch_size", 1)
    quant = result.get("quantization")
    return (
        f"vllm_trace_{result['model_id'].split('/')[-1]}_{device.upper()}"
        f"_{result.get('mode', 'eager')}{pass_tag}"
        f"_ctx{result.get('context_len_aligned') or result.get('context_len') or 0}"
        f"_in{result.get('query_len') or 0}_out{gen}_bs{bs}"
        f"_tp{result.get('tp_size', 1) or 1}"
        f"{f'_{quant}' if quant else ''}"
        f"_{result.get('profiled_layers', 'all')}layers"
        f"{'.json.gz' if trace_path.endswith('.gz') else '.json'}")


def trace_path_for(result: dict, which: str) -> str | None:
    """The stored trace file a download request refers to."""
    if which == "prefill":
        return result.get("prefill_trace_file")
    if which == "decode":
        return result.get("decode_trace_file") or result.get("trace_file")
    return result.get("trace_file")


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
    rank_files = _rank0_first(rank_files)

    profile_result = {
        "model_id": model_id,
        "mode": mode,
        "batch_size": batch_size,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "tp_size": tp_size,
        "quantization": quantization,
        "summary": summary,
        "profiled_layers": profiled_layers,
        "actual_layers": actual_layers,
        "layer_scale": layer_scale,
        "trace_file": trace_file if trace_file is not None else rank_files[0],
    }

    # Reconstruct the model graph directly from the profiler trace. This is the
    # single source of truth: the flat op breakdown below is an aggregation of
    # it, not a second parse of the trace, so the table and the tree can never
    # disagree. A failure here is fatal — a result without a graph has nothing
    # in it.
    graph = build_graph_from_trace(
        rank_files[0],
        summary=summary,
        tp_size=tp_size,
        batch_size=batch_size,
        quantization=quantization,
        query_len=query_len,
        context_len=context_len,
    )
    graph["profiled_layers"] = profiled_layers
    graph["actual_layers"] = actual_layers
    graph["layer_scale"] = layer_scale
    profile_result["graph"] = graph

    ops = summarize_ops(graph)
    profile_result["ops"] = ops
    profile_result["backends"] = backend_totals(ops)
    profile_result["total_device_time_us"] = round(
        sum(o["device_time_us"] for o in ops), 2)

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
