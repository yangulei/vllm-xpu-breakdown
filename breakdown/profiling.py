# SPDX-License-Identifier: Apache-2.0
"""Profiling: run vLLM once, and turn the traces into a reconstructed graph.

This is everything between "the user pressed Profile" and "there is a graph":
building the engine for a reduced-layer, prefix-cached, exact-length run,
installing the capture-time hooks, choosing which of the TP ranks' traces to
read, and merging a two-pass (separate prefill/decode batch) run into one
result.

It lives here rather than in ``app.py`` because none of it is about HTTP: the
CLI, the tests and the fixture capture tool all need the same run, and the web
route is one caller among several.
"""
from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from breakdown import runs
from breakdown.model_info import (
    fetch_model_config, min_profile_layers, summarize_config)
from breakdown.op_breakdown import backend_totals, summarize_ops
from breakdown.trace import build_graph_from_trace
from breakdown.trace_common import _detect_device_via_torch

logger = logging.getLogger("vllm_xpu_breakdown")

#: The accelerator this host has. Cached at import: it cannot change while the
#: process runs.
DEVICE = _detect_device_via_torch() or "xpu"

#: The stage name the profile's runs are stored under (see :mod:`breakdown.runs`).
STAGE = "profile"

# ---- Config Cache ----
# Persists successfully loaded model configs to disk so they appear as suggestions.

_CONFIG_CACHE_DIR = Path(__file__).parent / "output" / "config_cache"


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


#: The current (or last) profiling run. A profile is the input to every later
#: stage, so it is also **persisted** to ``output/profile/<run_id>/state.json``
#: (:func:`save_state`): a server restart, or a second browser tab, used to lose
#: the run everything downstream is derived from, and the only way back was to
#: profile again - minutes on a real model.
_profile_state: dict[str, Any] = {
    "status": "idle",   # idle | running | done | error
    "result": None,
    "error": None,
    "model_id": None,
    "settings": None,
    "run_id": None,
}


_profile_lock = threading.Lock()


def state() -> dict[str, Any]:
    """The current profiling state (restored from disk on first use)."""
    with _profile_lock:
        if _profile_state["status"] == "idle":
            _restore_latest()
        return _profile_state


def begin(model_id: str, settings: dict[str, Any]) -> str:
    """Mark a run as started and return its id."""
    run_id = runs.new_run_id(model_id.split("/")[-1])
    with _profile_lock:
        _profile_state.update({"status": "running", "result": None,
                               "error": None, "model_id": model_id,
                               "settings": settings, "run_id": run_id})
    return run_id


def save_state() -> None:
    """Persist the current state, so it outlives this process."""
    run_id = _profile_state.get("run_id")
    if not run_id:
        return
    try:
        runs.write_state(STAGE, run_id, _profile_state)
    except OSError:
        logger.warning("could not persist the profile run", exc_info=True)


def _restore_latest() -> None:
    """Adopt the newest completed run on disk, if there is one."""
    found = runs.latest_state(STAGE)
    if not found:
        return
    run_id, saved = found
    if saved.get("status") != "done" or not saved.get("result"):
        return
    _profile_state.update(saved)
    _profile_state["run_id"] = run_id
    logger.info("restored profile run %s (%s)", run_id, saved.get("model_id"))


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


def _scheduler_pin(prefill_batch: int, decode_batch: int,
                   query_len: int) -> dict[str, int]:
    """Engine settings that make every step run the *full* requested batch.

    Left to its defaults, vLLM's continuous-batching scheduler caps
    per-iteration concurrency (by ``max_num_seqs``, and by how many sequences'
    KV fits in cache) and runs an oversized batch in *partial-batch waves* - a
    batch of 32 dispatched as 29 + 3. Each wave has a different row count, so
    its ops neither symbolize to ``B`` nor merge with the full-batch ops, and
    the reconstructed decode graph grows duplicated ``29``/``3`` nodes.

    ``max_num_seqs`` admits the whole batch in one iteration;
    ``max_num_batched_tokens`` is sized to also admit a whole batch's prefill
    tokens in a single step (prefill pass: ``prefill_batch x query_len``;
    decode pass: ``decode_batch`` single-token prefills) so a full-shape step
    is never chunked. If the batch's KV does not fit device memory, raise
    ``gpu_memory_utilization`` or lower Context/Batch rather than letting the
    run silently split.
    """
    max_batch = max(int(prefill_batch), int(decode_batch))
    prefill_step_tokens = int(prefill_batch) * max(int(query_len), 1)
    return {
        "max_num_seqs": max_batch,
        "max_num_batched_tokens": max(prefill_step_tokens, max_batch, 2048),
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
    try:
        from vllm import LLM, SamplingParams, TokensPrompt

        # Fetch model config for analysis
        try:
            config = fetch_model_config(model_id)
            summary = summarize_config(config)
        except Exception:
            config = {}
            summary = {}

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

        # ``LLM.apply_model`` ships a *callable* to the worker process, which
        # vLLM refuses to serialize unless this is set — it raises
        # ``TypeError: Object of type <class 'function'> is not serializable``.
        # Both capture-time paths go through ``apply_model``: the ``module::``
        # span hooks (the primary source of real module names) and the
        # ``named_modules()`` reference tree. Without it BOTH fail, and the run
        # silently degrades to the heavy meta-device fallback with class-only
        # names. The callables we send are our own module-level functions, not
        # untrusted input. Must be set before the engine core process is
        # spawned, since the worker reads it at startup.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

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


        engine_kwargs.update(
            _scheduler_pin(pf_batch, dc_batch, int(query_len or 1)))

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
            tree with exact names. This is the *only* source of real module
            names, so a failure is logged loudly: the run still produces a
            graph, but every module falls back to a class heuristic.
            """
            try:
                from breakdown.module_hooks import install_module_span_hooks_on
                counts = llm.apply_model(install_module_span_hooks_on)
                total = sum(c for c in (counts or []) if isinstance(c, int))
                if not total:
                    logger.error(
                        "module span hooks: apply_model returned %r — no hooks "
                        "installed, module names will be class-only", counts)
                    return False
                logger.info("module span hooks: installed %d hooks across %d "
                            "worker(s)", total, len(counts or []))
            except Exception:
                logger.error("module span hooks: install FAILED; the trace will "
                             "carry no module:: spans and module names will fall "
                             "back to the name overlay", exc_info=True)
                return False
            # Kernel-launch spans: the operands of kernels launched straight
            # from Python (Triton, pybind11 extensions), which leave no cpu_op
            # and so no recorded shapes. Not fatal - without them such ops stay
            # shape-less and the replay benchmark reports them as uncovered.
            try:
                from breakdown.kernel_hooks import install_kernel_span_hooks_on
                kcounts = llm.apply_model(install_kernel_span_hooks_on)
                ktotal = sum(c for c in (kcounts or []) if isinstance(c, int))
                logger.info("kernel span hooks: installed %d hooks across %d "
                            "worker(s)", ktotal, len(kcounts or []))
            except Exception:
                logger.warning("kernel span hooks: install failed; "
                               "Python-launched kernels will have no recorded "
                               "operands", exc_info=True)
            return True

        def _remove_span_hooks(installed: bool) -> None:
            if not installed:
                return
            for mod_name, fn_name in (("breakdown.module_hooks",
                                       "remove_module_span_hooks_on"),
                                      ("breakdown.kernel_hooks",
                                       "remove_kernel_span_hooks_on")):
                try:
                    mod = importlib.import_module(mod_name)
                    llm.apply_model(getattr(mod, fn_name))
                except Exception:
                    logger.warning("%s: remove failed", mod_name, exc_info=True)

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
                tp_size=tp_size,
                batch_size=bsz,
                mode=mode,
                max_model_len=max_model_len,
                max_tokens=max_tokens,
                quantization=quantization,
                profiled_layers=profiled_layers,
                actual_layers=actual_layers,
                layer_scale=layer_scale,
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
        save_state()

    except Exception:
        with _profile_lock:
            _profile_state["status"] = "error"
            _profile_state["error"] = traceback.format_exc()
        save_state()
    finally:
        os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


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


def is_running() -> bool:
    with _profile_lock:
        return _profile_state["status"] == "running"


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

    with _profile_lock:
        _profile_state.clear()
        _profile_state.update({
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
        with _profile_lock:
            _profile_state["status"] = "done"
            _profile_state["result"] = result
            _profile_state["error"] = None
    except Exception:
        err = traceback.format_exc()
        with _profile_lock:
            _profile_state["status"] = "error"
            _profile_state["error"] = err
        save_state()
        return False, err
    save_state()
    return True, ""