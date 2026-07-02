#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
vLLM-XPU Ops/Kernels Breakdown — Web Application.

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
import os
import sys
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakdown.analyzer import AnalyzedOp, analyze_ops
from breakdown.classifier import Backend, classify_op
from breakdown.graph_from_trace import build_graph_from_trace
from breakdown.model_graph import (
    build_model_graph,
    min_profile_layers,
)
from breakdown.model_info import fetch_model_config, get_dim_symbols, summarize_config
from breakdown.registry import ALL_VLLM_XPU_OPS

app = Flask(__name__, static_folder="static")

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
) -> dict:
    """Parse one or more trace files and build the profile result dict.

    Shared by the live profiler (``_run_profile``) and the trace-upload
    endpoint so both paths reconstruct the model graph and op breakdown the
    same way. ``rank_files`` is rank-0 first; with TP>1 the remaining ranks are
    used only to average device time.
    """
    from breakdown.trace_parser import parse_trace_file

    op_dicts = parse_trace_file(rank_files[0])

    # If multi-rank, average device times across ranks
    if tp_size > 1 and len(rank_files) > 1:
        for extra_file in rank_files[1:]:
            extra_ops = parse_trace_file(extra_file)
            extra_timing = {
                (o["name"], o.get("input_shapes", "")): o.get("device_time_us", 0)
                for o in extra_ops
            }
            for op in op_dicts:
                key = (op["name"], op.get("input_shapes", ""))
                op["device_time_us"] = (
                    op.get("device_time_us", 0) + extra_timing.get(key, 0)
                )
        for op in op_dicts:
            op["device_time_us"] = op.get("device_time_us", 0) / tp_size

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
        )
        graph["profiled_layers"] = profiled_layers
        graph["actual_layers"] = actual_layers
        graph["layer_scale"] = layer_scale
        profile_result["graph"] = graph
    except Exception:
        pass  # Graph reconstruction is best-effort

    return profile_result


def _run_profile(model_id: str, mode: str, max_model_len: int,
                 batch_size: int, max_tokens: int, prompt: str,
                 num_profile_layers: int | None = None,
                 tp_size: int = 1,
                 quantization: str | None = None,
                 gpu_memory_utilization: float | None = None):
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
    """
    global _profile_state
    try:
        from vllm import LLM, SamplingParams

        # Fetch model config for analysis
        try:
            config = fetch_model_config(model_id)
            summary = summarize_config(config)
            dim_symbols = get_dim_symbols(summary)
        except Exception:
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

        # Quantization method
        if quantization:
            engine_kwargs["quantization"] = quantization

        # Override layer count for reduced-layer profiling.
        if profiled_layers < actual_layers:
            # Some multimodal models (e.g. MiniMax-M3) nest the decoder layer
            # count under ``text_config``. A top-level ``num_hidden_layers``
            # override is silently ignored there, so the full model is built
            # and exhausts device memory (UR_RESULT_ERROR_DEVICE_LOST). Pass a
            # callable override (vLLM applies callables in place, preserving the
            # rest of the config) that sets the count where it actually lives.
            # Must be a module-level partial so it pickles for the spawned
            # EngineCore subprocess.
            engine_kwargs["hf_overrides"] = functools.partial(
                _set_num_hidden_layers, n=profiled_layers
            )

        # Set compile / eager mode
        if mode == "compile":
            os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"
            engine_kwargs["enforce_eager"] = False
        else:
            os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)
            engine_kwargs["enforce_eager"] = True

        llm = LLM(**engine_kwargs)

        sampling_params = SamplingParams(max_tokens=max_tokens)

        # Use chat() if model supports it, else fall back to generate().
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        conversations = [conversation] * batch_size
        prompts = [prompt] * batch_size

        # First warmup pass also detects chat-template support.
        try:
            llm.chat(conversations, sampling_params, use_tqdm=False)
            use_chat = True
        except Exception:
            # Model may not have a chat template — use raw generate
            llm.generate(prompts, sampling_params, use_tqdm=False)
            use_chat = False

        def _run_inference():
            if use_chat:
                llm.chat(conversations, sampling_params, use_tqdm=False)
            else:
                llm.generate(prompts, sampling_params, use_tqdm=False)

        # Warm up before profiling. Warmup primes Triton JIT compilation /
        # kernel autotuning so the profiled trace reflects steady-state timing,
        # not one-time compilation overhead (important for fused/sparse-kernel
        # models such as MiniMax-M3). The detection pass above is warmup #1;
        # run 2 more for 3 warmups total.
        for _ in range(2):
            _run_inference()

        # --- Profiled run ---
        llm.start_profile()
        _run_inference()
        llm.stop_profile()

        # --- Parse trace files ---
        # With TP>1, vLLM produces one trace file per rank.
        # We parse rank-0 trace as the canonical ops (all ranks have the
        # same op sequence with identical per-rank shapes).
        trace_files = sorted(
            [os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
             if f.endswith(".json") or f.endswith(".json.gz")],
            key=os.path.getmtime, reverse=True,
        )

        if not trace_files:
            raise RuntimeError(
                f"No trace files found in {trace_dir}. "
                "Profiling may have failed in the worker process."
            )

        # With TP>1, take the tp_size most recent files (one per rank).
        # Parse rank-0 (most recent) for ops; average timing across all ranks.
        rank_files = trace_files[:tp_size]

        profile_result = _build_result_from_traces(
            rank_files,
            model_id=model_id,
            summary=summary,
            dim_symbols=dim_symbols,
            tp_size=tp_size,
            batch_size=batch_size,
            mode=mode,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            quantization=quantization,
            profiled_layers=profiled_layers,
            actual_layers=actual_layers,
            layer_scale=layer_scale,
        )

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

    with _profile_lock:
        _profile_state = {
            "status": "running",
            "result": None,
            "error": None,
            "model_id": model_id,
            "settings": {
                "mode": mode,
                "batch_size": batch_size,
                "max_model_len": max_model_len,
                "tp_size": tp_size,
                "quantization": quantization,
            },
        }

    thread = threading.Thread(
        target=_run_profile,
        args=(model_id, mode, max_model_len, batch_size, max_tokens, prompt,
              num_profile_layers, tp_size, quantization, gpu_memory_utilization),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "status": "running"})


@app.route("/api/profile/upload", methods=["POST"])
def upload_profile():
    """Reconstruct the model graph and op breakdown from uploaded trace(s).

    Accepts a multipart form with one or more ``trace`` files (a torch profiler
    Chrome trace, ``.json`` or ``.json.gz``; with TP>1 upload one file per rank,
    rank-0 first) plus optional form fields:

      - ``model_id``: HF id used to fetch config for shape symbols / summary
      - ``tensor_parallel_size`` / ``tp_size``: ranks represented by the uploads
      - ``batch_size``, ``quantization``, ``mode``
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
    mode = form.get("mode", "eager")
    tp_size = int(form.get("tensor_parallel_size") or form.get("tp_size") or 1)
    batch_size = int(form.get("batch_size") or 1)
    quantization = form.get("quantization") or None
    if quantization in ("", "auto", "none"):
        quantization = None

    # Persist uploads under output/traces so the trace-download endpoint works.
    from werkzeug.utils import secure_filename
    trace_dir = os.path.abspath("output/traces")
    os.makedirs(trace_dir, exist_ok=True)
    saved: list[str] = []
    for f in files:
        name = secure_filename(f.filename) or "uploaded_trace.json"
        dest = os.path.join(trace_dir, name)
        f.save(dest)
        saved.append(dest)

    # Fetch model config for shape symbols / summary (best-effort).
    try:
        summary = summarize_config(fetch_model_config(model_id)) if model_id else {}
        dim_symbols = get_dim_symbols(summary) if summary else {}
    except Exception:
        summary = {}
        dim_symbols = {}

    actual_layers = form.get("actual_layers") or summary.get("num_layers")
    actual_layers = int(actual_layers) if actual_layers else None
    profiled_layers = form.get("num_profile_layers") or actual_layers
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
                "tp_size": tp_size,
                "quantization": quantization,
                "uploaded": True,
            },
        }

    # Optional: reconstruct real module attribute names by instantiating the
    # model on ``meta`` (no weights). Heavy + network-dependent, so it is
    # opt-in via env var. Without it, uploaded traces keep heuristic names.
    ref_module_tree = None
    if model_id and os.environ.get("VLLM_XPU_BREAKDOWN_META_NAMES") == "1":
        try:
            from breakdown.module_naming import ref_tree_from_config
            ref_module_tree = ref_tree_from_config(
                fetch_model_config(model_id),
                dtype=summary.get("dtype", "bfloat16"),
                model_id=model_id,
                allow_remote_code=(
                    os.environ.get("VLLM_XPU_BREAKDOWN_TRUST_REMOTE_CODE") == "1"
                ),
            )
        except Exception:
            ref_module_tree = None

    try:
        result = _build_result_from_traces(
            saved[:tp_size] if len(saved) >= tp_size else saved,
            model_id=model_id,
            summary=summary,
            dim_symbols=dim_symbols,
            tp_size=tp_size,
            batch_size=batch_size,
            mode=mode,
            quantization=quantization,
            profiled_layers=profiled_layers,
            actual_layers=actual_layers,
            layer_scale=layer_scale,
            ref_module_tree=ref_module_tree,
        )
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
    # Don't expose internal trace_file path; indicate availability
    client_result = {k: v for k, v in result.items() if k != "trace_file"}
    client_result["has_trace"] = bool(result.get("trace_file"))
    return jsonify({"ok": True, "data": client_result})


@app.route("/api/profile/trace")
def download_trace():
    """Download the profiled trace file with a descriptive filename."""
    with _profile_lock:
        if _profile_state["status"] != "done" or not _profile_state.get("result"):
            return jsonify({"ok": False, "error": "No profile result available"}), 404
        result = _profile_state["result"]

    trace_path = result.get("trace_file")
    if not trace_path or not os.path.isfile(trace_path):
        return jsonify({"ok": False, "error": "Trace file not found"}), 404

    # Build a descriptive filename. A profiling run captures both the prefill
    # (prompt) and decode (generation) phases in one trace, so label it
    # "prefill+decode". Encode the engine max_model_len as "maxlen" (it is the
    # KV budget, not the processed context length — labeling it "ctx" was
    # misleading) and the generated token count as "gen".
    # vllm_trace_{model}_{mode}_prefill+decode_bs{bs}_maxlen{n}_gen{n}_tp{tp}_{n}layers.json.gz
    model_short = result["model_id"].replace("/", "_")
    mode = result.get("mode", "eager")
    bs = result.get("batch_size", 1)
    maxlen = result.get("max_model_len", "")
    gen = result.get("max_tokens", "")
    tp = result.get("tp_size", 1) or 1
    quant = result.get("quantization")
    layers = result.get("profiled_layers", "all")
    ext = ".json.gz" if trace_path.endswith(".gz") else ".json"
    quant_part = f"_{quant}" if quant else ""
    gen_part = f"_gen{gen}" if gen else ""
    download_name = (
        f"vllm_trace_{model_short}_{mode}_prefill+decode_bs{bs}"
        f"_maxlen{maxlen}{gen_part}_tp{tp}{quant_part}_{layers}layers{ext}"
    )

    return send_file(
        trace_path,
        mimetype="application/gzip" if ext == ".json.gz" else "application/json",
        as_attachment=True,
        download_name=download_name,
    )


# Roles whose 2nd input tensor (index 1) is a weight matrix
_WEIGHT_ROLES = {
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
    "expert_gate_up", "expert_down",
    "shared_expert_gate_up", "shared_expert_down",
    "q_compress", "q_decompress", "kv_compress",
    "router_gate", "lm_head", "vl_projector",
    "vit_qkv_proj", "vit_o_proj", "vit_mlp_up", "vit_mlp_down",
    "patch_embed", "mlp_up", "mlp_down",
}


def _bytes_to_dtype(nbytes: int) -> str:
    """Convert byte count to short dtype name."""
    return {4: "fp32", 2: "bf16", 1: "fp8"}.get(nbytes, f"{nbytes * 8}bit")


def _quant_dtype_name(quant: str | None) -> str | None:
    """Map quantization method to weight dtype short name."""
    if not quant or quant == "None":
        return None
    names = {
        "fp8": "fp8", "gptq": "int4", "gptq_marlin": "int4",
        "awq": "int4", "awq_marlin": "int4", "marlin": "int4",
        "squeezellm": "int4", "bitsandbytes": "nf4", "gguf": "q4",
        "int4": "int4", "int8": "int8",
    }
    return names.get(quant.lower(), quant.lower())


def _get_tensor_dtype(tensor_idx: int, role: str,
                      graph_cfg: dict) -> str:
    """Get the dtype label for a specific tensor in an op.

    Mirrors the frontend getOpDtypes logic: weight tensors (index 1) of
    projection ops use the weight dtype; everything else uses activation dtype.
    """
    act_dtype = _bytes_to_dtype(graph_cfg.get("dtype_bytes", 2))
    quant = graph_cfg.get("quantization")
    if quant:
        w_dtype = _quant_dtype_name(quant) or _bytes_to_dtype(
            graph_cfg.get("weight_dtype_bytes", 2)
        )
    else:
        w_dtype = act_dtype

    if tensor_idx == 1 and role in _WEIGHT_ROLES:
        return w_dtype
    return act_dtype


def _flatten_graph_nodes(node: dict, depth: int = 0,
                         rows: list | None = None,
                         parent_repeat: int = 1) -> list[dict]:
    """Flatten a hierarchical graph tree into rows for the hierarchy sheet."""
    if rows is None:
        rows = []

    # Effective repeat = own repeat_count × parent's repeat
    own_repeat = node.get("repeat_count", 1)
    effective_repeat = own_repeat * parent_repeat

    # Add module row
    rows.append({
        "depth": depth,
        "name": node.get("name", ""),
        "path": node.get("path", ""),
        "module_type": node.get("module_type", ""),
        "repeat_count": own_repeat,
        "effective_repeat": effective_repeat,
        "total_memory": node.get("total_memory", 0),
        "total_flops": node.get("total_flops", 0),
        "total_ai": node.get("total_ai", 0),
        "ops": node.get("ops", []),
    })

    # Recurse into children
    for child in node.get("children", []):
        _flatten_graph_nodes(child, depth + 1, rows, effective_repeat)

    return rows


# ---- Shape Matrix Export (single model, multi-config sweep) ----

# Max total rows to prevent excessive memory/time
_MAX_MATRIX_ROWS = 50000


def _format_op_shape_with_dtypes(
    op: dict, symbols: dict[str, int], graph_cfg: dict
) -> str:
    """Format op shapes as concrete values with per-tensor dtypes.

    Example: "[128, 2560, bf16] × [2560, 6144, fp8]"
    Resolves symbolic dims (including composite like "S+C") via the symbols dict.
    """
    op_shapes = op.get("input_shapes", [])
    if not op_shapes:
        return "—"
    role = op.get("role", "")
    parts = []
    for shape_idx, shape in enumerate(op_shapes):
        if isinstance(shape, list):
            tensor_dtype = _get_tensor_dtype(shape_idx, role, graph_cfg)
            dims = []
            for dim in shape:
                dims.append(str(_resolve_dim(dim, symbols)))
            if tensor_dtype:
                dims.append(tensor_dtype)
            parts.append("[" + ", ".join(dims) + "]")
        else:
            parts.append(str(_resolve_dim(shape, symbols)))
    return " × ".join(parts)


def _safe_arithmetic_eval(expr: str) -> int:
    """Safely evaluate a simple arithmetic expression (integers, +, -, *, /).

    Only allows integer literals and the operators +, -, *, /.
    Division is performed as integer (floor) division.
    Raises ValueError for anything else.
    """
    import ast

    # Normalize "/" to "//" for integer division in eval
    expr = expr.replace("//", "/").replace("/", "//")

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        elif isinstance(node, (ast.Constant,)):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Non-numeric constant: {node.value}")
        elif not isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv,
                                   ast.USub, ast.UAdd)):
            raise ValueError(f"Unsupported node: {type(node).__name__}")
    return int(eval(compile(tree, "<dim>", "eval")))  # noqa: S307


def _resolve_dim(dim, symbols: dict[str, int]):
    """Resolve a dimension value to a concrete integer if possible."""
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        # Direct lookup
        if dim in symbols:
            return symbols[dim]
        # Try evaluating composite expressions like "S+C", "2·I"
        # Replace symbol names with their values and evaluate
        expr = dim
        # Sort by length descending to avoid partial replacements
        for name in sorted(symbols.keys(), key=len, reverse=True):
            expr = expr.replace(name, str(symbols[name]))
        # Replace middle-dot with *
        expr = expr.replace("·", "*")
        try:
            return _safe_arithmetic_eval(expr)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return dim
    return dim


# Config-dependent variable symbols that should stay symbolic
_VARIABLE_SYMS = {"S", "B", "C", "TP"}


def _is_variable_composite(expr: str) -> bool:
    """Check if an expression is composed entirely of variable symbols.

    Handles both additive (S+C) and multiplicative (B·S) composites.
    """
    # Split on + and · to get individual parts
    parts = expr.replace("·", "+").split("+")
    return all(p.strip() in _VARIABLE_SYMS for p in parts)


def _partially_resolve_dim(dim, symbols: dict[str, int],
                           full_symbols: dict[str, int] | None = None,
                           tp_divided: set[str] | None = None):
    """Resolve dim keeping only S/B/C/TP symbolic, resolving all else to numbers.

    Model constants from config.json are shown as numbers. When a dimension
    contains "/TP", it's shown as "value/TP" using the full undivided config
    value from the symbols dict.

    The full_symbols and tp_divided params are accepted for backwards
    compatibility but ignored — the graph now embeds /TP directly in shapes
    and symbols already contain original (undivided) values.
    """
    if isinstance(dim, (int, float)):
        return str(int(dim))

    s = str(dim)

    # Pure variable symbol → keep as-is
    if s in _VARIABLE_SYMS:
        return s

    # Composite of only variable symbols (e.g. "S+C", "B·S") → keep as-is
    if _is_variable_composite(s):
        return s

    # Handle "/TP" suffix: resolve the base part, keep /TP
    if s.endswith("/TP"):
        base = s[:-3]  # strip "/TP"
        resolved_base = _resolve_constant_expr(base, symbols)
        return f"{resolved_base}/TP"

    # Check if s is a known symbol directly (handles names with · like "n_h·D_qh")
    if s in symbols:
        return str(symbols[s])

    # Check for multiply composites containing a variable (e.g., "B·S·K")
    if "·" in s:
        parts = s.split("·")
        has_variable = any(p in _VARIABLE_SYMS for p in parts)
        if has_variable:
            # Partially resolve: keep variable parts, resolve constants
            resolved_parts = []
            for p in parts:
                if p in _VARIABLE_SYMS:
                    resolved_parts.append(p)
                elif p in symbols:
                    resolved_parts.append(str(symbols[p]))
                elif p.isdigit():
                    resolved_parts.append(p)
                else:
                    resolved_parts.append(p)
            return "·".join(resolved_parts)
        else:
            # All parts are constants — compute product
            product = 1
            for p in parts:
                val = symbols.get(p)
                if val is not None:
                    product *= val
                elif p.isdigit():
                    product *= int(p)
            return str(product)

    # Pure constant — fully resolve
    if s in symbols:
        return str(symbols[s])
    resolved = _resolve_dim(s, symbols)
    return str(resolved)


def _resolve_constant_expr(expr: str, symbols: dict[str, int]) -> str:
    """Resolve a constant expression (no variables) to its numeric value.

    Handles symbols like "QKV", "n_h·d", "2·I", and plain numbers.
    """
    if expr in symbols:
        return str(symbols[expr])
    if "·" in expr:
        parts = expr.split("·")
        product = 1
        for p in parts:
            val = symbols.get(p)
            if val is not None:
                product *= val
            elif p.isdigit():
                product *= int(p)
            else:
                return expr  # can't resolve
        return str(product)
    if expr.isdigit():
        return expr
    return expr


@app.route("/api/export/shape-matrix", methods=["POST"])
def export_shape_matrix():
    """Export op shapes/dtypes for the current model across configurations.

    Produces a flat Excel table where each row is one
    (Phase, SeqLen, CtxLen, BatchSize, TP, Op) combination.
    Columns for Phase/SeqLen/CtxLen/BatchSize/TP enable Excel filtering
    to select any desired configuration subset.

    Prefill configs: sweep seq_lens × context_lens × batch_sizes × tp_sizes
      (assumes chunked prefill; seq_len = chunk size)
    Decode configs: seq_len fixed to 1, sweep context_lens × batch_sizes × tp_sizes
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    data = request.json or {}

    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"ok": False, "error": "No model_id specified"}), 400

    # Prefill settings
    prefill_seq_lens = data.get("prefill_seq_lens",
                                [128, 256, 512, 1024, 2048, 4096, 8192])
    prefill_ctx_lens = data.get("prefill_ctx_lens", [0, 8192])
    prefill_batch_sizes = data.get("prefill_batch_sizes", [1])

    # Decode settings (seq_len always 1)
    decode_ctx_lens = data.get("decode_ctx_lens", [8192])
    decode_batch_sizes = data.get("decode_batch_sizes",
                                  [1, 2, 4, 8, 16, 32, 64, 128])

    # TP sizes
    tp_sizes = data.get("tp_sizes", [1, 2, 4, 8])

    # Quantization normalization
    quantization = data.get("quantization", None)
    if quantization == "auto":
        quantization = None
    elif quantization == "none":
        quantization = "none"

    # Validate inputs
    if not isinstance(prefill_seq_lens, list) or not prefill_seq_lens:
        return jsonify({"ok": False,
                        "error": "prefill_seq_lens must be a non-empty list"}), 400
    if not isinstance(tp_sizes, list) or not tp_sizes:
        return jsonify({"ok": False,
                        "error": "tp_sizes must be a non-empty list"}), 400
    if not isinstance(decode_ctx_lens, list) or not decode_ctx_lens:
        return jsonify({"ok": False,
                        "error": "decode_ctx_lens must be a non-empty list"}), 400
    if not isinstance(decode_batch_sizes, list) or not decode_batch_sizes:
        return jsonify({"ok": False,
                        "error": "decode_batch_sizes must be a non-empty list"}), 400

    # Build list of all configurations to sweep (always both phases)
    configs: list[dict] = []
    for seq in prefill_seq_lens:
        for ctx in prefill_ctx_lens:
            for bs in prefill_batch_sizes:
                for tp in tp_sizes:
                    configs.append({
                        "phase": "prefill",
                        "seq_len": seq,
                        "ctx_len": ctx,
                        "batch_size": bs,
                        "tp_size": tp,
                        "prefill_len": seq,
                        "decode_batch": bs,
                        "context_len": ctx,
                    })
    for ctx in decode_ctx_lens:
        for bs in decode_batch_sizes:
            for tp in tp_sizes:
                configs.append({
                    "phase": "decode",
                    "seq_len": 1,
                    "ctx_len": ctx,
                    "batch_size": bs,
                    "tp_size": tp,
                    "prefill_len": 1,
                    "decode_batch": bs,
                    "context_len": ctx,
                })

    if not configs:
        return jsonify({"ok": False,
                        "error": "No configurations generated."}), 400

    # Fetch model config
    try:
        config = fetch_model_config(model_id)
        summary = summarize_config(config)
    except Exception as e:
        return jsonify({"ok": False,
                        "error": f"Failed to fetch model config: {e}"}), 400

    # Estimate row count (configs × ~ops_per_config) for limit check
    # Use first config to count ops
    test_graph = build_model_graph(
        summary, prefill_len=1, decode_batch=1, context_len=1,
        tp_size=tp_sizes[0], quantization=quantization,
    )
    test_tree = test_graph.get("prefill") or test_graph.get("decode")
    test_ops_count = 0
    if test_tree:
        for node in _flatten_graph_nodes(test_tree):
            test_ops_count += len(node["ops"])
    estimated_rows = len(configs) * test_ops_count
    if estimated_rows > _MAX_MATRIX_ROWS:
        return jsonify({
            "ok": False,
            "error": f"Too many rows ({estimated_rows}). Max is {_MAX_MATRIX_ROWS}. "
                     "Reduce seq_lens, batch_sizes, ctx_lens, or tp_sizes."
        }), 400

    # Build flat table data
    wb = Workbook()

    header_font = Font(bold=True, size=10, color="FFFFFF")
    header_fill = PatternFill(start_color="1A1A2E", end_color="1A1A2E",
                              fill_type="solid")
    thin_border = Border(bottom=Side(style="thin", color="E0E0E0"))

    # Sheet name: use model short name (last part of model_id)
    model_short = model_id.split("/")[-1] if "/" in model_id else model_id
    # Sanitize for Excel sheet name (max 31 chars, no special chars)
    sheet_name = model_short[:31].replace("[", "").replace("]", "")

    ws = wb.active
    ws.title = sheet_name

    headers = [
        "Phase", "Seq Len", "Ctx Len", "Batch Size", "TP",
        "Module", "Op Name", "Backend", "Layers",
        "Symbolic Shape", "Shape",
        "Memory (bytes)", "FLOPs", "AI",
    ]
    for col, hdr in enumerate(headers, 1):
        c = ws.cell(1, col, hdr)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")

    row = 2
    for cfg in configs:
        graph = build_model_graph(
            summary,
            prefill_len=cfg["prefill_len"],
            decode_batch=cfg["decode_batch"],
            context_len=cfg["context_len"],
            tp_size=cfg["tp_size"],
            quantization=quantization,
        )
        graph_cfg = graph.get("config", {})
        symbols = graph.get("symbols", {})
        tree = graph.get(cfg["phase"])
        if not tree:
            continue

        flat_nodes = _flatten_graph_nodes(tree)
        for node_info in flat_nodes:
            effective_repeat = node_info.get("effective_repeat", 1)
            for op in node_info["ops"]:
                shape_str = _format_op_shape_with_dtypes(op, symbols, graph_cfg)

                # Symbolic shape: keep only config variables (S, B, C, TP)
                # symbolic, resolve model constants to config.json numbers
                sym_shapes = op.get("input_shapes", [])
                if sym_shapes:
                    sym_parts = []
                    for s in sym_shapes:
                        if isinstance(s, list):
                            dims = [_partially_resolve_dim(d, symbols)
                                    for d in s]
                            sym_parts.append("[" + ", ".join(dims) + "]")
                        else:
                            sym_parts.append(
                                _partially_resolve_dim(s, symbols))
                    symbolic_str = " × ".join(sym_parts)
                else:
                    symbolic_str = "—"

                mem_bytes = op.get("memory_bytes", 0)
                flops = op.get("flops", 0)
                ai = round(flops / mem_bytes, 2) if mem_bytes > 0 else 0

                # Merge module path and op role into single column
                path = node_info["path"]
                role = op.get("role", "")
                module_col = f"{path}.{role}" if role else path

                ws.cell(row, 1, cfg["phase"])
                ws.cell(row, 2, cfg["seq_len"])
                ws.cell(row, 3, cfg["ctx_len"])
                ws.cell(row, 4, cfg["batch_size"])
                ws.cell(row, 5, cfg["tp_size"])
                ws.cell(row, 6, module_col)
                ws.cell(row, 7, op.get("name", ""))
                ws.cell(row, 8, op.get("backend", ""))
                ws.cell(row, 9, effective_repeat)
                ws.cell(row, 10, symbolic_str)
                ws.cell(row, 11, shape_str)
                ws.cell(row, 12, mem_bytes)
                ws.cell(row, 13, flops)
                ws.cell(row, 14, ai)

                for c in range(1, len(headers) + 1):
                    ws.cell(row, c).border = thin_border
                row += 1

    # AutoFit column widths by sampling header + first/last 100 data rows
    sample_rows = list(range(1, min(row, 102)))  # header + first 100
    if row > 202:
        sample_rows += list(range(row - 100, row))  # last 100
    elif row > 102:
        sample_rows += list(range(102, row))
    for col_idx in range(1, len(headers) + 1):
        max_len = 0
        col_letter = get_column_letter(col_idx)
        for r in sample_rows:
            val = ws.cell(r, col_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = min(max_len + 2, 80)

    ws.freeze_panes = "A2"
    if row > 2:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{row - 1}"

    # Write to buffer
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    model_name = model_id.replace("/", "_")
    # Resolve effective quantization for filename (align with what model actually uses)
    effective_quant = quantization
    if not effective_quant or effective_quant == "none":
        effective_quant = summary.get("quant_method")
    quant_tag = effective_quant if effective_quant else "none"
    filename = f"vllm_xpu_shape_matrix_{model_name}_{quant_tag}.xlsx"

    return Response(
        buf.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM-XPU Breakdown Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Starting vLLM-XPU Breakdown at http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=args.debug)
