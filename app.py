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
import io
import json
import os
import sys
import threading
import traceback
from dataclasses import asdict
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakdown.analyzer import AnalyzedOp, analyze_ops
from breakdown.classifier import Backend, classify_op
from breakdown.model_graph import (
    annotate_graph_from_modules,
    annotate_graph_timing,
    build_model_graph,
    min_profile_layers,
)
from breakdown.model_info import fetch_model_config, get_dim_symbols, summarize_config
from breakdown.registry import ALL_VLLM_XPU_OPS

app = Flask(__name__, static_folder="static")

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


# ---- Model Catalog API ----

from breakdown.model_catalog import (
    CATALOG,
    catalog_summary,
    get_model,
    get_models_by_type,
    get_vllm_models,
)


@app.route("/api/catalog")
def get_catalog():
    """Return the full model catalog, optionally filtered.

    Query params:
        type   – filter by model type (LLM, MLLM, T2I, T2V, Audio, etc.)
        priority – filter by priority (H, M, L)
        vllm   – if "true", only show vLLM-compatible models
    """
    model_type = request.args.get("type")
    priority = request.args.get("priority")
    vllm_only = request.args.get("vllm", "").lower() == "true"

    if vllm_only:
        models = get_vllm_models()
    elif model_type:
        models = get_models_by_type(model_type)
    else:
        models = list(CATALOG)

    if priority:
        models = [m for m in models if m.priority == priority]

    return jsonify({
        "ok": True,
        "models": [
            {
                "name": m.name,
                "hf_id": m.hf_id,
                "model_type": m.model_type,
                "precision": m.precision,
                "owner": m.owner,
                "focus": m.focus,
                "priority": m.priority,
                "in_cri_plan": m.in_cri_plan,
                "status": m.status,
                "vllm_supported": m.vllm_supported,
            }
            for m in models
        ],
        "summary": catalog_summary(),
    })


@app.route("/api/catalog/<name>")
def get_catalog_model(name: str):
    """Return details for a specific catalog model by name."""
    model = get_model(name)
    if not model:
        return jsonify({"ok": False, "error": f"Model '{name}' not found in catalog"}), 404
    return jsonify({
        "ok": True,
        "model": {
            "name": model.name,
            "hf_id": model.hf_id,
            "model_type": model.model_type,
            "precision": model.precision,
            "owner": model.owner,
            "focus": model.focus,
            "priority": model.priority,
            "in_cri_plan": model.in_cri_plan,
            "status": model.status,
            "vllm_supported": model.vllm_supported,
        },
    })


# ---- Model API ----

@app.route("/api/model/<path:model_id>")
def get_model_config(model_id: str):
    """Fetch config.json from HuggingFace and return summary."""
    try:
        config = fetch_model_config(model_id)
        summary = summarize_config(config)
        return jsonify({"ok": True, "config": config, "summary": summary})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/model/<path:model_id>/graph")
def get_model_graph(model_id: str):
    """Build static model graph (no profiling needed)."""
    try:
        config = fetch_model_config(model_id)
        summary = summarize_config(config)

        prefill_len = request.args.get("prefill_len", 128, type=int)
        decode_batch = request.args.get("decode_batch", 1, type=int)
        context_len = request.args.get("context_len", 4096, type=int)
        tp_size = request.args.get("tp_size", 1, type=int)
        quantization = request.args.get("quantization", None, type=str)
        # "auto" = use model's built-in quant config; "none" = force no quantization
        if quantization == "auto":
            quantization = None  # let build_model_graph read from model summary
        elif quantization == "none":
            quantization = "none"  # explicit override to disable quant

        graph = build_model_graph(summary,
                                  prefill_len=prefill_len,
                                  decode_batch=decode_batch,
                                  context_len=context_len,
                                  tp_size=tp_size,
                                  quantization=quantization)
        min_layers = min_profile_layers(summary)
        return jsonify({
            "ok": True, "graph": graph, "summary": summary,
            "min_profile_layers": min_layers,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ---- Profile API ----

def _run_profile(model_id: str, mode: str, max_model_len: int,
                 batch_size: int, max_tokens: int, prompt: str,
                 num_profile_layers: int | None = None,
                 tp_size: int = 1,
                 quantization: str | None = None):
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
    """
    global _profile_state
    try:
        from vllm import LLM, SamplingParams

        from breakdown.trace_parser import parse_trace_file, parse_trace_with_modules

        # Fetch model config for analysis
        try:
            config = fetch_model_config(model_id)
            summary = summarize_config(config)
            dim_symbols = get_dim_symbols(summary)
        except Exception:
            summary = {}
            dim_symbols = {}

        # Check for known unsupported architectures on XPU
        arch = summary.get("architecture", "")
        _MLA_ARCHS = {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
                      "DeepseekV4ForCausalLM", "GlmMoeDsaForCausalLM"}
        if arch in _MLA_ARCHS:
            raise RuntimeError(
                f"{model_id} uses MLA (Multi-Head Latent Attention) which "
                f"requires FlashAttention — not available on XPU. "
                f"Static analysis works; profiling is not yet supported for "
                f"this architecture on XPU hardware."
            )

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

        # Quantization method
        if quantization:
            engine_kwargs["quantization"] = quantization

        # Override layer count for reduced-layer profiling.
        if profiled_layers < actual_layers:
            engine_kwargs["hf_overrides"] = {
                "num_hidden_layers": profiled_layers,
            }

        # Set compile / eager mode
        if mode == "compile":
            os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"
            engine_kwargs["enforce_eager"] = False
        else:
            os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)
            engine_kwargs["enforce_eager"] = True

        llm = LLM(**engine_kwargs)

        sampling_params = SamplingParams(max_tokens=max_tokens)

        # Use chat() if model supports it, else fall back to generate()
        try:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            conversations = [conversation] * batch_size
            # Test that chat template works
            llm.chat(conversations, sampling_params, use_tqdm=False)
            use_chat = True
        except Exception:
            # Model may not have a chat template — use raw generate
            prompts = [prompt] * batch_size
            llm.generate(prompts, sampling_params, use_tqdm=False)
            use_chat = False

        # --- Profiled run ---
        llm.start_profile()
        if use_chat:
            llm.chat(conversations, sampling_params, use_tqdm=False)
        else:
            llm.generate(prompts, sampling_params, use_tqdm=False)
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
        op_dicts = parse_trace_file(rank_files[0])

        # If multi-rank, average device times across ranks
        if tp_size > 1 and len(rank_files) > 1:
            # Build timing map from additional ranks and average
            for extra_file in rank_files[1:]:
                extra_ops = parse_trace_file(extra_file)
                extra_timing = {
                    (o["name"], o.get("input_shapes", "")): o.get("device_time_us", 0)
                    for o in extra_ops
                }
                for op in op_dicts:
                    key = (op["name"], op.get("input_shapes", ""))
                    extra_t = extra_timing.get(key, 0)
                    op["device_time_us"] = (
                        op.get("device_time_us", 0) + extra_t
                    )
            # Average across ranks
            for op in op_dicts:
                op["device_time_us"] = op.get("device_time_us", 0) / tp_size

        if not op_dicts:
            raise RuntimeError(
                f"No ops found in trace file {trace_files[0]}. "
                "The worker may not have captured any events."
            )

        # Analyze
        analyzed = analyze_ops(
            op_dicts,
            dim_symbols=dim_symbols,
            batch_size=batch_size,
            seq_len=None,
            model_dtype=summary.get("dtype", "bfloat16"),
            num_layers=summary.get("num_layers"),
        )

        # Build result
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
            "tp_size": tp_size,
            "quantization": quantization,
            "summary": summary,
            "total_device_time_us": total_dev,
            "total_cpu_time_us": sum(o.cpu_time_us for o in analyzed),
            "backends": backend_totals,
            "ops": [o.to_dict() for o in analyzed],
            # Layer scaling info for 1-layer profiling
            "profiled_layers": profiled_layers,
            "actual_layers": actual_layers,
            "layer_scale": layer_scale,
            # Trace file path for download
            "trace_file": rank_files[0],
        }

        # Build annotated graph — same tree view with timing overlaid
        # Use actual_layers in the graph so repeat_count is correct
        try:
            graph = build_model_graph(summary, prefill_len=128,
                                      decode_batch=batch_size,
                                      context_len=max_model_len,
                                      tp_size=tp_size,
                                      quantization=quantization)
            # Try module-path-based annotation first (more precise)
            module_ops = parse_trace_with_modules(rank_files[0])
            if module_ops:
                annotate_graph_from_modules(graph, module_ops)
            else:
                # Fall back to name+shape matching
                annotate_graph_timing(graph, op_dicts)
            profile_result["graph"] = graph
        except Exception:
            pass  # Graph annotation is best-effort

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
              num_profile_layers, tp_size, quantization),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "status": "running"})


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

    # Build a descriptive filename:
    # vllm_trace_{model}_{mode}_bs{batch}_ctx{ctx}_tp{tp}_layers{n}.json.gz
    model_short = result["model_id"].replace("/", "_")
    mode = result.get("mode", "eager")
    bs = result.get("batch_size", 1)
    ctx = result.get("max_model_len", "")
    tp = result.get("tp_size", 1) or 1
    quant = result.get("quantization")
    layers = result.get("profiled_layers", "all")
    ext = ".json.gz" if trace_path.endswith(".gz") else ".json"
    quant_part = f"_{quant}" if quant else ""
    download_name = f"vllm_trace_{model_short}_{mode}_bs{bs}_ctx{ctx}_tp{tp}{quant_part}_{layers}layers{ext}"

    return send_file(
        trace_path,
        mimetype="application/gzip" if ext == ".json.gz" else "application/json",
        as_attachment=True,
        download_name=download_name,
    )


# ---- Demo/mock data for UI development ----

@app.route("/api/demo")
def demo_data():
    """Return mock profiling data for UI development without running a model."""
    mock_ops = [
        {"name": "aten::mm", "backend": "torch-xpu-ops", "category": "aten-xpu",
         "input_shapes": "[[4, 128, 3072], [3072, 3072]]", "count": 64,
         "device_time_us": 32000, "cpu_time_us": 1200},
        {"name": "aten::addmm", "backend": "torch-xpu-ops", "category": "aten-xpu",
         "input_shapes": "[[3072], [512, 3072], [3072, 8192]]", "count": 32,
         "device_time_us": 28000, "cpu_time_us": 900},
        {"name": "rms_norm", "backend": "vllm-xpu-kernels", "category": "vllm-xpu-kernels (general)",
         "input_shapes": "[[4, 128, 3072], [3072]]", "count": 64,
         "device_time_us": 8000, "cpu_time_us": 500},
        {"name": "silu_and_mul", "backend": "vllm-xpu-kernels", "category": "vllm-xpu-kernels (general)",
         "input_shapes": "[[4, 128, 16384]]", "count": 32,
         "device_time_us": 6000, "cpu_time_us": 300},
        {"name": "rotary_embedding", "backend": "vllm-xpu-kernels", "category": "vllm-xpu-kernels (general)",
         "input_shapes": "[[128], [4, 128, 32, 96], [4, 128, 8, 96]]", "count": 32,
         "device_time_us": 4000, "cpu_time_us": 200},
        {"name": "reshape_and_cache_flash", "backend": "vllm-xpu-kernels", "category": "vllm-xpu-kernels (cache)",
         "input_shapes": "[[4, 128, 8, 96], [4, 128, 8, 96]]", "count": 32,
         "device_time_us": 5000, "cpu_time_us": 400},
        {"name": "fused_add_rms_norm", "backend": "vllm-xpu-kernels", "category": "vllm-xpu-kernels (general)",
         "input_shapes": "[[4, 128, 3072], [4, 128, 3072], [3072]]", "count": 32,
         "device_time_us": 4500, "cpu_time_us": 250},
        {"name": "triton_flash_attn_fwd", "backend": "triton", "category": "triton-compiled",
         "input_shapes": "[[4, 128, 32, 96], [4, 128, 8, 96], [4, 128, 8, 96]]", "count": 32,
         "device_time_us": 20000, "cpu_time_us": 1500},
        {"name": "aten::embedding", "backend": "torch-xpu-ops", "category": "aten-xpu",
         "input_shapes": "[[151936, 3072], [512]]", "count": 2,
         "device_time_us": 800, "cpu_time_us": 100},
        {"name": "aten::softmax", "backend": "torch-xpu-ops", "category": "aten-xpu",
         "input_shapes": "[[4, 128, 151936]]", "count": 1,
         "device_time_us": 2000, "cpu_time_us": 50},
        {"name": "static_scaled_fp8_quant", "backend": "vllm-xpu-kernels", "category": "vllm-xpu-kernels (general)",
         "input_shapes": "[[4, 128, 3072], [1]]", "count": 32,
         "device_time_us": 3000, "cpu_time_us": 200},
    ]

    mock_summary = {
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "hidden_size": 3072,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 96,
        "intermediate_size": 8192,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "dtype": "bfloat16",
        "is_moe": False,
        "num_experts": None,
        "num_experts_per_tok": None,
        "quant_method": None,
        "rope_type": None,
    }

    dim_symbols = get_dim_symbols(mock_summary)

    analyzed = analyze_ops(
        mock_ops,
        dim_symbols=dim_symbols,
        batch_size=4,
        seq_len=128,
        model_dtype="bfloat16",
        num_layers=32,
    )

    total_dev = sum(o.device_time_us for o in analyzed)
    backend_totals: dict[str, dict] = {}
    for b in Backend:
        ops = [o for o in analyzed if o.backend == b.value]
        dev = sum(o.device_time_us for o in ops)
        backend_totals[b.value] = {
            "device_time_us": dev,
            "pct": round(dev / total_dev * 100, 1) if total_dev > 0 else 0,
            "num_ops": len(ops),
            "num_calls": sum(o.call_count for o in ops),
        }

    return jsonify({
        "ok": True,
        "data": {
            "model_id": "Qwen/Qwen3-4B (demo)",
            "mode": "eager",
            "summary": mock_summary,
            "total_device_time_us": total_dev,
            "total_cpu_time_us": sum(o.cpu_time_us for o in analyzed),
            "backends": backend_totals,
            "ops": [o.to_dict() for o in analyzed],
        },
    })


# ---- Excel Export ----

# Config keys written to the Summary sheet with their row numbers.
# These are referenced by formulas in the Operations sheet.
_SUMMARY_CONFIG_KEYS = [
    "hidden_size", "num_layers", "num_heads", "num_kv_heads",
    "head_dim", "intermediate_size", "vocab_size",
]

# Symbol → formula fragment mapping.
# Each entry maps a symbolic shape token to an Excel formula reference
# using the named row in the Summary sheet (column B, rows defined by
# _SUMMARY_CONFIG_START_ROW offset).
_SUMMARY_CONFIG_START_ROW = 7  # Row where first config value is written


def _symbol_to_cell_ref(symbol: str, cfg_cell_map: dict[str, str]) -> str | None:
    """Map a symbolic dimension name to a Summary sheet cell reference.

    Returns an Excel formula fragment like 'Summary!$B$7' or a compound
    expression like 'Summary!$B$7*Summary!$B$11' for composite symbols.
    Returns None if the symbol cannot be resolved.
    """
    # Direct match
    if symbol in cfg_cell_map:
        return cfg_cell_map[symbol]

    # Composite symbols
    if symbol == "n_h·d" and "n_h" in cfg_cell_map and "d" in cfg_cell_map:
        return f"{cfg_cell_map['n_h']}*{cfg_cell_map['d']}"
    if symbol == "QKV" and "n_h" in cfg_cell_map and "n_kv" in cfg_cell_map and "d" in cfg_cell_map:
        return f"({cfg_cell_map['n_h']}+2*{cfg_cell_map['n_kv']})*{cfg_cell_map['d']}"
    if symbol == "2·I" and "I" in cfg_cell_map:
        return f"2*{cfg_cell_map['I']}"

    return None


def _shape_to_formula(shape: list, cfg_cell_map: dict[str, str],
                      symbols: dict[str, int]) -> str | None:
    """Convert a symbolic shape list to an Excel formula showing concrete dims.

    Returns a formula string like '="[128, "&Summary!$B$7&"]"' or None if
    the shape contains no symbolic dimensions.
    """
    has_symbol = False
    parts: list[str] = []
    for dim in shape:
        if isinstance(dim, str):
            ref = _symbol_to_cell_ref(dim, cfg_cell_map)
            if ref:
                has_symbol = True
                parts.append(("ref", ref))
            elif dim in symbols:
                # Known symbol with concrete value but no cell ref (e.g. S, B, C)
                parts.append(("lit", str(symbols[dim])))
            else:
                parts.append(("lit", dim))
        else:
            parts.append(("lit", str(dim)))

    if not has_symbol:
        return None

    # Build CONCATENATE formula: ="[" & part & ", " & part & "]"
    formula_parts = ['"["']
    for idx, (kind, val) in enumerate(parts):
        if idx > 0:
            formula_parts.append('", "')
        if kind == "ref":
            formula_parts.append(val)
        else:
            formula_parts.append(f'"{val}"')
    formula_parts.append('"]"')
    return "=" + "&".join(formula_parts)


def _shape_to_concrete(shape: list, symbols: dict[str, int]) -> str:
    """Convert a symbolic shape list to a concrete string with numeric values."""
    concrete = []
    for dim in shape:
        if isinstance(dim, str) and dim in symbols:
            concrete.append(str(symbols[dim]))
        elif isinstance(dim, int):
            concrete.append(str(dim))
        else:
            concrete.append(str(dim))
    return "[" + ", ".join(concrete) + "]"


def _normalize_dtype(dtype: str) -> str:
    """Normalize dtype string to short form (e.g. bfloat16 -> bf16)."""
    mapping = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float8": "fp8",
    }
    return mapping.get(dtype, dtype)


def _shape_concrete_with_dtype(op: dict, symbols: dict[str, int]) -> str:
    """Build concrete shape string with dtype appended inside each bracket.

    Example: "[128, 2560, bf16] × [2560, 6144, bf16]"
    """
    shapes = op.get("input_shapes")
    if not shapes or not isinstance(shapes, list):
        return "—"
    dtype = _normalize_dtype(op.get("dtype") or "")
    parts = []
    for shape in shapes:
        if isinstance(shape, list):
            concrete = []
            for dim in shape:
                if isinstance(dim, str) and dim in symbols:
                    concrete.append(str(symbols[dim]))
                elif isinstance(dim, int):
                    concrete.append(str(dim))
                else:
                    concrete.append(str(dim))
            if dtype:
                concrete.append(dtype)
            parts.append("[" + ", ".join(concrete) + "]")
        else:
            parts.append(str(shape))
    return " × ".join(parts)


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


@app.route("/api/export/excel", methods=["POST"])
def export_excel():
    """Export breakdown results to Excel with formulas preserved."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    data = request.json
    if not data or "ops" not in data:
        return jsonify({"ok": False, "error": "No data to export"}), 400

    wb = Workbook()

    # ---- Sheet 1: Summary ----
    ws_sum = wb.active
    ws_sum.title = "Summary"
    title_font = Font(bold=True, size=14)
    header_font = Font(bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill(start_color="1A1A2E", end_color="1A1A2E",
                              fill_type="solid")
    thin_border = Border(
        bottom=Side(style="thin", color="E0E0E0"),
    )

    ws_sum["A1"] = "vLLM-XPU Ops/Kernels Breakdown"
    ws_sum["A1"].font = title_font
    ws_sum["A2"] = f"Model: {data.get('model_id', 'N/A')}"
    ws_sum["A3"] = f"Mode: {data.get('mode', 'N/A')}"
    ws_sum["A4"] = (
        f"Batch Size: {data.get('batch_size', 'N/A')} | "
        f"Context Len: {data.get('max_model_len', 'N/A')} | "
        f"TP: {data.get('tp_size', 1)}"
        + (f" | Quant: {data.get('quantization')}" if data.get('quantization') else "")
    )

    # Model summary — write config values in named cells for formula reference
    summary = data.get("summary", {})
    # Map: symbol name → "Summary!$B$<row>" for formula references
    cfg_cell_map: dict[str, str] = {}
    # Map: config key → row number (for building cfg_cell_map)
    cfg_row_map: dict[str, int] = {}

    if summary:
        row = 5
        ws_sum.cell(row, 1, "Model Configuration").font = Font(bold=True, size=12)
        row += 1
        # Write architecture and dtype first (non-numeric, not formula targets)
        ws_sum.cell(row, 1, "architecture").font = Font(bold=True)
        ws_sum.cell(row, 2, str(summary.get("architecture", "")))
        row += 1

        for key in _SUMMARY_CONFIG_KEYS:
            if key in summary:
                ws_sum.cell(row, 1, key).font = Font(bold=True)
                val = summary[key]
                if isinstance(val, (int, float)):
                    ws_sum.cell(row, 2, val)
                else:
                    ws_sum.cell(row, 2, str(val))
                cfg_row_map[key] = row
                row += 1

        # dtype row
        ws_sum.cell(row, 1, "dtype").font = Font(bold=True)
        ws_sum.cell(row, 2, str(summary.get("dtype", "bfloat16")))
        row += 1
        # is_moe row
        ws_sum.cell(row, 1, "is_moe").font = Font(bold=True)
        ws_sum.cell(row, 2, str(summary.get("is_moe", False)))
        row += 1

        # Build symbol → cell reference map
        _symbol_key_map = {
            "H": "hidden_size",
            "n_h": "num_heads",
            "n_kv": "num_kv_heads",
            "d": "head_dim",
            "I": "intermediate_size",
            "V": "vocab_size",
        }
        for sym, key in _symbol_key_map.items():
            if key in cfg_row_map:
                cfg_cell_map[sym] = f"Summary!$B${cfg_row_map[key]}"

    # Get symbols dict from graph data or build from summary
    graph_data = data.get("graph")
    symbols: dict[str, int] = {}
    graph_cfg: dict = {}
    if graph_data and "symbols" in graph_data:
        symbols = graph_data["symbols"]
    if graph_data and "config" in graph_data:
        graph_cfg = graph_data["config"]
    if not symbols and summary:
        # Fallback: build symbols from summary
        for key, val in summary.items():
            if isinstance(val, int):
                if key == "hidden_size":
                    symbols["H"] = val
                elif key == "num_heads":
                    symbols["n_h"] = val
                elif key == "num_kv_heads":
                    symbols["n_kv"] = val
                elif key == "head_dim":
                    symbols["d"] = val
                elif key == "intermediate_size":
                    symbols["I"] = val
                elif key == "vocab_size":
                    symbols["V"] = val

    # Backend summary
    backends = data.get("backends", {})
    if backends:
        row += 1
        ws_sum.cell(row, 1, "Backend Distribution").font = Font(bold=True,
                                                                  size=12)
        row += 1
        for col, hdr in enumerate(["Backend", "Device Time (µs)", "% Time",
                                    "Ops", "Calls"], 1):
            c = ws_sum.cell(row, col, hdr)
            c.font = header_font
            c.fill = header_fill
        row += 1
        for name, b in backends.items():
            if b.get("num_ops", 0) == 0:
                continue
            ws_sum.cell(row, 1, name)
            ws_sum.cell(row, 2, b["device_time_us"])
            ws_sum.cell(row, 3, b["pct"] / 100)
            ws_sum.cell(row, 3).number_format = '0.0%'
            ws_sum.cell(row, 4, b["num_ops"])
            ws_sum.cell(row, 5, b["num_calls"])
            row += 1

    ws_sum.column_dimensions["A"].width = 28
    ws_sum.column_dimensions["B"].width = 20

    # ---- Sheet 2: Operations with formulas ----
    ws = wb.create_sheet("Operations")

    headers = ["Op Name", "Backend", "Shape (symbolic)", "Shape (concrete)",
               "Shape (concrete + dtype)",
               "×Layers", "Calls", "Device Time (µs)", "% Time",
               "Memory (bytes)", "FLOPs", "Arithmetic Intensity"]
    for col, hdr in enumerate(headers, 1):
        c = ws.cell(1, col, hdr)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")

    ops = data["ops"]

    for i, op in enumerate(ops):
        r = i + 2  # data starts at row 2
        ws.cell(r, 1, op["name"])
        ws.cell(r, 2, op["backend"])

        # Shape (symbolic) — original shapes with dimension names
        shapes = op.get("input_shapes")
        if shapes:
            ws.cell(r, 3, json.dumps(shapes))
        else:
            ws.cell(r, 3, "—")

        # Shape (concrete) — resolved to numeric values using formulas
        if shapes and isinstance(shapes, list) and len(shapes) > 0:
            # Try to build a formula referencing Summary cells
            shape_formulas = []
            has_any_formula = False
            concrete_parts = []
            for shape in shapes:
                if isinstance(shape, list):
                    formula = _shape_to_formula(shape, cfg_cell_map, symbols)
                    if formula:
                        has_any_formula = True
                        shape_formulas.append(formula)
                    concrete_parts.append(
                        _shape_to_concrete(shape, symbols)
                    )

            if has_any_formula and len(shapes) == 1 and shape_formulas:
                # Single shape: use formula directly
                ws.cell(r, 4).value = shape_formulas[0]
            elif has_any_formula and shape_formulas:
                # Multiple shapes: Excel formulas for multi-tensor
                # expressions become unreadable; use resolved values instead
                ws.cell(r, 4, ", ".join(concrete_parts))
            else:
                # No symbolic dims: just show concrete values
                ws.cell(r, 4, ", ".join(concrete_parts))
        else:
            ws.cell(r, 4, "—")

        ws.cell(r, 5, _shape_concrete_with_dtype(op, symbols))

        ws.cell(r, 6, op.get("layer_count", 1))
        ws.cell(r, 7, op.get("call_count", 0))
        ws.cell(r, 8, op.get("device_time_us", 0))

        # % Time as formula: device_time / SUM(device_time_column)
        last_data_row = len(ops) + 1
        col_h = get_column_letter(8)  # H = Device Time
        ws.cell(r, 9).value = f"={col_h}{r}/SUM({col_h}$2:{col_h}${last_data_row})"
        ws.cell(r, 9).number_format = '0.0%'

        ws.cell(r, 10, op.get("memory_bytes", 0))

        flops = op.get("flops", 0)
        ws.cell(r, 11, flops)

        # Arithmetic Intensity as formula: FLOPs / Memory (avoid div-by-zero)
        col_j = get_column_letter(10)   # J = Memory
        col_k = get_column_letter(11)   # K = FLOPs
        ws.cell(r, 12).value = (
            f'=IF({col_j}{r}=0,"—",{col_k}{r}/{col_j}{r})'
        )
        ws.cell(r, 12).number_format = '0.00'

    # Totals row
    total_row = len(ops) + 2
    ws.cell(total_row, 1, "TOTAL").font = Font(bold=True)
    col_g = get_column_letter(7)
    col_h = get_column_letter(8)
    ws.cell(total_row, 7).value = f"=SUM({col_g}2:{col_g}{len(ops)+1})"
    ws.cell(total_row, 7).font = Font(bold=True)
    ws.cell(total_row, 8).value = f"=SUM({col_h}2:{col_h}{len(ops)+1})"
    ws.cell(total_row, 8).font = Font(bold=True)
    ws.cell(total_row, 9, 1.0)
    ws.cell(total_row, 9).number_format = '0.0%'
    ws.cell(total_row, 9).font = Font(bold=True)

    # Column widths
    col_widths = [45, 18, 50, 50, 50, 8, 8, 15, 10, 15, 15, 15]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    # Freeze header row
    ws.freeze_panes = "A2"

    # Auto-filter
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(ops)+1}"

    # Apply border to data rows
    for r in range(2, len(ops) + 2):
        for col in range(1, len(headers) + 1):
            ws.cell(r, col).border = thin_border

    # ---- Sheet 3: Model Hierarchy ----
    if graph_data:
        phase = data.get("phase", "prefill")
        tree = graph_data.get(phase) or graph_data.get("prefill")
        if tree:
            ws_hier = wb.create_sheet("Model Hierarchy")

            hier_headers = ["Module", "Path", "Type", "×Repeat",
                            "Memory (bytes)", "FLOPs", "AI",
                            "Op Role", "Op Name", "Op Backend", "Op Shape",
                            "Sub Op Time"]
            for col, hdr in enumerate(hier_headers, 1):
                c = ws_hier.cell(1, col, hdr)
                c.font = header_font
                c.fill = header_fill
                c.alignment = Alignment(horizontal="center")

            flat_nodes = _flatten_graph_nodes(tree)
            hier_row = 2
            indent_fill = PatternFill(start_color="F5F5F5", end_color="F5F5F5",
                                      fill_type="solid")
            module_font = Font(bold=True)

            for node_info in flat_nodes:
                depth = node_info["depth"]
                indent = "  " * depth

                # Module header row
                ws_hier.cell(hier_row, 1, indent + node_info["name"])
                ws_hier.cell(hier_row, 1).font = module_font
                ws_hier.cell(hier_row, 2, node_info["path"])
                ws_hier.cell(hier_row, 3, node_info["module_type"])
                repeat = node_info["repeat_count"]
                if repeat > 1:
                    ws_hier.cell(hier_row, 4, repeat)
                ws_hier.cell(hier_row, 5, node_info["total_memory"])
                ws_hier.cell(hier_row, 6, node_info["total_flops"])
                ai = node_info["total_ai"]
                if ai:
                    ws_hier.cell(hier_row, 7, ai)
                    ws_hier.cell(hier_row, 7).number_format = '0.00'

                # Apply light background for module rows
                for col in range(1, len(hier_headers) + 1):
                    ws_hier.cell(hier_row, col).fill = indent_fill
                    ws_hier.cell(hier_row, col).border = thin_border

                hier_row += 1

                # Op rows under this module
                for op in node_info["ops"]:
                    op_indent = "  " * (depth + 1)
                    op_role = op.get("role", "")
                    ws_hier.cell(hier_row, 8, op_role)
                    # Use high-level name when sub-ops show kernel details
                    if op.get("sub_ops"):
                        op_name = op.get("name", "")
                    else:
                        op_name = op.get("profiled_name") or op.get("name", "")
                    ws_hier.cell(hier_row, 9, op_name)
                    ws_hier.cell(hier_row, 10, op.get("backend", ""))

                    # Show concrete op shapes with per-tensor dtype
                    op_shapes = op.get("input_shapes", [])
                    if op_shapes:
                        concrete_shapes = []
                        for shape_idx, shape in enumerate(op_shapes):
                            if isinstance(shape, list):
                                # Determine per-tensor dtype
                                tensor_dtype = _get_tensor_dtype(
                                    shape_idx, op_role, graph_cfg
                                )
                                parts = []
                                for dim in shape:
                                    if isinstance(dim, str) and dim in symbols:
                                        parts.append(str(symbols[dim]))
                                    elif isinstance(dim, int):
                                        parts.append(str(dim))
                                    else:
                                        parts.append(str(dim))
                                if tensor_dtype:
                                    parts.append(tensor_dtype)
                                concrete_shapes.append(
                                    "[" + ", ".join(parts) + "]"
                                )
                            else:
                                concrete_shapes.append(str(shape))
                        ws_hier.cell(
                            hier_row, 11, " × ".join(concrete_shapes)
                        )
                    else:
                        ws_hier.cell(hier_row, 11, "—")

                    ws_hier.cell(hier_row, 1, op_indent + "↳ " + op_role)
                    for col in range(1, len(hier_headers) + 1):
                        ws_hier.cell(hier_row, col).border = thin_border
                    hier_row += 1

                    # Sub-ops rows (constituent kernels of complex ops)
                    for sub_op in op.get("sub_ops", []):
                        sub_indent = "  " * (depth + 2)
                        sname = sub_op.get("name", "")
                        ws_hier.cell(hier_row, 1, sub_indent + "⤷ " + sname)
                        ws_hier.cell(hier_row, 9, sname)
                        # Show timing in dedicated Sub Op Time column
                        if sub_op.get("device_time_us", 0) > 0:
                            ws_hier.cell(hier_row, 12, f'{sub_op["device_time_us"]:.1f}µs')
                        for col in range(1, len(hier_headers) + 1):
                            ws_hier.cell(hier_row, col).border = thin_border
                            ws_hier.cell(hier_row, col).font = Font(
                                size=9, color="888888"
                            )
                        hier_row += 1

            # Column widths for hierarchy sheet
            hier_col_widths = [35, 35, 25, 8, 15, 15, 10, 18, 35, 18, 55, 12]
            for i, w in enumerate(hier_col_widths, 1):
                ws_hier.column_dimensions[get_column_letter(i)].width = w

            ws_hier.freeze_panes = "A2"

    # Write to buffer
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    model_name = data.get("model_id", "breakdown").replace("/", "_")
    mode = data.get("mode", "eager")
    bs = data.get("batch_size", "")
    ctx = data.get("max_model_len", "")
    tp = data.get("tp_size", "")
    quant = data.get("quantization")

    # Build descriptive filename encoding profile settings
    parts = [f"vllm_xpu_breakdown_{model_name}_{mode}"]
    if bs:
        parts.append(f"bs{bs}")
    if ctx:
        parts.append(f"ctx{ctx}")
    if tp and int(tp) > 1:
        parts.append(f"tp{tp}")
    if quant:
        parts.append(quant)
    filename = "_".join(parts) + ".xlsx"

    return Response(
        buf.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/api/export/static-graph", methods=["POST"])
def export_static_graph():
    """Export the static model graph breakdown to Excel (no profiling needed)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    data = request.json
    if not data or "graph" not in data:
        return jsonify({"ok": False, "error": "No graph data to export"}), 400

    graph_data = data["graph"]
    summary = data.get("summary", {})
    model_id = data.get("model_id", "unknown")
    phase = data.get("phase", "prefill")

    wb = Workbook()

    title_font = Font(bold=True, size=14)
    header_font = Font(bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill(start_color="1A1A2E", end_color="1A1A2E",
                              fill_type="solid")
    thin_border = Border(
        bottom=Side(style="thin", color="E0E0E0"),
    )

    # ---- Sheet 1: Summary ----
    ws_sum = wb.active
    ws_sum.title = "Summary"

    ws_sum["A1"] = "vLLM-XPU Static Model Graph Breakdown"
    ws_sum["A1"].font = title_font
    ws_sum["A2"] = f"Model: {model_id}"
    ws_sum["A3"] = f"Phase: {phase}"

    graph_cfg = graph_data.get("config", {})
    tp_size = graph_cfg.get("tp_size", 1)
    quant = graph_cfg.get("quantization")
    ws_sum["A4"] = (
        f"Prefill Len: {graph_cfg.get('prefill_len', 'N/A')} | "
        f"Decode Batch: {graph_cfg.get('decode_batch', 'N/A')} | "
        f"Context Len: {graph_cfg.get('context_len', 'N/A')} | "
        f"TP: {tp_size}"
        + (f" | Quant: {quant}" if quant else "")
    )

    # Write model config
    row = 6
    if summary:
        ws_sum.cell(row, 1, "Model Configuration").font = Font(bold=True, size=12)
        row += 1
        ws_sum.cell(row, 1, "architecture").font = Font(bold=True)
        ws_sum.cell(row, 2, str(summary.get("architecture", "")))
        row += 1

        for key in _SUMMARY_CONFIG_KEYS:
            if key in summary:
                ws_sum.cell(row, 1, key).font = Font(bold=True)
                val = summary[key]
                if isinstance(val, (int, float)):
                    ws_sum.cell(row, 2, val)
                else:
                    ws_sum.cell(row, 2, str(val))
                row += 1

        ws_sum.cell(row, 1, "dtype").font = Font(bold=True)
        ws_sum.cell(row, 2, str(summary.get("dtype", "bfloat16")))
        row += 1
        ws_sum.cell(row, 1, "is_moe").font = Font(bold=True)
        ws_sum.cell(row, 2, str(summary.get("is_moe", False)))
        row += 1

    # Write symbols
    symbols = graph_data.get("symbols", {})
    if symbols:
        row += 1
        ws_sum.cell(row, 1, "Dimension Symbols").font = Font(bold=True, size=12)
        row += 1
        for sym, val in symbols.items():
            ws_sum.cell(row, 1, sym).font = Font(bold=True)
            ws_sum.cell(row, 2, val)
            row += 1

    ws_sum.column_dimensions["A"].width = 28
    ws_sum.column_dimensions["B"].width = 20

    # ---- Sheet 2: Model Hierarchy ----
    tree = graph_data.get(phase) or graph_data.get("prefill")
    if tree:
        ws_hier = wb.create_sheet("Model Hierarchy")

        hier_headers = ["Module", "Path", "Type", "×Repeat",
                        "Memory (bytes)", "FLOPs", "AI",
                        "Op Role", "Op Name", "Op Backend", "Op Shape"]
        for col, hdr in enumerate(hier_headers, 1):
            c = ws_hier.cell(1, col, hdr)
            c.font = header_font
            c.fill = header_fill
            c.alignment = Alignment(horizontal="center")

        flat_nodes = _flatten_graph_nodes(tree)
        hier_row = 2
        indent_fill = PatternFill(start_color="F5F5F5", end_color="F5F5F5",
                                  fill_type="solid")
        module_font = Font(bold=True)

        for node_info in flat_nodes:
            depth = node_info["depth"]
            indent = "  " * depth

            # Module header row
            ws_hier.cell(hier_row, 1, indent + node_info["name"])
            ws_hier.cell(hier_row, 1).font = module_font
            ws_hier.cell(hier_row, 2, node_info["path"])
            ws_hier.cell(hier_row, 3, node_info["module_type"])
            repeat = node_info["repeat_count"]
            if repeat > 1:
                ws_hier.cell(hier_row, 4, repeat)
            ws_hier.cell(hier_row, 5, node_info["total_memory"])
            ws_hier.cell(hier_row, 6, node_info["total_flops"])
            ai = node_info["total_ai"]
            if ai:
                ws_hier.cell(hier_row, 7, ai)
                ws_hier.cell(hier_row, 7).number_format = '0.00'

            for col in range(1, len(hier_headers) + 1):
                ws_hier.cell(hier_row, col).fill = indent_fill
                ws_hier.cell(hier_row, col).border = thin_border

            hier_row += 1

            # Op rows under this module
            for op in node_info["ops"]:
                op_role = op.get("role", "")
                ws_hier.cell(hier_row, 8, op_role)
                op_name = op.get("name", "")
                ws_hier.cell(hier_row, 9, op_name)
                ws_hier.cell(hier_row, 10, op.get("backend", ""))

                # Show concrete op shapes with per-tensor dtype
                op_shapes = op.get("input_shapes", [])
                if op_shapes:
                    concrete_shapes = []
                    for shape_idx, shape in enumerate(op_shapes):
                        if isinstance(shape, list):
                            tensor_dtype = _get_tensor_dtype(
                                shape_idx, op_role, graph_cfg
                            )
                            parts = []
                            for dim in shape:
                                if isinstance(dim, str) and dim in symbols:
                                    parts.append(str(symbols[dim]))
                                elif isinstance(dim, int):
                                    parts.append(str(dim))
                                else:
                                    parts.append(str(dim))
                            if tensor_dtype:
                                parts.append(tensor_dtype)
                            concrete_shapes.append(
                                "[" + ", ".join(parts) + "]"
                            )
                        else:
                            concrete_shapes.append(str(shape))
                    ws_hier.cell(
                        hier_row, 11, " × ".join(concrete_shapes)
                    )
                else:
                    ws_hier.cell(hier_row, 11, "—")

                op_indent = "  " * (depth + 1)
                ws_hier.cell(hier_row, 1, op_indent + "↳ " + op_role)
                for col in range(1, len(hier_headers) + 1):
                    ws_hier.cell(hier_row, col).border = thin_border
                hier_row += 1

        # Column widths
        hier_col_widths = [35, 35, 25, 8, 15, 15, 10, 18, 35, 18, 55]
        for i, w in enumerate(hier_col_widths, 1):
            ws_hier.column_dimensions[get_column_letter(i)].width = w

        ws_hier.freeze_panes = "A2"

    # Write to buffer
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    model_name = model_id.replace("/", "_")
    parts = [f"vllm_xpu_static_graph_{model_name}_{phase}"]
    if tp_size > 1:
        parts.append(f"tp{tp_size}")
    if quant:
        parts.append(quant)
    filename = "_".join(parts) + ".xlsx"

    return Response(
        buf.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
