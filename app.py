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
import json
import os
import sys
import threading
import traceback
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakdown.analyzer import AnalyzedOp, analyze_ops
from breakdown.classifier import Backend, classify_op
from breakdown.model_info import fetch_model_config, get_dim_symbols, summarize_config
from breakdown.profiler import _is_overhead_event
from breakdown.registry import ALL_VLLM_XPU_OPS

app = Flask(__name__, static_folder="static")

# Global state for async profiling
_profile_state = {
    "status": "idle",   # idle | running | done | error
    "result": None,
    "error": None,
    "model_id": None,
}
_profile_lock = threading.Lock()


# ---- Static file serving ----

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


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


# ---- Profile API ----

def _run_profile(model_id: str, mode: str, max_model_len: int,
                 batch_size: int, max_tokens: int, prompt: str):
    """Run profiling in a background thread."""
    global _profile_state
    try:
        import torch
        from vllm import LLM, SamplingParams

        from breakdown.profiler import (ProfileConfig, _sync_device,
                                        parse_events, simple_profile_context)

        # Fetch model config for analysis
        try:
            config = fetch_model_config(model_id)
            summary = summarize_config(config)
            dim_symbols = get_dim_symbols(summary)
        except Exception:
            summary = {}
            dim_symbols = {}

        engine_kwargs: dict = {
            "model": model_id,
            "max_model_len": max_model_len,
        }

        # Set compile mode
        if mode == "compile":
            os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"
        else:
            os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)

        llm = LLM(**engine_kwargs)

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        conversations = [conversation] * batch_size
        sampling_params = SamplingParams(max_tokens=max_tokens)

        # --- Warmup OUTSIDE profiler ---
        llm.chat(conversations, sampling_params, use_tqdm=False)
        _sync_device()

        # --- Profile with simple context (no schedule) ---
        prof_config = ProfileConfig(
            output_dir="output",
            warmup_steps=0,
            active_steps=1,
        )

        with simple_profile_context(prof_config) as prof:
            llm.chat(conversations, sampling_params, use_tqdm=False)
            # _sync_device() is called automatically inside simple_profile_context

        # Parse events (overhead events are filtered out)
        result = parse_events(prof, prof_config, filter_overhead=True)

        # Convert to dicts for analyzer
        op_dicts = []
        for op in result.ops:
            op_dicts.append({
                "name": op.name,
                "backend": op.backend.value,
                "category": op.category,
                "device_time_us": op.device_time_us,
                "cpu_time_us": op.cpu_time_us,
                "count": op.count,
                "input_shapes": op.input_shapes,
            })

        # Analyze
        analyzed = analyze_ops(
            op_dicts,
            dim_symbols=dim_symbols,
            batch_size=batch_size,
            seq_len=None,  # determined from profiling
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
            "summary": summary,
            "total_device_time_us": total_dev,
            "total_cpu_time_us": sum(o.cpu_time_us for o in analyzed),
            "backends": backend_totals,
            "ops": [o.to_dict() for o in analyzed],
        }

        with _profile_lock:
            _profile_state["status"] = "done"
            _profile_state["result"] = profile_result
            _profile_state["error"] = None

        # Cleanup compile env
        os.environ.pop("VLLM_TORCH_COMPILE_LEVEL", None)

    except Exception as e:
        with _profile_lock:
            _profile_state["status"] = "error"
            _profile_state["error"] = traceback.format_exc()


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

    with _profile_lock:
        _profile_state = {
            "status": "running",
            "result": None,
            "error": None,
            "model_id": model_id,
        }

    thread = threading.Thread(
        target=_run_profile,
        args=(model_id, mode, max_model_len, batch_size, max_tokens, prompt),
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
        return jsonify({"ok": True, "data": _profile_state["result"]})


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM-XPU Breakdown Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Starting vLLM-XPU Breakdown at http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=args.debug)
