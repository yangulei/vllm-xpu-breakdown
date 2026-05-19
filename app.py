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
        _MLA_ARCHS = {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"}
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
                                      tp_size=tp_size)
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
    # vllm_trace_{model}_{mode}_tp{tp}_layers{n}.json.gz
    model_short = result["model_id"].replace("/", "_")
    mode = result.get("mode", "eager")
    tp = result.get("summary", {}).get("tp_size", 1) or 1
    layers = result.get("profiled_layers", "all")
    ext = ".json.gz" if trace_path.endswith(".gz") else ".json"
    download_name = f"vllm_trace_{model_short}_{mode}_tp{tp}_{layers}layers{ext}"

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

    # Model summary
    summary = data.get("summary", {})
    if summary:
        row = 5
        ws_sum.cell(row, 1, "Model Configuration").font = Font(bold=True, size=12)
        row += 1
        for key in ("architecture", "hidden_size", "num_layers", "num_heads",
                     "num_kv_heads", "head_dim", "intermediate_size",
                     "vocab_size", "dtype", "is_moe"):
            if key in summary:
                ws_sum.cell(row, 1, key).font = Font(bold=True)
                ws_sum.cell(row, 2, str(summary[key]))
                row += 1

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

    headers = ["Op Name", "Backend", "Shape", "dtype", "×Layers",
               "Calls", "Device Time (µs)", "% Time",
               "Memory (bytes)", "FLOPs", "Arithmetic Intensity"]
    for col, hdr in enumerate(headers, 1):
        c = ws.cell(1, col, hdr)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")

    ops = data["ops"]

    # Write total device time in a named cell for formula reference
    # Place it in a helper cell: the sum will be computed by formula
    # Row 2..N+1 are data rows; we use column G for device time
    total_dev_time_us = data.get("total_device_time_us", 0)

    for i, op in enumerate(ops):
        r = i + 2  # data starts at row 2
        ws.cell(r, 1, op["name"])
        ws.cell(r, 2, op["backend"])

        shapes = op.get("input_shapes")
        if shapes:
            ws.cell(r, 3, json.dumps(shapes))
        else:
            ws.cell(r, 3, "—")

        ws.cell(r, 4, op.get("dtype") or "—")
        ws.cell(r, 5, op.get("layer_count", 1))
        ws.cell(r, 6, op.get("call_count", 0))
        ws.cell(r, 7, op.get("device_time_us", 0))

        # % Time as formula: device_time / SUM(device_time_column)
        last_data_row = len(ops) + 1
        col_g = get_column_letter(7)  # G = Device Time
        ws.cell(r, 8).value = f"={col_g}{r}/SUM({col_g}$2:{col_g}${last_data_row})"
        ws.cell(r, 8).number_format = '0.0%'

        ws.cell(r, 9, op.get("memory_bytes", 0))

        flops = op.get("flops", 0)
        ws.cell(r, 10, flops)

        # Arithmetic Intensity as formula: FLOPs / Memory (avoid div-by-zero)
        col_i = get_column_letter(9)   # I = Memory
        col_j = get_column_letter(10)  # J = FLOPs
        ws.cell(r, 11).value = (
            f'=IF({col_i}{r}=0,"—",{col_j}{r}/{col_i}{r})'
        )
        ws.cell(r, 11).number_format = '0.00'

    # Totals row
    total_row = len(ops) + 2
    ws.cell(total_row, 1, "TOTAL").font = Font(bold=True)
    col_f = get_column_letter(6)
    col_g = get_column_letter(7)
    ws.cell(total_row, 6).value = f"=SUM({col_f}2:{col_f}{len(ops)+1})"
    ws.cell(total_row, 6).font = Font(bold=True)
    ws.cell(total_row, 7).value = f"=SUM({col_g}2:{col_g}{len(ops)+1})"
    ws.cell(total_row, 7).font = Font(bold=True)
    ws.cell(total_row, 8, 1.0)
    ws.cell(total_row, 8).number_format = '0.0%'
    ws.cell(total_row, 8).font = Font(bold=True)

    # Column widths
    col_widths = [45, 18, 50, 10, 8, 8, 15, 10, 15, 15, 15]
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

    # Write to buffer
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    model_name = data.get("model_id", "breakdown").replace("/", "_")
    mode = data.get("mode", "eager")
    filename = f"vllm_xpu_breakdown_{model_name}_{mode}.xlsx"

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
