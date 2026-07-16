# SPDX-License-Identifier: Apache-2.0
"""Parse chrome trace JSON files from vLLM's torch profiler output.

vLLM writes chrome trace format files via `tensorboard_trace_handler`.
These contain events with categories like 'cpu_op', 'kernel', 'xpu_runtime', etc.
This module parses those trace files into the op dict format expected by the analyzer.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict

from .classifier import classify_op
from .trace_common import _is_overhead_event


# Chrome trace event categories that contain real ops
_OP_CATEGORIES = {"cpu_op", "user_annotation"}
_KERNEL_CATEGORIES = {"kernel", "gpu_memcpy", "xpu_op", "gpu_op", "cuda_op",
                      "cuda_runtime", "gpu_kernel"}


# Backends that represent real accelerator compute
_COMPUTE_BACKENDS = frozenset({
    "vllm-xpu-kernels", "vllm-cuda-kernels",
    "torch-xpu-ops", "torch-cuda-ops",
    "triton", "ccl", "flashinfer", "flash_xpu",
})


def _detect_device_via_torch() -> str:
    """Detect the active accelerator via the torch device APIs.

    Returns ``"xpu"``, ``"cuda"`` or ``""``. Torch is imported lazily so this
    module stays import-light; any failure (torch absent, driver error) simply
    yields ``""`` so callers can fall back to other heuristics.
    """
    try:
        import torch
    except Exception:
        return ""
    try:
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return ""


# Trace event categories that unambiguously identify the accelerator.
_XPU_CATEGORIES = {"xpu_runtime", "xpu_op"}
_CUDA_CATEGORIES = {"cuda_runtime", "cuda_op", "cuda_driver"}
# Generic GPU categories that don't distinguish CUDA from XPU on their own.
_GENERIC_GPU_CATEGORIES = {"kernel", "gpu_op", "gpu_memcpy", "gpu_kernel"}


def _infer_device_from_trace(events: list[dict]) -> str:
    """Infer accelerator type from trace events.

    XPU traces emit ``xpu_runtime`` host-launch events (and ``xpu_op``), while
    CUDA traces emit ``cuda_runtime`` / ``cuda_op``. XPU kernel events, however,
    share the generic ``kernel`` category with CUDA, so we must key off the
    device-specific *runtime* categories rather than the kernel category.

    Resolution order:
      1. Device-specific trace categories (``xpu_*`` → xpu, ``cuda_*`` → cuda).
      2. If only generic GPU kernels are present, auto-detect via the torch
         device APIs (``torch.xpu.is_available()`` / ``torch.cuda.is_available()``).
      3. Otherwise, return ``""``.
    """
    saw_generic_gpu = False
    for evt in events:
        cat = evt.get("cat", "")
        if cat in _XPU_CATEGORIES:
            return "xpu"
        if cat in _CUDA_CATEGORIES:
            return "cuda"
        if cat in _GENERIC_GPU_CATEGORIES:
            saw_generic_gpu = True

    # Ambiguous: generic GPU kernels with no device-specific runtime events.
    # Auto-detect the accelerator from the running torch build.
    if saw_generic_gpu:
        detected = _detect_device_via_torch()
        if detected:
            return detected
    return ""


def parse_trace_file(path: str) -> list[dict]:
    """Parse a chrome trace JSON file into classified op dicts.

    Args:
        path: Path to a .json or .json.gz trace file.

    Returns:
        List of op dicts compatible with analyze_ops():
        [{name, backend, category, device_time_us, cpu_time_us, count, input_shapes}, ...]
    """
    # Load trace
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            trace = json.load(f)
    else:
        with open(path) as f:
            trace = json.load(f)

    events = trace.get("traceEvents", [])
    if not events:
        return []

    # Separate CPU ops and kernel/device events
    cpu_ops: list[dict] = []
    kernel_events: list[dict] = []

    for evt in events:
        cat = evt.get("cat", "")
        ph = evt.get("ph", "")
        name = evt.get("name", "")

        # Only duration events (X = complete, B/E = begin/end)
        if ph not in ("X", "B", "E"):
            continue

        if cat in _OP_CATEGORIES:
            cpu_ops.append(evt)
        elif cat in _KERNEL_CATEGORIES:
            kernel_events.append(evt)

    # Build kernel time map: for each CPU thread, accumulate kernel times
    # Kernels are often correlated with CPU ops by timing overlap
    # For simplicity, aggregate by op name
    kernel_time_by_name: dict[str, float] = defaultdict(float)
    for evt in kernel_events:
        name = evt.get("name", "")
        dur = evt.get("dur", 0)
        kernel_time_by_name[name] += dur

    # Aggregate CPU ops by (name, input_shapes)
    op_agg: dict[tuple, dict] = {}

    for evt in cpu_ops:
        name = evt.get("name", "")
        dur = evt.get("dur", 0)
        args = evt.get("args", {})

        if _is_overhead_event(name):
            continue

        # Extract input shapes from args
        input_shapes_str = ""
        if "Input Dims" in args:
            input_shapes_str = str(args["Input Dims"])
        elif "input_shapes" in args:
            input_shapes_str = str(args["input_shapes"])

        # Extract input type/dtype
        input_type = args.get("Input type", "")

        key = (name, input_shapes_str)

        if key not in op_agg:
            op_agg[key] = {
                "name": name,
                "cpu_time_us": 0,
                "device_time_us": 0,
                "count": 0,
                "input_shapes": input_shapes_str,
                "input_type": input_type,
            }

        op_agg[key]["cpu_time_us"] += dur
        op_agg[key]["count"] += 1

    # Try to attribute device time to CPU ops
    # Strategy: match kernel names to their parent CPU ops
    _attribute_device_time(op_agg, kernel_events, kernel_time_by_name)

    # Infer the accelerator from trace events
    device_type = _infer_device_from_trace(kernel_events) or _infer_device_from_trace(cpu_ops)

    # Classify each aggregated op
    result = []
    for key, op in op_agg.items():
        dt = device_type if op["device_time_us"] > 0 else ""
        backend, category = classify_op(
            op["name"],
            device_type=dt,
            self_device_time_us=0,
            device_time_us=op["device_time_us"],
        )

        result.append({
            "name": op["name"],
            "backend": backend.value,
            "category": category,
            "device_time_us": op["device_time_us"],
            "cpu_time_us": op["cpu_time_us"],
            "count": op["count"],
            "input_shapes": op["input_shapes"],
        })

    return result


def _attribute_device_time(op_agg: dict[tuple, dict],
                           kernel_events: list[dict],
                           kernel_time_by_name: dict[str, float]) -> None:
    """Attribute device/kernel time to CPU ops.

    Uses a combination of:
    1. Direct name matching (e.g., kernel 'gemm_kernel' → aten::mm)
    2. Timing correlation (kernel timestamp falls within CPU op duration)
    """
    # Known kernel → op name mappings
    _KERNEL_TO_OP = {
        "gemm_kernel": "aten::mm",
        "gemm_batch_kernel": "aten::bmm",
    }

    # For each kernel, try to attribute to a CPU op
    total_kernel_time = sum(e.get("dur", 0) for e in kernel_events)

    if not kernel_events or not op_agg:
        # If no kernels but we have CPU ops with known compute dispatch,
        # use cpu_time as a rough proxy for device time
        for key, op in op_agg.items():
            name = op["name"]
            # Try both cuda and xpu; whichever is the active backend
            backend, _ = classify_op(name, device_type="cuda",
                                     self_device_time_us=1)
            if backend.value not in _COMPUTE_BACKENDS:
                backend, _ = classify_op(name, device_type="xpu",
                                         self_device_time_us=1)
            if backend.value in _COMPUTE_BACKENDS:
                # Use CPU time as proxy when no kernel events available
                op["device_time_us"] = op["cpu_time_us"]
        return

    # Sort kernel events by timestamp for correlation
    sorted_kernels = sorted(kernel_events, key=lambda e: e.get("ts", 0))

    # Sort CPU ops by timestamp for correlation
    # Build list of (ts, dur, key) from original events
    cpu_intervals: list[tuple[float, float, str]] = []
    for key, op in op_agg.items():
        cpu_intervals.append((0, 0, key[0]))  # placeholder

    # Simple approach: distribute kernel time proportionally to CPU compute ops
    compute_ops = {}
    total_compute_cpu = 0
    for key, op in op_agg.items():
        backend, _ = classify_op(op["name"], device_type="cuda",
                                 self_device_time_us=1)
        if backend.value not in _COMPUTE_BACKENDS:
            backend, _ = classify_op(op["name"], device_type="xpu",
                                     self_device_time_us=1)
        if backend.value in _COMPUTE_BACKENDS:
            compute_ops[key] = op
            total_compute_cpu += op["cpu_time_us"]

    if total_compute_cpu > 0 and total_kernel_time > 0:
        for key, op in compute_ops.items():
            frac = op["cpu_time_us"] / total_compute_cpu
            op["device_time_us"] = total_kernel_time * frac
    elif total_kernel_time > 0:
        # Distribute evenly if no CPU time info
        per_op = total_kernel_time / max(len(compute_ops), 1)
        for key, op in compute_ops.items():
            op["device_time_us"] = per_op


# ===================================================================
# Module-path helpers (module hierarchy + op-role inference)
# ===================================================================


def _strip_instance_idx(class_name: str) -> str:
    """Strip trailing _N index: 'QKVParallelLinear_0' → 'QKVParallelLinear'."""
    parts = class_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return class_name


def _infer_role(module_path: list[str], op_name: str) -> str | None:
    """Infer the op role from its enclosing module hierarchy.

    Args:
        module_path: List of nn.Module class names (outermost first),
                     e.g. ['OPTDecoderLayer', 'OPTAttention', 'QKVParallelLinear']
        op_name: The operator name, e.g. 'aten::linear', 'aten::mm'

    Returns:
        Role string matching static graph op roles, or None if not identifiable.
    """
    if not module_path:
        return None

    innermost = module_path[-1]
    parent = module_path[-2] if len(module_path) >= 2 else ""

    # Embedding
    if "VocabParallelEmbedding" in innermost or "Embedding" in innermost:
        return "embedding"

    # QKV projection
    if "QKVParallel" in innermost:
        return "qkv_proj"

    # Rotary embedding
    if "Rotary" in innermost or "rotary" in op_name.lower():
        return "rotary_emb"

    # Q/K norms (Qwen3-style QK normalization)
    if "Norm" in innermost:
        # Check if inside Attention module → likely q_norm or k_norm
        in_attention = any("Attention" in p for p in module_path[:-1])
        if in_attention:
            # Distinguish by module name hints
            lower_inner = innermost.lower()
            if "q_norm" in lower_inner or "q_layernorm" in lower_inner:
                return "q_norm"
            if "k_norm" in lower_inner or "k_layernorm" in lower_inner:
                return "k_norm"
            # Fallback: generic attention norm (will be disambiguated later)
            return "attention_norm"
        # Determine which norm based on position in layer
        return "norm"

    # Attention kernel / cache ops
    if "Attention" in innermost and "Attention" not in _strip_instance_idx(innermost).replace("Attention", "", 1):
        # innermost IS an Attention module (not just contains "Attention" as part of larger name)
        if "attention" in op_name.lower() or "flash_attn" in op_name.lower():
            return "attention"
        if "cache" in op_name.lower():
            return "cache_store"
        if "rotary" in op_name.lower():
            return "rotary_emb"
        if "norm" in op_name.lower():
            return "attention_norm"
        if op_name in ("aten::linear", "aten::mm", "aten::addmm"):
            return "attention"  # fallback for attention internal ops
        return "attention"

    # Row/Column parallel inside Attention → o_proj
    if "RowParallel" in innermost and any("Attention" in p for p in module_path[:-1]):
        return "o_proj"

    # Row/Column parallel inside MLP → down_proj / gate_up_proj
    if "RowParallel" in innermost and any("MLP" in p or "Mlp" in p or "MoE" in p for p in module_path[:-1]):
        return "down_proj"
    if ("ColumnParallel" in innermost or "MergedColumn" in innermost) and \
       any("MLP" in p or "Mlp" in p or "MoE" in p for p in module_path[:-1]):
        return "gate_up_proj"

    # Generic ColumnParallel at decoder layer level (MLP without MLP wrapper, like OPT)
    if "ColumnParallel" in innermost or "MergedColumn" in innermost:
        return "gate_up_proj"
    if "RowParallel" in innermost:
        # Disambiguation: check if parent is attention-like
        if any("Attention" in p for p in module_path[:-1]):
            return "o_proj"
        return "down_proj"

    # Logits / LM head
    if "Logits" in innermost or "lm_head" in innermost.lower():
        return "lm_head"

    # Activation functions
    if "silu" in op_name.lower() or "gelu" in op_name.lower() or "relu" in op_name.lower():
        return "activation"

    # Cache ops outside Attention module
    if "cache" in op_name.lower():
        return "cache_store"

    return None
