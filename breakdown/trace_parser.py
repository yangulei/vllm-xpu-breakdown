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
from .profiler import _is_overhead_event


# Chrome trace event categories that contain real ops
_OP_CATEGORIES = {"cpu_op", "user_annotation"}
_KERNEL_CATEGORIES = {"kernel", "gpu_memcpy", "xpu_op", "gpu_op"}


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

    # Classify each aggregated op
    result = []
    for key, op in op_agg.items():
        backend, category = classify_op(
            op["name"],
            device_type="xpu" if op["device_time_us"] > 0 else "",
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
        # If no kernels but we have CPU ops with known XPU dispatch,
        # use cpu_time as a rough proxy for device time
        for key, op in op_agg.items():
            name = op["name"]
            backend, _ = classify_op(name, device_type="xpu",
                                     self_device_time_us=1)
            if backend.value in ("vllm-xpu-kernels", "torch-xpu-ops", "triton"):
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
        backend, _ = classify_op(op["name"], device_type="xpu",
                                 self_device_time_us=1)
        if backend.value in ("vllm-xpu-kernels", "torch-xpu-ops", "triton"):
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
# Module-path-aware trace parsing (requires with_stack=True)
# ===================================================================

import re

_MODULE_RE = re.compile(r"^nn\.Module:\s*(.+)$")
_LAYER_IDX_RE = re.compile(r"DecoderLayer_(\d+)|TransformerLayer_(\d+)")


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

    # Attention kernel / cache ops
    if "Attention" in innermost and "Attention" not in _strip_instance_idx(innermost).replace("Attention", "", 1):
        # innermost IS an Attention module (not just contains "Attention" as part of larger name)
        if "attention" in op_name.lower() or "flash_attn" in op_name.lower():
            return "attention"
        if "cache" in op_name.lower():
            return "cache_store"
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

    # Norms
    if "Norm" in innermost:
        # Determine which norm based on position in layer
        # Check if this is pre-attention or post-attention by looking at siblings
        return "norm"

    # Logits / LM head
    if "Logits" in innermost or "lm_head" in innermost.lower():
        return "lm_head"

    # Activation functions
    if "silu" in op_name.lower() or "gelu" in op_name.lower() or "relu" in op_name.lower():
        return "activation"

    return None


def _extract_layer_idx(module_path: list[str]) -> int | None:
    """Extract decoder layer index from module path."""
    for mod in module_path:
        m = _LAYER_IDX_RE.search(mod)
        if m:
            return int(m.group(1) or m.group(2))
    return None


def parse_trace_with_modules(path: str) -> list[dict]:
    """Parse a trace file and annotate each op with its module hierarchy.

    Requires the trace to be captured with `with_stack=True` so that
    python_function events (including nn.Module annotations) are present.

    Returns:
        List of dicts:
        [{name, dur_us, device_time_us, module_path, layer_idx, role,
          input_shapes, count}, ...]
        Aggregated by (module_path_normalized, role, op_name).
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

    # Separate event types
    cpu_ops = []
    module_events = []  # nn.Module: events from python_function
    kernel_events = []

    for evt in events:
        cat = evt.get("cat", "")
        ph = evt.get("ph", "")
        if ph != "X":
            continue
        if cat == "cpu_op":
            cpu_ops.append(evt)
        elif cat == "python_function":
            name = evt.get("name", "")
            if name.startswith("nn.Module:"):
                module_events.append(evt)
        elif cat in _KERNEL_CATEGORIES:
            kernel_events.append(evt)

    if not cpu_ops:
        return []

    # Determine the worker thread (cpu_ops all share a tid)
    worker_tid = cpu_ops[0].get("tid")

    # Filter module events to same thread and sort by timestamp
    module_events = sorted(
        [e for e in module_events if e.get("tid") == worker_tid],
        key=lambda e: e.get("ts", 0),
    )

    # Build interval tree for module events (simple list-based for now)
    # Each entry: (start, end, class_name_with_idx)
    module_intervals = []
    for evt in module_events:
        ts = evt.get("ts", 0)
        dur = evt.get("dur", 0)
        m = _MODULE_RE.match(evt.get("name", ""))
        if m:
            module_intervals.append((ts, ts + dur, m.group(1)))

    # Sort by start time, then by duration descending (outermost first)
    module_intervals.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # For each cpu_op, find enclosing modules
    # Aggregate by (normalized_module_path, op_name)
    op_agg: dict[tuple, dict] = {}

    # Compute total kernel time for device attribution
    total_kernel_time = sum(e.get("dur", 0) for e in kernel_events)

    for evt in cpu_ops:
        name = evt.get("name", "")
        if _is_overhead_event(name):
            continue

        ts = evt.get("ts", 0)
        dur = evt.get("dur", 0)
        args = evt.get("args", {})

        # Find enclosing nn.Module events
        enclosing = []
        for (start, end, class_name) in module_intervals:
            if start <= ts and ts <= end:
                enclosing.append((end - start, class_name))

        # Sort by duration descending (outermost first)
        enclosing.sort(key=lambda x: -x[0])
        raw_path = [cls for (_, cls) in enclosing]
        norm_path = [_strip_instance_idx(cls) for cls in raw_path]

        # Extract layer index and role
        layer_idx = _extract_layer_idx(raw_path)
        role = _infer_role(norm_path, name)

        # Input shapes
        input_shapes_str = ""
        if "Input Dims" in args:
            input_shapes_str = str(args["Input Dims"])

        # Build aggregation key: (tuple of normalized path, role or op_name)
        # Role takes precedence for matching; fall back to op_name
        agg_key = (tuple(norm_path), role or name, name, layer_idx)

        if agg_key not in op_agg:
            op_agg[agg_key] = {
                "name": name,
                "module_path": norm_path,
                "layer_idx": layer_idx,
                "role": role,
                "cpu_time_us": 0.0,
                "device_time_us": 0.0,
                "count": 0,
                "input_shapes": input_shapes_str,
            }

        op_agg[agg_key]["cpu_time_us"] += dur
        op_agg[agg_key]["count"] += 1

    # Attribute device time proportionally (same approach as parse_trace_file)
    compute_ops = {}
    total_compute_cpu = 0
    for key, op in op_agg.items():
        backend, _ = classify_op(op["name"], device_type="xpu",
                                 self_device_time_us=1)
        if backend.value in ("vllm-xpu-kernels", "torch-xpu-ops", "triton"):
            compute_ops[key] = op
            total_compute_cpu += op["cpu_time_us"]

    if total_compute_cpu > 0 and total_kernel_time > 0:
        for key, op in compute_ops.items():
            frac = op["cpu_time_us"] / total_compute_cpu
            op["device_time_us"] = total_kernel_time * frac
    elif total_kernel_time > 0 and compute_ops:
        per_op = total_kernel_time / len(compute_ops)
        for key, op in compute_ops.items():
            op["device_time_us"] = per_op

    return list(op_agg.values())
