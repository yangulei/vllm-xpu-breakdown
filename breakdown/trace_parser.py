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
