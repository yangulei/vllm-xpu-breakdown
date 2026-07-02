# SPDX-License-Identifier: Apache-2.0
"""Profiler wrapper — captures op-level traces from vLLM inference."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from .classifier import Backend, ClassificationResult, OpRecord, classify_op


@dataclass
class ProfileConfig:
    """Configuration for a profiling run."""
    output_dir: str = "output"
    # Activities to trace
    trace_xpu: bool = True
    trace_cpu: bool = True
    trace_cuda: bool = True
    # Profiler options
    record_shapes: bool = True
    with_stack: bool = True
    with_flops: bool = True
    profile_memory: bool = False
    # Schedule: warmup N steps, then profile M steps
    warmup_steps: int = 1
    active_steps: int = 1
    # Top-N for reports
    top_n: int = 30


def _detect_device() -> str:
    """Return 'cuda', 'xpu', or 'cpu' based on available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _get_activities(config: ProfileConfig) -> list[ProfilerActivity]:
    acts = []
    if config.trace_cpu:
        acts.append(ProfilerActivity.CPU)
    device = _detect_device()
    if device == "cuda" and config.trace_cuda:
        if hasattr(ProfilerActivity, "CUDA"):
            acts.append(ProfilerActivity.CUDA)
    elif device == "xpu" and config.trace_xpu:
        if hasattr(ProfilerActivity, "XPU"):
            acts.append(ProfilerActivity.XPU)
    return acts


def _sync_device():
    """Synchronize the active accelerator to ensure all async work is captured."""
    try:
        device = _detect_device()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "xpu":
            torch.xpu.synchronize()
    except Exception:
        pass


# Events that are profiler infrastructure, not real ops. The classification
# logic lives in the torch-free ``trace_common`` module so offline trace parsing
# doesn't require torch; re-exported here for backward compatibility.
from .trace_common import (  # noqa: F401
    _OVERHEAD_EVENTS,
    _OVERHEAD_PREFIXES,
    _is_overhead_event,
)


@contextmanager
def profile_context(config: ProfileConfig):
    """Context manager that yields a torch.profiler.profile instance.

    Usage:
        with profile_context(config) as prof:
            # run inference
            prof.step()  # call after each iteration
    """
    os.makedirs(config.output_dir, exist_ok=True)
    activities = _get_activities(config)

    trace_path = os.path.join(config.output_dir, "trace.json")

    def trace_handler(p):
        p.export_chrome_trace(trace_path)

    prof = profile(
        activities=activities,
        schedule=schedule(
            wait=0,
            warmup=config.warmup_steps,
            active=config.active_steps,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
        record_shapes=config.record_shapes,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
        profile_memory=config.profile_memory,
    )

    with prof:
        yield prof


@contextmanager
def simple_profile_context(config: ProfileConfig):
    """Simple profiler context — profiles the entire block, no schedule.

    Use this when doing warmup externally. Ensures XPU sync before exit.

    Usage:
        # warmup outside
        model(input)
        sync_device()
        # profile
        with simple_profile_context(config) as prof:
            model(input)
    """
    os.makedirs(config.output_dir, exist_ok=True)
    activities = _get_activities(config)
    trace_path = os.path.join(config.output_dir, "trace.json")

    prof = profile(
        activities=activities,
        record_shapes=config.record_shapes,
        with_stack=config.with_stack,
        with_flops=config.with_flops,
        profile_memory=config.profile_memory,
    )

    with prof:
        yield prof
        # Sync device INSIDE the profiler context to capture all async work
        _sync_device()

    # Export trace after profiler finishes
    try:
        prof.export_chrome_trace(trace_path)
    except Exception:
        pass


def parse_events(prof: profile, config: ProfileConfig | None = None,
                 filter_overhead: bool = True) -> ClassificationResult:
    """Parse profiler events into classified OpRecords.

    Uses key_averages for the summary view (aggregated by op name + shape).
    Filters out profiler overhead events by default.
    """
    result = ClassificationResult()

    # Use key_averages grouped by input shape for richer info
    try:
        averages = prof.key_averages(group_by_input_shape=True)
    except Exception:
        averages = prof.key_averages()

    for evt in averages:
        name = evt.key

        # Skip overhead events
        if filter_overhead and _is_overhead_event(name):
            continue

        cpu_time = evt.cpu_time_total
        device_time = evt.device_time_total  # works for both CUDA and XPU

        # Determine device type from the event
        device_type = ""
        if hasattr(evt, "device_type"):
            dt = str(evt.device_type).lower()
            if "cuda" in dt:
                device_type = "cuda"
            elif "xpu" in dt:
                device_type = "xpu"
            elif "cpu" in dt:
                device_type = "cpu"
            else:
                device_type = dt

        # Use self device time and total device time for classification
        self_device = getattr(evt, "self_device_time_total", device_time)

        backend, category = classify_op(
            name,
            device_type=device_type,
            self_device_time_us=self_device,
            device_time_us=device_time,
        )

        input_shapes_str = ""
        if hasattr(evt, "input_shapes") and evt.input_shapes:
            input_shapes_str = str(evt.input_shapes)

        op = OpRecord(
            name=name,
            backend=backend,
            category=category,
            cpu_time_us=cpu_time,
            device_time_us=device_time,
            count=evt.count,
            input_shapes=input_shapes_str,
            device_type=device_type,
        )
        result.ops.append(op)
        result.total_device_time_us += device_time
        result.total_cpu_time_us += cpu_time

    return result


def parse_raw_events(prof: profile) -> list[dict]:
    """Parse raw profiler events into a list of dicts for detailed analysis.

    This preserves per-invocation data (not aggregated).
    """
    records = []
    try:
        events = prof.events()
    except Exception:
        return records

    for evt in events:
        name = evt.key
        if _is_overhead_event(name):
            continue

        device_time = getattr(evt, "device_time_total", 0) or 0
        self_device = getattr(evt, "self_device_time_total", 0) or 0
        device_type = str(getattr(evt, "device_type", ""))

        backend, category = classify_op(
            name,
            device_type=device_type,
            self_device_time_us=self_device,
            device_time_us=device_time,
        )

        records.append({
            "name": name,
            "backend": backend.value,
            "category": category,
            "cpu_time_us": evt.cpu_time_total,
            "device_time_us": device_time,
            "self_cpu_time_us": evt.self_cpu_time_total,
            "self_device_time_us": self_device,
            "device_type": device_type,
            "thread_id": getattr(evt, "thread", 0),
        })

    return records


def export_classified_events(result: ClassificationResult,
                             output_dir: str) -> None:
    """Export classified events to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "classified_ops.json")

    data = {
        "total_device_time_us": result.total_device_time_us,
        "total_cpu_time_us": result.total_cpu_time_us,
        "ops": [asdict(op) for op in result.ops],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
