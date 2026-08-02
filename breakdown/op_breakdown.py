# SPDX-License-Identifier: Apache-2.0
"""Flat op breakdown, derived from the reconstructed graph.

The reconstructed graph is the single source of truth for what ran: it links
every device kernel to its launch site, restricts the decode phase to
steady-state full-batch steps, and collapses repeated layers into a repeat
count. Aggregating *it* therefore gives a flat "which op, which backend, how
much device time" view that agrees with the tree the user is looking at.

This replaces the retired ``trace_parser`` pipeline, which parsed the trace a
second time and attributed device time by splitting each kernel's duration
across ops in proportion to their CPU time — a guess that disagreed with the
graph whenever it mattered.
"""
from __future__ import annotations

from typing import Any, Iterator

from .classifier import Backend


def iter_ops(node: dict | None, repeat: int = 1) -> Iterator[tuple[dict, int]]:
    """Yield ``(op, effective_repeat)`` for every op under ``node``.

    ``effective_repeat`` multiplies the repeat counts of all enclosing modules,
    so an op inside a layer group collapsed to ``x57`` is counted 57 times —
    which is what "how much time does this op cost the model" means.
    """
    if not node:
        return
    repeat = repeat * (node.get("repeat_count", 1) or 1)
    for op in node.get("ops") or []:
        yield op, repeat
    for child in node.get("children") or []:
        yield from iter_ops(child, repeat)


def summarize_ops(graph: dict | None,
                  phases: tuple[str, ...] = ("prefill", "decode"),
                  ) -> list[dict[str, Any]]:
    """Aggregate a graph's ops by dispatch name, hottest first.

    Each entry carries the op's backend, its total device time (repeat-weighted)
    and which phases it ran in.
    """
    agg: dict[str, dict[str, Any]] = {}
    for phase in phases:
        for op, repeat in iter_ops((graph or {}).get(phase)):
            name = op.get("name") or ""
            if not name:
                continue
            entry = agg.setdefault(name, {
                "op": name,
                "backend": op.get("backend", ""),
                "device_time_us": 0.0,
                "calls": 0,
                "phases": [],
            })
            entry["device_time_us"] += float(op.get("device_time_us") or 0.0) * repeat
            entry["calls"] += repeat
            if phase not in entry["phases"]:
                entry["phases"].append(phase)
    ops = sorted(agg.values(), key=lambda e: -e["device_time_us"])
    for entry in ops:
        entry["device_time_us"] = round(entry["device_time_us"], 2)
    return ops


def backend_totals(ops: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-backend device-time totals and shares over ``summarize_ops`` output.

    This is the breakdown the tool is named for: which backend — vllm-xpu-kernels,
    torch-xpu-ops, triton, ccl, … — owns the model's device time.
    """
    total = sum(o["device_time_us"] for o in ops)
    totals: dict[str, dict[str, Any]] = {}
    for backend in Backend:
        matching = [o for o in ops if o["backend"] == backend.value]
        device_us = sum(o["device_time_us"] for o in matching)
        totals[backend.value] = {
            "device_time_us": round(device_us, 2),
            "pct": round(device_us / total * 100, 1) if total > 0 else 0.0,
            "num_ops": len(matching),
            "num_calls": sum(o["calls"] for o in matching),
        }
    return totals
