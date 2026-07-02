# SPDX-License-Identifier: Apache-2.0
"""Report generators — console, CSV, JSON output."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict

from .classifier import Backend, ClassificationResult, OpRecord


def _pct(part: float, total: float) -> str:
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def _time_fmt(us: float) -> str:
    """Format microseconds into a human-readable string."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    elif us >= 1_000:
        return f"{us / 1_000:.2f}ms"
    else:
        return f"{us:.0f}µs"


# ---- Console Report ----

def print_summary(result: ClassificationResult, top_n: int = 30) -> str:
    """Generate and return a console-friendly summary report."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  vLLM Ops/Kernels Breakdown Report")
    lines.append("=" * 80)
    lines.append("")

    total_dev = result.total_device_time_us
    total_cpu = result.total_cpu_time_us

    lines.append(f"  Total device time: {_time_fmt(total_dev)}")
    lines.append(f"  Total CPU time:    {_time_fmt(total_cpu)}")
    lines.append(f"  Total ops:         {len(result.ops)}")
    lines.append("")

    # ---- Backend breakdown ----
    lines.append("-" * 80)
    lines.append("  Backend Breakdown (by device time)")
    lines.append("-" * 80)

    backend_stats: dict[Backend, dict] = {}
    for backend in Backend:
        backend_stats[backend] = {
            "device_time_us": 0.0,
            "cpu_time_us": 0.0,
            "count": 0,
            "ops": 0,
        }

    for op in result.ops:
        b = backend_stats[op.backend]
        b["device_time_us"] += op.device_time_us
        b["cpu_time_us"] += op.cpu_time_us
        b["count"] += op.count
        b["ops"] += 1

    header = f"  {'Backend':<25} {'DevTime':>12} {'%DevTime':>10} {'#Ops':>8} {'#Calls':>10}"
    lines.append(header)
    lines.append("  " + "-" * 67)

    for backend in Backend:
        s = backend_stats[backend]
        if s["ops"] == 0:
            continue
        lines.append(
            f"  {backend.value:<25} "
            f"{_time_fmt(s['device_time_us']):>12} "
            f"{_pct(s['device_time_us'], total_dev):>10} "
            f"{s['ops']:>8} "
            f"{s['count']:>10}"
        )
    lines.append("")

    # ---- Category breakdown ----
    lines.append("-" * 80)
    lines.append("  Category Breakdown (by device time)")
    lines.append("-" * 80)

    cat_stats: dict[str, dict] = defaultdict(
        lambda: {"device_time_us": 0.0, "count": 0, "ops": 0}
    )
    for op in result.ops:
        c = cat_stats[op.category]
        c["device_time_us"] += op.device_time_us
        c["count"] += op.count
        c["ops"] += 1

    sorted_cats = sorted(cat_stats.items(),
                         key=lambda x: x[1]["device_time_us"], reverse=True)

    header = f"  {'Category':<35} {'DevTime':>12} {'%DevTime':>10} {'#Ops':>8}"
    lines.append(header)
    lines.append("  " + "-" * 67)
    for cat, s in sorted_cats:
        if s["device_time_us"] == 0 and s["ops"] < 3:
            continue
        lines.append(
            f"  {cat:<35} "
            f"{_time_fmt(s['device_time_us']):>12} "
            f"{_pct(s['device_time_us'], total_dev):>10} "
            f"{s['ops']:>8}"
        )
    lines.append("")

    # ---- Top-N ops by device time ----
    lines.append("-" * 80)
    lines.append(f"  Top {top_n} Ops by Device Time")
    lines.append("-" * 80)

    sorted_ops = sorted(result.ops, key=lambda o: o.device_time_us,
                        reverse=True)

    header = (
        f"  {'Op Name':<40} {'Backend':<20} {'DevTime':>12} "
        f"{'%DevTime':>8} {'#Calls':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * 90)

    for op in sorted_ops[:top_n]:
        name = op.name[:40]
        lines.append(
            f"  {name:<40} {op.backend.value:<20} "
            f"{_time_fmt(op.device_time_us):>12} "
            f"{_pct(op.device_time_us, total_dev):>8} "
            f"{op.count:>8}"
        )
    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report


# ---- CSV Export ----

def export_csv(result: ClassificationResult, output_dir: str) -> str:
    """Export ops to CSV. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ops_breakdown.csv")

    sorted_ops = sorted(result.ops, key=lambda o: o.device_time_us,
                        reverse=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "op_name", "backend", "category", "device_time_us",
            "cpu_time_us", "count", "device_time_pct", "input_shapes",
        ])
        total = result.total_device_time_us
        for op in sorted_ops:
            pct = (op.device_time_us / total * 100) if total > 0 else 0
            writer.writerow([
                op.name, op.backend.value, op.category,
                f"{op.device_time_us:.1f}", f"{op.cpu_time_us:.1f}",
                op.count, f"{pct:.2f}", op.input_shapes,
            ])
    return path


# ---- JSON Export ----

def export_json(result: ClassificationResult, output_dir: str) -> str:
    """Export full classification result to JSON. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ops_breakdown.json")

    total_dev = result.total_device_time_us

    # Backend summary
    backend_summary = {}
    for backend in Backend:
        ops = [o for o in result.ops if o.backend == backend]
        dev_time = sum(o.device_time_us for o in ops)
        backend_summary[backend.value] = {
            "device_time_us": dev_time,
            "pct_device_time": (dev_time / total_dev * 100) if total_dev > 0 else 0,
            "num_ops": len(ops),
            "num_calls": sum(o.count for o in ops),
        }

    data = {
        "summary": {
            "total_device_time_us": total_dev,
            "total_cpu_time_us": result.total_cpu_time_us,
            "total_unique_ops": len(result.ops),
            "backends": backend_summary,
        },
        "ops": sorted(
            [asdict(op) for op in result.ops],
            key=lambda o: o["device_time_us"],
            reverse=True,
        ),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path
