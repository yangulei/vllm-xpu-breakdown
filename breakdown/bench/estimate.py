# SPDX-License-Identifier: Apache-2.0
"""Per-op time budgets for the replay runner.

A worker process is killed when it exceeds its budget, and a killed worker
loses nothing already written (results stream case by case) - but it does lose
the cases it had not reached. So the budget must be *derived*, not guessed:

    timeout = startup + cases x (measurement budget + per-case overhead) x safety

``startup`` is the process's import cost (torch + vLLM is tens of seconds, not
milliseconds) and ``per-case overhead`` covers argument materialization, warmup,
the probe and - on a cold cache - kernel compilation. Both are **calibrated
from previous runs' actual wall time** when any exist, so the estimate improves
with use instead of staying a constant someone picked once.

The roofline helpers here serve a second purpose: turning an op's analytic
work (FLOPs / bytes) into the *lower bound* a measurement should respect. A
replayed latency below the roofline bound means the replay did not do the work
(an early-exit kernel, an empty index map), which the report flags.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

#: Seconds a worker spends importing torch/vLLM before its first case.
DEFAULT_STARTUP_S = 60.0
#: Seconds of non-measurement work per case (materialize, warmup, probe).
DEFAULT_CASE_OVERHEAD_S = 3.0
#: Multiplier applied to the estimate before it becomes a hard kill.
DEFAULT_SAFETY = 3.0
#: Never kill a worker faster than this, whatever the estimate says.
MIN_TIMEOUT_S = 120
MAX_TIMEOUT_S = 7200
#: Conservative rate at which operand memory can be allocated and filled.
ALLOC_BYTES_PER_S = 2e9


def kernel_seconds(flops: float, nbytes: float, peak_tflops: float,
                   peak_bw_gbs: float, util: float = 0.5) -> float:
    """Roofline runtime of one case at ``util`` of peak."""
    util = max(min(util, 1.0), 1e-3)
    t_c = (flops / (peak_tflops * 1e12 * util)) if flops > 0 else 0.0
    t_m = (nbytes / (peak_bw_gbs * 1e9 * util)) if nbytes > 0 else 0.0
    return max(t_c, t_m)


def roofline_bound_us(flops: float, nbytes: float,
                      peaks: dict[str, float]) -> tuple[float, str]:
    """``(fastest possible microseconds, bound)`` at 100 % of peak."""
    t_c = (flops / (peaks["tflops"] * 1e12)) if flops > 0 else 0.0
    t_m = (nbytes / (peaks["bw_gbs"] * 1e9)) if nbytes > 0 else 0.0
    if t_c >= t_m:
        return t_c * 1e6, "compute"
    return t_m * 1e6, "memory"


def utilization(latency_us: float, flops: float, nbytes: float,
                peaks: dict[str, float]) -> tuple[float, str]:
    """``(achieved fraction of peak, which roof)`` for a measured case."""
    if latency_us <= 0:
        return 0.0, "unknown"
    secs = latency_us / 1e6
    u_c = (flops / secs) / (peaks["tflops"] * 1e12) if flops > 0 else 0.0
    u_m = (nbytes / secs) / (peaks["bw_gbs"] * 1e9) if nbytes > 0 else 0.0
    return (u_c, "compute") if u_c >= u_m else (u_m, "memory")


def op_timeout(n_cases: int, budget_s: float,
               startup_s: float = DEFAULT_STARTUP_S,
               case_overhead_s: float = DEFAULT_CASE_OVERHEAD_S,
               safety: float = DEFAULT_SAFETY,
               alloc_bytes: float = 0.0) -> int:
    """Budget for one op's worker.

    ``alloc_bytes`` is the total operand memory the op's cases allocate; an
    ``lm_head`` matmul materializes a several-hundred-megabyte weight per case,
    which costs far more wall time than the measurement itself and is the
    difference between a comfortable budget and a killed worker.
    """
    est = (startup_s + n_cases * (budget_s + case_overhead_s)
           + alloc_bytes / ALLOC_BYTES_PER_S)
    return int(max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, est * safety)))


def previous_run_results(root: str, limit: int = 5) -> list[dict[str, Any]]:
    """``run_result.json`` of the most recent runs, newest first."""
    if not os.path.isdir(root):
        return []
    out: list[dict[str, Any]] = []
    for name in sorted(os.listdir(root), reverse=True):
        if name.startswith(".") or len(out) >= limit:
            continue
        path = os.path.join(root, name, "run_result.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


def calibrate(run_results: Iterable[dict[str, Any]]
              ) -> tuple[float, float]:
    """``(startup_s, case_overhead_s)`` fitted to previous runs' wall time.

    A least-squares fit would be overkill for two parameters and a handful of
    points: ops with a single case pin the startup cost, and the slope of
    ``(seconds - startup) / cases`` over the multi-case ops pins the per-case
    overhead.
    """
    singles: list[float] = []
    multi: list[tuple[int, float]] = []
    for res in run_results or []:
        for op in res.get("ops") or []:
            cases = int(op.get("cases") or 0)
            secs = float(op.get("seconds") or 0)
            if cases <= 0 or secs <= 0 or op.get("timed_out"):
                continue
            if cases == 1:
                singles.append(secs)
            else:
                multi.append((cases, secs))
    startup = min(singles) if singles else DEFAULT_STARTUP_S
    if not multi:
        return startup, DEFAULT_CASE_OVERHEAD_S
    slopes = [max((s - startup) / c, 0.0) for c, s in multi]
    slopes.sort()
    per_case = slopes[len(slopes) // 2]
    return startup, max(per_case, 0.5)


def plan(case_counts: dict[str, int], budget_s: float, root: str | None = None,
         safety: float = DEFAULT_SAFETY,
         alloc_bytes: dict[str, float] | None = None) -> dict[str, int]:
    """op -> timeout seconds, calibrated against previous runs when present."""
    startup, per_case = (calibrate(previous_run_results(root))
                         if root else (DEFAULT_STARTUP_S,
                                       DEFAULT_CASE_OVERHEAD_S))
    return {op: op_timeout(n, budget_s, startup, per_case, safety,
                           (alloc_bytes or {}).get(op, 0.0))
            for op, n in case_counts.items()}
