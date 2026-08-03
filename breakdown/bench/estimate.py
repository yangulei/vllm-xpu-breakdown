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

The roofline that turns an op's analytic work into a predicted latency lives
in :mod:`breakdown.cost`, with the rest of the cost model. This module used to
re-export a dozen of its names so callers had "one import site"; that only
made ``estimate`` look like it owned a cost model it does not, and half the
re-exports had no caller. Callers ask :mod:`breakdown.cost` directly.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

from ..cost import compute_peak, effective_bw_gbs, kernel_seconds

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

#: Bounds of the *adaptive* per-case measurement budget (seconds). There is no
#: user-facing "budget / case" knob: the budget a case needs is a property of
#: the shape being replayed, which the profile already tells us.
MIN_BUDGET_S = 0.1
MAX_BUDGET_S = 2.0
#: Timed windows a case should get. ``timing.plan_window`` clamps the window
#: count to [MIN_WINDOWS, MAX_WINDOWS]; the budget only has to be long enough
#: that a slow kernel still gets a statistically usable number of them.
TARGET_WINDOWS = 12
#: Fraction of peak a replayed kernel is assumed to reach when its latency has
#: to be *predicted* from the analytic shape cost (no traced time available).
#: Deliberately pessimistic: under-predicting the latency under-budgets the
#: measurement, which costs windows; over-predicting only costs wall time.
ASSUMED_UTIL = 0.25


def case_seconds(case: Any, peaks: dict[str, float]) -> float:
    """Predicted device seconds of one replayed call of ``case``.

    The profile is the best predictor it has: a case whose swept shapes equal
    the recorded ones carries the trace's own device time. Everything else is
    predicted from the case's analytic work at :data:`ASSUMED_UTIL` of the roof
    the op can actually reach.
    """
    traced = float(getattr(case, "traced_device_time_us", 0.0) or 0.0)
    if getattr(case, "traced_comparable", False) and traced > 0:
        return traced / 1e6
    op = getattr(case, "op", None)
    flops = float(getattr(case, "flops", 0.0) or 0.0)
    nbytes = float(getattr(case, "nbytes", 0.0) or 0.0)
    bw, _ = effective_bw_gbs(nbytes, peaks)
    return kernel_seconds(flops, nbytes, compute_peak(peaks, op)[0], bw,
                          ASSUMED_UTIL)


def case_budget(case: Any, peaks: dict[str, float]) -> float:
    """Adaptive measurement budget (seconds) for one case.

    A budget must buy a usable number of timed windows, and a window costs
    ``max(kernel time, TARGET_WINDOW_S)`` - the repetition target that
    amortizes the device-event floor. So the budget scales with the shape being
    replayed instead of being a constant the user is asked to guess.
    """
    from breakdown.bench import timing

    window = max(case_seconds(case, peaks), timing.TARGET_WINDOW_S)
    return max(MIN_BUDGET_S, min(MAX_BUDGET_S, TARGET_WINDOWS * window))


def op_budgets(cases_by_op: dict[str, list], peaks: dict[str, float]
               ) -> dict[str, float]:
    """op -> adaptive budget, sized by the op's most expensive case.

    One worker measures all of an op's cases with one budget, so the op takes
    the largest budget any of its cases needs (bounded by
    :data:`MAX_BUDGET_S`); a cheap shape simply finishes its windows early.
    """
    return {op: round(max((case_budget(c, peaks) for c in cases),
                          default=MIN_BUDGET_S), 3)
            for op, cases in cases_by_op.items()}


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


def plan(case_counts: dict[str, int], budget_s: float | dict[str, float],
         root: str | None = None,
         safety: float = DEFAULT_SAFETY,
         alloc_bytes: dict[str, float] | None = None) -> dict[str, int]:
    """op -> timeout seconds, calibrated against previous runs when present.

    ``budget_s`` may be a single budget or the per-op budgets
    :func:`op_budgets` derived from the profiled shapes, in which case each
    op's timeout follows the budget its own cases were given.
    """
    startup, per_case = (calibrate(previous_run_results(root))
                         if root else (DEFAULT_STARTUP_S,
                                       DEFAULT_CASE_OVERHEAD_S))
    budgets = (budget_s if isinstance(budget_s, dict)
               else {op: float(budget_s) for op in case_counts})
    return {op: op_timeout(n, budgets.get(op, MIN_BUDGET_S), startup, per_case,
                           safety, (alloc_bytes or {}).get(op, 0.0))
            for op, n in case_counts.items()}
