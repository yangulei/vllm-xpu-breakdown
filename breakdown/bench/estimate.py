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

Two properties of that roofline are deliberate:

* **The bound comes from arithmetic intensity, not from the measurement.** An
  op is compute-bound iff its AI (FLOP/byte) is at or above the machine balance
  ``peak FLOPS / peak bandwidth``. Comparing the two achieved utilizations and
  taking the larger - the previous rule - labels a GEMM that ran at 30 % of
  peak FLOPS "memory-bound" and a pure-gather kernel "compute-bound".
* **A cache-resident op is measured against cache bandwidth.** The benchmark
  repeats a kernel on the same operands inside one timed window, so an op whose
  footprint fits in the last-level cache legitimately exceeds the DRAM peak.
  Charging it to DRAM produced "utilization 300 % of peak" warnings that said
  nothing about the kernel; the honest roof is the cache one.
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


def cache_resident(nbytes: float, peaks: dict[str, float]) -> bool:
    """Does this op's working set fit in the device's last-level cache?

    ``nbytes`` is the op's analytic traffic, which for a single replayed call is
    also its footprint (each operand is read once). The benchmark repeats a
    kernel inside one timed window on the *same* operands, so an op whose
    footprint fits in cache is served by the cache on every repetition after the
    first - and is bounded by cache bandwidth, not DRAM bandwidth.
    """
    cap = float(peaks.get("cache_bytes") or 0)
    return bool(cap) and 0 < nbytes <= cap


def effective_bw_gbs(nbytes: float, peaks: dict[str, float]
                     ) -> tuple[float, str]:
    """``(bandwidth roof GB/s, which memory level)`` for this op's footprint.

    A kernel whose operands are cache-resident routinely exceeds the DRAM peak;
    measuring it against DRAM produced "utilization 300 % of peak" warnings that
    said nothing about the kernel. The right roof for such an op is the
    last-level-cache bandwidth (see :data:`breakdown.bench.devices.SKU_PEAKS`).
    """
    cbw = float(peaks.get("cache_bw_gbs") or 0)
    if cbw and cache_resident(nbytes, peaks):
        return cbw, "cache"
    return float(peaks["bw_gbs"]), "dram"


def ridge_ai(peaks: dict[str, float], bw_gbs: float | None = None) -> float:
    """Machine balance in FLOP/byte: the roofline's ridge point.

    An op with a higher arithmetic intensity than this is compute-bound, one
    below it is memory-bound. That comparison - **not** "whichever utilization
    number comes out larger" - is what defines the bound.
    """
    bw = float(bw_gbs if bw_gbs is not None else peaks["bw_gbs"])
    if bw <= 0:
        return 0.0
    return (float(peaks["tflops"]) * 1e12) / (bw * 1e9)


def op_ai(flops: float, nbytes: float) -> float:
    """The op's arithmetic intensity in FLOP/byte."""
    if nbytes <= 0:
        return float("inf") if flops > 0 else 0.0
    return flops / nbytes


def bound_of(flops: float, nbytes: float,
             peaks: dict[str, float]) -> tuple[str, str]:
    """``(bound, memory level)`` from the op's AI against the machine balance.

    The previous rule compared the two *utilizations* and took the larger, which
    mislabels ops systematically: a GEMM measured below peak FLOPS came out
    "memory" and a bandwidth-starved gather came out "compute". The bound is a
    property of the op and the machine, not of how well the kernel did.
    """
    bw, level = effective_bw_gbs(nbytes, peaks)
    if flops <= 0:
        return ("memory" if nbytes > 0 else "unknown"), level
    if nbytes <= 0:
        return "compute", level
    return ("compute" if op_ai(flops, nbytes) >= ridge_ai(peaks, bw)
            else "memory"), level


def kernel_seconds(flops: float, nbytes: float, peak_tflops: float,
                   peak_bw_gbs: float, util: float = 0.5) -> float:
    """Roofline runtime of one case at ``util`` of peak."""
    util = max(min(util, 1.0), 1e-3)
    t_c = (flops / (peak_tflops * 1e12 * util)) if flops > 0 else 0.0
    t_m = (nbytes / (peak_bw_gbs * 1e9 * util)) if nbytes > 0 else 0.0
    return max(t_c, t_m)


def roofline_bound_us(flops: float, nbytes: float,
                      peaks: dict[str, float]) -> tuple[float, str]:
    """``(fastest possible microseconds, bound)`` at 100 % of peak.

    The memory term uses the roof the op's footprint actually sees (cache or
    DRAM), so a cache-resident kernel is not told it beat the speed of light.
    """
    bound, _ = bound_of(flops, nbytes, peaks)
    bw, _ = effective_bw_gbs(nbytes, peaks)
    t_c = (flops / (peaks["tflops"] * 1e12)) if flops > 0 else 0.0
    t_m = (nbytes / (bw * 1e9)) if nbytes > 0 else 0.0
    return ((t_c if bound == "compute" else t_m) * 1e6), bound


def utilization(latency_us: float, flops: float, nbytes: float,
                peaks: dict[str, float]) -> tuple[float, str]:
    """``(achieved fraction of the relevant roof, which roof)``.

    The roof is selected by the op's arithmetic intensity (see :func:`bound_of`)
    and, for a memory-bound op, by whether its footprint is cache-resident.
    """
    util, bound, _ = utilization_detail(latency_us, flops, nbytes, peaks)
    return util, bound


def utilization_detail(latency_us: float, flops: float, nbytes: float,
                       peaks: dict[str, float]) -> tuple[float, str, str]:
    """``(utilization, bound, memory level)`` for a measured case."""
    bound, level = bound_of(flops, nbytes, peaks)
    if latency_us <= 0:
        return 0.0, bound, level
    secs = latency_us / 1e6
    if bound == "compute":
        peak = float(peaks["tflops"]) * 1e12
        return ((flops / secs) / peak if peak > 0 else 0.0), bound, level
    if bound == "memory":
        bw, level = effective_bw_gbs(nbytes, peaks)
        roof = bw * 1e9
        return ((nbytes / secs) / roof if roof > 0 else 0.0), bound, level
    return 0.0, bound, level


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
