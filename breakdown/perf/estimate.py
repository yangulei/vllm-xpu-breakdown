# SPDX-License-Identifier: Apache-2.0
"""Estimate how long benchmarking an op will take, and size its timeout.

A single fixed timeout cannot fit this pipeline: ``rms_norm`` finishes in under
a minute while ``gemm`` sweeps 113 shapes across three providers. Too small and
a long op is killed *and loses every shape it had already measured* (micro_perf
writes its jsonl at the end of the op); too large and a genuinely hung kernel
holds the GPU for hours.

The estimate is built from what is already known about each shape:

* the Shape Matrix carries the **FLOPs and bytes** of every emitted case;
* a measured record of *any* shape of that op gives its **achieved
  utilization** (:func:`breakdown.perf.rank.utilization`) - i.e. what fraction
  of the device roofline this kernel actually reaches;
* so an unmeasured shape's device latency is ``work / (peak * util)``.

micro_perf then spends a roughly fixed *measurement budget* per case (it picks
its iteration count so measuring takes ~50 ms, clamped to 2..1000 iterations -
100..1000 for flash-attention ops), plus a per-case overhead that is dominated
by kernel compilation, not by the kernel. That overhead is calibrated from the
wall time of previous runs of the same op when history is available, so the
estimate self-corrects run over run.
"""
from __future__ import annotations

import glob
import os
from statistics import median
from typing import Any, Iterable

from breakdown.perf import devices

#: Fraction of the roofline an unmeasured kernel is assumed to reach. Kept
#: pessimistic: under-estimating utilization over-estimates the timeout, which
#: only costs patience, while the opposite loses a run's results.
DEFAULT_UTIL = 0.10

#: micro_perf's ``max_test_time`` - the wall time it targets per measured case.
MEASURE_BUDGET_S = 0.05
MIN_ITERS = 2
#: flash-attention ops force at least 100 iterations (``_min_test_iters``).
ATTN_MIN_ITERS = 100
MAX_ITERS = 1000
ATTN_OPS = ("flash_attention", "msa_sparse_attn")

#: Interpreter + torch + backend init before the first case of an op.
OP_STARTUP_S = 90.0
#: Per-case cost that is not the kernel: tensor allocation, the latency probe,
#: micro_perf's inter-case sleeps and - dominant on XPU - per-shape SYCL/Triton
#: compilation. Calibrated from history when the op has been run before.
DEFAULT_CASE_OVERHEAD_S = 5.0

#: Multiplier on the estimate, and the range a timeout is clamped to.
DEFAULT_SAFETY = 2.0
MIN_TIMEOUT_S = 600
MAX_TIMEOUT_S = 6 * 3600


def kernel_seconds(flops: float, nbytes: float, peak_tflops: float,
                   peak_bw_gbs: float, util: float = DEFAULT_UTIL) -> float:
    """Device latency of one case, from its work and the op's achieved util."""
    util = max(1e-3, min(1.0, util))
    compute = (flops or 0.0) / (peak_tflops * 1e12 * util)
    memory = (nbytes or 0.0) / (peak_bw_gbs * 1e9 * util)
    return max(compute, memory)


def iterations(latency_s: float, op: str = "") -> int:
    """Iteration count micro_perf will pick for a case of this latency."""
    lo = ATTN_MIN_ITERS if op in ATTN_OPS else MIN_ITERS
    if latency_s <= 0:
        return MAX_ITERS
    want = -(-int(MEASURE_BUDGET_S * 1e6) // max(1, int(latency_s * 1e6)))
    return max(lo, min(want, MAX_ITERS))


def case_seconds(latency_s: float, op: str = "",
                 overhead_s: float = DEFAULT_CASE_OVERHEAD_S) -> float:
    """Wall time of one benchmarked case: probe + measurement + overhead."""
    probe = 4 * latency_s          # 2 warmup + 2 probe iterations
    measure = iterations(latency_s, op) * latency_s
    return overhead_s + probe + measure


def op_seconds(op: str, costs: list[dict[str, float]], peaks: dict[str, float],
               util: float = DEFAULT_UTIL, providers: int = 1,
               overhead_s: float = DEFAULT_CASE_OVERHEAD_S,
               cases: int | None = None) -> float:
    """Estimated wall time for benchmarking every case of ``op``."""
    providers = max(1, providers)
    peak_tf, peak_bw = peaks["tflops"], peaks["bw_gbs"]
    if costs:
        total = sum(
            case_seconds(kernel_seconds(c.get("flops", 0.0),
                                        c.get("bytes", 0.0),
                                        peak_tf, peak_bw, util), op, overhead_s)
            for c in costs)
        # cases without a cost row (older runs) still take the overhead
        missing = max(0, (cases or len(costs)) - len(costs))
        total += missing * case_seconds(0.0, op, overhead_s)
    else:
        total = (cases or 0) * case_seconds(0.0, op, overhead_s)
    return OP_STARTUP_S + providers * total


def op_timeout(seconds: float, safety: float = DEFAULT_SAFETY,
               lo: int = MIN_TIMEOUT_S, hi: int = MAX_TIMEOUT_S) -> int:
    """Clamp an estimate into a usable timeout."""
    return int(max(lo, min(hi, round(seconds * max(1.0, safety)))))


# ---------------------------------------------------------------------------
# calibration from measured data
# ---------------------------------------------------------------------------
def op_utilization(records: Iterable[dict[str, Any]],
                   peaks: dict[str, float]) -> dict[str, float]:
    """op -> median achieved utilization, from measured micro_perf records.

    The whole point of using the *measured* util of a profiled shape: a kernel
    that reaches 4 % of roofline keeps reaching roughly 4 % at neighbouring
    shapes, so it predicts its own unmeasured shapes far better than the
    roofline does.
    """
    from breakdown.perf.rank import utilization  # local: avoids a cycle

    per_op: dict[str, list[float]] = {}
    for rec in records:
        op = rec.get("op")
        if not op:
            continue
        util, _bound = utilization(rec, peaks["bw_gbs"], peaks["tflops"])
        if util and util > 0:
            per_op.setdefault(op, []).append(util)
    return {op: median(v) for op, v in per_op.items() if v}


def op_case_overhead(run_results: Iterable[dict[str, Any]]
                     ) -> dict[str, float]:
    """op -> measured non-kernel seconds per case, from previous runs.

    Compile time per shape dwarfs the kernel on a first XPU run, and only a
    previous run of the same op can say by how much.
    """
    per_op: dict[str, list[float]] = {}
    for res in run_results:
        for o in res.get("ops") or []:
            cases, secs = o.get("cases") or 0, o.get("seconds") or 0.0
            if not o.get("ok") or cases <= 0 or secs <= 0:
                continue
            per_op.setdefault(o["op"], []).append(
                max(0.0, (secs - OP_STARTUP_S)) / cases)
    return {op: median(v) for op, v in per_op.items() if v}


def previous_run_results(perf_root: str, limit: int = 5) -> list[dict]:
    """``run_result.json`` of the most recent runs (newest first)."""
    import json

    paths = sorted(glob.glob(os.path.join(perf_root, "*", "reports",
                                          "run_result.json")),
                   key=os.path.getmtime, reverse=True)
    out = []
    for p in paths[:limit]:
        try:
            with open(p) as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


def provider_count(op: str, micro_perf_dir: str | None = None,
                   backend: str = "INTEL") -> int:
    """How many vendor impls micro_perf will benchmark for this op.

    Each provider re-runs every case, so it multiplies the op's wall time.
    """
    mp = micro_perf_dir or (str(devices.micro_perf_dir())
                            if devices.micro_perf_dir() else None)
    if not mp:
        return 1
    hits = glob.glob(os.path.join(mp, "vendor_ops", backend, "ops", "*",
                                  f"{op}.py"))
    return max(1, len(hits))


def plan_for_run(workloads_dir: str, perf_root: str | None = None,
                 sku: str = devices.DEFAULT_SKU, backend: str = "INTEL",
                 safety: float = DEFAULT_SAFETY,
                 records: Iterable[dict[str, Any]] | None = None,
                 ) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Per-op timeouts for a run's emitted workloads.

    Calibrates against the most recent previous runs found under ``perf_root``
    (their measured utilization and per-case wall time), so the second run of a
    model is budgeted from what the first one actually cost.
    """
    from breakdown.perf import reports as perf_reports, store, workloads as wl

    root = perf_root or store.perf_root()
    costs = wl.read_costs(workloads_dir)
    counts = wl.case_counts(workloads_dir)
    prev = previous_run_results(root)
    recs = list(records) if records is not None else []
    if not recs:
        for res in prev:
            rdir = res.get("reports_dir")
            if rdir and os.path.isdir(rdir):
                try:
                    recs += perf_reports.records(res.get("backend") or backend,
                                                 rdir)
                except Exception:  # noqa: BLE001 - calibration is best-effort
                    continue
    return plan(costs, case_counts=counts, records=recs, run_results=prev,
                sku=sku, backend=backend, safety=safety)


def plan(costs: dict[str, list[dict[str, float]]],
         case_counts: dict[str, int] | None = None,
         records: Iterable[dict[str, Any]] | None = None,
         run_results: Iterable[dict[str, Any]] | None = None,
         sku: str = devices.DEFAULT_SKU, backend: str = "INTEL",
         micro_perf_dir: str | None = None, safety: float = DEFAULT_SAFETY,
         floor: int = MIN_TIMEOUT_S, ceiling: int = MAX_TIMEOUT_S,
         ) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Per-op timeouts plus the estimate that produced each one.

    Returns ``(timeouts, detail)``; ``detail[op]`` carries the inputs so a run
    can explain why an op was given the budget it was.
    """
    pk = devices.peaks(sku)
    utils = op_utilization(records or [], pk)
    overheads = op_case_overhead(run_results or [])
    counts = dict(case_counts or {})
    for op, rows in costs.items():
        counts.setdefault(op, len(rows))

    timeouts: dict[str, int] = {}
    detail: dict[str, dict[str, Any]] = {}
    for op, n in counts.items():
        if n <= 0:
            continue
        util = utils.get(op, DEFAULT_UTIL)
        overhead = overheads.get(op, DEFAULT_CASE_OVERHEAD_S)
        providers = provider_count(op, micro_perf_dir, backend)
        est = op_seconds(op, costs.get(op) or [], pk, util=util,
                         providers=providers, overhead_s=overhead, cases=n)
        timeouts[op] = op_timeout(est, safety, floor, ceiling)
        detail[op] = {
            "cases": n, "providers": providers, "util": round(util, 4),
            "case_overhead_s": round(overhead, 2),
            "estimated_s": round(est, 1), "timeout_s": timeouts[op],
            "util_source": "measured" if op in utils else "default",
            "overhead_source": "history" if op in overheads else "default",
        }
    return timeouts, detail
