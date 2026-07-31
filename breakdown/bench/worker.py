# SPDX-License-Identifier: Apache-2.0
"""Benchmark one op's cases, in its own process.

One op per process is not an optimization, it is containment: a kernel that
aborts (``TORCH_CHECK``), hangs, or wedges the device takes down only its own
op, and every other op's results are already on disk. Results are streamed to
``results.jsonl`` case by case for the same reason - a run killed halfway still
says exactly what it measured.

Run standalone::

    python -m breakdown.bench.worker --cases cases.json --op aten::linear \\
        --out results.jsonl --device xpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Iterable

from breakdown.bench import inputs as inputs_mod
from breakdown.bench import recipes, resolve, timing
from breakdown.bench.spec import BenchCase, shape_key


def bench_env(cache_dir: str, base: dict | None = None) -> dict[str, str]:
    """Environment for a worker: persistent kernel caches.

    Without them every process re-pays SYCL AOT / Triton JIT on its first case,
    which dominates a short sweep and poisons the first measurement of each op.
    oneAPI itself is sourced by the caller's shell - ``setvars.sh`` reads unset
    variables, so sourcing it from a ``set -u`` script kills the script.
    """
    env = dict(base if base is not None else os.environ)
    sycl = os.path.join(cache_dir, "sycl")
    triton = os.path.join(cache_dir, "triton")
    os.makedirs(sycl, exist_ok=True)
    os.makedirs(triton, exist_ok=True)
    env.setdefault("SYCL_CACHE_PERSISTENT", "1")
    env.setdefault("SYCL_CACHE_DIR", sycl)
    env.setdefault("TRITON_CACHE_DIR", triton)
    return env


def _record(case: BenchCase, status: str, m: timing.Measurement | None = None,
            error: str = "", detail: str = "") -> dict[str, Any]:
    rec: dict[str, Any] = {
        "case_id": case.case_id,
        "op": case.op,
        "shape_key": shape_key(case.op, case.args),
        "shape": case.shape_label,
        "status": status,
        "device": case.device,
        "phase": case.phase,
        "seq_len": case.seq_len,
        "ctx_len": case.ctx_len,
        "batch_size": case.batch_size,
        "points": case.points,
        "tp": case.tp,
        "module": case.module,
        "role": case.role,
        "backend": case.backend,
        "layers": case.layers,
        "flops": case.flops,
        "bytes": case.nbytes,
        "traced_device_time_us": case.traced_device_time_us,
        "traced_comparable": case.traced_comparable,
        "error": error,
        "detail": detail,
    }
    if m is not None:
        rec.update({
            "latency_us": m.latency_us, "mean_us": m.mean_us,
            "min_us": m.min_us, "p10_us": m.p10_us, "p90_us": m.p90_us,
            "stdev_us": m.stdev_us, "iters": m.iters, "reps": m.reps,
            "windows": m.windows, "overhead_us": m.overhead_us,
            "notes": m.notes,
        })
    return rec


def run_case(case: BenchCase, device: str,
             budget: float = timing.DEFAULT_BUDGET_S,
             flush_cache: bool = True) -> dict[str, Any]:
    """Resolve, materialize and time one case; never raises."""
    reason = recipes.SKIP_REASONS.get(case.op)
    if reason:
        return _record(case, "skipped", error="", detail=reason)
    if resolve.is_collective(case.op):
        # Collectives need every rank; the single-process worker would only
        # measure a hang. They are handled by breakdown.bench.collective.
        return _record(case, "collective",
                       detail="needs a multi-rank launch (bench.collective)")
    try:
        res = resolve.resolve(case.op, case.args)
    except resolve.NotReplayable as exc:
        return _record(case, "not_replayable", detail=str(exc))
    except resolve.ResolveError as exc:
        return _record(case, "unresolved", error=str(exc))

    try:
        call = recipes.build_args(case, res, device)
    except inputs_mod.MissingSynthesizer as exc:
        return _record(case, "needs_synthesizer", error=str(exc))
    except (inputs_mod.ArgBuildError, Exception) as exc:  # noqa: BLE001
        return _record(case, "arg_error",
                       error=f"{type(exc).__name__}: {exc}",
                       detail=traceback.format_exc(limit=3))

    single = recipes.SINGLE_REP.get(case.op)
    try:
        m = timing.measure(res.fn, call.args, device, kwargs=call.kwargs,
                           mutated=call.mutated, budget=budget,
                           reps=1 if single else None,
                           flush_cache=flush_cache)
    except Exception as exc:               # noqa: BLE001 - a case never kills
        return _record(case, "failed",     # the op's remaining cases
                       error=f"{type(exc).__name__}: {exc}",
                       detail=traceback.format_exc(limit=3))
    if single and m.ok:
        m.notes.append(f"one call per timed window: {single}")
    if not m.ok:
        return _record(case, "failed", m, error=m.error)
    return _record(case, "ok", m)


def run_op(cases: Iterable[BenchCase], device: str, out_path: str,
           budget: float = timing.DEFAULT_BUDGET_S,
           flush_cache: bool = True) -> list[dict[str, Any]]:
    """Benchmark every case of one op, appending each result as it lands."""
    records: list[dict[str, Any]] = []
    for case in cases:
        rec = run_case(case, device, budget=budget, flush_cache=flush_cache)
        records.append(rec)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        # An op's cases can each hold hundreds of megabytes of operands (an
        # lm_head weight); without releasing them the worker OOMs partway
        # through its own sweep.
        _release(device)
    return records


def _release(device: str) -> None:
    try:
        import gc

        import torch
        gc.collect()
        mod = getattr(torch, device, None)
        if mod is not None and hasattr(mod, "empty_cache"):
            mod.empty_cache()
    except Exception:                       # noqa: BLE001 - best effort
        pass


def load_cases(path: str, op: str | None = None) -> list[BenchCase]:
    with open(path) as fh:
        raw = json.load(fh)
    cases = [BenchCase.from_dict(c) for c in raw]
    return [c for c in cases if op is None or c.op == op]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", required=True, help="cases.json of the run")
    p.add_argument("--op", help="benchmark only this op (default: all)")
    p.add_argument("--out", required=True, help="results.jsonl to append to")
    p.add_argument("--device", default=None, help="xpu | cuda | cpu")
    p.add_argument("--budget", type=float, default=timing.DEFAULT_BUDGET_S,
                   help="seconds of measurement per case")
    p.add_argument("--no-flush-cache", action="store_true")
    a = p.parse_args(argv)

    from breakdown.bench import devices

    device = a.device or devices.detect_device()
    cases = load_cases(a.cases, a.op)
    if not cases:
        print(f"no cases for op={a.op}", file=sys.stderr)
        return 1
    recs = run_op(cases, device, a.out, budget=a.budget,
                  flush_cache=not a.no_flush_cache)
    ok = sum(1 for r in recs if r["status"] == "ok")
    print(f"{a.op or 'all'}: {ok}/{len(recs)} cases measured")
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
