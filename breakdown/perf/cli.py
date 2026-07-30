# SPDX-License-Identifier: Apache-2.0
"""``python -m breakdown.perf`` - the headless entry point to the perf pipeline.

Every stage runs without the web app::

    python -m breakdown.perf emit    --matrix m.xlsx --config summary.json
    python -m breakdown.perf run     --run-id <id> --backend INTEL --devices 0
    python -m breakdown.perf rank    --run-id <id>
    python -m breakdown.perf bench   --op rms_norm --case '{...}'
    python -m breakdown.perf report  --run-id <id>
    python -m breakdown.perf history --compare <base> <new>
    python -m breakdown.perf all     --matrix m.xlsx --config summary.json

``all`` is the one-command path: emit -> run -> rank -> merge -> ingest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from breakdown.perf import (
    bench_case,
    devices,
    estimate,
    history,
    rank as rank_mod,
    reports,
    runner,
    store,
    workloads as wl,
)
from breakdown.perf.matrix_reader import read_matrix, rows_to_oprows
from breakdown.perf.op_map import ModelConfig

#: Fallback when an op has no estimate (no cost data, new op).
DEFAULT_TIMEOUT = 3600


def _intlist(s: str | None) -> set[int] | None:
    if not s:
        return None
    return {int(x) for x in str(s).split(",") if str(x).strip()}


def _rows_from_matrix(path: str):
    """OpRows from a Shape Matrix .xlsx (rows built in-process skip this)."""
    return read_matrix(path)


def _row_filter(a) -> wl.RowFilter | None:
    if getattr(a, "smoke", False):
        return wl.RowFilter.smoke(tp=_intlist(a.tp))
    f = wl.RowFilter(
        tp=_intlist(a.tp),
        phases=set(a.phases.split(",")) if getattr(a, "phases", None) else None,
        prefill_seq_lens=_intlist(getattr(a, "prefill_seq_lens", None)),
        prefill_ctx_lens=_intlist(getattr(a, "prefill_ctx_lens", None)),
        prefill_batch_sizes=_intlist(getattr(a, "prefill_batch_sizes", None)),
        decode_ctx_lens=_intlist(getattr(a, "decode_ctx_lens", None)),
        decode_batch_sizes=_intlist(getattr(a, "decode_batch_sizes", None)),
    )
    return f


def _model_config(path: str | None) -> ModelConfig:
    if path and os.path.isfile(path):
        return ModelConfig.from_summary(path)
    return ModelConfig()


def _resolve_run(a) -> store.RunPaths:
    run_id = a.run_id or store.make_run_id(
        getattr(a, "model_id", None) or "model", int(_first(a.tp) or 1),
        getattr(a, "backend", "INTEL"))
    return store.run_paths(run_id, getattr(a, "perf_root", None)).ensure()


def _first(v) -> int | None:
    s = _intlist(v)
    return sorted(s)[0] if s else None


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def cmd_emit(a) -> int:
    paths = _resolve_run(a)
    rows = _rows_from_matrix(a.matrix)
    cfg = _model_config(a.config)
    buckets, cov = wl.emit(rows, cfg, a.dispatch, _row_filter(a),
                           dense_sweep=not a.no_dense_sweep)
    wl.write(buckets, cov, paths.workloads)
    with open(paths.coverage, "w") as fh:
        json.dump(cov.to_dict(), fh, indent=2)
    if a.matrix != paths.matrix and not os.path.exists(paths.matrix):
        try:
            os.link(os.path.abspath(a.matrix), paths.matrix)
        except OSError:
            pass
    store.RunMeta(run_id=paths.run_id, model_id=a.model_id or "",
                  backend=a.backend, dispatch=a.dispatch,
                  tp=int(_first(a.tp) or 1), smoke=bool(a.smoke),
                  sweep={"matrix": os.path.abspath(a.matrix)}).write(paths)
    print(f"run_id     : {paths.run_id}")
    for group in wl.GROUPS:
        n = sum(len(v) for v in (buckets.get(group) or {}).values())
        print(f"{group:<11}: {n} cases across "
              f"{len(buckets.get(group) or {})} ops")
    if cov.unmapped_breakdown_ops:
        print(f"UNMAPPED   : {sorted(cov.unmapped_breakdown_ops)}",
              file=sys.stderr)
        return 1
    print("coverage   : 0 unmapped ops")
    return 0


def cmd_run(a) -> int:
    paths = _resolve_run(a)
    if not os.path.isdir(paths.workloads) or not os.listdir(paths.workloads):
        print(f"error: no workloads for run {paths.run_id}; run 'emit' first",
              file=sys.stderr)
        return 2
    timeouts, detail, fallback = _timeout_plan(a, paths)
    if detail:
        print("timeout plan (estimated runtime -> budget):")
        for op, d in sorted(detail.items(), key=lambda kv: -kv[1]["timeout_s"]):
            print(f"  {op:<26} {d['cases']:>4} cases x{d['providers']} prov  "
                  f"util {d['util']:.3f} ({d['util_source']})  "
                  f"est {d['estimated_s']:.0f}s  timeout {d['timeout_s']}s")
    res = runner.run(
        paths.workloads, paths.reports, backend=a.backend,
        devices_arg=a.devices, ccl_devices=a.ccl_devices,
        groups=[g for g in a.groups.split(",") if g],
        tasks=[t for t in a.tasks.split(",") if t] if a.tasks else None,
        timeout=fallback, timeouts=timeouts, cache_dir=paths.cache,
        on_op=_print_op)
    print(f"\nreports: {paths.reports}")
    if not res.ok:
        print(f"{len(res.failed)} op(s) FAILED", file=sys.stderr)
        return 1
    return 0


def _print_op(r) -> None:
    status = "ok" if r.ok else "FAILED"
    line = (f"  {r.op:<26} [{r.group}] {status} "
            f"({r.cases} cases, {r.seconds}s/{r.timeout}s)")
    if r.failed_cases:
        line += f" {r.failed_cases} case error(s)"
    print(line)
    # Per-case kernel errors are the actionable output of a run: a legitimate
    # workload the kernel rejects is a bug to fix, so it is printed in full
    # rather than collapsed into the op's status.
    for msg in r.errors:
        print(f"      ! {msg}")
    if r.error and r.error not in r.errors:
        print(f"      ! {r.error}")


def _timeout_plan(a, paths) -> tuple[dict[str, int], dict, int]:
    """(per-op timeouts, why, fallback) honouring ``--timeout``."""
    if str(a.timeout).lower() != "auto":
        return {}, {}, int(a.timeout)
    try:
        timeouts, detail = estimate.plan_for_run(
            paths.workloads, perf_root=store.perf_root(a.perf_root),
            backend=a.backend, safety=a.timeout_scale)
    except Exception as exc:  # noqa: BLE001 - never block a run on estimation
        print(f"warning: timeout estimation failed ({exc}); using {DEFAULT_TIMEOUT}s",
              file=sys.stderr)
        return {}, {}, DEFAULT_TIMEOUT
    return timeouts, detail, DEFAULT_TIMEOUT


def cmd_rank(a) -> int:
    paths = _resolve_run(a)
    meta = store.read_meta(paths)
    matrix = a.matrix or (meta.get("sweep") or {}).get("matrix") or paths.matrix
    if not os.path.isfile(matrix):
        print(f"error: no shape matrix for run {paths.run_id} ({matrix}); pass "
              f"--matrix or run 'emit' first", file=sys.stderr)
        return 2
    rows = _rows_from_matrix(matrix)
    reports_dir = a.reports or paths.reports
    recs = reports.records(a.backend, reports_dir)
    if not recs:
        print(f"error: no benchmark reports under {reports_dir}; run "
              f"'python -m breakdown.perf run --run-id {paths.run_id}' first",
              file=sys.stderr)
        return 2
    rc = rank_mod.RankConfig(
        dispatch=a.dispatch, tp=_first(a.tp),
        phases=tuple(p for p in a.phases.split(",") if p),
        points={k: v for k, v in (
            ("prefill", _point(a.prefill_point)),
            ("decode", _point(a.decode_point))) if v},
        phase_weight=_weights(a.phase_weight),
        sku=a.sku, peak_bw_gbs=a.peak_bw_gbs, peak_tflops=a.peak_tflops,
        target_util=a.target_util, top=a.top, backend=a.backend,
        provenance={"run_id": paths.run_id,
                    "commits": meta.get("commits") or {}})
    doc = rank_mod.rank(rows, recs, _model_config(a.config), rc)
    with open(a.out or paths.targets, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(rank_mod.format_table(doc))
    print(f"\nwrote {a.out or paths.targets}")
    return 0


def _point(s: str | None) -> tuple | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    vals = [int(p) if p and p.lower() not in ("none", "-") else None
            for p in parts]
    while len(vals) < 3:
        vals.append(None)
    return tuple(vals[:3])


def _weights(s: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in (s or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = float(v)
    return out


def cmd_bench(a) -> int:
    recs = bench_case.bench(a.op, bench_case.load_case(a.case), a.backend,
                            a.provider, a.repeat, a.reps, a.warmup)
    for r in recs:
        print(bench_case.format_result(r))
    if a.json:
        bench_case.write_json(recs, a.json)
    return 0


def cmd_report(a) -> int:
    paths = _resolve_run(a)
    specs = a.reports or [f"{a.backend}={paths.reports}"]
    info = reports.merge(specs, out=a.out or paths.merged)
    print(json.dumps(info, indent=2))
    return 0


def cmd_history(a) -> int:
    root = store.perf_root(a.perf_root)
    conn = history.connect(history.db_path(root))
    if a.compare:
        base, new = a.compare
        rows = history.compare(conn, base, new, a.threshold)
        if not rows:
            print(f"no change beyond {a.threshold:.0%} between {base} and {new}")
            return 0
        for r in rows:
            print(f"{r['kind']:<11} {r['op']:<24}{r['provider']:<17}"
                  f"{r['base_us']:>10.2f} -> {r['new_us']:>10.2f} "
                  f"({r['delta_pct']:+.1f}%)  {json.dumps(r['args'])[:60]}")
        return 0
    for r in history.runs(conn):
        print(f"{r['run_id']:<44} {r['created'] or '':<22} "
              f"{r['model_id'] or '':<28} {r['backend'] or ''}")
    return 0


def cmd_ingest(a) -> int:
    paths = _resolve_run(a)
    meta = store.read_meta(paths) or {"run_id": paths.run_id}
    targets: dict[str, Any] = {}
    if os.path.isfile(paths.targets):
        with open(paths.targets) as fh:
            targets = json.load(fh)
    conn = history.connect(history.db_path(store.perf_root(a.perf_root)))
    n = history.ingest(conn, meta,
                       reports.records(a.backend, paths.reports), targets)
    print(f"ingested {n} cases from {paths.run_id}")
    return 0


def cmd_all(a) -> int:
    rc = cmd_emit(a)
    if rc:
        return rc
    a.run_id = a.run_id or _latest_run_id(a)
    for step in (cmd_run, cmd_rank, cmd_report, cmd_ingest):
        rc = step(a)
        if rc and step is cmd_run and not a.keep_going:
            return rc
    return 0


def _latest_run_id(a) -> str | None:
    runs = store.list_runs(getattr(a, "perf_root", None))
    return runs[0]["run_id"] if runs else None


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m breakdown.perf", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf-root", default=None,
                    help="artifact root (default output/perf)")
    ap.add_argument("--run-id", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, matrix=True):
        if matrix:
            p.add_argument("--matrix", default=None,
                           help="Shape Matrix .xlsx")
        p.add_argument("--config", default=None,
                       help="model config summary JSON (summarize_config)")
        p.add_argument("--model-id", default=None)
        p.add_argument("--dispatch", default="xpu", choices=["xpu", "cuda"])
        p.add_argument("--backend", default="INTEL", choices=["INTEL", "GPU"])
        p.add_argument("--tp", default="4")

    p = sub.add_parser("emit", help="Shape Matrix -> micro_perf workloads")
    common(p)
    p.add_argument("--phases", default=None)
    p.add_argument("--prefill-seq-lens", default=None)
    p.add_argument("--prefill-ctx-lens", default=None)
    p.add_argument("--prefill-batch-sizes", default=None)
    p.add_argument("--decode-ctx-lens", default=None)
    p.add_argument("--decode-batch-sizes", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="screening tier: a few shapes per op")
    p.add_argument("--no-dense-sweep", action="store_true")
    p.set_defaults(func=cmd_emit)

    p = sub.add_parser("run", help="benchmark the workloads (one op/process)")
    common(p, matrix=False)
    p.add_argument("--devices", default="0")
    p.add_argument("--ccl-devices", default=None)
    p.add_argument("--groups", default=",".join(wl.GROUPS))
    p.add_argument("--tasks", default=None)
    p.add_argument("--timeout", default="auto",
               help="seconds per op, or 'auto' to size each op's "
                    "budget from its estimated runtime")
    p.add_argument("--timeout-scale", type=float,
                   default=estimate.DEFAULT_SAFETY,
                   help="safety multiplier on the estimate")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("rank", help="rank ops by optimization value")
    common(p)
    p.add_argument("--reports", default=None)
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--prefill-point", default=None, metavar="SEQ,CTX,BS")
    p.add_argument("--decode-point", default=None, metavar="SEQ,CTX,BS")
    p.add_argument("--phase-weight", default="prefill=1,decode=1")
    p.add_argument("--sku", default=None, choices=sorted(devices.SKU_PEAKS))
    p.add_argument("--peak-bw-gbs", type=float, default=None)
    p.add_argument("--peak-tflops", type=float, default=None)
    p.add_argument("--target-util", type=float,
                   default=rank_mod.DEFAULT_TARGET_UTIL)
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("bench", help="one op-case in its own process")
    p.add_argument("--op", required=True)
    p.add_argument("--case", required=True, help="JSON or @file.json")
    p.add_argument("--backend", default="INTEL", choices=["INTEL", "GPU"])
    p.add_argument("--provider", default=None, help="name, or 'all'")
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("report", help="merge report trees into a workbook")
    common(p, matrix=False)
    p.add_argument("--reports", action="append", default=None,
                   metavar="[LABEL=]DIR")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("ingest", help="store a run's cases in the history db")
    common(p, matrix=False)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("history", help="list runs / compare two runs")
    p.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"), default=None)
    p.add_argument("--threshold", type=float, default=0.10)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("all", help="emit -> run -> rank -> report -> ingest")
    common(p)
    p.add_argument("--devices", default="0")
    p.add_argument("--ccl-devices", default=None)
    p.add_argument("--groups", default=",".join(wl.GROUPS))
    p.add_argument("--tasks", default=None)
    p.add_argument("--timeout", default="auto",
               help="seconds per op, or 'auto' to size each op's "
                    "budget from its estimated runtime")
    p.add_argument("--timeout-scale", type=float,
                   default=estimate.DEFAULT_SAFETY,
                   help="safety multiplier on the estimate")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-dense-sweep", action="store_true")
    p.add_argument("--keep-going", action="store_true",
                   help="rank even if some ops failed to benchmark")
    p.add_argument("--phases", default="prefill,decode")
    p.add_argument("--prefill-seq-lens", default=None)
    p.add_argument("--prefill-ctx-lens", default=None)
    p.add_argument("--prefill-batch-sizes", default=None)
    p.add_argument("--decode-ctx-lens", default=None)
    p.add_argument("--decode-batch-sizes", default=None)
    p.add_argument("--prefill-point", default=None)
    p.add_argument("--decode-point", default=None)
    p.add_argument("--phase-weight", default="prefill=1,decode=1")
    p.add_argument("--sku", default=None, choices=sorted(devices.SKU_PEAKS))
    p.add_argument("--peak-bw-gbs", type=float, default=None)
    p.add_argument("--peak-tflops", type=float, default=None)
    p.add_argument("--target-util", type=float,
                   default=rank_mod.DEFAULT_TARGET_UTIL)
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--reports", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if getattr(a, "cmd", None) in ("emit", "all") and not a.matrix:
        ap.error("--matrix is required")
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
