# SPDX-License-Identifier: Apache-2.0
"""``python -m breakdown.bench`` - the replay benchmark, headless.

Stages, each usable on its own::

    plan     profile/trace -> shape sweep -> replay cases (+ what can't replay)
    run      benchmark the cases, one worker process per op
    rank     results -> ranked optimization targets (targets.json)
    report   results -> workbook / console summary
    case     re-run a single case (the command a target hands the optimizer)
    history  ingest a run, list runs, or diff two runs per shape
    all      plan + run + rank + report

Every stage takes ``--run <id>`` and writes into ``output/bench/<id>/``; the
web UI (``/api/bench/*``) is a wrapper around exactly these calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from breakdown.bench import (
    devices, history as history_mod, rank as rank_mod, reports, runner, store,
)
from breakdown.bench.spec import BenchCase, build_cases


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _intlist(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in str(s).replace(" ", "").split(",") if x]


def _resolve_run(a) -> store.RunPaths:
    if a.run:
        return store.run_paths(a.run, a.root)
    latest = store.list_runs(a.root)
    if not latest:
        raise SystemExit("no runs under output/bench - start with `plan`")
    return store.run_paths(latest[0]["run_id"], a.root)


def _load_cases(paths: store.RunPaths) -> list[BenchCase]:
    with open(paths.cases) as fh:
        return [BenchCase.from_dict(c) for c in json.load(fh)]


def _sweep(a) -> list[dict[str, Any]]:
    """The (phase, S, C, B, TP) points to replay at."""
    from breakdown.shape_matrix import build_configs

    return build_configs(
        prefill_seq_lens=_intlist(a.prefill_seq_lens) or [a.query_len],
        prefill_ctx_lens=_intlist(a.prefill_ctx_lens) or [a.context_len],
        prefill_batch_sizes=_intlist(a.prefill_batch_sizes) or [1],
        decode_ctx_lens=_intlist(a.decode_ctx_lens) or [a.context_len],
        decode_batch_sizes=_intlist(a.decode_batch_sizes) or [a.batch_size],
        tp_sizes=_intlist(a.tp_sizes) or [a.tp],
    )


def _graph_from_args(a) -> tuple[dict, dict]:
    """``(graph template, model summary)`` from a trace or a saved result."""
    if a.result:
        with open(a.result) as fh:
            res = json.load(fh)
        graph = res.get("graph") or res
        return graph, res.get("summary") or {}
    if not a.trace:
        raise SystemExit("plan needs --trace (a profiler trace) or --result "
                         "(a saved /api/profile/result payload)")
    from breakdown.graph_from_trace import build_graph_from_trace
    from breakdown.model_info import fetch_model_config, summarize_config

    if a.summary:
        with open(a.summary) as fh:
            summary = json.load(fh)
    elif a.model:
        summary = summarize_config(fetch_model_config(a.model))
    else:
        raise SystemExit("plan needs --model or --summary to symbolize shapes")
    graph = build_graph_from_trace(
        a.trace, summary, tp_size=a.tp, batch_size=a.batch_size,
        query_len=a.query_len, context_len=a.context_len,
        quantization=a.quantization or None)
    return graph, summary


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_plan(a) -> int:
    from breakdown.bench import resolve
    from breakdown.shape_matrix import build_rows

    graph, summary = _graph_from_args(a)
    rows = build_rows(graph, _sweep(a))
    device = a.device or devices.detect_device()
    cases, cov = build_cases(rows, device=device)

    run_id = a.run or store.make_run_id(
        a.model or summary.get("architecture") or "model", a.tp, device)
    paths = store.run_paths(run_id, a.root).ensure()
    runner.write_cases(cases, paths.cases)

    # Classify up front so the plan says what will *not* be measured, and why,
    # before an hour of benchmarking rather than after.
    status: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.op in status:
            continue
        st, detail = resolve.classify(case.op, case.args)
        status[case.op] = {"status": st, "detail": detail,
                           "backend": case.backend}
    cov["ops_by_status"] = {}
    for op, s in status.items():
        cov["ops_by_status"].setdefault(s["status"], []).append(op)
    cov["op_status"] = status
    cov["device"] = device
    cov["sweep"] = _sweep(a)
    store.write_json(paths.plan, cov)
    store.RunMeta(run_id=run_id, model_id=a.model or "", device=device,
                  tp=a.tp, device_name=devices.device_name(device),
                  sku=devices.sku_for_device(devices.device_name(device)),
                  sweep={"configs": len(cov["sweep"])}).write(paths)

    print(f"run {run_id}: {len(cases)} cases over {cov['ops']} ops "
          f"({cov['total_rows']} matrix rows)")
    for st in sorted(cov["ops_by_status"]):
        ops = cov["ops_by_status"][st]
        print(f"  {st:<16} {len(ops):>3}  {', '.join(sorted(ops)[:6])}"
              f"{' …' if len(ops) > 6 else ''}")
    return 0


def cmd_run(a) -> int:
    paths = _resolve_run(a)
    cases = _load_cases(paths)
    device = a.device or store.read_meta(paths).get("device") \
        or devices.detect_device()
    ops = _split(a.ops)
    res = runner.run(cases, paths, device, budget=a.budget, ops=ops,
                     flush_cache=not a.no_flush_cache,
                     in_process=a.in_process,
                     on_op=lambda r: print(
                         f"{r.op[:46]:<46} {'ok ' if r.ok else 'FAIL'} "
                         f"{r.measured}/{r.cases} cases {r.seconds:>6.1f}s "
                         f"{r.error[:60]}"))
    print(f"\n{sum(o.measured for o in res.ops)} cases measured; "
          f"{len(res.failed)} ops with no measurement")
    print(f"results: {paths.results}")
    return 0 if res.ok else 1


def cmd_rank(a) -> int:
    paths = _resolve_run(a)
    records = store.read_results(paths.results)
    if not records:
        raise SystemExit(f"no results in {paths.results} - run the benchmark")
    meta = store.read_meta(paths)
    rc = rank_mod.RankConfig(
        target_util=a.target_util, tp=a.tp_filter, top=a.top,
        min_share=a.min_share, run_id=paths.run_id,
        points=_points(a), provenance={"commits": meta.get("commits") or {},
                                       "run": meta.get("run_id")})
    doc = rank_mod.rank(records, rc)
    store.write_json(paths.targets, doc)
    print(rank_mod.format_table(doc))
    print(f"\ntargets: {paths.targets}")
    return 0


def cmd_report(a) -> int:
    paths = _resolve_run(a)
    records = store.read_results(paths.results)
    if not records:
        raise SystemExit(f"no results in {paths.results}")
    print(reports.format_summary(records))
    if a.xlsx:
        targets = None
        if os.path.isfile(paths.targets):
            with open(paths.targets) as fh:
                targets = json.load(fh)
        meta = store.read_meta(paths)
        peaks = devices.peaks(meta.get("sku") or devices.DEFAULT_SKU)
        reports.write_workbook(records, paths.report, peaks, targets)
        print(f"\nworkbook: {paths.report}")
    return 0


def cmd_case(a) -> int:
    """Re-run one case - the command a ranked target hands the optimizer."""
    from breakdown.bench import worker

    paths = _resolve_run(a)
    cases = _load_cases(paths)
    sel = [c for c in cases
           if (a.case_id and c.case_id == a.case_id) or
           (a.op and c.op == a.op)]
    if not sel:
        raise SystemExit("no case matched --case-id / --op")
    device = a.device or store.read_meta(paths).get("device") \
        or devices.detect_device()
    for case in sel[:a.limit]:
        rec = worker.run_case(case, device, budget=a.budget)
        print(f"{rec['status']:<16} {rec['op']:<44} {rec['shape']}")
        if rec["status"] == "ok":
            print(f"  latency {rec['latency_us']:.3f} us  "
                  f"(p10 {rec['p10_us']:.3f} / p90 {rec['p90_us']:.3f}, "
                  f"{rec['reps']}x{rec['windows']}), traced "
                  f"{rec['traced_device_time_us']} us")
        else:
            print(f"  {rec.get('error') or rec.get('detail')}")
    return 0


def cmd_history(a) -> int:
    root = store.bench_root(a.root)
    conn = history_mod.connect(history_mod.db_path(root))
    if a.base and a.new:
        diffs = history_mod.compare(conn, a.base, a.new, a.threshold)
        if not diffs:
            print("no change beyond the threshold")
            return 0
        print(f"{'op':<44}{'base_us':>10}{'new_us':>10}{'delta':>9}  shape")
        for d in diffs:
            print(f"{d['op'][:44]:<44}{d['base_us']:>10.2f}{d['new_us']:>10.2f}"
                  f"{d['delta_pct']:>8.1f}%  {d['shape']}")
        return 0
    if a.ingest:
        paths = _resolve_run(a)
        records = store.read_results(paths.results)
        targets = None
        if os.path.isfile(paths.targets):
            with open(paths.targets) as fh:
                targets = json.load(fh)
        n = history_mod.ingest(conn, store.read_meta(paths), records, targets)
        print(f"ingested {n} cases from {paths.run_id}")
        return 0
    for r in history_mod.runs(conn):
        print(f"{r['run_id']:<48}{r['created'] or '':<22}{r['model_id'] or ''}")
    return 0


def cmd_all(a) -> int:
    rc = cmd_plan(a)
    if rc:
        return rc
    if not a.run:
        a.run = store.list_runs(a.root)[0]["run_id"]
    cmd_run(a)
    rc = cmd_rank(a)
    cmd_report(a)
    if a.ingest_history:
        a.ingest, a.base, a.new = True, None, None
        cmd_history(a)
    return rc


def _split(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _points(a) -> dict[str, tuple | None]:
    out: dict[str, tuple | None] = {}
    for phase, raw in (("prefill", a.prefill_point),
                       ("decode", a.decode_point)):
        if raw:
            parts = _intlist(raw) or []
            if len(parts) != 3:
                raise SystemExit(f"--{phase}-point wants seq,ctx,batch")
            out[phase] = tuple(parts)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m breakdown.bench",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, sweep: bool = False):
        sp.add_argument("--run", help="run id (default: the most recent)")
        sp.add_argument("--root", help="override output/bench")
        sp.add_argument("--device", help="xpu | cuda | cpu")
        if sweep:
            sp.add_argument("--trace", help="profiler trace to reconstruct")
            sp.add_argument("--result", help="saved /api/profile/result JSON")
            sp.add_argument("--model", help="HF model id (for the config)")
            sp.add_argument("--summary", help="config-summary JSON")
            sp.add_argument("--tp", type=int, default=1)
            sp.add_argument("--batch-size", type=int, default=1,
                            dest="batch_size")
            sp.add_argument("--query-len", type=int, default=1024,
                            dest="query_len")
            sp.add_argument("--context-len", type=int, default=0,
                            dest="context_len")
            sp.add_argument("--quantization", default="")
            sp.add_argument("--prefill-seq-lens", dest="prefill_seq_lens")
            sp.add_argument("--prefill-ctx-lens", dest="prefill_ctx_lens")
            sp.add_argument("--prefill-batch-sizes", dest="prefill_batch_sizes")
            sp.add_argument("--decode-ctx-lens", dest="decode_ctx_lens")
            sp.add_argument("--decode-batch-sizes", dest="decode_batch_sizes")
            sp.add_argument("--tp-sizes", dest="tp_sizes")

    def bench_opts(sp):
        sp.add_argument("--budget", type=float, default=0.5,
                        help="seconds of measurement per case")
        sp.add_argument("--ops", help="comma-separated ops to benchmark")
        sp.add_argument("--no-flush-cache", action="store_true")
        sp.add_argument("--in-process", action="store_true",
                        help="skip per-op process isolation (debugging only)")

    def rank_opts(sp):
        sp.add_argument("--target-util", type=float,
                        default=rank_mod.DEFAULT_TARGET_UTIL,
                        dest="target_util")
        sp.add_argument("--tp-filter", type=int, dest="tp_filter")
        sp.add_argument("--top", type=int, default=0)
        sp.add_argument("--min-share", type=float, default=0.0,
                        dest="min_share")
        sp.add_argument("--prefill-point", dest="prefill_point",
                        help="seq,ctx,batch to rank prefill at")
        sp.add_argument("--decode-point", dest="decode_point",
                        help="seq,ctx,batch to rank decode at")

    sp = sub.add_parser("plan", help="build the replay cases")
    common(sp, sweep=True)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run", help="benchmark the cases")
    common(sp)
    bench_opts(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("rank", help="rank optimization targets")
    common(sp)
    rank_opts(sp)
    sp.set_defaults(func=cmd_rank)

    sp = sub.add_parser("report", help="summarize a run")
    common(sp)
    sp.add_argument("--xlsx", action="store_true", help="also write the workbook")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("case", help="re-run a single case")
    common(sp)
    sp.add_argument("--case-id", dest="case_id")
    sp.add_argument("--op")
    sp.add_argument("--limit", type=int, default=5)
    sp.add_argument("--budget", type=float, default=0.5)
    sp.set_defaults(func=cmd_case)

    sp = sub.add_parser("history", help="ingest / list / diff runs")
    common(sp)
    sp.add_argument("--ingest", action="store_true")
    sp.add_argument("--base")
    sp.add_argument("--new")
    sp.add_argument("--threshold", type=float, default=0.10)
    sp.set_defaults(func=cmd_history)

    sp = sub.add_parser("all", help="plan + run + rank + report")
    common(sp, sweep=True)
    bench_opts(sp)
    rank_opts(sp)
    sp.add_argument("--xlsx", action="store_true")
    sp.add_argument("--ingest-history", action="store_true",
                    dest="ingest_history")
    sp.set_defaults(func=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
