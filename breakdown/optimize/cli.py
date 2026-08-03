# SPDX-License-Identifier: Apache-2.0
"""``python -m breakdown.optimize {candidates,prompt,start,status,stop}``.

Headless parity with ``/api/optimize/*``: the web UI is a wrapper, so a kernel
session can always be opened (or inspected) from a terminal on the machine that
owns the GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from ..core import devices as bench_devices
from ..bench import store as bench_store
from .manager import MANAGER
from .prompt import build_prompt, candidates, targets_by_op
from .session import default_workspace_root, session_paths


def load_targets(run_id: str) -> dict[str, Any]:
    paths = bench_store.run_paths(run_id)
    if not os.path.isfile(paths.targets):
        raise SystemExit(
            f"run '{run_id}' has no targets.json - rank it first "
            f"(python -m breakdown.bench rank --run {run_id})")
    with open(paths.targets, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_candidates(args: argparse.Namespace) -> int:
    doc = load_targets(args.run)
    rows = candidates(doc, args.phase)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'#':>3}  {'op':<44} {'action':<16} {'us':>10}  launchable")
    for row in rows:
        mark = "yes" if row["launchable"] else f"no - {row['reason'][:60]}"
        print(f"{row['rank'] or 0:>3}  {str(row['op'])[:44]:<44} "
              f"{str(row['action'])[:16]:<16} {row['e2e_us'] or 0:>10.1f}  {mark}")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    doc = load_targets(args.run)
    target = targets_by_op(doc, args.phase).get(args.op)
    if target is None:
        raise SystemExit(f"'{args.op}' is not a ranked target of {args.run}")
    paths = session_paths(args.run, args.op)
    text = build_prompt(target, doc, run_id=args.run, phase=args.phase,
                        device_ids=bench_devices.parse_device_ids(args.devices)
                        or None,
                        workspace_root=args.cwd or default_workspace_root(),
                        artifact_dir=paths.dir)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(args.out)
    else:
        sys.stdout.write(text)
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    doc = load_targets(args.run)
    ids = bench_devices.parse_device_ids(args.devices)
    kind = args.device or bench_devices.detect_device()
    err = bench_devices.validate_device_ids(ids, kind)
    if err:
        raise SystemExit(err)
    state = MANAGER.start(run_id=args.run, doc=doc, ops=args.ops,
                          phase=args.phase,
                          workspace_root=args.cwd or default_workspace_root(),
                          device_kind=kind, device_ids=ids or None,
                          spawn=not args.dry_run)
    _print_sessions(state["sessions"])
    if args.dry_run or not args.wait:
        return 0
    while MANAGER.any_active(args.run):
        time.sleep(5)
    _print_sessions([s.to_dict() for s in MANAGER.sessions(args.run)])
    return 0 if all(s.state == "done" for s in MANAGER.sessions(args.run)) else 1


def cmd_status(args: argparse.Namespace) -> int:
    sessions = [s.to_dict() for s in MANAGER.sessions(args.run)]
    if not sessions:
        path = session_paths(args.run, "_index").index
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                sessions = json.load(fh).get("sessions", [])
    if args.json:
        print(json.dumps(sessions, indent=2))
        return 0
    _print_sessions(sessions)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    stopped = MANAGER.stop(args.run, args.op)
    print(f"stopped: {', '.join(stopped) if stopped else '(nothing active)'}")
    return 0


def _print_sessions(sessions: list[dict[str, Any]]) -> None:
    print(f"{'op':<44} {'state':<9} {'gpu':<8} {'queue':>5} {'exit':>5}")
    for s in sessions:
        gpu = ",".join(str(i) for i in s.get("device_ids") or []) or "-"
        print(f"{str(s.get('op'))[:44]:<44} {str(s.get('state')):<9} "
              f"{gpu:<8} {str(s.get('queue_position') or '-'):>5} "
              f"{str(s.get('exit_code') if s.get('exit_code') is not None else '-'):>5}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m breakdown.optimize",
        description="Open Copilot CLI kernel sessions for ranked targets "
                    "(one GPU per session).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True, help="benchmark run id")
        p.add_argument("--phase", default="prefill",
                       choices=["prefill", "decode"],
                       help="which phase's ranking to use (default: prefill)")

    p = sub.add_parser("candidates", help="ranked ops and their launchability")
    common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("prompt", help="print one op's optimization brief")
    common(p)
    p.add_argument("--op", required=True)
    p.add_argument("--devices", default="", help="device indexes, e.g. 0,1")
    p.add_argument("--cwd", default="", help="workspace root")
    p.add_argument("--out", default="", help="write to this file")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("start", help="open sessions for one or more ops")
    common(p)
    p.add_argument("--ops", nargs="+", required=True)
    p.add_argument("--devices", default="",
                   help="device indexes to use (blank = all present); "
                        "each session owns one of them exclusively")
    p.add_argument("--device", default="", help="device kind (xpu/cuda)")
    p.add_argument("--cwd", default="", help="workspace root")
    p.add_argument("--dry-run", action="store_true",
                   help="write the briefs and commands, spawn nothing")
    p.add_argument("--wait", action="store_true",
                   help="block until every session ends")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("status", help="sessions of a run")
    common(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("stop", help="stop a run's sessions")
    common(p)
    p.add_argument("--op", default=None, help="only this op")
    p.set_defaults(func=cmd_stop)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
