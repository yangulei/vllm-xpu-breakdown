# SPDX-License-Identifier: Apache-2.0
"""Trim a raw torch-profiler trace into a small, committable test fixture.

The reconstruction only reads three things: the worker thread's host events
(module spans, cpu_ops, python frames), the device kernel events, and the
runtime launch events that link the two. Everything else in a vLLM trace —
notably the ~93k ``shm_broadcast`` python frames on the IPC threads — is dead
weight for a fixture.

The trim is verified, not assumed: ``make_fixture`` reconstructs the graph from
the original and from the trimmed trace and refuses to write a fixture whose
graph differs.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.graph_from_trace import (  # noqa: E402
    _DEVICE_KERNEL_CATEGORIES,
    _RUNTIME_CATEGORIES,
    MODULE_SPAN_PREFIX,
    _load_trace,
    build_graph_from_trace,
)

# Inter-process plumbing frames: vLLM's shared-memory broadcast and the
# distributed utils poll loops. They launch no kernel and dispatch no op; they
# are simply the largest thing in a TP trace (~93k of 192k events).
IPC_FRAME_FILES = ("shm_broadcast.py", "distributed/utils.py")

KEEP_CATS = set(_DEVICE_KERNEL_CATEGORIES) | set(_RUNTIME_CATEGORIES) | {
    "cpu_op", "user_annotation", "python_function", "cuda_driver",
}


def _worker_tid(events: list[dict]) -> object:
    counts: dict = {}
    for e in events:
        if (e.get("cat") == "user_annotation"
                and str(e.get("name", "")).startswith(MODULE_SPAN_PREFIX)):
            counts[e.get("tid")] = counts.get(e.get("tid"), 0) + 1
    if counts:
        return max(counts, key=counts.get)
    # Span-less trace (e.g. the archived CUDA reference runs, captured before
    # the span hooks existed and not reproducible on this host): fall back to
    # the busiest cpu_op thread, which is what the reconstruction does too.
    for e in events:
        if e.get("cat") == "cpu_op":
            counts[e.get("tid")] = counts.get(e.get("tid"), 0) + 1
    if not counts:
        raise SystemExit("trace has neither module:: spans nor cpu_op events")
    return max(counts, key=counts.get)


def trim(trace: dict, drop_plumbing_frames: bool = True) -> dict:
    events = trace.get("traceEvents", [])
    tid = _worker_tid(events)

    # Timestamps of everything that constitutes *compute*: host op events and
    # kernel launch sites. A python frame that encloses none of them is pure
    # plumbing (vLLM's ~93k ``shm_broadcast`` IPC frames dominate a TP trace)
    # and cannot be promoted to a module by any rule, present or future.
    marks = sorted(
        e["ts"] for e in events
        if e.get("tid") == tid and e.get("ts") is not None
        and (e.get("cat") == "cpu_op" or e.get("cat") in _RUNTIME_CATEGORIES)
    )

    def encloses_compute(e: dict) -> bool:
        ts = e.get("ts")
        if ts is None:
            return True
        i = bisect.bisect_left(marks, ts)
        return i < len(marks) and marks[i] <= ts + (e.get("dur") or 0)

    out = []
    for e in events:
        cat = e.get("cat")
        if cat is None:
            out.append(e)          # metadata records (process/thread names)
            continue
        if cat in _DEVICE_KERNEL_CATEGORIES:
            out.append(e)
            continue
        if cat not in KEEP_CATS:
            continue
        if e.get("tid") != tid:
            continue
        if cat == "python_function":
            if any(f in str(e.get("name", "")) for f in IPC_FRAME_FILES):
                continue
            if drop_plumbing_frames and not encloses_compute(e):
                continue
        out.append(e)
    trimmed = {k: v for k, v in trace.items() if k != "traceEvents"}
    trimmed["traceEvents"] = out
    return trimmed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--query-len", type=int, default=None)
    ap.add_argument("--context-len", type=int, default=None)
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep every worker-thread python frame (larger, but "
                         "byte-faithful when the compute filter perturbs "
                         "sibling ordering)")
    args = ap.parse_args()

    from breakdown.model_info import fetch_model_config, summarize_config
    summary = summarize_config(fetch_model_config(args.model))

    kw = dict(summary=summary, tp_size=args.tp, batch_size=args.batch,
              query_len=args.query_len, context_len=args.context_len)

    original = _load_trace(args.src)
    trimmed = trim(original, drop_plumbing_frames=not args.keep_frames)
    print(f"events {len(original['traceEvents'])} -> {len(trimmed['traceEvents'])}")

    tmp = args.dst + ".tmp.json.gz"
    with gzip.open(tmp, "wt") as f:
        json.dump(trimmed, f)

    before = build_graph_from_trace(args.src, **kw)
    after = build_graph_from_trace(tmp, **kw)
    if json.dumps(before, sort_keys=True, default=str) != \
            json.dumps(after, sort_keys=True, default=str):
        os.remove(tmp)
        raise SystemExit("REFUSED: trimming changed the reconstructed graph")

    os.replace(tmp, args.dst)
    print(f"ok: {args.dst} {os.path.getsize(args.dst)/1024:.0f} KiB "
          f"(from {os.path.getsize(args.src)/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
