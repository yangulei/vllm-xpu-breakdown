# SPDX-License-Identifier: Apache-2.0
"""Turn a run's ``results.jsonl`` into a readable workbook.

Three sheets, in the order a reader needs them:

``Summary``   one row per op: e2e weight, utilization, action, coverage.
``Cases``     every measured case, with the profile's device time beside the
              replayed latency - the run's own fidelity check.
``Coverage``  what was *not* measured and why (collectives without ranks,
              context-bound wrappers, ops needing a synthesizer), because a
              silent omission is the one failure mode a benchmark cannot afford.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any, Iterable

from breakdown.bench import devices, estimate

COLUMNS = [
    ("op", "Op"), ("backend", "Backend"), ("phase", "Phase"),
    ("shape", "Shape"), ("status", "Status"),
    ("latency_us", "Latency (us)"), ("p10_us", "p10 (us)"),
    ("p90_us", "p90 (us)"), ("stdev_us", "Stdev (us)"),
    ("traced_device_time_us", "Traced (us)"),
    ("replay_vs_traced", "Replay/Traced"),
    ("util", "Util"), ("bound", "Bound"),
    ("memory_level", "Roof"), ("ai", "AI (flop/byte)"),
    ("layers", "Layers"), ("weighted_us", "Weighted (us)"),
    ("seq_len", "Seq Len"), ("ctx_len", "Ctx Len"),
    ("batch_size", "Batch"), ("tp", "TP"),
    ("flops", "FLOPs"), ("bytes", "Memory (bytes)"),
    ("reps", "Reps"), ("windows", "Windows"), ("iters", "Iters"),
    ("module", "Module"), ("error", "Error"), ("detail", "Detail"),
]


def enrich(records: Iterable[dict], peaks: dict[str, float] | None = None
           ) -> list[dict[str, Any]]:
    """Add utilization, weighted time and the traced-vs-replayed ratio."""
    peaks = peaks or devices.peaks(devices.DEFAULT_SKU)
    out = []
    for r in records:
        r = dict(r)
        lat = float(r.get("latency_us") or 0)
        calls = max(int(r.get("layers") or 1), 1)
        if lat > 0:
            util, bound, level = estimate.utilization_detail(
                lat, float(r.get("flops") or 0), float(r.get("bytes") or 0),
                peaks)
            r["util"] = round(util, 3)
            r["bound"] = bound
            r["memory_level"] = level
            r["ai"] = round(estimate.op_ai(float(r.get("flops") or 0),
                                           float(r.get("bytes") or 0)), 3)
            r["weighted_us"] = round(lat * calls, 1)
        traced = float(r.get("traced_device_time_us") or 0)
        if traced > 0 and lat > 0 and r.get("traced_comparable"):
            r["replay_vs_traced"] = round(lat / traced, 3)
        out.append(r)
    return out


def summarize(records: Iterable[dict]) -> list[dict[str, Any]]:
    """One row per op: how much it costs, how well it runs, what was measured."""
    by_op: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_op[r.get("op", "?")].append(r)
    rows = []
    for op, recs in by_op.items():
        ok = [r for r in recs if r.get("status") == "ok"]
        statuses = Counter(r.get("status") for r in recs)
        weighted = sum(float(r.get("weighted_us") or 0) for r in ok)
        utils = [float(r["util"]) for r in ok if r.get("util") is not None]
        ratios = [float(r["replay_vs_traced"]) for r in ok
                  if r.get("replay_vs_traced") is not None]
        rows.append({
            "Op": op,
            "Backend": next((r.get("backend") for r in recs if r.get("backend")),
                            ""),
            "Cases": len(recs),
            "Measured": len(ok),
            "Weighted (us)": round(weighted, 1),
            "Median latency (us)": round(
                sorted(float(r.get("latency_us") or 0) for r in ok)[len(ok) // 2],
                3) if ok else None,
            "Mean util": round(sum(utils) / len(utils), 3) if utils else None,
            "Replay/Traced": round(sum(ratios) / len(ratios), 3)
            if ratios else None,
            "Statuses": ", ".join(f"{k}:{v}" for k, v in
                                  sorted(statuses.items())),
        })
    rows.sort(key=lambda r: -(r["Weighted (us)"] or 0))
    return rows


def coverage(records: Iterable[dict]) -> list[dict[str, Any]]:
    """Every op that was *not* measured, with the reason it was not."""
    rows: dict[tuple, dict[str, Any]] = {}
    for r in records:
        status = r.get("status")
        if status == "ok":
            continue
        key = (r.get("op"), status)
        entry = rows.setdefault(key, {
            "Op": r.get("op"), "Backend": r.get("backend", ""),
            "Status": status, "Cases": 0,
            "Reason": r.get("detail") or r.get("error") or "",
            "Traced (us)": 0.0,
        })
        entry["Cases"] += 1
        entry["Traced (us)"] = round(
            entry["Traced (us)"] + float(r.get("traced_device_time_us") or 0), 1)
    out = list(rows.values())
    out.sort(key=lambda r: -r["Traced (us)"])
    return out


def write_workbook(records: list[dict], path: str,
                   peaks: dict[str, float] | None = None,
                   targets: dict | None = None) -> str:
    """Write the report workbook (requires pandas + an Excel writer)."""
    import pandas as pd

    rich = enrich(records, peaks)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cases = pd.DataFrame([{label: r.get(key) for key, label in COLUMNS}
                          for r in rich])
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame(summarize(rich)).to_excel(xl, sheet_name="Summary",
                                               index=False)
        cases.to_excel(xl, sheet_name="Cases", index=False)
        cov = coverage(rich)
        if cov:
            pd.DataFrame(cov).to_excel(xl, sheet_name="Coverage", index=False)
        if targets and targets.get("targets"):
            pd.DataFrame([{
                "Rank": t["rank"], "Op": t["op"], "Backend": t["backend"],
                "e2e (us)": t["e2e_us"], "Share": t["share_of_e2e"],
                "Util": t["roofline"]["util"], "Bound": t["roofline"]["bound"],
                "Roof": t["roofline"].get("memory_level", ""),
                "Savings (us)": t["savings_us"]["total"],
                "Action": t["action"],
                "Kernel dir": (t.get("kernel") or {}).get("kernel_dir", ""),
                "Flags": "; ".join(t.get("flags") or []),
            } for t in targets["targets"]]).to_excel(
                xl, sheet_name="Targets", index=False)
    return path


def format_summary(records: list[dict]) -> str:
    """Console summary of a run."""
    rich = enrich(records)
    rows = summarize(rich)
    total = sum(r["Weighted (us)"] or 0 for r in rows)
    out = [f"{len(rows)} ops, "
           f"{sum(r['Measured'] for r in rows)}/{sum(r['Cases'] for r in rows)}"
           f" cases measured, weighted {total / 1000:.2f} ms/step\n"]
    hdr = (f"{'op':<44}{'backend':<18}{'weighted_us':>12}{'util':>7}"
           f"{'repl/trace':>11}  statuses")
    out += [hdr, "-" * len(hdr)]
    for r in rows:
        util = f"{r['Mean util'] * 100:>6.0f}%" if r["Mean util"] else "     —"
        ratio = (f"{r['Replay/Traced']:>11.2f}" if r["Replay/Traced"]
                 else "          —")
        out.append(f"{r['Op'][:44]:<44}{(r['Backend'] or '')[:18]:<18}"
                   f"{r['Weighted (us)']:>12.1f}{util}{ratio}  {r['Statuses']}")
    return "\n".join(out)
