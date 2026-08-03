# SPDX-License-Identifier: Apache-2.0
"""Turn a run's ``results.jsonl`` into a readable workbook.

The workbook is the run's **single deliverable**: the shape space it swept, the
cost it measured there, and the ranking that follows. Sheets, in the order a
reader needs them:

``Info``         what the run was: model, profiled config, caveats, validation.
``Summary``      one row per op: e2e weight, utilization, action, coverage.
``Targets``      the ranking, and one sheet per phase when the run ranked
                 prefill and decode separately.
``<op>``         **one sheet per op** with every case measured for it - a single
                 flat Cases sheet mixed dozens of ops with different shape
                 vocabularies into one table, which is unreadable exactly when
                 it matters (comparing a kernel's own shapes against each other).
``Coverage``     what was *not* measured and why (collectives without ranks,
                 context-bound wrappers, ops needing a synthesizer), because a
                 silent omission is the one failure mode a benchmark cannot
                 afford.
``Shape Matrix`` the sweep itself - every (phase, S, C, B, TP, op) point with
                 its shapes, dtypes, memory and FLOPs. This is the same table
                 the standalone Shape Matrix export produces, and it belongs
                 here because it is the benchmark's *input*: the cases are
                 built from these rows, so the measured latencies are only
                 interpretable against them. It is last because it is by far
                 the longest sheet.
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from breakdown import cost
from breakdown.core import devices
from breakdown.shape_matrix import MATRIX_HEADERS

COLUMNS = [
    ("op", "Op"), ("backend", "Backend"), ("phase", "Phase"),
    ("shape", "Shape"), ("status", "Status"),
    ("latency_us", "Latency (us)"), ("p10_us", "p10 (us)"),
    ("p90_us", "p90 (us)"), ("stdev_us", "Stdev (us)"),
    ("traced_device_time_us", "Traced (us)"),
    ("replay_vs_traced", "Replay/Traced"),
    ("util", "Util"), ("util_dram", "Util (DRAM)"),
    ("bound", "Bound"), ("unit", "Roof"), ("ai", "AI (flop/byte)"),
    ("ridge_ai", "Ridge AI"),
    ("layers", "Layers"), ("weighted_us", "Weighted (us)"),
    ("seq_len", "Seq Len"), ("ctx_len", "Ctx Len"),
    ("batch_size", "Batch"), ("tp", "TP"),
    ("flops", "FLOPs"), ("nbytes", "Memory (bytes)"),
    ("reps", "Reps"), ("windows", "Windows"), ("iters", "Iters"),
    ("module", "Module"), ("error", "Error"), ("detail", "Detail"),
]

#: Excel forbids these in a sheet name and caps it at 31 characters.
_SHEET_BAD = re.compile(r"[\[\]:*?/\\]")


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
            d = cost.roofline_detail(
                lat, float(r.get("flops") or 0), float(r.get("nbytes") or 0),
                peaks, r.get("op"))
            r["util"] = round(d["util"], 3)
            r["util_dram"] = round(d["util_dram"], 3) or None
            r["effective_util"] = round(d["effective_util"], 3)
            r["bound"] = d["bound"]
            r["memory_level"] = d["memory_level"]
            r["unit"] = d["unit"]
            r["ai"] = round(d["ai"], 3) if d.get("ai") is not None else None
            r["ridge_ai"] = round(d["ridge_ai"], 2)
            r["weighted_us"] = round(lat * calls, 1)
        traced = float(r.get("traced_device_time_us") or 0)
        if traced > 0 and lat > 0 and r.get("traced_comparable"):
            r["replay_vs_traced"] = round(lat / traced, 3)
        out.append(r)
    return out


def sheet_name(op: str, used: set[str]) -> str:
    """A unique, Excel-legal sheet name for an op's dispatch name."""
    base = _SHEET_BAD.sub(".", str(op).replace("::", ".")).strip() or "op"
    name = base[:31]
    if name in used:
        for i in range(2, 100):
            suffix = f"~{i}"
            name = base[:31 - len(suffix)] + suffix
            if name not in used:
                break
    used.add(name)
    return name


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
                   targets: dict | None = None,
                   matrix: dict | None = None) -> str:
    """Write the report workbook (requires pandas + an Excel writer).

    ``matrix`` is the run's persisted Shape Matrix (``rows.json``:
    ``{"info": [[key, value], ...], "rows": [...]}``). It is the sweep the
    cases were built from, so it ships in the same workbook as the
    measurements instead of being a separate download the reader has to
    correlate by hand.
    """
    import pandas as pd

    rich = enrich(records, peaks)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    summary = summarize(rich)
    by_op: dict[str, list[dict]] = defaultdict(list)
    for r in rich:
        by_op[r.get("op", "?")].append(r)
    # Op columns drop the redundant "Op" and sort prefill before decode so a
    # kernel's two regimes read as two blocks.
    op_cols = [(k, label) for k, label in COLUMNS if k != "op"]
    phase_order = {"prefill": 0, "decode": 1}
    matrix = matrix or {}

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        info = matrix.get("info") or []
        if info:
            pd.DataFrame([{"Key": k, "Value": v} for k, v in info]).to_excel(
                xl, sheet_name="Info", index=False)
        pd.DataFrame(summary).to_excel(xl, sheet_name="Summary", index=False)
        used: set[str] = {"Info", "Summary", "Coverage", "Targets",
                          "Shape Matrix"}
        if targets and targets.get("targets"):
            pd.DataFrame(target_rows(targets["targets"])).to_excel(
                xl, sheet_name="Targets", index=False)
            for phase, sec in (targets.get("by_phase") or {}).items():
                if not sec.get("targets"):
                    continue
                pd.DataFrame(target_rows(sec["targets"])).to_excel(
                    xl, sheet_name=f"Targets {phase}"[:31], index=False)
        for row in summary:
            op = row["Op"]
            recs = sorted(by_op.get(op, []),
                          key=lambda r: (phase_order.get(r.get("phase"), 9),
                                         -float(r.get("weighted_us") or 0)))
            pd.DataFrame([{label: r.get(key) for key, label in op_cols}
                          for r in recs]).to_excel(
                xl, sheet_name=sheet_name(op, used), index=False)
        cov = coverage(rich)
        if cov:
            pd.DataFrame(cov).to_excel(xl, sheet_name="Coverage", index=False)
        rows = matrix.get("rows") or []
        if rows:
            pd.DataFrame([{h: r.get(h) for h in MATRIX_HEADERS} for r in rows]
                         ).to_excel(xl, sheet_name="Shape Matrix", index=False)
    return path


def target_rows(targets: list[dict]) -> list[dict[str, Any]]:
    """The ranking as spreadsheet rows."""
    return [{
        "Rank": t["rank"], "Op": t["op"], "Backend": t["backend"],
        "e2e (us)": t["e2e_us"], "Share": t["share_of_e2e"],
        "Util": t["roofline"]["util"],
        "Util (DRAM)": t["roofline"].get("util_dram"),
        "Util (headroom)": t["roofline"].get("effective_util",
                                             t["roofline"]["util"]),
        "Bound": t["roofline"]["bound"],
        "Roof": t["roofline"].get("unit",
                                  t["roofline"].get("memory_level", "")),
        "AI (flop/byte)": t["roofline"].get("ai"),
        "Savings (us)": t["savings_us"]["total"],
        "Action": t["action"],
        "Kernel dir": (t.get("kernel") or {}).get("kernel_dir", ""),
        "Flags": "; ".join(t.get("flags") or []),
    } for t in targets]


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
