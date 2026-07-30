# SPDX-License-Identifier: Apache-2.0
"""Read micro_perf report trees: records for the ranker, workbook for humans.

``records()`` flattens a report tree into one dict per benchmarked case (the
input :mod:`breakdown.perf.rank` consumes). ``merge()`` writes the
cross-platform Excel workbook.

Reads the ``reports*/<BACKEND>/<device>/<op>/<provider>/<op>-<provider>.jsonl``
trees produced by ``run_benchmark.sh`` and writes:

* **Summary** — one row per (op, provider, platform): case count and
  min/median/max latency, plus best bandwidth/TFLOPS, with the XPU-vs-CUDA
  median-latency speedup where both platforms ran the same op.
* **Platform compare** — per op, the median latency of each platform side by
  side (apple-to-apple view over the ops both dispatch).
* **one sheet per op** — one row per benchmarked *shape* (the arguments both
  platforms share, ``arg.`` prefix stripped, sorted by size), with each
  platform's metrics side by side as ``latency_us_CUDA`` / ``latency_us_XPU``
  etc., so the same shape can be compared across platforms on one line.

Usage::

    python3 converter/merge_reports.py --out reports_merged.xlsx \\
        --reports CUDA=reports_cuda --reports XPU=reports

``--reports`` may be given several times as ``LABEL=path`` (or just ``path``,
which then takes its label from the backend dir inside it). Missing report
trees are skipped with a warning.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from typing import Any

import pandas as pd

# metric key -> friendly column name (order matters for the detail sheets)
_METRICS = [
    ("latency(us)", "latency_us"),
    ("mem_bw(GB/s)", "mem_bw_GBs"),
    ("calc_flops_power(tflops)", "tflops"),
    ("algo_bw(GB/s)", "algo_bw_GBs"),
    ("bus_bw(GB/s)", "bus_bw_GBs"),
    ("read_bytes(B)", "read_bytes"),
    ("write_bytes(B)", "write_bytes"),
    ("io_bytes(B)", "io_bytes"),
    ("calc_flops", "calc_flops"),
    ("calc_mem_ratio", "calc_mem_ratio"),
]

_INVALID_SHEET = re.compile(r"[\[\]:*?/\\]")


def _sheet_name(op: str, used: set[str]) -> str:
    name = _INVALID_SHEET.sub("_", op)[:31] or "op"
    base, i = name, 1
    while name in used:
        suffix = f"~{i}"
        name = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def records(label: str, root: str) -> list[dict[str, Any]]:
    """Flatten every jsonl under a report tree into records.

    Report trees are ``.../<BACKEND>/<device>/<op>/<provider>/<file>.jsonl``;
    they may sit under extra grouping dirs (e.g. ``reports/compute/INTEL/...``),
    so the tree is walked recursively and the fields are read off the path tail.
    """
    out: list[dict[str, Any]] = []
    pattern = os.path.join(root, "**", "*.jsonl")
    for path in sorted(glob.glob(pattern, recursive=True)):
        parts = path.split(os.sep)
        if len(parts) < 5:
            continue
        backend, device, op, provider = parts[-5], parts[-4], parts[-3], parts[-2]
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            targets = raw.get("targets") or {}
            if "latency(us)" not in targets:
                continue
            rec: dict[str, Any] = {
                "platform": label,
                "backend": backend,
                "device": raw.get("sku_name", device),
                "op": raw.get("op_name", op),
                "provider": raw.get("provider", provider),
            }
            for key, val in (raw.get("arguments") or {}).items():
                if isinstance(val, (list, dict)):
                    val = json.dumps(val)
                rec[f"arg.{key}"] = val
            for key, col in _METRICS:
                if key in targets:
                    rec[col] = targets[key]
            kernels = targets.get("kernels")
            if kernels:
                rec["kernels"] = ", ".join(map(str, kernels))
            out.append(rec)
    return out


def _stats(values: list[float]) -> tuple[float, float, float]:
    return min(values), st.median(values), max(values)


def _numeric(series: pd.Series) -> pd.Series:
    """Numeric view of a column for sorting; non-numeric values sort last."""
    return pd.to_numeric(series, errors="coerce")


def _op_sheet(grp: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Pivot one op's cases into one row per shape, platforms side by side.

    Columns shared by every platform (the ``arg.*`` describing the shape, and
    ``provider`` when both platforms happen to use the same one) become the
    merge key and keep their bare name; everything that varies per platform
    (metrics, platform-only arguments, differing providers) is suffixed with
    the platform label -- ``latency_us_CUDA`` / ``latency_us_XPU``. Where a
    platform ran several providers for the op, the suffix carries the provider
    too (``latency_us_XPU_flash_xpu``) so they stay distinguishable.
    """
    grp = grp.dropna(axis=1, how="all")
    present = [l for l in labels if l in set(grp["platform"])]
    per_platform = {l: grp[grp["platform"] == l].dropna(axis=1, how="all")
                    for l in present}

    argcols = [c for c in grp.columns if c.startswith("arg.")]
    # shape key: arguments every platform reports (so the merge can line up)
    keycols = [c for c in argcols
               if all(c in sub.columns for sub in per_platform.values())]
    providers = {l: sorted(set(sub["provider"])) for l, sub in per_platform.items()}
    single_provider = all(len(p) == 1 for p in providers.values())
    same_provider = single_provider and len({p[0] for p in providers.values()}) == 1
    if same_provider:
        keycols = ["provider"] + keycols

    # one "series" per (platform, provider) that needs its own metric columns
    series: list[tuple[str, str | None, str]] = []
    for label in present:
        for provider in providers[label]:
            suffix = label if len(providers[label]) == 1 else f"{label}_{provider}"
            series.append((label, provider, suffix))

    metriccols = [c for _, c in _METRICS if c in grp]
    merged: pd.DataFrame | None = None
    extra_order: list[str] = []
    for label, provider, suffix in series:
        sub = per_platform[label]
        sub = sub[sub["provider"] == provider].dropna(axis=1, how="all")
        # the provider is already spelled out in the suffix when a platform ran
        # more than one, so only carry it as a column when it is not
        drop = keycols + ["op", "platform", "device", "backend"]
        if same_provider or len(providers[label]) > 1:
            drop.append("provider")
        others = [c for c in sub.columns if c not in drop]
        cols = keycols + others
        piece = sub[cols].rename(
            columns={c: f"{c}_{suffix}" for c in others})
        extra_order += [f"{c}_{suffix}" for c in others if c not in metriccols]
        piece = piece.drop_duplicates(subset=keycols)
        merged = piece if merged is None else merged.merge(
            piece, on=keycols, how="outer")

    assert merged is not None
    merged.insert(0, "op", grp["op"].iloc[0])

    # shape-derived metrics (bytes / FLOPs) come from the shared op_def, so they
    # are identical for every platform+provider at a given shape -- collapse
    # those into a single unsuffixed column instead of repeating them N times.
    for metric in metriccols:
        cols = [f"{metric}_{s}" for _, _, s in series if f"{metric}_{s}" in merged]
        if len(cols) < 2:
            continue
        combined = merged[cols[0]]
        agree = True
        for c in cols[1:]:
            both = merged[cols[0]].notna() & merged[c].notna()
            if not merged.loc[both, cols[0]].equals(merged.loc[both, c]):
                agree = False
                break
            combined = combined.combine_first(merged[c])
        if agree:
            merged = merged.drop(columns=cols[1:]).rename(
                columns={cols[0]: metric})
            merged[metric] = combined

    merged.columns = [c[4:] if c.startswith("arg.") else c
                      for c in merged.columns]
    keycols = [c[4:] if c.startswith("arg.") else c for c in keycols]

    # metric-major ordering so the platforms sit next to each other
    ordered = ["op"] + keycols
    for metric in metriccols:
        ordered += [c for c in ([metric] + [f"{metric}_{s}" for _, _, s in series])
                    if c in merged.columns and c not in ordered]
    ordered += [c for c in extra_order if c in merged.columns]
    ordered += [c for c in merged.columns if c not in ordered]
    merged = merged[ordered]

    # sort by the shape key, numerically where the values are numbers
    if keycols:
        order = sorted(keycols, key=lambda c: (c != "num_tokens", keycols.index(c)))
        merged = merged.sort_values(
            by=order, key=lambda s: _numeric(s).fillna(float("inf"))
            if _numeric(s).notna().any() else s.astype(str))
    return merged.reset_index(drop=True)


def merge(report_specs: list[str], out: str = "reports_merged.xlsx",
          baseline: str | None = None, log=print) -> dict[str, Any]:
    """Merge report trees into one workbook.

    ``report_specs`` are ``LABEL=DIR`` (or bare ``DIR``, labelled from the
    backend directory inside it). Returns a small summary dict.
    """
    class args:  # noqa: N801  (keeps the ported body readable)
        pass
    args.reports = report_specs
    args.out = out
    args.baseline = baseline


    all_records: list[dict[str, Any]] = []
    labels: list[str] = []
    for spec in args.reports:
        label, _, path = spec.partition("=")
        if not path:
            path, label = label, ""
        if not os.path.isdir(path):
            log(f"warning: no such report dir, skipped: {path}")
            continue
        if not label:
            subs = [d for d in sorted(os.listdir(path))
                    if os.path.isdir(os.path.join(path, d))]
            label = subs[0] if subs else os.path.basename(path.rstrip("/"))
        recs = records(label, path)
        if not recs:
            log(f"warning: no benchmark records under {path}, skipped")
            continue
        log(f"{label:8s} {len(recs):5d} cases from {path}")
        all_records += recs
        labels.append(label)

    if not all_records:
        raise SystemExit("no report records found — nothing to merge")

    df = pd.DataFrame(all_records)
    baseline = args.baseline or labels[0]

    # ---------------- summary: one row per (op, provider, platform) ---------
    summary_rows = []
    for (op, provider, platform), grp in df.groupby(
            ["op", "provider", "platform"], sort=True):
        lat = [v for v in grp.get("latency_us", []) if pd.notna(v)]
        if not lat:
            continue
        lo, med, hi = _stats(list(lat))
        row = {
            "op": op, "provider": provider, "platform": platform,
            "device": grp["device"].iloc[0],
            "cases": len(grp),
            "latency_min_us": round(lo, 3),
            "latency_median_us": round(med, 3),
            "latency_max_us": round(hi, 3),
            "latency_total_us": round(sum(lat), 3),
        }
        for col, label in (("mem_bw_GBs", "mem_bw_max_GBs"),
                           ("tflops", "tflops_max"),
                           ("algo_bw_GBs", "algo_bw_max_GBs"),
                           ("bus_bw_GBs", "bus_bw_max_GBs")):
            if col in grp:
                vals = [v for v in grp[col] if pd.notna(v)]
                if vals:
                    row[label] = round(max(vals), 3)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["op", "platform", "provider"]).reset_index(drop=True)

    # -------- platform compare: median latency per op, platform columns -----
    med_by_op: dict[str, dict[str, float]] = defaultdict(dict)
    for op, grp in df.groupby("op"):
        for platform, sub in grp.groupby("platform"):
            lat = [v for v in sub["latency_us"] if pd.notna(v)]
            if lat:
                # best provider on that platform represents it
                best = min(
                    (st.median([v for v in p["latency_us"] if pd.notna(v)])
                     for _, p in sub.groupby("provider")
                     if any(pd.notna(v) for v in p["latency_us"])),
                    default=st.median(lat))
                med_by_op[op][platform] = round(best, 3)
    compare_rows = []
    for op in sorted(med_by_op):
        row: dict[str, Any] = {"op": op}
        for label in labels:
            row[f"{label}_median_us"] = med_by_op[op].get(label)
        others = [l for l in labels if l != baseline]
        base_v = med_by_op[op].get(baseline)
        for other in others:
            other_v = med_by_op[op].get(other)
            if base_v and other_v:
                row[f"{other}_vs_{baseline}_speedup"] = round(base_v / other_v, 3)
        row["platforms"] = ", ".join(
            l for l in labels if l in med_by_op[op])
        compare_rows.append(row)
    compare = pd.DataFrame(compare_rows)

    # ---------------------------- write workbook ---------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        if len(labels) > 1:
            compare.to_excel(writer, sheet_name="Platform compare", index=False)

        used = {"Summary", "Platform compare"}
        for op, grp in df.groupby("op", sort=True):
            sheet = _op_sheet(grp, labels)
            sheet.to_excel(writer, sheet_name=_sheet_name(op, used), index=False)

        # auto-ish column widths
        for ws in writer.book.worksheets:
            for cells in ws.columns:
                width = max((len(str(c.value)) for c in cells if c.value is not None),
                            default=8)
                ws.column_dimensions[cells[0].column_letter].width = min(
                    max(width + 2, 10), 42)
            ws.freeze_panes = "A2"

    log(f"\nwrote {args.out}")
    log(f"  Summary: {len(summary)} (op, provider, platform) rows")
    if len(labels) > 1:
        log(f"  Platform compare: {len(compare)} ops across {labels}")
    log(f"  detail sheets: {df['op'].nunique()} ops, {len(df)} cases total")

    return {"out": out, "labels": labels, "cases": len(df),
            "ops": int(df["op"].nunique())}
