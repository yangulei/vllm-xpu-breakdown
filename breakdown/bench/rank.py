# SPDX-License-Identifier: Apache-2.0
"""Rank benchmarked ops by end-to-end optimization value.

The step between *"every op is benchmarked"* and *"optimize this kernel"*. A
latency table cannot say what is worth a session; two signals together can:

1. **e2e weight** - how often the op runs, from the Shape Matrix ``Layers``
   count carried on each case. A 10 us op in 57 layers outranks a 500 us op
   that runs once.
2. **roofline headroom** - the measured latency against the device peaks, using
   the op's analytic FLOPs / bytes. An op at or above ``target_util`` is
   ``at_roofline``: it is dropped *before* a session is spent on it. Which roof
   applies is decided by the op's *arithmetic intensity* against the machine
   balance, and a cache-resident op is charged to cache bandwidth rather than
   DRAM (see :mod:`breakdown.bench.estimate`).

There is deliberately no third "faster provider" signal. Replay measures the
kernel vLLM actually dispatched - there is no second implementation to compare
against - so a provider-gap column would either be empty or invented. What
replaces it is the **traced-vs-replayed** check: a replay that is much faster
than the profile's device time for the same shape means the replay is not doing
the model's work, and the target is flagged rather than trusted.

The result is ``targets.json``, the versioned contract consumed by the
``xpu-kernel-optimizer`` skill: kernel dir/files, build/bench/test commands,
baseline latency, roofline bound and the shapes to optimize at.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from breakdown.bench import devices, estimate

#: Bump when the meaning of a ``targets.json`` field changes; the optimizer
#: skill reads this before trusting the file.
#:
#: v2 - replay model: ``provider``/``switch_provider`` removed, ops keyed by
#: dispatch name, ``traced_device_time_us`` added.
#: v3 - roofline: ``roofline.bound`` now comes from the op's arithmetic
#: intensity against the machine balance (not from whichever utilization was
#: larger), and a cache-resident op is measured against the last-level-cache
#: bandwidth - ``roofline.memory_level`` says which roof was used, alongside
#: the new ``cache_bw_gbs`` / ``cache_bytes`` / ``ridge_ai`` fields.
SCHEMA_VERSION = 3

#: Fraction of the roofline treated as "done" - above it, only a redesign helps.
DEFAULT_TARGET_UTIL = 0.8

#: A replayed latency below this fraction of the profile's device time means
#: the replay is not doing the same work (an early-exit kernel, an index map
#: that touches nothing) and the target is flagged instead of trusted.
FIDELITY_FLOOR = 0.25

#: Above this utilization the roofline is not believable: Memory/FLOPs are
#: analytic estimates (an embedding or a paged-KV insert is charged for the
#: whole table it *could* read, not the handful of rows it touches), so a
#: 350x-of-peak number means the cost model is wrong for that op, not that the
#: kernel is done. Such ops are reported as ``check_cost_model`` rather than
#: silently retired as ``at_roofline``. This is now the *last* resort: an op
#: that merely ran out of a cache is no longer flagged here, because the
#: cache-bandwidth roof explains it (:func:`estimate.effective_bw_gbs`).
MAX_CREDIBLE_UTIL = 1.2

_KERNEL_SOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "kernel_sources.json")


@dataclass
class RankConfig:
    target_util: float = DEFAULT_TARGET_UTIL
    phases: tuple[str, ...] = ("prefill", "decode")
    phase_weight: dict[str, float] = field(
        default_factory=lambda: {"prefill": 1.0, "decode": 1.0})
    #: phase -> (seq_len, ctx_len, batch_size); ``None`` = the most benchmarked
    points: dict[str, tuple | None] = field(default_factory=dict)
    tp: int | None = None
    sku: str = ""
    peak_bw_gbs: float = 0.0
    peak_tflops: float = 0.0
    top: int = 0
    min_share: float = 0.0
    shapes_per_target: int = 3
    run_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def load_kernel_sources(path: str | None = None) -> dict:
    with open(path or _KERNEL_SOURCES_PATH) as fh:
        return json.load(fh)


def kernel_info(sources: dict, op: str, backend: str) -> dict[str, Any]:
    """Where this op's kernel lives, if it has editable source at all.

    An op with no entry (an ATen op backed by oneDNN, a collective backed by
    oneCCL) is not un-optimizable - but the lever is dispatch, layout or runtime
    configuration, not a kernel edit, which is what ``tune_config`` says.
    """
    base = dict((sources.get("backends") or {}).get(backend) or {})
    base.update((sources.get("ops") or {}).get(op) or {})
    return base


def _point_of(rec: dict) -> tuple:
    return (rec.get("seq_len"), rec.get("ctx_len"), rec.get("batch_size"))


def _points_of(rec: dict, phase: str) -> list[tuple]:
    """Every sweep point this record stands for, within ``phase``.

    A case whose operands do not depend on a swept dimension is measured once
    and represents several points (``BenchCase.points``). Matching only its
    stored coordinates would drop it from every operating point but the first -
    which silently removed the MoE grouped GEMM, the dominant kernel, from the
    ranking.
    """
    pts = rec.get("points") or []
    out = [tuple(p[1:]) for p in pts
           if isinstance(p, (list, tuple)) and len(p) == 4 and p[0] == phase]
    if out:
        return out
    return [_point_of(rec)] if rec.get("phase") == phase else []


def pick_point(records: list[dict], phase: str,
               want: tuple | None = None) -> tuple | None:
    """The operating point to rank at: the requested one, else the busiest."""
    pts = Counter(p for r in records if r.get("status") == "ok"
                  for p in _points_of(r, phase))
    if not pts:
        return None
    if want is not None:
        for p in pts:
            if tuple(p) == tuple(want):
                return p
    return pts.most_common(1)[0][0]


def _fidelity(rec: dict) -> tuple[float, str]:
    """``(replayed / traced, note)`` for a case measured at the profiled shape."""
    traced = float(rec.get("traced_device_time_us") or 0)
    lat = float(rec.get("latency_us") or 0)
    if not rec.get("traced_comparable") or traced <= 0 or lat <= 0:
        return 0.0, ""
    ratio = lat / traced
    if ratio < FIDELITY_FLOOR:
        return ratio, (f"replay is {1 / max(ratio, 1e-9):.1f}x faster than the "
                       f"profiled device time - the replayed arguments may not "
                       f"reproduce the model's work")
    return ratio, ""


def rank(records: Iterable[dict], rc: RankConfig | None = None,
         kernel_sources: dict | None = None) -> dict[str, Any]:
    """Rank ops by the end-to-end time an optimization would recover."""
    rc = rc or RankConfig()
    recs = [r for r in records if r.get("status") == "ok"]
    if not recs:
        raise ValueError("no successful benchmark records to rank")
    if rc.tp is not None:
        recs = [r for r in recs if int(r.get("tp") or 1) == rc.tp] or recs

    device = Counter(r.get("device") for r in recs).most_common(1)[0][0]
    sku = rc.sku or devices.sku_for_device(devices.device_name(device))
    peak = devices.peaks(sku)
    peaks = dict(peak)
    peaks["bw_gbs"] = rc.peak_bw_gbs or peak["bw_gbs"]
    peaks["tflops"] = rc.peak_tflops or peak["tflops"]
    sources = kernel_sources or load_kernel_sources()

    points = {ph: pick_point(recs, ph, rc.points.get(ph)) for ph in rc.phases}
    op_time: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    op_cases: dict[str, list[dict]] = defaultdict(list)
    op_backend: dict[str, Counter] = defaultdict(Counter)
    op_util: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    op_flags: dict[str, list[str]] = defaultdict(list)

    for r in recs:
        for ph in rc.phases:
            if points.get(ph) is None:
                continue
            if points[ph] not in _points_of(r, ph):
                continue
            op = r["op"]
            calls = max(int(r.get("layers") or 1), 1)
            lat = float(r.get("latency_us") or 0)
            weighted = lat * calls
            op_time[op][ph] += weighted
            op_backend[op][r.get("backend") or ""] += 1
            util, bound, level = estimate.utilization_detail(
                lat, float(r.get("flops") or 0), float(r.get("bytes") or 0),
                peaks)
            op_util[op].append((weighted, util, bound, level))
            ratio, note = _fidelity(r)
            if note:
                op_flags[op].append(note)
            op_cases[op].append({
                "phase": ph, "calls": calls, "shape": r.get("shape"),
                "latency_us": round(lat, 3), "weighted_us": round(weighted, 1),
                "traced_device_time_us": r.get("traced_device_time_us"),
                "replay_vs_traced": round(ratio, 3) if ratio else None,
                "case_id": r.get("case_id"),
            })

    if not op_time:
        raise ValueError(
            "no benchmark record fell on an operating point - check the phase "
            "and tp filters")

    def total(op: str) -> float:
        return sum(op_time[op][p] * rc.phase_weight.get(p, 1.0)
                   for p in rc.phases)

    grand = sum(total(op) for op in op_time)
    targets: list[dict[str, Any]] = []
    for op in op_time:
        t = total(op)
        uw = op_util[op]
        wsum = sum(w for w, _, _, _ in uw) or 1.0
        util = sum(w * u for w, u, _, _ in uw) / wsum
        bound = Counter(b for _, _, b, _ in uw).most_common(1)[0][0]
        level = Counter(l for _, _, _, l in uw).most_common(1)[0][0]
        backend = (op_backend[op].most_common(1)[0][0]
                   if op_backend[op] else "")
        info = kernel_info(sources, op, backend)
        buildable = bool(info.get("build_cmd"))
        credible = util <= MAX_CREDIBLE_UTIL
        headroom = max(0.0, 1.0 - util / rc.target_util)
        save = t * headroom if (buildable and credible) else 0.0
        if not credible:
            action = "check_cost_model"
            roof = ("cache" if level == "cache" else "DRAM")
            op_flags[op].append(
                f"roofline utilization {util * 100:.0f}% exceeds the {roof} "
                f"{bound} peak - the analytic FLOPs/bytes for this op overstate "
                f"the traffic it really does, so its headroom cannot be trusted")
        elif util >= rc.target_util:
            action = "at_roofline"
        elif buildable:
            action = "optimize_kernel"
        else:
            action = "tune_config"

        shapes = sorted(op_cases[op], key=lambda c: -c["weighted_us"])
        targets.append({
            "op": op,
            "backend": backend,
            "e2e_us": round(t, 1),
            "phase_us": {ph: round(op_time[op][ph], 1) for ph in rc.phases},
            "calls": sum(c["calls"] for c in op_cases[op]),
            "roofline": {"util": round(util, 3), "bound": bound,
                         "memory_level": level,
                         "peak_bw_gbs": peaks["bw_gbs"],
                         "peak_tflops": peaks["tflops"],
                         "cache_bw_gbs": peaks.get("cache_bw_gbs", 0.0),
                         "cache_bytes": peaks.get("cache_bytes", 0.0),
                         "ridge_ai": round(estimate.ridge_ai(peaks), 2),
                         "target_util": rc.target_util},
            "savings_us": {"optimize_kernel": round(save, 1),
                           "total": round(save, 1)},
            "action": action,
            "flags": sorted(set(op_flags[op])),
            "kernel": info,
            "top_shapes": [_shape_entry(op, c, rc)
                           for c in shapes[:rc.shapes_per_target]],
        })

    for t in targets:
        t["share_of_e2e"] = round(t["e2e_us"] / grand, 4) if grand else 0.0
    targets = [t for t in targets if t["share_of_e2e"] >= rc.min_share]
    targets.sort(key=lambda t: (-t["savings_us"]["total"], -t["e2e_us"]))
    for i, t in enumerate(targets, 1):
        t["rank"] = i
    if rc.top:
        targets = targets[:rc.top]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": "replay",
        "device": device,
        "sku": sku,
        "peaks": peaks,
        "target_util": rc.target_util,
        "tp": rc.tp,
        "run_id": rc.run_id,
        "operating_points": {
            p: ({"seq_len": points[p][0], "ctx_len": points[p][1],
                 "batch_size": points[p][2]} if points.get(p) else None)
            for p in rc.phases},
        "phase_weight": dict(rc.phase_weight),
        "e2e_us_total": round(grand, 1),
        "provenance": rc.provenance,
        "targets": targets,
    }


def _shape_entry(op: str, case: dict, rc: RankConfig) -> dict[str, Any]:
    run = rc.run_id or "<run_id>"
    bench = (f"python3 -m breakdown.bench case --run {run} "
             f"--case-id {case['case_id']}")
    return {
        "phase": case["phase"],
        "calls": case["calls"],
        "shape": case["shape"],
        "latency_us": case["latency_us"],
        "weighted_us": case["weighted_us"],
        "traced_device_time_us": case["traced_device_time_us"],
        "replay_vs_traced": case["replay_vs_traced"],
        "bench_cmd": bench,
        "profile_cmd": f"unitrace -d --chrome-kernel-logging {bench}",
    }


def _bound_label(roofline: dict[str, Any]) -> str:
    """``memory`` on the cache roof reads ``mem/$`` so the roof is visible."""
    bound = roofline.get("bound") or ""
    if bound == "memory" and roofline.get("memory_level") == "cache":
        return "mem/$"
    return bound


def format_table(doc: dict[str, Any]) -> str:
    """The ranking as a console table."""
    out = [
        f"device      : {doc['device']} / {doc['sku']}  (peaks "
        f"{doc['peaks']['bw_gbs']:.0f} GB/s DRAM"
        + (f" / {doc['peaks']['cache_bw_gbs']:.0f} GB/s cache"
           if doc['peaks'].get('cache_bw_gbs') else "")
        + f", {doc['peaks']['tflops']:.1f} "
        f"TFLOPS, target util {doc['target_util']:.0%})",
    ]
    for phase, p in doc["operating_points"].items():
        if p:
            out.append(f"{phase:<12}: seq={p['seq_len']} ctx={p['ctx_len']} "
                       f"bs={p['batch_size']}")
    out.append(f"e2e op time : {doc['e2e_us_total'] / 1000:.2f} ms/step over "
               f"{len(doc['targets'])} ops\n")
    hdr = (f"{'#':>2} {'op':<44}{'backend':<18}{'e2e_us':>10}{'share':>7}"
           f"{'util':>6} {'bound':<8}{'save_us':>9}  action")
    out += [hdr, "-" * len(hdr)]
    for t in doc["targets"]:
        flag = "  !" if t["flags"] else ""
        out.append(
            f"{t['rank']:>2} {t['op'][:44]:<44}{t['backend'][:18]:<18}"
            f"{t['e2e_us']:>10.1f}{t['share_of_e2e'] * 100:>6.1f}%"
            f"{t['roofline']['util'] * 100:>5.0f}% "
            f"{_bound_label(t['roofline']):<8}"
            f"{t['savings_us']['total']:>9.1f}  {t['action']}{flag}")
    flagged = [t for t in doc["targets"] if t["flags"]]
    if flagged:
        out.append("\n! fidelity warnings:")
        for t in flagged:
            out.append(f"  {t['op']}: {t['flags'][0]}")
    return "\n".join(out)
