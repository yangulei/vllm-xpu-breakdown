# SPDX-License-Identifier: Apache-2.0
"""Rank benchmarked ops by end-to-end optimization value.

The step between *"every op is benchmarked"* and *"optimize this kernel"*. A
latency table cannot say what is worth a session; three signals together can:

1. **e2e weight** - how often the op runs, from the Shape Matrix ``Layers``
   column at one operating point. A 10 us op in 57 layers outranks a 500 us op
   that runs once.
2. **roofline headroom** - achieved bandwidth / FLOPS vs the device peaks, from
   the ``io_bytes`` / ``calc_flops`` micro_perf already records. An op at or
   above ``target_util`` is ``at_roofline``: it is dropped *before* a session is
   spent on it.
3. **provider gap** - a faster provider than the one actually dispatched is a
   free win (``switch_provider``: a dispatch/config change, not a kernel
   rewrite), and must be harvested before any kernel work.

The result is ``opt_targets.json``, the versioned contract consumed by the
``xpu-kernel-optimizer`` skill: kernel dir/files, build/bench/profile/test
commands, baseline latency, roofline bound and the shapes to optimize at.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from breakdown.perf import devices
from breakdown.perf.matrix_reader import OpRow, unique_rows
from breakdown.perf.op_map import ModelConfig, get_dispatch

#: Bump when the meaning of an ``opt_targets.json`` field changes; the
#: optimizer skill reads this before trusting the file.
SCHEMA_VERSION = 1

#: Fraction of the roofline treated as "done" - above it, only a redesign helps.
DEFAULT_TARGET_UTIL = 0.8

#: Minimum relative gain before a provider switch is worth reporting.
DEFAULT_SWITCH_MARGIN = 0.10

#: breakdown "Backend" column (and classifier backends) -> micro_perf provider
BACKEND_ALIASES = {
    "flash_xpu": "flash_xpu", "xattention": "flash_xpu", "deepklox": "flash_xpu",
    "triton": "triton", "triton_xpu": "triton",
    "vllm_xpu_kernels": "vllm_xpu_kernels", "vllm-xpu-kernels": "vllm_xpu_kernels",
    "_c": "vllm_xpu_kernels", "_xpu_c": "vllm_xpu_kernels",
    "_moe_c": "vllm_xpu_kernels", "custom": "vllm_xpu_kernels",
    "sycl": "vllm_xpu_kernels", "vllm_cuda_kernels": "vllm_cuda_kernels",
    "flashinfer": "flashinfer_gemma",
    "onednn": "onednn", "aten": "torch", "torch": "torch",
    "torch-xpu-ops": "torch", "pytorch": "torch",
    "xccl": "xccl", "ccl": "xccl", "oneccl": "xccl",
    "sycl_tla": "sycl_tla", "cutlass": "sycl_tla", "sycl_ext": "sycl_ext",
    "ipex": "ipex",
}

_KERNEL_SOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "kernel_sources.json")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def norm(v: Any) -> str:
    """Normalized string form of a case-argument value (1 == 1.0 == '1')."""
    if isinstance(v, float) and math.isfinite(v) and float(v).is_integer():
        v = int(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    return str(v).strip()


def _asint(v: Any) -> Any:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_kernel_sources(path: str | None = None) -> dict:
    with open(path or _KERNEL_SOURCES_PATH) as fh:
        return json.load(fh)


def kernel_info(sources: dict, op: str, provider: str) -> dict:
    """Kernel location + build/test commands for an (op, provider)."""
    info = dict(sources.get("providers", {}).get(provider, {}))
    info.update(sources.get("ops", {}).get(f"{provider}:{op}", {}))
    return info


def index_records(records: Iterable[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        args = {k[4:]: norm(v) for k, v in r.items()
                if k.startswith("arg.") and v is not None}
        idx[r["op"]].append({**r, "_args": args})
    return idx


def match_case(records: list[dict], args: dict) -> list[dict]:
    """Records whose arguments agree with every key of ``args``."""
    want = {k: norm(v) for k, v in args.items()}
    return [r for r in records
            if all(k in r["_args"] and r["_args"][k] == v
                   for k, v in want.items())]


def utilization(rec: dict, peak_bw: float, peak_tf: float) -> tuple[float, str]:
    """Roofline fraction this case achieves, and which resource binds it."""
    lat_s = (rec.get("latency_us") or 0.0) / 1e6
    bw = rec.get("mem_bw_GBs")
    if not bw and rec.get("io_bytes") and lat_s > 0:
        bw = rec["io_bytes"] / lat_s / 1e9
    tf = rec.get("tflops")
    if not tf and rec.get("calc_flops") and lat_s > 0:
        tf = rec["calc_flops"] / lat_s / 1e12
    u_bw = (bw or 0.0) / peak_bw
    u_tf = (tf or 0.0) / peak_tf
    return max(u_bw, u_tf), ("memory" if u_bw >= u_tf else "compute")


# --------------------------------------------------------------------------
# call weighting
# --------------------------------------------------------------------------
@dataclass
class WeightedCase:
    op: str
    args: dict
    calls: int


def pick_operating_point(rows: list[OpRow], phase: str) -> tuple:
    """Heaviest sweep point of a phase: the one worth optimizing for.

    prefill -> longest sequence at the smallest context; decode -> largest
    context at the median batch.
    """
    pts = {(_asint(r.seq_len), _asint(r.ctx_len), _asint(r.batch_size))
           for r in rows if r.phase == phase}
    pts = {p for p in pts if any(x is not None for x in p)}
    if not pts:
        return (None, None, None)
    if phase == "prefill":
        seq = max((p[0] for p in pts if p[0] is not None), default=None)
        cand = [p for p in pts if p[0] == seq]
        ctx = min((p[1] for p in cand if p[1] is not None), default=None)
        cand = [p for p in cand if p[1] == ctx]
        bs = min((p[2] for p in cand if p[2] is not None), default=None)
        return (seq, ctx, bs)
    ctx = max((p[1] for p in pts if p[1] is not None), default=None)
    cand = sorted(p for p in pts if p[1] == ctx and p[2] is not None)
    return (None, ctx, cand[len(cand) // 2][2] if cand else None)


def weigh_cases(rows: list[OpRow], cfg: ModelConfig, phase: str,
                point: tuple | None = None, tp: int | None = None,
                dispatch: str = "xpu") -> tuple[list[WeightedCase],
                                                dict[str, Counter], tuple]:
    """Micro_perf cases at one operating point, weighted by call count.

    ``calls`` sums the ``Layers`` of every module that dispatches the op, i.e.
    how many times one forward pass runs it.
    """
    mod = get_dispatch(dispatch)
    sel = [r for r in rows if tp is None or _asint(r.tp) == tp]
    if not point or all(x is None for x in point):
        point = pick_operating_point(sel, phase)
    seq, ctx, bs = point

    def keeps(r: OpRow) -> bool:
        if r.phase != phase:
            return False
        if seq is not None and phase == "prefill" and _asint(r.seq_len) != seq:
            return False
        if ctx is not None and _asint(r.ctx_len) != ctx:
            return False
        if bs is not None and _asint(r.batch_size) != bs:
            return False
        return True

    weights: dict[tuple, WeightedCase] = {}
    backends: dict[str, Counter] = defaultdict(Counter)
    for r in unique_rows([r for r in sel if keeps(r)],
                         getattr(mod, "DENSE_SWEEP_OPS", set())):
        if r.op_name in mod.SKIP_OPS:
            continue
        adapter = mod.ADAPTERS.get(r.op_name)
        if adapter is None:
            continue
        layers = _asint(r.layers) or 1
        for ec in adapter(r, cfg) or []:
            key = (ec.op, json.dumps({k: norm(v) for k, v in ec.args.items()},
                                     sort_keys=True))
            wc = weights.setdefault(key, WeightedCase(ec.op, ec.args, 0))
            wc.calls += layers
            backends[ec.op][(r.backend or "").strip().lower()] += layers
    return list(weights.values()), backends, (seq, ctx, bs)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
@dataclass
class RankConfig:
    dispatch: str = "xpu"
    tp: int | None = 4
    phases: tuple[str, ...] = ("prefill", "decode")
    points: dict[str, tuple] = field(default_factory=dict)
    phase_weight: dict[str, float] = field(default_factory=dict)
    sku: str | None = None
    peak_bw_gbs: float | None = None
    peak_tflops: float | None = None
    target_util: float = DEFAULT_TARGET_UTIL
    switch_margin: float = DEFAULT_SWITCH_MARGIN
    min_share: float = 0.005
    top: int = 0
    shapes_per_target: int = 3
    backend: str = "INTEL"
    provenance: dict[str, Any] = field(default_factory=dict)


def rank(rows: list[OpRow], records: Iterable[dict], cfg: ModelConfig,
         rc: RankConfig | None = None,
         kernel_sources: dict | None = None) -> dict[str, Any]:
    """Rank ops by the end-to-end time an optimization would recover."""
    rc = rc or RankConfig()
    ridx = index_records(records)
    if not ridx:
        raise ValueError("no benchmark records to rank")
    device = Counter(r.get("device") for rs in ridx.values()
                     for r in rs).most_common(1)[0][0]
    sku = rc.sku or devices.sku_for_device(device)
    peak = devices.peaks(sku)
    peak_bw = rc.peak_bw_gbs or peak["bw_gbs"]
    peak_tf = rc.peak_tflops or peak["tflops"]
    sources = kernel_sources or load_kernel_sources()

    op_time: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    op_util: dict[str, dict[str, list[tuple[float, float, str]]]] = defaultdict(
        lambda: defaultdict(list))
    op_cases: dict[str, list[dict]] = defaultdict(list)
    op_backend: dict[str, Counter] = defaultdict(Counter)
    points: dict[str, tuple] = {}
    unmatched: Counter = Counter()

    for phase in rc.phases:
        cases, backends, point = weigh_cases(
            rows, cfg, phase, rc.points.get(phase), rc.tp, rc.dispatch)
        points[phase] = point
        for op, ctr in backends.items():
            op_backend[op].update(ctr)
        for wc in cases:
            hits = match_case(ridx.get(wc.op, []), wc.args)
            if not hits:
                unmatched[wc.op] += 1
                continue
            best_us = min(h["latency_us"] for h in hits)
            op_cases[wc.op].append({
                "phase": phase, "calls": wc.calls, "args": wc.args,
                "best_latency_us": best_us,
                "weighted_us": best_us * wc.calls,
            })
            for h in hits:
                t = h["latency_us"] * wc.calls
                op_time[wc.op][h["provider"]][phase] += t
                u, bound = utilization(h, peak_bw, peak_tf)
                op_util[wc.op][h["provider"]].append((t, u, bound))

    if not op_time:
        raise ValueError(
            "no benchmarked case matched the shape matrix - check tp / "
            "dispatch / that the reports and workloads came from the same run")

    def total(op: str, prov: str) -> float:
        return sum(op_time[op][prov][p] * rc.phase_weight.get(p, 1.0)
                   for p in rc.phases)

    targets: list[dict[str, Any]] = []
    grand = 0.0
    for op in op_time:
        provs = list(op_time[op])
        dispatched, source = None, "assumed"
        for be, _ in op_backend.get(op, Counter()).most_common():
            alias = BACKEND_ALIASES.get(be)
            if alias in provs:
                dispatched, source = alias, "trace"
                break
        if dispatched is None:
            dispatched = min(provs, key=lambda p: total(op, p))
        best = min(provs, key=lambda p: total(op, p))
        t_disp, t_best = total(op, dispatched), total(op, best)
        grand += t_disp

        uw = op_util[op][best]
        wsum = sum(w for w, _, _ in uw) or 1.0
        util = sum(w * u for w, u, _ in uw) / wsum
        bound = Counter(b for _, _, b in uw).most_common(1)[0][0]

        switch = (t_disp - t_best) if (
            best != dispatched and t_disp > 0
            and (t_disp - t_best) / t_disp >= rc.switch_margin) else 0.0
        info = kernel_info(sources, op, best)
        buildable = bool(info.get("build_cmd"))
        kernel_save = t_best * max(0.0, 1.0 - util / rc.target_util)
        if not buildable:
            # library / ATen / collective: a kernel session has nothing to edit,
            # the lever is dispatch, layout or runtime config
            kernel_save = 0.0
        if switch > 0 and switch >= kernel_save:
            action = "switch_provider"
        elif kernel_save > 0:
            action = "optimize_kernel"
        elif util >= rc.target_util:
            action = "at_roofline"
        else:
            action = "tune_config" if not buildable else "at_roofline"

        shapes = sorted(op_cases[op], key=lambda c: -c["weighted_us"])
        targets.append({
            "op": op,
            "dispatched_provider": dispatched,
            "dispatched_provider_source": source,
            "best_provider": best,
            "providers": {p: {ph: round(op_time[op][p][ph], 1)
                              for ph in rc.phases} for p in provs},
            "e2e_us": round(t_disp, 1),
            "e2e_us_best_provider": round(t_best, 1),
            "phase_us": {ph: round(op_time[op][dispatched][ph], 1)
                         for ph in rc.phases},
            "calls": sum(c["calls"] for c in op_cases[op]),
            "roofline": {"util": round(util, 3), "bound": bound,
                         "peak_bw_gbs": peak_bw, "peak_tflops": peak_tf,
                         "target_util": rc.target_util},
            "savings_us": {"switch_provider": round(switch, 1),
                           "optimize_kernel": round(kernel_save, 1),
                           "total": round(switch + kernel_save, 1)},
            "action": action,
            "kernel": info,
            "top_shapes": [_shape_entry(op, best, c, rc)
                           for c in shapes[:rc.shapes_per_target]],
        })

    for t in targets:
        t["share_of_e2e"] = round(t["e2e_us"] / grand, 4) if grand else 0.0
    targets = [t for t in targets if t["share_of_e2e"] >= rc.min_share]
    targets.sort(key=lambda t: -t["savings_us"]["total"])
    for i, t in enumerate(targets, 1):
        t["rank"] = i
    if rc.top:
        targets = targets[:rc.top]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "sku": sku,
        "peaks": {"bw_gbs": peak_bw, "tflops": peak_tf},
        "target_util": rc.target_util,
        "dispatch": rc.dispatch,
        "tp": rc.tp,
        "operating_points": {p: {"seq_len": points[p][0],
                                 "ctx_len": points[p][1],
                                 "batch_size": points[p][2]}
                             for p in rc.phases},
        "phase_weight": dict(rc.phase_weight),
        "e2e_us_total": round(grand, 1),
        "unmatched_cases": dict(unmatched),
        "provenance": rc.provenance,
        "targets": targets,
    }


def _shape_entry(op: str, provider: str, case: dict,
                 rc: RankConfig) -> dict[str, Any]:
    args = json.dumps(case["args"], sort_keys=True)
    bench = (f"python3 -m breakdown.perf bench --backend {rc.backend} "
             f"--op {op} --provider {provider} --repeat 20 --case '{args}'")
    return {
        "phase": case["phase"],
        "calls": case["calls"],
        "latency_us": round(case["best_latency_us"], 2),
        "weighted_us": round(case["weighted_us"], 1),
        "args": case["args"],
        "bench_cmd": bench,
        "profile_cmd": f"unitrace -d --chrome-kernel-logging {bench}",
    }


def format_table(doc: dict[str, Any]) -> str:
    """The ranking as a console table."""
    out = [
        f"device      : {doc['device']}  (peaks {doc['peaks']['bw_gbs']:.0f} "
        f"GB/s, {doc['peaks']['tflops']:.1f} TFLOPS, target util "
        f"{doc['target_util']:.0%})",
    ]
    for phase, p in doc["operating_points"].items():
        out.append(f"{phase:<12}: seq={p['seq_len']} ctx={p['ctx_len']} "
                   f"bs={p['batch_size']}")
    if doc["unmatched_cases"]:
        n = sum(doc["unmatched_cases"].values())
        out.append(f"unmatched   : {n} matrix cases had no benchmark record "
                   f"({', '.join(sorted(doc['unmatched_cases']))})")
    out.append(f"e2e op time : {doc['e2e_us_total'] / 1000:.2f} ms/step over "
               f"{len(doc['targets'])} ops\n")
    hdr = (f"{'#':>2} {'op':<24}{'provider':<17}{'e2e_us':>10}{'share':>7}"
           f"{'util':>6} {'bound':<8}{'save_us':>9}  action")
    out += [hdr, "-" * len(hdr)]
    for t in doc["targets"]:
        extra = (f" -> {t['best_provider']}"
                 if t["action"] == "switch_provider" else "")
        out.append(
            f"{t['rank']:>2} {t['op']:<24}{t['dispatched_provider']:<17}"
            f"{t['e2e_us']:>10.1f}{t['share_of_e2e'] * 100:>6.1f}%"
            f"{t['roofline']['util'] * 100:>5.0f}% {t['roofline']['bound']:<8}"
            f"{t['savings_us']['total']:>9.1f}  {t['action']}{extra}")
    return "\n".join(out)
