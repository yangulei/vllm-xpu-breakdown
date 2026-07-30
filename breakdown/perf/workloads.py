# SPDX-License-Identifier: Apache-2.0
"""Shape-Matrix rows -> micro_perf workload JSON + a coverage report.

Every *dispatched* op must map to the exact kernel that runs it; anything that
doesn't is reported as unmapped (and treated as an error by the runner) rather
than approximated, because an approximated op silently corrupts the ranking
downstream.

Workloads are grouped so they can be launched separately:

``compute``     single-device compute ops
``collective``  all_reduce / all_gather (need every rank)
``msa``         MiniMax-M3 sparse attention (its own providers)
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from breakdown.perf.matrix_reader import OpRow, unique_rows
from breakdown.perf.op_map import ModelConfig, get_dispatch, common

GROUPS = ("compute", "collective", "msa")


@dataclass
class RowFilter:
    """Optional row filters; ``None`` means "keep the whole sweep"."""

    tp: set[int] | None = None
    phases: set[str] | None = None
    prefill_seq_lens: set[int] | None = None
    prefill_ctx_lens: set[int] | None = None
    prefill_batch_sizes: set[int] | None = None
    decode_ctx_lens: set[int] | None = None
    decode_batch_sizes: set[int] | None = None

    #: A screening tier: a few representative shapes per op is enough to rank
    #: optimization targets, and takes minutes instead of an hour.
    @classmethod
    def smoke(cls, tp: set[int] | None = None) -> "RowFilter":
        return cls(tp=tp, prefill_seq_lens={128, 2048, 8192},
                   decode_batch_sizes={1, 32, 128})

    def keeps(self, row: OpRow) -> bool:
        if self.tp is not None and _asint(row.tp) not in self.tp:
            return False
        if self.phases and row.phase not in self.phases:
            return False
        if row.phase == "prefill":
            pairs = ((self.prefill_seq_lens, row.seq_len),
                     (self.prefill_ctx_lens, row.ctx_len),
                     (self.prefill_batch_sizes, row.batch_size))
        else:
            pairs = ((self.decode_ctx_lens, row.ctx_len),
                     (self.decode_batch_sizes, row.batch_size))
        return all(want is None or _asint(got) in want for want, got in pairs)


@dataclass
class Coverage:
    """What the emission mapped, skipped and failed to map."""

    dispatch: str
    total_rows: int = 0
    unique_op_shapes: int = 0
    dense_sweep: bool = True
    cases: dict[str, dict[str, int]] = field(default_factory=dict)
    mapped_op_case_counts: dict[str, int] = field(default_factory=dict)
    unmapped_breakdown_ops: dict[str, int] = field(default_factory=dict)
    skipped_framework_ops: dict[str, int] = field(default_factory=dict)
    invalid_cases: dict[str, dict[str, int]] = field(default_factory=dict)
    mapping_table: list[dict[str, Any]] = field(default_factory=list)
    #: op -> per-case ``{"flops": .., "bytes": ..}``; drives the runtime
    #: estimate that sets each op's benchmark timeout. Written to its own
    #: ``cost.json`` rather than into the coverage report.
    case_costs: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unmapped_breakdown_ops

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch": self.dispatch,
            "dense_sweep": self.dense_sweep,
            "total_rows": self.total_rows,
            "unique_op_shapes": self.unique_op_shapes,
            "mapped_micro_perf_cases": self.cases,
            "mapped_op_case_counts": self.mapped_op_case_counts,
            "unmapped_breakdown_ops": self.unmapped_breakdown_ops,
            "skipped_framework_ops": self.skipped_framework_ops,
            "invalid_cases": self.invalid_cases,
            "mapping_table": self.mapping_table,
        }


def _asint(v: Any) -> Any:
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def _num(v: Any) -> float:
    """A finite non-negative float, or 0.0 (matrix cells may be NaN/blank)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f and f not in (float("inf"), float("-inf")) and f > 0 else 0.0


def _case_key(op: str, args: dict) -> tuple:
    return (op, tuple(sorted((k, str(v)) for k, v in args.items())))


def emit(rows: list[OpRow], cfg: ModelConfig, dispatch: str = "xpu",
         row_filter: RowFilter | None = None,
         dense_sweep: bool = True) -> tuple[dict[str, dict[str, list[dict]]],
                                            Coverage]:
    """Map rows to micro_perf cases.

    Returns ``(buckets, coverage)`` where ``buckets[group][op]`` is the list of
    de-duplicated case-argument dicts.
    """
    mod = get_dispatch(dispatch)
    dense = set(getattr(mod, "DENSE_SWEEP_OPS", set())) if dense_sweep else set()
    constraints = getattr(mod, "CASE_CONSTRAINTS", {})

    kept = [r for r in rows if row_filter is None or row_filter.keeps(r)]
    uniq = unique_rows(kept, dense)

    buckets: dict[str, dict[str, list[dict]]] = {g: defaultdict(list)
                                                 for g in GROUPS}
    costs: dict[str, list[dict[str, float]]] = defaultdict(list)
    cov = Coverage(dispatch=dispatch, total_rows=len(kept),
                   unique_op_shapes=len(uniq), dense_sweep=dense_sweep)
    seen_cases: set[tuple] = set()
    mapped_ops: Counter = Counter()
    unmapped: Counter = Counter()
    skipped: Counter = Counter()
    invalid: dict[str, Counter] = defaultdict(Counter)
    mapping_rows: list[dict[str, Any]] = []

    for r in uniq:
        if r.op_name in mod.SKIP_OPS:
            skipped[r.op_name] += 1
            continue
        adapter = mod.ADAPTERS.get(r.op_name)
        if adapter is None:
            unmapped[r.op_name] += 1
            continue
        cases = adapter(r, cfg)
        if not cases:
            unmapped[r.op_name] += 1
            continue
        for ec in cases:
            # A case the kernel would reject is dropped here: micro_perf runs
            # one process per op, so a single TORCH_CHECK abort loses every
            # other shape of that op.
            reason = common.check_case(ec.op, ec.args, constraints)
            if reason:
                invalid[ec.op][reason] += 1
                continue
            mapped_ops[ec.op] += 1
            mapping_rows.append({
                "breakdown_op": r.op_name, "backend": r.backend,
                "phase": r.phase, "module": r.module_attr,
                "micro_perf_op": ec.op, "note": ec.note,
            })
            key = _case_key(ec.op, ec.args)
            if key in seen_cases:
                continue
            seen_cases.add(key)
            group = ("collective" if ec.op in mod.COLLECTIVE_OPS else
                     "msa" if ec.op in mod.MSA_OPS else "compute")
            buckets[group][ec.op].append(ec.args)
            costs[ec.op].append({
                "flops": _num(ec.flops if ec.flops is not None else r.flops),
                "bytes": _num(ec.nbytes if ec.nbytes is not None
                              else r.memory_bytes),
            })

    seen_map: set[tuple] = set()
    for m in mapping_rows:
        k = (m["breakdown_op"], m["phase"], m["micro_perf_op"], m["module"])
        if k not in seen_map:
            seen_map.add(k)
            cov.mapping_table.append(m)
    cov.cases = {g: {k: len(v) for k, v in buckets[g].items()} for g in GROUPS}
    cov.mapped_op_case_counts = dict(mapped_ops)
    cov.unmapped_breakdown_ops = dict(unmapped)
    cov.skipped_framework_ops = dict(skipped)
    cov.invalid_cases = {op: dict(c) for op, c in invalid.items()}
    cov.case_costs = dict(costs)
    return {g: dict(buckets[g]) for g in GROUPS}, cov


def write(buckets: dict[str, dict[str, list[dict]]], cov: Coverage,
          out_dir: str) -> dict[str, str]:
    """Write ``workloads/<group>/<group>_ops.json`` + ``coverage.json``."""
    paths: dict[str, str] = {}
    for group in GROUPS:
        d = os.path.join(out_dir, group)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{group}_ops.json")
        with open(path, "w") as fh:
            json.dump(buckets.get(group) or {}, fh, indent=2)
        paths[group] = path
    cov_path = os.path.join(out_dir, "coverage.json")
    with open(cov_path, "w") as fh:
        json.dump(cov.to_dict(), fh, indent=2)
    paths["coverage"] = cov_path
    # cost.json sits *outside* the group dirs on purpose: micro_perf's launcher
    # parses every json under --task_dir and would treat it as a workload.
    cost_path = os.path.join(out_dir, "cost.json")
    with open(cost_path, "w") as fh:
        json.dump(cov.case_costs, fh, indent=2)
    paths["cost"] = cost_path
    return paths


def read_costs(workloads_dir: str) -> dict[str, list[dict[str, float]]]:
    """Per-case cost written by :func:`write` (empty if the run predates it)."""
    path = os.path.join(workloads_dir, "cost.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def case_counts(workloads_dir: str,
                groups: Iterable[str] = GROUPS) -> dict[str, int]:
    """op -> number of emitted cases, read back from the workload JSONs."""
    out: dict[str, int] = {}
    for group in groups:
        path = os.path.join(workloads_dir, group, f"{group}_ops.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        for op, cases in data.items():
            out[op] = len(cases)
    return out


def ops_in(path: str) -> list[str]:
    """Op names inside a workload group directory or file."""
    files = []
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                 if f.endswith(".json")]
    elif os.path.isfile(path):
        files = [path]
    ops: list[str] = []
    for f in files:
        with open(f) as fh:
            ops += list(json.load(fh))
    return list(dict.fromkeys(ops))
