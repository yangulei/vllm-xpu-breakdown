# SPDX-License-Identifier: Apache-2.0
"""Shape-Matrix rows -> :class:`BenchCase` replay specs.

A row already carries everything the replay needs: the dispatch name, the
*swept* per-tensor dims (re-resolved at this config), the recorded per-tensor
dtypes, and — new for replay — the full ordered **argument slots** of the
original call (``_input_args``), including the non-tensor arguments and the
values the profiler recorded for them.

Building a case is therefore a substitution: walk the recorded slots in order
and replace each tensor slot's dims with the swept dims for that operand. Slot
order is preserved exactly, because :func:`~breakdown.bench.resolve.resolve`
aligns the slots positionally with the op's registered schema.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

#: Ops that are pure framework plumbing: they dispatch no device work worth
#: optimizing, and replaying them measures allocator/bookkeeping noise.
SKIP_OPS = frozenset({
    "TorchDynamo Cache Lookup",
    "aten::detach", "aten::detach_", "aten::t", "aten::transpose",
    "aten::permute", "aten::view", "aten::reshape", "aten::expand",
    "aten::slice", "aten::select", "aten::unbind", "aten::squeeze",
    "aten::unsqueeze", "aten::contiguous", "aten::item", "aten::_to_copy",
    "aten::empty", "aten::empty_like", "aten::new_empty", "aten::movedim",
    "aten::flatten", "aten::narrow", "aten::alias", "aten::as_strided",
    # allocation / dtype plumbing: the trace records these against a factory
    # overload whose recorded slots (ScalarList sizes, ScalarType enums,
    # Device, MemoryFormat) describe *how to allocate*, not a kernel worth
    # optimizing.
    "aten::to", "aten::clone", "aten::zeros", "aten::ones", "aten::arange",
    "aten::full", "aten::zeros_like", "aten::ones_like", "aten::copy_",
    "aten::pin_memory", "aten::lift_fresh",
})

#: Op-name prefixes that are compiled-region markers, not ops.
SKIP_PREFIXES = ("Torch-Compiled Region", "ProfilerStep", "Optimizer.",
                 "cudaLaunch", "Memcpy", "Memset")


@dataclass
class BenchCase:
    """One replayable invocation: an op plus fully-specified arguments."""

    op: str                                  # "aten::linear", "_C::silu_and_mul"
    args: list[dict] = field(default_factory=list)   # ordered arg slots
    device: str = "xpu"

    # provenance / sweep point
    phase: str = ""
    seq_len: Any = None
    ctx_len: Any = None
    batch_size: Any = None
    #: every ``(phase, seq_len, ctx_len, batch_size)`` this case stands for.
    #: An op whose operands do not depend on a swept dimension produces the
    #: *same* case at several sweep points; keeping only the first point would
    #: make the ranking's operating-point filter drop it entirely (the MoE
    #: grouped GEMM disappeared from the targets this way).
    points: list = field(default_factory=list)
    tp: int = 1
    module: str = ""
    module_type: str = ""
    role: str = ""
    backend: str = ""                        # trace backend classification
    layers: int = 1                          # modules dispatching this op

    # analytic cost (drives iteration budget + roofline utilization)
    flops: float = 0.0
    nbytes: float = 0.0

    #: device time the profile measured for this op at the *profiled* point.
    #: Only meaningful when the case's shapes equal the recorded ones; it is
    #: the replay's built-in ground truth (see ``traced_comparable``).
    traced_device_time_us: float = 0.0
    traced_comparable: bool = False

    case_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchCase":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def tensor_args(self) -> list[dict]:
        out = []
        for a in self.args:
            if a.get("kind") == "tensor":
                out.append(a)
            elif a.get("kind") == "tensorlist":
                out.extend(a.get("items") or [])
        return out

    @property
    def shape_label(self) -> str:
        """Human-readable operand shapes, e.g. ``[32, 6144]bf16 x [2304, 6144]``."""
        parts = []
        for t in self.tensor_args:
            dims = ",".join(str(d) for d in t.get("dims") or [])
            parts.append(f"[{dims}]{_short_dtype(t.get('dtype'))}")
        return " x ".join(parts) or "—"


_DTYPE_SHORT = {
    "bfloat16": "bf16", "float16": "f16", "float": "f32", "float32": "f32",
    "double": "f64", "long int": "i64", "long": "i64", "int": "i32",
    "int32": "i32", "int64": "i64", "char": "i8", "unsigned char": "u8",
    "bool": "b8", "float8_e4m3fn": "fp8", "float8_e5m2": "fp8",
}


def _short_dtype(dt: str | None) -> str:
    return _DTYPE_SHORT.get((dt or "").lower(), (dt or ""))


def is_skipped(op_name: str) -> bool:
    if op_name in SKIP_OPS:
        return True
    return any(op_name.startswith(p) for p in SKIP_PREFIXES)


def case_signature(op: str, args: Iterable[dict]) -> str:
    """Stable id of an (op, fully-specified arguments) pair."""
    payload = json.dumps([op, list(args)], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def shape_key(op: str, args: Iterable[dict]) -> str:
    """Id of an (op, operand shapes/dtypes) pair - stable across arg values.

    Used to join a benchmark result back to the matrix rows that produced it,
    and as the history key so a regression is attributable per shape.
    """
    tensors: list[Any] = []
    for a in args:
        if a.get("kind") == "tensor":
            tensors.append([a.get("dims"), a.get("dtype")])
        elif a.get("kind") == "tensorlist":
            tensors.append([[i.get("dims"), i.get("dtype")]
                            for i in a.get("items") or []])
    payload = json.dumps([op, tensors], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _substitute_dims(slots: list[dict], swept: list[list[int]]) -> list[dict]:
    """Recorded slots with each tensor operand's dims replaced by the sweep.

    ``swept`` is the row's resolved shape list, which is parallel to the
    *tensor* operands in slot order (``TensorList`` entries flattened) - the
    same ordering :func:`breakdown.graph_from_trace._parse_input_dims_types`
    produces, so index ``i`` of ``swept`` is the ``i``-th tensor operand.
    """
    out: list[dict] = []
    i = 0
    for slot in slots:
        kind = slot.get("kind")
        if kind == "tensor":
            new = dict(slot)
            if i < len(swept):
                new["dims"] = [int(d) for d in swept[i]]
                new.pop("strides", None)   # dims changed: strides no longer valid
            i += 1
            out.append(new)
        elif kind == "tensorlist":
            items = []
            for it in slot.get("items") or []:
                new_it = dict(it)
                if i < len(swept):
                    new_it["dims"] = [int(d) for d in swept[i]]
                    new_it.pop("strides", None)
                i += 1
                items.append(new_it)
            out.append({"kind": "tensorlist", "items": items})
        else:
            out.append(dict(slot))
    return out


def _slots_from_shapes(shapes: list[list[int]],
                       dtypes: list[str]) -> list[dict]:
    """Fallback slots for an op with no recorded arg structure.

    Synthetic kernel ops (``triton::``/``flash_xpu::``/``flashinfer::``) have no
    ``cpu_op``, so their operands were *reconstructed* rather than recorded.
    They get positional tensor slots; a recipe supplies the rest.
    """
    return [{"kind": "tensor", "dims": [int(d) for d in s],
             "dtype": (dtypes[i] if i < len(dtypes) else "")}
            for i, s in enumerate(shapes)]


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f and abs(f) != float("inf") else 0.0


def build_cases(rows: list[dict[str, Any]], device: str = "xpu",
                dedup: bool = True) -> tuple[list[BenchCase], dict[str, Any]]:
    """Rows -> de-duplicated replay cases plus a coverage summary.

    Rows differing only in a sweep dimension that no operand depends on collapse
    into one case (dedup by ``(op, args)``); the first row's provenance is kept
    and ``layers`` becomes the max seen, so ranking still weighs the op by how
    many modules dispatch it.
    """
    cases: dict[str, BenchCase] = {}
    skipped: dict[str, int] = {}
    no_shape: dict[str, int] = {}
    for row in rows:
        op = row.get("Op Name") or ""
        if not op:
            continue
        if is_skipped(op):
            skipped[op] = skipped.get(op, 0) + 1
            continue
        swept = row.get("_resolved_shapes") or []
        slots = row.get("_input_args") or []
        if slots:
            args = _substitute_dims(slots, swept)
        elif swept:
            args = _slots_from_shapes(swept, row.get("_input_dtypes") or [])
        else:
            no_shape[op] = no_shape.get(op, 0) + 1
            continue
        recorded = row.get("_recorded_shapes") or []
        comparable = bool(recorded) and [list(map(int, s)) for s in recorded] == \
            [list(map(int, s)) for s in swept]
        case = BenchCase(
            op=op, args=args, device=device,
            phase=row.get("Phase", ""), seq_len=row.get("Seq Len"),
            ctx_len=row.get("Ctx Len"), batch_size=row.get("Batch Size"),
            tp=_int(row.get("TP"), 1), module=row.get("Module", ""),
            module_type=row.get("_module_type", ""), role=row.get("_op_role", ""),
            backend=row.get("Backend", ""), layers=_int(row.get("Layers"), 1),
            flops=_float(row.get("FLOPs")), nbytes=_float(row.get("Memory (bytes)")),
            traced_device_time_us=_float(row.get("_device_time_us")),
            traced_comparable=comparable,
        )
        case.case_id = case_signature(case.op, case.args)
        case.points = [[case.phase, case.seq_len, case.ctx_len,
                        case.batch_size]]
        if not dedup:
            cases[f"{case.case_id}:{len(cases)}"] = case
            continue
        prev = cases.get(case.case_id)
        if prev is None:
            cases[case.case_id] = case
        else:
            prev.layers = max(prev.layers, case.layers)
            if case.points[0] not in prev.points:
                prev.points.append(case.points[0])
            if case.traced_comparable and not prev.traced_comparable:
                prev.traced_device_time_us = case.traced_device_time_us
                prev.traced_comparable = True
    out = list(cases.values())
    coverage = {
        "total_rows": len(rows),
        "cases": len(out),
        "ops": len({c.op for c in out}),
        "skipped_framework_ops": skipped,
        "rows_without_shapes": no_shape,
    }
    return out, coverage


def group_by_op(cases: list[BenchCase]) -> dict[str, list[BenchCase]]:
    """Cases keyed by op - the unit of process isolation in the runner."""
    out: dict[str, list[BenchCase]] = {}
    for c in cases:
        out.setdefault(c.op, []).append(c)
    return out
