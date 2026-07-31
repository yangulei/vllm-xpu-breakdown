# SPDX-License-Identifier: Apache-2.0
"""Materialize the arguments of a replayed op.

The recorded slots say *what shape and dtype* each operand had; the op's
registered schema says *what each argument means*. Combining the two is what
makes replay generic:

* a ``Tensor`` argument is allocated at the swept dims and recorded dtype;
* a ``Tensor?`` whose slot the profiler recorded as empty becomes ``None``;
* a scalar argument takes the value the profiler recorded in ``Concrete
  Inputs``; failing that the schema default; failing that a synthesizer.

**Integer tensors are never filled randomly.** Index-style operands (paged-KV
``slot_mapping``, MoE ``topk_ids`` / ``rows_per_expert`` /
``unpermuted_row_to_permuted_row``, ``positions``, block tables) are what make a
kernel read the memory it is supposed to read; garbage there either aborts the
kernel or measures a degenerate access pattern. Every integer tensor therefore
needs a **synthesizer** - one registered for its ``(op, argument name)``, one
registered for the argument name alone, or an explicit generic rule. When none
applies, :class:`MissingSynthesizer` is raised and the case is reported, never
silently guessed.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from breakdown.bench.resolve import Resolved

#: trace dtype token -> torch dtype attribute name
DTYPE_MAP: dict[str, str] = {
    "bfloat16": "bfloat16", "float16": "float16", "half": "float16",
    "float": "float32", "float32": "float32", "double": "float64",
    "float64": "float64", "long int": "int64", "long": "int64",
    "int64": "int64", "int": "int32", "int32": "int32", "short": "int16",
    "int16": "int16", "char": "int8", "int8": "int8",
    "unsigned char": "uint8", "uint8": "uint8", "byte": "uint8",
    "bool": "bool", "c10::bfloat16": "bfloat16",
    "float8_e4m3fn": "float8_e4m3fn", "float8_e5m2": "float8_e5m2",
    "float8_e4m3fnuz": "float8_e4m3fnuz",
}

_INT_DTYPES = {"int64", "int32", "int16", "int8", "uint8"}

_OPTIONAL = re.compile(r"\?$")
_LIST = re.compile(r"\[\d*\]$")


class MissingSynthesizer(RuntimeError):
    """An integer/index operand has no registered way to be filled."""


class ArgBuildError(RuntimeError):
    """The recorded slots could not be aligned with the op's schema."""


@dataclass
class Ctx:
    """What a synthesizer may look at to produce a valid operand."""

    op: str
    arg_name: str
    dims: list[int]
    dtype: str
    device: str
    resolved: Resolved
    #: every slot of the call, so a synthesizer can size itself against a
    #: sibling operand (rows of the hidden state, expert count of the weights…)
    slots: list[dict] = field(default_factory=list)
    #: already-materialized positional arguments (earlier slots only)
    built: list[Any] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.dims:
            n *= max(int(d), 1)
        return n

    def tensor_dims(self, pred: Callable[[dict], bool] | None = None
                    ) -> list[list[int]]:
        out = []
        for s in self.slots:
            if s.get("kind") == "tensor" and (pred is None or pred(s)):
                out.append([int(d) for d in s.get("dims") or []])
            elif s.get("kind") == "tensorlist":
                for it in s.get("items") or []:
                    out.append([int(d) for d in it.get("dims") or []])
        return out

    def max_dim(self, default: int = 1) -> int:
        vals = [d for dims in self.tensor_dims() for d in dims]
        return max(vals) if vals else default


Synth = Callable[[Ctx], Any]

#: ``(op, arg_name)`` -> synthesizer. Consulted before the name-only registry.
SYNTHESIZERS: dict[tuple[str, str], Synth] = {}

#: ``arg_name`` -> synthesizer, shared across ops that use the same convention
#: (``positions``, ``slot_mapping``, ``topk_ids``, …).
NAME_SYNTHESIZERS: dict[str, Synth] = {}


def synthesizer(*names: str, op: str | None = None) -> Callable[[Synth], Synth]:
    """Register a synthesizer for one or more argument names."""

    def deco(fn: Synth) -> Synth:
        for n in names:
            if op:
                SYNTHESIZERS[(op, n)] = fn
            else:
                NAME_SYNTHESIZERS[n] = fn
        return fn

    return deco


# ---------------------------------------------------------------------------
# generic index-operand synthesizers
# ---------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _dtype_of(ctx: Ctx):
    torch = _torch()
    return getattr(torch, DTYPE_MAP.get((ctx.dtype or "").lower(), "int64"))


def _arange(ctx: Ctx, high: int | None = None):
    """``0..n-1`` reshaped to the operand - a dense, in-range index map."""
    torch = _torch()
    n = ctx.numel
    limit = high if high and high > 0 else n
    t = torch.arange(n, dtype=_dtype_of(ctx), device=ctx.device) % max(limit, 1)
    return t.reshape(ctx.dims) if ctx.dims else t.reshape(())


@synthesizer("positions", "position_ids", "input_positions")
def _positions(ctx: Ctx):
    """Token positions: contiguous, so RoPE reads the whole cos/sin cache."""
    torch = _torch()
    cache = max((d[0] for d in ctx.tensor_dims() if len(d) == 2 and d[0] > 1024),
                default=0)
    n = ctx.numel
    start = 0
    if cache:
        start = max(0, min(cache - n, cache // 2))
    t = torch.arange(start, start + n, dtype=_dtype_of(ctx), device=ctx.device)
    return t.reshape(ctx.dims) if ctx.dims else t.reshape(())


@synthesizer("slot_mapping", "index_slot_mapping", "kv_slot_mapping")
def _slot_mapping(ctx: Ctx):
    """Distinct paged-KV slots: every token writes its own cache line."""
    return _arange(ctx)


@synthesizer("block_tables", "block_table", "kv_block_table")
def _block_tables(ctx: Ctx):
    return _arange(ctx)


@synthesizer("topk_ids", "topk_indices", "expert_ids", "selected_experts")
def _topk_ids(ctx: Ctx):
    """Routed expert ids: spread across all experts so every expert is hit."""
    torch = _torch()
    experts = _expert_count(ctx)
    n = ctx.numel
    t = torch.arange(n, dtype=_dtype_of(ctx), device=ctx.device) % max(experts, 1)
    return t.reshape(ctx.dims) if ctx.dims else t.reshape(())


@synthesizer("token_expert_indices", "unpermuted_row_to_permuted_row",
             "permuted_row_to_unpermuted_row", "sorted_token_ids",
             "expert_map", "expert_offsets", "sort_indices")
def _row_map(ctx: Ctx):
    return _arange(ctx)


@synthesizer("rows_per_expert", "num_tokens_per_expert", "expert_num_tokens",
             "tokens_per_expert", "group_sizes")
def _rows_per_expert(ctx: Ctx):
    """A balanced routing: rows split evenly over the experts, summing exactly.

    An unbalanced or non-summing partition makes a grouped GEMM either skip
    work or run off the end of its input, so this must be exact.
    """
    torch = _torch()
    groups = max(ctx.numel, 1)
    rows = _routed_rows(ctx, groups)
    base, rem = divmod(rows, groups)
    vals = [base + (1 if i < rem else 0) for i in range(groups)]
    t = torch.tensor(vals, dtype=_dtype_of(ctx), device=ctx.device)
    return t.reshape(ctx.dims) if ctx.dims else t


@synthesizer("seq_lens", "context_lens", "kv_lens", "seq_lens_tensor",
             "query_lens", "num_computed_tokens")
def _seq_lens(ctx: Ctx):
    torch = _torch()
    ctx_len = int(ctx.values.get("ctx_len") or 0) or 1
    t = torch.full((ctx.numel,), ctx_len, dtype=_dtype_of(ctx),
                   device=ctx.device)
    return t.reshape(ctx.dims) if ctx.dims else t


@synthesizer("cu_seqlens", "cu_seqlens_q", "cu_seqlens_k", "query_start_loc",
             "seq_start_loc")
def _cu_seqlens(ctx: Ctx):
    """A cumulative-length vector: strictly increasing, starting at 0."""
    torch = _torch()
    n = max(ctx.numel, 1)
    step = max(int(ctx.values.get("seq_len") or 1), 1)
    t = torch.arange(n, dtype=_dtype_of(ctx), device=ctx.device) * step
    return t.reshape(ctx.dims) if ctx.dims else t


def _expert_count(ctx: Ctx) -> int:
    """Expert count inferred from a sibling operand or the recorded value."""
    for key in ("num_experts", "total_experts_num", "local_experts_num",
                "n_experts"):
        v = ctx.values.get(key)
        if isinstance(v, int) and v > 0:
            return v
    # a 3-D expert weight [E, K, N], or a gating output [tokens, E]
    for dims in ctx.tensor_dims():
        if len(dims) == 3 and dims[0] > 1:
            return dims[0]
    for dims in ctx.tensor_dims():
        if len(dims) == 2 and dims[1] > 8:
            return dims[1]
    return max(ctx.numel, 1)


def _routed_rows(ctx: Ctx, groups: int) -> int:
    """Total rows a grouped GEMM will consume, from the widest 2-D operand."""
    best = 0
    for dims in ctx.tensor_dims():
        if len(dims) == 2:
            best = max(best, dims[0])
    return best or groups


# ---------------------------------------------------------------------------
# scalar parsing
# ---------------------------------------------------------------------------
def parse_scalar(text: str) -> Any:
    """The value the profiler recorded for a non-tensor argument.

    ``Concrete Inputs`` are strings: ``"7."``, ``"False"``, ``"[32, 2048]"``.
    Returns ``None`` when nothing was recorded.
    """
    s = (text or "").strip()
    if not s:
        return None
    if s in ("True", "False"):
        return s == "True"
    if s in ("None", "null"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _is_optional(t: str) -> bool:
    return t.endswith("?") or t.startswith("Optional")


def _base_type(t: str) -> str:
    t = t.strip()
    t = _OPTIONAL.sub("", t)
    if t.startswith("Optional[") and t.endswith("]"):
        t = t[len("Optional["):-1]
    return t


def _coerce(value: Any, type_str: str) -> Any:
    base = _base_type(type_str)
    if value is None:
        return None
    if base in ("int", "SymInt"):
        return int(value)
    if base == "float":
        return float(value)
    if base == "bool":
        return bool(value)
    if base == "str":
        return str(value)
    if base == "Scalar":
        return value
    if base in ("int[]", "SymInt[]", "float[]", "bool[]") or _LIST.search(base):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    return value


# ---------------------------------------------------------------------------
# tensor materialization
# ---------------------------------------------------------------------------
def torch_dtype(name: str):
    torch = _torch()
    attr = DTYPE_MAP.get((name or "").lower())
    if attr is None:
        return torch.bfloat16
    return getattr(torch, attr, torch.bfloat16)


def make_tensor(dims: list[int], dtype_name: str, device: str,
                ctx: Ctx | None = None) -> Any:
    """Allocate one operand.

    Floating tensors get small normal values (large magnitudes make fp16/bf16
    kernels take denormal/inf paths that are not representative) generated
    **directly in the target dtype**: an ``lm_head`` weight is hundreds of
    megabytes, and materializing it in fp32 first would double the allocation
    and dominate the case's wall time. Integer tensors never land here without
    a synthesizer - see :func:`build_args`.
    """
    torch = _torch()
    dt = torch_dtype(dtype_name)
    shape = [int(d) for d in dims]
    if dt.is_floating_point:
        try:
            return torch.randn(shape, device=device, dtype=dt) * 0.1
        except RuntimeError:
            # a few dtypes (fp8 variants) have no native normal_ kernel
            return (torch.randn(shape, device=device, dtype=torch.float32)
                    * 0.1).to(dt)
    if dt == torch.bool:
        return torch.zeros(shape, device=device, dtype=dt)
    if str(dt).startswith("torch.float8"):
        t = torch.randn(shape, device=device, dtype=torch.float32) * 0.1
        return t.to(dt)
    if ctx is not None:
        raise MissingSynthesizer(
            f"{ctx.op}: integer operand '{ctx.arg_name}' {shape} ({dtype_name}) "
            f"has no registered synthesizer - register one in "
            f"breakdown.bench.inputs.SYNTHESIZERS so the kernel reads a valid "
            f"index map")
    return torch.zeros(shape, device=device, dtype=dt)


def _synth_for(op: str, name: str) -> Synth | None:
    return SYNTHESIZERS.get((op, name)) or NAME_SYNTHESIZERS.get(name)


def _tensor_arg(slot: dict, name: str, ctx_values: dict, case_op: str,
                device: str, resolved: Resolved, slots: list[dict],
                built: list[Any], zeroed: bool = False) -> Any:
    dims = [int(d) for d in slot.get("dims") or []]
    dtype = slot.get("dtype") or ""
    if zeroed:
        import torch
        return torch.zeros([int(d) for d in dims], device=device,
                           dtype=torch_dtype(dtype))
    ctx = Ctx(op=case_op, arg_name=name, dims=dims, dtype=dtype, device=device,
              resolved=resolved, slots=slots, built=built, values=ctx_values)
    attr = DTYPE_MAP.get(dtype.lower(), "")
    if attr in _INT_DTYPES:
        fn = _synth_for(case_op, name)
        if fn is None:
            raise MissingSynthesizer(
                f"{case_op}: integer operand '{name}' {dims} ({dtype}) has no "
                f"registered synthesizer - a random index map would abort the "
                f"kernel or measure a degenerate access pattern. Register one "
                f"in breakdown.bench.inputs.SYNTHESIZERS.")
        return fn(ctx)
    fn = _synth_for(case_op, name)
    if fn is not None:
        return fn(ctx)
    return make_tensor(dims, dtype, device)


@dataclass
class Call:
    """A fully-materialized invocation: ``fn(*args, **kwargs)``."""

    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: the operands the op writes into, so timing can restore them between
    #: measurement windows
    mutated: list[Any] = field(default_factory=list)


def build_args(case, resolved: Resolved, device: str,
               extra_values: dict | None = None,
               output_names: Iterable[str] = ()) -> Call:
    """Build one replayed call.

    Recorded slots align positionally with the schema's arguments (the profiler
    records one slot per dispatched argument). When they do not - a synthetic
    kernel op whose operands were reconstructed rather than recorded - tensor
    slots are assigned to the schema's tensor arguments in order and the rest
    fall back to schema defaults. Keyword-only schema arguments are passed as
    keywords, never positionally.

    ``output_names`` are arguments the op *writes* but whose schema carries no
    alias annotation (a counter accumulated with atomics, a scatter map). They
    are allocated zeroed - never index-synthesized, since the kernel fills them
    - and registered as mutated so they are reset between windows.
    """
    slots = list(case.args)
    outputs = set(output_names or ())
    if resolved.kind != "torch_op" or not resolved.arg_names:
        return _build_positional(case, slots, device, resolved, extra_values)

    names, types = resolved.arg_names, resolved.arg_types
    values = _context_values(case, slots, names, types, extra_values)
    aligned = _align(slots, names, types, case.op)
    kwonly = set(resolved.kwarg_only)
    mutates = set(resolved.mutates)

    call = Call()
    built: list[Any] = []
    for i, name in enumerate(names):
        t = types[i]
        slot = aligned[i]
        base = _base_type(t)
        if base in ("Tensor", "Tensor[]", "List[Tensor]"):
            if slot is None or slot.get("kind") in ("none", None):
                if _is_optional(t) or resolved.defaults[i] is not None:
                    value = None if base == "Tensor" else []
                    _place(call, built, i, name, value, kwonly)
                    continue
                raise ArgBuildError(
                    f"{case.op}: required tensor argument '{name}' was not "
                    f"recorded in the trace")
            if slot.get("kind") == "tensorlist" or base != "Tensor":
                items = slot.get("items") or ([slot] if slot.get("kind") ==
                                              "tensor" else [])
                value = [_tensor_arg(it, name, values, case.op, device,
                                     resolved, slots, built,
                                     zeroed=name in outputs) for it in items]
            else:
                value = _tensor_arg(slot, name, values, case.op, device,
                                    resolved, slots, built,
                                    zeroed=name in outputs)
            if i in mutates or name in outputs:
                call.mutated.extend(value if isinstance(value, list) else [value])
            _place(call, built, i, name, value, kwonly)
            continue
        # non-tensor argument
        recorded = parse_scalar(slot.get("value")) if slot else None
        if recorded is None:
            recorded = resolved.defaults[i]
        if recorded is None:
            # Only an op-specific synthesizer may fill a *scalar*: the
            # name-keyed registry is tensor-oriented (``other`` is an
            # elementwise tensor operand for ``add`` but a number for
            # ``mul_.Scalar``), so consulting it here would hand the op a
            # tensor where its schema wants a number.
            fn = SYNTHESIZERS.get((case.op, name))
            if fn is not None:
                recorded = fn(Ctx(op=case.op, arg_name=name, dims=[], dtype="",
                                  device=device, resolved=resolved, slots=slots,
                                  built=built, values=values))
        if recorded is None:
            recorded = _infer_scalar(name, base, values)
        if recorded is None and not _is_optional(t):
            raise ArgBuildError(
                f"{case.op}: argument '{name}' ({t}) has no recorded value, no "
                f"schema default and no synthesizer")
        _place(call, built, i, name, _coerce(recorded, t), kwonly)
    return call


def _place(call: Call, built: list[Any], index: int, name: str, value: Any,
           kwonly: set[int]) -> None:
    built.append(value)
    if index in kwonly:
        call.kwargs[name] = value
    else:
        call.args.append(value)


def _build_positional(case, slots: list[dict], device: str,
                      resolved: Resolved, extra_values: dict | None) -> Call:
    """No schema (a Python-API kernel): materialize the slots in order."""
    values = dict(extra_values or {})
    call = Call()
    built: list[Any] = []
    for i, slot in enumerate(slots):
        kind = slot.get("kind")
        name = f"arg{i}"
        if kind == "tensor":
            value = _tensor_arg(slot, name, values, case.op, device,
                                resolved, slots, built)
        elif kind == "tensorlist":
            value = [_tensor_arg(it, name, values, case.op, device,
                                 resolved, slots, built)
                     for it in slot.get("items") or []]
        elif kind == "scalar":
            value = parse_scalar(slot.get("value"))
        else:
            value = None
        built.append(value)
        call.args.append(value)
    return call


def _align(slots: list[dict], names: list[str], types: list[str],
           op: str) -> list[dict | None]:
    """Recorded slots -> one slot per schema argument.

    The common case is a 1:1 positional match. When the trace recorded fewer
    slots (reconstructed operands, or a call that relied on defaults), tensor
    slots are assigned to the schema's tensor arguments in order so the shapes
    still land on the right operands.
    """
    if len(slots) == len(names):
        return list(slots)
    out: list[dict | None] = [None] * len(names)
    tensors = [s for s in slots if s.get("kind") in ("tensor", "tensorlist")]
    scalars = [s for s in slots if s.get("kind") == "scalar"]
    ti = si = 0
    for i, t in enumerate(types):
        base = _base_type(t)
        if base in ("Tensor", "Tensor[]"):
            if ti < len(tensors):
                out[i] = tensors[ti]
                ti += 1
        elif si < len(scalars):
            out[i] = scalars[si]
            si += 1
    if ti < len(tensors):
        raise ArgBuildError(
            f"{op}: {len(tensors)} recorded tensor operands do not fit the "
            f"schema's {sum(1 for t in types if _base_type(t).startswith('Tensor'))}"
            f" tensor arguments")
    return out


def _context_values(case, slots: list[dict], names: list[str],
                    types: list[str], extra: dict | None) -> dict[str, Any]:
    """Scalar arguments the call recorded, by name, plus the sweep point.

    Synthesizers need these: ``num_experts`` bounds ``topk_ids``, ``block_size``
    bounds a block table, the sweep's ``ctx_len`` sizes ``seq_lens``.
    """
    values: dict[str, Any] = {
        "phase": case.phase, "seq_len": case.seq_len, "ctx_len": case.ctx_len,
        "batch_size": case.batch_size, "tp": case.tp,
    }
    values.update(extra or {})
    for i, name in enumerate(names):
        if i >= len(slots):
            break
        slot = slots[i]
        if slot.get("kind") == "scalar":
            v = parse_scalar(slot.get("value"))
            if v is not None:
                values[name] = v
    return values


def _infer_scalar(name: str, base: str, values: dict[str, Any]) -> Any:
    """Last resort for a scalar the trace did not record.

    Only unambiguous cases: a scale/eps/alpha with a conventional value, and
    booleans that default to ``False``. Anything else stays ``None`` so the
    caller raises rather than inventing a number that changes the work done.
    """
    low = name.lower()
    if base == "bool":
        return False
    if base == "float":
        if "eps" in low:
            return 1e-6
        if "scale" in low:
            return 1.0
        if low in ("alpha", "beta"):
            return 1.0 if low == "alpha" else 0.0
    if base in ("Scalar", "number"):
        # An elementwise scalar operand (``mul_``'s multiplier) that the
        # profiler did not record. Its *value* does not change the work the
        # kernel does - only the numbers - and 1 is safe for a divisor.
        return 1.0
    if base in ("int", "SymInt"):
        if "block_size" in low:
            return 16
        if low in ("dim", "axis"):
            return -1
    return None
