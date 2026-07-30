# SPDX-License-Identifier: Apache-2.0
"""Normalized op records (:class:`OpRow`) from Shape Matrix rows or an .xlsx.

In-process the perf pipeline gets its rows straight from
:func:`breakdown.perf.shape_matrix.build_rows` (:func:`rows_to_oprows`); the
``.xlsx`` reader (:func:`read_matrix`) exists for matrices carried in from
another machine (e.g. the CUDA reference box).

The Shape Matrix (sheet 0) has one row per
(Phase, Seq Len, Ctx Len, Batch Size, TP, Module, Op Name) combination. Each row
carries a per-tensor concrete Shape (e.g. ``[128, 6144, bf16] x [2304, 6144, bf16]``)
and a Symbolic Shape (``[S, 6144] x [9216/TP, 6144]``).

This module parses those into structured :class:`OpRow` records and provides a
de-duplication keyed on (op_name, module_leaf, phase, concrete_shape+dtypes) so a
converter emits one benchmark case per genuinely distinct shape instead of once
per sweep point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Separator between per-tensor blocks in the Shape / Symbolic Shape columns.
_TENSOR_SEP = re.compile(r"\s*[x×]\s*")

# dtype tokens that may appear as the trailing element of a concrete tensor block.
_DTYPE_TOKENS = {
    "bf16", "fp16", "fp32", "f32", "f16", "bfloat16", "float16", "float32",
    "fp8", "fp8_e4m3", "fp8_e5m2", "int8", "uint8", "i8", "u8",
    "int32", "i32", "int64", "i64", "long int", "long", "bool", "int4",
}


@dataclass
class TensorShape:
    """One tensor operand: concrete dims + dtype, and its symbolic form."""

    dims: list[int]
    dtype: str
    symbolic: str  # e.g. "[S, 9216/TP, 6144]"

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"T({self.dims}, {self.dtype})"


@dataclass
class OpRow:
    phase: str
    seq_len: Any
    ctx_len: Any
    batch_size: Any
    tp: int
    module: str
    op_name: str
    backend: str
    layers: Any
    tensors: list[TensorShape]
    symbolic_raw: str
    shape_raw: str
    memory_bytes: Any = None
    flops: Any = None
    ai: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def module_leaf(self) -> str:
        """Last path element of Module, e.g. ``QKVParallelLinear.qkv_proj``."""
        if not self.module:
            return ""
        return self.module.split("/")[-1]

    @property
    def module_attr(self) -> str:
        """Attribute name after the class, e.g. ``qkv_proj`` from
        ``QKVParallelLinear.qkv_proj``. Falls back to the leaf itself."""
        leaf = self.module_leaf
        return leaf.split(".", 1)[1] if "." in leaf else leaf


def _parse_tensor_block(block: str, symbolic_block: str) -> TensorShape | None:
    """Parse ``[128, 6144, bf16]`` (concrete) aligned with ``[S, 6144]`` (symbolic)."""
    inner = block.strip().strip("[]").strip()
    if inner == "":
        return None
    parts = [p.strip() for p in inner.split(",")]
    dtype = "bf16"
    # trailing dtype token (may be a two-word token like "long int")
    if parts and (parts[-1].lower() in _DTYPE_TOKENS or "int" in parts[-1].lower()
                  or "float" in parts[-1].lower()):
        dtype = parts[-1]
        parts = parts[:-1]
    dims: list[int] = []
    for p in parts:
        try:
            dims.append(int(p))
        except ValueError:
            # non-integer concrete dim (shouldn't happen in the concrete column)
            dims.append(p)  # type: ignore[arg-type]
    return TensorShape(dims=dims, dtype=_norm_dtype(dtype),
                       symbolic=symbolic_block.strip())


def _norm_dtype(d: str) -> str:
    d = d.strip().lower()
    return {
        "bf16": "bfloat16", "bfloat16": "bfloat16",
        "fp16": "float16", "f16": "float16", "float16": "float16",
        "fp32": "float32", "f32": "float32", "float32": "float32",
        "i32": "int32", "int32": "int32",
        "i64": "int64", "int64": "int64", "long int": "int64", "long": "int64",
        "i8": "int8", "int8": "int8", "u8": "uint8", "uint8": "uint8",
        "fp8": "float8_e4m3", "fp8_e4m3": "float8_e4m3", "fp8_e5m2": "float8_e5m2",
        "bool": "bool", "int4": "int4",
    }.get(d, d)


def _parse_shapes(shape_raw: str, symbolic_raw: str) -> list[TensorShape]:
    if not isinstance(shape_raw, str) or shape_raw.strip() == "":
        return []
    conc_blocks = _TENSOR_SEP.split(shape_raw.strip())
    sym_blocks = _TENSOR_SEP.split(symbolic_raw.strip()) if isinstance(
        symbolic_raw, str) else []
    tensors: list[TensorShape] = []
    for i, blk in enumerate(conc_blocks):
        sym = sym_blocks[i] if i < len(sym_blocks) else ""
        t = _parse_tensor_block(blk, sym)
        if t is not None:
            tensors.append(t)
    return tensors


def read_matrix(xlsx_path: str, sheet: str | int = 0) -> list[OpRow]:
    """Read the Shape Matrix sheet into a list of :class:`OpRow`."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    rows: list[OpRow] = []
    for _, r in df.iterrows():
        shape_raw = r.get("Shape", "")
        sym_raw = r.get("Symbolic Shape", "")
        rows.append(OpRow(
            phase=str(r.get("Phase", "")),
            seq_len=r.get("Seq Len"),
            ctx_len=r.get("Ctx Len"),
            batch_size=r.get("Batch Size"),
            tp=int(r.get("TP", 1)),
            module=str(r.get("Module", "")),
            op_name=str(r.get("Op Name", "")),
            backend=str(r.get("Backend", "")),
            layers=r.get("Layers"),
            tensors=_parse_shapes(shape_raw, sym_raw),
            symbolic_raw=str(sym_raw),
            shape_raw=str(shape_raw),
            memory_bytes=r.get("Memory (bytes)"),
            flops=r.get("FLOPs"),
            ai=r.get("AI"),
        ))
    return rows


def sweep_point(row: OpRow) -> tuple:
    """The (seq, ctx, batch) sweep coordinate this row was captured at."""
    return (row.seq_len, row.ctx_len, row.batch_size)


def dedup_key(row: OpRow, dense_sweep_ops: set[str] | None = None) -> tuple:
    """Key that collapses identical op shapes across the sweep explosion.

    Ops listed in ``dense_sweep_ops`` are *not* collapsed across sweep points:
    raw kernel rows (e.g. the triton MSA kernels) carry no operand shapes in
    the trace, so their tensor signature is identical at every seq/ctx/batch
    point and the default key would keep only one row per phase. Including the
    sweep coordinate for those ops yields the full dense sweep instead.
    """
    tensor_sig = tuple(
        (tuple(t.dims), t.dtype) for t in row.tensors
    )
    extra: tuple = ()
    if dense_sweep_ops and row.op_name in dense_sweep_ops:
        extra = sweep_point(row)
    return (row.op_name, row.module_attr, row.phase, row.tp, tensor_sig, extra)


def unique_rows(rows: list[OpRow],
                dense_sweep_ops: set[str] | None = None) -> list[OpRow]:
    """Return one representative OpRow per :func:`dedup_key`."""
    seen: dict[tuple, OpRow] = {}
    for r in rows:
        k = dedup_key(r, dense_sweep_ops)
        if k not in seen:
            seen[k] = r
    return list(seen.values())


def rows_to_oprows(rows: list[dict]) -> list[OpRow]:
    """In-memory Shape-Matrix rows -> :class:`OpRow` (no spreadsheet detour).

    ``rows`` are the dicts from :func:`breakdown.perf.shape_matrix.build_rows`,
    which already carry resolved integer dims, so the string shape column is
    only parsed as a fallback for rows that came from an .xlsx.
    """
    out: list[OpRow] = []
    for r in rows:
        resolved = r.get("_resolved_shapes")
        dtypes = r.get("_input_dtypes") or []
        shape_raw = str(r.get("Shape", "") or "")
        sym_raw = str(r.get("Symbolic Shape", "") or "")
        if resolved:
            sym_blocks = _TENSOR_SEP.split(sym_raw.strip()) if sym_raw else []
            tensors = [
                TensorShape(
                    dims=[int(d) for d in dims],
                    dtype=_norm_dtype(dtypes[i]) if i < len(dtypes) and dtypes[i]
                    else "bfloat16",
                    symbolic=sym_blocks[i] if i < len(sym_blocks) else "",
                )
                for i, dims in enumerate(resolved)
            ]
        else:
            tensors = _parse_shapes(shape_raw, sym_raw)
        out.append(OpRow(
            phase=str(r.get("Phase", "")),
            seq_len=r.get("Seq Len"),
            ctx_len=r.get("Ctx Len"),
            batch_size=r.get("Batch Size"),
            tp=int(r.get("TP", 1) or 1),
            module=str(r.get("Module", "")),
            op_name=str(r.get("Op Name", "")),
            backend=str(r.get("Backend", "")),
            layers=r.get("Layers"),
            tensors=tensors,
            symbolic_raw=sym_raw,
            shape_raw=shape_raw,
            memory_bytes=r.get("Memory (bytes)"),
            flops=r.get("FLOPs"),
            ai=r.get("AI"),
        ))
    return out
