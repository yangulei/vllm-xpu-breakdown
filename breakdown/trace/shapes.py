# SPDX-License-Identifier: Apache-2.0
"""Span-less fallbacks: shapes for ops the trace does not record.

A capture made with :mod:`breakdown.kernel_hooks` records a Python-launched
kernel's real operands, so these passes only fire for archived or third-party
traces - and for the one gap no hook closes, a collective's dtype-less
``TensorList``.
"""
from __future__ import annotations


from ..cost import DTYPE_BYTES
from .rules import (
    _WEIGHT_PLUMBING_OPS, _is_hidden_state_op, _msa_kernel_layout)
from .forest import _Raw


def _infer_hidden_activation_ops(roots: list[_Raw], hidden_size: int | None
                                 ) -> None:
    """Fill missing shape/dtype on residual-stream ops from a neighbour.

    Two different gaps, one fix. A TP collective records only a dtype-less
    ``TensorList``, which no capture-time hook changes — the shape is genuinely
    absent from the dispatcher's own record. A Python-launched norm kernel
    carries no ``cpu_op`` at all; a capture made with
    :mod:`breakdown.kernel_hooks` records its real operands, so only an un-hooked
    trace reaches this pass for that case.

    Both operate on the residual hidden state ``[tokens, H]``, so
    for each such op missing a shape and/or a real dtype we borrow both from the
    op nearest in execution order that carries a genuine ``[tokens, H]`` tensor
    (2-D, trailing dim == ``hidden_size``, real dtype) — picking the nearest by
    timestamp keeps the correct per-step token count (prefill ``S`` vs decode
    ``B``). No-op when ``hidden_size`` is unknown or no reference exists.
    """
    if not hidden_size:
        return
    H = int(hidden_size)
    ops: list[_Raw] = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.kind == "op":
            ops.append(n)
        stack.extend(n.children)

    # Reference hidden-state tensors: (ts, [tokens, H], dtype).
    refs: list[tuple[float, list[int], str]] = []
    for o in ops:
        for sh, dt in zip(o.shapes, o.dtypes):
            if len(sh) == 2 and sh[1] == H and dt in DTYPE_BYTES:
                refs.append((o.ts, list(sh), dt))
                break
    if not refs:
        return

    for o in ops:
        need_shape = not o.shapes
        need_dtype = not any(d in DTYPE_BYTES for d in o.dtypes)
        if not (need_shape or need_dtype) or not _is_hidden_state_op(o.label):
            continue
        _ts, ref_shape, ref_dt = min(refs, key=lambda r: abs(r[0] - o.ts))
        if need_shape:
            o.shapes = [list(ref_shape)]
            o.dtypes = [ref_dt]
        else:
            o.dtypes = [ref_dt for _ in o.shapes]
        o.dtype = ref_dt


def _infer_attention_kernel_shapes(roots: list[_Raw], summary: dict,
                                   tp_size: int) -> None:
    """Reconstruct shape/dtype for shape-less MiniMax-M3 MSA / indexer kernels.

    **Fallback only.** A capture made with :mod:`breakdown.kernel_hooks` records
    these kernels' real operands in a ``kernel::`` span, so they arrive here
    already shaped and this pass skips them. It exists for traces captured
    without the hooks (archived CUDA references, third-party traces), where the
    alternative is an op with no shape at all — and therefore no memory, no
    FLOPs, no roofline and no replay.

    It is a *guess from the wrapper signature*, not a measurement: the MSA
    sparse-attention and lightning-indexer kernels carry no ``cpu_op`` on either
    backend (XPU ``flash_xpu`` SYCL kernels launched from ``xattention.py``;
    CUDA ``triton.jit`` kernels launched from
    ``common/ops/{sparse_attn,index_topk}.py``), so ``_attribute_kernels``
    surfaces them with no shape at all. Their primary-tensor layout is fixed by
    the wrapper signatures, so we rebuild it from the config (per-rank
    ``num_heads``/``head_dim`` for the attend kernels,
    ``sparse_num_index_heads``/``sparse_index_dim`` for the indexer kernels,
    ``sparse_topk_blocks`` for the top-k block ids) and the per-step token count
    ``total_q`` — taken as the leading dim of the nearest neighbouring activation
    op (so prefill gets ``S`` and decode ``B``). Kernels with no known layout
    fall back to borrowing the neighbour's activation shape so they at least
    carry the token dim + dtype. No-op without neighbour activations.

    Do not extend this table for a new kernel: capture with the hooks instead.
    """
    ops: list[_Raw] = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.kind == "op":
            ops.append(n)
        stack.extend(n.children)

    # Neighbour activation references: (ts, leading token dim, first shape, dtype).
    act_refs: list[tuple[float, int, list[int], str]] = []
    tok_refs: list[tuple[float, int]] = []
    H = summary.get("hidden_size")
    for o in ops:
        if not o.shapes or len(o.shapes[0]) < 2:
            continue
        dt = next((d for d in o.dtypes if d in DTYPE_BYTES), "")
        shp = list(o.shapes[0])
        act_refs.append((o.ts, shp[0], shp, dt or "bfloat16"))
        # Token-count references are restricted to genuine residual hidden
        # states: a ``[tokens, H]`` tensor on an op that isn't weight plumbing.
        # A weight is also ``[out_features, H]``, so an ``aten::t`` on one would
        # otherwise hand the MSA kernels the weight's out-features as their row
        # count (the symptom: ``[n_h·d/TP, n_idx/TP, d]`` instead of ``[B, ...]``).
        if (H and len(shp) == 2 and shp[1] == int(H)
                and o.label.split("::")[-1].lower() not in _WEIGHT_PLUMBING_OPS):
            tok_refs.append((o.ts, shp[0]))
    if not act_refs:
        return
    if not tok_refs:  # hidden size unknown / no hidden-state op — best effort
        tok_refs = [(ts, tok) for ts, tok, _shp, _dt in act_refs]

    tp = max(1, int(tp_size or 1))
    n_h = summary.get("num_heads")
    d = summary.get("head_dim")
    n_idx = summary.get("sparse_num_index_heads")
    idx_d = summary.get("sparse_index_dim")
    topk_blocks = summary.get("sparse_topk_blocks")

    for o in ops:
        if o.shapes:
            continue
        layout = _msa_kernel_layout(o.label)
        # ``flash_xpu`` ops with no recognised layout still get the neighbour
        # fallback (the whole namespace is MSA); other ops are left untouched.
        if layout is None and "flash_xpu" not in o.label.lower():
            continue
        _ts, _tok, act_shape, dt = min(act_refs, key=lambda r: abs(r[0] - o.ts))
        token = min(tok_refs, key=lambda r: abs(r[0] - o.ts))[1]
        shape: list[int] | None = None
        if layout == "attn" and n_h and d:
            shape = [token, max(1, int(n_h) // tp), int(d)]
        elif layout == "index" and n_idx and idx_d:
            shape = [token, max(1, int(n_idx) // tp), int(idx_d)]
        elif layout == "topk" and n_idx and topk_blocks:
            # The top-k kernels read a ``[n_idx, total_q, max_block]`` score and
            # write ``[n_idx, total_q, topk]`` block ids. Only the top-k width is
            # config-derivable (``max_block`` depends on the runtime max seq len),
            # so report the block-id tensor — int32, not the activation dtype.
            shape = [max(1, int(n_idx) // tp), token, int(topk_blocks)]
            dt = "int32"
        if shape is None:
            shape = list(act_shape)  # fallback: representative activation
        o.shapes = [shape]
        o.dtypes = [dt]
        o.dtype = dt
