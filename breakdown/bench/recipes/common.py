# SPDX-License-Identifier: Apache-2.0
"""Device-independent replay recipes and index-operand synthesizers.

The rule these encode: **an integer tensor is an index until proven otherwise.**
:mod:`breakdown.bench.inputs` refuses to fill one at random, so every integer
operand a dispatched op takes needs a registration here (or a name already
covered by the generic index synthesizers). That is deliberate - a random
``slot_mapping`` makes a paged-KV kernel scatter over the whole cache, and a
random ``rows_per_expert`` makes a grouped GEMM read past its input.
"""
from __future__ import annotations

from breakdown.bench.inputs import Ctx, synthesizer

#: Elementwise operands that happen to be integer tensors (position counters,
#: token id vectors). They are *data*, not indices, so a dense ascending fill is
#: both valid and representative; it starts at 1 so an op that divides or takes
#: a remainder by this operand cannot hit a division by zero.
_ELEMENTWISE = (
    "self", "other", "input", "tensor", "src", "end", "mat1", "mat2",
    "tensor1", "tensor2", "batch1", "batch2", "condition", "grad_output",
)


@synthesizer(*_ELEMENTWISE)
def _elementwise(ctx: Ctx):
    import torch
    from breakdown.bench.inputs import DTYPE_MAP, make_tensor

    attr = DTYPE_MAP.get((ctx.dtype or "").lower(), "")
    if attr not in ("int64", "int32", "int16", "int8", "uint8"):
        return make_tensor(ctx.dims, ctx.dtype, ctx.device)
    dt = getattr(torch, attr)
    n = ctx.numel
    t = torch.arange(1, n + 1, dtype=dt, device=ctx.device)
    return t.reshape(ctx.dims) if ctx.dims else t.reshape(())


@synthesizer("indices", "index", "idx", "input_ids", "token_ids")
def _gather_index(ctx: Ctx):
    """A gather index bounded by the table it indexes into.

    The bound is the leading dim of the largest 2-D operand (an embedding table
    or a KV cache); indexing past it is an out-of-bounds read, and clamping
    everything to 0 would measure a single cache line instead of a real gather.
    """
    import torch
    from breakdown.bench.inputs import DTYPE_MAP

    dt = getattr(torch, DTYPE_MAP.get((ctx.dtype or "").lower(), "int64"))
    rows = 0
    for dims in ctx.tensor_dims():
        if len(dims) >= 2:
            rows = max(rows, dims[0])
    n = ctx.numel
    high = rows or n
    t = torch.arange(n, dtype=dt, device=ctx.device) % max(high, 1)
    return t.reshape(ctx.dims) if ctx.dims else t.reshape(())


@synthesizer("weight", "bias", "scale", "scales", "cos_sin_cache",
             "q_norm_weight", "k_norm_weight")
def _plain_tensor(ctx: Ctx):
    """Named weight-like operands: ordinary values, never index treatment."""
    from breakdown.bench.inputs import make_tensor
    return make_tensor(ctx.dims, ctx.dtype, ctx.device)
