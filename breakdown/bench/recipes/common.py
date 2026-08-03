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
from breakdown.bench.recipes.table import register
from breakdown.core import dtypes


def skip(op: str, reason: str) -> None:
    """Declare that ``op`` must not be replayed, and why."""
    register(op, skip=reason)

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
    from breakdown.bench.inputs import make_tensor

    if not dtypes.is_integer(ctx.dtype):
        return make_tensor(ctx.dims, ctx.dtype, ctx.device)
    dt = getattr(torch, dtypes.torch_name(ctx.dtype))
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
    dt = getattr(torch, dtypes.name_or(ctx.dtype, "int64"))
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


# Dispatch *wrappers* that cannot be invoked outside a live vLLM forward pass.
# The kernels they launch are separate ops in the reconstructed graph and are
# benchmarked directly, so refusing the wrapper loses nothing - and saying so
# is the point: the plan reports it with the reason instead of dropping it.
skip("vllm::unified_attention",
     "reads KV cache + attention metadata from vLLM's forward context")
skip("vllm::moe_forward_shared",
     "fused MoE dispatch wrapper; its router/expert/shared-expert kernels "
     "are benchmarked as their own ops")
skip("vllm::moe_forward",
     "fused MoE dispatch wrapper; its constituent kernels are benchmarked "
     "as their own ops")
