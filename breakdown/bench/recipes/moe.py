# SPDX-License-Identifier: Apache-2.0
"""Replay recipes for the MoE path: routing, permutation, grouped GEMM.

These were split across ``xpu.py`` and ``cuda.py``, which looked like a device
split but was not one: a synthesizer is registered by *argument name*, and both
files' registrations went into the same global table, so a CUDA run inherited
the XPU ones and vice versa. The split described where a recipe was first
needed, not where it applies. What these actually have in common is the MoE
dataflow -- a router picks experts, the tokens are permuted into
expert-contiguous order, one grouped GEMM runs them, and the result is
permuted back -- and every operand below is a step in it.
"""
from __future__ import annotations

from breakdown.bench.inputs import Ctx, synthesizer
from breakdown.bench.recipes import outputs, values
from breakdown.core import dtypes

# ---------------------------------------------------------------------------
# vllm-xpu-kernels MoE
# ---------------------------------------------------------------------------
# ``remap_hidden_states`` permutes the routed tokens into expert-contiguous
# order. Two of its arguments look like inputs but are *outputs*:
# ``rows_per_expert`` is a per-expert counter accumulated with atomics, and
# ``unpermuted_row_to_permuted_row`` is the scatter map it fills. vLLM allocates
# them fresh (zeroed) every forward; a benchmark that reuses them without
# resetting makes the offsets grow on every call until the scatter writes past
# the end of the destination - which does not raise, it takes the device down
# (UR_RESULT_ERROR_DEVICE_LOST). Hence: zeroed, reset between windows, and one
# call per window so the accumulation cannot compound inside a window either.
outputs("_moe_C::remap_hidden_states", "rows_per_expert",
        "unpermuted_row_to_permuted_row",
        single_rep="rows_per_expert accumulates offsets with atomics; a second "
                   "call inside a timed window would scatter out of bounds")

# ``topk_sigmoid`` writes all three of its leading tensors (the schema does mark
# them, so they are already reset) - listed here only for the token->expert map
# it fills, which must start empty.
outputs("_moe_C::moe_gather", "output")

# The grouped GEMM's row partition must sum to exactly the rows of ptr_A, and
# its length must equal num_experts; the generic ``rows_per_expert`` synthesizer
# derives both from the sibling operands, so only the expert count needs
# pinning when the trace recorded it as a scalar under a different name.
values("_xpu_C::cutlass_grouped_gemm_interface", is_B_int4=False,
       is_B_mxfp4=False)


@synthesizer("ptr_A", "ptr_B", "ptr_D", "ptr_scales", "ptr_bias")
def _grouped_gemm_operand(ctx: Ctx):
    """The grouped GEMM's operands are plain data despite the pointer names."""
    from breakdown.bench.inputs import make_tensor
    return make_tensor(ctx.dims, ctx.dtype, ctx.device)


# ---------------------------------------------------------------------------
# Triton fused-MoE alignment buffers
# ---------------------------------------------------------------------------
# ``moe_align_block_size`` produces a *padded* token->expert ordering: the
# sorted-token buffer is longer than the token count and the tail is filled with
# a sentinel equal to the token count. Reproducing that padding matters, because
# the expert GEMM skips padded rows - filling the buffer densely would make it
# do strictly more work than it does in the model.
@synthesizer("sorted_token_ids", "sorted_ids")
def _sorted_token_ids(ctx: Ctx):
    import torch
    dt = getattr(torch, dtypes.name_or(ctx.dtype, "int32"))
    n = ctx.numel
    tokens = 0
    for dims in ctx.tensor_dims():
        if len(dims) == 2:
            tokens = max(tokens, dims[0])
    tokens = tokens or n
    t = torch.full((n,), tokens, dtype=dt, device=ctx.device)
    t[:min(n, tokens)] = torch.arange(min(n, tokens), dtype=dt,
                                      device=ctx.device)
    return t.reshape(ctx.dims) if ctx.dims else t


@synthesizer("expert_ids_ptr", "num_tokens_post_pad", "num_tokens_post_padded")
def _post_pad(ctx: Ctx):
    import torch
    dt = getattr(torch, dtypes.name_or(ctx.dtype, "int32"))
    n = max(ctx.numel, 1)
    t = torch.zeros((n,), dtype=dt, device=ctx.device)
    return t.reshape(ctx.dims) if ctx.dims else t
