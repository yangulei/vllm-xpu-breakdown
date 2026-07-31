# SPDX-License-Identifier: Apache-2.0
"""XPU-specific replay recipes (vllm-xpu-kernels, IPEX, xattention)."""
from __future__ import annotations

from breakdown.bench.inputs import Ctx, synthesizer
from breakdown.bench.recipes import outputs, skip, values

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
# MiniMax-M3 fused QK-norm + RoPE + KV insert
# ---------------------------------------------------------------------------
@synthesizer("kv_cache", "index_cache", "key_cache", "value_cache",
             "qkv", "q_out", "index_q_out")
def _cache_operand(ctx: Ctx):
    from breakdown.bench.inputs import make_tensor
    return make_tensor(ctx.dims, ctx.dtype, ctx.device)


# ---------------------------------------------------------------------------
# ops that exist on XPU but must not be replayed standalone
# ---------------------------------------------------------------------------
skip("vllm::xpu_topk_topp_sampler",
     "sampling wrapper: draws on a generator and per-request sampling "
     "metadata that only exist inside a running engine")
