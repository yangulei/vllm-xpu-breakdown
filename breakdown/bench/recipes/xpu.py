# SPDX-License-Identifier: Apache-2.0
"""XPU-specific replay recipes (vllm-xpu-kernels, IPEX, xattention)."""
from __future__ import annotations

from breakdown.bench.inputs import ArgBuildError, Ctx, synthesizer
from breakdown.bench.recipes import outputs, override, values

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
# sampling
# ---------------------------------------------------------------------------
# ``xpu_topk_topp_sampler`` looked context-bound - it is handed the engine's
# RNG state and the per-request sampling metadata - so it was skipped. But none
# of that is *context*: the schema is
#
#   (Tensor random_sampled, Tensor? logits_to_return, Tensor logits,
#    Tensor? k, Tensor? p, str logprobs_mode, Tensor? seeds, float lambda_)
#
# and every one of those is a plain value. ``seeds`` is just the philox
# ``(seed, offset)`` pair the caller reads out of the default generator - a
# two-element **CPU** int64 tensor, which is why it needs a recipe rather than
# the generic builder (which allocates on the device). ``random_sampled`` is
# the output the kernel writes. With those two supplied the sampler replays
# exactly as dispatched, and it matters: it runs over the full ``[B, V]``
# logits every decode step.
@override("vllm::xpu_topk_topp_sampler")
def _topk_topp_sampler(case, resolved, device: str):
    import torch

    from breakdown.bench.inputs import Call, make_tensor, torch_dtype

    logits_slot = None
    for slot in case.args:
        dims = slot.get("dims") or [] if slot.get("kind") == "tensor" else []
        if len(dims) == 2:
            logits_slot = slot
            break
    if logits_slot is None:
        raise ArgBuildError(
            f"{case.op}: no [batch, vocab] logits operand in the recorded call")
    rows, vocab = (int(d) for d in logits_slot["dims"])
    logits = make_tensor([rows, vocab], logits_slot.get("dtype") or "float",
                         device)

    # The kernel writes one sampled token id per row.
    random_sampled = torch.empty(rows, dtype=torch.int64, device=device)
    # philox (seed, offset), read from the generator by the caller - on CPU.
    seeds = torch.tensor([0, 0], dtype=torch.int64, device="cpu")

    # ``logprobs_mode`` decides whether the kernel also materializes the
    # processed logits. The profiler records a string argument as an empty
    # slot, so take the one the trace implies: no returned logits tensor means
    # the cheap raw mode, which is what vLLM runs by default.
    returns_logits = _slot_is_tensor(case.args, skip=logits_slot)
    logprobs_mode = "processed_logprobs" if returns_logits else "raw_logprobs"
    logits_to_return = (torch.empty_like(logits) if returns_logits else None)

    lambda_ = 1.0
    for slot in case.args:
        if slot.get("kind") == "scalar":
            try:
                lambda_ = float(str(slot.get("value")).rstrip("."))
            except (TypeError, ValueError):
                pass
    del torch_dtype
    return Call(args=[random_sampled, logits_to_return, logits, None, None,
                      logprobs_mode, seeds, lambda_],
                mutated=[random_sampled]
                + ([logits_to_return] if logits_to_return is not None else []))


def _slot_is_tensor(slots, skip) -> bool:
    """Was ``logits_to_return`` (the 2nd slot) recorded as a real tensor?"""
    for slot in slots:
        if slot is skip:
            continue
        if slot.get("kind") == "tensor" and len(slot.get("dims") or []) == 2:
            return True
    return False
