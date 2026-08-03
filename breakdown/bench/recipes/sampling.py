# SPDX-License-Identifier: Apache-2.0
"""Replay recipes for the sampler."""
from __future__ import annotations

from breakdown.bench.inputs import ArgBuildError
from breakdown.bench.recipes import override

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
