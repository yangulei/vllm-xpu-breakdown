# SPDX-License-Identifier: Apache-2.0
"""CUDA-specific replay recipes (Triton MoE, FlashInfer norms)."""
from __future__ import annotations

from breakdown.bench.inputs import Ctx, synthesizer

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
    from breakdown.bench.inputs import DTYPE_MAP

    dt = getattr(torch, DTYPE_MAP.get((ctx.dtype or "").lower(), "int32"))
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
    from breakdown.bench.inputs import DTYPE_MAP

    dt = getattr(torch, DTYPE_MAP.get((ctx.dtype or "").lower(), "int32"))
    n = max(ctx.numel, 1)
    t = torch.zeros((n,), dtype=dt, device=ctx.device)
    return t.reshape(ctx.dims) if ctx.dims else t
