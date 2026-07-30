# SPDX-License-Identifier: Apache-2.0
"""Shared types for the op maps.

:class:`ModelConfig` is the structural side-input an adapter needs but the
shapes don't carry (expert count, top-k, rope dim, sparse block size, …). It is
fed straight from :func:`breakdown.model_info.summarize_config`, so the config
summary is derived in one place instead of being hand-copied next to a matrix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    num_heads: int = 64
    num_kv_heads: int = 4
    head_dim: int = 128
    hidden_size: int = 6144
    intermediate_size: int = 12288          # dense MLP
    moe_intermediate_size: int = 3072       # per-expert
    num_experts: int = 128
    num_experts_per_tok: int = 4            # topk
    n_shared_experts: int = 1
    sparse_index_dim: int = 128             # lightning-indexer head dim
    sparse_num_index_heads: int = 4
    sparse_topk_blocks: int = 16
    sparse_block_size: int = 128
    vocab_size: int = 200064
    rope_dim: int = 64                      # rotary_dim (partial: n_h=64)

    @classmethod
    def from_config_summary(cls, c: dict[str, Any]) -> "ModelConfig":
        """Build from a :func:`breakdown.model_info.summarize_config` dict."""
        def g(*names, default=None):
            for n in names:
                if c.get(n) is not None:
                    return c[n]
            return default

        return cls(
            num_heads=g("num_heads", "num_attention_heads", default=64),
            num_kv_heads=g("num_kv_heads", "num_key_value_heads", default=4),
            head_dim=g("head_dim", default=128),
            hidden_size=g("hidden_size", default=6144),
            intermediate_size=g("intermediate_size", default=12288),
            moe_intermediate_size=g("moe_intermediate_size", default=3072),
            num_experts=g("num_experts", "n_routed_experts", default=128),
            num_experts_per_tok=g("num_experts_per_tok", default=4),
            n_shared_experts=g("n_shared_experts", default=1),
            sparse_index_dim=g("sparse_index_dim", default=128),
            sparse_num_index_heads=g("sparse_num_index_heads", default=4),
            sparse_topk_blocks=g("sparse_topk_blocks", default=16),
            sparse_block_size=g("sparse_block_size", default=128),
            vocab_size=g("vocab_size", default=200064),
        )

    @classmethod
    def from_summary(cls, path: str) -> "ModelConfig":
        """Build from a config-summary JSON file (offline / other machine)."""
        with open(path) as fh:
            return cls.from_config_summary(json.load(fh))


#: Historical name from the session converter; kept so ported adapters read the
#: same and external scripts keep working.
M3Config = ModelConfig


@dataclass
class EmittedCase:
    op: str                 # micro_perf op name
    args: dict[str, Any]    # micro_perf case arguments
    note: str = ""          # how the mapping was derived (coverage report)
    flops: float | None = None   # cost of this case, for runtime estimation
    nbytes: float | None = None


# ---------------------------------------------------------------------------
# case validity
# ---------------------------------------------------------------------------
#: Case arguments that may legitimately be 0 (an empty KV cache, no context).
#: Every *other* integer argument is a tensor extent, and a 0-sized extent is a
#: shape-derivation artefact - benchmarking it either aborts the whole op with a
#: kernel ``TORCH_CHECK`` or measures nothing at all.
ZERO_OK_KEYS = frozenset({
    "cache_len", "ctx_len", "kv_cache_len", "past_len", "prefix_len",
    "num_prefill_tokens", "num_decode_tokens", "seed",
})


def positive_dims(args: dict[str, Any]) -> str | None:
    """``None`` if every extent argument is positive, else why it is not."""
    for k, v in args.items():
        if isinstance(v, bool) or not isinstance(v, int):
            continue
        if k in ZERO_OK_KEYS:
            continue
        if v <= 0:
            return f"non-positive {k}={v}"
    return None


def check_case(op: str, args: dict[str, Any],
               constraints: dict[str, Any] | None = None) -> str | None:
    """Reason this case must not be benchmarked, or ``None`` if it is fine."""
    reason = positive_dims(args)
    if reason:
        return reason
    for check in (constraints or {}).get(op, ()):
        reason = check(args)
        if reason:
            return reason
    return None
