# SPDX-License-Identifier: Apache-2.0
"""Map vllm-xpu-breakdown Shape-Matrix ops -> xpu-perf/micro_perf workload cases.

Each breakdown trace op is translated to zero or more micro_perf op cases. The
Shape Matrix rows are *already the resolved sweep* (concrete per-rank dims at each
seq/ctx/batch point), so an adapter reads concrete dims from the row plus the
model config side-input (`M3Config`) for structural args the shape doesn't carry
(num_experts, topk, rope_dim, block_size, ...).

An adapter returns a list of ``EmittedCase(op, args, note)``. Fused breakdown ops
(``fused_minimax_m3_qknorm_rope_kv_insert``) expand into several micro_perf ops.
Ops with no micro_perf equivalent are recorded via :data:`UNMAPPED` /
:data:`SKIP` and surfaced in the coverage report.
"""
from __future__ import annotations

from typing import Callable

from breakdown.perf.matrix_reader import OpRow
from breakdown.perf.op_map import common
from breakdown.perf.op_map.common import EmittedCase, ModelConfig, M3Config


SKIP_OPS = {
    "aten::add", "aten::clone", "aten::detach_", "aten::div_", "aten::item",
    "aten::masked_fill_", "aten::movedim", "aten::mul_", "aten::new_empty",
    "aten::to", "aten::unbind", "aten::zeros", "aten::embedding",
    "vllm::xpu_topk_topp_sampler", "TorchDynamo Cache Lookup",
    "Torch-Compiled Region: 0/1",
    "vllm::moe_forward_shared",
}


def _tok(row: OpRow) -> int:
    """New tokens processed this row: seq_len for prefill, batch for decode."""
    try:
        if row.phase == "prefill":
            return int(row.seq_len)
        return int(row.batch_size)
    except (TypeError, ValueError):
        # fall back to leading concrete dim of first tensor
        return int(row.tensors[0].dims[0]) if row.tensors else 1


def _attn_sbc(row: OpRow) -> tuple[int, int, int]:
    """(batch_size, q_len, cache_len) for an attention-family op."""
    b = int(row.batch_size) if row.batch_size is not None else 1
    c = int(row.ctx_len) if row.ctx_len is not None else 0
    if row.phase == "prefill":
        q = int(row.seq_len)
    else:
        q = 1
    return b, q, c


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
def adapt_linear(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """aten::linear -> gemm (arg_type=default, concrete per-rank M/K/N)."""
    if len(row.tensors) < 2:
        return []
    a, w = row.tensors[0], row.tensors[1]
    # input [M, K]; weight [N, K]  (torch linear stores weight out-major)
    if len(a.dims) < 2 or len(w.dims) < 2:
        return []
    M, K = int(a.dims[0]), int(a.dims[1])
    N = int(w.dims[0])
    dt = a.dtype if a.dtype in ("float32", "float16", "bfloat16") else "bfloat16"
    return [EmittedCase("gemm", {
        "arg_type": "default", "M": M, "K": K, "N": N, "dtype": dt,
    }, note=f"linear {row.module_attr}")]


def adapt_attention(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """vllm::unified_attention_with_output -> flash_attention (paged)."""
    if not row.tensors:
        return []
    q = row.tensors[0]
    qh = int(q.dims[1]) if len(q.dims) >= 3 else max(1, cfg.num_heads)
    hd = int(q.dims[2]) if len(q.dims) >= 3 else cfg.head_dim
    kvh = cfg.num_kv_heads
    if len(row.tensors) >= 2 and len(row.tensors[1].dims) >= 3:
        kvh = int(row.tensors[1].dims[1])
    b, ql, cl = _attn_sbc(row)
    mode = "prefill" if row.phase == "prefill" else "decode"
    return [EmittedCase("flash_attention", {
        "arg_type": "llm", "attn_mode": mode,
        "q_head_num": qh, "kv_head_num": kvh, "head_dim": hd,
        "batch_size": b, "q_len": ql, "cache_len": cl,
        "block_size": cfg.sparse_block_size,
        "dtype": "bfloat16", "cache_dtype": "bfloat16", "dst_dtype": "bfloat16",
    }, note=f"dense attn {mode}")]


def adapt_kv_update(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """vllm::unified_kv_cache_update -> store_kv_cache."""
    kvh = cfg.num_kv_heads
    hd = cfg.head_dim
    if row.tensors and len(row.tensors[0].dims) >= 3:
        kvh = int(row.tensors[0].dims[1])
        hd = int(row.tensors[0].dims[2])
    b, ql, cl = _attn_sbc(row)
    qh = max(1, _per_rank(cfg.num_heads, row.tp))
    mode = "prefill" if row.phase == "prefill" else "decode"
    return [EmittedCase("store_kv_cache", {
        "arg_type": "llm", "attn_mode": mode,
        "q_head_num": qh, "kv_head_num": kvh, "head_dim": hd,
        "batch_size": b, "q_len": ql, "cache_len": cl,
        "block_size": cfg.sparse_block_size,
        "dtype": "bfloat16", "cache_dtype": "bfloat16",
    }, note="kv cache update")]


def adapt_qknorm_rope_kv(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_C::fused_minimax_m3_qknorm_rope_kv_insert -> fused_qknorm_rope_kv (exact).

    Single fused kernel (QK-norm + partial RoPE), not a 3-op decomposition.
    """
    qkv = row.tensors[0] if row.tensors else None
    if qkv is None or len(qkv.dims) < 2:
        return []
    nt = int(qkv.dims[0])
    nh = _per_rank(cfg.num_heads, row.tp)
    nkv = _per_rank(cfg.num_kv_heads, row.tp)
    head_dim = cfg.head_dim
    rotary_dim = cfg.rope_dim
    max_position = 131072
    # cos_sin_cache is the 4th traced tensor: [max_position, rotary_dim].
    if len(row.tensors) >= 4 and len(row.tensors[3].dims) >= 2:
        try:
            max_position = int(row.tensors[3].dims[0])
            rotary_dim = int(row.tensors[3].dims[1])
        except (TypeError, ValueError):
            pass
    return [EmittedCase("fused_qknorm_rope_kv", {
        "arg_type": "llm", "num_tokens": nt,
        "num_heads": nh, "num_kv_heads": nkv, "head_dim": head_dim,
        "rotary_dim": rotary_dim, "max_position": max_position,
        "eps": 1e-6, "dtype": "bfloat16",
    }, note="fused QK-norm + partial RoPE (exact _C op)")]


def adapt_topk_sigmoid(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_moe_C::topk_sigmoid -> topk_sigmoid (exact sigmoid gating)."""
    if not row.tensors or len(row.tensors[0].dims) < 2:
        return []
    tw = row.tensors[0]  # topk_weights [num_tokens, topk]
    nt = int(tw.dims[0])
    topk = int(tw.dims[1])
    E = cfg.num_experts
    if len(row.tensors) >= 4 and len(row.tensors[3].dims) >= 2:
        try:
            E = int(row.tensors[3].dims[1])  # gating_output [num_tokens, E]
        except (TypeError, ValueError):
            pass
    has_bias = len(row.tensors) >= 5
    return [EmittedCase("topk_sigmoid", {
        "arg_type": "llm", "num_tokens": nt, "num_experts": E, "topk": topk,
        "has_bias": has_bias, "renormalize": True, "dtype": "float32",
    }, note="sigmoid MoE gating (exact _moe_C op)")]


def adapt_moe_gather(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_moe_C::moe_gather -> moe_gather (exact combine; concrete dims)."""
    if not row.tensors or len(row.tensors[0].dims) < 2:
        return []
    out = row.tensors[0]
    nt, H = int(out.dims[0]), int(out.dims[1])
    topk = cfg.num_experts_per_tok
    if len(row.tensors) >= 3 and len(row.tensors[2].dims) >= 2:
        try:
            topk = int(row.tensors[2].dims[1])
        except (TypeError, ValueError):
            pass
    return [EmittedCase("moe_gather", {
        "arg_type": "llm", "num_tokens": nt, "hidden_size": H,
        "num_experts": cfg.num_experts, "topk": topk,
        "dtype": "bfloat16",
    }, note="MoE combine/gather (exact _moe_C op)")]


def adapt_grouped_gemm(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_xpu_C::cutlass_grouped_gemm_interface -> cutlass_grouped_gemm (exact bf16).

    Concrete per-rank dims straight from the traced tensors:
    A [total_rows, K], B [num_experts, K, N].
    """
    if len(row.tensors) < 2 or len(row.tensors[1].dims) < 3:
        return []
    a, w = row.tensors[0], row.tensors[1]
    try:
        total_rows = int(a.dims[0])
        E, K, N = int(w.dims[0]), int(w.dims[1]), int(w.dims[2])
    except (TypeError, ValueError):
        return []
    return [EmittedCase("cutlass_grouped_gemm", {
        "arg_type": "llm", "total_rows": total_rows, "K": K, "N": N,
        "num_experts": E, "dtype": "bfloat16",
    }, note="expert grouped gemm (bf16, exact _xpu_C op)")]


def adapt_silu_clamp(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_C::silu_and_mul_with_clamp -> silu_and_mul_with_clamp (exact)."""
    out = row.tensors[0] if row.tensors else None
    if out is None or len(out.dims) < 2:
        return []
    nt, d = int(out.dims[0]), int(out.dims[1])
    return [EmittedCase("silu_and_mul_with_clamp", {
        "arg_type": "llm", "num_tokens": nt, "intermediate_size": d,
        "swiglu_limit": 7.0, "dtype": "bfloat16",
    }, note="clamped SwiGLU (exact _C op)")]


def adapt_swigluoai(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_C::swigluoai_and_mul -> swigluoai_and_mul (exact)."""
    out = row.tensors[0] if row.tensors else None
    if out is None or len(out.dims) < 2:
        return []
    nt, d = int(out.dims[0]), int(out.dims[1])
    return [EmittedCase("swigluoai_and_mul", {
        "arg_type": "llm", "num_tokens": nt, "intermediate_size": d,
        "alpha": 1.702, "limit": 7.0, "dtype": "bfloat16",
    }, note="SwiGLU-OAI activation (exact _C op)")]


def adapt_remap(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_moe_C::remap_hidden_states -> remap_hidden_states (exact)."""
    if not row.tensors or len(row.tensors[0].dims) < 2:
        return []
    hs = row.tensors[0]
    nt, H = int(hs.dims[0]), int(hs.dims[1])
    E = cfg.num_experts
    topk = cfg.num_experts_per_tok
    if len(row.tensors) >= 3 and row.tensors[2].dims:
        try:
            E = int(row.tensors[2].dims[0])
        except (TypeError, ValueError):
            pass
    if len(row.tensors) >= 5 and len(row.tensors[4].dims) >= 2:
        try:
            topk = int(row.tensors[4].dims[1])
        except (TypeError, ValueError):
            pass
    return [EmittedCase("remap_hidden_states", {
        "arg_type": "llm", "num_tokens": nt, "hidden_size": H,
        "num_experts": E, "topk": topk, "dtype": "bfloat16",
    }, note="MoE token permute (exact _moe_C op)")]


def adapt_rms_norm(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """triton::_gemma_rmsnorm_kernel -> rms_norm (basic, default arg)."""
    if not row.tensors:
        return []
    t = row.tensors[0]
    M = int(t.dims[0]); H = int(t.dims[1]) if len(t.dims) >= 2 else cfg.hidden_size
    return [EmittedCase("rms_norm", {
        "arg_type": "default", "batch_size": M, "dim_size": H,
        "dtype": "bfloat16",
    }, note="gemma rmsnorm")]


def adapt_add_rms_norm(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """triton::_gemma_fused_add_rmsnorm_kernel -> add_rms_norm (llm)."""
    if not row.tensors:
        return []
    t = row.tensors[0]
    M = int(t.dims[0]); H = int(t.dims[1]) if len(t.dims) >= 2 else cfg.hidden_size
    return [EmittedCase("add_rms_norm", {
        "arg_type": "llm", "num_tokens": M, "hidden_size": H,
        "dtype": "bfloat16",
    }, note="gemma fused add-rmsnorm")]


def adapt_allreduce(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    if not row.tensors:
        return []
    t = row.tensors[0]
    nt = int(t.dims[0]); H = int(t.dims[1]) if len(t.dims) >= 2 else cfg.hidden_size
    return [EmittedCase("all_reduce", {
        "arg_type": "llm", "world_size": row.tp,
        "num_tokens": nt, "hidden_size": H, "dtype": "bfloat16",
    }, note="tp all_reduce")]


def adapt_allgather(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    if not row.tensors:
        return []
    t = row.tensors[0]
    nt = int(t.dims[0]); H = int(t.dims[1]) if len(t.dims) >= 2 else cfg.hidden_size
    return [EmittedCase("all_gather", {
        "arg_type": "llm", "world_size": row.tp,
        "num_tokens": nt, "hidden_size": H, "dtype": "bfloat16",
    }, note="tp all_gather")]


# --- MSA (P2.5 new ops); emitted here, op_defs added in xpu-perf ------------
def adapt_msa_index_score(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    b, ql, cl = _attn_sbc(row)
    return [EmittedCase("msa_index_score", {
        "arg_type": "llm", "attn_mode": _mode(row),
        "num_index_heads": _per_rank(cfg.sparse_num_index_heads, row.tp),
        "index_head_dim": cfg.sparse_index_dim,
        "block_size": cfg.sparse_block_size,
        "batch_size": b, "q_len": ql, "cache_len": cl, "dtype": "bfloat16",
    }, note="MSA lightning-indexer score")]


def adapt_msa_index_topk(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    b, ql, cl = _attn_sbc(row)
    return [EmittedCase("msa_index_topk", {
        "arg_type": "llm", "attn_mode": _mode(row),
        "num_index_heads": _per_rank(cfg.sparse_num_index_heads, row.tp),
        "index_head_dim": cfg.sparse_index_dim,
        "block_size": cfg.sparse_block_size, "topk_blocks": cfg.sparse_topk_blocks,
        "batch_size": b, "q_len": ql, "cache_len": cl, "dtype": "bfloat16",
    }, note="MSA index top-k block select")]


def adapt_msa_sparse_attn(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    q = row.tensors[0] if row.tensors else None
    qh = int(q.dims[1]) if q and len(q.dims) >= 3 else _per_rank(cfg.num_heads, row.tp)
    hd = int(q.dims[2]) if q and len(q.dims) >= 3 else cfg.head_dim
    b, ql, cl = _attn_sbc(row)
    return [EmittedCase("msa_sparse_attn", {
        "arg_type": "llm", "attn_mode": _mode(row),
        "q_head_num": qh, "kv_head_num": _per_rank(cfg.num_kv_heads, row.tp),
        "head_dim": hd, "block_size": cfg.sparse_block_size,
        "topk_blocks": cfg.sparse_topk_blocks,
        "batch_size": b, "q_len": ql, "cache_len": cl, "dtype": "bfloat16",
    }, note="MSA block-sparse attend")]


def _mode(row: OpRow) -> str:
    return "prefill" if row.phase == "prefill" else "decode"


def _per_rank(v: int, tp: int) -> int:
    return max(1, v // max(1, tp))


# op_name -> adapter
ADAPTERS: dict[str, Callable[[OpRow, M3Config], list[EmittedCase]]] = {
    "aten::linear": adapt_linear,
    "vllm::unified_attention_with_output": adapt_attention,
    "vllm::unified_kv_cache_update": adapt_kv_update,
    "_C::fused_minimax_m3_qknorm_rope_kv_insert": adapt_qknorm_rope_kv,
    "_moe_C::topk_sigmoid": adapt_topk_sigmoid,
    "_moe_C::moe_gather": adapt_moe_gather,
    "_moe_C::remap_hidden_states": adapt_remap,
    "_xpu_C::cutlass_grouped_gemm_interface": adapt_grouped_gemm,
    "_C::silu_and_mul_with_clamp": adapt_silu_clamp,
    "_C::swigluoai_and_mul": adapt_swigluoai,
    "triton::_gemma_rmsnorm_kernel": adapt_rms_norm,
    "triton::_gemma_fused_add_rmsnorm_kernel": adapt_add_rms_norm,
    "c10d::allreduce_": adapt_allreduce,
    "c10d::_allgather_base_": adapt_allgather,
    "flash_xpu::minimax_m3_index_score": adapt_msa_index_score,
    "flash_xpu::minimax_m3_index_decode": adapt_msa_index_score,
    "flash_xpu::minimax_m3_index_topk": adapt_msa_index_topk,
    "flash_xpu::minimax_m3_sparse_attn": adapt_msa_sparse_attn,
    "flash_xpu::minimax_m3_sparse_attn_decode": adapt_msa_sparse_attn,
}

# micro_perf op -> which JSON group it belongs to (compute vs collective vs msa)
COLLECTIVE_OPS = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}
MSA_OPS = {"msa_index_score", "msa_index_topk", "msa_sparse_attn"}

# MSA ops are derived purely from the sweep coordinate (seq/ctx/batch), so they
# must not be collapsed by the shape de-duplicator -- see
# ``matrix_reader.dedup_key``. Kept identical in spirit to the CUDA map so both
# platforms sweep the same points.
DENSE_SWEEP_OPS = {
    "flash_xpu::minimax_m3_index_score",
    "flash_xpu::minimax_m3_index_decode",
    "flash_xpu::minimax_m3_index_topk",
    "flash_xpu::minimax_m3_sparse_attn",
    "flash_xpu::minimax_m3_sparse_attn_decode",
}

# Emission-time shape gates. Policy: only *impossible* shapes are dropped here
# (that is :func:`common.positive_dims` - a 0-sized extent is a shape-derivation
# artefact, never a workload). Kernel *capability* limits are deliberately NOT
# listed: e.g. the M3 sparse-attention kernel tiles a fixed 16-head GQA group
# (``TORCH_CHECK(q.size(1) == num_kv_heads * 16)`` in deepklox
# ``msa_interface.cpp``), which TP > num_kv_heads cannot satisfy - that is a
# real workload the benchmark must *report as failing* so the kernel gets
# fixed, not a case to hide.
CASE_CONSTRAINTS: dict[str, tuple] = {}
