# SPDX-License-Identifier: Apache-2.0
"""CUDA op map: Shape-Matrix ops -> xpu-perf/micro_perf workload cases.

Companion to :mod:`op_map` (which targets the Intel XPU dispatch). The CUDA
graph of MiniMax-M3 dispatches several functional blocks to different kernels,
so every adapter here maps a breakdown op to *its own* exact micro_perf op:

===========================================  ==========================================
breakdown op (CUDA)                          micro_perf op / provider
===========================================  ==========================================
``aten::linear``                             ``gemm`` (torch)
``_C::fused_minimax_m3_qknorm_rope_kv_...``  ``fused_qknorm_rope_kv`` dense|sparse
``vllm::unified_attention_with_output``      ``flash_attention`` (vllm_flash_attn)
``vllm::unified_kv_cache_update``            ``store_kv_cache`` (reshape_and_cache_flash)
``_moe_C::topk_sigmoid``                     ``topk_sigmoid``
``_C::silu_and_mul_with_clamp``              ``silu_and_mul_with_clamp``
``_moe_C::moe_align_block_size``             ``moe_align_block_size``
``_moe_C::moe_sum``                          ``moe_sum``
``triton::fused_moe_kernel``                 ``fused_moe_gemm`` gemm1 + gemm2 (triton)
``flashinfer::gemma_rmsnorm``                ``rms_norm`` (flashinfer_gemma)
``flashinfer::gemma_fused_add_rmsnorm``      ``add_rms_norm`` (flashinfer_gemma)
``triton::_index_block_score_kernel`` &      ``msa_index_score`` (triton)
``triton::_decode_index_score_kernel``
``triton::_topk_index*_kernel``              ``msa_index_topk`` (triton)
``triton::_gqa_sparse_*`` &                  ``msa_sparse_attn`` (triton)
``triton::_merge_topk_attn_out_kernel``
``vllm::all_reduce`` / ``vllm::all_gather``  ``all_reduce`` / ``all_gather`` (NCCL)
``aten::embedding``                          ``embedding``
``aten::add`` / ``aten::div_``               ``add`` / ``div``
``aten::masked_fill_``                       ``masked_fill``
``aten::softmax`` / ``argmax``               ``softmax`` / ``argmax``
``aten::exponential_``                       ``exponential``
===========================================  ==========================================

Composite parents (``vllm::moe_forward_shared``) and pure framework/plumbing
regions (``aten::new_empty``, ``aten::to``, dynamo regions) are ``SKIP``ped:
their exact leaf kernels are each benchmarked, so mapping the parent would
double-count. ``aten::to`` is a dtype-conversion plumbing op that the XPU map
skips too — keeping both platforms on the same skip set is what makes the
Shape-Matrix coverage apple-to-apple.
"""
from __future__ import annotations

from typing import Callable

from breakdown.perf.matrix_reader import OpRow
from breakdown.perf.op_map.common import EmittedCase, M3Config, ModelConfig
from breakdown.perf.op_map.xpu import _per_rank, _mode, _attn_sbc

# Framework/plumbing regions and composite parents: no benchmarkable leaf kernel
# of their own.
SKIP_OPS = {
    "aten::new_empty", "aten::clone", "aten::detach_", "aten::item",
    "aten::movedim", "aten::unbind", "aten::zeros", "aten::to",
    "TorchDynamo Cache Lookup", "Torch-Compiled Region: 0/1",
    "vllm::moe_forward_shared",
}

_DT = ("float32", "float16", "bfloat16")


def _dtype_of(row: OpRow, idx: int = 0, default: str = "bfloat16") -> str:
    if len(row.tensors) > idx and row.tensors[idx].dtype in _DT:
        return row.tensors[idx].dtype
    return default


def _dims2(row: OpRow, idx: int = 0) -> tuple[int, int] | None:
    """Concrete [M, N] of the ``idx``-th traced tensor, if it has >= 2 dims."""
    if len(row.tensors) <= idx:
        return None
    d = row.tensors[idx].dims
    if len(d) < 2:
        return None
    try:
        return int(d[0]), int(d[1])
    except (TypeError, ValueError):
        return None


def _tokens(row: OpRow, cfg: M3Config) -> int:
    """New tokens this row processes (prefill: seq_len, decode: batch)."""
    try:
        return int(row.seq_len) if row.phase == "prefill" else int(row.batch_size)
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------------------
# dense compute
# ---------------------------------------------------------------------------
def adapt_linear(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """aten::linear -> gemm (concrete per-rank M/K/N)."""
    a = _dims2(row, 0)
    w = _dims2(row, 1)
    if a is None or w is None:
        return []
    M, K = a
    N = w[0]
    if K == 0 or N == 0 or M == 0:
        # o_proj rows whose K symbolizes to the (unresolved) context dim
        K = cfg.hidden_size if K == 0 else K
        N = cfg.hidden_size if N == 0 else N
        if M == 0:
            return []
    return [EmittedCase("gemm", {
        "arg_type": "default", "M": M, "K": K, "N": N,
        "dtype": _dtype_of(row),
    }, note=f"linear {row.module_attr}")]


def adapt_qknorm_rope_kv(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """_C::fused_minimax_m3_qknorm_rope_kv_insert -> fused_qknorm_rope_kv.

    Dense layers call the 9-arg form; sparse layers additionally norm/rope the
    lightning-index branch and scatter K/V + index-K into the paged caches, so
    the sparse rows map onto ``variant="sparse"``.
    """
    qkv = _dims2(row, 0)
    if qkv is None:
        return []
    nt, qkv_dim = qkv
    nh = _per_rank(cfg.num_heads, row.tp)
    nkv = _per_rank(cfg.num_kv_heads, row.tp)
    rotary_dim = cfg.rope_dim
    max_position = 131072
    if len(row.tensors) >= 4 and len(row.tensors[3].dims) >= 2:
        try:
            max_position = int(row.tensors[3].dims[0])
            rotary_dim = int(row.tensors[3].dims[1])
        except (TypeError, ValueError):
            pass

    args = {
        "arg_type": "llm", "num_tokens": nt,
        "num_heads": nh, "num_kv_heads": nkv, "head_dim": cfg.head_dim,
        "rotary_dim": rotary_dim, "max_position": max_position,
        "eps": 1e-6, "dtype": "bfloat16",
    }
    dense_dim = (nh + 2 * nkv) * cfg.head_dim
    if qkv_dim > dense_dim:
        # [q|k|v|index_q|index_k] -> sparse layer
        n_idx = _per_rank(cfg.sparse_num_index_heads, row.tp)
        args.update({
            "variant": "sparse",
            "num_index_heads": n_idx,
            "index_head_dim": cfg.sparse_index_dim,
            "block_size": cfg.sparse_block_size,
            "cache_len": int(row.ctx_len) if row.ctx_len else 0,
        })
        note = "fused QK-norm + RoPE + KV/index-cache insert (sparse layer)"
    else:
        args["variant"] = "dense"
        note = "fused QK-norm + partial RoPE (dense layer)"
    return [EmittedCase("fused_qknorm_rope_kv", args, note=note)]


def adapt_attention(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """vllm::unified_attention_with_output -> flash_attention (paged, vllm FA)."""
    if not row.tensors:
        return []
    q = row.tensors[0]
    qh = int(q.dims[1]) if len(q.dims) >= 3 else _per_rank(cfg.num_heads, row.tp)
    hd = int(q.dims[2]) if len(q.dims) >= 3 else cfg.head_dim
    kvh = _per_rank(cfg.num_kv_heads, row.tp)
    if len(row.tensors) >= 2 and len(row.tensors[1].dims) >= 3:
        kvh = int(row.tensors[1].dims[1])
    b, ql, cl = _attn_sbc(row)
    return [EmittedCase("flash_attention", {
        "arg_type": "llm", "attn_mode": _mode(row),
        "q_head_num": qh, "kv_head_num": kvh, "head_dim": hd,
        "batch_size": b, "q_len": ql, "cache_len": cl,
        "block_size": cfg.sparse_block_size,
        "dtype": "bfloat16", "cache_dtype": "bfloat16", "dst_dtype": "bfloat16",
    }, note=f"dense paged attention {_mode(row)}")]


def adapt_kv_update(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """vllm::unified_kv_cache_update -> store_kv_cache (reshape_and_cache_flash)."""
    kvh = _per_rank(cfg.num_kv_heads, row.tp)
    hd = cfg.head_dim
    if row.tensors and len(row.tensors[0].dims) >= 3:
        kvh = int(row.tensors[0].dims[1])
        hd = int(row.tensors[0].dims[2])
    b, ql, cl = _attn_sbc(row)
    return [EmittedCase("store_kv_cache", {
        "arg_type": "llm", "attn_mode": _mode(row),
        "q_head_num": _per_rank(cfg.num_heads, row.tp),
        "kv_head_num": kvh, "head_dim": hd,
        "batch_size": b, "q_len": ql, "cache_len": cl,
        "block_size": cfg.sparse_block_size,
        "paged_cache_layout": "offset_major",
        "dtype": "bfloat16", "cache_dtype": "bfloat16",
    }, note="paged KV cache write")]


def adapt_silu_clamp(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    out = _dims2(row, 0)
    if out is None:
        return []
    nt, d = out
    return [EmittedCase("silu_and_mul_with_clamp", {
        "arg_type": "llm", "num_tokens": nt, "intermediate_size": d,
        "swiglu_limit": 7.0, "dtype": "bfloat16",
    }, note="clamped SwiGLU (exact _C op)")]


def adapt_topk_sigmoid(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    tw = _dims2(row, 0)
    if tw is None:
        return []
    nt, topk = tw
    E = cfg.num_experts
    if len(row.tensors) >= 4 and len(row.tensors[3].dims) >= 2:
        try:
            E = int(row.tensors[3].dims[1])
        except (TypeError, ValueError):
            pass
    return [EmittedCase("topk_sigmoid", {
        "arg_type": "llm", "num_tokens": nt, "num_experts": E, "topk": topk,
        "has_bias": len(row.tensors) >= 5, "renormalize": True,
        "dtype": "float32",
    }, note="sigmoid MoE gating (exact _moe_C op)")]


# ---------------------------------------------------------------------------
# CUDA MoE path: align -> fused_moe_kernel (x2) -> moe_sum
# ---------------------------------------------------------------------------
def adapt_moe_align(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    ti = _dims2(row, 0)  # topk_ids [num_tokens, topk]
    if ti is None:
        return []
    nt, topk = ti
    return [EmittedCase("moe_align_block_size", {
        "arg_type": "llm", "num_tokens": nt, "topk": topk,
        "num_experts": cfg.num_experts, "block_size_m": 64, "dtype": "int32",
    }, note="MoE token permute/pad (exact _moe_C op)")]


def adapt_moe_sum(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    if not row.tensors or len(row.tensors[0].dims) < 3:
        return []
    d = row.tensors[0].dims
    try:
        nt, topk, H = int(d[0]), int(d[1]), int(d[2])
    except (TypeError, ValueError):
        return []
    return [EmittedCase("moe_sum", {
        "arg_type": "llm", "num_tokens": nt, "topk": topk, "hidden_size": H,
        "dtype": "bfloat16",
    }, note="MoE combine (exact _moe_C op)")]


def adapt_fused_moe(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """triton::fused_moe_kernel -> fused_moe_gemm gemm1 + gemm2.

    The trace records no shapes for raw triton kernels, so the (per-rank) dims
    come from the row's sweep point plus the model config. vLLM launches this
    kernel twice per MoE layer (w1 then w2); both are emitted.
    """
    nt = _tokens(row, cfg)
    base = {
        "arg_type": "llm", "num_tokens": nt,
        "hidden_size": cfg.hidden_size,
        "moe_intermediate_size": _per_rank(cfg.moe_intermediate_size, row.tp),
        "num_experts": cfg.num_experts, "topk": cfg.num_experts_per_tok,
        "dtype": "bfloat16",
    }
    return [
        EmittedCase("fused_moe_gemm", dict(base, stage="gemm1"),
                    note="MoE expert GEMM1 (triton fused_moe_kernel)"),
        EmittedCase("fused_moe_gemm", dict(base, stage="gemm2"),
                    note="MoE expert GEMM2 (triton fused_moe_kernel)"),
    ]


# ---------------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------------
def adapt_gemma_rms_norm(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    t = _dims2(row, 0)
    M, H = t if t else (_tokens(row, cfg), cfg.hidden_size)
    return [EmittedCase("rms_norm", {
        "arg_type": "default", "batch_size": M, "dim_size": H,
        "dtype": "bfloat16",
    }, note="gemma rmsnorm (flashinfer)")]


def adapt_gemma_add_rms_norm(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    t = _dims2(row, 0)
    M, H = t if t else (_tokens(row, cfg), cfg.hidden_size)
    return [EmittedCase("add_rms_norm", {
        "arg_type": "llm", "num_tokens": M, "hidden_size": H,
        "dtype": "bfloat16",
    }, note="gemma fused add-rmsnorm (flashinfer)")]


# ---------------------------------------------------------------------------
# MSA (triton on CUDA)
#
# The Shape Matrix carries real shapes for these kernels since breakdown
# 25d4a34 ("Reconstruct shapes for MiniMax-M3 MSA/indexer kernels on CUDA"), so
# the head counts / head dims / top-k come from the trace rather than from the
# config. The sweep coordinate still supplies batch / q_len / cache_len (a
# kernel's operand shape does not carry the context length), which is why these
# ops stay in ``DENSE_SWEEP_OPS``.
# ---------------------------------------------------------------------------
def _dims3(row: OpRow, idx: int = 0) -> tuple[int, int, int] | None:
    """Concrete first three dims of the ``idx``-th traced tensor."""
    if len(row.tensors) <= idx:
        return None
    d = row.tensors[idx].dims
    if len(d) < 3:
        return None
    try:
        return int(d[0]), int(d[1]), int(d[2])
    except (TypeError, ValueError):
        return None


def _msa_common(row: OpRow, cfg: M3Config) -> dict:
    b, ql, cl = _attn_sbc(row)
    return {
        "arg_type": "llm", "attn_mode": _mode(row),
        "num_index_heads": _per_rank(cfg.sparse_num_index_heads, row.tp),
        "index_head_dim": cfg.sparse_index_dim,
        "block_size": cfg.sparse_block_size,
        "topk_blocks": cfg.sparse_topk_blocks,
        "batch_size": b, "q_len": ql, "cache_len": cl, "dtype": "bfloat16",
    }


def adapt_msa_index_score(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """indexer score kernels; idx_q is [tokens, n_idx/TP, index_head_dim]."""
    args = _msa_common(row, cfg)
    d = _dims3(row)
    if d:
        args["num_index_heads"], args["index_head_dim"] = d[1], d[2]
    return [EmittedCase("msa_index_score", args,
                        note="MSA lightning-indexer score (triton)")]


def adapt_msa_index_score_topk(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """decode top-k kernels: fused into ``minimax_m3_index_decode``, so they run
    as ``msa_index_score(decode)``, but their tensor is the top-k layout
    [n_idx/TP, tokens, K_topk] rather than idx_q."""
    args = _msa_common(row, cfg)
    d = _dims3(row)
    if d:
        args["num_index_heads"], args["topk_blocks"] = d[0], d[2]
    return [EmittedCase("msa_index_score", args,
                        note="MSA indexer score+top-k, decode (triton)")]


def adapt_msa_index_topk(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """top-k select kernels; the score/index tensor is [n_idx/TP, tokens, K]."""
    args = _msa_common(row, cfg)
    d = _dims3(row)
    if d:
        args["num_index_heads"], args["topk_blocks"] = d[0], d[2]
    return [EmittedCase("msa_index_topk", args,
                        note="MSA index top-k block select (triton)")]


def adapt_msa_sparse_attn(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    """sparse attend kernels; q is [tokens, n_h/TP, head_dim]."""
    args = _msa_common(row, cfg)
    d = _dims3(row)
    qh = d[1] if d else _per_rank(cfg.num_heads, row.tp)
    hd = d[2] if d else cfg.head_dim
    args.update({
        "q_head_num": qh,
        "kv_head_num": _per_rank(cfg.num_kv_heads, row.tp),
        "head_dim": hd,
    })
    return [EmittedCase("msa_sparse_attn", args,
                        note="MSA block-sparse attend (triton)")]


# ---------------------------------------------------------------------------
# collectives (NCCL, via vllm:: wrappers on CUDA)
# ---------------------------------------------------------------------------
def _collective(row: OpRow, cfg: M3Config, op: str) -> list[EmittedCase]:
    t = _dims2(row, 0)
    if t is None:
        return []
    nt, H = t
    return [EmittedCase(op, {
        "arg_type": "llm", "world_size": row.tp,
        "num_tokens": nt, "hidden_size": H, "dtype": _dtype_of(row),
    }, note=f"tp {op} (NCCL)")]


def adapt_allreduce(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _collective(row, cfg, "all_reduce")


def adapt_allgather(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _collective(row, cfg, "all_gather")


# ---------------------------------------------------------------------------
# aten leaf kernels (embedding / sampler / elementwise)
# ---------------------------------------------------------------------------
def adapt_embedding(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    w = _dims2(row, 0)  # weight [vocab/TP, hidden]
    if w is None:
        return []
    vocab, H = w
    nt = _tokens(row, cfg)
    if len(row.tensors) >= 2 and row.tensors[1].dims:
        try:
            nt = int(row.tensors[1].dims[0])
        except (TypeError, ValueError):
            pass
    return [EmittedCase("embedding", {
        "arg_type": "default", "src_batch_size": vocab,
        "dst_batch_size": nt, "dim_size": H, "dtype": _dtype_of(row),
    }, note="vocab-parallel embedding lookup")]


def _elementwise(row: OpRow, cfg: M3Config, op: str,
                 note: str) -> list[EmittedCase]:
    t = _dims2(row, 0)
    if t is None:
        return []
    M, N = t
    if M == 0 or N == 0:
        return []
    return [EmittedCase(op, {
        "arg_type": "default", "batch_size": M, "dim_size": N,
        "dtype": _dtype_of(row),
    }, note=note)]


def adapt_add(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _elementwise(row, cfg, "add", "residual add")


def adapt_div(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _elementwise(row, cfg, "div", "sampler temperature / renorm divide")


def adapt_masked_fill(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _elementwise(row, cfg, "masked_fill", "vocab-parallel mask")


def adapt_softmax(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _elementwise(row, cfg, "softmax", "sampler softmax")


def adapt_argmax(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _elementwise(row, cfg, "argmax", "greedy sample argmax")


def adapt_exponential(row: OpRow, cfg: M3Config) -> list[EmittedCase]:
    return _elementwise(row, cfg, "exponential", "sampler exponential noise")


ADAPTERS: dict[str, Callable[[OpRow, M3Config], list[EmittedCase]]] = {
    # dense compute
    "aten::linear": adapt_linear,
    "_C::fused_minimax_m3_qknorm_rope_kv_insert": adapt_qknorm_rope_kv,
    "vllm::unified_attention_with_output": adapt_attention,
    "vllm::unified_kv_cache_update": adapt_kv_update,
    "_C::silu_and_mul_with_clamp": adapt_silu_clamp,
    "_moe_C::topk_sigmoid": adapt_topk_sigmoid,
    # CUDA MoE
    "_moe_C::moe_align_block_size": adapt_moe_align,
    "_moe_C::moe_sum": adapt_moe_sum,
    "triton::fused_moe_kernel": adapt_fused_moe,
    # norms
    "flashinfer::gemma_rmsnorm": adapt_gemma_rms_norm,
    "flashinfer::gemma_fused_add_rmsnorm": adapt_gemma_add_rms_norm,
    # MSA (triton)
    "triton::_index_block_score_kernel": adapt_msa_index_score,
    "triton::_decode_index_score_kernel": adapt_msa_index_score,
    "triton::_topk_index_kernel": adapt_msa_index_topk,
    # decode top-k is fused into minimax_m3_index_decode (score+top-k in one
    # wrapper), which msa_index_score(decode) runs -- mapping these to
    # msa_index_topk would be a prefill-only op with no decode entry point.
    "triton::_topk_index_partial_kernel": adapt_msa_index_score_topk,
    "triton::_topk_index_merge_kernel": adapt_msa_index_score_topk,
    "triton::_gqa_sparse_fwd_kernel": adapt_msa_sparse_attn,
    "triton::_gqa_sparse_decode_kernel": adapt_msa_sparse_attn,
    "triton::_merge_topk_attn_out_kernel": adapt_msa_sparse_attn,
    # collectives
    "vllm::all_reduce": adapt_allreduce,
    "vllm::all_gather": adapt_allgather,
    # aten leaves
    "aten::embedding": adapt_embedding,
    "aten::add": adapt_add,
    "aten::div_": adapt_div,
    "aten::masked_fill_": adapt_masked_fill,
    "aten::softmax": adapt_softmax,
    "aten::argmax": adapt_argmax,
    "aten::exponential_": adapt_exponential,
}

COLLECTIVE_OPS = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}
MSA_OPS = {"msa_index_score", "msa_index_topk", "msa_sparse_attn"}

# Breakdown ops whose trace rows carry no operand shapes (raw triton kernels are
# recorded with an empty Shape column), so every sweep point looks identical to
# the shape de-duplicator and only one case per phase would survive. Their
# adapters derive the case entirely from the sweep coordinate
# (seq/ctx/batch via ``_attn_sbc``), so de-dup must keep the sweep point to
# produce the full dense sweep.
DENSE_SWEEP_OPS = {
    "triton::_index_block_score_kernel",
    "triton::_decode_index_score_kernel",
    "triton::_topk_index_kernel",
    "triton::_topk_index_partial_kernel",
    "triton::_topk_index_merge_kernel",
    "triton::_gqa_sparse_fwd_kernel",
    "triton::_gqa_sparse_decode_kernel",
    "triton::_merge_topk_attn_out_kernel",
}

# Emission-time shape gates - see the XPU map for the policy (only impossible
# shapes are dropped; kernel capability limits must be reported by the run).
CASE_CONSTRAINTS: dict[str, tuple] = {}
