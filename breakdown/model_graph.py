# SPDX-License-Identifier: Apache-2.0
"""Static model graph builder — derives op dispatch from model architecture.

Builds a hierarchical module tree with ops, shapes, memory, and FLOPs for
each submodule, WITHOUT requiring model weights or actual profiling. This
enables analysis of models that don't fit in current hardware.

Supports architecture families:
  - Llama-like (Llama, Mistral, Yi)
  - Qwen2/Qwen3
  - More can be added via ARCH_SPECS registry

Each architecture is defined declaratively as a tree of ModuleSpec → OpSpec.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ===================================================================
# Data structures
# ===================================================================


@dataclass
class OpNode:
    """A single operation in the model graph."""
    name: str                    # e.g. "aten::mm", "rms_norm"
    role: str                    # semantic role: "qkv_proj", "o_proj", etc.
    backend: str                 # likely backend dispatch
    input_shapes: list[list]     # symbolic shapes (using dim names)
    output_shape: list           # symbolic output shape
    memory_bytes: int = 0        # estimated memory traffic
    flops: int = 0              # estimated FLOPs
    phase: str = "both"          # "prefill", "decode", or "both"


@dataclass
class ModuleNode:
    """A module in the model hierarchy."""
    name: str                    # display name: "self_attn", "mlp", etc.
    path: str                    # full path: "model.layers.*.self_attn"
    module_type: str             # class name: "Qwen3Attention"
    ops: list[OpNode] = field(default_factory=list)
    children: list["ModuleNode"] = field(default_factory=list)
    repeat_count: int = 1        # ×N for repeated layers
    total_memory: int = 0        # aggregate memory (self + children)
    total_flops: int = 0         # aggregate FLOPs (self + children)

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict."""
        ai = (self.total_flops / self.total_memory) if self.total_memory > 0 else 0
        d: dict[str, Any] = {
            "name": self.name,
            "path": self.path,
            "module_type": self.module_type,
            "repeat_count": self.repeat_count,
            "total_memory": self.total_memory,
            "total_flops": self.total_flops,
            "total_ai": round(ai, 2),
            "ops": [
                {
                    "name": op.name,
                    "role": op.role,
                    "backend": op.backend,
                    "input_shapes": op.input_shapes,
                    "output_shape": op.output_shape,
                    "memory_bytes": op.memory_bytes,
                    "flops": op.flops,
                    "ai": round(op.flops / op.memory_bytes, 2) if op.memory_bytes > 0 else 0,
                    "phase": op.phase,
                }
                for op in self.ops
            ],
            "children": [c.to_dict() for c in self.children],
        }
        return d


# ===================================================================
# Shape & cost helpers
# ===================================================================

def _mm_mem(M: int, K: int, N: int, dtype_bytes: int = 2,
            weight_dtype_bytes: int | None = None) -> int:
    """Memory for matmul: read A[M,K] + B[K,N] + write C[M,N].

    For quantized models, weights (B matrix) use weight_dtype_bytes while
    activations (A matrix) and output (C matrix) use dtype_bytes.
    """
    w_bytes = weight_dtype_bytes if weight_dtype_bytes is not None else dtype_bytes
    return M * K * dtype_bytes + K * N * w_bytes + M * N * dtype_bytes


def _mm_flops(M: int, K: int, N: int) -> int:
    """FLOPs for matmul: 2*M*K*N."""
    return 2 * M * K * N


def _norm_mem(tokens: int, dim: int, dtype_bytes: int = 2) -> int:
    """Memory for norm: read input + weight + write output."""
    return tokens * dim * dtype_bytes * 3


def _norm_flops(tokens: int, dim: int) -> int:
    """FLOPs for RMSNorm: ~5 ops per element."""
    return tokens * dim * 5


def _activation_mem(tokens: int, dim: int, dtype_bytes: int = 2) -> int:
    """Memory for silu_and_mul: read 2*dim, write dim."""
    return tokens * dim * dtype_bytes * 3


def _activation_flops(tokens: int, dim: int) -> int:
    """FLOPs for SiLU+Mul: ~4 ops per element."""
    return tokens * dim * 4


def _softmax_mem(tokens: int, seq_len: int, n_heads: int,
                 dtype_bytes: int = 2) -> int:
    """Memory for attention softmax."""
    return tokens * seq_len * n_heads * dtype_bytes * 3


def _softmax_flops(tokens: int, seq_len: int, n_heads: int) -> int:
    """FLOPs for attention (QK + softmax + AV)."""
    return 2 * tokens * seq_len * n_heads * 2  # approximate


# ===================================================================
# Architecture specs (declarative)
# ===================================================================

def _build_attention_ops(
    cfg: dict, phase: str, tokens: str, seq: str
) -> list[OpNode]:
    """Build ops for attention module."""
    H = cfg["hidden_size"]
    n_h = cfg["num_heads"]
    n_kv = cfg["num_kv_heads"]
    d = cfg["head_dim"]
    qkv_size = (n_h + 2 * n_kv) * d
    dtype_bytes = cfg["dtype_bytes"]
    w_bytes = cfg.get("weight_dtype_bytes", dtype_bytes)
    T = cfg.get(f"_{tokens}", tokens)  # numeric if available
    S = cfg.get(f"_{seq}", seq)

    ops: list[OpNode] = []

    # QKV projection
    ops.append(OpNode(
        name="aten::mm", role="qkv_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "QKV"]],
        output_shape=[tokens, "QKV"],
        memory_bytes=_mm_mem(T, H, qkv_size, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, qkv_size) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # Q/K norms (Qwen3-specific)
    if cfg.get("has_qk_norm"):
        for role, sym in [("q_norm", "n_h"), ("k_norm", "n_kv")]:
            n = n_h if role == "q_norm" else n_kv
            ops.append(OpNode(
                name="rms_norm", role=role,
                backend="vllm-xpu-kernels",
                input_shapes=[[tokens, sym, "d"]],
                output_shape=[tokens, sym, "d"],
                memory_bytes=_norm_mem(T * n, d, dtype_bytes) if isinstance(T, int) else 0,
                flops=_norm_flops(T * n, d) if isinstance(T, int) else 0,
                phase=phase,
            ))

    # Rotary embedding
    ops.append(OpNode(
        name="rotary_embedding", role="rotary_emb",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "n_h", "d"], [tokens, "n_kv", "d"]],
        output_shape=[tokens, "n_h", "d"],
        memory_bytes=(T * (n_h + n_kv) * d * dtype_bytes * 3) if isinstance(T, int) else 0,
        flops=(T * (n_h + n_kv) * d * 6) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # Attention kernel
    attn_backend = "vllm-xpu-kernels"
    if phase == "prefill":
        ops.append(OpNode(
            name="flash_attn_varlen_fwd", role="attention",
            backend=attn_backend,
            input_shapes=[[tokens, "n_h", "d"], [seq, "n_kv", "d"], [seq, "n_kv", "d"]],
            output_shape=[tokens, "n_h", "d"],
            memory_bytes=0,  # complex to estimate
            flops=(2 * T * S * n_h * d) if isinstance(T, int) and isinstance(S, int) else 0,
            phase="prefill",
        ))
    else:
        ops.append(OpNode(
            name="paged_attention", role="attention",
            backend=attn_backend,
            input_shapes=[[tokens, "n_h", "d"], [seq, "n_kv", "d"]],
            output_shape=[tokens, "n_h", "d"],
            memory_bytes=0,
            flops=0,
            phase="decode",
        ))

    # KV cache store
    ops.append(OpNode(
        name="reshape_and_cache_flash", role="cache_store",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "n_kv", "d"], [tokens, "n_kv", "d"]],
        output_shape=[],
        memory_bytes=(T * n_kv * d * dtype_bytes * 2) if isinstance(T, int) else 0,
        flops=0,
        phase=phase,
    ))

    # Output projection
    ops.append(OpNode(
        name="aten::mm", role="o_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "n_h·d"], ["n_h·d", "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_mm_mem(T, n_h * d, H, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, n_h * d, H) if isinstance(T, int) else 0,
        phase=phase,
    ))

    return ops


def _build_mlp_ops(cfg: dict, phase: str, tokens: str) -> list[OpNode]:
    """Build ops for MLP module."""
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    dtype_bytes = cfg["dtype_bytes"]
    w_bytes = cfg.get("weight_dtype_bytes", dtype_bytes)
    T = cfg.get(f"_{tokens}", tokens)

    ops: list[OpNode] = []

    # Gate+Up fused projection
    ops.append(OpNode(
        name="aten::mm", role="gate_up_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "2·I"]],
        output_shape=[tokens, "2·I"],
        memory_bytes=_mm_mem(T, H, 2 * I, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, 2 * I) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # SiLU and Mul activation
    ops.append(OpNode(
        name="silu_and_mul", role="activation",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "2·I"]],
        output_shape=[tokens, "I"],
        memory_bytes=_activation_mem(T, I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_activation_flops(T, I) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # Down projection
    ops.append(OpNode(
        name="aten::mm", role="down_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "I"], ["I", "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_mm_mem(T, I, H, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, I, H) if isinstance(T, int) else 0,
        phase=phase,
    ))

    return ops


def _build_norm_op(cfg: dict, role: str, tokens: str, phase: str) -> OpNode:
    """Build RMSNorm op."""
    H = cfg["hidden_size"]
    dtype_bytes = cfg["dtype_bytes"]
    T = cfg.get(f"_{tokens}", tokens)

    fused = "fused_add_" if role.startswith("post") else ""
    return OpNode(
        name=f"{fused}rms_norm", role=role,
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_norm_mem(T, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_norm_flops(T, H) if isinstance(T, int) else 0,
        phase=phase,
    )


def _build_decoder_layer(
    cfg: dict, arch_family: str, phase: str, tokens: str, seq: str
) -> ModuleNode:
    """Build one decoder layer module."""
    attn_type = f"{arch_family}Attention"
    mlp_type = f"{arch_family}MLP"

    # Input LayerNorm
    input_norm = ModuleNode(
        name="input_layernorm", path="model.layers.*.input_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "input_layernorm", tokens, phase)],
    )

    # Attention
    attn_ops = _build_attention_ops(cfg, phase, tokens, seq)
    attention = ModuleNode(
        name="self_attn", path="model.layers.*.self_attn",
        module_type=attn_type, ops=attn_ops,
    )

    # Post-attention LayerNorm
    post_norm = ModuleNode(
        name="post_attention_layernorm",
        path="model.layers.*.post_attention_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "post_attention_layernorm", tokens, phase)],
    )

    # MLP
    mlp_ops = _build_mlp_ops(cfg, phase, tokens)
    mlp = ModuleNode(
        name="mlp", path="model.layers.*.mlp",
        module_type=mlp_type, ops=mlp_ops,
    )

    layer = ModuleNode(
        name="decoder_layer", path="model.layers.*",
        module_type=f"{arch_family}DecoderLayer",
        children=[input_norm, attention, post_norm, mlp],
        repeat_count=cfg["num_layers"],
    )
    return layer


def _build_moe_layer(
    cfg: dict, arch_family: str, phase: str, tokens: str, seq: str
) -> ModuleNode:
    """Build one MoE decoder layer (Mixtral/DeepSeek style)."""
    num_experts = cfg.get("num_experts", 8)
    top_k = cfg.get("num_experts_per_tok", 2)
    n_shared = cfg.get("n_shared_experts", 0)
    H = cfg["hidden_size"]
    # MoE may use a different intermediate size than dense layers
    moe_I = cfg.get("moe_intermediate_size") or cfg["intermediate_size"]
    I = cfg["intermediate_size"]
    dtype_bytes = cfg["dtype_bytes"]
    T = cfg.get(f"_{tokens}", tokens)

    # Input LayerNorm
    input_norm = ModuleNode(
        name="input_layernorm", path="model.layers.*.input_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "input_layernorm", tokens, phase)],
    )

    # Attention (same as dense)
    attn_ops = _build_attention_ops(cfg, phase, tokens, seq)
    attention = ModuleNode(
        name="self_attn", path="model.layers.*.self_attn",
        module_type=f"{arch_family}Attention", ops=attn_ops,
    )

    # Post-attention LayerNorm
    post_norm = ModuleNode(
        name="post_attention_layernorm",
        path="model.layers.*.post_attention_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "post_attention_layernorm", tokens, phase)],
    )

    # MoE block
    gate_op = OpNode(
        name="aten::mm", role="router_gate",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "E"]],
        output_shape=[tokens, "E"],
        memory_bytes=_mm_mem(T, H, num_experts, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, num_experts) if isinstance(T, int) else 0,
        phase=phase,
    )
    align_op = OpNode(
        name="moe_align_block_size", role="moe_routing",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "E"]],
        output_shape=[tokens],
        memory_bytes=0, flops=0, phase=phase,
    )
    # Expert computation (top_k experts active) — use moe_intermediate_size
    moe_I_sym = f"I_moe" if moe_I != I else "I"
    moe_2I_sym = f"2·I_moe" if moe_I != I else "2·I"
    expert_mm1 = OpNode(
        name="aten::mm", role="expert_gate_up",
        backend="torch-xpu-ops",
        input_shapes=[[f"{tokens}·K", "H"], ["H", moe_2I_sym]],
        output_shape=[f"{tokens}·K", moe_2I_sym],
        memory_bytes=_mm_mem(T * top_k, H, 2 * moe_I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T * top_k, H, 2 * moe_I) if isinstance(T, int) else 0,
        phase=phase,
    )
    expert_act = OpNode(
        name="silu_and_mul", role="expert_activation",
        backend="vllm-xpu-kernels",
        input_shapes=[[f"{tokens}·K", moe_2I_sym]],
        output_shape=[f"{tokens}·K", moe_I_sym],
        memory_bytes=_activation_mem(T * top_k, moe_I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_activation_flops(T * top_k, moe_I) if isinstance(T, int) else 0,
        phase=phase,
    )
    expert_mm2 = OpNode(
        name="aten::mm", role="expert_down",
        backend="torch-xpu-ops",
        input_shapes=[[f"{tokens}·K", moe_I_sym], [moe_I_sym, "H"]],
        output_shape=[f"{tokens}·K", "H"],
        memory_bytes=_mm_mem(T * top_k, moe_I, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T * top_k, moe_I, H) if isinstance(T, int) else 0,
        phase=phase,
    )

    moe_ops = [gate_op, align_op, expert_mm1, expert_act, expert_mm2]

    # Shared experts (DeepSeek-style)
    if n_shared > 0:
        shared_mm1 = OpNode(
            name="aten::mm", role="shared_expert_gate_up",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, "H"], ["H", moe_2I_sym]],
            output_shape=[tokens, moe_2I_sym],
            memory_bytes=_mm_mem(T, H, 2 * moe_I * n_shared, dtype_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, H, 2 * moe_I * n_shared) if isinstance(T, int) else 0,
            phase=phase,
        )
        shared_act = OpNode(
            name="silu_and_mul", role="shared_expert_activation",
            backend="vllm-xpu-kernels",
            input_shapes=[[tokens, moe_2I_sym]],
            output_shape=[tokens, moe_I_sym],
            memory_bytes=_activation_mem(T, moe_I * n_shared, dtype_bytes) if isinstance(T, int) else 0,
            flops=_activation_flops(T, moe_I * n_shared) if isinstance(T, int) else 0,
            phase=phase,
        )
        shared_mm2 = OpNode(
            name="aten::mm", role="shared_expert_down",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, moe_I_sym], [moe_I_sym, "H"]],
            output_shape=[tokens, "H"],
            memory_bytes=_mm_mem(T, moe_I * n_shared, H, dtype_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, moe_I * n_shared, H) if isinstance(T, int) else 0,
            phase=phase,
        )
        moe_ops.extend([shared_mm1, shared_act, shared_mm2])

    moe = ModuleNode(
        name="moe", path="model.layers.*.mlp",
        module_type=f"{arch_family}MoE",
        ops=moe_ops,
    )

    layer = ModuleNode(
        name="decoder_layer", path="model.layers.*",
        module_type=f"{arch_family}DecoderLayer",
        children=[input_norm, attention, post_norm, moe],
        repeat_count=cfg["num_layers"],
    )
    return layer


def _build_mla_attention_ops(
    cfg: dict, phase: str, tokens: str, seq: str
) -> list[OpNode]:
    """Build ops for Multi-head Latent Attention (DeepSeek-V2/V3).

    MLA compresses KV into a low-rank latent space, reducing KV cache size.
    Key ops: Q projection → latent KV projection → decompress → attention.
    """
    H = cfg["hidden_size"]
    n_h = cfg["num_heads"]
    n_kv = cfg["num_kv_heads"]
    d = cfg["head_dim"]
    dtype_bytes = cfg["dtype_bytes"]
    w_bytes = cfg.get("weight_dtype_bytes", dtype_bytes)
    T = cfg.get(f"_{tokens}", tokens)
    S = cfg.get(f"_{seq}", seq)

    # MLA-specific dimensions (from config or estimated)
    kv_lora_rank = cfg.get("kv_lora_rank", 512)
    q_lora_rank = cfg.get("q_lora_rank", 0)
    qk_nope_head_dim = cfg.get("qk_nope_head_dim", d // 2)
    qk_rope_head_dim = cfg.get("qk_rope_head_dim", d - qk_nope_head_dim)

    ops: list[OpNode] = []

    # Q projection (may go through LoRA compression)
    if q_lora_rank > 0:
        # Compressed Q: H → q_lora_rank → n_h * d
        ops.append(OpNode(
            name="aten::mm", role="q_compress",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, "H"], ["H", "Q_r"]],
            output_shape=[tokens, "Q_r"],
            memory_bytes=_mm_mem(T, H, q_lora_rank, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, H, q_lora_rank) if isinstance(T, int) else 0,
            phase=phase,
        ))
        ops.append(OpNode(
            name="rms_norm", role="q_norm",
            backend="vllm-xpu-kernels",
            input_shapes=[[tokens, "Q_r"]],
            output_shape=[tokens, "Q_r"],
            memory_bytes=_norm_mem(T, q_lora_rank, dtype_bytes) if isinstance(T, int) else 0,
            flops=_norm_flops(T, q_lora_rank) if isinstance(T, int) else 0,
            phase=phase,
        ))
        out_q = n_h * (qk_nope_head_dim + qk_rope_head_dim)
        ops.append(OpNode(
            name="aten::mm", role="q_decompress",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, "Q_r"],
                          ["Q_r", "n_h·D_qh"]],
            output_shape=[tokens, "n_h·D_qh"],
            memory_bytes=_mm_mem(T, q_lora_rank, out_q, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, q_lora_rank, out_q) if isinstance(T, int) else 0,
            phase=phase,
        ))
    else:
        # Standard Q projection
        qkv_size = (n_h + 2 * n_kv) * d
        ops.append(OpNode(
            name="aten::mm", role="qkv_proj",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, "H"], ["H", "QKV"]],
            output_shape=[tokens, "QKV"],
            memory_bytes=_mm_mem(T, H, qkv_size, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, H, qkv_size) if isinstance(T, int) else 0,
            phase=phase,
        ))

    # KV compression: H → kv_lora_rank (latent)
    ops.append(OpNode(
        name="aten::mm", role="kv_compress",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "KV_r"]],
        output_shape=[tokens, "KV_r"],
        memory_bytes=_mm_mem(T, H, kv_lora_rank, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, kv_lora_rank) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # KV norm
    ops.append(OpNode(
        name="rms_norm", role="kv_norm",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "KV_r"]],
        output_shape=[tokens, "KV_r"],
        memory_bytes=_norm_mem(T, kv_lora_rank, dtype_bytes) if isinstance(T, int) else 0,
        flops=_norm_flops(T, kv_lora_rank) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # RoPE on the rope portion
    ops.append(OpNode(
        name="deepseek_scaling_rope", role="rotary_emb",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "n_h", "D_rope"]],
        output_shape=[tokens, "n_h", "D_rope"],
        memory_bytes=(T * n_h * qk_rope_head_dim * dtype_bytes * 3) if isinstance(T, int) else 0,
        flops=(T * n_h * qk_rope_head_dim * 6) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # MLA attention kernel
    ops.append(OpNode(
        name="gdn_attention", role="attention",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "n_h", "d"], [seq, "KV_r"]],
        output_shape=[tokens, "n_h", "d"],
        memory_bytes=0,
        flops=(2 * T * S * n_h * d) if isinstance(T, int) and isinstance(S, int) else 0,
        phase=phase,
    ))

    # KV cache (compressed — stores latent, not full K/V)
    ops.append(OpNode(
        name="concat_and_cache_mla", role="cache_store",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "KV_r"]],
        output_shape=[],
        memory_bytes=(T * kv_lora_rank * dtype_bytes) if isinstance(T, int) else 0,
        flops=0,
        phase=phase,
    ))

    # Output projection
    ops.append(OpNode(
        name="aten::mm", role="o_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "n_h·d"], ["n_h·d", "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_mm_mem(T, n_h * d, H, dtype_bytes, w_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, n_h * d, H) if isinstance(T, int) else 0,
        phase=phase,
    ))

    return ops


def _build_mla_decoder_layer(
    cfg: dict, arch_family: str, phase: str, tokens: str, seq: str
) -> ModuleNode:
    """Build one decoder layer with MLA attention (DeepSeek-V2/V3)."""
    input_norm = ModuleNode(
        name="input_layernorm", path="model.layers.*.input_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "input_layernorm", tokens, phase)],
    )

    attn_ops = _build_mla_attention_ops(cfg, phase, tokens, seq)
    attention = ModuleNode(
        name="self_attn", path="model.layers.*.self_attn",
        module_type=f"{arch_family}MLAAttention", ops=attn_ops,
    )

    post_norm = ModuleNode(
        name="post_attention_layernorm",
        path="model.layers.*.post_attention_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "post_attention_layernorm", tokens, phase)],
    )

    mlp_ops = _build_mlp_ops(cfg, phase, tokens)
    mlp = ModuleNode(
        name="mlp", path="model.layers.*.mlp",
        module_type=f"{arch_family}MLP", ops=mlp_ops,
    )

    layer = ModuleNode(
        name="decoder_layer", path="model.layers.*",
        module_type=f"{arch_family}DecoderLayer",
        children=[input_norm, attention, post_norm, mlp],
        repeat_count=cfg["num_layers"],
    )
    return layer


def _build_mla_moe_layer(
    cfg: dict, arch_family: str, phase: str, tokens: str, seq: str
) -> ModuleNode:
    """Build one MoE decoder layer with MLA attention (DeepSeek-V2/V3)."""
    num_experts = cfg.get("num_experts", 8)
    top_k = cfg.get("num_experts_per_tok", 2)
    n_shared = cfg.get("n_shared_experts", 0)
    H = cfg["hidden_size"]
    moe_I = cfg.get("moe_intermediate_size") or cfg["intermediate_size"]
    I = cfg["intermediate_size"]
    dtype_bytes = cfg["dtype_bytes"]
    T = cfg.get(f"_{tokens}", tokens)

    input_norm = ModuleNode(
        name="input_layernorm", path="model.layers.*.input_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "input_layernorm", tokens, phase)],
    )

    attn_ops = _build_mla_attention_ops(cfg, phase, tokens, seq)
    attention = ModuleNode(
        name="self_attn", path="model.layers.*.self_attn",
        module_type=f"{arch_family}MLAAttention", ops=attn_ops,
    )

    post_norm = ModuleNode(
        name="post_attention_layernorm",
        path="model.layers.*.post_attention_layernorm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "post_attention_layernorm", tokens, phase)],
    )

    # MoE block (same structure as standard MoE)
    moe_I_sym = "I_moe" if moe_I != I else "I"
    moe_2I_sym = "2·I_moe" if moe_I != I else "2·I"

    gate_op = OpNode(
        name="aten::mm", role="router_gate",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "E"]],
        output_shape=[tokens, "E"],
        memory_bytes=_mm_mem(T, H, num_experts, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, num_experts) if isinstance(T, int) else 0,
        phase=phase,
    )
    align_op = OpNode(
        name="moe_align_block_size", role="moe_routing",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, "E"]],
        output_shape=[tokens],
        memory_bytes=0, flops=0, phase=phase,
    )
    expert_mm1 = OpNode(
        name="aten::mm", role="expert_gate_up",
        backend="torch-xpu-ops",
        input_shapes=[[f"{tokens}·K", "H"], ["H", moe_2I_sym]],
        output_shape=[f"{tokens}·K", moe_2I_sym],
        memory_bytes=_mm_mem(T * top_k, H, 2 * moe_I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T * top_k, H, 2 * moe_I) if isinstance(T, int) else 0,
        phase=phase,
    )
    expert_act = OpNode(
        name="silu_and_mul", role="expert_activation",
        backend="vllm-xpu-kernels",
        input_shapes=[[f"{tokens}·K", moe_2I_sym]],
        output_shape=[f"{tokens}·K", moe_I_sym],
        memory_bytes=_activation_mem(T * top_k, moe_I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_activation_flops(T * top_k, moe_I) if isinstance(T, int) else 0,
        phase=phase,
    )
    expert_mm2 = OpNode(
        name="aten::mm", role="expert_down",
        backend="torch-xpu-ops",
        input_shapes=[[f"{tokens}·K", moe_I_sym], [moe_I_sym, "H"]],
        output_shape=[f"{tokens}·K", "H"],
        memory_bytes=_mm_mem(T * top_k, moe_I, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T * top_k, moe_I, H) if isinstance(T, int) else 0,
        phase=phase,
    )

    moe_ops = [gate_op, align_op, expert_mm1, expert_act, expert_mm2]

    # Shared experts
    if n_shared > 0:
        shared_mm1 = OpNode(
            name="aten::mm", role="shared_expert_gate_up",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, "H"], ["H", moe_2I_sym]],
            output_shape=[tokens, moe_2I_sym],
            memory_bytes=_mm_mem(T, H, 2 * moe_I * n_shared, dtype_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, H, 2 * moe_I * n_shared) if isinstance(T, int) else 0,
            phase=phase,
        )
        shared_act = OpNode(
            name="silu_and_mul", role="shared_expert_activation",
            backend="vllm-xpu-kernels",
            input_shapes=[[tokens, moe_2I_sym]],
            output_shape=[tokens, moe_I_sym],
            memory_bytes=_activation_mem(T, moe_I * n_shared, dtype_bytes) if isinstance(T, int) else 0,
            flops=_activation_flops(T, moe_I * n_shared) if isinstance(T, int) else 0,
            phase=phase,
        )
        shared_mm2 = OpNode(
            name="aten::mm", role="shared_expert_down",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, moe_I_sym], [moe_I_sym, "H"]],
            output_shape=[tokens, "H"],
            memory_bytes=_mm_mem(T, moe_I * n_shared, H, dtype_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, moe_I * n_shared, H) if isinstance(T, int) else 0,
            phase=phase,
        )
        moe_ops.extend([shared_mm1, shared_act, shared_mm2])

    moe = ModuleNode(
        name="moe", path="model.layers.*.mlp",
        module_type=f"{arch_family}MoE", ops=moe_ops,
    )

    layer = ModuleNode(
        name="decoder_layer", path="model.layers.*",
        module_type=f"{arch_family}DecoderLayer",
        children=[input_norm, attention, post_norm, moe],
        repeat_count=cfg["num_layers"],
    )
    return layer


def _build_vision_encoder(
    cfg: dict, phase: str, tokens: str
) -> ModuleNode:
    """Build ops for a Vision Transformer (ViT) encoder.

    Used by VL models (Qwen2.5-VL, InternVL, etc.) to encode image patches.
    """
    # Vision encoder dimensions (from config or defaults)
    vit_hidden = cfg.get("vit_hidden_size", 1024)
    vit_layers = cfg.get("vit_num_layers", 24)
    vit_heads = cfg.get("vit_num_heads", 16)
    vit_inter = cfg.get("vit_intermediate_size", vit_hidden * 4)
    patch_size = cfg.get("patch_size", 14)
    image_size = cfg.get("image_size", 448)
    dtype_bytes = cfg["dtype_bytes"]

    num_patches = (image_size // patch_size) ** 2
    T = num_patches

    # Patch embedding (conv2d → flatten)
    patch_embed = OpNode(
        name="aten::conv2d", role="patch_embed",
        backend="torch-xpu-ops",
        input_shapes=[["N_img", "3", str(image_size), str(image_size)]],
        output_shape=["N_img", str(num_patches), str(vit_hidden)],
        memory_bytes=(3 * patch_size * patch_size * vit_hidden * dtype_bytes),
        flops=(num_patches * 3 * patch_size * patch_size * vit_hidden * 2),
        phase=phase,
    )

    # Self-attention per ViT layer
    vit_d = vit_hidden // vit_heads
    qkv_proj = OpNode(
        name="aten::mm", role="vit_qkv_proj",
        backend="torch-xpu-ops",
        input_shapes=[[str(T), str(vit_hidden)],
                      [str(vit_hidden), str(3 * vit_hidden)]],
        output_shape=[str(T), str(3 * vit_hidden)],
        memory_bytes=_mm_mem(T, vit_hidden, 3 * vit_hidden, dtype_bytes),
        flops=_mm_flops(T, vit_hidden, 3 * vit_hidden),
        phase=phase,
    )
    vit_attn = OpNode(
        name="aten::scaled_dot_product_attention", role="vit_attention",
        backend="torch-xpu-ops",
        input_shapes=[[str(vit_heads), str(T), str(vit_d)],
                      [str(vit_heads), str(T), str(vit_d)],
                      [str(vit_heads), str(T), str(vit_d)]],
        output_shape=[str(vit_heads), str(T), str(vit_d)],
        memory_bytes=(T * T * vit_heads * dtype_bytes * 3),
        flops=(2 * T * T * vit_heads * vit_d),
        phase=phase,
    )
    vit_o_proj = OpNode(
        name="aten::mm", role="vit_o_proj",
        backend="torch-xpu-ops",
        input_shapes=[[str(T), str(vit_hidden)],
                      [str(vit_hidden), str(vit_hidden)]],
        output_shape=[str(T), str(vit_hidden)],
        memory_bytes=_mm_mem(T, vit_hidden, vit_hidden, dtype_bytes),
        flops=_mm_flops(T, vit_hidden, vit_hidden),
        phase=phase,
    )

    # MLP per ViT layer
    vit_mlp_up = OpNode(
        name="aten::mm", role="vit_mlp_up",
        backend="torch-xpu-ops",
        input_shapes=[[str(T), str(vit_hidden)],
                      [str(vit_hidden), str(vit_inter)]],
        output_shape=[str(T), str(vit_inter)],
        memory_bytes=_mm_mem(T, vit_hidden, vit_inter, dtype_bytes),
        flops=_mm_flops(T, vit_hidden, vit_inter),
        phase=phase,
    )
    vit_act = OpNode(
        name="aten::gelu", role="vit_activation",
        backend="torch-xpu-ops",
        input_shapes=[[str(T), str(vit_inter)]],
        output_shape=[str(T), str(vit_inter)],
        memory_bytes=(T * vit_inter * dtype_bytes * 2),
        flops=(T * vit_inter * 4),
        phase=phase,
    )
    vit_mlp_down = OpNode(
        name="aten::mm", role="vit_mlp_down",
        backend="torch-xpu-ops",
        input_shapes=[[str(T), str(vit_inter)],
                      [str(vit_inter), str(vit_hidden)]],
        output_shape=[str(T), str(vit_hidden)],
        memory_bytes=_mm_mem(T, vit_inter, vit_hidden, dtype_bytes),
        flops=_mm_flops(T, vit_inter, vit_hidden),
        phase=phase,
    )

    vit_layer = ModuleNode(
        name="vit_layer", path="visual.encoder.layers.*",
        module_type="ViTLayer",
        ops=[qkv_proj, vit_attn, vit_o_proj, vit_mlp_up, vit_act, vit_mlp_down],
        repeat_count=vit_layers,
    )

    vit_encoder = ModuleNode(
        name="visual_encoder", path="visual",
        module_type="VisionEncoder",
        ops=[patch_embed],
        children=[vit_layer],
    )
    return vit_encoder


def _build_vl_projector(
    cfg: dict, phase: str, tokens: str
) -> ModuleNode:
    """Build the visual-to-language projector (bridge between ViT and LLM)."""
    vit_hidden = cfg.get("vit_hidden_size", 1024)
    H = cfg["hidden_size"]
    dtype_bytes = cfg["dtype_bytes"]

    num_patches = cfg.get("_num_patches", 1024)

    proj_op = OpNode(
        name="aten::mm", role="vl_projector",
        backend="torch-xpu-ops",
        input_shapes=[[str(num_patches), str(vit_hidden)],
                      [str(vit_hidden), "H"]],
        output_shape=[str(num_patches), "H"],
        memory_bytes=_mm_mem(num_patches, vit_hidden, H, dtype_bytes),
        flops=_mm_flops(num_patches, vit_hidden, H),
        phase=phase,
    )

    return ModuleNode(
        name="projector", path="visual.projector",
        module_type="VLProjector",
        ops=[proj_op],
    )


def _build_encoder_model(
    cfg: dict, phase: str, tokens: str
) -> ModuleNode:
    """Build encoder-only model graph (BERT/RoBERTa for embedding/reranking).

    Encoder models process all tokens in parallel (no causal mask).
    """
    H = cfg["hidden_size"]
    n_h = cfg["num_heads"]
    d = cfg.get("head_dim", H // n_h)
    I = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    num_layers = cfg["num_layers"]
    dtype_bytes = cfg["dtype_bytes"]
    T = cfg.get(f"_{tokens}", tokens)

    # Token embedding + position embedding
    embed = ModuleNode(
        name="embeddings", path="model.embeddings",
        module_type="Embeddings",
        ops=[
            OpNode(
                name="aten::embedding", role="word_embedding",
                backend="torch-xpu-ops",
                input_shapes=[["V", "H"], [tokens]],
                output_shape=[tokens, "H"],
                memory_bytes=(T * H * dtype_bytes) if isinstance(T, int) else 0,
                flops=0, phase=phase,
            ),
            OpNode(
                name="aten::embedding", role="position_embedding",
                backend="torch-xpu-ops",
                input_shapes=[[str(512), "H"], [tokens]],
                output_shape=[tokens, "H"],
                memory_bytes=(T * H * dtype_bytes) if isinstance(T, int) else 0,
                flops=0, phase=phase,
            ),
            OpNode(
                name="aten::layer_norm", role="embed_norm",
                backend="torch-xpu-ops",
                input_shapes=[[tokens, "H"]],
                output_shape=[tokens, "H"],
                memory_bytes=_norm_mem(T, H, dtype_bytes) if isinstance(T, int) else 0,
                flops=_norm_flops(T, H) if isinstance(T, int) else 0,
                phase=phase,
            ),
        ],
    )

    # Encoder layers (bidirectional self-attention + MLP)
    qkv_proj = OpNode(
        name="aten::mm", role="qkv_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", str(3 * H)]],
        output_shape=[tokens, str(3 * H)],
        memory_bytes=_mm_mem(T, H, 3 * H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, 3 * H) if isinstance(T, int) else 0,
        phase=phase,
    )
    attention = OpNode(
        name="aten::scaled_dot_product_attention", role="attention",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "n_h", "d"], [tokens, "n_h", "d"]],
        output_shape=[tokens, "n_h", "d"],
        memory_bytes=0, flops=0, phase=phase,
    )
    o_proj = OpNode(
        name="aten::mm", role="o_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_mm_mem(T, H, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, H) if isinstance(T, int) else 0,
        phase=phase,
    )
    attn_norm = OpNode(
        name="aten::layer_norm", role="attn_norm",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_norm_mem(T, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_norm_flops(T, H) if isinstance(T, int) else 0,
        phase=phase,
    )
    mlp_up = OpNode(
        name="aten::mm", role="mlp_up",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "I"]],
        output_shape=[tokens, "I"],
        memory_bytes=_mm_mem(T, H, I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, I) if isinstance(T, int) else 0,
        phase=phase,
    )
    mlp_act = OpNode(
        name="aten::gelu", role="activation",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "I"]],
        output_shape=[tokens, "I"],
        memory_bytes=_activation_mem(T, I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_activation_flops(T, I) if isinstance(T, int) else 0,
        phase=phase,
    )
    mlp_down = OpNode(
        name="aten::mm", role="mlp_down",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "I"], ["I", "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_mm_mem(T, I, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, I, H) if isinstance(T, int) else 0,
        phase=phase,
    )
    mlp_norm = OpNode(
        name="aten::layer_norm", role="mlp_norm",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"]],
        output_shape=[tokens, "H"],
        memory_bytes=_norm_mem(T, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_norm_flops(T, H) if isinstance(T, int) else 0,
        phase=phase,
    )

    encoder_layer = ModuleNode(
        name="encoder_layer", path="model.encoder.layers.*",
        module_type="TransformerEncoderLayer",
        ops=[qkv_proj, attention, o_proj, attn_norm,
             mlp_up, mlp_act, mlp_down, mlp_norm],
        repeat_count=num_layers,
    )

    # Pooling (mean or CLS)
    pool = ModuleNode(
        name="pooler", path="model.pooler",
        module_type="Pooler",
        ops=[OpNode(
            name="aten::mean", role="pooling",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, "H"]],
            output_shape=["1", "H"],
            memory_bytes=(T * H * dtype_bytes * 2) if isinstance(T, int) else 0,
            flops=(T * H) if isinstance(T, int) else 0,
            phase=phase,
        )],
    )

    root = ModuleNode(
        name="model", path="",
        module_type="EncoderModel",
        children=[embed, encoder_layer, pool],
    )
    return root


# ===================================================================
# Public API
# ===================================================================

# Architecture family detection from HuggingFace architecture name
_ARCH_FAMILY_MAP: dict[str, str] = {
    # --- Llama family (GQA, RoPE, SwiGLU) ---
    "LlamaForCausalLM": "Llama",
    "MistralForCausalLM": "Llama",
    "YiForCausalLM": "Llama",
    "InternLM2ForCausalLM": "Llama",
    "MiMoForCausalLM": "Llama",       # Xiaomi MiMo uses Llama architecture
    "Kimi2ForCausalLM": "Llama",       # Kimi-K2 uses Llama-like architecture
    "StepForCausalLM": "Llama",        # Step models use Llama-like architecture
    # --- Qwen family ---
    "Qwen2ForCausalLM": "Qwen2",
    "Qwen3ForCausalLM": "Qwen3",
    "Qwen3MoeForCausalLM": "Qwen3Moe",
    # --- GLM family (ChatGLM / GLM-4 / GLM-5) ---
    "ChatGLMForConditionalGeneration": "GLM4",
    "ChatGLMModel": "GLM4",
    "Glm4ForCausalLM": "GLM4",
    # --- DeepSeek MLA family ---
    "DeepseekV2ForCausalLM": "DeepSeekV2",
    "DeepseekV3ForCausalLM": "DeepSeekV3",
    # --- MoE models ---
    "MixtralForCausalLM": "Mixtral",
    "HunYuanMoEV1ForCausalLM": "Hunyuan",
    "MiniMaxM1ForCausalLM": "MiniMax",
    # --- Vision-Language (VL) models ---
    "Qwen2_5_VLForConditionalGeneration": "Qwen2VL",
    "Qwen2VLForConditionalGeneration": "Qwen2VL",
    "Qwen3VLForConditionalGeneration": "Qwen3VL",
    "Qwen3OmniForConditionalGeneration": "Qwen3VL",
    "InternVLChatModel": "InternVL",
    # --- Embedding / Reranker ---
    "XLMRobertaModel": "RoBERTa",
    "XLMRobertaForSequenceClassification": "RoBERTa",
    "BertModel": "BERT",
    "BertForSequenceClassification": "BERT",
    "Qwen3ForSequenceClassification": "Qwen3",
    # --- Diffusion (static analysis only — not vLLM-served) ---
    "FluxTransformer2DModel": "Flux",
    "HunyuanDiT2DModel": "HunyuanDiT",
    "WanTransformer3DModel": "Wan",
    "LTXVideoTransformer3DModel": "LTX",
    "CogVideoXTransformer3DModel": "CogVideoX",
    "UNetSpatioTemporalConditionModel": "SVD",
    "HunyuanVideoTransformer3DModel": "HunyuanVideo",
}

# Architectures that have QK normalization
_HAS_QK_NORM = {"Qwen3", "Qwen3Moe", "DeepSeekV2", "DeepSeekV3", "GLM4"}

# Architectures that use Multi-head Latent Attention (MLA)
_MLA_ARCHS = {"DeepSeekV2", "DeepSeekV3"}

# Vision-language architecture families
_VL_ARCHS = {"Qwen2VL", "Qwen3VL", "InternVL"}

# Encoder-only architecture families
_ENCODER_ARCHS = {"RoBERTa", "BERT"}

# Diffusion architecture families (non-vLLM)
_DIFFUSION_ARCHS = {"Flux", "HunyuanDiT", "Wan", "LTX", "CogVideoX",
                     "SVD", "HunyuanVideo"}


def _dtype_bytes(dtype: str) -> int:
    """Get bytes per element for dtype."""
    mapping = {
        "float32": 4, "float16": 2, "bfloat16": 2,
        "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1, "int4": 1,
    }
    # Strip torch. prefix
    dt = dtype.replace("torch.", "")
    return mapping.get(dt, 2)


def _quant_weight_bytes(quant_method: str) -> int:
    """Get effective bytes per weight element for a quantization method.

    Weight-only quantization reduces weight storage while keeping activations
    at full precision. This returns the per-element byte cost of stored weights.
    """
    # Map quantization method → effective weight bytes
    quant_bytes = {
        "fp8": 1,              # FP8 (E4M3 or E5M2)
        "gptq": 1,            # GPTQ typically 4-bit packed (0.5B), but with scales ~1B effective
        "gptq_marlin": 1,     # GPTQ with Marlin kernel (4-bit)
        "awq": 1,             # AWQ 4-bit, similar to GPTQ
        "awq_marlin": 1,      # AWQ with Marlin kernel
        "marlin": 1,          # Marlin 4-bit
        "squeezellm": 1,      # SqueezeLLM 4-bit
        "bitsandbytes": 1,    # BitsAndBytes (NF4/INT8)
        "gguf": 1,            # GGUF various quant levels
        "int8": 1,            # INT8 weight-only
        "int4": 1,            # INT4 weight-only (0.5B per element, but metadata overhead → ~1B)
    }
    return quant_bytes.get(quant_method.lower(), 2)


def _bytes_to_dtype_name(nbytes: int, quant_method: str | None = None) -> str:
    """Convert byte count (and optional quant method) to a display dtype name."""
    if quant_method:
        return quant_method.upper()
    return {4: "fp32", 2: "bf16", 1: "fp8"}.get(nbytes, f"{nbytes*8}bit")


def min_profile_layers(model_summary: dict) -> int:
    """Compute minimum layers to profile to capture all unique layer types.

    For pure dense or pure MoE models: 1 layer suffices.
    For hybrid models (e.g. DeepSeek with first_k_dense_replace=3):
      need first_k_dense + 1 to capture both dense and MoE layers.
    """
    first_k = model_summary.get("first_k_dense_replace", 0) or 0
    if first_k > 0 and model_summary.get("is_moe"):
        return first_k + 1
    return 1


def build_model_graph(
    model_summary: dict,
    prefill_len: int | None = None,
    decode_batch: int | None = None,
    context_len: int | None = None,
    tp_size: int = 1,
    quantization: str | None = None,
) -> dict:
    """Build static model graph from model summary (from model_info.py).

    Args:
        model_summary: output of summarize_config() — contains architecture,
                       hidden_size, num_layers, num_heads, etc.
        prefill_len: prefill sequence length (query length for prefill phase)
        decode_batch: decode batch size
        context_len: KV cache length for decode phase attention
        tp_size: tensor parallel size (splits heads, intermediate, vocab)
        quantization: quantization method (e.g. "fp8", "gptq", "awq").
            Affects weight dtype_bytes and adds dequant ops to the graph.

    Returns:
        Dict with "prefill" and "decode" trees, plus metadata.
    """
    arch = model_summary.get("architecture", "")
    family = _ARCH_FAMILY_MAP.get(arch, "Unknown")
    is_moe = model_summary.get("is_moe", False)
    is_vl = family in _VL_ARCHS
    is_encoder = family in _ENCODER_ARCHS
    is_mla = family in _MLA_ARCHS
    is_diffusion = family in _DIFFUSION_ARCHS

    # Encoder-only models have a different graph structure
    if is_encoder:
        return _build_encoder_graph(model_summary, family, prefill_len, tp_size)

    # Diffusion models: return a placeholder noting static analysis is limited
    if is_diffusion:
        return _build_diffusion_placeholder(model_summary, family)

    # Full (un-split) dimensions for reference
    full_num_heads = model_summary.get("num_heads") or 1
    full_num_kv = model_summary.get("num_kv_heads", full_num_heads)
    full_intermediate = model_summary.get("intermediate_size") or 1
    full_vocab = model_summary.get("vocab_size") or 1

    # TP-split dimensions (column-parallel splits output, row-parallel splits input)
    tp_num_heads = full_num_heads // tp_size
    tp_num_kv = full_num_kv // tp_size
    tp_intermediate = full_intermediate // tp_size
    tp_vocab = full_vocab // tp_size

    # Build config dict — uses TP-adjusted dims for per-rank shapes
    cfg: dict[str, Any] = {
        "hidden_size": model_summary["hidden_size"],
        "num_layers": model_summary["num_layers"],
        "num_heads": tp_num_heads,
        "num_kv_heads": tp_num_kv,
        "head_dim": model_summary.get("head_dim", model_summary["hidden_size"] // full_num_heads),
        "intermediate_size": tp_intermediate,
        "vocab_size": tp_vocab,
        "dtype_bytes": _dtype_bytes(model_summary.get("dtype", "bfloat16")),
        "has_qk_norm": family in _HAS_QK_NORM,
        "tp_size": tp_size,
        "is_mla": is_mla,
        "is_vl": is_vl,
    }

    # Quantization: adjust weight dtype and track quant method
    # Activations stay at model dtype; weights use reduced precision
    # "none" means explicitly no quantization (overrides model config)
    if quantization and quantization.lower() == "none":
        quant = None
    else:
        quant = quantization or model_summary.get("quant_method")
    if quant:
        cfg["quantization"] = quant
        cfg["weight_dtype_bytes"] = _quant_weight_bytes(quant)
    else:
        cfg["quantization"] = None
        cfg["weight_dtype_bytes"] = cfg["dtype_bytes"]

    if is_moe:
        cfg["num_experts"] = model_summary.get("num_experts", 8)
        cfg["num_experts_per_tok"] = model_summary.get("num_experts_per_tok", 2)
        raw_moe_I = model_summary.get("moe_intermediate_size")
        cfg["moe_intermediate_size"] = (raw_moe_I // tp_size) if raw_moe_I else None
        cfg["n_shared_experts"] = model_summary.get("n_shared_experts", 0)

    # MLA-specific config (DeepSeek-V2/V3)
    if is_mla:
        cfg["kv_lora_rank"] = model_summary.get("kv_lora_rank", 512)
        cfg["q_lora_rank"] = model_summary.get("q_lora_rank", 0)
        cfg["qk_nope_head_dim"] = model_summary.get("qk_nope_head_dim", 128)
        cfg["qk_rope_head_dim"] = model_summary.get("qk_rope_head_dim", 64)

    # VL-specific config
    if is_vl:
        cfg["vit_hidden_size"] = model_summary.get("vit_hidden_size", 1024)
        cfg["vit_num_layers"] = model_summary.get("vit_num_layers", 24)
        cfg["vit_num_heads"] = model_summary.get("vit_num_heads", 16)
        cfg["vit_intermediate_size"] = model_summary.get(
            "vit_intermediate_size",
            cfg["vit_hidden_size"] * 4,
        )
        cfg["patch_size"] = model_summary.get("patch_size", 14)
        cfg["image_size"] = model_summary.get("image_size", 448)
        num_patches = (cfg["image_size"] // cfg["patch_size"]) ** 2
        cfg["_num_patches"] = num_patches

    # Hybrid dense/MoE detection
    first_k_dense = model_summary.get("first_k_dense_replace", 0) or 0
    cfg["first_k_dense_replace"] = first_k_dense

    # Set numeric token counts if specified
    if prefill_len:
        cfg["_S"] = prefill_len
    if decode_batch:
        cfg["_B"] = decode_batch
    if prefill_len and decode_batch:
        cfg["_B·S"] = decode_batch * prefill_len
    if context_len:
        cfg["_C"] = context_len
    # kv_len: for prefill = S + C, for decode = C
    if prefill_len and context_len:
        cfg["_S+C"] = prefill_len + context_len
    if context_len:
        cfg["_kv_len"] = context_len  # decode kv_len = C

    result: dict[str, Any] = {
        "architecture": arch,
        "family": family,
        "model_type": "mllm" if is_vl else ("moe" if is_moe else "llm"),
        "quantization": quant,
        "config": {
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"],
            "num_kv_heads": cfg["num_kv_heads"],
            "head_dim": cfg["head_dim"],
            "intermediate_size": cfg["intermediate_size"],
            "vocab_size": cfg["vocab_size"],
            "is_moe": is_moe,
            "first_k_dense_replace": first_k_dense,
            "tp_size": tp_size,
            "quantization": quant,
            "dtype_bytes": cfg["dtype_bytes"],
            "weight_dtype_bytes": cfg["weight_dtype_bytes"],
        },
        # Symbol → concrete value mapping (TP-divided, for numeric resolution)
        "symbols": {
            "H": cfg["hidden_size"],
            "n_h": cfg["num_heads"],
            "n_kv": cfg["num_kv_heads"],
            "d": cfg["head_dim"],
            "I": cfg["intermediate_size"],
            "2·I": 2 * cfg["intermediate_size"],
            "V": cfg["vocab_size"],
            "QKV": (cfg["num_heads"] + 2 * cfg["num_kv_heads"]) * cfg["head_dim"],
            "n_h·d": cfg["num_heads"] * cfg["head_dim"],
        },
        # Full (undivided) config.json values for symbolic display
        "full_symbols": {
            "H": model_summary["hidden_size"],
            "n_h": full_num_heads,
            "n_kv": full_num_kv,
            "d": cfg["head_dim"],
            "I": full_intermediate,
            "2·I": 2 * full_intermediate,
            "V": full_vocab,
            "QKV": (full_num_heads + 2 * full_num_kv) * cfg["head_dim"],
            "n_h·d": full_num_heads * cfg["head_dim"],
        },
        # Symbols that are divided by TP (for symbolic display as "value/TP")
        "tp_divided": ["n_h", "n_kv", "I", "2·I", "V", "QKV", "n_h·d"],
    }

    if tp_size > 1:
        result["tp_note"] = f"Per-rank shapes (TP={tp_size})"

    # Add phase-specific symbols
    if prefill_len:
        result["symbols"]["S"] = prefill_len
    if decode_batch:
        result["symbols"]["B"] = decode_batch
    if context_len is not None:
        result["symbols"]["C"] = context_len
    if prefill_len:
        result["symbols"]["S+C"] = prefill_len + (context_len or 0)
    if is_moe:
        result["symbols"]["E"] = cfg.get("num_experts", 8)
        result["symbols"]["K"] = cfg.get("num_experts_per_tok", 2)
        result["full_symbols"]["E"] = cfg.get("num_experts", 8)
        result["full_symbols"]["K"] = cfg.get("num_experts_per_tok", 2)
        moe_I = cfg.get("moe_intermediate_size")
        if moe_I and moe_I != cfg["intermediate_size"]:
            result["symbols"]["I_moe"] = moe_I
            result["symbols"]["2·I_moe"] = 2 * moe_I
            # Full undivided MoE intermediate
            raw_moe_I = model_summary.get("moe_intermediate_size") or moe_I
            result["full_symbols"]["I_moe"] = raw_moe_I
            result["full_symbols"]["2·I_moe"] = 2 * raw_moe_I
            result["tp_divided"].append("I_moe")
            result["tp_divided"].append("2·I_moe")
        if cfg.get("n_shared_experts", 0) > 0:
            result["symbols"]["n_shared"] = cfg["n_shared_experts"]
            result["full_symbols"]["n_shared"] = cfg["n_shared_experts"]

    # MLA-specific symbols
    if is_mla:
        q_lora_rank = cfg.get("q_lora_rank", 0)
        kv_lora_rank = cfg.get("kv_lora_rank", 512)
        qk_nope_head_dim = cfg.get("qk_nope_head_dim", 128)
        qk_rope_head_dim = cfg.get("qk_rope_head_dim", 64)
        if q_lora_rank > 0:
            result["symbols"]["Q_r"] = q_lora_rank
            result["symbols"]["n_h·D_qh"] = (
                cfg["num_heads"] * (qk_nope_head_dim + qk_rope_head_dim)
            )
            result["full_symbols"]["Q_r"] = q_lora_rank
            result["full_symbols"]["n_h·D_qh"] = (
                full_num_heads * (qk_nope_head_dim + qk_rope_head_dim)
            )
            result["tp_divided"].append("n_h·D_qh")
        result["symbols"]["KV_r"] = kv_lora_rank
        result["symbols"]["D_rope"] = qk_rope_head_dim
        result["full_symbols"]["KV_r"] = kv_lora_rank
        result["full_symbols"]["D_rope"] = qk_rope_head_dim

    # Build prefill and decode graphs
    # Both phases use B·S as the token dimension for alignment:
    #   Prefill: S = prefill_len (e.g. 128), so B·S = batch × seq_len
    #   Decode:  S = 1 (always), so B·S = batch × 1 = batch
    # kv_len:
    #   Prefill: S+C per sequence (query tokens + prior context)
    #   Decode:  C (full context length)
    for phase, tok_var, seq_var in [
        ("prefill", "B·S", "S+C"),
        ("decode", "B·S", "C"),
    ]:
        root = _build_full_model(cfg, family, is_moe, phase, tok_var, seq_var)
        _compute_totals(root)
        result[phase] = root.to_dict()

    return result


def _build_full_model(
    cfg: dict, family: str, is_moe: bool, phase: str,
    tokens: str, seq: str,
) -> ModuleNode:
    """Build the complete model tree."""
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    dtype_bytes = cfg["dtype_bytes"]
    T = cfg.get(f"_{tokens}", tokens)
    first_k_dense = cfg.get("first_k_dense_replace", 0)
    is_mla = cfg.get("is_mla", False)
    is_vl = cfg.get("is_vl", False)

    children: list[ModuleNode] = []

    # Vision encoder (for VL models, prefill phase only)
    if is_vl and phase == "prefill":
        vit = _build_vision_encoder(cfg, phase, tokens)
        children.append(vit)
        proj = _build_vl_projector(cfg, phase, tokens)
        children.append(proj)

    # Text embedding
    embed = ModuleNode(
        name="embed_tokens", path="model.embed_tokens",
        module_type="VocabParallelEmbedding",
        ops=[OpNode(
            name="aten::embedding", role="embedding",
            backend="torch-xpu-ops",
            input_shapes=[["V", "H"], [tokens]],
            output_shape=[tokens, "H"],
            memory_bytes=(T * H * dtype_bytes) if isinstance(T, int) else 0,
            flops=0,
            phase=phase,
        )],
    )
    children.append(embed)

    # Decoder layers — handle MLA, hybrid dense/MoE, standard dense/MoE
    layers: list[ModuleNode] = []

    # Choose the right layer builders based on architecture
    if is_mla:
        # MLA architectures (DeepSeek-V2/V3)
        dense_builder = _build_mla_decoder_layer
        moe_builder = _build_mla_moe_layer
    else:
        dense_builder = _build_decoder_layer
        moe_builder = _build_moe_layer

    if is_moe and first_k_dense > 0:
        # Hybrid: first N layers are dense, rest are MoE
        dense_layer = dense_builder(cfg, family, phase, tokens, seq)
        dense_layer.name = "dense_layer"
        dense_layer.module_type = f"{family}DenseLayer"
        dense_layer.repeat_count = first_k_dense
        layers.append(dense_layer)

        num_moe_layers = cfg["num_layers"] - first_k_dense
        moe_layer = moe_builder(cfg, family, phase, tokens, seq)
        moe_layer.name = "moe_layer"
        moe_layer.module_type = f"{family}MoELayer"
        moe_layer.repeat_count = num_moe_layers
        layers.append(moe_layer)
    elif is_moe:
        layer = moe_builder(cfg, family, phase, tokens, seq)
        layers.append(layer)
    else:
        layer = dense_builder(cfg, family, phase, tokens, seq)
        layers.append(layer)

    children.extend(layers)

    # Final norm
    final_norm = ModuleNode(
        name="norm", path="model.norm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "final_norm", tokens, phase)],
    )
    children.append(final_norm)

    # LM head
    lm_head = ModuleNode(
        name="lm_head", path="lm_head",
        module_type="ParallelLMHead",
        ops=[OpNode(
            name="aten::mm", role="lm_head",
            backend="torch-xpu-ops",
            input_shapes=[[tokens, "H"], ["H", "V"]],
            output_shape=[tokens, "V"],
            memory_bytes=_mm_mem(T, H, V, dtype_bytes) if isinstance(T, int) else 0,
            flops=_mm_flops(T, H, V) if isinstance(T, int) else 0,
            phase=phase,
        )],
    )
    children.append(lm_head)

    model_type = f"{family}ForCausalLM"
    if is_vl:
        model_type = f"{family}ForConditionalGeneration"

    root = ModuleNode(
        name="model", path="",
        module_type=model_type,
        children=children,
    )
    return root


def _build_encoder_graph(
    model_summary: dict, family: str,
    seq_len: int | None = None, tp_size: int = 1,
) -> dict:
    """Build graph for encoder-only models (BERT/RoBERTa for embedding/reranking)."""
    full_num_heads = model_summary.get("num_heads") or 12
    cfg: dict[str, Any] = {
        "hidden_size": model_summary.get("hidden_size", 768),
        "num_layers": model_summary.get("num_layers", 12),
        "num_heads": full_num_heads // tp_size,
        "head_dim": model_summary.get("head_dim",
                                       model_summary.get("hidden_size", 768) // full_num_heads),
        "intermediate_size": model_summary.get("intermediate_size", 3072) // tp_size,
        "vocab_size": model_summary.get("vocab_size", 30522) // tp_size,
        "dtype_bytes": _dtype_bytes(model_summary.get("dtype", "float32")),
    }
    if seq_len:
        cfg["_S"] = seq_len

    result: dict[str, Any] = {
        "architecture": model_summary.get("architecture", ""),
        "family": family,
        "model_type": "encoder",
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "symbols": {
            "H": cfg["hidden_size"],
            "n_h": cfg["num_heads"],
            "d": cfg["head_dim"],
            "I": cfg["intermediate_size"],
            "V": cfg["vocab_size"],
        },
    }
    if seq_len:
        result["symbols"]["S"] = seq_len

    # Encoder models only have a "forward" phase (no prefill/decode split)
    root = _build_encoder_model(cfg, "prefill", "S")
    _compute_totals(root)
    result["prefill"] = root.to_dict()
    # No decode phase for encoder models
    result["decode"] = None

    return result


def _build_diffusion_placeholder(model_summary: dict, family: str) -> dict:
    """Build placeholder graph for diffusion models.

    Diffusion models (FLUX, Wan, etc.) are not served by vLLM and have
    fundamentally different op patterns (iterative denoising).
    Static analysis provides basic structure info.
    """
    return {
        "architecture": model_summary.get("architecture", ""),
        "family": family,
        "model_type": "diffusion",
        "note": (
            f"Diffusion models ({family}) use iterative denoising pipelines "
            f"(typically via diffusers, not vLLM). Static op graph analysis "
            f"is limited. Use profiling with the appropriate pipeline framework."
        ),
        "config": {
            "hidden_size": model_summary.get("hidden_size"),
            "num_layers": model_summary.get("num_layers"),
        },
        "prefill": None,
        "decode": None,
    }


def _compute_totals(node: ModuleNode) -> tuple[int, int]:
    """Recursively compute total memory and FLOPs for a module."""
    mem = sum(op.memory_bytes for op in node.ops)
    flops = sum(op.flops for op in node.ops)

    for child in node.children:
        c_mem, c_flops = _compute_totals(child)
        mult = child.repeat_count
        mem += c_mem * mult
        flops += c_flops * mult

    node.total_memory = mem
    node.total_flops = flops
    return mem, flops


# ===================================================================
# Timing annotation — merge profiling data into static graph
# ===================================================================

def _normalize_trace_name(name: str) -> str:
    """Strip C++ binding prefixes from trace op names."""
    for prefix in ("_C_cache_ops::", "_C::", "_xpu_C::", "torch::ops::"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# Kernel names that are functionally equivalent to aten::mm (matmul variants)
_MM_EQUIVALENT_NAMES = frozenset((
    "fp8_gemm_w8a16",
    "fp8_gemm",
    "gemm_w8a16",
    "int4_gemm",
    "int8_gemm",
    "cutlass_gemm",
    "cublas_gemm",
    "marlin_gemm",
    "machete_gemm",
    "awq_gemm",
))


def _is_matmul_op(norm_name: str) -> bool:
    """Check if a normalized op name represents a matmul/gemm operation."""
    if norm_name == "aten::mm":
        return True
    # Check base name (without any suffix)
    base = norm_name.split("_forward")[0]
    return base in _MM_EQUIVALENT_NAMES or "gemm" in norm_name.lower()


def _parse_shapes_str(s: str) -> list[list[int]]:
    """Parse shape string like '[[1, 2560], [2560, 6144]]' to nested list."""
    if not s or s == "—":
        return []
    import ast
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [
                [int(d) for d in shape if isinstance(d, (int, float))]
                for shape in parsed
                if isinstance(shape, (list, tuple))
            ]
    except (ValueError, SyntaxError):
        pass
    return []


def _resolve_dim(d: Any, symbols: dict) -> Any:
    """Resolve a symbolic dimension to its concrete value."""
    if isinstance(d, (int, float)):
        return int(d)
    s = str(d)
    if s in symbols:
        return symbols[s]
    parts = s.split("·")
    if len(parts) > 1:
        prod = 1
        for p in parts:
            if p in symbols:
                prod *= symbols[p]
            elif p.isdigit():
                prod *= int(p)
            else:
                return s
        return prod
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def _resolve_shapes(shapes: list[list], symbols: dict) -> list[tuple]:
    """Resolve all symbolic dims in shapes to concrete values."""
    return [tuple(_resolve_dim(d, symbols) for d in shape) for shape in shapes]


def _first_input_nontok_dims(shapes: list) -> tuple:
    """Extract non-token dims from the first input shape."""
    if not shapes:
        return ()
    first = shapes[0]
    if isinstance(first, (list, tuple)) and len(first) > 1:
        return tuple(first[1:])
    return ()


def annotate_graph_timing(graph: dict, trace_ops: list[dict]) -> None:
    """Annotate graph ops in-place with measured timing from profiling.

    Matches trace ops to graph ops using name + shape signatures.
    For aten::mm, matches by weight-matrix shape (2nd tensor).
    For other ops, matches by normalized name + first-input non-token dims.
    Falls back to name-only matching for unique op names.

    Args:
        graph: Output of build_model_graph() — modified in place.
        trace_ops: Raw op dicts from trace_parser.parse_trace_file().
    """
    symbols = graph.get("symbols", {})

    # Build lookup: (normalized_name, signature) → {device_us, cpu_us, count, raw_name}
    lookup: dict[tuple, dict] = {}
    # Also build name-only lookup for fallback
    name_lookup: dict[str, dict] = {}
    for op in trace_ops:
        raw_name = op.get("name", "")
        norm = _normalize_trace_name(raw_name)
        shapes = _parse_shapes_str(op.get("input_shapes", ""))
        dur_dev = op.get("device_time_us", 0) or 0
        dur_cpu = op.get("cpu_time_us", 0) or 0
        count = op.get("count", 1)

        # Build key based on op type
        if _is_matmul_op(norm) and len(shapes) >= 2:
            key = ("aten::mm", tuple(shapes[1]))
        else:
            key = (norm, _first_input_nontok_dims(shapes))

        if key not in lookup:
            lookup[key] = {"device_us": 0.0, "cpu_us": 0.0, "count": 0,
                           "raw_name": raw_name}
        lookup[key]["device_us"] += dur_dev
        lookup[key]["cpu_us"] += dur_cpu
        lookup[key]["count"] += count

        # Name-only aggregation for fallback
        if norm not in name_lookup:
            name_lookup[norm] = {"device_us": 0.0, "cpu_us": 0.0, "count": 0,
                                 "raw_name": raw_name}
        name_lookup[norm]["device_us"] += dur_dev
        name_lookup[norm]["cpu_us"] += dur_cpu
        name_lookup[norm]["count"] += count

    # Walk graph and annotate
    matched = 0
    total_ops = 0
    for phase in ("prefill", "decode"):
        if phase in graph:
            m, t = _annotate_node(graph[phase], lookup, name_lookup, symbols)
            matched += m
            total_ops += t

    graph["has_timing"] = True
    graph["timing_matched"] = matched
    graph["timing_total_ops"] = total_ops


def _annotate_node(
    node: dict, lookup: dict, name_lookup: dict, symbols: dict
) -> tuple[int, int]:
    """Annotate a serialized node dict in-place. Returns (matched, total)."""
    node_time = 0.0
    node_cpu = 0.0
    matched = 0
    total = 0

    for op in node.get("ops", []):
        total += 1
        resolved = _resolve_shapes(op["input_shapes"], symbols)

        # Build key to match against lookup
        if _is_matmul_op(op["name"]) and len(resolved) >= 2:
            key = ("aten::mm", tuple(resolved[1]))
        else:
            norm = _normalize_trace_name(op["name"])
            key = (norm, _first_input_nontok_dims(resolved))

        info = lookup.get(key)

        # Fallback: try name-only match for ops with unique/complex args
        if info is None:
            norm = _normalize_trace_name(op["name"])
            info = name_lookup.get(norm)

        if info is not None:
            per_call_dev = info["device_us"] / max(info["count"], 1)
            per_call_cpu = info["cpu_us"] / max(info["count"], 1)
            op["device_time_us"] = round(per_call_dev, 2)
            op["cpu_time_us"] = round(per_call_cpu, 2)
            op["profiled_calls"] = info["count"]
            op["profiled_name"] = info.get("raw_name", "")
            node_time += per_call_dev
            node_cpu += per_call_cpu
            matched += 1

    child_time = 0.0
    child_cpu = 0.0
    for child in node.get("children", []):
        cm, ct = _annotate_node(child, lookup, name_lookup, symbols)
        matched += cm
        total += ct
        mult = child.get("repeat_count", 1)
        child_time += child.get("total_device_time_us", 0) * mult
        child_cpu += child.get("total_cpu_time_us", 0) * mult

    node["total_device_time_us"] = round(node_time + child_time, 2)
    node["total_cpu_time_us"] = round(node_cpu + child_cpu, 2)
    return matched, total


# ===================================================================
# Module-path-based annotation (stack-aware profiling)
# ===================================================================

def annotate_graph_from_modules(graph: dict, module_ops: list[dict]) -> None:
    """Annotate graph using module-path-enriched trace ops.

    This uses the nn.Module hierarchy from profiling with with_stack=True
    to precisely assign timing to static graph nodes by matching roles
    within the module tree.

    Args:
        graph: Output of build_model_graph() — modified in place.
        module_ops: Output of parse_trace_with_modules().
    """
    if not module_ops:
        return

    # Build lookup: role → list of op timing entries
    # Group by (layer_idx, role) for per-layer ops, and (None, role) for global ops
    role_timing: dict[tuple, list[dict]] = {}
    for op in module_ops:
        role = op.get("role")
        if not role:
            continue
        layer_idx = op.get("layer_idx")
        key = (layer_idx, role)
        if key not in role_timing:
            role_timing[key] = []
        role_timing[key].append(op)

    # Compute average timing per role across all layers
    # For repeated layers, average the timing across layer instances
    role_avg: dict[str, dict] = {}
    role_by_layer: dict[str, list[dict]] = {}

    for (layer_idx, role), ops in role_timing.items():
        if role not in role_by_layer:
            role_by_layer[role] = []
        total_dev = sum(o.get("device_time_us", 0) for o in ops)
        total_cpu = sum(o.get("cpu_time_us", 0) for o in ops)
        total_count = sum(o.get("count", 0) for o in ops)
        # Pick the op with the most time as representative name
        best_op = max(ops, key=lambda o: o.get("device_time_us", 0))
        role_by_layer[role].append({
            "device_us": total_dev,
            "cpu_us": total_cpu,
            "count": total_count,
            "layer_idx": layer_idx,
            "raw_name": best_op.get("name", ""),
            "ops": ops,  # keep all ops for sub_ops
        })

    for role, layer_entries in role_by_layer.items():
        n_layers = len(layer_entries)
        avg_dev = sum(e["device_us"] for e in layer_entries) / max(n_layers, 1)
        avg_cpu = sum(e["cpu_us"] for e in layer_entries) / max(n_layers, 1)
        avg_count = sum(e["count"] for e in layer_entries) / max(n_layers, 1)
        # Use the raw_name from the entry with the highest device time
        best_entry = max(layer_entries, key=lambda e: e["device_us"])
        # Aggregate sub-ops across layers (average per layer)
        sub_ops_agg: dict[str, dict] = {}
        for entry in layer_entries:
            for sub_op in entry.get("ops", []):
                sname = sub_op.get("name", "")
                if sname not in sub_ops_agg:
                    sub_ops_agg[sname] = {"device_us": 0.0, "cpu_us": 0.0,
                                          "count": 0, "n_entries": 0}
                sub_ops_agg[sname]["device_us"] += sub_op.get("device_time_us", 0)
                sub_ops_agg[sname]["cpu_us"] += sub_op.get("cpu_time_us", 0)
                sub_ops_agg[sname]["count"] += sub_op.get("count", 0)
                sub_ops_agg[sname]["n_entries"] += 1
        # Average across layers
        sub_ops_list = []
        for sname, sagg in sub_ops_agg.items():
            nl = max(sagg["n_entries"], 1)
            sub_ops_list.append({
                "name": sname,
                "device_time_us": round(sagg["device_us"] / nl, 2),
                "cpu_time_us": round(sagg["cpu_us"] / nl, 2),
                "count": max(1, sagg["count"] // nl),
            })
        sub_ops_list.sort(key=lambda x: -x["device_time_us"])
        role_avg[role] = {
            "device_us": avg_dev,
            "cpu_us": avg_cpu,
            "count": avg_count,
            "n_layers": n_layers,
            "raw_name": best_entry.get("raw_name", ""),
            "sub_ops": sub_ops_list if len(sub_ops_list) > 1 else [],
        }

    # Also aggregate norm roles by position within layer
    # Norms: distinguish by their position (1st norm before attn, 2nd after)
    # We handle this by matching "norm" roles in order within each layer
    _assign_norm_positions(module_ops, role_timing, role_avg)

    # Walk graph and annotate ops by their role
    matched = 0
    total_ops = 0
    for phase in ("prefill", "decode"):
        if phase in graph:
            m, t = _annotate_node_by_role(graph[phase], role_avg)
            matched += m
            total_ops += t

    graph["has_timing"] = True
    graph["timing_matched"] = matched
    graph["timing_total_ops"] = total_ops
    graph["timing_method"] = "module_path"


def _assign_norm_positions(module_ops: list[dict],
                           role_timing: dict, role_avg: dict) -> None:
    """Distinguish norm ops by position (input_layernorm vs post_attention_layernorm).

    Also handles attention_norm → q_norm / k_norm disambiguation.
    Norms within a layer are initially assigned role='norm' or 'attention_norm'.
    We differentiate them by their timestamp order within each layer.
    """
    from collections import defaultdict

    # Group generic norm ops by layer, sorted by their aggregation order
    layer_norms: dict[int | None, list[dict]] = defaultdict(list)
    layer_attn_norms: dict[int | None, list[dict]] = defaultdict(list)
    for op in module_ops:
        role = op.get("role")
        layer_idx = op.get("layer_idx")
        if layer_idx is None:
            continue
        if role == "norm":
            layer_norms[layer_idx].append(op)
        elif role == "attention_norm":
            layer_attn_norms[layer_idx].append(op)

    # For each layer, assign positional norms
    # Convention: 1st norm = input_layernorm, 2nd = post_attention_layernorm
    norm_position_timing: dict[str, list[dict]] = defaultdict(list)
    for layer_idx, norms in layer_norms.items():
        for i, norm in enumerate(norms):
            if i == 0:
                pos_role = "input_layernorm"
            elif i == 1:
                pos_role = "post_attention_layernorm"
            else:
                pos_role = f"norm_{i}"
            norm_position_timing[pos_role].append(norm)

    # Attention norms: 1st = q_norm, 2nd = k_norm
    attn_norm_timing: dict[str, list[dict]] = defaultdict(list)
    for layer_idx, norms in layer_attn_norms.items():
        for i, norm in enumerate(norms):
            if i == 0:
                pos_role = "q_norm"
            elif i == 1:
                pos_role = "k_norm"
            else:
                pos_role = f"attention_norm_{i}"
            attn_norm_timing[pos_role].append(norm)

    # Add positional norm averages to role_avg
    norm_entry = role_avg.get("norm", {})
    attn_norm_entry = role_avg.get("attention_norm", {})

    for pos_role, ops in norm_position_timing.items():
        if ops:
            avg_cpu = sum(o.get("cpu_time_us", 0) for o in ops) / len(ops)
            if norm_entry and norm_entry.get("cpu_us", 0) > 0:
                dev_ratio = norm_entry["device_us"] / norm_entry["cpu_us"]
            else:
                dev_ratio = 1.0
            role_avg[pos_role] = {
                "device_us": avg_cpu * dev_ratio,
                "cpu_us": avg_cpu,
                "count": len(ops),
                "n_layers": len(ops),
                "raw_name": norm_entry.get("raw_name", ""),
            }

    for pos_role, ops in attn_norm_timing.items():
        if ops and pos_role not in role_avg:
            avg_cpu = sum(o.get("cpu_time_us", 0) for o in ops) / len(ops)
            source = attn_norm_entry or norm_entry
            if source and source.get("cpu_us", 0) > 0:
                dev_ratio = source["device_us"] / source["cpu_us"]
            else:
                dev_ratio = 1.0
            role_avg[pos_role] = {
                "device_us": avg_cpu * dev_ratio,
                "cpu_us": avg_cpu,
                "count": len(ops),
                "n_layers": len(ops),
                "raw_name": source.get("raw_name", ""),
            }


def _annotate_node_by_role(
    node: dict, role_avg: dict
) -> tuple[int, int]:
    """Annotate a serialized node dict in-place using role-based matching."""
    node_time = 0.0
    node_cpu = 0.0
    matched = 0
    total = 0

    for op in node.get("ops", []):
        total += 1
        role = op.get("role", "")

        # Try exact role match
        info = role_avg.get(role)

        # Fallback chain for norms
        if info is None:
            if role in ("q_norm", "k_norm"):
                # Try generic attention_norm, then norm
                info = role_avg.get("attention_norm") or role_avg.get("norm")
            elif role in ("input_layernorm", "post_attention_layernorm", "final_norm"):
                info = role_avg.get("norm")
            elif role == "rotary_emb":
                # rotary_embedding may not have been identified by role
                info = role_avg.get("rotary_embedding")

        if info is not None:
            per_call_dev = info["device_us"] / max(info["count"], 1)
            per_call_cpu = info["cpu_us"] / max(info["count"], 1)
            op["device_time_us"] = round(per_call_dev, 2)
            op["cpu_time_us"] = round(per_call_cpu, 2)
            op["profiled_calls"] = int(info["count"])
            op["profiled_name"] = info.get("raw_name", "")
            # Attach sub-ops breakdown for complex ops
            if info.get("sub_ops"):
                op["sub_ops"] = info["sub_ops"]
            node_time += per_call_dev
            node_cpu += per_call_cpu
            matched += 1

    child_time = 0.0
    child_cpu = 0.0
    for child in node.get("children", []):
        cm, ct = _annotate_node_by_role(child, role_avg)
        matched += cm
        total += ct
        mult = child.get("repeat_count", 1)
        child_time += child.get("total_device_time_us", 0) * mult
        child_cpu += child.get("total_cpu_time_us", 0) * mult

    node["total_device_time_us"] = round(node_time + child_time, 2)
    node["total_cpu_time_us"] = round(node_cpu + child_cpu, 2)
    return matched, total
