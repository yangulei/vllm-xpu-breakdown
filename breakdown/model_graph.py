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

def _mm_mem(M: int, K: int, N: int, dtype_bytes: int = 2) -> int:
    """Memory for matmul: read A[M,K] + B[K,N] + write C[M,N]."""
    return (M * K + K * N + M * N) * dtype_bytes


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
    T = cfg.get(f"_{tokens}", tokens)  # numeric if available
    S = cfg.get(f"_{seq}", seq)

    ops: list[OpNode] = []

    # QKV projection
    ops.append(OpNode(
        name="aten::mm", role="qkv_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "QKV"]],
        output_shape=[tokens, "QKV"],
        memory_bytes=_mm_mem(T, H, qkv_size, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, qkv_size) if isinstance(T, int) else 0,
        phase=phase,
    ))

    # Q/K norms (Qwen3-specific)
    if cfg.get("has_qk_norm"):
        for role in ("q_norm", "k_norm"):
            n = n_h if role == "q_norm" else n_kv
            ops.append(OpNode(
                name="rms_norm", role=role,
                backend="vllm-xpu-kernels",
                input_shapes=[[tokens, str(n), "d"]],
                output_shape=[tokens, str(n), "d"],
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
            input_shapes=[[tokens, "n_h", "d"], ["cache_len", "n_kv", "d"]],
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
        memory_bytes=_mm_mem(T, n_h * d, H, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, n_h * d, H) if isinstance(T, int) else 0,
        phase=phase,
    ))

    return ops


def _build_mlp_ops(cfg: dict, phase: str, tokens: str) -> list[OpNode]:
    """Build ops for MLP module."""
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    dtype_bytes = cfg["dtype_bytes"]
    T = cfg.get(f"_{tokens}", tokens)

    ops: list[OpNode] = []

    # Gate+Up fused projection
    ops.append(OpNode(
        name="aten::mm", role="gate_up_proj",
        backend="torch-xpu-ops",
        input_shapes=[[tokens, "H"], ["H", "2·I"]],
        output_shape=[tokens, "2·I"],
        memory_bytes=_mm_mem(T, H, 2 * I, dtype_bytes) if isinstance(T, int) else 0,
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
        memory_bytes=_mm_mem(T, I, H, dtype_bytes) if isinstance(T, int) else 0,
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
        input_shapes=[[tokens, "H"], ["H", str(num_experts)]],
        output_shape=[tokens, str(num_experts)],
        memory_bytes=_mm_mem(T, H, num_experts, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T, H, num_experts) if isinstance(T, int) else 0,
        phase=phase,
    )
    align_op = OpNode(
        name="moe_align_block_size", role="moe_routing",
        backend="vllm-xpu-kernels",
        input_shapes=[[tokens, str(num_experts)]],
        output_shape=[tokens],
        memory_bytes=0, flops=0, phase=phase,
    )
    # Expert computation (top_k experts active) — use moe_intermediate_size
    moe_I_sym = f"I_moe" if moe_I != I else "I"
    moe_2I_sym = f"2·I_moe" if moe_I != I else "2·I"
    expert_mm1 = OpNode(
        name="aten::mm", role="expert_gate_up",
        backend="torch-xpu-ops",
        input_shapes=[[f"{tokens}·{top_k}", "H"], ["H", moe_2I_sym]],
        output_shape=[f"{tokens}·{top_k}", moe_2I_sym],
        memory_bytes=_mm_mem(T * top_k, H, 2 * moe_I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_mm_flops(T * top_k, H, 2 * moe_I) if isinstance(T, int) else 0,
        phase=phase,
    )
    expert_act = OpNode(
        name="silu_and_mul", role="expert_activation",
        backend="vllm-xpu-kernels",
        input_shapes=[[f"{tokens}·{top_k}", moe_2I_sym]],
        output_shape=[f"{tokens}·{top_k}", moe_I_sym],
        memory_bytes=_activation_mem(T * top_k, moe_I, dtype_bytes) if isinstance(T, int) else 0,
        flops=_activation_flops(T * top_k, moe_I) if isinstance(T, int) else 0,
        phase=phase,
    )
    expert_mm2 = OpNode(
        name="aten::mm", role="expert_down",
        backend="torch-xpu-ops",
        input_shapes=[[f"{tokens}·{top_k}", moe_I_sym], [moe_I_sym, "H"]],
        output_shape=[f"{tokens}·{top_k}", "H"],
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


# ===================================================================
# Public API
# ===================================================================

# Architecture family detection from HuggingFace architecture name
_ARCH_FAMILY_MAP: dict[str, str] = {
    "LlamaForCausalLM": "Llama",
    "MistralForCausalLM": "Llama",
    "YiForCausalLM": "Llama",
    "InternLM2ForCausalLM": "Llama",
    "Qwen2ForCausalLM": "Qwen2",
    "Qwen3ForCausalLM": "Qwen3",
    "MixtralForCausalLM": "Mixtral",
    "DeepseekV2ForCausalLM": "DeepSeekV2",
    "DeepseekV3ForCausalLM": "DeepSeekV3",
    "Qwen3MoeForCausalLM": "Qwen3Moe",
}

# Architectures that have QK normalization
_HAS_QK_NORM = {"Qwen3", "Qwen3Moe", "DeepSeekV2", "DeepSeekV3"}


def _dtype_bytes(dtype: str) -> int:
    """Get bytes per element for dtype."""
    mapping = {
        "float32": 4, "float16": 2, "bfloat16": 2,
        "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1, "int4": 1,
    }
    # Strip torch. prefix
    dt = dtype.replace("torch.", "")
    return mapping.get(dt, 2)


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
    tp_size: int = 1,
) -> dict:
    """Build static model graph from model summary (from model_info.py).

    Args:
        model_summary: output of summarize_config() — contains architecture,
                       hidden_size, num_layers, num_heads, etc.
        prefill_len: prefill sequence length (for shape estimation)
        decode_batch: decode batch size (for shape estimation)
        tp_size: tensor parallel size (splits heads, intermediate, vocab)

    Returns:
        Dict with "prefill" and "decode" trees, plus metadata.
    """
    arch = model_summary.get("architecture", "")
    family = _ARCH_FAMILY_MAP.get(arch, "Unknown")
    is_moe = model_summary.get("is_moe", False)

    # Full (un-split) dimensions for reference
    full_num_heads = model_summary["num_heads"]
    full_num_kv = model_summary.get("num_kv_heads", full_num_heads)
    full_intermediate = model_summary["intermediate_size"]
    full_vocab = model_summary["vocab_size"]

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
    }

    if is_moe:
        cfg["num_experts"] = model_summary.get("num_experts", 8)
        cfg["num_experts_per_tok"] = model_summary.get("num_experts_per_tok", 2)
        raw_moe_I = model_summary.get("moe_intermediate_size")
        cfg["moe_intermediate_size"] = (raw_moe_I // tp_size) if raw_moe_I else None
        cfg["n_shared_experts"] = model_summary.get("n_shared_experts", 0)

    # Hybrid dense/MoE detection
    first_k_dense = model_summary.get("first_k_dense_replace", 0) or 0
    cfg["first_k_dense_replace"] = first_k_dense

    # Set numeric token counts if specified
    if prefill_len:
        cfg["_S"] = prefill_len
    if decode_batch:
        cfg["_B"] = decode_batch

    result: dict[str, Any] = {
        "architecture": arch,
        "family": family,
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
        },
        # Symbol → concrete value mapping for tooltips
        # When TP>1, dimensions shown are per-rank
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
    }

    if tp_size > 1:
        result["tp_note"] = f"Per-rank shapes (TP={tp_size})"

    # Add phase-specific symbols
    if prefill_len:
        result["symbols"]["S"] = prefill_len
    if decode_batch:
        result["symbols"]["B"] = decode_batch
    if is_moe:
        result["symbols"][str(cfg.get("num_experts", 8))] = cfg.get("num_experts", 8)
        moe_I = cfg.get("moe_intermediate_size")
        if moe_I and moe_I != cfg["intermediate_size"]:
            result["symbols"]["I_moe"] = moe_I
            result["symbols"]["2·I_moe"] = 2 * moe_I
        if cfg.get("n_shared_experts", 0) > 0:
            result["symbols"]["n_shared"] = cfg["n_shared_experts"]

    # Build prefill and decode graphs
    for phase, tok_var, seq_var in [
        ("prefill", "S", "S"),
        ("decode", "B", "cache_len"),
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

    # Embedding
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

    # Decoder layers — handle hybrid dense/MoE
    layers: list[ModuleNode] = []
    if is_moe and first_k_dense > 0:
        # Hybrid: first N layers are dense, rest are MoE
        dense_layer = _build_decoder_layer(cfg, family, phase, tokens, seq)
        dense_layer.name = "dense_layer"
        dense_layer.module_type = f"{family}DenseLayer"
        dense_layer.repeat_count = first_k_dense
        layers.append(dense_layer)

        num_moe_layers = cfg["num_layers"] - first_k_dense
        moe_layer = _build_moe_layer(cfg, family, phase, tokens, seq)
        moe_layer.name = "moe_layer"
        moe_layer.module_type = f"{family}MoELayer"
        moe_layer.repeat_count = num_moe_layers
        layers.append(moe_layer)
    elif is_moe:
        layer = _build_moe_layer(cfg, family, phase, tokens, seq)
        layers.append(layer)
    else:
        layer = _build_decoder_layer(cfg, family, phase, tokens, seq)
        layers.append(layer)

    # Final norm
    final_norm = ModuleNode(
        name="norm", path="model.norm",
        module_type="RMSNorm",
        ops=[_build_norm_op(cfg, "final_norm", tokens, phase)],
    )

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

    root = ModuleNode(
        name="model", path="",
        module_type=f"{family}ForCausalLM",
        children=[embed] + layers + [final_norm, lm_head],
    )
    return root


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

    # Build lookup: (normalized_name, signature) → {device_us, cpu_us, count}
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
        if norm == "aten::mm" and len(shapes) >= 2:
            key = ("aten::mm", tuple(shapes[1]))
        else:
            key = (norm, _first_input_nontok_dims(shapes))

        if key not in lookup:
            lookup[key] = {"device_us": 0.0, "cpu_us": 0.0, "count": 0}
        lookup[key]["device_us"] += dur_dev
        lookup[key]["cpu_us"] += dur_cpu
        lookup[key]["count"] += count

        # Name-only aggregation for fallback
        if norm not in name_lookup:
            name_lookup[norm] = {"device_us": 0.0, "cpu_us": 0.0, "count": 0}
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
        if op["name"] == "aten::mm" and len(resolved) >= 2:
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
