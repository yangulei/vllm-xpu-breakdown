# SPDX-License-Identifier: Apache-2.0
"""Registry of known vllm-xpu-kernels ops.

Provides both a static list (extracted from torch_bindings.cpp) and runtime
introspection of loaded torch.ops namespaces.
"""

from __future__ import annotations

# --- Static registry extracted from vllm-xpu-kernels/csrc/ ---
# These are the op names as they appear in TORCH_LIBRARY registrations.

# csrc/torch_bindings.cpp  (TORCH_EXTENSION_NAME = _C)
_C_OPS: set[str] = {
    "weak_ref_tensor",
    "rms_norm",
    "fused_add_rms_norm",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant",
    "rms_norm_static_fp8_quant",
    "fused_add_rms_norm_static_fp8_quant",
    "silu_and_mul",
    "silu_and_mul_quant",
    "mul_and_silu",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "fatrelu_and_mul",
    "gelu_fast",
    "gelu_new",
    "gelu_quick",
    "rotary_embedding",
    "fused_qk_norm_rope",
    "static_scaled_fp8_quant",
    "dynamic_scaled_fp8_quant",
    "dynamic_per_token_scaled_fp8_quant",
    "per_token_group_fp8_quant",
    "per_token_group_quant_mxfp4",
    "swigluoai_and_mul",
    "relu2_no_mul",
    "swiglustep_and_mul",
    "get_xpu_view_from_cpu_tensor",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "xpu_memcpy_sync",
    "merge_attn_states",
    "fused_minimax_m3_qknorm_rope_kv_insert",
}

# csrc/torch_bindings.cpp  (CONCAT(TORCH_EXTENSION_NAME, _cache_ops))
_C_CACHE_OPS: set[str] = {
    "reshape_and_cache",
    "reshape_and_cache_flash",
    "concat_and_cache_mla",
    "gather_cache",
    "convert_fp8",
    "swap_blocks",
    "swap_blocks_batch",
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "gather_and_maybe_dequant_cache",
}

# csrc/moe/torch_bindings.cpp  (TORCH_EXTENSION_NAME = _moe_C)
_MOE_C_OPS: set[str] = {
    "moe_sum",
    "moe_align_block_size",
    "batched_moe_align_block_size",
    "moe_lora_align_block_size",
    "grouped_topk",
    "fused_grouped_topk",
    "topk_softmax",
    "topk_sigmoid",
    "moe_gather",
    "fused_moe_prologue",
    "init_expert_map",
    "remap_hidden_states",
}

# csrc/xpu/torch_bindings.cpp  (TORCH_EXTENSION_NAME = _xpu_C)
_XPU_C_OPS: set[str] = {
    "fp8_gemm",
    "fp8_gemm_w8a16",
    "fp4_gemm",
    "int4_gemm_w4a16",
    "int4_gemm_w4a8",
    "cutlass_grouped_gemm_interface",
    "deepseek_scaling_rope",
    "cutlass_paged_decode",
    "bgmv_shrink",
    "bgmv_expand",
    "bgmv_expand_slice",
    "gdn_attention",
    "is_bmg",
    "is_pvc",
    "exponential_2d_",
    "topk_topp_sampler",
}

# ---- CUDA-specific ops (from vLLM main repo csrc/) ----
# These are ops registered under _C/_C_cache_ops on CUDA builds that differ
# from the XPU set, plus CUDA-only kernels (flash_attn, marlin, cutlass, etc.)
_CUDA_C_OPS: set[str] = {
    # Attention
    "paged_attention_v1",
    "paged_attention_v2",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    # Quantization / GEMM
    "marlin_gemm",
    "marlin_qqq_gemm",
    "gptq_marlin_gemm",
    "gptq_marlin_24_gemm",
    "gptq_marlin_repack",
    "awq_gemm",
    "awq_dequantize",
    "cutlass_scaled_mm",
    "cutlass_scaled_mm_azp",
    "cutlass_moe_mm",
    # Quantized cache
    "fp8_marlin_gemm",
    "machete_gemm",
    "machete_prepack",
    # LoRA
    "sgmv_shrink",
    "sgmv_expand",
    "sgmv_expand_slice",
    # Misc CUDA kernels
    "aqlm_gemm",
    "aqlm_dequant",
    "ggml_dequantize",
    "ggml_mul_mat_vec",
    "ggml_mul_mat_a8",
}

# Module-to-ops mapping
STATIC_REGISTRY: dict[str, set[str]] = {
    "_C": _C_OPS,
    "_C_cache_ops": _C_CACHE_OPS,
    "_moe_C": _MOE_C_OPS,
    "_xpu_C": _XPU_C_OPS,
    "_cuda_C": _CUDA_C_OPS,
}

# Flat set for quick lookup
ALL_VLLM_XPU_OPS: set[str] = set()
for ops in STATIC_REGISTRY.values():
    ALL_VLLM_XPU_OPS.update(ops)

# Category labels
MODULE_CATEGORIES: dict[str, str] = {
    "_C": "vllm-xpu-kernels (general)",
    "_C_cache_ops": "vllm-xpu-kernels (cache)",
    "_moe_C": "vllm-xpu-kernels (MoE)",
    "_xpu_C": "vllm-xpu-kernels (XPU-specific)",
    "_cuda_C": "vllm-cuda-kernels (CUDA-specific)",
}


def discover_runtime_ops() -> dict[str, list[str]]:
    """Introspect loaded torch.ops to find vllm custom-kernel ops at runtime.

    Returns a dict mapping namespace to list of op names.
    Only works after vllm (xpu or cuda) has been imported.
    """
    import torch
    result: dict[str, list[str]] = {}
    for ns_name in ("_C", "_C_cache_ops", "_moe_C", "_xpu_C", "_cuda_C", "vllm"):
        ns = getattr(torch.ops, ns_name, None)
        if ns is None:
            continue
        ops = []
        for name in dir(ns):
            if not name.startswith("_"):
                ops.append(name)
        if ops:
            result[ns_name] = ops
    return result


def get_op_module(op_name: str) -> str | None:
    """Return the module key for a known vllm-xpu-kernels op, or None."""
    for module, ops in STATIC_REGISTRY.items():
        if op_name in ops:
            return module
    return None


def get_op_category(op_name: str) -> str | None:
    """Return a human-readable category for a known op, or None."""
    module = get_op_module(op_name)
    if module:
        return MODULE_CATEGORIES.get(module, module)
    return None
