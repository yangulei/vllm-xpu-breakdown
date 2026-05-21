# SPDX-License-Identifier: Apache-2.0
"""Fetch and summarize HuggingFace model configs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def fetch_model_config(model_id: str) -> dict[str, Any]:
    """Fetch config.json from HuggingFace Hub.

    Respects HF_ENDPOINT env variable for mirror support (e.g., hf-mirror.com).
    """
    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_id, "config.json")
        with open(path) as f:
            return json.load(f)
    except ImportError:
        pass

    # Direct HTTP fallback
    import urllib.request
    import urllib.error

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    url = f"{endpoint}/{model_id}/resolve/main/config.json"

    try:
        req = urllib.request.Request(url)
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to fetch config for {model_id}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to {endpoint}: {e.reason}") from e


def _get_vit_config(config: dict[str, Any], key: str) -> Any | None:
    """Extract a vision encoder config value from nested VL model configs.

    VL models store vision config in different places:
    - Qwen2.5-VL: config["vision_config"]["hidden_size"]
    - InternVL: config["vision_config"]["hidden_size"]
    - Some models: config["visual"]["hidden_size"]
    """
    for section in ("vision_config", "visual", "visual_config",
                    "vision_tower_config"):
        sub = config.get(section)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return None


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract key model properties from config.json."""
    archs = config.get("architectures", [])
    architecture = archs[0] if archs else config.get("model_type", "Unknown")

    hidden_size = config.get("hidden_size")
    num_layers = config.get("num_hidden_layers")
    num_heads = config.get("num_attention_heads")
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim")
    if head_dim is None and hidden_size and num_heads:
        head_dim = hidden_size // num_heads
    intermediate_size = config.get("intermediate_size") or config.get("ffn_dim")
    vocab_size = config.get("vocab_size")
    max_position = config.get("max_position_embeddings")
    dtype = config.get("torch_dtype") or config.get("dtype", "unknown")

    # MoE detection
    num_experts = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
    )
    is_moe = num_experts is not None and num_experts > 1
    num_experts_per_tok = (
        config.get("num_experts_per_tok")
        or config.get("n_group_top_k", config.get("top_k"))
    )

    # Quantization config
    quant_config = config.get("quantization_config")
    quant_method = None
    if quant_config:
        quant_method = quant_config.get("quant_method", "unknown")

    return {
        "architecture": architecture,
        "model_type": config.get("model_type", "unknown"),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position,
        "dtype": dtype,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "quant_method": quant_method,
        "rope_type": config.get("rope_scaling", {}).get("type") if config.get("rope_scaling") else None,
        # Hybrid dense/MoE: first N layers use dense MLP
        "first_k_dense_replace": config.get("first_k_dense_replace", 0),
        # MoE layer frequency (1 = every layer is MoE, 2 = every other, etc.)
        "moe_layer_freq": config.get("moe_layer_freq", 1),
        # Qwen-style: decoder_sparse_step (1 = all MoE, 2 = alternating)
        "decoder_sparse_step": config.get("decoder_sparse_step", 0),
        # MoE intermediate size (may differ from dense intermediate_size)
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        # Shared experts
        "n_shared_experts": config.get("n_shared_experts", 0),
        # MLA (Multi-head Latent Attention) — DeepSeek-V2/V3
        "kv_lora_rank": config.get("kv_lora_rank"),
        "q_lora_rank": config.get("q_lora_rank"),
        "qk_nope_head_dim": config.get("qk_nope_head_dim"),
        "qk_rope_head_dim": config.get("qk_rope_head_dim"),
        # Vision encoder (VL models)
        "vit_hidden_size": _get_vit_config(config, "hidden_size"),
        "vit_num_layers": _get_vit_config(config, "num_hidden_layers"),
        "vit_num_heads": _get_vit_config(config, "num_attention_heads"),
        "vit_intermediate_size": _get_vit_config(config, "intermediate_size"),
        "patch_size": _get_vit_config(config, "patch_size"),
        "image_size": _get_vit_config(config, "image_size"),
    }


# Known config dimension names for shape symbolization
def get_dim_symbols(summary: dict[str, Any]) -> dict[int, str]:
    """Build a mapping from literal dimension values to symbolic names.

    Returns {dim_value: symbol_name} for dimensions known from the config.
    """
    symbols: dict[int, str] = {}
    if summary.get("hidden_size"):
        symbols[summary["hidden_size"]] = "H"
    if summary.get("num_heads"):
        symbols[summary["num_heads"]] = "n_h"
    if summary.get("num_kv_heads") and summary["num_kv_heads"] != summary.get("num_heads"):
        symbols[summary["num_kv_heads"]] = "n_kv"
    if summary.get("head_dim"):
        symbols[summary["head_dim"]] = "d"
    if summary.get("intermediate_size"):
        symbols[summary["intermediate_size"]] = "I"
    if summary.get("vocab_size"):
        symbols[summary["vocab_size"]] = "V"
    if summary.get("num_experts"):
        symbols[summary["num_experts"]] = "E"

    # Derived dimensions
    h = summary.get("hidden_size") or 0
    n_h = summary.get("num_heads") or 0
    n_kv = summary.get("num_kv_heads") or n_h
    d_head = summary.get("head_dim") or 0
    inter = summary.get("intermediate_size") or 0

    if n_h and d_head:
        qkv = n_h * d_head + 2 * n_kv * d_head
        if qkv not in symbols:
            symbols[qkv] = "QKV"
        kv_dim = 2 * n_kv * d_head
        if kv_dim not in symbols:
            symbols[kv_dim] = "2·n_kv·d"
    if inter:
        if inter * 2 not in symbols:
            symbols[inter * 2] = "2·I"

    # MLA dimensions (DeepSeek-V2/V3)
    kv_lora = summary.get("kv_lora_rank")
    if kv_lora:
        symbols[kv_lora] = "kv_lora"
    q_lora = summary.get("q_lora_rank")
    if q_lora:
        symbols[q_lora] = "q_lora"

    # Vision encoder dimensions (VL models)
    vit_h = summary.get("vit_hidden_size")
    if vit_h and vit_h not in symbols:
        symbols[vit_h] = "H_vit"

    return symbols
