# SPDX-License-Identifier: Apache-2.0
"""Fetch and summarize HuggingFace model configs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _normalize_layerwise_value(value: Any) -> Any:
    """Collapse per-layer config lists to a representative scalar when possible.

    Static graph generation assumes one configuration per repeated layer block.
    Some model configs encode MoE settings as one value per layer. If all
    entries are identical, treat them as a scalar. Otherwise, fall back to the
    first non-null entry so graph building can proceed with a representative
    layer configuration.
    """
    if not isinstance(value, list):
        return value
    if not value:
        return None

    normalized = [_normalize_layerwise_value(item) for item in value]
    first = normalized[0]
    if all(item == first for item in normalized):
        return first
    for item in normalized:
        if item is not None:
            return item
    return None


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
    except Exception:
        pass  # Fall through to HTTP fallback

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


def _leading_dense_layers(moe_layer_freq: Any) -> int:
    """Derive the number of leading dense layers from a moe_layer_freq spec.

    Some models (e.g. MiniMax-M3) express the dense/MoE layout as a per-layer
    list where 0 = dense MLP and 1 = MoE block. The leading run of zeros is the
    equivalent of ``first_k_dense_replace``. Scalar values carry no dense prefix.
    """
    if isinstance(moe_layer_freq, (list, tuple)):
        count = 0
        for v in moe_layer_freq:
            if v:
                break
            count += 1
        return count
    return 0


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract key model properties from config.json."""
    archs = config.get("architectures", [])
    architecture = archs[0] if archs else config.get("model_type", "Unknown")

    # Some multimodal models (e.g. MiniMax-M3) nest the language-model
    # parameters under ``text_config``. Read text params from there first,
    # falling back to the top level for flat configs (Llama, Qwen, etc.).
    tcfg = config.get("text_config")
    if not isinstance(tcfg, dict):
        tcfg = {}

    def tget(key: str, default: Any = None) -> Any:
        if key in tcfg:
            return tcfg[key]
        return config.get(key, default)

    hidden_size = tget("hidden_size")
    num_layers = tget("num_hidden_layers")
    num_heads = tget("num_attention_heads")
    num_kv_heads = tget("num_key_value_heads", num_heads)
    head_dim = tget("head_dim")
    if head_dim is None and hidden_size and num_heads:
        head_dim = hidden_size // num_heads
    # Dense MLP size. MiniMax-M3 names the dense-layer FFN dimension
    # ``dense_intermediate_size`` while reusing ``intermediate_size`` for the
    # MoE expert width.
    dense_intermediate = tget("dense_intermediate_size")
    intermediate_size = (
        dense_intermediate
        or tget("intermediate_size")
        or tget("ffn_dim")
    )
    vocab_size = tget("vocab_size")
    max_position = tget("max_position_embeddings")
    dtype = (
        tget("torch_dtype")
        or config.get("torch_dtype")
        or config.get("dtype", "unknown")
    )

    # MoE detection
    num_experts = (
        tget("num_local_experts")
        or tget("num_experts")
        or tget("n_routed_experts")
    )
    is_moe = num_experts is not None and num_experts > 1
    num_experts_per_tok = _normalize_layerwise_value(
        tget("num_experts_per_tok")
        or tget("n_group_top_k")
        or tget("top_k")
        or tget("moe_topk")
    )

    # MoE expert intermediate size. When a model carries a separate dense FFN
    # size (M3), the plain ``intermediate_size`` is the per-expert width.
    moe_intermediate_size = _normalize_layerwise_value(tget("moe_intermediate_size"))
    if moe_intermediate_size is None and dense_intermediate:
        moe_intermediate_size = _normalize_layerwise_value(tget("intermediate_size"))

    n_shared_experts = _normalize_layerwise_value(
        tget("n_shared_experts", tget("num_shared_expert", 0))
    )

    # Dense/MoE layout: scalar ``first_k_dense_replace`` or a per-layer
    # ``moe_layer_freq`` list whose leading zeros mark the dense prefix.
    first_k_dense = tget("first_k_dense_replace", 0)
    moe_layer_freq = tget("moe_layer_freq", 1)
    if not first_k_dense:
        first_k_dense = _leading_dense_layers(moe_layer_freq)

    # Quantization config
    quant_config = config.get("quantization_config") or tcfg.get("quantization_config")
    quant_method = None
    if quant_config:
        quant_method = quant_config.get("quant_method", "unknown")

    rope_scaling = tget("rope_scaling") or config.get("rope_scaling")

    # DeepSeek-style sparse attention (lightning indexer + top-k block
    # selection). MiniMax-M3 carries this under text_config.sparse_attention_config.
    sparse_cfg = tget("sparse_attention_config")
    if not isinstance(sparse_cfg, dict):
        sparse_cfg = {}
    use_sparse_attention = bool(sparse_cfg.get("use_sparse_attention"))

    return {
        "architecture": architecture,
        "model_type": config.get("model_type", "unknown"),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        # True when the decoder layer count lives under ``text_config`` (nested
        # multimodal configs, e.g. MiniMax-M3) rather than at the top level. Used
        # to target a reduced-layer ``hf_overrides`` dict at the right key.
        "layers_under_text_config": "num_hidden_layers" in tcfg,
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
        "rope_type": rope_scaling.get("type") if isinstance(rope_scaling, dict) else None,
        # Hybrid dense/MoE: first N layers use dense MLP
        "first_k_dense_replace": first_k_dense,
        # MoE layer frequency (1 = every layer is MoE, 2 = every other, etc.)
        "moe_layer_freq": moe_layer_freq,
        # Qwen-style: decoder_sparse_step (1 = all MoE, 2 = alternating)
        "decoder_sparse_step": tget("decoder_sparse_step", 0),
        # MoE intermediate size (may differ from dense intermediate_size)
        "moe_intermediate_size": moe_intermediate_size,
        # Shared experts
        "n_shared_experts": n_shared_experts,
        # MLA (Multi-head Latent Attention) — DeepSeek-V2/V3/V4
        "kv_lora_rank": tget("kv_lora_rank"),
        "q_lora_rank": tget("q_lora_rank"),
        "qk_nope_head_dim": tget("qk_nope_head_dim"),
        "qk_rope_head_dim": tget("qk_rope_head_dim"),
        # Value head dim (may differ from head_dim, e.g. GLM5)
        "v_head_dim": tget("v_head_dim"),
        # V4 grouped low-rank output projection
        "o_lora_rank": tget("o_lora_rank"),
        "o_groups": tget("o_groups"),
        # DeepSeek-style sparse attention (lightning indexer) — MiniMax-M3
        "sparse_attention": use_sparse_attention,
        "sparse_index_dim": sparse_cfg.get("sparse_index_dim"),
        "sparse_num_index_heads": sparse_cfg.get("sparse_num_index_heads"),
        "sparse_topk_blocks": sparse_cfg.get("sparse_topk_blocks"),
        "sparse_block_size": sparse_cfg.get("sparse_block_size"),
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
