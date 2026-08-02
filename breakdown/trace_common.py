# SPDX-License-Identifier: Apache-2.0
"""Torch-free helpers shared by trace parsing / reconstruction.

Kept free of any PyTorch/vLLM imports so that static analysis and offline trace
reconstruction work without an ML stack installed.
"""

from __future__ import annotations

# Events that are profiler infrastructure, not real ops
_OVERHEAD_EVENTS = {
    "ProfilerStep*",
    "Optimizer.step#SGD.step",
    "Optimizer.step#Adam.step",
    "Optimizer.step#AdamW.step",
    "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__",
}

_OVERHEAD_PREFIXES = (
    "profiler::",
    "autograd::engine",
    "torch::autograd::",
)


# ===================================================================
# Capture-time module spans (research R1)
# ===================================================================
#
# The torch profiler only labels ``nn.Module`` forward frames with their
# *class* (``nn.Module: RMSNorm_2``), so same-class siblings (``q_norm`` vs
# ``k_norm``) are indistinguishable and previously had to be recovered by
# aligning a ``named_modules()`` reference tree onto the trace afterwards.
#
# Instead we open a ``record_function`` span named ``module::<qualified_name>``
# inside a ``register_forward_pre_hook`` / ``register_forward_hook`` pair on every
# module (see :mod:`breakdown.module_hooks`). These emit ``user_annotation``
# trace events that nest by time-containment exactly like the module forwards
# and carry the **real attribute path** (``model.layers.0.self_attn.q_norm``),
# so the reconstruction reads exact names straight from the trace — no overlay,
# no registration-order assumption. The label also embeds the class so the
# reconstructed node keeps a ``module_type`` for signatures/extrapolation.
#
# Label grammar: ``module::<qualified_name>::<ClassName>``. Qualified names are
# dot-separated attribute paths and never contain ``::``; the class is a bare
# identifier; so the class is recovered with a single ``rpartition("::")``.

MODULE_SPAN_PREFIX = "module::"


def module_span_label(qualified_name: str, cls: str) -> str:
    """Build the ``record_function`` label for a module's forward span."""
    return f"{MODULE_SPAN_PREFIX}{qualified_name}::{cls}"


def parse_module_span(name: str) -> tuple[str, str] | None:
    """Parse a ``module::<qname>::<Cls>`` label into ``(qualified_name, cls)``.

    Returns ``None`` when ``name`` is not a module span. The class may be empty
    for labels written without one (``module::<qname>``).
    """
    if not name.startswith(MODULE_SPAN_PREFIX):
        return None
    body = name[len(MODULE_SPAN_PREFIX):]
    qname, sep, cls = body.rpartition("::")
    if not sep:
        return body, ""
    return qname, cls


def module_span_display_name(qualified_name: str, cls: str) -> str:
    """Human-friendly node name for a module identified by its attribute path.

    A ``ModuleList`` element (numeric
    leaf, e.g. ``model.layers.0``) is a repeated-group representative, so it takes
    the list attribute (``layers``) — or ``decoder_layer`` when the class is a
    decoder layer/block — rather than its index, so structurally-identical
    siblings collapse. Any other module uses its own attribute (leaf) name. The
    root (empty ``qualified_name``) returns ``""`` so callers fall back to a
    class-based display name.
    """
    if not qualified_name:
        return ""
    leaf = qualified_name.rpartition(".")[2]
    if leaf.isdigit():
        low = (cls or "").lower()
        if "decoderlayer" in low or low.endswith("layer") or "block" in low:
            return "decoder_layer"
        parent = qualified_name.rpartition(".")[0].rpartition(".")[2]
        return parent or "layer"
    return leaf


def _is_overhead_event(name: str) -> bool:
    """Return True if this event is profiler/framework overhead to filter out."""
    if name in _OVERHEAD_EVENTS:
        return True
    for prefix in _OVERHEAD_PREFIXES:
        if name.startswith(prefix):
            return True
    # Filter out low-level XPU/SYCL kernel events and runtime calls.
    # These are children of aten:: ops and including them double-counts time.
    if name.startswith(("ur", "ze")) and not name.startswith("aten::"):
        # Level Zero / Unified Runtime calls (urEnqueueKernelLaunch, etc.)
        if any(c.isupper() for c in name[2:5]):
            return True
    if name.startswith("at::native::xpu::"):
        return True
    # Filter out CUDA runtime / driver API calls (children of aten:: ops).
    if name.startswith(("cudaLaunch", "cudaMalloc", "cudaFree",
                        "cudaMemcpy", "cudaMemset", "cudaStream",
                        "cudaEvent", "cudaDeviceSynchronize")):
        return True
    if name.startswith("at::native::") and "cuda" in name.lower():
        return True
    # Raw SYCL kernel names: contain template brackets or are pure C++ symbols
    if "<" in name and "::" in name and not name.startswith(("aten::", "_C::",
            "_C_cache_ops::", "_moe_C::", "_xpu_C::", "_cuda_C::", "triton")):
        return True
    # Bare kernel function names (gemm_kernel, etc.) — no aten:: prefix
    if name in ("gemm_kernel", "gemm_batch_kernel"):
        return True
    return False


# ---------------------------------------------------------------------------
# Device detection and module/op role heuristics
#
# These moved here from the retired ``trace_parser`` module, whose flat-op
# pipeline (a second, cruder device-time attribution by proportional CPU time)
# was superseded by the graph reconstruction. Only these torch-free helpers
# were still used.
# ---------------------------------------------------------------------------

def _detect_device_via_torch() -> str:
    """Detect the active accelerator via the torch device APIs.

    Returns ``"xpu"``, ``"cuda"`` or ``""``. Torch is imported lazily so this
    module stays import-light; any failure (torch absent, driver error) simply
    yields ``""`` so callers can fall back to other heuristics.
    """
    try:
        import torch
    except Exception:
        return ""
    try:
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return ""


# Trace event categories that unambiguously identify the accelerator.

_XPU_CATEGORIES = {"xpu_runtime", "xpu_op"}
_CUDA_CATEGORIES = {"cuda_runtime", "cuda_op", "cuda_driver"}
# Generic GPU categories that don't distinguish CUDA from XPU on their own.
_GENERIC_GPU_CATEGORIES = {"kernel", "gpu_op", "gpu_memcpy", "gpu_kernel"}


def _infer_device_from_trace(events: list[dict]) -> str:
    """Infer accelerator type from trace events.

    XPU traces emit ``xpu_runtime`` host-launch events (and ``xpu_op``), while
    CUDA traces emit ``cuda_runtime`` / ``cuda_op``. XPU kernel events, however,
    share the generic ``kernel`` category with CUDA, so we must key off the
    device-specific *runtime* categories rather than the kernel category.

    Resolution order:
      1. Device-specific trace categories (``xpu_*`` → xpu, ``cuda_*`` → cuda).
      2. If only generic GPU kernels are present, auto-detect via the torch
         device APIs (``torch.xpu.is_available()`` / ``torch.cuda.is_available()``).
      3. Otherwise, return ``""``.
    """
    saw_generic_gpu = False
    for evt in events:
        cat = evt.get("cat", "")
        if cat in _XPU_CATEGORIES:
            return "xpu"
        if cat in _CUDA_CATEGORIES:
            return "cuda"
        if cat in _GENERIC_GPU_CATEGORIES:
            saw_generic_gpu = True

    # Ambiguous: generic GPU kernels with no device-specific runtime events.
    # Auto-detect the accelerator from the running torch build.
    if saw_generic_gpu:
        detected = _detect_device_via_torch()
        if detected:
            return detected
    return ""



def _strip_instance_idx(class_name: str) -> str:
    """Strip trailing _N index: 'QKVParallelLinear_0' → 'QKVParallelLinear'."""
    parts = class_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return class_name


def _infer_role(module_path: list[str], op_name: str) -> str | None:
    """Infer the op role from its enclosing module hierarchy.

    Args:
        module_path: List of nn.Module class names (outermost first),
                     e.g. ['OPTDecoderLayer', 'OPTAttention', 'QKVParallelLinear']
        op_name: The operator name, e.g. 'aten::linear', 'aten::mm'

    Returns:
        Role string matching static graph op roles, or None if not identifiable.
    """
    if not module_path:
        return None

    innermost = module_path[-1]
    parent = module_path[-2] if len(module_path) >= 2 else ""

    # Embedding
    if "VocabParallelEmbedding" in innermost or "Embedding" in innermost:
        return "embedding"

    # QKV projection
    if "QKVParallel" in innermost:
        return "qkv_proj"

    # Rotary embedding
    if "Rotary" in innermost or "rotary" in op_name.lower():
        return "rotary_emb"

    # Q/K norms (Qwen3-style QK normalization)
    if "Norm" in innermost:
        # Check if inside Attention module → likely q_norm or k_norm
        in_attention = any("Attention" in p for p in module_path[:-1])
        if in_attention:
            # Distinguish by module name hints
            lower_inner = innermost.lower()
            if "q_norm" in lower_inner or "q_layernorm" in lower_inner:
                return "q_norm"
            if "k_norm" in lower_inner or "k_layernorm" in lower_inner:
                return "k_norm"
            # Fallback: generic attention norm (will be disambiguated later)
            return "attention_norm"
        # Determine which norm based on position in layer
        return "norm"

    # Attention kernel / cache ops
    if "Attention" in innermost and "Attention" not in _strip_instance_idx(innermost).replace("Attention", "", 1):
        # innermost IS an Attention module (not just contains "Attention" as part of larger name)
        if "attention" in op_name.lower() or "flash_attn" in op_name.lower():
            return "attention"
        if "cache" in op_name.lower():
            return "cache_store"
        if "rotary" in op_name.lower():
            return "rotary_emb"
        if "norm" in op_name.lower():
            return "attention_norm"
        if op_name in ("aten::linear", "aten::mm", "aten::addmm"):
            return "attention"  # fallback for attention internal ops
        return "attention"

    # Row/Column parallel inside Attention → o_proj
    if "RowParallel" in innermost and any("Attention" in p for p in module_path[:-1]):
        return "o_proj"

    # Row/Column parallel inside MLP → down_proj / gate_up_proj
    if "RowParallel" in innermost and any("MLP" in p or "Mlp" in p or "MoE" in p for p in module_path[:-1]):
        return "down_proj"
    if ("ColumnParallel" in innermost or "MergedColumn" in innermost) and \
       any("MLP" in p or "Mlp" in p or "MoE" in p for p in module_path[:-1]):
        return "gate_up_proj"

    # Generic ColumnParallel at decoder layer level (MLP without MLP wrapper, like OPT)
    if "ColumnParallel" in innermost or "MergedColumn" in innermost:
        return "gate_up_proj"
    if "RowParallel" in innermost:
        # Disambiguation: check if parent is attention-like
        if any("Attention" in p for p in module_path[:-1]):
            return "o_proj"
        return "down_proj"

    # Logits / LM head
    if "Logits" in innermost or "lm_head" in innermost.lower():
        return "lm_head"

    # Activation functions
    if "silu" in op_name.lower() or "gelu" in op_name.lower() or "relu" in op_name.lower():
        return "activation"

    # Cache ops outside Attention module
    if "cache" in op_name.lower():
        return "cache_store"

    return None
