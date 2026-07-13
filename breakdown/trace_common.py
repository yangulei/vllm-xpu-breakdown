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

    Mirrors ``module_naming._display_name``: a ``ModuleList`` element (numeric
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
