# SPDX-License-Identifier: Apache-2.0
"""Torch-free helpers shared by trace parsing and reconstruction.

Kept free of any PyTorch/vLLM imports so that static analysis and offline trace
reconstruction work without an ML stack installed -- and, since this module is
the *shared* half, free of model vocabulary too. What lives here is how a span
is encoded and decoded, how a launcher frame is recognized, and which events
are profiler infrastructure: facts about the trace format, not about vLLM.

The model vocabulary lives in :mod:`breakdown.trace.rules`. ``_infer_role``
used to sit here and matched on ``QKVParallelLinear``, ``RowParallelLinear``,
``VocabParallelEmbedding`` and a dozen more -- exactly the coupling this
module's contract says it does not have.
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


# ===================================================================
# Kernel-launch spans (``kernel::<json>``)
# ===================================================================
# A kernel launched straight from Python — a Triton ``JITFunction``, a pybind11
# extension entry point — emits no ``cpu_op``, so the trace records neither its
# operands nor the function that launched it. ``breakdown.kernel_hooks`` opens a
# ``record_function`` span at the launch carrying both, as base64-encoded JSON:
#
#     kernel::<base64(json)>
#
# Base64 rather than raw JSON because torch's chrome-trace writer emits an
# event's name **unescaped**: a quote in the label produces a trace file that is
# not valid JSON. The payload is ``{"file","line","func","args":[<slot>, ...]}``
# and the slots use the same schema as ``graph_from_trace._parse_input_args``
# (kind ``tensor`` / ``tensorlist`` / ``scalar`` / ``none`` / ``opaque``), so the
# replay benchmark consumes a Python-launched kernel exactly like a dispatched
# one.

KERNEL_SPAN_PREFIX = "kernel::"


def kernel_span_label(payload: dict) -> str:
    """Build the ``record_function`` label for a kernel-launch span."""
    import base64
    import json
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return KERNEL_SPAN_PREFIX + base64.b64encode(blob).decode("ascii")


def parse_kernel_span(name: str) -> dict | None:
    """Parse a ``kernel::<base64(json)>`` label, or ``None`` if it is not one."""
    if not name.startswith(KERNEL_SPAN_PREFIX):
        return None
    import base64
    import binascii
    import json
    try:
        blob = base64.b64decode(name[len(KERNEL_SPAN_PREFIX):], validate=True)
        payload = json.loads(blob)
    except (ValueError, binascii.Error):
        return None
    return payload if isinstance(payload, dict) else None


#: Python packages that only *dispatch* a kernel launch rather than define it:
#: torch's own machinery, Triton's runtime, and the stdlib. The frame that
#: launched a kernel is the innermost enclosing frame outside these.
LAUNCH_MACHINERY = (
    "/torch/", "/triton/", "/_dynamo/", "/_inductor/",
    "contextlib.py", "<built-in>", "<string>",
)

#: Function names that identify a *dispatch* wrapper rather than the kernel: a
#: JIT runner, a callable object's entry point. They name the mechanism, not the
#: work, and their file is the JIT layer rather than the kernel's source, so
#: they are useless both as a label and as a replay entry point.
DISPATCH_FUNCS = frozenset({
    "run", "launch", "call", "__call__", "execute", "invoke", "wrapper",
})


def is_launcher_frame(file: str, func: str) -> bool:
    """Can ``file``/``func`` be the *definition site* of a kernel launch?

    The single rule shared by the capture-time hook (which walks the live
    Python stack) and the trace reader (which walks recorded
    ``python_function`` events), so a span and a recovered frame always name
    the same function.
    """
    if not func or func.startswith("_") or func.endswith(">"):
        return False                     # private helper or <listcomp>/<lambda>
    if func in DISPATCH_FUNCS:
        return False
    return not any(fragment in file for fragment in LAUNCH_MACHINERY)


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


