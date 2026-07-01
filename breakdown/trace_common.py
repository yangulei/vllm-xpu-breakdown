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
    # Raw SYCL kernel names: contain template brackets or are pure C++ symbols
    if "<" in name and "::" in name and not name.startswith(("aten::", "_C::",
            "_C_cache_ops::", "_moe_C::", "_xpu_C::", "triton")):
        return True
    # Bare kernel function names (gemm_kernel, etc.) — no aten:: prefix
    if name in ("gemm_kernel", "gemm_batch_kernel"):
        return True
    return False
