# SPDX-License-Identifier: Apache-2.0
"""Runtime patches for vLLM to enable proper XPU profiling.

vLLM's gpu_worker.py hardcodes `activities=["CPU", "CUDA"]` for the torch
profiler. On XPU hardware, `ProfilerActivity.CUDA` may be silently disabled,
resulting in no device-level events. This module patches the worker at import
time to use `["CPU", "XPU"]` when running on XPU.

Usage:
    from breakdown.xpu_patch import patch_vllm_xpu_profiler
    patch_vllm_xpu_profiler()   # call before creating LLM instance
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_patched = False


def patch_vllm_xpu_profiler() -> bool:
    """Monkey-patch vLLM's worker to use XPU profiler activities.

    Returns True if the patch was applied, False if not needed or failed.
    """
    global _patched
    if _patched:
        return True

    try:
        import torch
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            logger.debug("XPU not available, skipping profiler patch")
            return False
    except ImportError:
        return False

    try:
        from torch.profiler import ProfilerActivity
        if not hasattr(ProfilerActivity, "XPU"):
            logger.debug("ProfilerActivity.XPU not available")
            return False
    except ImportError:
        return False

    # Patch the TorchProfilerWrapper to use XPU activities
    try:
        from vllm.profiler.torch_profiler import TorchProfilerWrapper
        original_init = TorchProfilerWrapper.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Replace CUDA activity with XPU in the profiler activities
            if hasattr(self, "_profiler") and self._profiler is not None:
                activities = self._profiler.activities
                if (ProfilerActivity.CUDA in activities
                        and ProfilerActivity.XPU not in activities):
                    activities.discard(ProfilerActivity.CUDA)
                    activities.add(ProfilerActivity.XPU)
                    logger.info("Patched profiler activities: CUDA → XPU")

        TorchProfilerWrapper.__init__ = patched_init
        _patched = True
        logger.info("Applied vLLM XPU profiler patch")
        return True
    except ImportError:
        pass

    # Fallback: patch the worker's _setup_profiler directly
    try:
        import vllm.v1.worker.gpu_worker as gpu_worker_mod
        original_setup = gpu_worker_mod.Worker._setup_profiler

        def patched_setup(self, profiler_config):
            result = original_setup(self, profiler_config)
            # After setup, check if the profiler uses CUDA and swap to XPU
            if hasattr(self, "profiler") and self.profiler is not None:
                profiler = self.profiler
                if hasattr(profiler, "_profiler") and profiler._profiler:
                    activities = profiler._profiler.activities
                    if (ProfilerActivity.CUDA in activities
                            and ProfilerActivity.XPU not in activities):
                        activities.discard(ProfilerActivity.CUDA)
                        activities.add(ProfilerActivity.XPU)
                        logger.info(
                            "Patched worker profiler activities: CUDA → XPU")
            return result

        gpu_worker_mod.Worker._setup_profiler = patched_setup
        _patched = True
        logger.info("Applied vLLM worker profiler patch (fallback)")
        return True
    except (ImportError, AttributeError) as e:
        logger.warning("Failed to apply XPU profiler patch: %s", e)
        return False
