# SPDX-License-Identifier: Apache-2.0
"""Op classifier — categorizes profiled ops by dispatch backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .core import opnames
from .registry import ALL_VLLM_XPU_OPS, get_op_category


class Backend(str, Enum):
    """Dispatch backend for an operation."""
    VLLM_XPU_KERNELS = "vllm-xpu-kernels"
    FLASHINFER = "flashinfer"
    FLASH_XPU = "flash_xpu"
    TRITON = "triton"
    TORCH_XPU_OPS = "torch-xpu-ops"
    CPU = "cpu"
    FRAMEWORK = "framework"
    CCL = "ccl"
    VLLM_CUDA_KERNELS = "vllm-cuda-kernels"
    TORCH_CUDA_OPS = "torch-cuda-ops"


@dataclass
class OpRecord:
    """A single profiled operation with classification."""
    name: str
    backend: Backend
    category: str  # more specific label within backend
    cpu_time_us: float = 0.0
    device_time_us: float = 0.0
    count: int = 1
    input_shapes: str = ""
    device_type: str = ""

    @property
    def total_time_us(self) -> float:
        return self.device_time_us if self.device_time_us > 0 else self.cpu_time_us


@dataclass
class ClassificationResult:
    """Full classification of a profiled run."""
    ops: list[OpRecord] = field(default_factory=list)
    total_device_time_us: float = 0.0
    total_cpu_time_us: float = 0.0

    @property
    def by_backend(self) -> dict[Backend, list[OpRecord]]:
        result: dict[Backend, list[OpRecord]] = {b: [] for b in Backend}
        for op in self.ops:
            result[op.backend].append(op)
        return result


def classify_op(name: str, device_type: str = "",
                self_device_time_us: float = 0.0,
                device_time_us: float = 0.0) -> tuple[Backend, str]:
    """Classify a single op by name and device context.

    The vocabulary -- which namespaces are vLLM's, which names are collectives,
    which kernel libraries fingerprint how, what counts as plumbing -- lives in
    :mod:`breakdown.core.opnames`. This function is only the *order* in which
    those questions are asked, and the order is load-bearing: see the comment
    on each step.

    Returns an ``(backend, category)`` pair.
    """
    stripped = opnames.split(name)[1] or name
    ns = opnames.namespace_of(name)
    has_device_time = self_device_time_us > 0 or device_time_us > 0
    is_xpu = device_type in ("xpu", "XPU")
    is_cuda = device_type in ("cuda", "CUDA")

    def _vllm(default: str) -> tuple[Backend, str]:
        cat = get_op_category(stripped) or default
        if is_cuda:
            return Backend.VLLM_CUDA_KERNELS, cat.replace("xpu", "cuda")
        return Backend.VLLM_XPU_KERNELS, cat

    def _aten() -> tuple[Backend, str]:
        if is_cuda:
            return Backend.TORCH_CUDA_OPS, "aten-cuda"
        return Backend.TORCH_XPU_OPS, "aten-xpu"

    # 1. Collectives are their own category, independent of compute backend:
    #    a tensor-parallel all-reduce is not "an XPU kernel that was slow", it
    #    is interconnect time, and mixing it into a compute backend hides that.
    if opnames.is_collective(name):
        return Backend.CCL, "collective-comm"

    # 2. vLLM's own kernels, by registry membership or by namespace. Shared ops
    #    such as rms_norm exist in both the CUDA and XPU builds, so the device
    #    decides which they are reported as.
    if stripped in ALL_VLLM_XPU_OPS:
        return _vllm("vllm-kernels")
    if ns == opnames.CUDA_KERNEL_NAMESPACE:
        cat = get_op_category(stripped) or "vllm-cuda-kernels"
        return Backend.VLLM_CUDA_KERNELS, cat
    if ns in opnames.VLLM_KERNEL_NAMESPACES:
        return _vllm("vllm-kernels")

    # 3. Kernel libraries, fingerprinted from the symbol. FlashInfer and
    #    xattention must be probed before Triton: a Python-launched kernel has
    #    no cpu_op, so its synthetic name is triton::-prefixed even when the
    #    kernel inside it is hand-written SYCL.
    library = opnames.library_of(name)
    if library == "flashinfer":
        return Backend.FLASHINFER, "flashinfer-kernel"
    if library == "flash_xpu":
        return Backend.FLASH_XPU, "xattention-kernel"
    if library == "triton":
        return Backend.TRITON, "triton-compiled"

    # 4. Plumbing. is_framework consults the compute set first, so a real
    #    kernel is never lost to a prefix it merely starts with.
    if opnames.is_framework(name):
        return Backend.FRAMEWORK, "framework-overhead"

    # 5. ATen compute, either by name or -- for anything the name list has not
    #    caught up with -- by the fact that it spent time on the device.
    if opnames.is_aten_compute(name):
        if is_cuda:
            return Backend.TORCH_CUDA_OPS, "aten-cuda"
        if is_xpu or has_device_time:
            return Backend.TORCH_XPU_OPS, "aten-xpu"
        return Backend.CPU, "aten-cpu"
    if ns == "aten":
        return _aten() if has_device_time else (Backend.FRAMEWORK,
                                                "aten-overhead")

    # 6. Anything left that never reached the device.
    if device_type in ("cpu", "CPU") or not has_device_time:
        return Backend.CPU, "cpu"
    return Backend.FRAMEWORK, "unknown"
