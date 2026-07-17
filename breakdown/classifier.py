# SPDX-License-Identifier: Apache-2.0
"""Op classifier — categorizes profiled ops by dispatch backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


# Collective-communication (oneCCL / NCCL) op indicators. These are the
# distributed tensor-parallel / pipeline-parallel comm calls (all_reduce,
# all_gather, reduce_scatter, all_to_all, ...) that should be attributed to the
# communication library rather than to a compute backend. Bare "gather"/
# "scatter" are intentionally excluded so cache/MoE gather ops aren't misfiled.
_CCL_NAMESPACES = ("c10d::", "ccl::", "oneccl::", "nccl::")
_CCL_KEYWORDS = (
    "all_reduce", "allreduce",
    "all_gather", "allgather",
    "reduce_scatter", "reducescatter",
    "all_to_all", "alltoall",
    "_allgather_base", "_reduce_scatter_base",
)


def _is_ccl(name: str) -> bool:
    """Return True if the op is a collective-communication (CCL) call."""
    low = name.lower()
    if low.startswith(_CCL_NAMESPACES):
        return True
    return any(kw in low for kw in _CCL_KEYWORDS)


# Op name substrings that indicate FlashInfer kernels. FlashInfer RMSNorm /
# fused-add-RMSNorm / attention kernels are launched directly from Python (no
# ``aten``/``_C`` cpu_op), so they surface as synthetic kernel ops whose name
# embeds ``flashinfer`` (e.g.
# ``kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_...``).
_FLASHINFER_INDICATORS = (
    "flashinfer",
)

# Op name substrings that indicate xattention SYCL kernels (MiniMax-M3 MSA on
# XPU). The lightning-indexer (block score + top-k) and block-sparse GQA attend
# kernels live in the ``xattention._C`` (``flash_xpu``) extension and are
# launched directly from Python via the ``xattention.py`` wrappers (no
# ``aten``/``_C`` cpu_op), so they surface as synthetic kernel ops whose name
# embeds ``flash_xpu`` (e.g. ``flash_xpu::(anonymous namespace)::
# index_score_kernel_t``, ``flash_xpu::msa_index_topk_xpu(...)``).
_FLASH_XPU_INDICATORS = (
    "flash_xpu",
)

# Op name prefixes/substrings that indicate Triton-compiled kernels
_TRITON_INDICATORS = (
    "triton_",
    "triton::",
    "Triton",
    "_triton_",
    "tt.",
    "CompiledFxGraph",
)

# Framework/overhead ops that are not real compute
_FRAMEWORK_PREFIXES = (
    "profiler::",
    "autograd::",
    "torch::autograd::",
    "aten::empty",
    "aten::zeros",
    "aten::ones",
    "aten::to",
    "aten::copy_",
    "aten::contiguous",
    "aten::view",
    "aten::reshape",
    "aten::expand",
    "aten::permute",
    "aten::transpose",
    "aten::slice",
    "aten::select",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::narrow",
    "aten::cat",
    "aten::stack",
    "aten::split",
    "aten::chunk",
    "aten::flatten",
    "aten::unflatten",
    "aten::detach",
    "aten::clone",
    "aten::_unsafe_view",
    "aten::as_strided",
    "aten::t",
    "aten::_to_copy",
    "record_function",
)

# ATen ops that dispatch to real XPU compute (torch-xpu-ops / oneDNN)
_ATEN_COMPUTE_OPS = {
    "aten::linear",
    "aten::mm",
    "aten::bmm",
    "aten::addmm",
    "aten::matmul",
    "aten::_scaled_mm",
    "aten::conv1d",
    "aten::conv2d",
    "aten::embedding",
    "aten::layer_norm",
    "aten::batch_norm",
    "aten::group_norm",
    "aten::softmax",
    "aten::_softmax",
    "aten::log_softmax",
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::gelu",
    "aten::relu",
    "aten::silu",
    "aten::sigmoid",
    "aten::tanh",
    "aten::mul",
    "aten::add",
    "aten::sub",
    "aten::div",
    "aten::sum",
    "aten::mean",
    "aten::max",
    "aten::min",
    "aten::pow",
    "aten::rsqrt",
    "aten::sqrt",
    "aten::exp",
    "aten::log",
    "aten::where",
    "aten::index_select",
    "aten::gather",
    "aten::scatter",
    "aten::scatter_",
    "aten::topk",
    "aten::sort",
    "aten::argmax",
    "aten::argmin",
    "aten::cumsum",
    "aten::arange",
    "aten::fill_",
    "aten::index_put_",
    "aten::masked_fill_",
}


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


def _strip_namespace(name: str) -> str:
    """Strip namespace prefix from op name: '_C::rms_norm' -> 'rms_norm'."""
    if "::" in name:
        return name.split("::")[-1]
    return name


def classify_op(name: str, device_type: str = "",
                self_device_time_us: float = 0.0,
                device_time_us: float = 0.0) -> tuple[Backend, str]:
    """Classify a single op by name and device context.

    Args:
        name: Op name from profiler (e.g. "aten::mm", "rms_norm")
        device_type: Device type string (e.g. "xpu", "cuda", "cpu")
        self_device_time_us: Self device time (excludes children)
        device_time_us: Total device time (includes children)

    Returns (backend, category) tuple.
    """
    stripped = _strip_namespace(name)
    has_device_time = self_device_time_us > 0 or device_time_us > 0
    is_xpu = device_type in ("xpu", "XPU")
    is_cuda = device_type in ("cuda", "CUDA")

    # 0. Collective communication (oneCCL / NCCL) — tensor/pipeline-parallel
    #    comm calls form their own category, independent of compute backend.
    if _is_ccl(name):
        return Backend.CCL, "collective-comm"

    # 1. Check against vllm custom-kernels registry (shared ops like rms_norm
    #    exist on both CUDA and XPU builds of vLLM)
    if stripped in ALL_VLLM_XPU_OPS:
        cat = get_op_category(stripped) or "vllm-kernels"
        if is_cuda:
            return Backend.VLLM_CUDA_KERNELS, cat.replace("xpu", "cuda")
        return Backend.VLLM_XPU_KERNELS, cat

    # Also check full name patterns for vllm custom ops
    for prefix in ("_C::", "_C_cache_ops::", "_moe_C::", "_xpu_C::", "vllm::"):
        if name.startswith(prefix):
            cat = get_op_category(stripped) or "vllm-kernels"
            if is_cuda:
                return Backend.VLLM_CUDA_KERNELS, cat.replace("xpu", "cuda")
            return Backend.VLLM_XPU_KERNELS, cat

    # 1b. CUDA-specific vllm kernel namespace
    if name.startswith("_cuda_C::"):
        cat = get_op_category(stripped) or "vllm-cuda-kernels"
        return Backend.VLLM_CUDA_KERNELS, cat

    # 2. Check for FlashInfer kernels (before Triton — the synthetic op name is
    #    ``triton::``-prefixed but the embedded kernel symbol is FlashInfer's).
    low = name.lower()
    for indicator in _FLASHINFER_INDICATORS:
        if indicator in low:
            return Backend.FLASHINFER, "flashinfer-kernel"

    # 2b. Check for xattention SYCL kernels (MiniMax-M3 MSA on XPU) — before
    #     Triton, since the fallback synthetic name would otherwise be
    #     ``triton::``-prefixed while these are hand-tuned SYCL kernels.
    for indicator in _FLASH_XPU_INDICATORS:
        if indicator in low:
            return Backend.FLASH_XPU, "xattention-kernel"

    # 3. Check for Triton kernels
    for indicator in _TRITON_INDICATORS:
        if indicator in name:
            return Backend.TRITON, "triton-compiled"

    # 4. Check for framework overhead (before aten compute check)
    for prefix in _FRAMEWORK_PREFIXES:
        if name.startswith(prefix):
            return Backend.FRAMEWORK, "framework-overhead"

    # 5. ATen compute ops on accelerator → torch-{cuda,xpu}-ops
    if name in _ATEN_COMPUTE_OPS or stripped in _ATEN_COMPUTE_OPS:
        if is_cuda:
            return Backend.TORCH_CUDA_OPS, "aten-cuda"
        elif is_xpu or has_device_time:
            return Backend.TORCH_XPU_OPS, "aten-xpu"
        else:
            return Backend.CPU, "aten-cpu"

    # 6. Any aten:: op with device time → torch-{cuda,xpu}-ops
    if name.startswith("aten::") and has_device_time:
        if is_cuda:
            return Backend.TORCH_CUDA_OPS, "aten-cuda"
        return Backend.TORCH_XPU_OPS, "aten-xpu"

    # 7. Any aten:: op without device time → framework
    if name.startswith("aten::"):
        return Backend.FRAMEWORK, "aten-overhead"

    # 8. CPU-only ops
    if device_type in ("cpu", "CPU") or not has_device_time:
        return Backend.CPU, "cpu"

    return Backend.FRAMEWORK, "unknown"
