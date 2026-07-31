# SPDX-License-Identifier: Apache-2.0
"""Device detection and roofline peaks for the replay benchmark.

``util = achieved / peak`` decides whether an op is worth an optimization
session, so the peaks must match the silicon actually being measured. XPU and
CUDA are equal first-class targets: the device kind is detected from torch and
the SKU from the device name.
"""
from __future__ import annotations

import os
import shutil

# Roofline peaks per SKU.
#
# ``cache_bytes`` / ``cache_bw_gbs`` describe the **last-level cache roof**. A
# replayed kernel whose whole footprint fits there is served by the cache on
# every repetition inside a timed window, so measuring it against DRAM makes it
# look like it beat the memory system (the old "utilization exceeds peak"
# warnings). The BMG numbers are *measured on an Intel Arc Pro B60* with a
# bf16 copy sweep: ~1.2 TB/s while both buffers fit in the 18 MB Xe2 L2/LLC,
# dropping to ~0.4 TB/s (DRAM) once they do not.
#
# ``tflops`` is the **matrix-engine** peak (XMX / Tensor Core) and
# ``vector_tflops`` the **vector-engine** peak (XVE / CUDA core). An op that
# never issues a matrix instruction - an RMSNorm, a gather, an elementwise
# activation - cannot reach the XMX peak, so charging it to that roof made
# every such op look like it had ~99 % headroom. The unit names are carried
# here too so a report can say *which* hardware unit is the roof (``XMX`` /
# ``XVE`` / ``DRAM`` / ``L3-Cache``) instead of the opaque "compute"/"memory".
# Intel Xe2: one Xe-core issues 8x more bf16 matrix FLOPs than vector FLOPs.
SKU_PEAKS: dict[str, dict[str, float]] = {
    # Intel Arc Pro B60 / Battlemage (Xe2): 456 GB/s GDDR6, 98.3 TFLOPS bf16 XMX
    "BMG": {"bw_gbs": 456.0, "tflops": 98.3, "vector_tflops": 12.3,
            "cache_bytes": 18 * 1024 ** 2, "cache_bw_gbs": 1200.0,
            "matrix_unit": "XMX", "vector_unit": "XVE",
            "cache_name": "L3-Cache"},
    # Crescent Island / Xe3P - placeholder until public numbers land
    "CRI": {"bw_gbs": 456.0, "tflops": 98.3, "vector_tflops": 12.3,
            "cache_bytes": 18 * 1024 ** 2, "cache_bw_gbs": 1200.0,
            "matrix_unit": "XMX", "vector_unit": "XVE",
            "cache_name": "L3-Cache"},
    # NVIDIA RTX PRO 5000 Blackwell (the CUDA reference box): 48 MB L2
    "BLACKWELL_RTX_PRO_5000": {"bw_gbs": 1344.0, "tflops": 250.0,
                               "vector_tflops": 31.2,
                               "cache_bytes": 48 * 1024 ** 2,
                               "cache_bw_gbs": 4000.0,
                               "matrix_unit": "Tensor", "vector_unit": "CUDA",
                               "cache_name": "L2-Cache"},
    "H100": {"bw_gbs": 3350.0, "tflops": 989.0, "vector_tflops": 67.0,
             "cache_bytes": 50 * 1024 ** 2, "cache_bw_gbs": 8000.0,
             "matrix_unit": "Tensor", "vector_unit": "CUDA",
             "cache_name": "L2-Cache"},
    "A100": {"bw_gbs": 2039.0, "tflops": 312.0, "vector_tflops": 19.5,
             "cache_bytes": 40 * 1024 ** 2, "cache_bw_gbs": 5000.0,
             "matrix_unit": "Tensor", "vector_unit": "CUDA",
             "cache_name": "L2-Cache"},
}

#: device-name substring -> SKU key, so a run self-identifies its roofline
_DEVICE_HINTS = (
    ("arc(tm) pro b", "BMG"),
    ("arc b", "BMG"),
    ("battlemage", "BMG"),
    ("crescent", "CRI"),
    ("rtx pro 5000", "BLACKWELL_RTX_PRO_5000"),
    ("h100", "H100"),
    ("a100", "A100"),
)

DEFAULT_SKU = "BMG"

#: torch device kinds the replay engine supports, in detection order.
DEVICE_KINDS = ("xpu", "cuda")


def sku_for_device(device_name: str | None, default: str = DEFAULT_SKU) -> str:
    """Best-effort SKU key for a torch device name."""
    name = (device_name or "").lower()
    for hint, sku in _DEVICE_HINTS:
        if hint in name:
            return sku
    return default


def peaks(sku: str) -> dict[str, float]:
    return SKU_PEAKS.get(sku, SKU_PEAKS[DEFAULT_SKU])


def detect_device(prefer: str | None = None) -> str:
    """``"xpu"`` / ``"cuda"`` / ``"cpu"`` - what replay can actually run on.

    ``prefer`` (``$BREAKDOWN_BENCH_DEVICE``) wins when that backend is really
    available; otherwise the first available accelerator is used. Kept
    torch-optional so the planning/ranking stages import on a GPU-less box.
    """
    want = prefer or os.environ.get("BREAKDOWN_BENCH_DEVICE")
    try:
        import torch
    except ImportError:
        return "cpu"
    avail = []
    for kind in DEVICE_KINDS:
        mod = getattr(torch, kind, None)
        try:
            if mod is not None and mod.is_available():
                avail.append(kind)
        except (RuntimeError, AssertionError):
            continue
    if want and want.split(":")[0] in avail:
        return want.split(":")[0]
    return avail[0] if avail else "cpu"


def device_name(kind: str, index: int = 0) -> str:
    try:
        import torch
    except ImportError:
        return ""
    mod = getattr(torch, kind, None)
    if mod is None:
        return ""
    try:
        return str(mod.get_device_name(index))
    except (RuntimeError, AssertionError, AttributeError):
        return ""


def device_count(kind: str) -> int:
    try:
        import torch
    except ImportError:
        return 0
    mod = getattr(torch, kind, None)
    try:
        return int(mod.device_count()) if mod is not None else 0
    except (RuntimeError, AttributeError):
        return 0


def has_unitrace() -> bool:
    return shutil.which("unitrace") is not None
