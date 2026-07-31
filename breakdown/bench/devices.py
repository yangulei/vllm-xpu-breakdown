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
SKU_PEAKS: dict[str, dict[str, float]] = {
    # Intel Arc Pro B60 / Battlemage (Xe2): 456 GB/s GDDR6, 98.3 TFLOPS bf16 XMX
    "BMG": {"bw_gbs": 456.0, "tflops": 98.3},
    # Crescent Island / Xe3P - placeholder until public numbers land
    "CRI": {"bw_gbs": 456.0, "tflops": 98.3},
    # NVIDIA RTX PRO 5000 Blackwell (the CUDA reference box)
    "BLACKWELL_RTX_PRO_5000": {"bw_gbs": 1344.0, "tflops": 250.0},
    "H100": {"bw_gbs": 3350.0, "tflops": 989.0},
    "A100": {"bw_gbs": 2039.0, "tflops": 312.0},
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
