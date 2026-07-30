# SPDX-License-Identifier: Apache-2.0
"""Device peaks and external-tool discovery for the perf pipeline."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Roofline peaks per SKU. ``util = achieved / peak`` decides whether an op is
# worth an optimization session, so these must match the silicon being measured.
SKU_PEAKS: dict[str, dict[str, float]] = {
    # Intel Arc Pro B60 / Battlemage (Xe2): 456 GB/s GDDR6, 98.3 TFLOPS bf16 XMX
    "BMG": {"bw_gbs": 456.0, "tflops": 98.3},
    # Crescent Island / Xe3P - placeholder until public numbers land
    "CRI": {"bw_gbs": 456.0, "tflops": 98.3},
    # NVIDIA RTX PRO 5000 Blackwell (the CUDA reference box)
    "BLACKWELL_RTX_PRO_5000": {"bw_gbs": 1344.0, "tflops": 250.0},
}

# device name substring -> SKU key, so a report tree self-identifies
_DEVICE_HINTS = (
    ("arc(tm) pro b", "BMG"),
    ("arc b", "BMG"),
    ("battlemage", "BMG"),
    ("crescent", "CRI"),
    ("rtx pro 5000", "BLACKWELL_RTX_PRO_5000"),
)

DEFAULT_SKU = "BMG"


def sku_for_device(device_name: str | None, default: str = DEFAULT_SKU) -> str:
    """Best-effort SKU key for a micro_perf ``sku_name``."""
    name = (device_name or "").lower()
    for hint, sku in _DEVICE_HINTS:
        if hint in name:
            return sku
    return default


def peaks(sku: str) -> dict[str, float]:
    return SKU_PEAKS.get(sku, SKU_PEAKS[DEFAULT_SKU])


def xpu_perf_home() -> Path | None:
    """Locate the xpu-perf checkout (``$XPU_PERF_HOME``, else workspace guess).

    The pipeline *invokes* xpu-perf; it never vendors it, so the path is
    configuration, not a dependency.
    """
    env = os.environ.get("XPU_PERF_HOME")
    if env and (Path(env) / "projects" / "micro_perf").is_dir():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "xpu-perf"
        if (cand / "projects" / "micro_perf").is_dir():
            return cand
    return None


def micro_perf_dir() -> Path | None:
    home = xpu_perf_home()
    return home / "projects" / "micro_perf" if home else None


def oneapi_setvars() -> Path | None:
    p = Path("/opt/intel/oneapi/setvars.sh")
    return p if p.is_file() else None


def has_unitrace() -> bool:
    return shutil.which("unitrace") is not None
