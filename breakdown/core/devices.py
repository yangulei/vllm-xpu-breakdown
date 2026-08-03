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
from typing import Any, Iterable

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


#: Environment variable that restricts which physical devices a *child*
#: process sees. Applied to replay workers (and therefore to the peer ranks of
#: a collective), which is the only place a device selection can be honoured:
#: both runtimes read it at driver init, so it cannot be changed inside a
#: process that has already touched the device.
VISIBILITY_ENV = {"xpu": "ZE_AFFINITY_MASK", "cuda": "CUDA_VISIBLE_DEVICES"}


def available(kind: str | None = None) -> dict[str, Any]:
    """The devices that are actually present: ``{kind, indexes, names}``.

    The UI offers device *indexes*, not a free-text device string, so the
    selection can be checked against what exists instead of failing deep inside
    a worker with a driver error.
    """
    kind = kind or detect_device()
    count = device_count(kind)
    return {"kind": kind, "count": count, "indexes": list(range(count)),
            "names": [device_name(kind, i) for i in range(count)]}


def parse_device_ids(spec: Any) -> list[int]:
    """``"0, 2,3"`` / ``[0, 2]`` -> ``[0, 2, 3]``; empty -> ``[]`` (= all).

    Raises ``ValueError`` on a token that is not a device index.
    """
    if spec is None or spec == "":
        return []
    items = spec if isinstance(spec, (list, tuple)) else str(spec).split(",")
    out: list[int] = []
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            raise ValueError(f"'{token}' is not a device index") from None
        if idx < 0:
            raise ValueError(f"device index {idx} is negative")
        if idx not in out:
            out.append(idx)
    return out


def validate_device_ids(ids: Iterable[int], kind: str | None = None,
                        need: int = 0) -> str | None:
    """Why these device indexes cannot be used, or ``None`` if they can.

    ``need`` is the number of devices the request requires (a TP=4 run needs
    four), so an under-sized selection is refused up front rather than
    discovered when the collective fails to form.
    """
    kind = kind or detect_device()
    ids = list(ids)
    count = device_count(kind)
    if not count:
        return f"no {kind} devices are available on this host"
    missing = [i for i in ids if i >= count]
    if missing:
        plural = "s" if len(missing) > 1 else ""
        return (f"{kind} device{plural} "
                f"{', '.join(str(i) for i in missing)} not available - this "
                f"host has {count} ({', '.join(str(i) for i in range(count))})")
    selected = len(ids) or count
    if need and selected < need:
        return (f"{need} devices are required but {selected} "
                f"{'is' if selected == 1 else 'are'} selected")
    return None


def visibility_env(kind: str, ids: Iterable[int]) -> dict[str, str]:
    """Environment restricting a child process to ``ids`` (empty = no change)."""
    ids = list(ids)
    var = VISIBILITY_ENV.get(kind)
    if not ids or not var:
        return {}
    return {var: ",".join(str(i) for i in ids)}


def has_unitrace() -> bool:
    return shutil.which("unitrace") is not None
