# SPDX-License-Identifier: Apache-2.0
"""Run ONE micro_perf op case in its own process - bench, profile or validate.

Single source of truth for op inputs: tensors come from the same ``vendor_ops``
implementation ``launch.py`` uses, so a number measured here is directly
comparable to the report tree and an optimization session never needs its own
benchmark harness (which is how shape drift creeps in).

Three uses:

* **bench** - warm up, then time ``--repeat`` batched launches and print the
  median. This is the ``--bench-cmd`` for the ``xpu-kernel-optimizer`` skill;
  it prints a machine-readable ``latency_us=`` line.
* **profile** - the same command under ``unitrace``; ``--repeat`` amortises the
  per-launch submit overhead so the kernel dominates the capture.
* **isolation** - one case per process, so a device fault cannot cascade into
  the rest of a sweep.

torch and the micro_perf registry are imported lazily, so importing this module
costs nothing on a GPU-less box.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Any

from breakdown.perf import devices


def load_case(spec: str) -> dict:
    """Case arguments from a JSON string or ``@file.json``."""
    if spec.startswith("@"):
        with open(spec[1:]) as fh:
            return json.load(fh)
    return json.loads(spec)


def _provider_registry(backend: str):
    micro_perf = devices.micro_perf_dir()
    if micro_perf is None:
        raise FileNotFoundError(
            "xpu-perf/projects/micro_perf not found - set $XPU_PERF_HOME")
    backends_dir = str(micro_perf / "backends")
    if backends_dir not in sys.path:
        sys.path.insert(0, backends_dir)
    try:
        from xpu_perf.micro_perf.core.op import ProviderRegistry
    except ImportError as exc:  # pragma: no cover - depends on the checkout
        raise ImportError(
            f"could not import micro_perf from {micro_perf}: {exc}. The "
            "xpu-perf checkout may be too old or not installed "
            "(pip install -e xpu-perf)") from exc
    ProviderRegistry.load_all_vendor_impls(
        micro_perf / "op_defs", [micro_perf / "vendor_ops" / backend / "ops"])
    return ProviderRegistry


def bench(op: str, case: dict, backend: str = "INTEL",
          provider: str | None = None, repeat: int = 20, reps: int = 10,
          warmup: int = 3) -> list[dict[str, Any]]:
    """Time one case for one (or every) provider registered for the op."""
    import torch  # lazy: keeps the module importable without a GPU stack

    registry = _provider_registry(backend)
    dev = "cuda" if backend == "GPU" else "xpu"

    class _Backend:
        def get_torch_device_name(self):
            return dev

    providers = registry.OP_MAPPING[op]
    if provider in (None, ""):
        names = [next(iter(providers))]
    elif provider == "all":
        names = list(providers)
    else:
        names = [provider]

    sync = torch.cuda.synchronize if dev == "cuda" else torch.xpu.synchronize
    out: list[dict[str, Any]] = []
    for name in names:
        inst = providers[name](case, _Backend())
        tensors = inst.create_tensors(1)[0]
        for _ in range(warmup):
            inst.core_run(tensors)
        sync()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            for _ in range(repeat):
                inst.core_run(tensors)
            sync()
            samples.append((time.perf_counter_ns() - t0) / 1e3 / repeat)
        us = statistics.median(samples)
        io = getattr(inst, "io_bytes", 0) or 0
        flops = getattr(inst, "calc_flops", 0) or 0
        out.append({
            "op": op, "provider": name, "backend": backend, "arguments": case,
            "latency_us": round(us, 3),
            "min_us": round(min(samples), 3),
            "spread_pct": round((max(samples) - min(samples)) / us * 100, 1),
            "mem_bw_GBs": round(io / (us / 1e6) / 1e9, 1) if io and us else None,
            "tflops": round(flops / (us / 1e6) / 1e12, 2) if flops and us else None,
        })
    return out


def format_result(rec: dict[str, Any]) -> str:
    """The line the optimizer skill parses for its metric."""
    return (f"OK {rec['op']}[{rec['provider']}] latency_us={rec['latency_us']} "
            f"mem_bw_GBs={rec['mem_bw_GBs']} tflops={rec['tflops']} "
            f"spread={rec['spread_pct']}%")


def write_json(records: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)
