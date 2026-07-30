# SPDX-License-Identifier: Apache-2.0
"""Shape-Matrix op -> micro_perf op case mapping, per platform dispatch.

``classifier.py`` answers *which backend runs this op*; the op map answers
*which micro_perf op benchmarks it, with which arguments*. Both describe the
same dispatch, which is why they live in one repo.

An adapter takes an :class:`~breakdown.perf.matrix_reader.OpRow` plus a
:class:`~breakdown.perf.op_map.common.ModelConfig` and returns zero or more
:class:`~breakdown.perf.op_map.common.EmittedCase`. Ops with no equivalent are
surfaced in the coverage report rather than silently approximated.
"""
from __future__ import annotations

from types import ModuleType

from breakdown.perf.op_map.common import EmittedCase, M3Config, ModelConfig

DISPATCHES = ("xpu", "cuda")


def get_dispatch(name: str) -> ModuleType:
    """Return the op-map module for a platform dispatch (``xpu`` / ``cuda``)."""
    if name == "xpu":
        from breakdown.perf.op_map import xpu as mod
    elif name == "cuda":
        from breakdown.perf.op_map import cuda as mod
    else:
        raise ValueError(f"unknown dispatch {name!r}; expected one of "
                         f"{DISPATCHES}")
    return mod


__all__ = ["DISPATCHES", "EmittedCase", "M3Config", "ModelConfig",
           "get_dispatch"]
