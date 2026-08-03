# SPDX-License-Identifier: Apache-2.0
"""Per-op replay recipes: the exceptions to schema-driven argument building.

:mod:`breakdown.bench.inputs` builds a call from the op's schema plus the
recorded slots, which covers the great majority of dispatched ops. A *recipe*
is the escape hatch for the rest, and there is exactly one record per op - see
:mod:`.table` for what a recipe may say and why each field exists. This module
is the registration API and the one place a call is materialized.

Recipes are grouped by what they are *about* -- :mod:`.attention`, :mod:`.moe`,
:mod:`.sampling`, and :mod:`.common` for the operand shapes any op can have --
not by device. They used to be split into ``xpu`` and ``cuda``, which looked
like a device split but was not one: a synthesizer is registered by *argument
name* into a single global table, so a CUDA run inherited the XPU
registrations regardless. All are imported here; registering is cheap and
import-safe, since nothing touches a device.
"""
from __future__ import annotations

from typing import Any, Callable

from breakdown.bench import inputs as _inputs
from breakdown.bench.recipes.table import RECIPES, OpRecipe, recipe, register
from breakdown.bench.resolve import Resolved


def override(op: str) -> Callable[[Callable[..., list[Any]]],
                                  Callable[..., list[Any]]]:
    """Register a full argument override for ``op``."""

    def deco(fn: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
        register(op, build=fn)
        return fn

    return deco


def skip(op: str, reason: str) -> None:
    """Declare that ``op`` must not be replayed, and why."""
    register(op, skip=reason)


def values(op: str, **kwargs: Any) -> None:
    """Constants an op's synthesizers need that the trace does not record."""
    register(op, values=kwargs)


def outputs(op: str, *names: str, single_rep: str = "") -> None:
    """Declare arguments an op writes but whose schema does not say so."""
    register(op, outputs=names, **({"single_rep": single_rep}
                                   if single_rep else {}))


def entry(op: str, module: str, attribute: str) -> None:
    """Declare the context-free entry point to replay ``op`` through."""
    register(op, entry=(module, attribute))


def build_args(case, resolved: Resolved, device: str):
    """The materialized call for a case, recipe first then the generic builder.

    Returns an :class:`breakdown.bench.inputs.Call`.
    """
    rec = recipe(case.op)
    if rec.build is not None:
        out = rec.build(case, resolved, device)
        if isinstance(out, _inputs.Call):
            return out
        return _inputs.Call(args=list(out))
    return _inputs.build_args(case, resolved, device,
                              extra_values=rec.values or None,
                              output_names=rec.outputs)


from breakdown.bench.recipes import common  # noqa: E402,F401  (registers)
from breakdown.bench.recipes import attention  # noqa: E402,F401
from breakdown.bench.recipes import moe  # noqa: E402,F401
from breakdown.bench.recipes import sampling  # noqa: E402,F401

__all__ = ["RECIPES", "OpRecipe", "build_args", "entry", "outputs",
           "override", "recipe", "register", "skip", "values"]
