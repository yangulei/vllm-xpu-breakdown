# SPDX-License-Identifier: Apache-2.0
"""Per-op replay recipes: the exceptions to schema-driven argument building.

:mod:`breakdown.bench.inputs` builds a call from the op's schema plus the
recorded slots, which covers the great majority of dispatched ops. A *recipe* is
the escape hatch for the rest:

* an **argument override** - the whole positional list is built by hand (a
  Python-API kernel with no schema, an op whose operands must satisfy a
  relationship the schema cannot express);
* **extra context values** an op's synthesizers need (expert count, block size);
* a **skip reason** - an op that is real but must not be replayed, recorded with
  why, so the plan reports it instead of silently dropping it.

Device-specific registrations live in :mod:`.xpu` and :mod:`.cuda`; both are
imported here so a recipe is available regardless of which device the run
targets (registering is cheap and import-safe - nothing touches a device).
"""
from __future__ import annotations

from typing import Any, Callable

from breakdown.bench import inputs as _inputs
from breakdown.bench.resolve import Resolved

#: op -> ``fn(case, resolved, device) -> list[Any]`` full argument override
OVERRIDES: dict[str, Callable[..., list[Any]]] = {}

#: op -> constants merged into the synthesizer context
EXTRA_VALUES: dict[str, dict[str, Any]] = {}

#: op -> why it must not be replayed (reported, never silently dropped)
SKIP_REASONS: dict[str, str] = {}

#: op -> argument names the kernel *writes* although the schema does not say so
#: (an atomically-accumulated counter, a scatter map). They are allocated
#: zeroed and reset between measurement windows.
OUTPUT_ARGS: dict[str, tuple[str, ...]] = {}

#: op -> reason it must be measured one call per window. An op that accumulates
#: into an output cannot be repeated inside a timed window: the second call
#: starts from the first one's result, and for an offset counter that means
#: writing past the end of the buffer (device lost). Such ops trade timer
#: resolution for correctness.
SINGLE_REP: dict[str, str] = {}


def override(op: str) -> Callable[[Callable[..., list[Any]]],
                                  Callable[..., list[Any]]]:
    def deco(fn: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
        OVERRIDES[op] = fn
        return fn

    return deco


def skip(op: str, reason: str) -> None:
    SKIP_REASONS[op] = reason


def values(op: str, **kwargs: Any) -> None:
    EXTRA_VALUES.setdefault(op, {}).update(kwargs)


def outputs(op: str, *names: str, single_rep: str = "") -> None:
    """Declare arguments an op writes but whose schema does not say so."""
    OUTPUT_ARGS[op] = tuple(dict.fromkeys(OUTPUT_ARGS.get(op, ()) + names))
    if single_rep:
        SINGLE_REP[op] = single_rep


def build_args(case, resolved: Resolved, device: str):
    """The materialized call for a case, recipe first then the generic builder.

    Returns an :class:`breakdown.bench.inputs.Call`.
    """
    fn = OVERRIDES.get(case.op)
    if fn is not None:
        out = fn(case, resolved, device)
        if isinstance(out, _inputs.Call):
            return out
        return _inputs.Call(args=list(out))
    return _inputs.build_args(case, resolved, device,
                              extra_values=EXTRA_VALUES.get(case.op),
                              output_names=OUTPUT_ARGS.get(case.op, ()))


from breakdown.bench.recipes import common  # noqa: E402,F401  (registers)
from breakdown.bench.recipes import attention  # noqa: E402,F401
from breakdown.bench.recipes import cuda  # noqa: E402,F401
from breakdown.bench.recipes import xpu  # noqa: E402,F401

__all__ = ["OVERRIDES", "EXTRA_VALUES", "SKIP_REASONS", "OUTPUT_ARGS",
           "SINGLE_REP", "build_args", "override", "skip", "values",
           "outputs"]
