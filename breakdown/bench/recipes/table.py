# SPDX-License-Identifier: Apache-2.0
"""One table of per-op replay exceptions.

Everything the replay knows about a *specific* op lives in one record. It used
to live in seven parallel dictionaries — ``OVERRIDES``, ``EXTRA_VALUES``,
``SKIP_REASONS``, ``OUTPUT_ARGS``, ``SINGLE_REP`` in the recipes package plus
``PYTHON_API`` and ``NOT_REPLAYABLE`` in ``resolve`` — all keyed by the same op
name, so answering "what does the benchmark do with this op" meant reading
seven places and knowing which took precedence.

A recipe is an **exception**, not a description: the replay resolves the
dispatch name and rebuilds the recorded operands on its own, and the great
majority of ops need no entry here at all. What genuinely needs one:

``entry``
    The op's replay entry point is a *different* function from the one the
    trace recorded — a context-bound wrapper (``vllm::unified_attention_-
    with_output``) whose kernel one level down takes the context as plain
    arguments.
``build``
    The whole argument list is built by hand, because the operands must satisfy
    a relationship no schema expresses (a paged KV cache consistent with its
    block table and sequence lengths).
``values``
    Constants an op's synthesizers need and the trace does not record (expert
    count, block size).
``outputs`` / ``single_rep``
    Arguments the kernel *writes* although the schema does not say so. They are
    allocated zeroed and reset between windows; one that is accumulated with
    atomics additionally cannot be repeated inside a timed window (the second
    call starts from the first one's result, and for an offset counter that
    means writing past the end of the buffer — device lost).
``skip``
    The op is real but must not be replayed, recorded with *why* so the plan
    reports it instead of silently dropping it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable


@dataclass(frozen=True)
class OpRecipe:
    """Everything the replay does differently for one op."""

    op: str
    #: ``(module, attribute)`` of the context-free entry point to call instead.
    entry: tuple[str, str] | None = None
    #: ``fn(case, resolved, device) -> list | Call`` full argument override.
    build: Callable[..., Any] | None = None
    #: constants merged into the synthesizer context
    values: dict[str, Any] = field(default_factory=dict)
    #: argument names the kernel writes although the schema does not say so
    outputs: tuple[str, ...] = ()
    #: why the op must be measured one call per timed window
    single_rep: str = ""
    #: why the op must not be replayed at all
    skip: str = ""


RECIPES: dict[str, OpRecipe] = {}


def register(op: str, **fields: Any) -> OpRecipe:
    """Add to (or extend) the recipe for ``op``.

    Extending rather than replacing so a device module can add an override to
    an op whose skip reason or values were declared elsewhere.
    """
    current = RECIPES.get(op, OpRecipe(op=op))
    merged = dict(fields)
    if "values" in merged:
        merged["values"] = {**current.values, **merged["values"]}
    if "outputs" in merged:
        merged["outputs"] = tuple(dict.fromkeys(
            current.outputs + tuple(merged["outputs"])))
    RECIPES[op] = replace(current, **merged)
    return RECIPES[op]


def recipe(op: str) -> OpRecipe:
    """The recipe for ``op`` — an empty one when it needs no exception."""
    return RECIPES.get(op) or OpRecipe(op=op)


#: Extra entry points, so a kernel that lives outside vLLM (a research branch,
#: a private extension) can be registered without editing a source file::
#:
#:     BREAKDOWN_BENCH_PYTHON_API=/path/api.json
#:     {"flash_xpu::my_kernel": ["my_pkg.wrappers", "my_kernel"]}
#:
#: Loaded here rather than in ``resolve`` because this module has no imports of
#: its own, so it can be read from anywhere in the package without a cycle.
_API_ENV = "BREAKDOWN_BENCH_PYTHON_API"


def _load_entry_overrides() -> None:
    path = os.environ.get(_API_ENV)
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path) as fh:
            extra = json.load(fh)
    except (OSError, ValueError):
        return
    for op, target in (extra or {}).items():
        if isinstance(target, (list, tuple)) and len(target) == 2:
            register(str(op), entry=(str(target[0]), str(target[1])))


_load_entry_overrides()
