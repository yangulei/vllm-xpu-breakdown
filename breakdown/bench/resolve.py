# SPDX-License-Identifier: Apache-2.0
"""Dispatch name -> callable, with its registered schema.

The trace records the *dispatch* name of every op vLLM ran, so replay does not
need an adapter table: ``aten::linear`` is ``torch.ops.aten.linear`` and
``_C::silu_and_mul_with_clamp`` is ``torch.ops._C.silu_and_mul_with_clamp``.
The op's registered **schema** then says what each positional argument is
(``Tensor``, ``Tensor?``, ``int``, ``float``, ``str``, ``SymInt[]``, …), which
is what lets :mod:`breakdown.bench.inputs` fill the slots the profiler could
not record a value for.

Two things are not plain dispatcher ops:

* **Synthetic kernel ops.** Kernels launched straight from Python via Triton /
  FlashInfer / the xattention SYCL extension have no ``cpu_op``, so the
  reconstruction names them after the public API frame that launched them
  (``triton::_gemma_rmsnorm_kernel``, ``flash_xpu::minimax_m3_sparse_attn``).
  Those resolve through the *launcher frame the trace recorded* - import that
  file, take that attribute.
* **Context-bound wrappers.** ``vllm::unified_attention_with_output`` takes a
  ``layer_name`` and reads the KV cache and attention metadata out of vLLM's
  *forward context*; the dispatcher op cannot be invoked standalone. Where the
  wrapper has a **context-free kernel entry point** (the paged FlashAttention
  varlen call, the KV-cache write), it is replayed through that entry point with
  a synthesized paged KV cache (see
  :mod:`breakdown.bench.recipes.attention`). Both facts live in that op's
  :class:`breakdown.bench.recipes.table.OpRecipe`: its ``entry`` is consulted
  *before* its ``skip`` precisely so an entry point wins. A wrapper with no
  context-free entry point (the fused MoE dispatch) carries only ``skip`` and
  is refused with that reason and reported in the plan - the kernels it
  launches are separate ops in the graph and are benchmarked on their own.

Nothing is guessed: an unknown op raises :class:`ResolveError`.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from breakdown.core.opnames import (
    COLLECTIVE_NAMESPACES, PYTHON_LAUNCHED_NAMESPACES, REGISTRAR_MODULES)

class ResolveError(RuntimeError):
    """The op could not be resolved to something callable."""


class NotReplayable(RuntimeError):
    """The op exists but cannot be invoked standalone (with the reason)."""


@dataclass
class Resolved:
    """A callable plus whatever description of its arguments exists."""

    op: str
    fn: Callable[..., Any]
    kind: str                       # "torch_op" | "python_api"
    schema: Any = None              # torch FunctionSchema when kind=torch_op
    arg_names: list[str] = None     # type: ignore[assignment]
    arg_types: list[str] = None     # type: ignore[assignment]
    defaults: list[Any] = None      # type: ignore[assignment]
    mutates: list[int] = None       # type: ignore[assignment]
    #: indices of arguments the schema marks keyword-only (``*, Scalar
    #: alpha=1``). Passing one positionally is a TypeError, so the argument
    #: builder must split them out.
    kwarg_only: list[int] = None    # type: ignore[assignment]
    returns_none: bool = False

    def __post_init__(self) -> None:
        self.arg_names = self.arg_names or []
        self.arg_types = self.arg_types or []
        self.defaults = self.defaults or []
        self.mutates = self.mutates or []
        self.kwarg_only = self.kwarg_only or []


def split_name(op: str) -> tuple[str, str]:
    ns, _, name = op.partition("::")
    return (ns, name) if name else ("aten", ns)


def is_collective(op: str) -> bool:
    """True if replaying this op needs more than one rank.

    Namespace only, deliberately: the launch path this gates spawns a process
    group, so it must fire on the ops that really are ``c10d`` calls and not on
    a compute kernel whose *name* happens to contain "all_reduce".
    """
    return split_name(op)[0] in COLLECTIVE_NAMESPACES


def _import_registrars(ns: str) -> None:
    for mod in REGISTRAR_MODULES.get(ns, ()):
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 - a missing optional layer is fine
            continue


def _slot_profile(slots: list[dict] | None) -> tuple[int, int] | None:
    """``(total slots, tensor slots)`` recorded for the call."""
    if not slots:
        return None
    tensors = sum(1 for s in slots
                  if s.get("kind") in ("tensor", "tensorlist"))
    return len(slots), tensors


def _schema_profile(schema: Any) -> tuple[int, int]:
    tensors = 0
    for arg in schema.arguments:
        t = str(arg.type)
        if t.startswith("Tensor") or t.startswith("Optional[Tensor") or \
                t.startswith("List[Tensor"):
            tensors += 1
    return len(schema.arguments), tensors


def _overload(packet, slots: list[dict] | None = None) -> Any:
    """Pick the overload the trace actually dispatched.

    ``aten::add`` has ``Tensor``, ``Scalar``, ``int``, ``float``, … overloads
    with wildly different signatures; ``default`` on such a packet is the *int*
    one, which has no tensor arguments at all. So the recorded slot profile
    (how many arguments, how many of them tensors) selects the overload, and
    only a tie falls back to ``default``.
    """
    names = list(packet.overloads())
    if not names:
        raise ResolveError("op has no overloads")
    cands = [n for n in names if n != "out"] or names
    want = _slot_profile(slots)

    def score(n: str) -> tuple:
        ov = getattr(packet, n)
        try:
            nargs, ntensors = _schema_profile(ov._schema)
        except (AttributeError, RuntimeError):
            return (-99, 0, 0)
        if want is None:
            return (0, n == "default", ntensors)
        s = 0
        if ntensors == want[1]:
            s += 4
        if nargs == want[0]:
            s += 2
        if nargs >= want[0]:
            s += 1
        return (s, n == "default", ntensors)

    return getattr(packet, max(cands, key=score))


def _from_schema(schema: Any) -> dict[str, Any]:
    names, types, defaults, mutates, kwonly = [], [], [], [], []
    for i, arg in enumerate(schema.arguments):
        names.append(arg.name)
        types.append(str(arg.type))
        defaults.append(arg.default_value if arg.has_default_value() else None)
        if getattr(arg, "kwarg_only", False):
            kwonly.append(i)
        alias = getattr(arg, "alias_info", None)
        if alias is not None and getattr(alias, "is_write", False):
            mutates.append(i)
    return {"arg_names": names, "arg_types": types, "defaults": defaults,
            "mutates": mutates, "kwarg_only": kwonly,
            "returns_none": len(schema.returns) == 0}


def _import_file(path: str) -> Any:
    """Import the module that lives at ``path``, by file location.

    The trace records where a kernel's launcher *is*, not what it is importable
    as, and the two differ: the same xattention wrapper is
    ``vllm/models/minimax_m3/xpu/ops/xattention.py`` in one checkout and
    ``vllm/model_executor/models/minimax_m3/xattention.py`` in another. Import
    by a guessed dotted path and the op is unresolvable in half the installs.
    But importing every file under a synthetic top-level name is also wrong:
    package-relative imports such as ``from .chunk_intra import ...`` then fail
    with "no known parent package". Recover the dotted name from the contiguous
    ``__init__.py`` chain when one exists, verify that it resolves to the exact
    recorded file, and only then fall back to a location import for standalone
    launchers.
    """
    if not os.path.isfile(path):
        raise ResolveError(f"the launcher file no longer exists: {path}")
    # Prefer an already-imported module for this file: re-importing it under a
    # synthetic name would create a *second* copy of its globals, and a Triton
    # kernel's compiled-kernel cache lives in those globals.
    target = os.path.abspath(path)
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and os.path.abspath(f) == target:
            return mod
    package_name = _package_name(path)
    if package_name:
        try:
            mod = importlib.import_module(package_name)
        except Exception as exc:  # noqa: BLE001 - report the real import error
            raise ResolveError(f"cannot import {path}: {exc}") from exc
        imported = getattr(mod, "__file__", None)
        if imported and os.path.abspath(imported) == target:
            return mod
    spec = importlib.util.spec_from_file_location(
        "breakdown_launcher_" + str(abs(hash(path))), path)
    if spec is None or spec.loader is None:
        raise ResolveError(f"cannot load a module from {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 - any import error is a resolve error
        raise ResolveError(f"cannot import {path}: {exc}") from exc
    return mod


def _package_name(path: str) -> str:
    """Return the dotted package name implied by adjacent ``__init__.py`` files."""
    stem, ext = os.path.splitext(os.path.basename(path))
    if ext != ".py":
        return ""
    parts = [] if stem == "__init__" else [stem]
    parent = os.path.dirname(os.path.abspath(path))
    while os.path.isfile(os.path.join(parent, "__init__.py")):
        parts.insert(0, os.path.basename(parent))
        parent = os.path.dirname(parent)
    return ".".join(parts) if len(parts) > 1 else ""


def resolve(op: str, slots: list[dict] | None = None,
            launch: dict | None = None) -> Resolved:
    """Resolve a dispatch name to a callable, or explain why it cannot be.

    ``slots`` are the call's recorded argument slots; they select the overload
    (see :func:`_overload`). ``launch`` is the ``{file, line, func}`` frame the
    profiler recorded for a kernel with no dispatcher op. Raises
    :class:`NotReplayable` for context-bound wrappers and :class:`ResolveError`
    for anything genuinely unknown - never a silent fallback, because a wrong
    callable measures a wrong kernel.
    """
    # A declared entry point comes first: an op that has one is replayable
    # through it, which beats the skip reason on the same dispatch name.
    from breakdown.bench.recipes.table import recipe as _recipe
    rec = _recipe(op)
    if rec.entry:
        mod_name, attr = rec.entry
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            raise ResolveError(f"cannot import {mod_name}: {exc}") from exc
        fn = getattr(mod, attr, None)
        if fn is None:
            raise ResolveError(f"{mod_name} has no attribute {attr}")
        return Resolved(op=op, fn=fn, kind="python_api")
    if rec.skip:
        raise NotReplayable(rec.skip)

    ns, name = split_name(op)
    if ns in PYTHON_LAUNCHED_NAMESPACES:
        # A kernel launched straight from Python. The profiler recorded the
        # function that launched it, so there is nothing to look up: import
        # that file and take that attribute.
        if launch and launch.get("file") and launch.get("func"):
            mod = _import_file(launch["file"])
            fn = getattr(mod, launch["func"], None)
            if fn is None:
                raise ResolveError(
                    f"{launch['file']} has no attribute {launch['func']}")
            return Resolved(op=op, fn=fn, kind="python_api")
        raise ResolveError(
            "synthetic kernel op with no recorded launcher frame; re-profile "
            "with a current build, or declare its entry point with "
            "breakdown.bench.recipes.entry()")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dep here
        raise ResolveError("torch is not installed") from exc

    packet = _lookup(torch, ns, name)
    if packet is None:
        _import_registrars(ns)
        packet = _lookup(torch, ns, name)
    if packet is None:
        raise ResolveError(
            f"torch.ops.{ns}.{name} is not registered (import the module that "
            f"defines it, or add it to REGISTRAR_MODULES)")
    ov = _overload(packet, slots)
    schema = ov._schema
    return Resolved(op=op, fn=ov, kind="torch_op", schema=schema,
                    **_from_schema(schema))


def _lookup(torch_mod, ns: str, name: str):
    try:
        namespace = getattr(torch_mod.ops, ns)
        packet = getattr(namespace, name)
    except (AttributeError, RuntimeError):
        return None
    try:
        packet.overloads()
    except (AttributeError, RuntimeError):
        return None
    return packet


def classify(op: str, slots: list[dict] | None = None,
             launch: dict | None = None) -> tuple[str, str]:
    """``(status, detail)`` without importing heavy modules where avoidable.

    ``status`` is ``replayable`` | ``not_replayable`` | ``unresolved`` |
    ``collective``.
    """
    from breakdown.bench import recipes

    rec = recipes.recipe(op)
    if rec.skip and not rec.entry:
        return "not_replayable", rec.skip
    if is_collective(op):
        return "collective", "needs a multi-rank launch"
    try:
        r = resolve(op, slots, launch=launch)
    except NotReplayable as exc:
        return "not_replayable", str(exc)
    except ResolveError as exc:
        return "unresolved", str(exc)
    return "replayable", r.kind
