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
  Those resolve through :data:`PYTHON_API` to the wrapper function.
* **Context-bound wrappers.** ``vllm::unified_attention_with_output`` takes a
  ``layer_name`` and reads the KV cache and attention metadata out of vLLM's
  *forward context*; the dispatcher op cannot be invoked standalone. Where the
  wrapper has a **context-free kernel entry point** (the paged FlashAttention
  varlen call, the KV-cache write), it is replayed through that entry point with
  a synthesized paged KV cache - see
  :mod:`breakdown.bench.recipes.attention`; :data:`PYTHON_API` is consulted
  *before* :data:`NOT_REPLAYABLE` precisely so such an entry point wins. A
  wrapper with no context-free entry point (the fused MoE dispatch) is still
  refused with an explicit reason (:data:`NOT_REPLAYABLE`) and reported in the
  plan - the kernels it launches are separate ops in the graph and are
  benchmarked on their own.

Nothing is guessed: an unknown op raises :class:`ResolveError`.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

#: namespace -> modules whose import registers that namespace's custom ops.
#: vLLM registers lazily, so ``torch.ops.vllm.<op>`` only exists once the layer
#: module that defines it has been imported.
REGISTRAR_MODULES: dict[str, tuple[str, ...]] = {
    "_C": ("vllm._custom_ops",),
    "_C_cache_ops": ("vllm._custom_ops",),
    "_moe_C": ("vllm._custom_ops", "vllm.model_executor.layers.fused_moe"),
    "_xpu_C": ("vllm._custom_ops", "vllm._ipex_ops"),
    "vllm": (
        "vllm._custom_ops",
        "vllm._xpu_ops",
        "vllm.attention.layer",
        "vllm.model_executor.layers.fused_moe.layer",
        "vllm.model_executor.layers.fused_moe",
        "vllm.v1.sample.ops.topk_topp_sampler",
        "vllm.distributed.parallel_state",
    ),
}

#: Ops that cannot be invoked outside a live vLLM forward pass, with the reason
#: reported in the plan. These are dispatch *wrappers*: the kernels they launch
#: are separate ops in the reconstructed graph and are benchmarked directly.
#:
#: An entry here is only reached when the op has **no** :data:`PYTHON_API`
#: entry; attention and the KV-cache write do have one (their kernels take the
#: cache and the sequence metadata as plain arguments), so they are replayed
#: rather than refused.
NOT_REPLAYABLE: dict[str, str] = {
    "vllm::unified_attention":
        "reads KV cache + attention metadata from vLLM's forward context",
    "vllm::moe_forward_shared":
        "fused MoE dispatch wrapper; its router/expert/shared-expert kernels "
        "are benchmarked as their own ops",
    "vllm::moe_forward":
        "fused MoE dispatch wrapper; its constituent kernels are benchmarked "
        "as their own ops",
}

#: Op name -> ``(module, attribute)`` for the *redirects*: an op whose replay
#: entry point is a different function from the one the trace recorded.
#:
#: This table used to also list every Python-launched kernel (the xattention
#: SYCL wrappers, the FlashInfer norms) by dotted module path. That was a guess
#: at something the trace states outright, and it was wrong: the recorded
#: launcher for MiniMax-M3's indexer is
#: ``vllm/models/minimax_m3/xpu/ops/xattention.py`` while the table said
#: ``vllm.model_executor.models.minimax_m3.xattention``, so all four MSA ops
#: failed to resolve on the very model they were added for. Such kernels are now
#: resolved from their recorded launcher frame (see :func:`resolve`).
#:
#: What legitimately remains are context-bound *wrappers*: they have a real
#: dispatcher ``cpu_op`` (so no launcher frame is recorded) and cannot run
#: outside a forward pass, but one level down there is a kernel entry point that
#: takes the context as plain arguments.
PYTHON_API: dict[str, tuple[str, str]] = {
    # Context-bound wrappers replayed through their context-free kernel entry
    # point. ``fa_utils`` re-exports the platform's implementation (the
    # vllm-xpu-kernels varlen FlashAttention on XPU, vllm_flash_attn on CUDA),
    # so this one mapping covers both devices. The paged KV cache, block table
    # and sequence metadata the wrapper would have read from the forward
    # context are synthesized by
    # :mod:`breakdown.bench.recipes.attention`.
    "vllm::unified_attention_with_output":
        ("vllm.v1.attention.backends.fa_utils", "flash_attn_varlen_func"),
    "vllm::unified_kv_cache_update":
        ("vllm.v1.attention.backends.fa_utils", "reshape_and_cache_flash"),
}

#: Namespaces handled by the collective path, not by direct replay.
COLLECTIVE_NAMESPACES = ("c10d", "ccl", "nccl", "xccl")

#: Extra ``PYTHON_API`` entries, so a kernel that lives outside vLLM (a research
#: branch, a private extension) can be registered without editing this file::
#:
#:     BREAKDOWN_BENCH_PYTHON_API=/path/api.json
#:     {"flash_xpu::my_kernel": ["my_pkg.wrappers", "my_kernel"]}
_API_ENV = "BREAKDOWN_BENCH_PYTHON_API"


def _load_api_overrides() -> None:
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
            PYTHON_API[str(op)] = (str(target[0]), str(target[1]))


_load_api_overrides()


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
    ns = split_name(op)[0]
    return ns in COLLECTIVE_NAMESPACES


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
    by dotted path and the op is unresolvable in half the installs; import by
    location and it always resolves to the code that actually ran.
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
    # PYTHON_API first: an op listed there has a context-free entry point, which
    # beats a NOT_REPLAYABLE refusal for the same dispatch name.
    if op in PYTHON_API:
        mod_name, attr = PYTHON_API[op]
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            raise ResolveError(f"cannot import {mod_name}: {exc}") from exc
        fn = getattr(mod, attr, None)
        if fn is None:
            raise ResolveError(f"{mod_name} has no attribute {attr}")
        return Resolved(op=op, fn=fn, kind="python_api")
    if op in NOT_REPLAYABLE:
        raise NotReplayable(NOT_REPLAYABLE[op])

    ns, name = split_name(op)
    if ns in ("triton", "flashinfer", "flash_xpu"):
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
            "with a current build, or add it to "
            "breakdown.bench.resolve.PYTHON_API")

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
    if op in NOT_REPLAYABLE and op not in PYTHON_API:
        return "not_replayable", NOT_REPLAYABLE[op]
    from breakdown.bench import recipes

    reason = recipes.SKIP_REASONS.get(op)
    if reason:
        return "skipped", reason
    if is_collective(op):
        return "collective", "needs a multi-rank launch"
    try:
        r = resolve(op, slots, launch=launch)
    except NotReplayable as exc:
        return "not_replayable", str(exc)
    except ResolveError as exc:
        return "unresolved", str(exc)
    return "replayable", r.kind
