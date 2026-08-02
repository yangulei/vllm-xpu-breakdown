# SPDX-License-Identifier: Apache-2.0
"""Capture-time kernel-launch spans — operands for kernels with no ``cpu_op``.

:mod:`breakdown.module_hooks` records the *structure* of a forward pass at
capture time; this module records the *arguments* of the kernel launches inside
it that the profiler cannot see.

A kernel dispatched through ``torch.ops`` leaves a ``cpu_op`` event carrying its
operand shapes, dtypes and concrete values, which is what the replay benchmark
rebuilds the call from. A kernel launched **straight from Python** leaves
nothing: a Triton ``JITFunction`` and a pybind11 extension entry point both go
from Python to the driver without passing through the dispatcher. The trace
then shows only the device kernel and the Python frames around it — enough to
name the launcher, not enough to call it again. Earlier code papered over this
by *reconstructing* likely shapes from the model config (one hardcoded layout
table per kernel family), which is a guess that silently rots when a kernel's
signature changes.

The general fix is to record the arguments where they exist: at the launch.
Two hooks cover every Python-launched kernel in practice:

* ``triton.runtime.jit.JITFunction.run`` — one patch, every Triton kernel.
* Native extension modules (``xattention._C``, ``flashinfer``'s JIT modules) —
  their callables are wrapped in place.

Each hook opens a ``record_function`` span labelled
``kernel::<json>`` (see :func:`breakdown.trace_common.kernel_span_label`)
carrying the *launcher frame* — the innermost public Python frame outside the
launch machinery, i.e. the function a replay must call — together with that
function's parameters, described with the same slot schema the profiler uses
for a ``cpu_op``. So the benchmark replays ``gemma_rmsnorm(x, weight, eps)``
with the tensors it really got, not with a layout inferred from the config.

The installer is a module-level function so it can be shipped across the vLLM
worker boundary via ``LLM.apply_model``, exactly like the module hooks.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any

from .trace_common import is_launcher_frame, kernel_span_label

#: Top-level packages whose native extensions are infrastructure, not kernels.
#: Wrapping them would add spans (and overhead) to work that never reaches the
#: device. Everything else that ships a ``.so`` is treated as a kernel library,
#: so a new extension needs no registration.
_EXT_DENY = frozenset({
    "torch", "torchgen", "numpy", "scipy", "pandas", "pyarrow", "PIL",
    "zmq", "msgspec", "yaml", "regex", "charset_normalizer", "cffi",
    "_cffi_backend", "sentencepiece", "tokenizers", "google", "grpc",
    "psutil", "lxml", "markupsafe", "sqlalchemy", "ujson", "orjson",
    "pydantic_core", "cv2", "av", "zstandard", "xxhash", "numba",
})

#: Explicit extension module list, overriding discovery::
#:
#:     BREAKDOWN_KERNEL_EXT_MODULES=xattention._C,my_kernels._C
_EXT_ENV = "BREAKDOWN_KERNEL_EXT_MODULES"

_MAX_SEQ_SLOT = 8       # describe at most this many items of a tensor sequence


# ===================================================================
# Argument description (same slot schema as a cpu_op's recorded inputs)
# ===================================================================

def _describe(value: Any, name: str) -> dict:
    """Describe one argument as a replay slot.

    Mirrors ``graph_from_trace._parse_input_args``: ``tensor`` carries dims /
    dtype / strides, ``tensorlist`` carries its items, ``scalar`` carries the
    value as text (parsed back by ``bench.inputs.parse_scalar``), ``none`` is an
    omitted optional. Anything else is ``opaque`` — recorded with its type so a
    failed replay says *what* it could not rebuild instead of guessing.
    """
    torch = sys.modules.get("torch")
    if value is None:
        return {"kind": "none", "name": name, "value": "None"}
    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "kind": "tensor", "name": name,
            "dims": [int(d) for d in value.shape],
            "dtype": str(value.dtype).replace("torch.", ""),
            "strides": [int(s) for s in value.stride()],
        }
    if isinstance(value, bool):
        return {"kind": "scalar", "name": name, "type": "bool",
                "value": "True" if value else "False"}
    if isinstance(value, (int, float, str)):
        return {"kind": "scalar", "name": name,
                "type": type(value).__name__, "value": str(value)}
    if isinstance(value, (list, tuple)):
        items = list(value)[:_MAX_SEQ_SLOT]
        if torch is not None and items and all(
                isinstance(v, torch.Tensor) for v in items):
            return {"kind": "tensorlist", "name": name,
                    "items": [_describe(v, name) for v in items]}
        if all(isinstance(v, (int, float, bool)) for v in items):
            return {"kind": "scalar", "name": name, "type": "int[]",
                    "value": "[" + ", ".join(str(v) for v in items) + "]"}
    if torch is not None and isinstance(value, torch.dtype):
        return {"kind": "scalar", "name": name, "type": "ScalarType",
                "value": str(value)}
    return {"kind": "opaque", "name": name, "type": type(value).__name__}


def _frame_args(frame: Any) -> list[dict]:
    """Describe a frame's declared parameters, in signature order."""
    code = frame.f_code
    count = code.co_argcount + code.co_kwonlyargcount
    names = code.co_varnames[:count]
    local = frame.f_locals
    return [_describe(local.get(n), n) for n in names]


# ===================================================================
# Launcher-frame lookup on the live stack
# ===================================================================

_SELF_FILE = os.path.abspath(__file__)


def _launcher_payload() -> dict | None:
    """The innermost public frame outside the launch machinery, described.

    The same rule the trace reader applies to recorded ``python_function``
    events (:func:`breakdown.trace_common.is_launcher_frame`), applied to the
    live stack — so the span and the trace always name the same function.
    """
    frame = sys._getframe(1)
    while frame is not None:
        file = frame.f_code.co_filename
        if os.path.abspath(file) != _SELF_FILE and is_launcher_frame(
                file, frame.f_code.co_name):
            return {"file": file, "line": frame.f_lineno,
                    "func": frame.f_code.co_name, "args": _frame_args(frame)}
        frame = frame.f_back
    return None


def _open_span():
    """Open a ``kernel::`` span for the current launch, or ``None``.

    Returns the entered ``record_function`` so the caller closes it in a
    ``finally``; a plain function rather than a context manager because this
    runs on every kernel launch.
    """
    try:
        payload = _launcher_payload()
        if payload is None:
            return None
        from torch.profiler import record_function
        span = record_function(kernel_span_label(payload))
        span.__enter__()
        return span
    except Exception:  # noqa: BLE001 - instrumentation must never break a run
        return None


# ===================================================================
# Installation
# ===================================================================

_PATCHES: list[tuple[Any, str, Any]] = []


def _patch(owner: Any, attr: str, wrapper_factory) -> bool:
    original = getattr(owner, attr, None)
    if original is None or getattr(original, "_breakdown_kernel_span", False):
        return False
    wrapper = wrapper_factory(original)
    wrapper._breakdown_kernel_span = True          # type: ignore[attr-defined]
    try:
        setattr(owner, attr, wrapper)
    except Exception:  # noqa: BLE001 - immutable extension module
        return False
    _PATCHES.append((owner, attr, original))
    return True


def _make_wrapper(original):
    def wrapper(*args, **kwargs):
        span = _open_span()
        if span is None:
            return original(*args, **kwargs)
        try:
            return original(*args, **kwargs)
        finally:
            try:
                span.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
    try:
        wrapper.__name__ = getattr(original, "__name__", "kernel")
        wrapper.__doc__ = getattr(original, "__doc__", None)
    except Exception:  # noqa: BLE001
        pass
    return wrapper


def _extension_modules() -> list[Any]:
    """Loaded native extension modules that may launch device kernels."""
    explicit = os.environ.get(_EXT_ENV, "").strip()
    if explicit:
        names = [n.strip() for n in explicit.split(",") if n.strip()]
        return [sys.modules[n] for n in names if n in sys.modules]
    out = []
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] in _EXT_DENY:
            continue
        file = getattr(mod, "__file__", None) or ""
        if file.endswith((".so", ".pyd", ".dylib")):
            out.append(mod)
    return out


def install_kernel_span_hooks() -> int:
    """Install the kernel-launch spans; returns the number of hooks installed.

    Idempotent: an already-patched callable is skipped, so calling this twice
    does not nest spans.
    """
    count = 0
    try:
        from triton.runtime.jit import JITFunction
    except Exception:  # noqa: BLE001 - Triton is optional
        JITFunction = None                        # type: ignore[assignment]
    if JITFunction is not None:
        count += _patch(JITFunction, "run", _make_wrapper)

    for mod in _extension_modules():
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            fn = getattr(mod, attr, None)
            if callable(fn) and not isinstance(fn, type):
                count += _patch(mod, attr, _make_wrapper)
    return count


def remove_kernel_span_hooks() -> None:
    """Restore every callable patched by :func:`install_kernel_span_hooks`."""
    while _PATCHES:
        owner, attr, original = _PATCHES.pop()
        try:
            setattr(owner, attr, original)
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def kernel_span_hooks():
    """Context manager wrapping install/remove for in-process use."""
    count = install_kernel_span_hooks()
    try:
        yield count
    finally:
        remove_kernel_span_hooks()


def install_kernel_span_hooks_on(model: Any) -> int:
    """``apply_model``-friendly installer (the model argument is unused).

    The patches are process-global — a Triton kernel is launched by whichever
    module happens to own it — so there is nothing to attach to the model; the
    signature exists so the hook ships across the vLLM worker boundary the same
    way :func:`breakdown.module_hooks.install_module_span_hooks_on` does.
    """
    return install_kernel_span_hooks()


def remove_kernel_span_hooks_on(model: Any) -> bool:
    """``apply_model``-friendly remover."""
    remove_kernel_span_hooks()
    return True
