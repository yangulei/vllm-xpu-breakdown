# SPDX-License-Identifier: Apache-2.0
"""Capture-time module-boundary spans (research recommendation R1).

The torch profiler labels ``nn.Module`` forward frames with their *class* only
(``nn.Module: RMSNorm_2``), so same-class siblings (``q_norm``/``k_norm``,
``input_layernorm``/``post_attention_layernorm``) are indistinguishable in the
raw trace. The previous design recovered the real names *after the fact* by
aligning a ``named_modules()`` reference tree onto the reconstructed tree
(:mod:`breakdown.module_naming`) — a brittle step that assumes same-class
siblings run in registration order and has to unwrap ``*Model`` wrapper levels
the trace omits.

This module records the real names **at capture time** instead: it installs a
``register_forward_pre_hook`` / ``register_forward_hook`` pair on every module
that opens a ``record_function("module::<qualified_name>::<Cls>")`` span around
the forward. Those emit ``user_annotation`` trace events that nest by
time-containment exactly like the module forwards and carry the exact attribute
path, so :mod:`breakdown.graph_from_trace` reconstructs the tree with real names
directly — no overlay, no ordering assumption, causally correct even under async
execution because the span is opened at dispatch time.

The installer is a module-level function so it can be shipped across the vLLM
worker boundary via ``LLM.apply_model`` (which runs it in the process that owns
the model, where the forwards — and the profiler — actually run).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from .trace_common import module_span_label


def install_module_span_hooks(model: Any) -> list[Any]:
    """Install forward hooks that emit ``module::`` spans on every submodule.

    Registers a pre/post hook pair on each entry of ``model.named_modules()``
    (including the root). The pre-hook opens a ``torch.profiler.record_function``
    span named ``module::<qualified_name>::<ClassName>``; the post-hook closes it.
    The open spans are stashed on the hook state keyed by module id so the
    matching close fires even for re-entrant / shared modules.

    Returns the list of ``RemovableHandle`` objects; pass them to
    :func:`remove_module_span_hooks` (or use :func:`module_span_hooks`) to
    uninstall. Safe to call with ``torch`` absent only if never reached — it
    imports ``torch`` lazily so the rest of the package stays import-light.
    """
    from torch.profiler import record_function

    handles: list[Any] = []
    # Per-module stack of open record_function contexts (a module may be entered
    # re-entrantly; a stack keeps pre/post correctly paired).
    open_spans: dict[int, list[Any]] = {}

    def make_hooks(qualified_name: str, cls: str):
        label = module_span_label(qualified_name, cls)

        def pre_hook(module, args):
            rf = record_function(label)
            rf.__enter__()
            open_spans.setdefault(id(module), []).append(rf)

        def post_hook(module, args, output):
            stack = open_spans.get(id(module))
            if stack:
                rf = stack.pop()
                rf.__exit__(None, None, None)

        return pre_hook, post_hook

    for name, module in model.named_modules():
        pre, post = make_hooks(name, type(module).__name__)
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    return handles


def remove_module_span_hooks(handles: list[Any]) -> None:
    """Remove hooks installed by :func:`install_module_span_hooks`."""
    for h in handles or []:
        try:
            h.remove()
        except Exception:
            pass


@contextmanager
def module_span_hooks(model: Any):
    """Context manager wrapping install/remove for in-process use.

    Prefer this when the model lives in the current process. For vLLM's worker
    boundary use :func:`install_module_span_hooks` via ``LLM.apply_model``.
    """
    handles = install_module_span_hooks(model)
    try:
        yield handles
    finally:
        remove_module_span_hooks(handles)


# ``LLM.apply_model`` runs a callable in the worker process that owns the model
# (and the profiler). The handles it returns are not needed on the caller side,
# so stash them on the model and provide a matching remover that reads them back.
_SPAN_HANDLES_ATTR = "_xpu_breakdown_span_handles"


def install_module_span_hooks_on(model: Any) -> int:
    """``apply_model``-friendly installer; stashes handles on the model.

    Returns the number of hooks installed (picklable across the worker boundary).
    """
    handles = install_module_span_hooks(model)
    setattr(model, _SPAN_HANDLES_ATTR, handles)
    return len(handles)


def remove_module_span_hooks_on(model: Any) -> bool:
    """``apply_model``-friendly remover for hooks installed on ``model``."""
    handles = getattr(model, _SPAN_HANDLES_ATTR, None)
    if handles:
        remove_module_span_hooks(handles)
        try:
            delattr(model, _SPAN_HANDLES_ATTR)
        except Exception:
            pass
    return True

