# SPDX-License-Identifier: Apache-2.0
"""The time-containment forest: module and op nodes, and the passes that
restructure it (hoist a module out of the fused op that wrapped it, coalesce a
module recorded twice in one forward) before device time rolls up.
"""
from __future__ import annotations


from typing import Any
from ..core.dtypes import is_known as is_dtype
from ..trace_common import (
    _is_overhead_event, _strip_instance_idx, module_span_display_name,
    parse_module_span)
from .rules import _functional_module_class
from .events import _parse_input_args, _parse_input_dims_types, _worker_tid


# ===================================================================
# Raw nesting tree (modules + ops) via time-containment
# ===================================================================

class _Raw:
    """A node in the raw trace nesting tree (a module or a leaf op)."""

    __slots__ = ("kind", "label", "ts", "end", "dur", "ext", "shapes",
                 "dtype", "dtypes", "children", "self_dev", "sub_dev",
                 "attr_name", "arg_slots", "launch")

    def __init__(self, kind: str, label: str, ts: float, dur: float):
        self.kind = kind          # "module" or "op"
        self.label = label        # class name (module) or op name (op)
        self.ts = ts
        self.dur = dur
        self.end = ts + dur
        self.ext: int | None = None
        self.shapes: list[list[int]] = []
        self.dtype = ""
        self.dtypes: list[str] = []   # per-tensor recorded dtypes (aligned)
        self.children: list[_Raw] = []
        self.self_dev = 0.0       # device us launched directly by this op
        self.sub_dev = 0.0        # device us of this node + all descendants
        self.attr_name = ""       # real module attribute name (q_norm, ...)
        self.arg_slots: list[dict] = []  # full ordered call args (replay)
        # {file, line, func} of the Python frame that launched this kernel;
        # only set for ops with no cpu_op (Triton / extension kernels).
        self.launch: dict[str, Any] | None = None


def _deepest_at(roots: list[_Raw], ts: float) -> _Raw | None:
    """Deepest node (module or op) in the forest whose interval contains ``ts``.

    Siblings in the forest never overlap (they were built by strict time
    containment), so descending into the first child that contains ``ts`` finds
    the tightest enclosing node — the physical launch site of a kernel.
    """
    node: _Raw | None = None
    level = roots
    while True:
        nxt = None
        for c in level:
            if c.ts <= ts < c.end:
                nxt = c
                break
        if nxt is None:
            break
        node = nxt
        level = nxt.children
    return node


def _enclosing_module(node: _Raw, module_of: dict[int, _Raw]) -> _Raw | None:
    """Walk up from ``node`` to the nearest ancestor module (inclusive)."""
    cur: _Raw | None = node
    while cur is not None:
        if cur.kind == "module":
            return cur
        cur = module_of.get(id(cur))
    return None


def _build_raw_forest(events: list[dict]) -> list[_Raw]:
    """Build the module/op nesting forest for the busiest worker thread.

    Two capture modes are supported, chosen automatically:

    * **Named-span mode** (preferred) — when the trace carries capture-time
      ``user_annotation`` spans named ``module::<qname>::<Cls>`` (emitted by
      :mod:`breakdown.module_hooks`), module boundaries come from those spans, so
      every module node has its **real attribute name** (``q_norm``/``k_norm``,
      ``self_attn``, ...) straight from the trace — no reference-tree overlay,
      no registration-order assumption. The class-only ``nn.Module:`` frames are
      ignored to avoid duplicating the tree.
    * **Span-less mode** — a trace captured without the hooks (an archived or
      third-party trace) falls back to the class-only ``nn.Module: <Cls>_<idx>``
      ``python_function`` events, so the tree is structurally correct but the
      module labels are class heuristics rather than attribute paths.

    Returns the raw forest only. Kernel attribution and the restructuring
    passes that follow it belong to the caller, so the pass order stays visible
    in one place - see :func:`breakdown.trace.graph.build_forest`.
    """
    worker_tid, named_span_mode = _worker_tid(events)
    if worker_tid is None:
        return []

    nodes: list[_Raw] = []
    for e in events:
        if e.get("tid") != worker_tid or e.get("ph") != "X":
            continue
        cat = e.get("cat")
        name = e.get("name", "")
        ts = e.get("ts", 0)
        dur = e.get("dur", 0) or 0
        if cat == "user_annotation":
            # Only meaningful in named-span mode; carries the real module path.
            if named_span_mode:
                parsed = parse_module_span(name)
                if parsed is not None:
                    qname, cls = parsed
                    node = _Raw("module", cls or "Module", ts, dur)
                    node.attr_name = module_span_display_name(qname, cls)
                    nodes.append(node)
            continue
        if cat == "python_function" and name.startswith("nn.Module:"):
            # In named-span mode the module boundaries come from the spans; the
            # class-only frames would duplicate the tree, so skip them.
            if named_span_mode:
                continue
            cls = name.split("nn.Module:", 1)[1].strip()
            nodes.append(_Raw("module", cls, ts, dur))
        elif cat == "python_function":
            mapped = _functional_module_class(name)
            if mapped:
                cls, display = mapped
                node = _Raw("module", cls, ts, dur)
                if display:
                    node.attr_name = display
                nodes.append(node)
        elif cat == "cpu_op":
            if _is_overhead_event(name):
                continue
            n = _Raw("op", name, ts, dur)
            a = e.get("args", {})
            n.ext = a.get("External id")
            n.shapes, n.dtypes = _parse_input_dims_types(a)
            n.arg_slots = _parse_input_args(a)
            n.dtype = next((d for d in n.dtypes if is_dtype(d)), "")
            nodes.append(n)

    if not nodes:
        return []

    # Outer intervals first: earlier start, and for equal starts the longer
    # (containing) interval first.
    nodes.sort(key=lambda n: (n.ts, -n.end))

    roots: list[_Raw] = []
    stack: list[_Raw] = []
    for n in nodes:
        while stack and stack[-1].end < n.end:
            stack.pop()
        if stack:
            stack[-1].children.append(n)
        else:
            roots.append(n)
        stack.append(n)

    return roots


def _hoist_modules_under_ops(roots: list[_Raw]) -> list[_Raw]:
    """Re-parent modules time-contained under an op to their nearest module.

    vLLM dispatches several fused blocks as a single custom ``cpu_op`` whose
    implementation *internally calls real ``nn.Module`` forwards*. The clearest
    case is the MoE block: ``vllm::moe_forward_shared`` wraps the
    ``shared_experts`` MLP (``MergedColumnParallelLinear`` → ``SiluAndMul`` →
    ``RowParallelLinear``) plus the router/expert math, so by time-containment
    the whole ``shared_experts`` module subtree nests **under the op event**
    rather than beside it. Reconstruction treats modules as the tree skeleton and
    only surfaces a module's *direct* child modules/ops (``_module_children`` /
    ``_direct_ops``), so any module buried under an op was silently dropped —
    e.g. the ``FusedMoE`` node showed only its flat op list and the shared
    experts' ``gate_up_proj``/``down_proj`` matmuls vanished from the graph.

    This lifts every module whose enclosing parent is an op up to its nearest
    ancestor module (the module that owns the wrapping op), preserving each
    module's own subtree. Order is restored later from timestamps
    (``_merge_modules`` sorts a node's children by ``ts``), so the hoisted module
    slots in at the point it actually ran. Run **after** ``_attribute_kernels``
    (which relies on the non-overlapping time-containment forest for launch-site
    lookup) and **before** ``_compute_sub_dev`` (which then rolls device time up
    through the corrected parentage, so a hoisted subtree is counted once, under
    its module, not also inside the wrapping op).
    """
    parent: dict[int, _Raw] = {}
    stack = list(roots)
    while stack:
        n = stack.pop()
        for c in n.children:
            parent[id(c)] = n
            stack.append(c)

    def nearest_module_ancestor(node: _Raw) -> _Raw | None:
        cur = parent.get(id(node))
        while cur is not None and cur.kind != "module":
            cur = parent.get(id(cur))
        return cur

    # Collect (module, target) using the *original* parentage before mutating.
    moves: list[tuple[_Raw, _Raw | None]] = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        stack.extend(n.children)
        if n.kind == "module":
            p = parent.get(id(n))
            if p is not None and p.kind == "op":
                moves.append((n, nearest_module_ancestor(n)))

    for mod, target in moves:
        parent[id(mod)].children.remove(mod)
        if target is None:
            roots.append(mod)
        else:
            target.children.append(mod)
        parent[id(mod)] = target
    return roots


def _coalesce_duplicate_child_modules(roots: list[_Raw]) -> None:
    """Merge sibling module nodes that share the same profiler instance label.

    A single module *object*'s forward can be recorded more than once within one
    parent forward. vLLM's MoE shared-experts overlap is the clearest case:
    ``shared_experts`` is entered once as an **empty shell** whose compute is
    fused into the sibling ``vllm::moe_forward_shared`` custom op (hoisted out
    empty by :func:`_hoist_modules_under_ops`) and once as the **real MLP**
    forward. Both events carry the *identical* profiler label (e.g.
    ``SharedExperts_0`` twice), so they are the same object and must collapse to
    one node — otherwise the reconstructed graph shows a spurious empty
    ``SharedExperts`` sibling next to the real one. The order varies (empty-first
    in prefill, empty-last in decode), so this keys purely on the shared label,
    unions the duplicates' child ops/modules into the earliest occurrence and
    sums their directly-launched device time.

    Distinct sibling modules have distinct instance labels (``_0``/``_1``/…), so
    this is a no-op for them. **Only real profiler module events** (``nn.Module:
    <Cls>_<idx>``, which carry a per-object instance index) are eligible: two
    events sharing ``SharedExperts_0`` are the same object. Synthetic
    functional-frame modules (``_FUNCTIONAL_MODULE_FRAMES`` →
    ``FusedAllreduceGemmaRMSNorm``, ``FusedTopKBiasRouter``, ``XpuFusedMoE``, …)
    have a **bare class label with no index**, so genuinely-distinct occurrences
    (e.g. a decoder layer's pre- and post-attention ``fused_allreduce_gemma_-
    rms_norm``) legitimately share a label and must **not** be merged — they are
    skipped here (``_strip_instance_idx`` is a no-op on them).

    Run **after** :func:`_hoist_modules_under_ops` (which relocates the empty
    shell to become a sibling of the real forward) and **before**
    :func:`_compute_sub_dev` (so the unioned subtree's device time rolls up once,
    under the single surviving node).
    """
    stack: list[_Raw] = list(roots)
    while stack:
        n = stack.pop()
        seen: dict[str, _Raw] = {}
        dups: set[int] = set()
        changed: list[_Raw] = []
        for cm in n.children:
            if cm.kind != "module":
                continue
            # Only real instance-indexed module events (SharedExperts_0) are the
            # same object when repeated; synthetic frame modules (no index) may
            # legitimately repeat as distinct siblings — leave them alone.
            if _strip_instance_idx(cm.label) == cm.label:
                continue
            rep = seen.get(cm.label)
            if rep is None:
                seen[cm.label] = cm
            else:
                rep.children.extend(cm.children)
                rep.self_dev += cm.self_dev
                rep.ts = min(rep.ts, cm.ts)
                rep.end = max(rep.end, cm.end)
                if not rep.attr_name:
                    rep.attr_name = cm.attr_name
                dups.add(id(cm))
                changed.append(rep)
        if dups:
            n.children = [c for c in n.children if id(c) not in dups]
            for rep in changed:
                rep.children.sort(key=lambda x: (x.ts, -x.end))
        stack.extend(c for c in n.children if c.kind == "module")


def _compute_sub_dev(roots: list[_Raw]) -> None:
    """Post-order fill of ``sub_dev`` (self + descendant device time)."""
    for r in roots:
        stack = [(r, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                node.sub_dev = node.self_dev + sum(c.sub_dev for c in node.children)
            else:
                stack.append((node, True))
                for c in node.children:
                    stack.append((c, False))


def _forest_has_named_modules(roots: list[_Raw]) -> bool:
    """True if any module node already carries a real attribute name.

    Set only when the trace was captured with :mod:`breakdown.module_hooks`
    spans, in which case the reference-tree overlay is unnecessary (names are
    already exact).
    """
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.kind == "module" and n.attr_name:
            return True
        stack.extend(n.children)
    return False
