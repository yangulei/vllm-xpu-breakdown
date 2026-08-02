# SPDX-License-Identifier: Apache-2.0
"""Inference steps, and which of them are prefill and which are decode.
"""
from __future__ import annotations


from ..trace_common import _strip_instance_idx
from .forest import _Raw


# ===================================================================
# Phase detection
# ===================================================================

def _pass_token_dim(root: _Raw) -> int:
    """Estimate the number of tokens processed in one forward pass.

    Uses the row count (first dim) of matmul-like activation inputs, which is
    ``num_tokens`` for prefill and ``num_running_seqs`` for decode.
    """
    best = 0
    stack = list(root.children)
    while stack:
        n = stack.pop()
        if n.kind == "op":
            base = n.label.split("::")[-1].lower()
            if base in ("mm", "addmm", "linear", "matmul", "bmm") and n.shapes:
                first = n.shapes[0]
                if len(first) >= 2:
                    best = max(best, first[0] if base != "addmm" else
                               (n.shapes[1][0] if len(n.shapes) > 1 and
                                len(n.shapes[1]) >= 1 else first[0]))
        stack.extend(n.children)
    return best


def _subtree_module_count(root: _Raw) -> int:
    """Number of module nodes in ``root``'s subtree (including itself)."""
    n = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind == "module":
            n += 1
        stack.extend(node.children)
    return n


def _partition_steps(roots: list[_Raw]) -> tuple[list[dict], str]:
    """Group top-level module roots into inference steps.

    A step is one iteration of the engine: the main model forward followed by
    its post-processing roots (``LogitsProcessor``, ``Sampler`` ...). The main
    model class is the module-root class with the largest module subtree — the
    full model forward is structurally deep (decoder layers, attention, MLP, ...)
    while post-processing roots (``LogitsProcessor``, ``Sampler``) are shallow.
    Device time is *not* used to pick the main class: the ``lm_head`` vocab
    projection inside ``LogitsProcessor`` (``V``-wide matmul, run once per decode
    step) can outweigh the whole model forward, which previously mis-selected
    ``LogitsProcessor`` as the main class and collapsed the prefill pass.

    Returns ``(steps, main_class)`` where each step is
    ``{"main": _Raw|None, "roots": [_Raw, ...], "token": int}``.
    """
    module_roots = sorted([r for r in roots if r.kind == "module"],
                          key=lambda r: r.ts)
    if not module_roots:
        return [], ""

    size_by_class: dict[str, int] = {}
    for r in module_roots:
        norm = _strip_instance_idx(r.label)
        size_by_class[norm] = max(size_by_class.get(norm, 0),
                                  _subtree_module_count(r))
    main_class = max(size_by_class, key=size_by_class.get)

    steps: list[dict] = []
    cur: dict | None = None
    for r in module_roots:
        if _strip_instance_idx(r.label) == main_class:
            cur = {"main": r, "roots": [r], "token": _pass_token_dim(r)}
            steps.append(cur)
        else:
            if cur is None:
                cur = {"main": None, "roots": [], "token": 0}
                steps.append(cur)
            cur["roots"].append(r)
    return steps, main_class


def _classify_steps(roots: list[_Raw], batch_size: int
                    ) -> tuple[list[_Raw], list[_Raw], int, int]:
    """Split all module roots into prefill and decode by their step's phase.

    A **decode** step advances every running sequence by one token, so its
    forward processes ``num_running_seqs`` tokens — at most ``batch_size`` (fewer
    while the batch ramps up/down). A **prefill** step ingests a multi-token
    prompt, so its forward processes *more* than one token per running sequence.
    A step is therefore prefill iff its token dim exceeds ``batch_size``.

    This is deliberately compared against ``batch_size`` rather than picking the
    largest-token step as prefill: in a two-pass profile the decode pass drives
    a full ``batch_size``-wide decode while its own (prefix-cached) prompts
    prefill only one new token each — i.e. the decode steps have *more* rows than
    the prefill microsteps, so a "max token == prefill" rule would invert the
    phases. Comparing to ``batch_size`` also classifies each chunk of a chunked
    prefill correctly.

    Only **steady-state, full-batch decode steps are kept**. vLLM admits the
    ``batch_size`` sequences into the running batch in ramp-up waves, so the
    early decode steps process *fewer* than ``batch_size`` sequences (their norm/
    rotary/matmul ops carry partial row counts like ``2``/``4``/``28``/``30``
    instead of ``32``). Those partial-batch steps are transient and would show up
    as spurious literal-int (non-``B``) nodes in the reconstructed graph, so the
    decode phase is restricted to the steps whose token dim equals the **maximum
    observed decode batch** (the steady state). If the configured batch is never
    fully reached (e.g. KV-cache pressure caps it below ``batch_size``), the
    largest batch actually run is used — which is the honest steady state.

    Among those steady-state steps the **first is additionally dropped** as
    warmup: the initial full-batch decode forward pays one-time costs (KV/
    allocator warmup, oneDNN/Triton plan + autotune caching under
    ``torch.compile``) that would skew the per-op latency average. Both filters
    are guarded so the decode phase is never emptied — at least one steady-state
    step always remains.

    Returns ``(prefill_roots, decode_roots, n_prefill_steps, n_decode_steps)``.
    """
    steps, _ = _partition_steps(roots)
    if not steps:
        return [], [], 0, 0

    threshold = max(int(batch_size), 1)

    prefill_steps = [s for s in steps if s["token"] > threshold]
    decode_steps = [s for s in steps if s["token"] <= threshold]

    # Restrict decode to steady-state, full-batch steps: drop ramp-up/partial
    # batches (fewer running seqs than the steady state) which would otherwise
    # appear as spurious partial-row nodes. Guarded so decode is never emptied.
    if decode_steps:
        max_decode = max(s["token"] for s in decode_steps)
        steady = [s for s in decode_steps if s["token"] == max_decode]
        if steady:
            decode_steps = steady

    # Drop the warmup first steady decode step (guarded so decode is never emptied).
    if len(decode_steps) >= 2:
        decode_steps = decode_steps[1:]

    prefill: list[_Raw] = []
    decode: list[_Raw] = []
    for s in prefill_steps:
        prefill.extend(s["roots"])
    for s in decode_steps:
        decode.extend(s["roots"])
    return prefill, decode, len(prefill_steps), len(decode_steps)
