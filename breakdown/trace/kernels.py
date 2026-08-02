# SPDX-License-Identifier: Apache-2.0
"""Device time -> leaf ops.

Every device kernel is located at its *host launch site* and attributed to the
deepest node containing it, so a kernel launched straight from Python surfaces
as an op of the module that launched it. A capture-time ``kernel::`` span
(see :mod:`breakdown.kernel_hooks`) additionally carries that call's operands.
"""
from __future__ import annotations


from typing import Any
from ..trace_common import is_launcher_frame, parse_kernel_span
from .rules import (
    _DEVICE_KERNEL_CATEGORIES, _PLUMBING_OPS, _RUNTIME_CATEGORIES,
    _clean_kernel_name, _synthetic_op_label)
from .events import _PY_FRAME_RE
from .forest import _Raw, _deepest_at, _enclosing_module


def _collect_kernel_launches(events: list[dict], worker_tid: Any
                             ) -> list[tuple[float, str, float,
                                             dict[str, Any] | None]]:
    """Collect every device kernel as ``(host_launch_ts, name, device_us,
    launcher_frame)``, where the frame is ``{file, line, func}`` of the Python
    function that launched it (``None`` when a ``cpu_op`` covers it).

    Only genuine *device* events (``_DEVICE_KERNEL_CATEGORIES`` — real
    ``kernel``/``gpu_memcpy``/``gpu_memset``) are surfaced. Each is linked to the
    host-side launch call that issued it (the "flow arrow" you see in the trace
    viewer) via the correlation id: ``kernel.correlation → xpu_runtime`` /
    ``cuda_runtime`` / ``cuda_driver``. The runtime launch event carries a
    timestamp on the worker thread, which is exactly where the launch sits inside
    the module/op nesting tree. Attributing kernels by this *launch site* (rather
    than by ``External id`` bookkeeping) is robust to ``torch.compile`` —
    fused/compiled regions and eager kernels are handled identically, because
    both physically launch from within the module that owns them.

    Host-side launch-API events (``_RUNTIME_CATEGORIES``) are **never** surfaced
    as launches on a trace that has real device-kernel events: they carry no
    device time and pure bookkeeping calls (``cudaEventQuery``,
    ``cudaStreamWaitEvent``, ...) would otherwise be mistaken for kernels. The
    only exception is a *runtime-only* trace (no device-kernel events captured at
    all): there the launch-API call is the sole evidence of GPU work, so an
    actual launch call (``*Launch*`` / ``*Enqueue*``) is emitted as a fallback.

    Fallback: when a device kernel has no matching runtime event on the worker
    thread (common for flash-attention kernels launched via custom CUDA graphs or
    internal streams), we fall back to matching via ``External id`` — the kernel's
    ``External id`` links back to the CPU op that issued it, whose timestamp
    provides the launch site.
    """
    corr_to_rt: dict[int, dict] = {}
    kernel_corrs: set = set()
    has_device_kernel = False
    for evt in events:
        cat = evt.get("cat")
        if cat in _RUNTIME_CATEGORIES:
            corr = evt.get("args", {}).get("correlation")
            if corr is not None:
                corr_to_rt[corr] = evt
        elif cat in _DEVICE_KERNEL_CATEGORIES:
            has_device_kernel = True
            # Correlations backed by an actual device kernel / memcpy event.
            corr = evt.get("args", {}).get("correlation")
            if corr is not None:
                kernel_corrs.add(corr)

    # Build External-id → CPU op timestamp map for fallback attribution
    ext_to_ts: dict[int, float] = {}
    for evt in events:
        if evt.get("cat") == "cpu_op" and evt.get("tid") == worker_tid:
            ext = evt.get("args", {}).get("External id")
            if ext is not None:
                ext_to_ts[ext] = evt.get("ts", 0)

    # The Python frames that can be a kernel's definition site. A kernel with
    # no ``cpu_op`` was launched straight from Python; a capture-time
    # ``kernel::`` span records both the frame and the operands, and the frame
    # heuristic recovers just the frame from an un-hooked trace.
    kernel_spans = _collect_kernel_spans(events, worker_tid)
    launcher_frames = ([] if kernel_spans
                       else _collect_launcher_frames(events, worker_tid))

    def launcher(ts: float) -> dict[str, Any] | None:
        return (_span_at(ts, kernel_spans) if kernel_spans
                else _launcher_at(ts, launcher_frames))

    launches: list[tuple[float, str, float, dict[str, Any] | None]] = []
    for evt in events:
        cat = evt.get("cat")
        args = evt.get("args", {})
        corr = args.get("correlation")
        name = evt.get("name", "")
        dur = evt.get("dur", 0) or 0
        if cat in _DEVICE_KERNEL_CATEGORIES:
            # A real device kernel — locate its host launch site.
            rt = corr_to_rt.get(corr) if corr is not None else None
            if rt is not None and rt.get("tid") == worker_tid:
                ts = rt.get("ts", 0)
            else:
                # Fallback: use External id to find the issuing CPU op's ts.
                ext = args.get("External id")
                if ext is None or ext not in ext_to_ts:
                    continue
                ts = ext_to_ts[ext]
            launches.append((ts, name, dur, launcher(ts)))
        elif cat in _RUNTIME_CATEGORIES and not has_device_kernel:
            # Runtime-only trace: no device-kernel events were captured, so the
            # launch-API call is the only signal of GPU work. Emit actual launch
            # calls (not bookkeeping) that sit on the worker thread.
            if corr in kernel_corrs or evt.get("tid") != worker_tid:
                continue
            low = name.lower()
            if "launch" not in low and "enqueue" not in low:
                continue
            ts = evt.get("ts", 0)
            launches.append((ts, name, dur, launcher(ts)))
    return launches


def _collect_kernel_spans(events: list[dict], worker_tid: Any
                          ) -> list[tuple[float, float, dict[str, Any]]]:
    """Capture-time ``kernel::`` launch spans, as ``(ts, end, payload)``.

    A span records what the trace cannot: the launcher frame *and* the operands
    it was called with (see :mod:`breakdown.kernel_hooks`). When present it is
    authoritative — the frame heuristic below is the fallback for traces
    captured without the hooks.
    """
    spans: list[tuple[float, float, dict[str, Any]]] = []
    for evt in events:
        if evt.get("cat") != "user_annotation" or evt.get("tid") != worker_tid:
            continue
        payload = parse_kernel_span(evt.get("name", ""))
        if payload is None:
            continue
        ts = evt.get("ts", 0)
        spans.append((ts, ts + (evt.get("dur", 0) or 0), payload))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    return spans


def _span_at(launch_ts: float,
             spans: list[tuple[float, float, dict[str, Any]]]
             ) -> dict[str, Any] | None:
    """The innermost kernel span containing ``launch_ts``."""
    best: dict[str, Any] | None = None
    best_dur = float("inf")
    for ts, end, payload in spans:
        if ts <= launch_ts < end and (end - ts) < best_dur:
            best_dur, best = end - ts, payload
    return best


def _collect_launcher_frames(events: list[dict], worker_tid: Any
                             ) -> list[tuple[float, float, str, int, str]]:
    """Python frames that can be the *definition site* of a kernel launch.

    Returns ``(ts, end, file, line, func)`` for every public Python frame on the
    worker thread that is not launch machinery (see
    :func:`breakdown.trace_common.is_launcher_frame`).

    This is the **fallback** for a trace captured without the kernel-launch
    hooks: it recovers the launcher's name and file (so the op is readable and
    the replay knows what to import) but not its operands. A hooked capture
    carries both in a ``kernel::`` span.
    """
    frames: list[tuple[float, float, str, int, str]] = []
    for evt in events:
        if evt.get("cat") != "python_function" or evt.get("tid") != worker_tid:
            continue
        match = _PY_FRAME_RE.match(evt.get("name", ""))
        if not match:
            continue                     # "nn.Module: X", "<built-in ...>", ...
        file, func = match.group("file"), match.group("func").strip()
        if not is_launcher_frame(file, func):
            continue
        ts = evt.get("ts", 0)
        frames.append((ts, ts + (evt.get("dur", 0) or 0), file,
                       int(match.group("line")), func))
    frames.sort(key=lambda f: (f[0], -(f[1] - f[0])))
    return frames


def _launcher_at(launch_ts: float,
                 frames: list[tuple[float, float, str, int, str]]
                 ) -> dict[str, Any] | None:
    """The innermost launcher frame containing ``launch_ts``.

    Innermost, not outermost: the frame closest to the launch is the one that
    actually called the kernel, so it names the kernel and is the entry point a
    replay must call. An outer frame is the caller of the launcher, which is a
    different (usually context-bound) function.
    """
    best: dict[str, Any] | None = None
    best_dur = float("inf")
    for ts, end, file, line, func in frames:
        if ts <= launch_ts < end:
            dur = end - ts
            if dur < best_dur:
                best_dur = dur
                best = {"file": file, "line": line, "func": func}
    return best


def _attribute_kernels(roots: list[_Raw],
                       launches: list[tuple[float, str, float, dict[str, Any] | None]]
                       ) -> None:
    """Attribute every device kernel to its host launch site.

    * If the launch sits inside a real (non-plumbing) op — ``aten::mm``,
      ``c10d::allreduce_``, ``vllm::unified_attention_with_output`` ... — the
      kernel's device time is added to that op.
    * If it sits directly in a module (or only inside tensor-plumbing ops), the
      kernel is surfaced as a synthetic ``triton::<kernel>`` op on the enclosing
      module. This is how Triton-compiled kernels (RMSNorm, the lightning
      indexer, block-sparse attention) — which never emit an ``aten``/``_C``
      ``cpu_op`` — become visible. FlashInfer kernels are named after their
      public API frame (``flashinfer::gemma_fused_add_rmsnorm``) when known.
    * If the launch site has **no enclosing module at all** (a module-less
      top-level op subtree — e.g. a bare sampler/logits op that isn't part of
      any decoder module), the kernel's device time is folded into that deepest
      op's own time rather than **silently dropped**, so no device time is ever
      lost. Such launch sites lie outside the reconstructed phase trees, so they
      are not shown as phase leaves, but their time stays conserved for the op
      breakdown / reports. Every kernel launched *inside a module subtree* — i.e.
      every in-phase kernel — always lands on a leaf op via the two cases above.
    """
    # Parent map (child id → parent) for walking up to the enclosing module.
    parent: dict[int, _Raw] = {}
    stack = list(roots)
    while stack:
        n = stack.pop()
        for c in n.children:
            parent[id(c)] = n
            stack.append(c)

    # Accumulate synthetic op device time per (module, kernel-name) so repeated
    # launches across forward passes collapse into one op node per module.
    synth: dict[tuple[int, str], _Raw] = {}
    for ts, name, dur, frame in launches:
        api_name = (frame or {}).get("func")
        node = _deepest_at(roots, ts)
        if node is None:
            continue
        if node.kind == "op" and node.label not in _PLUMBING_OPS:
            node.self_dev += dur
            continue
        mod = _enclosing_module(node, parent)
        if mod is None:
            # No enclosing module (module-less top-level op subtree). Don't drop
            # the device time — fold it into the deepest op so it is conserved
            # and still attributed to an op leaf.
            if node.kind == "op":
                node.self_dev += dur
            continue
        key = (id(mod), api_name or _clean_kernel_name(name))
        op = synth.get(key)
        if op is None:
            op = _Raw("op", _synthetic_op_label(name, api_name), ts, 0.0)
            # Where this kernel's Python entry point lives, and what it was
            # called with. The replay benchmark imports exactly this file and
            # attribute and rebuilds exactly these operands, instead of
            # guessing a module path from the op name and a layout from the
            # model config.
            op.launch = {k: frame[k] for k in ("file", "line", "func")
                         if k in (frame or {})} or None
            _apply_recorded_args(op, (frame or {}).get("args"))
            synth[key] = op
            mod.children.append(op)
            parent[id(op)] = mod
        op.self_dev += dur


def _apply_recorded_args(op: _Raw, slots: Any) -> None:
    """Attach a kernel span's recorded operands to a synthetic op.

    Fills the same three fields a ``cpu_op`` would: the full ordered argument
    slots (for replay), and the tensor shapes / dtypes (for the shape, memory
    and FLOPs analysis), kept aligned. A slot the hook could not describe
    (``opaque``) is kept in its position so the argument list stays aligned with
    the function's signature — the replay reports it rather than guessing.
    """
    if not isinstance(slots, list) or not slots:
        return
    op.arg_slots = [dict(s) for s in slots if isinstance(s, dict)]
    shapes: list[list[int]] = []
    dtypes: list[str] = []
    for slot in op.arg_slots:
        items = (slot.get("items") or []) if slot.get("kind") == "tensorlist" \
            else ([slot] if slot.get("kind") == "tensor" else [])
        for item in items:
            dims = [int(d) for d in item.get("dims") or []]
            if not dims:
                continue
            shapes.append(dims)
            dtypes.append(str(item.get("dtype") or ""))
    op.shapes = shapes
    op.dtypes = dtypes
    op.dtype = dtypes[0] if dtypes else ""


def _kernel_leaf_coverage(
    roots: list[_Raw],
    launches: list[tuple[float, str, float, dict[str, Any] | None]],
) -> dict[str, float]:
    """Read-only classifier: where does each collected device kernel land?

    Mirrors :func:`_attribute_kernels` **without mutating** the forest, so it can
    be run on the raw forest (before attribution) to verify coverage. For each
    collected launch it decides whether the device time lands on a leaf op (a
    real op, a module's synthetic op, or a folded module-less op) or is dropped
    because the launch ``ts`` falls in a gap outside every worker-thread node.

    Returns totals in microseconds and counts::

        {"total_us", "on_leaf_us", "dropped_gap_us",
         "n_total", "n_on_leaf", "n_dropped_gap"}

    A well-formed trace has ``dropped_gap_us`` limited to kernels launched
    *between* module forwards (warm-up / metadata prep), never to kernels
    launched inside a kept prefill/decode step subtree.
    """
    parent: dict[int, _Raw] = {}
    stack = list(roots)
    while stack:
        n = stack.pop()
        for c in n.children:
            parent[id(c)] = n
            stack.append(c)

    total = on_leaf = dropped = 0.0
    n_total = n_leaf = n_drop = 0
    for ts, _name, dur, _api in launches:
        total += dur
        n_total += 1
        node = _deepest_at(roots, ts)
        if node is None:
            dropped += dur
            n_drop += 1
            continue
        if node.kind == "op" and node.label not in _PLUMBING_OPS:
            on_leaf += dur
            n_leaf += 1
            continue
        mod = _enclosing_module(node, parent)
        if mod is None:
            # Folded into the deepest op (module-less launch site) — conserved.
            on_leaf += dur
            n_leaf += 1
            continue
        on_leaf += dur
        n_leaf += 1
    return {"total_us": total, "on_leaf_us": on_leaf, "dropped_gap_us": dropped,
            "n_total": n_total, "n_on_leaf": n_leaf, "n_dropped_gap": n_drop}
