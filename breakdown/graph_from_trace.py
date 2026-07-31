# SPDX-License-Identifier: Apache-2.0
"""Profile-first model graph reconstruction.

Rebuilds the model's module/op tree *directly from a torch profiler trace*
instead of deriving it statically from the HuggingFace config. The trace is the
ground truth for what actually executed on the hardware, so the reconstructed
graph tracks whatever vLLM/backend actually dispatched — no hardcoded structure.

What the trace gives us (captured with ``with_stack=True`` and
``record_shapes=True``):

* ``python_function`` events named ``nn.Module: <ClassName>_<idx>`` — these nest
  by time-containment and reconstruct the real module hierarchy
  (``DecoderLayer → Attention → QKVParallelLinear`` ...).
* ``cpu_op`` events carrying ``Input Dims`` (shapes), ``Input type`` (dtypes) and
  an ``External id``.
* ``kernel`` / ``gpu_memcpy`` events whose device time is attributed to the
  module/op that launched them, by **launch-site containment**: each kernel is
  linked to its host launch call (``kernel.correlation → xpu_runtime`` — the
  "flow arrow" in the trace viewer), whose timestamp falls inside the enclosing
  module/op interval on the worker thread. This is robust to ``torch.compile``:
  Triton-compiled kernels (RMSNorm, lightning indexer, block-sparse attention)
  that never emit an ``aten``/``_C`` ``cpu_op`` surface as ``triton::`` ops on
  their module, and fused/eager kernels are handled identically.

The output dict is a serialized module tree (``prefill`` / ``decode`` trees,
``symbols``, ``config``) that the web UI and the Shape Matrix export both render
/ consume unchanged.
"""

from __future__ import annotations

import ast
import gzip
import json
from typing import Any

from .analyzer import DTYPE_BYTES, dtype_size, estimate_flops, estimate_memory
from .classifier import classify_op
from .trace_common import (
    MODULE_SPAN_PREFIX,
    _is_overhead_event,
    module_span_display_name,
    parse_module_span,
)
from .trace_parser import _infer_device_from_trace, _infer_role, _strip_instance_idx

# Chrome-trace categories that carry actual *device* (GPU/XPU) work — real
# compute kernels, memcpys and memsets. Every event in one of these categories
# must be attributed to a leaf op (that is the whole point of the reconstruction:
# no device time is lost). Host-side launch-API events (``_RUNTIME_CATEGORIES``,
# below) are deliberately **not** in this set: they carry no device time, they
# only pinpoint where a kernel was launched. Historically ``cuda_runtime`` was
# lumped in here, which caused pure host bookkeeping calls that launch nothing
# (``cudaEventQuery``, ``cudaStreamWaitEvent``, ``cudaDeviceGetAttribute``,
# ``cudaStreamIsCapturing``, ``cudaEventRecord``) to be mistaken for kernel
# launches and surfaced as bogus ``triton::cudaEventQuery`` leaf ops.
_DEVICE_KERNEL_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "xpu_op",
                             "gpu_op", "cuda_op", "gpu_kernel"}

# Host-side launch-API event categories (``cudaLaunchKernelExC``,
# ``cuLaunchKernelEx``, ``urEnqueueKernelLaunch``, ...). These correlate to a
# device kernel and are used to locate its launch site; they are not surfaced as
# ops themselves when the kernel they launch is already surfaced. ``cuda_driver``
# is the CUDA *driver* launch API (``cuLaunchKernel``/``cuLaunchKernelEx``) that
# Triton uses directly — without it, Triton kernels (e.g. the MoE
# ``fused_moe_kernel`` grouped GEMM) have no runtime-API launch event, so
# launch-site lookup falls back to ``External id`` and misattributes them to the
# enclosing custom op's start (collapsing all expert GEMM time into
# ``vllm::moe_forward_shared`` instead of the ``moe`` node).
_RUNTIME_CATEGORIES = {"cuda_runtime", "cuda_driver", "xpu_runtime"}

# Ops that are pure tensor plumbing — kept out of the reconstructed op lists to
# avoid drowning the real compute ops. They carry no device time anyway.
_PLUMBING_OPS = frozenset({
    "aten::slice", "aten::as_strided", "aten::view", "aten::reshape",
    "aten::select", "aten::expand", "aten::unsqueeze", "aten::squeeze",
    "aten::t", "aten::transpose", "aten::permute", "aten::contiguous",
    "aten::detach", "aten::empty", "aten::empty_like", "aten::empty_strided",
    "aten::resize_", "aten::narrow", "aten::split", "aten::split_with_sizes",
    "aten::chunk", "aten::flatten", "aten::_unsafe_view", "aten::alias",
    "aten::lift_fresh", "aten::set_", "aten::_reshape_alias",
})

# Module display names that are valid semantic roles for their contained ops.
# When a module's resolved name (from ref_tree or heuristic) is in this set,
# all ops inside it inherit this role — overriding path-based inference that
# can be wrong due to GPU async timing causing incorrect time-containment.
_KNOWN_MODULE_ROLES = frozenset({
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
    "embedding", "lm_head", "norm", "q_norm", "k_norm",
    "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
})


# ===================================================================
# Trace loading + low-level event extraction
# ===================================================================

def _load_trace(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _parse_input_dims(args: dict) -> list[list[int]]:
    """Extract numeric input shapes from a cpu_op's args."""
    return _parse_input_dims_types(args)[0]


def _normalize_dtype(t: Any) -> str:
    """Normalize a trace dtype token to a ``DTYPE_BYTES`` key.

    ``'c10::BFloat16' → 'bfloat16'``, ``'Float' → 'float'``,
    ``'c10::Float8_e4m3fn' → 'float8_e4m3fn'``. Unknown tokens are lowered and
    returned as-is (``dtype_size`` then falls back to 2 bytes).
    """
    if not t:
        return ""
    name = str(t).split("::")[-1].lower().replace("torch.", "")
    name = name.replace("half", "float16")
    return name


def _parse_input_args(args: dict) -> list[dict]:
    """Full ordered argument slots of a ``cpu_op``, for benchmark replay.

    Unlike :func:`_parse_input_dims_types` — which keeps only the tensor
    operands, because that is all the shape/memory analysis needs — this keeps
    **every** slot in its original position, so
    :mod:`breakdown.bench` can rebuild the call: which slot is a tensor (with
    its dims/dtype/strides), which is a ``TensorList``, and which is a scalar /
    ``None`` (with the value the profiler recorded in ``Concrete Inputs``).

    Slot kinds: ``tensor`` | ``tensorlist`` | ``scalar`` | ``none``.
    """
    dims = args.get("Input Dims")
    if not isinstance(dims, (list, tuple)):
        return []
    types = args.get("Input type")
    types = types if isinstance(types, (list, tuple)) else []
    strides = args.get("Input Strides")
    strides = strides if isinstance(strides, (list, tuple)) else []
    concrete = args.get("Concrete Inputs")
    concrete = concrete if isinstance(concrete, (list, tuple)) else []

    def at(seq, i):
        return seq[i] if i < len(seq) else None

    def ints(v) -> list[int]:
        if not isinstance(v, (list, tuple)):
            return []
        return [int(d) for d in v if isinstance(d, (int, float))]

    out: list[dict] = []
    for i, entry in enumerate(dims):
        raw_t = at(types, i)
        dt = _normalize_dtype(raw_t)
        st = at(strides, i)
        val = at(concrete, i)
        val = "" if val is None else str(val)
        if isinstance(entry, (list, tuple)) and entry and all(
                isinstance(x, (list, tuple)) for x in entry):
            # TensorList: one nesting level deeper (c10d::allreduce_, foreach).
            elem_dt = dt if dt in DTYPE_BYTES else ""
            items = []
            for j, sub in enumerate(entry):
                items.append({
                    "dims": ints(sub), "dtype": elem_dt,
                    "strides": ints(at(st, j) if isinstance(st, (list, tuple))
                                    else None),
                })
            out.append({"kind": "tensorlist", "items": items})
        elif isinstance(entry, (list, tuple)) and entry:
            out.append({"kind": "tensor", "dims": ints(entry), "dtype": dt,
                        "strides": ints(st)})
        elif dt in DTYPE_BYTES:
            # A 0-dim tensor: empty dims but a real dtype token.
            out.append({"kind": "tensor", "dims": [], "dtype": dt,
                        "strides": []})
        elif raw_t:
            out.append({"kind": "scalar", "type": str(raw_t), "value": val})
        else:
            out.append({"kind": "none", "value": val})
    return out


def _parse_input_dims_types(args: dict) -> tuple[list[list[int]], list[str]]:
    """Extract numeric input shapes and per-tensor dtypes, kept aligned.

    Only non-empty tensor inputs are retained (scalars / empty inputs dropped);
    the returned dtype list is parallel to the shape list, so ``dtypes[i]`` is
    the recorded dtype of the tensor with shape ``shapes[i]``.
    """
    dims = args.get("Input Dims")
    if dims is None:
        raw = args.get("input_shapes")
        if isinstance(raw, str):
            try:
                dims = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                dims = None
    if not isinstance(dims, (list, tuple)):
        return [], []
    types = args.get("Input type")
    if not isinstance(types, (list, tuple)):
        types = []
    shapes: list[list[int]] = []
    dtypes: list[str] = []
    for i, tensor in enumerate(dims):
        if not isinstance(tensor, (list, tuple)) or not tensor:
            continue
        raw_dt = _normalize_dtype(types[i] if i < len(types) else "")
        # A ``TensorList`` input nests one level deeper: its "Input Dims" entry
        # is a *list of per-tensor shape lists* rather than a single shape, e.g.
        # ``c10d::allreduce_`` records ``[[2, 6144]]`` (a one-element list of
        # tensors) with ``Input type`` ``'TensorList'``. Without unwrapping the
        # container level the whole entry was dropped, so collective/foreach ops
        # surfaced with no shape and no dtype. Surface each contained tensor's
        # shape; the container label is not an element dtype, so leave the dtype
        # empty for the residual-stream inference pass to fill from a neighbour.
        if all(isinstance(t, (list, tuple)) for t in tensor):
            list_dt = "" if raw_dt not in DTYPE_BYTES else raw_dt
            for sub in tensor:
                shape = [int(d) for d in sub if isinstance(d, (int, float))]
                if shape:
                    shapes.append(shape)
                    dtypes.append(list_dt)
            continue
        shape = [int(d) for d in tensor if isinstance(d, (int, float))]
        if shape:
            shapes.append(shape)
            dtypes.append(raw_dt)
    return shapes, dtypes


def _first_dtype(args: dict) -> str:
    """First concrete input dtype, e.g. 'c10::BFloat16' → 'bfloat16'."""
    for name in _parse_input_dims_types(args)[1]:
        if name in DTYPE_BYTES:
            return name
    return ""


def _collect_kernel_launches(events: list[dict], worker_tid: Any
                             ) -> list[tuple[float, str, float, str | None]]:
    """Collect every device kernel as ``(host_launch_ts, name, device_us,
    friendly_api_name)``.

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

    # Public FlashInfer API frames on the worker thread, so a Python-direct
    # FlashInfer kernel can be named after its public API entrypoint (e.g.
    # ``flashinfer/norm/__init__.py(433): gemma_fused_add_rmsnorm``) instead of
    # the raw cutlass functor symbol.
    fi_api_frames = _collect_flashinfer_api_frames(events, worker_tid)
    # Public xattention API frames on the worker thread, so a Python-direct
    # MiniMax-M3 MSA SYCL kernel can be named after its ``xattention.py``
    # wrapper (e.g. ``minimax_m3_index_score``, ``minimax_m3_sparse_attn``)
    # instead of the raw ``flash_xpu::(anonymous namespace)::...`` symbol.
    fx_api_frames = _collect_flash_xpu_api_frames(events, worker_tid)

    def _friendly(name: str, ts: float) -> str | None:
        return (_flashinfer_api_name(name, ts, fi_api_frames)
                or _flash_xpu_api_name(name, ts, fx_api_frames))

    launches: list[tuple[float, str, float, str | None]] = []
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
            launches.append((ts, name, dur, _friendly(name, ts)))
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
            launches.append((ts, name, dur, _friendly(name, ts)))
    return launches


def _collect_flashinfer_api_frames(events: list[dict], worker_tid: Any
                                   ) -> list[tuple[float, float, str]]:
    """Enclosing public FlashInfer API frames as ``(ts, end, funcname)``.

    FlashInfer's public entrypoints live in ``flashinfer/**/__init__.py`` (e.g.
    ``gemma_fused_add_rmsnorm``, ``gemma_rmsnorm``). We keep only those public
    (non-underscore) ``__init__.py`` frames on the worker thread so a
    Python-direct FlashInfer kernel can be named after the API that launched it
    rather than the raw cutlass functor symbol — matching the readable XPU
    kernel names.
    """
    frames: list[tuple[float, float, str]] = []
    for evt in events:
        if evt.get("cat") != "python_function" or evt.get("tid") != worker_tid:
            continue
        nm = evt.get("name", "")
        if "flashinfer" not in nm or "__init__.py(" not in nm:
            continue
        func = nm.rsplit("): ", 1)[-1].strip()
        if not func or func.startswith("_"):
            continue
        ts = evt.get("ts", 0)
        frames.append((ts, ts + (evt.get("dur", 0) or 0), func))
    return frames


def _flashinfer_api_name(kernel_name: str, launch_ts: float,
                         fi_api_frames: list[tuple[float, float, str]]
                         ) -> str | None:
    """Public FlashInfer API name for a kernel launched at ``launch_ts``.

    Returns the outermost enclosing public ``__init__.py`` FlashInfer frame's
    function name (``gemma_fused_add_rmsnorm``), or ``None`` when the kernel is
    not a FlashInfer kernel or no such frame contains the launch.
    """
    if "flashinfer" not in kernel_name.lower():
        return None
    best: str | None = None
    best_dur = -1.0
    for ts, end, func in fi_api_frames:
        if ts <= launch_ts < end:
            dur = end - ts
            if dur > best_dur:
                best, best_dur = func, dur
    return best


def _collect_flash_xpu_api_frames(events: list[dict], worker_tid: Any
                                  ) -> list[tuple[float, float, str]]:
    """Enclosing public xattention API frames as ``(ts, end, funcname)``.

    The MiniMax-M3 MSA SYCL kernels (lightning indexer + block-sparse attend)
    are launched from the thin ``xattention.py`` wrappers
    (``minimax_m3_index_score``, ``minimax_m3_index_topk``,
    ``minimax_m3_index_decode``, ``minimax_m3_sparse_attn``,
    ``minimax_m3_sparse_attn_decode``). We keep only those public
    (non-underscore) ``xattention.py`` frames on the worker thread so a
    Python-direct ``flash_xpu`` kernel can be named after the API that launched
    it rather than the raw ``flash_xpu::(anonymous namespace)::...`` symbol.
    """
    frames: list[tuple[float, float, str]] = []
    for evt in events:
        if evt.get("cat") != "python_function" or evt.get("tid") != worker_tid:
            continue
        nm = evt.get("name", "")
        if "xattention.py(" not in nm:
            continue
        func = nm.rsplit("): ", 1)[-1].strip()
        if not func or func.startswith("_"):
            continue
        ts = evt.get("ts", 0)
        frames.append((ts, ts + (evt.get("dur", 0) or 0), func))
    return frames


def _flash_xpu_api_name(kernel_name: str, launch_ts: float,
                        fx_api_frames: list[tuple[float, float, str]]
                        ) -> str | None:
    """Public xattention API name for a ``flash_xpu`` kernel at ``launch_ts``.

    Returns the outermost enclosing public ``xattention.py`` frame's function
    name (``minimax_m3_index_score``), or ``None`` when the kernel is not an
    xattention (``flash_xpu``) kernel or no such frame contains the launch.
    """
    if "flash_xpu" not in kernel_name.lower():
        return None
    best: str | None = None
    best_dur = -1.0
    for ts, end, func in fx_api_frames:
        if ts <= launch_ts < end:
            dur = end - ts
            if dur > best_dur:
                best, best_dur = func, dur
    return best


# ===================================================================
# Raw nesting tree (modules + ops) via time-containment
# ===================================================================

class _Raw:
    """A node in the raw trace nesting tree (a module or a leaf op)."""

    __slots__ = ("kind", "label", "ts", "end", "dur", "ext", "shapes",
                 "dtype", "dtypes", "children", "self_dev", "sub_dev",
                 "attr_name", "arg_slots")

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


def _clean_kernel_name(name: str) -> str:
    """Strip the cutlass/functor object-repr + tensor-ptr tail from a raw
    device-kernel symbol so synthetic op names stay readable.

    e.g. ``kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128...___T_0``
    → ``kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel``.
    """
    idx = name.find("_object_at")
    if idx > 0:
        name = name[:idx]
    return name


def _synthetic_op_label(name: str, api_name: str | None = None) -> str:
    """Namespaced display label for a Python-direct device kernel (no cpu_op).

    FlashInfer kernels get a ``flashinfer::`` namespace and MiniMax-M3 MSA
    xattention SYCL kernels get a ``flash_xpu::`` namespace (rather than
    ``triton::``, which misrepresented them as Triton-compiled); every other
    orphan kernel keeps ``triton::``. When the launching public API name is
    known (``api_name``, e.g. ``gemma_fused_add_rmsnorm`` /
    ``minimax_m3_index_score``) it is used in place of the raw cutlass functor /
    ``flash_xpu::(anonymous namespace)::...`` symbol so the label is short and
    matches the readable API; otherwise the ``_object_at...`` object-repr tail
    is dropped from the raw symbol so the name stays readable.
    """
    clean = _clean_kernel_name(name)
    low = clean.lower()
    if "flash_xpu" in low:
        return "flash_xpu::" + (api_name or clean)
    if api_name:
        return "flashinfer::" + api_name
    if "flashinfer" in low:
        return "flashinfer::" + clean
    return "triton::" + clean


def _attribute_kernels(roots: list[_Raw],
                       launches: list[tuple[float, str, float, str | None]]
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
    for ts, name, dur, api_name in launches:
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
            synth[key] = op
            mod.children.append(op)
            parent[id(op)] = mod
        op.self_dev += dur


def _kernel_leaf_coverage(
    roots: list[_Raw],
    launches: list[tuple[float, str, float, str | None]],
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


# Residual-stream ops whose shape/dtype the trace does not record on the op
# itself: tensor-parallel collectives (``c10d::allreduce_`` records its tensor
# as a ``TensorList`` with no element dtype) and the RMSNorm/LayerNorm kernels
# that vLLM launches straight from Python via Triton/FlashInfer (no ``cpu_op``,
# so they surface as synthetic ``triton::``/``flashinfer::`` ops with no shape
# at all). All of them operate on the residual hidden state ``[tokens, H]`` in
# the model's activation dtype, so their shape/dtype can be recovered from the
# nearest neighbouring op that *does* carry a hidden-state tensor.
_COLLECTIVE_KEYWORDS = (
    "allreduce", "all_reduce", "allgather", "all_gather",
    "reduce_scatter", "reducescatter", "all_to_all", "alltoall",
)


def _is_hidden_state_op(label: str) -> bool:
    """True for a residual-stream op (norm / collective) worth inferring shapes.

    Restricted to norm-family and collective ops so attention kernels, ``zeros``,
    ``item`` and other genuinely shape-less / differently-shaped ops are left
    untouched (they must not inherit the ``[tokens, H]`` residual shape).
    """
    low = label.lower()
    if ("rmsnorm" in low or "rms_norm" in low
            or "layernorm" in low or "layer_norm" in low):
        return True
    return any(k in low for k in _COLLECTIVE_KEYWORDS)


def _infer_hidden_activation_ops(roots: list[_Raw], hidden_size: int | None
                                 ) -> None:
    """Fill missing shape/dtype on residual-stream ops from a neighbour.

    Norm kernels (Triton/FlashInfer, launched straight from Python) carry no
    ``cpu_op`` and so no shape, and TP collectives record only a dtype-less
    ``TensorList``. Both operate on the residual hidden state ``[tokens, H]``, so
    for each such op missing a shape and/or a real dtype we borrow both from the
    op nearest in execution order that carries a genuine ``[tokens, H]`` tensor
    (2-D, trailing dim == ``hidden_size``, real dtype) — picking the nearest by
    timestamp keeps the correct per-step token count (prefill ``S`` vs decode
    ``B``). No-op when ``hidden_size`` is unknown or no reference exists.
    """
    if not hidden_size:
        return
    H = int(hidden_size)
    ops: list[_Raw] = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.kind == "op":
            ops.append(n)
        stack.extend(n.children)

    # Reference hidden-state tensors: (ts, [tokens, H], dtype).
    refs: list[tuple[float, list[int], str]] = []
    for o in ops:
        for sh, dt in zip(o.shapes, o.dtypes):
            if len(sh) == 2 and sh[1] == H and dt in DTYPE_BYTES:
                refs.append((o.ts, list(sh), dt))
                break
    if not refs:
        return

    for o in ops:
        need_shape = not o.shapes
        need_dtype = not any(d in DTYPE_BYTES for d in o.dtypes)
        if not (need_shape or need_dtype) or not _is_hidden_state_op(o.label):
            continue
        _ts, ref_shape, ref_dt = min(refs, key=lambda r: abs(r[0] - o.ts))
        if need_shape:
            o.shapes = [list(ref_shape)]
            o.dtypes = [ref_dt]
        else:
            o.dtypes = [ref_dt for _ in o.shapes]
        o.dtype = ref_dt


# MiniMax-M3 MSA (sparse attention + lightning indexer) kernels are launched
# straight from Python with no ``cpu_op`` on **both** backends — XPU dispatches
# hand-tuned SYCL kernels from the ``xattention.py`` wrappers (surfacing as
# ``flash_xpu::<api>`` ops), CUDA dispatches ``triton.jit`` kernels from
# ``models/minimax_m3/common/ops/{sparse_attn,index_topk}.py`` (surfacing as
# ``triton::<kernel>`` ops). Either way they carry no shape at all.
#
# Their primary tensor layout is fixed by the wrapper signatures, so it can be
# rebuilt from the model config + the step's token count (``total_q`` — ``S``
# prefill / ``B`` decode):
#   ``attn``  — block-sparse GQA attend query/output ``[total_q, n_h, d]``
#   ``index`` — lightning-indexer query ``[total_q, n_idx, idx_d]``
#   ``topk``  — indexer top-k block ids ``[n_idx, total_q, topk_blocks]``
# Matching is by substring on the op's base name so it is device-agnostic;
# ``topk`` is probed first because the XPU API name ``minimax_m3_index_topk``
# also contains the ``index`` family's prefix.
_MSA_KERNEL_LAYOUTS: tuple[tuple[tuple[str, ...], str], ...] = (
    # indexer top-k (XPU: minimax_m3_index_topk; CUDA: _topk_index[_partial
    # |_merge]_kernel)
    (("index_topk", "topk_index"), "topk"),
    # block-sparse GQA attend (XPU: minimax_m3_sparse_attn[_decode];
    # CUDA: _gqa_sparse_{fwd,decode}_kernel + _merge_topk_attn_out_kernel)
    (("sparse_attn", "gqa_sparse", "merge_topk_attn"), "attn"),
    # lightning-indexer block score (XPU: minimax_m3_index_score /
    # minimax_m3_index_decode; CUDA: _index_block_score / _decode_index_score)
    (("index_score", "index_block_score", "index_decode"), "index"),
)


# Ops that merely re-view a *weight* tensor. A weight is ``[out_features, H]``,
# i.e. shaped exactly like a residual hidden state, so these must be excluded
# when inferring a step's token count from a neighbouring ``[tokens, H]`` op.
_WEIGHT_PLUMBING_OPS = {"t", "transpose", "permute", "detach"}


def _msa_kernel_layout(op_label: str) -> str | None:
    """Return the MSA primary-tensor layout for an op label, if it is one."""
    base = op_label.split("::")[-1].lower()
    for subs, layout in _MSA_KERNEL_LAYOUTS:
        if any(s in base for s in subs):
            return layout
    return None


def _infer_attention_kernel_shapes(roots: list[_Raw], summary: dict,
                                   tp_size: int) -> None:
    """Reconstruct shape/dtype for shape-less MiniMax-M3 MSA / indexer kernels.

    The MSA sparse-attention and lightning-indexer kernels carry no ``cpu_op`` on
    either backend (XPU ``flash_xpu`` SYCL kernels launched from
    ``xattention.py``; CUDA ``triton.jit`` kernels launched from
    ``common/ops/{sparse_attn,index_topk}.py``), so ``_attribute_kernels``
    surfaces them with no shape at all. Their primary-tensor layout is fixed by
    the wrapper signatures, so we rebuild it from the config (per-rank
    ``num_heads``/``head_dim`` for the attend kernels,
    ``sparse_num_index_heads``/``sparse_index_dim`` for the indexer kernels,
    ``sparse_topk_blocks`` for the top-k block ids) and the per-step token count
    ``total_q`` — taken as the leading dim of the nearest neighbouring activation
    op (so prefill gets ``S`` and decode ``B``). Kernels with no known layout
    fall back to borrowing the neighbour's activation shape so they at least
    carry the token dim + dtype. No-op without neighbour activations.
    """
    ops: list[_Raw] = []
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n.kind == "op":
            ops.append(n)
        stack.extend(n.children)

    # Neighbour activation references: (ts, leading token dim, first shape, dtype).
    act_refs: list[tuple[float, int, list[int], str]] = []
    tok_refs: list[tuple[float, int]] = []
    H = summary.get("hidden_size")
    for o in ops:
        if not o.shapes or len(o.shapes[0]) < 2:
            continue
        dt = next((d for d in o.dtypes if d in DTYPE_BYTES), "")
        shp = list(o.shapes[0])
        act_refs.append((o.ts, shp[0], shp, dt or "bfloat16"))
        # Token-count references are restricted to genuine residual hidden
        # states: a ``[tokens, H]`` tensor on an op that isn't weight plumbing.
        # A weight is also ``[out_features, H]``, so an ``aten::t`` on one would
        # otherwise hand the MSA kernels the weight's out-features as their row
        # count (the symptom: ``[n_h·d/TP, n_idx/TP, d]`` instead of ``[B, ...]``).
        if (H and len(shp) == 2 and shp[1] == int(H)
                and o.label.split("::")[-1].lower() not in _WEIGHT_PLUMBING_OPS):
            tok_refs.append((o.ts, shp[0]))
    if not act_refs:
        return
    if not tok_refs:  # hidden size unknown / no hidden-state op — best effort
        tok_refs = [(ts, tok) for ts, tok, _shp, _dt in act_refs]

    tp = max(1, int(tp_size or 1))
    n_h = summary.get("num_heads")
    d = summary.get("head_dim")
    n_idx = summary.get("sparse_num_index_heads")
    idx_d = summary.get("sparse_index_dim")
    topk_blocks = summary.get("sparse_topk_blocks")

    for o in ops:
        if o.shapes:
            continue
        layout = _msa_kernel_layout(o.label)
        # ``flash_xpu`` ops with no recognised layout still get the neighbour
        # fallback (the whole namespace is MSA); other ops are left untouched.
        if layout is None and "flash_xpu" not in o.label.lower():
            continue
        _ts, _tok, act_shape, dt = min(act_refs, key=lambda r: abs(r[0] - o.ts))
        token = min(tok_refs, key=lambda r: abs(r[0] - o.ts))[1]
        shape: list[int] | None = None
        if layout == "attn" and n_h and d:
            shape = [token, max(1, int(n_h) // tp), int(d)]
        elif layout == "index" and n_idx and idx_d:
            shape = [token, max(1, int(n_idx) // tp), int(idx_d)]
        elif layout == "topk" and n_idx and topk_blocks:
            # The top-k kernels read a ``[n_idx, total_q, max_block]`` score and
            # write ``[n_idx, total_q, topk]`` block ids. Only the top-k width is
            # config-derivable (``max_block`` depends on the runtime max seq len),
            # so report the block-id tensor — int32, not the activation dtype.
            shape = [max(1, int(n_idx) // tp), token, int(topk_blocks)]
            dt = "int32"
        if shape is None:
            shape = list(act_shape)  # fallback: representative activation
        o.shapes = [shape]
        o.dtypes = [dt]
        o.dtype = dt


# vLLM V1 runs the sampler (and similar post-processing) functionally rather
# than as an ``nn.Module``, so the profiler emits its top-level call as a
# source-located ``python_function`` frame (e.g. ".../sample/sampler.py(72):
# __call__") instead of ``nn.Module: Sampler``. Without a module boundary the
# sampler's ops become bare op roots and get dropped by ``_partition_steps``
# (which keeps only module roots), so the reconstructed tree stops at
# ``LogitsProcessor``. Map the sampler's ``__call__`` frame to a synthetic
# ``Sampler`` module so its ops attach to a proper node.
#
# The same trick surfaces the MoE routing and expert compute. vLLM dispatches
# the whole MoE block as one fused custom op (``vllm::moe_forward_shared``)
# whose Python body calls the router (``fused_topk_bias`` — sigmoid/topk/gather)
# and the routed-expert kernels as plain functions, not ``nn.Module`` forwards.
# The expert compute has a per-backend entry frame: XPU dispatches it through
# ``xpu_fused_moe`` (``fused_moe_interface.py`` — grouped GEMM/remap/gather),
# CUDA through the Triton modular kernel ``apply`` (``experts/triton_moe.py`` —
# ``moe_align_block_size`` → ``fused_moe_kernel`` grouped GEMM → activation →
# ``moe_sum``). Without a boundary their ops and kernels collapse into the single
# ``moe_forward_shared`` op node, so the ``FusedMoE`` graph showed neither the
# router nor the experts (only the hoisted ``shared_experts`` MLP). Promoting the
# frames to synthetic modules makes ``FusedMoE`` read
# ``shared_experts → router → moe → reduce``; each is then hoisted out of the
# wrapping op by ``_hoist_modules_under_ops``.
#
# It also groups the fused all-reduce + RMSNorm. Gemma-style models (MiniMax-M3)
# fuse the residual tensor-parallel all-reduce with the following RMSNorm as
# ``fused_allreduce_gemma_rms_norm``, a ``python_function`` that wraps both the
# ``c10d::allreduce_`` op **and** the ``MiniMAXGemmaRMSNorm`` module. Without a
# boundary the all-reduce and the norm float up as two unrelated siblings of the
# decoder layer (a bare ``c10d::allreduce_`` op next to a ``post_attention_-
# layernorm`` norm), which reads as an unexplained "norm" at the layer edge.
# Promoting the frame makes it a parent node ``fused_allreduce_gemma_rms_norm →
# {allreduce, norm}`` so the fusion is explicit.
#
# ``(path_substr, funcname, synthetic_class, display_name)`` — ``display_name``
# is the attribute-style label shown in the graph (``None`` → derive from the
# class). Only the outermost matching frame becomes a boundary per step.
_FUNCTIONAL_MODULE_FRAMES = (
    ("sample/sampler.py", "__call__", "Sampler", None),
    ("fused_topk_bias_router.py", "fused_topk_bias", "FusedTopKBiasRouter",
     "router"),
    ("fused_moe_interface.py", "xpu_fused_moe", "XpuFusedMoE", "moe"),
    ("experts/triton_moe.py", "apply", "TritonExperts", "moe"),
    ("fused_allreduce_gemma_rms_norm.py", "fused_allreduce_gemma_rms_norm",
     "FusedAllreduceGemmaRMSNorm", "fused_allreduce_gemma_rms_norm"),
)


def _functional_module_class(name: str) -> tuple[str, str | None] | None:
    """Return ``(class, display_name)`` for a functional (non-nn.Module) frame.

    ``name`` is a torch-profiler python_function label of the form
    ``"<path>(<lineno>): <func>"``. Returns the mapped ``(synthetic_class,
    display_name)`` when the frame is a recognised functional module boundary
    (``display_name`` may be ``None`` to derive it from the class), else
    ``None``.
    """
    head, sep, func = name.partition("): ")
    if not sep:
        return None
    path = head.rsplit("(", 1)[0]
    for path_substr, funcname, cls, display in _FUNCTIONAL_MODULE_FRAMES:
        if funcname == func and path_substr in path:
            return cls, display
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
    * **Legacy mode** — older traces (and third-party / upload traces) without
      those spans fall back to the class-only ``nn.Module: <Cls>_<idx>``
      ``python_function`` events; names are recovered afterwards by the
      ``module_naming`` overlay (``_apply_ref_names``).
    """
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"
               and e.get("ph") == "X"]
    if not cpu_ops:
        return []

    # Choose the worker thread that ran the model forward. When capture-time
    # module spans are present (research R1), the thread carrying them is an
    # unambiguous anchor to the forward (R6): the "busiest cpu_op thread" guess
    # can pick the wrong worker under tensor parallelism, where several threads
    # dispatch ops. Fall back to the busiest cpu_op thread for legacy traces
    # without spans.
    span_tid_counts: dict[Any, int] = {}
    for e in events:
        if (e.get("ph") == "X" and e.get("cat") == "user_annotation"
                and str(e.get("name", "")).startswith(MODULE_SPAN_PREFIX)):
            span_tid_counts[e.get("tid")] = span_tid_counts.get(e.get("tid"), 0) + 1
    if span_tid_counts:
        worker_tid = max(span_tid_counts, key=span_tid_counts.get)
        named_span_mode = True
    else:
        tid_counts: dict[Any, int] = {}
        for e in cpu_ops:
            tid_counts[e.get("tid")] = tid_counts.get(e.get("tid"), 0) + 1
        worker_tid = max(tid_counts, key=tid_counts.get)
        named_span_mode = False

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
            n.dtype = next((d for d in n.dtypes if d in DTYPE_BYTES), "")
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

    _attribute_kernels(roots, _collect_kernel_launches(events, worker_tid))
    # Surface modules that a fused custom op wraps (see _hoist_modules_under_ops)
    # before rolling up device time so their subtree isn't double-counted.
    roots = _hoist_modules_under_ops(roots)
    # Collapse the same module object recorded twice in one forward (MoE
    # shared-experts overlap) into a single node before device time rolls up.
    _coalesce_duplicate_child_modules(roots)
    _compute_sub_dev(roots)
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


# ===================================================================
# Merge / collapse into display tree
# ===================================================================

def _op_signature(n: _Raw) -> tuple:
    return (n.label, tuple(tuple(s) for s in n.shapes))


def _module_children(node: _Raw) -> list[_Raw]:
    return [c for c in node.children if c.kind == "module"]


def _apply_ref_names(roots: list[_Raw], ref_tree: dict) -> None:
    """Overlay real module attribute names onto the raw module forest.

    Uses the reference module-name tree (from ``module_naming.build_ref_tree``,
    derived from the live model's ``named_modules()``) to assign each raw module
    node its attribute name — ``q_norm``/``k_norm``, ``input_layernorm``,
    ``self_attn``, ... — by matching children on ``(class, order)``. Applied on
    the *raw* forest before finalization so the recovered names also feed the
    structural signature, keeping distinctly-named siblings (q_norm vs k_norm)
    from collapsing while genuinely-repeated layers still merge.
    """
    from .module_naming import (_display_name, _effective_ref_children,
                                _find_ref_desc, _match_ref)

    def align(raw_node: _Raw, rnode: dict) -> None:
        raw_node.attr_name = _display_name(rnode)
        raw_mod_children = _module_children(raw_node)
        present = {_strip_instance_idx(cm.label) for cm in raw_mod_children}
        rchildren = _effective_ref_children(rnode, present)
        used = [False] * len(rchildren)
        for cm in raw_mod_children:
            j = _match_ref(rchildren, used, _strip_instance_idx(cm.label))
            if j is not None:
                used[j] = True
                align(cm, rchildren[j])

    root_cls = ref_tree.get("cls")
    for root in roots:
        rc = _strip_instance_idx(root.label)
        if rc == root_cls:
            align(root, ref_tree)
        else:
            r = _find_ref_desc(ref_tree, rc)
            if r is not None:
                align(root, r)


def _direct_ops(node: _Raw) -> list[_Raw]:
    """Op children that belong to this module (not nested in a child module)."""
    return [c for c in node.children if c.kind == "op"
            and c.label not in _PLUMBING_OPS]


def _merge_modules(instances: list[_Raw], n_forward: int) -> dict:
    """Merge equivalent module instances into one averaged display node.

    ``instances`` are all the raw module nodes that map to the same logical
    module across the merged forward passes. Their per-instance ops and child
    modules are aligned by position and averaged by ``n_forward`` (the number of
    forward passes) so the node shows *one forward's* cost.
    """
    base = instances[0]

    # --- aggregate direct ops by (name, shapes, occurrence-within-instance) ---
    # A single module instance can dispatch the *same* op (identical name +
    # shapes) more than once. A TP decoder layer, for example, runs two
    # identical ``c10d::allreduce_`` residual reductions: the post-attention one
    # (before ``post_attention_layernorm``) and the post-MLP one. The post-MLP
    # allreduce is dispatched after the layer's forward returns, so by
    # time-containment it lands at the *start* of the next layer — giving that
    # layer two same-signature allreduces. Keying only by (name, shapes) merges
    # them into one node positioned at the first (leading) occurrence, which
    # hides the post-attention allreduce entirely. Index each occurrence within
    # its instance — mirroring the child-module grouping below — so repeated ops
    # stay distinct and still align by position across merged forward passes.
    op_groups: dict[tuple, dict] = {}
    op_order: list[tuple] = []
    for inst in instances:
        op_occ: dict[tuple, int] = {}
        for op in _direct_ops(inst):
            sig = _op_signature(op)
            occ = op_occ.get(sig, 0)
            op_occ[sig] = occ + 1
            gkey = (sig, occ)
            g = op_groups.get(gkey)
            if g is None:
                g = {"raw": op, "dev": 0.0, "host": 0.0, "count": 0}
                op_groups[gkey] = g
                op_order.append(gkey)
            g["dev"] += op.sub_dev
            g["host"] += op.dur
            g["count"] += 1

    # --- align child modules by (class, occurrence index within parent) ---
    child_groups: dict[tuple, list[_Raw]] = {}
    child_order: list[tuple] = []
    for inst in instances:
        seen: dict[str, int] = {}
        for cm in _module_children(inst):
            norm = _strip_instance_idx(cm.label)
            idx = seen.get(norm, 0)
            seen[norm] = idx + 1
            key = (norm, idx)
            if key not in child_groups:
                child_groups[key] = []
                child_order.append(key)
            child_groups[key].append(cm)

    # --- combined execution-order layout (base instance, chronological) ---
    # Direct ops and child modules are interleaved by timestamp here so the
    # finalized node can render them in the order they actually executed,
    # instead of emitting all ops before all children (which floated e.g. the
    # decoder layer's post-attention ``allreduce`` to the top of the layer and
    # the attention's ``qknorm_rope`` op ahead of ``qkv_proj``). Synthetic
    # ``triton::`` ops are appended to a module's child list out of order, so we
    # sort by ``ts`` to recover true execution order.
    order_index: dict[tuple, int] = {}
    order_by_name: dict[str, int] = {}
    seen_child: dict[str, int] = {}
    seen_op: dict[tuple, int] = {}
    pos = 0
    for c in sorted(base.children, key=lambda n: n.ts):
        if c.kind == "op":
            if c.label in _PLUMBING_OPS:
                continue
            sig = _op_signature(c)
            occ = seen_op.get(sig, 0)
            seen_op[sig] = occ + 1
            key = ("op", (sig, occ))
            order_by_name.setdefault(c.label, pos)
        elif c.kind == "module":
            norm = _strip_instance_idx(c.label)
            idx = seen_child.get(norm, 0)
            seen_child[norm] = idx + 1
            key = ("child", (norm, idx))
        else:
            continue
        if key not in order_index:
            order_index[key] = pos
            pos += 1

    return {
        "module_type": _strip_instance_idx(base.label),
        "attr_name": base.attr_name,
        "op_groups": op_groups,
        "op_order": op_order,
        "child_groups": child_groups,
        "child_order": child_order,
        "order_index": order_index,
        "order_by_name": order_by_name,
        "layout_len": pos,
        "n_forward": n_forward,
    }


def _finalize_node(
    merged: dict,
    name: str,
    path: str,
    repeat_count: int,
    symbols_val: dict[int, str],
    dtype_bytes: int,
    token_symbol: str,
    token_val: int,
    device_type: str = "cuda",
) -> dict:
    """Turn a merged module description into the serialized display dict."""
    n_forward = merged["n_forward"]
    module_path = _split_path_types(path)
    display_name = merged.get("attr_name") or name

    # GPU-only naming/role fixes. On CUDA, torch.compile/CUDA-graph async timing
    # corrupts the trace's time-containment nesting, so path-based role inference
    # and parent-based module naming are unreliable. These corrections are gated
    # to CUDA so the XPU path (accurate eager nesting) keeps its original
    # behaviour and its distinct symbol/naming mapping.
    is_cuda = (device_type or "").lower() == "cuda"

    # When the module has a resolved semantic identity (e.g. "down_proj"), the
    # module's *projection* op inherits that role — overriding path-based
    # inference that can be wrong on GPU. Only the matmul-family op adopts the
    # role; communication ops (all_reduce/all_gather) keep their own role.
    module_role_override = display_name if (is_cuda and
                                            display_name in _KNOWN_MODULE_ROLES) \
        else None

    ops_out: list[dict] = []
    node_dev = 0.0
    node_cpu = 0.0
    node_mem = 0
    node_flops = 0
    order_index = merged.get("order_index", {})
    order_by_name = merged.get("order_by_name", {})
    extra_order = merged.get("layout_len", 0)
    for gkey in merged["op_order"]:
        g = merged["op_groups"][gkey]
        raw = g["raw"]
        # Per-forward, per-module-instance averages.
        dev = g["dev"] / n_forward
        cpu = g["host"] / n_forward
        count = max(1, round(g["count"] / n_forward))

        shapes = raw.shapes
        mem = estimate_memory(raw.label, shapes, dtype_bytes)
        flops = estimate_flops(raw.label, shapes)
        backend, category = classify_op(
            raw.label,
            device_type=device_type if dev > 0 else "",
            self_device_time_us=dev,
            device_time_us=dev,
        )
        # CUDA-only role corrections: keep collective-comm ops labelled as their
        # own op (not the enclosing projection), and let the projection matmul
        # inherit the module's semantic role.
        cuda_role = None
        if is_cuda:
            low_op = raw.label.lower()
            op_base = low_op.split("::")[-1]
            if "all_reduce" in low_op or "allreduce" in low_op:
                cuda_role = "all_reduce"
            elif "all_gather" in low_op or "allgather" in low_op:
                cuda_role = "all_gather"
            elif "reduce_scatter" in low_op or "reducescatter" in low_op:
                cuda_role = "reduce_scatter"
            elif module_role_override and op_base in (
                    "mm", "addmm", "linear", "matmul", "bmm"):
                cuda_role = module_role_override
        role = cuda_role \
            or _infer_role(module_path + [merged["module_type"]], raw.label) \
            or raw.label.split("::")[-1]
        sym_shapes = [
            _symbolize(s, symbols_val, token_symbol, token_val) for s in shapes
        ]
        out_shape = _symbolize(_output_shape(raw.label, shapes),
                               symbols_val, token_symbol, token_val)
        op_order_pos = order_index.get(("op", gkey))
        if op_order_pos is None:
            op_order_pos = order_by_name.get(raw.label)
        if op_order_pos is None:
            op_order_pos = extra_order
            extra_order += 1
        ops_out.append({
            "name": raw.label,
            "role": role,
            "backend": backend.value,
            "category": category,
            "input_shapes": sym_shapes,
            "output_shape": out_shape,
            "recorded_shapes": [list(s) for s in shapes],
            "input_dtypes": list(raw.dtypes),
            "input_args": [dict(s) for s in raw.arg_slots],
            "memory_bytes": mem,
            "flops": flops,
            "ai": round(flops / mem, 2) if mem > 0 else 0,
            "phase": "both",
            "device_time_us": round(dev, 2),
            "cpu_time_us": round(cpu, 2),
            "count": count,
            "order": op_order_pos,
        })
        node_dev += dev
        node_cpu += cpu
        node_mem += mem
        node_flops += flops

    raw_children: list[dict] = []
    for key in merged["child_order"]:
        insts = merged["child_groups"][key]
        norm, occ_idx = key
        child_merged = _merge_modules(insts, n_forward)
        child_name = child_merged.get("attr_name")
        if not child_name and is_cuda:
            # CUDA-only: shape-based o_proj/down_proj disambiguation, robust to
            # the corrupted time-containment nesting seen on GPU traces.
            child_name = _rowparallel_shape_role(norm, child_merged, symbols_val)
        if not child_name:
            child_name = _disambiguate_child_name(
                norm, occ_idx, merged, is_cuda=is_cuda)
        child = _finalize_node(
            child_merged,
            name=child_name,
            path=path + "/" + norm,
            repeat_count=1,
            symbols_val=symbols_val,
            dtype_bytes=dtype_bytes,
            token_symbol=token_symbol,
            token_val=token_val,
            device_type=device_type,
        )
        child_order_pos = order_index.get(("child", key))
        if child_order_pos is None:
            child_order_pos = extra_order
            extra_order += 1
        child["order"] = child_order_pos
        raw_children.append(child)

    # Collapse runs of adjacent structurally-identical siblings (e.g. the
    # repeated decoder layers) into a single node with repeat_count = run length.
    children_out = _collapse_repeats(raw_children)
    for child in children_out:
        rep = child["repeat_count"]
        node_dev += child["total_device_time_us"] * rep
        node_cpu += child["total_cpu_time_us"] * rep
        node_mem += child["total_memory"] * rep
        node_flops += child["total_flops"] * rep

    ai = (node_flops / node_mem) if node_mem > 0 else 0
    return {
        "name": display_name,
        "path": path,
        "module_type": merged["module_type"],
        "repeat_count": repeat_count,
        "total_memory": node_mem,
        "total_flops": node_flops,
        "total_ai": round(ai, 2),
        "total_device_time_us": round(node_dev, 2),
        "total_cpu_time_us": round(node_cpu, 2),
        "ops": ops_out,
        "children": children_out,
    }


def _struct_sig(node: dict) -> tuple:
    """Structural signature (ignores timing) for detecting repeated siblings."""
    return (
        node["module_type"],
        node.get("name"),
        node["repeat_count"],
        tuple((o["name"], o["role"],
               tuple(tuple(s) for s in o["input_shapes"]))
              for o in node["ops"]),
        tuple(_struct_sig(c) for c in node["children"]),
    )


def _average_nodes(nodes: list[dict], repeat_mult: int) -> dict:
    """Average timing across structurally-identical nodes; set repeat_count.

    ``memory``/``flops`` are structural (identical across the run) and kept as-is;
    device/cpu time is averaged so the node represents one instance.
    """
    k = len(nodes)
    base = dict(nodes[0])
    base["repeat_count"] = nodes[0]["repeat_count"] * repeat_mult
    base["total_device_time_us"] = round(
        sum(n["total_device_time_us"] for n in nodes) / k, 2)
    base["total_cpu_time_us"] = round(
        sum(n["total_cpu_time_us"] for n in nodes) / k, 2)

    base["ops"] = []
    for i, op in enumerate(nodes[0]["ops"]):
        merged_op = dict(op)
        merged_op["device_time_us"] = round(
            sum(n["ops"][i]["device_time_us"] for n in nodes) / k, 2)
        merged_op["cpu_time_us"] = round(
            sum(n["ops"][i]["cpu_time_us"] for n in nodes) / k, 2)
        base["ops"].append(merged_op)

    base["children"] = [
        _average_nodes([n["children"][i] for n in nodes], 1)
        for i in range(len(nodes[0]["children"]))
    ]
    return base


def _collapse_repeats(children: list[dict]) -> list[dict]:
    """Merge maximal runs of adjacent structurally-identical children."""
    if len(children) <= 1:
        return children
    out: list[dict] = []
    run: list[dict] = [children[0]]
    run_sig = _struct_sig(children[0])
    for child in children[1:]:
        sig = _struct_sig(child)
        if sig == run_sig:
            run.append(child)
        else:
            out.append(run[0] if len(run) == 1 else _average_nodes(run, len(run)))
            run = [child]
            run_sig = sig
    out.append(run[0] if len(run) == 1 else _average_nodes(run, len(run)))
    return out


def _build_phase_tree(roots: list[_Raw], n_steps: int,
                      symbols_val: dict[int, str], dtype_bytes: int,
                      token_symbol: str, token_val: int,
                      device_type: str = "cuda") -> dict | None:
    """Build one phase tree from all module roots assigned to that phase.

    Roots of the same class (the model, the logits processor, the sampler ...)
    are merged and averaged over ``n_steps``. When more than one root class is
    present they are wrapped under a synthetic per-step node.
    """
    if not roots:
        return None
    n_steps = max(1, n_steps)

    groups: dict[str, list[_Raw]] = {}
    order: list[str] = []
    for r in roots:
        norm = _strip_instance_idx(r.label)
        if norm not in groups:
            groups[norm] = []
            order.append(norm)
        groups[norm].append(r)

    finalized: list[dict] = []
    for norm in order:
        merged = _merge_modules(groups[norm], n_steps)
        finalized.append(_finalize_node(
            merged,
            name=_module_display_name(norm),
            path=norm,
            repeat_count=1,
            symbols_val=symbols_val,
            dtype_bytes=dtype_bytes,
            token_symbol=token_symbol,
            token_val=token_val,
            device_type=device_type,
        ))

    if len(finalized) == 1:
        return finalized[0]

    # Multiple top-level roots per step → synthetic wrapper.
    dev = sum(c["total_device_time_us"] for c in finalized)
    cpu = sum(c["total_cpu_time_us"] for c in finalized)
    mem = sum(c["total_memory"] for c in finalized)
    flops = sum(c["total_flops"] for c in finalized)
    return {
        "name": "step",
        "path": "step",
        "module_type": "InferenceStep",
        "repeat_count": 1,
        "total_memory": mem,
        "total_flops": flops,
        "total_ai": round(flops / mem, 2) if mem > 0 else 0,
        "total_device_time_us": round(dev, 2),
        "total_cpu_time_us": round(cpu, 2),
        "ops": [],
        "children": finalized,
    }


# ===================================================================
# Shape symbolization + naming helpers
# ===================================================================

def _symbolize(shape: list[int], symbols_val: dict[int, str],
               token_symbol: str, token_val: int) -> list:
    """Replace known dimension values with symbol names.

    The token dim (``S`` prefill / ``B`` decode) is recognised at the leading
    position (where it always wins over a coincidental config-value match) and
    also at any later position when the value isn't a known config dim — some
    tensors are token-major only on their second axis (the MSA indexer's
    ``[n_idx, total_q, ...]`` score / top-k tensors), and leaving those as a bare
    integer would hand them a meaningless observed-value symbol later.
    """
    out: list = []
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            out.append(dim)
        elif i == 0 and token_val and dim == token_val:
            out.append(token_symbol)
        elif dim in symbols_val:
            out.append(symbols_val[dim])
        elif token_val and dim == token_val:
            out.append(token_symbol)
        else:
            out.append(dim)
    return out


def _output_shape(op_name: str, shapes: list[list[int]]) -> list[int]:
    """Best-effort output shape for common ops (matmul → [M, N])."""
    base = op_name.split("::")[-1].lower()
    if base in ("mm", "linear", "matmul") and len(shapes) >= 2:
        if len(shapes[0]) >= 1 and len(shapes[1]) >= 1:
            return list(shapes[0][:-1]) + [shapes[1][-1]]
    if base == "addmm" and len(shapes) >= 3:
        return [shapes[1][0], shapes[2][-1]]
    if base == "bmm" and len(shapes) >= 2 and len(shapes[0]) >= 3:
        return [shapes[0][0], shapes[0][1], shapes[1][-1]]
    return list(shapes[0]) if shapes else []


def _module_display_name(cls: str) -> str:
    """Human-friendly short name for a module class."""
    hints = {
        "attention": "self_attn", "attn": "self_attn",
        "mlp": "mlp", "decoderlayer": "layer", "layer": "layer",
        "embedding": "embed", "rmsnorm": "norm", "layernorm": "norm",
        "qkvparallellinear": "qkv_proj", "rowparallellinear": "o_proj",
        "mergedcolumnparallellinear": "gate_up_proj",
        "columnparallellinear": "proj",
    }
    low = cls.lower()
    for key, name in hints.items():
        if key in low:
            return name
    return cls


def _split_path_types(path: str) -> list[str]:
    """Return the list of module class names along a '/'-joined path."""
    return [p for p in path.split("/") if p]


def _rowparallel_shape_role(cls: str, child_merged: dict,
                            symbols_val: dict[int, str]) -> str | None:
    """Disambiguate a RowParallelLinear as o_proj vs down_proj by shape.

    Both the attention output projection (``o_proj``) and the MLP/MoE down
    projection (``down_proj``) are ``RowParallelLinear``, so class name alone is
    ambiguous. Parent-based heuristics are unreliable on GPU where async timing
    corrupts the trace's time-containment nesting (an attention ``o_proj`` can
    end up nested under an MoE block, or vice-versa). The projection's matmul
    input feature dimension is unambiguous instead:

    * ``o_proj``  input feature ≈ ``n_h·d`` (attention hidden, ~``H``)
    * ``down_proj`` input feature = ``intermediate`` (``I`` / ``I_moe``)

    We read the matmul's input feature dim, map it to a known symbol, and pick
    the role accordingly. Returns ``None`` when the class isn't RowParallelLinear
    or the shape can't be resolved to a known symbol (caller then falls back to
    the parent heuristic).
    """
    if "rowparallel" not in cls.lower():
        return None
    for sig in child_merged["op_order"]:
        raw = child_merged["op_groups"][sig]["raw"]
        base = raw.label.split("::")[-1].lower()
        if base not in ("mm", "addmm", "linear", "matmul"):
            continue
        # Activation input: [M, K]; for addmm it's the 2nd arg.
        act = raw.shapes[1] if base == "addmm" and len(raw.shapes) > 1 \
            else (raw.shapes[0] if raw.shapes else None)
        if not act:
            continue
        k = act[-1]
        sym = symbols_val.get(k, "")
        # Strip a trailing "/TP" so per-rank shards match the base symbol.
        base_sym = sym.split("/")[0] if sym else ""
        if base_sym in ("I", "I_moe", "2·I"):
            return "down_proj"
        if base_sym in ("H", "n_h·d", "QKV"):
            return "o_proj"
    return None


def _disambiguate_child_name(cls: str, occ_idx: int, parent_merged: dict,
                             is_cuda: bool = False) -> str:
    """Generate a display name for a child module, disambiguating by position.

    When multiple children share the same class (e.g. two RMSNorm inside
    Attention → q_norm and k_norm), use positional heuristics to distinguish
    them instead of showing the same generic name for both.

    ``is_cuda`` gates the GPU-only RowParallelLinear parent heuristics (which
    compensate for corrupted trace nesting on CUDA); the XPU path keeps the
    original generic naming.
    """
    parent_type = parent_merged.get("module_type", "").lower()
    low = cls.lower()

    # Norm modules inside Attention: first = q_norm, second = k_norm
    if ("norm" in low) and ("attention" in parent_type or "attn" in parent_type):
        # Count how many same-class norm siblings exist
        norm_count = sum(
            1 for key in parent_merged["child_order"]
            if "norm" in key[0].lower()
        )
        if norm_count >= 2:
            if occ_idx == 0:
                return "q_norm"
            elif occ_idx == 1:
                return "k_norm"

    # Norm modules inside DecoderLayer: first = input_layernorm,
    # second = post_attention_layernorm
    if ("norm" in low) and ("layer" in parent_type or "decoder" in parent_type):
        norm_count = sum(
            1 for key in parent_merged["child_order"]
            if "norm" in key[0].lower()
        )
        if norm_count >= 2:
            if occ_idx == 0:
                return "input_layernorm"
            elif occ_idx == 1:
                return "post_attention_layernorm"
            elif occ_idx == 2:
                return "pre_feedforward_layernorm"

    # Linear projections: RowParallelLinear is the attention output (o_proj) but
    # also the MLP/MoE down projection (down_proj) — both share the class, so the
    # generic ``_module_display_name`` (which maps RowParallelLinear → o_proj)
    # mislabels the MLP one as o_proj whenever the reference-name overlay fails
    # to tag it. Disambiguate by the parent module: a RowParallelLinear inside an
    # MLP/expert/feedforward module is always the down projection; inside
    # attention it's the output projection. This is **device-agnostic** — the
    # reference-name overlay can miss modules for reasons unrelated to CUDA async
    # timing. On XPU the shared_experts MLP is hoisted out of the fused
    # ``moe_forward_shared`` op (``_hoist_modules_under_ops``), so it sits under
    # ``FusedMoE`` while the reference tree lists it under ``MoE.shared_experts``;
    # alignment can't match it, leaving its ``down_proj`` unnamed. Without this
    # the shared expert's down projection read as ``o_proj`` even though the dense
    # MLP's (overlay-named) down_proj was correct.
    is_mlp = ("mlp" in parent_type or "moe" in parent_type
              or "expert" in parent_type or "feedforward" in parent_type)
    is_attn = "attention" in parent_type or "attn" in parent_type
    if "rowparallel" in low:
        if is_mlp:
            # Count RowParallelLinear siblings in this parent
            row_parallel_count = sum(
                1 for key in parent_merged["child_order"]
                if "rowparallel" in key[0].lower()
            )
            if row_parallel_count <= 1:
                return "down_proj"
            # Multiple RowParallel inside MLP: only the last is down_proj, the
            # earlier ones are misplaced from Attention. This happens on GPU
            # where async timing can time-contain Attention's RowParallelLinear
            # inside MLP; XPU nesting is reliable and normally hits the count<=1
            # branch above, so gate the compensation to CUDA.
            if is_cuda:
                last_rp_idx = max(
                    key[1] for key in parent_merged["child_order"]
                    if "rowparallel" in key[0].lower()
                )
                if occ_idx == last_rp_idx:
                    return "down_proj"
                return "o_proj"
            return "down_proj"
        if is_attn:
            return "o_proj"
    if "mergedcolumn" in low or "columnparallel" in low:
        if is_mlp:
            return "gate_up_proj"

    return _module_display_name(cls)


# ===================================================================
# Symbol table
# ===================================================================

def _build_symbol_tables(summary: dict, tp_size: int
                         ) -> tuple[dict[int, str], dict[str, int]]:
    """Return (value→symbol for shape rewriting, symbol→value for the UI)."""
    val_to_sym: dict[int, str] = {}
    sym_to_val: dict[str, int] = {}

    def add(sym: str, val: int | None):
        if not val or val <= 0:
            return
        sym_to_val.setdefault(sym, val)
        val_to_sym.setdefault(val, sym)
        if tp_size > 1 and val % tp_size == 0:
            val_to_sym.setdefault(val // tp_size, f"{sym}/TP")

    H = summary.get("hidden_size")
    n_h = summary.get("num_heads")
    n_kv = summary.get("num_kv_heads", n_h)
    d = summary.get("head_dim")
    if H and n_h and not d:
        d = H // n_h
    inter = summary.get("intermediate_size")
    vocab = summary.get("vocab_size")

    add("H", H)
    add("n_h", n_h)
    if n_kv and n_kv != n_h:
        add("n_kv", n_kv)
    add("d", d)
    add("I", inter)
    add("V", vocab)
    if n_h and d:
        add("n_h·d", n_h * d)
        add("QKV", (n_h + 2 * (n_kv or n_h)) * d)
    if inter:
        add("2·I", 2 * inter)
    if summary.get("num_experts"):
        add("E", summary["num_experts"])
    if summary.get("moe_intermediate_size"):
        add("I_moe", summary["moe_intermediate_size"])
        # The MoE gate_up projection is the two halves fused, exactly like the
        # dense ``2·I``; without this the width falls through to an
        # observed-value symbol and reads as a meaningless ``N``.
        add("2·I_moe", 2 * summary["moe_intermediate_size"])
    # Rope cos/sin cache length (``[max_position, rotary_dim]``). Not divided by
    # TP (the position table is replicated per rank).
    max_pos = summary.get("max_position_embeddings")
    if max_pos:
        sym_to_val.setdefault("P", max_pos)
        val_to_sym.setdefault(max_pos, "P")
    # DeepSeek/MiniMax-M3 sparse attention fuses the lightning-indexer's query
    # and key projections into the qkv_proj, so its output width is the plain
    # QKV plus ``2·(n_index_heads·index_dim)``. Register the augmented width as a
    # distinct symbol so the sparse qkv_proj / normed-qkv shapes symbolize
    # (``QKV_idx/TP``) instead of leaking a per-rank integer.
    if summary.get("sparse_attention") and n_h and d:
        n_idx = summary.get("sparse_num_index_heads")
        idx_d = summary.get("sparse_index_dim")
        if n_idx and idx_d:
            qkv = (n_h + 2 * (n_kv or n_h)) * d
            add("QKV_idx", qkv + 2 * n_idx * idx_d)
    sym_to_val["TP"] = tp_size
    return val_to_sym, sym_to_val


# ===================================================================
# Public entry point
# ===================================================================

def _recompute_totals(node: dict) -> None:
    """Post-order recompute of a node's aggregate totals from its ops and
    children (each child folded ``total × repeat_count``). Used after the layer
    repeat counts are rescaled by extrapolation."""
    dev = sum(o["device_time_us"] for o in node["ops"])
    cpu = sum(o["cpu_time_us"] for o in node["ops"])
    mem = sum(o["memory_bytes"] for o in node["ops"])
    flops = sum(o["flops"] for o in node["ops"])
    for c in node["children"]:
        _recompute_totals(c)
        rep = c["repeat_count"]
        dev += c["total_device_time_us"] * rep
        cpu += c["total_cpu_time_us"] * rep
        mem += c["total_memory"] * rep
        flops += c["total_flops"] * rep
    node["total_device_time_us"] = round(dev, 2)
    node["total_cpu_time_us"] = round(cpu, 2)
    node["total_memory"] = mem
    node["total_flops"] = flops
    node["total_ai"] = round(flops / mem, 2) if mem > 0 else 0


def _extrapolate_decoder_layers(tree: dict, num_layers: int | None) -> None:
    """Rescale decoder-layer repeat counts to the model's true layer count.

    When profiling ran with a reduced ``num_hidden_layers`` (to fit memory), the
    trace only contains a handful of decoder layers. The dense prefix is captured
    in full, but the repeated (MoE) body is under-represented. We add the missing
    layers to the *last* decoder-layer group — which, for dense-prefix MoE models
    (DeepSeek, MiniMax-M3, Qwen-MoE), is the MoE layer that repeats for the rest
    of the network. Totals are recomputed afterwards so parents stay consistent.
    """
    if not num_layers:
        return

    def find_layer_siblings(node: dict) -> list[dict] | None:
        layers = [c for c in node["children"]
                  if "DecoderLayer" in c["module_type"]]
        if layers:
            return layers
        for c in node["children"]:
            found = find_layer_siblings(c)
            if found is not None:
                return found
        return None

    layers = find_layer_siblings(tree)
    if not layers:
        return
    profiled = sum(c["repeat_count"] for c in layers)
    if num_layers > profiled:
        layers[-1]["repeat_count"] += num_layers - profiled
        _recompute_totals(tree)


_ATTENTION_OP_NAMES = frozenset({
    "vllm::unified_attention_with_output",
    "vllm::unified_attention",
})


def _is_attention_op(op: dict) -> bool:
    name = op.get("name", "")
    if name in _ATTENTION_OP_NAMES:
        return True
    low = name.lower()
    return ("attention" in low or "flash_attn" in low
            or op.get("role") == "attention")


def _annotate_attention_kv(node: dict, n_kv: int | None,
                           token_sym: str = "S",
                           kv_sym: str = "S+C",
                           kv_rows: int = 0, n_seqs: int = 1,
                           dtype_bytes: int = 2) -> None:
    """Rewrite attention key/value row lengths to the KV the call really reads.

    Paged/prefix-cached attention only records the *new* tokens as the op's
    key/value inputs (``[S, n_kv, d]``); the cached context never appears as a
    tensor dim. To make the attended context visible, the key/value input rows
    (and the KV-shaped inputs generally) have their leading token symbol
    replaced with the KV length the call actually reads: ``S+C`` for the single
    prefill sequence, and ``B·C`` for a decode step, where each of the ``B``
    sequences reads its own ``C``-token context (so the *total* KV traffic is
    ``B·C``, which is what the memory estimate needs; ``estimate_flops`` divides
    it back by the sequence count so each query is only charged for its own
    context). Leaving decode at a bare ``B`` left the heaviest op in the model
    looking like it read a few kilobytes. Query/output rows (``[S, n_h, d]``) are
    left untouched — there are still ``S`` query positions producing ``S``
    outputs. Key/value rows are identified by their second dim being the KV-head
    count (GQA); when heads are indistinguishable (MHA) the canonical vLLM
    ``[query, key, value, output]`` argument order (inputs 1 and 2) is used.
    """
    for op in node.get("ops", []):
        if not _is_attention_op(op):
            continue
        shapes = op.get("input_shapes") or []
        rewritten: list[int] = []
        kv_by_heads = False
        if n_kv:
            for i, row in enumerate(shapes):
                if (isinstance(row, list) and len(row) >= 2
                        and row[0] == token_sym and _dim_is(row[1], n_kv)):
                    row[0] = kv_sym
                    rewritten.append(i)
                    kv_by_heads = True
        if not kv_by_heads:
            # Fall back to vLLM arg order: inputs[1] = key, inputs[2] = value.
            for i in (1, 2):
                if (i < len(shapes) and isinstance(shapes[i], list)
                        and shapes[i] and shapes[i][0] == token_sym):
                    shapes[i][0] = kv_sym
                    rewritten.append(i)
        _recost_attention_op(op, rewritten, kv_rows, n_seqs, dtype_bytes)
    for child in node.get("children", []):
        _annotate_attention_kv(child, n_kv, token_sym, kv_sym, kv_rows,
                               n_seqs, dtype_bytes)


def _recost_attention_op(op: dict, rewritten: list[int], kv_rows: int,
                         n_seqs: int, dtype_bytes: int) -> None:
    """Recompute an attention op's analytic cost once its KV rows are known.

    ``_finalize_node`` costs every op while the tree is being built, i.e. from
    the *recorded* KV rows - the new tokens only. For attention that is the
    whole point of the annotation above: the call really reads ``context+query``
    (prefill) or ``batch x context`` (decode) KV rows. Without this the graph
    view charged prefill attention ~65x too little work at a 2048-token context.
    """
    recorded = op.get("recorded_shapes") or []
    if not rewritten or not kv_rows or not recorded:
        return
    numeric = [list(s) for s in recorded]
    for i in rewritten:
        if i < len(numeric) and numeric[i]:
            numeric[i][0] = int(kv_rows)
    from .shape_derive import _profile_op_memory

    dtypes = op.get("input_dtypes") or []
    mem = (_profile_op_memory(op["name"], numeric, dtypes, dtype_bytes)
           if dtypes else estimate_memory(op["name"], numeric, dtype_bytes))
    flops = estimate_flops(op["name"], numeric, n_seqs=max(n_seqs, 1))
    op["memory_bytes"] = mem
    op["flops"] = flops
    op["ai"] = round(flops / mem, 2) if mem > 0 else 0


def _dim_is(sym: Any, value: int) -> bool:
    """True if a (possibly symbolic) shape entry equals the integer ``value``."""
    if isinstance(sym, int):
        return sym == value
    # Symbolic head-count dims render as "n_kv" / "n_kv/TP"; treat any n_kv label
    # as the KV-head dimension.
    return isinstance(sym, str) and sym.split("/")[0] in ("n_kv", "n_h·d_kv")


# Concrete dims at or below this value are structural constants (k/v pair,
# real/imag split, singleton broadcasts), not dimensions to symbolize.
_RUNTIME_TRIVIAL_MAX = 2

# Op-name substrings → the symbol base for their run-specific allocation dims.
# Ordered: the first family whose substring appears in the op name wins.
_RUNTIME_DIM_FAMILIES: tuple[tuple[tuple[str, ...], str], ...] = (
    # MiniMax-M3 sparse-attention indexer top-k block count (config
    # ``sparse_topk_blocks``); probed first so it isn't absorbed by the MoE
    # family via a ``topk`` substring.
    (("index_topk", "topk_index"), "K_topk"),
    (("kv_insert", "reshape_and_cache", "kv_cache", "paged_attention",
      "block_table", "kv_update"), "N_kv"),
    (("moe", "expert", "silu_and_mul", "fused_moe", "topk_bias"), "M_moe"),
)


def _runtime_family_base(op_name: str, ndim: int) -> str:
    """Pick the observed-symbol base for a concrete runtime dim.

    ``N_kv`` for paged KV-cache allocation dims, ``M_moe`` for MoE expert-GEMM
    routed-token rows (multi-dim activations), ``N_moe`` for the MoE block-align
    metadata scratch buffers (1-D), and a generic ``N`` for anything else so the
    "no concrete structural values" invariant always holds.
    """
    low = op_name.lower()
    for subs, base in _RUNTIME_DIM_FAMILIES:
        if any(s in low for s in subs):
            if base == "M_moe" and ndim < 2:
                # 1-D moe_align_block_size scratch (sorted-token / expert-block
                # buffers) rather than an expert-GEMM activation row count.
                return "N_moe"
            return base
    return "N"


def _iter_ops(node: dict):
    if not node:
        return
    for op in node.get("ops", []):
        yield op
    for c in node.get("children", []):
        yield from _iter_ops(c)


def _symbolize_msa_dims(trees: list[tuple[dict | None, str, int]],
                        sym_to_val: dict[str, int],
                        summary: dict, tp_size: int) -> None:
    """Give the MiniMax-M3 MSA indexer dims their own symbols.

    The shapes reconstructed by :func:`_infer_attention_kernel_shapes` contain
    three dims that plain value→symbol mapping gets wrong, because M3's config
    makes them collide with unrelated model dims:

    * the **index-head count** (``sparse_num_index_heads``) equals
      ``num_kv_heads``, so it renders ``n_kv``/``n_kv/TP``;
    * the **top-k block count** (``sparse_topk_blocks``, 16) can equal a sharded
      head count (``n_h/TP`` = 64/4 at TP=4), so it renders ``n_h/TP`` — which
      would then wrongly scale with TP in the Shape Matrix sweep;
    * the top-k tensors are **token-major on their second axis**
      (``[n_idx, total_q, topk]``), where the token count can collide with a
      config dim (``S`` = 2048 = ``n_h·d/TP``) and lose its ``S``/``B`` symbol.

    All three are rewritten in place to the right symbol (``n_idx``/``n_idx/TP``,
    ``K_topk``, and the phase's token symbol) using each op's ``recorded_shapes``
    to confirm the concrete value. Runs after the phase trees are built (so the
    numeric shapes are already symbolized) and before
    :func:`_symbolize_runtime_dims`.
    """
    n_idx = summary.get("sparse_num_index_heads")
    topk_blocks = summary.get("sparse_topk_blocks")
    if not summary.get("sparse_attention") or not n_idx:
        return
    tp = max(1, int(tp_size or 1))
    if int(n_idx) % tp:
        return  # per-rank index-head count isn't a clean n_idx/TP shard
    n_idx_sym = "n_idx/TP" if tp > 1 else "n_idx"
    for tree, token_symbol, token_val in trees:
        for op in _iter_ops(tree):
            layout = _msa_kernel_layout(op.get("name", ""))
            if layout not in ("index", "topk"):
                continue
            head_pos = 1 if layout == "index" else 0
            n_idx_rank = max(1, int(n_idx) // tp)
            recorded = op.get("recorded_shapes") or []
            for i, shp in enumerate(op.get("input_shapes") or []):
                if not isinstance(shp, list) or len(shp) != 3:
                    continue
                rec = recorded[i] if i < len(recorded) else []
                if len(rec) != 3 or rec[head_pos] != n_idx_rank:
                    continue  # not the reconstructed indexer tensor
                shp[head_pos] = n_idx_sym
                sym_to_val.setdefault("n_idx", int(n_idx))
                if layout == "topk":
                    if token_val and rec[1] == token_val:
                        shp[1] = token_symbol
                    if topk_blocks and rec[2] == int(topk_blocks):
                        shp[2] = "K_topk"
                        sym_to_val.setdefault("K_topk", int(topk_blocks))


def _symbolize_moe_routed_rows(trees: list[tuple[dict | None, str, int]],
                               sym_to_val: dict[str, int],
                               summary: dict) -> None:
    """Symbolize the MoE routed-token row count as ``topk·S`` / ``topk·B``.

    An MoE block expands every token into ``num_experts_per_tok`` routed rows,
    so the permuted hidden states, the grouped-GEMM ``M`` and the gather
    destination are all ``tokens × topk`` rows. That value is not a config
    constant, so it used to fall through to :func:`_symbolize_runtime_dims`,
    which freezes it at the value it happened to have while profiling.

    Freezing it is not a cosmetic problem: the Shape Matrix and the replay
    benchmark sweep ``S``/``B``, so the token operand scaled while the routed
    operand did not, and the two stopped describing the same call. The kernels
    then rejected their own recorded shapes — ``remap_hidden_states`` with
    *"remapped_hidden_states must be [num_rows * TopK, hidden_size]"* and the
    MoE grouped GEMM (the dominant kernel of an MoE model) with
    *"ptr_A.size(1) must match ptr_B.size(1)"*.

    Emitting the **expression** keeps the relationship: ``_resolve_dim``
    evaluates ``topk·S`` against whatever ``S`` the configuration sweeps to.
    """
    topk = summary.get("num_experts_per_tok")
    if not summary.get("num_experts") or not topk or topk < 2:
        return
    sym_to_val.setdefault("topk", int(topk))
    for tree, token_sym, token_val in trees:
        if not tree or not token_val:
            continue
        target = int(token_val) * int(topk)
        expr = f"topk·{token_sym}"
        for op in _iter_ops(tree):
            for shp in (op.get("input_shapes") or []):
                if isinstance(shp, list):
                    _rewrite_routed(shp, target, expr, token_sym, int(topk))
            out = op.get("output_shape")
            if isinstance(out, list):
                _rewrite_routed(out, target, expr, token_sym, int(topk))


def _rewrite_routed(shape: list, target: int, expr: str, token_sym: str,
                    topk: int) -> None:
    """``tokens×topk`` → ``topk·S``; the per-token expert axis → ``topk``.

    The router rule is applied **first** and its axis is then excluded from the
    routed-rows rule. Otherwise a profile whose token count is 1 (a decode pass
    at batch 1) makes ``tokens × topk == topk``, so the router's ``[tokens,
    topk]`` operand matches the routed-rows rule and becomes ``[B, topk·B]`` -
    an expert fan-out that *scales with the swept batch*, which is exactly the
    bug this pass exists to prevent.

    The router rule is deliberately narrow - a bare ``8`` is far too common to
    map globally - and fires only on the ``[tokens, topk]`` router outputs
    (``topk_ids`` / ``topk_weights``), the only place the expert fan-out appears
    as its own axis.
    """
    router_axis = -1
    if (len(shape) == 2 and shape[0] == token_sym
            and isinstance(shape[1], int) and shape[1] == topk):
        shape[1] = "topk"
        router_axis = 1
    for j, dim in enumerate(shape):
        if j == router_axis or not isinstance(dim, int) or dim != target:
            continue
        if target == topk and not (j == 0 and len(shape) > 1):
            # tokens == 1 makes the two rules numerically indistinguishable;
            # routed rows are always a *row count*, so only the leading axis of
            # a multi-dim operand is safe to claim.
            continue
        shape[j] = expr


def _symbolize_runtime_dims(trees: list[dict | None],
                            sym_to_val: dict[str, int]) -> None:
    """Replace remaining concrete integer dims with observed-value symbols.

    Structural/config dims are already symbolized by :func:`_build_symbol_tables`
    and the token/context passes; what remains are run-specific allocation sizes
    (paged KV-cache slot counts, CUDA Triton-MoE routed-token / block-align
    buffers). Each distinct ``(base, value)`` gets a stable symbol (``N_kv``,
    ``M_moe``, ``N_moe``, or generic ``N`` — suffixed ``2``/``3``/… when a base
    carries several distinct values) recorded in ``sym_to_val`` so nothing
    structural is left as a bare integer while the concrete number stays in the
    legend. Trivial dims (``≤2``) are left untouched.
    """
    # Pass 1: collect distinct concrete values per family base (deterministic).
    per_base: dict[str, set[int]] = {}
    for tree in trees:
        for op in _iter_ops(tree):
            name = op.get("name", "")
            for shp in (op.get("input_shapes") or []):
                if not isinstance(shp, list):
                    continue
                ndim = len(shp)
                for dim in shp:
                    if isinstance(dim, int) and dim > _RUNTIME_TRIVIAL_MAX:
                        base = _runtime_family_base(name, ndim)
                        per_base.setdefault(base, set()).add(dim)

    if not per_base:
        return

    # Assign a stable symbol to each distinct value: largest value keeps the bare
    # base, further distinct values get numeric suffixes (sorted descending).
    value_to_sym: dict[tuple[str, int], str] = {}
    for base, values in per_base.items():
        for i, val in enumerate(sorted(values, reverse=True)):
            sym = base if i == 0 else f"{base}{i + 1}"
            value_to_sym[(base, val)] = sym
            sym_to_val.setdefault(sym, val)

    # Pass 2: rewrite the shapes in place.
    for tree in trees:
        for op in _iter_ops(tree):
            name = op.get("name", "")
            for shp in (op.get("input_shapes") or []):
                if not isinstance(shp, list):
                    continue
                ndim = len(shp)
                for j, dim in enumerate(shp):
                    if isinstance(dim, int) and dim > _RUNTIME_TRIVIAL_MAX:
                        base = _runtime_family_base(name, ndim)
                        shp[j] = value_to_sym[(base, dim)]
            out = op.get("output_shape")
            if isinstance(out, list):
                ndim = len(out)
                for j, dim in enumerate(out):
                    if isinstance(dim, int) and dim > _RUNTIME_TRIVIAL_MAX:
                        base = _runtime_family_base(name, ndim)
                        key = (base, dim)
                        if key in value_to_sym:
                            out[j] = value_to_sym[key]


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


def build_graph_from_trace(
    trace_path: str,
    summary: dict | None = None,
    tp_size: int = 1,
    batch_size: int = 1,
    quantization: str | None = None,
    ref_module_tree: dict | None = None,
    query_len: int | None = None,
    context_len: int | None = None,
) -> dict:
    """Reconstruct a model graph purely from a torch profiler trace.

    Args:
        trace_path: path to a ``.json`` / ``.json.gz`` chrome trace captured with
            ``with_stack=True`` and ``record_shapes=True``.
        summary: optional ``summarize_config()`` output, used only to symbolize
            dimensions and populate the symbol legend. Reconstruction works
            without it (dims stay numeric).
        tp_size: tensor-parallel size the trace was captured at (per-rank shapes).
        batch_size: request batch size, used for prefill/decode disambiguation.
        quantization: quant method, surfaced in ``config`` for the UI's dtype hints.
        ref_module_tree: optional reference module-name tree (from
            ``module_naming.build_ref_tree``) used to overlay real attribute
            names (``q_norm``/``k_norm``, ``input_layernorm``, ...) onto the
            trace-reconstructed module nodes. When omitted, nodes keep their
            class-name-derived heuristic labels.
        query_len: number of new prefill tokens (``S``); currently informational.
        context_len: prefix-cached context length (already floored to a KV-block
            boundary). Added to the symbol legend as ``C`` and, combined with the
            prefill token count, as ``S+C`` so attention KV dims symbolize.

    Returns:
        Dict with ``prefill`` / ``decode`` trees (either may be ``None``),
        ``symbols``, ``config`` and timing metadata (a serialized module tree
        the web UI and Shape Matrix export consume) plus per-op ``device_time_us``.
    """
    summary = summary or {}
    trace = _load_trace(trace_path)
    events = trace.get("traceEvents", [])
    if not events:
        return {"prefill": None, "decode": None, "symbols": {},
                "config": {}, "has_timing": False,
                "error": "empty trace"}

    roots = _build_raw_forest(events)
    if not roots:
        return {"prefill": None, "decode": None, "symbols": {},
                "config": {}, "has_timing": False,
                "error": "no module/op events (trace missing with_stack?)"}

    # Overlay real module attribute names (q_norm/k_norm, input_layernorm, ...)
    # from the reference module tree onto the class-name-based raw module nodes.
    # Done before phase building so recovered names feed the structural signature
    # and keep distinctly-named siblings from collapsing together.
    #
    # Skipped when the trace already carries capture-time module-name spans
    # (breakdown.module_hooks): those give exact names on the raw forest with no
    # alignment, so the reference-tree overlay is redundant. The overlay remains
    # the fallback for legacy / upload traces without spans.
    captured_names = _forest_has_named_modules(roots)
    if ref_module_tree and not captured_names:
        try:
            _apply_ref_names(roots, ref_module_tree)
        except Exception:
            pass  # naming enrichment is best-effort

    # Recover shape/dtype for residual-stream ops the trace leaves shape-less:
    # TP collectives (dtype-less ``TensorList``) and Python-launched norm kernels
    # (no ``cpu_op``). Borrow ``[tokens, H]`` + dtype from the nearest neighbour.
    _infer_hidden_activation_ops(roots, summary.get("hidden_size"))
    # Reconstruct shape/dtype for shape-less ``flash_xpu`` MSA attention kernels
    # (also ``cpu_op``-less) from their fixed wrapper layout + config dims.
    _infer_attention_kernel_shapes(roots, summary, tp_size)

    prefill_passes, decode_passes, n_pre, n_dec = _classify_steps(
        roots, batch_size)

    # Infer accelerator type from the trace events
    device_type = _infer_device_from_trace(events)

    dtype = summary.get("dtype", "bfloat16")
    dtype_bytes = dtype_size(dtype)
    val_to_sym, sym_to_val = _build_symbol_tables(summary, tp_size)

    prefill_tokens = max((_pass_token_dim(p) for p in prefill_passes), default=0)
    decode_tokens = max((_pass_token_dim(p) for p in decode_passes), default=0)

    # Register the prefix-cached context length as ``C`` (and the full attended
    # KV length ``context+query`` as ``S+C``) in the symbol legend. ``context_len``
    # is already floored to a KV-block boundary by the caller.
    #
    # The value→symbol direction uses ``setdefault``, so a **config structural
    # dim always wins over the context length**. Paged attention never records
    # the context as a tensor dim (the cached KV lives in the block cache), so
    # nothing in the trace legitimately *is* ``C``; ``_annotate_attention_kv``
    # writes the ``S+C`` KV rows explicitly instead. Letting ``C`` overwrite a
    # config value is therefore never right and is actively destructive when the
    # two collide: Qwen3-30B-A3B has ``hidden_size == 2048`` and the default
    # profiling context is also 2048, so every ``H`` dim symbolized to ``C`` and
    # then swept with the *context* in the Shape Matrix / benchmark — hidden
    # dims became 0 at ctx=0 (``rms_norm`` divided by zero and took the worker
    # down with SIGFPE; the MoE grouped GEMM rejected its operands).
    ctx = int(context_len) if context_len else 0
    if ctx > 0:
        val_to_sym.setdefault(ctx, "C")
        sym_to_val["C"] = ctx
        if prefill_tokens:
            val_to_sym.setdefault(ctx + prefill_tokens, "S+C")
            sym_to_val["S+C"] = ctx + prefill_tokens

    prefill_tree = _build_phase_tree(
        prefill_passes, n_pre, val_to_sym, dtype_bytes, "S", prefill_tokens,
        device_type=device_type)
    decode_tree = _build_phase_tree(
        decode_passes, n_dec, val_to_sym, dtype_bytes, "B", decode_tokens,
        device_type=device_type)

    # Reduced-layer profiling (app.py caps num_hidden_layers to save memory)
    # captures only a few decoder layers. Extrapolate the repeat counts back to
    # the model's true layer count so the tree reads e.g. ``x57`` MoE layers.
    num_layers = summary.get("num_layers")
    for tree in (prefill_tree, decode_tree):
        if tree:
            _extrapolate_decoder_layers(tree, num_layers)

    # Surface the prefix-cached context in attention. Paged attention records
    # only the *new* tokens in the op's key/value inputs ([S, n_kv, d]); the
    # context length lives in the block cache / seqlen metadata, never as a
    # tensor dim, so it can't be symbolized from the trace. When a context was
    # served from the prefix cache, rewrite the attention key/value rows to the
    # full attended KV length ``S+C`` so the graph shows the query attending
    # ``context+query`` keys (the query/output rows stay ``S``).
    if ctx > 0 and prefill_tokens and prefill_tree:
        # one prefill sequence attending context+query keys
        _annotate_attention_kv(prefill_tree, n_kv=summary.get("num_kv_heads"),
                               kv_rows=ctx + prefill_tokens, n_seqs=1,
                               dtype_bytes=dtype_bytes)
        _recompute_totals(prefill_tree)
    if ctx > 0 and decode_tokens and decode_tree:
        # B sequences each attending their own context
        _annotate_attention_kv(decode_tree, n_kv=summary.get("num_kv_heads"),
                               token_sym="B", kv_sym="B·C",
                               kv_rows=ctx * decode_tokens,
                               n_seqs=decode_tokens, dtype_bytes=dtype_bytes)
        _recompute_totals(decode_tree)

    if prefill_tokens:
        sym_to_val["S"] = prefill_tokens
    if decode_tokens:
        sym_to_val["B"] = decode_tokens

    # Symbolize any remaining concrete integer dims that aren't derivable from
    # the model config or S/B/C — chiefly run-specific allocation sizes: the
    # paged KV-cache slot count (``N_kv``) and the CUDA Triton-MoE routed-token /
    # block-align scratch buffers (``M_moe`` / ``N_moe``). Each distinct value is
    # assigned an observed-value symbol recorded in the legend so the shape reads
    # a symbol while the concrete number stays available for tooltips/exports.
    _symbolize_msa_dims([(prefill_tree, "S", prefill_tokens),
                         (decode_tree, "B", decode_tokens)],
                        sym_to_val, summary, tp_size)
    _symbolize_moe_routed_rows([(prefill_tree, "S", prefill_tokens),
                                (decode_tree, "B", decode_tokens)],
                               sym_to_val, summary)
    _symbolize_runtime_dims([prefill_tree, decode_tree], sym_to_val)

    total_ops = 0
    for tree in (prefill_tree, decode_tree):
        if tree:
            total_ops += _count_ops(tree)

    return {
        "architecture": summary.get("architecture", ""),
        "family": summary.get("family", ""),
        "prefill": prefill_tree,
        "decode": decode_tree,
        "symbols": sym_to_val,
        "config": {
            "tp_size": tp_size,
            "quantization": quantization or summary.get("quant_method"),
            "dtype_bytes": dtype_bytes,
            "weight_dtype_bytes": dtype_bytes,
            "num_layers": summary.get("num_layers"),
        },
        "has_timing": True,
        "has_module_names": captured_names or bool(ref_module_tree),
        "timing_matched": total_ops,
        "timing_total_ops": total_ops,
        "timing_method": "trace_reconstruction",
        "source": "profile",
    }


def _count_ops(node: dict) -> int:
    n = len(node.get("ops", []))
    for c in node.get("children", []):
        n += _count_ops(c)
    return n
