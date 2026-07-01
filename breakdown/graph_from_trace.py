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
* ``kernel`` / ``gpu_memcpy`` events whose device time is attributed back to the
  launching ``cpu_op`` via ``kernel.correlation → xpu_runtime.correlation →
  xpu_runtime.External id → cpu_op.External id``.

The output dict matches the serialized shape produced by
``model_graph.build_model_graph`` (``prefill`` / ``decode`` trees, ``symbols``,
``config``) so the existing web UI renders it unchanged.
"""

from __future__ import annotations

import ast
import gzip
import json
from typing import Any

from .analyzer import DTYPE_BYTES, dtype_size, estimate_flops, estimate_memory
from .classifier import classify_op
from .trace_common import _is_overhead_event
from .trace_parser import _infer_role, _strip_instance_idx

# Chrome-trace categories that carry device (GPU/XPU) kernel time.
_KERNEL_CATEGORIES = {"kernel", "gpu_memcpy", "xpu_op", "gpu_op"}

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
    dims = args.get("Input Dims")
    if dims is None:
        raw = args.get("input_shapes")
        if isinstance(raw, str):
            try:
                dims = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                dims = None
    if not isinstance(dims, (list, tuple)):
        return []
    shapes: list[list[int]] = []
    for tensor in dims:
        if isinstance(tensor, (list, tuple)) and tensor:
            shape = [int(d) for d in tensor if isinstance(d, (int, float))]
            if shape:
                shapes.append(shape)
    return shapes


def _first_dtype(args: dict) -> str:
    """First concrete input dtype, e.g. 'c10::BFloat16' → 'bfloat16'."""
    types = args.get("Input type")
    if isinstance(types, (list, tuple)):
        for t in types:
            if not t:
                continue
            name = str(t).split("::")[-1].lower()
            name = name.replace("bfloat16", "bfloat16").replace("half", "float16")
            if name in DTYPE_BYTES:
                return name
    return ""


def _build_device_time_map(events: list[dict]) -> dict[int, float]:
    """Map ``cpu_op External id → total device (kernel) microseconds``.

    Kernels don't always carry the cpu_op's External id directly, so resolve
    through the runtime correlation table:
    ``kernel.correlation → xpu_runtime.External id``.
    """
    corr_to_ext: dict[int, int] = {}
    for evt in events:
        if evt.get("cat") in ("xpu_runtime", "cuda_runtime"):
            a = evt.get("args", {})
            corr = a.get("correlation")
            ext = a.get("External id")
            if corr is not None and ext is not None:
                corr_to_ext[corr] = ext

    ext_to_dev: dict[int, float] = {}
    for evt in events:
        if evt.get("cat") not in _KERNEL_CATEGORIES:
            continue
        a = evt.get("args", {})
        dur = evt.get("dur", 0) or 0
        ext = corr_to_ext.get(a.get("correlation"), a.get("External id"))
        if ext is None:
            continue
        ext_to_dev[ext] = ext_to_dev.get(ext, 0.0) + dur
    return ext_to_dev


# ===================================================================
# Raw nesting tree (modules + ops) via time-containment
# ===================================================================

class _Raw:
    """A node in the raw trace nesting tree (a module or a leaf op)."""

    __slots__ = ("kind", "label", "ts", "end", "dur", "ext", "shapes",
                 "dtype", "children", "self_dev", "sub_dev")

    def __init__(self, kind: str, label: str, ts: float, dur: float):
        self.kind = kind          # "module" or "op"
        self.label = label        # class name (module) or op name (op)
        self.ts = ts
        self.dur = dur
        self.end = ts + dur
        self.ext: int | None = None
        self.shapes: list[list[int]] = []
        self.dtype = ""
        self.children: list[_Raw] = []
        self.self_dev = 0.0       # device us launched directly by this op
        self.sub_dev = 0.0        # device us of this node + all descendants


def _build_raw_forest(events: list[dict], ext_to_dev: dict[int, float]
                      ) -> list[_Raw]:
    """Build the module/op nesting forest for the busiest worker thread."""
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"
               and e.get("ph") == "X"]
    if not cpu_ops:
        return []

    # The worker thread runs the model forward; pick the tid with most cpu_ops.
    tid_counts: dict[Any, int] = {}
    for e in cpu_ops:
        tid_counts[e.get("tid")] = tid_counts.get(e.get("tid"), 0) + 1
    worker_tid = max(tid_counts, key=tid_counts.get)

    nodes: list[_Raw] = []
    for e in events:
        if e.get("tid") != worker_tid or e.get("ph") != "X":
            continue
        cat = e.get("cat")
        name = e.get("name", "")
        ts = e.get("ts", 0)
        dur = e.get("dur", 0) or 0
        if cat == "python_function" and name.startswith("nn.Module:"):
            cls = name.split("nn.Module:", 1)[1].strip()
            nodes.append(_Raw("module", cls, ts, dur))
        elif cat == "cpu_op":
            if _is_overhead_event(name):
                continue
            n = _Raw("op", name, ts, dur)
            a = e.get("args", {})
            n.ext = a.get("External id")
            n.shapes = _parse_input_dims(a)
            n.dtype = _first_dtype(a)
            n.self_dev = ext_to_dev.get(n.ext, 0.0) if n.ext is not None else 0.0
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

    _compute_sub_dev(roots)
    return roots


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


def _partition_steps(roots: list[_Raw]) -> tuple[list[dict], str]:
    """Group top-level module roots into inference steps.

    A step is one iteration of the engine: the main model forward followed by
    its post-processing roots (``LogitsProcessor``, ``Sampler`` ...). The main
    model class is the module-root class that accounts for the most device time.

    Returns ``(steps, main_class)`` where each step is
    ``{"main": _Raw|None, "roots": [_Raw, ...], "token": int}``.
    """
    module_roots = sorted([r for r in roots if r.kind == "module"],
                          key=lambda r: r.ts)
    if not module_roots:
        return [], ""

    dev_by_class: dict[str, float] = {}
    for r in module_roots:
        norm = _strip_instance_idx(r.label)
        dev_by_class[norm] = dev_by_class.get(norm, 0.0) + r.sub_dev
    main_class = max(dev_by_class, key=dev_by_class.get)

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

    Returns ``(prefill_roots, decode_roots, n_prefill_steps, n_decode_steps)``.
    """
    steps, _ = _partition_steps(roots)
    if not steps:
        return [], [], 0, 0

    main_tokens = [s["token"] for s in steps if s["main"] is not None]
    max_tok = max(main_tokens) if main_tokens else 0
    min_tok = min(main_tokens) if main_tokens else 0

    prefill: list[_Raw] = []
    decode: list[_Raw] = []
    n_pre = n_dec = 0
    for s in steps:
        tok = s["token"]
        if max_tok == min_tok:
            # Only one kind of step captured — decide against the batch size.
            is_prefill = max_tok > max(batch_size, 1)
        else:
            is_prefill = tok == max_tok
        if is_prefill:
            prefill.extend(s["roots"])
            n_pre += 1
        else:
            decode.extend(s["roots"])
            n_dec += 1
    return prefill, decode, n_pre, n_dec


# ===================================================================
# Merge / collapse into display tree
# ===================================================================

def _op_signature(n: _Raw) -> tuple:
    return (n.label, tuple(tuple(s) for s in n.shapes))


def _module_children(node: _Raw) -> list[_Raw]:
    return [c for c in node.children if c.kind == "module"]


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

    # --- aggregate direct ops by (name, shapes) ---
    op_groups: dict[tuple, dict] = {}
    op_order: list[tuple] = []
    for inst in instances:
        for op in _direct_ops(inst):
            sig = _op_signature(op)
            g = op_groups.get(sig)
            if g is None:
                g = {"raw": op, "dev": 0.0, "host": 0.0, "count": 0}
                op_groups[sig] = g
                op_order.append(sig)
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

    return {
        "module_type": _strip_instance_idx(base.label),
        "op_groups": op_groups,
        "op_order": op_order,
        "child_groups": child_groups,
        "child_order": child_order,
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
) -> dict:
    """Turn a merged module description into the serialized display dict."""
    n_forward = merged["n_forward"]
    module_path = _split_path_types(path)

    ops_out: list[dict] = []
    node_dev = 0.0
    node_cpu = 0.0
    node_mem = 0
    node_flops = 0
    for sig in merged["op_order"]:
        g = merged["op_groups"][sig]
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
            device_type="xpu" if dev > 0 else "",
            self_device_time_us=dev,
            device_time_us=dev,
        )
        role = _infer_role(module_path + [merged["module_type"]], raw.label) \
            or raw.label.split("::")[-1]
        sym_shapes = [
            _symbolize(s, symbols_val, token_symbol, token_val) for s in shapes
        ]
        out_shape = _symbolize(_output_shape(raw.label, shapes),
                               symbols_val, token_symbol, token_val)
        ops_out.append({
            "name": raw.label,
            "role": role,
            "backend": backend.value,
            "category": category,
            "input_shapes": sym_shapes,
            "output_shape": out_shape,
            "memory_bytes": mem,
            "flops": flops,
            "ai": round(flops / mem, 2) if mem > 0 else 0,
            "phase": "both",
            "device_time_us": round(dev, 2),
            "cpu_time_us": round(cpu, 2),
            "count": count,
        })
        node_dev += dev
        node_cpu += cpu
        node_mem += mem
        node_flops += flops

    raw_children: list[dict] = []
    for key in merged["child_order"]:
        insts = merged["child_groups"][key]
        norm, _ = key
        child_merged = _merge_modules(insts, n_forward)
        child = _finalize_node(
            child_merged,
            name=_module_display_name(norm),
            path=path + "/" + norm,
            repeat_count=1,
            symbols_val=symbols_val,
            dtype_bytes=dtype_bytes,
            token_symbol=token_symbol,
            token_val=token_val,
        )
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
        "name": name,
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
                      token_symbol: str, token_val: int) -> dict | None:
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
    """Replace known dimension values with symbol names."""
    out: list = []
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            out.append(dim)
        elif i == 0 and token_val and dim == token_val:
            out.append(token_symbol)
        elif dim in symbols_val:
            out.append(symbols_val[dim])
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
    sym_to_val["TP"] = tp_size
    return val_to_sym, sym_to_val


# ===================================================================
# Public entry point
# ===================================================================

def build_graph_from_trace(
    trace_path: str,
    summary: dict | None = None,
    tp_size: int = 1,
    batch_size: int = 1,
    quantization: str | None = None,
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

    Returns:
        Dict with ``prefill`` / ``decode`` trees (either may be ``None``),
        ``symbols``, ``config`` and timing metadata — matching the serialized
        format of ``build_model_graph`` plus per-op ``device_time_us``.
    """
    summary = summary or {}
    trace = _load_trace(trace_path)
    events = trace.get("traceEvents", [])
    if not events:
        return {"prefill": None, "decode": None, "symbols": {},
                "config": {}, "has_timing": False,
                "error": "empty trace"}

    ext_to_dev = _build_device_time_map(events)
    roots = _build_raw_forest(events, ext_to_dev)
    if not roots:
        return {"prefill": None, "decode": None, "symbols": {},
                "config": {}, "has_timing": False,
                "error": "no module/op events (trace missing with_stack?)"}

    prefill_passes, decode_passes, n_pre, n_dec = _classify_steps(
        roots, batch_size)

    dtype = summary.get("dtype", "bfloat16")
    dtype_bytes = dtype_size(dtype)
    val_to_sym, sym_to_val = _build_symbol_tables(summary, tp_size)

    prefill_tokens = max((_pass_token_dim(p) for p in prefill_passes), default=0)
    decode_tokens = max((_pass_token_dim(p) for p in decode_passes), default=0)

    prefill_tree = _build_phase_tree(
        prefill_passes, n_pre, val_to_sym, dtype_bytes, "S", prefill_tokens)
    decode_tree = _build_phase_tree(
        decode_passes, n_dec, val_to_sym, dtype_bytes, "B", decode_tokens)

    if prefill_tokens:
        sym_to_val["S"] = prefill_tokens
    if decode_tokens:
        sym_to_val["B"] = decode_tokens

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
