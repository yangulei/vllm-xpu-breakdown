# SPDX-License-Identifier: Apache-2.0
"""One node per module: merge a module's instances across forward passes,
name its children, and collapse structurally-identical repeated siblings.
"""
from __future__ import annotations


from ..analyzer import estimate_flops, estimate_memory
from ..classifier import classify_op
from ..trace_common import _infer_role, _strip_instance_idx
from .rules import (
    _KNOWN_MODULE_ROLES, _PLUMBING_OPS, _disambiguate_child_name,
    _module_display_name, _output_shape, _rowparallel_shape_role)
from .forest import _Raw


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

    # A module with a resolved semantic identity ("down_proj", "q_proj", ...)
    # lends it to its own projection matmul: the module knows what it is, the
    # op's path does not. Only the matmul-family op adopts it; a communication
    # op keeps its own role.
    module_role_override = (display_name if display_name in _KNOWN_MODULE_ROLES
                            else None)

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
        # A collective is its own role, never the enclosing projection's; a
        # projection matmul takes the role of the module that owns it.
        low_op = raw.label.lower()
        op_base = low_op.split("::")[-1]
        op_role = None
        if "all_reduce" in low_op or "allreduce" in low_op:
            op_role = "all_reduce"
        elif "all_gather" in low_op or "allgather" in low_op:
            op_role = "all_gather"
        elif "reduce_scatter" in low_op or "reducescatter" in low_op:
            op_role = "reduce_scatter"
        elif module_role_override and op_base in (
                "mm", "addmm", "linear", "matmul", "bmm"):
            op_role = module_role_override
        role = op_role \
            or _infer_role(module_path + [merged["module_type"]], raw.label) \
            or raw.label.split("::")[-1]
        # Shapes stay numeric here. Symbolization is a single ordered pass
        # over the finished trees (:mod:`breakdown.trace.symbols`) - doing it
        # per node would mean resolving a dim without knowing which other dims
        # the run produced, which is exactly what step 5 needs.
        sym_shapes = [list(s) for s in shapes]
        out_shape = _output_shape(raw.label, shapes)
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
            # For a kernel launched straight from Python (no cpu_op), the file
            # and function that launched it — the replay's entry point.
            "launch": dict(raw.launch) if raw.launch else None,
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
        if not child_name:
            # The span-less fallbacks: name a RowParallelLinear by the shape of
            # its own weight, then by its parent's type.
            child_name = (_rowparallel_shape_role(norm, child_merged,
                                                  symbols_val)
                          or _disambiguate_child_name(norm, occ_idx, merged))
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


def _split_path_types(path: str) -> list[str]:
    """Return the list of module class names along a '/'-joined path."""
    return [p for p in path.split("/") if p]
