# SPDX-License-Identifier: Apache-2.0
"""Symbolic shape derivation shared by the Shape Matrix export and the perf pipeline.

Everything here is pure: it turns a reconstructed graph's symbolic shapes plus a
config point (S/B/C/TP) into concrete dims, dtypes, memory and FLOPs. It is
deliberately free of Flask and of torch so both ``app.py`` and
``breakdown.shape_matrix`` (which must import on a machine with no GPU) can
use it.

Extracted verbatim from ``app.py`` - see ``breakdown/shape_matrix.py`` for
the row builder that consumes it.
"""
from __future__ import annotations

from breakdown.core import dims
from breakdown.core.dtypes import label as dtype_label, label_for_bytes
from breakdown.cost import _prod, op_bytes


# Roles whose 2nd input tensor (index 1) is a weight matrix
_WEIGHT_ROLES = {
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
    "expert_gate_up", "expert_down",
    "shared_expert_gate_up", "shared_expert_down",
    "q_compress", "q_decompress", "kv_compress",
    "router_gate", "lm_head", "vl_projector",
    "vit_qkv_proj", "vit_o_proj", "vit_mlp_up", "vit_mlp_down",
    "patch_embed", "mlp_up", "mlp_down",
}


def _bytes_to_dtype(nbytes: int) -> str:
    """The dtype label a configured element width implies."""
    return label_for_bytes(nbytes)


def _quant_dtype_name(quant: str | None) -> str | None:
    """Map quantization method to weight dtype short name."""
    if not quant or quant == "None":
        return None
    names = {
        "fp8": "fp8", "gptq": "int4", "gptq_marlin": "int4",
        "awq": "int4", "awq_marlin": "int4", "marlin": "int4",
        "squeezellm": "int4", "bitsandbytes": "nf4", "gguf": "q4",
        "int4": "int4", "int8": "int8",
    }
    return names.get(quant.lower(), quant.lower())


def _get_tensor_dtype(tensor_idx: int, role: str,
                      graph_cfg: dict) -> str:
    """Get the dtype label for a specific tensor in an op.

    Mirrors the frontend getOpDtypes logic: weight tensors (index 1) of
    projection ops use the weight dtype; everything else uses activation dtype.
    """
    act_dtype = _bytes_to_dtype(graph_cfg.get("dtype_bytes", 2))
    quant = graph_cfg.get("quantization")
    if quant:
        w_dtype = _quant_dtype_name(quant) or _bytes_to_dtype(
            graph_cfg.get("weight_dtype_bytes", 2)
        )
    else:
        w_dtype = act_dtype

    if tensor_idx == 1 and role in _WEIGHT_ROLES:
        return w_dtype
    return act_dtype


def _flatten_graph_nodes(node: dict, depth: int = 0,
                         rows: list | None = None,
                         parent_repeat: int = 1) -> list[dict]:
    """Flatten a hierarchical graph tree into rows for the hierarchy sheet."""
    if rows is None:
        rows = []

    # Effective repeat = own repeat_count × parent's repeat
    own_repeat = node.get("repeat_count", 1)
    effective_repeat = own_repeat * parent_repeat

    # Add module row
    rows.append({
        "depth": depth,
        "name": node.get("name", ""),
        "path": node.get("path", ""),
        "module_type": node.get("module_type", ""),
        "repeat_count": own_repeat,
        "effective_repeat": effective_repeat,
        "total_memory": node.get("total_memory", 0),
        "total_flops": node.get("total_flops", 0),
        "total_ai": node.get("total_ai", 0),
        "ops": node.get("ops", []),
    })

    # Recurse into children
    for child in node.get("children", []):
        _flatten_graph_nodes(child, depth + 1, rows, effective_repeat)

    return rows


# ---- Shape Matrix Export (single model, multi-config sweep) ----

# Max total rows to prevent excessive memory/time
_MAX_MATRIX_ROWS = 50000


def _format_op_shape_with_dtypes(
    op: dict, symbols: dict[str, int], graph_cfg: dict,
    recorded_dtypes: list[str] | None = None,
) -> str:
    """Format op shapes as concrete values with per-tensor dtypes.

    Example: "[128, 2560, bf16] × [2560, 6144, fp8]"
    Resolves symbolic dims (including composite like "S+C") via the symbols dict.

    When ``recorded_dtypes`` is given (the real per-tensor dtypes captured in the
    profiling trace, aligned with ``op["input_shapes"]``), it takes precedence
    over the config-driven ``_get_tensor_dtype`` heuristic so the exported dtype
    is exactly what the op actually ran.
    """
    op_shapes = op.get("input_shapes", [])
    if not op_shapes:
        return "—"
    role = op.get("role", "")
    parts = []
    for shape_idx, shape in enumerate(op_shapes):
        if isinstance(shape, list):
            tensor_dtype = None
            if recorded_dtypes and shape_idx < len(recorded_dtypes):
                tensor_dtype = dtype_label(recorded_dtypes[shape_idx])
            if not tensor_dtype:
                tensor_dtype = _get_tensor_dtype(shape_idx, role, graph_cfg)
            dims = []
            for dim in shape:
                dims.append(str(_resolve_dim(dim, symbols)))
            if tensor_dtype:
                dims.append(tensor_dtype)
            parts.append("[" + ", ".join(dims) + "]")
        else:
            parts.append(str(_resolve_dim(shape, symbols)))
    return " × ".join(parts)


#: Resolving a symbolic dim is :mod:`breakdown.core.dims`; these two names are
#: how the shape pipeline has always spelled it, kept so call sites read the
#: same. The parser replaced a textual substitution feeding an ``eval``.
_resolve_dim = dims.resolve
_partially_resolve_dim = dims.resolve_display


def _prod_ints(shape) -> int:
    """Product of a shape's dims; 0 if any is symbolic (kept for callers)."""
    return _prod(shape)


def _profile_op_memory(op_name: str, shapes: list[list[int]],
                       dtypes: list[str], act_bytes: int) -> int:
    """Bytes an op moves, using the *recorded per-tensor* dtypes.

    A thin name over :func:`breakdown.cost.op_bytes`, which is the single cost
    model - the graph's ``memory_bytes``, this export's ``Memory (bytes)`` and
    the benchmark's roofline must be the same number.
    """
    return op_bytes(op_name, shapes, dtypes, act_bytes)


def _config_symbols(base_symbols: dict[str, int], cfg: dict) -> dict[str, int]:
    """Return a copy of a profile graph's symbol table with the config
    variables (S/B/C/S+C/TP) overridden for one matrix configuration.

    Config *constants* (H, I, n_h·d, V, ...) are kept as-is so the op template's
    symbolic shapes resolve to this configuration's concrete dims.
    """
    sym = dict(base_symbols)
    if cfg["phase"] == "prefill":
        s_val = int(cfg["seq_len"])
    else:
        s_val = 1  # decode advances each sequence by one token
    c_val = int(cfg.get("ctx_len") or 0)
    sym["S"] = s_val
    sym["B"] = int(cfg["batch_size"])
    sym["C"] = c_val
    sym["S+C"] = s_val + c_val
    sym["TP"] = int(cfg["tp_size"])
    return sym


def _resolve_shape_ints(input_shapes, symbols: dict[str, int]) -> list[list[int]]:
    """Resolve an op's (symbolic) input shapes to concrete integer tensor shapes.

    Only fully-resolvable tensor (list) shapes with all-integer dims are kept;
    scalars and shapes with a dim that can't be resolved to an int are dropped,
    so ``estimate_memory``/``estimate_flops`` see a clean ``list[list[int]]``.
    """
    out: list[list[int]] = []
    for shape in input_shapes or []:
        if not isinstance(shape, list):
            continue
        dims = [_resolve_dim(d, symbols) for d in shape]
        if all(isinstance(d, int) for d in dims):
            out.append(dims)
    return out


def _validate_derived_shapes(template: dict) -> dict:
    """Round-trip check that the symbolic derivation reproduces reality.

    Resolving each op's symbolic ``input_shapes`` at the *profiled* config (the
    template's own ``symbols``) must reproduce the numeric shape recorded in the
    trace (``recorded_shapes``). Context-annotated KV dims (``C`` / ``S+C``) are
    excluded because the context is deliberately added, not recorded; dims that
    don't resolve to an int are skipped (can't compare). Returns counts + a few
    mismatch examples for the Info sheet.
    """
    syms = template.get("symbols", {})
    total = matched = 0
    examples: list[str] = []
    for phase in ("prefill", "decode"):
        tree = template.get(phase)
        if not tree:
            continue
        for node in _flatten_graph_nodes(tree):
            for op in node["ops"]:
                recorded = op.get("recorded_shapes")
                sym = op.get("input_shapes")
                if not recorded or not sym:
                    continue
                for rec, ss in zip(recorded, sym):
                    if not isinstance(ss, list) or not isinstance(rec, list):
                        continue
                    if any(isinstance(d, str) and d in ("C", "S+C", "B·C")
                           for d in ss):
                        continue  # deliberately context-annotated KV row
                    resolved = [_resolve_dim(d, syms) for d in ss]
                    if not all(isinstance(d, int) for d in resolved):
                        continue  # unresolved dim — nothing to compare
                    total += 1
                    if resolved == list(rec):
                        matched += 1
                    elif len(examples) < 6:
                        examples.append(
                            f"{op.get('name', '')} {ss}\u2192{resolved} vs {rec}")
    return {"total": total, "matched": matched,
            "mismatched": total - matched, "examples": examples}


def annotate_display_shapes(graph: dict) -> dict:
    """Give every op in a graph the shapes a reader should see.

    The browser used to work this out for itself: it had its own symbol
    resolver (``symTooltip``), its own byte-width-to-dtype table, its own copy
    of the quantization name map and of the set of roles whose second operand
    is a weight -- four transcriptions of Python that already existed, and one
    of them disagreed. ``symTooltip`` split on the multiply sign only, so an
    additive composite like ``S+C`` resolved to nothing and the reader was
    shown a symbol where every other dimension showed a number.

    So the server says it once. Each op gains::

        op["display"] = {"sym": [[dim, ...], ...],      # display form
                         "concrete": [[int, ...], ...] | None,
                         "dtypes": [label, ...]}

    Structured, not markup: what a dimension *is* belongs to the pipeline, how
    it looks belongs to the page.
    """
    symbols = dict(graph.get("symbols") or {})
    cfg = graph.get("config") or {}

    def annotate(node: dict) -> None:
        for op in node.get("ops") or []:
            shapes = op.get("input_shapes") or []
            sym = [[dims.resolve_display(d, symbols) for d in row]
                   if isinstance(row, list) else [dims.resolve_display(row, symbols)]
                   for row in shapes]
            concrete: list[list[int]] | None = []
            for row in shapes:
                cells = row if isinstance(row, list) else [row]
                resolved = [dims.resolve(d, symbols) for d in cells]
                if not all(isinstance(v, int) for v in resolved):
                    concrete = None
                    break
                concrete.append(resolved)
            op["display"] = {
                "sym": sym,
                "concrete": concrete,
                "dtypes": [_get_tensor_dtype(i, op.get("role") or "", cfg)
                           for i in range(len(shapes))],
            }
        for child in node.get("children") or []:
            annotate(child)

    for phase in ("prefill", "decode"):
        if graph.get(phase):
            annotate(graph[phase])
    return graph
