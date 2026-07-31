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

from typing import Any

from breakdown.analyzer import dtype_size, estimate_flops, estimate_memory


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
    """Convert byte count to short dtype name."""
    return {4: "fp32", 2: "bf16", 1: "fp8"}.get(nbytes, f"{nbytes * 8}bit")


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
                tensor_dtype = _friendly_dtype(recorded_dtypes[shape_idx])
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


def _safe_arithmetic_eval(expr: str) -> int:
    """Safely evaluate a simple arithmetic expression (integers, +, -, *, /).

    Only allows integer literals and the operators +, -, *, /.
    Division is performed as integer (floor) division.
    Raises ValueError for anything else.
    """
    import ast

    # Normalize "/" to "//" for integer division in eval
    expr = expr.replace("//", "/").replace("/", "//")

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        elif isinstance(node, (ast.Constant,)):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Non-numeric constant: {node.value}")
        elif not isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv,
                                   ast.USub, ast.UAdd)):
            raise ValueError(f"Unsupported node: {type(node).__name__}")
    return int(eval(compile(tree, "<dim>", "eval")))  # noqa: S307


def _resolve_dim(dim, symbols: dict[str, int]):
    """Resolve a dimension value to a concrete integer if possible.

    A dim that is *sharded* over TP (``4/TP``) is clamped to at least 1: when
    TP exceeds the number of KV heads/experts, the engine replicates the shard
    across ranks instead of giving some ranks an empty tensor, so a resolved 0
    is a division artefact that would otherwise emit degenerate benchmark
    shapes (``kv_head_num=0``, ``K=0``) that every kernel rejects.
    """
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        # Direct lookup
        if dim in symbols:
            return symbols[dim]
        # Try evaluating composite expressions like "S+C", "2·I"
        # Replace symbol names with their values and evaluate
        expr = dim
        sharded = "TP" in dim and "/" in dim
        # Sort by length descending to avoid partial replacements
        for name in sorted(symbols.keys(), key=len, reverse=True):
            expr = expr.replace(name, str(symbols[name]))
        # Replace middle-dot with *
        expr = expr.replace("·", "*")
        try:
            value = _safe_arithmetic_eval(expr)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return dim
        return max(1, value) if sharded else value
    return dim


# Config-dependent variable symbols that should stay symbolic
_VARIABLE_SYMS = {"S", "B", "C", "TP"}


def _is_variable_composite(expr: str) -> bool:
    """Check if an expression is composed entirely of variable symbols.

    Handles both additive (S+C) and multiplicative (B·S) composites.
    """
    # Split on + and · to get individual parts
    parts = expr.replace("·", "+").split("+")
    return all(p.strip() in _VARIABLE_SYMS for p in parts)


def _partially_resolve_dim(dim, symbols: dict[str, int],
                           full_symbols: dict[str, int] | None = None,
                           tp_divided: set[str] | None = None):
    """Resolve dim keeping only S/B/C/TP symbolic, resolving all else to numbers.

    Model constants from config.json are shown as numbers. When a dimension
    contains "/TP", it's shown as "value/TP" using the full undivided config
    value from the symbols dict.

    The full_symbols and tp_divided params are accepted for backwards
    compatibility but ignored — the graph now embeds /TP directly in shapes
    and symbols already contain original (undivided) values.
    """
    if isinstance(dim, (int, float)):
        return str(int(dim))

    s = str(dim)

    # Pure variable symbol → keep as-is
    if s in _VARIABLE_SYMS:
        return s

    # Composite of only variable symbols (e.g. "S+C", "B·S") → keep as-is
    if _is_variable_composite(s):
        return s

    # Handle "/TP" suffix: resolve the base part, keep /TP
    if s.endswith("/TP"):
        base = s[:-3]  # strip "/TP"
        resolved_base = _resolve_constant_expr(base, symbols)
        return f"{resolved_base}/TP"

    # Check if s is a known symbol directly (handles names with · like "n_h·D_qh")
    if s in symbols:
        return str(symbols[s])

    # Check for multiply composites containing a variable (e.g., "B·S·K")
    if "·" in s:
        parts = s.split("·")
        has_variable = any(p in _VARIABLE_SYMS for p in parts)
        if has_variable:
            # Partially resolve: keep variable parts, resolve constants
            resolved_parts = []
            for p in parts:
                if p in _VARIABLE_SYMS:
                    resolved_parts.append(p)
                elif p in symbols:
                    resolved_parts.append(str(symbols[p]))
                elif p.isdigit():
                    resolved_parts.append(p)
                else:
                    resolved_parts.append(p)
            return "·".join(resolved_parts)
        else:
            # All parts are constants — compute product
            product = 1
            for p in parts:
                val = symbols.get(p)
                if val is not None:
                    product *= val
                elif p.isdigit():
                    product *= int(p)
            return str(product)

    # Pure constant — fully resolve
    if s in symbols:
        return str(symbols[s])
    resolved = _resolve_dim(s, symbols)
    return str(resolved)


def _resolve_constant_expr(expr: str, symbols: dict[str, int]) -> str:
    """Resolve a constant expression (no variables) to its numeric value.

    Handles symbols like "QKV", "n_h·d", "2·I", and plain numbers.
    """
    if expr in symbols:
        return str(symbols[expr])
    if "·" in expr:
        parts = expr.split("·")
        product = 1
        for p in parts:
            val = symbols.get(p)
            if val is not None:
                product *= val
            elif p.isdigit():
                product *= int(p)
            else:
                return expr  # can't resolve
        return str(product)
    if expr.isdigit():
        return expr
    return expr


# ---- Profile-derived Shape Matrix helpers ----

_FRIENDLY_DTYPE = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "fp16": "fp16",
    "float32": "fp32", "float": "fp32", "fp32": "fp32",
    "float8_e4m3fn": "fp8e4m3", "float8_e5m2": "fp8e5m2", "fp8": "fp8",
    "int8": "int8", "uint8": "uint8",
    "int64": "i64", "int32": "i32", "int16": "i16", "long": "i64", "int": "i32",
    "bool": "bool",
}


def _friendly_dtype(name: str) -> str:
    """Short display name for a recorded trace dtype ('bfloat16' → 'bf16')."""
    if not name:
        return ""
    return _FRIENDLY_DTYPE.get(name.lower(), name.lower())


def _prod_ints(shape) -> int:
    p = 1
    for d in shape:
        if isinstance(d, int):
            p *= d
        else:
            return 0
    return p


_MM_OP_BASES = {"mm", "addmm", "linear", "matmul", "bmm", "_scaled_mm",
                "fp8_gemm", "fp4_gemm", "int4_gemm_w4a16", "int4_gemm_w4a8"}

#: Ops that *index into a table* rather than stream it. ``(table operand index,
#: operand whose element count is the number of rows looked up)``.
#:
#: Charging these for the whole table is not a rounding error: an embedding is
#: charged for the entire vocabulary matrix and a RoPE call for the entire
#: ``[max_position, head_dim]`` cos/sin cache, when both touch only one row per
#: token. That produced "utilization 37000 % of peak" - a number that says
#: nothing about the kernel and pushed the op into ``check_cost_model`` instead
#: of giving it an honest roofline.
_TABLE_LOOKUP_OPS: dict[str, tuple[int, int]] = {
    "embedding": (0, 1),          # (weight [V, H], indices [T])
    "rotary_embedding": (3, 0),   # (cos_sin_cache [P, d], positions [T])
}


def _lookup_reads(base: str, shapes: list[list[int]], dtypes: list[str],
                  act_bytes: int) -> int | None:
    """Bytes a table-lookup op really reads, or ``None`` if the rule doesn't fit."""
    spec = _TABLE_LOOKUP_OPS.get(base)
    if spec is None:
        return None
    t_i, i_i = spec
    if t_i >= len(shapes) or i_i >= len(shapes):
        return None
    table, index = shapes[t_i], shapes[i_i]
    if len(table) < 2:
        return None
    rows = _prod_ints(index)
    if rows <= 0 or rows >= table[0]:
        return None                  # touches (at least) the whole table anyway
    row_bytes = _prod_ints(table[1:]) * (
        dtype_size(dtypes[t_i]) if t_i < len(dtypes) and dtypes[t_i]
        else act_bytes)
    total = rows * row_bytes
    for i, s in enumerate(shapes):
        if i == t_i:
            continue
        n = _prod_ints(s)
        if n <= 0:
            continue
        total += n * (dtype_size(dtypes[i]) if i < len(dtypes) and dtypes[i]
                      else act_bytes)
    return total


def _profile_op_memory(op_name: str, shapes: list[list[int]],
                       dtypes: list[str], act_bytes: int) -> int:
    """Estimate op memory using the *recorded per-tensor* dtypes.

    Reads are sized per input tensor with its own dtype (so an fp8/int4 weight
    counts 1 byte while a bf16 activation counts 2) — more accurate than a
    single global dtype for quantized ops. The write (output) is sized at the
    activation dtype. Falls back to 0 when shapes aren't concrete.
    """
    if not shapes:
        return 0
    base = op_name.split("::")[-1].lower()
    lookup = _lookup_reads(base, shapes, dtypes, act_bytes)
    if lookup is not None:
        # The gathered rows are also what gets written back.
        return lookup + _prod_ints(shapes[_TABLE_LOOKUP_OPS[base][1]]) * \
            _prod_ints(shapes[_TABLE_LOOKUP_OPS[base][0]][1:]) * act_bytes
    reads = 0
    for i, s in enumerate(shapes):
        n = _prod_ints(s)
        if n == 0:
            # A genuinely empty operand costs nothing; it must not zero the
            # whole estimate. vLLM's attention op is dispatched with an empty
            # ``kv_cache_dummy_dep`` tensor purely to order it against the KV
            # write, and aborting here left the heaviest op in the profile with
            # no analytic cost at all (roofline bound "unknown", 100 % apparent
            # headroom). Unresolvable shapes never reach here - they are dropped
            # by ``_resolve_shape_ints`` before this point.
            continue
        b = dtype_size(dtypes[i]) if i < len(dtypes) and dtypes[i] else act_bytes
        reads += n * b
    if (base in _MM_OP_BASES and len(shapes) >= 2
            and len(shapes[0]) >= 2 and len(shapes[1]) >= 2):
        out = _prod_ints(shapes[0][:-1]) * shapes[1][-1]
    else:
        out = _prod_ints(shapes[0])
    return reads + out * act_bytes


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
