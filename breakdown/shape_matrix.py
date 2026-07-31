# SPDX-License-Identifier: Apache-2.0
"""Shape Matrix rows: the reconstructed graph swept over (S, C, B, TP).

One row per (Phase, Seq Len, Ctx Len, Batch Size, TP, Module, Op). The rows are
the *pipeline's* data structure - the ``.xlsx`` export
(:mod:`breakdown.shape_matrix_xlsx`) is one serialization of them, and
:mod:`breakdown.bench.spec` consumes them directly, so no benchmark run has to
round-trip through a spreadsheet.

The profile contributes the accurate op set, recorded shapes/dtypes and
backends; Memory/FLOPs are analytic functions of (op, shape, dtype) recomputed
per config - not measured values.
"""
from __future__ import annotations

from typing import Any

from breakdown.shape_derive import (
    _MAX_MATRIX_ROWS,
    _config_symbols,
    _flatten_graph_nodes,
    _format_op_shape_with_dtypes,
    _partially_resolve_dim,
    _profile_op_memory,
    _resolve_shape_ints,
    estimate_flops,
    estimate_memory,
)

MAX_MATRIX_ROWS = _MAX_MATRIX_ROWS

#: Column order of the exported matrix; also the row dict keys.
MATRIX_HEADERS = [
    "Phase", "Seq Len", "Ctx Len", "Batch Size", "TP",
    "Module", "Op Name", "Backend", "Layers",
    "Symbolic Shape", "Shape",
    "Memory (bytes)", "FLOPs", "AI",
]

DEFAULT_SWEEP: dict[str, list[int]] = {
    "prefill_seq_lens": [128, 256, 512, 1024, 2048, 4096, 8192],
    "prefill_ctx_lens": [0, 8192],
    "prefill_batch_sizes": [1],
    "decode_ctx_lens": [8192],
    "decode_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
    "tp_sizes": [1, 2, 4, 8],
}


def build_configs(
    prefill_seq_lens: list[int],
    prefill_ctx_lens: list[int],
    prefill_batch_sizes: list[int],
    decode_ctx_lens: list[int],
    decode_batch_sizes: list[int],
    tp_sizes: list[int],
) -> list[dict[str, Any]]:
    """Cross product of the sweep; decode seq_len is always 1."""
    configs: list[dict[str, Any]] = []
    for seq in prefill_seq_lens:
        for ctx in prefill_ctx_lens:
            for bs in prefill_batch_sizes:
                for tp in tp_sizes:
                    configs.append({"phase": "prefill", "seq_len": seq,
                                    "ctx_len": ctx, "batch_size": bs,
                                    "tp_size": tp})
    for ctx in decode_ctx_lens:
        for bs in decode_batch_sizes:
            for tp in tp_sizes:
                configs.append({"phase": "decode", "seq_len": 1,
                                "ctx_len": ctx, "batch_size": bs,
                                "tp_size": tp})
    return configs


def ops_per_config(template: dict) -> int:
    """Op count of one config, for the row-limit guard."""
    tree = template.get("prefill") or template.get("decode")
    if not tree:
        return 0
    return sum(len(node["ops"]) for node in _flatten_graph_nodes(tree))


def estimate_row_count(template: dict, configs: list[dict]) -> int:
    return len(configs) * ops_per_config(template)


def build_rows(template: dict, configs: list[dict]) -> list[dict[str, Any]]:
    """Sweep the reconstructed graph over ``configs`` into flat matrix rows."""
    graph_cfg = template.get("config", {})
    base_symbols = template.get("symbols", {})
    pdtype_bytes = graph_cfg.get("dtype_bytes", 2)
    rows: list[dict[str, Any]] = []

    for cfg in configs:
        tree = template.get(cfg["phase"])
        if not tree:
            continue
        symbols = _config_symbols(base_symbols, cfg)

        for node_info in _flatten_graph_nodes(tree):
            effective_repeat = node_info.get("effective_repeat", 1)
            for op in node_info["ops"]:
                # Prefer the real per-tensor dtypes recorded in the trace.
                recorded_dtypes = op.get("input_dtypes")
                shape_str = _format_op_shape_with_dtypes(
                    op, symbols, graph_cfg, recorded_dtypes=recorded_dtypes)

                # Symbolic shape: keep only config variables (S, B, C, TP)
                # symbolic, resolve model constants to numbers.
                sym_shapes = op.get("input_shapes", [])
                if sym_shapes:
                    sym_parts = []
                    for s in sym_shapes:
                        if isinstance(s, list):
                            dims = [_partially_resolve_dim(d, symbols)
                                    for d in s]
                            sym_parts.append("[" + ", ".join(dims) + "]")
                        else:
                            sym_parts.append(_partially_resolve_dim(s, symbols))
                    symbolic_str = " × ".join(sym_parts)
                else:
                    symbolic_str = "—"

                # Recompute Memory/FLOPs from this config's resolved shapes so
                # every row is self-consistent; Memory uses the recorded
                # per-tensor dtypes for accuracy.
                resolved = _resolve_shape_ints(sym_shapes, symbols)
                op_name = op.get("name", "")
                if recorded_dtypes:
                    mem_bytes = _profile_op_memory(
                        op_name, resolved, recorded_dtypes, pdtype_bytes)
                else:
                    mem_bytes = estimate_memory(op_name, resolved, pdtype_bytes)
                # Attention's key/value rows are the whole batch's KV read in
                # *decode* (``B·C``); in prefill they are the single sequence's
                # ``S+C``, which carries no batch factor. Dividing a prefill row
                # by the batch would understate the heaviest op by exactly the
                # batch size.
                n_seqs = (int(cfg.get("batch_size") or 1)
                          if cfg["phase"] == "decode" else 1)
                flops = estimate_flops(op_name, resolved, n_seqs=n_seqs)
                ai = round(flops / mem_bytes, 2) if mem_bytes > 0 else 0

                path = node_info["path"]
                role = op.get("role", "")
                rows.append({
                    "Phase": cfg["phase"],
                    "Seq Len": cfg["seq_len"],
                    "Ctx Len": cfg["ctx_len"],
                    "Batch Size": cfg["batch_size"],
                    "TP": cfg["tp_size"],
                    "Module": f"{path}.{role}" if role else path,
                    "Op Name": op_name,
                    "Backend": op.get("backend", ""),
                    "Layers": effective_repeat,
                    "Symbolic Shape": symbolic_str,
                    "Shape": shape_str,
                    "Memory (bytes)": mem_bytes,
                    "FLOPs": flops,
                    "AI": ai,
                    # kept out of the export, used by the replay benchmark
                    "_resolved_shapes": resolved,
                    "_input_dtypes": list(recorded_dtypes or []),
                    "_input_args": op.get("input_args") or [],
                    "_recorded_shapes": op.get("recorded_shapes") or [],
                    "_device_time_us": op.get("device_time_us", 0),
                    "_op_role": role,
                    "_module_type": node_info.get("module_type", ""),
                })
    return rows


def build_info_rows(model_id: str, template: dict,
                    profile_settings: dict | None) -> list[tuple[str, Any]]:
    """Provenance / caveat rows so consumers know what the shapes mean."""
    from breakdown.shape_derive import _validate_derived_shapes

    ps = profile_settings or {}
    pcfg = template.get("config", {})
    info_rows: list[tuple[str, Any]] = [
        ("Shape source", "profile-derived (grounded in a profiling run)"),
        ("Model", model_id),
        ("Profiled query_len (S)", ps.get("query_len")),
        ("Profiled context_len (C)", ps.get("context_len")),
        ("Profiled decode batch (B)", ps.get("decode_batch_size")),
        ("Profiled TP", pcfg.get("tp_size", ps.get("tp_size"))),
        ("Profiled quantization", pcfg.get("quantization")),
        ("Profiled mode", ps.get("mode")),
        ("", ""),
        ("How rows are derived",
         "The profile contributes the accurate op set, real recorded "
         "shapes and backends. Shapes are re-resolved per config "
         "(S/B/C/TP). Memory/FLOPs are then analytic functions of "
         "(op, shape, dtype), recomputed per config — NOT measured values."),
        ("Memory/FLOPs are estimates",
         "Heuristic (op+shape) estimates, not measured; e.g. attention "
         "FLOPs are not modeled. Only the op set and shapes come from the "
         "trace. Measured device time is intentionally not in this sweep."),
        ("Caveat — op set / TP",
         "The op set (TP collectives, MoE routing, chunked-prefill splits) "
         "is fixed at the profiled config. Profile at each TP you need — "
         "sweeping TP only divides /TP dims, it does not add comm ops that "
         "weren't profiled."),
        ("Context (C) is parametric",
         "No need to profile per context. One base profile with any "
         "non-zero context captures C as S+C on the prefill attention KV "
         "rows; every other context length is then derived by resolving C. "
         "A base profile with context=0 leaves KV rows as S, so context "
         "can't be derived — profile with a small non-zero context."),
        ("Caveat — timing",
         "Device time is valid only at the profiled point and is not "
         "included in this sweep; rows carry shapes/memory/FLOPs only."),
    ]
    # Round-trip validation: re-resolving each op's symbolic shape at the
    # profiled config must reproduce the shape actually recorded in the
    # trace — a self-consistency proof for the derivation machinery.
    vres = _validate_derived_shapes(template)
    if vres["total"]:
        pct = 100.0 * vres["matched"] / vres["total"]
        info_rows.append((
            "Shape validation",
            f"{vres['matched']}/{vres['total']} op input shapes "
            f"({pct:.1f}%) re-resolve at the profiled config to exactly the "
            "shape recorded in the trace (context-annotated KV rows "
            "excluded). This validates the symbolic derivation.",
        ))
        if vres["mismatched"]:
            info_rows.append((
                "Validation mismatches",
                f"{vres['mismatched']} mismatched. Examples: "
                + " | ".join(vres["examples"]),
            ))
    return info_rows
