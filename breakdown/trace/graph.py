# SPDX-License-Identifier: Apache-2.0
"""The reconstruction: trace file -> serialized module/op graph.

The pass order is flat and sequential on purpose; see
:func:`build_graph_from_trace`.
"""
from __future__ import annotations


from ..analyzer import dtype_size, estimate_flops, estimate_memory
from ..trace_common import _infer_device_from_trace
from .rules import _DEVICE_KERNEL_CATEGORIES, _is_attention_op
from .events import _load_trace, _worker_tid
from .forest import (
    _Raw, _build_raw_forest, _coalesce_duplicate_child_modules,
    _compute_sub_dev, _deepest_at, _forest_has_named_modules,
    _hoist_modules_under_ops)
from .kernels import (
    _attribute_kernels, _collect_kernel_launches, _kernel_leaf_coverage)
from .shapes import (
    _infer_attention_kernel_shapes, _infer_hidden_activation_ops)
from .phases import _classify_steps, _pass_token_dim
from .symbols import (
    _build_symbol_tables, _dim_is, _symbolize_moe_routed_rows,
    _symbolize_msa_dims, _symbolize_runtime_dims)
from .collapse import _build_phase_tree


# ===================================================================
# Pass 1: the forest
# ===================================================================

def build_forest(events: list[dict]) -> tuple[list[_Raw], float]:
    """Trace events -> the module/op forest, with device time attributed.

    The order of these passes is load-bearing, and is stated here rather than
    hidden inside any one of them:

    1. **nest** - module and op events nest by time-containment.
    2. **attribute** - every device kernel is charged to the deepest node
       containing its *host launch site*. This must run on the raw,
       non-overlapping forest: the launch-site lookup needs the intervals as
       recorded, before anything is re-parented.
    3. **hoist** - a module whose forward was wrapped in a fused custom op is
       lifted back beside it (vLLM's MoE block wraps ``shared_experts``).
    4. **coalesce** - the same module object recorded twice in one forward
       (the MoE shared-experts overlap) becomes one node.
    5. **roll up** - device time sums post-order. It must be last, or a hoisted
       subtree's time would be counted both under its module and inside the op
       that wrapped it.

    Returns ``(roots, device_us)``, the second being the run's **total**
    collected device time. The caller reports it as ``timing_device_us`` so the
    conservation invariant - no kernel's time is dropped - is checkable from
    outside.
    """
    roots = _build_raw_forest(events)
    if not roots:
        return [], 0.0
    worker_tid, _named = _worker_tid(events)
    launches = _collect_kernel_launches(events, worker_tid)
    device_us = sum(dur for _ts, _n, dur, _f in launches)
    _attribute_kernels(roots, launches)
    roots = _hoist_modules_under_ops(roots)
    _coalesce_duplicate_child_modules(roots)
    _compute_sub_dev(roots)
    return roots, device_us


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
    from ..shape_derive import _profile_op_memory

    dtypes = op.get("input_dtypes") or []
    mem = (_profile_op_memory(op["name"], numeric, dtypes, dtype_bytes)
           if dtypes else estimate_memory(op["name"], numeric, dtype_bytes))
    flops = estimate_flops(op["name"], numeric, n_seqs=max(n_seqs, 1))
    op["memory_bytes"] = mem
    op["flops"] = flops
    op["ai"] = round(flops / mem, 2) if mem > 0 else 0


def kernel_coverage(trace_path: str, batch_size: int = 1) -> dict[str, float]:
    """Where every device kernel of a trace lands — a public diagnostic.

    The reconstruction claims two things about device time: that no kernel is
    collected which is not a real kernel (a host-side ``cudaEventQuery``
    launches nothing), and that no collected kernel's time is silently dropped.
    Both are properties of a *trace*, so this exposes them without importing
    the passes that check them.

    Returns the :func:`_kernel_leaf_coverage` totals plus ``n_device_events``
    (the trace's real device-kernel count, which ``n_total`` must equal) and
    ``n_in_step`` / ``in_step_us`` (the launches inside a kept prefill/decode
    step, which must all land on a leaf).
    """
    events = _load_trace(trace_path).get("traceEvents", [])
    roots, _device_us = build_forest(events)
    if not roots:
        return {"n_total": 0, "n_device_events": 0, "n_in_step": 0}
    worker_tid, _named = _worker_tid(events)
    launches = _collect_kernel_launches(events, worker_tid)
    out = dict(_kernel_leaf_coverage(roots, launches))
    out["n_device_events"] = sum(
        1 for e in events if e.get("cat") in _DEVICE_KERNEL_CATEGORIES)
    prefill, decode, _, _ = _classify_steps(roots, batch_size)
    spans = [(r.ts, r.end) for r in prefill + decode]
    in_step = [(ts, dur) for ts, _n, dur, _f in launches
               if any(a <= ts < b for a, b in spans)]
    out["n_in_step"] = len(in_step)
    out["in_step_us"] = sum(d for _t, d in in_step)
    out["n_in_step_dropped"] = sum(
        1 for ts, _d in in_step if _deepest_at(roots, ts) is None)
    return out


def build_graph_from_trace(
    trace_path: str,
    summary: dict | None = None,
    tp_size: int = 1,
    batch_size: int = 1,
    quantization: str | None = None,
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

    roots, device_us = build_forest(events)
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
        "has_module_names": _forest_has_named_modules(roots),
        "timing_device_us": round(device_us, 6),
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
