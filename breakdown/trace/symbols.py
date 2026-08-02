# SPDX-License-Identifier: Apache-2.0
"""Concrete dims -> symbolic expressions.

A reconstructed shape is only useful for a *sweep* if its dimensions say what
they are (``S``, ``B``, ``H``, ``n_h/TP``, ``topk*S``) rather than what they
happened to be. This module owns the symbol tables and every pass that
rewrites a dim.
"""
from __future__ import annotations


from typing import Any
from .rules import _msa_kernel_layout


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
