# SPDX-License-Identifier: Apache-2.0
"""Concrete dims -> symbolic expressions.

A reconstructed shape is only useful for a *sweep* if its dimensions say what
they are (``S``, ``B``, ``H``, ``n_h/TP``, ``topk·S``) rather than what they
happened to be: the Shape Matrix and the replay benchmark re-resolve every
shape at each swept operating point, so a dim left as the integer it happened
to have stops describing the same call as soon as anything moves.

There is **one** resolution, applied to every dim in a fixed order:

1. **token** — the pass's own token count (``S`` prefill, ``B`` decode).
2. **derived** — an expression over the sweep variables, e.g. an MoE block's
   routed rows ``topk·S``. An expression, not a value, because the value moves
   with the sweep.
3. **scoped constant** — a config constant that only applies to certain ops.
   Needed because a value can mean two things at once: MiniMax-M3's index-head
   count equals ``num_kv_heads`` and its top-k block count equals ``n_h/TP``,
   so a value-keyed table alone cannot represent both.
4. **constant** — a config constant, or its ``/TP`` shard.
5. **allocation** — what remains is a run-specific allocation size (a paged
   KV-cache slot count, an MoE scratch buffer). It gets an observed-value
   symbol so nothing structural is left as a bare integer, with the value
   recorded in the legend.

Rules that need a whole shape rather than a single dim (the MoE router's
per-token expert axis) are applied first, per shape. Trivial dims (``<= 2``:
the k/v pair, a broadcast) are deliberately left concrete.
"""
from __future__ import annotations

from typing import Any, Callable

from .rules import _msa_kernel_layout

# Concrete dims at or below this value are structural constants (k/v pair,
# real/imag split, singleton broadcasts), not dimensions to symbolize.
_TRIVIAL_MAX = 2


# ===================================================================
# The symbol table
# ===================================================================

class SymbolTable:
    """Config constants, their ``/TP`` shards, and the scoped exceptions.

    ``value`` maps a concrete dim to its symbol. ``scoped`` holds the
    exceptions a value-keyed map cannot express: ``(op predicate, axis, value,
    symbol)``, consulted before ``value`` for the ops it applies to. ``legend``
    is the symbol -> value table the UI and the exports read.
    """

    def __init__(self, summary: dict, tp_size: int) -> None:
        self.tp = max(1, int(tp_size or 1))
        self.value: dict[int, str] = {}
        self.legend: dict[str, int] = {}
        #: ``(applies, axis, ndim, value, symbol)`` - see :meth:`add_scoped`.
        self.scoped: list[
            tuple[Callable[[str], bool], int, int, int, str]] = []
        #: ``(applies, axis, ndim)`` - axes that carry the *token* dim for
        #: certain ops even though the value collides with a config constant.
        self.token_axes: list[tuple[Callable[[str], bool], int, int]] = []
        self._build(summary)
        self.legend["TP"] = self.tp

    # -- construction ------------------------------------------------
    def add(self, sym: str, val: int | None, shard: bool = True) -> None:
        """Register a config constant, and its per-rank shard when it splits."""
        if not val or val <= 0:
            return
        self.legend.setdefault(sym, val)
        self.value.setdefault(val, sym)
        if shard and self.tp > 1 and val % self.tp == 0:
            self.value.setdefault(val // self.tp, f"{sym}/TP")

    def add_scoped(self, sym: str, val: int | None, axis: int, ndim: int,
                   applies: Callable[[str], bool], shard: bool = True) -> None:
        """Register a constant that only holds for the ops ``applies`` matches.

        ``axis``/``ndim`` are the position and rank it occupies in those ops'
        shapes (``-1`` for any), which is what keeps the entry from claiming a
        same-valued dim elsewhere in the same op. Scoped entries are why a
        colliding value can still be named: the same ``16`` is ``K_topk`` on an
        indexer top-k tensor and ``n_h/TP`` everywhere else.
        """
        if not val or val <= 0:
            return
        self.legend.setdefault(sym, val)
        per_rank = val
        if shard and self.tp > 1 and val % self.tp == 0:
            per_rank = val // self.tp
        name = f"{sym}/TP" if per_rank != val else sym
        self.scoped.append((applies, axis, ndim, per_rank, name))

    def _build(self, summary: dict) -> None:
        H = summary.get("hidden_size")
        n_h = summary.get("num_heads")
        n_kv = summary.get("num_kv_heads", n_h)
        d = summary.get("head_dim")
        if H and n_h and not d:
            d = H // n_h
        inter = summary.get("intermediate_size")

        self.add("H", H)
        self.add("n_h", n_h)
        if n_kv and n_kv != n_h:
            self.add("n_kv", n_kv)
        self.add("d", d)
        self.add("I", inter)
        self.add("V", summary.get("vocab_size"))
        if n_h and d:
            self.add("n_h·d", n_h * d)
            self.add("QKV", (n_h + 2 * (n_kv or n_h)) * d)
        if inter:
            self.add("2·I", 2 * inter)
        self.add("E", summary.get("num_experts"))
        if summary.get("moe_intermediate_size"):
            self.add("I_moe", summary["moe_intermediate_size"])
            # The MoE gate_up projection is the two halves fused, exactly like
            # the dense ``2·I``; without this the width falls through to an
            # observed-value symbol and reads as a meaningless ``N``.
            self.add("2·I_moe", 2 * summary["moe_intermediate_size"])
        # The rope cos/sin cache length (``[max_position, rotary_dim]``) is
        # replicated per rank, so it has no ``/TP`` shard.
        self.add("P", summary.get("max_position_embeddings"), shard=False)

        # DeepSeek / MiniMax-M3 sparse attention fuses the lightning indexer's
        # query and key projections into qkv_proj, so its output width is the
        # plain QKV plus 2*(n_index_heads*index_dim).
        if summary.get("sparse_attention") and n_h and d:
            n_idx = summary.get("sparse_num_index_heads")
            idx_d = summary.get("sparse_index_dim")
            if n_idx and idx_d:
                qkv = (n_h + 2 * (n_kv or n_h)) * d
                self.add("QKV_idx", qkv + 2 * n_idx * idx_d)
            # The indexer's own dims collide with unrelated model dims on M3
            # (index heads == num_kv_heads; top-k blocks == n_h/TP at TP=4), and
            # a top-k block count rendered as ``n_h/TP`` would then wrongly
            # *scale with TP* when the matrix sweeps it. Scope them to the
            # indexer kernels, where they are unambiguous.
            self.add_scoped("n_idx", n_idx, axis=1, ndim=3,
                            applies=_is_index_kernel)
            self.add_scoped("n_idx", n_idx, axis=0, ndim=3,
                            applies=_is_topk_kernel)
            self.add_scoped("K_topk", summary.get("sparse_topk_blocks"),
                            axis=2, ndim=3, applies=_is_topk_kernel,
                            shard=False)
            # The indexer's top-k tensors are token-major on their *second*
            # axis, where the token count can equal a config dim (S = 2048 =
            # n_h*d/TP) and would otherwise lose its S/B symbol.
            self.token_axes.append((_is_topk_kernel, 1, 3))
            # The block-sparse attend kernel takes the indexer's selection as
            # ``[n_idx/TP, tokens, K_topk]`` alongside its own
            # ``[tokens, n_h/TP, d]`` query, so every one of those three dims
            # collides with a different model constant.
            self.add_scoped("n_idx", n_idx, axis=0, ndim=3,
                            applies=_is_attn_kernel)
            self.add_scoped("K_topk", summary.get("sparse_topk_blocks"),
                            axis=2, ndim=3, applies=_is_attn_kernel,
                            shard=False)
            self.token_axes.append((_is_attn_kernel, 1, 3))
        if summary.get("num_experts") and \
                int(summary.get("num_experts_per_tok") or 0) >= 2:
            self.legend.setdefault("topk", int(summary["num_experts_per_tok"]))

    # -- resolution --------------------------------------------------
    def scoped_symbol(self, op_name: str, axis: int, ndim: int,
                      dim: int) -> str | None:
        for applies, want_axis, want_ndim, value, sym in self.scoped:
            if value == dim and (want_axis < 0 or want_axis == axis) \
                    and (want_ndim < 0 or want_ndim == ndim) \
                    and applies(op_name):
                return sym
        return None

    def is_token_axis(self, op_name: str, axis: int, ndim: int) -> bool:
        return any(a == axis and n == ndim and applies(op_name)
                   for applies, a, n in self.token_axes)


def _is_index_kernel(op_name: str) -> bool:
    return _msa_kernel_layout(op_name) == "index"


def _is_topk_kernel(op_name: str) -> bool:
    return _msa_kernel_layout(op_name) == "topk"


def _is_attn_kernel(op_name: str) -> bool:
    return _msa_kernel_layout(op_name) == "attn"


#: Ops of an MoE block, where a dim may be the *routed* row count
#: ``tokens x topk``. The expression is scoped to them because that product
#: collides with ordinary model dims: at S=2048/topk=4 it is 8192 = ``n_h*d``,
#: and at B=32/topk=4 it is 128 = ``d``. Claiming it globally would rename an
#: attention head dim after the router.
_MOE_OP_SUBSTRINGS = ("moe", "expert", "silu_and_mul", "swiglu",
                      "remap_hidden_states", "grouped_gemm", "topk_")


def _is_moe_op(op_name: str) -> bool:
    low = op_name.lower()
    return any(s in low for s in _MOE_OP_SUBSTRINGS)


# ===================================================================
# Allocation families
# ===================================================================
# What no config explains is a run-specific *allocation*: how many paged
# KV-cache slots the engine reserved, how big the MoE block-align scratch is.
# Those are still not integers a sweep may freeze and silently reuse, so each
# distinct value gets a stable observed-value symbol whose number lives in the
# legend. The family only decides the symbol's *name*, so a reader can tell a
# KV-cache slot count from an MoE buffer; the first matching substring wins.
_ALLOCATION_FAMILIES: tuple[tuple[tuple[str, ...], str], ...] = (
    # Probed first so it isn't absorbed by the MoE family via ``topk``.
    (("index_topk", "topk_index"), "K_topk"),
    (("kv_insert", "reshape_and_cache", "kv_cache", "paged_attention",
      "block_table", "kv_update"), "N_kv"),
    (("moe", "expert", "silu_and_mul", "fused_moe", "topk_bias"), "M_moe"),
)


def _allocation_base(op_name: str, ndim: int) -> str:
    """The observed-symbol base for a run-specific allocation dim."""
    low = op_name.lower()
    for subs, base in _ALLOCATION_FAMILIES:
        if any(s in low for s in subs):
            # A 1-D MoE buffer is block-align scratch, not an expert-GEMM row
            # count, so it gets its own base rather than a suffixed M_moe.
            if base == "M_moe" and ndim < 2:
                return "N_moe"
            return base
    return "N"


# ===================================================================
# The pass
# ===================================================================

def symbolize_trees(trees: list[tuple[dict | None, str, int]],
                    table: SymbolTable, summary: dict) -> None:
    """Rewrite every op shape of every phase tree — steps 1-4.

    ``trees`` is ``[(tree, token_symbol, token_value), ...]``: one entry per
    phase, since the token dim is what distinguishes them.
    """
    topk = int(summary.get("num_experts_per_tok") or 0)
    if not summary.get("num_experts"):
        topk = 0
    for tree, token_sym, token_val in trees:
        for op in _iter_ops(tree):
            name = op.get("name", "")
            for shape in (op.get("input_shapes") or []):
                if isinstance(shape, list):
                    _resolve_shape(shape, name, table, token_sym,
                                   int(token_val or 0), topk)
            out = op.get("output_shape")
            if isinstance(out, list):
                _resolve_shape(out, name, table, token_sym,
                               int(token_val or 0), topk)


def symbolize_allocations(trees: list[dict | None],
                          legend: dict[str, int]) -> None:
    """Step 5: what no rule explained is a run-specific allocation size.

    A separate entry point because the attention KV annotation runs between the
    two: it rewrites the attention key/value rows to the length actually
    attended, and those rows must not have been claimed as an allocation first.
    It is also global — a value can only be given a stable symbol once every
    phase has been seen, or the same buffer would be ``N`` in prefill and
    ``N2`` in decode.
    """
    per_base: dict[str, set[int]] = {}
    for tree in trees:
        for op in _iter_ops(tree):
            for shape in (op.get("input_shapes") or []):
                if not isinstance(shape, list):
                    continue
                for dim in shape:
                    if isinstance(dim, int) and dim > _TRIVIAL_MAX:
                        base = _allocation_base(op.get("name", ""), len(shape))
                        per_base.setdefault(base, set()).add(dim)
    if not per_base:
        return

    # Largest value keeps the bare base; further distinct values are suffixed,
    # so the naming is stable across runs over the same shape set.
    assigned: dict[tuple[str, int], str] = {}
    for base, values in per_base.items():
        for i, val in enumerate(sorted(values, reverse=True)):
            sym = base if i == 0 else f"{base}{i + 1}"
            assigned[(base, val)] = sym
            legend.setdefault(sym, val)

    for tree in trees:
        for op in _iter_ops(tree):
            name = op.get("name", "")
            shapes = list(op.get("input_shapes") or [])
            if isinstance(op.get("output_shape"), list):
                shapes.append(op["output_shape"])
            for shape in shapes:
                for j, dim in enumerate(shape):
                    if isinstance(dim, int) and dim > _TRIVIAL_MAX:
                        key = (_allocation_base(name, len(shape)), dim)
                        if key in assigned:
                            shape[j] = assigned[key]


def _resolve_shape(shape: list, op_name: str, table: SymbolTable,
                   token_sym: str, token_val: int, topk: int) -> None:
    """Steps 1-4, in place, for one shape."""
    # The routed-rows expression only applies inside an MoE block (see
    # ``_MOE_OP_SUBSTRINGS``); elsewhere the same product is an ordinary dim.
    routed = (token_val * topk
              if (token_val and topk and _is_moe_op(op_name)) else 0)

    # Shape-level rule, applied first: the MoE router's outputs are
    # ``[tokens, topk]``, the only place the expert fan-out is its own axis. It
    # must win over the routed-rows rule below, because a single-token pass
    # makes ``tokens*topk == topk`` and the two become indistinguishable — and
    # calling that axis ``topk·B`` would make the expert fan-out scale with the
    # swept batch.
    router_axis = -1
    if (routed and len(shape) == 2 and shape[0] == token_val
            and shape[1] == topk):
        router_axis = 1

    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            continue
        if i == router_axis:
            shape[i] = "topk"
            continue
        # 1. the token dim. At the leading axis it always wins over a
        #    coincidental config match; later it wins only when nothing else
        #    claims the value (some tensors are token-major on axis 1).
        if token_val and dim == token_val and (
                i == 0 or dim not in table.value
                or table.is_token_axis(op_name, i, len(shape))):
            shape[i] = token_sym
            continue
        # 2. an expression over the sweep variables. Routed rows are always a
        #    *row count*, so only the leading axis of a multi-dim operand may
        #    claim them.
        if routed and dim == routed and (routed != topk
                                         or (i == 0 and len(shape) > 1)):
            shape[i] = f"topk·{token_sym}"
            continue
        # 3. a constant that only holds for this op.
        scoped = table.scoped_symbol(op_name, i, len(shape), dim)
        if scoped:
            shape[i] = scoped
            continue
        # 4. a config constant.
        if dim in table.value:
            shape[i] = table.value[dim]


def _iter_ops(node: dict | None):
    if not node:
        return
    for op in node.get("ops", []):
        yield op
    for c in node.get("children", []):
        yield from _iter_ops(c)


def _dim_is(sym: Any, value: int) -> bool:
    """True if a (possibly symbolic) shape entry is the KV-head dimension."""
    if isinstance(sym, int):
        return sym == value
    return isinstance(sym, str) and sym.split("/")[0] == "n_kv"
