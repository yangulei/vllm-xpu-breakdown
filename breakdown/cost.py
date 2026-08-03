# SPDX-License-Identifier: Apache-2.0
"""The one cost model: bytes, FLOPs, arithmetic intensity, and the roofline.

Every stage of the pipeline needs the same three numbers for an op — how many
bytes it moves, how many FLOPs it does, and what that implies against the
hardware — and they must be the *same* numbers, or the graph's ``ai`` column
and the benchmark's roofline verdict describe different kernels. They used to
be computed in three places (``analyzer``, ``shape_derive`` and
``bench/estimate``) with three different sets of rules.

Four rules earn their keep here, each because getting it wrong produced a
number that said nothing about the kernel:

* **A table-lookup op is charged for the rows it reads.** Charging an embedding
  for the whole vocabulary matrix, or a RoPE call for the whole
  ``[max_position, head_dim]`` cache, produced "utilization 37000 % of peak".
* **An empty operand contributes 0 bytes rather than zeroing the estimate.**
  vLLM dispatches attention with an empty ``kv_cache_dummy_dep`` purely to
  order it against the KV write.
* **Attention has an explicit FLOPs model.** Without it the heaviest op in the
  profile had zero analytic work, so its bound came out ``unknown`` and it
  ranked as if it had 100 % headroom.
* **A per-tensor dtype is used when the trace recorded one**, so an fp8 weight
  counts one byte while its bf16 activation counts two.

The roofline helpers here serve a second purpose: turning an op's analytic
work (FLOPs / bytes) into the *lower bound* a measurement should respect. A
replayed latency below the roofline bound means the replay did not do the work
(an early-exit kernel, an empty index map), which the report flags.

Two properties of that roofline are deliberate:

* **The bound comes from arithmetic intensity, not from the measurement.** An
  op is compute-bound iff its AI (FLOP/byte) is at or above the machine balance
  ``peak FLOPS / peak bandwidth``. Comparing the two achieved utilizations and
  taking the larger - the previous rule - labels a GEMM that ran at 30 % of
  peak FLOPS "memory-bound" and a pure-gather kernel "compute-bound".
* **A cache-resident op is measured against cache bandwidth.** The benchmark
  repeats a kernel on the same operands inside one timed window, so an op whose
  footprint fits in the last-level cache legitimately exceeds the DRAM peak.
  Charging it to DRAM produced "utilization 300 % of peak" warnings that said
  nothing about the kernel; the honest roof is the cache one.

Two refinements make the answer name a *hardware unit* rather than a category:

* **The compute roof is the unit the op can actually issue to.** Only
  matrix-family ops (GEMM, attention, convolution) reach the XMX / Tensor peak;
  an RMSNorm or a gather issues vector instructions and is bounded by the XVE /
  CUDA-core peak, which on Xe2 is 8x lower. Charging every op to XMX made all
  the elementwise kernels look like they had ~99 % headroom. So the roof is
  reported as ``XMX`` / ``XVE`` / ``DRAM`` / ``L3-Cache``.
* **A cache-resident op is also scored against DRAM.** Ops whose footprint fits
  in the last-level cache are usually already-optimal streaming kernels; scored
  only against the (much higher) cache bandwidth they appear to have large
  headroom that does not exist. :func:`roofline_detail` therefore reports both
  utilizations and an ``effective_util`` = max of the two, which is what the
  ranking's headroom uses, while the credibility check still uses the cache
  number so a cache-resident kernel is not flagged as "above peak".
"""
from __future__ import annotations

import re
from typing import Any

from .core.dtypes import size as _element_bytes

def dtype_size(dtype_str: str) -> float:
    """Bytes per element for a dtype string (default bf16).

    The table lives in :mod:`breakdown.core.dtypes`; this is the cost model's
    name for it. Fractional for sub-byte packed types, so an int4 weight is
    counted at the 0.5 bytes/element it actually occupies.
    """
    return _element_bytes(dtype_str)


def _prod(shape: Any) -> int:
    """Product of a shape's *concrete* dims; 0 if any dim is symbolic."""
    p = 1
    for d in shape or ():
        if isinstance(d, int):
            p *= d
        else:
            return 0
    return p


def _prod_loose(shape: Any) -> int:
    """Product of a shape's concrete dims, ignoring symbolic ones."""
    p = 1
    for d in shape or ():
        if isinstance(d, int):
            p *= d
    return p


# ===================================================================
# Bytes
# ===================================================================

_MM_BASES = frozenset({
    "mm", "addmm", "linear", "matmul", "bmm", "_scaled_mm", "fp8_gemm",
    "fp4_gemm", "int4_gemm_w4a16", "int4_gemm_w4a8",
    "cutlass_grouped_gemm_interface",
})

#: Ops that *index into a large operand* rather than stream it:
#: ``base -> (indexed operand, operand whose element count is the number of
#: rows touched, whether those rows are also written)``.
#:
#: This is the same structure in three guises - an embedding table indexed by
#: token ids, a rope cos/sin cache indexed by positions, and a **paged KV
#: cache** indexed by a block table - and getting it wrong is not a rounding
#: error. Charging the whole operand made an embedding read the entire
#: vocabulary matrix, a rope call the entire ``[max_position, head_dim]``
#: cache, and MiniMax-M3's block-sparse attention its whole 450 MB block pool
#: when it touches 33 blocks: "utilization 3902 % of peak", a number that says
#: nothing about the kernel and retires the op into ``check_cost_model``
#: instead of giving it an honest roofline.
_TABLE_LOOKUP_OPS: dict[str, tuple[int, int, bool]] = {
    "embedding": (0, 1, True),          # (weight [V, H], indices [T])
    "rotary_embedding": (3, 0, True),   # (cos_sin_cache [P, d], positions [T])
    # The fused qk-norm + rope + KV insert carries the same rope cache,
    # indexed by the same positions.
    "fused_minimax_m3_qknorm_rope_kv_insert": (3, 4, True),
    # Paged caches, indexed by a block table: read only, and the output is
    # already one of the operands.
    "minimax_m3_index_score": (1, 2, False),
    "minimax_m3_index_decode": (1, 2, False),
    "minimax_m3_sparse_attn": (1, 3, False),
    "minimax_m3_sparse_attn_decode": (1, 3, False),
}


def _tensor_bytes(shape: Any, index: int, dtypes: list[str] | None,
                  act_bytes: int) -> int:
    n = _prod(shape)
    if n <= 0:
        return 0
    if dtypes and index < len(dtypes) and dtypes[index]:
        return n * dtype_size(dtypes[index])
    return n * act_bytes


def _lookup_bytes(base: str, shapes: list, dtypes: list[str] | None,
                  act_bytes: int) -> int | None:
    """Bytes a table-lookup op really moves, or ``None`` if the rule misfits."""
    spec = _TABLE_LOOKUP_OPS.get(base)
    if spec is None:
        return None
    t_i, i_i, writes_rows = spec
    if t_i >= len(shapes) or i_i >= len(shapes):
        return None
    table, index = shapes[t_i], shapes[i_i]
    if len(table) < 2:
        return None
    rows = _prod(index)
    if rows <= 0 or rows >= (table[0] if isinstance(table[0], int) else 0):
        return None                  # touches (at least) the whole table anyway
    row_elems = _prod(table[1:])
    row_bytes = row_elems * (dtype_size(dtypes[t_i])
                             if dtypes and t_i < len(dtypes) and dtypes[t_i]
                             else act_bytes)
    total = rows * row_bytes
    for i, s in enumerate(shapes):
        if i != t_i:
            total += _tensor_bytes(s, i, dtypes, act_bytes)
    if writes_rows:
        # The gathered rows are also what gets written back.
        total += rows * row_elems * act_bytes
    return total


def op_bytes(op_name: str, shapes: list, dtypes: list[str] | None = None,
             act_bytes: int = 2) -> int:
    """Bytes an op moves: every input read once, plus its output written once.

    ``dtypes`` are the trace's recorded per-tensor dtypes, used when present;
    otherwise every tensor is sized at ``act_bytes``. Returns 0 when no shape
    is concrete.
    """
    if not shapes:
        return 0
    base = op_name.split("::")[-1].lower()
    lookup = _lookup_bytes(base, shapes, dtypes, act_bytes)
    if lookup is not None:
        return lookup
    reads = sum(_tensor_bytes(s, i, dtypes, act_bytes)
                for i, s in enumerate(shapes))
    if (base in _MM_BASES and len(shapes) >= 2
            and len(shapes[0]) >= 2 and len(shapes[1]) >= 2
            and isinstance(shapes[1][-1], int)):
        out = _prod(shapes[0][:-1]) * shapes[1][-1]
    else:
        out = _prod(shapes[0])
    return reads + out * act_bytes


# ===================================================================
# FLOPs
# ===================================================================

_ATTENTION_BASES = ("unified_attention", "flash_attn", "paged_attention",
                    "sparse_attn", "attention_with_output")


def is_attention(op_name: str) -> bool:
    base = op_name.split("::")[-1].lower()
    return any(k in base for k in _ATTENTION_BASES)


def _attention_flops(shapes: list, n_seqs: int = 1) -> int:
    """QK^T + PV for a ``[tokens, heads, head_dim]`` attention call.

    Causality is deliberately *not* discounted: a decode step attends the whole
    cached context with no masking at all, and for a prefill over a long cached
    context the masked fraction is small. Halving it would understate the op
    the ranking cares most about.
    """
    q = next((s for s in shapes if len(s) == 3), None)
    if q is None:
        return 0
    kv = next((s for s in shapes[1:] if len(s) == 3), q)
    tokens, heads, dim = q[0], q[1], q[2]
    # A batched decode reads ``batch x context`` KV rows in total, but each
    # query attends only its own context; dividing by the sequence count keeps
    # it from being charged ``batch`` times its work.
    kv_per_seq = kv[0] / max(n_seqs, 1)
    return int(2 * 2 * tokens * kv_per_seq * heads * dim)


#: Elementwise / reduction op families and their FLOPs per element.
_PER_ELEMENT: tuple[tuple[frozenset[str], int], ...] = (
    (frozenset({"mul", "add", "sub", "div", "relu"}), 1),
    (frozenset({"rsqrt", "sqrt", "exp", "log"}), 2),
    (frozenset({"silu", "sigmoid", "tanh", "gelu"}), 4),
    (frozenset({"softmax", "_softmax", "log_softmax"}), 5),
    (frozenset({"silu_and_mul", "mul_and_silu", "gelu_and_mul",
                "gelu_tanh_and_mul", "swigluoai_and_mul",
                "swiglustep_and_mul"}), 5),
)

#: Name *substrings* and their FLOPs per element, probed after the exact names.
_PER_ELEMENT_SUBSTR: tuple[tuple[str, int], ...] = (
    ("norm", 5), ("rotary", 6), ("rope", 6), ("topk", 10),
)


def op_flops(op_name: str, shapes: list, n_seqs: int = 1) -> int:
    """FLOPs an op performs.

    ``n_seqs`` is the number of independent sequences the call covers; it only
    matters for attention (see :func:`_attention_flops`).
    """
    if not shapes:
        return 0
    base = op_name.split("::")[-1].lower()

    if base in ("mm", "linear", "_scaled_mm", "fp8_gemm", "fp4_gemm",
                "int4_gemm_w4a16", "int4_gemm_w4a8"):
        if len(shapes) >= 2 and len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
            M = _prod_loose(shapes[0][:-1])
            K = shapes[0][-1] if isinstance(shapes[0][-1], int) else 1
            N = shapes[1][-1] if isinstance(shapes[1][-1], int) else 1
            return 2 * M * K * N
    if base == "bmm":
        if len(shapes) >= 2 and len(shapes[0]) >= 3 and len(shapes[1]) >= 3:
            dims = [d if isinstance(d, int) else 1
                    for d in (shapes[0][0], shapes[0][1], shapes[0][2],
                              shapes[1][2])]
            return 2 * dims[0] * dims[1] * dims[2] * dims[3]
    if base == "addmm":
        if len(shapes) >= 3 and len(shapes[1]) >= 2 and len(shapes[2]) >= 2:
            M = shapes[1][0] if isinstance(shapes[1][0], int) else 1
            K = shapes[1][1] if isinstance(shapes[1][1], int) else 1
            N = shapes[2][1] if isinstance(shapes[2][1], int) else 1
            return 2 * M * K * N + M * N  # matmul + add
    if "grouped_gemm" in base:
        # A [M, K] x B [E, K, N] -> D [M, N]: every row goes through exactly
        # one expert, so the work is a plain M*K*N - the expert count
        # multiplies the *weights read*, not the arithmetic. Without this the
        # dominant kernel of an MoE model had zero FLOPs, hence an arithmetic
        # intensity of 0 and an unconditional "memory-bound" verdict.
        if len(shapes) >= 2 and len(shapes[0]) == 2 and len(shapes[1]) == 3:
            return 2 * _prod(shapes[0]) * (shapes[1][2]
                                           if isinstance(shapes[1][2], int)
                                           else 1)
    if base == "matmul":
        if len(shapes) >= 2 and shapes[0] and shapes[1]:
            K = shapes[0][-1] if isinstance(shapes[0][-1], int) else 1
            N = shapes[1][-1] if isinstance(shapes[1][-1], int) else 1
            return 2 * _prod_loose(shapes[0][:-1]) * K * N

    for names, per in _PER_ELEMENT:
        if base in names:
            return _prod_loose(shapes[0]) * per
    for substr, per in _PER_ELEMENT_SUBSTR:
        if substr in base:
            return _prod_loose(shapes[0]) * per
    if is_attention(op_name):
        return _attention_flops(shapes, n_seqs)
    return 0


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    """FLOP per byte — where an op sits relative to the machine balance."""
    return (flops / nbytes) if nbytes > 0 else 0.0


# ===================================================================
# The roofline
# ===================================================================

#: An op reaches the matrix-engine (XMX / Tensor) peak only if it issues matrix
#: instructions. Everything else - norms, activations, gathers, collectives -
#: is bounded by the vector engine, which on Xe2 is 8x slower.
_MATRIX_OP_RE = re.compile(
    r"(^|_)((b?add)?(b)?mm|matmul|linear|gemm|einsum|attn|attention"
    r"|conv\d?d?)(_|$)")


def uses_matrix_engine(op: str | None) -> bool:
    """Does this dispatch name denote a matrix-engine (GEMM-family) op?"""
    if not op:
        return False
    name = str(op).split("::")[-1].lower().lstrip("_")
    return bool(_MATRIX_OP_RE.search(name))


def compute_peak(peaks: dict[str, float],
                 op: str | None = None) -> tuple[float, str]:
    """``(peak TFLOPS, unit name)`` for the compute roof this op can reach.

    ``op is None`` keeps the matrix peak, so callers that do not know the op
    behave exactly as before.
    """
    if op is not None and not uses_matrix_engine(op):
        vec = float(peaks.get("vector_tflops") or 0)
        if vec > 0:
            return vec, str(peaks.get("vector_unit") or "vector")
    return float(peaks["tflops"]), str(peaks.get("matrix_unit") or "matrix")


def memory_unit(peaks: dict[str, float], level: str) -> str:
    """Human name of the memory roof: ``DRAM`` or the SKU's cache name."""
    if level == "cache":
        return str(peaks.get("cache_name") or "cache")
    return "DRAM"


def roof_unit(peaks: dict[str, float], bound: str, level: str,
              op: str | None = None) -> str:
    """The hardware unit that bounds this op (``XMX``/``XVE``/``DRAM``/...)."""
    if bound == "compute":
        return compute_peak(peaks, op)[1]
    if bound == "memory":
        return memory_unit(peaks, level)
    return "—"


def cache_resident(nbytes: float, peaks: dict[str, float]) -> bool:
    """Does this op's working set fit in the device's last-level cache?

    ``nbytes`` is the op's analytic traffic, which for a single replayed call is
    also its footprint (each operand is read once). The benchmark repeats a
    kernel inside one timed window on the *same* operands, so an op whose
    footprint fits in cache is served by the cache on every repetition after the
    first - and is bounded by cache bandwidth, not DRAM bandwidth.
    """
    cap = float(peaks.get("cache_bytes") or 0)
    return bool(cap) and 0 < nbytes <= cap


def effective_bw_gbs(nbytes: float, peaks: dict[str, float]
                     ) -> tuple[float, str]:
    """``(bandwidth roof GB/s, which memory level)`` for this op's footprint.

    A kernel whose operands are cache-resident routinely exceeds the DRAM peak;
    measuring it against DRAM produced "utilization 300 % of peak" warnings that
    said nothing about the kernel. The right roof for such an op is the
    last-level-cache bandwidth (see :data:`breakdown.bench.devices.SKU_PEAKS`).
    """
    cbw = float(peaks.get("cache_bw_gbs") or 0)
    if cbw and cache_resident(nbytes, peaks):
        return cbw, "cache"
    return float(peaks["bw_gbs"]), "dram"


def ridge_ai(peaks: dict[str, float], bw_gbs: float | None = None,
             op: str | None = None) -> float:
    """Machine balance in FLOP/byte: the roofline's ridge point.

    An op with a higher arithmetic intensity than this is compute-bound, one
    below it is memory-bound. That comparison - **not** "whichever utilization
    number comes out larger" - is what defines the bound. The compute term is
    the peak of the unit the op can issue to (matrix vs vector), so a vector op
    is not held to the XMX ridge it could never approach.
    """
    bw = float(bw_gbs if bw_gbs is not None else peaks["bw_gbs"])
    if bw <= 0:
        return 0.0
    return (compute_peak(peaks, op)[0] * 1e12) / (bw * 1e9)


def op_ai(flops: float, nbytes: float) -> float:
    """The op's arithmetic intensity in FLOP/byte."""
    if nbytes <= 0:
        return float("inf") if flops > 0 else 0.0
    return flops / nbytes


def bound_of(flops: float, nbytes: float, peaks: dict[str, float],
             op: str | None = None) -> tuple[str, str]:
    """``(bound, memory level)`` from the op's AI against the machine balance.

    The previous rule compared the two *utilizations* and took the larger, which
    mislabels ops systematically: a GEMM measured below peak FLOPS came out
    "memory" and a bandwidth-starved gather came out "compute". The bound is a
    property of the op and the machine, not of how well the kernel did.
    """
    bw, level = effective_bw_gbs(nbytes, peaks)
    if flops <= 0:
        return ("memory" if nbytes > 0 else "unknown"), level
    if nbytes <= 0:
        return "compute", level
    return ("compute" if op_ai(flops, nbytes) >= ridge_ai(peaks, bw, op)
            else "memory"), level


def kernel_seconds(flops: float, nbytes: float, peak_tflops: float,
                   peak_bw_gbs: float, util: float = 0.5) -> float:
    """Roofline runtime of one case at ``util`` of peak."""
    util = max(min(util, 1.0), 1e-3)
    t_c = (flops / (peak_tflops * 1e12 * util)) if flops > 0 else 0.0
    t_m = (nbytes / (peak_bw_gbs * 1e9 * util)) if nbytes > 0 else 0.0
    return max(t_c, t_m)


def roofline_bound_us(flops: float, nbytes: float, peaks: dict[str, float],
                      op: str | None = None) -> tuple[float, str]:
    """``(fastest possible microseconds, bound)`` at 100 % of peak.

    The memory term uses the roof the op's footprint actually sees (cache or
    DRAM), so a cache-resident kernel is not told it beat the speed of light,
    and the compute term uses the unit the op can issue to.
    """
    bound, _ = bound_of(flops, nbytes, peaks, op)
    bw, _ = effective_bw_gbs(nbytes, peaks)
    t_c = (flops / (compute_peak(peaks, op)[0] * 1e12)) if flops > 0 else 0.0
    t_m = (nbytes / (bw * 1e9)) if nbytes > 0 else 0.0
    return ((t_c if bound == "compute" else t_m) * 1e6), bound


def utilization(latency_us: float, flops: float, nbytes: float,
                peaks: dict[str, float],
                op: str | None = None) -> tuple[float, str]:
    """``(achieved fraction of the relevant roof, which roof)``.

    The roof is selected by the op's arithmetic intensity (see :func:`bound_of`)
    and, for a memory-bound op, by whether its footprint is cache-resident.
    """
    d = roofline_detail(latency_us, flops, nbytes, peaks, op)
    return d["util"], d["bound"]


def utilization_detail(latency_us: float, flops: float, nbytes: float,
                       peaks: dict[str, float], op: str | None = None
                       ) -> tuple[float, str, str]:
    """``(utilization, bound, memory level)`` for a measured case."""
    d = roofline_detail(latency_us, flops, nbytes, peaks, op)
    return d["util"], d["bound"], d["memory_level"]


def roofline_detail(latency_us: float, flops: float, nbytes: float,
                    peaks: dict[str, float], op: str | None = None
                    ) -> dict[str, Any]:
    """Everything the report and the ranking need about one measured case.

    Keys: ``util`` (against the roof the op actually sees), ``util_dram`` (the
    same measurement against DRAM bandwidth, for a memory-bound op),
    ``effective_util`` = max of the two, ``bound``, ``memory_level``, ``unit``
    (the hardware unit that bounds it), ``ai`` and ``ridge_ai``.

    ``effective_util`` exists because a cache-resident op scored only against
    the cache roof looks like it has headroom it does not have: such ops are
    typically already-optimal streaming kernels running at the DRAM roof. The
    ranking spends its headroom budget on ``effective_util`` while the
    credibility check keeps using ``util`` (so a cache-resident kernel is never
    flagged as "above peak").
    """
    bound, level = bound_of(flops, nbytes, peaks, op)
    ai = op_ai(flops, nbytes)
    peak_flops, unit_name = compute_peak(peaks, op)
    out: dict[str, Any] = {
        "bound": bound, "memory_level": level,
        "unit": roof_unit(peaks, bound, level, op),
        "ai": ai if ai != float("inf") else None,
        "ridge_ai": ridge_ai(peaks, effective_bw_gbs(nbytes, peaks)[0], op),
        "compute_unit": unit_name, "compute_peak_tflops": peak_flops,
        "util": 0.0, "util_dram": 0.0, "effective_util": 0.0,
    }
    if latency_us <= 0:
        return out
    secs = latency_us / 1e6
    if bound == "compute":
        peak = peak_flops * 1e12
        out["util"] = (flops / secs) / peak if peak > 0 else 0.0
        out["effective_util"] = out["util"]
        return out
    if bound == "memory":
        bw, _ = effective_bw_gbs(nbytes, peaks)
        achieved = nbytes / secs
        out["util"] = achieved / (bw * 1e9) if bw > 0 else 0.0
        dram = float(peaks.get("bw_gbs") or 0)
        out["util_dram"] = achieved / (dram * 1e9) if dram > 0 else 0.0
        out["effective_util"] = max(out["util"], out["util_dram"])
    return out



# ===================================================================
# Backwards-compatible names
# ===================================================================
# ``estimate_memory``/``estimate_flops`` are what the graph reconstruction
# calls; they are this model with no recorded dtypes.

def estimate_memory(op_name: str, shapes: list, dtype_bytes: int = 2) -> int:
    return op_bytes(op_name, shapes, None, dtype_bytes)


def estimate_flops(op_name: str, shapes: list, n_seqs: int = 1) -> int:
    return op_flops(op_name, shapes, n_seqs)
