# SPDX-License-Identifier: Apache-2.0
"""The one place op names mean something.

Before this module the same question was asked in eight files. "Is this a
matmul?" was answered by four separately-maintained tuples; "is this a
collective?" by two; "is this MoE compute?" by two overlapping substring lists
that had already drifted apart. Adding one op to vLLM meant finding all of
them.

The rule here is: **the data lives once, the questions stay separate.** Some of
those questions genuinely differ -- "should reconstruction list this op?",
"should the classifier call it compute?" and "should the benchmark replay it?"
are three different questions with three legitimately different answers -- so
this module does not collapse them into one predicate. It collapses the
*vocabulary* they share, and then each question is one small function over it.

Everything is a fact about a *name*, so this module stays torch-free and
import-light: the offline paths (rebuilding a graph from an uploaded trace,
producing a shape matrix on a machine with no GPU) depend on it.

Names arrive as ``namespace::base`` (``_C::rms_norm``, ``aten::mm``,
``c10d::allreduce_``) or bare. :func:`base_of` is the normalizer every
consumer used to open-code as ``name.split("::")[-1].lower()``.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def split(op_name: str | None) -> tuple[str, str]:
    """``"_C::rms_norm"`` -> ``("_C", "rms_norm")``; bare names get ``""``."""
    name = op_name or ""
    ns, sep, base = name.partition("::")
    return (ns, base) if sep else ("", name)


def base_of(op_name: str | None) -> str:
    """The op's own name, lowercased and without its namespace.

    ``"aten::_scaled_mm"`` -> ``"_scaled_mm"``. This is the key every kind
    table below is written in terms of.
    """
    return split(op_name)[1].lower()


def namespace_of(op_name: str | None) -> str:
    return split(op_name)[0]


# ---------------------------------------------------------------------------
# Namespaces: who produced this op, and how it is invoked
# ---------------------------------------------------------------------------
#: Namespaces holding vLLM's own custom kernels. The classifier used to carry
#: a hand-copied duplicate of this tuple, which is exactly the kind of copy
#: that goes stale when a namespace is added.
VLLM_KERNEL_NAMESPACES: tuple[str, ...] = (
    "_C", "_C_cache_ops", "_moe_C", "_xpu_C", "vllm")

#: The CUDA-only kernel namespace, kept separate so a CUDA trace is reported
#: as such rather than silently as XPU.
CUDA_KERNEL_NAMESPACE = "_cuda_C"

#: Collective-communication namespaces. ``classifier`` and ``bench.resolve``
#: each had their own list; ``oneccl`` was in one and ``xccl`` only in the
#: other, so the same op could be a collective to one stage and not the other.
COLLECTIVE_NAMESPACES: tuple[str, ...] = (
    "c10d", "ccl", "oneccl", "nccl", "xccl")

#: Collective *operations*, for traces that record the bare function name with
#: no namespace. Bare ``gather``/``scatter`` are deliberately absent: the cache
#: and MoE gathers are compute, not communication, and would be misfiled.
COLLECTIVE_KEYWORDS: tuple[str, ...] = (
    "all_reduce", "allreduce",
    "all_gather", "allgather",
    "reduce_scatter", "reducescatter",
    "all_to_all", "alltoall",
    "_allgather_base", "_reduce_scatter_base",
)

#: Namespaces whose ops are launched straight from Python rather than through
#: the ATen dispatcher, so they have no schema to resolve and must be replayed
#: through the launcher frame the capture recorded.
PYTHON_LAUNCHED_NAMESPACES: tuple[str, ...] = (
    "triton", "flashinfer", "flash_xpu")

#: namespace -> modules whose import registers that namespace's custom ops.
#: vLLM registers lazily, so ``torch.ops.vllm.<op>`` only exists once the layer
#: module that defines it has been imported.
REGISTRAR_MODULES: dict[str, tuple[str, ...]] = {
    "_C": ("vllm._custom_ops",),
    "_C_cache_ops": ("vllm._custom_ops",),
    "_moe_C": ("vllm._custom_ops", "vllm.model_executor.layers.fused_moe"),
    "_xpu_C": ("vllm._custom_ops", "vllm._ipex_ops"),
    "vllm": (
        "vllm._custom_ops",
        "vllm._xpu_ops",
        "vllm.attention.layer",
        "vllm.model_executor.layers.fused_moe.layer",
        "vllm.model_executor.layers.fused_moe",
        "vllm.v1.sample.ops.topk_topp_sampler",
        "vllm.distributed.parallel_state",
    ),
}

#: Kernel-library fingerprints, probed in order against the *whole* name.
#:
#: A Python-launched kernel has no ``cpu_op``, so reconstruction synthesizes an
#: op name from the device kernel symbol. FlashInfer's and xattention's symbols
#: land inside a ``triton::``-prefixed synthetic name, so they must be probed
#: before Triton or a hand-written SYCL kernel would be reported as compiled
#: Triton output.
KERNEL_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("flashinfer", "flashinfer"),
    ("flash_xpu", "flash_xpu"),
)

#: Substrings marking a Triton-compiled kernel. Case-sensitive: ``Triton`` and
#: ``CompiledFxGraph`` appear with that capitalization in torch.compile output.
TRITON_INDICATORS: tuple[str, ...] = (
    "triton_", "triton::", "Triton", "_triton_", "tt.", "CompiledFxGraph")


def is_collective(op_name: str | None) -> bool:
    """True for a tensor/pipeline-parallel communication call."""
    low = (op_name or "").lower()
    ns = namespace_of(low)
    if ns in COLLECTIVE_NAMESPACES:
        return True
    return any(kw in low for kw in COLLECTIVE_KEYWORDS)


def is_python_launched(op_name: str | None) -> bool:
    """True for an op with no dispatcher schema, launched from Python."""
    return namespace_of(op_name or "") in PYTHON_LAUNCHED_NAMESPACES


def library_of(op_name: str | None) -> str:
    """The kernel library a name fingerprints as, or ``""``.

    Returns one of ``flashinfer``, ``flash_xpu``, ``triton``.
    """
    name = op_name or ""
    low = name.lower()
    for needle, library in KERNEL_LIBRARIES:
        if needle in low:
            return library
    if any(ind in name for ind in TRITON_INDICATORS):
        return "triton"
    return ""


# ---------------------------------------------------------------------------
# Kinds: what an op does
# ---------------------------------------------------------------------------
#: The plain matrix multiplies. This set was written out *four* times -- twice
#: in trace.rules, once in trace.collapse, once in trace.phases -- so adding
#: ``aten::_int_mm`` meant four edits, and missing one was silent.
MATMUL_BASES = frozenset({"mm", "addmm", "linear", "matmul", "bmm"})

#: Matmuls plus the quantized GEMM entry points, which compute the same
#: ``2*M*K*N`` from the same two operands and differ only in operand dtype.
GEMM_BASES = MATMUL_BASES | frozenset({
    "_scaled_mm", "fp8_gemm", "fp4_gemm",
    "int4_gemm_w4a16", "int4_gemm_w4a8"})

#: Ops whose output is ``[*A.shape[:-1], B.shape[-1]]`` rather than A's shape,
#: which is what the byte count needs to size the write.
MM_OUTPUT_BASES = GEMM_BASES | frozenset({"cutlass_grouped_gemm_interface"})

#: Attention dispatch names, matched as substrings of the base name.
ATTENTION_BASES: tuple[str, ...] = (
    "unified_attention", "flash_attn", "paged_attention", "sparse_attn",
    "attention_with_output")

#: An op reaches the matrix-engine (XMX / Tensor) peak only if it issues matrix
#: instructions. Everything else -- norms, activations, gathers, collectives --
#: is bounded by the vector engine, which on Xe2 is 8x slower, so scoring them
#: against the matrix peak made every elementwise kernel look ~99 % idle.
MATRIX_OP_RE = re.compile(
    r"(^|_)((b?add)?(b)?mm|matmul|linear|gemm|einsum|attn|attention"
    r"|conv\d?d?)(_|$)")

#: MoE compute. One list, used both to scope the routed-row expression
#: ``topk*S`` and to name an MoE scratch allocation; they were two lists that
#: shared three entries and disagreed about the rest.
MOE_SUBSTRINGS: tuple[str, ...] = (
    "moe", "expert", "silu_and_mul", "swiglu", "remap_hidden_states",
    "grouped_gemm", "topk_")

#: Elementwise / reduction families and their FLOPs per element. These are
#: estimates for *relative* ranking, not vendor-published counts.
PER_ELEMENT: tuple[tuple[frozenset[str], int], ...] = (
    (frozenset({"mul", "add", "sub", "div", "relu"}), 1),
    (frozenset({"rsqrt", "sqrt", "exp", "log"}), 2),
    (frozenset({"silu", "sigmoid", "tanh", "gelu"}), 4),
    (frozenset({"softmax", "_softmax", "log_softmax"}), 5),
    (frozenset({"silu_and_mul", "mul_and_silu", "gelu_and_mul",
                "gelu_tanh_and_mul", "swigluoai_and_mul",
                "swiglustep_and_mul"}), 5),
)

#: Name *substrings* and their FLOPs per element, probed after exact names.
PER_ELEMENT_SUBSTR: tuple[tuple[str, int], ...] = (
    ("norm", 5), ("rotary", 6), ("rope", 6), ("topk", 10))

#: Ops that *index into* a large operand rather than stream it:
#: ``base -> (indexed operand, operand whose element count is the number of
#: rows touched, whether those rows are also written)``.
#:
#: This is one structure in three guises -- an embedding table indexed by token
#: ids, a rope cos/sin cache indexed by positions, and a paged KV cache indexed
#: by a block table -- and getting it wrong is not a rounding error. Charging
#: the whole operand made an embedding read the entire vocabulary matrix and
#: MiniMax-M3's block-sparse attention its whole 450 MB block pool when it
#: touches 33 blocks: "utilization 3902 % of peak", a number that says nothing
#: about the kernel and retires the op into ``check_cost_model`` instead of
#: giving it an honest roofline.
TABLE_LOOKUP_OPS: dict[str, tuple[int, int, bool]] = {
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


def is_matmul(op_name: str | None) -> bool:
    """True for a plain matrix multiply (not the quantized entry points)."""
    return base_of(op_name) in MATMUL_BASES


def is_attention(op_name: str | None) -> bool:
    base = base_of(op_name)
    return any(k in base for k in ATTENTION_BASES)


def is_moe(op_name: str | None) -> bool:
    low = (op_name or "").lower()
    return any(s in low for s in MOE_SUBSTRINGS)


def uses_matrix_engine(op_name: str | None) -> bool:
    """True if the op can reach the matrix-engine peak."""
    if not op_name:
        return False
    return bool(MATRIX_OP_RE.search(base_of(op_name).lstrip("_")))


def flops_per_element(op_name: str | None) -> int:
    """FLOPs per output element for an elementwise/reduction op, else 0."""
    base = base_of(op_name)
    for names, per in PER_ELEMENT:
        if base in names:
            return per
    for substr, per in PER_ELEMENT_SUBSTR:
        if substr in base:
            return per
    return 0


def table_lookup(op_name: str | None) -> tuple[int, int, bool] | None:
    """The indexed-operand rule for this op, or ``None``."""
    return TABLE_LOOKUP_OPS.get(base_of(op_name))


# ---------------------------------------------------------------------------
# Plumbing: ops that move or re-view tensors instead of computing
# ---------------------------------------------------------------------------
# Three stages ask about plumbing and want different answers, so the shared
# vocabulary is defined once and each stage's set is composed from it:
#
#   reconstruction  - "list this op in the graph?"   (VIEW + allocation)
#   classification  - "call this op compute?"        (prefix match, wider)
#   benchmark       - "replay this op?"              (VIEW + ALLOC + COPY)
#
#: Pure re-views: same storage, different metadata. No device work at all.
VIEW_BASES = frozenset({
    "view", "reshape", "expand", "permute", "transpose", "t", "squeeze",
    "unsqueeze", "slice", "select", "narrow", "flatten", "as_strided",
    "alias", "detach", "contiguous", "_unsafe_view", "_reshape_alias",
})

#: Tensor factories. The trace records these against an overload whose slots
#: (ScalarList sizes, ScalarType enums, Device, MemoryFormat) describe *how to
#: allocate*, not a kernel worth optimizing.
ALLOC_BASES = frozenset({
    "empty", "empty_like", "empty_strided", "new_empty", "zeros", "ones",
    "full", "zeros_like", "ones_like", "arange",
})

#: Copies and dtype conversions.
COPY_BASES = frozenset({
    "to", "_to_copy", "clone", "copy_", "pin_memory", "lift_fresh"})

#: Ops that merely re-view a *weight*. A weight is ``[out_features, H]``, i.e.
#: shaped exactly like a residual hidden state, so these must be excluded when
#: inferring a step's token count from a neighbouring ``[tokens, H]`` op.
WEIGHT_PLUMBING_BASES = frozenset({"t", "transpose", "permute", "detach"})

#: What reconstruction leaves out of the op lists, so the real compute ops are
#: not drowned. These carry no device time anyway.
PLUMBING_OPS = frozenset(
    f"aten::{b}" for b in (
        VIEW_BASES
        | {"empty", "empty_like", "empty_strided", "resize_", "split",
           "split_with_sizes", "chunk", "set_", "lift_fresh"}
    ))

#: What the benchmark refuses to replay: replaying these measures the
#: allocator and the bookkeeping, not a kernel.
SKIP_OPS = frozenset(
    [f"aten::{b}" for b in (VIEW_BASES | ALLOC_BASES | COPY_BASES
                            | {"detach_", "unbind", "item", "movedim"})]
    + ["TorchDynamo Cache Lookup"])

#: Whole-name prefixes that are compiled-region or profiler markers, not ops.
SKIP_PREFIXES: tuple[str, ...] = (
    "Torch-Compiled Region", "ProfilerStep", "Optimizer.",
    "cudaLaunch", "Memcpy", "Memset")

#: ATen ops that dispatch real accelerator compute (torch-xpu-ops / oneDNN).
#:
#: This set is a *fast path*, not a gate: an ``aten::`` op with device time is
#: treated as compute whether or not it is listed here, so the set going stale
#: costs a category label, not an op.
ATEN_COMPUTE_OPS = frozenset({
    "linear", "mm", "bmm", "addmm", "matmul", "_scaled_mm",
    "conv1d", "conv2d", "embedding",
    "layer_norm", "batch_norm", "group_norm",
    "softmax", "_softmax", "log_softmax",
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "gelu", "relu", "silu", "sigmoid", "tanh",
    "mul", "add", "sub", "div", "sum", "mean", "max", "min", "pow",
    "rsqrt", "sqrt", "exp", "log", "where",
    "index_select", "gather", "scatter", "scatter_",
    "topk", "sort", "argmax", "argmin", "cumsum", "arange",
    "fill_", "index_put_", "masked_fill_",
})

#: What the classifier calls framework overhead rather than compute. Prefixes,
#: so ``aten::empty`` also covers ``aten::empty_like``.
FRAMEWORK_PREFIXES: tuple[str, ...] = tuple(sorted(
    {"profiler::", "autograd::", "torch::autograd::", "record_function"}
    | {f"aten::{b}" for b in (
        VIEW_BASES | {"empty", "zeros", "ones", "to", "copy_", "clone",
                      "cat", "stack", "split", "chunk", "unflatten",
                      "_to_copy"})}
))


def is_aten_compute(op_name: str | None) -> bool:
    """True for an ATen op known to dispatch accelerator compute."""
    return base_of(op_name) in ATEN_COMPUTE_OPS


def is_framework(op_name: str | None) -> bool:
    """True if the op is plumbing rather than compute.

    The compute set is consulted **first**, and that ordering is the whole
    point. ``FRAMEWORK_PREFIXES`` is a shorthand -- ``aten::empty`` is meant to
    stand for ``aten::empty_like`` too -- but a prefix does not know where a
    name ends, so ``aten::t`` also swallowed ``aten::topk`` and ``aten::tanh``.
    Both are real kernels that burn real device time, and both were being
    reported as framework overhead: dropped from the ops table, from the
    backend distribution, and from the benchmark's op list. The sampler's topk
    is not a rounding error to lose.

    Rather than enumerate every suffix the shorthand was standing in for, the
    rule is stated directly: a name we know to be compute is compute, whatever
    else it happens to start with.
    """
    if is_aten_compute(op_name):
        return False
    return (op_name or "").startswith(FRAMEWORK_PREFIXES)


def is_plumbing(op_name: str | None) -> bool:
    """True if reconstruction should leave this op out of the graph."""
    return (op_name or "") in PLUMBING_OPS


def is_weight_plumbing(op_name: str | None) -> bool:
    return base_of(op_name) in WEIGHT_PLUMBING_BASES


def is_skipped(op_name: str | None) -> bool:
    """True if the benchmark should not try to replay this op."""
    name = op_name or ""
    return name in SKIP_OPS or name.startswith(SKIP_PREFIXES)


# ---------------------------------------------------------------------------
# Family tables: first matching substring wins
# ---------------------------------------------------------------------------
#: A run-specific *allocation* -- how many paged KV-cache slots the engine
#: reserved, how big the MoE block-align scratch is -- is explained by no
#: config, so it gets a stable observed-value symbol whose number lives in the
#: legend. The family only decides the symbol's *name*, so a reader can tell a
#: KV-cache slot count from an MoE buffer.
ALLOCATION_FAMILIES: tuple[tuple[tuple[str, ...], str], ...] = (
    # Probed first so it is not absorbed by the MoE family via ``topk``.
    (("index_topk", "topk_index"), "K_topk"),
    (("kv_insert", "reshape_and_cache", "kv_cache", "paged_attention",
      "block_table", "kv_update"), "N_kv"),
    (MOE_SUBSTRINGS, "M_moe"),
)

#: Tensor-layout family of a sparse-attention (MSA) kernel, for the shape
#: fallback that runs when a trace has no recorded operands. ``topk`` is probed
#: first because the XPU API name ``minimax_m3_index_topk`` also contains the
#: ``index`` family's marker.
MSA_KERNEL_LAYOUTS: tuple[tuple[tuple[str, ...], str], ...] = (
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


def first_family(op_name: str | None,
                 families: tuple[tuple[tuple[str, ...], str], ...]) -> str:
    """The label of the first family whose marker appears in the name.

    ``ALLOCATION_FAMILIES`` and ``MSA_KERNEL_LAYOUTS`` are the same shape of
    table probed the same way; this is that probe, written once. Both are
    **order-dependent** -- see the comments on each -- which is why they are
    tuples rather than dicts.
    """
    low = (op_name or "").lower()
    for markers, label in families:
        if any(m in low for m in markers):
            return label
    return ""
