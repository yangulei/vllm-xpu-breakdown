# SPDX-License-Identifier: Apache-2.0
"""Model- and backend-specific vocabulary.

Everything the reconstruction knows that is *not* derivable from a trace: the
categories a chrome trace uses, the ops that are plumbing rather than compute,
the frames vLLM runs a fused block through, and the display names a module
class maps to. Kept in one place so a new model or backend is a table entry,
and so the passes stay free of names.
"""
from __future__ import annotations




# Chrome-trace categories that carry actual *device* (GPU/XPU) work — real
# compute kernels, memcpys and memsets. Every event in one of these categories
# must be attributed to a leaf op (that is the whole point of the reconstruction:
# no device time is lost). Host-side launch-API events (``_RUNTIME_CATEGORIES``,
# below) are deliberately **not** in this set: they carry no device time, they
# only pinpoint where a kernel was launched. Historically ``cuda_runtime`` was
# lumped in here, which caused pure host bookkeeping calls that launch nothing
# (``cudaEventQuery``, ``cudaStreamWaitEvent``, ``cudaDeviceGetAttribute``,
# ``cudaStreamIsCapturing``, ``cudaEventRecord``) to be mistaken for kernel
# launches and surfaced as bogus ``triton::cudaEventQuery`` leaf ops.
_DEVICE_KERNEL_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "xpu_op",
                             "gpu_op", "cuda_op", "gpu_kernel"}


# Host-side launch-API event categories (``cudaLaunchKernelExC``,
# ``cuLaunchKernelEx``, ``urEnqueueKernelLaunch``, ...). These correlate to a
# device kernel and are used to locate its launch site; they are not surfaced as
# ops themselves when the kernel they launch is already surfaced. ``cuda_driver``
# is the CUDA *driver* launch API (``cuLaunchKernel``/``cuLaunchKernelEx``) that
# Triton uses directly — without it, Triton kernels (e.g. the MoE
# ``fused_moe_kernel`` grouped GEMM) have no runtime-API launch event, so
# launch-site lookup falls back to ``External id`` and misattributes them to the
# enclosing custom op's start (collapsing all expert GEMM time into
# ``vllm::moe_forward_shared`` instead of the ``moe`` node).
_RUNTIME_CATEGORIES = {"cuda_runtime", "cuda_driver", "xpu_runtime"}


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


# Module display names that are valid semantic roles for their contained ops.
# When a module's resolved name (from ref_tree or heuristic) is in this set,
# all ops inside it inherit this role — overriding path-based inference that
# can be wrong due to GPU async timing causing incorrect time-containment.
_KNOWN_MODULE_ROLES = frozenset({
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
    "embedding", "lm_head", "norm", "q_norm", "k_norm",
    "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
})


def _clean_kernel_name(name: str) -> str:
    """Strip the cutlass/functor object-repr + tensor-ptr tail from a raw
    device-kernel symbol so synthetic op names stay readable.

    e.g. ``kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128...___T_0``
    → ``kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel``.
    """
    idx = name.find("_object_at")
    if idx > 0:
        name = name[:idx]
    return name


def _synthetic_op_label(name: str, api_name: str | None = None) -> str:
    """Namespaced display label for a Python-direct device kernel (no cpu_op).

    The namespace comes from the *kernel symbol*, which is a fact about where
    the kernel was compiled: a MiniMax-M3 MSA xattention SYCL kernel reads
    ``flash_xpu::`` and a FlashInfer cutlass kernel ``flashinfer::`` rather than
    ``triton::``, which would misrepresent them as Triton-compiled. Everything
    else keeps ``triton::``.

    For the two extension backends the launching API name is used in place of
    the raw symbol (``flash_xpu::(anonymous namespace)::index_score_kernel_t``
    → ``flash_xpu::minimax_m3_index_score``), since the raw functor symbol is
    unreadable. A Triton kernel's own symbol is already its name, so it is kept
    as-is; its launcher is recorded separately on the op for replay.
    """
    clean = _clean_kernel_name(name)
    low = clean.lower()
    if "flash_xpu" in low:
        return "flash_xpu::" + (api_name or clean)
    if "flashinfer" in low:
        return "flashinfer::" + (api_name or clean)
    return "triton::" + clean


# Residual-stream ops whose shape/dtype the trace does not record on the op
# itself: tensor-parallel collectives (``c10d::allreduce_`` records its tensor
# as a ``TensorList`` with no element dtype) and the RMSNorm/LayerNorm kernels
# that vLLM launches straight from Python via Triton/FlashInfer (no ``cpu_op``,
# so they surface as synthetic ``triton::``/``flashinfer::`` ops with no shape
# at all). All of them operate on the residual hidden state ``[tokens, H]`` in
# the model's activation dtype, so their shape/dtype can be recovered from the
# nearest neighbouring op that *does* carry a hidden-state tensor.
_COLLECTIVE_KEYWORDS = (
    "allreduce", "all_reduce", "allgather", "all_gather",
    "reduce_scatter", "reducescatter", "all_to_all", "alltoall",
)


def _is_hidden_state_op(label: str) -> bool:
    """True for a residual-stream op (norm / collective) worth inferring shapes.

    Restricted to norm-family and collective ops so attention kernels, ``zeros``,
    ``item`` and other genuinely shape-less / differently-shaped ops are left
    untouched (they must not inherit the ``[tokens, H]`` residual shape).
    """
    low = label.lower()
    if ("rmsnorm" in low or "rms_norm" in low
            or "layernorm" in low or "layer_norm" in low):
        return True
    return any(k in low for k in _COLLECTIVE_KEYWORDS)


# MiniMax-M3 MSA (sparse attention + lightning indexer) kernels are launched
# straight from Python with no ``cpu_op`` on **both** backends — XPU dispatches
# hand-tuned SYCL kernels from the ``xattention.py`` wrappers (surfacing as
# ``flash_xpu::<api>`` ops), CUDA dispatches ``triton.jit`` kernels from
# ``models/minimax_m3/common/ops/{sparse_attn,index_topk}.py`` (surfacing as
# ``triton::<kernel>`` ops). Either way they carry no shape at all.
#
# Their primary tensor layout is fixed by the wrapper signatures, so it can be
# rebuilt from the model config + the step's token count (``total_q`` — ``S``
# prefill / ``B`` decode):
#   ``attn``  — block-sparse GQA attend query/output ``[total_q, n_h, d]``
#   ``index`` — lightning-indexer query ``[total_q, n_idx, idx_d]``
#   ``topk``  — indexer top-k block ids ``[n_idx, total_q, topk_blocks]``
# Matching is by substring on the op's base name so it is device-agnostic;
# ``topk`` is probed first because the XPU API name ``minimax_m3_index_topk``
# also contains the ``index`` family's prefix.
_MSA_KERNEL_LAYOUTS: tuple[tuple[tuple[str, ...], str], ...] = (
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


# Ops that merely re-view a *weight* tensor. A weight is ``[out_features, H]``,
# i.e. shaped exactly like a residual hidden state, so these must be excluded
# when inferring a step's token count from a neighbouring ``[tokens, H]`` op.
_WEIGHT_PLUMBING_OPS = {"t", "transpose", "permute", "detach"}


def _msa_kernel_layout(op_label: str) -> str | None:
    """Return the MSA primary-tensor layout for an op label, if it is one."""
    base = op_label.split("::")[-1].lower()
    for subs, layout in _MSA_KERNEL_LAYOUTS:
        if any(s in base for s in subs):
            return layout
    return None


# vLLM V1 runs the sampler (and similar post-processing) functionally rather
# than as an ``nn.Module``, so the profiler emits its top-level call as a
# source-located ``python_function`` frame (e.g. ".../sample/sampler.py(72):
# __call__") instead of ``nn.Module: Sampler``. Without a module boundary the
# sampler's ops become bare op roots and get dropped by ``_partition_steps``
# (which keeps only module roots), so the reconstructed tree stops at
# ``LogitsProcessor``. Map the sampler's ``__call__`` frame to a synthetic
# ``Sampler`` module so its ops attach to a proper node.
#
# The same trick surfaces the MoE routing and expert compute. vLLM dispatches
# the whole MoE block as one fused custom op (``vllm::moe_forward_shared``)
# whose Python body calls the router (``fused_topk_bias`` — sigmoid/topk/gather)
# and the routed-expert kernels as plain functions, not ``nn.Module`` forwards.
# The expert compute has a per-backend entry frame: XPU dispatches it through
# ``xpu_fused_moe`` (``fused_moe_interface.py`` — grouped GEMM/remap/gather),
# CUDA through the Triton modular kernel ``apply`` (``experts/triton_moe.py`` —
# ``moe_align_block_size`` → ``fused_moe_kernel`` grouped GEMM → activation →
# ``moe_sum``). Without a boundary their ops and kernels collapse into the single
# ``moe_forward_shared`` op node, so the ``FusedMoE`` graph showed neither the
# router nor the experts (only the hoisted ``shared_experts`` MLP). Promoting the
# frames to synthetic modules makes ``FusedMoE`` read
# ``shared_experts → router → moe → reduce``; each is then hoisted out of the
# wrapping op by ``_hoist_modules_under_ops``.
#
# It also groups the fused all-reduce + RMSNorm. Gemma-style models (MiniMax-M3)
# fuse the residual tensor-parallel all-reduce with the following RMSNorm as
# ``fused_allreduce_gemma_rms_norm``, a ``python_function`` that wraps both the
# ``c10d::allreduce_`` op **and** the ``MiniMAXGemmaRMSNorm`` module. Without a
# boundary the all-reduce and the norm float up as two unrelated siblings of the
# decoder layer (a bare ``c10d::allreduce_`` op next to a ``post_attention_-
# layernorm`` norm), which reads as an unexplained "norm" at the layer edge.
# Promoting the frame makes it a parent node ``fused_allreduce_gemma_rms_norm →
# {allreduce, norm}`` so the fusion is explicit.
#
# ``(path_substr, funcname, synthetic_class, display_name)`` — ``display_name``
# is the attribute-style label shown in the graph (``None`` → derive from the
# class). Only the outermost matching frame becomes a boundary per step.
#
# Why a table and not a structural rule. The obvious generalisation — "a Python
# frame that encloses compute no child module covers becomes a module" — was
# measured on the canonical MiniMax-M3 TP4 trace: of 91 900 candidate frames on
# the worker thread, 409 are inside a module and enclose two or more leaves, and
# they nest (``tensor_model_parallel_all_reduce`` → ``all_reduce`` →
# ``all_reduce`` → ``all_reduce`` is four of them around one collective). A rule
# with that hit rate does not produce a readable tree; it produces a deeper one.
# What the entries below actually encode is a *semantic* judgement — this
# function is a block of the model — which is model vocabulary, so it lives here
# with the rest of it. Adding a model means adding a line, not a pass.
_FUNCTIONAL_MODULE_FRAMES = (
    ("sample/sampler.py", "__call__", "Sampler", None),
    ("fused_topk_bias_router.py", "fused_topk_bias", "FusedTopKBiasRouter",
     "router"),
    ("fused_moe_interface.py", "xpu_fused_moe", "XpuFusedMoE", "moe"),
    ("experts/triton_moe.py", "apply", "TritonExperts", "moe"),
    ("fused_allreduce_gemma_rms_norm.py", "fused_allreduce_gemma_rms_norm",
     "FusedAllreduceGemmaRMSNorm", "fused_allreduce_gemma_rms_norm"),
)


def _functional_module_class(name: str) -> tuple[str, str | None] | None:
    """Return ``(class, display_name)`` for a functional (non-nn.Module) frame.

    ``name`` is a torch-profiler python_function label of the form
    ``"<path>(<lineno>): <func>"``. Returns the mapped ``(synthetic_class,
    display_name)`` when the frame is a recognised functional module boundary
    (``display_name`` may be ``None`` to derive it from the class), else
    ``None``.
    """
    head, sep, func = name.partition("): ")
    if not sep:
        return None
    path = head.rsplit("(", 1)[0]
    for path_substr, funcname, cls, display in _FUNCTIONAL_MODULE_FRAMES:
        if funcname == func and path_substr in path:
            return cls, display
    return None


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


def _rowparallel_shape_role(cls: str, child_merged: dict,
                            symbols_val: dict[int, str]) -> str | None:
    """Disambiguate a RowParallelLinear as o_proj vs down_proj by shape.

    Both the attention output projection (``o_proj``) and the MLP/MoE down
    projection (``down_proj``) are ``RowParallelLinear``, so class name alone is
    ambiguous. Parent-based heuristics are unreliable on GPU where async timing
    corrupts the trace's time-containment nesting (an attention ``o_proj`` can
    end up nested under an MoE block, or vice-versa). The projection's matmul
    input feature dimension is unambiguous instead:

    * ``o_proj``  input feature ≈ ``n_h·d`` (attention hidden, ~``H``)
    * ``down_proj`` input feature = ``intermediate`` (``I`` / ``I_moe``)

    We read the matmul's input feature dim, map it to a known symbol, and pick
    the role accordingly. Returns ``None`` when the class isn't RowParallelLinear
    or the shape can't be resolved to a known symbol (caller then falls back to
    the parent heuristic).
    """
    if "rowparallel" not in cls.lower():
        return None
    for sig in child_merged["op_order"]:
        raw = child_merged["op_groups"][sig]["raw"]
        base = raw.label.split("::")[-1].lower()
        if base not in ("mm", "addmm", "linear", "matmul"):
            continue
        # Activation input: [M, K]; for addmm it's the 2nd arg.
        act = raw.shapes[1] if base == "addmm" and len(raw.shapes) > 1 \
            else (raw.shapes[0] if raw.shapes else None)
        if not act:
            continue
        k = act[-1]
        sym = symbols_val.get(k, "")
        # Strip a trailing "/TP" so per-rank shards match the base symbol.
        base_sym = sym.split("/")[0] if sym else ""
        if base_sym in ("I", "I_moe", "2·I"):
            return "down_proj"
        if base_sym in ("H", "n_h·d", "QKV"):
            return "o_proj"
    return None


def _disambiguate_child_name(cls: str, occ_idx: int, parent_merged: dict) -> str:
    """Generate a display name for a child module, disambiguating by position.

    When multiple children share the same class (e.g. two RMSNorm inside
    Attention → q_norm and k_norm), use positional heuristics to distinguish
    them instead of showing the same generic name for both.

    This is only reached for a module the capture-time spans did not name, so
    it is a fallback for archived and third-party traces. Every rule here is
    device-agnostic: a naming ambiguity is a property of the model's class
    reuse, not of the runtime.
    """
    parent_type = parent_merged.get("module_type", "").lower()
    low = cls.lower()

    # Norm modules inside Attention: first = q_norm, second = k_norm
    if ("norm" in low) and ("attention" in parent_type or "attn" in parent_type):
        # Count how many same-class norm siblings exist
        norm_count = sum(
            1 for key in parent_merged["child_order"]
            if "norm" in key[0].lower()
        )
        if norm_count >= 2:
            if occ_idx == 0:
                return "q_norm"
            elif occ_idx == 1:
                return "k_norm"

    # Norm modules inside DecoderLayer: first = input_layernorm,
    # second = post_attention_layernorm
    if ("norm" in low) and ("layer" in parent_type or "decoder" in parent_type):
        norm_count = sum(
            1 for key in parent_merged["child_order"]
            if "norm" in key[0].lower()
        )
        if norm_count >= 2:
            if occ_idx == 0:
                return "input_layernorm"
            elif occ_idx == 1:
                return "post_attention_layernorm"
            elif occ_idx == 2:
                return "pre_feedforward_layernorm"

    # Linear projections: RowParallelLinear is the attention output (o_proj) but
    # also the MLP/MoE down projection (down_proj) — both share the class, so the
    # generic ``_module_display_name`` (which maps RowParallelLinear → o_proj)
    # mislabels the MLP one as o_proj whenever the reference-name overlay fails
    # to tag it. Disambiguate by the parent module: a RowParallelLinear inside an
    # MLP/expert/feedforward module is always the down projection; inside
    # attention it's the output projection. This is **device-agnostic** — the
    # reference-name overlay can miss modules for reasons unrelated to CUDA async
    # timing. On XPU the shared_experts MLP is hoisted out of the fused
    # ``moe_forward_shared`` op (``_hoist_modules_under_ops``), so it sits under
    # ``FusedMoE`` while the reference tree lists it under ``MoE.shared_experts``;
    # alignment can't match it, leaving its ``down_proj`` unnamed. Without this
    # the shared expert's down projection read as ``o_proj`` even though the dense
    # MLP's (overlay-named) down_proj was correct.
    is_mlp = ("mlp" in parent_type or "moe" in parent_type
              or "expert" in parent_type or "feedforward" in parent_type)
    is_attn = "attention" in parent_type or "attn" in parent_type
    if "rowparallel" in low:
        if is_mlp:
            # Count RowParallelLinear siblings in this parent
            row_parallel_count = sum(
                1 for key in parent_merged["child_order"]
                if "rowparallel" in key[0].lower()
            )
            if row_parallel_count <= 1:
                return "down_proj"
            # Several RowParallelLinear inside one MLP: only the last is the
            # down projection; the earlier ones are attention output
            # projections that async timing time-contained here. An accurate
            # capture hits the count<=1 branch above, so this only fires where
            # the nesting is already known to be wrong.
            last_rp_idx = max(
                key[1] for key in parent_merged["child_order"]
                if "rowparallel" in key[0].lower()
            )
            return "down_proj" if occ_idx == last_rp_idx else "o_proj"
        if is_attn:
            return "o_proj"
    if "mergedcolumn" in low or "columnparallel" in low:
        if is_mlp:
            return "gate_up_proj"

    return _module_display_name(cls)


_ATTENTION_OP_NAMES = frozenset({
    "vllm::unified_attention_with_output",
    "vllm::unified_attention",
})


def _is_attention_op(op: dict) -> bool:
    name = op.get("name", "")
    if name in _ATTENTION_OP_NAMES:
        return True
    low = name.lower()
    return ("attention" in low or "flash_attn" in low
            or op.get("role") == "attention")
