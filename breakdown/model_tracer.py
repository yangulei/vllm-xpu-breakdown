# SPDX-License-Identifier: Apache-2.0
"""Trace-based model graph builder.

Instead of hand-writing a builder per architecture family (see
``model_graph.py``), this module derives the op graph from the **real vLLM
model** via a single ``torch.export`` symbolic trace:

  1. Instantiate the actual vLLM ``nn.Module`` offline on the ``meta`` device
     (no weights, no HF download) from a model ``config.json``.
  2. ``torch.export`` it once with the token dimension kept symbolic.
  3. Walk the exported graph, grouping ops back onto their owning modules via
     each node's ``nn_module_stack`` metadata, and reading symbolic shapes from
     ``meta['val']``.

The backend of every op is resolved from its *real* dispatched op name through
the shared ``classifier``/``registry`` — so there is a single source of truth
for backend assignment, and no hardcoded backend strings per op.

This requires PyTorch and vLLM to be importable (see README "Requirements").
On any failure the caller falls back to the legacy static builders.

Scope: dense, single-stack, TP=1, unquantized models. MoE and multimodal
models stay on the static builders — a single symbolic export does not fully
capture fused-expert (FusedMoE) or vision-tower compute, so tracing them would
silently under-count cost. The caller (``build_model_graph``) enforces this and
this module additionally validates layer homogeneity, raising on violation.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import threading
from typing import Any

from .classifier import classify_op
from .model_graph import (
    ModuleNode,
    OpNode,
    _compute_totals,
    _dtype_bytes,
    fused_moe_expert_ops,
)

# Set once per process.
_DIST_INITIALIZED = False

# Instantiation + export mutate global state (vLLM current-config, distributed
# init, and the RMSNorm monkeypatch). Serialize the whole critical section so a
# threaded Flask server can't interleave two traces and corrupt that state.
_TRACE_LOCK = threading.RLock()

# vLLM dispatch wrappers that aren't plain aten/_C ops. Map the exported target
# (without the ``vllm.`` namespace and ``.default`` overload) to a
# (display_name, backend, role) triple.
_VLLM_DISPATCH_OPS: dict[str, tuple[str, str, str]] = {
    "unified_attention_with_output": ("unified_attention", "vllm-xpu-kernels", "attention"),
    "unified_attention": ("unified_attention", "vllm-xpu-kernels", "attention"),
    "unified_kv_cache_update": ("reshape_and_cache", "vllm-xpu-kernels", "cache_store"),
    # MLA (DeepSeek-V2/V3) — latent-attention counterparts of the dense ops.
    "unified_mla_attention_with_output": ("unified_mla_attention", "vllm-xpu-kernels", "attention"),
    "unified_mla_attention": ("unified_mla_attention", "vllm-xpu-kernels", "attention"),
    "unified_mla_kv_cache_update": ("reshape_and_cache", "vllm-xpu-kernels", "cache_store"),
}

# torch.export collapses the whole FusedMoE expert path into a single opaque
# ``vllm.moe_forward*`` call (classified as framework and otherwise dropped). We
# detect it by name, drop the opaque op, and splice in the real routed/shared
# expert GEMM+activation ops via ``fused_moe_expert_ops`` (single source of truth
# shared with the static builder).
_MOE_OPAQUE_OPS = frozenset({"moe_forward", "moe_forward_shared", "fused_moe"})

# Targets that carry no real compute — excluded from the static graph (parity
# with the legacy builders, which only emit meaningful ops).
_FRAMEWORK_TARGETS = re.compile(
    r"^aten\.(view|_unsafe_view|reshape|expand|permute|transpose|t|slice|select|"
    r"unsqueeze|squeeze|narrow|cat|stack|split|split_with_sizes|chunk|flatten|"
    r"unflatten|clone|contiguous|detach|to|_to_copy|copy_|empty|empty_like|"
    r"zeros|ones|fill_|as_strided|sym_size|sym_numel|_assert_tensor_metadata|"
    r"lift_fresh|alias)\b"
)


@contextlib.contextmanager
def _export_friendly_rmsnorm():
    """Make ``RMSNorm`` export-traceable.

    ``RMSNorm.forward_native`` passes ``self.weight.data``; accessing ``.data``
    on a ``Parameter`` during ``torch.export`` turns it into a lifted *constant*
    that becomes a fake tensor ("fake tensor in constant's list"), which aborts
    whole-model export. Here we temporarily pass the ``Parameter`` itself so it
    is lifted as a proper input. Only active during the export call; vLLM
    runtime behavior is untouched.
    """
    from vllm import ir
    from vllm.model_executor.layers.layernorm import RMSNorm

    original = RMSNorm.forward_native

    def patched(self, x, residual=None):
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        return ir.ops.fused_add_rms_norm.maybe_inplace(
            x,
            residual,
            self.weight if self.pass_weight_add else None,
            self.variance_epsilon,
            self.variance_size_override,
        )

    RMSNorm.forward_native = patched
    try:
        yield
    finally:
        RMSNorm.forward_native = original


def _ensure_distributed() -> None:
    """Initialize a single-process (TP=1) distributed environment once."""
    global _DIST_INITIALIZED
    if _DIST_INITIALIZED:
        return
    from vllm.distributed import init_distributed_environment

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        # Pick a free ephemeral port to avoid collisions with leftover
        # processes (e.g. a stale `python app.py`) holding a fixed port.
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="env://",
        local_rank=0, backend="gloo",
    )
    _DIST_INITIALIZED = True


def _expert_parallel_missing(model_config) -> bool:
    """True when the current model is MoE but vLLM's expert-parallel (EP) group
    was never created. ``model_parallel_is_initialized`` only tracks TP/PP, and
    ``initialize_model_parallel`` builds the EP group only when the *current*
    config is MoE — so tracing a dense model first leaves a later MoE model
    without an EP group. Isolated here because vLLM exposes no public predicate
    (``get_ep_group`` asserts) and we must read the private ``_EP`` slot."""
    if not model_config.is_moe:
        return False
    from vllm.distributed import parallel_state as _ps
    return _ps.model_parallel_is_initialized() and _ps._EP is None


def _instantiate_model(raw_config: dict, dtype: str, workdir: str):
    """Instantiate the real vLLM model on ``meta`` from a config dict.

    Returns ``(model, vllm_config)``. Offline (no weights, no HF download).
    ``workdir`` is a caller-owned temp directory used to stage ``config.json``.
    """
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized
    from vllm.model_executor.model_loader.utils import initialize_model
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _ensure_distributed()

    with open(os.path.join(workdir, "config.json"), "w") as f:
        # Export/trace UNQUANTIZED: vLLM's fp8/awq/gptq weight processing breaks
        # under fake-tensor export on ``meta`` (e.g. fp8 ``split_with_sizes``
        # mismatch with no real weights). Quantization only changes weight
        # storage bytes, which the caller applies analytically via
        # ``weight_dtype_bytes`` — so strip it before building the module.
        stripped = {k: v for k, v in raw_config.items()
                    if k not in ("quantization_config", "quantization",
                                 "compression_config")}
        json.dump(stripped, f)

    model_config = ModelConfig(
        model=workdir, tokenizer=workdir, dtype=_normalize_dtype(dtype), seed=0,
        enforce_eager=True, skip_tokenizer_init=True,
    )
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config):
        # ``ensure_model_parallel_initialized``/``model_parallel_is_initialized``
        # only track the TP and PP groups — NOT the expert-parallel (EP) group,
        # which ``initialize_model_parallel`` creates only when the *current*
        # model is MoE. So if a dense model was traced first, the EP group is
        # never built and a later MoE export fails with "expert parallel group
        # is not initialized". Rebuild the (TP=1) model-parallel state under this
        # MoE config so the EP group is created.
        if _expert_parallel_missing(model_config):
            from vllm.distributed import parallel_state as _ps
            _ps.destroy_model_parallel()
        ensure_model_parallel_initialized(1, 1)
        with torch.device("meta"):
            model = initialize_model(vllm_config=vllm_config, model_class=None)
    return model, vllm_config


def _normalize_dtype(dtype: str) -> str:
    d = (dtype or "bfloat16").lower()
    if d in ("bfloat16", "bf16"):
        return "bfloat16"
    if d in ("float16", "fp16", "half"):
        return "float16"
    if d in ("float32", "fp32", "float"):
        return "float32"
    return "bfloat16"


def _export_model(model, vllm_config, n_tokens: int = 8):
    """Single-shot ``torch.export`` with a symbolic token dimension."""
    import torch
    from torch.export import Dim
    from vllm.forward_context import set_forward_context

    ids = torch.zeros(n_tokens, dtype=torch.long, device="meta")
    pos = torch.zeros(n_tokens, dtype=torch.long, device="meta")
    tok = Dim("num_tokens", min=2, max=2**20)
    with set_forward_context(None, vllm_config), _export_friendly_rmsnorm():
        return torch.export.export(
            model, (ids, pos),
            dynamic_shapes=({0: tok}, {0: tok}),
            strict=False,
        )


# -------------------------------------------------------------------
# Exported-graph -> ModuleNode tree
# -------------------------------------------------------------------

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _normalize_target(target: str) -> str:
    """``aten.linear.default`` -> ``aten::linear``; ``_C.rms_norm.default`` -> ``rms_norm``."""
    parts = target.split(".")
    # drop trailing overload (default / int / Tensor / memory_format ...)
    if len(parts) >= 2:
        parts = parts[:-1]
    ns = parts[0]
    name = ".".join(parts[1:]) if len(parts) > 1 else parts[0]
    if ns == "aten":
        return f"aten::{name}"
    if ns in ("_C", "_C_cache_ops", "_moe_C", "_xpu_C"):
        return name  # custom vllm-xpu-kernels op
    if ns == "vllm":
        return name
    return target


def _resolve_op(target: str) -> tuple[str, str, str] | None:
    """Return (display_name, backend, role) or None if the op is framework/no-op."""
    if _FRAMEWORK_TARGETS.match(target):
        return None
    norm = _normalize_target(target)

    if norm in _VLLM_DISPATCH_OPS:
        return _VLLM_DISPATCH_OPS[norm]

    backend, category = classify_op(norm, device_type="xpu", device_time_us=1.0)
    backend_str = backend.value if hasattr(backend, "value") else str(backend)
    if backend_str == "framework":
        return None
    role = _role_for(norm)
    return norm, backend_str, role


def _role_for(norm_name: str) -> str:
    n = norm_name.replace("aten::", "")
    if "rms_norm" in n or "layer_norm" in n:
        return "norm"
    if "silu" in n or "gelu" in n or "act" in n:
        return "activation"
    if "rotary" in n:
        return "rotary_emb"
    if "linear" in n or "mm" in n or "gemm" in n or "matmul" in n:
        return "proj"
    if "embedding" in n:
        return "embedding"
    if "attention" in n or "attn" in n:
        return "attention"
    return n


def _tensor_shape(val) -> list | None:
    """Extract a shape (list of int|str) from a FakeTensor meta val, or None."""
    shape = getattr(val, "shape", None)
    if shape is None:
        return None
    out: list = []
    for d in shape:
        # A concrete dim is a plain Python int. The dynamic token dim is a
        # torch.SymInt -> keep it symbolic (int(SymInt) would silently return
        # the example hint, collapsing prefill/decode to the same size).
        if isinstance(d, int):
            out.append(int(d))
        else:
            out.append(str(d))
    return out


def _numeric(shape: list, token_value: int) -> list[int] | None:
    """Resolve a shape to ints, substituting the symbolic token dim."""
    res: list[int] = []
    for d in shape:
        if isinstance(d, int):
            res.append(d)
        else:
            res.append(token_value)  # the only symbol is the token dim
    return res


def _prod(xs: list[int]) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _module_path_and_classes(node) -> tuple[str, dict[str, str]]:
    """Return (owning_module_path, {path: class_name}) from nn_module_stack."""
    stack = node.meta.get("nn_module_stack") or {}
    path = ""
    classes: dict[str, str] = {}
    for _key, value in stack.items():
        mpath, mcls = value[0], value[1]
        if not isinstance(mpath, str) or "_empty_nn_module_stack" in mpath:
            continue
        cls_name = mcls
        if not isinstance(cls_name, str):
            cls_name = getattr(mcls, "__name__", str(mcls))
        else:
            cls_name = cls_name.rsplit(".", 1)[-1].strip("'\">")
        classes[mpath] = cls_name
        path = mpath  # last (deepest) wins
    return path, classes


class _TreeBuilder:
    """Builds a merged ModuleNode tree from per-op (module_path, OpNode)."""

    def __init__(self, root_name: str, root_type: str,
                 groups: list[tuple[int, int]] | None = None):
        self.path_class: dict[str, str] = {}
        self.ops_by_path: dict[str, list[OpNode]] = {}
        groups = groups or []
        self.count_by_rep: dict[int, int] = {rep: cnt for rep, cnt in groups}
        self.rep_set: set[int] = set(self.count_by_rep)
        self.single = len(groups) <= 1
        self.root_name = root_name
        self.root_type = root_type

    def _layer_token(self, idx: int) -> str:
        return "*" if self.single else str(idx)

    def _normalize_path(self, path: str) -> str | None:
        """Collapse ``layers.N`` -> ``layers.<token>``, keeping only the
        representative layer of each group. Returns None for dropped layers.
        ``token`` is ``*`` for a uniform stack (one group) or the rep index for
        hybrid stacks so distinct groups stay separate nodes."""
        m = _LAYER_RE.search(path)
        if not m:
            return path
        idx = int(m.group(1))
        if idx not in self.rep_set:
            return None
        token = self._layer_token(idx)
        return _LAYER_RE.sub(
            lambda mm: mm.group(0).replace(mm.group(1), token), path)

    def add(self, path: str, classes: dict[str, str], op: OpNode) -> None:
        for p, c in classes.items():
            np = self._normalize_path(p)
            if np:
                self.path_class.setdefault(np, c)
        np = self._normalize_path(path)
        if np is None:
            return
        self.ops_by_path.setdefault(np, []).append(op)

    def build(self) -> ModuleNode:
        root = ModuleNode(name=self.root_name, path="",
                          module_type=self.root_type)
        nodes: dict[str, ModuleNode] = {"": root}

        def get(path: str) -> ModuleNode:
            if path in nodes:
                return nodes[path]
            parent_path, _, leaf = path.rpartition(".")
            parent = get(parent_path)
            mtype = self.path_class.get(path, leaf)
            is_layer = parent_path.endswith("layers")
            name = "decoder_layer" if is_layer else leaf
            full = path if (path == "model" or path.startswith("model.")) \
                else "model." + path
            node = ModuleNode(name=name, path=full, module_type=mtype)
            if is_layer:
                if leaf == "*":
                    # Single uniform group → its count is the only entry.
                    cnt = next(iter(self.count_by_rep.values()), 1)
                else:
                    cnt = self.count_by_rep.get(int(leaf), 1)
                node.repeat_count = max(1, cnt)
            nodes[path] = node
            parent.children.append(node)
            return node

        for path, ops in self.ops_by_path.items():
            get(path).ops.extend(ops)
        return root


_TP_SHARDED_LINEAR = frozenset({
    "ColumnParallelLinear", "MergedColumnParallelLinear",
    "QKVParallelLinear", "QKVCrossParallelLinear", "RowParallelLinear",
    "ParallelLMHead",  # vocab-parallel logits projection
})

# Explicitly-replicated linears (full per rank), e.g. the MoE router gate.
_TP_REPLICATED_LINEAR = frozenset({"ReplicatedLinear"})


def _tp_shard_factor(path: str, role: str, name: str,
                     classes: dict, tp_size: int) -> int:
    """How an op's per-rank cost scales under tensor parallelism.

    The model is exported at TP=1 (full dims), so TP>1 is modelled analytically.
    Sharding is detected from vLLM's actual parallel-layer CLASS in the traced
    module chain (robust across architectures whose module attribute names differ
    — ``.mlp`` vs ``.feed_forward`` etc.):

    * ColumnParallel / MergedColumnParallel / QKVParallel / RowParallel linears
      and the vocab-parallel ``ParallelLMHead`` are sharded → 1/tp per rank.
    * ``ReplicatedLinear`` (e.g. the MoE router gate) is NOT in the set → full.
    * ``VocabParallelEmbedding`` shards the *table*, but the lookup op's traffic
      is the gathered rows + the all-reduced ``[T, H]`` output, which is not
      sharded — so it stays full.
    * The non-linear ops inside the attention block (the attention op itself,
      rotary, KV-cache store) shard with the head count; the MLP activation
      shards with the intermediate — keyed by role.

    FusedMoE expert ops are sharded inside ``fused_moe_expert_ops`` instead and
    never reach this function.

    First-order approximations (documented, acceptable for this estimator):
    * The whole op cost is divided by tp; the *replicated* activation input of a
      column-parallel GEMM (the ``[T, H]`` read) is thus slightly over-divided.
      FLOPs are exact (1/tp); memory error is bounded and small when weights /
      sharded activations dominate (decode, moderate seq).
    * GQA/MQA with ``tp_size > num_kv_heads``: vLLM replicates KV heads rather
      than sharding below one head, so KV-cache/attention cost is undercounted
      by up to ``tp_size / num_kv_heads``. Targets reasonable ``tp <= n_kv``.
    """
    if tp_size <= 1:
        return 1
    cls = set((classes or {}).values())
    if cls & _TP_SHARDED_LINEAR:
        return tp_size
    if role in ("attention", "rotary_emb", "cache_store", "activation"):
        return tp_size
    return 1


def build_traced_graph(
    raw_config: dict,
    model_summary: dict,
    symbols: dict,
    prefill_len: int | None = None,
    decode_batch: int | None = None,
    context_len: int | None = None,
    tp_size: int = 1,
    weight_dtype_bytes: int | None = None,
    extra_children: dict[str, list] | None = None,
) -> dict:
    """Build prefill/decode ModuleNode trees from a single export trace.

    Returns ``{"prefill": <tree dict>, "decode": <tree dict>}``. The caller
    merges these into the full result schema (architecture/symbols/config).
    Raises on any failure so the caller can fall back to the legacy builder.

    ``tp_size`` > 1 is applied analytically (see ``_tp_shard_factor``): the
    model is always exported at TP=1 and per-rank cost is divided by op role.

    ``weight_dtype_bytes`` (quantized models): the model is always exported and
    traced UNQUANTIZED (vLLM's fp8/awq weight processing breaks under fake-tensor
    export on ``meta``), and reduced weight precision is applied analytically to
    GEMM weight operands in ``_cost`` — mirroring the static builder.

    ``extra_children`` (multimodal): per-phase ``ModuleNode`` subtrees that
    torch.export does not capture and the caller supplies analytically — e.g. a
    VL model's vision tower + projector, which the text-only export trace skips
    (no image inputs). They are spliced into each phase's root *before* totals
    are computed so their cost is included. Keyed by phase name (e.g.
    ``{"prefill": [vit, projector]}``).
    """
    dtype = model_summary.get("dtype", "bfloat16")
    dtype_bytes = _dtype_bytes(dtype)
    w_dtype_bytes = weight_dtype_bytes if weight_dtype_bytes else dtype_bytes
    ctx = context_len or 0
    tp_size = max(1, int(tp_size or 1))

    with _TRACE_LOCK:
        workdir = tempfile.mkdtemp(prefix="vllm_breakdown_")
        try:
            model, vllm_config = _instantiate_model(raw_config, dtype, workdir)
            ep = _export_model(model, vllm_config)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    root_type = type(model).__name__

    # Collect compute ops with their owning module path + class chain.
    collected: list[tuple] = []
    moe_blocks: list[tuple[str, dict]] = []  # (experts_path, class_chain)
    _seen_moe: set[str] = set()
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        # FusedMoE expert path is one opaque op — drop it and remember the
        # owning ``*.experts`` module so we can splice real expert ops below.
        if _normalize_target(target) in _MOE_OPAQUE_OPS:
            path, classes = _module_path_and_classes(node)
            if path not in _seen_moe:
                _seen_moe.add(path)
                moe_blocks.append((path, classes))
            continue
        resolved = _resolve_op(target)
        if resolved is None:
            continue
        name, backend, role = resolved
        path, classes = _module_path_and_classes(node)
        # Input tensor shapes (from tensor-valued args).
        in_shapes: list = []
        for arg in node.all_input_nodes:
            val = arg.meta.get("val")
            s = _tensor_shape(val)
            if s is not None:
                in_shapes.append(s)
        out_shape = _tensor_shape(node.meta.get("val"))
        if out_shape is None and in_shapes:
            out_shape = in_shapes[0]
        collected.append((path, classes, name, backend, role, in_shapes, out_shape))

    # Collapse the decoder stack into one representative layer per group of
    # identically-shaped layers (uniform → 1 group; hybrid dense+MoE → 2+).
    # Raises only on multiple ModuleLists (vision tower + LM) → static fallback.
    groups = _group_layers(collected)
    rep_set = {rep for rep, _ in groups}

    # Fail-closed for analytic TP: the per-rank divisor relies on recognising the
    # parallel-linear CLASS of each projection. If a projection op carries no
    # known linear class (sharded *or* the explicitly-replicated router gate),
    # we cannot tell whether it shards and would silently overcount — so fall
    # back to the static builder rather than guess.
    if tp_size > 1:
        _known_linears = _TP_SHARDED_LINEAR | _TP_REPLICATED_LINEAR
        for (path, classes, name, _backend, role, *_rest) in collected:
            if role == "proj":
                cls = set((classes or {}).values())
                if not (cls & _known_linears):
                    raise ValueError(
                        f"projection {path!r} has no recognised parallel-linear "
                        f"class {sorted(cls)!r}; cannot model TP={tp_size}")

    # The opaque FusedMoE ops are dropped before layer grouping, so the MoE
    # group's representative is chosen from its (post-removal) signature. Guard
    # that at least one representative is a traced FusedMoE layer; otherwise
    # splicing per-block (which keeps only representatives) would silently drop
    # every expert op while still reporting "torch.export+fused_moe".
    if moe_blocks:
        moe_layer_idxs = set()
        for mpath, _ in moe_blocks:
            m = _LAYER_RE.search(mpath)
            if m:
                moe_layer_idxs.add(int(m.group(1)))
        if moe_layer_idxs and not (rep_set & moe_layer_idxs):
            raise ValueError(
                f"no representative layer {sorted(rep_set)} is a traced "
                f"FusedMoE block (MoE layers: {sorted(moe_layer_idxs)})")

    # Double-count guard: ``extra_children`` analytically supplies subtrees the
    # trace is assumed to OMIT (e.g. a VL vision tower — text-only export skips
    # it). If the trace unexpectedly DID capture vision/projector ops, splicing
    # would count them twice; raise so the caller falls back to the all-static
    # builder rather than over-report. (The multi-ModuleList guard in
    # ``_group_layers`` keys off ``layers.N`` and would miss a ``blocks.N``-style
    # vision tower, so this explicit check is required.)
    if extra_children:
        for (path, _classes, _name, _backend, role, *_rest) in collected:
            p = (path or "").lower()
            if (role and role.startswith(("vit_", "patch_", "vl_"))) or any(
                    tok in p for tok in ("visual", "vision", ".vit")):
                raise ValueError(
                    f"trace already contains vision op {path!r} (role={role!r});"
                    " analytic vision splice would double-count")

    # FusedMoE expert params for the spliced ops (only used when moe_blocks).
    moe_top_k = model_summary.get("num_experts_per_tok") or 2
    moe_num_experts = model_summary.get("num_experts") or 8
    moe_hidden = model_summary.get("hidden_size")
    moe_expert_I = (model_summary.get("moe_intermediate_size")
                    or model_summary.get("intermediate_size"))
    _shared_explicit = model_summary.get("shared_expert_intermediate_size")
    _n_shared = model_summary.get("n_shared_experts") or 0
    if _shared_explicit:
        moe_shared_I = _shared_explicit
    elif _n_shared:
        moe_shared_I = _n_shared * (moe_expert_I or 0)
    else:
        moe_shared_I = 0

    phases: dict[str, Any] = {}
    phase_tokens = {
        "prefill": (prefill_len or 128),
        "decode": (decode_batch or 1),
    }
    phase_symbol = {"prefill": "S", "decode": "B"}
    for phase, tok in phase_tokens.items():
        sym = phase_symbol[phase]
        builder = _TreeBuilder(root_name=root_type, root_type=root_type,
                               groups=groups)
        for (path, classes, name, backend, role, in_shapes, out_shape) in collected:
            mem, flops = _cost(name, role, in_shapes, out_shape, tok,
                               dtype_bytes, phase, ctx,
                               weight_dtype_bytes=w_dtype_bytes)
            factor = _tp_shard_factor(path, role, name, classes, tp_size)
            if factor > 1:
                mem //= factor
                flops //= factor
            op = OpNode(
                name=name, role=role, backend=backend,
                input_shapes=[_render_shape(s, sym) for s in in_shapes],
                output_shape=_render_shape(out_shape, sym) if out_shape else [],
                memory_bytes=mem, flops=flops, phase=phase,
            )
            builder.add(path, classes, op)
        # Splice the FusedMoE expert ops the export hid. ``builder.add`` drops
        # non-representative layers, so adding per block yields one copy.
        for (mpath, mclasses) in moe_blocks:
            for op in fused_moe_expert_ops(
                phase=phase, token_symbol=sym, token_value=tok,
                top_k=moe_top_k, hidden_size=moe_hidden,
                expert_intermediate=moe_expert_I,
                shared_intermediate=moe_shared_I,
                num_experts=moe_num_experts, dtype_bytes=dtype_bytes,
                tp_size=tp_size, weight_dtype_bytes=w_dtype_bytes,
            ):
                builder.add(mpath, mclasses, op)
        root = builder.build()
        # Splice analytic subtrees the export could not capture (e.g. VL vision
        # tower — the text-only trace skips it). Prepended so they precede the
        # language-model stack, matching the static builder's ordering.
        extras = (extra_children or {}).get(phase)
        if extras:
            root.children[:0] = list(extras)
        _compute_totals(root)
        phases[phase] = root.to_dict()
    src = "torch.export"
    if moe_blocks:
        src += "+fused_moe"
    if extra_children:
        src += "+vision"
    phases["graph_source"] = src
    return phases


_LAYER_SPLIT_RE = re.compile(r"^(?P<prefix>.*?)layers\.(?P<idx>\d+)(?:\.(?P<suffix>.*))?$")

# Residual-fusion turns a layer's leading ``rms_norm`` (first layer, no incoming
# residual) into ``fused_add_rms_norm`` (subsequent layers). Canonicalize so
# this benign difference doesn't look like a heterogeneous stack.
_CANON_OP = {"fused_add_rms_norm": "rms_norm"}


def _group_layers(collected: list) -> list[tuple[int, int]]:
    """Return ``[(representative_index, count), ...]`` for the decoder stack.

    The tracer keeps one representative layer per *group of identically-shaped
    layers* and repeats it ``count`` times. Uniform stacks yield a single group;
    hybrid stacks (e.g. DeepSeek ``first_k_dense_replace`` — leading dense layers
    then MoE layers) yield one group per distinct canonical op signature, each
    rendered as its own repeated decoder-layer node (mirroring the static
    builder's dense+MoE split).

    Still raises on multiple ``*.layers.N`` ModuleLists (vision tower + LM); that
    multi-stack case is handled separately. Layer 0's leading ``rms_norm`` (vs
    the residual-fused norm of later layers) is canonicalized so it does not look
    like its own group.

    Within a group the representative is the *modal* (steady-state) raw signature
    so the merged graph reflects residual-fused norms rather than layer 0's
    special-case leading norm. Groups are returned ordered by representative
    index to preserve stack order.
    """
    prefixes: set[str] = set()
    canon_by_idx: dict[int, set] = {}
    raw_by_idx: dict[int, set] = {}
    for (path, _classes, name, _backend, _role, _in, _out) in collected:
        m = _LAYER_SPLIT_RE.match(path)
        if not m:
            continue
        prefixes.add(m.group("prefix"))
        idx = int(m.group("idx"))
        suffix = m.group("suffix") or ""
        canon_by_idx.setdefault(idx, set()).add(
            (suffix, _CANON_OP.get(name, name)))
        raw_by_idx.setdefault(idx, set()).add((suffix, name))

    if not canon_by_idx:
        return []  # no repeated layers — nothing to collapse
    if len(prefixes) > 1:
        raise ValueError(
            f"multiple layer ModuleLists {sorted(prefixes)!r}; "
            "static builder handles multi-stack models")

    # Partition layer indices by canonical signature → one group each.
    by_signature: dict[frozenset, list[int]] = {}
    for idx, s in canon_by_idx.items():
        by_signature.setdefault(frozenset(s), []).append(idx)

    groups: list[tuple[int, int]] = []
    for members in by_signature.values():
        # Representative = largest raw-signature subgroup (steady state),
        # tie-broken toward higher indices, then its lowest index.
        raw_groups: dict[frozenset, list[int]] = {}
        for idx in members:
            raw_groups.setdefault(frozenset(raw_by_idx[idx]), []).append(idx)
        best = max(raw_groups.values(), key=lambda idxs: (len(idxs), max(idxs)))
        groups.append((min(best), len(members)))

    groups.sort(key=lambda g: g[0])
    return groups


def _render_shape(shape: list, token_symbol: str) -> list:
    """Replace the symbolic token dim with a readable symbol (``S``/``B``)."""
    return [d if isinstance(d, int) else token_symbol for d in shape]


def _cost(name: str, role: str, in_shapes: list, out_shape: list | None,
          token_value: int, dtype_bytes: int, phase: str,
          context_len: int, weight_dtype_bytes: int | None = None) -> tuple[int, int]:
    """Estimate memory (bytes touched) + FLOPs from symbolic shapes.

    Most ops use a generic input+output byte count. Attention and embedding
    are special-cased because the generic count is badly wrong for them.

    ``weight_dtype_bytes`` (quantized models): a GEMM's weight operand is the
    static 2-D input (the activation carries the symbolic token dim), so it is
    billed at the reduced weight precision while activations/outputs stay at the
    model dtype — mirroring the static builder's ``_mm_mem(weight_dtype_bytes=)``.
    """
    n = name.replace("aten::", "")
    w_bytes = weight_dtype_bytes if weight_dtype_bytes is not None else dtype_bytes

    if role == "attention" or "attention" in n:
        return _attention_cost(in_shapes, token_value, dtype_bytes,
                               phase, context_len)

    if "embedding" in n:
        # A lookup reads only the gathered rows (≈ output), not the whole
        # [vocab, hidden] table. Count rows read + output written.
        if out_shape:
            o = _numeric(out_shape, token_value)
            return 2 * _prod(o) * dtype_bytes, 0
        return 0, 0

    is_gemm = any(k in n for k in ("linear", "mm", "matmul", "gemm"))
    mem = 0
    for s in in_shapes:
        num = _numeric(s, token_value)
        if not num:
            continue
        # The weight operand of a GEMM is a static (all-int) ≥2-D tensor; under
        # quantization it is stored at reduced precision.
        bytes_per = dtype_bytes
        if (is_gemm and w_bytes != dtype_bytes
                and len(s) >= 2 and all(isinstance(d, int) for d in s)):
            bytes_per = w_bytes
        mem += _prod(num) * bytes_per
    if out_shape:
        num = _numeric(out_shape, token_value)
        if num:
            mem += _prod(num) * dtype_bytes

    flops = 0
    if is_gemm and out_shape:
        out_num = _numeric(out_shape, token_value)
        # find the contracted (K) dim from a 2D input that shares M with output
        k = None
        for s in in_shapes:
            sn = _numeric(s, token_value)
            if len(sn) >= 2 and sn[0] == out_num[0]:
                k = sn[-1]
                break
        if k is None:
            for s in in_shapes:
                sn = _numeric(s, token_value)
                if len(sn) >= 2:
                    k = sn[-1]
                    break
        if k is not None and out_num:
            flops = 2 * _prod(out_num) * k
    return mem, flops


def _attention_cost(in_shapes: list, token_value: int, dtype_bytes: int,
                    phase: str, context_len: int) -> tuple[int, int]:
    """Phase-aware attention cost including the KV-cache read over context.

    Unified attention traces identically for prefill and decode, so the cost
    model — not the graph — distinguishes them. Decode is dominated by reading
    the length-``C`` KV cache; the generic input/output count misses this.
    """
    # q is [T, n_h, d]; k/v are [T, n_kv, d]. Take the 3-D tensor inputs.
    threed = [s for s in in_shapes if len(s) == 3]
    if not threed:
        # Unexpected shape layout — fall back to a plain byte count.
        mem = sum(_prod(_numeric(s, token_value)) * dtype_bytes
                  for s in in_shapes)
        return mem, 0
    heads = [s[1] for s in threed if isinstance(s[1], int)]
    n_h = max(heads) if heads else 1
    n_kv = min(heads) if heads else 1
    d = next((s[2] for s in threed if isinstance(s[2], int)), 1)

    new = token_value  # query tokens this step (prefill: S, decode: B seqs)
    if phase == "decode":
        kv_len = context_len
    else:
        kv_len = token_value + context_len  # chunked prefill reads prior C too

    q_bytes = new * n_h * d * dtype_bytes          # read Q
    o_bytes = new * n_h * d * dtype_bytes           # write output
    kv_bytes = new * kv_len * n_kv * d * 2 * dtype_bytes  # read K and V cache
    mem = q_bytes + o_bytes + kv_bytes
    # QK^T then (softmax)·V, each ~2·(new·kv_len·n_h·d) flops.
    flops = 2 * (2 * new * kv_len * n_h * d)
    return mem, flops
