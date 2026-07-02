# SPDX-License-Identifier: Apache-2.0
"""Recover module *attribute* names for the profile-first graph.

The profile-first reconstruction in :mod:`breakdown.graph_from_trace` is
accurate — it reflects the real dispatched ops — but the torch profiler only
labels module events with their **class** name (``nn.Module: <Cls>_<idx>``).
Sibling modules of the same class (Qwen3 ``q_norm``/``k_norm``, both ``RMSNorm``;
``input_layernorm``/``post_attention_layernorm``; the two projections in an MLP,
...) are therefore indistinguishable and fall back to a generic heuristic name.

This module ports the *idea* from the ``torch_export`` branch: derive the exact
attribute name of every module from the **real** model's ``named_modules()``
(``model.layers.0.self_attn.q_norm`` → attribute ``q_norm``) and overlay those
names onto the trace-reconstructed tree via a structural alignment. Unlike
torch_export it does not rebuild the graph — the accurate profile-based tree is
kept, only its module *labels* are enriched.

The reference tree is obtained from the live vLLM model during profiling
(cheap — the model is already loaded), with an optional ``meta``-device
instantiation fallback for the offline / trace-upload path.

The pure tree functions (:func:`build_ref_tree`, :func:`enrich_graph_names`)
have no torch/vLLM dependency so they are unit-testable on any host.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("vllm_xpu_breakdown")

# ===================================================================
# Reference module-name tree (pure — torch-free)
# ===================================================================


def _is_numeric(name: str) -> bool:
    return name.isdigit()


def _make_node(attr: str, cls: str) -> dict:
    return {"attr": attr, "cls": cls, "children": [],
            "is_group": False, "group_size": 1}


def build_ref_tree(named_modules: list[tuple[str, str]]) -> dict | None:
    """Build a reference module tree from ``model.named_modules()`` output.

    ``named_modules`` is a list of ``(qualified_name, class_name)`` in
    registration order (the first entry is the root, whose qualified name is the
    empty string). Returns a nested tree::

        {"attr": str, "cls": str, "children": [...],
         "is_group": bool, "group_size": int}

    ``ModuleList``/indexed containers are *inlined* (their forward is never
    called, so they don't appear as trace module events) and consecutive numeric
    entries sharing a class collapse into a single representative marked
    ``is_group`` — mirroring how the trace collapses repeated decoder layers.
    """
    if not named_modules:
        return None

    nodes: dict[str, dict] = {}
    root: dict | None = None
    for path, cls in named_modules:
        leaf = path.rpartition(".")[2]
        node = _make_node(leaf, cls)
        nodes[path] = node
        if path == "":
            root = node
            continue
        parent_path = path.rpartition(".")[0]
        parent = nodes.get(parent_path)
        if parent is not None:
            parent["children"].append(node)

    if root is None:
        # No explicit root entry — synthesize one over the shallowest nodes.
        root = _make_node("", "")
        for path, node in nodes.items():
            if "." not in path:
                root["children"].append(node)

    _inline_containers(root)
    return root


def _inline_containers(node: dict) -> None:
    """Recursively inline indexed containers and collapse numeric siblings."""
    new_children: list[dict] = []
    for child in node["children"]:
        _inline_containers(child)
        if _is_indexed_container(child):
            new_children.extend(_collapse_numeric(child))
        else:
            new_children.append(child)
    node["children"] = new_children


def _is_indexed_container(node: dict) -> bool:
    """A container whose children are all numeric (``ModuleList``-style)."""
    kids = node["children"]
    return bool(kids) and all(_is_numeric(c["attr"]) for c in kids)


def _collapse_numeric(container: dict) -> list[dict]:
    """Collapse a container's numeric entries into per-class representatives.

    The representative keeps its own subtree (from the first entry of the group)
    and adopts the *container's* attribute name (e.g. ``layers``) so the display
    name reflects the meaningful list attribute rather than an index.
    """
    reps: dict[str, dict] = {}
    order: list[str] = []
    for entry in container["children"]:
        cls = entry["cls"]
        rep = reps.get(cls)
        if rep is None:
            rep = entry
            rep["attr"] = container["attr"]
            rep["is_group"] = True
            rep["group_size"] = 1
            reps[cls] = rep
            order.append(cls)
        else:
            rep["group_size"] += 1
    return [reps[c] for c in order]


# ===================================================================
# Alignment: overlay reference attribute names onto the trace tree
# ===================================================================


def _display_name(rnode: dict) -> str:
    """Human-friendly node name for a reference module."""
    attr = rnode.get("attr") or ""
    if rnode.get("is_group"):
        cls = (rnode.get("cls") or "").lower()
        if "decoderlayer" in cls or cls.endswith("layer") or "block" in cls:
            return "decoder_layer"
        return attr or "layer"
    return attr or rnode.get("cls") or ""


def _find_ref_desc(rnode: dict, cls: str, max_depth: int = 3) -> dict | None:
    """Find the shallowest descendant of ``rnode`` whose class is ``cls``."""
    if not cls:
        return None
    frontier = [(rnode, 0)]
    while frontier:
        node, depth = frontier.pop(0)
        for child in node["children"]:
            if child["cls"] == cls:
                return child
            if depth + 1 < max_depth:
                frontier.append((child, depth + 1))
    return None


def _effective_ref_children(rnode: dict, present: set[str]) -> list[dict]:
    """Reference children flattened past levels absent from the trace.

    vLLM wraps the decoder stack in an inner ``*Model`` module
    (``*ForCausalLM → *Model → [embed, layers, norm]``), but that wrapper's
    ``forward`` may not emit its own trace module event, so the trace nests the
    stack directly under ``*ForCausalLM``. When a reference child's class is not
    among the trace node's actual child classes (``present``), it is treated as a
    skipped level and replaced by *its* children, recursively — so matching lines
    up across the missing level instead of stalling.
    """
    out: list[dict] = []
    for rc in rnode["children"]:
        if rc["cls"] in present:
            out.append(rc)
        else:
            out.extend(_effective_ref_children(rc, present))
    return out


def _match_ref(rchildren: list[dict], used: list[bool],
               tcls: str | None) -> int | None:
    """Index of the reference child matching ``tcls`` (prefer unused).

    When the trace has more same-class siblings than the (collapsed) reference
    has representatives — e.g. a MoE stack's dense-prefix and MoE decoder-layer
    groups both map to one collapsed layer template — reuse the last matched
    representative so both groups still get their children named.
    """
    if not tcls:
        return None
    last = None
    for i, rc in enumerate(rchildren):
        if rc["cls"] == tcls:
            if not used[i]:
                return i
            last = i
    return last


def _align(tnode: dict, rnode: dict) -> None:
    """Assign ``tnode['name']`` from ``rnode`` and align children by class+order."""
    tnode["name"] = _display_name(rnode)
    tnode["module_name"] = rnode.get("attr") or ""
    tchildren = tnode.get("children", [])
    present = {tc.get("module_type") for tc in tchildren}
    rchildren = _effective_ref_children(rnode, present)
    used = [False] * len(rchildren)
    for tc in tchildren:
        j = _match_ref(rchildren, used, tc.get("module_type"))
        if j is not None:
            used[j] = True
            _align(tc, rchildren[j])


def _enrich_root(troot: dict, rroot: dict) -> None:
    """Align a phase tree root against the reference tree root."""
    tcls = troot.get("module_type")
    # Direct class match (trace root == reference root, e.g. *ForCausalLM).
    if tcls == rroot["cls"]:
        _align(troot, rroot)
        return
    # Reference root is an ancestor of the trace root (trace root == the inner
    # model class, reference root == the *ForCausalLM wrapper).
    r = _find_ref_desc(rroot, tcls)
    if r is not None:
        _align(troot, r)
        return
    # Trace root is a synthetic wrapper (``step`` / InferenceStep): align the
    # matching child instead.
    for tc in troot.get("children", []):
        if tc.get("module_type") == rroot["cls"]:
            _align(tc, rroot)
            return
    for tc in troot.get("children", []):
        r = _find_ref_desc(rroot, tc.get("module_type"))
        if r is not None:
            _align(tc, r)


def enrich_graph_names(graph: dict, ref_tree: dict | None) -> dict:
    """Overlay reference attribute names onto an already-built graph, in place.

    ``graph`` is a dict with ``prefill`` / ``decode`` trees. No-op when
    ``ref_tree`` is falsy. Returns ``graph`` for convenience.

    This is a *post-hoc* labeler that operates on the finalized (already
    repeat-collapsed) tree — it renames nodes but cannot split siblings that the
    reconstruction already merged. The production profile-first path instead
    applies names on the *raw* forest before collapse
    (``graph_from_trace._apply_ref_names``) so distinctly-named siblings
    (``q_norm``/``k_norm``) stay separate; this helper reuses the same alignment
    primitives and is handy for tests / static graphs.
    """
    if not ref_tree:
        return graph
    for phase in ("prefill", "decode"):
        tree = graph.get(phase)
        if tree:
            _enrich_root(tree, ref_tree)
    graph["has_module_names"] = True
    return graph


# ===================================================================
# Reference tree from a live / instantiated vLLM model (torch + vLLM)
# ===================================================================


def named_modules_from_model(model: Any) -> list[tuple[str, str]]:
    """``[(qualified_name, class_name), ...]`` for a live ``nn.Module``."""
    return [(name, type(mod).__name__) for name, mod in model.named_modules()]


def _collect_named_modules(model: Any) -> list[tuple[str, str]]:
    # Standalone (picklable) so it can run inside ``LLM.apply_model`` workers.
    return [(name, type(mod).__name__) for name, mod in model.named_modules()]


def _get_model_direct(llm: Any) -> Any | None:
    """Best-effort in-process traversal to the underlying ``nn.Module``."""
    chains = (
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "model_executor", "driver_worker", "worker",
         "model_runner", "model"),
        ("engine", "model_executor", "driver_worker", "model_runner", "model"),
        ("llm_engine", "engine_core", "engine_core", "model_executor",
         "driver_worker", "model_runner", "model"),
    )
    for chain in chains:
        obj = llm
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "named_modules"):
            return obj
    return None


def ref_tree_from_llm(llm: Any) -> dict | None:
    """Build a reference module-name tree from a loaded vLLM ``LLM``.

    Tries ``LLM.apply_model`` first (works across the worker boundary, returning
    picklable ``(name, class)`` strings), then falls back to direct in-process
    attribute traversal. Returns ``None`` on any failure — naming enrichment is
    strictly best-effort.
    """
    items: list[tuple[str, str]] | None = None
    apply_model = getattr(llm, "apply_model", None)
    if callable(apply_model):
        try:
            results = apply_model(_collect_named_modules)
            if isinstance(results, (list, tuple)) and results:
                first = results[0]
                items = first if (isinstance(first, list)
                                  and first and isinstance(first[0], tuple)) \
                    else results  # some versions return the list directly
        except Exception:
            logger.warning("ref_tree_from_llm: apply_model path failed",
                           exc_info=True)
            items = None
    if not items:
        model = _get_model_direct(llm)
        if model is not None:
            try:
                items = named_modules_from_model(model)
            except Exception:
                logger.warning("ref_tree_from_llm: direct traversal failed",
                               exc_info=True)
                items = None
        else:
            logger.warning(
                "ref_tree_from_llm: apply_model yielded nothing and no direct "
                "model path matched (vLLM engine layout unrecognized).")
    if not items:
        return None
    try:
        tree = build_ref_tree(list(items))
        if tree:
            logger.info("ref_tree_from_llm: read %d named modules (root=%s)",
                        len(items), tree.get("cls"))
        return tree
    except Exception:
        logger.warning("ref_tree_from_llm: build_ref_tree failed", exc_info=True)
        return None


def ref_tree_from_config(
    raw_config: dict,
    dtype: str = "bfloat16",
    model_id: str | None = None,
) -> dict | None:
    """Build a reference tree by instantiating the model on ``meta`` (no weights).

    Fallback for the offline / trace-upload path where no live ``LLM`` exists.
    Reuses the ``torch_export`` idea of a weightless ``meta``-device build purely
    to read ``named_modules()``. Heavy and network-dependent. Returns ``None`` on
    any failure.
    """
    import tempfile

    try:
        import torch  # noqa: F401
        from vllm.config import ModelConfig, VllmConfig
        from vllm.config.vllm import set_current_vllm_config
        from vllm.distributed import (
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )
        from vllm.model_executor.model_loader.utils import initialize_model
    except Exception:
        return None

    import json
    import socket

    workdir = tempfile.mkdtemp(prefix="xpu_breakdown_names_")
    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        try:
            init_distributed_environment(
                world_size=1, rank=0, distributed_init_method="env://",
                local_rank=0, backend="gloo",
            )
        except Exception:
            pass  # already initialized

        stripped = {k: v for k, v in raw_config.items()
                    if k not in ("quantization_config", "quantization",
                                 "compression_config")}
        with open(os.path.join(workdir, "config.json"), "w") as f:
            json.dump(stripped, f)

        model_config = ModelConfig(
            model=workdir, tokenizer=workdir, dtype=_normalize_dtype(dtype),
            seed=0, enforce_eager=True, skip_tokenizer_init=True,
            trust_remote_code=True,
        )
        vllm_config = VllmConfig(model_config=model_config)
        with set_current_vllm_config(vllm_config):
            ensure_model_parallel_initialized(1, 1)
            import torch as _torch
            with _torch.device("meta"):
                model = initialize_model(vllm_config=vllm_config,
                                         model_class=None)
        items = named_modules_from_model(model)
        return build_ref_tree(items)
    except Exception:
        return None
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def _normalize_dtype(dtype: str) -> str:
    d = (dtype or "bfloat16").lower()
    if d in ("bfloat16", "bf16"):
        return "bfloat16"
    if d in ("float16", "fp16", "half"):
        return "float16"
    if d in ("float32", "fp32", "float"):
        return "float32"
    return "bfloat16"
