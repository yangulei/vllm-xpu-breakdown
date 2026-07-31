# SPDX-License-Identifier: Apache-2.0
"""Replay paged attention and the KV-cache write outside a live engine.

Attention is normally the single most expensive op in the profile, and it was
also the one the benchmark refused: ``vllm::unified_attention_with_output``
takes a ``layer_name`` and pulls the KV cache, the block table and the sequence
metadata out of vLLM's *forward context*, so the dispatcher op cannot be called
standalone. Refusing it left the heaviest kernel unmeasured and therefore
un-rankable.

What the wrapper hides is not the kernel - it is the **context**. One level
down, the platform's varlen FlashAttention entry point
(``fa_utils.flash_attn_varlen_func``: the vllm-xpu-kernels SYCL kernel on XPU,
vllm_flash_attn on CUDA) takes exactly that context as plain arguments. So this
module rebuilds it:

* a **paged KV cache** ``[num_blocks, block_size, n_kv, d]`` in the NHD layout
  XPU requires, big enough to hold ``context + query`` tokens for every
  sequence, with each sequence given its **own** blocks - a shared or repeated
  block table would make every sequence hit the same cache lines and measure a
  cache-resident gather instead of the real paged access pattern;
* ``cu_seqlens_q`` / ``seqused_k`` describing the operating point the case was
  swept at (prefill: one sequence of ``S`` new tokens over ``C`` cached ones;
  decode: ``B`` sequences of one new token each);
* the same ``softmax_scale`` and ``causal`` flag vLLM passes.

The geometry is *read from the case*, never assumed: the head counts and head
dim come from the recorded query/key operands, the token count from the query's
leading dim, and the context length from the sweep point. Everything else that
would change the kernel's work (the block size) is explicit and overridable.

The KV-cache write (``vllm::unified_kv_cache_update`` → ``reshape_and_cache_flash``)
is rebuilt from the same cache, since it is the other half of the same context.
"""
from __future__ import annotations

import os
from typing import Any

from breakdown.bench.inputs import ArgBuildError, Call, make_tensor
from breakdown.bench.recipes import override

#: Paged KV-cache block size. Not recorded in the trace (it is engine
#: configuration, not an operand), but it changes the gather pattern the kernel
#: sees, so it is explicit rather than hidden. vLLM's default is 16; override
#: with ``BREAKDOWN_BENCH_KV_BLOCK_SIZE`` to match a run configured otherwise.
DEFAULT_KV_BLOCK_SIZE = 16
_BLOCK_ENV = "BREAKDOWN_BENCH_KV_BLOCK_SIZE"

#: Refuse to allocate a KV cache larger than this fraction of device memory -
#: an OOM mid-run would take out the whole op's worker, and a clear
#: "the operating point does not fit" is more useful than a device error.
MAX_CACHE_FRACTION = 0.5


def kv_block_size() -> int:
    try:
        return max(int(os.environ.get(_BLOCK_ENV, DEFAULT_KV_BLOCK_SIZE)), 1)
    except ValueError:
        return DEFAULT_KV_BLOCK_SIZE


class _Geometry:
    """The attention operating point, derived from the case's own operands."""

    def __init__(self, case) -> None:
        tensors = [t for t in case.tensor_args if len(t.get("dims") or []) == 3]
        if len(tensors) < 2:
            raise ArgBuildError(
                f"{case.op}: expected [tokens, heads, head_dim] query and key "
                f"operands, got {[t.get('dims') for t in case.tensor_args]}")
        q, k = tensors[0], tensors[1]
        qd = [int(x) for x in q["dims"]]
        kd = [int(x) for x in k["dims"]]
        self.dtype: str = q.get("dtype") or "bfloat16"
        self.tokens = max(qd[0], 1)
        self.n_h = max(qd[1], 1)
        self.d = max(qd[2], 1)
        self.n_kv = max(kd[1], 1)
        self.ctx = max(int(case.ctx_len or 0), 0)
        self.block = kv_block_size()

        # One sequence per batch entry, the tokens split across them. Prefill
        # sweeps batch=1 with S new tokens; decode sweeps B sequences advancing
        # one token each. A remainder is given to the leading sequences rather
        # than dropped, so the token count always matches the query operand.
        seqs = max(int(case.batch_size or 1), 1)
        seqs = min(seqs, self.tokens)
        base, rem = divmod(self.tokens, seqs)
        self.q_lens = [base + 1] * rem + [base] * (seqs - rem)
        self.seqs = seqs
        self.k_lens = [self.ctx + n for n in self.q_lens]
        self.max_q = max(self.q_lens)
        self.max_k = max(self.k_lens)
        # Blocks per sequence, each sequence getting a disjoint range.
        self.blocks_per_seq = max(
            (self.max_k + self.block - 1) // self.block, 1)
        self.num_blocks = self.blocks_per_seq * self.seqs

    def cache_bytes(self, itemsize: int) -> int:
        return (2 * self.num_blocks * self.block * self.n_kv * self.d
                * itemsize)


def _device_memory(device: str) -> int:
    """Total memory of the replay target, ``0`` when it cannot be determined.

    CPU is included deliberately: the capacity guard is what stops an
    unreachable operating point from turning into a multi-terabyte allocation
    that hangs the worker instead of failing it.
    """
    import torch

    mod = getattr(torch, device, None)
    if mod is not None and hasattr(mod, "get_device_properties"):
        try:
            return int(mod.get_device_properties(0).total_memory)
        except Exception:                   # noqa: BLE001 - no device / no props
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def _check_capacity(geo: _Geometry, device: str, itemsize: int) -> None:
    total = _device_memory(device)
    if not total:
        return
    need = geo.cache_bytes(itemsize)
    if need > total * MAX_CACHE_FRACTION:
        raise ArgBuildError(
            f"paged KV cache for this operating point needs "
            f"{need / 2 ** 30:.1f} GiB ({geo.num_blocks} blocks x "
            f"{geo.block} x {geo.n_kv} x {geo.d}), more than "
            f"{MAX_CACHE_FRACTION:.0%} of the device's "
            f"{total / 2 ** 30:.1f} GiB - benchmark this op at a smaller "
            f"context/batch")


def _build_cache(geo: _Geometry, device: str) -> tuple[Any, Any]:
    """``(key_cache, value_cache)`` in the ``[blocks, block, n_kv, d]`` NHD
    layout both the XPU and CUDA paged kernels expect."""
    dims = [geo.num_blocks, geo.block, geo.n_kv, geo.d]
    return (make_tensor(dims, geo.dtype, device),
            make_tensor(dims, geo.dtype, device))


def _block_table(geo: _Geometry, device: str):
    """Disjoint physical blocks per sequence, in the order they are read.

    Every sequence pointing at the same blocks would turn the paged gather into
    a cache hit and understate the kernel by a large factor.
    """
    import torch

    ids = torch.arange(geo.num_blocks, dtype=torch.int32, device=device)
    return ids.view(geo.seqs, geo.blocks_per_seq)


def _cu_seqlens(lens: list[int], device: str):
    import torch

    out = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    out[1:] = torch.tensor(lens, dtype=torch.int32, device=device).cumsum(0)
    return out


@override("vllm::unified_attention_with_output")
def _paged_attention(case, resolved, device: str) -> Call:
    """Replay attention through the platform's varlen FlashAttention call."""
    import torch

    geo = _Geometry(case)
    itemsize = torch.empty(0, dtype=_torch_dtype(geo.dtype)).element_size()
    _check_capacity(geo, device, itemsize)

    q = make_tensor([geo.tokens, geo.n_h, geo.d], geo.dtype, device)
    out = torch.empty_like(q)
    key_cache, value_cache = _build_cache(geo, device)
    seqused_k = torch.tensor(geo.k_lens, dtype=torch.int32, device=device)

    kwargs: dict[str, Any] = {
        "q": q, "k": key_cache, "v": value_cache, "out": out,
        "cu_seqlens_q": _cu_seqlens(geo.q_lens, device),
        "max_seqlen_q": geo.max_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": geo.max_k,
        "softmax_scale": geo.d ** -0.5,
        "causal": True,
        "block_table": _block_table(geo, device),
    }
    # ``out`` is fully overwritten by every call, so it needs no restoration
    # between windows; the cache is read-only here.
    return Call(args=[], kwargs=kwargs)


@override("vllm::unified_kv_cache_update")
def _kv_cache_update(case, resolved, device: str) -> Call:
    """Replay the paged KV-cache write (``reshape_and_cache_flash``)."""
    import torch

    geo = _Geometry(case)
    itemsize = torch.empty(0, dtype=_torch_dtype(geo.dtype)).element_size()
    _check_capacity(geo, device, itemsize)

    key = make_tensor([geo.tokens, geo.n_kv, geo.d], geo.dtype, device)
    value = make_tensor([geo.tokens, geo.n_kv, geo.d], geo.dtype, device)
    key_cache, value_cache = _build_cache(geo, device)

    # Each new token is written to the slot right after that sequence's cached
    # context - the real scatter pattern, and never twice to the same slot.
    slots: list[int] = []
    for i, n in enumerate(geo.q_lens):
        base = i * geo.blocks_per_seq * geo.block + geo.ctx
        slots.extend(range(base, base + n))
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)

    return Call(args=[key, value, key_cache, value_cache, slot_mapping,
                      "auto", scale, scale],
                mutated=[key_cache, value_cache])


def _torch_dtype(name: str):
    from breakdown.bench.inputs import torch_dtype
    return torch_dtype(name)
