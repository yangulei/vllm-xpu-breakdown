# SPDX-License-Identifier: Apache-2.0
"""Enhanced op analysis — shapes, memory, FLOPs, arithmetic intensity, layer merging."""

from __future__ import annotations

import math
from typing import Any

from .classifier import Backend

# dtype → bytes per element
DTYPE_BYTES: dict[str, int] = {
    "float32": 4, "fp32": 4, "float": 4,
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "fp8": 1,
    "int8": 1, "uint8": 1,
    "int4": 1,  # packed, but use 0.5 effectively
    "int32": 4, "int64": 8, "int16": 2,
    "bool": 1,
    # The profiler records ``Input type`` with C++ type names, so an index
    # tensor arrives as ``long int``, not ``int64``. Without these it fell back
    # to 2 bytes and every index/position operand was undercounted 4x.
    "long int": 8, "long": 8, "long long": 8, "unsigned long": 8,
    "int": 4, "unsigned int": 4, "short": 2, "unsigned short": 2,
    "char": 1, "signed char": 1, "unsigned char": 1, "byte": 1,
    "double": 8, "float64": 8,
}


def dtype_size(dtype_str: str) -> int:
    """Return bytes per element for a dtype string."""
    dtype_str = dtype_str.lower().replace("torch.", "")
    return DTYPE_BYTES.get(dtype_str, 2)  # default to 2 (bf16)


def _prod(shape: list[int]) -> int:
    """Product of shape dimensions, treating strings as 1."""
    result = 1
    for s in shape:
        if isinstance(s, int):
            result *= s
    return result


def symbolize_shape(shape: list[int], dim_symbols: dict[int, str],
                    batch_size: int | None = None,
                    seq_len: int | None = None) -> list[int | str]:
    """Replace known dimension values with symbolic names."""
    result: list[int | str] = []
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            result.append(dim)
            continue
        # First dim is often batch
        if i == 0 and batch_size is not None and dim == batch_size:
            result.append("B")
        # Sequence length (often 2nd dim, or first for 2D)
        elif seq_len is not None and dim == seq_len:
            result.append("S")
        elif dim in dim_symbols:
            result.append(dim_symbols[dim])
        else:
            result.append(dim)
    return result


def estimate_memory(op_name: str, shapes: list[list[int]],
                    dtype_bytes: int = 2) -> int:
    """Estimate total memory bytes transferred (reads + writes).

    Conservative estimate based on op type and tensor shapes.
    """
    if not shapes:
        return 0

    # For most ops: read all inputs + write all outputs
    # Output shape ≈ first input shape for element-wise ops
    total_elements = 0
    for s in shapes:
        total_elements += _prod(s)

    base = op_name.split("::")[-1].lower()

    # Matrix multiply: read A + B, write C
    if base in ("mm", "bmm", "addmm", "matmul", "linear", "_scaled_mm",
                "fp8_gemm", "fp4_gemm", "int4_gemm_w4a16", "int4_gemm_w4a8",
                "cutlass_grouped_gemm_interface"):
        if len(shapes) >= 2:
            read_bytes = sum(_prod(s) for s in shapes) * dtype_bytes
            # Output shape: [..., M, N] where A is [..., M, K] and B is [..., K, N]
            if len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
                out_elements = _prod(shapes[0][:-1]) * shapes[1][-1]
                return read_bytes + out_elements * dtype_bytes
            return read_bytes * 2

    # Attention: Q, K, V reads + output write
    if base in ("scaled_dot_product_attention",
                "_scaled_dot_product_flash_attention",
                "gdn_attention", "merge_attn_states"):
        return total_elements * dtype_bytes * 2  # read + write

    # Element-wise: read input + write output (same size)
    if base in ("silu_and_mul", "mul_and_silu", "gelu_and_mul",
                "gelu_tanh_and_mul", "gelu_fast", "gelu_new", "gelu_quick",
                "swigluoai_and_mul", "swiglustep_and_mul",
                "relu", "silu", "gelu", "sigmoid", "tanh",
                "fatrelu_and_mul", "relu2_no_mul",
                "mul", "add", "sub", "div", "rsqrt", "sqrt", "exp", "log"):
        if shapes:
            in_bytes = _prod(shapes[0]) * dtype_bytes
            return in_bytes * 2  # read + write

    # Normalization: read input + weight, write output
    if base in ("rms_norm", "fused_add_rms_norm", "layer_norm",
                "rms_norm_dynamic_per_token_quant", "rms_norm_per_block_quant",
                "rms_norm_static_fp8_quant",
                "fused_add_rms_norm_static_fp8_quant"):
        if shapes:
            in_bytes = _prod(shapes[0]) * dtype_bytes
            return in_bytes * 3  # read input + read weight + write output

    # Quantization: read input, write quantized output (smaller)
    if "quant" in base or "fp8" in base:
        if shapes:
            return _prod(shapes[0]) * dtype_bytes + _prod(shapes[0])  # input + output(1byte)

    # Cache ops: read + write cache blocks
    if "cache" in base or "reshape_and_cache" in base:
        return total_elements * dtype_bytes * 2

    # Default: read all inputs, assume output ≈ first input
    if shapes:
        read = total_elements * dtype_bytes
        write = _prod(shapes[0]) * dtype_bytes
        return read + write
    return 0


def estimate_flops(op_name: str, shapes: list[list[int]],
                   n_seqs: int = 1) -> int:
    """Estimate FLOPs for an operation based on its type and shapes.

    ``n_seqs`` is the number of independent sequences the call covers. It only
    matters for attention, where the key/value rows are the *total* KV read
    across the batch while each query attends only its own sequence's keys.
    """
    if not shapes:
        return 0

    base = op_name.split("::")[-1].lower()

    # Matrix multiply: 2*M*K*N
    if base in ("mm", "linear", "_scaled_mm", "fp8_gemm", "fp4_gemm",
                "int4_gemm_w4a16", "int4_gemm_w4a8"):
        if len(shapes) >= 2 and len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
            M = _prod(shapes[0][:-1])
            K = shapes[0][-1] if isinstance(shapes[0][-1], int) else 1
            N = shapes[1][-1] if isinstance(shapes[1][-1], int) else 1
            return 2 * M * K * N

    if base == "bmm":
        if len(shapes) >= 2 and len(shapes[0]) >= 3 and len(shapes[1]) >= 3:
            B = shapes[0][0] if isinstance(shapes[0][0], int) else 1
            M = shapes[0][1] if isinstance(shapes[0][1], int) else 1
            K = shapes[0][2] if isinstance(shapes[0][2], int) else 1
            N = shapes[1][2] if isinstance(shapes[1][2], int) else 1
            return 2 * B * M * K * N

    if base == "addmm":
        if len(shapes) >= 3 and len(shapes[1]) >= 2 and len(shapes[2]) >= 2:
            M = shapes[1][0] if isinstance(shapes[1][0], int) else 1
            K = shapes[1][1] if isinstance(shapes[1][1], int) else 1
            N = shapes[2][1] if isinstance(shapes[2][1], int) else 1
            return 2 * M * K * N + M * N  # matmul + add

    # Grouped (MoE expert) GEMM: A [M, K] x B [E, K, N] -> D [M, N]. Every row
    # goes through exactly one expert, so the work is a plain M*K*N - the
    # expert count multiplies the *weights read*, not the arithmetic. Without
    # this the dominant kernel of an MoE model had zero FLOPs, hence an
    # arithmetic intensity of 0 and an unconditional "memory-bound" verdict.
    if "grouped_gemm" in base:
        if (len(shapes) >= 2 and len(shapes[0]) == 2 and len(shapes[1]) == 3):
            return 2 * shapes[0][0] * shapes[0][1] * shapes[1][2]

    if base == "matmul":
        if len(shapes) >= 2 and shapes[0] and shapes[1]:
            K = shapes[0][-1] if isinstance(shapes[0][-1], int) else 1
            N = shapes[1][-1] if isinstance(shapes[1][-1], int) else 1
            batch = _prod(shapes[0][:-1])
            return 2 * batch * K * N

    # Element-wise: ~1-5 FLOPs per element
    if base in ("mul", "add", "sub", "div"):
        return _prod(shapes[0])
    if base in ("silu", "sigmoid", "tanh", "gelu"):
        return _prod(shapes[0]) * 4
    if base in ("relu"):
        return _prod(shapes[0])
    if base in ("rsqrt", "sqrt", "exp", "log"):
        return _prod(shapes[0]) * 2

    # Fused activation: ~5 FLOPs per element (silu + mul)
    if base in ("silu_and_mul", "mul_and_silu", "gelu_and_mul",
                "gelu_tanh_and_mul", "swigluoai_and_mul", "swiglustep_and_mul"):
        return _prod(shapes[0]) * 5

    # Normalization: ~5 FLOPs per element (mean, variance, normalize)
    if "norm" in base:
        return _prod(shapes[0]) * 5

    # Softmax: ~5 FLOPs per element
    if base in ("softmax", "_softmax", "log_softmax"):
        return _prod(shapes[0]) * 5

    # RoPE: ~6 FLOPs per element
    if "rotary" in base or "rope" in base:
        return _prod(shapes[0]) * 6

    # Topk: ~N*logK per element
    if "topk" in base:
        n = _prod(shapes[0])
        return n * 10  # rough estimate

    # Attention. The dispatched op carries [tokens, heads, head_dim] query and
    # [kv_tokens, kv_heads, head_dim] key/value operands (the key/value rows
    # already rewritten to the full attended length ``S+C`` by the graph
    # reconstruction), which is everything the two matmuls need:
    # QK^T and PV are each ``2 * q_tokens * kv_tokens * n_heads * head_dim``.
    # Without this, attention - normally the single most expensive op in the
    # profile - had *zero* analytic work, so its roofline bound came out
    # "unknown" and it was ranked as if it had 100 % headroom.
    if _is_attention(base):
        return _attention_flops(shapes, n_seqs)

    return 0


_ATTENTION_BASES = ("unified_attention", "flash_attn", "paged_attention",
                    "sparse_attn", "attention_with_output")


def _is_attention(base: str) -> bool:
    return any(k in base for k in _ATTENTION_BASES)


def _attention_flops(shapes: list[list[int]], n_seqs: int = 1) -> int:
    """QK^T + PV for a [tokens, heads, head_dim] attention call.

    Causality is deliberately *not* discounted: a decode step attends the whole
    cached context (no masking at all), and for a prefill over a long cached
    context the masked fraction is small. Halving it would understate the op
    the ranking cares most about.
    """
    q = next((s for s in shapes if len(s) == 3), None)
    if q is None:
        return 0
    kv = next((s for s in shapes[1:] if len(s) == 3), q)
    tokens, heads, dim = q[0], q[1], q[2]
    # Decode reads ``batch x context`` KV rows in total, but each of the batch's
    # queries attends only its own context - dividing by the sequence count is
    # what keeps a batched decode from being charged ``batch`` times its work.
    kv_per_seq = kv[0] / max(n_seqs, 1)
    return int(2 * 2 * tokens * kv_per_seq * heads * dim)
