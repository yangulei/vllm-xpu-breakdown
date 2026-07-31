#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the vLLM-XPU Ops/Kernels Breakdown pipeline.

Tests cover:
  - Model config fetching and summarization (Qwen/Qwen3-4B-Instruct-2507)
  - Op classification (5 backends)
  - Overhead event filtering
  - Shape symbolization with model-specific dimensions
  - Memory and FLOP estimation
  - Layer merging (including pre-aggregated ops)
  - Chrome trace file parsing
  - Full analyzer pipeline
  - Flask API endpoints (demo, model)
  - End-to-end with real trace file (if available)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.analyzer import (
    AnalyzedOp,
    analyze_ops,
    dtype_size,
    estimate_flops,
    estimate_memory,
    merge_layers,
    symbolize_shape,
)
from breakdown.classifier import Backend, classify_op
from breakdown.model_info import fetch_model_config, get_dim_symbols, summarize_config
from breakdown.profiler import _is_overhead_event
from breakdown.trace_parser import parse_trace_file


# ---- Qwen3-4B-Instruct-2507 model constants ----
QWEN3_4B_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN3_4B_EXPECTED = {
    "architecture": "Qwen3ForCausalLM",
    "hidden_size": 2560,
    "num_layers": 36,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 80,
    "intermediate_size": 9728,
    "vocab_size": 151936,
    "is_moe": False,
    "dtype": "bfloat16",
}

HUNYUAN_MOE_LIST_CONFIG = {
    "architectures": ["HunYuanMoEV1ForCausalLM"],
    "model_type": "hunyuan_v1_moe",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 128167,
    "torch_dtype": "bfloat16",
    "num_experts": 64,
    "moe_intermediate_size": [3072] * 32,
    "moe_topk": [8] * 32,
    "num_shared_expert": [1] * 32,
}

# Path to real trace file from profiling run (if available)
_TRACE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "output", "traces")


def _make_mock_trace(ops: list[dict], kernels: list[dict] | None = None) -> dict:
    """Build a chrome trace JSON structure for testing."""
    events = []
    ts = 1000000

    for op in ops:
        events.append({
            "ph": "X",
            "cat": "cpu_op",
            "name": op["name"],
            "ts": ts,
            "dur": op.get("dur", 100),
            "pid": 1,
            "tid": 1,
            "args": op.get("args", {}),
        })
        ts += op.get("dur", 100) + 10

    if kernels:
        kts = 1000050
        for k in kernels:
            events.append({
                "ph": "X",
                "cat": "kernel",
                "name": k["name"],
                "ts": kts,
                "dur": k.get("dur", 50),
                "pid": 0,
                "tid": 0,
            })
            kts += k.get("dur", 50) + 5

    return {"traceEvents": events}


def _get_real_trace_file() -> str | None:
    """Return path to the newest *loadable* real trace file, if any.

    The shared ``output/traces/`` dir may contain partial/corrupt stub files
    (e.g. zero-length or truncated ``.json.gz`` from another in-progress run), so
    skip any file that doesn't parse into a non-empty op list and return the
    newest one that does.
    """
    if not os.path.isdir(_TRACE_DIR):
        return None
    files = [os.path.join(_TRACE_DIR, f) for f in os.listdir(_TRACE_DIR)
             if f.endswith(".json") or f.endswith(".json.gz")]
    for path in sorted(files, key=os.path.getmtime, reverse=True):
        try:
            ops = parse_trace_file(path)
        except Exception:
            continue
        if ops:
            return path
    return None


def _find_multilayer_trace(num_layers: int, scan_limit: int = 12):
    """Find the newest trace whose per-layer ops repeat across ``num_layers``.

    Returns ``(path, parsed_ops)`` for the first (newest) trace that looks like a
    full-model run — some op is called a positive multiple of ``num_layers`` — or
    ``(None, None)`` if none of the newest ``scan_limit`` traces qualify (e.g.
    only reduced-layer profiling traces are present). Selection is based on the
    raw aggregated call count so it stays independent of the analyzer's own
    layer-detection logic under test.
    """
    if not os.path.isdir(_TRACE_DIR):
        return None, None
    files = [os.path.join(_TRACE_DIR, f) for f in os.listdir(_TRACE_DIR)
             if f.endswith(".json") or f.endswith(".json.gz")]
    for path in sorted(files, key=os.path.getmtime, reverse=True)[:scan_limit]:
        try:
            ops = parse_trace_file(path)
        except Exception:
            continue
        if any(o.get("count", 1) >= num_layers
               and o.get("count", 1) % num_layers == 0 for o in ops):
            return path, ops
    return None, None


# ===================================================================
# Model Info Tests
# ===================================================================

class TestModelInfo(unittest.TestCase):
    """Test model config fetching for Qwen/Qwen3-4B-Instruct-2507."""

    def test_fetch_config(self):
        config = fetch_model_config(QWEN3_4B_MODEL_ID)
        self.assertIsInstance(config, dict)
        self.assertIn("hidden_size", config)
        self.assertIn("num_hidden_layers", config)

    def test_summarize_config(self):
        config = fetch_model_config(QWEN3_4B_MODEL_ID)
        summary = summarize_config(config)

        self.assertEqual(summary["architecture"], QWEN3_4B_EXPECTED["architecture"])
        self.assertEqual(summary["hidden_size"], QWEN3_4B_EXPECTED["hidden_size"])
        self.assertEqual(summary["num_layers"], QWEN3_4B_EXPECTED["num_layers"])
        self.assertEqual(summary["num_heads"], QWEN3_4B_EXPECTED["num_heads"])
        self.assertEqual(summary["num_kv_heads"], QWEN3_4B_EXPECTED["num_kv_heads"])
        self.assertEqual(summary["intermediate_size"], QWEN3_4B_EXPECTED["intermediate_size"])
        self.assertEqual(summary["vocab_size"], QWEN3_4B_EXPECTED["vocab_size"])
        self.assertFalse(summary["is_moe"])
        self.assertEqual(summary["dtype"], QWEN3_4B_EXPECTED["dtype"])

    def test_dim_symbols(self):
        config = fetch_model_config(QWEN3_4B_MODEL_ID)
        summary = summarize_config(config)
        dim_symbols = get_dim_symbols(summary)

        self.assertEqual(dim_symbols[2560], "H")
        self.assertEqual(dim_symbols[9728], "I")
        self.assertEqual(dim_symbols[151936], "V")
        self.assertIn(32, dim_symbols)     # n_h
        self.assertIn(8, dim_symbols)      # n_kv

    def test_summarize_config_normalizes_layerwise_moe_lists(self):
        summary = summarize_config(HUNYUAN_MOE_LIST_CONFIG)

        self.assertEqual(summary["moe_intermediate_size"], 3072)
        self.assertEqual(summary["num_experts_per_tok"], 8)
        self.assertEqual(summary["n_shared_experts"], 1)


# ===================================================================
# Classifier Tests
# ===================================================================

class TestClassifier(unittest.TestCase):

    def test_vllm_xpu_kernels_stripped(self):
        backend, _ = classify_op("rms_norm")
        self.assertEqual(backend, Backend.VLLM_XPU_KERNELS)

    def test_vllm_xpu_kernels_namespaced(self):
        backend, _ = classify_op("_C::silu_and_mul")
        self.assertEqual(backend, Backend.VLLM_XPU_KERNELS)

    def test_vllm_cache_ops(self):
        backend, _ = classify_op("_C_cache_ops::reshape_and_cache_flash")
        self.assertEqual(backend, Backend.VLLM_XPU_KERNELS)

    def test_vllm_moe_ops(self):
        backend, _ = classify_op("_moe_C::moe_align_block_size")
        self.assertEqual(backend, Backend.VLLM_XPU_KERNELS)

    def test_vllm_xpu_specific(self):
        backend, _ = classify_op("_xpu_C::flash_attn_varlen_func")
        self.assertEqual(backend, Backend.VLLM_XPU_KERNELS)

    def test_vllm_namespace_dispatch_ops(self):
        # vLLM registered dispatch ops (attention core, kv-cache, MoE, sampler)
        # run vllm-xpu-kernels on XPU.
        for name in ("vllm::unified_attention_with_output",
                     "vllm::unified_kv_cache_update",
                     "vllm::moe_forward_shared",
                     "vllm::xpu_topk_topp_sampler"):
            backend, _ = classify_op(name, device_type="xpu",
                                     device_time_us=10.0)
            self.assertEqual(backend, Backend.VLLM_XPU_KERNELS, name)

    def test_triton_prefix(self):
        backend, _ = classify_op("triton_flash_attn_fwd")
        self.assertEqual(backend, Backend.TRITON)

    def test_triton_compiled_graph(self):
        backend, _ = classify_op("CompiledFxGraph_123")
        self.assertEqual(backend, Backend.TRITON)

    def test_flashinfer_rmsnorm_kernel(self):
        # Real synthetic op name from MiniMax-M3 CUDA profiling: FlashInfer
        # RMSNorm is launched directly from Python (no aten/_C cpu_op), so the
        # kernel symbol embeds "flashinfer" under the "triton::" synthetic prefix.
        name = ("triton::kernel_cutlass_kernel_flashinfernormkernelsrmsnorm"
                "RMSNormKernel_object_at__tensorptrbf16gmemalign128oi646144"
                "61441_tensorptrbf16gmemalign16o61441___T_0")
        backend, cat = classify_op(name, device_type="cuda", device_time_us=50)
        self.assertEqual(backend, Backend.FLASHINFER)
        self.assertEqual(cat, "flashinfer-kernel")

    def test_flashinfer_fused_add_rmsnorm_kernel(self):
        name = ("triton::kernel_cutlass_kernel_flashinfernormkernels"
                "fused_add_rmsnormFusedAddRMSNormKernel_object_at__T_0")
        backend, _ = classify_op(name, device_type="cuda", device_time_us=50)
        self.assertEqual(backend, Backend.FLASHINFER)

    def test_triton_fused_kernel(self):
        # Real kernel name from Qwen3 profiling
        backend, _ = classify_op("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0")
        self.assertEqual(backend, Backend.TRITON)

    def test_torch_xpu_ops_with_device_time(self):
        backend, _ = classify_op("aten::mm", device_time_us=100)
        self.assertEqual(backend, Backend.TORCH_XPU_OPS)

    def test_torch_xpu_ops_with_self_device_time(self):
        backend, _ = classify_op("aten::linear", self_device_time_us=50)
        self.assertEqual(backend, Backend.TORCH_XPU_OPS)

    def test_torch_xpu_ops_with_device_type(self):
        backend, _ = classify_op("aten::relu", device_type="xpu")
        self.assertEqual(backend, Backend.TORCH_XPU_OPS)

    def test_aten_compute_no_device_time_is_cpu(self):
        backend, _ = classify_op("aten::mm", device_time_us=0,
                                 self_device_time_us=0)
        self.assertEqual(backend, Backend.CPU)

    def test_framework_view_ops(self):
        for op in ["aten::view", "aten::reshape", "aten::permute",
                    "aten::transpose", "aten::contiguous"]:
            backend, _ = classify_op(op)
            self.assertEqual(backend, Backend.FRAMEWORK, f"{op} should be framework")

    def test_framework_memory_ops(self):
        for op in ["aten::empty", "aten::zeros", "aten::clone", "aten::to"]:
            backend, _ = classify_op(op)
            self.assertEqual(backend, Backend.FRAMEWORK, f"{op} should be framework")

    def test_ccl_collective_ops(self):
        # Collective-communication (oneCCL/NCCL) calls form their own category.
        for op in ["c10d::allreduce_", "c10d::allgather_",
                   "c10d::reduce_scatter_", "c10d::_allgather_base_",
                   "vllm::all_reduce", "all_to_all"]:
            backend, cat = classify_op(op, device_type="xpu",
                                       device_time_us=10.0)
            self.assertEqual(backend, Backend.CCL, f"{op} should be ccl")
            self.assertEqual(cat, "collective-comm")

    def test_ccl_does_not_capture_moe_gather(self):
        # Bare gather/scatter (cache/MoE) must not be misfiled as CCL.
        for op in ["moe_gather", "gather_cache"]:
            backend, _ = classify_op(op, device_type="xpu",
                                     device_time_us=10.0)
            self.assertNotEqual(backend, Backend.CCL, f"{op} should not be ccl")


# ===================================================================
# Overhead Filter Tests
# ===================================================================

class TestOverheadFilter(unittest.TestCase):

    def test_profiler_step_filtered(self):
        self.assertTrue(_is_overhead_event("ProfilerStep*"))

    def test_profiler_internal_filtered(self):
        self.assertTrue(_is_overhead_event("profiler::_record_function_enter"))
        self.assertTrue(_is_overhead_event("autograd::engine::evaluate"))

    def test_xpu_runtime_filtered(self):
        self.assertTrue(_is_overhead_event("urEnqueueKernelLaunch"))

    def test_xpu_dispatch_filtered(self):
        self.assertTrue(_is_overhead_event("at::native::xpu::ClampScalarFunctor"))

    def test_sycl_template_kernel_filtered(self):
        self.assertTrue(_is_overhead_event(
            "at::native::xpu::VectorizedElementwiseKernel<4, at::native::xpu::foo, int>"))

    def test_real_ops_not_filtered(self):
        for name in ["aten::mm", "rms_norm", "_C::silu_and_mul",
                      "triton_flash_attn_fwd", "aten::embedding",
                      "_xpu_C::flash_attn_varlen_func",
                      "_C_cache_ops::reshape_and_cache_flash",
                      "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0",
                      "## Call CompiledFxGraph foo ##"]:
            self.assertFalse(_is_overhead_event(name),
                             f"{name} should NOT be filtered")


# ===================================================================
# Shape Symbolization Tests
# ===================================================================

class TestShapeSymbolization(unittest.TestCase):

    def setUp(self):
        config = fetch_model_config(QWEN3_4B_MODEL_ID)
        summary = summarize_config(config)
        self.dim_symbols = get_dim_symbols(summary)

    def test_hidden_size(self):
        shape = symbolize_shape([128, 2560], self.dim_symbols,
                                batch_size=1, seq_len=128)
        self.assertEqual(shape, ["S", "H"])

    def test_attention_qkv(self):
        # Use seq_len=256 to avoid collision with head_dim=128
        shape = symbolize_shape([1, 256, 32, 128], self.dim_symbols,
                                batch_size=1, seq_len=256)
        self.assertEqual(shape, ["B", "S", "n_h", "d"])

    def test_kv_heads(self):
        shape = symbolize_shape([1, 256, 8, 128], self.dim_symbols,
                                batch_size=1, seq_len=256)
        self.assertEqual(shape, ["B", "S", "n_kv", "d"])

    def test_intermediate(self):
        shape = symbolize_shape([128, 9728], self.dim_symbols,
                                batch_size=1, seq_len=128)
        self.assertEqual(shape, ["S", "I"])

    def test_vocab(self):
        shape = symbolize_shape([151936, 2560], self.dim_symbols)
        self.assertEqual(shape, ["V", "H"])

    def test_unknown_dims_preserved(self):
        shape = symbolize_shape([7, 42], self.dim_symbols)
        self.assertEqual(shape, [7, 42])


# ===================================================================
# Memory & FLOP Estimation Tests
# ===================================================================

class TestEstimation(unittest.TestCase):

    def test_matmul_memory(self):
        mem = estimate_memory("aten::mm",
                              [[128, 2560], [2560, 2560]], dtype_bytes=2)
        expected_read = (128*2560 + 2560*2560) * 2
        expected_write = 128*2560 * 2
        self.assertEqual(mem, expected_read + expected_write)

    def test_rms_norm_memory(self):
        mem = estimate_memory("rms_norm", [[128, 2560], [2560]], dtype_bytes=2)
        self.assertEqual(mem, 128 * 2560 * 2 * 3)

    def test_silu_and_mul_memory(self):
        mem = estimate_memory("silu_and_mul", [[128, 19456]], dtype_bytes=2)
        self.assertEqual(mem, 128 * 19456 * 2 * 2)

    def test_empty_shapes(self):
        self.assertEqual(estimate_memory("aten::mm", []), 0)

    def test_matmul_flops(self):
        flops = estimate_flops("aten::mm", [[128, 2560], [2560, 2560]])
        self.assertEqual(flops, 2 * 128 * 2560 * 2560)

    def test_addmm_flops(self):
        flops = estimate_flops("aten::addmm",
                               [[2560], [128, 2560], [2560, 9728]])
        expected = 2 * 128 * 2560 * 9728 + 128 * 9728
        self.assertEqual(flops, expected)

    def test_norm_flops(self):
        flops = estimate_flops("rms_norm", [[128, 2560]])
        self.assertEqual(flops, 128 * 2560 * 5)


# ===================================================================
# Layer Merging Tests
# ===================================================================

class TestLayerMerging(unittest.TestCase):

    def test_merge_multiple_entries(self):
        """Multiple identical AnalyzedOp entries merge into one."""
        ops = [AnalyzedOp(name="aten::mm", backend="torch-xpu-ops",
                          category="aten-xpu",
                          input_shapes=[["S", "H"], ["H", "H"]],
                          input_shapes_raw=[[128, 2560], [2560, 2560]],
                          dtype="bfloat16", call_count=1, device_time_us=200)
               for _ in range(36)]

        merged = merge_layers(ops, num_layers=36)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].layer_count, 36)
        self.assertEqual(merged[0].call_count, 36)

    def test_detect_layer_in_preaggregated(self):
        """Pre-aggregated op with count=252 (7 iters × 36 layers) → ×36."""
        ops = [AnalyzedOp(name="aten::mm", backend="torch-xpu-ops",
                          category="aten-xpu",
                          input_shapes=[["B", "H"], ["H", "QKV"]],
                          input_shapes_raw=[[1, 2560], [2560, 6144]],
                          dtype="bfloat16", call_count=252, device_time_us=1000)]

        merged = merge_layers(ops, num_layers=36)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].layer_count, 36)

    def test_no_merge_different_shapes(self):
        ops = [
            AnalyzedOp(name="aten::mm", backend="torch-xpu-ops",
                       category="aten-xpu",
                       input_shapes=[["S", "H"], ["H", "H"]],
                       input_shapes_raw=[[128, 2560], [2560, 2560]],
                       dtype="bfloat16", device_time_us=200),
            AnalyzedOp(name="aten::mm", backend="torch-xpu-ops",
                       category="aten-xpu",
                       input_shapes=[["S", "H"], ["H", "I"]],
                       input_shapes_raw=[[128, 2560], [2560, 9728]],
                       dtype="bfloat16", device_time_us=400),
        ]
        merged = merge_layers(ops, num_layers=36)
        self.assertEqual(len(merged), 2)

    def test_single_call_no_layer_count(self):
        """Ops with small call counts shouldn't get layer_count."""
        ops = [AnalyzedOp(name="aten::embedding", backend="torch-xpu-ops",
                          category="aten-xpu",
                          input_shapes=[["V", "H"], ["S"]],
                          input_shapes_raw=[[151936, 2560], [128]],
                          dtype="bfloat16", call_count=1, device_time_us=100)]

        merged = merge_layers(ops, num_layers=36)
        self.assertEqual(merged[0].layer_count, 1)


# ===================================================================
# Trace Parser Tests
# ===================================================================

class TestTraceParser(unittest.TestCase):

    def test_parse_cpu_ops(self):
        trace = _make_mock_trace([
            {"name": "aten::mm", "dur": 500,
             "args": {"Input Dims": [[128, 2560], [2560, 2560]]}},
            {"name": "aten::mm", "dur": 300,
             "args": {"Input Dims": [[128, 2560], [2560, 9728]]}},
            {"name": "rms_norm", "dur": 100,
             "args": {"Input Dims": [[128, 2560], [2560]]}},
        ])
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(trace, f)
            path = f.name
        try:
            ops = parse_trace_file(path)
            self.assertGreater(len(ops), 0)
            names = [op["name"] for op in ops]
            self.assertIn("aten::mm", names)
            self.assertIn("rms_norm", names)
        finally:
            os.unlink(path)

    def test_overhead_filtered_in_trace(self):
        trace = _make_mock_trace([
            {"name": "ProfilerStep*", "dur": 10000},
            {"name": "aten::mm", "dur": 500},
            {"name": "profiler::_record_function_enter", "dur": 5},
        ])
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(trace, f)
            path = f.name
        try:
            ops = parse_trace_file(path)
            names = [op["name"] for op in ops]
            self.assertNotIn("ProfilerStep*", names)
            self.assertIn("aten::mm", names)
        finally:
            os.unlink(path)

    def test_device_time_attribution(self):
        trace = _make_mock_trace(
            ops=[{"name": "aten::mm", "dur": 500},
                 {"name": "rms_norm", "dur": 100}],
            kernels=[{"name": "gemm_kernel", "dur": 1000},
                     {"name": "norm_kernel", "dur": 200}],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(trace, f)
            path = f.name
        try:
            ops = parse_trace_file(path)
            for op in ops:
                if op["name"] in ("aten::mm", "rms_norm"):
                    self.assertGreater(op["device_time_us"], 0,
                                       f"{op['name']} should have device time")
        finally:
            os.unlink(path)

    def test_gzip_trace(self):
        import gzip as gz
        trace = _make_mock_trace([{"name": "aten::mm", "dur": 500}])
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name
            with gz.open(path, "wt") as gf:
                json.dump(trace, gf)
        try:
            ops = parse_trace_file(path)
            self.assertEqual(len(ops), 1)
        finally:
            os.unlink(path)

    def test_empty_trace(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"traceEvents": []}, f)
            path = f.name
        try:
            self.assertEqual(parse_trace_file(path), [])
        finally:
            os.unlink(path)


# ===================================================================
# Full Analyzer Pipeline Tests
# ===================================================================

class TestAnalyzerPipeline(unittest.TestCase):

    def setUp(self):
        config = fetch_model_config(QWEN3_4B_MODEL_ID)
        self.summary = summarize_config(config)
        self.dim_symbols = get_dim_symbols(self.summary)

    def test_analyze_produces_all_backends(self):
        mock_ops = [
            {"name": "aten::mm", "backend": "torch-xpu-ops", "category": "aten-xpu",
             "count": 36, "input_shapes": "[[1, 2560], [2560, 6144]]",
             "device_time_us": 1000, "cpu_time_us": 200},
            {"name": "rms_norm", "backend": "vllm-xpu-kernels",
             "category": "vllm-xpu-kernels (general)", "count": 72,
             "input_shapes": "[[1, 2560], [2560]]",
             "device_time_us": 500, "cpu_time_us": 100},
            {"name": "triton_flash_attn_fwd", "backend": "triton",
             "category": "triton-compiled", "count": 36,
             "input_shapes": "[[1, 32, 80], [1, 8, 80], [1, 8, 80]]",
             "device_time_us": 2000, "cpu_time_us": 500},
        ]
        analyzed = analyze_ops(mock_ops, dim_symbols=self.dim_symbols,
                               batch_size=1, seq_len=None,
                               model_dtype="bfloat16", num_layers=36)
        backends = {op.backend for op in analyzed}
        self.assertIn("vllm-xpu-kernels", backends)
        self.assertIn("torch-xpu-ops", backends)
        self.assertIn("triton", backends)

    def test_symbolic_shapes_present(self):
        mock_ops = [
            {"name": "aten::mm", "backend": "torch-xpu-ops", "category": "aten-xpu",
             "count": 1, "input_shapes": "[[1, 2560], [2560, 9728]]",
             "device_time_us": 100, "cpu_time_us": 20},
        ]
        analyzed = analyze_ops(mock_ops, dim_symbols=self.dim_symbols,
                               batch_size=1, seq_len=None, model_dtype="bfloat16")
        self.assertTrue(any(isinstance(d, str)
                            for s in analyzed[0].input_shapes for d in s))

    def test_arithmetic_intensity_matmul(self):
        mock_ops = [
            {"name": "aten::mm", "backend": "torch-xpu-ops", "category": "aten-xpu",
             "count": 1, "input_shapes": "[[128, 2560], [2560, 2560]]",
             "device_time_us": 100, "cpu_time_us": 20},
        ]
        analyzed = analyze_ops(mock_ops, dim_symbols=self.dim_symbols,
                               batch_size=1, seq_len=128, model_dtype="bfloat16")
        self.assertGreater(analyzed[0].arithmetic_intensity, 0)
        self.assertGreater(analyzed[0].flops, 0)
        self.assertGreater(analyzed[0].memory_bytes, 0)

    def test_to_dict_json_serializable(self):
        mock_ops = [
            {"name": "aten::mm", "backend": "torch-xpu-ops", "category": "aten-xpu",
             "count": 36, "input_shapes": "[[1, 2560], [2560, 6144]]",
             "device_time_us": 1000, "cpu_time_us": 200},
        ]
        analyzed = analyze_ops(mock_ops, dim_symbols=self.dim_symbols,
                               batch_size=1, seq_len=None, model_dtype="bfloat16",
                               num_layers=36)
        for op in analyzed:
            d = op.to_dict()
            for field in ("name", "backend", "input_shapes", "dtype",
                          "call_count", "layer_count", "device_time_us",
                          "memory_bytes", "flops", "arithmetic_intensity"):
                self.assertIn(field, d)
            json.dumps(d)  # must not raise


# ===================================================================
# Flask API Tests
# ===================================================================

class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_index_page(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"vLLM Ops/Kernels Breakdown", resp.data)

    def test_model_endpoint_qwen3(self):
        resp = self.client.get(f"/api/model/{QWEN3_4B_MODEL_ID}")
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["summary"]["architecture"], "Qwen3ForCausalLM")
        self.assertEqual(data["summary"]["hidden_size"], 2560)
        self.assertEqual(data["summary"]["num_layers"], 36)

    def test_model_endpoint_invalid(self):
        resp = self.client.get("/api/model/nonexistent/model-xyz-999")
        self.assertNotEqual(resp.status_code, 200)


# ===================================================================
# Query/Context Len profiling (Method 1: APC prefix)
# ===================================================================

class TestQueryContextProfiling(unittest.TestCase):
    """Wiring for the Query Len / Context Len prefix-cache profiling knobs."""

    def test_make_token_ids_deterministic_and_valid(self):
        from app import _make_token_ids
        a = _make_token_ids(48, 32000, seed=0)
        b = _make_token_ids(48, 32000, seed=0)
        c = _make_token_ids(48, 32000, seed=7)
        self.assertEqual(a, b)               # same seed -> identical prefix
        self.assertNotEqual(a, c)            # different seed -> different query
        self.assertEqual(len(a), 48)
        self.assertTrue(all(256 <= x < 32000 - 256 for x in a))
        self.assertEqual(_make_token_ids(0, 32000, 0), [])

    def test_get_block_size_and_vocab_fallbacks(self):
        from app import _get_block_size, _get_vocab_size

        class _CC:
            block_size = 64

        class _Cfg:
            cache_config = _CC()

        class _Eng:
            vllm_config = _Cfg()

        class _LLM:
            llm_engine = _Eng()

        self.assertEqual(_get_block_size(_LLM()), 64)
        self.assertEqual(_get_block_size(object(), default=16), 16)
        self.assertEqual(_get_vocab_size(object(), {"vocab_size": 40000}), 40000)
        self.assertEqual(_get_vocab_size(object(), {}, default=32000), 32000)

    def test_start_profile_passes_query_context_and_bumps_maxlen(self):
        import app as app_module
        captured = {}

        def _fake_thread(target=None, args=(), daemon=None):
            captured["args"] = args

            class _T:
                def start(self_inner):
                    pass
            return _T()

        client = app_module.app.test_client()
        with patch.object(app_module.threading, "Thread", _fake_thread), \
                patch.object(app_module, "_profile_state",
                             {"status": "idle", "result": None, "error": None,
                              "model_id": None}):
            resp = client.post("/api/profile", json={
                "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                "query_len": 2048,
                "context_len": 2048,
                "max_model_len": 4096,   # query+context, no decode headroom
                "max_tokens": 128,
            })
        self.assertTrue(resp.get_json()["ok"])
        args = captured["args"]
        # signature: (model_id, mode, max_model_len, batch_size, max_tokens,
        #  prompt, num_profile_layers, tp_size, quantization, gpu_mem,
        #  query_len, context_len, prefill_batch_size, decode_batch_size)
        self.assertEqual(args[-4], 2048)     # query_len
        self.assertEqual(args[-3], 2048)     # context_len
        max_model_len = args[2]
        # must be bumped to cover context + query + decode budget
        self.assertGreaterEqual(max_model_len, 2048 + 2048 + 128)


# ===================================================================
# End-to-End with Real Trace (if available)
# ===================================================================

class TestRealTrace(unittest.TestCase):
    """Tests using the real Qwen3-4B trace file from a profiling run."""

    def setUp(self):
        self.trace_file = _get_real_trace_file()
        if not self.trace_file:
            self.skipTest("No real trace file available in output/traces/")

        config = fetch_model_config(QWEN3_4B_MODEL_ID)
        self.summary = summarize_config(config)
        self.dim_symbols = get_dim_symbols(self.summary)

    def test_parse_ops_not_empty(self):
        ops = parse_trace_file(self.trace_file)
        self.assertGreater(len(ops), 10,
                           "Real trace should contain many ops")

    def test_has_all_backends(self):
        ops = parse_trace_file(self.trace_file)
        backends = {op["backend"] for op in ops}
        self.assertIn("vllm-xpu-kernels", backends,
                       "Should have vllm-xpu-kernels ops")
        self.assertIn("torch-xpu-ops", backends,
                       "Should have torch-xpu-ops ops")
        # triton only present in torch.compile mode traces

    def test_has_expected_ops(self):
        ops = parse_trace_file(self.trace_file)
        names = {op["name"] for op in ops}
        self.assertIn("aten::mm", names, "Should have aten::mm")
        self.assertIn("_C_cache_ops::reshape_and_cache_flash", names,
                       "Should have cache ops")

    def test_full_pipeline_produces_analyzed_ops(self):
        # Layer merging only applies to full-model traces, where each per-layer
        # op repeats once per layer. Reduced-layer profiling traces
        # (``num_profile_layers`` < ``num_layers``) legitimately contain no
        # multi-layer ops, so pick a genuinely multi-layer trace here — the
        # shared ``output/traces/`` dir may hold either kind. Selection uses the
        # raw per-op call count (a signal independent of the ``layer_count``
        # field the analyzer computes), so an analyzer regression that fails to
        # detect layer repetition still fails this test rather than skipping it.
        trace_file, ops = _find_multilayer_trace(num_layers=36)
        if not trace_file:
            self.skipTest("No full-model (multi-layer) trace available")

        analyzed = analyze_ops(ops, dim_symbols=self.dim_symbols,
                               batch_size=1, seq_len=None,
                               model_dtype="bfloat16", num_layers=36)
        self.assertGreater(len(analyzed), 0)

        # Check that layer_count > 1 for per-layer ops
        layer_ops = [o for o in analyzed if o.layer_count > 1]
        self.assertGreater(len(layer_ops), 0,
                           "Should detect per-layer ops with ×36")

    def test_symbolic_shapes_in_real_trace(self):
        ops = parse_trace_file(self.trace_file)
        analyzed = analyze_ops(ops, dim_symbols=self.dim_symbols,
                               batch_size=1, seq_len=None,
                               model_dtype="bfloat16", num_layers=36)
        has_symbol = False
        for op in analyzed:
            for shape in op.input_shapes:
                for dim in shape:
                    if isinstance(dim, str) and dim in ("H", "I", "V", "QKV",
                                                        "n_h", "n_kv", "d"):
                        has_symbol = True
        self.assertTrue(has_symbol, "Real trace should have symbolic dims")

    def test_json_serializable(self):
        ops = parse_trace_file(self.trace_file)
        analyzed = analyze_ops(ops, dim_symbols=self.dim_symbols,
                               batch_size=1, seq_len=None,
                               model_dtype="bfloat16", num_layers=36)
        result = [o.to_dict() for o in analyzed]
        # Must not raise
        serialized = json.dumps(result)
        self.assertGreater(len(serialized), 100)


# ===================================================================
# Dtype Tests
# ===================================================================

class TestDtypeSize(unittest.TestCase):

    def test_bfloat16(self):
        self.assertEqual(dtype_size("bfloat16"), 2)

    def test_float32(self):
        self.assertEqual(dtype_size("float32"), 4)

    def test_fp8(self):
        self.assertEqual(dtype_size("float8_e4m3fn"), 1)

    def test_torch_prefix(self):
        self.assertEqual(dtype_size("torch.bfloat16"), 2)


# ===================================================================
# Profile-first graph reconstruction (build_graph_from_trace)
# ===================================================================


def _synthetic_trace(steps):
    """Build a minimal chrome-trace dict for reconstruction tests.

    ``steps`` is a list of ``tokens`` (int); each produces one model forward
    with an embedding + two identical decoder layers (each a linear op). Kernel
    device time is linked to each cpu_op through the correlation → runtime →
    External id chain, exactly like real XPU traces.
    """
    events = []
    ext = [0]
    corr = [0]
    tid = 7

    def kernel_for(ext_id, ts, dur):
        corr[0] += 1
        c = corr[0]
        events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
                       "ts": ts, "dur": 0.1, "name": "urEnqueueKernelLaunch",
                       "args": {"correlation": c, "External id": ext_id}})
        events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                       "ts": ts + 1000, "dur": dur, "name": "gemm_xpu_kernel",
                       "args": {"correlation": c}})

    def op(name, ts, dur, shapes, kdur):
        ext[0] += 1
        e = ext[0]
        events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                       "ts": ts, "dur": dur, "name": name,
                       "args": {"External id": e, "Input Dims": shapes,
                                "Input type": ["c10::BFloat16", "c10::BFloat16"]}})
        if kdur:
            kernel_for(e, ts, kdur)

    def module(cls, ts, dur):
        events.append({"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
                       "ts": ts, "dur": dur, "name": f"nn.Module: {cls}"})

    t = 0.0
    for si, tokens in enumerate(steps):
        base = t
        module(f"TinyForCausalLM_{si}", base, 100)
        module(f"TinyModel_{si}", base + 1, 98)
        module(f"VocabParallelEmbedding_{si}", base + 2, 4)
        op("aten::embedding", base + 3, 2, [[32000, 16], [tokens]], 0)
        for li in range(2):
            lts = base + 10 + li * 40
            module(f"TinyDecoderLayer_{li}", lts, 38)
            module(f"TinyAttention_{li}", lts + 1, 16)
            op("aten::linear", lts + 2, 8,
               [[tokens, 16], [48, 16]], 5.0)  # qkv proj → kernel 5us
            module(f"TinyMLP_{li}", lts + 20, 16)
            op("aten::linear", lts + 21, 8,
               [[tokens, 16], [64, 16]], 7.0)  # mlp → kernel 7us
        t = base + 120
    return {"traceEvents": events}


class TestGraphFromTrace(unittest.TestCase):
    """Reconstruct a model graph directly from a profiler trace."""

    SUMMARY = {
        "architecture": "TinyForCausalLM", "hidden_size": 16, "num_heads": 3,
        "num_kv_heads": 3, "head_dim": 16, "intermediate_size": 64,
        "vocab_size": 32000, "num_layers": 2, "dtype": "bfloat16",
    }

    def _build(self, steps):
        from breakdown.graph_from_trace import build_graph_from_trace
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(_synthetic_trace(steps), f)
            path = f.name
        try:
            return build_graph_from_trace(path, self.SUMMARY,
                                          tp_size=1, batch_size=1)
        finally:
            os.unlink(path)

    def test_reconstructs_prefill_and_decode(self):
        g = self._build([8, 1, 1])  # 1 prefill (8 tokens) + 2 decode (1 token)
        self.assertIsNotNone(g["prefill"])
        self.assertIsNotNone(g["decode"])
        self.assertTrue(g["has_timing"])
        self.assertEqual(g["timing_method"], "trace_reconstruction")

    def test_phase_token_symbols(self):
        g = self._build([8, 1, 1])
        self.assertEqual(g["symbols"]["S"], 8)   # prefill token dim
        self.assertEqual(g["symbols"]["B"], 1)   # decode token dim

    def test_repeated_layers_collapsed(self):
        g = self._build([8])
        # Model → embed + collapsed decoder layer (×2) + (no final norm here)
        model = g["prefill"]["children"][0]
        layer_nodes = [c for c in model["children"]
                       if c["module_type"] == "TinyDecoderLayer"]
        self.assertEqual(len(layer_nodes), 1)
        self.assertEqual(layer_nodes[0]["repeat_count"], 2)

    def test_device_time_attributed_via_correlation(self):
        g = self._build([8])
        model = g["prefill"]["children"][0]
        layer = next(c for c in model["children"]
                     if c["module_type"] == "TinyDecoderLayer")
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "TinyAttention")
        linear = next(o for o in attn["ops"] if o["name"] == "aten::linear")
        self.assertAlmostEqual(linear["device_time_us"], 5.0, places=3)

    def test_ops_and_children_interleaved_by_execution_order(self):
        # A direct op of a module that executes *between* two child modules must
        # render between them (via its ``order`` field), not be hoisted above
        # both children. This mirrors the MiniMax-M3 case where the decoder
        # layer's post-attention ``c10d::allreduce_`` (a direct op, launched
        # after the attention submodule) was floating to the top of the layer.
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 200)
        mod("TinyDecoderLayer", 0, 1.0, 150)
        mod("TinyAttention", 0, 2.0, 30)          # child module (order 0)
        op("aten::linear", 3.0, 8, [[tokens, 16], [48, 16]], 5.0)
        # Direct op of the layer, launched *after* the attention submodule
        # closes (ts 50 > attention end 32) — floats up to the layer.
        op("c10d::allreduce_", 50.0, 8, [], 3.0)  # layer direct op (order 1)
        mod("TinyMLP", 0, 60.0, 40)               # child module (order 2)
        op("aten::linear", 61.0, 8, [[tokens, 16], [64, 16]], 7.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        layer = find(g["prefill"], "TinyDecoderLayer")
        self.assertIsNotNone(layer)
        allreduce = next(o for o in layer["ops"]
                         if o["name"] == "c10d::allreduce_")
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "TinyAttention")
        mlp = next(c for c in layer["children"]
                   if c["module_type"] == "TinyMLP")
        # Every direct op / child carries an execution-order index.
        for item in (allreduce, attn, mlp):
            self.assertIn("order", item)
        # The allreduce op must sort between attention and mlp, not before both.
        self.assertLess(attn["order"], allreduce["order"])
        self.assertLess(allreduce["order"], mlp["order"])

    def test_repeated_same_signature_op_kept_distinct(self):
        # A single module instance can dispatch the *same* op (identical name +
        # shapes) twice. This mirrors a TP MiniMax-M3 decoder layer, which runs
        # two identical ``c10d::allreduce_`` residual reductions: the post-MLP
        # one of the previous layer time-contains at *this* layer's start
        # (before ``input_layernorm``), and this layer's own post-attention one
        # sits before ``post_attention_layernorm``. Grouping ops only by
        # (name, shapes) collapsed both into one node at the leading position,
        # hiding the post-attention allreduce. Both must now survive, ordered.
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        tid = 7

        def op(name, ts, dur, shapes):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 200)
        mod("TinyDecoderLayer", 0, 1.0, 150)
        # Leading allreduce (previous layer's post-MLP residual, landed here).
        op("c10d::allreduce_", 2.0, 3, [[tokens, 16]])
        mod("TinyInputNorm", 0, 6.0, 5)
        mod("TinyAttention", 0, 12.0, 30)
        op("aten::mm", 13.0, 8, [[tokens, 16], [16, 16]])  # sets token dim
        # This layer's own post-attention allreduce — same name + shapes.
        op("c10d::allreduce_", 50.0, 3, [[tokens, 16]])
        mod("TinyPostNorm", 0, 55.0, 5)
        mod("TinyMLP", 0, 62.0, 40)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        layer = find(g["prefill"], "TinyDecoderLayer")
        self.assertIsNotNone(layer)
        allreduces = [o for o in layer["ops"]
                      if o["name"] == "c10d::allreduce_"]
        # Both identical-signature allreduces are preserved as distinct ops.
        self.assertEqual(len(allreduces), 2)
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "TinyAttention")
        post = next(c for c in layer["children"]
                    if c["module_type"] == "TinyPostNorm")
        # The second (post-attention) allreduce must render between attention
        # and post_attention_layernorm — not be merged away to the top.
        orders = sorted(o["order"] for o in allreduces)
        self.assertLess(orders[0], attn["order"])   # leading one is first
        self.assertLess(attn["order"], orders[1])   # post-attn one after attn
        self.assertLess(orders[1], post["order"])   # ...and before post-norm

    def test_module_wrapped_in_fused_op_is_hoisted(self):
        # vLLM dispatches the MoE block as a single fused custom op
        # (``vllm::moe_forward_shared``) whose implementation internally calls
        # the ``shared_experts`` MLP forward, so by time-containment the whole
        # MLP module subtree nests *under the op event*. Reconstruction only
        # surfaces a module's direct child modules/ops, so the wrapped MLP (and
        # its gate_up/down projections) used to vanish from the graph. The hoist
        # must lift it out to sit beside the op as a child of ``FusedMoE``, with
        # its own device time counted once (moved out of the op's rollup).
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur=0.0):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 300)
        mod("TinyDecoderLayer", 0, 1.0, 250)
        mod("TinyMoE", 0, 5.0, 240)
        op("aten::mm", 6.0, 4, [[tokens, 16], [16, 16]], 2.0)  # router/gate
        # Fused MoE op wrapping the shared-experts MLP module (nested under it).
        mod("FusedMoE", 0, 20.0, 200)
        op("vllm::moe_forward_shared", 22.0, 180, [[tokens, 16], [tokens, 4]],
           9.0)                                     # the op itself (routed moe)
        mod("TinyMLP", 0, 40.0, 120)                # shared_experts, WRAPPED
        op("aten::linear", 42.0, 8, [[tokens, 16], [64, 16]], 7.0)  # gate_up
        op("aten::linear", 60.0, 8, [[tokens, 32], [16, 32]], 6.0)  # down
        # Back at FusedMoE level, after the op closes: the TP reduce.
        op("c10d::allreduce_", 210.0, 5, [[tokens, 16]], 3.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        fused = find(g["prefill"], "FusedMoE")
        self.assertIsNotNone(fused)
        # The wrapped MLP was hoisted out of the op to a child of FusedMoE.
        mlp = next((c for c in fused["children"]
                    if c["module_type"] == "TinyMLP"), None)
        self.assertIsNotNone(mlp)
        # It kept its own projection ops.
        self.assertEqual(sum(1 for o in mlp["ops"]
                             if o["name"] == "aten::linear"), 2)
        # Device time is conserved and counted once: the op no longer includes
        # the MLP's kernels (7+6=13us), only its own (9us); the MLP holds 13us.
        moe_op = next(o for o in fused["ops"]
                      if o["name"] == "vllm::moe_forward_shared")
        self.assertAlmostEqual(moe_op["device_time_us"], 9.0, places=2)
        self.assertAlmostEqual(mlp["total_device_time_us"], 13.0, places=2)
        # Execution order: op (0) → hoisted MLP (1) → reduce (2).
        reduce_op = next(o for o in fused["ops"]
                         if o["name"] == "c10d::allreduce_")
        self.assertLess(moe_op["order"], mlp["order"])
        self.assertLess(mlp["order"], reduce_op["order"])

    def test_moe_router_and_experts_surfaced_from_functional_frames(self):
        # vLLM runs the MoE router (``fused_topk_bias``) and expert compute
        # (``xpu_fused_moe``) as plain ``python_function`` frames inside the
        # fused ``vllm::moe_forward_shared`` op — not as ``nn.Module`` forwards.
        # Without promoting those frames to synthetic module boundaries, their
        # ops (topk/gather routing, grouped-GEMM experts) collapse into the
        # single ``moe_forward_shared`` op node and the ``FusedMoE`` graph shows
        # neither the router nor the experts. They must surface as ``router`` and
        # ``moe`` children of ``FusedMoE`` (order: router → moe), with their ops
        # and device time moved out of the wrapping op.
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur=0.0):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        def pyfn(name, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur, "name": name})

        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 400)
        mod("TinyDecoderLayer", 0, 1.0, 350)
        mod("TinyMoE", 0, 5.0, 340)
        mod("FusedMoE", 0, 20.0, 300)
        # Fused MoE custom op wrapping the router + experts (nested under it).
        op("vllm::moe_forward_shared", 22.0, 280, [[tokens, 16], [tokens, 4]],
           5.0)
        # Router: fused_topk_bias python_function wrapping topk/gather.
        pyfn(".../fused_moe/router/fused_topk_bias_router.py(100): "
             "fused_topk_bias", 40.0, 60)
        op("aten::topk", 42.0, 20, [[tokens, 4]], 8.0)
        op("aten::gather", 66.0, 20, [[tokens, 4]], 4.0)
        # Experts: xpu_fused_moe python_function wrapping the grouped GEMM.
        pyfn(".../vllm_xpu_kernels/fused_moe_interface.py(515): xpu_fused_moe",
             120.0, 120)
        op("_xpu_C::cutlass_grouped_gemm_interface", 122.0, 100,
           [[tokens, 16], [16, 32]], 30.0)
        # Back at FusedMoE level, after the op closes: the TP reduce.
        op("c10d::allreduce_", 310.0, 5, [[tokens, 16]], 3.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        tree = g["prefill"] or g["decode"]
        fused = find(tree, "FusedMoE")
        self.assertIsNotNone(fused)
        children = {c["name"]: c for c in fused["children"]}
        # Router and experts surfaced as named children of FusedMoE.
        self.assertIn("router", children)
        self.assertIn("moe", children)
        router, moe = children["router"], children["moe"]
        # Each kept its own ops (moved out of the fused op).
        self.assertTrue(any(o["name"] == "aten::topk" for o in router["ops"]))
        self.assertTrue(any(o["name"] == "aten::gather" for o in router["ops"]))
        self.assertTrue(any(o["name"] == "_xpu_C::cutlass_grouped_gemm_interface"
                            for o in moe["ops"]))
        # Device time is conserved and counted once: the wrapping op keeps only
        # its own 5us; router holds 8+4=12us, experts hold 30us.
        moe_op = next(o for o in fused["ops"]
                      if o["name"] == "vllm::moe_forward_shared")
        self.assertAlmostEqual(moe_op["device_time_us"], 5.0, places=2)
        self.assertAlmostEqual(router["total_device_time_us"], 12.0, places=2)
        self.assertAlmostEqual(moe["total_device_time_us"], 30.0, places=2)
        # Execution order: router → moe → reduce.
        reduce_op = next(o for o in fused["ops"]
                         if o["name"] == "c10d::allreduce_")
        self.assertLess(router["order"], moe["order"])
        self.assertLess(moe["order"], reduce_op["order"])

    def test_duplicate_shared_experts_module_coalesced(self):
        # vLLM's MoE shared-experts overlap records the *same* module object's
        # forward twice within one parent forward: once as an empty shell whose
        # compute is fused into the sibling ``vllm::moe_forward_shared`` op
        # (hoisted out empty) and once as the real MLP forward. Both events carry
        # the identical profiler label (``SharedExperts_0`` twice), so the graph
        # used to show a spurious empty ``SharedExperts`` sibling next to the
        # real one. They must collapse to a single node holding the MLP, with
        # device time conserved. The empty-first order here mirrors the CUDA
        # prefill trace (decode has empty-last — coalescing is order-agnostic).
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur=0.0):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 400)
        mod("TinyDecoderLayer", 0, 1.0, 380)
        mod("TinyMoE", 0, 5.0, 370)
        mod("MoERunner", 0, 8.0, 360)
        # Fused MoE op (routed experts). The empty shared_experts shell is
        # recorded *inside* it and gets hoisted out empty.
        op("vllm::moe_forward_shared", 20.0, 180, [[tokens, 16], [tokens, 4]],
           9.0)
        mod("SharedExperts", 0, 30.0, 20)          # empty shell (no ops)
        # The real shared_experts forward, same object → same label, with MLP.
        mod("SharedExperts", 0, 210.0, 120)
        mod("TinyMLP", 0, 212.0, 110)
        op("aten::linear", 214.0, 8, [[tokens, 16], [64, 16]], 7.0)   # gate_up
        op("aten::linear", 232.0, 8, [[tokens, 32], [16, 32]], 6.0)   # down

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        runner = find(g["prefill"], "MoERunner")
        self.assertIsNotNone(runner)
        # Exactly one SharedExperts child — the empty duplicate is gone.
        shared = [c for c in runner["children"]
                  if c["module_type"] == "SharedExperts"]
        self.assertEqual(len(shared), 1)
        # The surviving node holds the real MLP subtree.
        mlp = find(shared[0], "TinyMLP")
        self.assertIsNotNone(mlp)
        self.assertEqual(sum(1 for o in mlp["ops"]
                             if o["name"] == "aten::linear"), 2)
        # Device time is conserved: the fused op keeps its own 9us; the shared
        # experts hold the MLP's 7+6=13us (not lost to the empty shell).
        moe_op = next(o for o in runner["ops"]
                      if o["name"] == "vllm::moe_forward_shared")
        self.assertAlmostEqual(moe_op["device_time_us"], 9.0, places=2)
        self.assertAlmostEqual(shared[0]["total_device_time_us"], 13.0, places=2)

    def test_synthetic_frame_duplicates_not_coalesced(self):
        # The duplicate-module coalescing must ONLY merge real instance-indexed
        # module events (``SharedExperts_0`` twice = same object). Synthetic
        # functional-frame modules (``_FUNCTIONAL_MODULE_FRAMES``) have a bare
        # class label with no index, so genuinely-distinct occurrences share a
        # label — a Gemma decoder layer has two ``fused_allreduce_gemma_rms_norm``
        # (pre- and post-attention). They must stay as two separate siblings, not
        # collapse into one.
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur=0.0):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        def pyfn(name, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur, "name": name})

        FRAME = (".../fused_allreduce_gemma_rms_norm.py(20): "
                 "fused_allreduce_gemma_rms_norm")
        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 400)
        mod("TinyDecoderLayer", 0, 1.0, 380)
        # Pre-attention fused norm (frame wraps allreduce + norm module).
        pyfn(FRAME, 10.0, 40)
        op("c10d::allreduce_", 12.0, 5, [[tokens, 16]], 3.0)
        mod("TinyRMSNorm", 0, 20.0, 20)
        op("aten::rms_norm", 22.0, 5, [[tokens, 16]], 2.0)
        # Attention in between.
        mod("TinyAttention", 0, 60.0, 40)
        op("aten::mm", 62.0, 8, [[tokens, 16], [16, 16]], 4.0)
        # Post-attention fused norm — SAME frame label, distinct occurrence.
        pyfn(FRAME, 120.0, 40)
        op("c10d::allreduce_", 122.0, 5, [[tokens, 16]], 3.0)
        mod("TinyRMSNorm", 1, 130.0, 20)
        op("aten::rms_norm", 132.0, 5, [[tokens, 16]], 2.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        layer = find(g["prefill"] or g["decode"], "TinyDecoderLayer")
        self.assertIsNotNone(layer)
        fused = [c for c in layer["children"]
                 if c["module_type"] == "FusedAllreduceGemmaRMSNorm"]
        # Two distinct fused norms, NOT collapsed into one.
        self.assertEqual(len(fused), 2)
        for fn in fused:
            self.assertTrue(any(o["name"] == "c10d::allreduce_"
                                for o in fn["ops"]))

    def test_fused_allreduce_gemma_rms_norm_grouped(self):
        # Gemma-style models (MiniMax-M3) fuse the residual TP all-reduce with
        # the following RMSNorm as ``fused_allreduce_gemma_rms_norm``, a
        # python_function that wraps both the ``c10d::allreduce_`` op and the
        # RMSNorm module. Without a boundary the all-reduce and norm float up as
        # two unrelated siblings of the decoder layer (a bare allreduce next to a
        # lone norm), which reads as an unexplained "norm" at the layer edge.
        # Promoting the frame must make it a parent node
        # ``fused_allreduce_gemma_rms_norm → {allreduce, norm}``.
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur=0.0):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        def pyfn(name, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur, "name": name})

        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 300)
        mod("TinyDecoderLayer", 0, 1.0, 250)
        mod("TinyAttention", 0, 2.0, 40)
        op("aten::mm", 5.0, 4, [[tokens, 16], [16, 16]], 2.0)
        # Fused all-reduce + RMSNorm: python_function wrapping allreduce + norm.
        pyfn(".../layers/fused_allreduce_gemma_rms_norm.py(103): "
             "fused_allreduce_gemma_rms_norm", 60.0, 80)
        op("c10d::allreduce_", 62.0, 20, [[tokens, 16]], 6.0)
        mod("GemmaRMSNorm", 0, 90.0, 30)
        op("aten::rms_norm", 92.0, 10, [[tokens, 16]], 4.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        tree = g["prefill"] or g["decode"]
        fused = find(tree, "FusedAllreduceGemmaRMSNorm")
        self.assertIsNotNone(fused)
        self.assertEqual(fused["name"], "fused_allreduce_gemma_rms_norm")
        # The all-reduce op is a direct child op of the fused node.
        self.assertTrue(any(o["name"] == "c10d::allreduce_"
                            for o in fused["ops"]))
        # The RMSNorm module nests under the fused node as a child.
        norm = next((c for c in fused["children"]
                     if c["module_type"] == "GemmaRMSNorm"), None)
        self.assertIsNotNone(norm)
        # Device time is conserved under the fused node (allreduce 6 + norm 4).
        self.assertAlmostEqual(fused["total_device_time_us"], 10.0, places=2)

    def test_tensorlist_collective_and_norm_kernel_shapes_recovered(self):
        # Two residual-stream ops surface shape-less in a real XPU trace and must
        # be recovered: (1) ``c10d::allreduce_`` records its tensor as a
        # ``TensorList`` — ``Input Dims`` = ``[[[2, H]], [], ...]`` with
        # ``Input type`` ``'TensorList'`` (an extra nesting level + no element
        # dtype) — which the parser used to drop; (2) the Gemma RMSNorm runs as a
        # Python-launched Triton kernel (no ``cpu_op``) so it has no shape at all.
        # Both operate on the residual ``[tokens, H]`` in the activation dtype, so
        # both must come out with shape ``[S, H]`` and dtype ``bfloat16`` inferred
        # from the neighbouring hidden-state op.
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7
        H = self.SUMMARY["hidden_size"]

        def kern(e, ts, dur, kname="g"):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": kname,
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, dims, types, kdur=0.0):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": dims,
                                    "Input type": types}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        # A Python-launched Triton norm kernel: a bare device kernel whose launch
        # (xpu_runtime) sits inside the norm module, with no backing cpu_op.
        def triton_kern(ts, dur, kname):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0]}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": kname,
                           "args": {"correlation": corr[0]}})

        tokens = 8
        bf16 = ["c10::BFloat16", "c10::BFloat16"]
        mod("TinyForCausalLM", 0, 0.0, 300)
        mod("TinyDecoderLayer", 0, 1.0, 250)
        # A neighbouring hidden-state op carrying the real [tokens, H] tensor.
        op("aten::mm", 5.0, 4, [[tokens, H], [H, H]], bf16, 2.0)
        # input_layernorm: a Triton norm kernel launched from Python, no cpu_op.
        mod("RMSNorm", 0, 20.0, 10)
        triton_kern(22.0, 3.0, "_gemma_rmsnorm_kernel")
        # TP all-reduce: TensorList input (extra nesting) + no element dtype.
        op("c10d::allreduce_", 40.0, 20,
           [[[tokens, H]], [], [], [], [], []],
           ["TensorList", "", "", "", "Scalar", "Scalar"], 6.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def all_ops(node, out):
            out.extend(node.get("ops", []))
            for c in node.get("children", []):
                all_ops(c, out)
            return out

        tree = g["prefill"] or g["decode"]
        ops = all_ops(tree, [])
        ar = next(o for o in ops if o["name"] == "c10d::allreduce_")
        self.assertEqual(ar["recorded_shapes"], [[tokens, H]])
        self.assertEqual(ar["input_shapes"], [["S", "H"]])
        self.assertEqual(ar["input_dtypes"], ["bfloat16"])
        norm = next(o for o in ops
                    if o["name"] == "triton::_gemma_rmsnorm_kernel")
        self.assertEqual(norm["recorded_shapes"], [[tokens, H]])
        self.assertEqual(norm["input_shapes"], [["S", "H"]])
        self.assertEqual(norm["input_dtypes"], ["bfloat16"])

    def test_cuda_triton_moe_experts_surfaced(self):
        # On CUDA the routed MoE experts run through the Triton modular kernel
        # ``experts/triton_moe.py(198): apply`` (a python_function, not an
        # nn.Module) inside the fused ``vllm::moe_forward_shared`` op, and the
        # grouped-GEMM ``fused_moe_kernel`` is launched via the CUDA *driver* API
        # (``cuLaunchKernelEx``, cat ``cuda_driver``) with no runtime-API launch
        # event. Two things must happen for the experts to be visible: (1) the
        # ``triton_moe.py apply`` frame is promoted to a ``moe`` module and
        # hoisted out of the op; (2) ``cuda_driver`` counts as a launch-site
        # category so the Triton kernel is attributed to its real launch site
        # (inside ``moe``) instead of falling back to External id and collapsing
        # into ``moe_forward_shared``'s start.
        from breakdown.graph_from_trace import build_graph_from_trace
        events = []
        ext = [0]
        corr = [0]
        tid = 7

        def driver_kern(ts, dur, kname):
            # A device kernel launched via the CUDA driver API: the launch event
            # (cat cuda_driver) sits on the worker thread at ``ts``; the kernel
            # event carries the device duration and links back by correlation.
            corr[0] += 1
            events.append({"ph": "X", "cat": "cuda_driver", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1,
                           "name": "cuLaunchKernelEx",
                           "args": {"correlation": corr[0]}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 5000, "dur": dur, "name": kname,
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur=0.0):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                corr[0] += 1
                events.append({"ph": "X", "cat": "cuda_runtime", "tid": tid,
                               "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                               "args": {"correlation": corr[0],
                                        "External id": ext[0]}})
                events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                               "ts": ts + 5000, "dur": kdur, "name": "g",
                               "args": {"correlation": corr[0]}})

        def mod(cls, idx, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{idx}"})

        def pyfn(name, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur, "name": name})

        tokens = 8
        mod("TinyForCausalLM", 0, 0.0, 500)
        mod("TinyDecoderLayer", 0, 1.0, 480)
        mod("TinyMoE", 0, 5.0, 460)
        mod("MoERunner", 0, 8.0, 450)
        # Fused MoE custom op wrapping the routed-expert Triton kernel. Its own
        # residual self-time is a small kernel (2us).
        op("vllm::moe_forward_shared", 20.0, 400, [[tokens, 16], [tokens, 4]],
           2.0)
        # The Triton experts modular-kernel apply frame (promoted to ``moe``).
        pyfn(".../fused_moe/experts/triton_moe.py(198): apply", 60.0, 300)
        op("_moe_C::moe_align_block_size", 66.0, 10, [[tokens, 4]], 3.0)
        # Grouped GEMM: driver-launched Triton kernel, no cpu_op, no runtime API.
        driver_kern(120.0, 40.0, "fused_moe_kernel")
        op("_C::silu_and_mul_with_clamp", 200.0, 8, [[tokens, 32]], 4.0)
        op("_moe_C::moe_sum", 300.0, 8, [[tokens, 4, 16]], 5.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def find(node, cls):
            if node.get("module_type") == cls:
                return node
            for c in node.get("children", []):
                r = find(c, cls)
                if r:
                    return r
            return None

        runner = find(g["prefill"] or g["decode"], "MoERunner")
        self.assertIsNotNone(runner)
        moe = next((c for c in runner["children"]
                    if c["name"] == "moe"), None)
        self.assertIsNotNone(moe)
        self.assertEqual(moe["module_type"], "TritonExperts")
        # The routed-expert ops surfaced on the moe node.
        names = {o["name"] for o in moe["ops"]}
        self.assertIn("_moe_C::moe_align_block_size", names)
        self.assertIn("_moe_C::moe_sum", names)
        # The driver-launched grouped GEMM surfaced as a triton op on moe.
        fmk = next((o for o in moe["ops"]
                    if o["name"] == "triton::fused_moe_kernel"), None)
        self.assertIsNotNone(fmk)
        self.assertEqual(fmk["backend"], "triton")
        self.assertAlmostEqual(fmk["device_time_us"], 40.0, places=2)
        # Device time moved out of the fused op onto moe (op keeps only its 2us
        # residual; moe holds 3 + 40 + 4 + 5 = 52us). Total is conserved.
        moe_op = next(o for o in runner["ops"]
                      if o["name"] == "vllm::moe_forward_shared")
        self.assertAlmostEqual(moe_op["device_time_us"], 2.0, places=2)
        self.assertAlmostEqual(moe["total_device_time_us"], 52.0, places=2)

    def test_rowparallel_in_mlp_named_down_proj_on_xpu(self):
        # RowParallelLinear defaults to ``o_proj`` (attention output), but inside
        # an MLP/expert module it is the ``down_proj``. The disambiguation used
        # to be gated to CUDA, so on XPU the shared_experts MLP hoisted out of a
        # fused MoE op (whose down_proj the reference-name overlay can't reach,
        # since the ref tree lists it under ``MoE.shared_experts`` while the trace
        # nests it under ``FusedMoE``) mislabeled its down projection as
        # ``o_proj`` — even though the dense MLP's overlay-named down_proj was
        # correct. The parent-type disambiguation must be device-agnostic.
        from breakdown.graph_from_trace import _disambiguate_child_name
        mlp_parent = {"module_type": "MiniMaxM3MLP",
                      "child_order": [("MergedColumnParallelLinear", 0),
                                      ("SiluAndMulWithClamp", 0),
                                      ("RowParallelLinear", 0)]}
        attn_parent = {"module_type": "MiniMaxM3Attention",
                       "child_order": [("RowParallelLinear", 0)]}
        # XPU (is_cuda=False): RowParallelLinear in an MLP -> down_proj.
        self.assertEqual(
            _disambiguate_child_name("RowParallelLinear", 2, mlp_parent,
                                     is_cuda=False),
            "down_proj")
        # ... but in attention it stays o_proj on XPU too.
        self.assertEqual(
            _disambiguate_child_name("RowParallelLinear", 0, attn_parent,
                                     is_cuda=False),
            "o_proj")
        # CUDA behavior is unchanged.
        self.assertEqual(
            _disambiguate_child_name("RowParallelLinear", 2, mlp_parent,
                                     is_cuda=True),
            "down_proj")

    def test_first_decode_step_dropped_from_average(self):
        # The first (warmup) decode step must be excluded from the decode
        # latency average. Here its kernel is 10x heavier than steady state;
        # the reported decode op time must be the steady-state value, not a
        # mean that includes the warmup step.
        from breakdown.graph_from_trace import build_graph_from_trace
        events, ext, corr, tid, midx = [], [0], [0], 7, [0]
        clock = [0.0]

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        def step(tokens, kdur):
            t0 = clock[0]
            mod("TinyForCausalLM", t0, 100)
            mod("TinyAttention", t0 + 1, 40)
            op("aten::linear", t0 + 2, 8, [[tokens, 16], [48, 16]], kdur)
            clock[0] = t0 + 120

        step(8, 5.0)      # prefill
        step(1, 100.0)    # decode warmup (heavy) — must be dropped
        step(1, 10.0)     # steady state
        step(1, 10.0)     # steady state

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def _find_linear(node):
            for o in node.get("ops", []):
                if o["name"] == "aten::linear":
                    return o
            for c in node.get("children", []):
                r = _find_linear(c)
                if r:
                    return r
            return None

        self.assertIsNotNone(g["decode"])
        lin = _find_linear(g["decode"])
        self.assertIsNotNone(lin)
        # Average of the two steady steps (10, 10) — warmup 100 excluded.
        self.assertAlmostEqual(lin["device_time_us"], 10.0, places=3)

    def test_single_decode_step_not_dropped(self):
        # Guard: with only one decode step the warmup drop must NOT empty the
        # decode phase (the drop only applies when >= 2 decode steps exist).
        g = self._build([8, 1])
        self.assertIsNotNone(g["prefill"])
        self.assertIsNotNone(g["decode"])

    def test_partial_batch_decode_steps_dropped(self):
        # vLLM admits the batch in ramp-up waves, so early decode steps run
        # fewer than `batch_size` sequences (partial rows like 2/30). Only the
        # steady-state full-batch (32) steps must survive, so the decode op row
        # dim symbolizes to B=32 rather than leaking literal 2/30 nodes.
        from breakdown.graph_from_trace import build_graph_from_trace
        summary = dict(self.SUMMARY)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            # 1 prefill (40 tok) + ramp decode (2, 30) + steady decode (32, 32)
            json.dump(_synthetic_trace([40, 2, 30, 32, 32]), f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=32)
        finally:
            os.unlink(path)

        self.assertIsNotNone(g["decode"])
        self.assertEqual(g["symbols"]["B"], 32)

        def _row_dims(node, acc):
            for o in node.get("ops", []):
                for s in o.get("input_shapes", []):
                    if s:
                        acc.append(s[0])
            for c in node.get("children", []):
                _row_dims(c, acc)
            return acc

        rows = _row_dims(g["decode"], [])
        # No partial-batch literal ints (2, 30) survived; batch dim is symbolic B.
        self.assertNotIn(2, rows)
        self.assertNotIn(30, rows)
        self.assertIn("B", rows)

    def test_shapes_symbolized(self):
        g = self._build([8])
        model = g["prefill"]["children"][0]
        layer = next(c for c in model["children"]
                     if c["module_type"] == "TinyDecoderLayer")
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "TinyAttention")
        linear = next(o for o in attn["ops"] if o["name"] == "aten::linear")
        # First input row dim → "S", hidden dim → "H"
        self.assertEqual(linear["input_shapes"][0], ["S", "H"])

    def test_context_len_symbolized_as_C(self):
        # When a prefix-cached prefill is profiled, the context length (and the
        # full attended KV length context+query) must symbolize as C / S+C.
        from breakdown.graph_from_trace import build_graph_from_trace
        events, ext, corr, tid, midx = [], [0], [0], 7, [0]

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        mod("TinyForCausalLM", 0, 400)
        mod("TinyAttention", 5, 80)
        op("aten::mm", 6, 4, [[8, 16], [16, 48]], 5.0)  # 8 query tokens => S
        # attention: context dim 100 => C, total kv 108 => S+C. (100 is chosen
        # not to collide with a config dim - a colliding value must resolve to
        # the config symbol, see test_config_dim_wins_over_a_colliding_context.)
        op("vllm::unified_attention_with_output", 10, 4,
           [[8, 48], [100, 16], [108, 16]], 3.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1, query_len=8,
                                       context_len=100)
        finally:
            os.unlink(path)

        self.assertEqual(g["symbols"]["C"], 100)
        self.assertEqual(g["symbols"]["S+C"], 108)

        def _find_attn(node):
            for o in node.get("ops", []):
                if "unified_attention" in o["name"]:
                    return o
            for c in node.get("children", []):
                r = _find_attn(c)
                if r:
                    return r
            return None

        attn_op = _find_attn(g["prefill"])
        self.assertIsNotNone(attn_op)
        self.assertIn(["C", "H"], attn_op["input_shapes"])
        self.assertIn(["S+C", "H"], attn_op["input_shapes"])

    def test_paged_attention_kv_rows_get_S_plus_C(self):
        # Real paged/prefix-cached attention records ONLY the new tokens in its
        # key/value inputs ([S, n_kv, d]); the context length is never a tensor
        # dim. With a context, key/value rows must be rewritten to the full
        # attended KV length S+C while query/output rows stay S.
        from breakdown.graph_from_trace import build_graph_from_trace
        events, ext, corr, tid, midx = [], [0], [0], 7, [0]

        def op(name, ts, dur, shapes):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        # summary: n_h=3, n_kv=1 (GQA so heads distinguish q from k/v), d=16.
        summary = {"architecture": "TinyForCausalLM", "hidden_size": 48,
                   "num_heads": 3, "num_kv_heads": 1, "head_dim": 16,
                   "intermediate_size": 64, "vocab_size": 32000,
                   "num_layers": 1, "dtype": "bfloat16"}
        mod("TinyForCausalLM", 0, 400)
        mod("TinyAttention", 5, 80)
        op("aten::mm", 6, 4, [[8, 48], [48, 80]])  # 8 query tokens => S
        # q [8,3,16], k [8,1,16], v [8,1,16], out [8,3,16] — all leading dim 8=S
        op("vllm::unified_attention_with_output", 10, 4,
           [[8, 3, 16], [8, 1, 16], [8, 1, 16], [8, 3, 16], []])

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=1,
                                       query_len=8, context_len=64)
        finally:
            os.unlink(path)

        def _find_attn(node):
            for o in node.get("ops", []):
                if "unified_attention" in o["name"]:
                    return o
            for c in node.get("children", []):
                r = _find_attn(c)
                if r:
                    return r
            return None

        attn = _find_attn(g["prefill"])
        self.assertIsNotNone(attn)
        shapes = attn["input_shapes"]
        # query + output rows stay S; key + value rows become S+C.
        self.assertEqual(shapes[0][0], "S")       # query
        self.assertEqual(shapes[1][0], "S+C")     # key
        self.assertEqual(shapes[2][0], "S+C")     # value
        self.assertEqual(shapes[3][0], "S")       # output
        self.assertEqual(g["symbols"]["S+C"], 72)

    def test_config_dim_wins_over_a_colliding_context(self):
        # Qwen3-30B-A3B has hidden_size == 2048 and the default profiling
        # context is also 2048. The context used to overwrite the config symbol
        # for that value, so every hidden dim symbolized as C and was then swept
        # with the *context* by the Shape Matrix / benchmark: hidden dims
        # collapsed to 0 at ctx=0 (rms_norm divided by zero and killed its
        # worker with SIGFPE, the MoE grouped GEMM rejected its operands).
        # Paged attention never records the context as a tensor dim, so a
        # config dim must always win.
        from breakdown.graph_from_trace import build_graph_from_trace
        events, ext, tid, midx = [], [0], 7, [0]

        def op(name, ts, dur, shapes):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        summary = dict(self.SUMMARY)          # hidden_size = 16
        mod("TinyForCausalLM", 0, 400)
        mod("TinyAttention", 5, 80)
        op("aten::mm", 6, 4, [[8, 16], [16, 48]])

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            # context == hidden_size: the collision that broke Qwen3-30B-A3B.
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=1,
                                       query_len=8, context_len=16)
        finally:
            os.unlink(path)

        self.assertEqual(g["symbols"]["C"], 16)

        def _find(node, name):
            for o in node.get("ops", []):
                if o["name"] == name:
                    return o
            for c in node.get("children", []):
                r = _find(c, name)
                if r:
                    return r
            return None

        mm = _find(g["prefill"], "aten::mm")
        self.assertIsNotNone(mm)
        self.assertEqual(mm["input_shapes"][0], ["S", "H"])
        self.assertEqual(mm["input_shapes"][1][0], "H")

    def test_moe_routed_rows_scale_with_the_token_dim(self):
        # An MoE block expands every token into num_experts_per_tok routed rows,
        # so the permuted hidden states and the grouped GEMM's M are tokens*topk.
        # Frozen at the profiled value (an observed-value symbol) they stopped
        # matching the token operand as soon as the Shape Matrix swept S, and
        # the kernels rejected their own shapes ("remapped_hidden_states must be
        # [num_rows * TopK, hidden_size]").
        from breakdown.graph_from_trace import build_graph_from_trace
        events, ext, tid, midx = [], [0], 7, [0]

        def op(name, ts, dur, shapes, types=None):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": types
                                    or ["c10::BFloat16"] * len(shapes)}})

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        summary = dict(self.SUMMARY)
        summary.update({"num_experts": 8, "num_experts_per_tok": 4,
                        "moe_intermediate_size": 12})
        mod("TinyForCausalLM", 0, 400)
        mod("TinyMoE", 5, 80)
        op("aten::mm", 6, 4, [[8, 16], [16, 48]])          # 8 tokens => S
        # routed rows = 8 * 4 = 32; the gate_up width is 2*12 = 24
        op("_moe_C::remap_hidden_states", 10, 4, [[8, 16], [32, 16], [8, 4]])
        op("_xpu_C::cutlass_grouped_gemm_interface", 15, 4,
           [[32, 16], [8, 16, 24], [32, 24]])

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=1,
                                       query_len=8)
        finally:
            os.unlink(path)

        def _find(node, name):
            for o in node.get("ops", []):
                if o["name"] == name:
                    return o
            for c in node.get("children", []):
                r = _find(c, name)
                if r:
                    return r
            return None

        remap = _find(g["prefill"], "_moe_C::remap_hidden_states")
        gemm = _find(g["prefill"], "_xpu_C::cutlass_grouped_gemm_interface")
        self.assertEqual(remap["input_shapes"][1], ["topk·S", "H"])
        self.assertEqual(remap["input_shapes"][2], ["S", "topk"])
        self.assertEqual(gemm["input_shapes"][0], ["topk·S", "H"])
        self.assertEqual(gemm["input_shapes"][2][0], "topk·S")
        # the fused gate_up width is a config constant, not an observed value
        self.assertEqual(gemm["input_shapes"][1][2], "2·I_moe")
        self.assertEqual(g["symbols"]["topk"], 4)

        # and it re-resolves at a *different* sweep point, which is the point
        from breakdown.shape_derive import _resolve_shape_ints
        syms = dict(g["symbols"], S=64)
        resolved = _resolve_shape_ints(gemm["input_shapes"], syms)
        self.assertEqual(resolved[0][0], 64 * 4)

    def test_router_axis_is_not_swept_when_the_profile_ran_one_token(self):
        # A decode pass at batch 1 makes tokens*topk == topk, so the routed-rows
        # rule would match the router's own [tokens, topk] operand and turn the
        # expert fan-out into topk*B - a width that scales with the swept batch,
        # which is the very bug the pass exists to prevent.
        from breakdown.graph_from_trace import _symbolize_moe_routed_rows

        summary = {"num_experts": 8, "num_experts_per_tok": 4}
        tree = {
            "name": "moe", "module_type": "MoE", "path": "moe",
            "repeat_count": 1, "children": [],
            "ops": [{"name": "_moe_C::remap_hidden_states",
                     "input_shapes": [["B", "H"], [4, "H"], ["B", 4]],
                     "output_shape": None}],
        }
        syms: dict = {}
        _symbolize_moe_routed_rows([(tree, "B", 1)], syms, summary)
        shapes = tree["ops"][0]["input_shapes"]
        self.assertEqual(shapes[2], ["B", "topk"])       # not ["B", "topk·B"]
        self.assertEqual(shapes[1], ["topk·B", "H"])     # routed rows still scale

    def test_graph_attention_flops_account_for_the_cached_context(self):
        # Ops are costed while the tree is built, i.e. from the recorded KV rows
        # (the new tokens only). Attention really reads context+query, so its
        # cost has to be recomputed once the KV rows are annotated - otherwise a
        # 2048-token context understates the heaviest op ~65x.
        from breakdown.graph_from_trace import _annotate_attention_kv

        op = {"name": "vllm::unified_attention_with_output",
              "input_shapes": [["S", "n_h", "d"], ["S", "n_kv", "d"],
                               ["S", "n_kv", "d"], ["S", "n_h", "d"]],
              "recorded_shapes": [[8, 4, 16], [8, 1, 16], [8, 1, 16],
                                  [8, 4, 16]],
              "input_dtypes": ["bfloat16"] * 4,
              "flops": 0, "memory_bytes": 0, "ai": 0}
        node = {"ops": [op], "children": []}
        _annotate_attention_kv(node, n_kv=1, kv_rows=8 + 64, n_seqs=1,
                               dtype_bytes=2)
        self.assertEqual(op["input_shapes"][1][0], "S+C")
        self.assertEqual(op["flops"], 2 * 2 * 8 * 72 * 4 * 16)
        self.assertGreater(op["memory_bytes"], 0)

    def test_prefill_attention_flops_are_not_divided_by_the_batch(self):
        # The sequence divisor exists for decode's B·C KV rows; a prefill row is
        # already per-sequence (S+C), so dividing it would understate attention
        # by exactly the prefill batch size.
        from breakdown.analyzer import estimate_flops
        from breakdown import shape_matrix

        shapes = [[8, 4, 16], [72, 1, 16], [72, 1, 16], [8, 4, 16]]
        one_seq = estimate_flops("vllm::unified_attention_with_output",
                                 shapes, n_seqs=1)
        self.assertEqual(one_seq, 2 * 2 * 8 * 72 * 4 * 16)

        template = {
            "prefill": {"name": "attn", "module_type": "Attention",
                        "path": "attn", "repeat_count": 1, "children": [],
                        "ops": [{"name": "vllm::unified_attention_with_output",
                                 "input_shapes": [["S", "n_h", "d"],
                                                  ["S+C", "n_kv", "d"],
                                                  ["S+C", "n_kv", "d"],
                                                  ["S", "n_h", "d"]],
                                 "input_dtypes": ["bfloat16"] * 4,
                                 "role": "attention"}]},
            "decode": None,
            "symbols": {"n_h": 4, "n_kv": 1, "d": 16, "S": 8, "C": 64,
                        "S+C": 72, "B": 1, "TP": 1},
            "config": {"tp_size": 1, "dtype_bytes": 2, "num_layers": 1},
        }
        cfgs = shape_matrix.build_configs(
            prefill_seq_lens=[8], prefill_ctx_lens=[64],
            prefill_batch_sizes=[4], decode_ctx_lens=[], decode_batch_sizes=[],
            tp_sizes=[1])
        rows = shape_matrix.build_rows(template, cfgs)
        self.assertEqual(rows[0]["FLOPs"], one_seq)

    def test_empty_trace(self):
        from breakdown.graph_from_trace import build_graph_from_trace
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": []}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY)
            self.assertIsNone(g["prefill"])
            self.assertIsNone(g["decode"])
            self.assertFalse(g["has_timing"])
        finally:
            os.unlink(path)

    def test_orphan_triton_kernel_surfaced_on_module(self):
        # A Triton-compiled kernel launches straight from Python with no cpu_op
        # wrapper (e.g. an RMSNorm). It must still surface as a ``triton::`` op on
        # its enclosing module, attributed by launch-site containment.
        from breakdown.graph_from_trace import build_graph_from_trace
        tid = 7
        events = [
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 0, "dur": 100, "name": "nn.Module: TinyForCausalLM_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 1, "dur": 98, "name": "nn.Module: TinyModel_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 10, "dur": 30, "name": "nn.Module: TinyDecoderLayer_0"},
            # A real matmul with a cpu_op + kernel (so a phase/token dim exists).
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 11, "dur": 8, "name": "aten::linear",
             "args": {"External id": 1, "Input Dims": [[8, 16], [48, 16]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            {"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
             "ts": 12, "dur": 0.1, "name": "urEnqueueKernelLaunch",
             "args": {"correlation": 1, "External id": 1}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 2000, "dur": 5.0, "name": "gemm_xpu_kernel",
             "args": {"correlation": 1}},
            # An orphan Triton norm kernel: only a runtime launch (inside the
            # decoder-layer module) + device kernel, NO cpu_op.
            {"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
             "ts": 30, "dur": 0.1, "name": "urEnqueueKernelLaunch",
             "args": {"correlation": 2, "External id": 999}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 3000, "dur": 4.0, "name": "_rms_norm_kernel",
             "args": {"correlation": 2}},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)
        model = g["prefill"]["children"][0]
        layer = next(c for c in model["children"]
                     if c["module_type"] == "TinyDecoderLayer")
        norm_op = next(o for o in layer["ops"]
                       if o["name"] == "triton::_rms_norm_kernel")
        self.assertEqual(norm_op["backend"], "triton")
        self.assertAlmostEqual(norm_op["device_time_us"], 4.0, places=3)

    def test_flashinfer_kernel_surfaced_launcher_suppressed(self):
        # FlashInfer RMSNorm launches directly from Python: a cuda_runtime
        # ``cudaLaunchKernelExC`` event (inside the decoder-layer module) backing
        # a device ``kernel`` whose symbol embeds "flashinfer". The kernel must
        # surface as a FlashInfer op; the launch-API runtime event must NOT
        # appear as a duplicate ``triton::cudaLaunchKernelExC`` op.
        from breakdown.graph_from_trace import build_graph_from_trace
        tid = 7
        fi_kernel = ("kernel_cutlass_kernel_flashinfernormkernelsrmsnorm"
                     "RMSNormKernel_object_at__T_0")
        events = [
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 0, "dur": 100, "name": "nn.Module: TinyForCausalLM_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 1, "dur": 98, "name": "nn.Module: TinyModel_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 10, "dur": 30, "name": "nn.Module: TinyDecoderLayer_0"},
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 11, "dur": 8, "name": "aten::linear",
             "args": {"External id": 1, "Input Dims": [[8, 16], [48, 16]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            {"ph": "X", "cat": "cuda_runtime", "tid": tid, "pid": tid,
             "ts": 12, "dur": 0.1, "name": "cudaLaunchKernelExC",
             "args": {"correlation": 1, "External id": 1}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 2000, "dur": 5.0, "name": "gemm_cuda_kernel",
             "args": {"correlation": 1}},
            # FlashInfer norm: cudaLaunchKernelExC (inside the layer) + kernel.
            {"ph": "X", "cat": "cuda_runtime", "tid": tid, "pid": tid,
             "ts": 30, "dur": 0.1, "name": "cudaLaunchKernelExC",
             "args": {"correlation": 2, "External id": 999}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 3000, "dur": 4.0, "name": fi_kernel,
             "args": {"correlation": 2}},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def all_ops(node):
            yield from node.get("ops", [])
            for c in node.get("children", []):
                yield from all_ops(c)

        ops = list(all_ops(g["prefill"]))
        # No launch-API event should be surfaced as an op.
        self.assertFalse(any("cudaLaunchKernelExC" in o["name"] for o in ops),
                         "runtime launch-API event must not surface as an op")
        fi_op = next(o for o in ops if "flashinfer" in o["name"].lower())
        self.assertEqual(fi_op["backend"], "flashinfer")
        self.assertAlmostEqual(fi_op["device_time_us"], 4.0, places=3)
        # Name is cleaned: "flashinfer::" namespace, no misleading "triton::"
        # prefix and no "_object_at..." cutlass object-repr tail.
        self.assertTrue(fi_op["name"].startswith("flashinfer::"), fi_op["name"])
        self.assertNotIn("triton::", fi_op["name"])
        self.assertNotIn("_object_at", fi_op["name"])

    def test_flashinfer_kernel_named_after_public_api_frame(self):
        # When the FlashInfer kernel is launched inside its public API python
        # frame (``flashinfer/norm/__init__.py(...): gemma_fused_add_rmsnorm``),
        # the synthetic op is named after that API — short and matching the
        # readable XPU kernel names — instead of the raw cutlass functor symbol.
        from breakdown.graph_from_trace import build_graph_from_trace
        tid = 7
        fi_kernel = ("kernel_cutlass_kernel_flashinfernormkernels"
                     "fused_add_rmsnormFusedAddRMSNormKernel_object_at__T_0")
        events = [
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 0, "dur": 100, "name": "nn.Module: TinyForCausalLM_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 1, "dur": 98, "name": "nn.Module: TinyModel_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 10, "dur": 30, "name": "nn.Module: TinyDecoderLayer_0"},
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 11, "dur": 8, "name": "aten::linear",
             "args": {"External id": 1, "Input Dims": [[8, 16], [48, 16]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            {"ph": "X", "cat": "cuda_runtime", "tid": tid, "pid": tid,
             "ts": 12, "dur": 0.1, "name": "cudaLaunchKernelExC",
             "args": {"correlation": 1, "External id": 1}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 2000, "dur": 5.0, "name": "gemm_cuda_kernel",
             "args": {"correlation": 1}},
            # Public FlashInfer API frame (outer) + private impl (inner) — the
            # public ``__init__.py`` name must win.
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 28, "dur": 8,
             "name": "flashinfer/norm/__init__.py(433): gemma_fused_add_rmsnorm"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 29, "dur": 6,
             "name": ("flashinfer/norm/kernels/fused_add_rmsnorm.py(1014): "
                      "fused_add_rmsnorm_cute")},
            {"ph": "X", "cat": "cuda_runtime", "tid": tid, "pid": tid,
             "ts": 30, "dur": 0.1, "name": "cudaLaunchKernelExC",
             "args": {"correlation": 2, "External id": 999}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 3000, "dur": 4.0, "name": fi_kernel,
             "args": {"correlation": 2}},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def all_ops(node):
            yield from node.get("ops", [])
            for c in node.get("children", []):
                yield from all_ops(c)

        ops = list(all_ops(g["prefill"]))
        fi_op = next(o for o in ops if "flashinfer" in o["name"].lower())
        self.assertEqual(fi_op["name"], "flashinfer::gemma_fused_add_rmsnorm")
        self.assertEqual(fi_op["backend"], "flashinfer")
        self.assertAlmostEqual(fi_op["device_time_us"], 4.0, places=3)

    def test_flash_xpu_kernel_named_after_xattention_api_frame(self):
        # MiniMax-M3 MSA (lightning indexer + block-sparse attend) runs on XPU
        # via the ``flash_xpu`` (``xattention._C``) SYCL kernels, launched
        # directly from the ``xattention.py`` wrappers with no aten/_C cpu_op.
        # The raw kernel symbol (``flash_xpu::(anonymous namespace)::
        # index_score_kernel_t``) is long and would misclassify as triton; the
        # synthetic op must instead be named after the public xattention API
        # frame (``flash_xpu::minimax_m3_index_score``) and classify as the
        # ``flash_xpu`` backend.
        from breakdown.graph_from_trace import build_graph_from_trace
        tid = 7
        fx_kernel = "flash_xpu::(anonymous namespace)::index_score_kernel_t"
        events = [
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 0, "dur": 100, "name": "nn.Module: TinyForCausalLM_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 1, "dur": 98, "name": "nn.Module: TinyModel_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 10, "dur": 30, "name": "nn.Module: TinyDecoderLayer_0"},
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 11, "dur": 8, "name": "aten::linear",
             "args": {"External id": 1, "Input Dims": [[8, 16], [48, 16]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            {"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
             "ts": 12, "dur": 0.1, "name": "urEnqueueKernelLaunch",
             "args": {"correlation": 1, "External id": 1}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 2000, "dur": 5.0, "name": "gemm_xpu_kernel",
             "args": {"correlation": 1}},
            # Public xattention API frame (outer) + pybind builtin (inner) — the
            # ``xattention.py`` wrapper name must win.
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 28, "dur": 8,
             "name": ("/x/vllm/models/minimax_m3/xpu/ops/xattention.py(63): "
                      "minimax_m3_index_score")},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 29, "dur": 6,
             "name": ("<built-in method minimax_m3_index_score of "
                      "pybind11_builtins.pybind11_detail_function_record>")},
            {"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
             "ts": 30, "dur": 0.1, "name": "urEnqueueKernelLaunch",
             "args": {"correlation": 2, "External id": 999}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 3000, "dur": 4.0, "name": fx_kernel,
             "args": {"correlation": 2}},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        def all_ops(node):
            yield from node.get("ops", [])
            for c in node.get("children", []):
                yield from all_ops(c)

        ops = list(all_ops(g["prefill"]))
        fx_op = next(o for o in ops if "flash_xpu" in o["name"].lower())
        self.assertEqual(fx_op["name"], "flash_xpu::minimax_m3_index_score")
        self.assertEqual(fx_op["backend"], "flash_xpu")
        self.assertAlmostEqual(fx_op["device_time_us"], 4.0, places=3)

    def test_flash_xpu_attention_kernel_shapes_reconstructed(self):
        # ``flash_xpu`` MSA kernels carry no cpu_op, so they surface shape-less.
        # Their primary-tensor layout is fixed by the xattention.py wrappers: the
        # block-sparse attend takes a query ``[total_q, num_heads, head_dim]`` and
        # the lightning indexer an index query
        # ``[total_q, num_index_heads, index_head_dim]``. The reconstruction must
        # rebuild both from the config + the step's token count (leading dim of a
        # neighbouring activation) — ``[S, n_h, d]`` and ``[S, n_idx, idx_d]``.
        from breakdown.graph_from_trace import build_graph_from_trace
        summary = dict(self.SUMMARY, head_dim=8, sparse_num_index_heads=2,
                       sparse_index_dim=4)
        tid = 7
        tokens = 8
        events = [
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 0, "dur": 100, "name": "nn.Module: TinyForCausalLM_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 5, "dur": 90, "name": "nn.Module: TinyAttention_0"},
            # Neighbouring activation op fixing the token count (leading dim).
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 10, "dur": 4, "name": "aten::linear",
             "args": {"External id": 1,
                      "Input Dims": [[tokens, 16], [48, 16]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            {"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
             "ts": 11, "dur": 0.1, "name": "l",
             "args": {"correlation": 1, "External id": 1}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 2000, "dur": 2.0, "name": "g", "args": {"correlation": 1}},
            # Indexer query kernel launched from the xattention.py wrapper.
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 20, "dur": 6,
             "name": ("/x/vllm/models/minimax_m3/xpu/ops/xattention.py(114): "
                      "minimax_m3_index_decode")},
            {"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
             "ts": 21, "dur": 0.1, "name": "l", "args": {"correlation": 2}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0, "ts": 2100,
             "dur": 3.0,
             "name": "flash_xpu::(anonymous namespace)::index_score_kernel_t",
             "args": {"correlation": 2}},
            # Block-sparse attend query kernel from the xattention.py wrapper.
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 40, "dur": 6,
             "name": ("/x/vllm/models/minimax_m3/xpu/ops/xattention.py(177): "
                      "minimax_m3_sparse_attn_decode")},
            {"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
             "ts": 41, "dur": 0.1, "name": "l", "args": {"correlation": 3}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0, "ts": 2200,
             "dur": 4.0,
             "name": "flash_xpu::(anonymous namespace)::sparse_attn_kernel_t",
             "args": {"correlation": 3}},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=1)
        finally:
            os.unlink(path)

        def all_ops(node, out):
            out.extend(node.get("ops", []))
            for c in node.get("children", []):
                all_ops(c, out)
            return out

        ops = all_ops(g["prefill"] or g["decode"], [])
        attn = next(o for o in ops
                    if o["name"] == "flash_xpu::minimax_m3_sparse_attn_decode")
        self.assertEqual(attn["recorded_shapes"], [[tokens, 3, 8]])  # [S, n_h, d]
        self.assertEqual(attn["input_shapes"], [["S", "n_h", "d"]])
        self.assertEqual(attn["input_dtypes"], ["bfloat16"])
        idx = next(o for o in ops
                   if o["name"] == "flash_xpu::minimax_m3_index_decode")
        self.assertEqual(idx["recorded_shapes"], [[tokens, 2, 4]])  # [S, n_idx, idx_d]
        self.assertEqual(idx["input_shapes"][0][0], "S")
        self.assertEqual(idx["input_dtypes"], ["bfloat16"])

    def test_cuda_msa_indexer_kernel_shapes_reconstructed(self):
        # On CUDA the MiniMax-M3 MSA / lightning-indexer kernels are ``triton.jit``
        # kernels launched straight from ``common/ops/{sparse_attn,index_topk}.py``
        # (no cpu_op), so they surface as shape-less ``triton::`` ops — the graph
        # and the Shape Matrix showed them with no shape at all. The layouts are
        # fixed by the wrapper signatures, so reconstruction must rebuild
        # ``[S, n_h/TP, d]`` for the block-sparse attend, ``[S, n_idx/TP, d]`` for
        # the indexer block score and ``[n_idx/TP, S, K_topk]`` (int32) for the
        # top-k block ids.
        from breakdown.graph_from_trace import build_graph_from_trace
        summary = dict(self.SUMMARY, hidden_size=32, num_heads=4, num_kv_heads=2,
                       head_dim=8, sparse_attention=True,
                       sparse_num_index_heads=2, sparse_index_dim=8,
                       sparse_topk_blocks=6)
        tid = 7
        tokens = 8

        def launch(corr, ts):
            return {"ph": "X", "cat": "cuda_driver", "tid": tid, "pid": tid,
                    "ts": ts, "dur": 0.1, "name": "cuLaunchKernelEx",
                    "args": {"correlation": corr}}

        def kern(corr, ts, name):
            return {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0, "ts": ts,
                    "dur": 3.0, "name": name, "args": {"correlation": corr}}

        events = [
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 0, "dur": 100, "name": "nn.Module: TinyForCausalLM_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 5, "dur": 90, "name": "nn.Module: TinyAttention_0"},
            # Residual hidden-state op fixing the step's token count.
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 10, "dur": 4, "name": "aten::linear",
             "args": {"External id": 1,
                      "Input Dims": [[tokens, 32], [48, 32]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            launch(1, 11),
            kern(1, 2000, "g"),
            # A weight transpose: its leading dim is a model dim, not the token
            # count — it must NOT be picked as the token reference.
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 14, "dur": 1, "name": "aten::t",
             "args": {"External id": 2, "Input Dims": [[48, 32]],
                      "Input type": ["c10::BFloat16"]}},
            launch(5, 20), kern(5, 2100, "_gqa_sparse_fwd_kernel"),
            launch(6, 30), kern(6, 2200, "_index_block_score_kernel"),
            launch(7, 40), kern(7, 2300, "_topk_index_kernel"),
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=2, batch_size=1)
        finally:
            os.unlink(path)

        def all_ops(node, out):
            out.extend(node.get("ops", []))
            for c in node.get("children", []):
                all_ops(c, out)
            return out

        ops = {o["name"]: o for o in all_ops(g["prefill"] or g["decode"], [])}
        attn = ops["triton::_gqa_sparse_fwd_kernel"]
        self.assertEqual(attn["recorded_shapes"], [[tokens, 2, 8]])
        self.assertEqual(attn["input_shapes"], [["S", "n_h/TP", "d"]])
        self.assertEqual(attn["input_dtypes"], ["bfloat16"])
        score = ops["triton::_index_block_score_kernel"]
        self.assertEqual(score["recorded_shapes"], [[tokens, 1, 8]])
        self.assertEqual(score["input_shapes"], [["S", "n_idx/TP", "d"]])
        topk = ops["triton::_topk_index_kernel"]
        self.assertEqual(topk["recorded_shapes"], [[1, tokens, 6]])
        self.assertEqual(topk["input_shapes"], [["n_idx/TP", "S", "K_topk"]])
        self.assertEqual(topk["input_dtypes"], ["int32"])
        # Symbols carry the concrete values so the Shape Matrix can resolve them.
        self.assertEqual(g["symbols"]["n_idx"], 2)
        self.assertEqual(g["symbols"]["K_topk"], 6)

    def test_layer_extrapolation_to_config_count(self):
        # Reduced-layer profiling captures 2 decoder layers; the config says the
        # model has 5. The unprofiled layers fold into the last decoder group.
        from breakdown.graph_from_trace import build_graph_from_trace
        summary = dict(self.SUMMARY, num_layers=5)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(_synthetic_trace([8]), f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=1)
        finally:
            os.unlink(path)
        model = g["prefill"]["children"][0]
        layers = [c for c in model["children"]
                  if c["module_type"] == "TinyDecoderLayer"]
        self.assertEqual(sum(c["repeat_count"] for c in layers), 5)
        self.assertEqual(layers[-1]["repeat_count"], 5)

    def test_runtime_bookkeeping_not_surfaced_as_kernel(self):
        # A CUDA trace carries host-side runtime bookkeeping events
        # (cudaEventQuery / cudaStreamWaitEvent) that launch NO device kernel.
        # They must never be collected as kernel launches (and so never surface
        # as bogus ``triton::cudaEventQuery`` leaf ops): the collected launch
        # count must equal the real device-kernel count.
        from breakdown.graph_from_trace import (_build_raw_forest,
                                                 _collect_kernel_launches)
        tid = 7
        events = [
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 0, "dur": 100, "name": "nn.Module: TinyForCausalLM_0"},
            {"ph": "X", "cat": "python_function", "tid": tid, "pid": tid,
             "ts": 10, "dur": 30, "name": "nn.Module: TinyDecoderLayer_0"},
            {"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
             "ts": 11, "dur": 8, "name": "aten::mm",
             "args": {"External id": 1, "Input Dims": [[8, 16], [16, 16]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            # Real device kernel launched by the mm (cuda_runtime backs it).
            {"ph": "X", "cat": "cuda_runtime", "tid": tid, "pid": tid,
             "ts": 12, "dur": 0.1, "name": "cudaLaunchKernel",
             "args": {"correlation": 1, "External id": 1}},
            {"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
             "ts": 2000, "dur": 5.0, "name": "gemm_cuda_kernel",
             "args": {"correlation": 1}},
            # Host-side bookkeeping that launches nothing — must be ignored.
            {"ph": "X", "cat": "cuda_runtime", "tid": tid, "pid": tid,
             "ts": 20, "dur": 3.0, "name": "cudaEventQuery",
             "args": {"correlation": 501}},
            {"ph": "X", "cat": "cuda_runtime", "tid": tid, "pid": tid,
             "ts": 25, "dur": 3.0, "name": "cudaStreamWaitEvent",
             "args": {"correlation": 502}},
        ]
        roots = _build_raw_forest(events)
        launches = _collect_kernel_launches(events, tid)
        n_device = sum(1 for e in events if e.get("cat") == "kernel")
        self.assertEqual(len(launches), n_device)
        names = [nm for _, nm, _, _ in launches]
        self.assertNotIn("cudaEventQuery", names)
        self.assertNotIn("cudaStreamWaitEvent", names)

    def test_module_less_kernel_time_conserved_not_dropped(self):
        # A device kernel launched inside a *module-less* top-level op subtree
        # (e.g. a bare sampler op with no enclosing module) must not be silently
        # dropped: its device time is folded into the deepest op so it is
        # conserved (coverage classifier reports it on a leaf, nothing dropped).
        from breakdown.graph_from_trace import (_Raw, _attribute_kernels,
                                                 _kernel_leaf_coverage)
        # Forest: a module with an mm op, plus a module-less top-level sampler op.
        mod = _Raw("module", "TinyDecoderLayer", 0.0, 100.0)
        mm = _Raw("op", "aten::mm", 10.0, 5.0)
        mod.children.append(mm)
        sampler = _Raw("op", "aten::topk", 200.0, 20.0)  # top-level, no module
        roots = [mod, sampler]
        launches = [
            (12.0, "gemm", 4.0, None),        # inside mm -> mm.self_dev
            (205.0, "topk_kernel", 6.0, None),  # inside module-less sampler op
        ]
        cov = _kernel_leaf_coverage(roots, launches)
        self.assertEqual(cov["n_dropped_gap"], 0)
        self.assertAlmostEqual(cov["dropped_gap_us"], 0.0, places=6)
        self.assertAlmostEqual(cov["on_leaf_us"], 10.0, places=6)
        # Attribution must conserve both kernels' time onto op leaves.
        _attribute_kernels(roots, launches)
        self.assertAlmostEqual(mm.self_dev, 4.0, places=6)
        self.assertAlmostEqual(sampler.self_dev, 6.0, places=6)

    def test_minimax_m3_traces_every_in_step_kernel_on_leaf(self):
        # Opt-in end-to-end coverage over the real MiniMax-M3 traces (XPU + CUDA,
        # prefill + decode): every device kernel launched inside a kept
        # prefill/decode step must land on a leaf op, and the collected launch
        # count must equal the real device-kernel count (no bookkeeping noise,
        # no silent drops). Skipped when the trace files are absent.
        import glob
        from breakdown.graph_from_trace import (
            _build_raw_forest, _classify_steps, _collect_kernel_launches,
            _deepest_at, _load_trace, _DEVICE_KERNEL_CATEGORIES, _PLUMBING_OPS)
        from breakdown.trace_common import MODULE_SPAN_PREFIX
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pattern = os.path.join(here, "output", "traces",
                               "vllm_trace_MiniMax-M3_*")
        files = sorted(glob.glob(pattern))
        if not files:
            self.skipTest("MiniMax-M3 trace files not present")

        def worker_of(events):
            span: dict = {}
            for e in events:
                if (e.get("ph") == "X" and e.get("cat") == "user_annotation"
                        and str(e.get("name", "")).startswith(
                            MODULE_SPAN_PREFIX)):
                    span[e.get("tid")] = span.get(e.get("tid"), 0) + 1
            if span:
                return max(span, key=span.get)
            cc: dict = {}
            for e in events:
                if e.get("cat") == "cpu_op":
                    cc[e.get("tid")] = cc.get(e.get("tid"), 0) + 1
            return max(cc, key=cc.get)

        checked = 0
        for f in files:
            # The shared traces dir may hold partial/other-run stub files; skip
            # anything that isn't a readable trace with events.
            try:
                events = _load_trace(f).get("traceEvents", [])
            except (OSError, ValueError):
                continue
            if not events:
                continue
            roots = _build_raw_forest(events)
            if not roots:
                continue
            worker = worker_of(events)
            launches = _collect_kernel_launches(events, worker)
            n_device = sum(1 for e in events
                           if e.get("cat") in _DEVICE_KERNEL_CATEGORIES)
            # Precision: no bookkeeping events collected as launches.
            self.assertEqual(len(launches), n_device, os.path.basename(f))
            bs = 32 if "decode" in os.path.basename(f) else 1
            prefill, decode, _, _ = _classify_steps(roots, bs)
            intervals = [(r.ts, r.end) for r in prefill + decode]

            def in_kept(ts, intervals=intervals):
                return any(a <= ts < b for a, b in intervals)

            checked += 1
            for ts, _nm, _dur, _api in launches:
                if not in_kept(ts):
                    continue
                node = _deepest_at(roots, ts)
                # Every in-step launch resolves to a node, and either lands on a
                # real op leaf or on an enclosing module (synthetic leaf op).
                self.assertIsNotNone(node, os.path.basename(f))

        if not checked:
            self.skipTest("no readable MiniMax-M3 trace files present")


class TestSymbolicShapeCompleteness(unittest.TestCase):
    """No concrete structural dims leak into symbolic shapes (MiniMax-M3)."""

    # MiniMax-M3 text_config dims: hybrid dense/MoE + DeepSeek-style sparse
    # attention (lightning indexer fused into qkv_proj) + a 1M rope cache.
    M3_SUMMARY = {
        "architecture": "MiniMaxM3SparseForCausalLM",
        "hidden_size": 6144, "num_heads": 64, "num_kv_heads": 4, "head_dim": 128,
        "intermediate_size": 12288, "moe_intermediate_size": 3072,
        "num_experts": 128, "vocab_size": 200064,
        "max_position_embeddings": 1048576, "num_layers": 60, "dtype": "bfloat16",
        "sparse_attention": True, "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
    }

    def test_config_symbols_include_sparse_qkv_and_rope_cache(self):
        from breakdown.graph_from_trace import _build_symbol_tables
        val_to_sym, sym_to_val = _build_symbol_tables(self.M3_SUMMARY, 4)
        # Rope cos/sin cache length = max_position_embeddings → P (not /TP).
        self.assertEqual(sym_to_val["P"], 1048576)
        self.assertEqual(val_to_sym[1048576], "P")
        # Sparse qkv_proj fuses the lightning-indexer q/k projections:
        # QKV_idx = (n_h + 2*n_kv)*d + 2*(n_index_heads*index_dim)
        #         = (64 + 8)*128 + 2*(4*128) = 9216 + 1024 = 10240.
        self.assertEqual(sym_to_val["QKV_idx"], 10240)
        self.assertEqual(val_to_sym[10240], "QKV_idx")
        self.assertEqual(val_to_sym[2560], "QKV_idx/TP")  # per-rank at TP=4
        # The plain (dense) QKV and its per-rank shard still resolve.
        self.assertEqual(val_to_sym[9216], "QKV")
        self.assertEqual(val_to_sym[2304], "QKV/TP")
        # Head-count per-rank dims resolve (position of the leak the user saw).
        self.assertEqual(val_to_sym[16], "n_h/TP")
        self.assertEqual(val_to_sym[1], "n_kv/TP")

    def test_runtime_dims_symbolized_with_observed_values(self):
        from breakdown.graph_from_trace import _symbolize_runtime_dims
        # A minimal tree carrying the run-specific allocation dims that aren't
        # config-derivable: paged KV-cache slots, MoE routed-token rows, and the
        # 1-D moe_align_block_size scratch buffers.
        tree = {
            "module_type": "Model", "ops": [], "children": [
                {"module_type": "MiniMaxM3SparseAttention", "children": [],
                 "ops": [{
                     "name": "_C::fused_minimax_m3_qknorm_rope_kv_insert",
                     "input_shapes": [["N", 2, "d", "n_kv/TP", "d"],
                                      [17286, "d", "d"]]}]},
                {"module_type": "TritonExperts", "children": [],
                 "ops": [
                     {"name": "_C::silu_and_mul_with_clamp",
                      "input_shapes": [[16384, "I_moe/TP"]]},
                     {"name": "_moe_C::moe_align_block_size",
                      "input_shapes": [[32640], [255]]},
                 ]},
            ],
        }
        # Fix the placeholder above: both kv rows share the same slot count.
        tree["children"][0]["ops"][0]["input_shapes"][0][0] = 17286
        sym_to_val: dict = {}
        _symbolize_runtime_dims([tree], sym_to_val)

        kv_op = tree["children"][0]["ops"][0]
        self.assertEqual(kv_op["input_shapes"][0], ["N_kv", 2, "d", "n_kv/TP", "d"])
        self.assertEqual(kv_op["input_shapes"][1], ["N_kv", "d", "d"])
        self.assertEqual(sym_to_val["N_kv"], 17286)

        experts = tree["children"][1]["ops"]
        # 2-D expert-GEMM row count → M_moe.
        self.assertEqual(experts[0]["input_shapes"][0], ["M_moe", "I_moe/TP"])
        self.assertEqual(sym_to_val["M_moe"], 16384)
        # 1-D moe_align scratch → N_moe (largest) / N_moe2 (next), deterministic.
        self.assertEqual(sym_to_val["N_moe"], 32640)
        self.assertEqual(sym_to_val["N_moe2"], 255)
        self.assertEqual(experts[1]["input_shapes"], [["N_moe"], ["N_moe2"]])

    def test_trivial_dims_left_concrete(self):
        from breakdown.graph_from_trace import _symbolize_runtime_dims
        # The k/v-pair constant 2 (and 0/1 placeholders) must stay literal.
        tree = {"module_type": "M", "children": [], "ops": [
            {"name": "_C::kv_cache_update", "input_shapes": [[2], [1], [0]]}]}
        sym_to_val: dict = {}
        _symbolize_runtime_dims([tree], sym_to_val)
        self.assertEqual(tree["ops"][0]["input_shapes"], [[2], [1], [0]])

    def test_minimax_m3_traces_no_concrete_structural_dims(self):
        # End-to-end over the real MiniMax-M3 traces (XPU + CUDA, prefill +
        # decode): after reconstruction no op input shape may carry a concrete
        # integer above the trivial threshold (2). Skipped when traces absent.
        import glob
        from breakdown.graph_from_trace import build_graph_from_trace, _load_trace
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = sorted(glob.glob(os.path.join(
            here, "output", "traces", "vllm_trace_MiniMax-M3_*_tp4_*layers.json.gz")))
        if not files:
            self.skipTest("MiniMax-M3 trace files not present")
        checked = 0
        for f in files:
            base = os.path.basename(f)
            # The shared traces dir may hold partial/LFS-stub files; skip any
            # that aren't a readable trace with events.
            try:
                if not _load_trace(f).get("traceEvents"):
                    continue
            except (OSError, ValueError):
                continue
            is_prefill = "prefill" in base
            g = build_graph_from_trace(
                f, self.M3_SUMMARY, tp_size=4,
                batch_size=(1 if is_prefill else 32),
                query_len=(2048 if is_prefill else 1), context_len=2048)

            def concretes(node, acc):
                if not node:
                    return
                for op in node.get("ops", []):
                    for shp in (op.get("input_shapes") or []):
                        if not isinstance(shp, list):
                            continue
                        for dim in shp:
                            if isinstance(dim, int) and dim > 2:
                                acc.append((op["name"], tuple(shp)))
                for c in node.get("children", []):
                    concretes(c, acc)

            acc: list = []
            concretes(g.get("prefill"), acc)
            concretes(g.get("decode"), acc)
            self.assertEqual(acc, [], f"{base}: concrete structural dims leaked")
            checked += 1
        if not checked:
            self.skipTest("no readable MiniMax-M3 trace files present")


class TestModuleNaming(unittest.TestCase):
    """Recover real module attribute names (q_norm/k_norm, ...) for the graph."""

    # named_modules() of a tiny Qwen3-style model: two RMSNorm siblings in
    # attention (q_norm/k_norm) plus per-layer norms, and a ModuleList of layers.
    NAMED_MODULES = [
        ("", "Qwen3ForCausalLM"),
        ("model", "Qwen3Model"),
        ("model.embed_tokens", "VocabParallelEmbedding"),
        ("model.layers", "ModuleList"),
        ("model.layers.0", "Qwen3DecoderLayer"),
        ("model.layers.0.self_attn", "Qwen3Attention"),
        ("model.layers.0.self_attn.qkv_proj", "QKVParallelLinear"),
        ("model.layers.0.self_attn.o_proj", "RowParallelLinear"),
        ("model.layers.0.self_attn.q_norm", "RMSNorm"),
        ("model.layers.0.self_attn.k_norm", "RMSNorm"),
        ("model.layers.0.mlp", "Qwen3MLP"),
        ("model.layers.0.input_layernorm", "RMSNorm"),
        ("model.layers.0.post_attention_layernorm", "RMSNorm"),
        ("model.layers.1", "Qwen3DecoderLayer"),
        ("model.layers.1.self_attn", "Qwen3Attention"),
        ("model.layers.1.self_attn.qkv_proj", "QKVParallelLinear"),
        ("model.layers.1.self_attn.o_proj", "RowParallelLinear"),
        ("model.layers.1.self_attn.q_norm", "RMSNorm"),
        ("model.layers.1.self_attn.k_norm", "RMSNorm"),
        ("model.layers.1.mlp", "Qwen3MLP"),
        ("model.layers.1.input_layernorm", "RMSNorm"),
        ("model.layers.1.post_attention_layernorm", "RMSNorm"),
        ("model.norm", "RMSNorm"),
        ("lm_head", "ParallelLMHead"),
    ]

    def _ref(self):
        from breakdown.module_naming import build_ref_tree
        return build_ref_tree(self.NAMED_MODULES)

    def test_build_ref_tree_inlines_modulelist(self):
        ref = self._ref()
        self.assertEqual(ref["cls"], "Qwen3ForCausalLM")
        model = next(c for c in ref["children"] if c["cls"] == "Qwen3Model")
        # ModuleList "layers" is inlined + collapsed into ONE representative.
        layers = [c for c in model["children"]
                  if c["cls"] == "Qwen3DecoderLayer"]
        self.assertEqual(len(layers), 1)
        self.assertTrue(layers[0]["is_group"])
        self.assertEqual(layers[0]["group_size"], 2)
        self.assertEqual(layers[0]["attr"], "layers")

    def test_build_ref_tree_preserves_sibling_norm_names(self):
        ref = self._ref()
        model = next(c for c in ref["children"] if c["cls"] == "Qwen3Model")
        layer = next(c for c in model["children"]
                     if c["cls"] == "Qwen3DecoderLayer")
        attn = next(c for c in layer["children"] if c["cls"] == "Qwen3Attention")
        norms = [c["attr"] for c in attn["children"] if c["cls"] == "RMSNorm"]
        self.assertEqual(norms, ["q_norm", "k_norm"])

    def _trace_tree(self):
        # A trace-reconstructed tree: module_type is the class only; the two
        # attention norms + two layer norms are indistinguishable siblings.
        def mod(mt, children=None):
            return {"name": mt, "module_type": mt, "repeat_count": 1,
                    "ops": [], "children": children or []}
        attn = mod("Qwen3Attention", [
            mod("RMSNorm"),  # q_norm (executes first)
            mod("RMSNorm"),  # k_norm
        ])
        layer = mod("Qwen3DecoderLayer", [
            mod("RMSNorm"),   # input_layernorm
            attn,
            mod("RMSNorm"),   # post_attention_layernorm
            mod("Qwen3MLP"),
        ])
        layer["repeat_count"] = 2
        model = mod("Qwen3Model", [
            mod("VocabParallelEmbedding"),
            layer,
            mod("RMSNorm"),  # final norm
        ])
        return mod("Qwen3ForCausalLM", [model])

    def test_enrich_assigns_attribute_names(self):
        from breakdown.module_naming import enrich_graph_names
        tree = self._trace_tree()
        enrich_graph_names({"prefill": tree, "decode": None}, self._ref())

        model = tree["children"][0]
        self.assertEqual(model["name"], "model")
        layer = next(c for c in model["children"]
                     if c["module_type"] == "Qwen3DecoderLayer")
        self.assertEqual(layer["name"], "decoder_layer")
        names = {c["module_type"]: c["name"] for c in layer["children"]}
        # Sibling RMSNorms disambiguated by execution/definition order.
        layer_norms = [c["name"] for c in layer["children"]
                       if c["module_type"] == "RMSNorm"]
        self.assertEqual(layer_norms,
                         ["input_layernorm", "post_attention_layernorm"])
        self.assertEqual(names["Qwen3Attention"], "self_attn")
        self.assertEqual(names["Qwen3MLP"], "mlp")

        attn = next(c for c in layer["children"]
                    if c["module_type"] == "Qwen3Attention")
        attn_norms = [c["name"] for c in attn["children"]
                      if c["module_type"] == "RMSNorm"]
        self.assertEqual(attn_norms, ["q_norm", "k_norm"])

    def test_enrich_handles_step_wrapper(self):
        # Multi-root phase trees are wrapped in a synthetic InferenceStep node.
        from breakdown.module_naming import enrich_graph_names
        inner = self._trace_tree()
        wrapper = {"name": "step", "module_type": "InferenceStep",
                   "repeat_count": 1, "ops": [], "children": [inner]}
        enrich_graph_names({"prefill": wrapper, "decode": None}, self._ref())
        self.assertEqual(inner["children"][0]["name"], "model")

    def test_enrich_noop_without_ref(self):
        from breakdown.module_naming import enrich_graph_names
        tree = self._trace_tree()
        enrich_graph_names({"prefill": tree, "decode": None}, None)
        # Names unchanged (still class-name placeholders).
        self.assertEqual(tree["children"][0]["name"], "Qwen3Model")

    def test_empty_named_modules(self):
        from breakdown.module_naming import build_ref_tree
        self.assertIsNone(build_ref_tree([]))

    def test_end_to_end_distinguishes_qnorm_knorm(self):
        # Full reconstruction: two structurally-identical RMSNorm siblings in
        # attention would collapse into one "norm ×2" node, but with the ref
        # tree they must stay separate q_norm / k_norm nodes.
        from breakdown.graph_from_trace import build_graph_from_trace
        from breakdown.module_naming import build_ref_tree

        events = []
        ext = [0]
        corr = [0]
        tid = 7
        midx = [0]

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        mod("Qwen3ForCausalLM", 0, 400)
        mod("Qwen3Model", 1, 398)
        mod("VocabParallelEmbedding", 2, 2)
        op("aten::embedding", 2, 1, [[32000, 16], [8]], 0)
        for li in range(2):
            b = 10 + li * 180
            mod("Qwen3DecoderLayer", b, 178)
            mod("RMSNorm", b + 1, 2)
            op("aten::rms_norm", b + 1, 1, [[8, 16]], 1.0)
            mod("Qwen3Attention", b + 5, 80)
            op("aten::linear", b + 6, 4, [[8, 16], [48, 16]], 5.0)
            mod("RMSNorm", b + 12, 2)   # q_norm
            op("aten::rms_norm", b + 12, 1, [[8, 16]], 1.0)
            mod("RMSNorm", b + 15, 2)   # k_norm
            op("aten::rms_norm", b + 15, 1, [[8, 16]], 1.0)
            mod("RMSNorm", b + 90, 2)   # post_attention_layernorm
            op("aten::rms_norm", b + 90, 1, [[8, 16]], 1.0)
            mod("Qwen3MLP", b + 95, 20)
            op("aten::linear", b + 96, 4, [[8, 16], [64, 16]], 7.0)

        ref = build_ref_tree(self.NAMED_MODULES)
        summary = {"architecture": "Qwen3ForCausalLM", "hidden_size": 16,
                   "num_heads": 3, "num_kv_heads": 3, "head_dim": 16,
                   "intermediate_size": 64, "vocab_size": 32000,
                   "num_layers": 2, "dtype": "bfloat16"}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=1,
                                       ref_module_tree=ref)
        finally:
            os.unlink(path)

        self.assertTrue(g["has_module_names"])
        model = g["prefill"]["children"][0]
        self.assertEqual(model["name"], "model")
        layer = next(c for c in model["children"]
                     if c["module_type"] == "Qwen3DecoderLayer")
        self.assertEqual(layer["name"], "decoder_layer")
        self.assertEqual(layer["repeat_count"], 2)  # layers still collapse
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "Qwen3Attention")
        norm_names = [c["name"] for c in attn["children"]
                      if c["module_type"] == "RMSNorm"]
        self.assertEqual(norm_names, ["q_norm", "k_norm"])
        # Each stays a single instance — NOT collapsed into "norm ×2".
        self.assertTrue(all(c["repeat_count"] == 1 for c in attn["children"]
                            if c["module_type"] == "RMSNorm"))

    def test_names_recovered_when_inner_model_level_absent(self):
        # Real vLLM traces often DON'T emit a module event for the inner
        # ``*Model`` wrapper — the decoder stack nests directly under
        # ``*ForCausalLM``. The reference tree still has that level, so alignment
        # must transparently unwrap it. Without the unwrap, child matching stalls
        # at the top and q_norm/k_norm stay "norm".
        from breakdown.graph_from_trace import build_graph_from_trace
        from breakdown.module_naming import build_ref_tree

        events = []
        ext = [0]
        corr = [0]
        tid = 7
        midx = [0]

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 1000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        # NOTE: no Qwen3Model event — layers nest directly under ForCausalLM.
        mod("Qwen3ForCausalLM", 0, 400)
        mod("VocabParallelEmbedding", 2, 2)
        op("aten::embedding", 2, 1, [[32000, 16], [8]], 0)
        for li in range(2):
            b = 10 + li * 180
            mod("Qwen3DecoderLayer", b, 178)
            mod("RMSNorm", b + 1, 2)
            op("aten::rms_norm", b + 1, 1, [[8, 16]], 1.0)
            mod("Qwen3Attention", b + 5, 80)
            op("aten::linear", b + 6, 4, [[8, 16], [48, 16]], 5.0)
            mod("RMSNorm", b + 12, 2)   # q_norm
            op("aten::rms_norm", b + 12, 1, [[8, 16]], 1.0)
            mod("RMSNorm", b + 15, 2)   # k_norm
            op("aten::rms_norm", b + 15, 1, [[8, 16]], 1.0)
            mod("RMSNorm", b + 90, 2)   # post_attention_layernorm
            op("aten::rms_norm", b + 90, 1, [[8, 16]], 1.0)
            mod("Qwen3MLP", b + 95, 20)
            op("aten::linear", b + 96, 4, [[8, 16], [64, 16]], 7.0)

        ref = build_ref_tree(self.NAMED_MODULES)
        summary = {"architecture": "Qwen3ForCausalLM", "hidden_size": 16,
                   "num_heads": 3, "num_kv_heads": 3, "head_dim": 16,
                   "intermediate_size": 64, "vocab_size": 32000,
                   "num_layers": 2, "dtype": "bfloat16"}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, summary, tp_size=1, batch_size=1,
                                       ref_module_tree=ref)
        finally:
            os.unlink(path)

        self.assertTrue(g["has_module_names"])
        # ForCausalLM's children were named across the missing model level.
        root = g["prefill"]
        self.assertEqual(root["module_type"], "Qwen3ForCausalLM")
        embed = next(c for c in root["children"]
                     if c["module_type"] == "VocabParallelEmbedding")
        self.assertEqual(embed["name"], "embed_tokens")
        layer = next(c for c in root["children"]
                     if c["module_type"] == "Qwen3DecoderLayer")
        self.assertEqual(layer["name"], "decoder_layer")
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "Qwen3Attention")
        norm_names = [c["name"] for c in attn["children"]
                      if c["module_type"] == "RMSNorm"]
        self.assertEqual(norm_names, ["q_norm", "k_norm"])


class TestPhasePartition(unittest.TestCase):
    """Prefill/decode step partition + main-model-class detection."""

    def _build(self, prefill_tokens, decode_steps, lm_head_dev):
        # Each engine step is [Model(deep), LogitsProcessor(shallow), Sampler].
        # The LogitsProcessor's lm_head matmul is intentionally the most
        # device-time-heavy op — main-class detection must NOT pick it.
        from breakdown import graph_from_trace as G
        events = []
        ext = [0]
        corr = [0]
        tid = 7
        midx = [0]
        clock = [0]

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 100000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        def step(tokens):
            t0 = clock[0]
            mod("LlamaForCausalLM", t0, 200)          # deep model
            mod("LlamaDecoderLayer", t0 + 1, 190)
            op("aten::mm", t0 + 2, 4, [[tokens, 16], [16, 16]], 2.0)
            t1 = t0 + 210
            mod("LogitsProcessor", t1, 30)            # shallow, huge lm_head op
            op("aten::mm", t1 + 1, 20, [[tokens, 16], [16, 32000]], lm_head_dev)
            t2 = t1 + 40
            mod("Sampler", t2, 10)
            clock[0] = t2 + 20

        step(prefill_tokens)
        for _ in range(decode_steps):
            step(1)

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            summary = {"architecture": "LlamaForCausalLM", "hidden_size": 16,
                       "num_heads": 1, "num_kv_heads": 1, "head_dim": 16,
                       "intermediate_size": 16, "vocab_size": 32000,
                       "num_layers": 1, "dtype": "bfloat16"}
            return G.build_graph_from_trace(path, summary, tp_size=1,
                                            batch_size=1)
        finally:
            os.unlink(path)

    def test_main_class_is_model_not_logits_processor(self):
        # lm_head op dominates device time, but the model subtree is deeper.
        g = self._build(prefill_tokens=8, decode_steps=3, lm_head_dev=999.0)
        # Both phases reconstructed and distinct (prefill S=8, decode B=1).
        self.assertIsNotNone(g["prefill"])
        self.assertIsNotNone(g["decode"])
        self.assertEqual(g["symbols"].get("S"), 8)
        self.assertEqual(g["symbols"].get("B"), 1)

    def _build_steps(self, step_tokens, batch_size):
        """Build a trace of engine steps with explicit per-step token dims."""
        from breakdown import graph_from_trace as G  # noqa: F401
        events = []
        ext = [0]
        corr = [0]
        tid = 7
        midx = [0]
        clock = [0]

        def kern(e, ts, dur):
            corr[0] += 1
            events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid,
                           "pid": tid, "ts": ts, "dur": 0.1, "name": "l",
                           "args": {"correlation": corr[0], "External id": e}})
            events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                           "ts": ts + 100000, "dur": dur, "name": "g",
                           "args": {"correlation": corr[0]}})

        def op(name, ts, dur, shapes, kdur):
            ext[0] += 1
            events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                           "ts": ts, "dur": dur, "name": name,
                           "args": {"External id": ext[0], "Input Dims": shapes,
                                    "Input type": ["c10::BFloat16"]}})
            if kdur:
                kern(ext[0], ts, kdur)

        def mod(cls, ts, dur):
            events.append({"ph": "X", "cat": "python_function", "tid": tid,
                           "pid": tid, "ts": ts, "dur": dur,
                           "name": f"nn.Module: {cls}_{midx[0]}"})
            midx[0] += 1

        def step(tokens):
            t0 = clock[0]
            mod("LlamaForCausalLM", t0, 200)
            mod("LlamaDecoderLayer", t0 + 1, 190)
            op("aten::mm", t0 + 2, 4, [[tokens, 16], [16, 16]], 2.0)
            t1 = t0 + 210
            mod("LogitsProcessor", t1, 30)
            op("aten::mm", t1 + 1, 20, [[tokens, 16], [16, 32000]], 1.0)
            clock[0] = t1 + 40

        for tk in step_tokens:
            step(tk)

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        summary = {"architecture": "LlamaForCausalLM", "hidden_size": 16,
                   "num_heads": 1, "num_kv_heads": 1, "head_dim": 16,
                   "intermediate_size": 16, "vocab_size": 32000,
                   "num_layers": 1, "dtype": "bfloat16"}
        try:
            from breakdown import graph_from_trace as G
            return G.build_graph_from_trace(path, summary, tp_size=1,
                                            batch_size=batch_size)
        finally:
            os.unlink(path)

    def test_decode_pass_batch_rows_not_misclassified(self):
        # Two-pass decode pass: query_len=1 means each sequence's single new
        # token is prefilled *individually* (1-row microsteps), while the batched
        # decode steps have batch_size rows. A "max-token = prefill" rule would
        # wrongly tag the batch-row decode steps as prefill and report B = 7.
        # token dim > batch_size is the correct discriminator: all steps here are
        # decode, and B must equal the batch (8).
        g = self._build_steps(step_tokens=[1, 1, 8, 8, 8, 7], batch_size=8)
        self.assertIsNone(g["prefill"])           # nothing exceeds batch_size
        self.assertIsNotNone(g["decode"])
        self.assertEqual(g["symbols"].get("B"), 8)

    def test_prefill_step_above_batch_is_prefill(self):
        # A genuine multi-token prefill (128 rows) alongside batch-row decode
        # steps (batch=8): the 128-row step is prefill (S=128), decode B=8.
        g = self._build_steps(step_tokens=[128, 8, 8, 8], batch_size=8)
        self.assertIsNotNone(g["prefill"])
        self.assertIsNotNone(g["decode"])
        self.assertEqual(g["symbols"].get("S"), 128)
        self.assertEqual(g["symbols"].get("B"), 8)


class TestLayerOverride(unittest.TestCase):
    """Reduced-layer ``hf_overrides`` construction (app._build_layer_override).

    Regression coverage for the FP8/quantization failure: vLLM's
    ``get_quant_config`` requires ``hf_overrides`` to be a *dict* when
    quantization is requested, so a callable override (used for the non-quant
    reduced-layer path) triggered:

        ValidationError: hf_overrides must be a dict for get_quant_config ...
    """

    def _build(self, *args, **kwargs):
        import app
        return app._build_layer_override(*args, **kwargs)

    def test_no_quant_returns_callable(self):
        # Without quantization we keep the picklable callable override.
        import functools
        ov = self._build(2, None, False)
        self.assertIsInstance(ov, functools.partial)

    def test_quant_flat_returns_dict(self):
        # With quantization (flat model) the override MUST be a plain dict so
        # get_quant_config accepts it. This is the exact FP8 regression.
        ov = self._build(2, "fp8", False)
        self.assertIsInstance(ov, dict)
        self.assertEqual(ov, {"num_hidden_layers": 2})
        self.assertNotIn("text_config", ov)

    def test_quant_nested_targets_text_config(self):
        # Nested multimodal configs (e.g. MiniMax-M3) keep the layer count under
        # text_config; the dict override must target it there.
        ov = self._build(3, "fp8", True)
        self.assertIsInstance(ov, dict)
        self.assertEqual(ov, {"text_config": {"num_hidden_layers": 3}})

    def test_quant_override_is_dict_for_all_methods(self):
        # Any truthy quantization method must yield a dict, never a callable.
        for q in ("fp8", "awq", "gptq", "int4", "bitsandbytes"):
            with self.subTest(quantization=q):
                self.assertIsInstance(self._build(1, q, False), dict)

    def test_callable_sets_flat_layers(self):
        import app

        class _Cfg:
            num_hidden_layers = 36
        cfg = _Cfg()
        app._build_layer_override(4, None, False)(cfg)
        self.assertEqual(cfg.num_hidden_layers, 4)

    def test_callable_sets_nested_layers(self):
        import app

        class _TextCfg:
            num_hidden_layers = 57

        class _Cfg:
            num_hidden_layers = 999
            text_config = _TextCfg()
        cfg = _Cfg()
        app._build_layer_override(5, None, True)(cfg)
        # The nested count is what actually drives the model build.
        self.assertEqual(cfg.text_config.num_hidden_layers, 5)


class TestLayersUnderTextConfigFlag(unittest.TestCase):
    """summarize_config exposes where the decoder layer count lives."""

    def test_flat_config_flag_false(self):
        s = summarize_config({
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 36,
            "hidden_size": 2560,
            "num_attention_heads": 32,
        })
        self.assertFalse(s["layers_under_text_config"])

    def test_nested_config_flag_true(self):
        s = summarize_config({
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 57,
                "hidden_size": 4096,
                "num_attention_heads": 32,
            },
        })
        self.assertTrue(s["layers_under_text_config"])
