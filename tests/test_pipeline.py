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
from breakdown.model_graph import build_model_graph


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
    """Return path to the real trace file if available."""
    if not os.path.isdir(_TRACE_DIR):
        return None
    files = [os.path.join(_TRACE_DIR, f) for f in os.listdir(_TRACE_DIR)
             if f.endswith(".json") or f.endswith(".json.gz")]
    if not files:
        return None
    return sorted(files, key=os.path.getmtime, reverse=True)[0]


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

    def test_build_model_graph_accepts_layerwise_moe_lists(self):
        summary = summarize_config(HUNYUAN_MOE_LIST_CONFIG)

        graph = build_model_graph(
            summary,
            prefill_len=128,
            decode_batch=1,
            context_len=4096,
            tp_size=1,
        )

        self.assertIn("prefill", graph)
        self.assertIn("decode", graph)


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


# ---- Offline MLA configs (no HuggingFace fetch / no GPU required) ----
# MLA (Multi-head Latent Attention) profiling is now supported on XPU via the
# TRITON_MLA backend (dense MLA) and the XPU_MLA_SPARSE backend (DeepSeek sparse
# attention). These configs let us exercise the static MLA graph builders
# without hitting the network.
_DEEPSEEK_V2_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "hidden_size": 5120,
    "num_hidden_layers": 4,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "head_dim": 192,
    "intermediate_size": 12288,
    "moe_intermediate_size": 1536,
    "vocab_size": 102400,
    "torch_dtype": "bfloat16",
    "n_routed_experts": 160,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
    "first_k_dense_replace": 1,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
}

# GLM-style MLA + DeepSeek Sparse Attention (DSA), routed to XPU_MLA_SPARSE.
_GLM_MOE_DSA_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "hidden_size": 4096,
    "num_hidden_layers": 3,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "head_dim": 64,
    "intermediate_size": 10944,
    "moe_intermediate_size": 1408,
    "vocab_size": 151552,
    "torch_dtype": "bfloat16",
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
}

# Backends MLA ops are allowed to run on (XPU-supported, no CPU fallback).
_XPU_SUPPORTED_BACKENDS = {"vllm-xpu-kernels", "torch-xpu-ops", "triton", "framework"}


def _collect_ops(node: dict) -> list[dict]:
    """Flatten all OpNode dicts in a serialized model-graph tree."""
    ops = list(node.get("ops", []))
    for child in node.get("children", []):
        ops.extend(_collect_ops(child))
    return ops


class TestMLAModelGraph(unittest.TestCase):
    """MLA architectures are supported on XPU — static graph must build and
    route attention to XPU backends (regression test for the removed profiling
    block that hard-rejected MLA models)."""

    def _build(self, config: dict) -> dict:
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config

        summary = summarize_config(config)
        return build_model_graph(
            summary, prefill_len=128, decode_batch=1, context_len=2048
        )

    def test_deepseek_v2_graph_builds(self):
        graph = self._build(_DEEPSEEK_V2_CONFIG)
        self.assertEqual(graph["family"], "DeepSeekV2")
        # MLA models have both prefill and decode phases (autoregressive).
        self.assertIsNotNone(graph["prefill"])
        self.assertIsNotNone(graph["decode"])

    def test_glm_moe_dsa_graph_builds(self):
        graph = self._build(_GLM_MOE_DSA_CONFIG)
        self.assertEqual(graph["family"], "GLM5MoE")
        self.assertIsNotNone(graph["prefill"])
        self.assertIsNotNone(graph["decode"])
        # GLM5 has v_head_dim != head_dim — exercises the differing-head-dim path.
        self.assertEqual(graph["symbols"].get("v_d"), 256)

    def test_mla_attention_op_present(self):
        for config in (_DEEPSEEK_V2_CONFIG, _GLM_MOE_DSA_CONFIG):
            graph = self._build(config)
            ops = _collect_ops(graph["prefill"])
            attn = [o for o in ops if o.get("role") == "attention"]
            self.assertTrue(
                attn, f"no attention op for {config['architectures'][0]}"
            )

    def test_mla_attention_not_named_gdn(self):
        # Regression: MLA attention must NOT be labeled "gdn_attention" — that is
        # the Gated Delta Net (linear attention) kernel used by Qwen3-Next, not
        # MLA. Expect the accurate flash/paged-decode kernel names per phase.
        expected = {
            "prefill": "flash_attn_varlen_fwd",
            "decode": "cutlass_paged_decode",
        }
        for config in (_DEEPSEEK_V2_CONFIG, _GLM_MOE_DSA_CONFIG):
            graph = self._build(config)
            for phase, name in expected.items():
                attn = [
                    o for o in _collect_ops(graph[phase])
                    if o.get("role") == "attention"
                ]
                names = {o["name"] for o in attn}
                self.assertNotIn(
                    "gdn_attention", names,
                    f"{config['architectures'][0]} {phase} mislabels MLA "
                    f"attention as gdn_attention",
                )
                self.assertIn(
                    name, names,
                    f"{config['architectures'][0]} {phase} missing MLA "
                    f"attention op {name}",
                )

    def test_mla_ops_use_xpu_backends(self):
        # Regression: MLA must not fall back to CPU on XPU.
        for config in (_DEEPSEEK_V2_CONFIG, _GLM_MOE_DSA_CONFIG):
            graph = self._build(config)
            for phase in ("prefill", "decode"):
                for op in _collect_ops(graph[phase]):
                    self.assertIn(
                        op["backend"],
                        _XPU_SUPPORTED_BACKENDS,
                        f"{config['architectures'][0]} {phase} op "
                        f"{op['name']} fell back to {op['backend']}",
                    )


# MiniMax-M3 is a vision-language MoE model with sparse attention. vLLM-XPU now
# supports it, so the static graph builder must handle its nested ``text_config``
# layout (text params under text_config, vision under vision_config), the
# per-layer ``moe_layer_freq`` dense/MoE split, and the separate dense vs. MoE
# intermediate sizes. M3 uses standard GQA (not MLA); the sparse indexer is not
# modeled as separate ops (same simplification as GLM5MoE DSA).
_MINIMAX_M3_CONFIG = {
    "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
    "model_type": "minimax_m3_vl",
    "torch_dtype": "bfloat16",
    "projector_hidden_act": "gelu",
    "vision_feature_layer": -1,
    "text_config": {
        "hidden_size": 6144,
        "intermediate_size": 3072,
        "dense_intermediate_size": 12288,
        "shared_intermediate_size": 3072,
        "num_hidden_layers": 60,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 200064,
        "use_qk_norm": True,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "n_shared_experts": 1,
        # First 3 layers dense (leading zeros), remainder MoE.
        "moe_layer_freq": [0, 0, 0] + [1] * 57,
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_index_dim": 128,
            "sparse_num_index_heads": 4,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
        },
    },
    "vision_config": {
        "hidden_size": 1280,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "intermediate_size": 5120,
        "patch_size": 14,
        "image_size": 2016,
    },
}


class TestMiniMaxM3ModelGraph(unittest.TestCase):
    """MiniMax-M3 (VL + MoE + sparse attention) is supported on XPU — the static
    graph must build from its nested text_config/vision_config layout and route
    every op to an XPU backend."""

    def _build(self, config: dict) -> dict:
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config

        summary = summarize_config(config)
        return build_model_graph(
            summary, prefill_len=128, decode_batch=1, context_len=2048
        )

    def test_summary_reads_nested_text_config(self):
        from breakdown.model_info import summarize_config

        summary = summarize_config(_MINIMAX_M3_CONFIG)
        self.assertEqual(summary["hidden_size"], 6144)
        self.assertEqual(summary["num_layers"], 60)
        self.assertEqual(summary["num_heads"], 64)
        self.assertEqual(summary["num_kv_heads"], 4)
        self.assertTrue(summary["is_moe"])
        self.assertEqual(summary["num_experts"], 128)
        self.assertEqual(summary["num_experts_per_tok"], 4)
        self.assertEqual(summary["n_shared_experts"], 1)
        # dense MLP uses dense_intermediate_size; experts use intermediate_size
        self.assertEqual(summary["intermediate_size"], 12288)
        self.assertEqual(summary["moe_intermediate_size"], 3072)
        # leading zeros of moe_layer_freq → dense prefix
        self.assertEqual(summary["first_k_dense_replace"], 3)
        # vision encoder dimensions resolved from vision_config
        self.assertEqual(summary["vit_hidden_size"], 1280)
        self.assertEqual(summary["vit_num_layers"], 32)

    def test_graph_builds(self):
        graph = self._build(_MINIMAX_M3_CONFIG)
        self.assertEqual(graph["family"], "MiniMaxM3")
        self.assertEqual(graph["model_type"], "mllm")
        self.assertIsNotNone(graph["prefill"])
        self.assertIsNotNone(graph["decode"])

    def test_hybrid_dense_moe_split(self):
        graph = self._build(_MINIMAX_M3_CONFIG)
        types = {
            ch["module_type"]: ch.get("repeat_count")
            for ch in graph["prefill"]["children"]
        }
        self.assertEqual(types.get("MiniMaxM3DenseLayer"), 3)
        self.assertEqual(types.get("MiniMaxM3MoELayer"), 57)

    def test_vision_encoder_and_shared_experts_present(self):
        graph = self._build(_MINIMAX_M3_CONFIG)
        roles = {o.get("role") for o in _collect_ops(graph["prefill"])}
        self.assertTrue(any(r and r.startswith("vit") for r in roles),
                        "vision encoder ops missing")
        self.assertIn("vl_projector", roles)
        self.assertTrue(any(r and "shared_expert" in r for r in roles),
                        "shared expert ops missing")
        self.assertIn("q_norm", roles)  # use_qk_norm=True

    def test_sparse_attention_indexer_in_moe_layers(self):
        # MoE (sparse) layers carry the lightning indexer + top-k + Triton
        # block-sparse attention; the dense prefix layers keep full attention
        # (no indexer). On XPU M3 dispatches to its own Triton kernels
        # (minimax_m3_*) and a fused vllm-xpu-kernels qknorm/rope/insert op,
        # not the DeepSeek indexer ops or flash_attn_varlen_fwd.
        graph = self._build(_MINIMAX_M3_CONFIG)

        def _attn_names(layer_type, phase):
            for ch in graph[phase]["children"]:
                if ch["module_type"] == layer_type:
                    for sub in ch["children"]:
                        if sub["name"] == "self_attn":
                            return {o["name"] for o in sub["ops"]}
            return set()

        for phase, attn, score in (
            ("prefill", "minimax_m3_sparse_attn", "minimax_m3_index_score"),
            ("decode", "minimax_m3_sparse_attn_decode", "minimax_m3_index_decode"),
        ):
            moe_names = _attn_names("MiniMaxM3MoELayer", phase)
            # Fused qknorm/rope/kv-insert custom op replaces norm/rotary/cache.
            self.assertIn("fused_minimax_m3_qknorm_rope_kv_insert", moe_names)
            # Triton lightning-indexer score + the Triton block-sparse attention.
            self.assertIn(score, moe_names)
            self.assertIn(attn, moe_names)
            # The sparse path must NOT fall back to dense flash/paged attention.
            self.assertNotIn("flash_attn_varlen_fwd", moe_names)
            self.assertNotIn("paged_attention", moe_names)
            # Prefill runs a dedicated top-k kernel; decode fuses it into score.
            if phase == "prefill":
                self.assertIn("minimax_m3_index_topk", moe_names)
            # Dense prefix layers keep full attention, no indexer.
            dense_names = _attn_names("MiniMaxM3DenseLayer", phase)
            self.assertNotIn("minimax_m3_index_score", dense_names)
            self.assertNotIn("minimax_m3_index_decode", dense_names)
            self.assertNotIn("fused_minimax_m3_qknorm_rope_kv_insert", dense_names)
            dense_attn = (
                "flash_attn_varlen_fwd" if phase == "prefill" else "paged_attention"
            )
            self.assertIn(dense_attn, dense_names)

    def test_ops_use_xpu_backends(self):
        graph = self._build(_MINIMAX_M3_CONFIG)
        for phase in ("prefill", "decode"):
            for op in _collect_ops(graph[phase]):
                self.assertIn(
                    op["backend"],
                    _XPU_SUPPORTED_BACKENDS,
                    f"MiniMax-M3 {phase} op {op['name']} fell back to "
                    f"{op['backend']}",
                )


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
        # attention: context dim 64 => C, total kv 72 => S+C
        op("vllm::unified_attention_with_output", 10, 4,
           [[8, 48], [64, 16], [72, 16]], 3.0)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"traceEvents": events}, f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1, query_len=8, context_len=64)
        finally:
            os.unlink(path)

        self.assertEqual(g["symbols"]["C"], 64)
        self.assertEqual(g["symbols"]["S+C"], 72)

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
