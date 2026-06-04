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

    def test_summarize_nested_text_config(self):
        """Multimodal models nest LLM dims under text_config (regression for
        the Qwen3.6-27B 'NoneType // int' graph crash)."""
        config = {
            "architectures": ["FakeVLForConditionalGeneration"],
            "model_type": "fake_vl",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 25600,
                "vocab_size": 152064,
                "max_position_embeddings": 262144,
            },
            "vision_config": {
                "hidden_size": 1280,
                "num_hidden_layers": 32,
            },
        }
        summary = summarize_config(config)
        # Core LLM dims must resolve from the nested text_config, not be None.
        self.assertEqual(summary["hidden_size"], 5120)
        self.assertEqual(summary["num_layers"], 64)
        self.assertEqual(summary["num_heads"], 40)
        self.assertEqual(summary["num_kv_heads"], 8)
        self.assertEqual(summary["head_dim"], 128)
        self.assertEqual(summary["intermediate_size"], 25600)
        self.assertEqual(summary["vocab_size"], 152064)

    def test_graph_build_nested_text_config_requires_raw_config(self):
        """Decoder graph building now requires raw_config (static builders
        removed). Without it, build_model_graph must hard-fail with a clear
        error rather than silently producing a static graph."""
        from breakdown.model_graph import build_model_graph
        config = {
            "architectures": ["FakeVLForConditionalGeneration"],
            "model_type": "fake_vl",
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 14336,
                "vocab_size": 152064,
            },
            "vision_config": {"hidden_size": 1280},
        }
        summary = summarize_config(config)
        with self.assertRaises(ValueError) as ctx:
            build_model_graph(summary, prefill_len=128)
        self.assertIn("raw_config", str(ctx.exception))

    def test_summarize_missing_dims_hard_fails_without_raw_config(self):
        """A config with no resolvable LLM dims and no raw_config must raise a
        clear error (the static graceful-degradation path has been removed)."""
        from breakdown.model_graph import build_model_graph
        config = {"architectures": ["MysteryForCausalLM"], "model_type": "mystery"}
        summary = summarize_config(config)
        with self.assertRaises(ValueError) as ctx:
            build_model_graph(summary, prefill_len=128)
        self.assertIn("raw_config", str(ctx.exception))

    def test_unsupported_vl_family_hard_fails(self):
        """A VL family not validated for tracing must hard-fail with a clear
        error even when raw_config IS provided (no static VL fallback)."""
        from breakdown.model_graph import build_model_graph
        config = {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "intermediate_size": 8192,
                "vocab_size": 152064,
            },
            "vision_config": {"hidden_size": 1280},
        }
        summary = summarize_config(config)
        with self.assertRaises(ValueError) as ctx:
            build_model_graph(summary, prefill_len=128, raw_config=config)
        self.assertIn("VL family", str(ctx.exception))


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
        self.assertIn(b"vLLM-XPU", resp.data)

    def test_demo_endpoint_ok(self):
        resp = self.client.get("/api/demo")
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertGreater(len(data["data"]["ops"]), 0)

    def test_demo_no_overhead(self):
        resp = self.client.get("/api/demo")
        data = resp.get_json()
        names = [op["name"] for op in data["data"]["ops"]]
        self.assertNotIn("ProfilerStep*", names)

    def test_demo_ops_fields(self):
        resp = self.client.get("/api/demo")
        data = resp.get_json()
        required = {"name", "backend", "input_shapes", "dtype",
                    "call_count", "layer_count", "device_time_us",
                    "memory_bytes", "flops", "arithmetic_intensity"}
        for op in data["data"]["ops"]:
            for field in required:
                self.assertIn(field, op, f"Missing '{field}' in {op['name']}")

    def test_demo_backend_pct_sums_to_100(self):
        resp = self.client.get("/api/demo")
        data = resp.get_json()
        total_pct = sum(b["pct"] for b in data["data"]["backends"].values())
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

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
        ops = parse_trace_file(self.trace_file)
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


# Backends ops are allowed to run on (XPU-supported, no CPU fallback).
_XPU_SUPPORTED_BACKENDS = {"vllm-xpu-kernels", "torch-xpu-ops", "triton", "framework"}


def _collect_ops(node: dict) -> list[dict]:
    """Flatten all OpNode dicts in a serialized model-graph tree."""
    ops = list(node.get("ops", []))
    for child in node.get("children", []):
        ops.extend(_collect_ops(child))
    return ops



# ---- Trace-based graph builder (model_tracer) ----

_TINY_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 4, "vocab_size": 1000,
    "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16", "tie_word_embeddings": False,
}

# Small DeepSeek-V2: exercises MLA attention + hybrid (1 dense + 2 MoE) layers.
_TINY_DEEPSEEK_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
    "hidden_size": 256, "intermediate_size": 512, "moe_intermediate_size": 128,
    "num_hidden_layers": 3, "num_attention_heads": 8, "num_key_value_heads": 8,
    "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
    "first_k_dense_replace": 1, "moe_layer_freq": 1, "vocab_size": 1000,
    "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
    "q_lora_rank": None, "kv_lora_rank": 128, "qk_rope_head_dim": 32,
    "qk_nope_head_dim": 64, "v_head_dim": 64, "topk_method": "greedy",
    "n_group": 1, "topk_group": 1, "routed_scaling_factor": 1.0,
    "scoring_func": "softmax", "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
}

# Small Qwen2.5-VL: exercises the VL path — text-only export captures the LM
# stack, the vision tower + projector are spliced analytically.
_TINY_QWEN2_5_VL_CONFIG = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "model_type": "qwen2_5_vl",
    "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 4, "vocab_size": 1000,
    "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16", "tie_word_embeddings": False,
    "vision_config": {
        "hidden_size": 128, "depth": 2, "num_heads": 8,
        "intermediate_size": 256, "patch_size": 14, "spatial_merge_size": 2,
        "in_chans": 3, "out_hidden_size": 256,
    },
    "rope_scaling": {"type": "mrope", "mrope_section": [4, 6, 6]},
}


def _tracer_available() -> bool:
    """torch.export tracing needs torch (with XPU) + vLLM importable."""
    try:
        import torch  # noqa: F401
        import vllm  # noqa: F401
    except Exception:
        return False
    return bool(getattr(getattr(torch, "xpu", None), "is_available", bool)())


@unittest.skipUnless(_tracer_available(),
                     "torch+vllm+XPU required for trace-based graph builder")
class TestTracedGraph(unittest.TestCase):
    """The torch.export builder derives the real op graph + backends with no
    per-architecture builder code."""

    @classmethod
    def setUpClass(cls):
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        summary = summarize_config(_TINY_LLAMA_CONFIG)
        cls.graph = build_model_graph(
            summary, prefill_len=128, decode_batch=1, context_len=4096,
            tp_size=1, quantization=None, raw_config=_TINY_LLAMA_CONFIG,
        )

    def test_uses_export_source(self):
        self.assertEqual(self.graph.get("graph_source"), "torch.export")

    def test_schema_parity(self):
        for key in ("architecture", "family", "model_type", "symbols",
                    "config", "prefill", "decode"):
            self.assertIn(key, self.graph)
        self.assertEqual(self.graph["architecture"], "LlamaForCausalLM")

    def test_layers_merged_with_repeat_count(self):
        # The repeated decoder layer must collapse to one node x num_layers.
        def find(node):
            if node.get("name") == "decoder_layer":
                return node
            for c in node.get("children", []):
                r = find(c)
                if r:
                    return r
            return None
        layer = find(self.graph["prefill"])
        self.assertIsNotNone(layer, "decoder_layer node not found")
        self.assertEqual(layer["repeat_count"],
                         _TINY_LLAMA_CONFIG["num_hidden_layers"])

    def test_real_ops_and_backends(self):
        ops = _collect_ops(self.graph["prefill"])
        names = {op["name"] for op in ops}
        backends = {op["name"]: op["backend"] for op in ops}
        # Real dispatched op names captured from the trace.
        self.assertIn("aten::linear", names)
        self.assertIn("silu_and_mul", names)
        self.assertIn("unified_attention", names)
        # Steady-state layers use the residual-fused norm.
        self.assertTrue(
            {"rms_norm", "fused_add_rms_norm"} & names,
            "no RMSNorm op captured")
        # Backends resolved through the shared classifier (single source).
        self.assertEqual(backends["aten::linear"], "torch-xpu-ops")
        self.assertEqual(backends["silu_and_mul"], "vllm-xpu-kernels")
        self.assertEqual(backends["unified_attention"], "vllm-xpu-kernels")
        norm_op = "fused_add_rms_norm" if "fused_add_rms_norm" in backends \
            else "rms_norm"
        self.assertEqual(backends[norm_op], "vllm-xpu-kernels")

    def test_ops_route_to_xpu_backends(self):
        for phase in ("prefill", "decode"):
            for op in _collect_ops(self.graph[phase]):
                self.assertIn(op["backend"], _XPU_SUPPORTED_BACKENDS,
                              f"{phase} op {op['name']} -> {op['backend']}")

    def test_token_dim_substituted_per_phase(self):
        # Prefill (128 tokens) must cost more than decode (1 token).
        self.assertGreater(self.graph["prefill"]["total_memory"],
                           self.graph["decode"]["total_memory"])
        self.assertGreater(self.graph["prefill"]["total_flops"],
                           self.graph["decode"]["total_flops"])

    def test_decode_attention_scales_with_context(self):
        # Decode attention reads the KV cache over context, so a longer context
        # must raise decode memory (regression: context was previously unused).
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        summary = summarize_config(_TINY_LLAMA_CONFIG)

        def decode_mem(ctx):
            g = build_model_graph(summary, prefill_len=128, decode_batch=1,
                                  context_len=ctx, raw_config=_TINY_LLAMA_CONFIG)
            return g["decode"]["total_memory"]
        self.assertGreater(decode_mem(8192), decode_mem(512))

    def test_tp_uses_tracer_and_shards_cost(self):
        # TP>1 is handled analytically by the tracer (export at TP=1, divide
        # per-rank cost by op role). Sharded ops (attention, projections) halve
        # at TP=2; replicated ops (norms) are unchanged.
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        summary = summarize_config(_TINY_LLAMA_CONFIG)
        g1 = build_model_graph(summary, prefill_len=128, decode_batch=1,
                               tp_size=1, raw_config=_TINY_LLAMA_CONFIG)
        g2 = build_model_graph(summary, prefill_len=128, decode_batch=1,
                               tp_size=2, raw_config=_TINY_LLAMA_CONFIG)
        self.assertEqual(g2.get("graph_source"), "torch.export")

        def role_flops(graph, role):
            return sum(o.get("flops", 0) for o in _collect_ops(graph["prefill"])
                       if o.get("role") == role)

        def role_mem(graph, role):
            return sum(o.get("memory_bytes", 0)
                       for o in _collect_ops(graph["prefill"])
                       if o.get("role") == role)

        attn1, attn2 = role_flops(g1, "attention"), role_flops(g2, "attention")
        self.assertGreater(attn1, 0)
        self.assertAlmostEqual(attn2 / attn1, 0.5, places=2)  # sharded
        norm1, norm2 = role_mem(g1, "norm"), role_mem(g2, "norm")
        self.assertGreater(norm1, 0)
        self.assertEqual(norm1, norm2)  # replicated

    def test_uniform_moe_uses_tracer_with_spliced_experts(self):
        # A model whose every layer is a MoE layer (uniform) is traced, and the
        # opaque FusedMoE export op is replaced with real expert/shared-expert
        # GEMM+activation ops spliced in from the shared FusedMoE builder.
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        moe = {
            "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
            "hidden_size": 256, "intermediate_size": 512,
            "moe_intermediate_size": 128, "shared_expert_intermediate_size": 256,
            "num_hidden_layers": 2, "num_attention_heads": 8,
            "num_key_value_heads": 4, "vocab_size": 1000,
            "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
            "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
            "norm_topk_prob": True, "torch_dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
        g = build_model_graph(summarize_config(moe), prefill_len=128,
                              decode_batch=1, raw_config=moe)
        self.assertEqual(g.get("graph_source"), "torch.export+fused_moe")
        roles = {op.get("role") for op in _collect_ops(g["prefill"])}
        # Routed + shared expert compute recovered from the opaque export op.
        for expected in ("expert_gate_up", "expert_activation", "expert_down",
                         "shared_expert_gate_up", "shared_expert_down"):
            self.assertIn(expected, roles)
        # The opaque FusedMoE op must NOT be counted as a real compute op.
        names = {op.get("name") for op in _collect_ops(g["prefill"])}
        self.assertFalse(any(n and "moe_forward" in n for n in names))

    def test_hybrid_moe_traces_with_grouped_layers(self):
        # A hybrid stack with interleaved dense + MoE layers is traced: layers
        # are grouped by op signature into separate repeated decoder-layer nodes
        # (one dense, one MoE), instead of falling back to the static builder.
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        moe = {
            "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
            "hidden_size": 256, "intermediate_size": 512,
            "moe_intermediate_size": 128, "shared_expert_intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8, "num_key_value_heads": 4,
            "vocab_size": 1000, "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5, "num_experts": 4, "num_experts_per_tok": 2,
            "decoder_sparse_step": 2, "mlp_only_layers": [],
            "norm_topk_prob": True, "torch_dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
        g = build_model_graph(summarize_config(moe), prefill_len=128,
                              decode_batch=1, raw_config=moe)
        self.assertEqual(g.get("graph_source"), "torch.export+fused_moe")

        # Collect the decoder-layer groups and the roles each contains.
        groups = []

        def walk(node):
            if node.get("name") == "decoder_layer":
                roles = set()

                def rs(m):
                    for o in m.get("ops", []):
                        roles.add(o.get("role"))
                    for c in m.get("children", []):
                        rs(c)
                rs(node)
                groups.append((node.get("repeat_count"), roles))
            for c in node.get("children", []):
                walk(c)
        walk(g["prefill"])

        # Two distinct decoder-layer groups: one MoE (expert ops), one dense.
        self.assertEqual(len(groups), 2)
        moe_groups = [gr for gr in groups if "expert_gate_up" in gr[1]]
        dense_groups = [gr for gr in groups if "activation" in gr[1]
                        and "expert_gate_up" not in gr[1]]
        self.assertEqual(len(moe_groups), 1)
        self.assertEqual(len(dense_groups), 1)
        # decoder_sparse_step=2 over 4 layers → 2 MoE + 2 dense.
        self.assertEqual(moe_groups[0][0], 2)
        self.assertEqual(dense_groups[0][0], 2)

    def test_deepseek_mla_hybrid_traces(self):
        # DeepSeek-V2 exercises MLA attention + a hybrid stack (1 leading dense
        # layer, then MoE layers). The tracer must classify the MLA attention
        # ops, group the layers into a dense node + a MoE node, and splice the
        # FusedMoE experts — all via torch.export, no static fallback.
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        g = build_model_graph(summarize_config(_TINY_DEEPSEEK_CONFIG),
                              prefill_len=128, decode_batch=1,
                              raw_config=_TINY_DEEPSEEK_CONFIG)
        self.assertEqual(g.get("graph_source"), "torch.export+fused_moe")

        groups = []

        def walk(node):
            if node.get("name") == "decoder_layer":
                roles = set()

                def rs(m):
                    for o in m.get("ops", []):
                        roles.add(o.get("role"))
                    for c in m.get("children", []):
                        rs(c)
                rs(node)
                groups.append((node.get("repeat_count"), roles))
            for c in node.get("children", []):
                walk(c)
        walk(g["prefill"])

        # One dense group (count 1) and one MoE group (count 2).
        self.assertEqual(len(groups), 2)
        moe = [gr for gr in groups if "expert_gate_up" in gr[1]]
        dense = [gr for gr in groups if "expert_gate_up" not in gr[1]]
        self.assertEqual(len(moe), 1)
        self.assertEqual(len(dense), 1)
        self.assertEqual(dense[0][0], 1)   # first_k_dense_replace=1
        self.assertEqual(moe[0][0], 2)     # remaining 2 layers
        # MLA attention is classified and counted in both groups.
        self.assertIn("attention", moe[0][1])
        self.assertIn("attention", dense[0][1])

    def test_qwen2_5_vl_traces_lm_and_splices_vision(self):
        # VL models export text-only (no image inputs), so torch.export captures
        # just the language-model stack — the vision tower + projector are spliced
        # analytically (prefill only). The merged graph must report the vision
        # ops in prefill, none in decode, and still trace the LM attention.
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        g = build_model_graph(summarize_config(_TINY_QWEN2_5_VL_CONFIG),
                              prefill_len=128, decode_batch=1,
                              raw_config=_TINY_QWEN2_5_VL_CONFIG)
        self.assertEqual(g.get("graph_source"), "torch.export+vision")

        def roles(node, acc):
            for o in node.get("ops", []):
                acc.add(o.get("role"))
            for c in node.get("children", []):
                roles(c, acc)
            return acc

        pre = roles(g["prefill"], set())
        dec = roles(g["decode"], set())
        # Vision tower + projector present in prefill...
        self.assertIn("vit_attention", pre)
        self.assertIn("patch_embed", pre)
        self.assertIn("vl_projector", pre)
        # ...the LM stack is traced (real attention op present)...
        self.assertIn("attention", pre)
        # ...and vision is absent from the autoregressive decode phase.
        self.assertFalse({"vit_attention", "patch_embed", "vl_projector"} & dec)
        # Vision subtree precedes the language-model stack.
        names = [c.get("name") for c in g["prefill"].get("children", [])]
        self.assertIn("visual_encoder", names)
        self.assertIn("language_model", names)
        self.assertLess(names.index("visual_encoder"),
                        names.index("language_model"))
        # The spliced vision subtree is aggregated into totals (guards against a
        # future refactor inserting it after _compute_totals): the root prefill
        # total must exceed the language-model subtree's own total.
        children = g["prefill"]["children"]
        vis = next(c for c in children if c["name"] == "visual_encoder")
        lm = next(c for c in children if c["name"] == "language_model")
        self.assertGreater(vis["total_flops"], 0)
        self.assertGreaterEqual(g["prefill"]["total_flops"],
                                vis["total_flops"] + lm["total_flops"])

    def test_moe_tp_shards_expert_compute(self):
        # Under TP>1 the spliced FusedMoE expert GEMMs are sharded on the expert
        # intermediate; per-rank expert FLOPs halve at TP=2.
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        moe = {
            "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
            "hidden_size": 256, "intermediate_size": 512,
            "moe_intermediate_size": 128, "shared_expert_intermediate_size": 256,
            "num_hidden_layers": 2, "num_attention_heads": 8,
            "num_key_value_heads": 4, "vocab_size": 1000,
            "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
            "num_experts": 4, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
            "norm_topk_prob": True, "torch_dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
        summary = summarize_config(moe)
        g1 = build_model_graph(summary, prefill_len=128, decode_batch=1,
                               tp_size=1, raw_config=moe)
        g2 = build_model_graph(summary, prefill_len=128, decode_batch=1,
                               tp_size=2, raw_config=moe)
        self.assertEqual(g2.get("graph_source"), "torch.export+fused_moe")

        def expert_flops(graph):
            return sum(o.get("flops", 0) for o in _collect_ops(graph["prefill"])
                       if o.get("role", "").startswith("expert_"))

        e1, e2 = expert_flops(g1), expert_flops(g2)
        self.assertGreater(e1, 0)
        self.assertAlmostEqual(e2 / e1, 0.5, places=2)

    def test_quant_traces_and_reduces_weight_bytes(self):
        # fp8 export breaks inside vLLM on meta tensors, so the tracer strips
        # quantization, traces in BF16, and applies reduced weight precision
        # analytically: projection memory drops (weights at 1B) while activations
        # stay BF16 — and the model still goes through the tracer (no fallback).
        from breakdown.model_graph import build_model_graph
        from breakdown.model_info import summarize_config
        q = dict(_TINY_LLAMA_CONFIG)
        q["quantization_config"] = {"quant_method": "fp8",
                                    "activation_scheme": "static"}
        sb = summarize_config(_TINY_LLAMA_CONFIG)
        sq = summarize_config(q)
        gb = build_model_graph(sb, prefill_len=128, decode_batch=1,
                               tp_size=1, raw_config=_TINY_LLAMA_CONFIG)
        gq = build_model_graph(sq, prefill_len=128, decode_batch=1,
                               tp_size=1, raw_config=q)
        self.assertEqual(gq.get("graph_source"), "torch.export")

        def proj_mem(graph):
            return sum(o.get("memory_bytes", 0)
                       for o in _collect_ops(graph["prefill"])
                       if o.get("role") == "proj")

        mb, mq = proj_mem(gb), proj_mem(gq)
        self.assertGreater(mb, 0)
        self.assertLess(mq, mb)       # weights at reduced precision
        self.assertGreater(mq, mb / 2)  # activations/outputs stay BF16

    def test_tp_shard_factor_keys_off_parallel_layer_class(self):
        # The TP sharding signal is vLLM's parallel-layer CLASS, not the module
        # attribute name — so architectures that name their blocks differently
        # (.feed_forward / .attention / .block_sparse_moe) still shard correctly.
        from breakdown.model_tracer import _tp_shard_factor as f
        # A column-parallel projection inside a non-".mlp"-named block shards.
        cls = {"model.layers.0.feed_forward.w1": "MergedColumnParallelLinear"}
        self.assertEqual(
            f("model.layers.0.feed_forward.w1", "proj", "aten::linear", cls, 4), 4)
        # A row-parallel projection inside a ".attention"-named block shards.
        cls = {"model.layers.0.attention.dense": "RowParallelLinear"}
        self.assertEqual(
            f("model.layers.0.attention.dense", "proj", "aten::linear", cls, 2), 2)
        # The MoE router gate is a ReplicatedLinear → stays full.
        cls = {"model.layers.0.block_sparse_moe.gate": "ReplicatedLinear"}
        self.assertEqual(
            f("model.layers.0.block_sparse_moe.gate", "proj", "aten::linear",
              cls, 8), 1)
        # Vocab-parallel embedding lookup output is all-reduced → stays full.
        cls = {"model.embed_tokens": "VocabParallelEmbedding"}
        self.assertEqual(
            f("model.embed_tokens", "embedding", "aten::embedding", cls, 4), 1)
        # ParallelLMHead is vocab-parallel on the output → shards.
        cls = {"lm_head": "ParallelLMHead"}
        self.assertEqual(f("lm_head", "proj", "aten::linear", cls, 4), 4)
        # Non-linear head/intermediate-sharded ops shard by role.
        for role in ("attention", "rotary_emb", "cache_store", "activation"):
            self.assertEqual(f("p", role, "x", {}, 4), 4)
        # Replicated norms / residuals stay full; TP=1 is always 1.
        self.assertEqual(f("p", "norm", "rms_norm", {}, 4), 1)
        self.assertEqual(f("p", "proj", "aten::linear",
                           {"p": "RowParallelLinear"}, 1), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
