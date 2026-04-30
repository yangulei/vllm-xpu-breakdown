#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for profiling with reduced layers.

These tests exercise the real vLLM profiling pipeline with dummy weights
and reduced layer count — the exact scenario that caused KeyError when
real weights were loaded for non-existent layers.

Requires XPU hardware. Run with: pytest tests/test_profile_reduced_layers.py -v
"""

import os
import shutil
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip all tests if XPU not available
try:
    import torch
    HAS_XPU = torch.xpu.is_available()
except (ImportError, AttributeError):
    HAS_XPU = False

pytestmark = pytest.mark.skipif(not HAS_XPU, reason="XPU hardware not available")

TRACE_DIR = os.path.abspath("output/test_traces")
MODEL_ID = "facebook/opt-125m"


@pytest.fixture(autouse=True)
def clean_trace_dir():
    """Ensure clean trace directory for each test."""
    if os.path.exists(TRACE_DIR):
        shutil.rmtree(TRACE_DIR)
    os.makedirs(TRACE_DIR, exist_ok=True)
    yield
    # Cleanup after test
    if os.path.exists(TRACE_DIR):
        shutil.rmtree(TRACE_DIR)


def _create_engine(num_hidden_layers=None):
    """Create a vLLM engine with profiling config and optional layer reduction."""
    from vllm import LLM

    kwargs = {
        "model": MODEL_ID,
        "max_model_len": 128,
        "enforce_eager": True,
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": TRACE_DIR,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": True,
            "torch_profiler_use_gzip": False,
        },
    }

    if num_hidden_layers is not None:
        kwargs["hf_overrides"] = {"num_hidden_layers": num_hidden_layers}
        kwargs["load_format"] = "dummy"

    return LLM(**kwargs)


def _get_trace_files():
    """Return sorted trace files (newest first)."""
    time.sleep(1)  # Allow trace flush
    files = []
    for f in os.listdir(TRACE_DIR):
        if f.endswith(".json") or f.endswith(".json.gz"):
            files.append(os.path.join(TRACE_DIR, f))
    return sorted(files, key=os.path.getmtime, reverse=True)


class TestReducedLayerProfiling:
    """Test profiling with reduced layers uses dummy weights correctly."""

    def test_single_layer_dummy_weights_no_keyerror(self):
        """Reduced-layer profiling must not crash with KeyError on weight loading.

        This was the original bug: setting num_hidden_layers=1 via hf_overrides
        caused vLLM to try loading checkpoint weights for layers that don't exist.
        Fix: use load_format='dummy' for reduced-layer profiling.
        """
        # opt-125m has 12 layers; reduce to 1
        llm = _create_engine(num_hidden_layers=1)
        assert llm is not None
        del llm

    def test_reduced_layer_profiling_produces_trace(self):
        """Profile with 1 layer and verify a trace file is produced."""
        from vllm import SamplingParams

        llm = _create_engine(num_hidden_layers=1)

        sampling = SamplingParams(max_tokens=8)
        prompt = "Hello, how are you?"

        # Warmup
        llm.generate([prompt], sampling, use_tqdm=False)

        # Profiled run
        llm.start_profile()
        llm.generate([prompt], sampling, use_tqdm=False)
        llm.stop_profile()

        del llm

        trace_files = _get_trace_files()
        assert len(trace_files) >= 1, f"No trace files in {TRACE_DIR}"
        assert os.path.getsize(trace_files[0]) > 1000, "Trace file too small"

    def test_reduced_layer_trace_parseable(self):
        """Trace from reduced-layer profiling must be parseable by our parser."""
        from vllm import SamplingParams

        from breakdown.trace_parser import parse_trace_file

        llm = _create_engine(num_hidden_layers=1)
        sampling = SamplingParams(max_tokens=8)
        prompt = "What is 2+2?"

        llm.generate([prompt], sampling, use_tqdm=False)
        llm.start_profile()
        llm.generate([prompt], sampling, use_tqdm=False)
        llm.stop_profile()
        del llm

        trace_files = _get_trace_files()
        assert trace_files, "No trace files produced"

        ops = parse_trace_file(trace_files[0])
        assert len(ops) > 0, "No ops parsed from trace"

        # Should contain typical ops (matmul, linear, etc.)
        op_names = {o["name"] for o in ops}
        assert len(op_names) > 1, f"Only got ops: {op_names}"

    def test_full_profile_pipeline_with_reduced_layers(self):
        """End-to-end test of _run_profile with reduced layers."""
        from app import _run_profile, _profile_state

        # _run_profile writes to output/traces/ — clean up after to avoid
        # interfering with TestRealTrace in test_pipeline.py
        main_trace_dir = os.path.abspath("output/traces")
        existing_before = set(os.listdir(main_trace_dir)) if os.path.isdir(main_trace_dir) else set()

        _run_profile(
            model_id=MODEL_ID,
            mode="eager",
            max_model_len=128,
            batch_size=1,
            max_tokens=8,
            prompt="What is 1+1?",
            num_profile_layers=1,
            tp_size=1,
        )

        assert _profile_state["status"] == "done", (
            f"Profile failed: {_profile_state.get('error', 'unknown')}"
        )
        result = _profile_state["result"]
        assert result["model_id"] == MODEL_ID
        assert result["layer_scale"] == 12  # opt-125m has 12 layers, profiled 1
        assert len(result["ops"]) > 0
        assert result["backends"]

        # Cleanup: remove traces written by this test
        if os.path.isdir(main_trace_dir):
            for f in os.listdir(main_trace_dir):
                if f not in existing_before:
                    os.remove(os.path.join(main_trace_dir, f))
