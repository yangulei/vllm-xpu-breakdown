#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for capture-time kernel-launch spans.

A kernel launched straight from Python (Triton, a pybind11 extension) leaves no
``cpu_op``, so the trace records neither its operands nor a replay entry point.
:mod:`breakdown.kernel_hooks` opens a ``kernel::`` span at the launch carrying
both; these tests cover the label round-trip, the live-stack launcher lookup,
the argument description, and end-to-end reconstruction of a synthetic op that
carries real shapes and argument slots.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown import kernel_hooks
from breakdown.trace import build_graph_from_trace
from breakdown.trace_common import (
    is_launcher_frame,
    kernel_span_label,
    parse_kernel_span,
)

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in this repo
    torch = None


class TestKernelSpanHelpers(unittest.TestCase):
    def test_label_roundtrip(self):
        payload = {"file": "/k/norm.py", "line": 12, "func": "gemma_rmsnorm",
                   "args": [{"kind": "tensor", "name": "x", "dims": [8, 16],
                             "dtype": "bfloat16", "strides": [16, 1]}]}
        self.assertEqual(parse_kernel_span(kernel_span_label(payload)), payload)

    def test_not_a_kernel_span(self):
        self.assertIsNone(parse_kernel_span("module::model.layers.0::Layer"))
        self.assertIsNone(parse_kernel_span("kernel::not base64!"))

    def test_launcher_frame_rule(self):
        self.assertTrue(is_launcher_frame("/k/norm.py", "gemma_rmsnorm"))
        # dispatch machinery: the mechanism, not the kernel
        self.assertFalse(is_launcher_frame("/x/triton/runtime/jit.py", "run"))
        self.assertFalse(is_launcher_frame("/x/torch/_ops.py", "gemma_rmsnorm"))
        self.assertFalse(is_launcher_frame("/k/norm.py", "_private"))
        self.assertFalse(is_launcher_frame("/k/norm.py", "<listcomp>"))


@unittest.skipIf(torch is None, "torch required")
class TestArgumentDescription(unittest.TestCase):
    def test_slot_kinds(self):
        t = torch.zeros(4, 8, dtype=torch.float16)
        self.assertEqual(
            kernel_hooks._describe(t, "x"),
            {"kind": "tensor", "name": "x", "dims": [4, 8],
             "dtype": "float16", "strides": [8, 1]})
        self.assertEqual(kernel_hooks._describe(None, "opt"),
                         {"kind": "none", "name": "opt", "value": "None"})
        self.assertEqual(kernel_hooks._describe(1e-6, "eps"),
                         {"kind": "scalar", "name": "eps", "type": "float",
                          "value": "1e-06"})
        self.assertEqual(kernel_hooks._describe(True, "causal"),
                         {"kind": "scalar", "name": "causal", "type": "bool",
                          "value": "True"})
        self.assertEqual(kernel_hooks._describe([1, 2], "shape"),
                         {"kind": "scalar", "name": "shape", "type": "int[]",
                          "value": "[1, 2]"})
        lst = kernel_hooks._describe([t, t], "tensors")
        self.assertEqual(lst["kind"], "tensorlist")
        self.assertEqual(len(lst["items"]), 2)

    def test_unrepresentable_argument_is_reported_not_guessed(self):
        slot = kernel_hooks._describe(object(), "ctx")
        self.assertEqual(slot["kind"], "opaque")
        self.assertEqual(slot["type"], "object")


@unittest.skipIf(torch is None, "torch required")
class TestLiveCapture(unittest.TestCase):
    """The hook records the *launcher's* arguments, not the dispatcher's."""

    def test_span_records_launcher_and_operands(self):
        from torch.profiler import ProfilerActivity, profile

        import types

        # Stands in for ``xattention._C`` - a native kernel module.
        ext = types.ModuleType("fake_ext_C")
        ext.__file__ = "/fake/ext/_C.so"

        def sparse_attn(q, out, scale):
            out.add_(scale)

        ext.sparse_attn = sparse_attn

        def my_kernel_wrapper(q, output, sm_scale=0.5):
            """The public entry point a replay must call."""
            ext.sparse_attn(q, output, sm_scale)

        kernel_hooks._patch(ext, "sparse_attn", kernel_hooks._make_wrapper)
        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                my_kernel_wrapper(torch.zeros(2, 3), torch.zeros(2, 3))
        finally:
            kernel_hooks.remove_kernel_span_hooks()

        payloads = []
        for evt in _events(prof):
            payload = parse_kernel_span(evt.get("name", ""))
            if payload:
                payloads.append(payload)
        self.assertEqual(len(payloads), 1, payloads)
        got = payloads[0]
        self.assertEqual(got["func"], "my_kernel_wrapper")
        self.assertEqual(got["file"], __file__)
        names = [s["name"] for s in got["args"]]
        self.assertEqual(names, ["q", "output", "sm_scale"])
        self.assertEqual(got["args"][0]["dims"], [2, 3])
        self.assertEqual(got["args"][2]["value"], "0.5")

    def test_hooks_are_idempotent_and_restore(self):
        import types

        Ext = types.ModuleType("fake_ext2")
        Ext.k = lambda: 1
        original = Ext.k
        self.assertTrue(kernel_hooks._patch(Ext, "k", kernel_hooks._make_wrapper))
        self.assertFalse(kernel_hooks._patch(Ext, "k", kernel_hooks._make_wrapper))
        self.assertEqual(Ext.k(), 1)
        kernel_hooks.remove_kernel_span_hooks()
        self.assertIs(Ext.k, original)


def _events(prof):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.json")
        prof.export_chrome_trace(path)
        with open(path) as fh:
            return json.load(fh)["traceEvents"]


SUMMARY = {
    "architecture": "Qwen3ForCausalLM", "hidden_size": 16, "num_heads": 3,
    "num_kv_heads": 3, "head_dim": 16, "intermediate_size": 64,
    "vocab_size": 32000, "num_layers": 1, "dtype": "bfloat16",
    "model_id": "test/Kernel-Span",
}


def _trace_with_kernel_span(with_span: bool) -> dict:
    """One decoder layer whose norm is a Python-launched Triton kernel."""
    from breakdown.trace_common import module_span_label

    tid = 7
    events: list[dict] = []

    def span(qname, cls, ts, dur):
        events.append({"ph": "X", "cat": "user_annotation", "tid": tid,
                       "pid": tid, "ts": ts, "dur": dur,
                       "name": module_span_label(qname, cls)})

    span("", "Qwen3ForCausalLM", 0, 200)
    span("model", "Qwen3Model", 1, 198)
    span("model.layers.0", "Qwen3DecoderLayer", 10, 150)
    span("model.layers.0.input_layernorm", "RMSNorm", 20, 40)
    # A real dispatched op, so the phase partition sees a token dim.
    events.append({"ph": "X", "cat": "cpu_op", "tid": tid, "pid": tid,
                   "ts": 70, "dur": 20, "name": "aten::linear",
                   "args": {"External id": 1, "Input Dims": [[8, 16], [64, 16]],
                            "Input type": ["c10::BFloat16", "c10::BFloat16"]}})
    events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
                   "ts": 70, "dur": 1, "name": "launch",
                   "args": {"correlation": 1, "External id": 1}})
    events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                   "ts": 1070, "dur": 5, "name": "gemm",
                   "args": {"correlation": 1}})

    if with_span:
        payload = {
            "file": "/k/gemma_rmsnorm.py", "line": 109, "func": "gemma_rmsnorm",
            "args": [
                {"kind": "tensor", "name": "x", "dims": [8, 16],
                 "dtype": "bfloat16", "strides": [16, 1]},
                {"kind": "tensor", "name": "weight", "dims": [16],
                 "dtype": "bfloat16", "strides": [1]},
                {"kind": "scalar", "name": "eps", "type": "float",
                 "value": "1e-06"},
            ],
        }
        events.append({"ph": "X", "cat": "user_annotation", "tid": tid,
                       "pid": tid, "ts": 25, "dur": 20,
                       "name": kernel_span_label(payload)})
    else:
        events.append({"ph": "X", "cat": "python_function", "tid": tid,
                       "pid": tid, "ts": 25, "dur": 20,
                       "name": "/k/gemma_rmsnorm.py(109): gemma_rmsnorm"})
    events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
                   "ts": 30, "dur": 1, "name": "launch",
                   "args": {"correlation": 2, "External id": 99}})
    events.append({"ph": "X", "cat": "kernel", "tid": 99, "pid": 0,
                   "ts": 1030, "dur": 9, "name": "_gemma_rmsnorm_kernel",
                   "args": {"correlation": 2}})
    return {"traceEvents": events}


def _find_op(node, name):
    for op in node.get("ops", []):
        if op["name"] == name:
            return op
    for child in node.get("children", []):
        found = _find_op(child, name)
        if found:
            return found
    return None


class TestKernelSpanReconstruction(unittest.TestCase):
    """A span turns a shape-less synthetic op into a fully-specified one."""

    def _graph(self, with_span: bool):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w") as fh:
                json.dump(_trace_with_kernel_span(with_span), fh)
            return build_graph_from_trace(path, SUMMARY, tp_size=1,
                                          batch_size=1)

    def test_span_supplies_operands_and_entry_point(self):
        graph = self._graph(True)
        tree = graph.get("prefill") or graph.get("decode")
        op = _find_op(tree, "triton::_gemma_rmsnorm_kernel")
        self.assertIsNotNone(op, json.dumps(tree)[:2000])
        self.assertEqual(op["recorded_shapes"], [[8, 16], [16]])
        self.assertEqual(op["input_dtypes"], ["bfloat16", "bfloat16"])
        self.assertEqual([s["name"] for s in op["input_args"]],
                         ["x", "weight", "eps"])
        self.assertEqual(op["launch"], {"file": "/k/gemma_rmsnorm.py",
                                        "line": 109, "func": "gemma_rmsnorm"})

    def test_without_a_span_the_frame_still_names_the_entry_point(self):
        graph = self._graph(False)
        tree = graph.get("prefill") or graph.get("decode")
        op = _find_op(tree, "triton::_gemma_rmsnorm_kernel")
        self.assertIsNotNone(op)
        self.assertEqual(op["launch"]["func"], "gemma_rmsnorm")
        self.assertEqual(op["input_args"], [])


if __name__ == "__main__":
    unittest.main()
