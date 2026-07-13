#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for capture-time module spans (research R1).

Covers the torch-free label helpers (:mod:`breakdown.trace_common`), the forward
hook emitter (:mod:`breakdown.module_hooks`), and end-to-end reconstruction of a
module tree with real attribute names directly from ``user_annotation`` spans
(no ``named_modules()`` overlay needed).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.trace_common import (
    module_span_display_name,
    module_span_label,
    parse_module_span,
)


class TestModuleSpanHelpers(unittest.TestCase):
    def test_roundtrip(self):
        lbl = module_span_label("model.layers.0.self_attn.q_norm", "RMSNorm")
        self.assertEqual(lbl, "module::model.layers.0.self_attn.q_norm::RMSNorm")
        self.assertEqual(parse_module_span(lbl),
                         ("model.layers.0.self_attn.q_norm", "RMSNorm"))

    def test_root_span(self):
        self.assertEqual(parse_module_span(module_span_label("", "Qwen3ForCausalLM")),
                         ("", "Qwen3ForCausalLM"))

    def test_non_span_returns_none(self):
        self.assertIsNone(parse_module_span("aten::mm"))
        self.assertIsNone(parse_module_span("nn.Module: RMSNorm_2"))

    def test_display_name_numeric_group_becomes_decoder_layer(self):
        self.assertEqual(
            module_span_display_name("model.layers.0", "Qwen3DecoderLayer"),
            "decoder_layer")

    def test_display_name_numeric_group_uses_list_attr(self):
        self.assertEqual(
            module_span_display_name("model.layers.3.mlp.experts.0", "Qwen3MoeMLP"),
            "experts")

    def test_display_name_leaf(self):
        self.assertEqual(
            module_span_display_name("model.layers.0.self_attn.q_norm", "RMSNorm"),
            "q_norm")

    def test_display_name_root_empty(self):
        self.assertEqual(module_span_display_name("", "Qwen3ForCausalLM"), "")


class TestModuleSpanHooks(unittest.TestCase):
    """The forward hooks emit (and cleanly remove) real-name spans."""

    def _profile_spans(self, install: bool):
        import torch
        import torch.nn as nn
        from torch.profiler import ProfilerActivity, profile

        from breakdown.module_hooks import module_span_hooks

        class Sub(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 8)

            def forward(self, x):
                return self.lin(x)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Sub()
                self.b = Sub()

            def forward(self, x):
                return self.b(self.a(x))

        net = Net()
        x = torch.randn(4, 8)

        def _run():
            with profile(activities=[ProfilerActivity.CPU]) as p:
                net(x)
            path = tempfile.mktemp(suffix=".json")
            p.export_chrome_trace(path)
            events = json.load(open(path))["traceEvents"]
            os.unlink(path)
            return [e["name"] for e in events
                    if e.get("cat") == "user_annotation"
                    and e["name"].startswith("module::")]

        if install:
            with module_span_hooks(net):
                return _run()
        return _run()

    def test_hooks_emit_real_name_spans(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        spans = self._profile_spans(install=True)
        self.assertIn("module::a::Sub", spans)
        self.assertIn("module::a.lin::Linear", spans)
        self.assertIn("module::b.lin::Linear", spans)

    def test_hooks_removed_after_context(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        self._profile_spans(install=True)  # installs + removes
        self.assertEqual(self._profile_spans(install=False), [])


if __name__ == "__main__":
    unittest.main()
