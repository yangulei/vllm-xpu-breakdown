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


def _span_trace():
    """A trace using capture-time ``module::`` spans (no ``nn.Module:`` frames).

    Two Qwen3-style decoder layers, each with an attention block containing two
    same-class ``RMSNorm`` siblings (``q_norm``/``k_norm``) plus the two layer
    norms — the exact case that class-only traces cannot disambiguate.
    """
    events = []
    ext = [0]
    corr = [0]
    tid = 7

    def kern(e, ts, dur):
        corr[0] += 1
        events.append({"ph": "X", "cat": "xpu_runtime", "tid": tid, "pid": tid,
                       "ts": ts, "dur": 0.1, "name": "l",
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

    def span(qname, cls, ts, dur):
        from breakdown.trace_common import module_span_label
        events.append({"ph": "X", "cat": "user_annotation", "tid": tid,
                       "pid": tid, "ts": ts, "dur": dur,
                       "name": module_span_label(qname, cls)})

    span("", "Qwen3ForCausalLM", 0, 400)
    span("model", "Qwen3Model", 1, 398)
    span("model.embed_tokens", "VocabParallelEmbedding", 2, 2)
    op("aten::embedding", 2, 1, [[32000, 16], [8]])
    for li in range(2):
        b = 10 + li * 180
        span(f"model.layers.{li}", "Qwen3DecoderLayer", b, 178)
        span(f"model.layers.{li}.input_layernorm", "RMSNorm", b + 1, 2)
        op("aten::rms_norm", b + 1, 1, [[8, 16]], 1.0)
        span(f"model.layers.{li}.self_attn", "Qwen3Attention", b + 5, 80)
        op("aten::linear", b + 6, 4, [[8, 16], [48, 16]], 5.0)
        span(f"model.layers.{li}.self_attn.q_norm", "RMSNorm", b + 12, 2)
        op("aten::rms_norm", b + 12, 1, [[8, 16]], 1.0)
        span(f"model.layers.{li}.self_attn.k_norm", "RMSNorm", b + 15, 2)
        op("aten::rms_norm", b + 15, 1, [[8, 16]], 1.0)
        span(f"model.layers.{li}.post_attention_layernorm", "RMSNorm", b + 90, 2)
        op("aten::rms_norm", b + 90, 1, [[8, 16]], 1.0)
        span(f"model.layers.{li}.mlp", "Qwen3MLP", b + 95, 20)
        op("aten::linear", b + 96, 4, [[8, 16], [64, 16]], 7.0)
    return {"traceEvents": events}


class TestNamedSpanReconstruction(unittest.TestCase):
    """Reconstruct real module names directly from capture-time spans."""

    SUMMARY = {
        "architecture": "Qwen3ForCausalLM", "hidden_size": 16, "num_heads": 3,
        "num_kv_heads": 3, "head_dim": 16, "intermediate_size": 64,
        "vocab_size": 32000, "num_layers": 2, "dtype": "bfloat16",
    }

    def _build(self, ref_tree=None):
        from breakdown.graph_from_trace import build_graph_from_trace
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(_span_trace(), f)
            path = f.name
        try:
            return build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                          batch_size=1,
                                          ref_module_tree=ref_tree)
        finally:
            os.unlink(path)

    def _layer(self, g):
        model = g["prefill"]["children"][0]
        self.assertEqual(model["name"], "model")
        return next(c for c in model["children"]
                    if c["module_type"] == "Qwen3DecoderLayer")

    def test_real_names_without_ref_tree(self):
        # Names come straight from the trace spans — no reference tree passed.
        g = self._build(ref_tree=None)
        self.assertTrue(g["has_module_names"])
        layer = self._layer(g)
        self.assertEqual(layer["name"], "decoder_layer")
        names = {c["module_type"]: c["name"] for c in layer["children"]}
        self.assertEqual(names["Qwen3Attention"], "self_attn")
        self.assertEqual(names["Qwen3MLP"], "mlp")
        layer_norms = [c["name"] for c in layer["children"]
                       if c["module_type"] == "RMSNorm"]
        self.assertEqual(layer_norms,
                         ["input_layernorm", "post_attention_layernorm"])

    def test_qnorm_knorm_distinguished(self):
        g = self._build(ref_tree=None)
        layer = self._layer(g)
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "Qwen3Attention")
        attn_norms = [c["name"] for c in attn["children"]
                      if c["module_type"] == "RMSNorm"]
        self.assertEqual(attn_norms, ["q_norm", "k_norm"])

    def test_layers_still_collapse(self):
        # Distinct per-layer qnames (layers.0/layers.1) must not defeat the
        # repeat-collapse: both map to the display name "decoder_layer".
        g = self._build(ref_tree=None)
        model = g["prefill"]["children"][0]
        layer_nodes = [c for c in model["children"]
                       if c["module_type"] == "Qwen3DecoderLayer"]
        self.assertEqual(len(layer_nodes), 1)
        self.assertEqual(layer_nodes[0]["repeat_count"], 2)

    def test_ref_tree_ignored_when_spans_present(self):
        # A (deliberately empty) ref tree must not override the exact span names.
        bogus = {"attr": "", "cls": "Qwen3ForCausalLM", "is_group": False,
                 "group_size": 1, "children": []}
        g = self._build(ref_tree=bogus)
        layer = self._layer(g)
        attn = next(c for c in layer["children"]
                    if c["module_type"] == "Qwen3Attention")
        attn_norms = [c["name"] for c in attn["children"]
                      if c["module_type"] == "RMSNorm"]
        self.assertEqual(attn_norms, ["q_norm", "k_norm"])


class TestWorkerThreadSelection(unittest.TestCase):
    """R6: the module-span thread is the worker, even if a decoy thread has
    more cpu_ops (the old 'busiest cpu_op tid' heuristic would mis-select it)."""

    SUMMARY = {
        "architecture": "Qwen3ForCausalLM", "hidden_size": 16, "num_heads": 3,
        "num_kv_heads": 3, "head_dim": 16, "intermediate_size": 64,
        "vocab_size": 32000, "num_layers": 1, "dtype": "bfloat16",
    }

    def _trace_with_decoy(self):
        events = _span_trace()["traceEvents"]
        # Add a decoy thread (tid=1) with MANY cpu_ops but no module spans, so
        # it dominates the raw cpu_op count.
        for i in range(500):
            events.append({"ph": "X", "cat": "cpu_op", "tid": 1, "pid": 1,
                           "ts": 5000 + i, "dur": 1, "name": "aten::decoy_add",
                           "args": {"External id": 10_000 + i,
                                    "Input Dims": [[4, 4]],
                                    "Input type": ["c10::BFloat16"]}})
        return {"traceEvents": events}

    def test_span_thread_selected_over_busier_decoy(self):
        from breakdown.graph_from_trace import build_graph_from_trace
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(self._trace_with_decoy(), f)
            path = f.name
        try:
            g = build_graph_from_trace(path, self.SUMMARY, tp_size=1,
                                       batch_size=1)
        finally:
            os.unlink(path)

        self.assertTrue(g["has_module_names"])

        def _op_names(node, acc):
            for o in node.get("ops", []):
                acc.append(o["name"])
            for c in node.get("children", []):
                _op_names(c, acc)
            return acc

        tree = g["prefill"] or g["decode"]
        self.assertIsNotNone(tree)
        # The decoy thread's ops must not leak into the reconstructed tree.
        self.assertNotIn("aten::decoy_add", _op_names(tree, []))
        # Real span names still recovered from the worker (span) thread.
        model = tree["children"][0]
        self.assertEqual(model["name"], "model")


if __name__ == "__main__":
    unittest.main()
