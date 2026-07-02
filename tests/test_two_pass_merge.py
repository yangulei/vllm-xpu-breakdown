#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the two-pass (separate prefill/decode batch) merge helper.

``_merge_two_pass_result`` splices a prefill-batch pass and a decode-batch pass
into one profile result: prefill graph tree from the prefill pass, decode graph
tree from the decode pass, symbols combined (S/S+C/C from prefill, B from
decode). No GPU/torch required — the helper is a pure dict transform.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import _merge_two_pass_result


def _fake_result(prefill_tree, decode_tree, symbols, *, batch_size):
    return {
        "batch_size": batch_size,
        "ops": [{"name": f"op@{batch_size}"}],
        "backends": {"triton": {"pct": 50.0}},
        "graph": {
            "architecture": "LlamaForCausalLM",
            "prefill": prefill_tree,
            "decode": decode_tree,
            "symbols": symbols,
            "config": {"tp_size": 1},
        },
    }


class TestTwoPassMerge(unittest.TestCase):
    def setUp(self):
        # Prefill pass: batch=1 -> faithful prefill tree, S=2048, B (its decode) = 1.
        self.pre = _fake_result(
            prefill_tree={"name": "prefill_root", "children": []},
            decode_tree={"name": "pre_decode_discard", "children": []},
            symbols={"S": 2048, "S+C": 4096, "C": 2048, "B": 1, "H": 4096},
            batch_size=1,
        )
        # Decode pass: batch=32 -> faithful decode tree, B=32, S (its prefill) = 1.
        self.dec = _fake_result(
            prefill_tree={"name": "dec_prefill_discard", "children": []},
            decode_tree={"name": "decode_root", "children": []},
            symbols={"S": 1, "C": 2048, "B": 32, "H": 4096},
            batch_size=32,
        )

    def test_prefill_tree_from_prefill_pass(self):
        merged = _merge_two_pass_result(self.pre, self.dec, 1, 32)
        self.assertEqual(merged["graph"]["prefill"]["name"], "prefill_root")

    def test_decode_tree_from_decode_pass(self):
        merged = _merge_two_pass_result(self.pre, self.dec, 1, 32)
        self.assertEqual(merged["graph"]["decode"]["name"], "decode_root")

    def test_symbols_combined(self):
        merged = _merge_two_pass_result(self.pre, self.dec, 1, 32)
        sym = merged["graph"]["symbols"]
        self.assertEqual(sym["S"], 2048)      # prefill token dim from prefill pass
        self.assertEqual(sym["S+C"], 4096)    # from prefill pass
        self.assertEqual(sym["B"], 32)        # decode batch from decode pass
        self.assertEqual(sym["H"], 4096)      # shared config constant

    def test_batch_size_metadata(self):
        merged = _merge_two_pass_result(self.pre, self.dec, 1, 32)
        self.assertEqual(merged["prefill_batch_size"], 1)
        self.assertEqual(merged["decode_batch_size"], 32)
        self.assertEqual(merged["batch_size"], 32)
        self.assertTrue(merged["two_pass"])

    def test_base_is_decode_pass(self):
        # Op breakdown / backends come from the (steady-state) decode pass.
        merged = _merge_two_pass_result(self.pre, self.dec, 1, 32)
        self.assertEqual(merged["ops"], self.dec["ops"])
        self.assertEqual(merged["backends"], self.dec["backends"])

    def test_missing_prefill_graph_is_safe(self):
        pre = dict(self.pre)
        pre["graph"] = None  # graph reconstruction failed for prefill pass
        merged = _merge_two_pass_result(pre, self.dec, 1, 32)
        self.assertIsNone(merged["graph"]["prefill"])
        self.assertEqual(merged["graph"]["decode"]["name"], "decode_root")

    def test_does_not_mutate_inputs(self):
        pre_before = self.pre["graph"]["prefill"]["name"]
        dec_before = self.dec["graph"]["decode"]["name"]
        _merge_two_pass_result(self.pre, self.dec, 1, 32)
        self.assertEqual(self.pre["graph"]["prefill"]["name"], pre_before)
        self.assertEqual(self.dec["graph"]["decode"]["name"], dec_before)
        self.assertNotIn("two_pass", self.dec)


if __name__ == "__main__":
    unittest.main()
