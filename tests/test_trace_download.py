#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the /api/profile/trace download endpoint.

Two-pass runs (separate prefill/decode batch sizes) write a separate trace per
pass. The endpoint must be able to serve either one via ``?pass=prefill`` or
``?pass=decode`` while staying backward-compatible for single-pass runs.
"""

from __future__ import annotations

import gzip
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from app import app, _merge_two_pass_result


def _write_trace(tmpdir: str, name: str, marker: str) -> str:
    path = os.path.join(tmpdir, name)
    with gzip.open(path, "wt") as fh:
        fh.write('{"marker": "%s"}' % marker)
    return path


class TestTraceDownload(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self._saved_state = app_module._profile_state
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        app_module._profile_state = self._saved_state
        self.tmp.cleanup()

    def _set_state(self, result: dict):
        app_module._profile_state = {
            "status": "done",
            "result": result,
            "error": None,
        }

    # --- two-pass ---------------------------------------------------------
    def _two_pass_result(self) -> dict:
        pre_file = _write_trace(self.tmp.name, "pre-rank0.json.gz", "prefill")
        dec_file = _write_trace(self.tmp.name, "dec-rank0.json.gz", "decode")
        pre = {"model_id": "org/M", "mode": "eager", "tp_size": 4,
               "trace_file": pre_file, "batch_size": 1, "max_tokens": 1,
               "query_len": 128}
        dec = {"model_id": "org/M", "mode": "eager", "tp_size": 4,
               "trace_file": dec_file, "batch_size": 8, "max_tokens": 16,
               "query_len": 128}
        return _merge_two_pass_result(pre, dec, prefill_bs=1, decode_bs=8)

    def test_merge_retains_both_trace_files(self):
        result = self._two_pass_result()
        self.assertTrue(result["two_pass"])
        self.assertTrue(result["prefill_trace_file"].endswith("pre-rank0.json.gz"))
        self.assertTrue(result["decode_trace_file"].endswith("dec-rank0.json.gz"))
        # Default trace_file stays the decode pass (backward compat).
        self.assertEqual(result["trace_file"], result["decode_trace_file"])

    def test_download_prefill_pass(self):
        self._set_state(self._two_pass_result())
        resp = self.client.get("/api/profile/trace?pass=prefill")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(gzip.decompress(resp.data), b'{"marker": "prefill"}')
        cd = resp.headers["Content-Disposition"]
        self.assertIn("_prefill_", cd)
        self.assertIn("_bs1_", cd)  # prefill batch
        self.assertIn("_out1_", cd)  # prefill generates 1 token

    def test_download_decode_pass(self):
        self._set_state(self._two_pass_result())
        resp = self.client.get("/api/profile/trace?pass=decode")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(gzip.decompress(resp.data), b'{"marker": "decode"}')
        cd = resp.headers["Content-Disposition"]
        self.assertIn("_decode_", cd)
        self.assertIn("_bs8_", cd)  # decode batch
        self.assertIn("_out16_", cd)

    def test_download_default_is_decode(self):
        self._set_state(self._two_pass_result())
        resp = self.client.get("/api/profile/trace")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(gzip.decompress(resp.data), b'{"marker": "decode"}')
        self.assertIn("_decode_", resp.headers["Content-Disposition"])

    def test_result_reports_both_traces_and_strips_paths(self):
        self._set_state(self._two_pass_result())
        data = self.client.get("/api/profile/result").get_json()["data"]
        self.assertTrue(data["has_prefill_trace"])
        self.assertTrue(data["has_decode_trace"])
        self.assertNotIn("trace_file", data)
        self.assertNotIn("prefill_trace_file", data)
        self.assertNotIn("decode_trace_file", data)

    # --- single-pass ------------------------------------------------------
    def test_single_pass_backward_compatible(self):
        one = _write_trace(self.tmp.name, "one-rank0.json.gz", "single")
        self._set_state({"model_id": "org/M", "mode": "eager", "tp_size": 1,
                         "trace_file": one, "batch_size": 1, "max_tokens": 16})
        data = self.client.get("/api/profile/result").get_json()["data"]
        self.assertTrue(data["has_trace"])
        self.assertFalse(data["has_prefill_trace"])
        self.assertFalse(data["has_decode_trace"])
        resp = self.client.get("/api/profile/trace")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(gzip.decompress(resp.data), b'{"marker": "single"}')
        # No pass tag for a single-pass run.
        self.assertNotIn("_prefill_", resp.headers["Content-Disposition"])
        self.assertNotIn("_decode_", resp.headers["Content-Disposition"])

    def test_missing_pass_file_returns_404(self):
        result = self._two_pass_result()
        result["prefill_trace_file"] = None
        self._set_state(result)
        resp = self.client.get("/api/profile/trace?pass=prefill")
        self.assertEqual(resp.status_code, 404)


class TestRank0Selection(unittest.TestCase):
    """Rank 0 is always the representative worker for TP>1 traces.

    Ranks 1..N idle longer on collectives (their allreduce device time is
    inflated by the wait to synchronize with rank 0), so the OP breakdown,
    reconstructed graph and downloadable trace must all be built from rank 0
    regardless of the (mtime) order the per-rank files arrive in.
    """

    def test_trace_rank_from_rank_marker(self):
        self.assertEqual(app_module._trace_rank(
            "dp0_pp0_tp2_dcp0_ep2_rank2.178.pt.trace.json.gz"), 2)
        self.assertEqual(app_module._trace_rank(
            "dp0_pp0_tp0_dcp0_ep0_rank0.178.pt.trace.json.gz"), 0)

    def test_trace_rank_none_for_merged_name(self):
        self.assertIsNone(app_module._trace_rank("merged.json.gz"))

    def test_rank0_first_reorders(self):
        files = [
            "dp0_pp0_tp2_dcp0_ep2_rank2.a.pt.trace.json.gz",
            "dp0_pp0_tp0_dcp0_ep0_rank0.b.pt.trace.json.gz",
            "dp0_pp0_tp3_dcp0_ep3_rank3.c.pt.trace.json.gz",
            "dp0_pp0_tp1_dcp0_ep1_rank1.d.pt.trace.json.gz",
        ]
        out = app_module._rank0_first(files)
        self.assertEqual(app_module._trace_rank(out[0]), 0)
        self.assertEqual([app_module._trace_rank(f) for f in out], [0, 1, 2, 3])

    def test_rank0_first_no_markers_unchanged(self):
        files = ["a.json.gz", "b.json.gz"]
        self.assertEqual(app_module._rank0_first(files), files)

    def test_rank0_first_single_file(self):
        self.assertEqual(app_module._rank0_first(["only.json.gz"]),
                         ["only.json.gz"])


if __name__ == "__main__":
    unittest.main()
