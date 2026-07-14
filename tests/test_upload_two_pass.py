#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the trace-upload endpoint's two-pass round-trip support.

The download endpoint writes a separate ``…_prefill_…`` and ``…_decode_…``
trace for a two-pass run. Uploading that pair on a GPU-less machine must
reconstruct **both** phases (not decode only) by mirroring the live two-pass
merge, recovering every profiled knob from the descriptive filenames.

``_build_result_from_traces`` is monkeypatched so no real trace/torch/XPU is
needed — these tests exercise the endpoint's filename parsing, pass grouping
and merge wiring, not trace parsing itself.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod
from app import _parse_trace_filename


PRE_NAME = "vllm_trace_MiniMax-M3_XPU_eager_prefill_ctx2048_in1536_out1_bs1_tp4_6layers.json.gz"
DEC_NAME = "vllm_trace_MiniMax-M3_XPU_eager_decode_ctx2048_in1536_out8_bs32_tp4_6layers.json.gz"


def _fake_result(*, prefill, decode, symbols, batch_size, trace_file):
    return {
        "model_id": "",
        "mode": "eager",
        "tp_size": 4,
        "quantization": None,
        "summary": {},
        "batch_size": batch_size,
        "ops": [{"name": f"op@{batch_size}"}],
        "backends": {"triton": {"pct": 50.0}},
        "trace_file": trace_file,
        "graph": {
            "architecture": "MiniMaxM3",
            "prefill": prefill,
            "decode": decode,
            "symbols": symbols,
            "config": {"tp_size": 4},
        },
    }


def _fake_build(rank_files, *, batch_size, query_len=None, **kw):
    """Stand in for _build_result_from_traces: return a per-pass fake result."""
    name = os.path.basename(rank_files[0])
    if "prefill" in name:
        return _fake_result(
            prefill={"name": "prefill_root", "children": []},
            decode=None,
            symbols={"S": 1536, "S+C": 3584, "C": 2048, "B": 1},
            batch_size=batch_size,
            trace_file=rank_files[0],
        )
    return _fake_result(
        prefill=None,
        decode={"name": "decode_root", "children": []},
        symbols={"S": 1, "C": 2048, "B": 32},
        batch_size=batch_size,
        trace_file=rank_files[0],
    )


class TestParseTraceFilename(unittest.TestCase):
    def test_decode_name(self):
        m = _parse_trace_filename(DEC_NAME)
        self.assertEqual(m["pass"], "decode")
        self.assertEqual(m["mode"], "eager")
        self.assertEqual(m["context_len"], 2048)
        self.assertEqual(m["query_len"], 1536)
        self.assertEqual(m["batch_size"], 32)
        self.assertEqual(m["tp"], 4)
        self.assertEqual(m["profiled_layers"], 6)
        self.assertIsNone(m["quantization"])

    def test_prefill_name(self):
        m = _parse_trace_filename(PRE_NAME)
        self.assertEqual(m["pass"], "prefill")
        self.assertEqual(m["batch_size"], 1)

    def test_all_layers_and_quant(self):
        m = _parse_trace_filename(
            "vllm_trace_Foo_CUDA_compile_ctx0_in128_out8_bs2_tp2_fp8_alllayers.json")
        self.assertIsNone(m["pass"])
        self.assertEqual(m["quantization"], "fp8")
        self.assertIsNone(m["profiled_layers"])  # "all" -> None

    def test_unrecognized_name(self):
        self.assertEqual(_parse_trace_filename("random.json"), {})


class _UploadTestBase(unittest.TestCase):
    def setUp(self):
        appmod.app.config["TESTING"] = True
        self.client = appmod.app.test_client()
        with appmod._profile_lock:
            appmod._profile_state = {"status": "idle", "result": None,
                                     "error": None, "model_id": None}

    def _post(self, names):
        data = {}
        data["trace"] = [
            (io.BytesIO(b"dummy-trace"), n) for n in names
        ]
        # Empty model_id skips the HF config fetch + meta reference tree.
        data["model_id"] = ""
        return self.client.post("/api/profile/upload", data=data,
                                content_type="multipart/form-data")


class TestUploadTwoPass(_UploadTestBase):
    def test_pair_reconstructs_both_phases(self):
        with mock.patch.object(appmod, "_build_result_from_traces",
                               side_effect=_fake_build):
            resp = self._post([DEC_NAME, PRE_NAME])  # order shouldn't matter
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["ok"])

        res = appmod._profile_state["result"]
        self.assertTrue(res["two_pass"])
        # Both phases present — the bug was "decode only".
        self.assertEqual(res["graph"]["prefill"]["name"], "prefill_root")
        self.assertEqual(res["graph"]["decode"]["name"], "decode_root")

    def test_batches_recovered_from_filenames(self):
        with mock.patch.object(appmod, "_build_result_from_traces",
                               side_effect=_fake_build):
            self._post([PRE_NAME, DEC_NAME])
        res = appmod._profile_state["result"]
        self.assertEqual(res["prefill_batch_size"], 1)
        self.assertEqual(res["decode_batch_size"], 32)
        self.assertEqual(res["batch_size"], 32)

    def test_symbols_combined(self):
        with mock.patch.object(appmod, "_build_result_from_traces",
                               side_effect=_fake_build):
            self._post([PRE_NAME, DEC_NAME])
        sym = appmod._profile_state["result"]["graph"]["symbols"]
        self.assertEqual(sym["S"], 1536)   # from prefill pass
        self.assertEqual(sym["S+C"], 3584)
        self.assertEqual(sym["B"], 32)     # from decode pass

    def test_both_trace_files_retained_for_download(self):
        with mock.patch.object(appmod, "_build_result_from_traces",
                               side_effect=_fake_build):
            self._post([PRE_NAME, DEC_NAME])
        res = appmod._profile_state["result"]
        self.assertTrue(res["prefill_trace_file"].endswith(".json.gz"))
        self.assertIn("prefill", os.path.basename(res["prefill_trace_file"]))
        self.assertIn("decode", os.path.basename(res["decode_trace_file"]))

    def test_result_endpoint_reports_both_traces(self):
        with mock.patch.object(appmod, "_build_result_from_traces",
                               side_effect=_fake_build):
            self._post([PRE_NAME, DEC_NAME])
        data = self.client.get("/api/profile/result").get_json()["data"]
        self.assertTrue(data["has_prefill_trace"])
        self.assertTrue(data["has_decode_trace"])
        # Internal absolute paths are not leaked to the client.
        self.assertNotIn("prefill_trace_file", data)

    def test_context_and_query_recovered(self):
        with mock.patch.object(appmod, "_build_result_from_traces",
                               side_effect=_fake_build):
            self._post([PRE_NAME, DEC_NAME])
        res = appmod._profile_state["result"]
        self.assertEqual(res["query_len"], 1536)
        self.assertEqual(res["context_len"], 2048)
        self.assertEqual(res["context_len_aligned"], 2048)


class TestUploadSinglePass(_UploadTestBase):
    def test_lone_decode_file_is_decode_only(self):
        with mock.patch.object(appmod, "_build_result_from_traces",
                               side_effect=_fake_build):
            resp = self._post([DEC_NAME])
        self.assertEqual(resp.status_code, 200)
        res = appmod._profile_state["result"]
        self.assertFalse(res.get("two_pass", False))
        self.assertEqual(res["graph"]["decode"]["name"], "decode_root")
        self.assertIsNone(res["graph"]["prefill"])


if __name__ == "__main__":
    unittest.main()
