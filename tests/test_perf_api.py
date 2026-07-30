# SPDX-License-Identifier: Apache-2.0
"""Tests for the /api/perf/* endpoints (no GPU: nothing is benchmarked)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from breakdown.perf import store
from tests.test_perf_rank import _fixture_records, _write_tree
from tests.test_perf_workloads import _mock_m3_graph

MOCK_CONFIG = {
    "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
    "model_type": "minimax_m3_vl", "hidden_size": 6144,
    "num_hidden_layers": 60, "num_attention_heads": 64,
    "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 12288,
    "vocab_size": 200064,
}

SWEEP = {
    "prefill_seq_lens": [128, 2048], "prefill_ctx_lens": [0],
    "prefill_batch_sizes": [1], "decode_ctx_lens": [8192],
    "decode_batch_sizes": [1, 32], "tp_sizes": [4],
}


class PerfApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self._saved = app_module._profile_state
        app_module._profile_state = {
            "status": "done",
            "result": {"graph": _mock_m3_graph()},
            "error": None,
            "model_id": "MiniMaxAI/MiniMax-M3",
            "settings": {"query_len": 128, "context_len": 0, "tp_size": 4,
                         "mode": "eager", "quantization": None},
        }
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["BREAKDOWN_PERF_ROOT"] = self._tmp.name

    def tearDown(self):
        app_module._profile_state = self._saved
        os.environ.pop("BREAKDOWN_PERF_ROOT", None)
        self._tmp.cleanup()

    def _emit(self, **extra):
        with patch("app.fetch_model_config", return_value=MOCK_CONFIG):
            return self.client.post("/api/perf/workloads", json={
                "model_id": "MiniMaxAI/MiniMax-M3", **SWEEP, **extra})


class TestWorkloadsEndpoint(PerfApiTest):
    def test_emits_workloads_into_an_owned_run_dir(self):
        r = self._emit()
        self.assertEqual(r.status_code, 200, r.json)
        run_id = r.json["run_id"]
        paths = store.run_paths(run_id)
        # the matrix must be kept: it can't be regenerated without a re-profile
        self.assertTrue(os.path.isfile(paths.matrix))
        self.assertTrue(os.path.isfile(paths.coverage))
        self.assertTrue(os.path.isfile(
            os.path.join(paths.workloads, "compute", "compute_ops.json")))
        self.assertEqual(r.json["coverage"]["unmapped_breakdown_ops"], {})

    def test_run_records_provenance(self):
        run_id = self._emit().json["run_id"]
        meta = store.read_meta(store.run_paths(run_id))
        self.assertEqual(meta["model_id"], "MiniMaxAI/MiniMax-M3")
        self.assertEqual(meta["tp"], 4)
        self.assertIn("commits", meta)

    def test_smoke_tier_emits_fewer_cases(self):
        full = self._emit(prefill_seq_lens=[128, 512, 2048],
                          decode_batch_sizes=[1, 8, 32])
        smoke = self._emit(prefill_seq_lens=[128, 512, 2048],
                           decode_batch_sizes=[1, 8, 32], smoke=True)

        def n(resp):
            cases = resp.json["coverage"]["mapped_micro_perf_cases"]
            return sum(sum(v.values()) for v in cases.values())

        self.assertLess(n(smoke), n(full))

    def test_requires_a_matching_profile(self):
        app_module._profile_state = {"status": "idle", "result": None,
                                     "error": None, "model_id": None,
                                     "settings": None}
        r = self._emit()
        self.assertEqual(r.status_code, 400)
        self.assertIn("profiling run", r.json["error"])

    def test_model_id_required(self):
        r = self.client.post("/api/perf/workloads", json={})
        self.assertEqual(r.status_code, 400)


class TestTargetsEndpoint(PerfApiTest):
    def _run_with_reports(self):
        run_id = self._emit().json["run_id"]
        paths = store.run_paths(run_id)
        with open(os.path.join(paths.workloads, "compute",
                               "compute_ops.json")) as fh:
            compute = json.load(fh)
        with open(os.path.join(paths.workloads, "msa", "msa_ops.json")) as fh:
            msa = json.load(fh)
        with open(os.path.join(paths.workloads, "collective",
                               "collective_ops.json")) as fh:
            coll = json.load(fh)
        _write_tree(paths.reports,
                    _fixture_records({"compute": compute, "msa": msa,
                                      "collective": coll}))
        return run_id

    def test_ranks_and_persists_targets(self):
        run_id = self._run_with_reports()
        with patch("app.fetch_model_config", return_value=MOCK_CONFIG):
            r = self.client.get(f"/api/perf/targets?run_id={run_id}&refresh=1")
        self.assertEqual(r.status_code, 200, r.json)
        doc = r.json["targets"]
        self.assertEqual(doc["targets"][0]["op"], "msa_sparse_attn")
        self.assertEqual(doc["targets"][0]["action"], "optimize_kernel")
        self.assertTrue(os.path.isfile(store.run_paths(run_id).targets))
        # the ranking is attributable to the run that produced it
        self.assertEqual(doc["provenance"]["run_id"], run_id)

    def test_cached_targets_are_served_without_recompute(self):
        run_id = self._run_with_reports()
        with patch("app.fetch_model_config", return_value=MOCK_CONFIG):
            self.client.get(f"/api/perf/targets?run_id={run_id}&refresh=1")
        r = self.client.get(f"/api/perf/targets?run_id={run_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["targets"]["targets"][0]["op"],
                         "msa_sparse_attn")

    def test_run_without_reports_is_rejected(self):
        run_id = self._emit().json["run_id"]
        with patch("app.fetch_model_config", return_value=MOCK_CONFIG):
            r = self.client.get(f"/api/perf/targets?run_id={run_id}&refresh=1")
        self.assertEqual(r.status_code, 400)
        self.assertIn("no benchmark reports", r.json["error"])

    def test_history_records_the_ranked_run(self):
        run_id = self._run_with_reports()
        with patch("app.fetch_model_config", return_value=MOCK_CONFIG):
            self.client.get(f"/api/perf/targets?run_id={run_id}&refresh=1")
        r = self.client.get("/api/perf/history")
        self.assertEqual(r.status_code, 200)
        self.assertIn(run_id, [x["run_id"] for x in r.json["runs"]])


class TestRunEndpoint(PerfApiTest):
    def test_unknown_run_is_rejected(self):
        r = self.client.post("/api/perf/run", json={"run_id": "nope"})
        self.assertEqual(r.status_code, 400)

    def test_listing_runs(self):
        run_id = self._emit().json["run_id"]
        r = self.client.get("/api/perf/runs")
        self.assertEqual(r.status_code, 200)
        self.assertIn(run_id, [x["run_id"] for x in r.json["runs"]])


if __name__ == "__main__":
    unittest.main()
