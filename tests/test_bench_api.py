# SPDX-License-Identifier: Apache-2.0
"""``/api/bench/*`` endpoints - no GPU required.

The web layer is a thin wrapper around :mod:`breakdown.bench`, so these tests
cover the wiring: that planning is refused without a matching profile, that the
plan reports what will *not* be measured, and that ranking/reporting read the
run's own artifacts.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402
from breakdown.bench import store  # noqa: E402


def _graph():
    def op(name, role, shapes, recorded, dtypes, args, backend="torch-xpu-ops"):
        return {"name": name, "role": role, "backend": backend,
                "input_shapes": shapes, "recorded_shapes": recorded,
                "input_dtypes": dtypes, "input_args": args,
                "memory_bytes": 111, "flops": 222, "device_time_us": 9.0}

    tree = {
        "name": "model", "module_type": "Qwen3Model", "path": "model",
        "repeat_count": 1, "ops": [], "children": [{
            "name": "self_attn", "module_type": "Attention",
            "path": "model/attn", "repeat_count": 36, "children": [],
            "ops": [op("aten::linear", "qkv_proj",
                       [["S", "H"], ["QKV/TP", "H"]],
                       [[128, 2560], [3840, 2560]],
                       ["bfloat16", "bfloat16"],
                       [{"kind": "tensor", "dims": [128, 2560],
                         "dtype": "bfloat16"},
                        {"kind": "tensor", "dims": [3840, 2560],
                         "dtype": "bfloat16"},
                        {"kind": "none", "value": ""}]),
                    op("vllm::unified_attention_with_output", "attention",
                       [["S", "n_h/TP", "d"]], [[128, 32, 80]], ["bfloat16"],
                       [{"kind": "tensor", "dims": [128, 32, 80],
                         "dtype": "bfloat16"}],
                       backend="vllm-xpu-kernels"),
                    op("vllm::moe_forward_shared", "moe",
                       [["S", "H"]], [[128, 2560]], ["bfloat16"],
                       [{"kind": "tensor", "dims": [128, 2560],
                         "dtype": "bfloat16"}],
                       backend="vllm-xpu-kernels")],
        }],
    }
    return {
        "prefill": tree, "decode": tree,
        "symbols": {"H": 2560, "n_h": 32, "n_kv": 8, "d": 80, "V": 151936,
                    "QKV": 3840, "TP": 1, "S": 128, "B": 1, "C": 0,
                    "S+C": 128},
        "config": {"tp_size": 1, "quantization": None, "dtype_bytes": 2,
                   "weight_dtype_bytes": 2, "num_layers": 36},
        "source": "profile",
    }


SWEEP = {
    "model_id": "Qwen/Qwen3-4B",
    "prefill_seq_lens": [128], "prefill_ctx_lens": [0],
    "prefill_batch_sizes": [1], "decode_ctx_lens": [0],
    "decode_batch_sizes": [1], "tp_sizes": [1],
}


class BenchApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["BREAKDOWN_BENCH_ROOT"] = self.tmp.name
        import app as app_module
        self._saved = app_module._profile_state
        app_module._profile_state = {
            "status": "done", "result": {"graph": _graph()}, "error": None,
            "model_id": "Qwen/Qwen3-4B",
            "settings": {"query_len": 128, "context_len": 0,
                         "decode_batch_size": 1, "tp_size": 1,
                         "quantization": None, "mode": "eager"},
        }

    def tearDown(self):
        import app as app_module
        app_module._profile_state = self._saved
        os.environ.pop("BREAKDOWN_BENCH_ROOT", None)
        self.tmp.cleanup()

    def _plan(self, **over):
        body = {**SWEEP, **over, "device": "cpu"}
        return self.client.post("/api/bench/plan", data=json.dumps(body),
                                content_type="application/json")


class TestPlan(BenchApiTest):
    def test_plan_builds_cases_and_reports_what_cannot_be_replayed(self):
        resp = self._plan()
        self.assertEqual(resp.status_code, 200, resp.data)
        data = json.loads(resp.data)
        self.assertTrue(data["ok"])
        cov = data["coverage"]
        self.assertGreaterEqual(data["cases"], 1)
        # attention is context-bound at the dispatcher level but has a
        # context-free kernel entry point, so it is planned as replayable
        self.assertEqual(
            cov["op_status"]["vllm::unified_attention_with_output"]["status"],
            "replayable")
        # a wrapper with no such entry point is still reported, never silently
        # dropped - its kernels are benchmarked as their own ops
        self.assertIn("vllm::moe_forward_shared",
                      cov["ops_by_status"].get("not_replayable", []))

    def test_plan_needs_a_model_id(self):
        resp = self.client.post("/api/bench/plan", data=json.dumps({}),
                                content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_plan_refuses_without_a_matching_profile(self):
        import app as app_module
        app_module._profile_state = {"status": "idle", "result": None}
        self.assertEqual(self._plan().status_code, 400)

    def test_plan_writes_the_run_artifacts(self):
        run_id = json.loads(self._plan().data)["run_id"]
        paths = store.run_paths(run_id, self.tmp.name)
        for path in (paths.cases, paths.plan, paths.run_json, paths.rows):
            self.assertTrue(os.path.isfile(path), path)

    def test_plan_persists_the_shape_matrix_rows(self):
        """The sweep's rows are the run's input, so the run keeps them.

        The report workbook ships them as its Shape Matrix sheet; re-deriving
        them at report time would silently follow a profile that has since been
        overwritten.
        """
        run_id = json.loads(self._plan().data)["run_id"]
        with open(store.run_paths(run_id, self.tmp.name).rows) as fh:
            matrix = json.load(fh)
        self.assertTrue(matrix["rows"])
        self.assertTrue(matrix["info"])
        self.assertIn("Op Name", matrix["rows"][0])
        self.assertIn("Phase", matrix["rows"][0])


class TestRunAndTargets(BenchApiTest):
    def test_run_requires_a_planned_run(self):
        resp = self.client.post("/api/bench/run",
                                data=json.dumps({"run_id": "nope"}),
                                content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_targets_without_results_is_an_error_not_an_empty_table(self):
        run_id = json.loads(self._plan().data)["run_id"]
        resp = self.client.get(f"/api/bench/targets?run_id={run_id}")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no benchmark results", json.loads(resp.data)["error"])

    def test_targets_rank_the_run_results(self):
        run_id = json.loads(self._plan().data)["run_id"]
        paths = store.run_paths(run_id, self.tmp.name)
        with open(paths.results, "w") as fh:
            for op, lat in (("aten::linear", 50.0),
                            ("_C::silu_and_mul_with_clamp", 8.0)):
                fh.write(json.dumps({
                    "op": op, "status": "ok", "device": "cpu",
                    "phase": "decode", "seq_len": 1, "ctx_len": 0,
                    "batch_size": 1, "tp": 1, "backend": "torch-xpu-ops",
                    "layers": 36, "latency_us": lat, "flops": 1e6,
                    "bytes": 1e5, "shape": "[1, 2560]",
                    "shape_key": op, "case_id": op,
                    "traced_device_time_us": 0, "traced_comparable": False,
                }) + "\n")
        resp = self.client.get(f"/api/bench/targets?run_id={run_id}&refresh=1")
        self.assertEqual(resp.status_code, 200, resp.data)
        doc = json.loads(resp.data)["targets"]
        self.assertEqual(doc["engine"], "replay")
        self.assertEqual({t["op"] for t in doc["targets"]},
                         {"aten::linear", "_C::silu_and_mul_with_clamp"})
        # the ranking is persisted next to the run
        self.assertTrue(os.path.isfile(paths.targets))

    def test_report_carries_the_runs_shape_matrix(self):
        """One download, not two.

        The measured latencies are only interpretable against the shapes they
        were taken at, and those shapes are what the cases were built from - so
        the sweep ships as a sheet of the same workbook.
        """
        run_id = json.loads(self._plan().data)["run_id"]
        paths = store.run_paths(run_id, self.tmp.name)
        with open(paths.results, "w") as fh:
            fh.write(json.dumps({
                "op": "aten::linear", "status": "ok", "device": "cpu",
                "phase": "decode", "seq_len": 1, "ctx_len": 0,
                "batch_size": 1, "tp": 1, "backend": "torch-xpu-ops",
                "layers": 36, "latency_us": 50.0, "flops": 1e6,
                "bytes": 1e5, "shape": "[1, 2560]",
                "shape_key": "aten::linear", "case_id": "aten::linear",
            }) + "\n")
        resp = self.client.get(f"/api/bench/report?run_id={run_id}")
        if resp.status_code == 500:  # pragma: no cover - pandas missing
            self.skipTest("pandas/openpyxl not available")
        self.assertEqual(resp.status_code, 200, resp.data)
        import io

        import openpyxl
        names = openpyxl.load_workbook(io.BytesIO(resp.data)).sheetnames
        self.assertIn("Shape Matrix", names)
        self.assertIn("Info", names)
        self.assertIn("Summary", names)
        self.assertIn("aten.linear", names)

    def test_results_endpoint_returns_summary_and_coverage(self):
        run_id = json.loads(self._plan().data)["run_id"]
        paths = store.run_paths(run_id, self.tmp.name)
        with open(paths.results, "w") as fh:
            fh.write(json.dumps({
                "op": "vllm::moe_forward_shared", "status": "not_replayable",
                "detail": "fused MoE dispatch wrapper", "backend": "vllm",
                "traced_device_time_us": 12.0, "layers": 1,
            }) + "\n")
        resp = self.client.get(f"/api/bench/results?run_id={run_id}")
        data = json.loads(resp.data)
        self.assertTrue(data["ok"])
        self.assertEqual(data["coverage"][0]["Op"],
                         "vllm::moe_forward_shared")

    def test_results_endpoint_filters_by_op(self):
        """The per-op detail view asks for one op, not the whole run."""
        run_id = json.loads(self._plan().data)["run_id"]
        paths = store.run_paths(run_id, self.tmp.name)
        with open(paths.results, "w") as fh:
            for op in ("_C::rms_norm", "aten::linear"):
                fh.write(json.dumps({
                    "op": op, "status": "ok", "latency_us": 5.0,
                    "layers": 36, "bytes": 1000.0, "flops": 0.0,
                    "phase": "decode", "seq_len": 1, "ctx_len": 2048,
                    "batch_size": 32, "tp": 1, "device": "xpu",
                }) + "\n")
        data = json.loads(self.client.get(
            f"/api/bench/results?run_id={run_id}&op=_C::rms_norm").data)
        self.assertTrue(data["ok"])
        self.assertEqual(data["op"], "_C::rms_norm")
        self.assertEqual({r["op"] for r in data["records"]}, {"_C::rms_norm"})
        self.assertIn("bw_gbs", data["peaks"])

    def test_runs_endpoint_lists_planned_runs(self):
        run_id = json.loads(self._plan().data)["run_id"]
        data = json.loads(self.client.get("/api/bench/runs").data)
        self.assertIn(run_id, [r["run_id"] for r in data["runs"]])


if __name__ == "__main__":
    unittest.main()
