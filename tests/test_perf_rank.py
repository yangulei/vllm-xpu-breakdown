# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ranking, the runner and the perf history (no GPU needed)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.perf import history, rank as rank_mod, reports, runner, store
from breakdown.perf import workloads as wl
from breakdown.perf.matrix_reader import rows_to_oprows
from breakdown.perf.op_map import ModelConfig
from tests.test_perf_workloads import _CFG, _rows

PEAK_BW = 456e9  # BMG GDDR6, bytes/s


def _rec(op, provider, args, latency_us, util=0.3, flops=0.0,
         device="Intel(R) Arc(TM) Pro B60 Graphics"):
    """A micro_perf report line whose io_bytes realizes a chosen utilization."""
    io = util * PEAK_BW * (latency_us / 1e6)
    return {
        "sku_name": device, "op_name": op, "provider": provider,
        "arguments": args,
        "targets": {
            "latency(us)": latency_us,
            "io_bytes(B)": io,
            "mem_bw(GB/s)": io / (latency_us / 1e6) / 1e9,
            "calc_flops": flops,
            "calc_flops_power(tflops)": flops / (latency_us / 1e6) / 1e12
            if flops else 0.0,
        },
    }


def _write_tree(root, records, backend="INTEL",
                device="Intel(R) Arc(TM) Pro B60 Graphics"):
    for r in records:
        d = os.path.join(root, backend, device, r["op_name"], r["provider"])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{r['op_name']}-{r['provider']}.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def _fixture_records(buckets):
    """Report records for every emitted case, with deliberate characteristics.

    * ``msa_sparse_attn`` — slow, far from the roofline → the top target
    * ``rms_norm`` — dispatched provider (triton) slower than an available one
      → a free ``switch_provider`` win
    * ``silu_and_mul_with_clamp`` — 90 % of peak bandwidth → ``at_roofline``
    * ``all_reduce`` — a collective, no kernel source → ``tune_config``
    """
    recs = []
    for group, ops in buckets.items():
        for op, cases in ops.items():
            for args in cases:
                if op == "msa_sparse_attn":
                    recs.append(_rec(op, "flash_xpu", args, 3000.0, util=0.35))
                    recs.append(_rec(op, "triton", args, 6000.0, util=0.18))
                elif op == "rms_norm":
                    recs.append(_rec(op, "triton", args, 100.0, util=0.2))
                    recs.append(_rec(op, "vllm_xpu_kernels", args, 20.0,
                                     util=0.5))
                elif op == "silu_and_mul_with_clamp":
                    recs.append(_rec(op, "vllm_xpu_kernels", args, 50.0,
                                     util=0.9))
                elif op == "all_reduce":
                    recs.append(_rec(op, "xccl", args, 80.0, util=0.1))
                else:
                    recs.append(_rec(op, "vllm_xpu_kernels", args, 200.0,
                                     util=0.4))
    return recs


def _ranked(**rc_kwargs):
    _, rows = _rows()
    oprows = rows_to_oprows(rows)
    buckets, _ = wl.emit(oprows, _CFG, "xpu")
    with tempfile.TemporaryDirectory() as d:
        _write_tree(d, _fixture_records(buckets))
        recs = reports.records("INTEL", d)
        rc = rank_mod.RankConfig(tp=4, **rc_kwargs)
        return rank_mod.rank(oprows, recs, _CFG, rc), buckets


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.doc, self.buckets = _ranked()
        self.by_op = {t["op"]: t for t in self.doc["targets"]}

    def test_schema_is_versioned(self):
        """The optimizer skill reads this before trusting the file."""
        self.assertEqual(self.doc["schema_version"], rank_mod.SCHEMA_VERSION)

    def test_slow_far_from_roofline_op_ranks_first(self):
        top = self.doc["targets"][0]
        self.assertEqual(top["op"], "msa_sparse_attn")
        self.assertEqual(top["action"], "optimize_kernel")
        self.assertEqual(top["best_provider"], "flash_xpu")

    def test_dispatched_provider_comes_from_the_trace_backend(self):
        t = self.by_op["msa_sparse_attn"]
        self.assertEqual(t["dispatched_provider"], "flash_xpu")
        self.assertEqual(t["dispatched_provider_source"], "trace")

    def test_faster_provider_is_a_free_win(self):
        t = self.by_op["rms_norm"]
        self.assertEqual(t["action"], "switch_provider")
        self.assertEqual(t["dispatched_provider"], "triton")
        self.assertEqual(t["best_provider"], "vllm_xpu_kernels")
        self.assertGreater(t["savings_us"]["switch_provider"], 0)

    def test_op_at_the_roofline_is_not_a_target(self):
        t = self.by_op["silu_and_mul_with_clamp"]
        self.assertEqual(t["action"], "at_roofline")
        self.assertEqual(t["savings_us"]["total"], 0.0)
        self.assertGreaterEqual(t["roofline"]["util"], 0.8)

    def test_library_op_without_kernel_source_is_config_tuning(self):
        self.assertEqual(self.by_op["all_reduce"]["action"], "tune_config")

    def test_call_weighting_uses_the_layers_column(self):
        """e2e time = per-call latency x how many layers dispatch the op."""
        t = self.by_op["msa_sparse_attn"]
        self.assertEqual(t["calls"] % 57, 0)
        self.assertAlmostEqual(t["e2e_us"], 3000.0 * t["calls"], places=1)

    def test_shares_sum_to_one(self):
        total = sum(t["share_of_e2e"] for t in self.doc["targets"])
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_targets_carry_an_executable_handoff(self):
        t = self.by_op["msa_sparse_attn"]
        self.assertIn("msa_sparse_attn.cpp", " ".join(t["kernel"]["files"]))
        self.assertTrue(t["kernel"]["build_cmd"])
        self.assertIn("pytest", t["kernel"]["test_cmd"])
        shape = t["top_shapes"][0]
        self.assertIn("breakdown.perf bench", shape["bench_cmd"])
        self.assertIn("unitrace", shape["profile_cmd"])
        # the bench command must carry a runnable case
        args = shape["bench_cmd"].split("--case '")[1].rstrip("'")
        self.assertEqual(json.loads(args), shape["args"])

    def test_sku_is_detected_from_the_device(self):
        self.assertEqual(self.doc["sku"], "BMG")
        self.assertAlmostEqual(self.doc["peaks"]["bw_gbs"], 456.0)

    def test_phase_weight_shifts_the_ranking(self):
        doc, _ = _ranked(phase_weight={"prefill": 0.0, "decode": 1.0})
        for t in doc["targets"]:
            self.assertAlmostEqual(t["e2e_us"], t["phase_us"]["decode"],
                                   places=1)

    def test_top_limits_the_output(self):
        doc, _ = _ranked(top=2)
        self.assertEqual(len(doc["targets"]), 2)

    def test_rank_without_records_is_rejected(self):
        _, rows = _rows()
        with self.assertRaises(ValueError):
            rank_mod.rank(rows_to_oprows(rows), [], _CFG)


class TestRunner(unittest.TestCase):
    """The runner's contract, exercised against a stub launcher (no GPU)."""

    def _stub_micro_perf(self, d, report_line=True, crash=False):
        mp = os.path.join(d, "micro_perf")
        os.makedirs(mp, exist_ok=True)
        body = "import sys, os, json\n"
        if crash:
            body += "print('RuntimeError: kernel exploded'); sys.exit(1)\n"
        else:
            body += (
                "args = sys.argv\n"
                "op = args[args.index('--task') + 1]\n"
                "rd = args[args.index('--report_dir') + 1]\n"
                "d = os.path.join(rd, 'INTEL', 'dev', op, 'prov')\n"
                "os.makedirs(d, exist_ok=True)\n")
            if report_line:
                body += (
                    "open(os.path.join(d, f'{op}-prov.jsonl'), 'w').write("
                    "json.dumps({'op_name': op, 'provider': 'prov',"
                    " 'arguments': {}, 'targets': {'latency(us)': 1.0}}) "
                    "+ chr(10))\n")
        with open(os.path.join(mp, "launch.py"), "w") as fh:
            fh.write(body)
        return mp

    def _workloads(self, d):
        wdir = os.path.join(d, "workloads")
        os.makedirs(os.path.join(wdir, "compute"), exist_ok=True)
        with open(os.path.join(wdir, "compute", "compute_ops.json"), "w") as fh:
            json.dump({"gemm": [{"M": 1}], "rms_norm": [{"n": 1}]}, fh)
        return wdir

    def test_runs_each_op_in_its_own_process(self):
        with tempfile.TemporaryDirectory() as d:
            mp = self._stub_micro_perf(d)
            res = runner.run(self._workloads(d), os.path.join(d, "reports"),
                             groups=["compute"], micro_perf_dir=mp)
            self.assertTrue(res.ok)
            self.assertEqual(sorted(o.op for o in res.ops),
                             ["gemm", "rms_norm"])
            self.assertTrue(all(o.cases == 1 for o in res.ops))

    def test_a_failing_op_does_not_abort_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            mp = self._stub_micro_perf(d, crash=True)
            res = runner.run(self._workloads(d), os.path.join(d, "reports"),
                             groups=["compute"], micro_perf_dir=mp)
            self.assertFalse(res.ok)
            self.assertEqual(len(res.ops), 2)          # both were attempted
            self.assertIn("RuntimeError", res.ops[0].error)

    def test_reports_inside_workloads_is_rejected(self):
        """The launcher parses every json under --task_dir."""
        with tempfile.TemporaryDirectory() as d:
            mp = self._stub_micro_perf(d)
            wdir = self._workloads(d)
            with self.assertRaises(ValueError):
                runner.run(wdir, os.path.join(wdir, "reports"),
                           groups=["compute"], micro_perf_dir=mp)

    def test_persistent_kernel_caches_are_set(self):
        with tempfile.TemporaryDirectory() as d:
            env = runner.bench_env(os.path.join(d, "cache"), base={})
            self.assertEqual(env["SYCL_CACHE_PERSISTENT"], "1")
            self.assertTrue(os.path.isdir(env["SYCL_CACHE_DIR"]))
            self.assertTrue(os.path.isdir(env["TRITON_CACHE_DIR"]))

    def test_missing_micro_perf_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                runner.run(self._workloads(d), os.path.join(d, "reports"),
                           micro_perf_dir=os.path.join(d, "nope"))


class TestHistory(unittest.TestCase):
    def _ingest(self, conn, run_id, latency):
        recs = [{"op": "gemm", "provider": "onednn", "arg.M": 128,
                 "latency_us": latency, "mem_bw_GBs": 1.0, "tflops": 1.0}]
        return history.ingest(conn, {"run_id": run_id, "created": run_id,
                                     "model_id": "m", "backend": "INTEL",
                                     "dispatch": "xpu", "tp": 4,
                                     "commits": {"vllm-xpu-kernels": "abc123"}},
                              recs)

    def test_regression_is_detected_at_identical_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            conn = history.connect(history.db_path(d))
            self._ingest(conn, "run-a", 100.0)
            self._ingest(conn, "run-b", 120.0)          # 20 % slower
            diffs = history.compare(conn, "run-a", "run-b")
            self.assertEqual(len(diffs), 1)
            self.assertEqual(diffs[0]["kind"], "regression")
            self.assertAlmostEqual(diffs[0]["delta_pct"], 20.0, places=1)

    def test_noise_below_threshold_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            conn = history.connect(history.db_path(d))
            self._ingest(conn, "run-a", 100.0)
            self._ingest(conn, "run-b", 103.0)
            self.assertEqual(history.compare(conn, "run-a", "run-b"), [])

    def test_runs_record_component_commits(self):
        with tempfile.TemporaryDirectory() as d:
            conn = history.connect(history.db_path(d))
            self._ingest(conn, "run-a", 100.0)
            row = history.runs(conn)[0]
            self.assertEqual(json.loads(row["commits"])["vllm-xpu-kernels"],
                             "abc123")


class TestStore(unittest.TestCase):
    def test_run_layout_is_self_contained(self):
        with tempfile.TemporaryDirectory() as d:
            rid = store.make_run_id("MiniMaxAI/MiniMax-M3", 4, "INTEL")
            p = store.run_paths(rid, d).ensure()
            self.assertTrue(rid.startswith("MiniMax-M3-tp4-intel-"))
            for path in (p.workloads, p.reports, p.cache):
                self.assertTrue(os.path.isdir(path))
            store.RunMeta(run_id=rid, model_id="MiniMaxAI/MiniMax-M3",
                          backend="INTEL", dispatch="xpu", tp=4).write(p)
            self.assertEqual(store.read_meta(p)["model_id"],
                             "MiniMaxAI/MiniMax-M3")
            self.assertEqual(store.list_runs(d)[0]["run_id"], rid)


if __name__ == "__main__":
    unittest.main()
