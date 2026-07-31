# SPDX-License-Identifier: Apache-2.0
"""Ranking, timing statistics, budgets and history - all GPU-free."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.bench import estimate, history, rank, reports, timing  # noqa: E402


def _rec(op, latency, layers=1, flops=0.0, nbytes=0.0, phase="decode",
         backend="vllm-xpu-kernels", traced=0.0, comparable=False, **kw):
    r = {
        "op": op, "status": "ok", "device": "xpu", "phase": phase,
        "seq_len": 1, "ctx_len": 2048, "batch_size": 32, "tp": 4,
        "backend": backend, "layers": layers, "latency_us": latency,
        "flops": flops, "bytes": nbytes, "shape": "[32, 6144]",
        "shape_key": f"{op}-key", "case_id": f"{op}-case",
        "traced_device_time_us": traced, "traced_comparable": comparable,
    }
    r.update(kw)
    return r


PEAKS = {"bw_gbs": 456.0, "tflops": 98.3}


class TestUtilization(unittest.TestCase):
    def test_memory_bound_utilization_is_against_the_bandwidth_peak(self):
        # 456 GB/s for 1 us moves 456e3 bytes; half of that is 50 % of peak.
        util, bound = estimate.utilization(1.0, 0.0, 228_000, PEAKS)
        self.assertEqual(bound, "memory")
        self.assertAlmostEqual(util, 0.5, places=2)

    def test_the_binding_roof_is_whichever_is_higher(self):
        util, bound = estimate.utilization(1.0, 98.3e6, 1.0, PEAKS)
        self.assertEqual(bound, "compute")
        self.assertAlmostEqual(util, 1.0, places=2)


class TestRank(unittest.TestCase):
    def test_call_count_outweighs_a_single_expensive_op(self):
        recs = [_rec("small_op_in_every_layer", 10.0, layers=57,
                     nbytes=1_000),
                _rec("big_op_once", 400.0, layers=1, nbytes=1_000)]
        doc = rank.rank(recs)
        self.assertEqual(doc["targets"][0]["op"], "small_op_in_every_layer")
        self.assertEqual(doc["schema_version"], rank.SCHEMA_VERSION)
        self.assertEqual(doc["engine"], "replay")

    def test_an_op_at_the_roofline_is_not_a_target(self):
        # 1 us at 456 GB/s = 456e3 bytes; 410e3 is ~90 % of peak.
        recs = [_rec("saturated", 1.0, layers=1, nbytes=410_000)]
        doc = rank.rank(recs)
        t = doc["targets"][0]
        self.assertEqual(t["action"], "at_roofline")
        self.assertEqual(t["savings_us"]["total"], 0.0)

    def test_op_with_no_editable_source_is_tune_config_not_a_kernel_session(self):
        recs = [_rec("aten::div", 10.0, backend="torch-xpu-ops", nbytes=1_000)]
        doc = rank.rank(recs)
        self.assertEqual(doc["targets"][0]["action"], "tune_config")

    def test_op_with_editable_source_and_headroom_is_a_kernel_session(self):
        recs = [_rec("_C::silu_and_mul_with_clamp", 100.0, layers=57,
                     nbytes=1_000)]
        doc = rank.rank(recs, kernel_sources=rank.load_kernel_sources())
        t = doc["targets"][0]
        self.assertEqual(t["action"], "optimize_kernel")
        self.assertTrue(t["kernel"]["build_cmd"])
        self.assertGreater(t["savings_us"]["total"], 0)

    def test_impossible_utilization_flags_the_cost_model_not_the_kernel(self):
        # The analytic bytes charge an embedding for the whole table it could
        # read; the resulting 300x-of-peak "utilization" must not retire the op
        # as done.
        recs = [_rec("aten::embedding", 1.0, nbytes=456_000_00 * 30)]
        doc = rank.rank(recs)
        t = doc["targets"][0]
        self.assertEqual(t["action"], "check_cost_model")
        self.assertTrue(t["flags"])

    def test_replay_far_faster_than_the_profile_is_flagged(self):
        recs = [_rec("suspicious", 1.0, traced=100.0, comparable=True,
                     nbytes=1_000)]
        doc = rank.rank(recs)
        self.assertTrue(any("faster than the profiled" in f
                            for f in doc["targets"][0]["flags"]))

    def test_only_the_chosen_operating_point_is_ranked(self):
        recs = [_rec("op", 10.0, batch_size=32), _rec("op", 10.0, batch_size=32),
                _rec("op", 999.0, batch_size=1)]
        doc = rank.rank(recs)
        self.assertEqual(doc["operating_points"]["decode"]["batch_size"], 32)
        self.assertEqual(doc["targets"][0]["e2e_us"], 20.0)

    def test_ranking_without_any_measurement_is_an_error_not_an_empty_table(self):
        with self.assertRaises(ValueError):
            rank.rank([_rec("op", 1.0, status="failed")])


class TestTimingPlan(unittest.TestCase):
    def test_a_fast_kernel_is_repeated_inside_the_window(self):
        reps, windows = timing.plan_window(5e-6)      # 5 us
        self.assertGreater(reps, 100)
        self.assertGreaterEqual(windows, timing.MIN_WINDOWS)

    def test_a_slow_kernel_is_not_repeated_into_a_huge_window(self):
        reps, _ = timing.plan_window(0.05)            # 50 ms
        self.assertEqual(reps, timing.MIN_REPS)

    def test_restorer_is_only_built_for_mutated_operands(self):
        self.assertIsNone(timing.make_restorer([]))


class TestEstimate(unittest.TestCase):
    def test_timeout_covers_startup_measurement_and_allocation(self):
        t = estimate.op_timeout(10, 0.5, startup_s=60, case_overhead_s=3,
                                safety=3, alloc_bytes=20e9)
        # 60 + 10*3.5 + 10 s of allocation, times 3
        self.assertGreater(t, 300)
        self.assertLessEqual(t, estimate.MAX_TIMEOUT_S)

    def test_timeout_never_drops_below_the_floor(self):
        self.assertGreaterEqual(estimate.op_timeout(1, 0.01, startup_s=1,
                                                    case_overhead_s=0,
                                                    safety=1),
                                estimate.MIN_TIMEOUT_S)

    def test_calibration_uses_previous_runs(self):
        prev = [{"ops": [{"cases": 1, "seconds": 20.0},
                         {"cases": 11, "seconds": 130.0}]}]
        startup, per_case = estimate.calibrate(prev)
        self.assertAlmostEqual(startup, 20.0)
        self.assertAlmostEqual(per_case, 10.0)

    def test_roofline_bound_is_the_fastest_possible_time(self):
        us, bound = estimate.roofline_bound_us(0.0, 456_000, PEAKS)
        self.assertEqual(bound, "memory")
        self.assertAlmostEqual(us, 1.0, places=2)


class TestReports(unittest.TestCase):
    def test_enrich_adds_utilization_and_the_fidelity_ratio(self):
        rich = reports.enrich([_rec("op", 2.0, nbytes=456_000, traced=4.0,
                                    comparable=True)], PEAKS)[0]
        self.assertAlmostEqual(rich["util"], 0.5, places=2)
        self.assertAlmostEqual(rich["replay_vs_traced"], 0.5, places=2)

    def test_coverage_lists_what_was_not_measured_and_why(self):
        recs = [_rec("ok_op", 1.0),
                _rec("wrapper", 0.0, status="not_replayable",
                     detail="needs forward context", traced=60.0)]
        cov = reports.coverage(reports.enrich(recs, PEAKS))
        self.assertEqual(len(cov), 1)
        self.assertEqual(cov[0]["Op"], "wrapper")
        self.assertEqual(cov[0]["Reason"], "needs forward context")


class TestHistory(unittest.TestCase):
    def test_only_shapes_present_in_both_runs_are_compared(self):
        with tempfile.TemporaryDirectory() as d:
            conn = history.connect(history.db_path(d))
            history.ingest(conn, {"run_id": "base"},
                           [_rec("op", 10.0), _rec("gone", 5.0)])
            history.ingest(conn, {"run_id": "new"},
                           [_rec("op", 15.0), _rec("fresh", 5.0)])
            diffs = history.compare(conn, "base", "new")
            self.assertEqual([d_["op"] for d_ in diffs], ["op"])
            self.assertEqual(diffs[0]["kind"], "regression")
            self.assertAlmostEqual(diffs[0]["delta_pct"], 50.0)

    def test_failed_cases_are_not_ingested(self):
        with tempfile.TemporaryDirectory() as d:
            conn = history.connect(history.db_path(d))
            n = history.ingest(conn, {"run_id": "r"},
                               [_rec("op", 0.0, status="failed")])
            self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
