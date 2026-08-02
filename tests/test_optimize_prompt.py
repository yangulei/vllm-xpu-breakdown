# SPDX-License-Identifier: Apache-2.0
"""The optimization brief and its refusal rules - no GPU, no agent.

The brief is the whole contract between the ranking and the kernel session, so
what it must carry (the measured baseline, the roofline it is judged against,
the commands that reproduce it) and what it must refuse (an op with no headroom
or no editable source) are worth pinning down.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.optimize import prompt as op_prompt  # noqa: E402


def _target(**over):
    target = {
        "op": "_C::rms_norm",
        "backend": "vllm-xpu-kernels",
        "rank": 1,
        "e2e_us": 4200.0,
        "share_of_e2e": 0.12,
        "calls": 56,
        "action": "optimize_kernel",
        "flags": [],
        "savings_us": {"total": 900.0},
        "roofline": {"util": 0.41, "util_dram": 0.41, "bound": "memory",
                     "unit": "DRAM", "memory_level": "dram", "ai": 0.5,
                     "ridge_ai": 27.0, "peak_bw_gbs": 456.0,
                     "peak_tflops": 98.3, "vector_tflops": 12.3,
                     "target_util": 0.8},
        "kernel": {"repo": "vllm-xpu-kernels",
                   "kernel_dir": "vllm-xpu-kernels/csrc",
                   "files": ["csrc/**/*.cpp"], "language": "sycl",
                   "build_cmd": "cd vllm-xpu-kernels && pip install -e .",
                   "test_cmd": "cd vllm-xpu-kernels && pytest tests -v",
                   "notes": "build MSA only"},
        "top_shapes": [
            {"phase": "prefill", "profiled": True, "calls": 56,
             "shape": "[2048,4096]bf16", "latency_us": 31.25,
             "weighted_us": 1750.0, "traced_device_time_us": 30.0,
             "replay_vs_traced": 1.04,
             "bench_cmd": "python3 -m breakdown.bench case --run R --case-id abc",
             "profile_cmd": "unitrace -d python3 -m breakdown.bench case --run R --case-id abc"},
            {"phase": "decode", "profiled": False, "calls": 56,
             "shape": "[32,4096]bf16", "latency_us": 6.5,
             "weighted_us": 364.0,
             "bench_cmd": "python3 -m breakdown.bench case --run R --case-id def"},
        ],
    }
    target.update(over)
    return target


def _doc(**over):
    doc = {
        "run_id": "R", "tp": 2, "device": "xpu", "sku": "BMG",
        "targets": [_target(action="at_roofline", rank=7, e2e_us=99.0)],
        "by_phase": {
            "prefill": {"targets": [_target()],
                        "operating_point": {"seq_len": 2048, "ctx_len": 0,
                                            "batch_size": 1, "profiled": True}},
            "decode": {"targets": [_target(op="_C::rms_norm", rank=3)],
                       "operating_point": {"seq_len": 1, "ctx_len": 2048,
                                           "batch_size": 32}},
        },
    }
    doc.update(over)
    return doc


class TestLaunchability(unittest.TestCase):
    def test_a_kernel_with_headroom_is_launchable(self):
        can, reason = op_prompt.launchability(_target())
        self.assertTrue(can)
        self.assertIn("headroom", reason)

    def test_an_op_at_the_roofline_is_refused(self):
        can, reason = op_prompt.launchability(_target(action="at_roofline"))
        self.assertFalse(can)
        self.assertIn("roof", reason)

    def test_an_untrusted_cost_model_is_refused(self):
        can, reason = op_prompt.launchability(_target(action="check_cost_model"))
        self.assertFalse(can)
        self.assertIn("cost model", reason)

    def test_an_op_without_editable_source_is_refused(self):
        no_build = _target()
        no_build["kernel"] = {**no_build["kernel"], "build_cmd": None}
        can, reason = op_prompt.launchability(no_build)
        self.assertFalse(can)
        self.assertIn("build command", reason)

    def test_a_collective_with_no_kernel_dir_is_refused(self):
        target = _target(op="c10d::allreduce_", action="tune_config")
        target["kernel"] = {"repo": "-", "kernel_dir": "-", "files": [],
                            "build_cmd": None}
        can, reason = op_prompt.launchability(target)
        self.assertFalse(can)

    def test_a_tunable_library_op_is_launchable_but_says_so(self):
        target = _target(action="tune_config")
        can, reason = op_prompt.launchability(target)
        self.assertTrue(can)
        self.assertIn("configuration", reason)


class TestCandidates(unittest.TestCase):
    def test_candidates_carry_the_reason_for_every_op(self):
        rows = op_prompt.candidates(_doc(), "prefill")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["launchable"])
        self.assertEqual(rows[0]["kernel_dir"], "vllm-xpu-kernels/csrc")

    def test_the_phase_ranking_wins_over_the_combined_one(self):
        # The combined row is at_roofline; the prefill row is not. A session is
        # opened for a phase, so it must be briefed with that phase's record.
        by_op = op_prompt.targets_by_op(_doc(), "prefill")
        self.assertEqual(by_op["_C::rms_norm"]["action"], "optimize_kernel")
        self.assertEqual(by_op["_C::rms_norm"]["rank"], 1)


class TestPrompt(unittest.TestCase):
    def setUp(self):
        self.text = op_prompt.build_prompt(
            _target(), _doc(), run_id="R", phase="prefill",
            device_ids=[2], device_kind="xpu",
            workspace_root="/ws", artifact_dir="/ws/out/op")

    def test_it_names_the_skill_that_owns_the_loop(self):
        self.assertIn(op_prompt.OPTIMIZER_SKILL, self.text)

    def test_it_carries_the_measured_baseline_and_its_commands(self):
        self.assertIn("31.250 us", self.text)          # the baseline latency
        self.assertIn("--case-id abc", self.text)      # how to reproduce it
        self.assertIn("unitrace", self.text)           # how to profile it
        self.assertIn("Acceptance criterion", self.text)

    def test_it_carries_the_roofline_it_is_judged_against(self):
        self.assertIn("DRAM", self.text)
        self.assertIn("41%", self.text)
        self.assertIn("456.0 GB/s", self.text)

    def test_it_carries_the_kernel_source_and_its_commands(self):
        self.assertIn("vllm-xpu-kernels/csrc", self.text)
        self.assertIn("pytest tests -v", self.text)
        self.assertIn("/ws", self.text)

    def test_it_states_the_device_the_session_owns(self):
        self.assertIn("xpu device 2", self.text)
        self.assertIn("ZE_AFFINITY_MASK", self.text)
        self.assertIn("exclusively", self.text)

    def test_it_names_the_profiled_operating_point(self):
        self.assertIn("seq=2048", self.text)
        self.assertIn("profiled", self.text)

    def test_flags_are_carried_so_the_baseline_is_not_blindly_trusted(self):
        target = _target(flags=["replay is 5x faster than the trace"])
        text = op_prompt.build_prompt(target, _doc(), phase="prefill")
        self.assertIn("5x faster", text)
        self.assertIn("Caveats", text)

    def test_a_refused_op_launched_anyway_is_told_why_it_was_refused(self):
        target = _target(action="at_roofline")
        text = op_prompt.build_prompt(target, _doc(), phase="prefill")
        self.assertIn("did not consider this op worth a session", text)

    def test_no_device_lease_is_stated_plainly(self):
        text = op_prompt.build_prompt(_target(), _doc(), phase="prefill")
        self.assertIn("No device restriction", text)


if __name__ == "__main__":
    unittest.main()
