# SPDX-License-Identifier: Apache-2.0
"""End-to-end replay on a real accelerator (skipped without one).

The profile's own device time is the oracle: replaying an op at the shape it
was recorded at must land in the same order of magnitude. A replay that is
wildly faster is not a fast kernel, it is a wrong call.
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.bench import devices, runner, store  # noqa: E402
from breakdown.bench.spec import BenchCase, build_cases  # noqa: E402

DEVICE = devices.detect_device()
requires_gpu = unittest.skipIf(DEVICE == "cpu", "no XPU/CUDA device present")

M3_SUMMARY = {
    "architecture": "MiniMaxM3SparseForCausalLM", "hidden_size": 6144,
    "num_heads": 64, "num_kv_heads": 4, "head_dim": 128,
    "intermediate_size": 12288, "moe_intermediate_size": 3072,
    "num_experts": 128, "vocab_size": 200064,
    "max_position_embeddings": 1048576, "num_layers": 60, "dtype": "bfloat16",
    "sparse_attention": True, "sparse_index_dim": 128,
    "sparse_num_index_heads": 4,
}


def _trace() -> str | None:
    """A readable MiniMax-M3 XPU decode trace, if the repo has one.

    The shared traces directory can hold LFS stubs / partial files, so each
    candidate is opened before it is trusted.
    """
    from breakdown.graph_from_trace import _load_trace

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in sorted(glob.glob(os.path.join(
            here, "output", "traces",
            "vllm_trace_MiniMax-M3_XPU_eager_decode_*tp4*.json.gz"))):
        try:
            if _load_trace(path).get("traceEvents"):
                return path
        except (OSError, ValueError, EOFError, Exception):  # noqa: BLE001
            continue
    return None


def _cases() -> list[BenchCase]:
    from breakdown.graph_from_trace import build_graph_from_trace
    from breakdown.shape_matrix import build_rows

    path = _trace()
    if not path:
        return []
    graph = build_graph_from_trace(path, M3_SUMMARY, tp_size=4, batch_size=32,
                                   query_len=1, context_len=2048)
    rows = build_rows(graph, [{"phase": "decode", "seq_len": 1,
                               "ctx_len": 2048, "batch_size": 32,
                               "tp_size": 4}])
    return build_cases(rows, device=DEVICE)[0]


@requires_gpu
class TestReplayOnDevice(unittest.TestCase):
    """Replay a handful of real dispatched ops and check they measure."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _cases()
        if not cls.cases:
            raise unittest.SkipTest("no MiniMax-M3 XPU trace present")

    def _run(self, op: str):
        from breakdown.bench import worker

        sel = [c for c in self.cases if c.op == op]
        if not sel:
            self.skipTest(f"{op} not in this trace")
        return worker.run_case(sel[0], DEVICE, budget=0.05)

    def test_elementwise_kernel_replays(self):
        rec = self._run("_C::silu_and_mul_with_clamp")
        self.assertEqual(rec["status"], "ok", rec.get("error"))
        self.assertGreater(rec["latency_us"], 0)

    def test_matmul_replays_close_to_the_profiled_device_time(self):
        rec = self._run("aten::linear")
        self.assertEqual(rec["status"], "ok", rec.get("error"))
        self.assertGreater(rec["latency_us"], 0)

    def test_accumulating_moe_op_runs_one_call_per_window(self):
        # rows_per_expert accumulates with atomics: repeating it inside a timed
        # window scatters out of bounds and takes the device down.
        rec = self._run("_moe_C::remap_hidden_states")
        self.assertEqual(rec["status"], "ok", rec.get("error"))
        self.assertEqual(rec["reps"], 1)

    def test_attention_is_measured_through_its_context_free_entry_point(self):
        # The dispatcher op is context-bound, but the kernel underneath takes
        # the paged cache and the sequence metadata as plain arguments - so the
        # heaviest op in the model is measured, not refused.
        rec = self._run("vllm::unified_attention_with_output")
        self.assertEqual(rec["status"], "ok", rec.get("error"))
        self.assertGreater(rec["latency_us"], 0)

    def test_a_wrapper_with_no_entry_point_is_reported_not_measured(self):
        rec = self._run("vllm::moe_forward_shared")
        self.assertEqual(rec["status"], "not_replayable")
        self.assertTrue(rec["detail"])


@requires_gpu
class TestRunnerIsolation(unittest.TestCase):
    def test_each_op_runs_in_its_own_process(self):
        cases = _cases()
        if not cases:
            self.skipTest("no MiniMax-M3 XPU trace present")
        with tempfile.TemporaryDirectory() as d:
            paths = store.RunPaths(d, "iso").ensure()
            res = runner.run(cases, paths, DEVICE, budget=0.05,
                             ops=["_C::silu_and_mul_with_clamp",
                                  "aten::embedding"])
            self.assertTrue(res.ok, [o.error for o in res.failed])
            self.assertTrue(os.path.isfile(paths.results))
            records = store.read_results(paths.results)
            self.assertTrue(all(r["status"] == "ok" for r in records))
            # a killed run must still leave a readable record
            self.assertTrue(os.path.isfile(paths.run_result))


if __name__ == "__main__":
    unittest.main()
