# SPDX-License-Identifier: Apache-2.0
"""Runner bookkeeping: import path, partial re-runs, probe restoration.

These cover failure modes that are invisible in a green run: a worker that
cannot import the repo, a subset re-run that deletes the rest of the results,
and an accumulating op whose probe call is not reset.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.bench import runner, store, timing  # noqa: E402
from breakdown.bench.spec import BenchCase  # noqa: E402


def _case(op: str) -> BenchCase:
    return BenchCase(op=op, args=[{"kind": "tensor", "dims": [4],
                                   "dtype": "bfloat16"}], device="cpu")


def _record(op: str) -> str:
    return json.dumps({"op": op, "status": "ok", "latency_us": 1.0}) + "\n"


class TestWorkerEnvironment(unittest.TestCase):
    def test_worker_imports_the_repo_despite_an_inherited_pythonpath(self):
        """An inherited PYTHONPATH must not become the worker's cwd.

        ``setdefault`` left a dev shell's PYTHONPATH in place, and using it as
        the working directory either raised FileNotFoundError (multi-entry) or
        started a worker that could not import ``breakdown`` - failing the whole
        run, not just one op.
        """
        with tempfile.TemporaryDirectory() as d:
            paths = store.RunPaths(d, "envtest").ensure()
            res = runner.run([_case("aten::relu")], paths, "cpu",
                             budget=0.02, timeouts={"aten::relu": 120},
                             env={"PYTHONPATH": "/opt/venv:/usr/lib",
                                  "PATH": os.environ.get("PATH", "")})
            self.assertEqual(len(res.ops), 1)
            records = store.read_results(paths.results)
            self.assertEqual(len(records), 1, res.ops[0].error)
            self.assertEqual(records[0]["status"], "ok",
                             records[0].get("error"))


class TestPartialRerun(unittest.TestCase):
    def _paths(self, d):
        paths = store.RunPaths(d, "partial").ensure()
        with open(paths.results, "w") as fh:
            fh.write(_record("op_a"))
            fh.write(_record("op_b"))
        return paths

    def test_rerunning_one_op_keeps_the_other_ops_results(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._paths(d)
            runner._drop_records(paths.results, {"op_a"})
            ops = [json.loads(l)["op"] for l in open(paths.results)]
            self.assertEqual(ops, ["op_b"])

    def test_a_full_run_starts_from_an_empty_results_file(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._paths(d)
            runner.run([_case("op_a"), _case("op_b")], paths, "cpu",
                       in_process=False, ops=["op_a", "op_b"],
                       timeouts={"op_a": 5, "op_b": 5})
            # both ops were selected -> previous results were cleared, and the
            # new (failing, since these ops do not exist) records replace them
            prev = [json.loads(l) for l in open(paths.results)]
            self.assertTrue(all(r.get("status") != "ok" for r in prev))


class TestProbeRestoration(unittest.TestCase):
    def test_probe_resets_the_operand_between_its_two_calls(self):
        """An accumulating op must not be run un-reset, not even by the probe.

        ``rows_per_expert`` grows with atomics; two un-restored calls leave it
        at a state the kernel never sees in the model, which is what the
        single-call-per-window recipe exists to prevent.
        """
        state = {"n": 0}
        restored = {"n": 0}

        def call():
            state["n"] += 1

        def restore():
            restored["n"] += 1
            state["n"] = 0

        timing.probe(call, "cpu", restore)
        self.assertEqual(restored["n"], 2)
        self.assertEqual(state["n"], 1)

    def test_probe_without_a_restorer_still_works(self):
        secs = timing.probe(lambda: None, "cpu")
        self.assertGreater(secs, 0)


if __name__ == "__main__":
    unittest.main()
