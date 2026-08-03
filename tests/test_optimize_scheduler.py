# SPDX-License-Identifier: Apache-2.0
"""One session owns one GPU - the scheduling rule, without a GPU or an agent.

An optimization session profiles and benchmarks continuously, so two sessions
sharing a device would measure each other's interference. These tests pin the
consequences: exclusivity, a FIFO queue for the surplus, a lease that is
released on *every* exit path, and an enforced single-device environment.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.core import devices as bench_devices  # noqa: E402
from breakdown.optimize import scheduler  # noqa: E402
from breakdown.optimize.manager import OptimizeManager  # noqa: E402

from test_optimize_prompt import _doc, _target  # noqa: E402


class TestDevicePool(unittest.TestCase):
    def pool(self, n):
        return scheduler.DevicePool("xpu", list(range(n)))

    def test_a_device_serves_one_session_at_a_time(self):
        pool = self.pool(2)
        self.assertEqual(pool.acquire("a"), [0])
        self.assertEqual(pool.acquire("b"), [1])
        self.assertIsNone(pool.acquire("c"))   # nothing free - it must wait
        pool.release("a")
        self.assertEqual(pool.acquire("c"), [0])

    def test_a_lease_is_idempotent_for_the_same_session(self):
        pool = self.pool(2)
        self.assertEqual(pool.acquire("a"), [0])
        self.assertEqual(pool.acquire("a"), [0])
        self.assertEqual(pool.free_ids(), [1])

    def test_a_multi_device_lease_is_all_or_nothing(self):
        pool = self.pool(2)
        pool.acquire("a")
        self.assertIsNone(pool.acquire("collective", need=2))
        self.assertEqual(pool.free_ids(), [1])   # the free one is not stranded
        pool.release("a")
        self.assertEqual(pool.acquire("collective", need=2), [0, 1])

    def test_an_impossible_request_fails_instead_of_waiting_forever(self):
        pool = self.pool(2)
        with self.assertRaises(scheduler.LeaseError):
            pool.acquire("tp4", need=4)

    def test_the_lease_is_enforced_in_the_child_environment(self):
        pool = self.pool(4)
        self.assertEqual(pool.env_for([2]), {"ZE_AFFINITY_MASK": "2"})
        self.assertEqual(scheduler.DevicePool("cuda", [0, 1]).env_for([1]),
                         {"CUDA_VISIBLE_DEVICES": "1"})

    def test_releasing_an_unknown_session_is_harmless(self):
        pool = self.pool(1)
        self.assertEqual(pool.release("nobody"), [])
        self.assertEqual(pool.free_ids(), [0])


_AGENT = """#!/bin/sh
echo "mask=$ZE_AFFINITY_MASK cwd=$PWD"
sleep "${FAKE_AGENT_SECONDS:-1}"
exit "${FAKE_AGENT_EXIT:-0}"
"""


class ManagerTest(unittest.TestCase):
    """A manager driven by a fake agent binary and a fixed device count."""

    devices = 2

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["BREAKDOWN_OPTIMIZE_ROOT"] = os.path.join(self.tmp, "out")
        self.agent = os.path.join(self.tmp, "fake-agent")
        with open(self.agent, "w", encoding="utf-8") as fh:
            fh.write(_AGENT)
        os.chmod(self.agent, 0o755)
        os.environ["COPILOT_BIN"] = self.agent
        self._avail = bench_devices.available
        bench_devices.available = lambda kind=None: {
            "kind": "xpu", "count": self.devices,
            "indexes": list(range(self.devices)),
            "names": ["fake"] * self.devices}
        self.mgr = OptimizeManager()

    def tearDown(self):
        self.mgr.shutdown()
        bench_devices.available = self._avail
        os.environ.pop("BREAKDOWN_OPTIMIZE_ROOT", None)
        os.environ.pop("COPILOT_BIN", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def doc(self, ops):
        doc = _doc()
        doc["by_phase"]["prefill"]["targets"] = [
            _target(op=op, rank=i + 1) for i, op in enumerate(ops)]
        doc["targets"] = list(doc["by_phase"]["prefill"]["targets"])
        doc["tp"] = 1
        return doc

    def start(self, ops, **over):
        return self.mgr.start(run_id="R", doc=self.doc(ops), ops=ops,
                              phase="prefill", device_kind="xpu",
                              workspace_root=self.tmp, **over)

    def wait(self, timeout=30):
        end = time.time() + timeout
        while time.time() < end and self.mgr.any_active("R"):
            time.sleep(0.2)


class TestSessions(ManagerTest):
    def test_only_as_many_sessions_run_as_there_are_gpus(self):
        os.environ["FAKE_AGENT_SECONDS"] = "2"
        state = self.start(["op::a", "op::b", "op::c"])
        by_op = {s["op"]: s for s in state["sessions"]}
        running = [s for s in by_op.values() if s["state"] == "running"]
        pending = [s for s in by_op.values() if s["state"] == "pending"]
        self.assertEqual(len(running), 2)
        self.assertEqual(len(pending), 1)
        # every running session holds a *different* device
        held = sorted(d for s in running for d in s["device_ids"])
        self.assertEqual(held, [0, 1])
        self.assertEqual(pending[0]["queue_position"], 1)
        self.wait()
        os.environ.pop("FAKE_AGENT_SECONDS", None)

    def test_a_freed_gpu_starts_the_queued_session(self):
        os.environ["FAKE_AGENT_SECONDS"] = "1"
        self.start(["op::a", "op::b", "op::c"])
        self.wait()
        states = {s.op: s.state for s in self.mgr.sessions("R")}
        self.assertEqual(set(states.values()), {"done"})
        # and no lease survived the run
        self.assertEqual(len(self.mgr.pool_snapshot()["free"]), self.devices)
        os.environ.pop("FAKE_AGENT_SECONDS", None)

    def test_the_session_only_sees_the_device_it_leased(self):
        self.start(["op::a"])
        self.wait()
        sess = self.mgr.get("R", "op::a")
        with open(sess.log_file, encoding="utf-8") as fh:
            logged = fh.read()
        self.assertIn(f"mask={sess.device_ids[0]}", logged)
        self.assertIn(f"cwd={self.tmp}", logged)

    def test_a_failing_agent_is_reported_and_releases_its_gpu(self):
        os.environ["FAKE_AGENT_EXIT"] = "3"
        self.start(["op::a"])
        self.wait()
        sess = self.mgr.get("R", "op::a")
        self.assertEqual(sess.state, "failed")
        self.assertEqual(sess.exit_code, 3)
        self.assertEqual(len(self.mgr.pool_snapshot()["free"]), self.devices)
        os.environ.pop("FAKE_AGENT_EXIT", None)

    def test_stopping_a_pending_session_drops_it_from_the_queue(self):
        os.environ["FAKE_AGENT_SECONDS"] = "3"
        self.start(["op::a", "op::b", "op::c"])
        pending = [s.op for s in self.mgr.sessions("R") if s.state == "pending"]
        self.mgr.stop("R", pending[0])
        self.assertEqual(self.mgr.get("R", pending[0]).state, "stopped")
        self.mgr.stop("R")
        self.wait()
        self.assertFalse(self.mgr.any_active("R"))
        self.assertEqual(len(self.mgr.pool_snapshot()["free"]), self.devices)
        os.environ.pop("FAKE_AGENT_SECONDS", None)

    def test_a_second_start_for_a_live_op_is_refused(self):
        os.environ["FAKE_AGENT_SECONDS"] = "3"
        self.start(["op::a"])
        with self.assertRaises(RuntimeError):
            self.start(["op::a"])
        self.mgr.stop("R")
        self.wait()
        os.environ.pop("FAKE_AGENT_SECONDS", None)

    def test_a_dry_run_writes_the_brief_and_spawns_nothing(self):
        state = self.start(["op::a"], spawn=False)
        sess = state["sessions"][0]
        self.assertEqual(sess["state"], "pending")
        self.assertIsNone(sess["pid"])
        self.assertTrue(os.path.isfile(sess["prompt_file"]))
        self.assertIn("xpu-kernel-optimizer",
                      open(sess["prompt_file"], encoding="utf-8").read())

    def test_the_run_records_what_it_did(self):
        self.start(["op::a"])
        self.wait()
        index = os.path.join(os.environ["BREAKDOWN_OPTIMIZE_ROOT"], "R",
                             "index.json")
        with open(index, encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["run_id"], "R")
        self.assertEqual(doc["sessions"][0]["op"], "op::a")
        self.assertNotIn("argv", doc["sessions"][0])   # it embeds the brief

    def test_the_fallback_command_is_pasteable(self):
        state = self.start(["op::a"], spawn=False)
        command = state["sessions"][0]["command"]
        self.assertIn("--allow-all-tools", command)
        self.assertIn("$(cat ", command)     # the brief is read from its file
        self.assertNotIn("xpu-kernel-optimizer", command)

    def test_artifacts_stay_where_the_session_started(self):
        # The state is written again from the wait thread after the agent
        # exits. Re-reading $BREAKDOWN_OPTIMIZE_ROOT there would send that
        # write wherever the environment points *then* - which, in a test run,
        # is another test's teardown, and in a server a stray output/ tree.
        root = os.environ["BREAKDOWN_OPTIMIZE_ROOT"]
        os.environ["FAKE_AGENT_SECONDS"] = "1"
        self.start(["op::a"])
        moved = os.path.join(self.tmp, "moved")
        os.environ["BREAKDOWN_OPTIMIZE_ROOT"] = moved
        try:
            self.wait()
        finally:
            os.environ["BREAKDOWN_OPTIMIZE_ROOT"] = root
        self.assertFalse(os.path.exists(moved))
        state = os.path.join(root, "R", "op-a", "session.json")
        with open(state, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["state"], "done")

    def test_an_unranked_op_is_refused(self):
        with self.assertRaises(KeyError):
            self.mgr.start(run_id="R", doc=self.doc(["op::a"]),
                           ops=["op::nope"], phase="prefill",
                           device_kind="xpu", workspace_root=self.tmp)


class TestNoDevices(ManagerTest):
    devices = 0

    def test_a_session_cannot_start_without_a_device(self):
        with self.assertRaises(RuntimeError):
            self.start(["op::a"])


if __name__ == "__main__":
    unittest.main()
