# SPDX-License-Identifier: Apache-2.0
"""``/api/optimize/*`` endpoints - no GPU, no real agent.

The web layer is a wrapper around :mod:`breakdown.optimize`, so these tests
cover the wiring: that a session needs a ranking, that the selection list
carries each op's launchability, that a spawn is bound to one device, and that
the copy-paste fallback works even where the Copilot CLI is not installed.
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

from app import app# noqa: E402
from breakdown.core import devices as bench_devices  # noqa: E402
from breakdown.bench import store as bench_store  # noqa: E402
from breakdown.optimize.manager import MANAGER  # noqa: E402

from test_optimize_prompt import _doc, _target  # noqa: E402

_AGENT = """#!/bin/sh
echo "mask=$ZE_AFFINITY_MASK"
sleep "${FAKE_AGENT_SECONDS:-1}"
"""


class OptimizeApiTest(unittest.TestCase):
    devices = 2

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["BREAKDOWN_BENCH_ROOT"] = os.path.join(self.tmp, "bench")
        os.environ["BREAKDOWN_OPTIMIZE_ROOT"] = os.path.join(self.tmp, "opt")
        self.agent = os.path.join(self.tmp, "fake-agent")
        with open(self.agent, "w", encoding="utf-8") as fh:
            fh.write(_AGENT)
        os.chmod(self.agent, 0o755)
        os.environ["COPILOT_BIN"] = self.agent
        self._avail = bench_devices.available
        self._detect = bench_devices.detect_device
        self._count = bench_devices.device_count
        bench_devices.available = lambda kind=None: {
            "kind": "xpu", "count": self.devices,
            "indexes": list(range(self.devices)),
            "names": ["fake"] * self.devices}
        bench_devices.detect_device = lambda prefer=None: "xpu"
        bench_devices.device_count = lambda kind: self.devices
        self.run_id = "R"
        paths = bench_store.run_paths(self.run_id).ensure()
        doc = _doc()
        doc["tp"] = 1
        doc["by_phase"]["prefill"]["targets"] = [
            _target(op="op::a", rank=1),
            _target(op="op::roof", rank=2, action="at_roofline"),
        ]
        doc["targets"] = list(doc["by_phase"]["prefill"]["targets"])
        with open(paths.targets, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        self.client = app.test_client()

    def tearDown(self):
        MANAGER.stop(self.run_id)
        # Wait for the *processes*, not just the states: `stop` marks a session
        # stopped as soon as the signal is sent, and a still-live child would
        # otherwise make the next test's start collide with it.
        end = time.time() + 20
        while time.time() < end and (MANAGER.any_active(self.run_id)
                                     or MANAGER._procs):
            time.sleep(0.2)
        MANAGER.shutdown()
        MANAGER._sessions.clear()
        bench_devices.available = self._avail
        bench_devices.detect_device = self._detect
        bench_devices.device_count = self._count
        for key in ("BREAKDOWN_BENCH_ROOT", "BREAKDOWN_OPTIMIZE_ROOT",
                    "COPILOT_BIN", "FAKE_AGENT_SECONDS"):
            os.environ.pop(key, None)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCandidates(OptimizeApiTest):
    def test_it_lists_the_phase_ranking_with_launchability(self):
        resp = self.client.get(
            f"/api/optimize/candidates?run_id={self.run_id}&phase=prefill")
        data = resp.get_json()
        self.assertTrue(data["ok"])
        by_op = {c["op"]: c for c in data["candidates"]}
        self.assertTrue(by_op["op::a"]["launchable"])
        self.assertFalse(by_op["op::roof"]["launchable"])
        self.assertIn("roof", by_op["op::roof"]["reason"])
        self.assertEqual(data["devices"]["count"], self.devices)
        self.assertTrue(data["workspace_root"])

    def test_an_unranked_run_is_refused_with_a_reason(self):
        resp = self.client.get("/api/optimize/candidates?run_id=nope")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("ranked targets", resp.get_json()["error"])


class TestPromptEndpoint(OptimizeApiTest):
    def test_it_returns_a_pasteable_command_without_spawning(self):
        resp = self.client.post("/api/optimize/prompt", json={
            "run_id": self.run_id, "phase": "prefill", "ops": ["op::a"],
            "device_ids": "1"})
        data = resp.get_json()
        self.assertTrue(data["ok"])
        sess = data["sessions"][0]
        self.assertIn("xpu-kernel-optimizer", sess["prompt"])
        self.assertIn("ZE_AFFINITY_MASK=1", sess["command"])
        self.assertIn("$(cat ", sess["command"])
        # the command reads the brief from disk, so it must have been written
        self.assertTrue(os.path.isfile(sess["prompt_file"]))
        self.assertEqual(MANAGER.sessions(self.run_id), [])

    def test_an_unknown_op_is_refused(self):
        resp = self.client.post("/api/optimize/prompt", json={
            "run_id": self.run_id, "ops": ["op::nope"]})
        self.assertEqual(resp.status_code, 400)


class TestStartStop(OptimizeApiTest):
    def test_start_binds_each_session_to_one_device(self):
        os.environ["FAKE_AGENT_SECONDS"] = "2"
        resp = self.client.post("/api/optimize/start", json={
            "run_id": self.run_id, "phase": "prefill",
            "ops": ["op::a", "op::roof"]})
        data = resp.get_json()
        self.assertTrue(data["ok"], data.get("error"))
        held = sorted(d for s in data["sessions"] for d in s["device_ids"])
        self.assertEqual(held, [0, 1])
        self.assertEqual(len(data["pool"]["free"]), 0)

    def test_a_bad_device_index_is_refused_before_anything_starts(self):
        resp = self.client.post("/api/optimize/start", json={
            "run_id": self.run_id, "ops": ["op::a"], "device_ids": "9"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not available", resp.get_json()["error"])

    def test_status_reports_the_queue_and_the_log_is_streamed(self):
        os.environ["FAKE_AGENT_SECONDS"] = "1"
        self.client.post("/api/optimize/start", json={
            "run_id": self.run_id, "phase": "prefill", "ops": ["op::a"]})
        end = time.time() + 20
        while time.time() < end and MANAGER.any_active(self.run_id):
            time.sleep(0.2)
        status = self.client.get(
            f"/api/optimize/status?run_id={self.run_id}").get_json()
        self.assertTrue(status["ok"])
        self.assertFalse(status["active"])
        self.assertEqual(status["sessions"][0]["state"], "done")
        log = self.client.get(
            f"/api/optimize/log?run_id={self.run_id}&op=op::a").get_json()
        self.assertIn("mask=", log["text"])
        self.assertTrue(log["eof"])
        # a second read from the returned offset yields nothing new
        again = self.client.get(
            f"/api/optimize/log?run_id={self.run_id}&op=op::a"
            f"&offset={log['offset']}").get_json()
        self.assertEqual(again["text"], "")

    def test_a_second_session_does_not_append_to_the_first_log(self):
        # The log is opened "wb", not "ab": appending across runs made the
        # streamed pane show the concatenation of every session ever opened for
        # that op (it read as the agent repeating itself), and it breaks the
        # /api/optimize/log offset contract.
        os.environ["FAKE_AGENT_SECONDS"] = "1"
        for _ in range(2):
            self.client.post("/api/optimize/start", json={
                "run_id": self.run_id, "phase": "prefill", "ops": ["op::a"]})
            end = time.time() + 20
            while time.time() < end and MANAGER.any_active(self.run_id):
                time.sleep(0.2)
        log = self.client.get(
            f"/api/optimize/log?run_id={self.run_id}&op=op::a").get_json()
        self.assertEqual(log["text"].count("mask="), 1, log["text"])

    def test_stopping_releases_the_devices(self):
        os.environ["FAKE_AGENT_SECONDS"] = "10"
        self.client.post("/api/optimize/start", json={
            "run_id": self.run_id, "phase": "prefill",
            "ops": ["op::a", "op::roof"]})
        resp = self.client.post("/api/optimize/stop",
                                json={"run_id": self.run_id})
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(sorted(data["stopped"]), ["op::a", "op::roof"])
        end = time.time() + 20
        while time.time() < end and MANAGER.any_active(self.run_id):
            time.sleep(0.2)
        self.assertEqual(len(MANAGER.pool_snapshot()["free"]), self.devices)

    def test_an_unknown_op_is_refused_with_a_readable_message(self):
        # KeyError's str() quotes its argument; this message reaches an alert().
        resp = self.client.post("/api/optimize/start", json={
            "run_id": self.run_id, "ops": ["op::nope"]})
        self.assertEqual(resp.status_code, 400)
        error = resp.get_json()["error"]
        self.assertTrue(error.startswith("'op::nope' is not"), error)

    def test_no_endpoint_ships_the_whole_brief_as_argv(self):
        os.environ["FAKE_AGENT_SECONDS"] = "5"
        started = self.client.post("/api/optimize/start", json={
            "run_id": self.run_id, "phase": "prefill",
            "ops": ["op::a"]}).get_json()
        status = self.client.get(
            f"/api/optimize/status?run_id={self.run_id}").get_json()
        stopped = self.client.post("/api/optimize/stop",
                                   json={"run_id": self.run_id}).get_json()
        for data in (started, status, stopped):
            self.assertNotIn("argv", data["sessions"][0])

    def test_a_non_numeric_log_offset_is_a_400_not_a_500(self):
        resp = self.client.get(
            f"/api/optimize/log?run_id={self.run_id}&op=op::a&offset=abc")
        self.assertEqual(resp.status_code, 400)

    def test_a_session_from_a_previous_server_is_not_reported_running(self):
        # The index on disk outlives the process, but its agents do not: the
        # atexit hook kills them, so a restored "running" would poll forever.
        MANAGER.stop(self.run_id)
        MANAGER._sessions.clear()
        index = os.path.join(os.environ["BREAKDOWN_OPTIMIZE_ROOT"],
                             self.run_id, "index.json")
        os.makedirs(os.path.dirname(index), exist_ok=True)
        with open(index, "w", encoding="utf-8") as fh:
            json.dump({"run_id": self.run_id,
                       "sessions": [{"op": "op::a", "state": "running"}]}, fh)
        data = self.client.get(
            f"/api/optimize/status?run_id={self.run_id}").get_json()
        self.assertFalse(data["active"])
        self.assertEqual(data["sessions"][0]["state"], "stopped")
        self.assertIn("restarted", data["sessions"][0]["error"])

    def test_a_missing_copilot_is_a_clear_error_not_a_traceback(self):
        os.environ["COPILOT_BIN"] = os.path.join(self.tmp, "does-not-exist")
        resp = self.client.post("/api/optimize/start", json={
            "run_id": self.run_id, "ops": ["op::a"]})
        self.assertEqual(resp.status_code, 501)
        self.assertIn("Copilot CLI", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
