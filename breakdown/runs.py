# SPDX-License-Identifier: Apache-2.0
"""Where a run's artifacts live, and what survives a restart.

The benchmark and the optimizer already owned their output
(``output/bench/<run_id>/``, ``output/optimize/<run_id>/``); the *profile* did
not. It lived in one in-process dict, so restarting the server — or opening the
UI in a second tab — lost the run that everything downstream is derived from,
and the only way back was to profile again (minutes on a real model).

A run is a directory under ``output/<stage>/<run_id>/`` with a ``state.json``.
Nothing here knows what a stage's state contains; a stage writes what it needs
and reads it back, so the store stays one concept rather than three.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any

#: Root for every stage's runs. Overridable so tests never touch the real one.
_ROOT_ENV = "BREAKDOWN_OUTPUT_ROOT"


def output_root() -> str:
    return os.environ.get(_ROOT_ENV) or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def stage_root(stage: str) -> str:
    return os.path.join(output_root(), stage)


def run_dir(stage: str, run_id: str, create: bool = False) -> str:
    path = os.path.join(stage_root(stage), run_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def new_run_id(prefix: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}" if prefix else stamp


def write_json(path: str, obj: Any, indent: int | None = None) -> str:
    """Write JSON so a reader never sees a half-written file.

    The UI polls these files while a worker thread writes them, and a truncated
    read is an error the reader cannot tell apart from a failed run. Writing to
    a temporary in the same directory and renaming makes the swap atomic, and a
    crash mid-write leaves the previous file intact rather than no file at all.

    There were two of these, one here and one in ``bench.store``, differing
    only in how they named the temporary.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=indent, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_state(stage: str, run_id: str, state: dict[str, Any]) -> str:
    """Persist a run's state atomically."""
    return write_json(
        os.path.join(run_dir(stage, run_id, create=True), "state.json"), state)


def read_state(stage: str, run_id: str) -> dict[str, Any] | None:
    path = os.path.join(run_dir(stage, run_id), "state.json")
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def list_runs(stage: str) -> list[str]:
    """Run ids of a stage, newest first (the ids are timestamp-ordered)."""
    root = stage_root(stage)
    if not os.path.isdir(root):
        return []
    runs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and not d.startswith(".")]
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(root, d)),
              reverse=True)
    return runs


def latest_state(stage: str) -> tuple[str, dict[str, Any]] | None:
    """The newest run of a stage that has a readable state, if any."""
    for run_id in list_runs(stage):
        state = read_state(stage, run_id)
        if state is not None:
            return run_id, state
    return None


class RunState(dict):
    """A stage's current (or last) job, and the lock guarding it.

    The profile and the benchmark each grew their own copy of this: a
    module-level ``{status, run_id, error, ...}`` dict, a ``threading.Lock``
    beside it, and the same four transitions written out by hand. They are one
    concept -- "a long job a worker thread runs and an HTTP route polls" -- so
    they get one implementation.

    It is a ``dict`` subclass rather than a wrapper because the routes and the
    tests already treat the state as a mapping, and because a test that swaps
    in a plain dict to stage a scenario should keep working.
    """

    #: ``idle`` before anything ran, then ``running`` -> ``done`` | ``error``.
    STATUSES = ("idle", "running", "done", "error")

    def __init__(self, **fields: Any) -> None:
        super().__init__(status="idle", run_id=None, error=None, **fields)
        #: Re-entrant so a transition may be taken while the caller holds it.
        self.lock = threading.RLock()
        self._initial = dict(self)

    def begin(self, run_id: str, **fields: Any) -> str:
        """Start a job: clear the previous result and stamp the new id."""
        with self.lock:
            self.update(self._initial)
            self.update(status="running", run_id=run_id, error=None, **fields)
        return run_id

    def finish(self, **fields: Any) -> None:
        with self.lock:
            self.update(status="done", error=None, **fields)

    def fail(self, error: BaseException | str) -> None:
        with self.lock:
            self.update(status="error", error=str(error))

    def reset(self) -> None:
        with self.lock:
            self.update(self._initial)

    @property
    def running(self) -> bool:
        with self.lock:
            return self.get("status") == "running"

    def snapshot(self) -> dict[str, Any]:
        """A plain-dict copy, safe to serialize while the worker writes."""
        with self.lock:
            return dict(self)
