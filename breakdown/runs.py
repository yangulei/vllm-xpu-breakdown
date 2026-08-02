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


def write_state(stage: str, run_id: str, state: dict[str, Any]) -> str:
    """Persist a run's state atomically.

    Atomically because the UI polls this file's stage while a worker thread
    writes it: a half-written state.json read by a poll is an error the reader
    cannot distinguish from a failed run.
    """
    path = os.path.join(run_dir(stage, run_id, create=True), "state.json")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


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
