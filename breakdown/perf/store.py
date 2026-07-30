# SPDX-License-Identifier: Apache-2.0
"""On-disk layout for perf runs, and their provenance.

Everything a run produces lives under one directory so it is reproducible and
attributable::

    output/perf/<run_id>/
        matrix.xlsx  workloads/  reports/  logs/
        coverage.json  opt_targets.json  reports_merged.xlsx  run.json
    output/perf/history.sqlite
    output/perf/.cache/{sycl,triton}      persistent kernel caches

``run.json`` records the model, sweep and the **git commit of every component
that can change a number** (the kernel repos, xpu-perf, this tool), so a
ranking is always attributable and a regression can be bisected. Keeping these
artifacts owned - rather than in a scratch directory - is deliberate: a lost
matrix means the whole benchmark is unreproducible without re-profiling on the
GPU.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from breakdown.perf import devices

#: Repos whose commit changes measured performance.
PROVENANCE_REPOS = (
    "vllm-xpu-kernels",
    "applications.ai.gpu.vllm-xpu-kernels",
    "applications.ai.gpu.deepklox",
    "sycl-tla",
    "intel-xpu-backend-for-triton",
    "vllm-xpu",
    "xpu-perf",
)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def perf_root(base: str | None = None) -> str:
    """``output/perf`` (override with ``$BREAKDOWN_PERF_ROOT``)."""
    if base:
        return base
    env = os.environ.get("BREAKDOWN_PERF_ROOT")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(repo, "output", "perf")


def make_run_id(model_id: str, tp: int, backend: str,
                when: datetime | None = None) -> str:
    model = _SAFE.sub("-", (model_id or "model").split("/")[-1])
    ts = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{model}-tp{tp}-{backend.lower()}-{ts}"


@dataclass
class RunPaths:
    root: str
    run_id: str

    @property
    def dir(self) -> str:
        return os.path.join(self.root, self.run_id)

    @property
    def workloads(self) -> str:
        return os.path.join(self.dir, "workloads")

    @property
    def reports(self) -> str:
        return os.path.join(self.dir, "reports")

    @property
    def matrix(self) -> str:
        return os.path.join(self.dir, "matrix.xlsx")

    @property
    def coverage(self) -> str:
        return os.path.join(self.dir, "coverage.json")

    @property
    def targets(self) -> str:
        return os.path.join(self.dir, "opt_targets.json")

    @property
    def merged(self) -> str:
        return os.path.join(self.dir, "reports_merged.xlsx")

    @property
    def run_json(self) -> str:
        return os.path.join(self.dir, "run.json")

    @property
    def cache(self) -> str:
        return os.path.join(self.root, ".cache")

    def ensure(self) -> "RunPaths":
        for d in (self.dir, self.workloads, self.reports, self.cache):
            os.makedirs(d, exist_ok=True)
        return self


def run_paths(run_id: str, base: str | None = None) -> RunPaths:
    return RunPaths(perf_root(base), run_id)


def list_runs(base: str | None = None) -> list[dict[str, Any]]:
    """Known runs, newest first, with whatever metadata they recorded."""
    root = perf_root(base)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root), reverse=True):
        d = os.path.join(root, name)
        if name.startswith(".") or not os.path.isdir(d):
            continue
        meta: dict[str, Any] = {"run_id": name, "dir": d}
        rj = os.path.join(d, "run.json")
        if os.path.isfile(rj):
            try:
                with open(rj) as fh:
                    meta.update(json.load(fh))
            except (OSError, json.JSONDecodeError):
                pass
        meta["has_targets"] = os.path.isfile(os.path.join(d, "opt_targets.json"))
        out.append(meta)
    return out


def _git_commit(path: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def component_commits() -> dict[str, str]:
    """Short git commit of every component that can move a measurement."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    workspace = os.path.dirname(repo)
    out: dict[str, str] = {}
    own = _git_commit(repo)
    if own:
        out["vllm-xpu-breakdown"] = own
    for name in PROVENANCE_REPOS:
        p = os.path.join(workspace, name)
        if os.path.isdir(p):
            c = _git_commit(p)
            if c:
                out[name] = c
    xp = devices.xpu_perf_home()
    if xp and "xpu-perf" not in out:
        c = _git_commit(str(xp))
        if c:
            out["xpu-perf"] = c
    return out


@dataclass
class RunMeta:
    run_id: str
    model_id: str
    backend: str
    dispatch: str
    tp: int
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"))
    sweep: dict[str, Any] = field(default_factory=dict)
    devices: str = ""
    smoke: bool = False
    commits: dict[str, str] = field(default_factory=component_commits)
    notes: str = ""

    def write(self, paths: RunPaths) -> str:
        paths.ensure()
        with open(paths.run_json, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        return paths.run_json


def read_meta(paths: RunPaths) -> dict[str, Any]:
    if not os.path.isfile(paths.run_json):
        return {}
    with open(paths.run_json) as fh:
        return json.load(fh)
