# SPDX-License-Identifier: Apache-2.0
"""On-disk layout for benchmark runs, and their provenance.

Everything a run produces lives under one directory so it is reproducible and
attributable::

    output/bench/<run_id>/
        cases.json          the replay plan (BenchCase list)
        plan.json           coverage: replayable / needs-recipe / skipped
        results.jsonl       one JSON record per benchmarked case
        run_result.json     per-op progress, rewritten after every op
        logs/<op>.log       worker stdout/stderr
        targets.json        ranked optimization targets
        report.xlsx         merged workbook
        run.json            model, sweep, device and component commits
    output/bench/history.sqlite
    output/bench/.cache/{sycl,triton}     persistent kernel caches

``run.json`` records the git commit of **every component that can move a
number** (kernel repos, vLLM, this tool), so a ranking is attributable and a
regression can be bisected to a kernel bump.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Repos whose commit changes measured performance.
PROVENANCE_REPOS = (
    "vllm-xpu-kernels",
    "applications.ai.gpu.vllm-xpu-kernels",
    "applications.ai.gpu.deepklox",
    "sycl-tla",
    "intel-xpu-backend-for-triton",
    "vllm-xpu",
    "vllm",
    "torch-xpu-ops",
)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def bench_root(base: str | None = None) -> str:
    """``output/bench`` (override with ``$BREAKDOWN_BENCH_ROOT``)."""
    if base:
        return base
    env = os.environ.get("BREAKDOWN_BENCH_ROOT")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(repo, "output", "bench")


def make_run_id(model_id: str, tp: int, device: str,
                when: datetime | None = None) -> str:
    model = _SAFE.sub("-", (model_id or "model").split("/")[-1])
    ts = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{model}-tp{tp}-{(device or 'dev').lower()}-{ts}"


@dataclass
class RunPaths:
    root: str
    run_id: str

    @property
    def dir(self) -> str:
        return os.path.join(self.root, self.run_id)

    @property
    def cases(self) -> str:
        return os.path.join(self.dir, "cases.json")

    @property
    def plan(self) -> str:
        return os.path.join(self.dir, "plan.json")

    @property
    def results(self) -> str:
        return os.path.join(self.dir, "results.jsonl")

    @property
    def run_result(self) -> str:
        return os.path.join(self.dir, "run_result.json")

    @property
    def logs(self) -> str:
        return os.path.join(self.dir, "logs")

    @property
    def matrix(self) -> str:
        return os.path.join(self.dir, "matrix.xlsx")

    @property
    def rows(self) -> str:
        return os.path.join(self.dir, "rows.json")

    @property
    def targets(self) -> str:
        return os.path.join(self.dir, "targets.json")

    @property
    def report(self) -> str:
        return os.path.join(self.dir, "report.xlsx")

    @property
    def run_json(self) -> str:
        return os.path.join(self.dir, "run.json")

    @property
    def cache(self) -> str:
        return os.path.join(self.root, ".cache")

    def ensure(self) -> "RunPaths":
        for d in (self.dir, self.logs, self.cache):
            os.makedirs(d, exist_ok=True)
        return self


def run_paths(run_id: str, base: str | None = None) -> RunPaths:
    return RunPaths(bench_root(base), run_id)


def list_runs(base: str | None = None) -> list[dict[str, Any]]:
    """Known runs, newest first, with whatever metadata they recorded."""
    root = bench_root(base)
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
        meta["has_results"] = os.path.isfile(os.path.join(d, "results.jsonl"))
        meta["has_targets"] = os.path.isfile(os.path.join(d, "targets.json"))
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
    return out


@dataclass
class RunMeta:
    run_id: str
    model_id: str
    device: str
    tp: int
    device_name: str = ""
    sku: str = ""
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"))
    sweep: dict[str, Any] = field(default_factory=dict)
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
    try:
        with open(paths.run_json) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: str, obj: Any) -> str:
    """Atomic JSON write (a run killed mid-write must not lose the old file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)
    return path


def read_results(path: str) -> list[dict[str, Any]]:
    """Records from a ``results.jsonl`` (tolerates a truncated last line)."""
    if not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out
