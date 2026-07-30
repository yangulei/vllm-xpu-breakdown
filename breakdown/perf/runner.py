# SPDX-License-Identifier: Apache-2.0
"""Run xpu-perf/micro_perf over emitted workloads, one op per process.

xpu-perf is **invoked, never vendored**: this module shells out to its
``launch.py`` so the upstream project stays a clean dependency (its op schemas
and vendor impls are contributed upstream; the orchestration lives here).

Two hard-won rules are encoded:

* **One op per process.** A single unhandled provider exception aborts an entire
  ``--task all`` run, losing every other op's results, and a bad kernel can take
  the device down with it. Ops are launched separately and failures are
  contained.
* **Persistent kernel caches.** Without ``SYCL_CACHE_PERSISTENT`` /
  ``TRITON_CACHE_DIR`` every run re-pays SYCL AOT and Triton JIT on the first
  case of each op, which dominates a short sweep and pollutes the first
  measurement.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from breakdown.perf import devices, workloads as wl

BACKENDS = ("INTEL", "GPU")

#: micro_perf error lines worth surfacing in the summary
_ERR_RE = re.compile(r"^[A-Za-z_.]*(Error|Exception):")


@dataclass
class OpResult:
    op: str
    group: str
    ok: bool
    cases: int
    seconds: float
    log: str
    error: str = ""
    timeout: int = 0
    failed_cases: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    backend: str
    reports_dir: str
    ops: list[OpResult] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0

    @property
    def failed(self) -> list[OpResult]:
        return [o for o in self.ops if not o.ok]

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reports_dir": self.reports_dir,
            "started": self.started,
            "finished": self.finished,
            "ok": self.ok,
            "ops": [asdict(o) for o in self.ops],
        }


def bench_env(cache_dir: str, base: dict | None = None) -> dict[str, str]:
    """Environment for a micro_perf launch: persistent kernel caches.

    oneAPI itself is sourced by the caller's shell (``setvars.sh`` sets dozens
    of variables and, being written for an interactive shell, reads unset ones -
    sourcing it under ``set -u`` silently kills the calling script).
    """
    env = dict(base if base is not None else os.environ)
    sycl = os.path.join(cache_dir, "sycl")
    triton = os.path.join(cache_dir, "triton")
    os.makedirs(sycl, exist_ok=True)
    os.makedirs(triton, exist_ok=True)
    env.setdefault("SYCL_CACHE_PERSISTENT", "1")
    env.setdefault("SYCL_CACHE_DIR", sycl)
    env.setdefault("TRITON_CACHE_DIR", triton)
    return env


def count_cases(reports_dir: str, op: str) -> int:
    """Benchmarked cases for an op, counted from its report jsonl.

    Counted from the reports rather than the log: micro_perf prints every
    result twice, so grepping the log double-counts.
    """
    n = 0
    for path in glob.glob(os.path.join(reports_dir, "**", "*.jsonl"),
                          recursive=True):
        if f"{os.sep}{op}{os.sep}" not in path:
            continue
        with open(path) as fh:
            n += sum(1 for line in fh if "latency(us)" in line)
    return n


def case_errors(log_text: str) -> tuple[int, list[str]]:
    """``(failed cases, distinct error lines)`` from an op's log.

    micro_perf catches a case's exception and prints the traceback, then
    carries on with the next shape, so a per-case failure must be *reported*
    rather than inferred from the op exiting non-zero. A kernel that rejects a
    legitimate workload (e.g. a GQA group the kernel cannot tile) shows up here
    with the exact shapes that failed, which is the signal worth acting on.
    """
    seen: dict[str, int] = {}
    total = 0
    for line in log_text.splitlines():
        s = line.strip()
        if _ERR_RE.match(s):
            total += 1
            seen[s[:200]] = seen.get(s[:200], 0) + 1
    # micro_perf's probe + measure phases can each raise for the same case
    return total, [f"{msg} (x{n})" if n > 1 else msg
                   for msg, n in sorted(seen.items(), key=lambda kv: -kv[1])]


def _launch_cmd(micro_perf: str, backend: str, devices_arg: str,
                task_dir: str, op: str, reports_dir: str) -> list[str]:
    return ["python3", "launch.py", "--backend", backend,
            "--device", devices_arg, "--task_dir", task_dir,
            "--task", op, "--report_dir", reports_dir]


def run(workloads_dir: str, reports_dir: str, backend: str = "INTEL",
        devices_arg: str = "0", ccl_devices: str | None = None,
        groups: Iterable[str] = wl.GROUPS, tasks: Iterable[str] | None = None,
        timeout: int = 1800, cache_dir: str | None = None,
        micro_perf_dir: str | None = None,
        on_op: Callable[[OpResult], None] | None = None,
        env: dict[str, str] | None = None,
        timeouts: dict[str, int] | None = None) -> RunResult:
    """Benchmark every op of the selected groups, one process per op.

    ``timeouts`` gives a per-op budget (see :mod:`breakdown.perf.estimate`);
    ``timeout`` is the fallback for ops it does not cover.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    mp = micro_perf_dir or (str(devices.micro_perf_dir())
                            if devices.micro_perf_dir() else None)
    if not mp or not os.path.isdir(mp):
        raise FileNotFoundError(
            "xpu-perf/projects/micro_perf not found - set $XPU_PERF_HOME to "
            "the xpu-perf checkout")
    # The launcher parses EVERY json under --task_dir, so reports must not live
    # inside the workload tree.
    if os.path.abspath(reports_dir).startswith(
            os.path.abspath(workloads_dir) + os.sep):
        raise ValueError("reports_dir must not live inside workloads_dir")

    os.makedirs(reports_dir, exist_ok=True)
    log_dir = os.path.join(reports_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_env = bench_env(cache_dir or os.path.join(reports_dir, ".cache"), env)
    result_path = os.path.join(reports_dir, "run_result.json")

    res = RunResult(backend=backend, reports_dir=reports_dir,
                    started=time.time())
    want = set(tasks) if tasks else None
    for group in groups:
        group_dir = os.path.join(workloads_dir, group)
        ops = wl.ops_in(group_dir)
        if want is not None:
            ops = [o for o in ops if o in want]
        devs = ccl_devices or devices_arg if group == "collective" else devices_arg
        for op in ops:
            log_path = os.path.join(log_dir, f"{group}-{op}.log")
            cmd = _launch_cmd(mp, backend, devs, group_dir, op, reports_dir)
            op_timeout = int((timeouts or {}).get(op, timeout))
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=mp, env=run_env,
                                      capture_output=True, text=True,
                                      timeout=op_timeout)
                out = (proc.stdout or "") + (proc.stderr or "")
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                out = (exc.stdout or "") + (exc.stderr or "") if isinstance(
                    exc.stdout, str) else ""
                timed_out = True
            with open(log_path, "w") as fh:
                fh.write(out)
            cases = count_cases(reports_dir, op)
            n_failed, errs = case_errors(out)
            error = f"TIMEOUT after {op_timeout}s" if timed_out else (
                errs[0] if errs else "")
            # A per-case kernel failure is reported, not fatal: the shapes that
            # did run are still the ranking input, and the failing shapes are
            # exactly what the op-level report needs to surface.
            r = OpResult(op=op, group=group,
                         ok=bool(cases and not timed_out),
                         cases=cases, seconds=round(time.time() - t0, 1),
                         log=log_path, error=error, timeout=op_timeout,
                         failed_cases=n_failed, errors=errs[:10])
            res.ops.append(r)
            # Written after every op: micro_perf only flushes its jsonl when an
            # op ends, so a run killed midway must still leave a readable
            # record of what completed.
            _write_result(res, result_path)
            if on_op:
                on_op(r)
    res.finished = time.time()
    _write_result(res, result_path)
    return res


def _write_result(res: RunResult, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res.to_dict(), fh, indent=2)
    os.replace(tmp, path)
