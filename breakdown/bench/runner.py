# SPDX-License-Identifier: Apache-2.0
"""Orchestrate a replay run: one worker process per op.

Process isolation is the load-bearing decision. A replayed kernel runs with
synthesized operands, so a shape or index the kernel cannot handle does not
merely raise - it can abort the process (``TORCH_CHECK``), or wedge the device
so that *every subsequent op in that process* fails with a device-lost error.
Isolating each op means one bad kernel costs one op's results, and the failure
is reported with the shapes that caused it (which is exactly the signal worth
acting on) instead of silently corrupting the rest of the sweep.

``run_result.json`` is rewritten after every op for the same reason: a run
killed midway must still say what completed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

from breakdown.bench import estimate, resolve, store
from breakdown.bench.spec import BenchCase, group_by_op
from breakdown.bench.worker import bench_env


@dataclass
class OpResult:
    op: str
    ok: bool = False
    cases: int = 0
    measured: int = 0
    failed: int = 0
    seconds: float = 0.0
    timeout: int = 0
    timed_out: bool = False
    log: str = ""
    error: str = ""
    statuses: dict[str, int] = field(default_factory=dict)


@dataclass
class RunResult:
    device: str
    run_dir: str
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
            "device": self.device, "run_dir": self.run_dir,
            "started": self.started, "finished": self.finished,
            "ok": self.ok, "ops": [asdict(o) for o in self.ops],
        }


def write_cases(cases: list[BenchCase], path: str) -> str:
    return store.write_json(path, [c.to_dict() for c in cases])


def _worker_cmd(cases_path: str, op: str, out_path: str, device: str,
                budget: float, flush_cache: bool) -> list[str]:
    cmd = [sys.executable, "-m", "breakdown.bench.worker",
           "--cases", cases_path, "--op", op, "--out", out_path,
           "--device", device, "--budget", str(budget)]
    if not flush_cache:
        cmd.append("--no-flush-cache")
    return cmd


def _statuses(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        s = r.get("status", "?")
        out[s] = out.get(s, 0) + 1
    return out


def run(cases: list[BenchCase], paths: store.RunPaths, device: str,
        budget: float = 0.5, ops: Iterable[str] | None = None,
        flush_cache: bool = True, timeouts: dict[str, int] | None = None,
        on_op: Callable[[OpResult], None] | None = None,
        env: dict[str, str] | None = None,
        in_process: bool = False) -> RunResult:
    """Benchmark every op of ``cases``, one worker process each.

    ``in_process`` runs the cases in this process instead - useful for tests and
    for a single-op debug session, but it gives up the crash containment that
    is the whole point of the subprocess path.
    """
    paths.ensure()
    write_cases(cases, paths.cases)
    grouped = group_by_op(cases)
    want = set(ops) if ops else None
    selected = {op: cs for op, cs in grouped.items()
                if want is None or op in want}

    run_env = bench_env(paths.cache, env)
    # The worker is launched from the repo root so ``breakdown`` is importable.
    # Prepend rather than setdefault: a dev shell usually already exports
    # PYTHONPATH, and using *its* value as the working directory either raises
    # FileNotFoundError (multi-entry) or leaves the repo off the import path.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    inherited = run_env.get("PYTHONPATH")
    run_env["PYTHONPATH"] = (f"{repo_root}{os.pathsep}{inherited}"
                             if inherited else repo_root)
    budgets = timeouts or estimate.plan(
        {op: len(cs) for op, cs in selected.items()}, budget, paths.root,
        alloc_bytes={op: sum(_operand_bytes(c) for c in cs)
                     for op, cs in selected.items()})

    res = RunResult(device=device, run_dir=paths.dir, started=time.time())
    # Re-running a subset of ops must not delete the rest of the run's
    # measurements: only a full run starts from an empty results file, and a
    # partial one drops just the selected ops' previous records.
    if want is None or set(selected) >= set(grouped):
        open(paths.results, "w").close()
    else:
        _drop_records(paths.results, set(selected))

    for op, op_cases in selected.items():
        t0 = time.time()
        log_path = os.path.join(paths.logs, _safe(op) + ".log")
        timeout = int(budgets.get(op, estimate.MIN_TIMEOUT_S))
        before = _count_lines(paths.results)
        timed_out = False
        error = ""
        if resolve.is_collective(op):
            error = _run_collective(op, op_cases, paths, device, budget,
                                    timeout, run_env, log_path)
        elif in_process:
            from breakdown.bench import worker
            try:
                worker.run_op(op_cases, device, paths.results, budget=budget,
                              flush_cache=flush_cache)
            except Exception as exc:              # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
        else:
            cmd = _worker_cmd(paths.cases, op, paths.results, device, budget,
                              flush_cache)
            try:
                proc = subprocess.run(cmd, env=run_env, capture_output=True,
                                      text=True, timeout=timeout,
                                      cwd=repo_root)
                out = (proc.stdout or "") + (proc.stderr or "")
                if proc.returncode not in (0, 2):
                    error = _last_error(out) or f"exit {proc.returncode}"
            except subprocess.TimeoutExpired as exc:
                out = _decode(exc.stdout) + _decode(exc.stderr)
                timed_out = True
                error = f"TIMEOUT after {timeout}s"
            with open(log_path, "w") as fh:
                fh.write(out)

        records = _tail_records(paths.results, before)
        statuses = _statuses(records)
        measured = statuses.get("ok", 0)
        r = OpResult(op=op, ok=bool(measured) and not timed_out,
                     cases=len(op_cases), measured=measured,
                     failed=len(records) - measured,
                     seconds=round(time.time() - t0, 1), timeout=timeout,
                     timed_out=timed_out, log=log_path, error=error,
                     statuses=statuses)
        res.ops.append(r)
        store.write_json(paths.run_result, res.to_dict())
        if on_op:
            on_op(r)

    res.finished = time.time()
    store.write_json(paths.run_result, res.to_dict())
    return res


def _run_collective(op: str, cases: list[BenchCase], paths: store.RunPaths,
                    device: str, budget: float, timeout: int,
                    env: dict[str, str], log_path: str) -> str:
    """Launch peer ranks for a collective, or record why it could not run.

    Refusing to run a collective on fewer devices than it was profiled with is
    deliberate: a 4-rank all-reduce replayed on 1 rank is not a slower
    all-reduce, it is a different operation, and recording it as a measurement
    would corrupt the ranking.
    """
    from breakdown.bench import collective, devices as dev_mod

    worlds = sorted({max(int(c.tp or 1), 1) for c in cases})
    have = dev_mod.device_count(device)
    logs: list[str] = []
    error = ""
    for world in worlds:
        if world <= 1:
            error = "profiled at TP=1: no collective to replay"
            _append(paths.results, [collective._rec(c, "skipped", world,
                                                    error=error)
                                    for c in cases if int(c.tp or 1) == world])
            continue
        if have < world:
            error = (f"needs {world} devices for TP={world}, {have} present")
            _append(paths.results, [collective._rec(c, "needs_ranks", world,
                                                    error=error)
                                    for c in cases if int(c.tp or 1) == world])
            continue
        ok, out = collective.launch(op, world, paths.cases, paths.results,
                                    device, budget, timeout, env)
        logs.append(out)
        if not ok:
            error = _last_error(out) or f"collective launch failed (TP={world})"
    if logs:
        with open(log_path, "w") as fh:
            fh.write("\n".join(logs))
    return error


def _append(path: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with open(path, "a") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _drop_records(path: str, ops: set[str]) -> None:
    """Remove previous records of ``ops``, keeping every other op's results."""
    if not os.path.isfile(path):
        return
    kept = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("op") in ops:
                    continue
            except ValueError:
                continue
            kept.append(line)
    with open(path, "w") as fh:
        for line in kept:
            fh.write(line + "\n")


def _operand_bytes(case: BenchCase) -> float:
    """Bytes the case's operands allocate, for the worker's time budget."""
    from breakdown.bench.inputs import DTYPE_MAP

    width = {"float64": 8, "int64": 8, "float32": 4, "int32": 4,
             "bfloat16": 2, "float16": 2, "int16": 2}
    total = 0.0
    for t in case.tensor_args:
        n = 1
        for d in t.get("dims") or []:
            n *= max(int(d), 1)
        total += n * width.get(DTYPE_MAP.get((t.get("dtype") or "").lower(),
                                             ""), 1)
    return total


def _safe(op: str) -> str:
    return op.replace("::", "__").replace("/", "_").replace(" ", "_")


def _decode(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v or ""


def _count_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path) as fh:
        return sum(1 for _ in fh)


def _tail_records(path: str, skip: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i < skip or not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _last_error(text: str) -> str:
    """The most informative line of a crashed worker's output."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for line in reversed(lines):
        if any(k in line for k in ("Error", "error", "Aborted", "Fatal",
                                   "TORCH_CHECK", "signal")):
            return line[:300]
    return lines[-1][:300] if lines else ""
