# SPDX-License-Identifier: Apache-2.0
"""Replay collective ops across the ranks they were profiled on.

A tensor-parallel all-reduce cannot be measured in one process: with no peers
the call either hangs or returns immediately, and either way the number is
meaningless. So collectives are launched as ``world_size`` peer processes (the
profiled ``TP``), each replaying the same collective on its own device, and
**rank 0's measurement is the one recorded** - exactly the convention the
profile itself uses, because ranks 1..N-1 additionally absorb the wait to
synchronize with rank 0 and would report inflated times.

The dispatch names are mapped to the ``torch.distributed`` entry points rather
than invoked through ``torch.ops.c10d`` directly: the raw ops take a
``ProcessGroup`` *TorchScript class* argument, which cannot be materialized
from a trace slot.

Launched by :mod:`breakdown.bench.runner`, or standalone::

    python -m breakdown.bench.collective --cases cases.json --out results.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable

from breakdown.bench import timing
from breakdown.bench.spec import BenchCase, shape_key
from breakdown.bench.types import case_record

#: dispatch name -> ``torch.distributed`` function name. Aliases included
#: because the trace records whichever wrapper the backend dispatched.
COLLECTIVE_FN: dict[str, str] = {
    "allreduce_": "all_reduce",
    "allreduce": "all_reduce",
    "allreduce_coalesced_": "all_reduce",
    "all_reduce": "all_reduce",
    "_allgather_base_": "all_gather_into_tensor",
    "allgather_": "all_gather_into_tensor",
    "allgather_into_tensor_coalesced_": "all_gather_into_tensor",
    "all_gather": "all_gather_into_tensor",
    "_reduce_scatter_base_": "reduce_scatter_tensor",
    "reduce_scatter_": "reduce_scatter_tensor",
    "reduce_scatter_tensor_coalesced_": "reduce_scatter_tensor",
    "reduce_scatter": "reduce_scatter_tensor",
    "alltoall_": "all_to_all_single",
    "alltoall_base_": "all_to_all_single",
    "all_to_all": "all_to_all_single",
}

#: Collective replay uses a **fixed** iteration schedule, identical on every
#: rank. A per-rank adaptive schedule (the probe the single-device path uses)
#: makes ranks issue different numbers of collective calls, which does not
#: merely skew the number - the ranks desynchronize and the transport runs out
#: of resources mid-run.
COLLECTIVE_REPS = 10
COLLECTIVE_WINDOWS = 5
COLLECTIVE_WARMUP = 3

#: torch.distributed backend per device kind.
BACKENDS = {"xpu": "xccl", "cuda": "nccl", "cpu": "gloo"}

#: How long a rank waits to form the group before giving up. Long enough for
#: three peers to import torch on a cold page cache, short enough that a rank
#: that will never arrive is reported instead of holding a GPU.
PG_TIMEOUT_S = 300

#: How many rendezvous a collective gets before it is reported as failed. See
#: :func:`launch` - the XCCL transport deadlocks intermittently, so one attempt
#: is not evidence that the op cannot be measured.
LAUNCH_ATTEMPTS = 3


def collective_fn_name(op: str) -> str | None:
    name = op.split("::")[-1]
    return COLLECTIVE_FN.get(name)


def _tensors(case: BenchCase) -> list[dict]:
    out = []
    for a in case.args:
        if a.get("kind") == "tensor":
            out.append(a)
        elif a.get("kind") == "tensorlist":
            out.extend(a.get("items") or [])
    return out


def build_call(case: BenchCase, device: str) -> tuple[Callable[[], Any], list]:
    """``(callable, tensors to keep alive)`` for one collective case."""
    import torch
    import torch.distributed as dist

    from breakdown.bench.inputs import torch_dtype

    fn_name = collective_fn_name(case.op)
    if fn_name is None:
        raise ValueError(f"no torch.distributed mapping for {case.op}")
    ts = _tensors(case)
    if not ts:
        raise ValueError(f"{case.op}: no tensor operands recorded")

    def alloc(spec: dict):
        dims = [int(d) for d in spec.get("dims") or []]
        dt = torch_dtype(spec.get("dtype") or "bfloat16")
        if dt.is_floating_point:
            return (torch.randn(dims, device=device, dtype=torch.float32)
                    * 0.1).to(dt)
        return torch.zeros(dims, device=device, dtype=dt)

    if fn_name == "all_reduce":
        t = alloc(ts[0])
        return (lambda: dist.all_reduce(t)), [t]

    # gather/scatter/all-to-all: the trace records both the big and the small
    # buffer; which is output vs input follows from the collective's direction.
    big, small = sorted(ts, key=lambda s: -_numel(s))[:2] if len(ts) > 1 \
        else (ts[0], ts[0])
    if fn_name == "all_gather_into_tensor":
        out_t, in_t = alloc(big), alloc(small)
        return (lambda: dist.all_gather_into_tensor(out_t, in_t)), [out_t, in_t]
    if fn_name == "reduce_scatter_tensor":
        out_t, in_t = alloc(small), alloc(big)
        return (lambda: dist.reduce_scatter_tensor(out_t, in_t)), [out_t, in_t]
    out_t, in_t = alloc(big), alloc(big)
    return (lambda: dist.all_to_all_single(out_t, in_t)), [out_t, in_t]


def _numel(spec: dict) -> int:
    n = 1
    for d in spec.get("dims") or []:
        n *= max(int(d), 1)
    return n


def _init_process_group(dist, torch, kind: str, rank: int, world: int) -> None:
    """Form the process group with this rank's device **named**.

    Without ``device_id`` the XCCL/NCCL backend has to guess which GPU the rank
    owns, because at ``init`` time the process has not touched the device yet
    (``set_device`` alone does not create a context). It logs
    ``"using GPU 0 as device used by this process is currently unknown ... can
    potentially cause a hang"`` and then a later ``barrier()`` runs on whatever
    device the current context happens to be - which is exactly the hang that
    was observed: the four ranks of the MiniMax-M3 TP=4 all-reduce would
    intermittently never return, the per-op timeout would fire, and the run
    died with ``TIMEOUT`` and no measurement. Naming the device makes the
    mapping explicit, so the group and every collective on it run on
    ``kind:rank``.

    ``PG_TIMEOUT_S`` bounds the rendezvous itself: a rank that cannot join is a
    failure to report, not a process to leave sitting on a GPU until the
    runner's wall clock kills it.
    """
    kwargs: dict[str, Any] = {
        "backend": BACKENDS.get(kind, "gloo"),
        "rank": rank,
        "world_size": world,
        "timeout": datetime.timedelta(seconds=PG_TIMEOUT_S),
    }
    if kind != "cpu" and getattr(torch, kind, None) is not None:
        kwargs["device_id"] = torch.device(kind, rank)
    try:
        dist.init_process_group(**kwargs)
    except (TypeError, ValueError):
        # Older torch builds do not accept device_id; the mapping is then only
        # as good as set_device(), which is what this function exists to avoid.
        kwargs.pop("device_id", None)
        dist.init_process_group(**kwargs)


def _sync_device(torch, kind: str) -> None:
    mod = getattr(torch, kind, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def rank_main(argv: list[str] | None = None) -> int:
    """One rank of a collective replay."""
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--op", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--budget", type=float, default=0.5)
    a = p.parse_args(argv)

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dev_mod = getattr(torch, a.device, None)
    if dev_mod is not None and hasattr(dev_mod, "set_device"):
        dev_mod.set_device(rank)
    device = f"{a.device}:{rank}"
    _init_process_group(dist, torch, a.device, rank, world)
    try:
        # The same warm-up vLLM's XPU worker performs right after it forms its
        # group: oneCCL builds its transport lazily, on the first collective.
        # Doing it here keeps that one-time cost out of the first case's
        # warm-up window, where it would be charged to the measurement.
        try:
            dist.all_reduce(torch.zeros(1, device=device))
            _sync_device(torch, a.device)
        except Exception:                           # noqa: BLE001
            pass
        with open(a.cases) as fh:
            raw = json.load(fh)
        cases = [BenchCase.from_dict(c) for c in raw
                 if c.get("op") == a.op and int(c.get("tp") or 1) == world]
        out_fh = open(a.out, "a") if rank == 0 else None
        for case in cases:
            try:
                call, keep = build_call(case, device)
            except Exception as exc:                # noqa: BLE001
                if out_fh:
                    out_fh.write(json.dumps(_rec(case, "arg_error", world,
                                                 error=str(exc))) + "\n")
                continue
            dist.barrier()
            m = timing.measure(lambda: call(), [], a.device.split(":")[0],
                               reps=COLLECTIVE_REPS,
                               windows=COLLECTIVE_WINDOWS,
                               warmup=COLLECTIVE_WARMUP, flush_cache=False)
            dist.barrier()
            del keep
            if out_fh:
                status = "ok" if m.ok else "failed"
                out_fh.write(json.dumps(
                    _rec(case, status, world, m=m, error=m.error)) + "\n")
                out_fh.flush()
        if out_fh:
            out_fh.close()
    finally:
        dist.destroy_process_group()
    return 0


def _rec(case: BenchCase, status: str, world: int,
         m: timing.Measurement | None = None, error: str = "") -> dict[str, Any]:
    """Rank 0's record. Ranks 1..N-1 absorb the wait to synchronize with it, so
    their latency is inflated; only rank 0's is recorded."""
    return case_record(case, status, shape_key(case.op, case.args),
                       measurement=m, error=error,
                       detail="rank 0 measurement", world_size=world)



def launch(op: str, world_size: int, cases_path: str, out_path: str,
           device: str, budget: float, timeout: int,
           env: dict[str, str] | None = None,
           port: int = 29591,
           attempts: int = LAUNCH_ATTEMPTS) -> tuple[bool, str]:
    """Start ``world_size`` peer ranks for one collective op, and retry a hang.

    The retry is not defensive programming, it is a measured property of the
    transport. A four-rank XCCL all-reduce on PCIe-connected Battlemage cards
    intermittently deadlocks *inside the device queue*: every rank enqueues its
    collectives and then all four block forever in ``torch.xpu.synchronize``.
    It reproduces in twenty lines of plain torch with none of this code
    involved, at roughly one attempt in three, and no oneCCL setting tried made
    it go away - so it cannot be configured out here, only survived. A fresh
    group succeeds where the wedged one did not, which is why each attempt gets
    a **new port**: a killed rank can leave the previous rendezvous port bound.

    Without this a single unlucky attempt ended the whole benchmark run: the
    op timed out, and the twenty-odd ops planned after it were never measured.

    Each attempt writes to its **own** file, which is then merged case by case.
    A hung attempt is rarely empty - rank 0 streams a record as each case
    finishes, so it typically measured the small shapes and wedged on a large
    one - and appending the next attempt's output directly would record those
    cases twice. The op's latency is then averaged over duplicate rows and its
    rank is wrong, which is worse than the timeout this is fixing. Merging by
    ``case_id`` also means a retry only has to succeed at what is still
    missing: once every expected case has a record the launch has *succeeded*,
    whether or not the final attempt exited cleanly.
    """
    want = _expected_case_ids(cases_path, op, world_size)
    last = ""
    for attempt in range(max(attempts, 1)):
        tmp = f"{out_path}.attempt{attempt}"
        _unlink(tmp)
        ok, out = _launch_once(op, world_size, cases_path, tmp, device,
                               budget, timeout, env, port + attempt * 8)
        _merge_records(tmp, out_path)
        _unlink(tmp)
        if ok or (want and want <= _case_ids(out_path)):
            return True, (f"[attempt {attempt + 1}]\n" + out) if attempt else out
        last = f"[attempt {attempt + 1}/{attempts}]\n{out}\n{last}"
    return False, last


def _expected_case_ids(cases_path: str, op: str, world: int) -> set[str]:
    """The cases this launch is responsible for measuring."""
    try:
        with open(cases_path) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return set()
    return {cid for cid in
            (BenchCase.from_dict(c).case_id for c in raw
             if c.get("op") == op and int(c.get("tp") or 1) == world) if cid}


def _case_ids(path: str) -> set[str]:
    return {_rec_key(r) for r in _read_records(path)}


def _rec_key(rec: dict[str, Any]) -> str:
    """Identity of a result row across attempts.

    ``case_id`` when the planner assigned one, otherwise the operating point
    itself. Keying on ``case_id`` alone would silently *drop* a record without
    one, which is the opposite of what the merge is for.
    """
    cid = rec.get("case_id")
    if cid:
        return str(cid)
    return "|".join(str(rec.get(k, "")) for k in
                    ("op", "shape", "phase", "seq_len", "ctx_len",
                     "batch_size", "tp"))


def _read_records(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _merge_records(src: str, dst: str) -> None:
    """Append ``src``'s records for cases ``dst`` does not already have."""
    have = _case_ids(dst)
    new = []
    for rec in _read_records(src):
        key = _rec_key(rec)
        if key in have:
            continue
        have.add(key)
        new.append(rec)
    if not new:
        return
    with open(dst, "a") as fh:
        for rec in new:
            fh.write(json.dumps(rec) + "\n")


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _rank_env(base: dict[str, str], rank: int, world_size: int) -> dict[str, str]:
    """One rank's environment.

    ``LOCAL_RANK``/``LOCAL_WORLD_SIZE`` are what oneCCL reads to work out how
    many ranks share the node; without them it logs *"could not get
    local_idx/count from environment variables"* and has to infer the topology
    through ATL. vLLM's own XPU worker sets exactly these two plus
    ``CCL_ATL_TRANSPORT`` before it forms its process group, and the replay has
    to stand in for that worker faithfully - it is measuring the same
    collective. ``CCL_LOCAL_RANK``/``CCL_LOCAL_SIZE`` are the same facts under
    the names oneCCL itself documents.
    """
    e = dict(base)
    e["RANK"] = str(rank)
    e["LOCAL_RANK"] = str(rank)
    e["LOCAL_WORLD_SIZE"] = str(world_size)
    e["CCL_LOCAL_RANK"] = str(rank)
    e["CCL_LOCAL_SIZE"] = str(world_size)
    e.setdefault("CCL_ATL_TRANSPORT", "ofi")
    # Per-rank Triton cache: concurrent JIT into one directory races.
    if e.get("TRITON_CACHE_DIR"):
        e["TRITON_CACHE_DIR"] = os.path.join(e["TRITON_CACHE_DIR"],
                                             f"rank{rank}")
        os.makedirs(e["TRITON_CACHE_DIR"], exist_ok=True)
    return e


def _launch_once(op: str, world_size: int, cases_path: str, out_path: str,
                 device: str, budget: float, timeout: int,
                 env: dict[str, str] | None, port: int) -> tuple[bool, str]:
    """One rendezvous: ``world_size`` peer ranks, or the reason they failed."""
    base = dict(env or os.environ)
    # oneCCL segfaults (SIGSEGV, no Python traceback) when the ranks run with a
    # persistent SYCL kernel cache: several ranks compile the collective's
    # kernels into the same cache concurrently and the runtime dies. Per-rank
    # cache directories are not enough - the variable itself is the trigger -
    # so collectives run with the SYCL cache disabled. They compile a handful of
    # kernels, so the cache buys them almost nothing anyway.
    for var in ("SYCL_CACHE_PERSISTENT", "SYCL_CACHE_DIR"):
        base.pop(var, None)
    base.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                 "WORLD_SIZE": str(world_size)})
    procs = []
    for rank in range(world_size):
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "breakdown.bench.collective",
             "--cases", cases_path, "--op", op, "--out", out_path,
             "--device", device, "--budget", str(budget)],
            env=_rank_env(base, rank, world_size),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True))
    deadline = time.time() + timeout
    logs = []
    timed_out = False
    for pr in procs:
        remaining = max(deadline - time.time(), 1)
        try:
            out, _ = pr.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            # Killing is not enough: the output buffered before the hang is the
            # only evidence of *where* it hung, and it is still in the pipe.
            # The old code replaced it with the string "TIMEOUT", so a hung
            # collective reported nothing at all.
            timed_out = True
            _kill_group(pr)
            try:
                out, _ = pr.communicate(timeout=30)
            except subprocess.TimeoutExpired:      # pragma: no cover - wedged
                out = ""
            out = (out or "") + f"\n[TIMEOUT after {timeout}s]"
        logs.append(out or "")
    ok = all(pr.returncode == 0 for pr in procs) and not timed_out
    codes = ", ".join(f"rank{i}={pr.returncode}" for i, pr in enumerate(procs))
    return ok, f"[exit codes] {codes}\n" + "\n".join(logs)


def _kill_group(pr: subprocess.Popen) -> None:
    """Kill a wedged rank *and anything it spawned*.

    Each rank runs in its own session (``start_new_session``) so the whole
    group can be signalled: a rank deadlocked in the driver can leave helper
    threads and processes holding the GPU, and the next attempt then fails for
    a second, unrelated reason.
    """
    try:
        os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pr.kill()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(rank_main())
