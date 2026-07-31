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
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable

from breakdown.bench import timing
from breakdown.bench.spec import BenchCase, shape_key

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
    dist.init_process_group(backend=BACKENDS.get(a.device, "gloo"),
                            rank=rank, world_size=world)
    try:
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
    rec = {
        "case_id": case.case_id, "op": case.op,
        "shape_key": shape_key(case.op, case.args), "shape": case.shape_label,
        "status": status, "device": case.device, "phase": case.phase,
        "seq_len": case.seq_len, "ctx_len": case.ctx_len,
        "batch_size": case.batch_size, "points": case.points,
        "tp": case.tp, "world_size": world,
        "module": case.module, "role": case.role, "backend": case.backend,
        "layers": case.layers, "flops": case.flops, "bytes": case.nbytes,
        "traced_device_time_us": case.traced_device_time_us,
        "traced_comparable": case.traced_comparable, "error": error,
        "detail": "rank 0 measurement",
    }
    if m is not None:
        rec.update({"latency_us": m.latency_us, "mean_us": m.mean_us,
                    "min_us": m.min_us, "p10_us": m.p10_us, "p90_us": m.p90_us,
                    "stdev_us": m.stdev_us, "iters": m.iters, "reps": m.reps,
                    "windows": m.windows, "overhead_us": m.overhead_us,
                    "notes": m.notes})
    return rec


def launch(op: str, world_size: int, cases_path: str, out_path: str,
           device: str, budget: float, timeout: int,
           env: dict[str, str] | None = None,
           port: int = 29591) -> tuple[bool, str]:
    """Start ``world_size`` peer ranks for one collective op."""
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
        e = dict(base)
        e["RANK"] = str(rank)
        e["LOCAL_RANK"] = str(rank)
        # Per-rank Triton cache: concurrent JIT into one directory races.
        if e.get("TRITON_CACHE_DIR"):
            e["TRITON_CACHE_DIR"] = os.path.join(e["TRITON_CACHE_DIR"],
                                                 f"rank{rank}")
            os.makedirs(e["TRITON_CACHE_DIR"], exist_ok=True)
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "breakdown.bench.collective",
             "--cases", cases_path, "--op", op, "--out", out_path,
             "--device", device, "--budget", str(budget)],
            env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True))
    deadline = time.time() + timeout
    logs = []
    for pr in procs:
        remaining = max(deadline - time.time(), 1)
        try:
            out, _ = pr.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            pr.kill()
            out = "TIMEOUT"
        logs.append(out or "")
    ok = all(pr.returncode == 0 for pr in procs)
    codes = ", ".join(f"rank{i}={pr.returncode}" for i, pr in enumerate(procs))
    return ok, f"[exit codes] {codes}\n" + "\n".join(logs)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(rank_main())
