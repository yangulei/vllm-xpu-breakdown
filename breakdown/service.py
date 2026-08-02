# SPDX-License-Identifier: Apache-2.0
"""The stage orchestration the web routes and the CLIs both call.

A route should decide *what the request asked for* and *what to answer with*;
everything between - which devices a run may use, how a benchmark is launched
in the background, which shape matrix belongs to a run, which ranking a
session should be briefed from - is the same work whether it was asked for
over HTTP or from ``python -m breakdown.bench``. It lives here so there is one
implementation of each, and so ``app.py`` is routes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict
from typing import Any

from breakdown import profiling, shape_matrix, shape_matrix_xlsx
from breakdown.shape_derive import _bytes_to_dtype
from breakdown.bench import devices as bench_devices
from breakdown.bench import rank as bench_rank
from breakdown.bench import runner as bench_runner
from breakdown.bench import spec as bench_spec
from breakdown.bench import store as bench_store
from breakdown.optimize import session as optimize_session
from breakdown.optimize.manager import MANAGER as OPTIMIZE_MANAGER

logger = logging.getLogger("vllm_xpu_breakdown")

#: The current (or last) benchmark job. Benchmark runs are long, so the web
#: layer starts one in a thread and polls it; the CLI runs it in the
#: foreground. Both go through the same runner.
_bench_state: dict[str, Any] = {
    "status": "idle",     # idle | running | done | error
    "run_id": None,
    "error": None,
    "ops": [],            # per-op progress
    "result": None,
}
_bench_lock = threading.Lock()

def _bench_device_ids(data: dict, device: str, need: int = 0
                      ) -> tuple[bool, list[int], str | None]:
    """``(ok, device_ids, error)`` for a request's device selection.

    A selection that names a device this host does not have, or fewer devices
    than the widest collective in the sweep needs, is refused before a run
    starts - the failure is otherwise a driver error inside a worker.
    """
    try:
        ids = bench_devices.parse_device_ids(data.get("device_ids"))
    except ValueError as exc:
        return False, [], str(exc)
    err = bench_devices.validate_device_ids(ids, device, need=need)
    return (err is None), ids, err


def _run_bench(run_id: str, device: str, budget: float | None,
               ops: list[str] | None, device_ids: list[int] | None = None
               ) -> None:
    paths = bench_store.run_paths(run_id)
    try:
        cases = [bench_spec.BenchCase.from_dict(c)
                 for c in json.load(open(paths.cases))]

        def progress(op_result):
            with _bench_lock:
                _bench_state["ops"].append(asdict(op_result))

        # Device visibility is inherited by the worker processes: both runtimes
        # read it at driver init, so pinning the selection here is what makes
        # the choice effective (including for a collective's peer ranks).
        env = {**os.environ,
               **bench_devices.visibility_env(device, device_ids or [])}
        result = bench_runner.run(cases, paths, device, budget=budget, ops=ops,
                                  env=env, on_op=progress)
        with _bench_lock:
            _bench_state["status"] = "done"
            _bench_state["result"] = result.to_dict()
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.exception("bench run failed")
        with _bench_lock:
            _bench_state["status"] = "error"
            _bench_state["error"] = str(exc)


def _bench_matrix(paths) -> dict | None:
    """A run's Shape Matrix rows for the report workbook.

    Runs planned after the matrix/benchmark merge persist ``rows.json`` at plan
    time. Older runs predate it, so fall back to rebuilding the rows from the
    sweep the plan recorded plus the current profile - correct whenever the
    profile still matches, and simply absent (no sheet) when it does not.
    """
    if os.path.isfile(paths.rows):
        try:
            with open(paths.rows) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None
    meta = bench_store.read_meta(paths)
    model_id = meta.get("model_id")
    configs = None
    if os.path.isfile(paths.plan):
        try:
            with open(paths.plan) as fh:
                configs = (json.load(fh) or {}).get("sweep")
        except (OSError, ValueError):
            configs = None
    if not model_id or not configs:
        return None
    template, settings, err = profiling._profile_template_for(model_id, None)
    if err:
        return None
    return {"model_id": model_id,
            "info": shape_matrix.build_info_rows(model_id, template, settings),
            "rows": shape_matrix.build_rows(template, configs)}


# ---- Optimize Kernels: hand a ranked target to a Copilot CLI session ----
# The benchmark answers *which* kernel is worth a session; these endpoints open
# it. One session owns one GPU exclusively (see breakdown/optimize/scheduler.py),
# so more selected kernels than devices simply queue.

def optimize_doc(run_id: str) -> tuple[dict | None, str | None]:
    """A run's ranked targets, or why they cannot be loaded."""
    if not run_id:
        return None, "run_id is required"
    paths = bench_store.run_paths(run_id)
    if not os.path.isfile(paths.targets):
        return None, (f"run '{run_id}' has no ranked targets yet - "
                      "run Bench & Rank first")
    try:
        with open(paths.targets) as fh:
            return json.load(fh), None
    except (OSError, ValueError) as exc:
        return None, f"could not read the ranking: {exc}"


def optimize_sessions(run_id: str) -> list[dict]:
    """A run's sessions as JSON, live ones first and the on-disk index after.

    ``argv`` is dropped: it embeds the whole multi-KB brief, and every endpoint
    that returns sessions must return the same shape.

    A session read back from ``index.json`` belongs to a *previous* server
    process, whose agents the atexit hook terminated - reporting one as
    ``running`` makes the UI poll forever for a process that is gone, so it is
    downgraded to ``stopped`` with that reason.
    """
    out = []
    for sess in OPTIMIZE_MANAGER.sessions(run_id or None):
        data = sess.to_dict()
        data.pop("argv", None)
        out.append(data)
    if out or not run_id:
        return out
    index = optimize_session.session_paths(run_id, "_index").index
    if not os.path.isfile(index):
        return out
    try:
        with open(index) as fh:
            out = json.load(fh).get("sessions", [])
    except (OSError, ValueError):
        return []
    for sess in out:
        sess.pop("argv", None)
        if sess.get("state") in ("pending", "running"):
            sess["state"] = "stopped"
            sess["error"] = (sess.get("error")
                             or "the server restarted while this session was "
                                "open, so its agent was terminated")
    return out


def shape_matrix_workbook(data: dict) -> tuple[bytes | None, str, str]:
    """The Shape Matrix workbook for a sweep request: ``(payload, name, err)``.

    The op set and the real shapes come from the latest completed profiling run
    for this model - its reconstructed graph is the symbolic template - and
    each swept configuration re-resolves S/B/C/TP and recomputes Memory/FLOPs
    from the resolved shapes. The op set is fixed at the profiled config (TP
    collectives, MoE routing, chunked-prefill splits), so a caller profiles at
    each TP it needs; S/B/C are parametric from one profile.
    """
    model_id = data.get("model_id")
    if not model_id:
        return None, "", "No model_id specified"

    sweep = {k: data.get(k, v) for k, v in shape_matrix.DEFAULT_SWEEP.items()}
    for key in ("prefill_seq_lens", "tp_sizes", "decode_ctx_lens",
                "decode_batch_sizes"):
        if not isinstance(sweep[key], list) or not sweep[key]:
            return None, "", f"{key} must be a non-empty list"

    configs = shape_matrix.build_configs(**sweep)
    if not configs:
        return None, "", "No configurations generated."

    template, profile_settings, err = profiling._profile_template_for(
        model_id, data.get("quantization"))
    if err:
        return None, "", err

    estimated = shape_matrix.estimate_row_count(template, configs)
    if estimated > shape_matrix.MAX_MATRIX_ROWS:
        return None, "", (
            f"Too many rows ({estimated}). Max is "
            f"{shape_matrix.MAX_MATRIX_ROWS}. Reduce seq_lens, batch_sizes, "
            f"ctx_lens, or tp_sizes.")

    rows = shape_matrix.build_rows(template, configs)
    info_rows = shape_matrix.build_info_rows(model_id, template,
                                             profile_settings)
    payload = shape_matrix_xlsx.write_workbook(
        rows, info_rows, shape_matrix_xlsx.sheet_name_for(model_id))

    # Tag with the quantization method, or the model's concrete activation
    # dtype (bf16/fp16) when the run is unquantized - never a bare "none".
    cfg = template.get("config", {})
    tag = cfg.get("quantization") or _bytes_to_dtype(cfg.get("dtype_bytes", 2))
    name = f"vllm_xpu_shape_matrix_{model_id.replace('/', '_')}_{tag}.xlsx"
    return payload, name, ""
