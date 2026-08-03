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
from typing import Any
from dataclasses import asdict

from breakdown import profiling, runs, shape_matrix, shape_matrix_xlsx
from breakdown.shape_derive import _bytes_to_dtype
from breakdown.core import devices as bench_devices
from breakdown.bench import history as bench_history
from breakdown.bench import rank as bench_rank
from breakdown.bench import resolve as bench_resolve
from breakdown.bench import runner as bench_runner
from breakdown.bench import spec as bench_spec
from breakdown.bench import store as bench_store
from breakdown.optimize import session as optimize_session
from breakdown.optimize.manager import MANAGER as OPTIMIZE_MANAGER

logger = logging.getLogger("vllm_xpu_breakdown")

#: The current (or last) benchmark job. Benchmark runs are long, so the web
#: layer starts one in a thread and polls it; the CLI runs it in the
#: foreground. Both go through the same runner.
_bench_state = runs.RunState(ops=[], result=None)

#: Kept as a name because the routes take it around the state they mutate.
_bench_lock = _bench_state.lock

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
        _bench_state.finish(result=result.to_dict())
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        logger.exception("bench run failed")
        _bench_state.fail(exc)


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


def plan(data: dict) -> tuple[dict, int]:
    """Sweep the profiled graph into a new benchmark run's replay cases.

    ``(payload, http_status)``. The whole body used to sit in the Flask route,
    which meant the CLI reached the same artifacts by a second, slightly
    different path -- and the two could drift without a test noticing. One
    function, two callers.
    """
    model_id = data.get("model_id")
    if not model_id:
        return {"ok": False, "error": "No model_id specified"}, 400

    template, settings, err = profiling._profile_template_for(
        model_id, data.get("quantization"))
    if err:
        return {"ok": False, "error": err}, 400

    sweep = {k: data.get(k, v) for k, v in shape_matrix.DEFAULT_SWEEP.items()}
    tp_sizes = sweep.get("tp_sizes") or [1]
    configs = shape_matrix.build_configs(**sweep)
    if shape_matrix.estimate_row_count(template, configs) > \
            shape_matrix.MAX_MATRIX_ROWS:
        return {"ok": False, "error": "Sweep too large - reduce it"}, 400

    rows = shape_matrix.build_rows(template, configs)
    device = data.get("device") or bench_devices.detect_device()
    # The largest swept TP is the widest collective the run will replay, so the
    # selection must contain at least that many devices.
    ok, device_ids, dev_err = _bench_device_ids(
        data, device, need=max(int(t) for t in tp_sizes))
    if not ok:
        return {"ok": False, "error": dev_err}, 400

    cases, coverage = bench_spec.build_cases(rows, device=device)
    run_id = data.get("run_id") or bench_store.make_run_id(
        model_id, int(tp_sizes[0]), device)
    paths = bench_store.run_paths(run_id).ensure()
    bench_runner.write_cases(cases, paths.cases)
    # The Shape Matrix is the run's input, not a separate artifact: persist the
    # swept rows so the report workbook can carry the shape space alongside the
    # measurements taken on it.
    bench_store.write_json(paths.rows, {
        "model_id": model_id,
        "quantization": data.get("quantization"),
        "info": shape_matrix.build_info_rows(model_id, template, settings),
        "rows": rows,
    })

    # Classify before benchmarking, so the plan can say what will *not* be
    # measured - and why - instead of the run silently omitting it.
    status: dict[str, Any] = {}
    for case in cases:
        if case.op not in status:
            st, detail = bench_resolve.classify(case.op, case.args,
                                                launch=case.launch)
            status[case.op] = {"status": st, "detail": detail,
                               "backend": case.backend}
    by_status: dict[str, list] = {}
    for op, st in status.items():
        by_status.setdefault(st["status"], []).append(op)
    coverage.update({"op_status": status, "ops_by_status": by_status,
                     "device": device, "device_ids": device_ids,
                     "sweep": configs})
    bench_store.write_json(paths.plan, coverage)
    name = bench_devices.device_name(device)
    bench_store.RunMeta(
        run_id=run_id, model_id=model_id, device=device, tp=int(tp_sizes[0]),
        device_name=name, device_ids=device_ids,
        sku=bench_devices.sku_for_device(name),
        smoke=bool(data.get("smoke")),
        sweep={**sweep, "configs": len(configs)}).write(paths)

    return ({"ok": True, "run_id": run_id, "rows": len(rows),
             "cases": len(cases), "coverage": coverage}, 200)


def targets(run_id: str | None, *, refresh: bool = False,
            target_util: float | None = None, top: int = 0
            ) -> tuple[dict, int]:
    """A run's ranked targets, ranking them if needed.

    ``targets.json`` is written once and served from disk afterwards, because
    ranking a large run is not free and the UI polls this. ``refresh`` re-ranks
    -- which is what you do after changing a tolerance.
    """
    if not run_id:
        runs_ = bench_store.list_runs()
        run_id = runs_[0]["run_id"] if runs_ else None
    if not run_id:
        return {"ok": False, "error": "no benchmark runs yet"}, 404

    paths = bench_store.run_paths(run_id)
    meta = bench_store.read_meta(paths)
    if not refresh and os.path.isfile(paths.targets):
        with open(paths.targets) as fh:
            return {"ok": True, "run_id": run_id, "targets": json.load(fh)}, 200

    records = bench_store.read_results(paths.results)
    if not records:
        return ({"ok": False,
                 "error": f"run {run_id} has no benchmark results"}, 400)

    rc = bench_rank.RankConfig(
        target_util=(bench_rank.DEFAULT_TARGET_UTIL if target_util is None
                     else float(target_util)),
        tp=meta.get("tp"), top=int(top), run_id=run_id,
        provenance={"run_id": run_id, "commits": meta.get("commits") or {}})
    try:
        doc = bench_rank.rank(records, rc)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}, 400
    bench_store.write_json(paths.targets, doc)

    # History is best-effort: a regression database that cannot be written is
    # not a reason to fail the ranking the caller asked for.
    try:
        conn = bench_history.connect(
            bench_history.db_path(bench_store.bench_root()))
        bench_history.ingest(conn, meta or {"run_id": run_id}, records, doc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bench history ingest failed: %s", exc)
    return {"ok": True, "run_id": run_id, "targets": doc}, 200
