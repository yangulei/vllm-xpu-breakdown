#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
vLLM Ops/Kernels Breakdown — Web Application.

Interactive web UI for profiling vLLM inference on Intel XPU and visualizing
the op dispatch breakdown.

Usage:
    python app.py [--port 8080] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import functools
import importlib
import io
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

# Force spawn for multiprocessing so vLLM's EngineCore doesn't hit
# "Cannot re-initialize CUDA in forked subprocess".
import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breakdown.cost import (
    dtype_size,
    estimate_flops,
    estimate_memory,
)
from breakdown.classifier import Backend, classify_op
from breakdown.op_breakdown import backend_totals, summarize_ops
from breakdown.trace import build_graph_from_trace
from breakdown.trace_common import _detect_device_via_torch
from breakdown import profiling, service
from breakdown.profiling import (
    _build_result_from_traces,
    _CONFIG_CACHE_DIR,
    _load_cached_config,
    _load_cached_model_ids,
    _merge_two_pass_result,
    _norm_quant,
    _parse_trace_filename,
    _profile_lock,
    _profile_template_for,
    _run_profile,
    _save_config_cache,
)
from breakdown.model_info import (
    fetch_model_config,
    min_profile_layers,
    summarize_config,
)
from breakdown.registry import ALL_VLLM_XPU_OPS
from breakdown import shape_matrix, shape_matrix_xlsx
from breakdown.bench import (
    devices as bench_devices,
    history as bench_history,
    rank as bench_rank,
    reports as bench_reports,
    resolve as bench_resolve,
    runner as bench_runner,
    spec as bench_spec,
    store as bench_store,
)
from breakdown.optimize import prompt as optimize_prompt
from breakdown.optimize import session as optimize_session
from breakdown.optimize.manager import MANAGER as OPTIMIZE_MANAGER

app = Flask(__name__, static_folder="static")


# Cached at import time — won't change during server lifetime.
_DEVICE = _detect_device_via_torch() or "xpu"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vllm_xpu_breakdown")

_CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)












# ---- Static file serving ----

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---- Model API ----

@app.route("/api/cached-models")
def get_cached_models():
    """Return list of model IDs whose config.json has been cached locally."""
    return jsonify({"ok": True, "models": _load_cached_model_ids()})


@app.route("/api/model/<path:model_id>")
def get_model_config(model_id: str):
    """Fetch config.json from HuggingFace (or cache) and return summary."""
    try:
        # Try cache first
        config = _load_cached_config(model_id)
        if config is None:
            config = fetch_model_config(model_id)
        # Cache on success
        _save_config_cache(model_id, config)
        summary = summarize_config(config)
        return jsonify({
            "ok": True,
            "config": config,
            "summary": summary,
            "min_profile_layers": min_profile_layers(summary),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


































@app.route("/api/devices")
def list_devices():
    """The accelerators present on this host, for the Device selector.

    The UI selects device *indexes*, so it needs to know which exist; a
    selection is then checked against this list before a profile or a benchmark
    starts.
    """
    return jsonify({"ok": True, **bench_devices.available(_DEVICE)})


@app.route("/api/profile", methods=["POST"])
def start_profile():
    """Start a profiling run. Non-blocking — poll /api/profile/status."""
    
    with _profile_lock:
        if profiling._profile_state["status"] == "running":
            return jsonify({"ok": False, "error": "Profiling already in progress"}), 409

    data = request.json or {}
    model_id = data.get("model_id", "")
    if not model_id:
        return jsonify({"ok": False, "error": "model_id is required"}), 400

    mode = data.get("mode", "eager")
    max_model_len = data.get("max_model_len", 4096)
    batch_size = data.get("batch_size", 1)
    max_tokens = data.get("max_tokens", 128)
    prompt = data.get("prompt", "Write a short essay about AI.")
    num_profile_layers = data.get("num_profile_layers")  # None = all layers
    tp_size = data.get("tensor_parallel_size", 1)
    quantization = data.get("quantization")  # None = no quantization
    gpu_memory_utilization = data.get("gpu_memory_utilization")  # None = vLLM default
    query_len = data.get("query_len")  # new prefill tokens (None = legacy prompt)
    context_len = data.get("context_len")  # cached prefix tokens (None/0 = none)
    # Per-phase batch sizes. When they differ, profiling runs two passes
    # (prefill@prefill_batch_size + decode@decode_batch_size) and merges them.
    # Absent → fall back to the single ``batch_size`` (legacy single pass).
    prefill_batch_size = data.get("prefill_batch_size")
    decode_batch_size = data.get("decode_batch_size")

    # Device selection: comma-separated indexes of the devices actually present.
    # A TP=N run needs N of them, so an under-sized or non-existent selection is
    # refused here rather than failing deep inside engine start-up.
    try:
        device_ids = bench_devices.parse_device_ids(data.get("device_ids"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    dev_err = bench_devices.validate_device_ids(device_ids, _DEVICE,
                                                need=int(tp_size or 1))
    if dev_err:
        return jsonify({"ok": False, "error": dev_err}), 400

    # The engine must fit the whole sequence it will ever see: cached context +
    # new query tokens + the decode tokens we generate. The frontend sizes
    # max_model_len from Query+Context; bump it to also cover the decode budget.
    if query_len:
        needed = int(query_len) + int(context_len or 0) + int(max_tokens) + 16
        if needed > int(max_model_len):
            max_model_len = needed

    with _profile_lock:
        profiling._profile_state = {
            "status": "running",
            "result": None,
            "error": None,
            "model_id": model_id,
            "settings": {
                "mode": mode,
                "batch_size": batch_size,
                "prefill_batch_size": prefill_batch_size,
                "decode_batch_size": decode_batch_size,
                "max_model_len": max_model_len,
                "tp_size": tp_size,
                "quantization": quantization,
                "query_len": query_len,
                "context_len": context_len,
                "device_ids": device_ids,
            },
        }

    thread = threading.Thread(
        target=_run_profile,
        args=(model_id, mode, max_model_len, batch_size, max_tokens, prompt,
              num_profile_layers, tp_size, quantization, gpu_memory_utilization,
              query_len, context_len, prefill_batch_size, decode_batch_size),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "status": "running"})


@app.route("/api/profile/upload", methods=["POST"])
def upload_profile():
    """Reconstruct the model graph and op breakdown from uploaded trace(s).

    Accepts a multipart form with one or more ``trace`` files (torch profiler
    Chrome traces, ``.json`` or ``.json.gz``). The upload path mirrors the live
    profiler so a **download -> upload round-trip** reconstructs the same graph
    (and drives the Shape Matrix export) on a machine **without an XPU/GPU**:

      - **Two-pass pair** — upload the separate ``…_prefill_…`` and
        ``…_decode_…`` files the download endpoint produces for a two-pass run
        (distinct prefill/decode batch sizes). Each pass's phase graph is built
        with that pass's batch/query size and merged via ``_merge_two_pass_result``
        (prefill tree + decode base), exactly like a live two-pass run — so the
        reconstructed graph has **both** phases, not decode only.
      - **Single trace** — one file (optionally several rank files for TP>1,
        in any order) reconstructs a single run; the **rank-0** file is used and
        the other ranks are ignored (rank 0 is the representative worker — see
        ``_rank0_first``).

    The descriptive download filenames are self-describing
    (``…_{mode}[_prefill|_decode]_ctx{C}_in{S}_out{gen}_bs{B}_tp{TP}[_{quant}]_{N}layers``),
    so mode/TP/quant/query/context/batch and the prefill|decode split are
    recovered from the names via ``_parse_trace_filename`` — no need to re-enter
    them. Explicit form fields still override, and legacy names without the
    encoded tail fall back to form fields:

      - ``model_id``: HF id used to fetch config for shape symbols / summary
      - ``tensor_parallel_size`` / ``tp_size``: ranks represented by the uploads
      - ``batch_size`` / ``prefill_batch_size`` / ``decode_batch_size``
      - ``quantization``, ``mode``, ``query_len``, ``context_len``
      - ``num_profile_layers`` / ``actual_layers``: for reduced-layer scaling

    Parsing is fast, so this runs synchronously and stores the result in the
    shared profile state so ``/api/profile/result`` and ``/api/profile/trace``
    work exactly as they do for a live profiling run.
    """
    
    if profiling.is_running():
        return jsonify({"ok": False,
                        "error": "Profiling already in progress"}), 409
    files = [f for f in request.files.getlist("trace") if f and f.filename]
    if not files:
        return jsonify({"ok": False, "error": "No trace file uploaded"}), 400
    saved = profiling.save_uploads(
        [(f.filename, f.save) for f in files])
    ok, error = profiling.build_from_uploads(saved, request.form)
    if not ok:
        return jsonify({"ok": False, "error": error}), 500
    return jsonify({"ok": True, "status": "done"})


@app.route("/api/profile/status")
def profile_status():
    """Check profiling status."""
    with _profile_lock:
        return jsonify({
            "status": profiling._profile_state["status"],
            "model_id": profiling._profile_state["model_id"],
            "settings": profiling._profile_state.get("settings"),
            "error": profiling._profile_state["error"],
        })


@app.route("/api/profile/result")
def profile_result():
    """Get profiling results (only available when status=done)."""
    with _profile_lock:
        if profiling._profile_state["status"] != "done":
            return jsonify({
                "ok": False,
                "status": profiling._profile_state["status"],
                "error": profiling._profile_state.get("error"),
            }), 202
        result = profiling._profile_state["result"]
    # Don't expose internal trace_file path(s); indicate availability instead.
    _internal = {"trace_file", "prefill_trace_file", "decode_trace_file"}
    client_result = {k: v for k, v in result.items() if k not in _internal}
    client_result["has_trace"] = bool(result.get("trace_file"))
    client_result["has_prefill_trace"] = bool(result.get("prefill_trace_file"))
    client_result["has_decode_trace"] = bool(result.get("decode_trace_file"))
    return jsonify({"ok": True, "data": client_result})


@app.route("/api/profile/trace")
def download_trace():
    """Download a profiled trace file with a descriptive filename.

    Two-pass runs (separate prefill/decode batch sizes) write a separate trace
    per pass. Use ``?pass=prefill`` or ``?pass=decode`` to pick one; without it
    the default (decode, i.e. the merged base) is served for backward compat.
    """
    with _profile_lock:
        if profiling._profile_state["status"] != "done" or not profiling._profile_state.get("result"):
            return jsonify({"ok": False, "error": "No profile result available"}), 404
        result = profiling._profile_state["result"]

    which = (request.args.get("pass") or "").strip().lower()
    if which == "prefill":
        trace_path = result.get("prefill_trace_file")
        pass_tag = "_prefill"
        pass_bs = result.get("prefill_batch_size")
        pass_gen = 1  # prefill pass generates a single token
    elif which == "decode":
        trace_path = result.get("decode_trace_file") or result.get("trace_file")
        pass_tag = "_decode"
        pass_bs = result.get("decode_batch_size", result.get("batch_size", 1))
        pass_gen = result.get("max_tokens", "")
    else:
        trace_path = result.get("trace_file")
        # Tag the filename only when the run actually has distinct passes.
        pass_tag = "_decode" if result.get("two_pass") else ""
        pass_bs = result.get("decode_batch_size", result.get("batch_size", 1))
        pass_gen = result.get("max_tokens", "")

    if not trace_path or not os.path.isfile(trace_path):
        return jsonify({"ok": False, "error": "Trace file not found"}), 404

    # Build a descriptive filename encoding the profiled configuration:
    #   vllm_trace_{model}_{mode}[_prefill|_decode]_ctx{context}_in{query}_out{gen}_bs{bs}_tp{tp}_{n}layers.json.gz
    # where "ctx" is the block-aligned prefix-cache context the prefill attends
    # to, "in" is the query length (new prefill tokens, S), "out" is the number
    # of generated decode tokens, and "bs" is the pass's batch. The model id is
    # reduced to its final path component (org prefix dropped).
    model_short = result["model_id"].split("/")[-1]
    mode = result.get("mode", "eager")
    bs = pass_bs if pass_bs is not None else result.get("batch_size", 1)
    ctx = result.get("context_len_aligned") or result.get("context_len") or 0
    qin = result.get("query_len") or 0
    gen = pass_gen
    tp = result.get("tp_size", 1) or 1
    quant = result.get("quantization")
    layers = result.get("profiled_layers", "all")
    device = _DEVICE.upper()
    ext = ".json.gz" if trace_path.endswith(".gz") else ".json"
    quant_part = f"_{quant}" if quant else ""
    download_name = (
        f"vllm_trace_{model_short}_{device}_{mode}{pass_tag}_ctx{ctx}_in{qin}"
        f"_out{gen}_bs{bs}_tp{tp}{quant_part}_{layers}layers{ext}"
    )

    return send_file(
        trace_path,
        mimetype="application/gzip" if ext == ".json.gz" else "application/json",
        as_attachment=True,
        download_name=download_name,
    )


from breakdown.shape_derive import (  # noqa: F401  (re-exported for tests)
    _MAX_MATRIX_ROWS,
    _WEIGHT_ROLES,
    _bytes_to_dtype,
)





@app.route("/api/export/shape-matrix", methods=["POST"])
def export_shape_matrix():
    """Export op shapes/dtypes across configurations, grounded in a profiling run.

    Produces a flat Excel table where each row is one
    (Phase, SeqLen, CtxLen, BatchSize, TP, Op) combination.
    Columns for Phase/SeqLen/CtxLen/BatchSize/TP enable Excel filtering
    to select any desired configuration subset.

    Prefill configs: sweep seq_lens × context_lens × batch_sizes × tp_sizes
      (assumes chunked prefill; seq_len = chunk size)
    Decode configs: seq_len fixed to 1, sweep context_lens × batch_sizes × tp_sizes

    The op set + real shapes come from the latest completed profiling run for
    this model (its reconstructed graph is used as a symbolic template): each
    config re-resolves S/B/C/TP and recomputes Memory/FLOPs from the resolved
    shapes (using the recorded per-tensor dtypes). The op set is fixed at the
    profiled config (TP collectives, MoE routing, chunked-prefill splits), so the
    caller profiles at each TP it needs; S/B/C are parametric from one profile.
    The frontend ensures a matching profile exists (reusing the latest run or
    launching a fresh one) before calling this endpoint.

    The rows themselves are built by :mod:`breakdown.shape_matrix`; this
    endpoint only validates the request and serializes. ``/api/perf/*`` consumes
    the same rows in-process, without going through Excel.
    """
    payload, filename, error = service.shape_matrix_workbook(request.json or {})
    if error:
        return jsonify({"ok": False, "error": error}), 400
    return Response(
        payload,
        mimetype="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



# ---------------------------------------------------------------------------
# Benchmark pipeline (/api/bench/*): profile -> replay -> ranked targets
#
# The web layer only wraps breakdown.bench; every stage also runs headless via
# ``python -m breakdown.bench``. Benchmark runs are long, so they use the same
# async job + lock pattern as profiling.
# ---------------------------------------------------------------------------





@app.route("/api/bench/plan", methods=["POST"])
def bench_plan():
    """Sweep the profiled graph into replay cases for a new benchmark run.

    The op set and the operands come from the profile, so this endpoint needs a
    completed profiling run for the requested model/quantization - the same
    precondition the Shape Matrix export has.
    """
    data = request.json or {}
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"ok": False, "error": "No model_id specified"}), 400

    template, settings, err = _profile_template_for(model_id,
                                                    data.get("quantization"))
    if err:
        return jsonify({"ok": False, "error": err}), 400

    sweep = {k: data.get(k, v) for k, v in shape_matrix.DEFAULT_SWEEP.items()}
    tp_sizes = sweep.get("tp_sizes") or [1]
    configs = shape_matrix.build_configs(**sweep)
    if shape_matrix.estimate_row_count(template, configs) > \
            shape_matrix.MAX_MATRIX_ROWS:
        return jsonify({"ok": False, "error": "Sweep too large - reduce it"}), 400

    rows = shape_matrix.build_rows(template, configs)
    device = data.get("device") or bench_devices.detect_device()
    # The largest swept TP is the widest collective the run will replay, so the
    # selection must contain at least that many devices.
    ok, device_ids, dev_err = service._bench_device_ids(data, device,
                                                need=max(int(t) for t in tp_sizes))
    if not ok:
        return jsonify({"ok": False, "error": dev_err}), 400
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
    bench_store.RunMeta(
        run_id=run_id, model_id=model_id, device=device, tp=int(tp_sizes[0]),
        device_name=bench_devices.device_name(device), device_ids=device_ids,
        sku=bench_devices.sku_for_device(bench_devices.device_name(device)),
        smoke=bool(data.get("smoke")),
        sweep={**sweep, "configs": len(configs)}).write(paths)

    return jsonify({"ok": True, "run_id": run_id, "rows": len(rows),
                    "cases": len(cases), "coverage": coverage})




@app.route("/api/bench/run", methods=["POST"])
def bench_run():
    """Replay a run's cases. Non-blocking - poll /api/bench/status.

    ``budget`` (seconds of measurement per case) is optional and normally
    omitted: the runner derives it per op from the profiled shapes, so the UI
    has no budget knob to guess at.
    """
    data = request.json or {}
    run_id = data.get("run_id")
    if not run_id:
        return jsonify({"ok": False, "error": "run_id is required"}), 400
    paths = bench_store.run_paths(run_id)
    if not os.path.isfile(paths.cases):
        return jsonify({"ok": False,
                        "error": f"no cases for run {run_id}"}), 400
    with service._bench_lock:
        if service._bench_state["status"] == "running":
            return jsonify({"ok": False,
                            "error": "A benchmark run is already in progress"}), 409
        service._bench_state.update({"status": "running", "run_id": run_id,
                             "error": None, "ops": [], "result": None})

    meta = bench_store.read_meta(paths)
    device = (data.get("device") or meta.get("device")
              or bench_devices.detect_device())
    ok, device_ids, dev_err = service._bench_device_ids(
        {"device_ids": data.get("device_ids", meta.get("device_ids"))},
        device, need=int(meta.get("tp") or 1))
    if not ok:
        with service._bench_lock:
            service._bench_state.update({"status": "idle", "run_id": None})
        return jsonify({"ok": False, "error": dev_err}), 400
    budget = data.get("budget")
    thread = threading.Thread(
        target=service._run_bench,
        args=(run_id, device, float(budget) if budget else None,
              data.get("ops"), device_ids),
        daemon=True)
    thread.start()
    return jsonify({"ok": True, "status": "running", "run_id": run_id})


@app.route("/api/bench/ops")
def bench_ops():
    """The dispatch names the latest profile ran - the Ops filter's checklist.

    The benchmark's op set is a property of the profile, so the filter offers
    exactly what was dispatched (framework plumbing excluded, since replaying
    it measures allocator noise) ordered by device time, i.e. by how much
    selecting it is worth.
    """
    with _profile_lock:
        status = profiling._profile_state["status"]
        result = profiling._profile_state.get("result")
        model_id = profiling._profile_state.get("model_id")
    graph = (result or {}).get("graph") if status == "done" and result else None
    if not graph:
        return jsonify({"ok": True, "model_id": model_id, "ops": []})

    ops = [op for op in summarize_ops(graph)
           if not bench_spec.is_skipped(op["op"])]
    return jsonify({"ok": True, "model_id": model_id, "ops": ops})


@app.route("/api/bench/status")
def bench_status():
    with service._bench_lock:
        return jsonify({"ok": True, **dict(service._bench_state)})


@app.route("/api/bench/runs")
def bench_runs():
    return jsonify({"ok": True, "runs": bench_store.list_runs()})


@app.route("/api/bench/results")
def bench_results():
    """A run's measured cases, enriched with utilization and the traced time.

    ``?op=`` narrows the payload to one dispatch name - what the UI's per-op
    detail view needs, instead of shipping every case of every op to render one.
    """
    run_id = request.args.get("run_id")
    runs = bench_store.list_runs()
    run_id = run_id or (runs[0]["run_id"] if runs else None)
    if not run_id:
        return jsonify({"ok": False, "error": "no benchmark runs yet"}), 404
    paths = bench_store.run_paths(run_id)
    meta = bench_store.read_meta(paths)
    records = bench_store.read_results(paths.results)
    op = request.args.get("op")
    if op:
        records = [r for r in records if r.get("op") == op]
    peaks = bench_devices.peaks(meta.get("sku") or bench_devices.DEFAULT_SKU)
    rich = bench_reports.enrich(records, peaks)
    return jsonify({"ok": True, "run_id": run_id, "op": op, "peaks": peaks,
                    "summary": bench_reports.summarize(rich),
                    "coverage": bench_reports.coverage(rich),
                    "records": rich})


@app.route("/api/bench/targets")
def bench_targets():
    """The ranked optimization targets of a run (recomputed with ?refresh=1)."""
    runs = bench_store.list_runs()
    run_id = request.args.get("run_id") or (runs[0]["run_id"] if runs else None)
    if not run_id:
        return jsonify({"ok": False, "error": "no benchmark runs yet"}), 404
    paths = bench_store.run_paths(run_id)
    meta = bench_store.read_meta(paths)
    if request.args.get("refresh") not in ("1", "true") and \
            os.path.isfile(paths.targets):
        with open(paths.targets) as fh:
            return jsonify({"ok": True, "run_id": run_id,
                            "targets": json.load(fh)})

    records = bench_store.read_results(paths.results)
    if not records:
        return jsonify({"ok": False,
                        "error": f"run {run_id} has no benchmark results"}), 400
    rc = bench_rank.RankConfig(
        target_util=float(request.args.get(
            "target_util", bench_rank.DEFAULT_TARGET_UTIL)),
        tp=meta.get("tp"), top=int(request.args.get("top", 0)),
        run_id=run_id,
        provenance={"run_id": run_id, "commits": meta.get("commits") or {}})
    try:
        doc = bench_rank.rank(records, rc)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    bench_store.write_json(paths.targets, doc)
    try:
        conn = bench_history.connect(
            bench_history.db_path(bench_store.bench_root()))
        bench_history.ingest(conn, meta or {"run_id": run_id}, records, doc)
    except Exception as exc:  # noqa: BLE001 - history is best-effort
        logger.warning("bench history ingest failed: %s", exc)
    return jsonify({"ok": True, "run_id": run_id, "targets": doc})




@app.route("/api/bench/report")
def bench_report():
    """Download a run's report workbook (built on demand).

    One download, not two: the workbook carries the run's Shape Matrix (the
    sweep the cases were built from) next to what was measured on it.
    """
    run_id = request.args.get("run_id")
    if not run_id:
        return jsonify({"ok": False, "error": "run_id is required"}), 400
    paths = bench_store.run_paths(run_id)
    records = bench_store.read_results(paths.results)
    if not records:
        return jsonify({"ok": False, "error": "run has no results"}), 400
    meta = bench_store.read_meta(paths)
    targets = None
    if os.path.isfile(paths.targets):
        with open(paths.targets) as fh:
            targets = json.load(fh)
    bench_reports.write_workbook(
        records, paths.report,
        bench_devices.peaks(meta.get("sku") or bench_devices.DEFAULT_SKU),
        targets, service._bench_matrix(paths))
    return send_file(paths.report, as_attachment=True,
                     download_name=f"{run_id}_bench.xlsx")


@app.route("/api/bench/history")
def bench_history_api():
    """Runs in the history db, or a per-shape diff of two runs."""
    conn = bench_history.connect(
        bench_history.db_path(bench_store.bench_root()))
    base, new = request.args.get("base"), request.args.get("new")
    if base and new:
        return jsonify({"ok": True, "base": base, "new": new,
                        "changes": bench_history.compare(
                            conn, base, new,
                            float(request.args.get("threshold", 0.10)))})
    return jsonify({"ok": True, "runs": bench_history.runs(conn)})






@app.route("/api/optimize/candidates")
def optimize_candidates():
    """The ranked ops of a phase and whether each is worth a kernel session."""
    run_id = request.args.get("run_id")
    if not run_id:
        runs = bench_store.list_runs()
        run_id = runs[0]["run_id"] if runs else ""
    phase = request.args.get("phase") or "prefill"
    device = bench_devices.detect_device()
    # The environment half of the answer (where sessions run, on what, with
    # which binary) does not depend on the ranking, so it is returned even when
    # the run has none - the form is then still usable.
    base = {
        "run_id": run_id, "phase": phase,
        "devices": bench_devices.available(device),
        "workspace_root": optimize_session.default_workspace_root(),
        "copilot": optimize_session.resolve_copilot(),
        "skill": optimize_prompt.OPTIMIZER_SKILL,
    }
    doc, err = service.optimize_doc(run_id)
    if err:
        return jsonify({"ok": False, "error": err, **base}), 400
    return jsonify({"ok": True, **base,
                    "candidates": optimize_prompt.candidates(doc, phase)})


@app.route("/api/optimize/prompt", methods=["POST"])
def optimize_prompt_api():
    """The brief and the exact command, without spawning anything.

    Spawning is a convenience: the same session can always be run by hand,
    which is the fallback when the server should not hold a long-lived agent.
    """
    data = request.json or {}
    doc, err = service.optimize_doc(data.get("run_id"))
    if err:
        return jsonify({"ok": False, "error": err}), 400
    phase = data.get("phase") or "prefill"
    by_op = optimize_prompt.targets_by_op(doc, phase)
    cwd = data.get("cwd") or optimize_session.default_workspace_root()
    try:
        ids = bench_devices.parse_device_ids(data.get("device_ids"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    out = []
    for op in data.get("ops") or []:
        target = by_op.get(op)
        if target is None:
            return jsonify({"ok": False,
                            "error": f"'{op}' is not a ranked target"}), 400
        paths = optimize_session.session_paths(data["run_id"], op).ensure()
        can, reason = optimize_prompt.launchability(target)
        text = optimize_prompt.build_prompt(
            target, doc, run_id=data["run_id"], phase=phase,
            device_ids=ids or None, workspace_root=cwd,
            artifact_dir=paths.dir)
        # The command reads the brief from disk, so the brief has to be there:
        # the fallback must work as-is, not only after a spawn.
        with open(paths.prompt, "w", encoding="utf-8") as fh:
            fh.write(text)
        argv = optimize_session.session_argv(text)
        out.append({
            "op": op, "launchable": can, "reason": reason, "prompt": text,
            "command": optimize_session.command_line(
                argv, cwd=cwd,
                env=bench_devices.visibility_env(
                    bench_devices.detect_device(), ids),
                prompt_file=paths.prompt),
            "prompt_file": paths.prompt,
        })
    return jsonify({"ok": True, "run_id": data["run_id"], "sessions": out})


@app.route("/api/optimize/start", methods=["POST"])
def optimize_start():
    """Open a session per selected kernel; each owns one GPU exclusively."""
    data = request.json or {}
    run_id = data.get("run_id")
    doc, err = service.optimize_doc(run_id)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    device = bench_devices.detect_device()
    ok, ids, dev_err = service._bench_device_ids(data, device)
    if not ok:
        return jsonify({"ok": False, "error": dev_err}), 400
    try:
        state = OPTIMIZE_MANAGER.start(
            run_id=run_id, doc=doc, ops=data.get("ops") or [],
            phase=data.get("phase") or "prefill",
            workspace_root=data.get("cwd") or None,
            device_kind=device, device_ids=ids or None,
            spawn=not data.get("dry_run"))
    except (ValueError, NotADirectoryError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except KeyError as exc:
        # KeyError's str() quotes its argument; the message is user-facing.
        return jsonify({"ok": False, "error": exc.args[0]}), 400
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 501
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    return jsonify({"ok": True, "run_id": state["run_id"],
                    "pool": state["pool"],
                    "sessions": service.optimize_sessions(run_id)})


@app.route("/api/optimize/status")
def optimize_status():
    """Every session of a run: state, the GPU it holds, its queue position."""
    run_id = request.args.get("run_id") or ""
    sessions = service.optimize_sessions(run_id)
    return jsonify({"ok": True, "run_id": run_id, "sessions": sessions,
                    "pool": OPTIMIZE_MANAGER.pool_snapshot(),
                    "active": any(s.get("state") in ("pending", "running")
                                  for s in sessions)})


@app.route("/api/optimize/log")
def optimize_log():
    """A session's log from ``offset`` - the UI polls this while it runs."""
    run_id, op = request.args.get("run_id"), request.args.get("op")
    if not run_id or not op:
        return jsonify({"ok": False, "error": "run_id and op are required"}), 400
    path = optimize_session.session_paths(run_id, op).log
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "offset must be an integer"}), 400
    if not os.path.isfile(path):
        return jsonify({"ok": True, "offset": 0, "text": "", "eof": True})
    size = os.path.getsize(path)
    if offset > size:      # the log was truncated/restarted - resend it whole
        offset = 0
    with open(path, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read()
    return jsonify({"ok": True, "offset": offset + len(chunk),
                    "text": chunk.decode("utf-8", "replace"),
                    "eof": offset + len(chunk) >= size})


@app.route("/api/optimize/stop", methods=["POST"])
def optimize_stop():
    """Stop one session or all of a run's; a freed GPU starts the next one."""
    data = request.json or {}
    run_id = data.get("run_id")
    if not run_id:
        return jsonify({"ok": False, "error": "run_id is required"}), 400
    stopped = OPTIMIZE_MANAGER.stop(run_id, data.get("op"))
    return jsonify({"ok": True, "run_id": run_id, "stopped": stopped,
                    "sessions": service.optimize_sessions(run_id)})


if __name__ == "__main__":
    import atexit

    # A session is a long-lived agent holding a GPU; do not leave orphans
    # (and their leases) behind when the server exits.
    atexit.register(OPTIMIZE_MANAGER.shutdown)

    parser = argparse.ArgumentParser(description="vLLM Breakdown Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Starting vLLM Breakdown at http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=args.debug)
