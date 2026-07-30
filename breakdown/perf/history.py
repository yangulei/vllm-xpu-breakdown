# SPDX-License-Identifier: Apache-2.0
"""Perf history: per-run case metrics in SQLite, and regression detection.

A single ranking says what to optimize *today*; the history says whether last
month's optimization survived a kernel-repo bump. Every benchmarked case is
stored keyed by (op, provider, shape) together with the run's component
commits, so two runs can be diffed op-by-op at identical shapes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    created    TEXT,
    model_id   TEXT,
    backend    TEXT,
    dispatch   TEXT,
    tp         INTEGER,
    device     TEXT,
    commits    TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    run_id     TEXT NOT NULL,
    op         TEXT NOT NULL,
    provider   TEXT NOT NULL,
    args_hash  TEXT NOT NULL,
    args       TEXT NOT NULL,
    latency_us REAL,
    mem_bw_gbs REAL,
    tflops     REAL,
    PRIMARY KEY (run_id, op, provider, args_hash)
);
CREATE INDEX IF NOT EXISTS cases_op_idx ON cases (op, provider, args_hash);
CREATE TABLE IF NOT EXISTS targets (
    run_id     TEXT NOT NULL,
    rank       INTEGER,
    op         TEXT NOT NULL,
    provider   TEXT,
    e2e_us     REAL,
    share      REAL,
    util       REAL,
    action     TEXT,
    savings_us REAL,
    PRIMARY KEY (run_id, op)
);
"""


def db_path(root: str) -> str:
    return os.path.join(root, "history.sqlite")


def connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def args_hash(args: dict[str, Any]) -> str:
    blob = json.dumps({k: str(v) for k, v in sorted(args.items())},
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def ingest(conn: sqlite3.Connection, run_meta: dict[str, Any],
           records: Iterable[dict[str, Any]],
           targets: dict[str, Any] | None = None) -> int:
    """Store one run's cases (and its ranking, if any). Returns cases stored."""
    conn.execute(
        "INSERT OR REPLACE INTO runs "
        "(run_id, created, model_id, backend, dispatch, tp, device, commits) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (run_meta.get("run_id"), run_meta.get("created"),
         run_meta.get("model_id"), run_meta.get("backend"),
         run_meta.get("dispatch"), run_meta.get("tp"),
         (targets or {}).get("device"),
         json.dumps(run_meta.get("commits") or {})))

    n = 0
    for r in records:
        args = {k[4:]: v for k, v in r.items()
                if k.startswith("arg.") and v is not None}
        conn.execute(
            "INSERT OR REPLACE INTO cases (run_id, op, provider, args_hash, "
            "args, latency_us, mem_bw_gbs, tflops) VALUES (?,?,?,?,?,?,?,?)",
            (run_meta.get("run_id"), r.get("op"), r.get("provider"),
             args_hash(args), json.dumps(args, sort_keys=True),
             r.get("latency_us"), r.get("mem_bw_GBs"), r.get("tflops")))
        n += 1

    for t in (targets or {}).get("targets", []):
        conn.execute(
            "INSERT OR REPLACE INTO targets (run_id, rank, op, provider, "
            "e2e_us, share, util, action, savings_us) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_meta.get("run_id"), t.get("rank"), t.get("op"),
             t.get("dispatched_provider"), t.get("e2e_us"),
             t.get("share_of_e2e"), (t.get("roofline") or {}).get("util"),
             t.get("action"), (t.get("savings_us") or {}).get("total")))
    conn.commit()
    return n


def runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    cur = conn.execute("SELECT * FROM runs ORDER BY created DESC LIMIT ?",
                       (limit,))
    return [dict(r) for r in cur.fetchall()]


def compare(conn: sqlite3.Connection, base_run: str, new_run: str,
            threshold: float = 0.10) -> list[dict[str, Any]]:
    """Per-(op, provider, shape) latency change between two runs.

    Positive ``delta_pct`` means the new run is **slower**. Only shapes present
    in both runs are compared, so a changed sweep cannot fake a regression.
    """
    cur = conn.execute(
        "SELECT b.op AS op, b.provider AS provider, b.args AS args, "
        "       b.latency_us AS base_us, n.latency_us AS new_us "
        "FROM cases b JOIN cases n "
        "  ON b.op = n.op AND b.provider = n.provider "
        " AND b.args_hash = n.args_hash "
        "WHERE b.run_id = ? AND n.run_id = ? "
        "  AND b.latency_us > 0 AND n.latency_us > 0",
        (base_run, new_run))
    out = []
    for r in cur.fetchall():
        delta = (r["new_us"] - r["base_us"]) / r["base_us"]
        if abs(delta) < threshold:
            continue
        out.append({
            "op": r["op"], "provider": r["provider"],
            "args": json.loads(r["args"]),
            "base_us": round(r["base_us"], 3), "new_us": round(r["new_us"], 3),
            "delta_pct": round(delta * 100, 1),
            "kind": "regression" if delta > 0 else "improvement",
        })
    out.sort(key=lambda d: -abs(d["delta_pct"]))
    return out
