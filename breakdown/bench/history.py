# SPDX-License-Identifier: Apache-2.0
"""Benchmark history: per-run case metrics in SQLite, and regression detection.

A single ranking says what to optimize *today*; the history says whether last
month's optimization survived a kernel-repo bump. Every measured case is stored
keyed by ``(op, shape signature)`` - the shape key, not the argument values, so
a run that swept different scalars still compares - together with the run's
component commits, so a regression can be attributed to a specific bump.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    created    TEXT,
    model_id   TEXT,
    device     TEXT,
    sku        TEXT,
    tp         INTEGER,
    commits    TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    run_id     TEXT NOT NULL,
    op         TEXT NOT NULL,
    backend    TEXT,
    shape_key  TEXT NOT NULL,
    shape      TEXT,
    phase      TEXT,
    latency_us REAL,
    util       REAL,
    traced_us  REAL,
    PRIMARY KEY (run_id, op, shape_key)
);
CREATE INDEX IF NOT EXISTS cases_op_idx ON cases (op, shape_key);
CREATE TABLE IF NOT EXISTS targets (
    run_id     TEXT NOT NULL,
    rank       INTEGER,
    op         TEXT NOT NULL,
    backend    TEXT,
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


def ingest(conn: sqlite3.Connection, run_meta: dict[str, Any],
           records: Iterable[dict[str, Any]],
           targets: dict[str, Any] | None = None) -> int:
    """Store one run's measured cases (and its ranking). Returns cases stored."""
    conn.execute(
        "INSERT OR REPLACE INTO runs "
        "(run_id, created, model_id, device, sku, tp, commits) "
        "VALUES (?,?,?,?,?,?,?)",
        (run_meta.get("run_id"), run_meta.get("created"),
         run_meta.get("model_id"), run_meta.get("device"),
         run_meta.get("sku") or (targets or {}).get("sku"),
         run_meta.get("tp"), json.dumps(run_meta.get("commits") or {})))

    n = 0
    for r in records:
        if r.get("status") != "ok":
            continue
        conn.execute(
            "INSERT OR REPLACE INTO cases (run_id, op, backend, shape_key, "
            "shape, phase, latency_us, util, traced_us) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_meta.get("run_id"), r.get("op"), r.get("backend"),
             r.get("shape_key"), r.get("shape"), r.get("phase"),
             r.get("latency_us"), r.get("util"),
             r.get("traced_device_time_us")))
        n += 1

    for t in (targets or {}).get("targets", []):
        conn.execute(
            "INSERT OR REPLACE INTO targets (run_id, rank, op, backend, "
            "e2e_us, share, util, action, savings_us) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_meta.get("run_id"), t.get("rank"), t.get("op"),
             t.get("backend"), t.get("e2e_us"), t.get("share_of_e2e"),
             (t.get("roofline") or {}).get("util"), t.get("action"),
             (t.get("savings_us") or {}).get("total")))
    conn.commit()
    return n


def runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    cur = conn.execute("SELECT * FROM runs ORDER BY created DESC LIMIT ?",
                       (limit,))
    return [dict(r) for r in cur.fetchall()]


def compare(conn: sqlite3.Connection, base_run: str, new_run: str,
            threshold: float = 0.10) -> list[dict[str, Any]]:
    """Per-(op, shape) latency change between two runs.

    Positive ``delta_pct`` means the new run is **slower**. Only shapes present
    in both runs are compared, so a changed sweep cannot fake a regression.
    """
    cur = conn.execute(
        "SELECT b.op AS op, b.backend AS backend, b.shape AS shape, "
        "       b.latency_us AS base_us, n.latency_us AS new_us "
        "FROM cases b JOIN cases n "
        "  ON b.op = n.op AND b.shape_key = n.shape_key "
        "WHERE b.run_id = ? AND n.run_id = ? "
        "  AND b.latency_us > 0 AND n.latency_us > 0",
        (base_run, new_run))
    out = []
    for r in cur.fetchall():
        delta = (r["new_us"] - r["base_us"]) / r["base_us"]
        if abs(delta) < threshold:
            continue
        out.append({
            "op": r["op"], "backend": r["backend"], "shape": r["shape"],
            "base_us": round(r["base_us"], 3), "new_us": round(r["new_us"], 3),
            "delta_pct": round(delta * 100, 1),
            "kind": "regression" if delta > 0 else "improvement",
        })
    out.sort(key=lambda d: -abs(d["delta_pct"]))
    return out
