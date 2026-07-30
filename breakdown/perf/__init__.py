# SPDX-License-Identifier: Apache-2.0
"""Perf pipeline: reconstructed graph -> shapes -> benchmarks -> ranked targets.

Stages (each usable standalone from :mod:`breakdown.perf.cli` or the
``/api/perf/*`` endpoints):

``shape_matrix``  graph + config sweep -> matrix rows (the export is one
                  serialization of these rows, not a transport format)
``op_map``        matrix row -> the micro_perf op case that benchmarks it
``workloads``     rows -> grouped micro_perf workload JSON + coverage report
``estimate``      per-case cost -> estimated runtime -> per-op benchmark timeout
``runner``        run xpu-perf/micro_perf per op, isolated, into a report tree
``reports``       report tree -> records / merged workbook
``rank``          calls x latency x roofline headroom -> opt_targets.json
``history``       per-run SQLite history for regression detection

This package is deliberately **import-light**: no torch, no vLLM, no Flask at
import time, so it works inside the web app and on a GPU-less box (the CUDA
analysis path). ``runner``/``bench_case`` import torch lazily when they run.
"""
from __future__ import annotations

__all__ = [
    "devices",
    "estimate",
    "op_map",
    "rank",
    "reports",
    "shape_matrix",
    "workloads",
]
