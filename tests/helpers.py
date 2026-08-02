# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers.

The reconstruction's public entry point takes a *file*, so nearly every test
writes its synthetic events to a temporary file. :func:`graph_of` does that
once, so tests assert through ``build_graph_from_trace`` — the public entry
point — instead of reaching into private passes that the refactor moves.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from breakdown.trace import build_graph_from_trace


def graph_of(events: list[dict], summary: dict, **kwargs: Any) -> dict:
    """Reconstruct a graph from an in-memory chrome-trace event list."""
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"traceEvents": events}, fh)
        return build_graph_from_trace(path, summary, **kwargs)
    finally:
        os.unlink(path)


def iter_ops(node: dict):
    """Every op in a reconstructed phase tree, depth-first."""
    for op in node.get("ops", []):
        yield op
    for child in node.get("children", []):
        yield from iter_ops(child)


def find_op(node: dict, name: str) -> dict | None:
    """The first op named ``name`` in a reconstructed phase tree."""
    for op in iter_ops(node):
        if op["name"] == name:
            return op
    return None


def device_time(node: dict) -> float:
    """Total device time on every op leaf of a phase tree."""
    return sum(op.get("device_time_us", 0.0) * op.get("count", 1)
               for op in iter_ops(node))
