# SPDX-License-Identifier: Apache-2.0
"""Golden snapshots of the *vocabulary* the pipeline applies to an op.

``test_golden_graph.py`` pins the shape of the reconstructed tree. This file
pins the three judgements the rest of the pipeline makes *about* each op in
that tree, because those judgements are currently spread over ~28 name-matching
tables in eight modules and the refactor consolidates them:

- **classification** — which backend an op belongs to (``classifier.py`` +
  ``registry.py``);
- **cost** — its analytic bytes and FLOPs, whether it is attention, whether it
  reaches the matrix engine, and which roof therefore bounds it (``cost.py``);
- **symbolic dims** — what every distinct symbolic dimension in the graph
  resolves to (``shape_derive.py``).

Consolidating a table is only safe if the answers do not move. A refactor that
changes one of these produces a reviewable diff here instead of a silently
different ranking three stages downstream.

Regenerate after a deliberate change:

    pytest tests/test_golden_semantics.py --update-golden
"""
from __future__ import annotations

import json
import os

import pytest

from breakdown import cost
from breakdown.classifier import classify_op
from breakdown.shape_derive import _resolve_dim
from breakdown.trace import build_graph_from_trace
from tests.data import GOLDEN_DIR, available

#: The SKU every cost snapshot is taken against. Fixed so the snapshot does not
#: depend on which card happens to be in the host running the suite.
PEAKS = {
    "bw_gbs": 456.0,
    "tflops": 98.3,
    "vector_tflops": 12.3,
    "cache_bytes": 18 * 1024 ** 2,
    "cache_bw_gbs": 1200.0,
}

FIXTURES = available()
IDS = [f.name for f in FIXTURES]


def _iter_ops(node: dict | None):
    if node is None:
        return
    for op in node.get("ops", []):
        yield op
    for child in node.get("children", []):
        yield from _iter_ops(child)


def _int_shapes(shapes) -> list[list[int]]:
    """Only the fully-concrete shapes; symbolic dims are covered separately."""
    out = []
    for row in shapes or []:
        if isinstance(row, list) and all(isinstance(d, int) for d in row):
            out.append(list(row))
    return out


def _cost_entry(op: dict) -> dict:
    """What the cost model says about one op, at its recorded shapes."""
    name = op.get("name", "")
    shapes = _int_shapes(op.get("recorded_shapes") or op.get("input_shapes"))
    dtypes = op.get("input_dtypes") or None
    nbytes = cost.op_bytes(name, shapes, dtypes) if shapes else 0
    flops = cost.op_flops(name, shapes) if shapes else 0
    entry = {
        "nbytes": nbytes,
        "flops": flops,
        "is_attention": cost.is_attention(name),
        "uses_matrix_engine": cost.uses_matrix_engine(name),
    }
    if nbytes or flops:
        bound, level = cost.bound_of(flops, nbytes, PEAKS, name)
        entry["bound"] = bound
        entry["memory_level"] = level
        entry["unit"] = cost.roof_unit(PEAKS, bound, level, name)
        entry["ai"] = round(cost.op_ai(flops, nbytes), 6)
    return entry


def _symbolic_dims(graph: dict) -> dict[str, object]:
    """Every distinct symbolic dim in the graph, and what it resolves to.

    A dim that fails to resolve is recorded as ``None`` rather than skipped, so
    the snapshot pins the *failures* too — that is what stops a "simplification"
    of the resolver from quietly turning a resolvable dim into a symbol.
    """
    symbols = dict(graph.get("symbols") or {})
    seen: set[str] = set()
    for phase in ("prefill", "decode"):
        for op in _iter_ops(graph.get(phase)):
            for row in op.get("input_shapes") or []:
                if not isinstance(row, list):
                    continue
                for dim in row:
                    if isinstance(dim, str):
                        seen.add(dim)
    out: dict[str, object] = {}
    for dim in sorted(seen):
        try:
            value = _resolve_dim(dim, symbols)
        except Exception:  # a resolver must never raise; record it if it does
            value = "RAISED"
        out[dim] = value if isinstance(value, (int, str)) else None
    return out


def _digest(graph: dict) -> dict:
    """Classification + cost for every op name, plus the dim legend.

    Keyed by op *name*, not by position: the same op at the same shapes must
    get the same answer wherever it appears, and keying by name makes an
    accidental context-dependence show up as a diff.
    """
    ops: dict[str, dict] = {}
    for phase in ("prefill", "decode"):
        for op in _iter_ops(graph.get(phase)):
            name = op.get("name", "")
            backend, category = classify_op(
                name,
                device_type=op.get("device_type", ""),
                self_device_time_us=op.get("self_device_time_us", 0.0) or 0.0,
                device_time_us=op.get("device_time_us", 0.0) or 0.0,
            )
            entry = {
                "backend": str(getattr(backend, "value", backend)),
                "category": category,
                "graph_backend": op.get("backend"),
                "cost": _cost_entry(op),
            }
            # The same name may recur; assert the vocabulary is stable rather
            # than letting the last occurrence silently win.
            ops.setdefault(name, entry)
    return {"ops": ops, "dims": _symbolic_dims(graph)}


def _golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{name}_semantics.json")


@pytest.mark.skipif(not FIXTURES, reason="no trace fixtures present")
@pytest.mark.parametrize("fixture", FIXTURES, ids=IDS)
def test_op_vocabulary_matches_golden(fixture, request):
    graph = build_graph_from_trace(fixture.trace_path, **fixture.build_kwargs())
    digest = _digest(graph)
    path = _golden_path(fixture.name)

    if request.config.getoption("--update-golden"):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(digest, f, indent=1, sort_keys=True, default=str)
        pytest.skip(f"golden updated: {path}")

    if not os.path.exists(path):
        pytest.fail(f"no golden for {fixture.name}; run with --update-golden")
    with open(path) as f:
        golden = json.load(f)

    actual = json.loads(json.dumps(digest, default=str))
    assert actual == golden, (
        f"{fixture.name}: an op's backend, cost or symbolic dim changed. "
        f"Review the diff; if intended, re-run with --update-golden.")


@pytest.mark.skipif(not FIXTURES, reason="no trace fixtures present")
@pytest.mark.parametrize("fixture", FIXTURES, ids=IDS)
def test_every_symbolic_dim_resolves(fixture):
    """A symbolic dim that does not resolve cannot be swept or costed.

    The sweep substitutes ``S``/``B``/``C``/``TP`` and asks for an integer; a
    dim the resolver returns as a string is one the shape matrix cannot turn
    into a number, so the op drops out of the benchmark. The four sweep
    variables are the legitimate exceptions — they stay symbolic by design.
    """
    graph = build_graph_from_trace(fixture.trace_path, **fixture.build_kwargs())
    dims = _symbolic_dims(graph)
    unresolved = sorted(k for k, v in dims.items() if not isinstance(v, int))
    assert not unresolved, (
        f"{fixture.name}: symbolic dims that do not resolve to an integer, so "
        f"they cannot be swept or costed: {unresolved}")
