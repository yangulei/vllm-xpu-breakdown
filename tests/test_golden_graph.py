# SPDX-License-Identifier: Apache-2.0
"""Golden-snapshot tests for the reconstruction pipeline.

These are the refactor's safety net. Every other test in this suite asserts one
rule against a synthetic trace built to exercise it; these assert the *whole*
pipeline against real captured profiles of the hardest model the repo targets
(MiniMax-M3, TP=4, 6 layers — see ``tests/data/__init__.py``).

The snapshot is deliberately *structural*, not byte-exact: it records the module
tree, every op's name/backend/symbolic shapes/dtypes, the symbol legend and the
op inventory — the things a reader of the graph relies on — but not raw device
times, which vary run to run and would make the snapshot untestable. Device time
is asserted as a *distribution* (which nodes carry time, and that time is
conserved), not as absolute microseconds.

Regenerate after a deliberate output change:

    pytest tests/test_golden_graph.py --update-golden
"""
from __future__ import annotations

import json
import os

import pytest

from breakdown.graph_from_trace import build_graph_from_trace
from tests.data import GOLDEN_DIR, available


def pytest_namespace():  # pragma: no cover - compatibility shim
    return {}


def _round(x: float | None, places: int = 4) -> float | None:
    return None if x is None else round(float(x), places)


def _node_digest(node: dict | None) -> dict | None:
    """A stable, reviewable description of one module node.

    Excludes absolute timings (run-to-run noise) but keeps *whether* a node
    carries device time, so an attribution regression that moves time between
    nodes is still caught.
    """
    if node is None:
        return None
    return {
        "name": node.get("name"),
        "module_type": node.get("module_type"),
        "repeat_count": node.get("repeat_count", 1),
        "carries_device_time": bool(node.get("total_device_time_us")),
        "ops": [
            {
                "name": op.get("name"),
                "backend": op.get("backend"),
                "shapes": op.get("input_shapes"),
                "dtypes": op.get("input_dtypes"),
                "role": op.get("role"),
                "launch": (lambda f: f and {
                    "file": os.path.basename(f.get("file", "")),
                    "func": f.get("func"),
                })(op.get("launch")),
                "order": op.get("order"),
                "carries_device_time": bool(op.get("device_time_us")),
            }
            for op in node.get("ops", [])
        ],
        "children": [_node_digest(c) for c in node.get("children", [])],
    }


def _graph_digest(graph: dict) -> dict:
    return {
        "symbols": graph.get("symbols"),
        "has_timing": graph.get("has_timing"),
        "prefill": _node_digest(graph.get("prefill")),
        "decode": _node_digest(graph.get("decode")),
    }


def _iter_ops(node: dict | None):
    if node is None:
        return
    for op in node.get("ops", []):
        yield op
    for child in node.get("children", []):
        yield from _iter_ops(child)


FIXTURES = available()
IDS = [f.name for f in FIXTURES]


@pytest.mark.skipif(not FIXTURES, reason="no trace fixtures present")
@pytest.mark.parametrize("fixture", FIXTURES, ids=IDS)
def test_graph_matches_golden(fixture, request):
    graph = build_graph_from_trace(fixture.trace_path, **fixture.build_kwargs())
    digest = _graph_digest(graph)

    if request.config.getoption("--update-golden"):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(fixture.golden_path, "w") as f:
            json.dump(digest, f, indent=1, sort_keys=True, default=str)
        pytest.skip(f"golden updated: {fixture.golden_path}")

    if not os.path.exists(fixture.golden_path):
        pytest.fail(f"no golden for {fixture.name}; run with --update-golden")
    with open(fixture.golden_path) as f:
        golden = json.load(f)

    actual = json.loads(json.dumps(digest, default=str))
    assert actual == golden, (
        f"{fixture.name}: reconstruction changed. Review the diff; if the "
        f"change is intended, re-run with --update-golden."
    )


@pytest.mark.skipif(not FIXTURES, reason="no trace fixtures present")
@pytest.mark.parametrize("fixture", FIXTURES, ids=IDS)
def test_expected_phase_is_reconstructed(fixture):
    """A pass profiled for one phase must actually yield that phase.

    The two-pass profiler keeps the prefill tree from the prefill pass and the
    decode tree from the decode pass; a phase-classification regression that
    drops one of them (``prefill: None``) is invisible in a merged result but
    fatal to everything downstream.
    """
    graph = build_graph_from_trace(fixture.trace_path, **fixture.build_kwargs())
    assert graph.get(fixture.phase) is not None, (
        f"{fixture.name} was profiled as a {fixture.phase} pass but "
        f"reconstruction produced no {fixture.phase} tree")


@pytest.mark.skipif(not FIXTURES, reason="no trace fixtures present")
@pytest.mark.parametrize("fixture", FIXTURES, ids=IDS)
def test_module_names_come_from_spans(fixture):
    """Capture-time spans are the primary naming path; assert they landed.

    This is the check that would have caught ``VLLM_ALLOW_INSECURE_SERIALIZATION``
    silently disabling ``apply_model``: the run still produced a graph, but every
    module carried a class-heuristic name instead of its real attribute path.
    """
    if not fixture.has_module_spans:
        pytest.skip("archived trace captured before the span hooks existed")
    graph = build_graph_from_trace(fixture.trace_path, **fixture.build_kwargs())
    names = set()

    def walk(node):
        if node is None:
            return
        if node.get("name"):
            names.add(node["name"])
        for child in node.get("children", []):
            walk(child)

    walk(graph.get(fixture.phase))
    # These are attribute paths that only a named_modules()-derived span can
    # supply; a class heuristic yields "norm"/"o_proj" style guesses instead.
    assert {"language_model", "model", "embed_tokens"} <= names, sorted(names)


#: Ops that legitimately carry no tensor shape, and the one that does not.
#:
#: ``TorchDynamo Cache Lookup`` / ``Torch-Compiled Region`` are compile-plumbing
#: markers, and ``aten::zeros`` takes a size, not a tensor — those three are
#: honest blanks.
#:
#: ``triton::fused_moe_kernel`` is **not**: it is the routed-expert grouped GEMM,
#: the single most expensive kernel in a MoE forward, and it is shape-less only
#: because Triton launches it straight from Python with no ``cpu_op`` to record
#: ``Input Dims`` on. Everything downstream — the sweep, the cost model, the
#: ranking, the replay — needs a shape, so today the heaviest kernel in the model
#: silently drops out of the pipeline. Capture-time kernel-launcher spans (M1)
#: close this; when they land, delete the entry and this test becomes its
#: regression guard.
KNOWN_SHAPELESS = {
    "TorchDynamo Cache Lookup",
    "Torch-Compiled Region: 0/1",
    "aten::zeros",
    "triton::fused_moe_kernel",
}


@pytest.mark.skipif(not FIXTURES, reason="no trace fixtures present")
@pytest.mark.parametrize("fixture", FIXTURES, ids=IDS)
def test_every_op_carries_a_shape(fixture):
    """No op in the graph may be shape-less, except the tracked exceptions.

    A shape-less op cannot be swept, costed, ranked or replayed, so it silently
    drops out of the pipeline at the first stage that needs a number. The
    allowlist is deliberately explicit so a *new* shape-less op fails here
    instead of quietly disappearing downstream.
    """
    graph = build_graph_from_trace(fixture.trace_path, **fixture.build_kwargs())
    missing = sorted({
        op["name"] for op in _iter_ops(graph.get(fixture.phase))
        if not op.get("input_shapes")
    })
    unexpected = [m for m in missing if m not in KNOWN_SHAPELESS]
    assert not unexpected, f"{fixture.name}: new ops without shapes: {unexpected}"
