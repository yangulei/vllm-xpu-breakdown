# SPDX-License-Identifier: Apache-2.0
"""Golden-snapshot fixtures for the reconstruction pipeline.

The refactor rewrites the trace → graph → rows → cases pipeline wholesale, so
the safety net cannot be "the unit tests still pass" — most of them assert on
synthetic traces built to exercise one rule each. What actually has to keep
working is the pipeline's output on a *real* profile of the hardest model the
repo targets.

`FIXTURES` names those profiles. Each is a rank-0 torch-profiler trace captured
from a real run and trimmed by ``tools/make_fixture.py`` (which refuses to write
a fixture whose reconstructed graph differs from the untrimmed original), plus
the exact reconstruction arguments the run was profiled with.

The canonical example is **MiniMax-M3, TP=4, 6 layers** — MoE with shared
experts, sparse (MSA) attention, Gemma fused norms, TP collectives and
Python-launched SYCL/Triton kernels, i.e. every mechanism the pipeline has.
The CUDA pair is archived reference data: it cannot be re-captured on this
host, so it is the only regression coverage for the driver-launched Triton MoE
and the FlashInfer norms.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(DATA_DIR, "golden")


@dataclass(frozen=True)
class Fixture:
    """One captured profile pass plus how to reconstruct it."""

    name: str
    trace: str
    config: str
    device: str
    phase: str
    tp_size: int
    batch_size: int
    query_len: int | None = None
    context_len: int | None = None
    has_module_spans: bool = True
    notes: str = ""

    @property
    def trace_path(self) -> str:
        return os.path.join(DATA_DIR, self.trace)

    @property
    def golden_path(self) -> str:
        return os.path.join(GOLDEN_DIR, f"{self.name}.json")

    def summary(self) -> dict:
        from breakdown.model_info import summarize_config
        with open(os.path.join(DATA_DIR, self.config)) as f:
            return summarize_config(json.load(f))

    def build_kwargs(self) -> dict:
        return {
            "summary": self.summary(),
            "tp_size": self.tp_size,
            "batch_size": self.batch_size,
            "query_len": self.query_len,
            "context_len": self.context_len,
        }


M3 = "config_MiniMax-M3.json"

FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        name="m3_tp4_6l_xpu_prefill",
        trace="m3_tp4_6l_xpu_prefill.json.gz",
        config=M3, device="xpu", phase="prefill",
        tp_size=4, batch_size=1, query_len=2048, context_len=2048,
        notes="Canonical example. Prefill pass of the two-pass run: batch 1, "
              "2048 new tokens attending a 2048-token prefix-cached context.",
    ),
    Fixture(
        name="m3_tp4_6l_xpu_decode",
        trace="m3_tp4_6l_xpu_decode.json.gz",
        config=M3, device="xpu", phase="decode",
        tp_size=4, batch_size=32, query_len=1, context_len=2048,
        notes="Canonical example. Decode pass: batch 32, one new token each.",
    ),
    Fixture(
        name="m3_tp4_6l_cuda_prefill",
        trace="m3_tp4_6l_cuda_prefill.json.gz",
        config=M3, device="cuda", phase="prefill",
        tp_size=4, batch_size=1, query_len=4096, context_len=2048,
        has_module_spans=False,
        notes="Archived CUDA reference; predates the span hooks, so module "
              "names come from class frames. Not reproducible on this host.",
    ),
    Fixture(
        name="m3_tp4_6l_cuda_decode",
        trace="m3_tp4_6l_cuda_decode.json.gz",
        config=M3, device="cuda", phase="decode",
        tp_size=4, batch_size=32, query_len=1, context_len=2048,
        has_module_spans=False,
        notes="Archived CUDA reference. Covers the driver-launched Triton MoE "
              "grouped GEMM and the FlashInfer norms.",
    ),
)

BY_NAME = {f.name: f for f in FIXTURES}


def available() -> list[Fixture]:
    """Fixtures whose trace file is actually present."""
    return [f for f in FIXTURES if os.path.exists(f.trace_path)]
