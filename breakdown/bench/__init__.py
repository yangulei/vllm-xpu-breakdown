# SPDX-License-Identifier: Apache-2.0
"""Native op-replay benchmark: breakdown -> benchmark -> optimization target.

The profiler trace already records, for every op vLLM dispatched, its exact
dispatch name, per-tensor shapes/dtypes/strides and the concrete values of its
non-tensor arguments. So instead of translating each op into a *substitute*
kernel in an external benchmark suite, this package **re-invokes the op that
actually ran**: it resolves the dispatch name to its callable
(:mod:`~breakdown.bench.resolve`), materializes the recorded operands
(:mod:`~breakdown.bench.inputs`) and times it on device
(:mod:`~breakdown.bench.timing`).

Coverage is therefore a property of the profile, not of a hand-maintained
adapter table: every op the model dispatched is benchmarkable, on XPU and CUDA
alike.

Pipeline::

    reconstructed graph -> shape_matrix rows -> BenchCase specs -> replay
        -> results.jsonl -> rank -> targets.json / history

Every stage runs headless via ``python -m breakdown.bench``; ``/api/bench/*``
and the web UI are thin wrappers.
"""
from __future__ import annotations

from breakdown.bench.spec import BenchCase, build_cases, case_signature

__all__ = ["BenchCase", "build_cases", "case_signature"]
