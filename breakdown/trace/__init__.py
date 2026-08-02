# SPDX-License-Identifier: Apache-2.0
"""Profile-first model-graph reconstruction.

Rebuilds the model's module/op tree *directly from a torch profiler trace*
rather than deriving it statically from the HuggingFace config. The trace is
the ground truth for what actually executed, so the reconstruction tracks
whatever vLLM and the backends dispatched - there is no hardcoded structure to
drift.

What the capture provides (``with_stack=True``, ``record_shapes=True``, plus
this project's own capture-time hooks):

* ``module::<qname>::<Cls>`` spans (:mod:`breakdown.module_hooks`) - the real
  module hierarchy with real attribute names. A trace without them falls back
  to the class-only ``nn.Module: <Cls>_<idx>`` frames.
* ``cpu_op`` events carrying ``Input Dims`` / ``Input type`` / ``Concrete
  Inputs`` - a dispatched op's operands.
* ``kernel::<...>`` spans (:mod:`breakdown.kernel_hooks`) - the operands of a
  kernel launched straight from Python, which leaves no ``cpu_op``.
* ``kernel`` / ``gpu_memcpy`` events, attributed to the module or op that
  launched them by **launch-site containment** (``kernel.correlation`` -> the
  runtime launch call -> the enclosing node on the worker thread).

The modules, in pipeline order:

===========  ===================================================
``rules``    model/backend vocabulary - the only place names live
``events``   reading the trace file
``forest``   the time-containment tree of modules and ops
``kernels``  device time -> leaf ops
``shapes``   span-less shape fallbacks
``phases``   steps -> prefill / decode
``symbols``  concrete dims -> symbolic expressions
``collapse`` merge instances, name children, collapse repeats
``graph``    the orchestration
===========  ===================================================
"""
from __future__ import annotations

from .graph import build_forest, build_graph_from_trace, kernel_coverage

__all__ = ["build_forest", "build_graph_from_trace", "kernel_coverage"]
