# SPDX-License-Identifier: Apache-2.0
"""Turn a ranked optimization target into a Copilot CLI kernel session.

The benchmark answers *which kernel is worth an optimization session*; this
package is the handoff that opens it. It contains **no** optimization strategy
of its own: the brief is built from ``targets.json`` (the contract the ranker
already produces) and handed to the ``xpu-kernel-optimizer`` skill, which owns
the Profile -> Analyze -> Optimize -> Validate loop.

Layout::

    prompt.py     target record -> markdown brief (+ launchability rules)
    session.py    session record, argv, output/optimize/<run_id>/<op>/ layout
    scheduler.py  the GPU pool: one session owns one device, surplus queues
    manager.py    spawn/track/stop the per-op copilot processes
    cli.py        python -m breakdown.optimize {prompt,start,status,stop}
"""
from __future__ import annotations

from .prompt import (
    OPTIMIZER_SKILL,
    build_prompt,
    candidates,
    launchability,
    targets_by_op,
)
from .scheduler import DevicePool, LeaseError
from .session import (
    OptimizeSession,
    default_workspace_root,
    optimize_root,
    resolve_copilot,
    session_argv,
    session_paths,
)

__all__ = [
    "OPTIMIZER_SKILL",
    "DevicePool",
    "LeaseError",
    "OptimizeSession",
    "build_prompt",
    "candidates",
    "default_workspace_root",
    "launchability",
    "optimize_root",
    "resolve_copilot",
    "session_argv",
    "session_paths",
    "targets_by_op",
]
