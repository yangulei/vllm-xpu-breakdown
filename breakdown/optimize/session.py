# SPDX-License-Identifier: Apache-2.0
"""One optimization session: its record, its argv, and where it writes.

Artifacts live beside the benchmark's, one directory per (run, op)::

    output/optimize/<run_id>/index.json          the run's sessions
    output/optimize/<run_id>/<op>/prompt.md      the brief handed to the agent
    output/optimize/<run_id>/<op>/command.txt    the exact command, for a terminal
    output/optimize/<run_id>/<op>/session.log    the agent's stdout+stderr
    output/optimize/<run_id>/<op>/session.json   state, device lease, exit code
    output/optimize/<run_id>/<op>/summary.md     written by the agent itself

``command.txt`` exists because spawning is a convenience, not a requirement:
the same session can always be run by hand, which is the fallback when the
server should not hold a long-lived agent.
"""
from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

from ..bench.store import _SAFE

#: States a session moves through. ``pending`` means "queued for a GPU".
STATES = ("pending", "running", "done", "failed", "stopped")


def optimize_root(base: str | None = None) -> str:
    """``output/optimize`` (override with ``$BREAKDOWN_OPTIMIZE_ROOT``)."""
    if base:
        return base
    env = os.environ.get("BREAKDOWN_OPTIMIZE_ROOT")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    return os.path.join(repo, "output", "optimize")


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def default_workspace_root() -> str:
    """The parent of this repo - where the kernel repos live.

    ``kernel_sources.json`` paths are relative to it, so it is the only
    directory from which a session can follow them.
    """
    env = os.environ.get("BREAKDOWN_WORKSPACE_ROOT")
    if env:
        return env
    return os.path.dirname(repo_root())


def op_slug(op: str) -> str:
    """A filesystem-safe directory name for a dispatch name."""
    return _SAFE.sub("-", op or "op").strip("-") or "op"


@dataclass
class SessionPaths:
    root: str
    run_id: str
    op: str

    @property
    def run_dir(self) -> str:
        return os.path.join(self.root, self.run_id)

    @property
    def dir(self) -> str:
        return os.path.join(self.run_dir, op_slug(self.op))

    @property
    def prompt(self) -> str:
        return os.path.join(self.dir, "prompt.md")

    @property
    def command(self) -> str:
        return os.path.join(self.dir, "command.txt")

    @property
    def log(self) -> str:
        return os.path.join(self.dir, "session.log")

    @property
    def state(self) -> str:
        return os.path.join(self.dir, "session.json")

    @property
    def summary(self) -> str:
        return os.path.join(self.dir, "summary.md")

    @property
    def index(self) -> str:
        return os.path.join(self.run_dir, "index.json")

    def ensure(self) -> "SessionPaths":
        os.makedirs(self.dir, exist_ok=True)
        return self


def session_paths(run_id: str, op: str, base: str | None = None) -> SessionPaths:
    return SessionPaths(optimize_root(base), run_id, op)


def resolve_copilot() -> str | None:
    """The Copilot CLI binary (``$COPILOT_BIN`` overrides), or ``None``.

    An override that does not exist resolves to ``None`` rather than being
    handed to ``Popen``: a missing binary should be one clear message, not a
    per-session ``FileNotFoundError`` after the GPUs were already leased.
    """
    override = os.environ.get("COPILOT_BIN")
    if override:
        if os.path.isabs(override):
            return override if os.access(override, os.X_OK) else None
        return shutil.which(override)
    return shutil.which("copilot")


def session_argv(prompt_text: str, *, binary: str | None = None,
                 log_dir: str | None = None) -> list[str]:
    """The headless Copilot CLI invocation for a session.

    ``-p`` takes the prompt *text*, so the brief is passed inline. Colour is
    disabled because the output is captured to a log file, not a terminal.
    """
    argv = [binary or resolve_copilot() or "copilot",
            "-p", prompt_text,
            "--allow-all-tools", "--allow-all-paths", "--no-color"]
    if log_dir:
        argv += ["--log-dir", log_dir]
    return argv


def command_line(argv: list[str], *, cwd: str | None = None,
                 env: dict[str, str] | None = None,
                 prompt_file: str | None = None) -> str:
    """The argv as a copy-pasteable shell line, env prefix included.

    The brief is thousands of characters, so the rendered line reads it from
    ``prompt.md`` instead of inlining it - otherwise the fallback command is
    not something a human can paste.
    """
    shown = list(argv)
    if prompt_file:
        for i, arg in enumerate(shown):
            if arg == "-p" and i + 1 < len(shown):
                shown[i + 1] = "\0PROMPT\0"
                break
    parts: list[str] = []
    if cwd:
        parts.append(f"cd {shlex.quote(cwd)} && ")
    for key, value in (env or {}).items():
        parts.append(f"{key}={shlex.quote(value)} ")
    line = "".join(parts) + " ".join(shlex.quote(a) for a in shown)
    if prompt_file:
        line = line.replace(shlex.quote("\0PROMPT\0"),
                            f'"$(cat {shlex.quote(prompt_file)})"')
    return line


@dataclass
class OptimizeSession:
    """One kernel optimization session and everything observable about it."""

    run_id: str
    op: str
    phase: str | None = None
    state: str = "pending"
    reason: str = ""
    #: Device indexes leased exclusively for this session.
    device_ids: list[int] = field(default_factory=list)
    device_kind: str = ""
    #: How many devices it needs (``tp`` of the profiled run).
    need_devices: int = 1
    queue_position: int | None = None
    pid: int | None = None
    exit_code: int | None = None
    started: str | None = None
    ended: str | None = None
    cwd: str = ""
    #: Where this session's artifacts live, resolved once when it is created.
    #: A background thread writes state after the session ends, so re-reading
    #: ``$BREAKDOWN_OPTIMIZE_ROOT`` at write time could land it elsewhere.
    root: str = ""
    argv: list[str] = field(default_factory=list)
    command: str = ""
    prompt_file: str = ""
    log_file: str = ""
    error: str = ""
    #: A trimmed copy of the ranked target, so the session is self-describing
    #: even after the benchmark run is re-ranked.
    target: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def active(self) -> bool:
        return self.state in ("pending", "running")
