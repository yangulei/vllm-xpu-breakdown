# SPDX-License-Identifier: Apache-2.0
"""Spawn, track and stop the per-kernel Copilot CLI sessions.

One process per selected kernel, each holding one GPU exclusively (see
:mod:`.scheduler`). Sessions beyond the device count stay ``pending`` in FIFO
order and are started by the release path when a GPU frees up, so the pool -
not a parallelism knob - is what bounds concurrency.

Everything a session produces is written under ``output/optimize/<run_id>/``
as it happens, so a server restart or a killed agent still leaves a readable
record of what ran, on which device, and how it ended.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

from ..bench import devices as bench_devices
from .prompt import build_prompt, launchability, targets_by_op
from .scheduler import DevicePool, LeaseError
from .session import (
    OptimizeSession,
    command_line,
    default_workspace_root,
    optimize_root,
    resolve_copilot,
    session_argv,
    session_paths,
)

logger = logging.getLogger("vllm_xpu_breakdown")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class OptimizeManager:
    """All optimization sessions of this server process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        #: (run_id, op) -> session
        self._sessions: dict[tuple[str, str], OptimizeSession] = {}
        self._procs: dict[tuple[str, str], subprocess.Popen] = {}
        #: The ranking document each session was created from, so its brief can
        #: be re-rendered once the device lease is known.
        self._docs: dict[tuple[str, str], dict[str, Any]] = {}
        self._binaries: dict[tuple[str, str], str | None] = {}
        self._queue: list[tuple[str, str]] = []
        self._pool: DevicePool | None = None

    # ---------------------------------------------------------------- state
    def sessions(self, run_id: str | None = None) -> list[OptimizeSession]:
        with self._lock:
            out = [s for k, s in self._sessions.items()
                   if run_id is None or k[0] == run_id]
        return sorted(out, key=lambda s: (s.state != "running",
                                          s.state != "pending", s.op))

    def get(self, run_id: str, op: str) -> OptimizeSession | None:
        with self._lock:
            return self._sessions.get((run_id, op))

    def pool_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._pool.snapshot() if self._pool else {}

    def any_active(self, run_id: str | None = None) -> bool:
        return any(s.active for s in self.sessions(run_id))

    # ---------------------------------------------------------------- start
    def start(self, *, run_id: str, doc: dict[str, Any], ops: Iterable[str],
              phase: str | None = None, workspace_root: str | None = None,
              device_kind: str | None = None,
              device_ids: list[int] | None = None,
              binary: str | None = None,
              spawn: bool = True) -> dict[str, Any]:
        """Queue one session per op; start as many as there are free GPUs."""
        ops = [op for op in ops if op]
        if not ops:
            raise ValueError("no ops selected")
        binary = binary or resolve_copilot()
        if spawn and not binary:
            raise FileNotFoundError(
                "the Copilot CLI ('copilot') was not found on PATH - install it "
                "or set $COPILOT_BIN, or use the copy-command fallback")

        kind = device_kind or bench_devices.detect_device()
        cwd = workspace_root or default_workspace_root()
        if not os.path.isdir(cwd):
            raise NotADirectoryError(f"workspace root '{cwd}' does not exist")

        by_op = targets_by_op(doc, phase)
        tp = int(doc.get("tp") or 1)
        # Resolved once: a session's artifacts must keep going to the same
        # place even if the environment changes while its agent is running.
        root = optimize_root()

        with self._lock:
            if self._pool is None or not self.any_active():
                self._pool = DevicePool(kind, device_ids or None)
            pool = self._pool
            if not pool.size:
                raise RuntimeError(
                    f"no {kind} devices are available to run a session on")

            created: list[OptimizeSession] = []
            for op in ops:
                key = (run_id, op)
                existing = self._sessions.get(key)
                if existing and existing.active:
                    raise RuntimeError(
                        f"an optimization session for '{op}' is already "
                        f"{existing.state}")
                target = by_op.get(op)
                if target is None:
                    raise KeyError(f"'{op}' is not a ranked target of {run_id}")
                can, reason = launchability(target)
                paths = session_paths(run_id, op, root).ensure()
                sess = OptimizeSession(
                    run_id=run_id, op=op, phase=phase, state="pending",
                    reason=reason, device_kind=kind,
                    need_devices=tp if _needs_all_ranks(target, tp) else 1,
                    cwd=cwd, root=root,
                    prompt_file=paths.prompt, log_file=paths.log,
                    target=_target_snapshot(target, can, reason))
                self._sessions[key] = sess
                self._procs.pop(key, None)
                self._docs[key] = doc
                self._binaries[key] = binary
                if key not in self._queue:
                    self._queue.append(key)
                created.append(sess)
                self._write_session(sess)

            self._plan_prompts(created, doc, phase, binary)
            if spawn:
                self._pump_locked()
            self._renumber_queue_locked()
            self._write_index(run_id)
            return {"run_id": run_id,
                    "sessions": [s.to_dict() for s in self.sessions(run_id)],
                    "pool": pool.snapshot()}

    def _plan_prompts(self, sessions: list[OptimizeSession],
                      doc: dict[str, Any], phase: str | None,
                      binary: str | None) -> None:
        """Write each session's brief and its copy-pasteable command."""
        for sess in sessions:
            self._render_locked(sess, binary)

    def _render_locked(self, sess: OptimizeSession,
                       binary: str | None = None) -> None:
        """(Re)write the brief, the argv and the fallback command line.

        Called once at queue time and again when the device lease is granted,
        because the brief names the device the session owns.
        """
        key = (sess.run_id, sess.op)
        doc = self._docs.get(key, {})
        binary = binary if binary is not None else self._binaries.get(key)
        paths = session_paths(sess.run_id, sess.op, sess.root or None).ensure()
        text = build_prompt(
            sess.target.get("target") or {}, doc,
            run_id=sess.run_id, phase=sess.phase,
            device_ids=sess.device_ids or None,
            device_kind=sess.device_kind,
            workspace_root=sess.cwd, artifact_dir=paths.dir)
        with open(paths.prompt, "w", encoding="utf-8") as fh:
            fh.write(text)
        sess.argv = session_argv(text, binary=binary)
        vis = (self._pool.env_for(sess.device_ids)
               if self._pool and sess.device_ids else {})
        sess.command = command_line(sess.argv, cwd=sess.cwd, env=vis,
                                    prompt_file=paths.prompt)
        with open(paths.command, "w", encoding="utf-8") as fh:
            fh.write(sess.command + "\n")
        self._write_session(sess)

    # ------------------------------------------------------------ scheduling
    def _pump_locked(self) -> None:
        """Start as many pending sessions as there are free devices."""
        pool = self._pool
        if pool is None:
            return
        for key in list(self._queue):
            sess = self._sessions.get(key)
            if sess is None or sess.state != "pending":
                self._queue.remove(key)
                continue
            try:
                ids = pool.acquire(f"{key[0]}::{key[1]}", sess.need_devices)
            except LeaseError as exc:
                self._fail_locked(sess, str(exc))
                self._queue.remove(key)
                continue
            if ids is None:
                continue          # no free GPU yet - stay queued, keep order
            self._queue.remove(key)
            self._spawn_locked(sess, ids)

    def _renumber_queue_locked(self) -> None:
        for pos, key in enumerate(self._queue):
            sess = self._sessions.get(key)
            if sess is not None:
                sess.queue_position = pos + 1
                self._write_session(sess)

    def _spawn_locked(self, sess: OptimizeSession, ids: list[int]) -> None:
        pool = self._pool
        assert pool is not None
        sess.device_ids = list(ids)
        sess.queue_position = None
        vis = pool.env_for(ids)
        paths = session_paths(sess.run_id, sess.op, sess.root or None)
        # The brief names the device it owns, so it is re-rendered now that the
        # lease is known - the agent must not run on a device it wasn't given.
        try:
            self._render_locked(sess)
        except OSError as exc:
            pool.release(f"{sess.run_id}::{sess.op}")
            self._fail_locked(sess, f"could not write the session brief: {exc}")
            return

        env = {**os.environ, **vis}
        try:
            # A session's log is its own: starting a new one truncates it, so
            # the streamed pane is this run's output and not a concatenation
            # of every session ever opened for this op.
            log = open(paths.log, "wb", buffering=0)
            proc = subprocess.Popen(  # noqa: S603 - argv is built, not shell
                sess.argv, cwd=sess.cwd, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            pool.release(f"{sess.run_id}::{sess.op}")
            self._fail_locked(sess, f"could not start the session: {exc}")
            return
        self._procs[(sess.run_id, sess.op)] = proc
        sess.pid = proc.pid
        sess.state = "running"
        sess.started = _now()
        self._write_session(sess)
        threading.Thread(target=self._wait, args=(sess.run_id, sess.op, proc, log),
                         daemon=True).start()
        logger.info("optimize: %s started on %s %s (pid %s)",
                    sess.op, sess.device_kind, ids, proc.pid)

    def _wait(self, run_id: str, op: str, proc: subprocess.Popen,
              log: Any) -> None:
        code = None
        try:
            code = proc.wait()
        finally:
            try:
                log.close()
            except OSError:
                pass
            with self._lock:
                sess = self._sessions.get((run_id, op))
                if sess is not None:
                    if sess.state != "stopped":
                        sess.state = "done" if code == 0 else "failed"
                        if code not in (0, None):
                            sess.error = sess.error or f"exit code {code}"
                    sess.exit_code = code
                    sess.ended = _now()
                    self._write_session(sess)
                if self._pool is not None:
                    self._pool.release(f"{run_id}::{op}")
                self._procs.pop((run_id, op), None)
                # A freed GPU is what starts the next queued kernel.
                self._pump_locked()
                self._renumber_queue_locked()
                self._write_index(run_id)
        logger.info("optimize: %s finished with %s", op, code)

    def _fail_locked(self, sess: OptimizeSession, error: str) -> None:
        sess.state = "failed"
        sess.error = error
        sess.ended = _now()
        sess.queue_position = None
        self._write_session(sess)

    # ----------------------------------------------------------------- stop
    def stop(self, run_id: str, op: str | None = None) -> list[str]:
        """Terminate running sessions / drop pending ones. Returns the ops."""
        stopped: list[str] = []
        with self._lock:
            for key, sess in list(self._sessions.items()):
                if key[0] != run_id or (op and key[1] != op):
                    continue
                if not sess.active:
                    continue
                proc = self._procs.get(key)
                sess.state = "stopped"
                sess.ended = _now()
                sess.queue_position = None
                if key in self._queue:
                    self._queue.remove(key)   # a pending session just leaves
                if proc is not None and proc.poll() is None:
                    _terminate(proc)
                else:
                    if self._pool is not None:
                        self._pool.release(f"{key[0]}::{key[1]}")
                self._write_session(sess)
                stopped.append(key[1])
            self._pump_locked()
            self._renumber_queue_locked()
            self._write_index(run_id)
        return stopped

    def shutdown(self) -> None:
        with self._lock:
            for key, proc in list(self._procs.items()):
                if proc.poll() is None:
                    _terminate(proc)
                self._procs.pop(key, None)
            self._queue.clear()
            if self._pool is not None:
                self._pool.release_all()

    # ------------------------------------------------------------ artifacts
    def _write_session(self, sess: OptimizeSession) -> None:
        paths = session_paths(sess.run_id, sess.op, sess.root or None)
        try:
            paths.ensure()
            data = sess.to_dict()
            data.pop("argv", None)     # it embeds the whole prompt
            with open(paths.state, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            logger.warning("optimize: could not write session state: %s", exc)

    def _write_index(self, run_id: str) -> None:
        of_run = self.sessions(run_id)
        paths = session_paths(run_id, "_index",
                              next((s.root for s in of_run if s.root), None))
        try:
            os.makedirs(paths.run_dir, exist_ok=True)
            sessions = []
            for sess in of_run:
                data = sess.to_dict()
                data.pop("argv", None)
                sessions.append(data)
            with open(paths.index, "w", encoding="utf-8") as fh:
                json.dump({"run_id": run_id, "updated": _now(),
                           "pool": self.pool_snapshot(),
                           "sessions": sessions}, fh, indent=2)
        except OSError as exc:
            logger.warning("optimize: could not write session index: %s", exc)


def _terminate(proc: subprocess.Popen) -> None:
    """Stop the agent and everything it launched (it has its own group)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass


def _needs_all_ranks(target: dict[str, Any], tp: int) -> bool:
    """Whether replaying this op needs the whole TP group (a collective does)."""
    if tp <= 1:
        return False
    backend = (target.get("backend") or "").lower()
    op = (target.get("op") or "").lower()
    return backend == "ccl" or op.startswith("c10d::") or "all_reduce" in op


def _target_snapshot(target: dict[str, Any], can: bool,
                     reason: str) -> dict[str, Any]:
    """What the session keeps of the ranking it was created from."""
    return {"target": target, "launchable": can, "reason": reason}


#: The server's manager. One per process, like the bench state.
MANAGER = OptimizeManager()
