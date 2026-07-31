# SPDX-License-Identifier: Apache-2.0
"""Measure one replayed call on device.

Each iteration is timed **individually** with device events, and everything
that is not the kernel - restoring operands a mutating op overwrote, flushing
the cache - happens between the end event of one iteration and the start event
of the next, so it never lands inside a measurement.

That per-iteration framing matters for replay specifically: most vLLM custom
ops write into an out-parameter or update a cache in place, so a naive "run it
N times inside one timing window" loop both accumulates garbage in the operands
(``mul_`` decays its input to denormals, ``add_`` diverges) and hides the
per-call launch cost.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

#: Total time budget per case: how long the whole measurement may take.
DEFAULT_BUDGET_S = 0.5
#: Kernel repetitions inside one timed window. Device event windows carry a
#: large fixed cost (~60-90 us on Level Zero), so a single-call window measures
#: the timer, not the kernel; repeating inside the window amortizes it.
TARGET_WINDOW_S = 0.02
MIN_REPS = 1
MAX_REPS = 2000
MIN_WINDOWS = 3
MAX_WINDOWS = 30
WARMUP_ITERS = 3

#: Bytes of scratch written to evict the last-level cache before a window.
#: Sized above BMG's 8 MB / Blackwell's 32 MB L2 so a small op does not measure
#: a cache-resident replay of the previous window.
FLUSH_BYTES = 64 << 20


@dataclass
class Measurement:
    ok: bool = True
    iters: int = 0
    reps: int = 0
    windows: int = 0
    warmup: int = 0
    latency_us: float = 0.0            # median device time per call
    mean_us: float = 0.0
    min_us: float = 0.0
    p10_us: float = 0.0
    p90_us: float = 0.0
    stdev_us: float = 0.0
    wall_us: float = 0.0
    overhead_us: float = 0.0           # measured empty-window cost, subtracted
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sync(device: str) -> None:
    import torch
    mod = getattr(torch, device, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def _event_pair(device: str):
    """A timing event pair, or ``(None, None)`` when the backend has none.

    CPU (and some accelerator builds) expose an ``Event`` class that takes no
    timing argument; falling back to wall-clock timing there is correct, and
    must not raise out of the measurement.
    """
    import torch
    if device == "cpu":
        return None, None
    mod = getattr(torch, device, None)
    ev = getattr(mod, "Event", None) if mod is not None else None
    if ev is None:
        return None, None
    try:
        return ev(enable_timing=True), ev(enable_timing=True)
    except Exception:                      # noqa: BLE001 - backend-specific
        try:
            return ev(True), ev(True)
        except Exception:                  # noqa: BLE001
            return None, None


class CacheFlusher:
    """Writes a buffer larger than the LLC, to defeat cache-resident replays."""

    def __init__(self, device: str, nbytes: int = FLUSH_BYTES):
        self.buf = None
        if device == "cpu":
            return
        try:
            import torch
            self.buf = torch.empty(nbytes // 4, dtype=torch.float32,
                                   device=device)
        except Exception:                  # noqa: BLE001 - flushing is optional
            self.buf = None

    def flush(self) -> None:
        if self.buf is not None:
            self.buf.zero_()


def make_restorer(mutated: list[Any]) -> Callable[[], None] | None:
    """Restore the operands a mutating op overwrites, between windows.

    Without this, window *N* of an in-place op does not do the same work as
    window 1 (values drift into denormals/inf, an accumulating buffer fills
    up), so the median would describe a different kernel than the one profiled.
    """
    saved: list[tuple[Any, Any]] = []
    for a in mutated or []:
        if a is None:
            continue
        if isinstance(a, (list, tuple)):
            for t in a:
                if hasattr(t, "clone"):
                    saved.append((t, t.clone()))
        elif hasattr(a, "clone"):
            saved.append((a, a.clone()))
    if not saved:
        return None

    def restore() -> None:
        for dst, src in saved:
            dst.copy_(src)

    return restore


def probe(call: Callable[[], Any], device: str,
          restore: Callable[[], None] | None = None) -> float:
    """One timed call after a warmup, in seconds - sets the window sizing.

    ``restore`` runs before *every* probe call. Without it the probe is the one
    place an accumulating op (``remap_hidden_states``, whose ``rows_per_expert``
    grows with atomics) would run twice un-reset - exactly what the op's
    single-call-per-window recipe exists to prevent.
    """
    if restore:
        restore()
    call()
    _sync(device)
    if restore:
        restore()
    t0 = time.perf_counter()
    call()
    _sync(device)
    return max(time.perf_counter() - t0, 1e-9)


#: Cached per (device) empty-window cost. Measuring it once per process is
#: enough: it is a property of the runtime's event machinery, not of the op.
_OVERHEAD_US: dict[str, float] = {}


def window_overhead_us(device: str, samples: int = 20) -> float:
    """Cost of an empty timed window, in microseconds.

    Device event windows are not free: on Level Zero an empty
    ``record / record / elapsed_time`` measures tens of microseconds, which is
    an order of magnitude more than a small elementwise kernel. Subtracting the
    measured floor (and repeating the kernel inside the window) is what makes a
    3 us norm distinguishable from a 6 us one.
    """
    if device in _OVERHEAD_US:
        return _OVERHEAD_US[device]
    start_ev, end_ev = _event_pair(device)
    if start_ev is None:
        _OVERHEAD_US[device] = 0.0
        return 0.0
    vals = []
    for _ in range(samples):
        _sync(device)
        start_ev.record()
        end_ev.record()
        _sync(device)
        vals.append(float(start_ev.elapsed_time(end_ev)) * 1000.0)
    _OVERHEAD_US[device] = round(min(vals), 3)
    return _OVERHEAD_US[device]


def plan_window(seconds: float, budget: float = DEFAULT_BUDGET_S
                ) -> tuple[int, int]:
    """``(reps per window, windows)`` for a kernel of ``seconds``."""
    reps = int(TARGET_WINDOW_S / max(seconds, 1e-9))
    reps = max(MIN_REPS, min(MAX_REPS, reps))
    per_window = reps * seconds
    windows = int(budget / max(per_window, 1e-9))
    windows = max(MIN_WINDOWS, min(MAX_WINDOWS, windows))
    return reps, windows


def measure(fn: Callable[..., Any], args: list[Any], device: str,
            kwargs: dict[str, Any] | None = None,
            mutated: list[Any] | None = None, reps: int | None = None,
            windows: int | None = None, warmup: int = WARMUP_ITERS,
            budget: float = DEFAULT_BUDGET_S,
            flush_cache: bool = True) -> Measurement:
    """Time ``fn(*args)`` on ``device`` and summarize the distribution.

    Each window runs the kernel ``reps`` times between one pair of device
    events; the empty-window cost is subtracted and the remainder divided by
    ``reps``. Operand restoration and cache flushing happen *between* windows,
    never inside one.
    """
    m = Measurement(warmup=warmup)
    kwargs = kwargs or {}

    def call():
        return fn(*args, **kwargs)

    restore = make_restorer(mutated or [])
    flusher = CacheFlusher(device) if flush_cache else None
    try:
        for _ in range(max(warmup, 1)):
            if restore:
                restore()
            call()
        _sync(device)
    except Exception as exc:               # noqa: BLE001 - reported per case
        m.ok = False
        m.error = f"{type(exc).__name__}: {exc}"
        return m

    if reps is None or windows is None:
        try:
            secs = probe(call, device, restore)
        except Exception as exc:           # noqa: BLE001
            m.ok = False
            m.error = f"{type(exc).__name__}: {exc}"
            return m
        auto_reps, auto_windows = plan_window(secs, budget)
        reps = reps or auto_reps
        windows = windows or auto_windows
    m.reps, m.windows = reps, windows
    m.iters = reps * windows
    m.overhead_us = window_overhead_us(device)

    start_ev, end_ev = _event_pair(device)
    samples: list[float] = []
    wall0 = time.perf_counter()
    try:
        for _ in range(windows):
            if restore:
                restore()
            if flusher:
                flusher.flush()
            _sync(device)
            if start_ev is not None:
                start_ev.record()
                for _ in range(reps):
                    call()
                end_ev.record()
                _sync(device)
                elapsed = float(start_ev.elapsed_time(end_ev)) * 1000.0
                samples.append(max(elapsed - m.overhead_us, 0.0) / reps)
            else:
                t0 = time.perf_counter()
                for _ in range(reps):
                    call()
                _sync(device)
                samples.append((time.perf_counter() - t0) * 1e6 / reps)
    except Exception as exc:               # noqa: BLE001
        m.ok = False
        m.error = f"{type(exc).__name__}: {exc}"
        return m
    m.wall_us = (time.perf_counter() - wall0) * 1e6

    samples.sort()
    m.latency_us = round(statistics.median(samples), 3)
    m.mean_us = round(statistics.fmean(samples), 3)
    m.min_us = round(samples[0], 3)
    m.p10_us = round(samples[max(0, int(0.10 * len(samples)) - 1)], 3)
    m.p90_us = round(samples[min(len(samples) - 1, int(0.90 * len(samples)))], 3)
    m.stdev_us = round(statistics.pstdev(samples), 3) if len(samples) > 1 else 0.0
    if start_ev is None:
        m.notes.append("wall-clock timing (no device events on this backend)")
    return m
