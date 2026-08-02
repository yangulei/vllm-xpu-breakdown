# SPDX-License-Identifier: Apache-2.0
"""The GPU pool: one session owns one device, exclusively.

An optimization session profiles and benchmarks continuously. Two agents
sharing a device would measure each other's interference and could accept a
change on a false number - the same reason the replay benchmark runs one op per
process. So a session is admitted only when a free device index exists, holds
it for its whole lifetime, and releases it on exit; the surplus waits in FIFO
order. Concurrency is therefore the size of the device pool, never a guessed
parallelism knob.

The lease is *enforced*, not advisory: the child's environment gets
``ZE_AFFINITY_MASK`` / ``CUDA_VISIBLE_DEVICES`` for exactly the leased indexes,
so everything the agent launches - builds, ``bench_cmd``, ``unitrace`` runs -
inherits the single-device view.
"""
from __future__ import annotations

import threading
from typing import Any

from ..bench import devices as bench_devices


class LeaseError(RuntimeError):
    """A lease can never be granted (not merely 'not yet')."""


class DevicePool:
    """Exclusive device leases with a FIFO wait list.

    Not a queue runner: it hands out and takes back leases, and the manager
    decides what to start. Keeping the policy here and the process handling in
    the manager is what makes the "one GPU, one session" rule testable without
    a GPU or a subprocess.
    """

    def __init__(self, kind: str, ids: list[int] | None = None) -> None:
        self.kind = kind
        avail = bench_devices.available(kind)
        self.all_ids: list[int] = list(ids) if ids else list(avail["indexes"])
        self._free: list[int] = list(self.all_ids)
        self._leases: dict[str, list[int]] = {}
        self._lock = threading.RLock()

    # -- introspection -------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.all_ids)

    def free_ids(self) -> list[int]:
        with self._lock:
            return list(self._free)

    def leases(self) -> dict[str, list[int]]:
        with self._lock:
            return {k: list(v) for k, v in self._leases.items()}

    def leased_by(self, key: str) -> list[int]:
        with self._lock:
            return list(self._leases.get(key, []))

    # -- leasing -------------------------------------------------------
    def acquire(self, key: str, need: int = 1) -> list[int] | None:
        """Lease ``need`` devices for ``key``; ``None`` if none are free yet.

        Raises :class:`LeaseError` when the pool could *never* satisfy the
        request, so an impossible session fails immediately instead of waiting
        forever behind a queue.
        """
        need = max(1, int(need))
        with self._lock:
            if need > self.size:
                raise LeaseError(
                    f"this op needs {need} {self.kind} devices but only "
                    f"{self.size} {'is' if self.size == 1 else 'are'} selected")
            if key in self._leases:
                return list(self._leases[key])
            if len(self._free) < need:
                return None
            got = [self._free.pop(0) for _ in range(need)]
            self._leases[key] = got
            return list(got)

    def release(self, key: str) -> list[int]:
        """Give a session's devices back; returns what was released."""
        with self._lock:
            ids = self._leases.pop(key, [])
            for i in ids:
                if i not in self._free:
                    self._free.append(i)
            self._free.sort()
            return ids

    def release_all(self) -> None:
        with self._lock:
            self._leases.clear()
            self._free = list(self.all_ids)

    def env_for(self, ids: list[int]) -> dict[str, str]:
        """The visibility environment that pins a child to ``ids``."""
        return bench_devices.visibility_env(self.kind, ids)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"kind": self.kind, "ids": list(self.all_ids),
                    "free": list(self._free), "leases": self.leases()}


def build_pool(kind: str | None = None,
               ids: list[int] | None = None) -> tuple[DevicePool, str | None]:
    """A pool for a device selection, or the reason it cannot be used."""
    kind = kind or bench_devices.detect_device()
    ids = list(ids or [])
    err = bench_devices.validate_device_ids(ids, kind)
    if err:
        return DevicePool(kind, ids or None), err
    return DevicePool(kind, ids or None), None
