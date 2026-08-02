# SPDX-License-Identifier: Apache-2.0
"""The data contracts of the replay benchmark, and their one set of names.

A case travels through four representations — a matrix row, a replay spec, a
measured record, a ranked target — and a value that changes name on the way is
a value a reader has to re-identify at every step. The analytic byte count did
exactly that: ``Memory (bytes)`` in the matrix, ``nbytes`` on the case,
``bytes`` in the record. It is ``nbytes`` everywhere now, and the matrix's
column header is the only place a display name appears.

The record builder lives here for the same reason: the direct worker and the
collective worker each built the measured record independently, so a field
added to one silently did not exist in the other.
"""
from __future__ import annotations

from typing import Any

#: The matrix column that carries each analytic quantity. The matrix is a
#: *display* artifact (it becomes a spreadsheet sheet), so its headers are
#: prose; everything downstream uses the field name.
MATRIX_COLUMNS = {
    "nbytes": "Memory (bytes)",
    "flops": "FLOPs",
    "op": "Op Name",
    "phase": "Phase",
    "seq_len": "Seq Len",
    "ctx_len": "Ctx Len",
    "batch_size": "Batch Size",
    "tp": "TP",
    "layers": "Layers",
    "module": "Module",
    "backend": "Backend",
}


def matrix_get(row: dict, field: str, default: Any = None) -> Any:
    """Read an analytic field from a matrix row by its *field* name."""
    return row.get(MATRIX_COLUMNS.get(field, field), default)


#: Fields every measured record carries, whatever measured it. ``value(case)``
#: reads it off the replay spec.
_CASE_FIELDS: tuple[tuple[str, str], ...] = (
    ("case_id", "case_id"), ("op", "op"), ("shape", "shape_label"),
    ("device", "device"), ("phase", "phase"), ("seq_len", "seq_len"),
    ("ctx_len", "ctx_len"), ("batch_size", "batch_size"),
    ("points", "points"), ("tp", "tp"), ("module", "module"),
    ("role", "role"), ("backend", "backend"), ("layers", "layers"),
    ("flops", "flops"), ("nbytes", "nbytes"),
    ("traced_device_time_us", "traced_device_time_us"),
    ("traced_comparable", "traced_comparable"),
)

#: Fields copied off a :class:`breakdown.bench.timing.Measurement`.
_MEASUREMENT_FIELDS = ("latency_us", "mean_us", "min_us", "p10_us", "p90_us",
                       "stdev_us", "iters", "reps", "windows", "overhead_us",
                       "notes")


def case_record(case: Any, status: str, shape_key: str,
                measurement: Any = None, error: str = "", detail: str = "",
                **extra: Any) -> dict[str, Any]:
    """One measured record, built the same way by every measurement path."""
    rec: dict[str, Any] = {name: getattr(case, attr)
                           for name, attr in _CASE_FIELDS}
    rec["shape_key"] = shape_key
    rec["status"] = status
    rec["error"] = error
    rec["detail"] = detail
    rec.update(extra)
    if measurement is not None:
        rec.update({f: getattr(measurement, f) for f in _MEASUREMENT_FIELDS})
    return rec
