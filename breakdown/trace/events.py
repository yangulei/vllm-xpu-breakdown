# SPDX-License-Identifier: Apache-2.0
"""Reading a chrome trace: the file, the argument slots, the worker thread.
"""
from __future__ import annotations

import ast
import gzip
import json
import re

from typing import Any
from ..cost import DTYPE_BYTES
from ..trace_common import MODULE_SPAN_PREFIX


# ===================================================================
# Trace loading + low-level event extraction
# ===================================================================

def _load_trace(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _normalize_dtype(t: Any) -> str:
    """Normalize a trace dtype token to a ``DTYPE_BYTES`` key.

    ``'c10::BFloat16' → 'bfloat16'``, ``'Float' → 'float'``,
    ``'c10::Float8_e4m3fn' → 'float8_e4m3fn'``. Unknown tokens are lowered and
    returned as-is (``dtype_size`` then falls back to 2 bytes).
    """
    if not t:
        return ""
    name = str(t).split("::")[-1].lower().replace("torch.", "")
    name = name.replace("half", "float16")
    return name


def _parse_input_args(args: dict) -> list[dict]:
    """Full ordered argument slots of a ``cpu_op``, for benchmark replay.

    Unlike :func:`_parse_input_dims_types` — which keeps only the tensor
    operands, because that is all the shape/memory analysis needs — this keeps
    **every** slot in its original position, so
    :mod:`breakdown.bench` can rebuild the call: which slot is a tensor (with
    its dims/dtype/strides), which is a ``TensorList``, and which is a scalar /
    ``None`` (with the value the profiler recorded in ``Concrete Inputs``).

    Slot kinds: ``tensor`` | ``tensorlist`` | ``scalar`` | ``none``.
    """
    dims = args.get("Input Dims")
    if not isinstance(dims, (list, tuple)):
        return []
    types = args.get("Input type")
    types = types if isinstance(types, (list, tuple)) else []
    strides = args.get("Input Strides")
    strides = strides if isinstance(strides, (list, tuple)) else []
    concrete = args.get("Concrete Inputs")
    concrete = concrete if isinstance(concrete, (list, tuple)) else []

    def at(seq, i):
        return seq[i] if i < len(seq) else None

    def ints(v) -> list[int]:
        if not isinstance(v, (list, tuple)):
            return []
        return [int(d) for d in v if isinstance(d, (int, float))]

    out: list[dict] = []
    for i, entry in enumerate(dims):
        raw_t = at(types, i)
        dt = _normalize_dtype(raw_t)
        st = at(strides, i)
        val = at(concrete, i)
        val = "" if val is None else str(val)
        if isinstance(entry, (list, tuple)) and entry and all(
                isinstance(x, (list, tuple)) for x in entry):
            # TensorList: one nesting level deeper (c10d::allreduce_, foreach).
            elem_dt = dt if dt in DTYPE_BYTES else ""
            items = []
            for j, sub in enumerate(entry):
                items.append({
                    "dims": ints(sub), "dtype": elem_dt,
                    "strides": ints(at(st, j) if isinstance(st, (list, tuple))
                                    else None),
                })
            out.append({"kind": "tensorlist", "items": items})
        elif isinstance(entry, (list, tuple)) and entry:
            out.append({"kind": "tensor", "dims": ints(entry), "dtype": dt,
                        "strides": ints(st)})
        elif dt in DTYPE_BYTES:
            # A 0-dim tensor: empty dims but a real dtype token.
            out.append({"kind": "tensor", "dims": [], "dtype": dt,
                        "strides": []})
        elif raw_t:
            out.append({"kind": "scalar", "type": str(raw_t), "value": val})
        else:
            out.append({"kind": "none", "value": val})
    return out


def _parse_input_dims_types(args: dict) -> tuple[list[list[int]], list[str]]:
    """Extract numeric input shapes and per-tensor dtypes, kept aligned.

    Only non-empty tensor inputs are retained (scalars / empty inputs dropped);
    the returned dtype list is parallel to the shape list, so ``dtypes[i]`` is
    the recorded dtype of the tensor with shape ``shapes[i]``.
    """
    dims = args.get("Input Dims")
    if dims is None:
        raw = args.get("input_shapes")
        if isinstance(raw, str):
            try:
                dims = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                dims = None
    if not isinstance(dims, (list, tuple)):
        return [], []
    types = args.get("Input type")
    if not isinstance(types, (list, tuple)):
        types = []
    shapes: list[list[int]] = []
    dtypes: list[str] = []
    for i, tensor in enumerate(dims):
        if not isinstance(tensor, (list, tuple)) or not tensor:
            continue
        raw_dt = _normalize_dtype(types[i] if i < len(types) else "")
        # A ``TensorList`` input nests one level deeper: its "Input Dims" entry
        # is a *list of per-tensor shape lists* rather than a single shape, e.g.
        # ``c10d::allreduce_`` records ``[[2, 6144]]`` (a one-element list of
        # tensors) with ``Input type`` ``'TensorList'``. Without unwrapping the
        # container level the whole entry was dropped, so collective/foreach ops
        # surfaced with no shape and no dtype. Surface each contained tensor's
        # shape; the container label is not an element dtype, so leave the dtype
        # empty for the residual-stream inference pass to fill from a neighbour.
        if all(isinstance(t, (list, tuple)) for t in tensor):
            list_dt = "" if raw_dt not in DTYPE_BYTES else raw_dt
            for sub in tensor:
                shape = [int(d) for d in sub if isinstance(d, (int, float))]
                if shape:
                    shapes.append(shape)
                    dtypes.append(list_dt)
            continue
        shape = [int(d) for d in tensor if isinstance(d, (int, float))]
        if shape:
            shapes.append(shape)
            dtypes.append(raw_dt)
    return shapes, dtypes


#: The launch-machinery vocabulary lives in :mod:`breakdown.trace_common` so the
#: capture-time hook (:mod:`breakdown.kernel_hooks`, which walks the *live*
#: stack) and this reader (which walks recorded ``python_function`` events) pick
#: the same frame by the same rule.

_PY_FRAME_RE = re.compile(r"^(?P<file>.+\.py)\((?P<line>\d+)\): (?P<func>.+)$")


def _worker_tid(events: list[dict]) -> tuple[Any, bool]:
    """The thread that ran the model forward, and whether it has module spans.

    When capture-time ``module::`` spans are present the thread carrying them
    is an unambiguous anchor; otherwise the busiest ``cpu_op`` thread is the
    best available guess (several threads dispatch ops under tensor
    parallelism). Returns ``(None, False)`` for a trace with no ``cpu_op``.
    """
    span_counts: dict[Any, int] = {}
    for e in events:
        if (e.get("ph") == "X" and e.get("cat") == "user_annotation"
                and str(e.get("name", "")).startswith(MODULE_SPAN_PREFIX)):
            span_counts[e.get("tid")] = span_counts.get(e.get("tid"), 0) + 1
    if span_counts:
        return max(span_counts, key=span_counts.get), True
    counts: dict[Any, int] = {}
    for e in events:
        if e.get("cat") == "cpu_op" and e.get("ph") == "X":
            counts[e.get("tid")] = counts.get(e.get("tid"), 0) + 1
    if not counts:
        return None, False
    return max(counts, key=counts.get), False
