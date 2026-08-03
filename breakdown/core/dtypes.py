# SPDX-License-Identifier: Apache-2.0
"""The one dtype table.

Five tables encoded this before, each a partial view of the same fact:

===========================  =====================================
``cost.DTYPE_BYTES``         token -> bytes per element
``bench.inputs.DTYPE_MAP``   token -> torch attribute name
``bench.spec._DTYPE_SHORT``  token -> short label ("f16")
``bench.runner._operand_bytes``  torch attribute -> bytes (6 of 20 types)
``shape_derive._FRIENDLY_DTYPE``  token -> short label ("fp16")
===========================  =====================================

They had drifted: the two label tables disagreed on the float types, and the
runner's private width table knew six dtypes, so an fp8 or int8 operand was
budgeted at one byte by accident rather than on purpose. A dtype is one thing;
it gets one record.

**Why the tokens are so various.** The profiler records ``Input type`` using
*C++* type names, so an index tensor arrives as ``long int``, not ``int64``.
Without those aliases every index and position operand was undercounted 4x.

**Why ``nbytes`` is a float.** ``int4`` packs two values per byte. It was
listed as 1 byte with a comment saying it should be 0.5, which overstated the
memory of every int4-quantized weight by 2x -- and therefore understated its
arithmetic intensity, which is what decides whether the ranking calls it
compute- or memory-bound. The comment is now the code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dtype:
    """One element type: what torch calls it, how wide it is, how it prints."""

    #: The ``torch.<name>`` attribute, which is also the canonical spelling.
    name: str
    #: Bytes per element. Fractional for sub-byte packed types.
    nbytes: float
    #: Short label for tables and shape strings.
    label: str


#: Canonical types, in a stable order.
DTYPES: tuple[Dtype, ...] = (
    Dtype("float64", 8, "fp64"),
    Dtype("float32", 4, "fp32"),
    Dtype("float16", 2, "fp16"),
    Dtype("bfloat16", 2, "bf16"),
    Dtype("float8_e4m3fn", 1, "fp8e4m3"),
    Dtype("float8_e5m2", 1, "fp8e5m2"),
    Dtype("float8_e4m3fnuz", 1, "fp8e4m3uz"),
    Dtype("int64", 8, "i64"),
    Dtype("int32", 4, "i32"),
    Dtype("int16", 2, "i16"),
    Dtype("int8", 1, "i8"),
    Dtype("uint8", 1, "u8"),
    Dtype("bool", 1, "b8"),
    # Packed two-per-byte; see the module docstring.
    Dtype("int4", 0.5, "i4"),
)

#: Every spelling the pipeline can meet -> canonical type. Includes the C++
#: type names the profiler emits and the abbreviations the UI and configs use.
_ALIASES: dict[str, str] = {
    "double": "float64", "fp64": "float64",
    "float": "float32", "fp32": "float32",
    "half": "float16", "fp16": "float16",
    "bf16": "bfloat16", "c10::bfloat16": "bfloat16",
    "c10::half": "float16",
    "fp8": "float8_e4m3fn", "c10::float8_e4m3fn": "float8_e4m3fn",
    "long int": "int64", "long": "int64", "long long": "int64",
    "unsigned long": "int64",
    "int": "int32", "unsigned int": "int32",
    "short": "int16", "unsigned short": "int16",
    "char": "int8", "signed char": "int8",
    "unsigned char": "uint8", "byte": "uint8",
}

_BY_NAME: dict[str, Dtype] = {d.name: d for d in DTYPES}
_BY_NAME.update({alias: _BY_NAME[canon] for alias, canon in _ALIASES.items()})

#: Assumed when a dtype is absent. Activations are half precision throughout.
DEFAULT = _BY_NAME["bfloat16"]

#: Integer types, i.e. the ones whose *values* are indices rather than data.
#: An integer operand cannot be filled with random noise -- it addresses memory
#: -- which is why the benchmark refuses one it has no synthesizer for.
_INTEGER = frozenset({"int64", "int32", "int16", "int8", "uint8", "int4"})


def normalize(token: str | None) -> str:
    """A dtype token reduced to its lookup key.

    Handles ``torch.bfloat16``/``BFloat16``/``c10::BFloat16`` alike, because
    the three arrive from the trace, from a config and from a schema.
    """
    return (token or "").strip().lower().replace("torch.", "")


def find(token: str | None) -> Dtype | None:
    """The dtype a token names, or ``None`` if it names none."""
    return _BY_NAME.get(normalize(token))


def is_known(token: str | None) -> bool:
    """True if the token names a dtype this pipeline understands.

    Reconstruction uses this to tell a recorded *dtype* apart from the other
    strings that share the ``Input type`` list (``ScalarType``, ``Device``, an
    empty slot for a non-tensor argument).
    """
    return normalize(token) in _BY_NAME


def size(token: str | None) -> float:
    """Bytes per element, defaulting to bf16 for an unrecognized token."""
    d = find(token)
    return (d or DEFAULT).nbytes


def torch_name(token: str | None) -> str:
    """The ``torch.<attr>`` spelling, for materializing an operand."""
    d = find(token)
    return (d or DEFAULT).name


def name_or(token: str | None, default: str) -> str:
    """The ``torch.<attr>`` spelling, or ``default`` if the token names none.

    Synthesizers differ in what an *absent* dtype should mean -- an index map
    defaults to ``int64``, a CUDA MoE sort buffer to ``int32`` -- so the
    default is the caller's to state rather than this table's to assume.
    """
    d = find(token)
    return d.name if d else default


def label(token: str | None) -> str:
    """Short display label; the original token if it names no dtype."""
    if not token:
        return ""
    d = find(token)
    return d.label if d else normalize(token)


def is_integer(token: str | None) -> bool:
    """True for index-like element types."""
    d = find(token)
    return d is not None and d.name in _INTEGER


#: The dtype a width implies, when a config gives bytes instead of a name.
#: Only these three are ambiguous in practice: a model states ``dtype_bytes``
#: and means the obvious float of that width.
_DEFAULT_FOR_BYTES: dict[int, str] = {8: "fp64", 4: "fp32", 2: "bf16", 1: "fp8"}


def label_for_bytes(nbytes: int) -> str:
    """The conventional dtype label for an element width.

    Used where a config carries ``dtype_bytes`` rather than a dtype name.
    """
    return _DEFAULT_FOR_BYTES.get(int(nbytes), f"{int(nbytes) * 8}bit")
