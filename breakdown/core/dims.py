# SPDX-License-Identifier: Apache-2.0
"""Symbolic shape dimensions, parsed rather than string-substituted.

A reconstructed shape is a mix of integers and small expressions over the
model's symbols: ``H``, ``n_h·d/TP``, ``S+C``, ``topk·S``, ``2·I_moe``. Three
places used to interpret those strings and no two agreed:

* ``shape_derive._resolve_dim`` replaced every symbol name with its value by
  textual substitution -- longest name first, so ``2·I`` was not eaten by
  ``I`` -- and then ran the result through ``eval`` behind an AST whitelist.
* ``shape_derive._partially_resolve_dim`` re-implemented the same walk to keep
  ``S``/``B``/``C``/``TP`` symbolic, with its own splitting rules.
* the browser had a third implementation in JavaScript, which silently failed
  on ``S+C``: it split on ``·`` only, so an additive composite came back as
  the bare string and the tooltip showed no number at all.

The awkward part -- and the reason substitution was reached for -- is that
``·`` is *both* the multiply operator and a character inside symbol names:
``n_h·d`` is one registered symbol, while ``topk·S`` is a product of two. A
grammar alone cannot tell them apart. So the lexer consults the symbol table
and takes the **longest registered name** at each position, which is exactly
the invariant the old ``sorted(symbols, key=len, reverse=True)`` was
protecting, now stated where it can be tested.

Rendering is byte-identical to what reconstruction wrote, so the JSON contract
and the golden snapshots are unchanged; this module changes how the strings
are *read*, not how they look.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Symbols the sweep varies. Everything else is a model constant fixed by
#: ``config.json``, so these are the ones a display form keeps symbolic.
VARIABLES: frozenset[str] = frozenset({"S", "B", "C", "TP"})

#: The multiplication sign used in shape strings (U+00B7 MIDDLE DOT). It reads
#: as multiplication without colliding with ``*``, which pandas and Excel
#: would treat as a formula.
MUL = "·"


# ---------------------------------------------------------------------------
# The expression type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sym:
    """A named dimension: ``H``, ``n_h·d``, ``S``."""

    name: str


@dataclass(frozen=True)
class BinOp:
    """``left <op> right`` where op is one of ``·``, ``+``, ``/``."""

    op: str
    left: "Dim"
    right: "Dim"


Dim = int | Sym | BinOp


class ParseError(ValueError):
    """The text is not a dimension expression this module understands."""


# ---------------------------------------------------------------------------
# Lexing and parsing
# ---------------------------------------------------------------------------
_OPS = (MUL, "+", "/")


def _tokens(text: str, symbols) -> list[object]:
    """Split into symbols, integers and operators, longest symbol first.

    ``symbols`` is any container of known names. Taking the longest match is
    what separates the single symbol ``n_h·d`` from the product ``topk·S``:
    both contain a middle dot, and only the symbol table knows which is which.
    """
    names = sorted((n for n in symbols if n), key=len, reverse=True)
    out: list[object] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _OPS:
            out.append(ch)
            i += 1
            continue
        for name in names:
            if text.startswith(name, i):
                out.append(Sym(name))
                i += len(name)
                break
        else:
            j = i
            while j < n and text[j].isdigit():
                j += 1
            if j > i:
                out.append(int(text[i:j]))
                i = j
                continue
            # An unregistered name: take everything up to the next operator.
            j = i
            while j < n and text[j] not in _OPS and not text[j].isspace():
                j += 1
            if j == i:
                raise ParseError(f"cannot lex {text!r} at {i}")
            out.append(Sym(text[i:j]))
            i = j
    return out


def parse(text: str, symbols=()) -> Dim:
    """Parse a dimension expression.

    Precedence is the conventional one: ``+`` binds loosest, then ``·``, then
    ``/``. ``·`` and ``/`` are left-associative, which matters for ``a/b·c``.
    """
    toks = _tokens(str(text), symbols)
    if not toks:
        raise ParseError("empty dimension")
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def atom() -> Dim:
        nonlocal pos
        tok = peek()
        if tok is None or tok in _OPS:
            raise ParseError(f"expected a value in {text!r}")
        pos += 1
        return tok  # type: ignore[return-value]

    def product() -> Dim:
        nonlocal pos
        node = atom()
        while peek() in (MUL, "/"):
            op = toks[pos]
            pos += 1
            node = BinOp(str(op), node, atom())
        return node

    def sum_() -> Dim:
        nonlocal pos
        node = product()
        while peek() == "+":
            pos += 1
            node = BinOp("+", node, product())
        return node

    node = sum_()
    if pos != len(toks):
        raise ParseError(f"trailing input in {text!r}")
    return node


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(dim: Dim) -> str:
    """The expression as a shape string, in the spelling reconstruction uses.

    No parentheses are emitted: every expression the pipeline builds is a flat
    left-associative chain, so the conventional precedence reads it back
    unchanged. An expression that would need brackets is not one this pipeline
    produces.
    """
    if isinstance(dim, int):
        return str(dim)
    if isinstance(dim, Sym):
        return dim.name
    return f"{render(dim.left)}{dim.op}{render(dim.right)}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _apply(op: str, a: int, b: int) -> int:
    if op == MUL:
        return a * b
    if op == "+":
        return a + b
    if b == 0:
        raise ZeroDivisionError(op)
    return a // b


def is_sharded(dim: Dim) -> bool:
    """True if the expression divides by ``TP``.

    A sharded dim is clamped to at least 1 by :func:`evaluate`: when TP exceeds
    the number of KV heads or experts, the engine replicates the shard across
    ranks rather than handing some rank an empty tensor. A resolved 0 is a
    division artefact, and it propagates into degenerate benchmark shapes
    (``kv_head_num=0``, ``K=0``) that every kernel rejects.
    """
    if isinstance(dim, BinOp):
        if dim.op == "/" and isinstance(dim.right, Sym) and dim.right.name == "TP":
            return True
        return is_sharded(dim.left) or is_sharded(dim.right)
    return False


def evaluate(dim: Dim, symbols: dict[str, int]) -> int:
    """Fold the expression to an integer. Raises if a symbol is unknown."""
    def go(node: Dim) -> int:
        if isinstance(node, int):
            return node
        if isinstance(node, Sym):
            if node.name not in symbols:
                raise KeyError(node.name)
            return int(symbols[node.name])
        return _apply(node.op, go(node.left), go(node.right))

    value = go(dim)
    return max(1, value) if is_sharded(dim) else value


def is_variable_name(name: str, keep: frozenset[str] = VARIABLES) -> bool:
    """True if a *symbol name* is built only from swept variables.

    The legend registers composites as names in their own right -- ``S+C`` is
    a symbol, not just an expression -- so the lexer takes ``S+C`` whole. For
    evaluation that is correct and convenient. For display it is not: folding
    it to a number would hide that this dimension grows with the sweep, which
    is the one thing the reader is looking for.
    """
    parts = [p for p in re.split(r"[" + MUL + r"+/]", name) if p]
    return bool(parts) and all(p in keep for p in parts)


def partial(dim: Dim, symbols: dict[str, int],
            keep: frozenset[str] = VARIABLES) -> Dim:
    """Fold every subtree that does not mention a kept symbol.

    This is the display form: model constants from ``config.json`` become
    numbers, while the symbols the sweep varies stay legible, so a reader sees
    ``2048·S`` rather than either ``n_h·d·S`` or a number that hides which
    dimension moves when the sweep moves.
    """
    def mentions_kept(node: Dim) -> bool:
        if isinstance(node, Sym):
            return node.name in keep or is_variable_name(node.name, keep)
        if isinstance(node, BinOp):
            return mentions_kept(node.left) or mentions_kept(node.right)
        return False

    def go(node: Dim) -> Dim:
        if not mentions_kept(node):
            try:
                return evaluate(node, symbols)
            except (KeyError, ZeroDivisionError):
                return node
        if isinstance(node, BinOp):
            return BinOp(node.op, go(node.left), go(node.right))
        return node

    return go(dim)


# ---------------------------------------------------------------------------
# The string-in / string-out helpers the pipeline actually calls
# ---------------------------------------------------------------------------
def resolve(dim, symbols: dict[str, int]):
    """A dim resolved to an integer, or returned unchanged if it cannot be.

    Returning the input on failure -- rather than raising or substituting a
    plausible number -- is the pipeline's rule that nothing is guessed: a dim
    that stays a string is visibly unresolved, and the tests assert that none
    survive to the sweep.
    """
    if isinstance(dim, int):
        return dim
    if not isinstance(dim, str):
        return dim
    if dim in symbols:
        return int(symbols[dim])
    try:
        return evaluate(parse(dim, symbols), symbols)
    except (ParseError, KeyError, ZeroDivisionError, OverflowError, ValueError):
        return dim


def resolve_display(dim, symbols: dict[str, int],
                    keep: frozenset[str] = VARIABLES) -> str:
    """A dim rendered with constants folded and swept symbols kept."""
    if isinstance(dim, (int, float)):
        return str(int(dim))
    text = str(dim)
    if text in keep or is_variable_name(text, keep):
        return text
    try:
        return render(partial(parse(text, symbols), symbols, keep))
    except (ParseError, ValueError):
        return text
