# SPDX-License-Identifier: Apache-2.0
"""Symbolic dimensions: the grammar, and the substitution hazards it removes."""
from __future__ import annotations

import unittest

from breakdown.core import dims
from breakdown.core.dims import MUL, BinOp, Sym

#: A legend of the shape the reconstruction actually produces, including the
#: composites it registers as names in their own right (``n_h·d``, ``2·I``,
#: ``S+C``) -- those are what make the grammar ambiguous without a symbol table.
SYMBOLS = {
    "H": 6144, "n_h": 64, "n_kv": 4, "d": 128, "n_h·d": 8192,
    "I": 1536, "2·I": 3072, "I_moe": 768, "2·I_moe": 1536,
    "V": 50016, "QKV": 9216, "topk": 4,
    "S": 2048, "B": 32, "C": 2048, "S+C": 4096, "TP": 4,
}


class TestLexingAgainstTheSymbolTable(unittest.TestCase):
    """The reason this is a lexer with a dictionary rather than a grammar."""

    def test_a_dot_inside_a_registered_name_is_not_multiplication(self):
        """``n_h·d`` is one symbol; ``topk·S`` is a product of two.

        Both contain a middle dot and only the symbol table can tell them
        apart. Taking the longest registered name at each position is the
        invariant the old ``sorted(symbols, key=len, reverse=True)``
        substitution was protecting.
        """
        self.assertEqual(dims.parse("n_h·d", SYMBOLS), Sym("n_h·d"))
        self.assertEqual(dims.parse("topk·S", SYMBOLS),
                         BinOp(MUL, Sym("topk"), Sym("S")))

    def test_the_longer_composite_wins_over_its_own_prefix(self):
        """``2·I`` and ``2·I_moe`` are both registered, and one prefixes the
        other. Shortest-first would resolve ``2·I_moe`` to ``3072`` followed by
        an unlexable tail."""
        self.assertEqual(dims.resolve("2·I_moe", SYMBOLS), 1536)
        self.assertEqual(dims.resolve("2·I", SYMBOLS), 3072)

    def test_an_unregistered_name_is_still_lexed_not_rejected(self):
        self.assertEqual(dims.parse("zzz/TP", SYMBOLS),
                         BinOp("/", Sym("zzz"), Sym("TP")))


class TestPrecedence(unittest.TestCase):
    def test_a_registered_name_wins_over_operator_precedence(self):
        """``topk·S+C`` reads as ``topk·(S+C)``, because ``S+C`` is a symbol.

        This is the lexer's dictionary beating the grammar, and it is the
        right answer here: the legend registered ``S+C`` as one dimension --
        the attended context length -- so a product against it means the whole
        context, not the new tokens alone. The pipeline never builds this
        string (``topk·S`` and ``S+C`` are each emitted whole), so the case is
        documented rather than relied upon.
        """
        self.assertEqual(dims.parse("topk·S+C", SYMBOLS),
                         BinOp(MUL, Sym("topk"), Sym("S+C")))

    def test_plus_binds_loosest_when_no_name_intervenes(self):
        no_composites = {k: v for k, v in SYMBOLS.items() if k != "S+C"}
        self.assertEqual(dims.parse("topk·S+C", no_composites),
                         BinOp("+", BinOp(MUL, Sym("topk"), Sym("S")),
                               Sym("C")))

    def test_products_and_divisions_are_left_associative(self):
        self.assertEqual(dims.resolve("n_h·d/TP", SYMBOLS), 2048)
        self.assertEqual(dims.resolve("H/TP", SYMBOLS), 1536)


class TestRoundTrip(unittest.TestCase):
    def test_rendering_reproduces_the_input(self):
        """The JSON contract and the golden snapshots depend on this."""
        for text in ("H", "H/TP", "n_h·d/TP", "topk·S", "S+C", "2·I_moe/TP",
                     "B·C", "999"):
            with self.subTest(dim=text):
                self.assertEqual(dims.render(dims.parse(text, SYMBOLS)), text)


class TestResolution(unittest.TestCase):
    def test_the_swept_variables_resolve_like_any_other_symbol(self):
        self.assertEqual(dims.resolve("S+C", SYMBOLS), 4096)
        self.assertEqual(dims.resolve("B·C", SYMBOLS), 65536)

    def test_an_unresolvable_dim_comes_back_unchanged(self):
        """Nothing is guessed: an unresolved dim stays visibly unresolved.

        Substituting a plausible number here would put a wrong shape into the
        sweep, and the benchmark would measure the wrong kernel and report it
        as a valid baseline.
        """
        self.assertEqual(dims.resolve("n_idx/TP", {"TP": 4}), "n_idx/TP")
        self.assertEqual(dims.resolve("nonsense", SYMBOLS), "nonsense")

    def test_an_integer_passes_through(self):
        self.assertEqual(dims.resolve(4096, SYMBOLS), 4096)


class TestSharding(unittest.TestCase):
    def test_a_shard_never_resolves_to_zero(self):
        """When TP exceeds the KV-head or expert count the engine replicates
        the shard rather than handing a rank an empty tensor. A resolved 0 is
        a division artefact that propagates into degenerate benchmark shapes
        (``kv_head_num=0``, ``K=0``) which every kernel rejects.
        """
        self.assertEqual(dims.resolve("n_kv/TP", {"n_kv": 4, "TP": 8}), 1)
        self.assertEqual(dims.resolve("n_kv/TP", {"n_kv": 4, "TP": 2}), 2)

    def test_an_unsharded_expression_is_not_clamped(self):
        self.assertEqual(dims.resolve("a+b", {"a": -3, "b": 3}), 0)


class TestDisplayForm(unittest.TestCase):
    def test_constants_fold_and_swept_symbols_stay(self):
        """A reader needs to see which dimension moves when the sweep moves."""
        self.assertEqual(dims.resolve_display("topk·S", SYMBOLS), "4·S")
        self.assertEqual(dims.resolve_display("H/TP", SYMBOLS), "6144/TP")
        self.assertEqual(dims.resolve_display("H", SYMBOLS), "6144")

    def test_a_registered_composite_of_variables_stays_symbolic(self):
        """``S+C`` is a *symbol*, so the lexer takes it whole -- correct for
        evaluation, wrong for display, where folding it to a number would hide
        that this dimension grows with the sweep."""
        self.assertEqual(dims.resolve_display("S+C", SYMBOLS), "S+C")

    def test_a_variable_is_not_folded_away_by_an_addition(self):
        """The old display resolver fell through to full resolution for any
        ``+`` composite that was not made *entirely* of variables, so a mixed
        expression lost its variable entirely and printed a bare number."""
        self.assertEqual(dims.resolve_display("H+S", SYMBOLS), "6144+S")

    def test_an_unresolvable_dim_displays_as_itself(self):
        self.assertEqual(dims.resolve_display("n_idx/TP", {"TP": 4}),
                         "n_idx/TP")


class TestNoEval(unittest.TestCase):
    def test_a_hostile_dim_cannot_execute(self):
        """The previous resolver compiled the substituted text and ran it
        through ``eval`` behind an AST whitelist. There is no evaluator now,
        so the whitelist is not a thing that can be got wrong.
        """
        for hostile in ("__import__('os').system('true')", "1;2", "S if S else C"):
            with self.subTest(dim=hostile):
                self.assertEqual(dims.resolve(hostile, SYMBOLS), hostile)


if __name__ == "__main__":
    unittest.main()
