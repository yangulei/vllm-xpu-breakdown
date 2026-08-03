# SPDX-License-Identifier: Apache-2.0
"""The one dtype table, and the drifts it exists to prevent."""
from __future__ import annotations

import unittest

from breakdown.core import dtypes


class TestLookup(unittest.TestCase):
    def test_cpp_type_names_from_the_profiler_resolve(self):
        """The profiler records ``Input type`` with C++ names, not torch ones.

        An index tensor arrives as ``long int``. Before these aliases existed
        it fell through to the bf16 default and every index and position
        operand was counted at 2 bytes instead of 8 -- a 4x undercount on the
        operands that dominate a paged-attention kernel's traffic.
        """
        for token in ("long int", "long", "long long", "unsigned long"):
            with self.subTest(token=token):
                self.assertEqual(dtypes.size(token), 8)
                self.assertEqual(dtypes.torch_name(token), "int64")

    def test_spelling_variants_agree(self):
        """The same type arrives spelled three ways from three sources."""
        for token in ("bfloat16", "bf16", "c10::bfloat16", "torch.bfloat16",
                      "BFloat16"):
            with self.subTest(token=token):
                self.assertEqual(dtypes.torch_name(token), "bfloat16")
                self.assertEqual(dtypes.label(token), "bf16")

    def test_an_unknown_token_is_not_a_dtype(self):
        """``is_known`` separates a dtype from the other Input type entries.

        The recorded type list also carries ``ScalarType``, ``Device`` and an
        empty slot for each non-tensor argument. Reconstruction uses this to
        find the *element* dtype among them.
        """
        for token in ("", "ScalarType", "Device", "TensorList", None):
            with self.subTest(token=token):
                self.assertFalse(dtypes.is_known(token))
        self.assertTrue(dtypes.is_known("float8_e4m3fn"))

    def test_an_unknown_token_falls_back_to_the_activation_dtype(self):
        self.assertEqual(dtypes.size("nonsense"), 2)
        self.assertEqual(dtypes.torch_name("nonsense"), "bfloat16")

    def test_name_or_lets_the_caller_state_its_own_default(self):
        """Synthesizers disagree about what an absent dtype should mean."""
        self.assertEqual(dtypes.name_or("", "int64"), "int64")
        self.assertEqual(dtypes.name_or("", "int32"), "int32")
        self.assertEqual(dtypes.name_or("long int", "int32"), "int64")


class TestPackedTypes(unittest.TestCase):
    def test_int4_is_half_a_byte(self):
        """int4 packs two values per byte.

        It was listed at 1 byte with a comment saying it should be 0.5, so
        every int4-quantized weight was reported at twice its real size. That
        halves the op's arithmetic intensity, which is the number that decides
        whether the ranking calls it compute- or memory-bound -- so the
        overstatement did not just misreport memory, it could pick the wrong
        roof.
        """
        self.assertEqual(dtypes.size("int4"), 0.5)

    def test_a_byte_wide_type_is_still_a_byte(self):
        for token in ("int8", "uint8", "float8_e4m3fn", "bool"):
            with self.subTest(token=token):
                self.assertEqual(dtypes.size(token), 1)


class TestLabels(unittest.TestCase):
    def test_one_label_per_type(self):
        """There were two label tables and they disagreed.

        The benchmark's shape strings called ``float16`` ``f16`` while the
        shape matrix called it ``fp16``, so the same operand read differently
        depending on which artifact you opened. One vocabulary: floats are
        ``fpNN`` (with ``bf16`` for bfloat), ints are ``iNN``/``uNN``.
        """
        self.assertEqual(dtypes.label("float16"), "fp16")
        self.assertEqual(dtypes.label("float32"), "fp32")
        self.assertEqual(dtypes.label("bfloat16"), "bf16")
        self.assertEqual(dtypes.label("int64"), "i64")
        self.assertEqual(dtypes.label("uint8"), "u8")
        self.assertEqual(dtypes.label("bool"), "b8")

    def test_the_two_fp8_encodings_are_distinguishable(self):
        """Both were labelled ``fp8``, which hid which one a kernel used."""
        self.assertEqual(dtypes.label("float8_e4m3fn"), "fp8e4m3")
        self.assertEqual(dtypes.label("float8_e5m2"), "fp8e5m2")

    def test_labels_are_unique(self):
        labels = [d.label for d in dtypes.DTYPES]
        self.assertEqual(len(labels), len(set(labels)), sorted(labels))

    def test_an_unlabelled_token_prints_as_itself(self):
        self.assertEqual(dtypes.label("Q4_K_M"), "q4_k_m")
        self.assertEqual(dtypes.label(""), "")

    def test_a_configured_width_implies_the_obvious_float(self):
        """Some configs carry ``dtype_bytes`` rather than a dtype name."""
        self.assertEqual(dtypes.label_for_bytes(4), "fp32")
        self.assertEqual(dtypes.label_for_bytes(2), "bf16")
        self.assertEqual(dtypes.label_for_bytes(1), "fp8")
        self.assertEqual(dtypes.label_for_bytes(3), "24bit")


class TestIntegerOperands(unittest.TestCase):
    def test_index_types_are_integers(self):
        """An integer operand addresses memory, so it cannot hold noise.

        This is the predicate behind the benchmark's refusal to invent an
        index map: a random one reads outside the buffer it indexes.
        """
        for token in ("int64", "int32", "int16", "int8", "uint8", "long int"):
            with self.subTest(token=token):
                self.assertTrue(dtypes.is_integer(token))

    def test_data_types_are_not(self):
        for token in ("bfloat16", "float32", "float8_e4m3fn", "bool", ""):
            with self.subTest(token=token):
                self.assertFalse(dtypes.is_integer(token))


class TestTableIntegrity(unittest.TestCase):
    def test_every_alias_points_at_a_real_type(self):
        names = {d.name for d in dtypes.DTYPES}
        for alias in dtypes._ALIASES.values():
            self.assertIn(alias, names)

    def test_canonical_names_are_torch_attributes(self):
        """``torch_name`` is used with ``getattr(torch, ...)``."""
        torch = __import__("torch")
        for d in dtypes.DTYPES:
            if d.name == "int4":
                continue  # packed; torch has no such dtype
            with self.subTest(dtype=d.name):
                self.assertTrue(hasattr(torch, d.name), d.name)


if __name__ == "__main__":
    unittest.main()
