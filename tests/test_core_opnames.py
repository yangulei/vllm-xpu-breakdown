# SPDX-License-Identifier: Apache-2.0
"""The one op-name vocabulary, and the drifts it exists to prevent."""
from __future__ import annotations

import unittest

from breakdown.classifier import Backend, classify_op
from breakdown.core import opnames


class TestNames(unittest.TestCase):
    def test_namespace_is_split_off_and_the_base_lowercased(self):
        self.assertEqual(opnames.split("_C::rms_norm"), ("_C", "rms_norm"))
        self.assertEqual(opnames.base_of("aten::_scaled_mm"), "_scaled_mm")
        self.assertEqual(opnames.base_of("triton::FusedMoE"), "fusedmoe")

    def test_a_bare_name_has_no_namespace(self):
        self.assertEqual(opnames.split("fused_moe_kernel"),
                         ("", "fused_moe_kernel"))


class TestPrefixShadowing(unittest.TestCase):
    """The bug the consolidation surfaced.

    ``FRAMEWORK_PREFIXES`` used ``aten::t`` as shorthand for the view ops, but
    a prefix does not know where a name ends, so it also swallowed
    ``aten::topk`` and ``aten::tanh``. Both are real kernels with real device
    time, and both were reported as framework overhead -- dropped from the ops
    table, from the backend distribution and from the benchmark's op list. The
    sampler's topk is not a rounding error to lose.
    """

    def test_a_compute_op_is_not_lost_to_a_prefix_it_starts_with(self):
        for name in ("aten::topk", "aten::tanh"):
            with self.subTest(op=name):
                self.assertFalse(opnames.is_framework(name))
                backend, _ = classify_op(name, "xpu", 5.0, 5.0)
                self.assertEqual(backend, Backend.TORCH_XPU_OPS)

    def test_the_plumbing_the_prefix_stood_for_is_still_plumbing(self):
        for name in ("aten::t", "aten::to", "aten::transpose", "aten::clone",
                     "aten::zeros", "aten::empty_like", "aten::view"):
            with self.subTest(op=name):
                self.assertTrue(opnames.is_framework(name))
                backend, _ = classify_op(name, "xpu", 5.0, 5.0)
                self.assertEqual(backend, Backend.FRAMEWORK)


class TestKinds(unittest.TestCase):
    def test_matmul_is_defined_once(self):
        """It used to be written out four times, in four files."""
        for name in ("aten::mm", "aten::addmm", "aten::linear",
                     "aten::matmul", "aten::bmm"):
            with self.subTest(op=name):
                self.assertTrue(opnames.is_matmul(name))
        self.assertFalse(opnames.is_matmul("aten::_scaled_mm"))

    def test_quantized_gemms_join_the_gemm_family_not_the_matmul_one(self):
        """They compute the same 2*M*K*N; only the operand dtype differs."""
        for base in ("_scaled_mm", "fp8_gemm", "fp4_gemm",
                     "int4_gemm_w4a16", "int4_gemm_w4a8"):
            with self.subTest(base=base):
                self.assertIn(base, opnames.GEMM_BASES)

    def test_only_matrix_ops_reach_the_matrix_peak(self):
        """Scoring a norm against XMX made every elementwise op ~99 % idle."""
        for name in ("aten::mm", "aten::linear", "_C::fp8_gemm",
                     "vllm::unified_attention"):
            with self.subTest(op=name):
                self.assertTrue(opnames.uses_matrix_engine(name))
        for name in ("_C::rms_norm", "_C::silu_and_mul", "aten::add",
                     "c10d::allreduce_"):
            with self.subTest(op=name):
                self.assertFalse(opnames.uses_matrix_engine(name))

    def test_attention_is_recognized_across_backends(self):
        for name in ("vllm::unified_attention", "vllm::unified_attention_with_output",
                     "flash_xpu::minimax_m3_sparse_attn", "_C::paged_attention_v1"):
            with self.subTest(op=name):
                self.assertTrue(opnames.is_attention(name))

    def test_a_table_lookup_op_declares_which_operand_it_indexes(self):
        """Charging the whole table gave "utilization 3902 % of peak"."""
        self.assertEqual(opnames.table_lookup("aten::embedding"), (0, 1, True))
        self.assertIsNone(opnames.table_lookup("aten::mm"))


class TestCollectives(unittest.TestCase):
    def test_the_namespace_list_no_longer_disagrees_between_stages(self):
        """classifier had ``oneccl``, bench.resolve had ``xccl``; now both."""
        for ns in ("c10d", "ccl", "oneccl", "nccl", "xccl"):
            with self.subTest(ns=ns):
                self.assertIn(ns, opnames.COLLECTIVE_NAMESPACES)

    def test_a_bare_collective_name_is_still_a_collective(self):
        self.assertTrue(opnames.is_collective("allreduce_"))
        self.assertTrue(opnames.is_collective("c10d::_allgather_base_"))

    def test_cache_and_moe_gathers_are_compute_not_communication(self):
        """Bare gather/scatter are deliberately absent from the keywords."""
        for name in ("_moe_C::moe_gather", "aten::gather", "aten::scatter_"):
            with self.subTest(op=name):
                self.assertFalse(opnames.is_collective(name))


class TestLibraries(unittest.TestCase):
    def test_flashinfer_and_xattention_are_probed_before_triton(self):
        """A Python-launched kernel gets a ``triton::``-prefixed synthetic name
        even when the kernel inside it is hand-written SYCL, so probing Triton
        first would report a SYCL kernel as compiled Triton output.
        """
        self.assertEqual(
            opnames.library_of("triton::kernel_flashinfernorm_rmsnorm"),
            "flashinfer")
        self.assertEqual(
            opnames.library_of("flash_xpu::minimax_m3_index_topk"), "flash_xpu")
        self.assertEqual(opnames.library_of("triton::fused_moe_kernel"),
                         "triton")
        self.assertEqual(opnames.library_of("aten::mm"), "")


class TestPlumbing(unittest.TestCase):
    def test_the_three_questions_stay_separate(self):
        """Reconstruction, classification and replay ask different things.

        ``aten::clone`` is plumbing to the classifier and unreplayable to the
        benchmark, but reconstruction still lists it -- it is a real copy that
        can carry device time, and hiding it would lose that time.
        """
        self.assertTrue(opnames.is_framework("aten::clone"))
        self.assertTrue(opnames.is_skipped("aten::clone"))
        self.assertFalse(opnames.is_plumbing("aten::clone"))

    def test_compiled_region_markers_are_not_ops(self):
        for name in ("Torch-Compiled Region: 0/1", "ProfilerStep#12",
                     "Memcpy DtoH", "cudaLaunchKernel"):
            with self.subTest(op=name):
                self.assertTrue(opnames.is_skipped(name))

    def test_a_real_kernel_is_replayable(self):
        for name in ("_C::rms_norm", "aten::linear", "_moe_C::topk_softmax"):
            with self.subTest(op=name):
                self.assertFalse(opnames.is_skipped(name))


class TestFamilies(unittest.TestCase):
    def test_the_family_probe_is_order_dependent(self):
        """``index_topk`` must be claimed by the topk family, not by MoE.

        ``minimax_m3_index_topk`` contains ``topk_``, which is also an MoE
        marker, so probing MoE first would name the indexer's block count after
        the router.
        """
        self.assertEqual(
            opnames.first_family("minimax_m3_index_topk",
                                 opnames.ALLOCATION_FAMILIES), "K_topk")
        self.assertEqual(
            opnames.first_family("minimax_m3_index_topk",
                                 opnames.MSA_KERNEL_LAYOUTS), "topk")

    def test_an_unfamiliar_name_matches_no_family(self):
        self.assertEqual(
            opnames.first_family("aten::mm", opnames.ALLOCATION_FAMILIES), "")

    def test_moe_markers_are_shared_with_the_allocation_family(self):
        """They were two lists agreeing on three entries and no more."""
        moe_family = next(markers for markers, label
                          in opnames.ALLOCATION_FAMILIES if label == "M_moe")
        self.assertIs(moe_family, opnames.MOE_SUBSTRINGS)


class TestNamespaces(unittest.TestCase):
    def test_the_classifier_no_longer_keeps_its_own_copy(self):
        """It had a hand-copied duplicate of the registry's namespace keys."""
        from breakdown.registry import STATIC_REGISTRY
        for ns in opnames.VLLM_KERNEL_NAMESPACES:
            if ns == "vllm":
                continue  # registered lazily, not in the static registry
            with self.subTest(ns=ns):
                self.assertIn(ns, STATIC_REGISTRY)

    def test_python_launched_namespaces_have_no_dispatcher_schema(self):
        for ns in ("triton", "flashinfer", "flash_xpu"):
            with self.subTest(ns=ns):
                self.assertTrue(opnames.is_python_launched(f"{ns}::whatever"))
        self.assertFalse(opnames.is_python_launched("aten::mm"))


if __name__ == "__main__":
    unittest.main()
