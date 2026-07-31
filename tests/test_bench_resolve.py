# SPDX-License-Identifier: Apache-2.0
"""Dispatch resolution and argument materialization.

These run on CPU: resolving an op and building its arguments never needs an
accelerator, which is the point - a wrong callable or a randomly-filled index
map must be caught long before a GPU run.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.bench import inputs as inputs_mod, recipes, resolve  # noqa: E402
from breakdown.bench.spec import BenchCase  # noqa: E402

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

requires_torch = unittest.skipUnless(HAS_TORCH, "torch not installed")


def _t(dims, dtype="bfloat16"):
    return {"kind": "tensor", "dims": list(dims), "dtype": dtype}


@requires_torch
class TestResolve(unittest.TestCase):
    def test_aten_op_resolves_with_its_schema(self):
        r = resolve.resolve("aten::linear")
        self.assertEqual(r.kind, "torch_op")
        self.assertEqual(r.arg_names, ["input", "weight", "bias"])

    def test_overload_is_chosen_by_the_recorded_slots_not_by_name(self):
        # aten::add's *default* overload is (Scalar a, Scalar b) - no tensors at
        # all. Picking it would fail the call; the recorded slots must select
        # add.Tensor.
        slots = [_t([32], "long int"), _t([32], "long int"),
                 {"kind": "scalar", "type": "Scalar", "value": "1"}]
        r = resolve.resolve("aten::add", slots)
        self.assertEqual(r.arg_names, ["self", "other", "alpha"])
        self.assertEqual(r.arg_types[1], "Tensor")

    def test_keyword_only_schema_args_are_recorded_as_such(self):
        r = resolve.resolve("aten::add", [_t([4]), _t([4]),
                                          {"kind": "scalar", "value": "1"}])
        # aten::add.Tensor declares `*, Scalar alpha=1`
        self.assertIn(2, r.kwarg_only)

    def test_context_bound_wrapper_is_refused_with_a_reason(self):
        with self.assertRaises(resolve.NotReplayable) as cm:
            resolve.resolve("vllm::unified_attention_with_output")
        self.assertIn("forward context", str(cm.exception))

    def test_unknown_op_raises_rather_than_guessing(self):
        with self.assertRaises(resolve.ResolveError):
            resolve.resolve("aten::definitely_not_an_op")

    def test_synthetic_kernel_without_api_entry_is_unresolved(self):
        status, detail = resolve.classify("triton::_some_jit_kernel")
        self.assertEqual(status, "unresolved")
        self.assertIn("PYTHON_API", detail)

    def test_collectives_are_routed_to_the_multi_rank_path(self):
        self.assertEqual(resolve.classify("c10d::allreduce_")[0], "collective")
        self.assertTrue(resolve.is_collective("c10d::_allgather_base_"))


@requires_torch
class TestBuildArgs(unittest.TestCase):
    DEV = "cpu"

    def _case(self, op, args, **kw):
        return BenchCase(op=op, args=args, device=self.DEV, **kw)

    def test_optional_tensor_recorded_empty_becomes_none(self):
        case = self._case("aten::linear",
                          [_t([8, 16]), _t([32, 16]), {"kind": "none"}])
        call = recipes.build_args(case, resolve.resolve("aten::linear",
                                                        case.args), self.DEV)
        self.assertEqual(len(call.args), 3)
        self.assertIsNone(call.args[2])
        self.assertEqual(list(call.args[0].shape), [8, 16])

    def test_keyword_only_arg_is_passed_as_a_keyword(self):
        case = self._case("aten::add", [_t([4]), _t([4]),
                                        {"kind": "scalar", "value": "1"}])
        call = recipes.build_args(case, resolve.resolve("aten::add", case.args),
                                  self.DEV)
        self.assertEqual(len(call.args), 2)
        self.assertEqual(call.kwargs, {"alpha": 1})

    def test_scalar_the_profiler_did_not_record_does_not_become_a_tensor(self):
        # ``other`` is an elementwise *tensor* operand for add and a *number*
        # for mul_.Scalar; the name-keyed synthesizer must not leak into the
        # scalar slot.
        case = self._case("aten::mul_",
                          [_t([4, 8]), {"kind": "scalar", "type": "double",
                                        "value": ""}])
        call = recipes.build_args(case, resolve.resolve("aten::mul_",
                                                        case.args), self.DEV)
        self.assertIsInstance(call.args[1], float)

    def test_integer_operand_without_a_synthesizer_is_a_hard_error(self):
        saved = inputs_mod.NAME_SYNTHESIZERS.pop("index", None)
        try:
            case = self._case("aten::index_select",
                              [_t([16, 4], "float"),
                               {"kind": "scalar", "value": "0"},
                               _t([4], "long int")])
            with self.assertRaises(inputs_mod.MissingSynthesizer) as cm:
                recipes.build_args(
                    case, resolve.resolve("aten::index_select", case.args),
                    self.DEV)
            self.assertIn("synthesizer", str(cm.exception))
        finally:
            if saved is not None:
                inputs_mod.NAME_SYNTHESIZERS["index"] = saved

    def test_gather_index_stays_inside_the_table_it_indexes(self):
        case = self._case("aten::index_select",
                          [_t([16, 4], "float"),
                           {"kind": "scalar", "value": "0"},
                           _t([32], "long int")])
        call = recipes.build_args(
            case, resolve.resolve("aten::index_select", case.args), self.DEV)
        idx = call.args[2]
        self.assertLess(int(idx.max()), 16)
        self.assertGreaterEqual(int(idx.min()), 0)

    def test_mutating_operands_are_registered_for_restoration(self):
        case = self._case("aten::mul_",
                          [_t([4, 8]), {"kind": "scalar", "value": "2"}])
        call = recipes.build_args(case, resolve.resolve("aten::mul_",
                                                        case.args), self.DEV)
        self.assertEqual(len(call.mutated), 1)

    def test_declared_output_args_are_zeroed_not_index_synthesized(self):
        # rows_per_expert is an *output* accumulated with atomics; filling it
        # with a balanced partition would make the kernel scatter out of bounds.
        self.assertIn("rows_per_expert",
                      recipes.OUTPUT_ARGS["_moe_C::remap_hidden_states"])
        self.assertIn("_moe_C::remap_hidden_states", recipes.SINGLE_REP)


class TestScalarParsing(unittest.TestCase):
    def test_concrete_inputs_are_parsed_from_their_recorded_strings(self):
        self.assertEqual(inputs_mod.parse_scalar("7."), 7.0)
        self.assertIs(inputs_mod.parse_scalar("False"), False)
        self.assertEqual(inputs_mod.parse_scalar("[32, 2048]"), [32, 2048])
        self.assertIsNone(inputs_mod.parse_scalar(""))


if __name__ == "__main__":
    unittest.main()
