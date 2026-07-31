# SPDX-License-Identifier: Apache-2.0
"""Replay-case construction: rows -> BenchCase, and the skip/dedup rules."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.bench.spec import (  # noqa: E402
    BenchCase, build_cases, case_signature, group_by_op, is_skipped, shape_key,
)


def _row(**kw):
    row = {
        "Phase": "decode", "Seq Len": 1, "Ctx Len": 2048, "Batch Size": 32,
        "TP": 4, "Module": "Model/Layer.qkv_proj", "Op Name": "aten::linear",
        "Backend": "torch-xpu-ops", "Layers": 57,
        "Memory (bytes)": 1000.0, "FLOPs": 2000.0,
        "_resolved_shapes": [[32, 6144], [2304, 6144]],
        "_input_dtypes": ["bfloat16", "bfloat16"],
        "_input_args": [
            {"kind": "tensor", "dims": [23, 6144], "dtype": "bfloat16",
             "strides": [6144, 1]},
            {"kind": "tensor", "dims": [2304, 6144], "dtype": "bfloat16",
             "strides": [6144, 1]},
            {"kind": "none", "value": ""},
        ],
        "_recorded_shapes": [[23, 6144], [2304, 6144]],
        "_device_time_us": 87.2,
        "_op_role": "qkv_proj", "_module_type": "QKVParallelLinear",
    }
    row.update(kw)
    return row


class TestBuildCases(unittest.TestCase):
    def test_swept_dims_replace_recorded_dims_in_order(self):
        cases, cov = build_cases([_row()])
        self.assertEqual(len(cases), 1)
        c = cases[0]
        # tensor slots take the swept dims; the non-tensor slot is untouched
        self.assertEqual(c.args[0]["dims"], [32, 6144])
        self.assertEqual(c.args[1]["dims"], [2304, 6144])
        self.assertEqual(c.args[2]["kind"], "none")
        # strides described the *recorded* dims and must not survive the sweep
        self.assertNotIn("strides", c.args[0])
        self.assertEqual(c.layers, 57)
        self.assertEqual(cov["ops"], 1)

    def test_traced_time_only_comparable_at_the_recorded_shape(self):
        swept = build_cases([_row()])[0][0]
        self.assertFalse(swept.traced_comparable)
        at_profile = build_cases([_row(
            _resolved_shapes=[[23, 6144], [2304, 6144]])])[0][0]
        self.assertTrue(at_profile.traced_comparable)
        self.assertEqual(at_profile.traced_device_time_us, 87.2)

    def test_framework_plumbing_is_skipped_with_a_count(self):
        rows = [_row(**{"Op Name": "aten::t"}), _row(**{"Op Name": "aten::to"}),
                _row()]
        cases, cov = build_cases(rows)
        self.assertEqual([c.op for c in cases], ["aten::linear"])
        self.assertEqual(cov["skipped_framework_ops"],
                         {"aten::t": 1, "aten::to": 1})

    def test_identical_cases_dedup_and_keep_the_largest_layer_count(self):
        cases, _ = build_cases([_row(Layers=3), _row(Layers=57)])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].layers, 57)

    def test_a_deduped_case_records_every_sweep_point_it_stands_for(self):
        # An op whose operands don't depend on the swept dimension is measured
        # once but belongs to *both* points; keeping only the first would drop
        # it from the ranking at every other operating point.
        cases, _ = build_cases([_row(**{"Batch Size": 1}),
                                _row(**{"Batch Size": 32})])
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].points,
                         [["decode", 1, 2048, 1], ["decode", 1, 2048, 32]])

    def test_reconstructed_ops_without_slots_fall_back_to_shapes(self):
        row = _row(**{"Op Name": "triton::_gemma_rmsnorm_kernel",
                      "_input_args": [],
                      "_resolved_shapes": [[32, 6144]],
                      "_input_dtypes": ["bfloat16"]})
        cases, _ = build_cases([row])
        self.assertEqual(cases[0].args,
                         [{"kind": "tensor", "dims": [32, 6144],
                           "dtype": "bfloat16"}])

    def test_row_with_no_shape_at_all_is_reported_not_dropped_silently(self):
        cases, cov = build_cases([_row(_input_args=[], _resolved_shapes=[])])
        self.assertEqual(cases, [])
        self.assertEqual(cov["rows_without_shapes"], {"aten::linear": 1})

    def test_tensorlist_slot_consumes_one_swept_shape_per_element(self):
        row = _row(**{
            "Op Name": "c10d::allreduce_",
            "_input_args": [
                {"kind": "tensorlist",
                 "items": [{"dims": [23, 6144], "dtype": "bfloat16"}]},
                {"kind": "scalar", "type": "Scalar", "value": "False"},
            ],
            "_resolved_shapes": [[32, 6144]],
            "_input_dtypes": ["bfloat16"],
        })
        c = build_cases([row])[0][0]
        self.assertEqual(c.args[0]["items"][0]["dims"], [32, 6144])
        self.assertEqual(c.args[1]["value"], "False")
        self.assertEqual(len(c.tensor_args), 1)


class TestKeys(unittest.TestCase):
    def test_shape_key_ignores_scalar_values_but_case_id_does_not(self):
        a = BenchCase(op="x", args=[{"kind": "tensor", "dims": [4],
                                     "dtype": "bfloat16"},
                                    {"kind": "scalar", "value": "1"}])
        b = BenchCase(op="x", args=[{"kind": "tensor", "dims": [4],
                                     "dtype": "bfloat16"},
                                    {"kind": "scalar", "value": "2"}])
        self.assertEqual(shape_key(a.op, a.args), shape_key(b.op, b.args))
        self.assertNotEqual(case_signature(a.op, a.args),
                            case_signature(b.op, b.args))

    def test_group_by_op_is_the_unit_of_process_isolation(self):
        cases = [BenchCase(op="a"), BenchCase(op="b"), BenchCase(op="a")]
        self.assertEqual({k: len(v) for k, v in group_by_op(cases).items()},
                         {"a": 2, "b": 1})

    def test_compiled_region_markers_are_not_ops(self):
        self.assertTrue(is_skipped("Torch-Compiled Region: 0/1"))
        self.assertTrue(is_skipped("TorchDynamo Cache Lookup"))
        self.assertFalse(is_skipped("aten::linear"))


if __name__ == "__main__":
    unittest.main()
