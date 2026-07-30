# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perf pipeline: graph -> matrix rows -> micro_perf workloads.

No GPU and no xpu-perf checkout required: everything here is pure mapping.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breakdown.perf import shape_matrix, shape_matrix_xlsx, workloads
from breakdown.perf.matrix_reader import (
    read_matrix,
    rows_to_oprows,
    unique_rows,
)
from breakdown.perf.op_map import ModelConfig, get_dispatch


def _op(name, role, shapes, dtypes, backend="vllm-xpu-kernels"):
    return {"name": name, "role": role, "backend": backend,
            "input_shapes": shapes, "recorded_shapes": [], "input_dtypes": dtypes,
            "memory_bytes": 0, "flops": 0}


def _leaf(name, cls, path, ops, repeat=1):
    return {"name": name, "module_type": cls, "path": path,
            "repeat_count": repeat, "ops": ops, "children": []}


def _mock_m3_graph():
    """A MiniMax-M3-shaped reconstructed graph covering all workload groups."""
    attn_ops = [
        _op("aten::linear", "qkv_proj", [["S", "H"], ["QKV/TP", "H"]],
            ["bfloat16", "bfloat16"], backend="torch-xpu-ops"),
        _op("vllm::unified_attention_with_output", "attention",
            [["S", "n_h/TP", "d"], ["S+C", "n_kv/TP", "d"],
             ["S+C", "n_kv/TP", "d"]],
            ["bfloat16", "bfloat16", "bfloat16"]),
        _op("flash_xpu::minimax_m3_sparse_attn", "sparse_attn",
            [["S", "n_h/TP", "d"]], ["bfloat16"], backend="flash_xpu"),
        _op("flash_xpu::minimax_m3_index_score", "index_score",
            [["S", "n_idx/TP", "idx_d"]], ["bfloat16"], backend="flash_xpu"),
        _op("c10d::allreduce_", "allreduce", [["S", "H"]], ["bfloat16"],
            backend="ccl"),
    ]
    mlp_ops = [
        _op("_C::silu_and_mul_with_clamp", "act", [["S", "2·I/TP"]],
            ["bfloat16"]),
        _op("triton::_gemma_rmsnorm_kernel", "norm", [["S", "H"]],
            ["bfloat16"], backend="triton"),
    ]

    def tree(tok):
        return {
            "name": "model", "module_type": "MiniMaxM3Model", "path": "model",
            "repeat_count": 1, "ops": [], "children": [
                _leaf("self_attn", "Attention", "model/attn", attn_ops,
                      repeat=57),
                _leaf("mlp", "MLP", "model/mlp", mlp_ops, repeat=57),
            ],
        }

    return {
        "prefill": tree("S"), "decode": tree("B"),
        "symbols": {"H": 6144, "n_h": 64, "n_kv": 4, "d": 128, "I": 12288,
                    "2·I": 24576, "QKV": 9216, "n_idx": 4, "idx_d": 128,
                    "V": 200064, "TP": 4, "S": 128, "B": 1, "C": 0,
                    "S+C": 128},
        "config": {"tp_size": 4, "quantization": None, "dtype_bytes": 2,
                   "num_layers": 60},
        "source": "profile",
    }


_CFG = ModelConfig()


def _rows(**sweep):
    base = dict(prefill_seq_lens=[128, 2048], prefill_ctx_lens=[0],
                prefill_batch_sizes=[1], decode_ctx_lens=[8192],
                decode_batch_sizes=[1, 32], tp_sizes=[4])
    base.update(sweep)
    graph = _mock_m3_graph()
    return graph, shape_matrix.build_rows(graph,
                                          shape_matrix.build_configs(**base))


class TestShapeMatrixRows(unittest.TestCase):
    def test_rows_cover_the_sweep(self):
        graph, rows = _rows()
        # 2 prefill + 2 decode configs x 7 ops
        self.assertEqual(len(rows), 4 * 7)
        self.assertEqual({r["Phase"] for r in rows}, {"prefill", "decode"})
        self.assertEqual({r["TP"] for r in rows}, {4})

    def test_layers_column_carries_the_call_count(self):
        """The e2e weighting depends on Layers = how often the module repeats."""
        _, rows = _rows()
        self.assertTrue(all(r["Layers"] == 57 for r in rows))

    def test_rows_resolve_symbols_per_config(self):
        _, rows = _rows()
        qkv = [r for r in rows
               if r["Op Name"] == "aten::linear" and r["Phase"] == "prefill"
               and r["Seq Len"] == 2048][0]
        # per-rank weight: QKV/TP = 9216/4
        self.assertEqual(qkv["_resolved_shapes"], [[2048, 6144], [2304, 6144]])

    def test_row_limit_estimate(self):
        graph = _mock_m3_graph()
        configs = shape_matrix.build_configs(
            prefill_seq_lens=[1], prefill_ctx_lens=[0], prefill_batch_sizes=[1],
            decode_ctx_lens=[1], decode_batch_sizes=[1], tp_sizes=[1])
        self.assertEqual(shape_matrix.ops_per_config(graph), 7)
        self.assertEqual(shape_matrix.estimate_row_count(graph, configs), 14)


class TestWorkloadEmission(unittest.TestCase):
    def test_every_dispatched_op_maps(self):
        _, rows = _rows()
        buckets, cov = workloads.emit(rows_to_oprows(rows), _CFG, "xpu")
        self.assertEqual(cov.unmapped_breakdown_ops, {})
        self.assertTrue(cov.ok)

    def test_cases_land_in_the_right_group(self):
        _, rows = _rows()
        buckets, _ = workloads.emit(rows_to_oprows(rows), _CFG, "xpu")
        self.assertIn("gemm", buckets["compute"])
        self.assertIn("all_reduce", buckets["collective"])
        self.assertIn("msa_sparse_attn", buckets["msa"])
        # the MSA ops must not leak into the single-device compute group
        self.assertNotIn("msa_sparse_attn", buckets["compute"])

    def test_cases_are_deduplicated(self):
        """The same resolved shape from several sweep points emits once."""
        _, rows = _rows(prefill_seq_lens=[128, 128])
        buckets, _ = workloads.emit(rows_to_oprows(rows), _CFG, "xpu")
        gemm = buckets["compute"]["gemm"]
        keys = {tuple(sorted(c.items())) for c in gemm}
        self.assertEqual(len(gemm), len(keys))

    def test_smoke_filter_shrinks_the_sweep(self):
        _, rows = _rows(prefill_seq_lens=[128, 512, 2048],
                        decode_batch_sizes=[1, 8, 32])
        oprows = rows_to_oprows(rows)
        full, _ = workloads.emit(oprows, _CFG, "xpu")
        smoke, _ = workloads.emit(oprows, _CFG, "xpu",
                                  row_filter=workloads.RowFilter.smoke())
        self.assertLess(sum(len(v) for v in smoke["compute"].values()),
                        sum(len(v) for v in full["compute"].values()))

    def test_skipped_ops_are_not_unmapped(self):
        graph = _mock_m3_graph()
        graph["prefill"]["children"][0]["ops"].append(
            _op("aten::clone", "", [["S", "H"]], ["bfloat16"],
                backend="framework"))
        rows = shape_matrix.build_rows(
            graph, shape_matrix.build_configs(
                prefill_seq_lens=[128], prefill_ctx_lens=[0],
                prefill_batch_sizes=[1], decode_ctx_lens=[8192],
                decode_batch_sizes=[1], tp_sizes=[4]))
        _, cov = workloads.emit(rows_to_oprows(rows), _CFG, "xpu")
        self.assertIn("aten::clone", cov.skipped_framework_ops)
        self.assertEqual(cov.unmapped_breakdown_ops, {})


class TestXlsxRoundTrip(unittest.TestCase):
    """In-process rows and a matrix carried in as .xlsx must map identically."""

    def test_roundtrip_emits_the_same_cases(self):
        _, rows = _rows()
        direct, cov_direct = workloads.emit(rows_to_oprows(rows), _CFG, "xpu")

        payload = shape_matrix_xlsx.write_workbook(rows, [("Model", "mock")],
                                                   "mock")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.xlsx")
            with open(path, "wb") as fh:
                fh.write(payload)
            via_xlsx, cov_xlsx = workloads.emit(
                rows_to_oprows([dict(r) for r in _as_dicts(read_matrix(path))]),
                _CFG, "xpu")

        for group in workloads.GROUPS:
            self.assertEqual(sorted(direct[group]), sorted(via_xlsx[group]),
                             f"{group} op set differs")
        self.assertEqual(cov_direct.unmapped_breakdown_ops,
                         cov_xlsx.unmapped_breakdown_ops)


def _as_dicts(oprows):
    """OpRow -> matrix-row dicts (the .xlsx reader's output re-flattened)."""
    return [{
        "Phase": r.phase, "Seq Len": r.seq_len, "Ctx Len": r.ctx_len,
        "Batch Size": r.batch_size, "TP": r.tp, "Module": r.module,
        "Op Name": r.op_name, "Backend": r.backend, "Layers": r.layers,
        "Symbolic Shape": r.symbolic_raw, "Shape": r.shape_raw,
        "Memory (bytes)": r.memory_bytes, "FLOPs": r.flops, "AI": r.ai,
    } for r in oprows]


class TestDispatches(unittest.TestCase):
    def test_both_dispatch_maps_load(self):
        for name in ("xpu", "cuda"):
            mod = get_dispatch(name)
            self.assertTrue(mod.ADAPTERS)
            self.assertTrue(mod.SKIP_OPS)

    def test_unknown_dispatch_rejected(self):
        with self.assertRaises(ValueError):
            get_dispatch("rocm")

    def test_model_config_from_summary_dict(self):
        cfg = ModelConfig.from_config_summary({
            "num_heads": 64, "num_kv_heads": 4, "hidden_size": 6144,
            "num_experts": 128, "num_experts_per_tok": 4,
            "sparse_topk_blocks": 16,
        })
        self.assertEqual(cfg.num_experts, 128)
        self.assertEqual(cfg.sparse_topk_blocks, 16)


class TestDedup(unittest.TestCase):
    def test_dense_sweep_ops_keep_every_sweep_point(self):
        """MSA kernels carry no operand shapes, so they must not collapse."""
        _, rows = _rows()
        oprows = rows_to_oprows(rows)
        mod = get_dispatch("xpu")
        msa = [r for r in oprows
               if r.op_name == "flash_xpu::minimax_m3_sparse_attn"]
        collapsed = unique_rows(msa, set())
        expanded = unique_rows(msa, mod.DENSE_SWEEP_OPS)
        self.assertGreater(len(expanded), len(collapsed))


if __name__ == "__main__":
    unittest.main()
