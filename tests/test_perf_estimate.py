# SPDX-License-Identifier: Apache-2.0
"""Degenerate-shape gating, runtime estimation and adaptive benchmark timeouts."""
from __future__ import annotations

import json
import os

import pytest

from breakdown.perf import estimate, runner, workloads as wl
from breakdown.perf.matrix_reader import OpRow, TensorShape
from breakdown.perf.op_map import ModelConfig
from breakdown.perf.op_map import common
from breakdown.shape_derive import _resolve_dim


# ---------------------------------------------------------------------------
# 0-dim shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("expr,tp,want", [
    ("4/TP", 8, 1),        # KV heads are replicated, never split to nothing
    ("4/TP", 2, 2),
    ("64/TP", 8, 8),
    ("200064/TP", 8, 25008),
])
def test_sharded_dim_never_resolves_to_zero(expr, tp, want):
    assert _resolve_dim(expr, {"S": 128, "C": 0, "B": 1, "TP": tp}) == want


def test_unsharded_dim_may_be_zero():
    # an empty KV cache is a real operating point, not an artefact
    assert _resolve_dim("C", {"C": 0, "TP": 1}) == 0


def test_positive_dims_flags_zero_extent():
    assert common.positive_dims({"M": 128, "K": 0, "N": 6144}) == \
        "non-positive K=0"
    assert common.positive_dims({"M": 128, "K": 64, "cache_len": 0}) is None


def test_check_case_allows_kernel_limits_through():
    # a kernel capability limit is NOT gated at emission: the benchmark must
    # report it so the kernel gets fixed
    args = {"q_head_num": 8, "kv_head_num": 1, "head_dim": 128}
    assert common.check_case("msa_sparse_attn", args, {}) is None


def _row(op_name, dims_a, dims_b, tp=8, flops=1e9, nbytes=1e6):
    return OpRow(
        phase="prefill", seq_len=128, ctx_len=0, batch_size=1, tp=tp,
        module="M/Linear.o_proj", op_name=op_name, backend="xpu", layers=1,
        tensors=[TensorShape(dims=dims_a, dtype="bfloat16", symbolic=""),
                 TensorShape(dims=dims_b, dtype="bfloat16", symbolic="")],
        symbolic_raw="", shape_raw="", flops=flops, memory_bytes=nbytes)


def test_emit_drops_degenerate_cases_and_reports_them():
    rows = [_row("aten::linear", [128, 0], [6144, 0]),
            _row("aten::linear", [128, 6144], [2304, 6144])]
    buckets, cov = wl.emit(rows, ModelConfig(), "xpu")
    assert buckets["compute"]["gemm"] == [
        {"arg_type": "default", "M": 128, "K": 6144, "N": 2304,
         "dtype": "bfloat16"}]
    assert cov.invalid_cases["gemm"] == {"non-positive K=0": 1}


def test_emit_records_per_case_cost(tmp_path):
    rows = [_row("aten::linear", [128, 6144], [2304, 6144],
                 flops=4.5e9, nbytes=3.2e7)]
    buckets, cov = wl.emit(rows, ModelConfig(), "xpu")
    assert cov.case_costs["gemm"] == [{"flops": 4.5e9, "bytes": 3.2e7}]
    wl.write(buckets, cov, str(tmp_path))
    assert wl.read_costs(str(tmp_path)) == {"gemm": [{"flops": 4.5e9,
                                                      "bytes": 3.2e7}]}
    assert wl.case_counts(str(tmp_path))["gemm"] == 1


def test_cost_json_is_not_a_workload_file(tmp_path):
    """micro_perf parses every json under a group dir - cost.json must not be one."""
    rows = [_row("aten::linear", [128, 6144], [2304, 6144])]
    buckets, cov = wl.emit(rows, ModelConfig(), "xpu")
    wl.write(buckets, cov, str(tmp_path))
    assert os.path.isfile(tmp_path / "cost.json")
    assert wl.ops_in(str(tmp_path / "compute")) == ["gemm"]


# ---------------------------------------------------------------------------
# runtime estimation
# ---------------------------------------------------------------------------
def test_kernel_seconds_uses_the_binding_resource():
    # 1 TFLOP at 100 TFLOPS peak and 10% util -> 100 ms
    assert estimate.kernel_seconds(1e12, 0, 100.0, 1000.0, 0.1) == \
        pytest.approx(0.1)
    # 100 GB at 1000 GB/s and 10% util -> 1 s, and memory binds
    assert estimate.kernel_seconds(1e12, 1e11, 100.0, 1000.0, 0.1) == \
        pytest.approx(1.0)


def test_measured_utilization_shortens_the_estimate():
    peaks = {"tflops": 100.0, "bw_gbs": 1000.0}
    costs = [{"flops": 1e12, "bytes": 0.0}] * 4
    slow = estimate.op_seconds("gemm", costs, peaks, util=0.02, overhead_s=0)
    fast = estimate.op_seconds("gemm", costs, peaks, util=0.50, overhead_s=0)
    assert slow > fast > estimate.OP_STARTUP_S


def test_iterations_follow_micro_perf_budget():
    # a 1 ms kernel gets ~50 iterations (50 ms budget)
    assert estimate.iterations(1e-3) == 50
    # a tiny kernel is capped
    assert estimate.iterations(1e-9) == estimate.MAX_ITERS
    # a 1 s kernel drops to the floor
    assert estimate.iterations(1.0) == estimate.MIN_ITERS
    # attention ops force a much higher floor
    assert estimate.iterations(1.0, "flash_attention") == estimate.ATTN_MIN_ITERS


def test_providers_multiply_the_estimate():
    peaks = {"tflops": 100.0, "bw_gbs": 1000.0}
    costs = [{"flops": 1e12, "bytes": 0.0}]
    one = estimate.op_seconds("gemm", costs, peaks, providers=1, overhead_s=1.0)
    three = estimate.op_seconds("gemm", costs, peaks, providers=3, overhead_s=1.0)
    assert three - estimate.OP_STARTUP_S == \
        pytest.approx(3 * (one - estimate.OP_STARTUP_S))


def test_op_utilization_from_measured_records():
    peaks = {"tflops": 100.0, "bw_gbs": 1000.0}
    recs = [{"op": "gemm", "latency_us": 1000.0, "tflops": 5.0},
            {"op": "gemm", "latency_us": 1000.0, "tflops": 15.0},
            {"op": "gemm", "latency_us": 1000.0, "tflops": 10.0}]
    assert estimate.op_utilization(recs, peaks)["gemm"] == pytest.approx(0.10)


def test_op_case_overhead_calibrates_from_history():
    prev = [{"ops": [{"op": "gemm", "ok": True, "cases": 10,
                      "seconds": estimate.OP_STARTUP_S + 40.0}]}]
    assert estimate.op_case_overhead(prev)["gemm"] == pytest.approx(4.0)
    # failed / empty ops carry no signal
    assert estimate.op_case_overhead(
        [{"ops": [{"op": "gemm", "ok": False, "cases": 0, "seconds": 99}]}]) == {}


def test_plan_gives_a_heavy_op_a_bigger_budget_than_a_light_one():
    costs = {"gemm": [{"flops": 1e13, "bytes": 1e9}] * 100,
             "rms_norm": [{"flops": 1e6, "bytes": 1e6}] * 3}
    timeouts, detail = estimate.plan(costs, micro_perf_dir="/nonexistent")
    assert timeouts["gemm"] > timeouts["rms_norm"]
    assert detail["gemm"]["cases"] == 100
    assert detail["gemm"]["util_source"] == "default"
    assert all(estimate.MIN_TIMEOUT_S <= t <= estimate.MAX_TIMEOUT_S
               for t in timeouts.values())


def test_plan_is_clamped():
    huge = {"gemm": [{"flops": 1e20, "bytes": 0.0}] * 1000}
    timeouts, _ = estimate.plan(huge, micro_perf_dir="/nonexistent")
    assert timeouts["gemm"] == estimate.MAX_TIMEOUT_S


# ---------------------------------------------------------------------------
# runner: per-case error reporting + durable progress
# ---------------------------------------------------------------------------
_LOG = """\
running gemm
Traceback (most recent call last):
  File "x.py", line 1, in run
RuntimeError: M3 expects gqa group size 16
Traceback (most recent call last):
RuntimeError: M3 expects gqa group size 16
ValueError: bad shape
done
"""


def test_case_errors_counts_and_dedups():
    failed, msgs = runner.case_errors(_LOG)
    assert failed == 3
    assert msgs[0] == "RuntimeError: M3 expects gqa group size 16 (x2)"
    assert "ValueError: bad shape" in msgs


def test_case_errors_on_clean_log():
    assert runner.case_errors("all good\n") == (0, [])


def test_per_op_timeout_is_used(monkeypatch, tmp_path):
    workloads = tmp_path / "workloads" / "compute"
    workloads.mkdir(parents=True)
    (workloads / "compute_ops.json").write_text(json.dumps(
        {"gemm": [{"M": 1}], "rms_norm": [{"M": 1}]}))
    mp = tmp_path / "micro_perf"
    mp.mkdir()
    monkeypatch.setattr(runner.devices, "micro_perf_dir", lambda: mp)

    seen: dict[str, int] = {}

    class _Proc:
        stdout, stderr, returncode = "ok\n", "", 0

    def fake_run(cmd, **kw):
        seen[cmd[cmd.index("--task") + 1]] = kw["timeout"]
        return _Proc()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "count_cases", lambda *a: 1)
    res = runner.run(str(tmp_path / "workloads"), str(tmp_path / "reports"),
                     groups=["compute"], timeout=1800,
                     timeouts={"gemm": 7200})
    assert seen == {"gemm": 7200, "rms_norm": 1800}
    assert [o.timeout for o in res.ops if o.op == "gemm"] == [7200]


def test_run_result_is_written_after_every_op(monkeypatch, tmp_path):
    """A run killed midway must still leave a record of what completed."""
    workloads = tmp_path / "workloads" / "compute"
    workloads.mkdir(parents=True)
    (workloads / "compute_ops.json").write_text(json.dumps(
        {"gemm": [{"M": 1}], "rms_norm": [{"M": 1}]}))
    mp = tmp_path / "micro_perf"
    mp.mkdir()
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(runner.devices, "micro_perf_dir", lambda: mp)

    class _Proc:
        stdout, stderr, returncode = "ok\n", "", 0

    written: list[int] = []

    def fake_run(cmd, **kw):
        path = reports_dir / "run_result.json"
        if path.exists():
            written.append(len(json.loads(path.read_text())["ops"]))
        return _Proc()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "count_cases", lambda *a: 1)
    runner.run(str(tmp_path / "workloads"), str(reports_dir),
               groups=["compute"])
    # the second op saw the first op's result already on disk
    assert written == [1]


def test_case_failures_do_not_fail_the_op(monkeypatch, tmp_path):
    workloads = tmp_path / "workloads" / "compute"
    workloads.mkdir(parents=True)
    (workloads / "compute_ops.json").write_text(json.dumps({"gemm": [{"M": 1}]}))
    mp = tmp_path / "micro_perf"
    mp.mkdir()
    monkeypatch.setattr(runner.devices, "micro_perf_dir", lambda: mp)

    class _Proc:
        stdout, stderr, returncode = _LOG, "", 0

    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: _Proc())
    monkeypatch.setattr(runner, "count_cases", lambda *a: 4)
    res = runner.run(str(tmp_path / "workloads"), str(tmp_path / "reports"),
                     groups=["compute"])
    op = res.ops[0]
    assert op.ok and op.cases == 4          # the shapes that ran are still data
    assert op.failed_cases == 3             # and the ones that did not are reported
    assert any("gqa group size 16" in m for m in op.errors)
