# SPDX-License-Identifier: Apache-2.0
"""Dispatch resolution and argument materialization.

These run on CPU: resolving an op and building its arguments never needs an
accelerator, which is the point - a wrong callable or a randomly-filled index
map must be caught long before a GPU run.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

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
            resolve.resolve("vllm::moe_forward_shared")
        self.assertIn("benchmarked as their own ops", str(cm.exception))

    def test_attention_resolves_through_its_context_free_entry_point(self):
        # The dispatcher op reads the KV cache and the metadata from vLLM's
        # forward context, but the kernel underneath takes both as plain
        # arguments - so attention, normally the heaviest op in the profile, is
        # replayed rather than refused.
        self.assertEqual(
            resolve.classify("vllm::unified_attention_with_output")[0],
            "replayable")
        self.assertEqual(
            resolve.classify("vllm::unified_kv_cache_update")[0], "replayable")

    def test_a_python_launched_kernel_resolves_from_its_recorded_frame(self):
        """The trace says where the launcher is; that beats guessing a path.

        A Triton kernel or a pybind11 extension entry point emits no dispatcher
        op, so there is nothing to look up in ``torch.ops``. The profiler
        records the Python frame that launched it, and the replay imports that
        exact file. This used to be a hardcoded dotted-module table, which was
        wrong for the model it was written for: MiniMax-M3's xattention
        wrappers live at ``vllm/models/minimax_m3/xpu/ops/xattention.py`` while
        the table said ``vllm.model_executor.models.minimax_m3.xattention``, so
        all four MSA ops were unresolvable.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "some_kernel_wrappers.py")
            with open(path, "w") as fh:
                fh.write("def my_fused_kernel(x):\n    return x\n")
            r = resolve.resolve("triton::_my_fused_kernel_impl", None,
                                launch={"file": path, "line": 1,
                                        "func": "my_fused_kernel"})
        self.assertEqual(r.kind, "python_api")
        self.assertEqual(r.fn.__name__, "my_fused_kernel")

    def test_a_recorded_package_file_keeps_relative_imports_working(self):
        """A file-location import must not erase the launcher's package.

        Kimi-K3's KDA launchers import sibling modules with ``from .``. Loading
        them under a synthetic top-level name made four real kernels unresolved
        with "attempted relative import with no known parent package".
        """
        with tempfile.TemporaryDirectory() as tmp:
            package = os.path.join(tmp, "launchers")
            os.mkdir(package)
            with open(os.path.join(package, "__init__.py"), "w"):
                pass
            with open(os.path.join(package, "helper.py"), "w") as fh:
                fh.write("VALUE = 7\n")
            path = os.path.join(package, "kernel.py")
            with open(path, "w") as fh:
                fh.write("from .helper import VALUE\n\n")
                fh.write("def launch(x):\n    return x + VALUE\n")
            with mock.patch.object(sys, "path", [tmp, *sys.path]):
                r = resolve.resolve(
                    "triton::_package_kernel", None,
                    launch={"file": path, "line": 3, "func": "launch"})
            self.assertEqual(r.fn(5), 12)

    def test_context_bound_mla_wrappers_are_explained_not_unresolved(self):
        for op in ("vllm::unified_mla_kv_cache_update",
                   "vllm::unified_mla_attention_with_output"):
            status, detail = resolve.classify(op)
            self.assertEqual(status, "not_replayable")
            self.assertIn("forward context", detail)

    def test_a_python_launched_kernel_without_a_frame_is_refused(self):
        # Never guessed: no recorded launcher means no callable, and a wrong
        # callable would measure a different kernel.
        with self.assertRaises(resolve.ResolveError):
            resolve.resolve("triton::_unknown_kernel")

    def test_unknown_op_raises_rather_than_guessing(self):
        with self.assertRaises(resolve.ResolveError):
            resolve.resolve("aten::definitely_not_an_op")

    def test_synthetic_kernel_without_api_entry_is_unresolved(self):
        status, detail = resolve.classify("triton::_some_jit_kernel")
        self.assertEqual(status, "unresolved")
        self.assertIn("entry()", detail)

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
        saved = inputs_mod.SYNTHESIZERS.pop(("", "index"), None)
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
                inputs_mod.SYNTHESIZERS[("", "index")] = saved

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
                      recipes.recipe("_moe_C::remap_hidden_states").outputs)
        self.assertTrue(
            recipes.recipe("_moe_C::remap_hidden_states").single_rep)


class TestScalarParsing(unittest.TestCase):
    def test_concrete_inputs_are_parsed_from_their_recorded_strings(self):
        self.assertEqual(inputs_mod.parse_scalar("7."), 7.0)
        self.assertIs(inputs_mod.parse_scalar("False"), False)
        self.assertEqual(inputs_mod.parse_scalar("[32, 2048]"), [32, 2048])
        self.assertIsNone(inputs_mod.parse_scalar(""))


if __name__ == "__main__":
    unittest.main()


@requires_torch
class TestAttentionRecipe(unittest.TestCase):
    """The paged-attention replay rebuilds the context the wrapper would read."""

    def _case(self, phase, tokens, ctx, batch, n_h=32, n_kv=4, d=128):
        return BenchCase(
            op="vllm::unified_attention_with_output", device="cpu",
            phase=phase, seq_len=tokens, ctx_len=ctx, batch_size=batch,
            args=[_t([tokens, n_h, d]), _t([tokens, n_kv, d]),
                  _t([tokens, n_kv, d]), _t([tokens, n_h, d])])

    def _call(self, case):
        from breakdown.bench.recipes import attention
        return attention._paged_attention(case, None, "cpu")

    def test_prefill_builds_one_sequence_over_the_cached_context(self):
        call = self._call(self._case("prefill", 128, 8192, 1))
        kw = call.kwargs
        # the paged cache holds context+query for the sequence, in NHD layout
        self.assertEqual(list(kw["k"].shape[1:]), [16, 4, 128])
        self.assertEqual(kw["max_seqlen_q"], 128)
        self.assertEqual(kw["max_seqlen_k"], 8192 + 128)
        self.assertEqual(kw["seqused_k"].tolist(), [8320])
        self.assertEqual(kw["cu_seqlens_q"].tolist(), [0, 128])
        self.assertTrue(kw["causal"])

    def test_decode_gives_every_sequence_its_own_blocks(self):
        # A shared block table would turn the paged gather into a cache hit and
        # understate the kernel by a large factor.
        call = self._call(self._case("decode", 8, 1024, 8))
        table = call.kwargs["block_table"]
        self.assertEqual(list(table.shape), [8, 65])
        self.assertEqual(len(set(table.flatten().tolist())), table.numel())
        self.assertEqual(call.kwargs["seqused_k"].tolist(), [1025] * 8)

    def test_an_operating_point_that_cannot_fit_is_refused_not_attempted(self):
        # Without the guard this becomes a multi-terabyte allocation that hangs
        # the worker instead of reporting a case that does not fit.
        from breakdown.bench.inputs import ArgBuildError
        from breakdown.bench.recipes import attention

        case = self._case("decode", 1024, 10_000_000, 1024)
        with self.assertRaises(ArgBuildError) as cm:
            attention._paged_attention(case, None, "cpu")
        self.assertIn("KV cache", str(cm.exception))

    def test_kv_cache_write_scatters_each_token_to_its_own_slot(self):
        from breakdown.bench.recipes import attention

        case = self._case("prefill", 4, 32, 1)
        case.op = "vllm::unified_kv_cache_update"
        call = attention._kv_cache_update(case, None, "cpu")
        slots = call.args[4].tolist()
        self.assertEqual(slots, [32, 33, 34, 35])
        self.assertEqual(len(set(slots)), len(slots))
        self.assertEqual(call.args[5], "auto")


@requires_torch
class TestSamplerRecipe(unittest.TestCase):
    def test_sampler_is_replayed_rather_than_skipped(self):
        # Nothing about the sampler is context-bound: the "generator state" it
        # is handed is a two-element philox (seed, offset) CPU tensor.
        self.assertFalse(recipes.recipe("vllm::xpu_topk_topp_sampler").skip)
        case = BenchCase(op="vllm::xpu_topk_topp_sampler", device="cpu",
                         phase="decode", batch_size=32,
                         args=[{"kind": "tensor", "dims": [32],
                                "dtype": "long int"},
                               {"kind": "none", "value": ""},
                               {"kind": "tensor", "dims": [32, 151936],
                                "dtype": "float"},
                               {"kind": "none", "value": ""},
                               {"kind": "none", "value": ""},
                               {"kind": "none", "value": ""},
                               {"kind": "tensor", "dims": [2],
                                "dtype": "long int"},
                               {"kind": "scalar", "type": "Scalar",
                                "value": "1."}])
        call = recipes.recipe("vllm::xpu_topk_topp_sampler").build(case, None,
                                                                "cpu")
        random_sampled, to_return, logits, k, p, mode, seeds, lam = call.args
        self.assertEqual(list(logits.shape), [32, 151936])
        self.assertEqual(list(random_sampled.shape), [32])
        self.assertEqual(str(seeds.device), "cpu")
        self.assertEqual(list(seeds.shape), [2])
        self.assertIsNone(to_return)
        self.assertEqual(mode, "raw_logprobs")
        self.assertEqual(lam, 1.0)


class TestWorkerEnvironment(unittest.TestCase):
    """Environment invariants that only surface as a slow or dead run."""

    def test_persistent_kernel_caches_are_pinned(self):
        # Without them every worker re-pays SYCL AOT / Triton JIT on its first
        # case, which dominates a short sweep and poisons its first
        # measurement.
        import tempfile

        from breakdown.bench.worker import bench_env
        with tempfile.TemporaryDirectory() as d:
            env = bench_env(d, base={})
        self.assertEqual(env["SYCL_CACHE_PERSISTENT"], "1")
        self.assertTrue(env["SYCL_CACHE_DIR"].startswith(d))
        self.assertTrue(env["TRITON_CACHE_DIR"].startswith(d))

    def test_collectives_disable_the_persistent_cache(self):
        # oneCCL segfaults with no Python traceback when SYCL_CACHE_PERSISTENT
        # is set, so the peer ranks must run without it.
        import inspect

        from breakdown.bench import collective
        src = inspect.getsource(collective)
        self.assertIn("SYCL_CACHE_PERSISTENT", src)

    def test_a_rank_is_told_how_many_peers_share_the_node(self):
        # Without LOCAL_RANK/LOCAL_WORLD_SIZE oneCCL cannot read the node
        # topology from the environment and has to infer it; vLLM's own XPU
        # worker sets both before forming its group, and the replay stands in
        # for that worker.
        from breakdown.bench.collective import _rank_env
        env = _rank_env({}, rank=2, world_size=4)
        self.assertEqual(env["RANK"], "2")
        self.assertEqual(env["LOCAL_RANK"], "2")
        self.assertEqual(env["LOCAL_WORLD_SIZE"], "4")
        self.assertEqual(env["CCL_ATL_TRANSPORT"], "ofi")

    def test_a_deadlocked_rendezvous_is_retried_on_a_fresh_port(self):
        # The XCCL transport deadlocks intermittently on PCIe-connected cards:
        # every rank enqueues its collectives and all of them then block in
        # synchronize(). One unlucky attempt used to end the whole run, with
        # every op planned after it left unmeasured.
        from breakdown.bench import collective

        seen: list[int] = []

        def fake_once(op, world, cases, out, device, budget, timeout, env,
                      port):
            seen.append(port)
            ok = len(seen) == 3
            return ok, f"port={port}"

        with mock.patch.object(collective, "_launch_once", fake_once):
            ok, log = collective.launch("c10d::allreduce_", 4, "c.json",
                                        "o.jsonl", "xpu", 0.1, 5,
                                        {}, port=29591, attempts=3)
        self.assertTrue(ok)
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3, "a killed rank can leave the "
                                            "previous port bound")
        self.assertIn("attempt 3", log)

    def test_a_retry_does_not_record_a_case_twice(self):
        # A hung attempt is rarely empty: rank 0 streams a record per case, so
        # it usually measured the small shapes and wedged on a large one.
        # Appending the next attempt's output on top recorded those cases
        # twice, and the op's latency was then averaged over duplicate rows.
        import json

        from breakdown.bench import collective

        with tempfile.TemporaryDirectory() as d:
            cases_path = os.path.join(d, "cases.json")
            out_path = os.path.join(d, "results.jsonl")
            cases = [BenchCase(op="c10d::allreduce_", tp=4, device="xpu",
                               args=[{"kind": "tensor", "dims": [n],
                                      "dtype": "bfloat16"}])
                     for n in (8, 16)]
            with open(cases_path, "w") as fh:
                json.dump([c.to_dict() for c in cases], fh)

            calls = {"n": 0}

            def fake_once(op, world, cpath, out, device, budget, timeout, env,
                          port):
                calls["n"] += 1
                # attempt 1 measures the first case then hangs; attempt 2
                # measures both.
                done = cases if calls["n"] > 1 else cases[:1]
                with open(out, "w") as fh:
                    for c in done:
                        fh.write(json.dumps(
                            collective._rec(c, "ok", world)) + "\n")
                return calls["n"] > 1, "log"

            with mock.patch.object(collective, "_launch_once", fake_once):
                ok, _ = collective.launch("c10d::allreduce_", 4, cases_path,
                                          out_path, "xpu", 0.1, 5, {},
                                          attempts=3)
            with open(out_path) as fh:
                ids = [json.loads(line)["case_id"] for line in fh if line.strip()]
        self.assertTrue(ok)
        self.assertEqual(len(ids), 2, "each case is recorded exactly once")
        self.assertEqual(sorted(ids), sorted(c.case_id for c in cases))

    def test_every_attempt_failing_is_reported_not_swallowed(self):
        from breakdown.bench import collective

        with mock.patch.object(collective, "_launch_once",
                               lambda *a, **k: (False, "TIMEOUT")):
            ok, log = collective.launch("c10d::allreduce_", 4, "c.json",
                                        "o.jsonl", "xpu", 0.1, 5, {},
                                        attempts=2)
        self.assertFalse(ok)
        self.assertIn("attempt 1/2", log)
        self.assertIn("attempt 2/2", log)


class TestOperandAllocation(unittest.TestCase):
    def test_float_operands_are_built_in_their_target_dtype(self):
        # Materializing a several-hundred-MB weight in fp32 and casting doubles
        # the allocation and dominates the case's wall time (it timed out the
        # worker before this was fixed).
        try:
            import torch
        except ImportError:
            self.skipTest("torch required")
        from breakdown.bench.inputs import make_tensor
        for name, want in (("bfloat16", torch.bfloat16),
                           ("float16", torch.float16)):
            t = make_tensor([4, 8], name, "cpu")
            self.assertIs(t.dtype, want)
