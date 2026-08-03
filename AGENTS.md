# AGENTS.md — vllm-xpu-breakdown

Instructions for AI agents working in this repository.

## What this is

A four-stage pipeline that answers "which kernel should I optimize next, and
by how much can it improve":

1. **Profile** — run vLLM once on Intel XPU and reconstruct the model's
   module/op tree *directly from the torch-profiler trace*.
2. **Sweep + replay** — turn those ops into a shape matrix, and re-invoke the
   ops vLLM actually dispatched at every swept operating point.
3. **Rank** — score each op by `calls x latency x roofline headroom` and say
   which are worth a session.
4. **Optimize** — hand a ranked target to a headless Copilot CLI session
   running the `xpu-kernel-optimizer` skill, one GPU per session.

Every stage runs headless (`python -m breakdown.bench`, `python -m
breakdown.optimize`); the web UI and the REST API are wrappers.

The canonical example the pipeline is validated against is **MiniMax-M3,
TP=4, 6 layers, XPU** — a hybrid dense/MoE VL model with sparse attention,
which exercises nearly every path in the codebase. Its traces are committed as
test fixtures (`tests/data/`).

## The shape of the codebase

Four *stages*, over one shared *vocabulary*.

```
breakdown/core/     the vocabulary: facts about names, shared by every stage
breakdown/trace/    stage 1: a trace becomes a model graph
breakdown/bench/    stages 2+3: ops become measured, ranked targets
breakdown/optimize/ stage 4: a ranked target becomes a kernel session
```

**`breakdown/core/` is the rule that keeps the rest honest.** Before it
existed, five facts were each encoded many times over — "what does this op
name mean" alone lived in ~28 tables across eight modules — and the copies had
drifted apart. If you find yourself writing a table of op names, dtypes or
symbols, it goes in `core/`, and the stage asks it a question.

- **`core/opnames.py`** — what a dispatch name means: its namespace, its kind
  (matmul / attention / MoE / elementwise / table-lookup / plumbing), whether
  it reaches the matrix engine, whether it is a collective. The *data* lives
  once; the questions stay separate, because "should reconstruction list this
  op?", "should the classifier call it compute?" and "should the benchmark
  replay it?" are three different questions with three legitimate answers.
- **`core/dtypes.py`** — one record per element type: torch name, bytes
  (fractional, so packed `int4` is honest), short label.
- **`core/dims.py`** — the symbolic-dimension expression type: `parse` /
  `render` / `resolve`. The lexer consults the symbol table because `·` is
  both the multiply operator and a character *inside* names (`n_h·d` is one
  symbol, `topk·S` is two).
- **`core/devices.py`** — SKU peaks, device detection, id parsing, the
  visibility env var. Three subsystems import it, so it is not inside one.

The cost model is `breakdown/cost.py` and reads `core/`; it stays at the top
level because it is a *computation*, not vocabulary.

## Deep dives

The narrative for each subsystem lives with its code. Read these before
changing anything in them:

- **`breakdown/trace/README.md`** — how a trace becomes a model graph.
- **`breakdown/bench/README.md`** — how those ops become measured, ranked targets.
- **`breakdown/optimize/README.md`** — how a ranked target becomes a kernel session.

Every non-obvious rule is documented on the function that holds it. If a
docstring explains *why*, that reason is the specification; do not delete it
while changing the code it guards.

## Project structure

```
app.py                  Flask routes only
static/index.html       Single-page UI (inline CSS/JS, no build step)
breakdown/
  core/                 The shared vocabulary — torch-free, import-light
    opnames.py            What a dispatch name means (the one op table)
    dtypes.py             One record per element type
    dims.py               Symbolic dimensions: parse / render / resolve
    devices.py            SKU peaks, device detection, visibility env
  cost.py               The one cost model: bytes, FLOPs, AI, and the roofline
  runs.py               output/<stage>/<run_id>/ + RunState + atomic JSON
  service.py            What a route does between request and response
  profiling/            Run vLLM once; traces -> a result
    runstate.py           The run the server remembers, and the config cache
    launch.py             Running vLLM under the profiler
    traces.py             Trace files -> one reconstructed result
    uploads.py            The same, from files a browser sent
  trace/                Graph reconstruction (rules, events, forest, kernels,
                        shapes, phases, symbols, collapse, graph)
  bench/                Replay benchmark (spec, resolve, inputs, recipes,
                        timing, worker, runner, collective, estimate, rank,
                        reports, store, history, types, cli)
  optimize/             Kernel sessions (prompt, session, scheduler, manager, cli)
  module_hooks.py       Capture-time module-name spans
  kernel_hooks.py       Capture-time kernel-launch spans (operands + launcher)
  op_breakdown.py       Flat op breakdown derived from the graph
  shape_derive.py       Symbolic shape resolution + the display annotation
  shape_matrix.py       Graph + config sweep -> matrix rows
  shape_matrix_xlsx.py  Excel serialization of those rows
  model_info.py         HuggingFace config fetch/summarize
  classifier.py         Op -> backend (the *order* the questions are asked in)
  registry.py           Known vllm-xpu-kernels ops
  trace_common.py       Torch-free trace-*format* helpers (spans, overhead)
tools/
  capture_fixture.py    Re-capture the canonical MiniMax-M3 TP4 6-layer profile
  make_fixture.py       Trim a trace to a fixture (refuses if the graph differs)
tests/                  see "Build, run, test"
```

## Build, run, test

```bash
pip install -r requirements.txt
python app.py --port 8080

# headless
python -m breakdown.bench {plan,run,rank,report,case,history,all}
python -m breakdown.optimize {candidates,prompt,start,status,stop}
```

```bash
# everything that does not need a GPU (~419 tests, ~65 s)
pytest tests -q -p no:cacheprovider \
  --ignore=tests/test_real_profile.py \
  --ignore=tests/test_bench_replay.py \
  --ignore=tests/test_profile_reduced_layers.py

# the two safety nets, for any change to reconstruction or the vocabulary
pytest tests/test_golden_graph.py tests/test_golden_semantics.py -q
pytest tests/test_golden_graph.py tests/test_golden_semantics.py -q --update-golden

# with a GPU
pytest tests -q
```

No linter or pre-commit is configured. `python -m pyflakes` is used ad hoc.

**The golden fixtures are the safety net, and there are two of them.**
`tests/data/` holds trimmed rank-0 traces of the canonical example (XPU and
CUDA, prefill and decode) plus two snapshots per fixture:

- **`test_golden_graph.py`** pins the *shape of the tree* — modules, ops,
  symbolic shapes, dtypes, where device time lands.
- **`test_golden_semantics.py`** pins the *judgements made about* each op —
  its backend and category, its analytic bytes and FLOPs, which roof bounds
  it, and what every symbolic dimension resolves to. This is what makes it
  safe to consolidate a table in `core/`: if an answer moves it shows up here
  as a reviewable diff, instead of as a different ranking three stages
  downstream.

Re-capture with `tools/capture_fixture.py` then `tools/make_fixture.py`; the
CUDA fixtures cannot be re-captured on an XPU host and exist to keep the
device-agnostic paths honest.

## Conventions

- Every Python file starts with `# SPDX-License-Identifier: Apache-2.0`.
- `from __future__ import annotations`, modern types (`dict[str, Any]`,
  `list[int] | None`).
- Assume torch-xpu and vLLM are installed and an Intel XPU is present.
  `core/` and `model_info.py` stay import-light so the offline paths are fast.
- Op shapes are symbolic (`"H"`, `"S"`, `"n_h·d/TP"`, `"topk·S"`). `/TP` marks
  a tensor-parallel shard and is present even at TP=1.
- A run's artifacts belong to the run: `output/<stage>/<run_id>/`.
- **One name per concept, end to end.** The analytic byte count is `nbytes`
  everywhere; only the Shape Matrix column header is prose.
- **Import the function, not the module,** when the module's name collides
  with a common local (`dtypes`, `state`). Several such collisions were found
  the hard way.
- **A mutable module-level object is reached through its module**, never
  imported by name: `from x import obj` takes a snapshot, so swapping `x.obj`
  afterwards has no effect on the importer.
- **Nothing is guessed.** An op whose operands cannot be rebuilt is *reported*
  with the reason, never filled with plausible data. A utilization above peak
  is flagged as a cost-model problem, not silently retired.

## How to add things

**A new model.** Nothing to build — the graph is reconstructed. Ensure
`model_info.summarize_config` extracts the model's key dims so shapes
symbolize, classify any novel ops, then profile it and check the graph and the
Shape Matrix.

**A new op or kernel.** Add the name to `registry.ALL_VLLM_XPU_OPS`. If it
needs a *fact* — it is a matmul, it indexes into a table, it is MoE compute —
that fact goes in `core/opnames.py`, in the one table for it, and every stage
picks it up. Benchmarking needs no adapter: the replay resolves the dispatch
name and rebuilds the recorded operands. It may need:

- an **input synthesizer** (`bench/inputs.py`) if the op takes an integer/index
  tensor whose name is not covered — otherwise the case is reported
  `needs_synthesizer`, never randomly filled;
- a **recipe** (`bench/recipes/table.py`, one record per op) declaring an
  `entry` point, a `build` override, extra `values`, `outputs`/`single_rep`, or
  a `skip` reason. Recipes are grouped by *subject* (`attention`, `moe`,
  `sampling`, `common`), not by device — a synthesizer is registered by
  argument name into one global table, so a device split was never real;
- an entry in `bench/kernel_sources.json` if it has editable kernel source, so
  its target carries build/test commands.

**A new fact in an optimization brief.** It goes in `optimize/prompt.py` (plus
a test in `tests/test_optimize_prompt.py`), never in the ranker. `targets.json`
is a versioned contract (`schema_version` 5); changing a field's meaning
requires a bump.

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/<hf_id>` | GET | Fetch/summarize HF model config (+ `min_profile_layers`) |
| `/api/cached-models` | GET | Previously loaded model ids |
| `/api/devices` | GET | Accelerators present on this host |
| `/api/profile` | POST | Start an async profiling run |
| `/api/profile/upload` | POST | Rebuild a profile from uploaded trace(s) |
| `/api/profile/status` | GET | Poll; restores the last run on a fresh server |
| `/api/profile/result` | GET | Ops + reconstructed graph, with display shapes |
| `/api/profile/trace` | GET | Raw trace (`?pass=prefill\|decode`) |
| `/api/export/shape-matrix` | POST | Profile-derived shape sweep, as Excel |
| `/api/bench/ops` | GET | Dispatch names the latest profile ran |
| `/api/bench/plan` | POST | Sweep the profiled graph into replay cases |
| `/api/bench/run` | POST | Replay a run's cases (async) |
| `/api/bench/status` | GET | Poll the benchmark run (per-op progress) |
| `/api/bench/runs` | GET | List bench runs |
| `/api/bench/results` | GET | Measured cases + summary + coverage (`?op=`) |
| `/api/bench/targets` | GET | Ranked targets (`targets.json`) |
| `/api/bench/report` | GET | The run's report workbook |
| `/api/bench/history` | GET | History, or `?base=&new=` regression diff |
| `/api/optimize/candidates` | GET | Ranked ops + whether each is worth a session |
| `/api/optimize/prompt` | POST | The brief + pasteable command, spawning nothing |
| `/api/optimize/start` | POST | One session per selected kernel (one GPU each) |
| `/api/optimize/status` | GET | Sessions: state, device, queue position |
| `/api/optimize/log` | GET | A session's log from `?offset=` |
| `/api/optimize/stop` | POST | Stop a session, or a run's sessions |

**Routes are thin by rule**: a route validates, calls one `service` function,
and serializes. When a route grows a body, that body belongs in `service.py` —
the HTTP path and the CLI must run the same code, or they drift and no test
notices. They did drift once: see the `max_model_len` row in the pitfall index.

## The UI

Three tabs — **Model Graph**, **Bench & Rank**, **Optimize Kernels** — and each
consumes the previous one's output. The page restores the latest profile,
benchmark run and sessions on load, and an empty tab states its precondition
and the next action rather than being blank.

**The page renders; it does not derive.** Shapes, dtypes and symbol resolution
arrive from the server as structured data (`op.display = {sym, concrete,
dtypes}`). The page used to compute all of it, with four transcriptions of
Python that already existed — and its symbol resolver disagreed with the
Python one on additive composites. If you are about to write model logic in
JavaScript, it belongs on the server.

Deliberate absences, each of which was tried and removed:

- **No Shape Matrix tab.** The matrix is the benchmark's input; the run's
  report workbook already carries it as a sheet.
- **No separate plan/run/rank buttons.** They can only run in that order, so
  choosing between them was never a decision. One `▶ Bench & Rank`.
- **No combined prefill+decode ranking in the UI.** The same kernel is a
  compute-bound GEMM at prefill and a memory-bound GEMV at decode; a combined
  row belongs to neither. (The combined list stays in `targets.json` for the
  optimizer skill.)
- **No candidate list on Optimize Kernels.** The ranked table *is* the
  selection; a session is opened with its row's `🚀 optimize` button.
- **No `max_parallel` knob.** Concurrency is the size of the GPU pool.
- **No budget-per-case knob.** The budget is derived from the profiled shapes.

## Pitfall index

Each row is an invariant that was learned from a specific wrong result. The
reasoning lives in the named function's docstring; the test is what stops it
coming back.

### The shared vocabulary (`breakdown/core/`)

| Symptom if broken | Holder | Test |
|---|---|---|
| A real kernel (`aten::topk`) is reported as framework overhead and vanishes from the ops table, the backend chart and the benchmark | `opnames.is_framework` (compute set consulted before the prefix shorthand) | `test_core_opnames.py::TestPrefixShadowing` |
| An int4 weight is charged twice its size, halving its AI and flipping which roof bounds it | `dtypes.DTYPES` (`nbytes` is a float) | `test_core_dtypes.py::TestPackedTypes` |
| An index operand is counted at 2 bytes instead of 8 | `dtypes._ALIASES` (the profiler emits C++ type names: `long int`) | `test_core_dtypes.py::TestLookup` |
| The same operand reads `f16` in one artifact and `fp16` in another | `dtypes.Dtype.label` (one vocabulary) | `test_core_dtypes.py::TestLabels` |
| `2·I_moe` resolves to `3072` plus an unlexable tail | `dims._tokens` (longest registered name wins) | `test_core_dims.py::TestLexingAgainstTheSymbolTable` |
| A display shape folds a swept variable into a bare number | `dims.partial`, `dims.is_variable_name` | `test_core_dims.py::TestDisplayForm` |
| A "symbolic" shape is a copy of the concrete one beside it (`[S, 2560] × [6144, 2560]`) | `dims.partial` (`keep=None` keeps every registered name) | `test_core_dims.py::TestDisplayForm`, `test_shape_matrix_export.py::test_symbolic_shapes_column` |
| A TP=1 profile has no `/TP`, so a TP sweep divides nothing and every TP collapses to one case | `symbols.SHARDED_SYMBOLS` (declared, not inferred from the arithmetic) | `test_pipeline.py::TestSymbolicShapeCompleteness` (`a_tp1_profile_still_marks_the_sharded_dims`) |
| A shard resolves to 0 and every kernel rejects the shape | `dims.is_sharded` (clamp to 1) | `test_core_dims.py::TestSharding` |
| `index_topk` is claimed by the MoE family and named after the router | `opnames.first_family` (order-dependent tables) | `test_core_opnames.py::TestFamilies` |
| An op is a collective to one stage and not to another | `opnames.COLLECTIVE_NAMESPACES` (one list) | `test_core_opnames.py::TestCollectives` |
| A hand-written SYCL kernel is reported as compiled Triton output | `opnames.library_of` (FlashInfer/flash_xpu probed first) | `test_core_opnames.py::TestLibraries` |

### Reconstruction (`breakdown/trace/`)

| Symptom if broken | Holder | Test |
|---|---|---|
| Module names collapse to class heuristics | `module_hooks.install_module_span_hooks`, `forest._build_raw_forest` | `test_module_spans.py::TestNamedSpanReconstruction` |
| Python-launched kernels have no shapes or cannot be replayed | `kernel_hooks`, `kernels._apply_recorded_args` | `test_kernel_spans.py` |
| Trace file is not valid JSON after a capture | `trace_common.kernel_span_label` (base64) | `test_kernel_spans.py::TestKernelSpanHelpers` |
| Prefill phase vanishes; both phases identical | `phases._partition_steps` (subtree size, not device time) | `test_pipeline.py::TestPhasePartition` |
| Phases inverted on a two-pass decode capture | `phases._classify_steps` (`token_dim > batch_size`) | `test_pipeline.py::TestPhasePartition` |
| Decode graph grows `28`/`30` nodes beside `B` | `phases._classify_steps` (steady-state filter) | `test_pipeline.py` (`partial_batch_decode_steps_dropped`) |
| Decode latency skewed by one-time costs | `phases._classify_steps` (first step is warmup) | `test_pipeline.py` (`first_decode_step_dropped_from_average`) |
| A layer's post-attention all-reduce disappears | `collapse._merge_modules` (occurrence-indexed ops) | `test_pipeline.py` (`repeated_same_signature_op_kept_distinct`) |
| MoE shared experts vanish from the graph | `forest._hoist_modules_under_ops` | `test_pipeline.py` (`module_wrapped_in_fused_op_is_hoisted`) |
| An empty duplicate module beside the real one | `forest._coalesce_duplicate_child_modules` | `test_pipeline.py` (`duplicate_shared_experts_module_coalesced`) |
| A layer appears to start with copy+allreduce | `collapse._finalize_node` (`order`) | `test_pipeline.py` (`op_order_interleaved_with_children`) |
| Expert GEMM time collapses into the wrapping op | `rules._RUNTIME_CATEGORIES` (includes `cuda_driver`) | `test_pipeline.py` (`cuda_triton_moe_experts_surfaced`) |
| Bogus `triton::cudaEventQuery` leaf ops | `kernels._collect_kernel_launches` | `test_pipeline.py` (`runtime_bookkeeping_not_surfaced_as_kernel`) |
| Device time silently dropped | `kernels._attribute_kernels`, `graph.kernel_coverage` | `test_pipeline.py` (`module_less_kernel_time_conserved`, `fixture_traces_every_in_step_kernel_on_leaf`) |
| Hidden dims become `0` at `ctx=0` (SIGFPE) | `graph.build_graph_from_trace` (`C` uses `setdefault`) | `test_pipeline.py` (`config_dim_wins_over_a_colliding_context`) |
| A swept MoE shape stops matching its token operand | `symbols._resolve_shape` (routed rows are `topk·S`) | `test_pipeline.py` (`moe_routed_rows_scale_with_the_token_dim`) |
| The expert fan-out scales with the swept batch | `symbols._resolve_shape` (router axis first) | `test_pipeline.py` (`router_axis_is_not_swept…`) |
| Indexer dims render as unrelated model dims | `symbols.SymbolTable.add_scoped` | `test_golden_graph.py` |
| A concrete structural integer leaks into a shape | `symbols.symbolize_allocations` | `test_pipeline.py::TestSymbolicShapeCompleteness` |
| Attention has no analytic cost, ranks as free | `graph._annotate_attention_kv`, `cost.op_flops` | `test_pipeline.py` (`graph_attention_flops_account_for_the_cached_context`) |
| TP graph built from the wrong rank | `profiling.traces._rank0_first` | `test_trace_download.py::TestRank0Selection` |
| A `_prefill`+`_decode` upload rebuilds decode only | `profiling.uploads.build_from_uploads` | `test_upload_two_pass.py` |

### Replay benchmark (`breakdown/bench/`)

| Symptom if broken | Holder | Test |
|---|---|---|
| A kernel reads outside its buffers, or the device is lost | `inputs.build_args` (no synthesizer -> refuse) | `test_bench_resolve.py` |
| An accumulated counter runs off the end of its buffer | `recipes.outputs(..., single_rep=)` | `test_bench_resolve.py`, `test_bench_rank.py` |
| A small kernel "measures" the timer | `timing.measure` (repeat inside the window, subtract the floor) | `test_bench_rank.py` (`a_fast_kernel_is_repeated_inside_the_window`) |
| Length synthesizers fall back to 1; attention 80x too fast | `inputs._build_positional` (derives the sweep context) | `test_bench_resolve.py` |
| An operand after a 0-dim tensor gets the wrong shape | `spec._substitute_dims` | `test_bench_spec.py` |
| The dominant MoE GEMM disappears from the targets | `spec.build_cases` (`BenchCase.points`) | `test_bench_rank.py` (`sweep_invariant_case_is_ranked_at_every_point`) |
| A GEMM at 30 % of peak is called "memory-bound" | `cost.bound_of` (AI vs machine balance) | `test_bench_rank.py` |
| "utilization 300 % of peak" on a streaming kernel | `cost.effective_bw_gbs` (cache roof) | `test_bench_rank.py` |
| Every elementwise kernel looks ~99 % idle | `cost.compute_peak` (vector vs matrix unit) | `test_bench_rank.py` |
| "utilization 3902 % of peak" on a paged-cache op | `opnames.TABLE_LOOKUP_OPS` | `test_pipeline.py::TestEstimation` |
| A replay far faster than the trace is trusted | `rank` fidelity check (`RankConfig.fidelity_floor`) | `test_bench_rank.py` (`replay_far_faster_than_the_profile_is_flagged`) |
| An unmodelled op is silently retired | `RankConfig.max_credible_util` -> `check_cost_model` | `test_bench_rank.py` |
| A bad shape wedges the device and fails every later op | `runner` (one process per op) | `test_bench_replay.py` (GPU) |
| Collective ranks desynchronize / oneCCL segfaults | `collective.launch` (fixed schedule, no persistent cache) | `test_bench_resolve.py::TestWorkerEnvironment` |
| Every worker re-pays AOT/JIT on its first case | `worker.bench_env` | `test_bench_resolve.py::TestWorkerEnvironment` |
| A several-hundred-MB operand times out its worker | `inputs.make_tensor` (target dtype) | `test_bench_resolve.py::TestOperandAllocation` |
| A partial re-run deletes the rest of the run | `runner.run` | `test_bench_rank.py` |

### Profiling, runs and sessions

| Symptom if broken | Holder | Test |
|---|---|---|
| Module and kernel spans silently absent | `profiling.launch` (`VLLM_ALLOW_INSECURE_SERIALIZATION`) | — (logged at ERROR with the hook count) |
| A headless profile dies at engine start-up: request longer than `max_model_len` | `profiling.fit_max_model_len` (applied in `_run_profile`, not in the route) | `test_pipeline.py::TestQueryContextProfiling` |
| Decode dispatched in partial waves (`29 + 3`) | `profiling.launch._scheduler_pin` | `test_two_pass_merge.py::TestSchedulerPin` |
| A prefill/decode batch pair collapses to one pass | `profiling.traces._merge_two_pass_result` | `test_two_pass_merge.py` |
| Downloading a trace and re-uploading it loses the config | `profiling.traces.trace_download_name` + `_parse_trace_filename` (inverses, kept adjacent) | `test_trace_download.py`, `test_upload_two_pass.py` |
| Swapping the run state has no effect on the code that reads it | `profiling.__init__.__getattr__` (forwards, never binds) | `test_upload_two_pass.py`, `test_bench_api.py` |
| The model config cache is orphaned by a file move | `profiling.runstate._CONFIG_CACHE_DIR` (anchored to the package, not to `__file__`) | — |
| A half-written `state.json` is read by a poll | `runs.write_json` (temp + rename) | `test_bench_api.py` |
| A restart loses the profile everything derives from | `runs`, `profiling.runstate` | `test_bench_api.py` |
| Two agents share a GPU and measure each other | `optimize/scheduler.DevicePool` | `test_optimize_scheduler.py` |
| A leaked lease strands a GPU ("the queue is stuck") | `optimize/manager` (release in `finally`) | `test_optimize_scheduler.py` |
| A session burns a GPU on an op with nothing to win | `optimize/prompt._REFUSALS` | `test_optimize_prompt.py` |
| The log pane shows every session ever run | `optimize/manager` (log opened `"wb"`) | `test_optimize_api.py` |
| A dead session is reported `running` forever | `service.optimize_sessions` | `test_optimize_api.py` |
| An endpoint ships the whole brief as `argv` | `service.optimize_sessions` | `test_optimize_api.py` |
| Session artifacts land in the wrong tree | `OptimizeSession.root` (pinned at creation) | `test_optimize_api.py` |

### The UI

| Symptom if broken | Holder | Test |
|---|---|---|
| A dimension reads as a number on one surface and a symbol on the other | `shape_derive.annotate_display_shapes` (the server resolves; the page renders) | `test_golden_semantics.py::TestDisplayShapes` |
| A half-resolved shape renders as numbers with a symbol hiding among them | `annotate_display_shapes` (`concrete` is all-or-nothing) | `test_golden_semantics.py::TestDisplayShapes` |
| A finished profile leaves the sweep on unrelated defaults | `applyProfileResult` -> `applyProfileToSweep` | — (checked in a browser) |

## Updating documentation

After a change that adds a feature, changes an API, moves files, or discovers
a new invariant:

1. **The deep dive** (`breakdown/*/README.md`) — the narrative for that
   subsystem.
2. **The docstring** on the function that holds the invariant — the reasoning,
   including the wrong number the naive version produced.
3. **This file** — the structure listing, the endpoint table, the conventions,
   and a row in the pitfall index (symptom -> holder -> test).
4. **README.md** — user-facing features, CLI, output formats.

Verify: `ls breakdown/ breakdown/core/ tests/` matches the structure listing,
and `grep "@app.route" app.py` matches the endpoint table.
