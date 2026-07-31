# AGENTS.md — vllm-xpu-breakdown

Instructions for AI agents working in this repository.

## Overview

This is a profiling and static-analysis tool for vLLM inference on Intel XPU (GPU) hardware. It visualizes which backend handles each operation during inference:
- **vllm-xpu-kernels** — Custom SYCL/DPC++ kernels
- **torch-xpu-ops** — PyTorch ATen on XPU (via oneDNN/oneMKL)
- **triton** — Intel Triton-compiled kernels
- **cpu** — CPU fallbacks
- **framework** — Tensor reshaping, profiler overhead

The tool's in-app **Model Graph is always reconstructed from a profiling run**;
there is no interactive static-graph view, and no config-driven graph builder.
**Dynamic profiling (profile-first)** — Runs actual inference via vLLM on Intel
XPU, then **reconstructs the model graph directly from the torch profiler trace**
(nn.Module call stack + `Input Dims` shapes + `Input type` dtypes + kernel device
time via the correlation→runtime→External-id chain). The reconstructed tree
reflects what actually executed, so it tracks whatever vLLM/the backends
dispatched instead of relying on a hand-maintained static graph. Both the web UI
graph and the Shape Matrix export are built from this reconstruction.

> **Environment assumption:** the profiling path assumes torch-xpu and vLLM are
> installed and an Intel XPU is available. (`model_info.py` happens to be
> import-light, but supporting a torch-free CPU-only install is no longer a
> design goal.)

## Project Structure

```
app.py                    — Flask web server (API + static serving + exports)
run_profile.py            — CLI profiling entry point
chat.py                   — Interactive chat with profiling
breakdown/
  graph_from_trace.py     — Profile-first graph reconstruction from a trace
  shape_derive.py         — Symbolic shape/dtype/memory resolution (torch-free, shared)
  shape_matrix.py         — Graph + config sweep → matrix rows (the export serializes them)
  shape_matrix_xlsx.py    — Excel serialization of matrix rows
  bench/                  — Replay benchmark: dispatched ops → measured → ranked targets
    spec.py               — Matrix rows → BenchCase replay specs (+ skip/dedup rules)
    resolve.py            — Dispatch name → callable + schema (PYTHON_API for kernel ops)
    inputs.py             — Schema-driven operand materialization + synthesizer registry
    recipes/{common,xpu,cuda}.py — Per-op overrides, output args, skip reasons
    recipes/attention.py    — Paged attention / KV-cache write replayed context-free
    timing.py             — Device-event windows, overhead subtraction, operand restore
    worker.py             — Benchmark one op in its own process → results.jsonl
    runner.py             — Orchestration, per-op timeouts, incremental run_result.json
    collective.py         — Multi-rank replay of c10d ops (rank 0 is recorded)
    estimate.py           — Roofline utilization + per-op time budgets
    rank.py               — calls × latency × roofline headroom → targets.json
    reports.py            — results.jsonl → summary / coverage / workbook
    store.py              — output/bench/<run_id>/ layout + provenance
    history.py            — SQLite history + regression detection
    kernel_sources.json   — op/backend → repo, files, build and test commands
    cli.py                — python -m breakdown.bench {plan,run,rank,report,case,history,all}
  module_hooks.py         — Capture-time module-name spans (forward hooks)
  module_naming.py        — Fallback: recover names from named_modules() overlay
  model_info.py           — HuggingFace config fetcher/summarizer + min_profile_layers
  analyzer.py             — Op analysis (shapes, memory, FLOPs, AI)
  classifier.py           — Op → backend classification
  registry.py             — Known vllm-xpu-kernels ops list
  trace_parser.py         — Chrome trace JSON parser + module/role helpers
  trace_common.py         — Torch-free trace helpers (overhead-event filtering)
  profiler.py             — vLLM profiler integration
  report.py               — Text/CSV/JSON report generation
  visualize.py            — Plotting utilities
static/
  index.html              — Single-page web UI (HTML + CSS + JS)
scripts/
  run_profile.sh          — Shell wrapper for profiling
  compare_modes.sh        — Compare eager vs compile modes
tests/
  test_pipeline.py              — Unit tests (requires torch)
  test_shape_matrix_export.py   — Shape Matrix Export endpoint tests
  test_bench_spec.py            — Bench: rows → replay cases, skip/dedup (no GPU)
  test_bench_resolve.py         — Bench: dispatch resolution + operand building (no GPU)
  test_bench_rank.py            — Bench: ranking, timing plan, budgets, history (no GPU)
  test_bench_api.py             — /api/bench/* endpoints (no GPU)
  test_bench_replay.py          — Bench: end-to-end replay on a real device (GPU)
  test_profile_reduced_layers.py
  test_real_profile.py          — Integration test (requires GPU)
```

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run web UI (static analysis works without loading model weights)
python app.py --port 8080

# Run CLI profiling (requires Intel XPU + vLLM)
python run_profile.py --model Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768
```

## Testing

```bash
# Unit tests (no GPU required)
pytest tests/test_pipeline.py -v

# Shape Matrix Export tests (no GPU required)
pytest tests/test_shape_matrix_export.py -v

# Replay benchmark tests (no GPU required)
pytest tests/test_bench_spec.py tests/test_bench_resolve.py \
       tests/test_bench_rank.py tests/test_bench_api.py -v

# Full integration (requires Intel XPU hardware)
pytest tests/ -v
```

## Key Architecture Decisions

> **MLA profiling is supported on XPU.** Dynamic profiling of MLA models
> (DeepSeek-V2/V3/V4, GLM-MoE-DSA) used to be hard-rejected in `app.py`
> because MLA required FlashAttention. vLLM-XPU now provides MLA backends
> (`TRITON_MLA` for dense MLA, `XPU_MLA_SPARSE` for DeepSeek sparse attention),
> so that block was removed. Do not re-add an MLA architecture guard in the
> profiling path.

### Profile-First Graph Reconstruction (`graph_from_trace.py`)

The model graph is reconstructed straight from the torch profiler trace.
`build_graph_from_trace(trace_path, summary, tp_size, batch_size, quantization)`
returns a serialized module/op tree (`{prefill, decode, symbols, config,
has_timing, timing_*, source}`) that the frontend and the Shape Matrix export
consume unchanged. How it works:

- **Module tree** — built from **capture-time module spans** when present
  (research R1, the primary path): `module_hooks` installs forward hooks that
  open a `record_function("module::<qualified_name>::<Cls>")` span around every
  module's forward, so the trace carries `user_annotation` events with the
  **real attribute path** (`model.layers.0.self_attn.q_norm`). `_build_raw_forest`
  auto-detects those spans on the worker thread and builds module nodes directly
  from them (real names, no overlay, no ordering assumption), skipping the
  class-only `nn.Module: <Cls>_<idx>` frames to avoid duplication. **Legacy
  fallback:** traces without spans (older runs / uploads) use the `nn.Module:
  <Cls>_<idx>` python_function events + the `module_naming` overlay. Either way,
  events nest by time-containment (sort by `(ts asc, end desc)`, pop a stack
  while the top ends before the current node).
- **Op shapes** — taken from each `cpu_op`'s `Input Dims` / `Input type`, then
  symbolized (`_symbolize`) against the model config so dims show as `S`, `B`,
  `C`, `H`, `/TP`, etc.
- **Device time** — kernels link to their launching `cpu_op` via
  `kernel.correlation → runtime.correlation→External id → cpu_op.External id`;
  kernel durations are summed per op (`self_dev`) and rolled up post-order
  (`sub_dev`).
- **Phase split** — `_partition_steps` groups module roots into inference steps
  (main model class starts each step; trailing LogitsProcessor/Sampler attach to
  it), classified prefill vs decode by the pass's max matmul token dim. The
  **main model class is chosen by largest module subtree**, NOT by device time:
  the `lm_head` vocab projection inside `LogitsProcessor` (a `V`-wide matmul run
  once per decode step) can outweigh the entire model forward, which previously
  mis-selected `LogitsProcessor` as the main class — collapsing every step to a
  1-token pass so the prefill phase vanished (`prefill: None`) and both phases
  looked identical. Do NOT revert `_partition_steps` to a `sub_dev`-max heuristic
  (see `_subtree_module_count` + `TestPhasePartition`).
- **Repeat collapse** — adjacent structurally-identical sibling layers merge into
  one node with `repeat_count`, timing averaged. Dense vs MoE layers have
  different structural signatures so they stay as separate runs. The structural
  signature includes the node's (recovered) `name`, so distinctly-named siblings
  (`q_norm`/`k_norm`) are kept apart while genuinely-repeated layers still merge.
  Capture-time spans name repeated `ModuleList` elements by their **list
  attribute** (`model.layers.0` → `decoder_layer`, not `0`) so they still merge.
- **Execution-order interleaving** — each direct op and child module of a node
  carries an `order` index (`_merge_modules` builds it from the base instance's
  children sorted by `ts`; `_finalize_node` stamps it on every op/child), so the
  UI renders a module's direct ops and submodules **interleaved in the order
  they ran** instead of all-ops-then-all-children. This matters because vLLM
  ops frequently nest one level *above* their semantic module: a cpu_op whose
  `ts` lands just after the child module's forward event closes (the forward
  returned before the async op ran) time-contains under the *parent*, e.g. the
  decoder layer's post-attention `c10d::allreduce_` + `aten::clone` (the XPU
  decomposition of `fused_allreduce_gemma_rms_norm` → allreduce + norm) and
  MiniMax-M3 attention's `fused_minimax_m3_qknorm_rope_kv_insert` (runs after
  `qkv_proj`). Without the `order` field these floated to the top of their
  parent, making the layer look like it *started* with copy+allreduce and
  putting `qknorm_rope` ahead of `qkv_proj`. The raw time-containment forest was
  already correct; only rendering reordered. Same-named ops whose exact
  `(name, shapes)` signature isn't in the base layout fall back to a
  name-keyed order so multi-shape merges keep them grouped at their slot. The
  frontend (`buildTreeNode`) sorts the combined op+child list by `order`
  (legacy results without `order` fall back to ops-then-children).
- **Repeated same-signature ops are occurrence-indexed, not merged.**
  `_merge_modules` groups a module's direct ops by `(name, shapes, occurrence
  index within the instance)` — mirroring the child-module grouping — **not**
  by `(name, shapes)` alone. A TP decoder layer dispatches two *identical*
  `c10d::allreduce_` residual reductions: its own **post-attention** one (before
  `post_attention_layernorm`) and the previous layer's **post-MLP** one, which
  is dispatched after that layer's forward returns and so time-contains at the
  *start* of this layer (before `input_layernorm`). Keying ops by `(name,
  shapes)` alone collapsed both into a single node at the leading position,
  **hiding the post-attention allreduce** in every layer after the first (the
  symptom: "allreduce before `post_attention_layernorm` missing in the 2nd/3rd
  layers"). Occurrence indexing keeps them distinct and still aligns each
  occurrence across merged forward passes. See
  `TestGraphFromTrace.test_repeated_same_signature_op_kept_distinct`.
- **Modules wrapped inside a fused custom op are hoisted out
  (`_hoist_modules_under_ops`).** vLLM dispatches some fused blocks as a single
  custom `cpu_op` whose implementation *internally calls real `nn.Module`
  forwards*, so by time-containment the wrapped module subtree nests **under the
  op event** rather than beside it. The clearest case is the MoE block:
  `vllm::moe_forward_shared` wraps the `shared_experts` MLP
  (`MergedColumnParallelLinear` → `SiluAndMul` → `RowParallelLinear`) plus the
  router/expert math, so the whole `shared_experts` subtree lands under the op.
  Reconstruction only surfaces a module's **direct** child modules/ops
  (`_module_children` / `_direct_ops`), so a module buried under an op was
  silently dropped — the `FusedMoE` node showed only its flat op list and the
  shared experts' `gate_up_proj`/`down_proj` matmuls **vanished from the graph**
  (symptom: "MoE layer missing details — should be shared_experts → router →
  moe → reduce"). `_hoist_modules_under_ops` lifts every module whose enclosing
  parent is an op up to its **nearest ancestor module** (preserving each
  module's own subtree); order is restored from timestamps by `_merge_modules`.
  It runs **after** `_attribute_kernels` (which needs the non-overlapping
  time-containment forest for launch-site lookup) and **before**
  `_compute_sub_dev` (so the hoisted subtree's device time rolls up **once**,
  under its module, not also inside the wrapping op — verified device-conserving).
  See `TestGraphFromTrace.test_module_wrapped_in_fused_op_is_hoisted`.
- **The same module object recorded twice in one forward is coalesced
  (`_coalesce_duplicate_child_modules`).** vLLM's MoE shared-experts overlap
  enters `shared_experts` **twice** within one MoE forward: once as an **empty
  shell** whose compute is fused into the sibling `vllm::moe_forward_shared`
  custom op (hoisted out empty by `_hoist_modules_under_ops`) and once as the
  **real MLP** forward. Both events carry the *identical* profiler instance label
  (`SharedExperts_0` twice), so they are the same object — but reconstruction
  used to render them as two sibling nodes, one a spurious **empty**
  `SharedExperts` next to the real one (symptom: "MiniMaxM3MoE graph
  inconsistent with the trace in prefill"). The empty/real order varies
  (empty-**first** in the CUDA prefill trace, empty-**last** in decode), so the
  coalesce keys purely on the shared full label, unions the duplicates' child
  ops/modules into the earliest occurrence and sums their directly-launched
  device time. Distinct siblings have distinct instance labels (`_0`/`_1`/…), so
  it's a no-op for them (a single un-duplicated module like the empty
  `MiniMaxM3IndexerTritonImpl` is left untouched). **Only real instance-indexed
  module events are eligible** (`_strip_instance_idx(label) != label`): synthetic
  functional-frame modules (`_FUNCTIONAL_MODULE_FRAMES` →
  `FusedAllreduceGemmaRMSNorm`, `FusedTopKBiasRouter`, `XpuFusedMoE`, …) carry a
  **bare class label with no index**, so genuinely-distinct repeats — e.g. a
  Gemma decoder layer's **two** `fused_allreduce_gemma_rms_norm` (pre- and
  post-attention) — legitimately share a label and must **not** be merged; they
  are skipped. Runs **after** `_hoist_modules_under_ops` (which relocates the
  empty shell to a sibling of the real forward) and **before** `_compute_sub_dev`
  (so the unioned subtree's device time rolls up once). Verified against the real
  MiniMax-M3 CUDA (duplicate shell present) **and** XPU (no duplicate — XPU
  doesn't double-record shared experts, so the coalesce is a clean no-op) traces.
  See `TestGraphFromTrace.test_duplicate_shared_experts_module_coalesced` and
  `test_synthetic_frame_duplicates_not_coalesced`.
- **A `RowParallelLinear` inside an MLP/expert module names to `down_proj` on
  every device, not just CUDA.** `RowParallelLinear` is both the attention
  output projection (`o_proj`) and the MLP/MoE down projection (`down_proj`), so
  the class-heuristic default (`_module_display_name` → `o_proj`) is wrong for
  the MLP one whenever the reference-name overlay didn't tag it.
  `_disambiguate_child_name` resolves it by **parent module type**
  (MLP/expert/feedforward → `down_proj`; attention → `o_proj`). This used to be
  gated to `is_cuda`, which broke the **hoisted `shared_experts` MLP on XPU**:
  after `_hoist_modules_under_ops` lifts it out of `moe_forward_shared` it sits
  under `FusedMoE`, but the reference tree lists it under `MoE.shared_experts`,
  so `_apply_ref_names` can't align it and its `down_proj` stayed unnamed →
  mislabeled `o_proj` (while the dense MLP's overlay-named `down_proj` was
  correct). The parent-type disambiguation is now device-agnostic; only the
  GPU-specific "multiple RowParallelLinear in one MLP, keep just the last" async
  compensation stays `is_cuda`-gated. See
  `TestGraphFromTrace.test_rowparallel_in_mlp_named_down_proj_on_xpu`.
- **MoE router / experts are surfaced from functional python_function frames
  (`_FUNCTIONAL_MODULE_FRAMES`).** vLLM runs the MoE routing (`fused_topk_bias`
  — sigmoid/topk/gather) and expert compute as plain `python_function` frames
  inside the fused `vllm::moe_forward_shared` op, **not** as `nn.Module`
  forwards. The routed-expert entry frame is **backend-specific**: XPU dispatches
  through `xpu_fused_moe` (`fused_moe_interface.py` — grouped
  GEMM/remap/gather/swiglu), CUDA through the Triton modular kernel `apply`
  (`experts/triton_moe.py` — `moe_align_block_size` → `fused_moe_kernel` grouped
  GEMM → activation → `moe_sum`). With no module boundary, their ops and kernels
  collapsed into the single `moe_forward_shared` op node, so the `FusedMoE` graph
  showed neither the router nor the experts (only the hoisted `shared_experts`
  MLP). `_functional_module_class` promotes those frames (and the V1 `Sampler`)
  to **synthetic modules** with explicit display names (`router`, `moe` — CUDA's
  Triton-experts class is `TritonExperts`, XPU's `XpuFusedMoE`);
  `_hoist_modules_under_ops` then lifts them out of the wrapping op, so
  `FusedMoE` reads `shared_experts → router → moe → reduce`. Each entry is
  `(path_substr, funcname, synthetic_class, display_name)`; device time is
  conserved (the wrapping op keeps only its residual self time). **On CUDA the
  routed-expert `fused_moe_kernel` grouped GEMM is Triton-launched via the CUDA
  *driver* API (`cuLaunchKernelEx`, cat `cuda_driver`), which must be a
  launch-site category (`_RUNTIME_CATEGORIES`) or the GEMM time falls back to
  External-id attribution and collapses into `moe_forward_shared`'s start instead
  of landing on `moe`** (see the launch-site pitfall below). See
  `TestGraphFromTrace.test_moe_router_and_experts_surfaced_from_functional_frames`
  and `test_cuda_triton_moe_experts_surfaced`.
- **The fused all-reduce + RMSNorm is grouped as a parent node
  (`_FUNCTIONAL_MODULE_FRAMES`).** Gemma-style models (MiniMax-M3) fuse the
  residual tensor-parallel all-reduce with the following RMSNorm as
  `fused_allreduce_gemma_rms_norm` — a `python_function` that wraps **both** the
  `c10d::allreduce_` op **and** the `MiniMAXGemmaRMSNorm` module. Without a
  boundary the all-reduce and the norm float up as two unrelated siblings of the
  decoder layer (a bare `c10d::allreduce_` op next to a lone norm), so a layer
  appeared to have an unexplained "norm" at its edges — and because vLLM
  dispatches the pre-attention fused norm right after the *previous* layer's
  forward returns, by time-containment it lands at the **start** of the current
  layer (so a layer could show a fused norm at both its beginning, the
  previous layer's tail, and after attention). Promoting the frame makes it a
  parent node `fused_allreduce_gemma_rms_norm → {allreduce, norm}` so the fusion
  is explicit. The frame path is MiniMax-specific, so other models are
  unaffected. The wrapped `MiniMAXGemmaRMSNorm` moves under the fused node
  (labeled `norm`); only the **bare** (un-fused) `input_layernorm` — the first
  layer's, which has no preceding all-reduce — stays a direct layer child. See
  `TestGraphFromTrace.test_fused_allreduce_gemma_rms_norm_grouped`.

The old `annotate_graph_timing` / `annotate_graph_from_modules` /
`parse_trace_with_modules` paths were **removed**. Do not reintroduce a
static-overlay step in the profiling worker.

### Module Attribute Naming — capture-time spans (`module_hooks.py`) + overlay fallback (`module_naming.py`)

Same-class sibling modules (Qwen3 `q_norm`/`k_norm`, both `RMSNorm`;
`input_layernorm`/`post_attention_layernorm`; ...) are indistinguishable in a
class-only trace. There are **two** ways to recover the real names; the first is
now primary:

**1. Capture-time spans (`module_hooks.py`, research R1 — primary).**
`install_module_span_hooks(model)` registers a `register_forward_pre_hook` /
`register_forward_hook` pair on every `named_modules()` entry that opens a
`record_function("module::<qualified_name>::<Cls>")` span around the forward.
These `user_annotation` events nest exactly like the module forwards and embed
the exact attribute path + class, so `graph_from_trace._build_raw_forest`
reconstructs the tree with real names **directly from the trace** — no
alignment, no registration-order assumption, causally correct even under async
execution (the span is opened at dispatch time). During live profiling
`_run_profile` installs the hooks in the worker via
`LLM.apply_model(install_module_span_hooks_on)` right before `start_profile` and
removes them after `stop_profile` (best-effort; warmup runs stay unhooked). The
label grammar and the display-name derivation (`module_span_display_name` — a
numeric `ModuleList` leaf like `layers.0` becomes `decoder_layer`/`layers` so
siblings still collapse) live torch-free in `trace_common`.

**2. `named_modules()` overlay (`module_naming.py`) — fallback.** Kept for legacy
/ upload traces that have no spans. `build_graph_from_trace` applies it only when
the reconstructed forest has no captured names (`_forest_has_named_modules`). It
ports the *idea* from the retired `torch_export` branch — derive names from the
model's `named_modules()` — and overlays them onto the accurate profile-based
tree instead of rebuilding it.

- **`build_ref_tree(named_modules)`** — pure/torch-free. Builds a reference tree
  of `{attr, cls, children, is_group, group_size}` from
  `[(qualified_name, class_name), ...]`. Indexed `ModuleList` containers are
  *inlined* (their `forward` is never called, so they emit no module event) and
  consecutive numeric entries of the same class collapse into one `is_group`
  representative — mirroring the trace's layer collapse.
- **`ref_tree_from_llm(llm)`** — extracts the tree from the *live* vLLM model
  during profiling (via `LLM.apply_model`, falling back to attribute traversal).
  Cheap: the model is already loaded.
- **`ref_tree_from_config(...)`** — `meta`-device instantiation fallback
  (always trusts remote code). Heavy + network. Used by the trace-upload path **and** as a live-path fallback: if
  `ref_tree_from_llm` returns `None` during profiling, `_run_profile` retries
  with `ref_tree_from_config`. The whole naming path logs its outcome
  (`vllm_xpu_breakdown` logger) — a "reference tree available but no
  names landed" warning means alignment (not acquisition) failed.
- **Alignment** — `graph_from_trace._apply_ref_names` walks the *raw* module
  forest against the reference tree, matching children greedily by
  `(class, order)` (reusing a matched representative when the trace has more
  same-class siblings than the collapsed reference, e.g. dense + MoE layer
  groups). It **unwraps reference levels absent from the trace**
  (`module_naming._effective_ref_children`): vLLM nests the decoder stack under
  an inner `*Model` module (`*ForCausalLM → *Model → [embed, layers, norm]`)
  whose `forward` often emits **no** trace module event, so the trace nests the
  stack directly under `*ForCausalLM`. Without the unwrap, child-class matching
  stalls at that missing level and every submodule keeps its class-heuristic name
  (`q_norm`/`k_norm` → `norm`). Applied **before** finalization/collapse so
  recovered names feed the structural signature.
  `build_graph_from_trace(..., ref_module_tree=...)` opts in; without a ref tree
  the class-heuristic names are used (backward compatible). The assumption:
  sibling modules **of the same class** execute in their registration/definition
  order (holds for q_norm-before-k_norm, the layer norms, etc.). Capture-time
  spans (path 1) do **not** rely on this assumption.

### Symbolic Shape System

Op shapes use symbolic expressions for dimensions:

- **`cfg` dict** stores original config.json values (e.g., `cfg["num_heads"] = 32`)
- **`cfg["_tp_*"]` keys** store TP-divided values for numeric calculations (e.g., `cfg["_tp_num_heads"] = 8` when TP=4)
- **Shape strings** always use `/TP` for TP-aware dims: `"n_h/TP"`, `"QKV/TP"`, `"I/TP"`, `"V/TP"`
- **`symbols` dict** in result contains original (undivided) values + `"TP": tp_size` (always present, even TP=1)
- **Variable symbols** stay symbolic in exports: `S` (seq_len), `B` (batch), `C` (context_len), `TP`
- Frontend `symTooltip()` resolves `/TP` suffix by dividing base value by TP
- **No concrete structural dims leak.** `_build_symbol_tables` registers the
  config-derived dims (`H`/`n_h`/`n_kv`/`d`/`I`/`I_moe`/`V`/`n_h·d`/`QKV`/`2·I`/`E`
  + their `/TP` shards) **plus**: `P` = `max_position_embeddings` (rope cos/sin
  cache length, **not** divided by TP — the table is replicated per rank) and,
  for DeepSeek/MiniMax-M3-style sparse attention, `QKV_idx` =
  `(n_h+2·n_kv)·d + 2·(sparse_num_index_heads·sparse_index_dim)` — the qkv_proj
  fuses the lightning-indexer's q/k projections, so its output width exceeds the
  plain `QKV` (M3: `QKV`=9216 vs `QKV_idx`=10240; per-rank `2304` vs `2560`).
  Run-specific **allocation** dims that aren't config/S/B/C-derivable are then
  symbolized by `_symbolize_runtime_dims` (after phase-tree build) with
  **observed-value** symbols recorded in the legend: `N_kv` (paged KV-cache slot
  count in `fused_minimax_m3_qknorm_rope_kv_insert` / cache ops — XPU `17286`,
  CUDA `114848`), `M_moe` (CUDA Triton-MoE expert-GEMM routed-token rows, e.g.
  `silu_and_mul_with_clamp` `[16384, I_moe/TP]`), and `N_moe`/`N_moe2` (1-D
  `moe_align_block_size` sorted-token / expert-block scratch buffers). Distinct
  values under one base are suffixed deterministically (largest keeps the bare
  base). The reconstructed MSA-indexer dims are symbolized just before that by
  `_symbolize_msa_dims` as `n_idx`/`n_idx/TP` (`sparse_num_index_heads`) and
  `K_topk` (`sparse_topk_blocks`) — they must not fall through to the generic
  observed-value symbols, nor to the colliding `n_kv/TP` / `n_h/TP`. Trivial dims (`≤2` — the k/v-pair `2`, `0`/`1` placeholders/broadcasts)
  are intentionally left concrete. Verified: reconstructing the four MiniMax-M3
  `tp4` traces (XPU/CUDA × prefill/decode) leaves **no** concrete structural
  integer `>2` in any op shape (`TestSymbolicShapeCompleteness`). Note: the
  per-rank MoE gate_up width `2·I_moe/TP`=1536 coincidentally equals `H/TP`, so
  it renders `H/TP` (symbolic, benign collision), as do `d`=`sparse_index_dim`=128
  and `rotary_dim`=`n_h`=64.

### Quantization Support

- Quantization affects `weight_dtype_bytes` (reduced from model dtype)
- `_get_tensor_dtype()` determines per-tensor dtype based on op role (weight vs activation)
- Supported: fp8, gptq, awq, marlin, bitsandbytes, int4, int8

### Shape Matrix Export (`/api/export/shape-matrix`)

Exports a flat Excel table sweeping across configurations:
- Prefill: seq_lens × ctx_lens × batch_sizes × tp_sizes
- Decode: seq_len=1 × ctx_lens × batch_sizes × tp_sizes
- Each row = one (Phase, SeqLen, CtxLen, BatchSize, TP, Op) combination
- Symbolic Shape column keeps S/B/C/TP symbolic, resolves model constants to numbers
- Row limit guard (`_MAX_MATRIX_ROWS = 50000`) prevents excessive generation

**Always profile-derived (there is no config-driven path).** The op set and real
shapes come from the latest completed profiling run for the model. The
reconstructed graph (`_profile_state["result"]["graph"]`) is used as an **op
template**: its real dispatched ops already carry symbolic shapes (`S`/`B`/`C`/
`n_h·d/TP`/`S+C`/...), the exact vocabulary the exporter resolves. For each
config, `_config_symbols` overrides `S`/`B`/`C`/`S+C`/`TP` in a copy of the
profile's `symbols`, every op's shapes are re-resolved (`_resolve_shape_ints`),
and **Memory/FLOPs are recomputed per config** via `_profile_op_memory` /
`estimate_flops`. An **Info** sheet records the profiled config, caveats, and a
validation summary. Filename is
`vllm_xpu_shape_matrix_<model>_<quant>.xlsx`.
> There is **no config-driven builder** — `model_graph.py` was deleted. The
> Shape Matrix is purely profile-derived; do not re-add a config `source` branch
> or a static graph builder.
- **Acquisition + invariants:** requires `_profile_state["status"]=="done"` and
  both the profiled `model_id` **and** the profiled quantization to match the
  requested ones (else 400) — quantization must match because the derived
  dtypes/memory are only valid for the quantization the run actually used
  (compared via the run's `settings["quantization"]`, `""`/`auto`/`none`→None).
  The **frontend guarantees** a matching run exists before calling:
  `ensureProfileForMatrix` reuses the latest completed run only when its model
  **and** `settings.quantization` match the Shape Matrix's `Quantization`
  selector, waits out any in-progress run, or launches a fresh profile
  (`buildProfileBody`, shared with `startProfile`, with `quantization`
  overridden to the selected one) via `runProfileForMatrix` and polls to
  completion. The op *set* is fixed at the
  profiled config — sweeping TP only divides `/TP` dims, it does **not**
  synthesize comm ops that weren't profiled, so profile at **each TP** you need.
  Query/batch/context (`S`/`B`/`C`) are **parametric** and need only one base
  profile: `C` enters solely via the prefill attention KV rows (`S+C`), which
  `graph_from_trace._annotate_attention_kv` rewrites **only when the base profile
  had a non-zero context**. So you do **not** profile per context — one profile
  with any (small, cheap) non-zero context makes every other context length
  derivable; a `context=0` base leaves KV rows as `S` and context can't be
  derived. Device time is not extrapolated into the sweep.
- **Accurate per-tensor dtypes + shape validation:** each reconstructed op carries
  `recorded_shapes` (the numeric shapes as captured) and `input_dtypes` (the real
  per-tensor dtypes from the trace's `Input type`, parsed aligned-with-shapes by
  `graph_from_trace._parse_input_dims_types`). The export uses `input_dtypes` for
  the **Shape** column dtype (via `_format_op_shape_with_dtypes(...,
  recorded_dtypes=...)`, falling back to the `_get_tensor_dtype` heuristic when
  absent) and for a dtype-accurate **Memory** estimate (`_profile_op_memory` sizes
  each input tensor by *its own* dtype — an fp8/int4 weight counts 1 byte while a
  bf16 activation counts 2). The Info sheet adds a **shape round-trip validation**
  (`_validate_derived_shapes`): re-resolving every op's symbolic `input_shapes` at
  the *profiled* config must reproduce `recorded_shapes` (context-annotated
  `S+C`/`C` KV dims excluded, since context is deliberately added, not recorded).
  Note: memory/FLOPs are still analytic estimates — only the op set, shapes,
  dtypes and backends come from the trace.

### Replay Benchmark (`breakdown/bench/`)

Answers the question the Shape Matrix cannot: **which kernel is worth an
optimization session**. It does so by **re-invoking the ops vLLM actually
dispatched** rather than benchmarking substitutes in an external suite. Stages:
rows (`shape_matrix`) → replay cases (`spec`) → measured cases (`worker`/
`runner`, budgeted by `estimate`) → ranked targets (`rank`) → history
(`history`). Every stage runs headless (`python -m breakdown.bench`);
`/api/bench/*` and the **Benchmark & Targets** tab are wrappers.

> The old `breakdown/perf/` pipeline (op-map adapters + a shell-out to
> `$XPU_PERF_HOME/projects/micro_perf/launch.py`) was **removed**. Do not
> reintroduce an op→micro_perf adapter table, a workloads JSON emission, or an
> `xpu-perf` dependency: coverage is now a property of the profile, not of a
> hand-maintained mapping.

- **Replay, not re-implementation.** The trace records each op's dispatch name,
  per-tensor shapes/dtypes/strides and the concrete values of its non-tensor
  arguments (`Concrete Inputs`), so `resolve` maps the name to
  `torch.ops.<ns>.<op>` and `inputs` materializes the recorded operands. Every
  op the profile dispatched is benchmarkable — on XPU and CUDA alike — without
  writing an adapter for it.
- **The rows are the transport, not the .xlsx.** `shape_matrix.build_rows`
  returns the matrix as dicts carrying `_input_args` (the full ordered argument
  slots); the export merely serializes them and `bench.spec` consumes them
  in-process. Do **not** reintroduce a graph → xlsx → pandas round-trip: the
  spreadsheet drops the non-tensor arguments, so a run built from it cannot
  replay anything.
- **The schema selects the overload; the slots select the schema.**
  `aten::add`'s *default* overload is `(Scalar a, Scalar b)` — no tensors at
  all. `resolve._overload` scores overloads against the recorded slot profile
  (argument count, tensor count), and keyword-only schema arguments (`*, Scalar
  alpha=1`) are passed as keywords. Picking by name would call the wrong kernel
  or fail outright.
- **An integer tensor is an index until proven otherwise.** A random
  `slot_mapping` makes a paged-KV kernel scatter across the whole cache; a
  random `rows_per_expert` makes a grouped GEMM read past its input. So
  `inputs` **refuses** to fill an integer operand without a registered
  synthesizer (`MissingSynthesizer`), and the case is reported as
  `needs_synthesizer`. Never relax this into a zeros/random default.
- **Some arguments are outputs the schema does not mark.**
  `_moe_C::remap_hidden_states` takes `rows_per_expert` as an argument but
  *accumulates into it with atomics*; reusing it across calls grows the offsets
  until the scatter writes out of bounds and takes the device down
  (`UR_RESULT_ERROR_DEVICE_LOST`, no traceback). `recipes.outputs(...)` declares
  such arguments so they are allocated zeroed, reset between windows, and
  measured **one call per timed window** (`SINGLE_REP`).
- **A context-bound wrapper is replayed through its context-free kernel entry
  point when it has one (`recipes/attention.py`).** `vllm::unified_attention_-
  with_output` and `vllm::unified_kv_cache_update` take a `layer_name` and pull
  the KV cache, block table and sequence metadata out of vLLM's *forward
  context*, so the **dispatcher op** cannot be called standalone — but what the
  wrapper hides is the context, not the kernel. One level down,
  `fa_utils.flash_attn_varlen_func` / `reshape_and_cache_flash` (the
  vllm-xpu-kernels SYCL kernels on XPU, vllm_flash_attn on CUDA) take exactly
  that context as plain arguments, so `resolve.PYTHON_API` maps both ops there
  (**checked before `NOT_REPLAYABLE`**) and a recipe rebuilds the context: a
  paged `[num_blocks, block_size, n_kv, d]` NHD cache holding `context+query`
  for every sequence, `cu_seqlens_q` / `seqused_k` for the swept operating
  point, and vLLM's own `softmax_scale`/`causal`. Attention is normally the
  heaviest op in the profile; refusing it left it unmeasured and therefore
  un-rankable. Two invariants: **every sequence gets its own blocks** (a shared
  block table turns the paged gather into a cache hit and understates the
  kernel by a large factor), and an operating point whose cache would not fit
  device memory is **refused with a reason** rather than attempted (the
  allocation would hang the worker instead of failing it). The block size is
  engine configuration, not an operand, so it is explicit —
  `DEFAULT_KV_BLOCK_SIZE` = 16, override with `BREAKDOWN_BENCH_KV_BLOCK_SIZE`.
  A wrapper with **no** context-free entry point (`vllm::moe_forward[_shared]`)
  is still reported `not_replayable` — the kernels it launches are separate ops
  in the graph and are benchmarked on their own.
- **Nothing about the sampler is context-bound.**
  `vllm::xpu_topk_topp_sampler` used to be skipped as needing "a generator and
  per-request metadata". It does not: its schema is all plain values, and the
  "generator state" is the philox `(seed, offset)` pair as a two-element **CPU**
  int64 tensor — the one thing the generic (device-allocating) builder cannot
  produce, hence the recipe in `recipes/xpu.py`. It runs over the full `[B, V]`
  logits every decode step, so it belongs in the ranking.
- **Timing subtracts a measured floor and repeats inside the window.** An empty
  device-event window costs ~60–90 µs on Level Zero, an order of magnitude more
  than a small elementwise kernel; a one-call-per-window loop measures the timer.
  `timing.measure` repeats the kernel `reps` times between one event pair,
  subtracts the calibrated empty-window cost and divides. Operand restoration
  and cache flushing happen *between* windows, never inside one.
- **Operands are allocated in their target dtype.** Building a several-hundred-MB
  `lm_head` weight in fp32 and casting doubles the allocation and dominates the
  case's wall time (it timed out the worker before this was fixed).
- **A case may stand for several sweep points.** An op whose operands do not
  depend on a swept dimension produces the *identical* case at several points,
  so `build_cases` de-duplicates it — but it records **every** point it covers
  (`BenchCase.points`). Ranking matches on membership: keying on the single
  stored point dropped the MoE grouped GEMM (the dominant kernel) from the
  targets whenever the operating point was not the first one swept.
- **A partial re-run must not delete the run.** `bench run --ops <one>` (and the
  UI's Ops box) re-measures a subset against an existing run id; the runner
  truncates `results.jsonl` only for a full run and otherwise drops just the
  selected ops' previous records.
- **The worker is launched from the repo root, with the repo *prepended* to any
  inherited `PYTHONPATH`.** A dev shell usually exports one already; using it as
  the working directory raised `FileNotFoundError` (multi-entry) or produced
  workers that could not import `breakdown` — failing every op, not just one.
- **One op per process, always.** A replayed kernel runs with synthesized
  operands, so a shape it cannot handle does not merely raise: it can abort the
  process or wedge the device so every *subsequent* op in it fails with a
  device-lost error. `runner` launches each op separately; results stream to
  `results.jsonl` case by case and `run_result.json` is rewritten after every op,
  so a run killed midway still says what completed.
- **Collectives run on the ranks they were profiled with.** `bench.collective`
  launches `world_size = TP` peers and records **rank 0** (ranks 1..N-1 absorb
  the wait to synchronize and report inflated times — the same reason the
  profile itself uses rank 0). Two hard-won rules: every rank must run an
  **identical, fixed** iteration schedule (a per-rank adaptive probe
  desynchronizes them and the transport runs out of resources), and the ranks
  must run with the **persistent SYCL cache disabled** — `SYCL_CACHE_PERSISTENT`
  makes oneCCL segfault with no Python traceback. Fewer devices than the
  profiled TP is reported as `needs_ranks`, never measured on fewer ranks.
- **Persistent kernel caches for everything else.** `worker.bench_env` pins
  `SYCL_CACHE_PERSISTENT`/`SYCL_CACHE_DIR`/`TRITON_CACHE_DIR` under
  `output/bench/.cache`; without them every worker re-pays AOT/JIT on its first
  case, which dominates a short sweep and poisons the first measurement.
- **Never source `setvars.sh` under `set -u`.** oneAPI's script reads unset
  variables, so a `set -u` shell dies silently (exit 127, no output) at that
  line. The runner sets only the cache variables and expects oneAPI to be
  sourced by the calling shell.
- **Timeouts are estimated, not guessed (`estimate.py`).**
  `startup + cases × (budget + per-case overhead) + operand bytes / alloc rate`,
  times a safety factor, with `startup` and the per-case overhead **calibrated
  from previous runs' wall time**.
- **Ranking = calls × latency × roofline headroom.** Calls come from the
  `Layers` count (how many modules dispatch the op at the chosen operating
  point), so a small op in 57 layers outranks a large one that runs once.
  Utilization is measured against the SKU peaks in `devices.py` (BMG: 456 GB/s
  DRAM, 98.3 TFLOPS bf16); at or above `target_util` (default 80 %) the op is
  `at_roofline` and is **not** a target. Ops with no editable kernel source
  (oneDNN/ATen/collectives) are `tune_config`. There is deliberately **no**
  `switch_provider` signal — replay measures the kernel that ran, so there is no
  second implementation to compare against.
- **The bound comes from arithmetic intensity, and a cache-resident op is
  charged to cache bandwidth (`estimate.py`).** Two rules, both deliberate:
  (1) an op is **compute-bound iff its AI (FLOP/byte) is at or above the machine
  balance** `peak FLOPS / peak bandwidth` (`estimate.ridge_ai`, ~215 flop/byte
  on BMG). The old rule compared the two *achieved* utilizations and took the
  larger, which labelled a GEMM running at 30 % of peak FLOPS "memory-bound"
  and a pure gather "compute-bound" — the bound is a property of the op and the
  machine, not of how well the kernel did. (2) The benchmark repeats a kernel
  on the **same operands** inside one timed window, so an op whose footprint
  fits the last-level cache is served by the cache and legitimately exceeds the
  DRAM peak; `estimate.effective_bw_gbs` measures it against
  `SKU_PEAKS[...]["cache_bw_gbs"]` instead (BMG: 18 MB at ~1.2 TB/s, *measured*
  on an Arc Pro B60 with a bf16 copy sweep). Before this, such ops were reported
  as "utilization 300 % of peak", a number that said nothing about the kernel.
  `roofline.memory_level` records which roof was used (the console table shows
  `mem/$`).
- **The roofline is only as honest as the FLOPs/bytes it is fed
  (`analyzer.py` / `shape_derive.py`).** Three cost-model rules earn their keep:
  attention has an explicit FLOPs model (`2·2·q·kv·heads·dim`, with
  `estimate_flops(..., n_seqs=batch)` dividing the batch's *total* KV rows back
  down so each query is charged only for its own context) — without it the
  heaviest op had zero analytic work, so its bound came out `unknown` and it
  ranked as if it had 100 % headroom; an **empty operand contributes 0 bytes
  rather than zeroing the estimate** (vLLM dispatches attention with an empty
  `kv_cache_dummy_dep` purely to order it against the KV write); and a
  **table-lookup op is charged for the rows it reads** (`_TABLE_LOOKUP_OPS`:
  `aten::embedding`, `_C::rotary_embedding`), not for the whole vocabulary
  matrix / cos-sin cache, which is what produced "37000 % of peak". Relatedly,
  `DTYPE_BYTES` knows the profiler's **C++ type names** (`long int` → 8), or
  every index/position operand is undercounted 4x.
- **Two honesty checks the ranking must keep.** (1) *Fidelity*: a case measured
  at the profiled shape carries the trace's own `device_time_us`; a replay far
  faster than it means the arguments do not reproduce the model's work, and the
  target is flagged instead of trusted. (2) *Cost model*: Memory/FLOPs are
  analytic estimates, so a utilization above `MAX_CREDIBLE_UTIL` is reported as
  `check_cost_model`, never silently retired as `at_roofline`. This is now the
  *last* resort: an op that merely ran out of a cache is explained by the cache
  roof, and the common table-lookup overcounts are fixed at the source, so a
  `check_cost_model` flag means a genuinely unmodelled op.
- **`targets.json` is a versioned contract** (`schema_version`, now **3**: v2
  was the replay model — provider fields removed, ops keyed by dispatch name,
  `traced_device_time_us` added — and v3 the roofline change: `roofline.bound`
  now comes from arithmetic intensity, with `memory_level` / `cache_bw_gbs` /
  `cache_bytes` / `ridge_ai` added) consumed by the `xpu-kernel-optimizer` skill:
  kernel dir/files, build/test commands, baseline latency, roofline bound and
  `bench_cmd`/`profile_cmd` per dominant shape. Changing a field's meaning
  requires bumping the version. `bench/kernel_sources.json` maps op/backend →
  repo, files, build and test command; **add an entry when a new op is
  benchmarked**, otherwise its target degrades to `tune_config` with no source.
- **Artifacts are owned.** Everything lands in `output/bench/<run_id>/`
  (`cases.json`, `plan.json`, `results.jsonl`, `run_result.json`, `logs/`,
  `targets.json`, `report.xlsx`) with a `run.json` recording the git commit of
  every component that can move a number, and cases are ingested into
  `output/bench/history.sqlite` so a regression can be attributed to a kernel
  bump (`/api/bench/history?base=&new=`).

### Op Classification (`classifier.py`)

Classifies ops by name prefix/pattern to backends. Priority order:
1. ccl (collective-comm: `c10d::`/`ccl::` namespaces or all_reduce/all_gather/reduce_scatter/all_to_all keywords)
2. vllm-xpu-kernels (exact match from registry)
3. flashinfer (op name contains `flashinfer` — FlashInfer RMSNorm/attention kernels launched directly from Python)
4. flash_xpu (op name contains `flash_xpu` — MiniMax-M3 MSA xattention SYCL kernels launched directly from Python)
5. triton (kernel name patterns)
6. torch-xpu-ops (aten:: ops that run on XPU)
7. cpu (fallback ops)
8. framework (reshaping, profiler markers)

CCL is checked **first** so tensor/pipeline-parallel comm calls (including
`vllm::all_reduce`-style ops) land in their own category rather than being
absorbed into vllm-xpu-kernels. Bare `gather`/`scatter` are intentionally
**not** CCL keywords so cache/MoE gather ops (`moe_gather`, `gather_cache`)
aren't misfiled.

**FlashInfer** is checked **before** Triton: FlashInfer RMSNorm /
fused-add-RMSNorm / attention kernels are launched straight from Python with no
`aten`/`_C` cpu_op, so they surface as synthetic kernel ops under the
`triton::` prefix (e.g.
`triton::kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_...`).
The embedded `flashinfer` symbol routes them to their own backend rather than
`triton`.

**flash_xpu** (MiniMax-M3 MSA / xattention) is likewise checked **before**
Triton: the lightning-indexer and block-sparse GQA attend SYCL kernels live in
the `xattention._C` (`flash_xpu`) extension and are launched straight from
the `xattention.py` wrappers with no `aten`/`_C` cpu_op, so they surface as
synthetic ops whose raw symbol embeds `flash_xpu` (e.g.
`flash_xpu::(anonymous namespace)::index_score_kernel_t`,
`flash_xpu::msa_index_topk_xpu(...)`). The embedded `flash_xpu` symbol routes
them to their own backend rather than `triton`.

## Conventions

- **License header** — All Python files start with `# SPDX-License-Identifier: Apache-2.0`
- **Type annotations** — Use `from __future__ import annotations` and modern syntax (`dict[str, Any]`, `list[int] | None`)
- **Environment** — Assume torch-xpu and vLLM are installed and an Intel XPU is available. `model_info.py` remains import-light (it doesn't pull in torch/vLLM at import time), which keeps config fetching fast, but supporting a torch-free CPU-only install is no longer a design requirement.
- **Symbolic shapes** — Op shapes use string symbols (`"H"`, `"S"`, `"n_h·d/TP"`) with `/TP` for TP-divided dims
- **TP-awareness** — All graph builders accept `tp_size`; shapes always show `/TP` for split dimensions; `cfg["_tp_*"]` keys hold divided values for numeric calculations

## Adding a New Model

The model graph is reconstructed from the trace, so a new architecture needs no
static builder. Ensure:
1. `breakdown/model_info.py` `summarize_config` extracts the model's key dims
   (hidden size, heads, intermediate, experts, MLA/VL nesting) so shapes
   symbolize (`S`/`B`/`C`/`H`/`I`/`n_h·d`/…) in the reconstruction.
2. Any novel ops are classified — see *Adding a New Op/Kernel*.
3. Profile it and confirm the reconstructed graph + Shape Matrix look right.

## Adding a New Op/Kernel

1. Add the op name to `breakdown/registry.py` `ALL_VLLM_XPU_OPS` set
2. If the op has a unique classification pattern, update `breakdown/classifier.py`
3. **Benchmarking needs no adapter** — the replay engine resolves the dispatch
   name and rebuilds the recorded operands. What it may need:
   - an **input synthesizer** if the op takes an integer/index tensor whose name
     isn't already covered (`breakdown/bench/inputs.py` or
     `breakdown/bench/recipes/`) — otherwise the case is reported as
     `needs_synthesizer`, never randomly filled;
   - a `recipes.outputs(...)` declaration if an argument is really an *output*
     the schema doesn't mark (especially one accumulated with atomics);
   - a `resolve.PYTHON_API` entry if it's a kernel launched straight from Python
     (Triton / FlashInfer / a SYCL extension) with no dispatcher op — or if it
     is a **context-bound wrapper with a context-free kernel entry point**, in
     which case add a `recipes.override` that rebuilds the context (see
     `recipes/attention.py`); `PYTHON_API` is consulted before `NOT_REPLAYABLE`;
   - an entry in `breakdown/bench/kernel_sources.json` if it has editable kernel
     source, so its target carries build/test commands

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/<hf_id>` | GET | Fetch/summarize HF model config (+ `min_profile_layers`) |
| `/api/cached-models` | GET | List previously loaded model IDs |
| `/api/profile` | POST | Start async profiling run |
| `/api/profile/upload` | POST | Reconstruct graph + op breakdown from uploaded trace file(s); a `_prefill`+`_decode` pair is merged into both phases (see *Trace upload round-trip*) |
| `/api/profile/status` | GET | Poll profiling status |
| `/api/profile/result` | GET | Fetch profiling result (ops + reconstructed graph) |
| `/api/profile/trace` | GET | Download raw trace file (two-pass runs: `?pass=prefill\|decode`; default = decode) |
| `/api/export/shape-matrix` | POST | Export profile-derived multi-config shape sweep to Excel |
| `/api/bench/plan` | POST | Sweep the profiled graph into replay cases (creates a bench run) |
| `/api/bench/run` | POST | Replay a run's cases (async — poll `/api/bench/status`) |
| `/api/bench/status` | GET | Poll the benchmark run (per-op progress) |
| `/api/bench/runs` | GET | List bench runs under `output/bench/` |
| `/api/bench/results` | GET | A run's measured cases + summary + coverage |
| `/api/bench/targets` | GET | Ranked optimization targets (`?run_id=`, `?refresh=1`, `?target_util=`) |
| `/api/bench/report` | GET | Download a run's report workbook |
| `/api/bench/history` | GET | Runs in the history db, or `?base=&new=` per-shape diff |

## Common Pitfalls

- **A config structural dim always beats the context length in the symbol
  table.** `build_graph_from_trace` registers `C` = `context_len` (and `S+C`)
  with **`setdefault`** on the value→symbol map, so a config dim wins a
  collision. It used to assign directly, on the theory that "the context dim
  wins over any coincidental config-value collision" — the opposite of what is
  right. Paged attention *never* records the context as a tensor dim (the
  cached KV lives in the block cache; `_annotate_attention_kv` writes the KV
  rows explicitly), so nothing in the trace legitimately *is* `C`, while a
  collision is destructive: Qwen3-30B-A3B has `hidden_size == 2048` and the
  default profiling context is also 2048, so every `H` dim symbolized to `C`
  and was then swept **with the context** by the Shape Matrix and the
  benchmark. At `ctx=0` hidden dims became **0** — `_C::rms_norm` divided by
  zero and killed its worker with SIGFPE (exit -8), and the MoE grouped GEMM
  rejected its own operands. See
  `TestGraphFromTrace.test_config_dim_wins_over_a_colliding_context`.
- **Dims derived from the token count must be symbolized as *expressions*, not
  frozen values (`_symbolize_moe_routed_rows`).** An MoE block expands every
  token into `num_experts_per_tok` routed rows, so the permuted hidden states,
  the grouped GEMM's `M` and the gather destination are all `tokens × topk`.
  As a plain observed-value symbol (`M_moe`) that dim froze at whatever it was
  while profiling, so as soon as the Shape Matrix swept `S` the token operand
  scaled and the routed operand did not — and the kernels rejected their own
  shapes (`remapped_hidden_states must be [num_rows * TopK, hidden_size]`,
  `ptr_A.size(1) must match ptr_B.size(1)`). They are now `topk·S` / `topk·B`,
  which `_resolve_dim` evaluates at each swept point; the per-token expert axis
  of the router outputs (`[tokens, topk]`) becomes `topk`, and the fused MoE
  gate_up width is registered as the config constant `2·I_moe`. Run this pass
  **before** `_symbolize_runtime_dims`, which would otherwise freeze the value
  first. See `TestGraphFromTrace.test_moe_routed_rows_scale_with_the_token_dim`.
- **Decode attention reads `B·C` KV rows, not `B`.** `_annotate_attention_kv`
  is applied to the decode tree too (with `token_sym="B"`, `kv_sym="B·C"`):
  each of the `B` sequences reads its own `C`-token context, so the *total* KV
  traffic is `B·C` — which is what the memory estimate needs, and what a
  replayed decode attention actually moves. `estimate_flops` divides it back by
  the sequence count so no query is charged for another sequence's context.
  Left at a bare `B`, the heaviest op in the model looked like it read a few
  kilobytes (util 0 %, bound `unknown`).
- **Query Len / Context Len drive a real prefix-cached prefill (Method 1 / APC).**
  Profiling no longer runs a fixed text prompt. `_run_profile` builds
  exact-length synthetic token prompts (`_make_token_ids`): `query_len` new
  tokens form the profiled prefill (`S`), and `context_len` (rounded **down** to
  a KV `block_size` boundary) is pre-computed in an **un-profiled warm pass** and
  served from the prefix cache (`enable_prefix_caching=True`) during the profiled
  run. So the profiled prefill computes only `query_len` tokens while attention
  still reads the full `context_len+query_len` KV. Key invariants: (1) warm the
  **context prefix alone**, then warm kernels with **distinct** query seeds per
  pass, and give the **profiled** run yet another seed set (`900000+b`) so its
  queries are never cache-hit; (2) the context prefix is shared across the batch
  (one warm pass) but each batch item gets a distinct query; (3) verify the hit
  via `outputs[0].num_cached_tokens >= ctx_aligned` — a miss recomputes the whole
  context (`S = context+query`) and is surfaced as `cache_hit_note`. `start_profile`
  bumps `max_model_len` to `context+query+max_tokens`. When `query_len` is absent
  (uploads/legacy clients) the old chat/text-prompt path still runs. The
  **block-aligned** context length (`ctx_aligned`) is threaded into
  `build_graph_from_trace(..., context_len=...)`, which registers `C =
  context_len` and `S+C = context+query` in the symbol legend. **Paged attention
  never records the context as a tensor dim** — the op's key/value inputs only
  carry the *new* `[S, n_kv, d]` tokens (verified: `unified_attention_with_output`
  =`[[S,n_h,d],[S,n_kv,d],[S,n_kv,d],[S,n_h,d],…]`), the cached context lives in
  the block cache / seqlen metadata. So `C` can't be *symbolized* from the trace;
  instead `_annotate_attention_kv` **rewrites the attention key/value rows from
  `S` to `S+C`** (query/output rows stay `S`) so the prefill graph shows the
  query attending `context+query` keys. Without threading `context_len`, KV rows
  stay `S` and the context is invisible.
- **Separate prefill / decode batch sizes are two profiled passes merged.** Real
  serving prefills ~1 sequence while decode batches 32/64/128, but a single
  `llm.generate` prefills and decodes the *same* batch (so `S` and `B` couple to
  one `batch_size`). When `prefill_batch_size != decode_batch_size`, `_run_profile`
  runs **two** profiled passes reusing the one loaded `LLM`: a **prefill pass** at
  `prefill_batch` with the real `query_len`, generating only **1** token so the
  trace holds exactly the prefill step (keep its `prefill` tree, `S`=query_len),
  and a **decode pass** at `decode_batch` with `query_len` forced to **1** (decode
  = 1 new token/seq; also avoids OOM from prefilling `decode_batch × query_len`),
  generating the full decode budget (keep its `decode` tree, `B`=decode_batch).
  `_profiled_pass` takes a `pass_max_tokens` so each pass generates only what its
  kept phase needs — the prefill pass no longer wastes time/trace on decode steps. `_merge_two_pass_result` splices them:
  decode pass is the base (its op/backend breakdown is the steady-state one), the
  prefill pass's `graph.prefill` is overlaid, and symbols are combined (`S`/`S+C`/`C`
  from prefill, `B` from decode). Each `_profiled_pass` snapshots `trace_dir` before
  its `start_profile` and returns only the file(s) it wrote, so the two passes'
  traces don't cross-contaminate. Because the merged result's base is the decode
  pass (`result = dict(dec)`), its `trace_file` is the *decode* trace — so a plain
  `/api/profile/trace` download is decode-only. `_merge_two_pass_result` therefore
  also stashes `prefill_trace_file` / `decode_trace_file`, and the download endpoint
  accepts `?pass=prefill|decode` (default = decode) to serve either; the frontend
  shows two buttons (Prefill/Decode Trace) for two-pass runs and the single
  "Download Trace" button otherwise (`has_prefill_trace`/`has_decode_trace` flags on
  `/api/profile/result`). Equal batches (or only legacy `batch_size`) → a
  single pass, identical to before. Frontend sends `prefill_batch_size` /
  `decode_batch_size`; `setPhase` no longer rewrites the inputs since one run now
  yields both phases. See `tests/test_two_pass_merge.py` and
  `tests/test_trace_download.py`.
- **Trace upload is a lossless round-trip — a `_prefill`+`_decode` pair rebuilds
  both phases (not decode only).** `/api/profile/upload` must mirror the live
  two-pass build so you can profile once on the XPU box, **download the separate
  prefill/decode traces**, and reconstruct the graph / drive the Shape Matrix
  export on a **GPU-less** machine. The download filenames are self-describing
  (`vllm_trace_<model>_<device>_<mode>[_prefill|_decode]_ctx<C>_in<S>_out<gen>_bs<B>_tp<TP>[_<quant>]_<N>layers`),
  so `upload_profile` parses each with `_parse_trace_filename` (`_TRACE_NAME_RE`)
  to recover mode/TP/quant/`query_len`/`context_len`/per-pass batch **and the
  prefill|decode split** — explicit form fields still override; legacy names
  without the encoded tail fall back to form fields. It groups the uploads by
  pass tag: a prefill file **and** a decode file → build each phase with its own
  batch/query (`res_pre` at `pf_batch`/`query_len`, `res_dec` at
  `dc_batch`/query=1) and splice via the **same** `_merge_two_pass_result`; the
  merged result stashes `prefill_trace_file`/`decode_trace_file` so re-download
  (`?pass=`) round-trips again. **Do NOT revert to treating multiple uploads as
  TP ranks** — the pre-fix bug built the graph only from `rank_files[0]` (the
  decode pass, which by design has only decode steps → `prefill: None`) and
  silently averaged the prefill trace in as a bogus rank, so uploading the pair
  "worked for Decode only". Untagged single files (optionally several rank files)
  still reconstruct one run — and, like the live path, the **rank-0 file is used
  and the other ranks are ignored** (see the rank-0 pitfall below), not averaged.
  The upload path now also
  threads recovered `query_len`/`context_len` into `_build_result_from_traces`
  so `C`/`S+C` symbolize (previously ignored → the Shape Matrix lost context).
  See `tests/test_upload_two_pass.py`.
- **Rank 0 is always the representative worker for TP>1 traces (not an average).**
  With tensor parallelism vLLM writes one trace per rank. Ranks 1..N idle much
  longer than rank 0 on collectives — their `c10d::allreduce_` device time is
  inflated by the wait to synchronize with rank 0 — so averaging across ranks (or
  picking whichever rank flushed last) skews the op breakdown. `_rank0_first`
  lifts the `rank0`/`tp0` file (parsed from the raw
  `dp0_pp0_tp<N>_…_rank<N>.….pt.trace.json.gz` name via `_trace_rank`) to the
  front of `rank_files`, and `_build_result_from_traces` builds the op breakdown,
  the reconstructed graph **and** the downloadable `trace_file` from that rank-0
  file only; the other ranks are ignored. This applies to both the live profiler
  and the upload path, and to each pass of a two-pass run. Do NOT reintroduce
  cross-rank device-time averaging or a mtime-based `rank_files[0]` primary.
  See `tests/test_trace_download.py::TestRank0Selection`.
- **The scheduler is pinned so decode runs the full batch every step.**
  `_run_profile` sets `max_num_seqs = max(prefill_batch, decode_batch)` (and
  `max_num_batched_tokens` large enough for a whole-batch prefill step) *before*
  constructing the `LLM`. Without this, vLLM's continuous-batching scheduler
  caps per-iteration concurrency (by its default `max_num_seqs` and by how many
  sequences' KV fits in cache) and dispatches an oversized batch in
  **partial-batch waves** — a batch of 32 as e.g. `29 + 3`. Each wave has a
  different row count (`num_running_seqs`), so `_symbolize` (which only maps the
  max decode row count to `B`) leaves the partial waves as literal ints, and
  `_merge_modules` (keys ops by `(name, shapes, occurrence)`) can't merge them —
  they surface as **duplicated `29`/`3` op nodes** in the decode graph instead of
  one `B` node. Pinning `max_num_seqs` makes every decode forward run all
  `decode_batch` sequences → a clean single `B`. Do NOT remove the pin; if a
  batch's KV won't fit device memory, raise `gpu_memory_utilization` or lower
  Context/Batch rather than reverting to the splitting default.
- **Phase classification is `token_dim > batch_size`, not "max-token step = prefill".**
  `graph_from_trace._classify_steps` labels a step **prefill** iff its forward
  processes *more than one token per running sequence* (`token > batch_size`); a
  **decode** step advances each running sequence by one token, so its token dim is
  `num_running_seqs` (≤ `batch_size`). The old rule (largest-token step = prefill)
  breaks the two-pass **decode pass**: with `query_len=1` and prefix caching, vLLM
  prefills each sequence's single new token *individually* (a **1-row** op) while
  decode runs the whole batch (**`batch_size`-row** ops) — so the decode steps have
  *more* rows than the prefill microsteps, and "max = prefill" would invert the
  phases and report `B = batch_size − 1` (a ramp step). Comparing to `batch_size`
  also classifies each chunk of a chunked prefill correctly. Verified end-to-end on
  XPU (Qwen3-4B): prefill@1 → `S=query_len`, decode@8 → `B=8`.
- **Only steady-state, full-batch decode steps are kept.**
  `graph_from_trace._classify_steps` restricts the decode phase to the steps
  whose token dim equals the **maximum observed decode batch** before building
  the phase tree. vLLM admits the `batch_size` sequences into the running batch
  in **ramp-up waves**, so the early decode steps process fewer sequences and
  their per-token ops (embedding, `rms_norm`/`fused_add_rms_norm`,
  `rotary_embedding`, dense matmuls) carry **partial row counts**
  (`2`/`4`/`28`/`30` instead of `32`). Those partial-batch steps are transient
  and would otherwise surface as spurious literal-int (non-`B`) nodes in the
  reconstructed decode graph — the symptom is duplicated near-`B` nodes like
  `28`/`30`. Keeping only the max-batch steps makes the batch dim symbolize
  cleanly to `B`. If the configured batch is never fully reached (e.g. KV-cache
  pressure caps it below `batch_size`), the largest batch actually run is used
  (the honest steady state). Among the surviving steady-state steps the **first
  is additionally dropped as warmup**: the initial full-batch decode forward pays
  one-time costs (KV/allocator warmup, oneDNN/Triton plan + autotune caching
  under `torch.compile`) that would skew the per-op latency average. Both filters
  are **guarded** so the decode phase is never emptied — at least one steady-state
  step always remains. The profiling UI exposes a **Decode Steps** control
  (`max_tokens`, default **8**) with a `>= 2` minimum. See
  `TestGraphFromTrace.test_first_decode_step_dropped_from_average` /
  `test_single_decode_step_not_dropped` / `test_partial_batch_decode_steps_dropped`.
  Do NOT remove the guards or the UI minimum.
- **Main model class is picked by subtree size, not device time.** The
  `LogitsProcessor`'s `lm_head` matmul (`V`-wide, once per decode step) can
  out-weigh the whole model forward, so selecting the main class by `sub_dev`
  wrongly picked `LogitsProcessor` — making every step a 1-token pass, so the
  prefill phase disappeared (`prefill: None`, blank graph until you click Decode)
  and prefill/decode looked identical. `_partition_steps` now uses
  `_subtree_module_count`. Do NOT revert to a device-time heuristic.
- **Module names come from capture-time spans; the overlay is a fallback.**
  `module_hooks` installs forward hooks that emit
  `record_function("module::<qname>::<Cls>")` `user_annotation` spans during the
  profiled generate, so `graph_from_trace._build_raw_forest` reads the real
  attribute path straight from the trace (no alignment, no ordering assumption).
  `build_graph_from_trace` only applies the `module_naming` overlay when the
  forest has **no** captured names (`_forest_has_named_modules` → false), i.e.
  for legacy/upload traces. Do NOT make the overlay unconditional — it would
  redundantly (and possibly wrongly) relabel a correctly-named span tree. When
  editing the span label format, update **both** `trace_common.module_span_label`
  and `parse_module_span` (the emitter in `module_hooks` and the parser in
  `graph_from_trace` share them).
- **Module-name alignment (fallback path) must unwrap `*Model` levels absent from the trace.**
  vLLM nests the decoder stack under an inner `*Model` module whose `forward`
  usually emits no trace module event, so the trace nests it directly under
  `*ForCausalLM`. `module_naming._effective_ref_children` flattens reference
  levels whose class isn't among a node's actual trace-child classes; without it,
  child matching stalls at the missing level and `q_norm`/`k_norm` stay `norm`.
  (Capture-time spans don't hit this — they carry the full path already.)
- **Device time is attributed by launch-site containment, not `External id`.**
  `graph_from_trace.py` links each device `kernel` to its host launch call
  (`kernel.correlation → xpu_runtime`, the "flow arrow") and attributes it to the
  deepest module/op interval containing the launch timestamp on the worker
  thread. This is deliberate: under `torch.compile` the `External id` bookkeeping
  points norm/indexer/sparse-attention kernels at compiled-region/plumbing
  cpu_ops (leaking their time), whereas launch-site containment is stable across
  eager and compiled passes. Do NOT revert to `External id` mapping
  (`_build_device_time_map` was replaced by `_collect_kernel_launches` +
  `_attribute_kernels`). **Research R2** (pure flow-based attribution + deleting
  the `is_cuda` async workaround) is a *deliberate deferral*: CUDA-graph launch
  drift is not fixed by R1's spans alone and cannot be validated on this XPU
  host, so the hardware-validated launch-site path and the `is_cuda`-gated
  corrections are kept as-is. **The launch-site categories include the CUDA
  *driver* API (`cuda_driver`: `cuLaunchKernel`/`cuLaunchKernelEx`), not just the
  runtime API (`cuda_runtime`) and XPU (`xpu_runtime`)** — Triton launches its
  kernels straight through the driver API, so the routed-MoE `fused_moe_kernel`
  grouped GEMM (and other Triton kernels) have **no** `cuda_runtime` launch
  event. Without `cuda_driver` in `_RUNTIME_CATEGORIES`, their launch-site lookup
  misses and falls back to `External id`, which points at the enclosing
  `vllm::moe_forward_shared` custom op's start — collapsing all the expert GEMM
  time into `moe_forward_shared` instead of the `moe` node (symptom: "the MoE
  after shared experts dispatched to triton_moe is missing from the graph").
  Driver and runtime launches never share a correlation id (verified), so adding
  `cuda_driver` doesn't double-count. See
  `TestGraphFromTrace.test_cuda_triton_moe_experts_surfaced`.
- **Worker thread is chosen by the module-span anchor (research R6), not raw
  cpu_op count.** `_build_raw_forest` picks the tid carrying the capture-time
  `module::` spans when present; only legacy (span-less) traces fall back to the
  busiest-cpu_op tid. This avoids mis-selecting a wrong thread under tensor
  parallelism (several threads dispatch ops). Do NOT revert to an unconditional
  busiest-cpu_op guess.
- **Triton kernels with no `cpu_op` surface as synthetic `triton::<kernel>` ops.**
  Kernels launched straight from Python via `triton.jit` (Gemma RMSNorm,
  MiniMax-M3 lightning indexer, block-sparse attention) never emit an
  `aten`/`_C` cpu_op. `_attribute_kernels` adds them as `triton::`-prefixed ops
  on their enclosing module (classified as `triton`, unless the kernel symbol
  embeds `flashinfer` → the FlashInfer backend). Real ops (`aten::mm`,
  `c10d::allreduce_`, `vllm::unified_attention_with_output`) get the kernel time
  added to themselves instead.
- **Residual-stream ops the trace leaves shape-less are inferred from a
  neighbour (`_infer_hidden_activation_ops`).** Two op classes carry no usable
  shape/dtype on the op event itself: (1) TP collectives — `c10d::allreduce_`
  records its tensor as a **`TensorList`** (`Input Dims` = `[[[2, H]], [], …]`,
  an *extra nesting level*, with `Input type` `'TensorList'` and **no element
  dtype**), which `_parse_input_dims_types` used to drop entirely (symptom:
  "allreduce shape+dtype missing"); (2) the Gemma/LayerNorm RMSNorm kernels
  vLLM launches straight from Python via Triton/FlashInfer
  (`triton::_gemma_rmsnorm_kernel`, `triton::_gemma_fused_add_rmsnorm_kernel`)
  have **no `cpu_op` and so no shape at all** (symptom: "MiniMAXGemmaRMSNorm
  shape+dtype missing"). `_parse_input_dims_types` now **unwraps the TensorList
  container** (surfacing each contained tensor's shape, dtype left empty), and
  `_infer_hidden_activation_ops` fills any remaining shape-less norm/collective
  op with the residual hidden state `[tokens, H]` + activation dtype **borrowed
  from the op nearest in execution order that carries a genuine `[tokens, H]`
  tensor** (2-D, trailing dim == `hidden_size`, real dtype). Nearest-by-`ts`
  keeps the correct per-step token dim (prefill `S` vs decode `B`). It is
  **restricted to norm-family + collective ops** (`_is_hidden_state_op`) so
  attention kernels (`flash_xpu::minimax_m3_sparse_attn_decode`), `aten::zeros`,
  `aten::item` and other differently-shaped / genuinely-input-less ops are left
  untouched — they must NOT inherit `[tokens, H]`. Runs on the raw forest after
  the reference-name overlay, before phase partition. See
  `TestGraphFromTrace.test_tensorlist_collective_and_norm_kernel_shapes_recovered`.
- **Shape-less MiniMax-M3 MSA / indexer kernels are reconstructed from their
  wrapper layout — on CUDA as well as XPU (`_infer_attention_kernel_shapes`).**
  The sparse-attention and lightning-indexer kernels carry no `cpu_op` on
  **either** backend — XPU runs `xattention._C` SYCL kernels launched from
  `xattention.py` (`flash_xpu::minimax_m3_sparse_attn[_decode]`,
  `minimax_m3_index_score`/`_decode`/`_topk`), CUDA runs `triton.jit` kernels
  launched from `models/minimax_m3/common/ops/{sparse_attn,index_topk}.py`
  (`triton::_gqa_sparse_{fwd,decode}_kernel`, `_merge_topk_attn_out_kernel`,
  `_index_block_score_kernel`, `_decode_index_score_kernel`,
  `_topk_index[_partial|_merge]_kernel`) — so they surface shape-less. Unlike
  the residual-stream ops their tensors are **not** `[tokens, H]`, so
  `_infer_hidden_activation_ops` deliberately skips them. Their primary-tensor
  layout is instead fixed by the wrapper signatures: the block-sparse **attend**
  kernels take a query `[total_q, num_heads, head_dim]` → `[S, n_h/TP, d]`, the
  lightning-**index** kernels an index query
  `[total_q, num_index_heads, index_head_dim]` → `[S, n_idx/TP, idx_d]`, and the
  indexer **top-k** kernels the block-id tensor `[n_idx/TP, total_q, topk]` →
  `[n_idx/TP, S, K_topk]` (int32; only the top-k width is config-derivable —
  the score width depends on the runtime max seq len).
  `_infer_attention_kernel_shapes` rebuilds the numeric shape from the config
  (per-rank `num_heads`/`head_dim`; `sparse_num_index_heads`/`sparse_index_dim`;
  `sparse_topk_blocks`) with `total_q` taken from the **nearest neighbouring
  residual hidden-state op** (2-D, trailing dim == `hidden_size`), so prefill
  gets `S` and decode `B`. That token reference **excludes weight-plumbing ops**
  (`_WEIGHT_PLUMBING_OPS` = `t`/`transpose`/`permute`/`detach`): a weight is also
  `[out_features, H]`, so an `aten::t` on one otherwise handed the kernels the
  weight's out-features as their row count (`[n_h·d/TP, n_idx/TP, d]` instead of
  `[B, …]`). `_MSA_KERNEL_LAYOUTS` maps kernel-name **substrings** to
  `topk`/`attn`/`index` so it is device-agnostic (`topk` is probed first because
  the XPU API name `minimax_m3_index_topk` also contains the `index` prefix);
  `flash_xpu` ops with no recognised layout still fall back to borrowing the
  neighbour's activation shape. Runs right after `_infer_hidden_activation_ops`.
  See `TestGraphFromTrace.test_flash_xpu_attention_kernel_shapes_reconstructed`
  and `test_cuda_msa_indexer_kernel_shapes_reconstructed`.
- **The MSA indexer dims get dedicated symbols (`_symbolize_msa_dims`).** Plain
  value→symbol mapping gets three of the reconstructed indexer dims wrong on M3:
  the index-head count equals `num_kv_heads` (renders `n_kv/TP`), the top-k block
  count (16) equals `n_h/TP` at TP=4 (renders `n_h/TP` — and would then wrongly
  **scale with TP** in the Shape Matrix sweep), and the top-k tensors are
  token-major on their **second** axis where the token count can collide with a
  config dim (`S`=2048=`n_h·d/TP`) and lose its `S`/`B` symbol. `_symbolize_msa_-
  dims` rewrites all three to `n_idx`/`n_idx/TP`, `K_topk` and the phase token
  symbol, guarded by each op's `recorded_shapes` so only the reconstructed
  tensors are touched. Runs after the phase trees are built and **before**
  `_symbolize_runtime_dims` (so those dims don't get a meaningless observed-value
  symbol). Relatedly, `_symbolize` now also recognises the token dim at a
  **non-leading** position when the value isn't a known config dim.
- **Only genuine device kernels are collected; host-side launch-API events are
  NOT surfaced as ops.** `_collect_kernel_launches` emits a launch **only** for a
  real *device* event (cat in `_DEVICE_KERNEL_CATEGORIES` = `kernel` /
  `gpu_memcpy` / `gpu_memset` / `xpu_op` / ...), located at its host launch site
  via `kernel.correlation → cuda_runtime`/`cuda_driver`/`xpu_runtime` (External-id
  fallback). Host launch-API events (`_RUNTIME_CATEGORIES`) are never surfaced on
  a trace that has device-kernel events — they carry no device time and are only
  launch-site locators. **Do NOT put `cuda_runtime` back into the device-kernel
  set:** it used to be in the kernel-category set, which meant pure host bookkeeping
  calls that launch nothing (`cudaEventQuery`, `cudaStreamWaitEvent`,
  `cudaDeviceGetAttribute`, `cudaStreamIsCapturing`, `cudaEventRecord`) were
  collected as fake kernel launches (on the MiniMax-M3 CUDA decode trace ~2500
  spurious "launches" of 4267 vs 1757 real device kernels) and could surface as
  bogus `triton::cudaEventQuery` leaf ops. The only exception is a *runtime-only*
  trace (no device-kernel events at all): there an actual `*Launch*`/`*Enqueue*`
  call on the worker thread is emitted as a fallback (bookkeeping still excluded).
  A launch that backs a real kernel is likewise not duplicated (else a
  Python-direct FlashInfer RMSNorm would show once as the kernel op and once as a
  `triton::cudaLaunchKernelExC` op). See
  `TestGraphFromTrace.test_flashinfer_kernel_surfaced_launcher_suppressed` and
  `test_runtime_bookkeeping_not_surfaced_as_kernel`.
- **Every device kernel launched inside a module subtree lands on a leaf op — no
  silent drops.** `_attribute_kernels` routes each launch to a leaf: inside a
  real op → that op's `self_dev`; inside a module (or plumbing op) → a synthetic
  `triton::`/`flashinfer::`/`flash_xpu::` op on the enclosing module. A launch
  whose site has **no enclosing module** (a module-less top-level op subtree,
  e.g. a bare sampler/logits op outside every decoder module) is **folded into
  the deepest op's `self_dev` rather than dropped**, so device time is conserved
  (it just isn't shown as a phase leaf, being outside the reconstructed phase
  trees). Verified on the four MiniMax-M3 traces (XPU/CUDA × prefill/decode):
  every kernel launched inside a kept prefill/decode step lands on a leaf, and
  the collected launch count equals the real device-kernel count. Only kernels
  launched **between** forward passes (e.g. `_compute_slot_mapping_kernel` KV
  metadata prep) fall in gaps and are out of step scope by design. The read-only
  `_kernel_leaf_coverage(roots, launches)` classifier reports
  `total`/`on_leaf`/`dropped_gap`. See
  `TestGraphFromTrace.test_module_less_kernel_time_conserved_not_dropped` and
  `test_minimax_m3_traces_every_in_step_kernel_on_leaf`.
- **FlashInfer kernels classify as the `flashinfer` backend and are named after
  their public API frame.** FlashInfer RMSNorm/fused-add-RMSNorm/attention
  kernels launch directly from Python (no `aten`/`_C` cpu_op), so
  `_attribute_kernels` surfaces them as synthetic ops on the enclosing module.
  Rather than the unreadable raw cutlass functor symbol
  (`flashinfer::kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel`),
  the op is named after the **public FlashInfer API python frame** that launched
  it — the outermost `flashinfer/**/__init__.py(...): <func>` frame whose
  function name is public (no leading `_`) — so the input_layernorm norm reads
  `flashinfer::gemma_fused_add_rmsnorm` (and `flashinfer::gemma_rmsnorm`),
  matching the readable XPU triton names (`_gemma_fused_add_rmsnorm_kernel`).
  `_collect_flashinfer_api_frames` gathers those frames on the worker thread and
  `_flashinfer_api_name` picks the one enclosing each kernel launch; the
  `flashinfer::` namespace prefix is kept so `classify_op` still routes any op
  whose name contains `flashinfer` to `Backend.FLASHINFER` (checked before the
  Triton pattern match). When no such API frame is found (legacy/synthetic
  traces) it falls back to the cleaned raw kernel symbol. See
  `TestGraphFromTrace.test_flashinfer_kernel_named_after_public_api_frame`.
- **xattention (MiniMax-M3 MSA) kernels classify as the `flash_xpu` backend and
  are named after their xattention API frame.** MiniMax-M3 sparse attention on
  XPU (the lightning indexer's block score + top-k, and the block-sparse GQA
  attend) runs as hand-tuned SYCL kernels in the `xattention._C`
  (`flash_xpu`) extension, launched directly from the `xattention.py` wrappers
  with no `aten`/`_C` cpu_op, so `_attribute_kernels` surfaces them as synthetic
  ops on the enclosing module (the `Indexer` / sparse-attention module). Rather
  than the long raw SYCL symbol (`flash_xpu::(anonymous namespace)::
  index_score_kernel_t`, `flash_xpu::msa_index_topk_xpu(at::Tensor const&, ...)::
  {lambda...`), the op is named after the **public xattention API python frame**
  that launched it — the outermost `xattention.py(...): <func>` frame whose
  function name is public (no leading `_`) — so they read
  `flash_xpu::minimax_m3_index_score`, `flash_xpu::minimax_m3_index_topk`,
  `flash_xpu::minimax_m3_sparse_attn` (prefill) and
  `flash_xpu::minimax_m3_index_decode`, `flash_xpu::minimax_m3_sparse_attn_decode`
  (decode). `_collect_flash_xpu_api_frames` gathers those frames on the worker
  thread and `_flash_xpu_api_name` picks the one enclosing each kernel launch;
  the `flash_xpu::` namespace prefix is kept so `classify_op` routes any op whose
  name contains `flash_xpu` to `Backend.FLASH_XPU` (checked **before** the Triton
  pattern match, since the fallback synthetic name would otherwise misclassify
  these SYCL kernels as `triton`). When no such API frame is found
  (legacy/synthetic traces) it falls back to the cleaned raw kernel symbol under
  `flash_xpu::`. See
  `TestGraphFromTrace.test_flash_xpu_kernel_named_after_xattention_api_frame`.
- **`vllm::` namespace ops classify as vllm-xpu-kernels.** `classify_op` maps the
  `vllm::` prefix (dispatch ops: `unified_attention_with_output`,
  `unified_kv_cache_update`, `moe_forward_shared`, `xpu_topk_topp_sampler`)
  alongside `_C::`/`_C_cache_ops::`/`_moe_C::`/`_xpu_C::`. Without this, dense
  attention showed as `framework`.
- **Reduced-layer profiling is extrapolated to the true layer count.**
  `_extrapolate_decoder_layers` folds unprofiled layers into the *last*
  `*DecoderLayer` sibling group (the MoE body for dense-prefix MoE models), so a
  1-MoE-layer reduced trace of MiniMax-M3 reads `x57`. Totals are recomputed via
  `_recompute_totals`. Triggers only when `summary["num_layers"]` exceeds the
  profiled decoder-layer count.
- `app.py` is ~1700 lines — use `view_range` to read targeted sections
- **Profiling reconstructs the graph from the trace** (`graph_from_trace.py`) —
  it does NOT overlay timing onto a static graph. The `annotate_graph_*` and
  `parse_trace_with_modules` helpers were removed; don't reintroduce them. There
  is **no interactive static-graph endpoint** (`/api/model/<id>/graph` and
  `/api/export/static-graph` were removed) and **no config-driven graph builder**
  (`model_graph.py` / `build_model_graph` were deleted). The Shape Matrix export
  is profile-derived (`graph_from_trace`); don't reintroduce a static builder.
- Encoder models have no autoregressive decode phase
- The web UI is a single HTML file with inline CSS/JS — no build step needed
- `model_info.py` fetches from HuggingFace API — tests should mock HTTP calls
- Shape strings contain `/TP` always (even when TP=1) — resolve via `symbols["TP"]`
- MLA models (DeepSeek-V2/V3/V4, GLM-MoE-DSA) are supported on XPU — do not
  re-add the removed profiling guard that rejected them
- Some VL models (e.g. MiniMax-M3, `MiniMaxM3SparseForConditionalGeneration`)
  nest language-model params under `text_config` and vision params under
  `vision_config`. `summarize_config` reads text params via a `text_config`
  fallback, derives `first_k_dense_replace` from the leading zeros of a
  per-layer `moe_layer_freq` list, and maps `dense_intermediate_size` → dense
  MLP / `intermediate_size` → MoE experts. M3 uses standard GQA (not MLA).

## Updating Documentation

When making significant changes to this repository, update documentation accordingly:

1. **AGENTS.md** — Update when:
   - New files/directories are added or removed (update Project Structure)
   - New API endpoints are added (update API Endpoints table)
   - Architecture decisions change (update Key Architecture Decisions)
   - Conventions change (update Conventions section)
   - New model types or builders are added (update builder table)
   - Common pitfalls are discovered (add to Common Pitfalls)

2. **README.md** — Update when:
   - User-facing features are added or changed (update Features list)
   - CLI interface changes (update CLI section)
   - New output formats are added (update Output table)
   - Architecture overview changes (update Architecture section)
   - New export/analysis capabilities are added

3. **When to update** — After any PR that adds features, changes APIs, modifies the project structure, or alters how the tool is used. Run `git log --oneline <last_doc_commit>..HEAD` to see what changed since docs were last updated.

4. **How to verify** — After updating, check that:
   - Project Structure listing matches actual files (`ls breakdown/ tests/ scripts/`)
   - API Endpoints table matches routes in `app.py` (`grep "@app.route" app.py`)
