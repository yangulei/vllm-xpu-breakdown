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
there is no interactive static-graph view.
1. **Dynamic profiling (profile-first)** — Runs actual inference via vLLM on
   Intel XPU, then **reconstructs the model graph directly from the torch
   profiler trace** (nn.Module call stack + `Input Dims` shapes + kernel device
   time via the correlation→runtime→External-id chain). The reconstructed tree
   reflects what actually executed, so it tracks whatever vLLM/the backends
   dispatched instead of relying on a hand-maintained static graph.
2. **Config-driven shape sweep** — `build_model_graph` (from the HuggingFace
   config) still powers the **Shape Matrix Excel export**
   (`/api/export/shape-matrix`), a sweep across seq/context/batch/TP. It is no
   longer exposed as an interactive graph endpoint.

> **Environment assumption:** the profiling path assumes torch-xpu and vLLM are
> installed and an Intel XPU is available. (`model_graph.py`/`model_info.py`
> happen to be import-light, but supporting a torch-free CPU-only install is no
> longer a design goal.)

## Project Structure

```
app.py                    — Flask web server (API + static serving + exports)
run_profile.py            — CLI profiling entry point
chat.py                   — Interactive chat with profiling
breakdown/
  model_graph.py          — Static (config-driven) model graph builder
  graph_from_trace.py     — Profile-first graph reconstruction from a trace
  module_naming.py        — Recover real module attribute names (q_norm/k_norm)
  model_info.py           — HuggingFace config fetcher and summarizer
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

# Full integration (requires Intel XPU hardware)
pytest tests/ -v
```

## Key Architecture Decisions

### Model Graph Builder (`model_graph.py`)

The config-driven shape builder. It powers the **Shape Matrix Excel export**
(no longer an interactive graph view). Key concepts:

- **`_ARCH_FAMILY_MAP`** — Maps HuggingFace `architectures` field to family names (35+ entries). This determines which graph builder is used.
- **Architecture category sets** — `_MLA_ARCHS`, `_VL_ARCHS`, `_ENCODER_ARCHS`, `_DIFFUSION_ARCHS` control routing in `build_model_graph()`.
- **`ModuleNode` / `OpNode`** — Tree structure representing model modules and their ops with shapes, backends, memory, and FLOPs.
- **Phase-aware** — Each model gets separate `prefill` and `decode` graphs (different token dimensions).

Architecture-specific builders:
| Builder | Used for |
|---------|----------|
| `_build_decoder_layer` | Standard GQA decoder (Llama, Qwen, Mistral) |
| `_build_moe_layer` | MoE decoder (Mixtral, Qwen-MoE) |
| `_build_mla_decoder_layer` | MLA attention (DeepSeek-V2/V3) |
| `_build_mla_moe_layer` | MLA + MoE (DeepSeek-V3) |
| `_build_vision_encoder` | ViT for VL models |
| `_build_encoder_model` | BERT/RoBERTa for embedding/reranker |
| `_build_diffusion_placeholder` | Diffusion models (not vLLM-served) |

> **MLA profiling is supported on XPU.** Dynamic profiling of MLA models
> (DeepSeek-V2/V3/V4, GLM-MoE-DSA) used to be hard-rejected in `app.py`
> because MLA required FlashAttention. vLLM-XPU now provides MLA backends
> (`TRITON_MLA` for dense MLA, `XPU_MLA_SPARSE` for DeepSeek sparse attention),
> so that block was removed. Do not re-add an MLA architecture guard in the
> profiling path. See `TestMLAModelGraph` in `tests/test_pipeline.py`.

### Profile-First Graph Reconstruction (`graph_from_trace.py`)

The profiling flow no longer overlays timing onto the static graph. Instead
`build_graph_from_trace(trace_path, summary, tp_size, batch_size, quantization)`
reconstructs the module/op tree straight from the torch profiler trace and
returns the same serialized shape (`{prefill, decode, symbols, config,
has_timing, timing_*, source}`) as `build_model_graph`, so the frontend renders
it unchanged. How it works:

- **Module tree** — `nn.Module: <Cls>_<idx>` python_function events nest by
  time-containment (sort by `(ts asc, end desc)`, pop a stack while the top ends
  before the current node).
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

The old `annotate_graph_timing` / `annotate_graph_from_modules` /
`parse_trace_with_modules` paths were **removed**. Do not reintroduce a
static-overlay step in the profiling worker.

### Module Attribute Naming (`module_naming.py`)

The profiler only labels module events with their **class** (`nn.Module:
<Cls>_<idx>`), so sibling modules of the same class (Qwen3 `q_norm`/`k_norm`,
both `RMSNorm`; `input_layernorm`/`post_attention_layernorm`; ...) are
indistinguishable and fall back to a class-heuristic name. `module_naming.py`
ports the *idea* from the (retired) `torch_export` branch — derive the real
attribute name of every module from the model's `named_modules()` — and overlays
those names onto the accurate profile-based tree instead of rebuilding it.

- **`build_ref_tree(named_modules)`** — pure/torch-free. Builds a reference tree
  of `{attr, cls, children, is_group, group_size}` from
  `[(qualified_name, class_name), ...]`. Indexed `ModuleList` containers are
  *inlined* (their `forward` is never called, so they emit no module event) and
  consecutive numeric entries of the same class collapse into one `is_group`
  representative — mirroring the trace's layer collapse.
- **`ref_tree_from_llm(llm)`** — extracts the tree from the *live* vLLM model
  during profiling (via `LLM.apply_model`, falling back to attribute traversal).
  Cheap: the model is already loaded. This is the primary path.
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
  order (holds for q_norm-before-k_norm, the layer norms, etc.).

### Symbolic Shape System

Op shapes use symbolic expressions for dimensions:

- **`cfg` dict** stores original config.json values (e.g., `cfg["num_heads"] = 32`)
- **`cfg["_tp_*"]` keys** store TP-divided values for numeric calculations (e.g., `cfg["_tp_num_heads"] = 8` when TP=4)
- **Shape strings** always use `/TP` for TP-aware dims: `"n_h/TP"`, `"QKV/TP"`, `"I/TP"`, `"V/TP"`
- **`symbols` dict** in result contains original (undivided) values + `"TP": tp_size` (always present, even TP=1)
- **Variable symbols** stay symbolic in exports: `S` (seq_len), `B` (batch), `C` (context_len), `TP`
- Frontend `symTooltip()` resolves `/TP` suffix by dividing base value by TP

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

### Op Classification (`classifier.py`)

Classifies ops by name prefix/pattern to backends. Priority order:
1. vllm-xpu-kernels (exact match from registry)
2. triton (kernel name patterns)
3. torch-xpu-ops (aten:: ops that run on XPU)
4. cpu (fallback ops)
5. framework (reshaping, profiler markers)

## Conventions

- **License header** — All Python files start with `# SPDX-License-Identifier: Apache-2.0`
- **Type annotations** — Use `from __future__ import annotations` and modern syntax (`dict[str, Any]`, `list[int] | None`)
- **Environment** — Assume torch-xpu and vLLM are installed and an Intel XPU is available. `model_graph.py`/`model_info.py` remain import-light (they don't pull in torch/vLLM at import time), which keeps static analysis fast, but supporting a torch-free CPU-only install is no longer a design requirement.
- **Symbolic shapes** — Op shapes use string symbols (`"H"`, `"S"`, `"n_h·d/TP"`) with `/TP` for TP-divided dims
- **TP-awareness** — All graph builders accept `tp_size`; shapes always show `/TP` for split dimensions; `cfg["_tp_*"]` keys hold divided values for numeric calculations

## Adding a New Model

1. If the architecture is new, add mapping in `_ARCH_FAMILY_MAP` in `model_graph.py`
2. If the attention/MLP pattern is novel, add a new builder function
3. Test with: `python -c "from breakdown.model_graph import build_model_graph; ..."`

## Adding a New Op/Kernel

1. Add the op name to `breakdown/registry.py` `ALL_VLLM_XPU_OPS` set
2. If the op has a unique classification pattern, update `breakdown/classifier.py`
3. Reference the op in the appropriate graph builder with correct shapes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/<hf_id>` | GET | Fetch/summarize HF model config (+ `min_profile_layers`) |
| `/api/cached-models` | GET | List previously loaded model IDs |
| `/api/profile` | POST | Start async profiling run |
| `/api/profile/upload` | POST | Reconstruct graph + op breakdown from uploaded trace file(s) |
| `/api/profile/status` | GET | Poll profiling status |
| `/api/profile/result` | GET | Fetch profiling result (ops + reconstructed graph) |
| `/api/profile/trace` | GET | Download raw trace file |
| `/api/export/shape-matrix` | POST | Export config-driven multi-config shape sweep to Excel |

## Common Pitfalls

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
  traces don't cross-contaminate. Equal batches (or only legacy `batch_size`) → a
  single pass, identical to before. Frontend sends `prefill_batch_size` /
  `decode_batch_size`; `setPhase` no longer rewrites the inputs since one run now
  yields both phases. See `tests/test_two_pass_merge.py`.
- **The scheduler is pinned so decode runs the full batch every step.**
  `_run_profile` sets `max_num_seqs = max(prefill_batch, decode_batch)` (and
  `max_num_batched_tokens` large enough for a whole-batch prefill step) *before*
  constructing the `LLM`. Without this, vLLM's continuous-batching scheduler
  caps per-iteration concurrency (by its default `max_num_seqs` and by how many
  sequences' KV fits in cache) and dispatches an oversized batch in
  **partial-batch waves** — a batch of 32 as e.g. `29 + 3`. Each wave has a
  different row count (`num_running_seqs`), so `_symbolize` (which only maps the
  max decode row count to `B`) leaves the partial waves as literal ints, and
  `_merge_modules` (keys ops by `(name, shapes)`) can't merge them — they surface
  as **duplicated `29`/`3` op nodes** in the decode graph instead of one `B`
  node. Pinning `max_num_seqs` makes every decode forward run all
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
- **The first decode step is always dropped from the decode average.**
  `graph_from_trace._classify_steps` discards the first decode step's roots
  before the phase tree is built: the initial decode forward after prefill pays
  one-time warmup costs (KV/allocator warmup, oneDNN/Triton plan + autotune
  caching under `torch.compile`) that would skew the steady-state per-op latency
  average. The drop is **guarded** (`len(decode_steps) >= 2`) so a phase with a
  single decode step is never emptied. The profiling UI exposes a **Decode
  Steps** control (`max_tokens`, default **8**) with a `>= 2` minimum so at least
  one steady-state step always remains after the drop. See
  `TestGraphFromTrace.test_first_decode_step_dropped_from_average` /
  `test_single_decode_step_not_dropped`. Do NOT remove the guard or the UI
  minimum.
- **Main model class is picked by subtree size, not device time.** The
  `LogitsProcessor`'s `lm_head` matmul (`V`-wide, once per decode step) can
  out-weigh the whole model forward, so selecting the main class by `sub_dev`
  wrongly picked `LogitsProcessor` — making every step a 1-token pass, so the
  prefill phase disappeared (`prefill: None`, blank graph until you click Decode)
  and prefill/decode looked identical. `_partition_steps` now uses
  `_subtree_module_count`. Do NOT revert to a device-time heuristic.
- **Module-name alignment must unwrap `*Model` levels absent from the trace.**
  vLLM nests the decoder stack under an inner `*Model` module whose `forward`
  usually emits no trace module event, so the trace nests it directly under
  `*ForCausalLM`. `module_naming._effective_ref_children` flattens reference
  levels whose class isn't among a node's actual trace-child classes; without it,
  child matching stalls at the missing level and `q_norm`/`k_norm` stay `norm`.
- **Device time is attributed by launch-site containment, not `External id`.**
  `graph_from_trace.py` links each device `kernel` to its host launch call
  (`kernel.correlation → xpu_runtime`, the "flow arrow") and attributes it to the
  deepest module/op interval containing the launch timestamp on the worker
  thread. This is deliberate: under `torch.compile` the `External id` bookkeeping
  points norm/indexer/sparse-attention kernels at compiled-region/plumbing
  cpu_ops (leaking their time), whereas launch-site containment is stable across
  eager and compiled passes. Do NOT revert to `External id` mapping
  (`_build_device_time_map` was replaced by `_collect_kernel_launches` +
  `_attribute_kernels`).
- **Triton kernels with no `cpu_op` surface as synthetic `triton::<kernel>` ops.**
  Kernels launched straight from Python via `triton.jit` (Gemma RMSNorm,
  MiniMax-M3 lightning indexer, block-sparse attention) never emit an
  `aten`/`_C` cpu_op. `_attribute_kernels` adds them as `triton::`-prefixed ops
  on their enclosing module (classified as `triton`). Real ops (`aten::mm`,
  `c10d::allreduce_`, `vllm::unified_attention_with_output`) get the kernel time
  added to themselves instead.
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
- `model_graph.py` is ~1600 lines — use `view_range` to read targeted sections
- `app.py` is ~1700 lines — use `view_range` to read targeted sections
- **Profiling reconstructs the graph from the trace** (`graph_from_trace.py`) —
  it does NOT overlay timing onto a static graph. The `annotate_graph_*` and
  `parse_trace_with_modules` helpers were removed; don't reintroduce them. There
  is **no interactive static-graph endpoint** (`/api/model/<id>/graph` and
  `/api/export/static-graph` were removed) — `build_model_graph` is only reached
  through the Shape Matrix export.
- The `_ARCH_FAMILY_MAP` keys must exactly match HuggingFace `architectures[0]` values
- Encoder models return `decode: None` (no autoregressive decode phase)
- Diffusion models return `prefill: None, decode: None` with a `note` field
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
- MiniMax-M3 sparse attention is modeled to match the **actual XPU dispatch**
  (not the DeepSeek/CUDA indexer ops). `_build_attention_ops(..., sparse=True)`
  emits: a fused `fused_minimax_m3_qknorm_rope_kv_insert` vllm-xpu-kernels op
  (replaces the dense q_norm/k_norm/rotary/reshape_and_cache — it also writes the
  K/V and index-key caches), the Triton lightning-indexer score
  (`minimax_m3_index_score` prefill / `minimax_m3_index_decode` decode), a Triton
  `minimax_m3_index_topk` (prefill only; decode fuses top-k into the score
  kernel), and the Triton block-sparse attention itself
  (`minimax_m3_sparse_attn` prefill / `minimax_m3_sparse_attn_decode` decode —
  the decode kernel merges its split-K partials internally, so there is no
  separate `merge_attn_states`). The index Q/K projection is fused into
  `qkv_proj`, so it is not a separate matmul. On XPU the flash/MSA sparse path is
  CUDA-SM100-only, so sparse attention always runs Triton — do NOT model it as
  `flash_attn_varlen_fwd`. Only the MoE layers are sparse; the dense prefix
  layers keep full attention. The MoE layer builder passes
  `sparse=cfg.get("sparse_attention")`.

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
   - Builder table matches functions in `model_graph.py` (`grep "^def _build_" breakdown/model_graph.py`)
