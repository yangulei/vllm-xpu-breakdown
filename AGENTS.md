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
  it), classified prefill vs decode by the pass's max matmul token dim.
- **Repeat collapse** — adjacent structurally-identical sibling layers merge into
  one node with `repeat_count`, timing averaged. Dense vs MoE layers have
  different structural signatures so they stay as separate runs.

The old `annotate_graph_timing` / `annotate_graph_from_modules` /
`parse_trace_with_modules` paths were **removed**. Do not reintroduce a
static-overlay step in the profiling worker.

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
