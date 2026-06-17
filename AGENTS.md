# AGENTS.md — vllm-xpu-breakdown

Instructions for AI agents working in this repository.

## Overview

This is a profiling and static-analysis tool for vLLM inference on Intel XPU (GPU) hardware. It visualizes which backend handles each operation during inference:
- **vllm-xpu-kernels** — Custom SYCL/DPC++ kernels
- **torch-xpu-ops** — PyTorch ATen on XPU (via oneDNN/oneMKL)
- **triton** — Intel Triton-compiled kernels
- **cpu** — CPU fallbacks
- **framework** — Tensor reshaping, profiler overhead

The tool has two modes:
1. **Static analysis** — Builds a model op graph from HuggingFace config without GPU
2. **Dynamic profiling** — Runs actual inference via vLLM and parses torch profiler traces

## Project Structure

```
app.py                    — Flask web server (API + static serving + exports)
run_profile.py            — CLI profiling entry point
chat.py                   — Interactive chat with profiling
breakdown/
  model_graph.py          — Static model graph builder (core engine)
  model_info.py           — HuggingFace config fetcher and summarizer
  analyzer.py             — Op analysis (shapes, memory, FLOPs, AI)
  classifier.py           — Op → backend classification
  registry.py             — Known vllm-xpu-kernels ops list
  trace_parser.py         — Chrome trace JSON parser
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

# Run web UI (no GPU needed for static analysis)
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

The core engine for static analysis. Key concepts:

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
- **No external ML dependencies for static analysis** — `model_graph.py` and `model_info.py` must work without PyTorch/vLLM installed
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
| `/api/model/<hf_id>` | GET | Fetch and summarize HF model config |
| `/api/model/<hf_id>/graph` | GET | Build static model graph |
| `/api/profile/start` | POST | Start async profiling |
| `/api/profile/status` | GET | Poll profiling status |
| `/api/profile/trace` | GET | Download raw trace file |
| `/api/export/shape-matrix` | POST | Export multi-config shape sweep to Excel |
| `/api/export/excel` | POST | Export profiled breakdown to Excel |
| `/api/export/static-graph` | POST | Export static graph breakdown to Excel |

## Common Pitfalls

- `model_graph.py` is ~1600 lines — use `view_range` to read targeted sections
- `app.py` is ~1900 lines — use `view_range` to read targeted sections
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
- MiniMax-M3 sparse attention (DeepSeek-style "lightning indexer") IS modeled:
  `_build_attention_ops(..., sparse=True)` adds an `indexer_proj` (aten::mm),
  `indexer_k_quant_and_cache`, `top_k_per_row_prefill`/`top_k_per_row_decode`,
  and `merge_attn_states` (all registered vllm-xpu-kernels). Only the MoE layers
  are sparse — the dense prefix layers keep full attention, matching
  `sparse_attention_freq` (= the dense/MoE split). The MoE layer builder passes
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
