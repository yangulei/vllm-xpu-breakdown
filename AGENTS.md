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
app.py                    — Flask web server (API + static serving)
run_profile.py            — CLI profiling entry point
chat.py                   — Interactive chat with profiling
breakdown/
  model_catalog.py        — Registry of 65+ target models with metadata
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
  run_catalog_models.sh   — Batch static analysis for catalog models
  compare_modes.sh        — Compare eager vs compile modes
tests/
  test_pipeline.py        — Unit tests
  test_profile_reduced_layers.py
  test_real_profile.py    — Integration test (requires GPU)
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

### Model Catalog (`model_catalog.py`)

Registry of target models with:
- HuggingFace IDs, precision targets, model type
- Owner, focus area, priority, CRI plan status
- `vllm_supported` flag (False for diffusion/video models)

Categories: LLM, MLLM, T2I, T2V, Audio, Embedding/Reranker, Segmentation, MTP

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
- **Symbolic shapes** — Op shapes use string symbols (`"H"`, `"S"`, `"n_h·d"`) when concrete values aren't available
- **TP-awareness** — All graph builders accept `tp_size` and produce per-rank shapes

## Adding a New Model

1. Add entry to `breakdown/model_catalog.py` in the appropriate category list
2. If the architecture is new, add mapping in `_ARCH_FAMILY_MAP` in `model_graph.py`
3. If the attention/MLP pattern is novel, add a new builder function
4. Test with: `python -c "from breakdown.model_graph import build_model_graph; ..."`

## Adding a New Op/Kernel

1. Add the op name to `breakdown/registry.py` `ALL_VLLM_XPU_OPS` set
2. If the op has a unique classification pattern, update `breakdown/classifier.py`
3. Reference the op in the appropriate graph builder with correct shapes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/<hf_id>` | GET | Fetch and summarize HF model config |
| `/api/model/<hf_id>/graph` | GET | Build static model graph |
| `/api/catalog` | GET | List models with `?type=`, `?priority=`, `?vllm=true` filters |
| `/api/catalog/<name>` | GET | Get single catalog model details |
| `/api/profile/start` | POST | Start async profiling |
| `/api/profile/status` | GET | Poll profiling status |
| `/api/profile/trace` | GET | Download raw trace file |

## Common Pitfalls

- `model_graph.py` is ~1500 lines — use `view_range` to read targeted sections
- The `_ARCH_FAMILY_MAP` keys must exactly match HuggingFace `architectures[0]` values
- Encoder models return `decode: None` (no autoregressive decode phase)
- Diffusion models return `prefill: None, decode: None` with a `note` field
- The web UI is a single HTML file with inline CSS/JS — no build step needed
- `model_info.py` fetches from HuggingFace API — tests should mock HTTP calls
