# Copilot Instructions — vllm-xpu-breakdown

## First Steps

Always read `AGENTS.md` in the project root before exploring or making changes. It contains the authoritative project structure, architecture, and conventions.

## Build & Run

```bash
pip install -r requirements.txt

# Web UI (no GPU needed for static analysis)
python app.py --port 8080

# CLI profiling (requires Intel XPU + vLLM installed)
python run_profile.py --model Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768
```

## Testing

```bash
# Unit tests (no GPU required)
pytest tests/test_pipeline.py -v

# Single test
pytest tests/test_pipeline.py::TestClassifier::test_vllm_xpu_kernels -v -s

# All tests including GPU integration
pytest tests/ -v
```

No linter or pre-commit is configured for this project.

## Architecture

Flask web app (`app.py`) + static SPA (`static/index.html`) backed by a `breakdown/` analysis engine.

**Two analysis modes:**
1. **Static** — Builds model op graph from HuggingFace config.json (no GPU, no model weights). Core logic in `model_graph.py`.
2. **Dynamic** — Runs vLLM inference with `torch.profiler`, parses Chrome trace JSON, classifies ops to backends.

**Key data flow:**
- `model_info.py` fetches HF config → `model_graph.py` builds `ModuleNode`/`OpNode` tree → `app.py` serializes to JSON → `index.html` renders tree
- `profiler.py` runs inference → `trace_parser.py` parses trace → `analyzer.py` computes shapes/memory/FLOPs → `classifier.py` assigns backends

**Backend classification priority:** vllm-xpu-kernels (exact match from registry) > triton (name patterns) > torch-xpu-ops (aten:: on XPU) > cpu > framework

## Key Conventions

- All Python files start with `# SPDX-License-Identifier: Apache-2.0`
- Use `from __future__ import annotations` and modern type syntax (`dict[str, Any]`, `list[int] | None`)
- `model_graph.py` and `model_info.py` must work without PyTorch/vLLM installed — static analysis has no ML dependencies
- Op shapes use string symbols (`"H"`, `"S"`, `"n_h·d"`) resolved to concrete values at display/export time
- All graph builders accept `tp_size` and produce per-rank shapes (dimensions divided by TP)
- The web UI is a single HTML file with inline CSS/JS — no bundler or build step
- `app.py` is large (~1300 lines) — use `view_range` to read targeted sections

## Adding a New Model Architecture

1. Add mapping in `_ARCH_FAMILY_MAP` in `model_graph.py` (key must exactly match HuggingFace `architectures[0]`)
2. Route to appropriate builder in `build_model_graph()` — check category sets: `_MLA_ARCHS`, `_VL_ARCHS`, `_ENCODER_ARCHS`, `_DIFFUSION_ARCHS`
3. If the attention/MLP pattern is novel, add a new `_build_*` function
4. Test: `python -c "from breakdown.model_graph import build_model_graph; from breakdown.model_info import fetch_model_config, summarize_config; c = fetch_model_config('org/model'); s = summarize_config(c); g = build_model_graph(s); print(g.keys())"`

## Adding a New Op/Kernel

1. Add op name to `ALL_VLLM_XPU_OPS` set in `breakdown/registry.py`
2. If the op has a unique classification pattern, update `breakdown/classifier.py`
3. Reference the op in the appropriate graph builder in `model_graph.py` with correct shapes

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/<hf_id>` | GET | Fetch and summarize HF model config |
| `/api/model/<hf_id>/graph` | GET | Build static model graph |
| `/api/catalog` | GET | List catalog models (filters: `?type=`, `?priority=`, `?vllm=true`) |
| `/api/profile/start` | POST | Start async profiling |
| `/api/profile/status` | GET | Poll profiling status |
| `/api/profile/trace` | GET | Download raw trace file |
| `/api/export/excel` | POST | Export profiled breakdown to Excel |
| `/api/export/static-graph` | POST | Export static graph breakdown to Excel |
