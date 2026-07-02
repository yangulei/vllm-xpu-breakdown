# Copilot Instructions — vllm-xpu-breakdown

## First Steps

Always read `AGENTS.md` in the project root before exploring or making changes. It contains the authoritative project structure, architecture, and conventions.

## Build & Run

```bash
pip install -r requirements.txt

# Web UI (static analysis works without loading model weights)
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

**Analysis model** — the in-app Model Graph is **always reconstructed from a profiling run**; there is no interactive static-graph view.
1. **Dynamic (profile-first)** — Runs vLLM inference on Intel XPU with `torch.profiler` (`with_stack` + `record_shapes`), then **reconstructs the model graph directly from the trace** in `graph_from_trace.py` (nn.Module call stack + `Input Dims` shapes + kernel device time via `correlation → runtime → External id`). The reconstructed tree reflects what actually executed, so it doesn't drift as vLLM/backends change.
2. **Config-driven shape sweep** — `build_model_graph` (in `model_graph.py`) derives op shapes/memory/FLOPs from `config.json` for the **Shape Matrix Excel export** (`/api/export/shape-matrix`). It is not exposed as an interactive graph endpoint.

**Key data flow:**
- `model_info.py` fetches HF config → `model_graph.py` builds `ModuleNode`/`OpNode` tree → `app.py` serializes to JSON → `index.html` renders tree
- `profiler.py` runs inference → `graph_from_trace.py` reconstructs the module/op tree from the trace (reusing `analyzer.py` shapes/memory/FLOPs + `classifier.py` backends) → `app.py` serializes to JSON. The removed `annotate_graph_*` / `parse_trace_with_modules` static-overlay path is not used anymore.
- `module_naming.py` recovers real module attribute names (`q_norm`/`k_norm`, `input_layernorm`, ...) from the live model's `named_modules()` (`ref_tree_from_llm`, captured during profiling) and overlays them onto the reconstructed tree (`graph_from_trace._apply_ref_names`, applied before repeat-collapse). The trace alone only exposes class names, so same-class siblings would otherwise be indistinguishable. Opt-in via `build_graph_from_trace(..., ref_module_tree=...)`. Alignment unwraps reference `*Model` levels absent from the trace (`_effective_ref_children`) — vLLM's inner `*Model.forward` often emits no module event, and without unwrapping, matching stalls and names fall back to `norm`.

**Two subtle invariants (don't regress):** (1) `_partition_steps` picks the main model class by **largest module subtree** (`_subtree_module_count`), NOT device time — the `LogitsProcessor` `lm_head` matmul can dominate `sub_dev`, which previously mis-selected it, dropped the prefill phase (`prefill: None`) and made both phases identical. (2) The frontend (`applyProfileResult`) auto-selects a phase that has a reconstructed tree so the graph shows immediately.

**Backend classification priority:** vllm-xpu-kernels (exact match from registry) > triton (name patterns) > torch-xpu-ops (aten:: on XPU) > cpu > framework

## Key Conventions

- All Python files start with `# SPDX-License-Identifier: Apache-2.0`
- Use `from __future__ import annotations` and modern type syntax (`dict[str, Any]`, `list[int] | None`)
- Assume torch-xpu + vLLM are installed and an Intel XPU is available; `model_graph.py`/`model_info.py` stay import-light but a torch-free CPU-only install is no longer a design goal
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
| `/api/model/<hf_id>` | GET | Fetch/summarize HF model config (+ `min_profile_layers`) |
| `/api/profile` | POST | Start async profiling |
| `/api/profile/upload` | POST | Reconstruct graph + op breakdown from uploaded trace file(s) |
| `/api/profile/status` | GET | Poll profiling status |
| `/api/profile/result` | GET | Fetch profiling result (ops + reconstructed graph) |
| `/api/profile/trace` | GET | Download raw trace file |
| `/api/export/shape-matrix` | POST | Export config-driven shape sweep to Excel |


<!-- headroom:rtk-instructions -->
# RTK (Rust Token Killer) - Token-Optimized Commands

When running shell commands, **always prefix with `rtk`**. This reduces context
usage by 60-90% with zero behavior change. If rtk has no filter for a command,
it passes through unchanged — so it is always safe to use.

## Key Commands
```bash
# Git (59-80% savings)
rtk git status          rtk git diff            rtk git log

# Files & Search (60-75% savings)
rtk ls <path>           rtk read <file>         rtk grep <pattern>
rtk find <pattern>      rtk diff <file>

# Test (90-99% savings) — shows failures only
rtk pytest tests/       rtk cargo test          rtk test <cmd>

# Build & Lint (80-90% savings) — shows errors only
rtk tsc                 rtk lint                rtk cargo build
rtk prettier --check    rtk mypy                rtk ruff check

# Analysis (70-90% savings)
rtk err <cmd>           rtk log <file>          rtk json <file>
rtk summary <cmd>       rtk deps                rtk env

# GitHub (26-87% savings)
rtk gh pr view <n>      rtk gh run list         rtk gh issue list

# Infrastructure (85% savings)
rtk docker ps           rtk kubectl get         rtk docker logs <c>

# Package managers (70-90% savings)
rtk pip list            rtk pnpm install        rtk npm run <script>
```

## Rules
- In command chains, prefix each segment: `rtk git add . && rtk git commit -m "msg"`
- For debugging, use raw command without rtk prefix
- `rtk proxy <cmd>` runs command without filtering but tracks usage
<!-- /headroom:rtk-instructions -->
