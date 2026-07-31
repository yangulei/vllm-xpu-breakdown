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

**Analysis model** — the in-app Model Graph is **always reconstructed from a profiling run**; there is no interactive static-graph view and no config-driven graph builder. **Dynamic (profile-first)** — Runs vLLM inference on Intel XPU with `torch.profiler` (`with_stack` + `record_shapes`), then **reconstructs the model graph directly from the trace** in `graph_from_trace.py` (nn.Module call stack + `Input Dims` shapes + `Input type` dtypes + kernel device time via `correlation → runtime → External id`). The reconstructed tree reflects what actually executed, so it doesn't drift as vLLM/backends change. Both the web UI graph and the Shape Matrix Excel export (`/api/export/shape-matrix`) are built from this reconstruction.

**Key data flow:**
- `profiler.py` runs inference → `graph_from_trace.py` reconstructs the module/op tree from the trace (reusing `analyzer.py` shapes/memory/FLOPs + `classifier.py` backends) → `app.py` serializes to JSON → `index.html` renders tree. The removed `annotate_graph_*` / `parse_trace_with_modules` static-overlay path and the deleted `model_graph.py` config builder are not used anymore.
- **Module names — capture-time spans (primary, `module_hooks.py`).** During profiling, `_run_profile` installs forward hooks (via `LLM.apply_model(install_module_span_hooks_on)`) that emit `record_function("module::<qualified_name>::<Cls>")` `user_annotation` spans around every module's forward. `graph_from_trace._build_raw_forest` auto-detects those spans and builds module nodes with **real attribute names straight from the trace** — no alignment, no registration-order assumption, correct even under async. Label helpers (`module_span_label`/`parse_module_span`/`module_span_display_name`) live torch-free in `trace_common.py`.
- **Module names — `named_modules()` overlay (fallback, `module_naming.py`).** For legacy/upload traces without spans, `module_naming.py` recovers names from the live model's `named_modules()` (`ref_tree_from_llm`) and overlays them (`graph_from_trace._apply_ref_names`, before repeat-collapse). `build_graph_from_trace` applies it **only** when the forest has no captured names (`_forest_has_named_modules`). Alignment unwraps reference `*Model` levels absent from the trace (`_effective_ref_children`) — vLLM's inner `*Model.forward` often emits no module event, and without unwrapping, matching stalls and names fall back to `norm`.

**Two subtle invariants (don't regress):** (1) `_partition_steps` picks the main model class by **largest module subtree** (`_subtree_module_count`), NOT device time — the `LogitsProcessor` `lm_head` matmul can dominate `sub_dev`, which previously mis-selected it, dropped the prefill phase (`prefill: None`) and made both phases identical. (2) The frontend (`applyProfileResult`) auto-selects a phase that has a reconstructed tree so the graph shows immediately.

**Query Len / Context Len profiling (Method 1 / APC):** `_run_profile` builds exact-length synthetic token prompts (`_make_token_ids`) instead of a fixed text prompt. `query_len` = new prefill tokens (`S`); `context_len` (floored to `block_size`) is pre-computed in an un-profiled warm pass and served from the prefix cache (`enable_prefix_caching=True`) during the profiled run, so the profiled prefill sees `S` new tokens attending to a `context_len`-token KV. Warm the context prefix alone first, warm kernels with distinct query seeds, and profile with yet another seed (`900000+b`) so profiled queries are never cache-hit; each batch item gets a distinct query over a shared context. A cache miss (`outputs[0].num_cached_tokens < ctx_aligned`) is surfaced as `cache_hit_note`. `start_profile` bumps `max_model_len` to `context+query+max_tokens`. Absent `query_len` → legacy text-prompt path.

**Replay benchmark (`breakdown/bench/`)** — the Shape Matrix rows become
**replay cases**: the trace records each op's dispatch name, per-tensor
shapes/dtypes/strides and its non-tensor argument values, so the benchmark
**re-invokes the op vLLM actually dispatched** (`torch.ops.<ns>.<op>`, or the
recorded Python API frame for Triton/FlashInfer/SYCL-extension kernels) instead
of benchmarking a substitute. There is no adapter table: coverage follows the
profile. Each op is replayed in its own process (a bad shape can wedge the
device), collectives are replayed on `TP` peer ranks with rank 0 recorded, and
a ranker scores ops by calls × latency × roofline headroom, producing
`output/bench/<run_id>/targets.json` for the `xpu-kernel-optimizer` skill. Runs
headless via `python -m breakdown.bench {plan,run,rank,report,case,history,all}`.
Integer/index operands are **never** filled randomly — an op without a
registered synthesizer is reported, not guessed. A context-bound wrapper with a
**context-free kernel entry point** is replayed through it — attention and the
KV-cache write rebuild a paged KV cache + block table + sequence metadata
(`bench/recipes/attention.py`), so the heaviest op in the model is measured
rather than refused; a wrapper with no such entry point (`vllm::moe_forward*`)
is still reported with a reason, and the kernels it launches are benchmarked as
their own ops. The roofline classifies an op as compute- or memory-bound by its
**arithmetic intensity vs the machine balance** (not by whichever utilization
came out larger), and charges a cache-resident op to the last-level-cache
bandwidth instead of DRAM. Timing repeats the kernel inside
a device-event window and subtracts the measured empty-window cost, because that
floor (~60-90 us on Level Zero) dwarfs a small kernel. The old `breakdown/perf/`
op-map + xpu-perf/micro_perf shell-out was removed; do not reintroduce it.

**Backend classification priority:** ccl (collective-comm: `c10d::`/`ccl::` or all_reduce/all_gather/reduce_scatter/all_to_all) > vllm-xpu-kernels (exact match from registry) > flashinfer (name contains `flashinfer`) > triton (name patterns) > torch-xpu-ops (aten:: on XPU) > cpu > framework

## Key Conventions

- All Python files start with `# SPDX-License-Identifier: Apache-2.0`
- Use `from __future__ import annotations` and modern type syntax (`dict[str, Any]`, `list[int] | None`)
- Assume torch-xpu + vLLM are installed and an Intel XPU is available; `model_info.py` stays import-light but a torch-free CPU-only install is no longer a design goal
- Op shapes use string symbols (`"H"`, `"S"`, `"n_h·d"`) resolved to concrete values at display/export time
- Graph reconstruction accepts `tp_size` and produces per-rank shapes (dimensions divided by TP)
- The web UI is a single HTML file with inline CSS/JS — no bundler or build step
- `app.py` is large (~1700 lines) — use `view_range` to read targeted sections

## Adding a New Model Architecture

The model graph is reconstructed from the trace, so no static builder is needed:
1. Ensure `breakdown/model_info.py` `summarize_config` extracts the model's key dims so shapes symbolize (`S`/`B`/`C`/`H`/`I`/`n_h·d`/…).
2. Classify any novel ops (see below).
3. Profile it and confirm the reconstructed graph + Shape Matrix look right.

## Adding a New Op/Kernel

1. Add op name to `ALL_VLLM_XPU_OPS` set in `breakdown/registry.py`
2. If the op has a unique classification pattern, update `breakdown/classifier.py`
3. Benchmarking needs no adapter. Add an input synthesizer
   (`breakdown/bench/inputs.py` / `recipes/`) if it takes an index tensor, a
   `resolve.PYTHON_API` entry if it is launched straight from Python, and an
   entry in `breakdown/bench/kernel_sources.json` if it has editable source

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
| `/api/bench/plan` | POST | Sweep the profiled graph into replay cases |
| `/api/bench/run` | POST | Replay a run's cases (async) |
| `/api/bench/status` | GET | Poll the benchmark run |
| `/api/bench/runs` | GET | List bench runs |
| `/api/bench/results` | GET | Measured cases + summary + coverage |
| `/api/bench/targets` | GET | Ranked optimization targets (`targets.json`) |
| `/api/bench/report` | GET | Download a run's report workbook |
| `/api/bench/history` | GET | Benchmark history / two-run regression diff |


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
