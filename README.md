# vLLM-XPU Ops/Kernels Breakdown

Profile vLLM inference on Intel XPU and visualize which backend handles each operation.

## Backends Tracked

| Backend | What it covers |
|---|---|
| **vllm-xpu-kernels** | Custom SYCL/DPC++ kernels (RMSNorm, activations, attention, MoE, quantization, cache ops) |
| **torch-xpu-ops** | Native PyTorch ATen operators on XPU (linear, matmul, embedding — via oneDNN/oneMKL) |
| **triton** | Triton-compiled kernels (attention backends, sampling, torch.compile output) |
| **cpu** | Operations running on CPU |
| **framework** | Tensor reshaping, memory ops, profiler overhead |

## Analysis Model

The in-app **Model Graph is always reconstructed from a profiling run** — there
is no static graph view.

- **Profiling (profile-first)** — Runs real vLLM inference on Intel XPU with
  `torch.profiler` (`with_stack` + `record_shapes`) and **reconstructs the model
  graph directly from the trace**: the captured module call stack rebuilds the
  module hierarchy, `Input Dims` give real op shapes, and kernel device time
  is attributed to each op through the `correlation → runtime → External id`
  chain. Module nodes carry their **real attribute names** (`q_norm`/`k_norm`,
  `self_attn`, ...) because profiling installs forward hooks that emit
  `record_function("module::<path>::<Cls>")` spans at capture time; a
  `named_modules()` overlay is the fallback for legacy/uploaded traces without
  those spans. Because the tree is derived from what actually executed, it tracks
  whatever vLLM/the backends dispatched and does not drift as vLLM evolves.
  Requires a working Intel XPU with torch-xpu and vLLM installed.
- **Static shape sweeps** — `build_model_graph` derives op shapes/memory/FLOPs
  from the HuggingFace `config.json` for the **Shape Matrix export** (a
  config-driven sweep across seq/context/batch/TP). It no longer powers an
  interactive graph view — the in-app Model Graph is always the profiled
  reconstruction.

## Supported Architectures

Standard GQA decoders (Llama, Qwen, Mistral), MoE (Mixtral, Qwen-MoE), VL,
and encoder/embedding models are supported for both static analysis and
profiling. **MLA (Multi-head Latent Attention) models** (DeepSeek-V2/V3/V4,
GLM-MoE-DSA) are now supported on XPU too — dense MLA routes to the
`TRITON_MLA` backend and DeepSeek sparse attention routes to the
`XPU_MLA_SPARSE` backend in vLLM-XPU. **MiniMax-M3** (vision-language MoE
with sparse attention) is supported on XPU as well — its nested
`text_config`/`vision_config` layout, per-layer dense/MoE split, shared
experts, and Triton lightning-indexer sparse attention (index score + top-k
block selection + block-sparse attend, the actual XPU dispatch) are modeled in
the static graph. Diffusion (T2I/T2V) models support
static analysis only (not vLLM-served).

## Web UI (Recommended)

Interactive web application for exploring the breakdown:

```bash
pip install -r requirements.txt
python app.py [--port 8080]
```

Then open `http://localhost:8080` in your browser.

**Features:**
- Search for any HuggingFace model by ID
- Auto-loads model config (architecture, layers, MoE, dtype)
- Toggle between eager and torch.compile mode
- Reconstructed model graph (from a profiling run) with symbolic shapes and TP-aware annotations
- Quantization support (fp8, gptq, awq) — affects weight dtype and memory estimates
- Shape Matrix Export: sweep across seq_len, batch_size, context_len, and TP configurations
- Rich ops table with:
  - Shapes with symbolic dimensions (B=batch, S=seq_len, H=hidden_size, etc.)
  - Per-tensor dtype tags (bf16, fp8, int4)
  - TP-divided dimensions shown as `symbol/TP` (e.g., `n_h/TP`, `QKV/TP`)
  - Merged duplicate layers with ×N repetition count
  - Memory estimation, FLOPs, and arithmetic intensity
  - Sortable and filterable by backend
- Backend distribution chart

## CLI (Quick One-Shot)

```bash
# Profile a model
python run_profile.py --model Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768

# Profile with custom prompt and batch size
python run_profile.py --model meta-llama/Llama-3.2-1B-Instruct \
    --prompt "Explain quantum computing" \
    --max-tokens 512 --batch-size 4

# Separate prefill / decode batch sizes (mirrors real serving: prefill ~1,
# decode 32/64/128). Runs two passes → output/prefill and output/decode.
python run_profile.py --model meta-llama/Llama-3.2-1B-Instruct \
    --prefill-batch-size 1 --decode-batch-size 32
```

All standard vLLM `EngineArgs` (e.g., `--model`, `--max-model-len`, `--dtype`) are passed through.

## Output

Reports are written to `output/` (or `--output-dir`):

| File | Description |
|---|---|
| `report.txt` | Console summary with backend breakdown and top-N ops |
| `ops_breakdown.csv` | Every op with backend, timing, and call count |
| `ops_breakdown.json` | Full structured data for programmatic analysis |
| `breakdown.html` | Static HTML report with charts and sortable tables |
| `trace.json` | Chrome trace (`chrome://tracing`) for timeline analysis |

## Shape Matrix Export

Export op shapes across multiple configurations for analysis:

```
POST /api/export/shape-matrix
{
  "model_id": "Qwen/Qwen3-4B",
  "source": "config",
  "prefill_seq_lens": [128, 1024, 4096],
  "prefill_ctx_lens": [0, 8192],
  "prefill_batch_sizes": [1],
  "decode_ctx_lens": [8192],
  "decode_batch_sizes": [1, 8, 32, 128],
  "tp_sizes": [1, 4],
  "quantization": "auto"
}
```

Produces an Excel file with one row per (Phase, SeqLen, CtxLen, BatchSize, TP, Op) combination. Columns include symbolic shapes (with `S`, `B`, `C`, `TP` kept as variables), concrete shapes with dtypes, memory, FLOPs, and arithmetic intensity.

**Two shape sources** (`source`):

- **`"config"`** (default) — a fast, analytic sweep derived from the HuggingFace
  `config.json`. The op set is *inferred*, no profiling run required.
- **`"profile"`** — grounds the sweep in a **real profiling run**: profile once at
  a typical config, and the actually-dispatched ops (with their symbolic shapes)
  become a template that is re-resolved for every other (seq/ctx/batch/TP) case,
  with memory/FLOPs recomputed per config. This makes the matrix accurate to what
  the model really executes on XPU — the intended input for downstream
  optimization work. Requires a completed profiling run for the same model
  (use the **Shape Source** control in the Shape Matrix tab: *use latest run* or
  *run a fresh profile now*). The exported workbook adds an **Info** sheet with the
  profiled config and caveats. Because the op set is fixed at the profiled config,
  profile at **each TP** you need (sweeping TP only divides `/TP` dims). Query,
  batch, and context (`S`/`B`/`C`) are **parametric** — you do **not** profile per
  context: one base profile with any small non-zero context captures `C` (as `S+C`
  on the attention KV rows) and every other context length is derived. (A base
  profile with `context=0` leaves the KV rows as `S`, so context can't be derived.)

## Comparing Eager vs Compiled

```bash
./scripts/compare_modes.sh --model Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768
```

## Architecture

```
app.py                  Web server (Flask) — model config, profiling, exports, REST API
run_profile.py          CLI entry point — standalone profiling + reports
static/index.html       Interactive frontend (SPA, vanilla JS)
breakdown/
  model_graph.py        Config-driven shape builder (Shape Matrix export sweep)
  graph_from_trace.py   Profile-first graph reconstruction (from torch profiler trace)
  module_hooks.py       Capture-time module-name spans (forward hooks; research R1)
  module_naming.py      Fallback name overlay from named_modules() (legacy/upload traces)
  model_info.py         HuggingFace model config fetching & summarization
  analyzer.py           Shape symbolization, memory/FLOPs estimation, layer merging
  profiler.py           torch.profiler wrapper (XPU activity, shapes, stacks)
  classifier.py         Op classification: vllm-xpu-kernels | triton | torch-xpu-ops | cpu
  registry.py           Known op list from vllm-xpu-kernels (68 ops across 4 modules)
  trace_parser.py       Chrome trace JSON parser + module/role inference helpers
  trace_common.py       Torch-free trace helpers (overhead filter, module-span labels)
  report.py             Console, CSV, JSON report generators
  visualize.py          Static HTML report generator
tests/
  test_pipeline.py            Unit tests (incl. graph reconstruction; requires torch)
  test_module_spans.py        Capture-time module-span emission + reconstruction tests
  test_shape_matrix_export.py Shape Matrix Export endpoint tests
  test_real_profile.py        Integration test (requires GPU)
```

## How Classification Works

1. Op name matched against the 68 registered vllm-xpu-kernels ops → **vllm-xpu-kernels**
2. Op name contains Triton indicators (`triton_`, `CompiledFxGraph`) → **triton**
3. `aten::` compute ops with XPU device time > 0 → **torch-xpu-ops**
4. Shape/view ops, profiler overhead → **framework**
5. Everything else with no device time → **cpu**

## Symbolic Shape System

Op shapes use symbolic expressions that represent model config values:

- **Config symbols** (resolved to numbers from `config.json`): `H`, `n_h`, `n_kv`, `d`, `I`, `V`, `QKV`, `n_h·d`
- **Variable symbols** (stay symbolic — user-defined): `S` (seq_len), `B` (batch), `C` (context_len), `TP` (tensor_parallel)
- **TP-divided dims** always show `/TP` suffix: `n_h/TP`, `QKV/TP`, `I/TP`, `V/TP`, etc.
- `TP` is always in the symbols dict (value = tp_size, even when 1)

This makes it easy to identify which dimensions are split across tensor-parallel ranks.
