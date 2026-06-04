# vLLM-XPU Ops/Kernels Breakdown

Profile vLLM inference on Intel XPU and visualize which backend handles each operation.

## Requirements / Environment Assumptions

This tool assumes it runs inside the standard vLLM-XPU development environment
(e.g. the `intel/vllm:*-xpu` Docker image):

- **PyTorch is installed** (with XPU support).
- **vLLM (vllm-xpu) is installed.**
- **An Intel XPU (GPU) is available.**

Both the static model-graph analysis and the dynamic profiling paths may import
PyTorch / vLLM and use the live model definitions and runtime dispatch. There is
no separate "no-GPU / no-ML-dependencies" mode — install
`requirements.txt` on top of an existing PyTorch + vLLM-XPU install.

### Model Graph: trace-based vs. static

The Model Graph (op tree + per-op backend, without profiling) is built two ways:

- **Trace-based (default for dense models).** The tool instantiates the *real*
  vLLM `nn.Module` offline on the `meta` device (no weights, no download) and
  runs a single `torch.export` symbolic trace with the token dimension kept
  symbolic. The op tree, op names, and backends come straight from the real
  dispatch, so the graph tracks vLLM/vllm-xpu-kernels changes automatically —
  no per-architecture builder code. Backends are resolved through the shared
  `classifier`, the single source of truth. Implemented in
  `breakdown/model_tracer.py`; the result carries `graph_source: "torch.export"`.
- **Static (fallback).** Hand-written per-architecture builders in
  `breakdown/model_graph.py`. Used for MoE, multimodal, tensor-parallel
  (`tp_size > 1`), or quantized models, and whenever tracing fails for any
  reason. The result carries `graph_source: "static"`.

The trace-based path is intentionally limited to dense, single-stack, TP=1,
unquantized models: a single symbolic export does not fully capture fused-expert
(`FusedMoE`) or vision-tower compute, so those stay on the static builders to
avoid silently under-counting cost.

## Backends Tracked

| Backend | What it covers |
|---|---|
| **vllm-xpu-kernels** | Custom SYCL/DPC++ kernels (RMSNorm, activations, attention, MoE, quantization, cache ops) |
| **torch-xpu-ops** | Native PyTorch ATen operators on XPU (linear, matmul, embedding — via oneDNN/oneMKL) |
| **triton** | Triton-compiled kernels (attention backends, sampling, torch.compile output) |
| **cpu** | Operations running on CPU |
| **framework** | Tensor reshaping, memory ops, profiler overhead |

## Supported Architectures

Standard GQA decoders (Llama, Qwen, Mistral), MoE (Mixtral, Qwen-MoE), VL,
and encoder/embedding models are supported for both static analysis and
profiling. **MLA (Multi-head Latent Attention) models** (DeepSeek-V2/V3/V4,
GLM-MoE-DSA) are now supported on XPU too — dense MLA routes to the
`TRITON_MLA` backend and DeepSeek sparse attention routes to the
`XPU_MLA_SPARSE` backend in vLLM-XPU. Diffusion (T2I/T2V) models support
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
- Static model graph with symbolic shapes and TP-aware annotations
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
- Model catalog with 65+ models across LLM, MLLM, T2I, T2V, Audio, Embedding

## CLI (Quick One-Shot)

```bash
# Profile a model
python run_profile.py --model Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768

# Profile with custom prompt and batch size
python run_profile.py --model meta-llama/Llama-3.2-1B-Instruct \
    --prompt "Explain quantum computing" \
    --max-tokens 512 --batch-size 4
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
  model_catalog.py      Registry of 65+ target models with metadata
  model_graph.py        Static model graph builder + trace/static dispatch (core engine)
  model_tracer.py       Trace-based graph builder (real vLLM nn.Module + torch.export)
  model_info.py         HuggingFace model config fetching & summarization
  analyzer.py           Shape symbolization, memory/FLOPs estimation, layer merging
  profiler.py           torch.profiler wrapper (XPU activity, shapes, stacks)
  classifier.py         Op classification: vllm-xpu-kernels | triton | torch-xpu-ops | cpu
  registry.py           Known op list from vllm-xpu-kernels (68 ops across 4 modules)
  trace_parser.py       Chrome trace JSON parser
  report.py             Console, CSV, JSON report generators
  visualize.py          Static HTML report generator
tests/
  test_pipeline.py            Unit tests (requires torch)
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
