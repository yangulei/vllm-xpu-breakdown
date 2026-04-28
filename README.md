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
- Rich ops table with:
  - Shapes with symbolic dimensions (B=batch, S=seq_len, H=hidden_size, etc.)
  - Merged duplicate layers with ×N repetition count
  - Memory estimation, FLOPs, and arithmetic intensity
  - Sortable and filterable by backend
- Backend distribution chart
- "Load Demo" button for UI preview without running profiling

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

## Comparing Eager vs Compiled

```bash
./scripts/compare_modes.sh --model Qwen/Qwen3-4B-Instruct-2507 --max-model-len 32768
```

## Architecture

```
app.py                  Web server (Flask) — model config, profiling, REST API
run_profile.py          CLI entry point — standalone profiling + reports
static/index.html       Interactive frontend (SPA, vanilla JS)
breakdown/
  model_info.py         HuggingFace model config fetching & summarization
  analyzer.py           Shape symbolization, memory/FLOPs estimation, layer merging
  profiler.py           torch.profiler wrapper (XPU activity, shapes, stacks)
  classifier.py         Op classification: vllm-xpu-kernels | triton | torch-xpu-ops | cpu
  registry.py           Known op list from vllm-xpu-kernels (68 ops across 4 modules)
  report.py             Console, CSV, JSON report generators
  visualize.py          Static HTML report generator
```

## How Classification Works

1. Op name matched against the 68 registered vllm-xpu-kernels ops → **vllm-xpu-kernels**
2. Op name contains Triton indicators (`triton_`, `CompiledFxGraph`) → **triton**
3. `aten::` compute ops with XPU device time > 0 → **torch-xpu-ops**
4. Shape/view ops, profiler overhead → **framework**
5. Everything else with no device time → **cpu**
