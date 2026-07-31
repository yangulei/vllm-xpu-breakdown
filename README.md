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
- **Shape Matrix export** — sweeps op shapes/memory/FLOPs across
  seq/context/batch/TP configurations, grounded in a real profiling run (the
  reconstructed ops are re-resolved per config). See below.

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
  "prefill_seq_lens": [128, 1024, 4096],
  "prefill_ctx_lens": [0, 8192],
  "prefill_batch_sizes": [1],
  "decode_ctx_lens": [8192],
  "decode_batch_sizes": [1, 8, 32, 128],
  "tp_sizes": [1, 4]
}
```

Produces an Excel file with one row per (Phase, SeqLen, CtxLen, BatchSize, TP, Op) combination. Columns include symbolic shapes (with `S`, `B`, `C`, `TP` kept as variables), concrete shapes with real per-tensor dtypes, memory, FLOPs, and arithmetic intensity.

**Grounded in a real profiling run.** The op set and real shapes come from an
actual profiling run: profile once at a typical config, and the
actually-dispatched ops (with their symbolic shapes and recorded per-tensor
dtypes) become a template that is re-resolved for every other
(seq/ctx/batch/TP) case, with memory/FLOPs recomputed per config. This makes the
matrix accurate to what the model really executes on XPU — the intended input
for downstream optimization work. The Shape Matrix tab handles this
automatically: it **reuses the latest completed run** for the model, or
**launches a fresh profile** (using the Profile tab settings) if none exists.
The exported workbook adds an **Info** sheet with the profiled config, caveats,
and a shape round-trip **validation** summary. Because the op set is fixed at
the profiled config, profile at **each TP** you need (sweeping TP only divides
`/TP` dims). Query, batch, and context (`S`/`B`/`C`) are **parametric** — you do
**not** profile per context: one base profile with any small non-zero context
captures `C` (as `S+C` on the attention KV rows) and every other context length
is derived. (A base profile with `context=0` leaves the KV rows as `S`, so
context can't be derived.) Memory/FLOPs remain analytic estimates (op+shape),
not measured.


## Replay Benchmark — what to optimize next

The Shape Matrix says *what runs at which shapes*; the replay benchmark says
**which kernel is worth a session**. It re-invokes **the ops vLLM actually
dispatched** — same kernel, same shapes and dtypes, rebuilt from the trace — and
ranks the ops by the end-to-end time an optimization would actually recover:

| signal | source | why it matters |
|---|---|---|
| calls x latency | Shape Matrix `Layers` at one operating point | a 10 us op in 57 layers beats a 500 us op that runs once |
| roofline headroom | analytic bytes/FLOPs vs the SKU peaks | an op already at >=80 % of peak is `at_roofline` - don't spend a session on it |
| which roof | the op's arithmetic intensity vs the machine balance | compute- or memory-bound is a property of the op, not of how the kernel did; a cache-resident op is measured against cache bandwidth, not DRAM |
| replay vs traced | the profile's own device time for the same op+shape | a replay far off the profiled time is not a valid baseline, and is flagged |

Because the benchmark *is* the dispatched op, coverage follows the profile:
there is no adapter table to extend and no external benchmark suite to install.

Use the **Benchmark & Targets** tab, or run it headless:

```bash
# one command: plan cases -> replay (one op per process) -> rank -> report
python -m breakdown.bench all --run m3 --trace output/traces/<trace>.json.gz \
    --model MiniMaxAI/MiniMax-M3 --tp 4 --batch-size 32 --context-len 2048 --xlsx

# or stage by stage
python -m breakdown.bench plan   --trace <trace> --model <id> --tp 4
python -m breakdown.bench run    --run <id> --budget 0.5
python -m breakdown.bench rank   --run <id> --target-util 0.8
python -m breakdown.bench report --run <id> --xlsx
python -m breakdown.bench case   --run <id> --case-id <id>     # re-run one shape
python -m breakdown.bench history --base <run> --new <run>
```

Attention, the KV-cache write and the sampler are benchmarked too. The first
two are context-bound only at the *dispatcher* level: one layer down the kernel
takes the paged KV cache, the block table and the sequence metadata as plain
arguments, so the benchmark rebuilds that context (own blocks per sequence,
`cu_seqlens_q` / `seqused_k` for the swept point) instead of refusing the
heaviest op in the model.

The plan reports up front what will **not** be measured and why — collectives
needing more ranks, dispatch wrappers with no context-free entry point (the
fused MoE forward), ops whose index operands have no synthesizer — so nothing
is silently omitted.

Output is `output/bench/<run_id>/targets.json`, the versioned handoff for the
`xpu-kernel-optimizer` skill: each target carries the kernel directory and
files, build/test commands, the baseline latency, the roofline bound, and a
ready-to-run `bench_cmd` / `profile_cmd` for the shapes that dominate. Every run
also records the git commit of each component that can move a number (kernel
repos, vLLM, this tool), and its cases are stored in
`output/bench/history.sqlite` so regressions are detectable across kernel bumps.

Needs only torch + vLLM on the machine (the same prerequisites as profiling);
each op is replayed in its own process, so a kernel that rejects a shape — or
takes the device down — costs only its own results.

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
  graph_from_trace.py   Profile-first graph reconstruction (from torch profiler trace)
  shape_derive.py       Symbolic shape resolution (shared by the export + benchmark)
  shape_matrix.py       Graph + config sweep -> matrix rows (the export serializes these)
  shape_matrix_xlsx.py  Excel serialization of those rows
  bench/                Replay benchmark: dispatched ops -> measured -> ranked targets
    spec.py               Matrix rows -> replay cases (skip/dedup rules)
    resolve.py            Dispatch name -> callable + schema (overload from the slots)
    inputs.py             Schema-driven operands + index-synthesizer registry
    recipes/              Per-op overrides, output args, skip reasons (xpu + cuda)
                          + attention.py: paged attention / KV write, context-free
    timing.py             Device-event windows, overhead subtraction, operand restore
    worker.py             Benchmark one op in its own process -> results.jsonl
    runner.py             Orchestration, per-op timeouts, incremental run_result.json
    collective.py         Multi-rank replay of c10d ops (rank 0 is recorded)
    estimate.py           Roofline (AI-based bound, DRAM/cache roof) + time budgets
    rank.py               calls x latency x roofline headroom -> targets.json
    reports.py            results.jsonl -> summary / coverage / workbook
    store.py              output/bench/<run_id>/ layout + run provenance
    history.py            SQLite history + regression detection
    cli.py                python -m breakdown.bench {plan,run,rank,report,case,history,all}
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
  test_bench_spec.py          Rows -> replay cases (skip/dedup, swept dims)
  test_bench_resolve.py       Dispatch resolution + operand materialization
  test_bench_rank.py          Ranking, timing plan, budgets, history
  test_bench_api.py           /api/bench/* endpoints
  test_bench_replay.py        End-to-end replay on a real device (requires GPU)
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
