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

## Reading Order

- **`breakdown/trace/README.md`** — how a trace becomes a model graph.
- **`breakdown/bench/README.md`** — how those ops become measured, ranked targets.
- **`breakdown/optimize/README.md`** — how a ranked target becomes a kernel session.
- **`AGENTS.md`** — structure, conventions, endpoints, and the pitfall index.

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
  `record_function("module::<path>::<Cls>")` spans at capture time. A second
  set of hooks records the *operands* of kernels launched straight from Python
  (Triton, pybind11 extensions), which leave no `cpu_op` and would otherwise
  have no shapes at all. A trace captured without either falls back to
  class-only module names and config-derived shapes. Because the tree is
  derived from what actually executed, it tracks whatever vLLM and the backends
  dispatched and does not drift as vLLM evolves. Requires a working Intel XPU
  with torch-xpu and vLLM installed.
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
block selection + block-sparse attend, the actual XPU dispatch) all reconstruct
from a profiling run, and every one of its ops replays in the benchmark.

## Web UI (Recommended)

Interactive web application for exploring the breakdown:

```bash
pip install -r requirements.txt
python app.py [--port 8080]
```

Then open `http://localhost:8080` in your browser.

**Features:**
- Three tabs: **Model Graph** (profile + reconstructed graph), **Bench & Rank**
  (one sweep form and one button: shapes → replay → ranked targets) and
  **Optimize Kernels** (manage the Copilot CLI kernel sessions opened with
  🚀 optimize from the ranked table — one GPU per session). A clicked ranked
  op expands in place with every case measured for it.
- The model, quantization and **device selection** are set once on *Model
  Graph* and used by both tabs (*Bench & Rank* has its own Devices field for
  when the replay should run elsewhere; blank there inherits Model Graph's). Devices are the comma-separated indexes of the
  accelerators this host actually has (blank = all of them); a selection naming
  a device that does not exist — or fewer devices than the TP size needs — is
  refused before anything starts, by the browser and again by the API
  (`GET /api/devices` lists what is present)
- Search for any HuggingFace model by ID
- Auto-loads model config (architecture, layers, MoE, dtype)
- Toggle between eager and torch.compile mode
- Reconstructed model graph (from a profiling run) with symbolic shapes and TP-aware annotations
- Quantization support (fp8, gptq, awq) — affects weight dtype and memory estimates
- Shape Matrix: sweep across seq_len, batch_size, context_len, and TP
  configurations — the benchmark's input, shipped as a sheet of its report (or
  downloadable on its own)
- Rich ops table with:
  - Shapes with symbolic dimensions (B=batch, S=seq_len, H=hidden_size, etc.)
  - Per-tensor dtype tags (bf16, fp8, int4)
  - TP-divided dimensions shown as `symbol/TP` (e.g., `n_h/TP`, `QKV/TP`)
  - Merged duplicate layers with ×N repetition count
  - Memory estimation, FLOPs, and arithmetic intensity
  - Sortable and filterable by backend
- Backend distribution chart

## Profiling

All profiling is done through the web UI. Start the server and use the browser
interface to configure model, batch sizes, TP, query/context lengths, and launch
a profile:

```bash
python app.py --port 8080
```

The canonical validation fixture is **MiniMax-M3, TP=4, 6 layers, XPU** (prefill:
batch 1, query 2048, context 2048; decode: batch 32). To re-capture it:

```bash
python tools/capture_fixture.py   # requires Intel XPU + vLLM
```

## Output

Profiling results are served through the web UI and the REST API. Downloadable
artifacts include the raw Chrome trace (`/api/profile/trace`), the Shape Matrix
Excel export (`/api/export/shape-matrix`), and the benchmark report workbook
(`/api/bench/report`).

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
for downstream optimization work. The sweep card at the top of the
**Bench & Rank** tab handles this automatically: it **reuses the latest
completed run** for the model, or **launches a fresh profile** (using the Model
Graph tab settings, including its quantization and device selection) if none
exists. The same sweep drives the replay benchmark's
cases, which is why the two share one tab.
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

The Shape Matrix and the replay benchmark are **one pipeline**: the matrix says
*what runs at which shapes*, and those exact rows become the benchmark's replay
cases, so the benchmark says **which kernel is worth a session**. It re-invokes
**the ops vLLM actually dispatched** — same kernel, same shapes and dtypes,
rebuilt from the trace — and
ranks the ops by the end-to-end time an optimization would actually recover:

| signal | source | why it matters |
|---|---|---|
| calls x latency | Shape Matrix `Layers` at one operating point | a 10 us op in 57 layers beats a 500 us op that runs once |
| roofline headroom | analytic bytes/FLOPs vs the SKU peaks | an op already at >=80 % of peak is `at_roofline` - don't spend a session on it |
| which roof | the op's arithmetic intensity vs the machine balance | compute- or memory-bound is a property of the op, not of how the kernel did; the roof is named as a hardware unit (`XMX` / `XVE` / `DRAM` / `L3-Cache`), a non-matrix op is scored against the vector-engine peak, and a cache-resident op is measured against cache bandwidth *and* DRAM (the headroom uses the larger) |
| per phase | prefill and decode ranked separately, as in the model graph | the same kernel is a compute-bound GEMM at prefill and a memory-bound GEMV at decode |
| replay vs traced | the profile's own device time for the same op+shape | a replay far off the profiled time is not a valid baseline, and is flagged |

Because the benchmark *is* the dispatched op, coverage follows the profile:
there is no adapter table to extend and no external benchmark suite to install.

### Using it from the web UI

The **Bench & Rank** tab is one form and one button. *Bench Settings* is the
shape sweep — the configuration space the ops are replayed over — plus three
controls:

| control | what it does |
|---|---|
| **TP Sizes** | tensor-parallel sizes to sweep; the widest one is also how many devices the run needs |
| **Devices** | which devices to *replay* on — blank inherits the Model Graph selection (the placeholder shows what that is). The replay does not have to run where the model was profiled: a TP=4 profile can be replayed on one card, or a sweep given the idle half of the box |
| **Kernels** | an Excel-style checkbox list of **the kernels/dispatch names this profile actually ran**, ordered by device time (searchable; default = all). The set is a property of the profile, so there is nothing to type |
| **Target util** | the fraction of its roofline at which an op is reported as `at_roofline` (done) rather than as a target |

**▶ Bench & Rank** then runs the whole pipeline: sweep the profiled graph into
replay cases → replay them (one op per process) → rank the results. They were
three buttons that could only ever be pressed in that order, so pressing them
is not a decision worth exposing. `📥 Report` downloads the run's workbook —
which carries the run's Shape Matrix as one of its sheets, so there is no
second download to choose between. A clicked target row expands under the
table with its measured cases. (The shape matrix on its own is still available headless via
`POST /api/export/shape-matrix`; it needs a profile but no benchmark.)

There is **no "budget / case" knob**: how long a case needs to be measured is a
property of the kernel being replayed, and the profile already knows it. The
runner sizes the per-op budget from the profiled shapes — the trace's own
device time for a case whose shapes match the recorded ones, the analytic
roofline cost otherwise — so a millisecond-scale GEMM gets a longer measurement
window than a microsecond-scale elementwise kernel, and each op's worker
timeout follows its own budget (see `breakdown/bench/estimate.py`:
`case_budget` / `op_budgets`).

Or run it headless:

```bash
# one command: plan cases -> replay (one op per process) -> rank -> report
python -m breakdown.bench all --run m3 --trace output/traces/<trace>.json.gz \
    --model MiniMaxAI/MiniMax-M3 --tp 4 --batch-size 32 --context-len 2048 --xlsx

# or stage by stage
python -m breakdown.bench plan   --trace <trace> --model <id> --tp 4
python -m breakdown.bench run    --run <id>            # budget derived per op
python -m breakdown.bench run    --run <id> --budget 0.5   # or pin it
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
ready-to-run `bench_cmd` / `profile_cmd` for the shapes that dominate. The
document holds a combined ranking plus a per-phase one (`by_phase.prefill` /
`by_phase.decode`), and the report workbook is the run's **single deliverable**:
`Info`, `Summary`, per-phase `Targets`, **one sheet per op**, `Coverage`, and
the run's own **`Shape Matrix`** — the sweep the cases were built from, because
a measured latency is only interpretable against the shape it was taken at. The
ranking is done at
the **profiled** operating point (the sweep point whose shapes are the ones the
trace recorded) whenever it was benchmarked, so the numbers are comparable to
the profile. In the web UI the Ranked-targets table shows one phase at a time
(Prefill / Decode), sorts on any column header, and expands a clicked op
**in place, under the table**, with every case measured for it **in that
phase** — the toggle moves the panel too, and 🚀 optimize sits at its foot. Every run
also records the git commit of each component that can move a number (kernel
repos, vLLM, this tool), and its cases are stored in
`output/bench/history.sqlite` so regressions are detectable across kernel bumps.

Needs only torch + vLLM on the machine (the same prerequisites as profiling);
each op is replayed in its own process, so a kernel that rejects a shape — or
takes the device down — costs only its own results.

## Optimize Kernels — open a session on what the ranking picked

The ranking says *which* kernel is worth an optimization session; this stage
opens it. Hit **🚀 optimize** on a ranked row — or in that op's case tile —
on **Bench & Rank**: the ranked table *is* the selection, so there is no second
list to pick from. The op gets a headless [Copilot CLI](https://github.com/github/copilot-cli) session
driven by the `xpu-kernel-optimizer` skill, briefed entirely from `targets.json`:
the backend and phase, the measured baseline latency at the profiled shape, the
roofline it is judged against, the kernel's repo/files/`build_cmd`/`test_cmd`,
and the exact `bench_cmd` / `profile_cmd` that reproduce and profile it. The
tool contributes no optimization strategy of its own — the skill owns the
Profile → Analyze → Optimize → Validate loop, the benchmark owns the numbers.

**One session owns one GPU, exclusively.** A session profiles and benchmarks
continuously, so two agents sharing a device would measure each other's
interference. Each session leases a single device for its whole lifetime
(enforced with `ZE_AFFINITY_MASK` / `CUDA_VISIBLE_DEVICES`, so the agent's
builds, benchmarks and `unitrace` runs all inherit it) and releases it on exit.
Open more sessions than you have devices and the surplus waits in a FIFO queue.
The **Optimize Kernels** tab manages them: each one's state, the GPU it holds,
its queue position, a stop button, and its streamed log (the log follows the tail only while you are at
the tail, so scrolling up to read something is not undone by new output).

**What it refuses to launch**, mirroring the benchmark's honesty rules: an op
that is `at_roofline` (no headroom left), one flagged `check_cost_model` (its
utilization is above peak, so the headroom cannot be trusted), and one with no
editable kernel source (ATen, collectives). 🚀 optimize on such an op states
the reason and asks before spending a GPU on it.

Sessions run from the **workspace root** (the parent of this repo, where the
kernel repos live — the paths in `bench/kernel_sources.json` are relative to
it) and write to `output/optimize/<run_id>/<op>/`: `prompt.md` (the brief),
`command.txt` (the same session as a pasteable shell line), `session.log`,
`session.json` and the agent's own `summary.md`. Spawning is a convenience, not
a requirement — if the Copilot CLI is not installed, 🚀 optimize prints the
brief and the command to run the same session in your own terminal.

Or run it headless:

```bash
# what the ranking picked, and whether each is worth a session
python -m breakdown.optimize candidates --run <run_id> --phase prefill

# the brief for one kernel (stdout), without starting anything
python -m breakdown.optimize prompt --run <run_id> --op _C::rms_norm

# open sessions; --devices bounds how many run at once (one each)
python -m breakdown.optimize start --run <run_id> --ops _C::rms_norm _moe_C::moe_gather \
    --devices 0,1 --wait

python -m breakdown.optimize status --run <run_id>
python -m breakdown.optimize stop   --run <run_id>
```

## Architecture

```
app.py                  Web server (Flask) — routes only
static/index.html       Interactive frontend (SPA, vanilla JS)
breakdown/
  profiling.py          Run vLLM once and turn the traces into a result
  service.py            What a route does between request and response
  runs.py               output/<stage>/<run_id>/state.json — a run survives a restart
  cost.py               The one cost model: bytes, FLOPs, AI, and the roofline
  trace/                Profile-first graph reconstruction — see its README.md
    rules.py              Model/backend vocabulary — the only place names live
    events.py             Reading the trace file
    forest.py             The time-containment tree of modules and ops
    kernels.py            Device time -> leaf ops
    shapes.py             Span-less shape fallbacks
    phases.py             Steps -> prefill / decode
    symbols.py            Concrete dims -> symbolic expressions (one resolution)
    collapse.py           Merge instances, name children, collapse repeats
    graph.py              The orchestration
  module_hooks.py       Capture-time module-name spans (real attribute paths)
  kernel_hooks.py       Capture-time kernel-launch spans (operands of Python-launched kernels)
  op_breakdown.py       Flat op breakdown derived from the graph
  shape_derive.py       Symbolic shape resolution (shared by the export + benchmark)
  shape_matrix.py       Graph + config sweep -> matrix rows
  shape_matrix_xlsx.py  Excel serialization of those rows
  bench/                Replay benchmark — see its README.md
    spec.py               Matrix rows -> replay cases (skip/dedup rules)
    types.py              The data contracts and the one record builder
    resolve.py            Dispatch name -> callable + schema (launch-frame resolution)
    inputs.py             Schema-driven operands + the index-synthesizer registry
    recipes/table.py      The one per-op recipe table (entry/build/values/outputs/skip)
    recipes/              Per-op recipes (common, xpu, cuda, attention)
    timing.py             Device-event windows, overhead subtraction, operand restore
    worker.py             Benchmark one op in its own process -> results.jsonl
    runner.py             Orchestration, per-op timeouts, incremental run_result.json
    collective.py         Multi-rank replay of c10d ops (rank 0 is recorded)
    estimate.py           Per-op time budgets (the roofline lives in cost.py)
    rank.py               calls x latency x roofline headroom -> targets.json
    reports.py            results.jsonl -> summary / coverage / workbook
    store.py              output/bench/<run_id>/ layout + run provenance
    history.py            SQLite history + regression detection
    cli.py                python -m breakdown.bench {plan,run,rank,report,case,history,all}
  optimize/             Ranked target -> Copilot CLI kernel session — see its README.md
    prompt.py             Target record -> the brief + the refusal rules
    session.py            Session record, argv, output/optimize/<run_id>/ layout
    scheduler.py          The GPU pool: exclusive leases, FIFO queue for the surplus
    manager.py            Spawn/track/stop the per-kernel copilot processes
    cli.py                python -m breakdown.optimize {candidates,prompt,start,status,stop}
  model_info.py         HuggingFace model config fetching & summarization
  profiler.py           torch.profiler wrapper (XPU activity, shapes, stacks)
  classifier.py         Op classification: vllm-xpu-kernels | triton | torch-xpu-ops | ...
  registry.py           Known op list from vllm-xpu-kernels
  trace_common.py       Torch-free trace helpers (span labels, overhead filter, device/role)
tools/
  make_fixture.py       Trim a full trace to a committable fixture (refuses if the graph differs)
  capture_fixture.py    Re-capture the canonical MiniMax-M3 TP4 6-layer profile
tests/
  conftest.py                 Repo root on sys.path + --update-golden flag
  helpers.py                  graph_of / find_op / iter_ops / device_time
  data/                       Golden fixtures (trimmed rank-0 traces + configs + snapshots)
  test_pipeline.py            Unit tests (incl. graph reconstruction; requires torch)
  test_golden_graph.py        Golden snapshots over the MiniMax-M3 TP4 6-layer fixtures
  test_module_spans.py        Capture-time module spans + reconstruction
  test_kernel_spans.py        Capture-time kernel-launch spans + reconstruction
  test_shape_matrix_export.py Shape Matrix export endpoint
  test_bench_spec.py          Rows -> replay cases (skip/dedup, swept dims)
  test_bench_resolve.py       Dispatch resolution + operand materialization
  test_bench_rank.py          Ranking, timing plan, budgets, history
  test_bench_api.py           /api/bench/* endpoints
  test_bench_replay.py        End-to-end replay on a real device (requires GPU)
  test_optimize_*.py          The brief, the GPU leases, /api/optimize/*
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
