# vLLM-XPU Ops/Kernels Breakdown

**Which kernel should I optimize next, and how much is there to win?**

Profile vLLM inference once on Intel XPU, replay the ops it actually
dispatched across the shapes you care about, rank them by the end-to-end time
an optimization would recover, and hand the winners to an AI agent.

```
 profile          sweep + replay          rank              optimize
 ───────          ──────────────          ────              ────────
 run vLLM  ──►  the ops it dispatched ──► calls x latency ──► one Copilot CLI
 once,          re-invoked at every       x roofline          session per
 reconstruct    swept (S,B,C,TP) point    headroom            kernel, one GPU
 the op tree                              -> targets.json     each
 from the trace
```

Each stage is usable on its own, from the web UI or headless.

## Install and run

```bash
pip install -r requirements.txt
python app.py --port 8080        # then open http://localhost:8080
```

Needs torch-xpu and vLLM, and an Intel XPU for the stages that touch hardware.
Reconstruction and the shape matrix work on any machine, from a saved trace.

The web UI accepts either a HuggingFace model ID or a local model directory
containing `config.json`, for example `/root/code/models/kimi-k3-xpu-text`.
Enter either value in **Model** and select **Load Config**; the same identifier
or path is passed to vLLM when profiling starts. Profiles use vLLM's text-only
mode, so local text-only copies of multimodal checkpoints do not need vision
processor assets. For recurrent linear-attention models such as Kimi-K3, the
UI selects Decode Batch 1 and profiling caps memory utilization at 0.5 to leave
headroom for the recurrent-state gather during XPU startup.

## The four stages

### 1. Profile — what actually runs

vLLM runs once under `torch.profiler`, and the model's module/op tree is
**reconstructed from the trace**. Nothing is inferred from a model definition,
so the tree tracks whatever vLLM and the backends really dispatched and does
not drift as vLLM evolves.

- Module nodes carry their **real attribute names** (`q_norm`, `self_attn`,
  `shared_experts`), because profiling installs forward hooks that emit
  `record_function("module::<path>::<Cls>")` spans at capture time.
- A second set of hooks records the **operands of kernels launched straight
  from Python** (Triton, pybind11 extensions), which leave no `cpu_op` and
  would otherwise have no shapes at all.
- Kernel device time is charged to the op that launched it, through the
  `correlation -> runtime -> External id` chain, and the total is conserved:
  no kernel's time is silently dropped.
- Shapes come out **symbolic** — `S` (query), `B` (batch), `C` (context),
  `TP`, plus model constants — so one profile parameterizes many operating
  points. `/TP` marks a tensor-parallel shard, and is present even at TP=1.

Profile at **each TP** you need. `S`, `B` and `C` are parametric from a single
profile (one small non-zero context is enough to capture `C`).

Traces can be downloaded and re-uploaded: the filename encodes the whole
configuration, so the round trip is lossless and a profile captured on the GPU
box can be analysed anywhere.

### 2. Sweep + replay — measure the ops, at your shapes

The reconstructed ops become a **shape matrix**: one row per
(phase, seq, context, batch, TP, op), with concrete shapes, per-tensor dtypes,
analytic bytes, FLOPs and arithmetic intensity.

Those exact rows become the benchmark's cases. The benchmark **re-invokes the
op vLLM dispatched** — same kernel, same shapes and dtypes, operands rebuilt
from the trace — so coverage follows the profile: there is no adapter table to
extend and no external benchmark suite to install.

Attention, the KV-cache write and the sampler are benchmarked too. They look
context-bound only at the dispatcher level; one layer down the kernel takes the
paged cache, the block table and the sequence metadata as plain arguments, so
the benchmark rebuilds that context rather than refusing the heaviest op in the
model.

Each op is replayed **in its own process**, so a kernel that rejects a shape —
or takes the device down — costs only its own results.

The plan reports up front what will **not** be measured and why: collectives
needing more ranks, dispatch wrappers with no context-free entry point, ops
whose index operands have no synthesizer. Nothing is silently omitted.

### 3. Rank — what is worth a session

| signal | why it matters |
|---|---|
| calls x latency | a 10 µs op in 57 layers beats a 500 µs op that runs once |
| roofline headroom | an op already at ≥80 % of peak is `at_roofline` — don't spend a session on it |
| which roof | compute- or memory-bound is a property of the op, not of the kernel. The roof is named as a hardware unit (`XMX` / `XVE` / `DRAM` / `L3-Cache`); a non-matrix op is scored against the vector engine, and a cache-resident op against cache bandwidth *and* DRAM |
| per phase | prefill and decode rank separately — the same kernel is a compute-bound GEMM at prefill and a memory-bound GEMV at decode |
| replay vs traced | a replay far off the profiled device time is not a valid baseline, and is flagged rather than trusted |

Output is `output/bench/<run_id>/targets.json`, plus a report workbook whose
sheets are `Info`, `Summary`, per-phase `Targets`, one sheet per op,
`Coverage`, and the run's own **Shape Matrix** — because a measured latency is
only interpretable against the shape it was taken at.

Every run records the git commit of each component that can move a number, and
its cases go into `output/bench/history.sqlite`, so regressions are detectable
across kernel bumps.

### 4. Optimize — open a session on what the ranking picked

Hit **🚀 optimize** on a ranked row. The op gets a headless
[Copilot CLI](https://github.com/github/copilot-cli) session driven by the
`xpu-kernel-optimizer` skill, briefed entirely from `targets.json`: the backend
and phase, the measured baseline at the profiled shape, the roofline it is
judged against, the kernel's repo/files/`build_cmd`/`test_cmd`, and the exact
`bench_cmd` / `profile_cmd` that reproduce and profile it.

This tool contributes **no optimization strategy of its own** — the skill owns
the Profile → Analyze → Optimize → Validate loop, the benchmark owns the
numbers.

**One session owns one GPU, exclusively.** A session profiles and benchmarks
continuously, so two agents sharing a device would measure each other's
interference. Each leases one device for its lifetime (`ZE_AFFINITY_MASK` /
`CUDA_VISIBLE_DEVICES`, inherited by its builds, benchmarks and `unitrace`
runs) and releases it on exit; surplus sessions wait in a FIFO queue.

**What it refuses to launch**, mirroring the benchmark's honesty rules: an op
that is `at_roofline` (nothing to win), one flagged `check_cost_model` (its
utilization is above peak, so the headroom cannot be trusted), and one with no
editable kernel source (ATen, collectives). It states the reason and asks
before spending a GPU.

Spawning is a convenience, not a requirement: if the Copilot CLI is not
installed, you get the brief and a pasteable command.

## The web UI

Three tabs, each consuming the previous one's output.

- **Model Graph** — the reconstructed tree with symbolic shapes, per-tensor
  dtypes, TP-aware annotations, merged repeated layers (`×N`), memory, FLOPs,
  arithmetic intensity and the backend distribution. Model, quantization and
  device selection are set once here and used by the other tabs.
- **Bench & Rank** — one sweep form and one button. It reuses the latest
  matching profile or launches a fresh one, sweeps, replays, ranks. A clicked
  target expands in place with every case measured for it. `📥 Report`
  downloads the workbook.
- **Optimize Kernels** — each session's state, the GPU it holds, its queue
  position, a stop button, and its streamed log.

The page restores the latest profile, benchmark run and sessions on load, and
an empty tab states its precondition rather than being blank.

## Headless

```bash
# one command: plan cases -> replay (one op per process) -> rank -> report
python -m breakdown.bench all --run m3 --trace output/traces/<trace>.json.gz \
    --model MiniMaxAI/MiniMax-M3 --tp 4 --batch-size 32 --context-len 2048 --xlsx

# or stage by stage
python -m breakdown.bench plan   --trace <trace> --model <id> --tp 4
python -m breakdown.bench run    --run <id>              # budget derived per op
python -m breakdown.bench rank   --run <id> --target-util 0.8
python -m breakdown.bench report --run <id> --xlsx
python -m breakdown.bench case   --run <id> --case-id <id>   # re-run one shape
python -m breakdown.bench history --base <run> --new <run>
```

```bash
# what the ranking picked, and whether each is worth a session
python -m breakdown.optimize candidates --run <run_id> --phase prefill

# the brief for one kernel (stdout), starting nothing
python -m breakdown.optimize prompt --run <run_id> --op _C::rms_norm

# open sessions; --devices bounds how many run at once (one each)
python -m breakdown.optimize start --run <run_id> --ops _C::rms_norm \
    --devices 0,1 --wait
python -m breakdown.optimize status --run <run_id>
python -m breakdown.optimize stop   --run <run_id>
```

The shape matrix on its own is `POST /api/export/shape-matrix` — it needs a
profile but no benchmark. The full REST API is listed in `AGENTS.md`.

## Backends tracked

| Backend | What it covers |
|---|---|
| **vllm-xpu-kernels** | Custom SYCL/DPC++ kernels (RMSNorm, activations, attention, MoE, quantization, cache ops) |
| **flash_xpu** | xattention SYCL kernels (sparse-attention index + block-sparse attend) |
| **flashinfer** | FlashInfer kernels |
| **triton** | Triton-compiled kernels (attention backends, sampling, `torch.compile` output) |
| **torch-xpu-ops** | ATen operators on XPU (linear, matmul, embedding — oneDNN/oneMKL) |
| **ccl** | Tensor-parallel collectives (all-reduce, all-gather, reduce-scatter) |
| **cpu** / **framework** | CPU work; tensor plumbing and profiler overhead |

Classification asks, in order: is it a collective? one of vLLM's own kernels?
a known kernel library? plumbing? an ATen op with device time? The vocabulary
behind those questions lives in one place (`breakdown/core/opnames.py`).

## Supported architectures

Standard GQA decoders (Llama, Qwen, Mistral), MoE (Mixtral, Qwen-MoE), VL and
encoder/embedding models. **MLA** models (DeepSeek-V2/V3/V4, GLM-MoE-DSA) are
supported on XPU — dense MLA routes to `TRITON_MLA`, DeepSeek sparse attention
to `XPU_MLA_SPARSE`. **MiniMax-M3** (vision-language MoE with sparse attention)
is the canonical validation example: its nested `text_config`/`vision_config`
layout, per-layer dense/MoE split, shared experts and Triton lightning-indexer
sparse attention all reconstruct from a profile, and every one of its ops
replays in the benchmark.

Adding a model needs no code: ensure `model_info.summarize_config` extracts its
key dims, then profile it.

## Design rules

These are the rules the code is held to; the reasoning for each specific case
lives on the function that holds it.

- **Nothing is guessed.** An op whose operands cannot be rebuilt is *reported*
  with the reason, never filled with plausible data. A utilization above peak
  is flagged as a cost-model problem, not silently retired.
- **One name per concept, end to end.** Facts about op names, dtypes and
  symbolic dimensions live in `breakdown/core/` and are asked, not
  re-transcribed. The page renders what the server resolved; it does not
  re-derive it.
- **Reasons are the specification.** Nearly every non-obvious rule exists
  because the naive version produced a specific wrong number, and that number
  is in the docstring.
- **The golden fixtures are the safety net.** Two snapshots of the canonical
  profile — the graph's shape, and the judgements made about each op — turn any
  behaviour change into a reviewable diff.

## Contributing

Read `AGENTS.md` first: structure, conventions, the endpoint table, and the
pitfall index (symptom → the function that holds the invariant → the test that
guards it). Then the deep dive for whatever you are changing:
`breakdown/trace/README.md`, `breakdown/bench/README.md`,
`breakdown/optimize/README.md`.

```bash
pytest tests -q -p no:cacheprovider \
  --ignore=tests/test_real_profile.py \
  --ignore=tests/test_bench_replay.py \
  --ignore=tests/test_profile_reduced_layers.py
```
