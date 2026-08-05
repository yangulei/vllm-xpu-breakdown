# breakdown/bench — Replay Benchmark

## Replay, Not Re-Implementation

The profiler trace records, for every op vLLM dispatched, its exact dispatch
name, per-tensor shapes/dtypes/strides, and the concrete values of its
non-tensor arguments. So instead of translating each op into a substitute
kernel in an external benchmark suite, this package **re-invokes the op that
actually ran**: it resolves the dispatch name to its callable, materializes the
recorded operands, and times it on device.

Coverage is a property of the profile, not of a hand-maintained adapter table.
Every op the model dispatched is benchmarkable, on XPU and CUDA alike, without
writing an adapter for it.

What an op *name* means — is it a matmul, a collective, plumbing, an op that
indexes into a table rather than streaming it — is not decided here. That
vocabulary lives in `breakdown/core/opnames.py`, shared with reconstruction and
the cost model, so a new op is described once and every stage picks it up.


## Stages

```
reconstructed graph → shape_matrix rows → BenchCase specs → replay
    → results.jsonl → rank → targets.json / history
```

Every stage runs headless via `python -m breakdown.bench {plan,run,rank,report,
case,history,all}`; `/api/bench/*` and the web UI are thin wrappers.

- **`breakdown/shape_matrix.py`** — Sweeps the profiled graph's ops across
  `(phase, S, C, B, TP)` configurations. The rows are the transport, not the
  `.xlsx`: they carry the full ordered argument slots, which the spreadsheet
  cannot represent.
- **`spec.py`** — Turns matrix rows into `BenchCase` replay specs. De-duplicates
  cases whose operands do not depend on a swept dimension (recording every point
  it covers in `case.points`). Skips framework plumbing.
- **`worker.py` / `runner.py`** — Replays each op in its own process (a bad
  shape can wedge the device); results stream to `results.jsonl`. The runner
  launches each op separately and rewrites `run_result.json` after every op, so
  a run killed midway still says what completed.
- **`rank.py`** — `calls × latency × roofline headroom`. Produces
  `targets.json` (schema v5), the versioned contract consumed by the
  `xpu-kernel-optimizer` skill.
- **`history.py`** — SQLite ingestion of results + two-run regression detection.


## Honesty Rules

### Integer operands

An integer tensor is an index until proven otherwise. A random `slot_mapping`
makes a paged-KV kernel scatter across the whole cache; a random
`rows_per_expert` makes a grouped GEMM read past its input. The replay
**refuses** to fill an integer operand without a registered synthesizer
(`MissingSynthesizer`), and the case is reported as `needs_synthesizer` — never
randomly filled.

### Output arguments

Some arguments are outputs the schema does not mark.
`_moe_C::remap_hidden_states` takes `rows_per_expert` as an argument but
accumulates into it with atomics; reusing it across calls grows the offsets
until the scatter writes out of bounds (`UR_RESULT_ERROR_DEVICE_LOST`, no
traceback). The recipe declares such arguments so they are allocated zeroed,
reset between windows, and measured one call per timed window (`single_rep`).

### Context-bound wrappers

`vllm::unified_attention_with_output` and `vllm::unified_kv_cache_update` pull
the KV cache, block table and sequence metadata out of vLLM's forward context,
so the dispatcher op cannot be called standalone. One level down, the SYCL /
vllm_flash_attn kernels take that context as plain arguments. The recipe's
`entry` field points the replay there, and `build` reconstructs a paged KV
cache consistent with its block table and sequence lengths. Attention is
normally the heaviest op in the model; refusing it would leave the dominant
kernel unmeasured and un-rankable.

The MLA wrappers are different: their context also owns the selected backend,
projection weights, and metadata that are not wrapper operands. Until a trace
records a context-free MLA kernel boundary, they are reported as
`not_replayable` with that reason rather than left `unresolved` or replayed with
invented state.

### Recorded launcher imports

Python-launched kernels resolve from the exact file captured in their launcher
frame. Package files keep their real dotted module name so sibling imports such
as `from .chunk_intra import ...` work; standalone files fall back to a direct
location import. The resolved module must still point to the exact recorded
file, so an installed package from another checkout is never substituted.

### Fidelity

A case measured at the profiled shape carries the trace's own `device_time_us`.
A replay far faster than that means the arguments do not reproduce the model's
work, and the target is flagged rather than trusted. (The symptom before this
check: a kernel reading 15 µs instead of 1223 µs because its index operand
was zero-filled.)

### Cost-model credibility

Utilization above `RankConfig.max_credible_util` means the analytic
FLOPs/bytes overstate this op's traffic. It is reported as `check_cost_model`,
never silently retired as `at_roofline` — so the brief tells the agent to fix
the cost model first. It and `fidelity_floor` are `RankConfig` fields rather
than constants, because re-ranking a finished run with a different tolerance is
exactly what you do when deciding whether an op really is at the roof.


## Timing

Each case is measured with device events: repeats inside a device-event window,
empty-window overhead subtracted. The empty-window floor is ~60–90 µs on Level
Zero — an order of magnitude more than a small elementwise kernel — so measuring
one call per window would report the timer, not the kernel.

Operand restoration (for mutating ops) and cache flushing happen *between*
windows, never inside one.


## One Process Per Op

A replayed kernel runs with synthesized operands, so a shape it cannot handle
does not merely raise: it can abort the process or wedge the device so every
subsequent op fails with a device-lost error. The runner launches each op in
its own process; results stream case-by-case.


## Collectives

`c10d::allreduce_` and friends run on `TP` peer ranks, recording **rank 0**
(ranks 1..N-1 absorb the synchronization wait and report inflated times).

Two hard-won rules:

1. **Identical, fixed iteration schedule on every rank.** A per-rank adaptive
   probe desynchronizes the ranks and the transport runs out of resources.
2. **`SYCL_CACHE_PERSISTENT` disabled.** The persistent SYCL cache makes
   oneCCL segfault with no Python traceback.

Three more, learned from a run that died at its third op:

3. **The group is formed with `device_id`.** Without it the backend has to
   guess which GPU the rank owns (it says so: *"using GPU 0 as device used by
   this process is currently unknown ... can potentially cause a hang"*), and a
   later `barrier()` runs on whatever device the current context happens to be.
4. **A hang is retried, on a fresh port.** The XCCL transport on
   PCIe-connected Battlemage cards intermittently deadlocks *inside the device
   queue*: every rank enqueues its collectives and all of them then block in
   `torch.xpu.synchronize`. It reproduces in twenty lines of plain torch with
   none of this code involved, so it cannot be configured away — only
   survived. One unlucky attempt used to end the whole run, leaving the twenty
   ops planned after it unmeasured.
5. **Attempts are merged by case, not appended.** A hung attempt is rarely
   empty — rank 0 streams a record as each case finishes — so appending the
   retry's output recorded the measured cases twice and averaged the op's
   latency over duplicate rows.

Fewer devices than the profiled TP is reported as `needs_ranks`, never measured
on fewer. A launch that fails every attempt records each of its cases as
`failed` with the timeout as the reason: only rank 0 writes results, so a group
that never forms writes nothing, and an op with no records is indistinguishable
from an op that was never planned.


## The Recipe Table (`recipes/table.py`)

Everything the replay knows about a *specific* op lives in one `OpRecipe`
record (it used to be seven parallel dictionaries). A recipe is an
**exception** — the replay resolves and rebuilds on its own, and the majority of
ops need no entry. Fields:

| Field | Purpose |
|-------|---------|
| `entry` | The replay entry point is a *different* function (a context-free kernel under a context-bound wrapper). Checked **before** `skip`. |
| `build` | The entire argument list is built by hand (paged KV cache + block table). |
| `values` | Constants the synthesizers need and the trace does not record (expert count, block size). |
| `outputs` | Arguments the kernel writes although the schema does not say so. |
| `single_rep` | An output accumulated with atomics — one call per window. |
| `skip` | The op is real but must not be replayed, with *why*. |

Precedence: `entry` is resolved before `skip` is checked, so an op that has a
context-free entry point is replayed even if the wrapper itself is not
replayable.


## The Roofline (`breakdown/cost.py`)

The cost model lives in `breakdown/cost.py` (shared by the graph reconstruction
and the ranking) rather than inside this package:

- **Bound from arithmetic intensity vs machine balance** — an op is
  compute-bound iff its AI ≥ `peak FLOPS / peak bandwidth` (the ridge point,
  ~215 flop/byte on BMG). The old rule compared achieved utilizations and took
  the larger, which labelled a GEMM running at 30 % of peak FLOPS
  "memory-bound".
- **Named hardware unit** — `XMX` / `XVE` / `DRAM` / `L3-Cache`. A non-matrix
  op (norms, activations, gathers) is scored against the vector-engine peak
  (`XVE`, 8× lower on Xe2), not the XMX peak. Charging RMSNorm to the 98.3
  TFLOPS XMX peak made every elementwise kernel show ~99 % headroom.
- **Cache vs DRAM** — a kernel whose footprint fits L3 is measured against
  cache bandwidth (~1.2 TB/s on BMG). It is *also* scored against DRAM; the
  headroom decision uses the larger utilization. Without this, cache-resident
  streaming kernels showed headroom that the model can never use — the symptom
  was `rms_norm`-class ops proposed as kernel sessions.
