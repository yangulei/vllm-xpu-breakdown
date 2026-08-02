# breakdown/trace — Profile-First Graph Reconstruction

## What This Is

The model graph is **reconstructed from a torch profiler trace**, not derived
statically from a HuggingFace config. The trace is ground truth for what
actually executed, so the reconstruction tracks whatever vLLM and the backends
dispatched — there is no hardcoded structure to drift.

The alternative — a config-driven graph builder (`model_graph.py`) — was deleted.
It required a per-architecture builder, silently drifted as vLLM changed ops,
and could not know what quantization or compilation actually dispatched. Profile
first; derive shapes from what ran.


## Module Order

Each module has one concern and no imports from modules below it in the list:

- **`rules`** — Model- and backend-specific vocabulary: chrome-trace categories,
  plumbing ops, functional module frames, display-name tables. The only place
  names live; the passes stay name-free.
- **`events`** — Reading the trace file: JSON/gzip load, argument-slot parsing
  (`_parse_input_args`), worker-thread selection.
- **`forest`** — The time-containment tree of modules and ops (`_Raw` nodes),
  plus three structural passes: hoist, coalesce, roll up device time.
- **`kernels`** — Device time → leaf ops. Every device kernel is located at its
  host launch site and attributed to the deepest module/op containing it.
- **`shapes`** — Span-less fallbacks: shapes for ops the trace does not record
  (collective `TensorList`, Python-launched kernels in un-hooked traces).
- **`phases`** — Steps → prefill / decode classification by token dim.
- **`symbols`** — Concrete dims → symbolic expressions (`S`, `B`, `H`,
  `n_h/TP`, `topk·S`).
- **`collapse`** — Merge instances across forward passes, name children, and
  collapse structurally-identical repeated siblings into one node with
  `repeat_count`.
- **`graph`** — The orchestration entry point (`build_forest`,
  `build_graph_from_trace`).


## Pass Order in `graph.build_forest`

The pipeline is sequential and the order is load-bearing:

1. **`_build_raw_forest`** — Time-containment nesting from the flat event list.
   Module spans give real names; functional frames (`_functional_module_class`)
   promote MoE router/experts/fused blocks.
2. **`_collect_kernel_launches` + `_attribute_kernels`** — Each device kernel is
   linked to its launch site on the worker thread; `_attribute_kernels` assigns
   it to the deepest module/op enclosing that site. Must run **before** hoist,
   because hoisting moves module subtrees and would break containment.
3. **`_hoist_modules_under_ops`** — A module wrapped inside a fused custom op
   (e.g. `shared_experts` MLP inside `vllm::moe_forward_shared`) is lifted to
   its nearest module ancestor. Must run **after** kernels (launch-site needs
   the non-overlapping forest) and **before** `_compute_sub_dev` (or the
   kernel time rolls up under the wrapping op *and* the hoisted module).
4. **`_coalesce_duplicate_child_modules`** — The same module object recorded
   twice in one forward (an empty-shell entry + the real forward) is unioned
   into the earliest occurrence. Only instance-indexed events are eligible;
   synthetic functional frames (bare class labels) are skipped so
   genuinely-distinct same-class siblings stay distinct.
5. **`_compute_sub_dev`** — Post-order roll-up of `self_dev` → `sub_dev`. Must
   be last: any structural move after this would double-count.


## Capture-Time Spans

Two hook modules inject information the stock torch profiler cannot record:

### Module spans (`breakdown/module_hooks.py`)

A `register_forward_pre_hook`/`register_forward_hook` pair on every
`named_modules()` entry opens a `record_function("module::<qname>::<Cls>")`
`user_annotation` span around each forward. These carry the **real attribute
path** (`model.layers.0.self_attn.q_norm`) and nest by wall time, so
`_build_raw_forest` reconstructs the tree with correct names directly — no
alignment, no registration-order assumption, correct even under async.

A trace without them (legacy, upload) falls back to the class-only
`nn.Module: <Cls>_<idx>` `python_function` events, which carry only the class
name and a per-class instance index.

### Kernel spans (`breakdown/kernel_hooks.py`)

A kernel launched straight from Python — a Triton `JITFunction`, a pybind11
extension entry point — emits no `cpu_op`, so the trace records neither its
operands nor the function that launched it. `kernel_hooks` patches the launch
path to open a `record_function("kernel::<base64(json)>")` span carrying the
launcher's `{file, line, func}` and the full argument slots in the same schema
as `_parse_input_args`.

**Why base64?** Torch's chrome-trace writer emits an event's name *unescaped*:
a quote in the label produces a trace file that is not valid JSON. Base64
avoids this without modifying torch.

Without kernel spans, `shapes.py` falls back to inferring operands from
neighbouring ops and the model config — a guess that can silently rot when a
kernel's signature changes.


## Symbol Resolution

`symbols.py` resolves every concrete integer dimension to a symbolic expression
so the shape is useful across a sweep. There is **one** ordered resolution,
applied per dim:

1. **Token** — the pass's own token count (`S` prefill, `B` decode), matched on
   the axis that carries it. Must be first: on MiniMax-M3 at context=2048,
   `hidden_size` == 2048 == the token dim, so letting the constant pass claim it
   first would make `S` invisible — and the Shape Matrix would sweep hidden_size
   with the context.

2. **Derived expression** — a dimension that scales with a swept variable:
   `topk·S` (MoE routed rows = `tokens × num_experts_per_tok`). An expression,
   not a frozen value, because the value moves with the sweep. Without this, the
   MoE grouped GEMM's `M` froze at its profiled value and the kernels rejected
   their own shapes as soon as `S` moved (`"ptr_A.size(1) must match
   ptr_B.size(1)"`).

3. **Scoped constant** — a config constant that applies only at a specific
   `(op_name, axis, ndim)`. Needed because a value can mean two things at once:
   MiniMax-M3's index-head count equals `num_kv_heads` (so a plain value lookup
   renders it `n_kv/TP`), and its top-k block count equals `n_h/TP` at TP=4 (so
   it would be labelled `n_h/TP` and then *swept with TP*). Scoped constants
   break the collision by restricting a symbol to the ops where it is
   semantically correct.

4. **Constant** — a config constant or its `/TP` shard, looked up from the
   value → symbol table.

5. **Allocation** — what remains is a run-specific allocation size (a paged
   KV-cache slot count, an MoE scratch buffer). It gets an observed-value
   symbol (`N_kv`, `M_moe`) so nothing structural is left as a bare integer.


## Phase Classification

`phases._classify_steps` labels each inference step:

- **Prefill** — the forward processes more than one token per running sequence
  (`token_dim > batch_size`). Using `max(token_dim)` = prefill failed the
  two-pass decode run: with `query_len=1`, vLLM prefills each sequence's single
  new token individually (a 1-row op) while decode runs the whole batch, so the
  decode steps had *more* rows than the prefill micro-steps.
- **Decode** — token_dim ≤ batch_size.

### Decode-step filters

Only steady-state, full-batch decode steps survive:

1. **Full-batch** — only steps whose token dim equals the *maximum observed
   decode batch*. vLLM ramps up the running batch in waves (2 → 4 → 28 → 30 →
   32), and partial-batch steps carry partial row counts that would not
   symbolize to `B`. The symptom was duplicated near-`B` nodes (`28`/`30`).
2. **First step dropped** — the initial full-batch step pays one-time costs
   (KV/allocator warmup, oneDNN/Triton plan + autotune caching) that would skew
   the per-op latency average.

Both filters are guarded so the decode phase is never emptied.
