# Copilot instructions — vllm-xpu-breakdown

**Read `AGENTS.md` first.** It is the authoritative index: project structure,
build/test commands, conventions, the API table, and the pitfall index
(symptom -> the function that holds the invariant -> the test that guards it).

Then read the deep dive for whatever you are changing:

- `breakdown/trace/README.md` — how a trace becomes a model graph.
- `breakdown/bench/README.md` — how those ops become measured, ranked targets.
- `breakdown/optimize/README.md` — how a ranked target becomes a kernel session.

## The shape of the thing

Four stages, each of which also runs headless:

1. **Profile** — run vLLM once on Intel XPU and reconstruct the module/op tree
   directly from the torch-profiler trace (`breakdown/trace/`).
2. **Sweep + replay** — turn those ops into a shape matrix and re-invoke the
   ops vLLM actually dispatched, at every swept point (`breakdown/bench/`).
3. **Rank** — `calls x latency x roofline headroom` -> `targets.json`.
4. **Optimize** — hand a ranked target to a Copilot CLI kernel session, one GPU
   each (`breakdown/optimize/`).

The web UI (`app.py` + `static/index.html`) is a wrapper over the same code the
CLIs call. `app.py` is routes; the work is in `breakdown/`.

## Rules of thumb

- **Reasons are the specification.** Nearly every non-obvious rule in this
  codebase exists because the naive version produced a specific wrong number.
  Those numbers are in the docstrings. Do not delete a docstring while changing
  the code it guards; update both.
- **Nothing is guessed.** An op whose operands cannot be rebuilt is *reported*
  with the reason, never filled with plausible data. A utilization above peak is
  flagged as a cost-model problem, not silently retired.
- **The golden fixtures are the safety net.** Any change to reconstruction must
  be reviewed as a diff of `tests/data/golden/`
  (`pytest tests/test_golden_graph.py --update-golden` to accept).
- **The canonical example is MiniMax-M3, TP=4, 6 layers, XPU.** It exercises
  hybrid dense/MoE, sparse attention, tensor-parallel collectives and
  Python-launched kernels; its traces are committed fixtures.

## Before you finish

```bash
pytest tests -q -p no:cacheprovider \
  --ignore=tests/test_real_profile.py \
  --ignore=tests/test_bench_replay.py \
  --ignore=tests/test_profile_reduced_layers.py
```

Then update the documentation that your change invalidates — see "Updating
documentation" at the end of `AGENTS.md`.
