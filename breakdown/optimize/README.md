# breakdown/optimize — Ranked Target → Kernel Session

## The Fourth Stage

The benchmark answers *which kernel is worth an optimization session*; this
package opens it. A ranked target becomes a markdown brief handed to a headless
Copilot CLI session (`copilot -p <brief> --allow-all-tools --allow-all-paths`)
running the `xpu-kernel-optimizer` skill.

This module contributes **no** optimization strategy. Every fact in the brief
comes from `targets.json` (baseline latency, roofline bound/unit/utilization,
kernel repo/dir/files, `build_cmd`/`test_cmd`/`bench_cmd`/`profile_cmd`). The
skill owns the Profile → Analyze → Optimize → Validate loop; the ranker owns
the numbers.


## One Session Owns One GPU

An optimization session profiles and benchmarks continuously. Two agents
sharing a device would measure each other's interference and could accept a
change on a false number — the same reason the replay benchmark runs one op per
process.

`scheduler.DevicePool` enforces the rule:

- A session is admitted only when a free device index exists; it holds the lease
  for its whole lifetime and releases it on exit.
- The surplus waits in FIFO order; a released device starts the next queued
  session.
- Concurrency is the pool size. There is deliberately **no** `max_parallel`
  knob — every available GPU should be working.
- The lease is enforced in the child's environment via `ZE_AFFINITY_MASK` /
  `CUDA_VISIBLE_DEVICES`, so builds, `bench_cmd` and `unitrace` runs all
  inherit a single-device view.
- A collective target (`ccl` / `c10d::`) leases `tp` devices at once, or waits;
  a request the pool could *never* satisfy raises `LeaseError` immediately
  rather than queuing forever.
- **Every exit path must release the lease** (normal, failed, stopped, spawn
  error, shutdown). A leaked lease strands a GPU for the rest of the server's
  life — the symptom is "the queue is stuck" with nothing running.


## Refusals (`launchability`)

`prompt._REFUSALS` is an ordered predicate table — first match wins. A GPU is
exclusive for the session's whole life, so opening one on an op with nothing to
win costs a card:

| Predicate | Reason |
|-----------|--------|
| `action == "at_roofline"` | The ranker found no structural headroom left — there is nothing for a session to win. |
| `action == "check_cost_model"` | Measured utilization is above peak, so headroom cannot be trusted — fix the cost model first. |
| No `kernel_dir` | No editable kernel source is registered (add one to `bench/kernel_sources.json`). |
| No `build_cmd` | The backend has no build command — its kernel is not editable here. |

Launching a refused op explicitly is still possible (the CLI/API allow it); the
brief then states the premise the agent must verify first.


## Artifacts

```
output/optimize/<run_id>/<op>/
    prompt.md       — the brief
    command.txt     — pasteable shell invocation (reads brief from prompt.md)
    session.log     — truncated at spawn; the streaming pane reads this
    session.json    — state record, rewritten on every transition
    summary.md      — written by the agent on completion
```

Design decisions:

- **Artifact root is pinned at creation** (`OptimizeSession.root`). The reaper
  thread writes state after the agent exits; re-reading `$BREAKDOWN_OPTIMIZE_-
  ROOT` at write time would send it wherever the environment points *then*.
- **A session restored from `index.json` is downgraded to `stopped`.** It
  belongs to a previous server process whose agents the `atexit` hook
  terminated. Reporting it as `running` makes the UI poll forever.
- **No endpoint returns `argv`.** It embeds the whole multi-KB brief; the
  response shape should not change size by two orders of magnitude.
- **The session log is opened `"wb"`.** Appending across runs made the streamed
  pane show the concatenation of every session ever opened for that op.


## Headless Parity

```bash
python -m breakdown.optimize candidates   # ranked ops + launchability
python -m breakdown.optimize prompt       # the brief + pasteable command
python -m breakdown.optimize start        # one session per selected kernel
python -m breakdown.optimize status       # state, leased device(s), queue
python -m breakdown.optimize stop         # stop a session / a run's sessions
```

The API and the web UI tab are wrappers; everything runs from the terminal.
