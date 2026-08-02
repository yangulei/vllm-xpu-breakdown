# SPDX-License-Identifier: Apache-2.0
"""A ranked target -> the brief a Copilot CLI kernel session starts from.

Pure and torch-free, so it is the unit-testable core of the handoff. Every fact
in the brief comes from ``targets.json``; nothing here invents an optimization
strategy, a baseline or a command - the ``xpu-kernel-optimizer`` skill owns the
loop and the ranker owns the numbers.

It also owns the **refusal rules** (:func:`launchability`), which mirror the
benchmark's honesty rules: an op that is already at the roofline, whose
utilization is above peak (so its headroom cannot be trusted), or that has no
editable kernel source is *reported with the reason* rather than handed to an
agent that would burn a GPU rediscovering it.
"""
from __future__ import annotations

from typing import Any

#: The skill that owns the optimization loop. It lives in ``~/.copilot/skills``
#: and is discovered by the CLI itself, so only its name is plumbed.
OPTIMIZER_SKILL = "xpu-kernel-optimizer"

#: Why a ranked target may not be worth a GPU: ``(predicate, reason)``, in
#: order, first match wins. A table rather than a chain of ifs because these
#: are the *policy* - one GPU is exclusive to one session for its whole life,
#: so opening a session on an op with nothing to win costs a card - and a
#: policy should be readable in one place.
_REFUSALS: tuple[tuple[Any, str], ...] = (
    (lambda t, k: t.get("action") == "at_roofline",
     "already at the hardware roof - the ranker found no structural "
     "headroom left, so a kernel session has nothing to win"),
    (lambda t, k: t.get("action") == "check_cost_model",
     "measured utilization is above peak, so the analytic FLOPs/bytes "
     "overstate this op's traffic and its headroom cannot be trusted - "
     "fix the cost model before opening a session"),
    (lambda t, k: not k.get("kernel_dir") or k.get("kernel_dir") == "-",
     "no editable kernel source is registered for this op "
     "(add an entry to breakdown/bench/kernel_sources.json)"),
    (lambda t, k: not k.get("build_cmd"),
     "this backend has no build command - its kernel is not editable here "
     "(the fix is usually to dispatch elsewhere, not to edit it)"),
)


def _kernel(target: dict[str, Any]) -> dict[str, Any]:
    return target.get("kernel") or {}


def launchability(target: dict[str, Any]) -> tuple[bool, str]:
    """``(can_launch, reason)`` for one ranked target.

    ``reason`` is always populated: for a refusal it says why, for a launchable
    target it says what the session is expected to do.
    """
    kernel = _kernel(target)
    for refuses, reason in _REFUSALS:
        if refuses(target, kernel):
            return False, reason

    if target.get("action") == "tune_config":
        return True, (
            "library/collective op: expect a configuration or dispatch change "
            "rather than a kernel rewrite")
    return True, "kernel session: there is headroom against the roofline"


def targets_by_op(doc: dict[str, Any],
                  phase: str | None = None) -> dict[str, dict[str, Any]]:
    """Ranked targets keyed by dispatch name; the *phase's* record wins.

    A kernel is a compute-bound GEMM at prefill and a memory-bound GEMV at
    decode, so the phase's own record is what a session must be briefed with -
    the combined row belongs to neither phase and is only a fallback for an op
    the phase did not rank.
    """
    section = (doc.get("by_phase") or {}).get(phase or "", {}) or {}
    by_op = {t.get("op"): t for t in section.get("targets") or []}
    for t in doc.get("targets") or []:
        by_op.setdefault(t.get("op"), t)
    return by_op


def candidates(doc: dict[str, Any], phase: str | None = None) -> list[dict[str, Any]]:
    """The ranked ops of a phase, each with why it can (or cannot) be optimized.

    This is the selection list the UI offers: the ranking decides the order,
    :func:`launchability` decides what is worth a GPU.
    """
    section = (doc.get("by_phase") or {}).get(phase or "", {}) or {}
    targets = section.get("targets") or doc.get("targets") or []
    out: list[dict[str, Any]] = []
    for target in targets:
        can, reason = launchability(target)
        kernel = _kernel(target)
        out.append({
            "op": target.get("op"),
            "rank": target.get("rank"),
            "backend": target.get("backend"),
            "action": target.get("action"),
            "e2e_us": target.get("e2e_us"),
            "share_of_e2e": target.get("share_of_e2e"),
            "savings_us": (target.get("savings_us") or {}).get("total"),
            "flags": target.get("flags") or [],
            "kernel_dir": kernel.get("kernel_dir"),
            "language": kernel.get("language"),
            "launchable": can,
            "reason": reason,
        })
    return out


def _fmt_pct(value: Any) -> str:
    return f"{float(value) * 100:.0f}%" if isinstance(value, (int, float)) else "-"


def _fmt_num(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "-"


def _roofline_lines(target: dict[str, Any]) -> list[str]:
    rf = target.get("roofline") or {}
    unit = rf.get("unit") or rf.get("bound") or "?"
    lines = [
        f"- Bound: **{rf.get('bound', '?')}** on **{unit}** "
        f"(arithmetic intensity {_fmt_num(rf.get('ai'), 2)} flop/byte vs "
        f"machine balance {_fmt_num(rf.get('ridge_ai'), 2)})",
        f"- Utilization of that roof: **{_fmt_pct(rf.get('util'))}** "
        f"(same measurement against DRAM: {_fmt_pct(rf.get('util_dram'))}; "
        f"target {_fmt_pct(rf.get('target_util'))})",
    ]
    if rf.get("memory_level") == "cache":
        lines.append(
            f"- The replay's footprint fits the {unit} "
            f"({rf.get('cache_bytes', '?')} B at {rf.get('cache_bw_gbs', '?')} GB/s), "
            "so it is served from cache; DRAM is the roof the model actually sees.")
    lines.append(
        f"- Device peaks: {rf.get('peak_tflops', '?')} TFLOPS matrix / "
        f"{rf.get('vector_tflops', '?')} TFLOPS vector / "
        f"{rf.get('peak_bw_gbs', '?')} GB/s DRAM")
    return lines


def _shape_lines(target: dict[str, Any], phase: str | None) -> list[str]:
    shapes = target.get("top_shapes") or []
    if phase:
        ordered = ([s for s in shapes if s.get("phase") == phase]
                   + [s for s in shapes if s.get("phase") != phase])
    else:
        ordered = list(shapes)
    lines: list[str] = []
    for shape in ordered:
        marks = []
        if shape.get("profiled"):
            marks.append("profiled shape")
        ratio = shape.get("replay_vs_traced")
        if isinstance(ratio, (int, float)):
            marks.append(f"replay/traced {ratio:.2f}")
        lines.append(
            f"- **{shape.get('phase', '?')}** `{shape.get('shape', '?')}` - "
            f"baseline **{_fmt_num(shape.get('latency_us'), 3)} us**, "
            f"{shape.get('calls', '?')} calls, "
            f"weighted {_fmt_num(shape.get('weighted_us'), 1)} us"
            + (f" ({', '.join(marks)})" if marks else ""))
        if shape.get("bench_cmd"):
            lines.append(f"  - benchmark: `{shape['bench_cmd']}`")
        if shape.get("profile_cmd"):
            lines.append(f"  - profile: `{shape['profile_cmd']}`")
    return lines or ["- (no measured shapes recorded for this op)"]


def _baseline(target: dict[str, Any], phase: str | None) -> dict[str, Any] | None:
    shapes = target.get("top_shapes") or []
    for want_profiled in (True, False):
        for shape in shapes:
            if phase and shape.get("phase") != phase:
                continue
            if want_profiled and not shape.get("profiled"):
                continue
            return shape
    return shapes[0] if shapes else None


def build_prompt(target: dict[str, Any], doc: dict[str, Any], *,
                 run_id: str | None = None, phase: str | None = None,
                 device_ids: list[int] | None = None,
                 device_kind: str | None = None,
                 workspace_root: str | None = None,
                 artifact_dir: str | None = None) -> str:
    """The markdown brief for one target's optimization session."""
    op = target.get("op") or "?"
    kernel = _kernel(target)
    can, reason = launchability(target)
    run_id = run_id or doc.get("run_id") or "?"
    point = ((doc.get("by_phase") or {}).get(phase or "", {}) or {}
             ).get("operating_point") or (doc.get("operating_points") or {}
                                          ).get(phase or "", {})
    baseline = _baseline(target, phase)
    devices = device_ids or []
    kind = device_kind or ("xpu" if str(doc.get("device", "")).startswith("xpu")
                           else doc.get("device") or "xpu")

    lines: list[str] = [
        f"# Optimize `{op}` on {doc.get('sku') or doc.get('device') or 'this GPU'}",
        "",
        f"Use the **{OPTIMIZER_SKILL}** skill to run its "
        "Profile -> Analyze -> Optimize -> Validate loop on this kernel. "
        "Everything below comes from a replay benchmark of the op vLLM "
        "actually dispatched, so the baseline and the commands are real - do "
        "not substitute a different kernel or a synthetic benchmark.",
        "",
        "## Target",
        "",
        f"- Op (dispatch name): `{op}`",
        f"- Backend: `{target.get('backend') or '-'}`",
        f"- Ranked #{target.get('rank', '?')} for "
        f"{phase or 'prefill+decode'}; action `{target.get('action') or '?'}` "
        f"({reason})",
        f"- End-to-end contribution: {_fmt_num(target.get('e2e_us'), 1)} us "
        f"({_fmt_pct(target.get('share_of_e2e'))} of the model's measured op "
        f"time), {target.get('calls', '?')} calls per pass",
        f"- Estimated headroom if optimized to the roof: "
        f"{_fmt_num((target.get('savings_us') or {}).get('total'), 1)} us",
        f"- Benchmark run: `{run_id}`"
        + (f", tensor parallel {doc.get('tp')}" if doc.get("tp") else ""),
    ]
    if point:
        lines.append(
            f"- Operating point ({phase or 'ranked'}): "
            f"seq={point.get('seq_len')} ctx={point.get('ctx_len')} "
            f"batch={point.get('batch_size')}"
            + (" (the shape the model was profiled at)"
               if point.get("profiled") else ""))

    lines += ["", "## Roofline (measured by the replay benchmark)", ""]
    lines += _roofline_lines(target)

    flags = target.get("flags") or []
    if flags:
        lines += ["", "## Caveats recorded by the ranker", ""]
        lines += [f"- {f}" for f in flags]
        lines.append(
            "- Treat the baseline with care: these flags mean the replay may "
            "not reproduce exactly what the model does.")

    lines += ["", "## Kernel source", "",
              f"- Repo: `{kernel.get('repo') or '-'}`",
              f"- Kernel dir: `{kernel.get('kernel_dir') or '-'}`",
              f"- Language: `{kernel.get('language') or '-'}`",
              f"- Files: {', '.join(f'`{f}`' for f in kernel.get('files') or []) or '-'}",
              f"- Build: `{kernel.get('build_cmd') or '-'}`",
              f"- Test: `{kernel.get('test_cmd') or '-'}`"]
    if kernel.get("notes"):
        lines.append(f"- Notes: {kernel['notes']}")
    if workspace_root:
        lines.append(
            f"- Those paths are relative to the workspace root `{workspace_root}`, "
            "which is this session's working directory.")

    lines += ["", "## Measured shapes (the baseline to beat)", ""]
    lines += _shape_lines(target, phase)

    lines += ["", "## This session's device", ""]
    if devices:
        var = "ZE_AFFINITY_MASK" if kind == "xpu" else "CUDA_VISIBLE_DEVICES"
        ids = ", ".join(str(i) for i in devices)
        lines.append(
            f"- You own **{kind} device {ids}** exclusively; "
            f"`{var}` is already set for this session, so every build, "
            "benchmark and profile you launch sees only that device. "
            "Other kernels may be optimized concurrently on the other "
            "devices - do not change or unset that variable, and do not run "
            "anything on a device you were not given.")
    else:
        lines.append(
            "- No device restriction was applied to this session; assume the "
            "whole machine and re-measure if numbers look contended.")

    lines += [
        "", "## What to do", "",
        "1. Reproduce the baseline with the benchmark command above and "
        "confirm you get a latency close to the one recorded here. If you "
        "cannot, stop and report why - an unreproducible baseline makes every "
        "later comparison meaningless.",
        "2. Profile it (the `profile_cmd` above uses `unitrace`) and identify "
        "the bottleneck against the roofline stated above.",
        "3. Iterate the skill's loop: one hypothesis per trial, build with the "
        "build command, keep a change only if it beats the baseline and the "
        "test command still passes, revert it otherwise.",
        "4. Re-measure the final accepted kernel once more on its own before "
        "reporting, so the number is not a contended one.",
    ]
    if artifact_dir:
        lines.append(
            f"5. Write a short summary to `{artifact_dir}/summary.md`: the "
            "baseline, each accepted change with its measurement, the final "
            "latency and speedup, and what is left on the table.")
    else:
        lines.append(
            "5. Finish with a short summary: the baseline, each accepted "
            "change with its measurement, the final latency and speedup, and "
            "what is left on the table.")

    if baseline and baseline.get("latency_us") is not None:
        lines += [
            "",
            f"**Acceptance criterion:** beat "
            f"{_fmt_num(baseline.get('latency_us'), 3)} us on "
            f"`{baseline.get('shape', '?')}` ({baseline.get('phase', '?')}) "
            "with the op's tests still passing.",
        ]
    if not can:
        lines += [
            "",
            f"> NOTE: the ranker did not consider this op worth a session "
            f"({reason}). It was launched explicitly anyway - verify that "
            "premise first and stop early if it holds.",
        ]
    return "\n".join(lines) + "\n"
