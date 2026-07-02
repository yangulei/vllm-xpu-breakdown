#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
vLLM Ops/Kernels Breakdown — main entry point.

Profiles a vLLM inference run on XPU or CUDA and generates dispatch
breakdown reports showing which ops go to vllm-kernels, torch-ops, or Triton.

Usage:
    python run_profile.py --model <model> [--max-model-len N] [--output-dir DIR]

All standard vLLM EngineArgs are supported.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure breakdown package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile vLLM inference on XPU/CUDA and generate ops breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Profile options
    profile_group = parser.add_argument_group("Profiling options")
    profile_group.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for output reports (default: output/)",
    )
    profile_group.add_argument(
        "--warmup-steps", type=int, default=1,
        help="Number of warmup iterations before profiling (default: 1)",
    )
    profile_group.add_argument(
        "--active-steps", type=int, default=1,
        help="Number of iterations to profile (default: 1)",
    )
    profile_group.add_argument(
        "--top-n", type=int, default=30,
        help="Number of top ops to show in report (default: 30)",
    )
    profile_group.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML report generation",
    )
    profile_group.add_argument(
        "--no-trace", action="store_true",
        help="Skip Chrome trace export",
    )
    profile_group.add_argument(
        "--profile-memory", action="store_true",
        help="Enable memory profiling (adds overhead)",
    )

    # Prompt options
    prompt_group = parser.add_argument_group("Prompt options")
    prompt_group.add_argument(
        "--prompt", type=str,
        default="Write a short essay about the importance of higher education.",
        help="User prompt for inference",
    )
    prompt_group.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens to generate (default: 256)",
    )
    prompt_group.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of concurrent requests (default: 1)",
    )
    prompt_group.add_argument(
        "--prefill-batch-size", type=int, default=None,
        help="Concurrent sequences for the prefill phase (real serving is "
             "usually 1). If it differs from --decode-batch-size, two profiled "
             "passes are run and written to <output>/prefill and "
             "<output>/decode. Defaults to --batch-size.",
    )
    prompt_group.add_argument(
        "--decode-batch-size", type=int, default=None,
        help="Concurrent sequences for the decode phase (often 32/64/128). "
             "See --prefill-batch-size. Defaults to --batch-size.",
    )

    return parser


def main():
    # Parse known args, pass the rest to vLLM EngineArgs
    parser = create_parser()
    args, vllm_args = parser.parse_known_args()

    # Late imports — vLLM is heavy
    print("[breakdown] Importing vLLM...")
    from vllm import LLM, EngineArgs, SamplingParams
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from breakdown.profiler import ProfileConfig, parse_events, profile_context
    from breakdown.report import export_csv, export_json, print_summary
    from breakdown.visualize import generate_html

    # Parse vLLM engine args
    engine_parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(engine_parser)
    engine_args_parsed = vars(engine_parser.parse_args(vllm_args))

    os.makedirs(args.output_dir, exist_ok=True)

    # Build the LLM instance
    print(f"[breakdown] Loading model: {engine_args_parsed.get('model', 'default')}...")
    llm = LLM(**engine_args_parsed)

    # Prepare the shared conversation template.
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    sampling_params = SamplingParams(max_tokens=args.max_tokens)

    def _profile_batch(batch_size: int, output_dir: str):
        """Profile one run at ``batch_size`` and write reports to ``output_dir``."""
        config = ProfileConfig(
            output_dir=output_dir,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            top_n=args.top_n,
            profile_memory=args.profile_memory,
        )
        os.makedirs(config.output_dir, exist_ok=True)
        conversations = [conversation] * batch_size

        total_steps = config.warmup_steps + config.active_steps
        print(f"[breakdown] Running {total_steps} steps at batch={batch_size} "
              f"({config.warmup_steps} warmup + {config.active_steps} profiled)...")

        with profile_context(config) as prof:
            for step in range(total_steps):
                phase = "warmup" if step < config.warmup_steps else "profile"
                print(f"[breakdown] Step {step + 1}/{total_steps} ({phase})")
                llm.chat(conversations, sampling_params, use_tqdm=False)
                prof.step()

        print("[breakdown] Profiling complete. Generating reports...")
        result = parse_events(prof, config)

        report_text = print_summary(result, top_n=config.top_n)
        report_path = os.path.join(config.output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        csv_path = export_csv(result, config.output_dir)
        json_path = export_json(result, config.output_dir)
        print(f"[breakdown] CSV:  {csv_path}")
        print(f"[breakdown] JSON: {json_path}")

        if not args.no_html:
            html_path = generate_html(result, config.output_dir)
            print(f"[breakdown] HTML: {html_path}")

        if not args.no_trace:
            trace_path = os.path.join(config.output_dir, "trace.json")
            if os.path.exists(trace_path):
                print(f"[breakdown] Chrome trace: {trace_path}")

        print(f"[breakdown] Reports written to {config.output_dir}/")

    # Resolve per-phase batch sizes. When they differ, real serving decouples
    # the phases (prefill ~1 sequence, decode many), so profile two passes into
    # separate report directories; otherwise a single run at --batch-size.
    pf = args.prefill_batch_size or args.batch_size
    dc = args.decode_batch_size or args.batch_size

    if pf != dc:
        print(f"[breakdown] Two-pass profiling: prefill batch={pf}, "
              f"decode batch={dc}")
        _profile_batch(pf, os.path.join(args.output_dir, "prefill"))
        _profile_batch(dc, os.path.join(args.output_dir, "decode"))
    else:
        _profile_batch(pf, args.output_dir)

    print(f"[breakdown] All reports written to {args.output_dir}/")


if __name__ == "__main__":
    main()
