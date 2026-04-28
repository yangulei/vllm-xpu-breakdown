#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a real profiling session with Qwen/Qwen3-4B-Instruct-2507 on XPU.

This script profiles the model using vLLM's native profiler (which runs inside
the worker process) and then parses the resulting trace file.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch
    from vllm import LLM, SamplingParams

    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    trace_dir = os.path.abspath("output/traces")
    os.makedirs(trace_dir, exist_ok=True)

    # Clean old traces
    for f in os.listdir(trace_dir):
        if f.endswith(".json") or f.endswith(".json.gz"):
            os.remove(os.path.join(trace_dir, f))

    print(f"[1/5] Creating LLM engine with profiler config...")
    llm = LLM(
        model=model_id,
        max_model_len=4096,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": trace_dir,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": True,
            "torch_profiler_use_gzip": False,
        },
    )

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    sampling_params = SamplingParams(max_tokens=64)

    print(f"[2/5] Warmup inference (no profiling)...")
    output = llm.chat([conversation], sampling_params, use_tqdm=False)
    print(f"  Warmup output: {output[0].outputs[0].text[:80]}...")

    print(f"[3/5] Starting profiler and running profiled inference...")
    llm.start_profile()
    output = llm.chat([conversation], sampling_params, use_tqdm=False)
    llm.stop_profile()
    print(f"  Profiled output: {output[0].outputs[0].text[:80]}...")

    # Wait a moment for trace to be written
    time.sleep(2)

    print(f"[4/5] Looking for trace files in {trace_dir}...")
    trace_files = []
    for f in os.listdir(trace_dir):
        full = os.path.join(trace_dir, f)
        if f.endswith(".json") or f.endswith(".json.gz"):
            size = os.path.getsize(full)
            print(f"  Found: {f} ({size / 1024:.1f} KB)")
            trace_files.append(full)

    if not trace_files:
        print("ERROR: No trace files found!")
        # Check if there are any files at all
        all_files = os.listdir(trace_dir)
        print(f"  Directory contents: {all_files}")
        sys.exit(1)

    # Parse the trace
    trace_files.sort(key=os.path.getmtime, reverse=True)
    trace_file = trace_files[0]

    print(f"[5/5] Parsing trace: {os.path.basename(trace_file)}...")

    # First, raw analysis of event categories
    import json
    import gzip as gz

    if trace_file.endswith(".gz"):
        with gz.open(trace_file, "rt") as f:
            trace = json.load(f)
    else:
        with open(trace_file) as f:
            trace = json.load(f)

    events = trace.get("traceEvents", [])
    print(f"  Total events: {len(events)}")

    cats = {}
    for e in events:
        cat = e.get("cat", "none")
        cats[cat] = cats.get(cat, 0) + 1
    print(f"  Categories: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")

    # Show sample cpu_op events
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]
    print(f"\n  CPU ops: {len(cpu_ops)}")
    if cpu_ops:
        # Unique names
        op_names = {}
        for e in cpu_ops:
            name = e.get("name", "?")
            dur = e.get("dur", 0)
            op_names[name] = op_names.get(name, 0) + dur
        print(f"  Unique op names: {len(op_names)}")
        for name, total_dur in sorted(op_names.items(), key=lambda x: -x[1])[:20]:
            print(f"    {name:45s}  total_dur={total_dur:>10.0f}µs")

    # Show kernel events
    kernel_events = [e for e in events if e.get("cat") in ("kernel", "xpu_op", "gpu_op")]
    print(f"\n  Kernel events: {len(kernel_events)}")
    if kernel_events:
        kernel_names = {}
        for e in kernel_events:
            name = e.get("name", "?")
            dur = e.get("dur", 0)
            kernel_names[name] = kernel_names.get(name, 0) + dur
        for name, total_dur in sorted(kernel_names.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name:60s}  total_dur={total_dur:>10.0f}µs")

    # Now parse with our trace_parser
    from breakdown.trace_parser import parse_trace_file
    from breakdown.model_info import fetch_model_config, summarize_config, get_dim_symbols
    from breakdown.analyzer import analyze_ops

    ops = parse_trace_file(trace_file)
    print(f"\n  Parsed ops (after filtering): {len(ops)}")
    for op in sorted(ops, key=lambda o: o["device_time_us"], reverse=True)[:15]:
        print(f"    {op['name']:40s} {op['backend']:18s} calls={op['count']:>4d} "
              f"dev={op['device_time_us']:>10.0f}µs cpu={op['cpu_time_us']:>10.0f}µs")

    # Full analysis
    config = fetch_model_config(model_id)
    summary = summarize_config(config)
    dim_symbols = get_dim_symbols(summary)

    analyzed = analyze_ops(ops, dim_symbols=dim_symbols, batch_size=1, seq_len=None,
                           model_dtype=summary.get("dtype", "bfloat16"),
                           num_layers=summary.get("num_layers"))
    print(f"\n  Analyzed ops (after merge): {len(analyzed)}")
    for op in analyzed[:15]:
        d = op.to_dict()
        shapes_str = str(d["input_shapes"])[:50] if d["input_shapes"] else "—"
        print(f"    {d['name']:40s} {d['backend']:18s} ×{d['layer_count']:<3d} "
              f"calls={d['call_count']:>4d} shapes={shapes_str}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
