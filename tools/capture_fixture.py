# SPDX-License-Identifier: Apache-2.0
"""Re-capture the canonical MiniMax-M3 TP4 6-layer profile.

Produces the two raw rank-0 traces (prefill and decode passes) that
``tools/make_fixture.py`` trims into ``tests/data/``. Requires 4 XPUs and the
MiniMax-M3 weights in the HF cache.

    python tools/capture_fixture.py
    python tools/make_fixture.py output/fixtures/m3_tp4_6l_xpu_prefill.raw.json.gz \
        tests/data/m3_tp4_6l_xpu_prefill.json.gz \
        --model MiniMaxAI/MiniMax-M3 --tp 4 --batch 1 \
        --query-len 2048 --context-len 2048
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as A  # noqa: E402

MODEL = "MiniMaxAI/MiniMax-M3"
OUT = os.path.abspath("output/fixtures")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    A._run_profile(
        model_id=MODEL, mode="eager", max_model_len=4096 + 64,
        batch_size=1, max_tokens=8, prompt="",
        num_profile_layers=6, tp_size=4, quantization=None,
        gpu_memory_utilization=0.85,
        query_len=2048, context_len=2048,
        prefill_batch_size=1, decode_batch_size=32,
    )
    state = A._profile_state
    if state.get("status") != "done":
        print("profiling failed:", state.get("error"))
        return 1
    result = state["result"]
    for tag in ("prefill", "decode"):
        src = result.get(f"{tag}_trace_file")
        if src and os.path.exists(src):
            dst = os.path.join(OUT, f"m3_tp4_6l_xpu_{tag}.raw.json.gz")
            shutil.copy2(src, dst)
            print(f"{tag}: {dst} ({os.path.getsize(dst)/1024:.0f} KiB)")
    print("symbols:", json.dumps(result.get("graph", {}).get("symbols"),
                                 default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
