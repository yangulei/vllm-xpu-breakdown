#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Profile a vLLM inference run on Intel XPU or NVIDIA CUDA and generate
# breakdown reports.
#
# Usage:
#   ./scripts/run_profile.sh --model Qwen/Qwen3-4B --max-model-len 32768
#   ./scripts/run_profile.sh --model meta-llama/Llama-3.2-1B-Instruct --output-dir output/llama
#
# All arguments are forwarded to run_profile.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Source oneAPI if available (Intel XPU)
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
fi

echo "=== vLLM Ops/Kernels Breakdown ==="
echo "Working directory: $PROJECT_DIR"
echo "Arguments: $@"
echo ""

python run_profile.py "$@"
