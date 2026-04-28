#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Compare eager vs torch.compile dispatch breakdown.
#
# Usage:
#   ./scripts/compare_modes.sh --model Qwen/Qwen3-4B --max-model-len 32768

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
fi

echo "=== Mode Comparison: Eager vs Compiled ==="
echo ""

# Run eager mode
echo "--- Eager Mode ---"
python run_profile.py --output-dir output/eager "$@"

# Run compiled mode (VLLM_TORCH_COMPILE_LEVEL=3)
echo ""
echo "--- Compiled Mode (torch.compile) ---"
VLLM_TORCH_COMPILE_LEVEL=3 python run_profile.py --output-dir output/compiled "$@"

echo ""
echo "=== Comparison Complete ==="
echo "Eager results:    output/eager/"
echo "Compiled results: output/compiled/"
echo ""
echo "Compare the JSON files:"
echo "  python -c \"import json; e=json.load(open('output/eager/ops_breakdown.json')); c=json.load(open('output/compiled/ops_breakdown.json')); print('Eager backends:', {k:v['pct_device_time'] for k,v in e['summary']['backends'].items() if v['num_ops']>0}); print('Compiled backends:', {k:v['pct_device_time'] for k,v in c['summary']['backends'].items() if v['num_ops']>0})\""
