#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run static analysis for all high-priority vLLM-supported models in the catalog.
#
# Usage:
#   ./scripts/run_catalog_models.sh                 # All high-priority models
#   ./scripts/run_catalog_models.sh --type LLM      # LLM only
#   ./scripts/run_catalog_models.sh --type MLLM     # Vision-language only
#   ./scripts/run_catalog_models.sh --all            # All models (not just high priority)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Source oneAPI if available
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
fi

MODEL_TYPE=""
ALL_MODELS=false
TP_SIZE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)  MODEL_TYPE="$2"; shift 2 ;;
        --all)   ALL_MODELS=true; shift ;;
        --tp)    TP_SIZE="$2"; shift 2 ;;
        *)       echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== vLLM-XPU Model Catalog — Static Analysis ==="
echo ""

python3 -c "
from breakdown.model_catalog import CATALOG, get_models_by_type, get_vllm_models
from breakdown.model_info import fetch_model_config, summarize_config
from breakdown.model_graph import build_model_graph
import json, sys

model_type = '${MODEL_TYPE}' or None
all_models = ${ALL_MODELS,,}  # bash lowercase
tp_size = ${TP_SIZE}

if model_type:
    models = get_models_by_type(model_type)
else:
    models = get_vllm_models()

if not all_models:
    models = [m for m in models if m.priority == 'H']

print(f'Processing {len(models)} models (tp_size={tp_size})...')
print()

results = {}
for m in models:
    if not m.hf_id:
        print(f'  SKIP {m.name:40s} — no HF ID (unreleased)')
        continue
    try:
        config = fetch_model_config(m.hf_id)
        summary = summarize_config(config)
        graph = build_model_graph(summary, tp_size=tp_size)
        family = graph.get('family', '?')
        model_type_g = graph.get('model_type', '?')
        print(f'  OK   {m.name:40s} family={family:12s} type={model_type_g}')
        results[m.name] = {'status': 'ok', 'family': family}
    except Exception as e:
        print(f'  FAIL {m.name:40s} — {e}')
        results[m.name] = {'status': 'error', 'error': str(e)}

print()
ok = sum(1 for r in results.values() if r['status'] == 'ok')
fail = sum(1 for r in results.values() if r['status'] == 'error')
skip = len(models) - len(results)
print(f'Results: {ok} OK, {fail} FAIL, {skip} SKIP')
"
