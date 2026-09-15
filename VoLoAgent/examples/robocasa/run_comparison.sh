#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# RoboCasa: Passthrough vs Orchestrated comparison evaluation
#
# Prerequisites:
#   - RoboCasa pi0 server on port 8002
#   - Orchestrator proxy on port 8019 (--vla-port 8002)
#   - RoboCasa conda env activated
#
# Usage:
#   bash examples/robocasa/run_comparison.sh [num_trials] [max_tasks]

set -euo pipefail

NUM_TRIALS=${1:-10}
MAX_TASKS=${2:-5}
TASK_SET=${3:-atomic_seen}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="results/robocasa/comparison_${TIMESTAMP}"

echo "═══════════════════════════════════════════════════════════"
echo "  RoboCasa Comparison: Passthrough vs Orchestrated"
echo "  Task set: ${TASK_SET}"
echo "  Trials/task: ${NUM_TRIALS}"
echo "  Max tasks: ${MAX_TASKS} (0=all)"
echo "  Results: ${BASE_DIR}"
echo "═══════════════════════════════════════════════════════════"

# 1. Passthrough (direct to VLA on port 8002)
echo ""
echo ">>> Phase 1: PASSTHROUGH (port 8002)"
python examples/robocasa/robocasa_eval_client.py \
    --port 8002 \
    --task-set ${TASK_SET} \
    --num-trials ${NUM_TRIALS} \
    --max-tasks ${MAX_TASKS} \
    --log-dir "${BASE_DIR}/passthrough"

# 2. Orchestrated (through proxy on port 8019)
echo ""
echo ">>> Phase 2: ORCHESTRATED (port 8019)"
python examples/robocasa/robocasa_eval_client.py \
    --port 8019 \
    --task-set ${TASK_SET} \
    --num-trials ${NUM_TRIALS} \
    --max-tasks ${MAX_TASKS} \
    --enable-depth \
    --log-dir "${BASE_DIR}/orchestrated"

# 3. Generate comparison report
echo ""
echo ">>> Phase 3: Generating comparison report"
python -c "
import json, pathlib

base = pathlib.Path('${BASE_DIR}')
p = json.load(open(base / 'passthrough/results.json'))
o = json.load(open(base / 'orchestrated/results.json'))

lines = [
    '# RoboCasa Comparison Report',
    '',
    '| Mode | Success Rate | Successes | Episodes |',
    '|------|-------------|-----------|----------|',
    f'| Passthrough | {p[\"success_rate\"]*100:.1f}% | {p[\"total_successes\"]} | {p[\"total_episodes\"]} |',
    f'| Orchestrated | {o[\"success_rate\"]*100:.1f}% | {o[\"total_successes\"]} | {o[\"total_episodes\"]} |',
    f'| **Delta** | **{(o[\"success_rate\"]-p[\"success_rate\"])*100:+.1f}pp** | | |',
    '',
    '## Per-Task Comparison',
    '',
    '| Task | Passthrough | Orchestrated | Delta |',
    '|------|-------------|--------------|-------|',
]
p_tasks = {t['task']: t for t in p['tasks']}
o_tasks = {t['task']: t for t in o['tasks']}
for name in p_tasks:
    pt = p_tasks[name]
    ot = o_tasks.get(name, {'success_rate': 0})
    delta = (ot['success_rate'] - pt['success_rate']) * 100
    lines.append(f'| {name} | {pt[\"success_rate\"]*100:.1f}% | {ot[\"success_rate\"]*100:.1f}% | {delta:+.1f}pp |')
lines.append('')

with open(base / 'COMPARISON.md', 'w') as f:
    f.write('\n'.join(lines))
print(f'Comparison saved to {base}/COMPARISON.md')
print('\n'.join(lines))
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Done! Results in ${BASE_DIR}"
echo "═══════════════════════════════════════════════════════════"
