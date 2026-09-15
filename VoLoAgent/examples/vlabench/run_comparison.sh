#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# VLABench comparison evaluation: passthrough vs orchestrated
# Requires: VLABench env (conda: vlabench), pi0.5 model on port 8002
#
# Usage:
#   # 1. Start VLABench model server (port 8002)
#   cd ~/VLABench/third_party/openpi
#   PYTHONPATH=src ~/openpi/.venv/bin/python scripts/serve_policy.py \
#       --port 8002 policy:checkpoint \
#       --policy.config=pi05_ft_vlabench_primitive \
#       --policy.dir=$HOME/vlabench-checkpoints/pi05-primitive-10task
#
#   # 2. Start orchestrator proxy (port 8019)
#   vlm-orchestrator --env libero --mode subgoal --vla-port 8002 --port 8019
#
#   # 3. Run this script
#   bash examples/vlabench/run_comparison.sh

set -euo pipefail

VLABENCH_ROOT="${VLABENCH_ROOT:-$HOME/VLABench/VLABench}"
PYTHON="${VLABENCH_PYTHON:-$HOME/miniconda3/envs/vlabench/bin/python}"
EVAL_SCRIPT="examples/vlabench/vlabench_eval_client.py"
TRACK="${TRACK:-track_1_in_distribution}"
N_EPISODE="${N_EPISODE:-3}"
MAX_TASKS="${MAX_TASKS:-0}"  # 0 = all tasks

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/vlabench/comparison_${TS}"
mkdir -p "${RESULTS_DIR}"

export VLABENCH_ROOT MUJOCO_GL=egl

echo "═══════════════════════════════════════════════════════════════"
echo " VLABench Comparison: Passthrough vs Orchestrated"
echo " Track: ${TRACK}, Episodes/task: ${N_EPISODE}"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo ">>> Phase 1: Passthrough (port 8002)"
${PYTHON} ${EVAL_SCRIPT} \
    --port 8002 \
    --eval-track "${TRACK}" \
    --n-episode "${N_EPISODE}" \
    --max-tasks "${MAX_TASKS}" \
    --log-dir "${RESULTS_DIR}/passthrough"

echo ""
echo ">>> Phase 2: Orchestrated (port 8019)"
${PYTHON} ${EVAL_SCRIPT} \
    --port 8019 \
    --eval-track "${TRACK}" \
    --n-episode "${N_EPISODE}" \
    --max-tasks "${MAX_TASKS}" \
    --enable-depth \
    --log-dir "${RESULTS_DIR}/orchestrated"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Results saved to ${RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════"
