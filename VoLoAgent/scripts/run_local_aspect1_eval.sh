#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Local Aspect-1 representative-task eval — 6 LH tasks × 1 episode in
# both passthrough and subgoal-vlm modes.
#
# Layout (matches docs/instructions/benchmark-launch-guide.md "RoboLab" track):
#   T1: pi0.5 policy server (openpi uv venv) on :8002
#   T2: orchestrator (vlm-orch conda env) on :8001
#   T3: robolab eval client (robolab conda env)
#
# Each service runs in its own tmux session so progress + logs are
# inspectable.  Per-mode log dir: ~/vlm-orchestrator/results/aspect1_local_<mode>/
#
# Usage:
#   bash scripts/run_local_aspect1_eval.sh
# Tmux:
#   tmux ls         # see live sessions
#   tmux attach -t <session>
#
# Pre-reqs (verified before launching):
#   - conda envs: openpi (.venv via uv), vlm-orch, robolab
#   - VLM_API_KEY in shell env (for the subgoal-mode VLM)

set -e

REPO_VLMORCH=${HOME}/vlm-orchestrator
REPO_OPENPI=${HOME}/openpi
REPO_ROBOLAB=${HOME}/robolab

POLICY_PORT=8002
ORCH_PORT=8001
GRASP_PORT=8003

POLICY_SESSION=aspect1_policy
ORCH_SESSION=aspect1_orch
GRASP_SESSION=aspect1_grasp
EVAL_LOG_DIR=$REPO_VLMORCH/results/aspect1_local_logs
mkdir -p "$EVAL_LOG_DIR"

GRASP_GRIPPER_CFG=$HOME/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml

REP_TASKS="RestackTopOnLooseTask CyclicReorderTowerTask"

# ── Pre-flight ──────────────────────────────────────────────────────
if [ -z "${VLM_API_KEY:-}" ]; then
  echo "[aspect1] WARN: VLM_API_KEY not set — subgoal mode will fail VLM calls"
fi

# Tear down any leftover sessions from a previous run
tmux kill-session -t $POLICY_SESSION 2>/dev/null || true
tmux kill-session -t $ORCH_SESSION 2>/dev/null || true
tmux kill-session -t $GRASP_SESSION 2>/dev/null || true

# ── T1: policy server ──────────────────────────────────────────────
echo "[aspect1] starting policy server in tmux:$POLICY_SESSION"
tmux new-session -d -s $POLICY_SESSION -c $REPO_OPENPI "
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py \
      --port $POLICY_PORT \
      policy:checkpoint \
      --policy.config=pi05_droid_jointpos \
      --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos \
      2>&1 | tee $EVAL_LOG_DIR/policy.log
"

echo "[aspect1] waiting for policy server on :$POLICY_PORT (up to 30 min, first run downloads the checkpoint)..."
for i in $(seq 1 900); do
  if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $POLICY_PORT)); s.close()" 2>/dev/null; then
    echo "[aspect1] policy server is up (after ${i}s)"
    break
  fi
  sleep 2
done
if ! python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $POLICY_PORT)); s.close()" 2>/dev/null; then
  echo "[aspect1] FATAL: policy server didn't come up. tail $EVAL_LOG_DIR/policy.log"
  exit 1
fi

# ── T1b: grasp server (graspgen env, port 8003) ────────────────────
# Required by --recovery-mode replan_grasp / vlm_grasp / grasp_first.
# Without it, grasp-tool recovery silently degenerates to replan-only
# (orchestrator falls back to local GroundingDINO and never gets a
# grasp pose), undermining mode comparisons.
if [ -f "$GRASP_GRIPPER_CFG" ]; then
  echo "[aspect1] starting grasp server in tmux:$GRASP_SESSION"
  tmux new-session -d -s $GRASP_SESSION -c $REPO_VLMORCH "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate graspgen
    python -m vlm_orchestrator.grasp.server \
        --gripper-config '$GRASP_GRIPPER_CFG' \
        --port $GRASP_PORT --verbose \
        --enable-sam2 --sam2-model facebook/sam2.1-hiera-small \
    --enable-curobo \
        --enable-gdino \
        2>&1 | tee $EVAL_LOG_DIR/grasp.log
  "
  echo "[aspect1] waiting for grasp server on :$GRASP_PORT (up to 5 min)..."
  for i in $(seq 1 150); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $GRASP_PORT)); s.close()" 2>/dev/null; then
      echo "[aspect1] grasp server is up (after ${i}s)"
      break
    fi
    sleep 2
  done
  if ! python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $GRASP_PORT)); s.close()" 2>/dev/null; then
    echo "[aspect1] FATAL: grasp server didn't come up. tail $EVAL_LOG_DIR/grasp.log"
    exit 1
  fi
else
  echo "[aspect1] FATAL: grasp gripper config not found at $GRASP_GRIPPER_CFG"
  echo "[aspect1] subgoal mode uses --recovery-mode replan_grasp; the grasp server is required."
  exit 1
fi

# ── Per-mode loop ───────────────────────────────────────────────────
run_mode() {
  local mode=$1
  local orch_args=$2
  local label="aspect1_local_${mode}"
  local log_dir=$REPO_VLMORCH/results/$label

  echo
  echo "═════════════════════════════════════════════════════════════"
  echo "[aspect1] mode=${mode}  log_dir=${log_dir}"
  echo "═════════════════════════════════════════════════════════════"
  mkdir -p "$log_dir"

  tmux kill-session -t $ORCH_SESSION 2>/dev/null || true

  tmux new-session -d -s $ORCH_SESSION -c $REPO_VLMORCH "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vlm-orch
    export VLM_API_KEY='${VLM_API_KEY:-}'
    vlm-orchestrator \
        --vla-host 127.0.0.1 --vla-port $POLICY_PORT \
        --port $ORCH_PORT \
        ${orch_args} \
        --log-dir $log_dir \
        --verbose \
        2>&1 | tee $EVAL_LOG_DIR/orch_${mode}.log
  "

  echo "[aspect1] waiting for orchestrator on :$ORCH_PORT..."
  for i in $(seq 1 300); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', $ORCH_PORT)); s.close()" 2>/dev/null; then
      echo "[aspect1] orchestrator is up (after ${i}s)"
      break
    fi
    sleep 2
  done

  echo "[aspect1] launching robolab eval client (1 ep × $(echo $REP_TASKS | wc -w) tasks)..."
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate robolab
  cd $REPO_ROBOLAB
  python policies/volo/run.py \
      --headless \
      --remote-host 127.0.0.1 --remote-port $ORCH_PORT \
      --num-runs 1 \
      --enable-subtask \
      --enable-gt-state \
      --video-mode all \
      --policy pi05 \
      --task-dirs robovolo \
      --output-folder-name $label \
      --task $REP_TASKS \
      2>&1 | tee $EVAL_LOG_DIR/eval_${mode}.log

  echo "[aspect1] mode=${mode} done"
  tmux kill-session -t $ORCH_SESSION 2>/dev/null || true
  conda deactivate
}

# Mode 1: passthrough (orchestrator does no orchestration; just forwards)
run_mode "pass" "--mode passthrough"

# Mode 2: subgoal + VLM monitor (Claude 4.6) — exercises the new
# Aspect-1 logger across a richer event distribution.
run_mode "subgoal" "--mode subgoal --failure-monitor vlm --recovery-mode replan_grasp --vlm-model YOUR_VLM_MODEL --use-front-camera --grasp-seg-mode gdino_sam2 --subgoal-timeout 9999"

# ── Teardown ───────────────────────────────────────────────────────
echo
echo "[aspect1] tearing down policy + grasp servers"
tmux kill-session -t $POLICY_SESSION 2>/dev/null || true
tmux kill-session -t $GRASP_SESSION 2>/dev/null || true

echo
echo "[aspect1] all done"
echo "  passthrough log dir: $REPO_VLMORCH/results/aspect1_local_pass/"
echo "  subgoal log dir:     $REPO_VLMORCH/results/aspect1_local_subgoal/"
echo "  service logs:        $EVAL_LOG_DIR/{policy,orch_pass,orch_subgoal,eval_pass,eval_subgoal}.log"
