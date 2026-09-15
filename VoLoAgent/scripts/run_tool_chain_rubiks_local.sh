#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Local tool_chain Rubik's-cube smoke — verifies two recent fixes:
#   1. PLACE_RELEASE_HEIGHT_M: 0.05 → 0.10
#      (place tool no longer plants the gripper into the bin)
#   2. GRASP_VERIFY_THRESHOLD: 0.05 → 0.02
#      (CLOSING phase no longer mis-classifies a partial-closure on
#       the Rubik's cube as empty air)
#
# Memory-aware config: SAM3 grasp + Claude (vlm_point) place — no
# local Molmo2 sidecar, so the GPU budget is just Isaac Sim (~20 GB) +
# grasp server with SAM3 + GraspGen (~17 GB) ≈ 37 GB / 48 GB.
#
# Env-var overrides:
#   TASKS              robolab --task list   (default InferClearTableTask)
#   NUM_EPISODES       episodes per task     (default 1)
#   USE_FRONT_CAMERA   "1" → pass --use-front-camera to the orchestrator
#                      (default 0 = oblique exterior camera).  Set to 1 to
#                      use the egocentric mirrored camera.
#   RUN_NAME           output folder name    (default tool_chain_rubiks_local)
#
# Default task (InferClearTableTask) is the scene where the two
# pre-fix bugs were observed in run lh-cs-tool-chain-sam3-molmo2-5.
#
# Services (all on this box):
#   :8003  grasp server (graspgen env) — SAM3 + GraspGen
#   :8001  orchestrator (vlm-orch env)
#   robolab eval client (robolab env, foreground)
#
# Outputs:
#   ~/vlm-orchestrator/results/tool_chain_rubiks_local/  (orchestrator)
#   ~/robolab/output/tool_chain_rubiks_local/            (videos + hdf5)

set -e

GRASP_PORT=8003
ORCH_PORT=8001

RUN_NAME=${RUN_NAME:-tool_chain_rubiks_local}
ORCH_LOG_DIR="$HOME/vlm-orchestrator/results/$RUN_NAME"
LOGS_DIR="$ORCH_LOG_DIR/_services"
mkdir -p "$LOGS_DIR"
rm -rf "$ORCH_LOG_DIR"
mkdir -p "$ORCH_LOG_DIR" "$LOGS_DIR"
rm -rf "$HOME/robolab/output/$RUN_NAME"

TASKS=${TASKS:-InferClearTableTask}
NUM_EPISODES=${NUM_EPISODES:-1}
USE_FRONT_CAMERA=${USE_FRONT_CAMERA:-0}

FRONT_CAM_FLAG=""
if [ "$USE_FRONT_CAMERA" = "1" ] || [ "$USE_FRONT_CAMERA" = "true" ]; then
  FRONT_CAM_FLAG="--use-front-camera"
fi

# ─── Pre-flight ──────────────────────────────────────────────────────
for env in graspgen vlm-orch robolab; do
  if ! conda env list | awk '{print $1}' | grep -qx "$env"; then
    echo "[run] conda env '$env' missing — abort"; exit 1
  fi
done
for port in $GRASP_PORT $ORCH_PORT; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 2

PIDS=()
cleanup() {
  echo
  echo "[run] cleaning up background services..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  for port in $GRASP_PORT $ORCH_PORT; do
    fuser -k "${port}/tcp" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_port() {
  local label="$1" port="$2" timeout="${3:-180}"
  local start
  start=$(date +%s)
  while ! python3 -c "import socket; s=socket.socket(); s.settimeout(1); \
                      s.connect(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
    if (( $(date +%s) - start > timeout )); then
      echo "[run] ❌ $label port $port never came up after ${timeout}s"
      return 1
    fi
    sleep 2
  done
  echo "[run] ✓ $label ready on :$port"
}

# ─── 1. Grasp server (SAM3 + GraspGen, no Molmo) ─────────────────────
GRIPPER_CFG=""
for cand in \
  "$HOME/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml" \
  "$HOME/graspgen/models/checkpoints/graspgen_franka_panda.yml"; do
  if [ -f "$cand" ]; then GRIPPER_CFG="$cand"; break; fi
done
[ -z "$GRIPPER_CFG" ] && { echo "[run] gripper cfg not found"; exit 1; }
echo "[run] gripper cfg: $GRIPPER_CFG"
echo "[run] starting grasp server on :$GRASP_PORT (SAM3 + GraspGen) ..."
conda run --no-capture-output -n graspgen \
  python -m vlm_orchestrator.grasp.server \
    --gripper-config "$GRIPPER_CFG" \
    --port $GRASP_PORT \
    --enable-sam3 --sam3-model facebook/sam3 \
    --enable-curobo \
    --verbose \
  > "$LOGS_DIR/grasp_server.log" 2>&1 &
PIDS+=($!)

# --- VLA stub no longer needed -------------------------------------
# tool_chain mode uses the in-process StubBackend (protocols/stub.py)
# so the orchestrator synthesizes the openpi metadata handshake
# itself.  --vla-host / --vla-port are still accepted by argparse but
# ignored when --mode tool_chain.

wait_port "grasp server" $GRASP_PORT 180

# ─── 2. Orchestrator (tool_chain, SAM3 grasp + vlm_point place) ──────
echo "[run] starting orchestrator on :$ORCH_PORT ..."
export GRASP_SERVER_HOST=127.0.0.1
export GRASP_SERVER_PORT=$GRASP_PORT
conda run --no-capture-output -n vlm-orch \
  vlm-orchestrator \
    --port $ORCH_PORT \
    --mode tool_chain \
    --vlm-model YOUR_VLM_MODEL \
    --grasp-seg-mode sam3 \
    --place-seg-mode vlm_point \
    --grasp-topdown-threshold 0.85 \
    $FRONT_CAM_FLAG \
    --log-dir "$ORCH_LOG_DIR" \
    --verbose \
  > "$ORCH_LOG_DIR/orchestrator.log" 2>&1 &
PIDS+=($!)
wait_port "orchestrator" $ORCH_PORT 60

# ─── 4. Robolab eval client ──────────────────────────────────────────
echo "[run] launching robolab eval (task=$TASKS, episodes=$NUM_EPISODES)"
echo "[run] orch log:     $ORCH_LOG_DIR/orchestrator.log"
echo "[run] service logs: $LOGS_DIR/"
echo
conda run --no-capture-output -n robolab \
  python "$HOME/robolab/policies/volo/run.py" \
    --headless \
    --remote-host 127.0.0.1 --remote-port $ORCH_PORT \
    --num-runs "$NUM_EPISODES" \
    --enable-subtask \
    --enable-gt-state \
    --video-mode all \
    --policy pi05 \
    --task-dirs robovolo \
    --output-folder-name "$RUN_NAME" \
    --task $TASKS

echo
echo "[run] done."
echo "  orch:   $ORCH_LOG_DIR"
echo "  videos: $HOME/robolab/output/$RUN_NAME"
