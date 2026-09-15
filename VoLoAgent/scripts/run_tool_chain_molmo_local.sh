#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Local end-to-end test of tool_chain mode with Molmo2 perception
# (molmo_sam2 grasp + molmo_point place).  No pi0.5 process — tool_chain
# bypasses the VLA entirely (in-process StubBackend in protocols/stub.py
# synthesizes the openpi metadata handshake).
#
# Service map (all on this box):
#   :8122  Molmo2 vLLM           (molmo-env env)
#   :8003  grasp server (SAM2)   (graspgen env)
#   :8001  orchestrator          (vlm-orch env)
#   robolab eval client          (robolab env, foreground)
#
# Outputs:
#   ~/vlm-orchestrator/results/tool_chain_molmo_local/  (orchestrator)
#   ~/robolab/output/tool_chain_molmo_local/            (videos + hdf5)
#
# All background processes are killed on EXIT.

set -e

# ─── Config ───────────────────────────────────────────────────────────
MOLMO_MODEL=${MOLMO_MODEL:-allenai/Molmo2-8B}
# NOTE: Molmo2-8B can't be served by vllm 0.9.1 (transformers conflict),
# so we use a small HF-transformers HTTP shim
# (vlm_orchestrator/utils/molmo2_hf_server.py) in the molmo-env env (which has
# transformers 4.57 + torch 2.7+cu128 + accelerate).  The shim
# exposes the same /v1/chat/completions endpoint our vision/molmo.py
# client expects.
MOLMO_PORT=8122
GRASP_PORT=8003
ORCH_PORT=8001

RUN_NAME="tool_chain_molmo_local"
ORCH_LOG_DIR="$HOME/vlm-orchestrator/results/$RUN_NAME"
LOGS_DIR="$ORCH_LOG_DIR/_services"
mkdir -p "$LOGS_DIR"
rm -rf "$ORCH_LOG_DIR"
mkdir -p "$ORCH_LOG_DIR" "$LOGS_DIR"
rm -rf "$HOME/robolab/output/$RUN_NAME"

# Tasks for the smoke run.  Trim to keep wallclock reasonable.
TASKS=${TASKS:-InferClearTableTask}
NUM_EPISODES=${NUM_EPISODES:-1}

# ─── Sanity checks ────────────────────────────────────────────────────
# molmo-env hosts the Molmo2 HF shim; graspgen runs the grasp server;
# vlm-orch hosts the orchestrator; robolab runs the eval client.
for env in molmo-env graspgen vlm-orch robolab; do
  if ! conda env list | awk '{print $1}' | grep -qx "$env"; then
    echo "[run] conda env '$env' missing — abort"; exit 1
  fi
done

# Kill anything left over on our service ports so we don't get a
# stale orchestrator binding port 8001 (etc.) from an earlier run.
for port in $MOLMO_PORT $GRASP_PORT $ORCH_PORT; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 2

PIDS=()
cleanup() {
  echo
  echo "[run] cleaning up background services..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  # Give them a moment, then hard-kill stragglers.
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_port() {
  local label="$1" port="$2" timeout="${3:-180}"
  local start=$(date +%s)
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

# ─── 1. Grasp server (also subprocess-spawns Molmo2 HF shim) ─────────
# `--enable-molmo` makes the grasp server fork the Molmo2 HF shim in
# the `molmo-env` env (which has the transformers/huggingface_hub
# versions Molmo2 needs).  Grasp server supervises the shim — kills
# it on exit.  This avoids the env conflict where graspgen's SAM3
# deps pin huggingface_hub < 0.30 while Molmo2 needs >= 0.30.
echo "[run] starting grasp server on :$GRASP_PORT (Molmo2 sidecar on :$MOLMO_PORT) ..."
GRIPPER_CFG=""
for cand in \
  "$HOME/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml" \
  "$HOME/graspgen/models/checkpoints/graspgen_franka_panda.yml"; do
  if [ -f "$cand" ]; then GRIPPER_CFG="$cand"; break; fi
done
if [ -z "$GRIPPER_CFG" ]; then
  echo "[run] could not find graspgen_franka_panda.yml — abort"; exit 1
fi
echo "[run] using gripper cfg: $GRIPPER_CFG"
conda run --no-capture-output -n graspgen \
  python -m vlm_orchestrator.grasp.server \
    --gripper-config "$GRIPPER_CFG" \
    --port $GRASP_PORT \
    --enable-sam2 --sam2-model facebook/sam2.1-hiera-small \
    --enable-curobo \
    --enable-molmo --molmo-model "$MOLMO_MODEL" --molmo-port $MOLMO_PORT \
    --molmo-quantize ${MOLMO_QUANTIZE:-int4} \
  > "$LOGS_DIR/grasp_server.log" 2>&1 &
# NOTE: --enable-sam3 / --enable-gdino intentionally omitted here.
# molmo_sam2 + molmo_point only need SAM2 (point-prompt segmentation)
# + Molmo2 (pointing).  Adding SAM3 + GDinoV2 costs ~5 GB GPU which
# pushes Isaac Sim out of memory on this 48 GB card with Molmo2-8B
# resident.
PIDS+=($!)

# --- VLA stub no longer needed -------------------------------------
# tool_chain mode uses the in-process StubBackend (protocols/stub.py)
# so the orchestrator synthesizes the openpi metadata handshake
# itself.  --vla-host / --vla-port are ignored when --mode tool_chain.

# Wait for grasp server + Molmo sidecar.  Grasp server boots first
# (SAM2 in ~10s), then it spawns the Molmo2 sidecar in the
# molmo-env env which takes ~60-90s to load weights.
wait_port "grasp server" $GRASP_PORT 120
wait_port "Molmo2 shim"  $MOLMO_PORT 240

# ─── 3. Orchestrator ─────────────────────────────────────────────────
echo "[run] starting orchestrator on :$ORCH_PORT ..."
export GRASP_SERVER_HOST=127.0.0.1
export GRASP_SERVER_PORT=$GRASP_PORT
export MOLMO_BASE_URL="http://127.0.0.1:$MOLMO_PORT/v1"
export MOLMO_MODEL
conda run --no-capture-output -n vlm-orch \
  vlm-orchestrator \
    --port $ORCH_PORT \
    --mode tool_chain \
    --vlm-model YOUR_VLM_MODEL \
    --grasp-seg-mode molmo_sam2 \
    --place-seg-mode molmo_point \
    --grasp-topdown-threshold 0.85 \
    --log-dir "$ORCH_LOG_DIR" \
    --verbose \
  > "$ORCH_LOG_DIR/orchestrator.log" 2>&1 &
PIDS+=($!)

wait_port "orchestrator" $ORCH_PORT 60

# ─── 5. Robolab eval client ──────────────────────────────────────────
echo "[run] launching robolab eval (tasks: $TASKS, episodes: $NUM_EPISODES)"
echo "[run] orchestrator log: $ORCH_LOG_DIR/orchestrator.log"
echo "[run] service logs:     $LOGS_DIR/"
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
echo "  orch:    $ORCH_LOG_DIR"
echo "  videos:  $HOME/robolab/output/$RUN_NAME"
