#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Local A/B: linear vs cuRobo motion planning in tool_chain mode.
#
# tool_chain bypasses the VLA entirely, so perception (SAM3 grasp +
# Claude vlm_point place), IK targets, and grasp/place POSES are
# identical across the two arms — the ONLY thing that changes is the
# joint-space trajectory between waypoints:
#   linear  = straight interpolate_joints (today's behavior)
#   curobo  = collision-aware plan_cspace on the grasp server
#             (requires the grasp server started with --enable-curobo,
#              cuRobo + cuda-core[cu12] in the graspgen env).
#
# One grasp server is shared by both arms (it always exposes
# /plan_motion once --enable-curobo is on; the 'linear' arm simply
# never calls it).  Each arm gets its own orchestrator + robolab eval,
# run SEQUENTIALLY so Isaac Sim + the grasp server never contend for
# GPU across arms.
#
# Env-var overrides:
#   TASKS          robolab --task list        (default: 3 single-pick-place
#                  tasks with distinct object geometries —
#                    RecoverFallenFruitTask  orange (sphere) → bowl
#                    InferReturnStrayTask    tuna can (cylinder) → bin
#                    RecoverToyInBowlTask    rubik's cube (box) → table)
#   NUM_EPISODES   episodes per task per arm  (default 2)
#   PLANNERS       space-sep arm list         (default "linear curobo")
#   RUN_NAME       output folder prefix       (default tool_chain_curobo_ab)
#   USE_FRONT_CAMERA "1" → --use-front-camera (default 0)
#
# Outputs (per arm):
#   ~/vlm-orchestrator/results/<RUN_NAME>_<planner>/   (orchestrator + logs)
#   ~/robolab/output/<RUN_NAME>_<planner>/             (videos + hdf5)
#
# Compare with:  scripts/compare_motion_planner_ab.py  (see bottom).

set -e

GRASP_PORT=${GRASP_PORT:-8003}
ORCH_PORT=${ORCH_PORT:-8001}

RUN_NAME=${RUN_NAME:-tool_chain_curobo_ab}
# Three single-pick-place tasks with distinct object geometries so the A/B
# exercises grasp/place trajectories across sphere / cylinder / box shapes.
TASKS=${TASKS:-"RecoverFallenFruitTask InferReturnStrayTask RecoverToyInBowlTask"}
NUM_EPISODES=${NUM_EPISODES:-2}
# robolab --task-dirs to search for the tasks named in TASKS. Stack tasks live
# in benchmark/ (StackYellowOnRedTask); pick-place tasks in robovolo/.
TASK_DIRS=${TASK_DIRS:-"robovolo"}
# Top-down grasp filter. Production default is 0.85, but that filters ~95%
# of GraspGen candidates ("Top-down filtering removed all grasps" 422s), so
# the grasp aborts/replans BEFORE the motion planner runs — starving the A/B
# of trajectory data. 0.3 (the grasp server's own default) lets many more
# grasps through so the motion planner is actually exercised. Both arms use
# the SAME value, so pose selection stays identical across arms; only the
# joint-space trajectory differs.
GRASP_TOPDOWN_THRESHOLD=${GRASP_TOPDOWN_THRESHOLD:-0.3}
PLANNERS=${PLANNERS:-"linear curobo"}
USE_FRONT_CAMERA=${USE_FRONT_CAMERA:-0}
# ENABLE_STACK_MODE=1 adds --enable-stack-mode so the VLM place tool can pick
# stack=True (current-EE rotation candidate) vs stack=False (top-down only).
ENABLE_STACK_MODE=${ENABLE_STACK_MODE:-0}

FRONT_CAM_FLAG=""
if [ "$USE_FRONT_CAMERA" = "1" ] || [ "$USE_FRONT_CAMERA" = "true" ]; then
  FRONT_CAM_FLAG="--use-front-camera"
fi

STACK_MODE_FLAG=""
if [ "$ENABLE_STACK_MODE" = "1" ] || [ "$ENABLE_STACK_MODE" = "true" ]; then
  STACK_MODE_FLAG="--enable-stack-mode"
fi

TOP_LOG_DIR="$HOME/vlm-orchestrator/results/${RUN_NAME}_services"
mkdir -p "$TOP_LOG_DIR"

# ─── Pre-flight ──────────────────────────────────────────────────────
for env in graspgen vlm-orch; do
  if ! conda env list | awk '{print $1}' | grep -qx "$env"; then
    echo "[run] conda env '$env' missing — abort"; exit 1
  fi
done
if [ ! -x "$HOME/robolab/.venv/bin/python" ]; then
  echo "[run] robolab uv venv missing ($HOME/robolab/.venv/bin/python) — abort"
  exit 1
fi

# Confirm cuRobo is importable in graspgen (only needed if 'curobo' arm requested).
if echo "$PLANNERS" | grep -qw curobo; then
  if ! conda run -n graspgen python -c "import curobo" 2>/dev/null; then
    echo "[run] ❌ cuRobo not importable in graspgen env — install with:"
    echo "        conda run -n graspgen pip install -e ~/code/curobo-cc"
    echo "        conda run -n graspgen pip install 'cuda-core[cu12]'"
    exit 1
  fi
fi

for port in $GRASP_PORT $ORCH_PORT; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 2

PIDS=()
cleanup() {
  echo
  echo "[run] cleaning up background services..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
  for port in $GRASP_PORT $ORCH_PORT; do fuser -k "${port}/tcp" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_port() {
  local label="$1" port="$2" timeout="${3:-180}"
  local start; start=$(date +%s)
  while ! python3 -c "import socket; s=socket.socket(); s.settimeout(1); \
                      s.connect(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
    if (( $(date +%s) - start > timeout )); then
      echo "[run] ❌ $label port $port never came up after ${timeout}s"; return 1
    fi
    sleep 2
  done
  echo "[run] ✓ $label ready on :$port"
}

wait_port_down() {
  local port="$1" timeout="${2:-30}"
  local start; start=$(date +%s)
  while python3 -c "import socket; s=socket.socket(); s.settimeout(1); \
                    s.connect(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
    (( $(date +%s) - start > timeout )) && break
    sleep 1
  done
}

# ─── 1. Grasp server (SAM3 + GraspGen + cuRobo), shared by both arms ─
GRIPPER_CFG=""
for cand in \
  "$HOME/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml" \
  "$HOME/graspgen/models/checkpoints/graspgen_franka_panda.yml"; do
  [ -f "$cand" ] && { GRIPPER_CFG="$cand"; break; }
done
[ -z "$GRIPPER_CFG" ] && { echo "[run] gripper cfg not found"; exit 1; }
echo "[run] gripper cfg: $GRIPPER_CFG"

CUROBO_FLAG=""
if echo "$PLANNERS" | grep -qw curobo; then CUROBO_FLAG="--enable-curobo"; fi

echo "[run] starting grasp server on :$GRASP_PORT (SAM3 + GraspGen ${CUROBO_FLAG}) ..."
conda run --no-capture-output -n graspgen \
  python -m vlm_orchestrator.grasp.server \
    --gripper-config "$GRIPPER_CFG" \
    --port $GRASP_PORT \
    --enable-sam3 --sam3-model facebook/sam3 \
    $CUROBO_FLAG \
    --verbose \
  > "$TOP_LOG_DIR/grasp_server.log" 2>&1 &
PIDS+=($!)
# cuRobo warmup adds ~20s on top of SAM3+GraspGen load.
wait_port "grasp server" $GRASP_PORT 300

# ─── 2. Per-planner arms (sequential) ────────────────────────────────
export GRASP_SERVER_HOST=127.0.0.1
export GRASP_SERVER_PORT=$GRASP_PORT

for PLANNER in $PLANNERS; do
  ARM_RUN="${RUN_NAME}_${PLANNER}"
  ORCH_LOG_DIR="$HOME/vlm-orchestrator/results/$ARM_RUN"
  rm -rf "$ORCH_LOG_DIR" "$HOME/robolab/output/$ARM_RUN"
  mkdir -p "$ORCH_LOG_DIR"

  echo
  echo "════════════════════════════════════════════════════════════"
  echo "[run] ARM: motion-planner=$PLANNER  → $ARM_RUN"
  echo "════════════════════════════════════════════════════════════"

  fuser -k "${ORCH_PORT}/tcp" 2>/dev/null || true
  wait_port_down $ORCH_PORT 15

  echo "[run] starting orchestrator (--motion-planner $PLANNER) on :$ORCH_PORT ..."
  conda run --no-capture-output -n vlm-orch \
    vlm-orchestrator \
      --port $ORCH_PORT \
      --mode tool_chain \
      --vlm-model YOUR_VLM_MODEL \
      --grasp-seg-mode sam3 \
      --place-seg-mode vlm_point \
      --grasp-topdown-threshold "$GRASP_TOPDOWN_THRESHOLD" \
      --motion-planner "$PLANNER" \
      $FRONT_CAM_FLAG \
      $STACK_MODE_FLAG \
      --log-dir "$ORCH_LOG_DIR" \
      --verbose \
    > "$ORCH_LOG_DIR/orchestrator.log" 2>&1 &
  ORCH_PID=$!
  PIDS+=($ORCH_PID)
  wait_port "orchestrator[$PLANNER]" $ORCH_PORT 60

  echo "[run] launching robolab eval (task=$TASKS, num-runs=$NUM_EPISODES)"
  # Native VoLo evaluation uses the dedicated RoboLab runner so depth,
  # calibration, GT state, and episode metadata reach the orchestrator.
  # Episodes per task = --num-envs (default 1) * --num-runs.
  ( cd "$HOME/robolab" && OMNI_KIT_ACCEPT_EULA=Y \
    .venv/bin/python policies/volo/run.py \
      --headless \
      --policy pi05 \
      --open-loop-horizon 8 \
      --remote-host 127.0.0.1 --remote-port $ORCH_PORT \
      --task-dirs $TASK_DIRS \
      --task $TASKS \
      --num-runs "$NUM_EPISODES" \
      --enable-subtask --enable-gt-state \
      --video-mode all \
      --output-folder-name "$ARM_RUN" ) \
    || echo "[run] ⚠ arm '$PLANNER' eval exited non-zero (continuing)"

  # Tear down this arm's orchestrator before the next arm.
  kill "$ORCH_PID" 2>/dev/null || true
  wait_port_down $ORCH_PORT 15

  echo "[run] arm '$PLANNER' done → $ORCH_LOG_DIR"
done

echo
echo "[run] all arms done. Compare with:"
echo "  conda run -n vlm-orch python scripts/compare_motion_planner_ab.py \\"
echo "    --run-prefix $RUN_NAME --planners $PLANNERS"
