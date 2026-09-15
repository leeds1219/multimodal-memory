#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Local A/B: stack=TRUE vs stack=FALSE placement, tool_chain mode, cuRobo planner.
#
# Two modes:
#   MODE=force   (default) — forces STACK_FORCE=true then =false, overriding the
#                 VLM. Isolates the effect of stack orientation + grasp top-down
#                 filter coupling on the SAME tasks.
#   MODE=vlm     — no force; the VLM decides stack per object (natural mode).
#                 Runs ONE arm; the per-tool stack choices are in rewrites.jsonl.
#
# Both use --motion-planner curobo --enable-stack-mode and GRASP_COLLISION_FREE=0
# (plain cuRobo — validated working).
#
# Env-var overrides:
#   MODE           force | vlm                       (default force)
#   TASKS          robolab --task list
#   TASK_DIRS      robolab --task-dirs               (default "benchmark robovolo")
#   NUM_EPISODES   episodes per task per arm         (default 1)
#   RUN_NAME       output folder prefix              (default stack_ab)
#   USE_FRONT_CAMERA "1" → --use-front-camera        (default 0)
#   PLACE_SEG_MODE place pointer                     (default molmo_point;
#                  set PLACE_SEG_MODE=sam3 to reuse the grasp server's SAM3
#                  and avoid spinning up a separate Molmo2 vLLM (GPU saving))
#   GRASP_TOPDOWN_THRESHOLD                          (default 0.3)
#
# Outputs (per arm):
#   ~/vlm-orchestrator/results/<RUN_NAME>_<arm>/   (orchestrator + logs)
#   ~/robolab/output/<RUN_NAME>_<arm>/             (videos + hdf5)
# where <arm> ∈ {stack_true, stack_false} for MODE=force, or {vlm} for MODE=vlm.

set -e

GRASP_PORT=${GRASP_PORT:-8003}
ORCH_PORT=${ORCH_PORT:-8001}
MODE=${MODE:-force}
RUN_NAME=${RUN_NAME:-stack_ab}
# StackYellowOnRedTask (benchmark/) needs stack=True. RecoverFallenFruitTask
# (robovolo/) is a drop-into-bowl → stack=False should keep more grasps.
TASKS=${TASKS:-"StackYellowOnRedTask RecoverFallenFruitTask"}
TASK_DIRS=${TASK_DIRS:-"benchmark robovolo"}
NUM_EPISODES=${NUM_EPISODES:-1}
GRASP_TOPDOWN_THRESHOLD=${GRASP_TOPDOWN_THRESHOLD:-0.3}
USE_FRONT_CAMERA=${USE_FRONT_CAMERA:-0}

FRONT_CAM_FLAG=""
if [ "$USE_FRONT_CAMERA" = "1" ] || [ "$USE_FRONT_CAMERA" = "true" ]; then
  FRONT_CAM_FLAG="--use-front-camera"
fi

# Arms depend on MODE.
if [ "$MODE" = "vlm" ]; then
  ARMS="vlm"
else
  ARMS="stack_true stack_false"
fi

TOP_LOG_DIR="$HOME/vlm-orchestrator/results/${RUN_NAME}_services"
mkdir -p "$TOP_LOG_DIR"

# ─── Pre-flight ──────────────────────────────────────────────────────
for env in graspgen vlm-orch; do
  conda env list | awk '{print $1}' | grep -qx "$env" || { echo "[run] conda env '$env' missing"; exit 1; }
done
[ -x "$HOME/robolab/.venv/bin/python" ] || { echo "[run] robolab venv missing"; exit 1; }
conda run -n graspgen python -c "import curobo" 2>/dev/null || { echo "[run] cuRobo not importable in graspgen"; exit 1; }

for port in $GRASP_PORT $ORCH_PORT; do fuser -k "${port}/tcp" 2>/dev/null || true; done
sleep 2

PIDS=()
cleanup() {
  echo; echo "[run] cleaning up..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
  for port in $GRASP_PORT $ORCH_PORT; do fuser -k "${port}/tcp" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_port() {
  local label="$1" port="$2" timeout="${3:-180}" start; start=$(date +%s)
  while ! python3 -c "import socket;s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',$port));s.close()" 2>/dev/null; do
    (( $(date +%s) - start > timeout )) && { echo "[run] ❌ $label :$port timeout"; return 1; }
    sleep 2
  done
  echo "[run] ✓ $label ready on :$port"
}
wait_port_down() {
  local port="$1" timeout="${2:-30}" start; start=$(date +%s)
  while python3 -c "import socket;s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',$port));s.close()" 2>/dev/null; do
    (( $(date +%s) - start > timeout )) && break; sleep 1
  done
}

# ─── 1. Grasp server (SAM3 + GraspGen + cuRobo), shared by all arms ─
GRIPPER_CFG=""
for cand in \
  "$HOME/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml" \
  "$HOME/graspgen/models/checkpoints/graspgen_franka_panda.yml"; do
  [ -f "$cand" ] && { GRIPPER_CFG="$cand"; break; }
done
[ -z "$GRIPPER_CFG" ] && { echo "[run] gripper cfg not found"; exit 1; }
echo "[run] gripper cfg: $GRIPPER_CFG"
echo "[run] MODE=$MODE  ARMS='$ARMS'  TASKS='$TASKS'"

echo "[run] starting grasp server on :$GRASP_PORT (SAM3 + GraspGen + cuRobo) ..."
conda run --no-capture-output -n graspgen \
  python -m vlm_orchestrator.grasp.server \
    --gripper-config "$GRIPPER_CFG" --port $GRASP_PORT \
    --enable-sam3 --sam3-model facebook/sam3 \
    --enable-curobo --verbose \
  > "$TOP_LOG_DIR/grasp_server.log" 2>&1 &
PIDS+=($!)
wait_port "grasp server" $GRASP_PORT 300

export GRASP_SERVER_HOST=127.0.0.1
export GRASP_SERVER_PORT=$GRASP_PORT
# Plain cuRobo — collision-free path OFF (validated working).
export GRASP_COLLISION_FREE=0

# ─── 2. Per-arm (sequential) ────────────────────────────────────────
for ARM in $ARMS; do
  ARM_RUN="${RUN_NAME}_${ARM}"
  ORCH_LOG_DIR="$HOME/vlm-orchestrator/results/$ARM_RUN"
  rm -rf "$ORCH_LOG_DIR" "$HOME/robolab/output/$ARM_RUN"
  mkdir -p "$ORCH_LOG_DIR"

  # Stack force override per arm.
  case "$ARM" in
    stack_true)  export STACK_FORCE=true  ;;
    stack_false) export STACK_FORCE=false ;;
    vlm)         unset STACK_FORCE        ;;
  esac

  echo; echo "════════════════════════════════════════════════════════════"
  echo "[run] ARM: $ARM   (STACK_FORCE=${STACK_FORCE:-<unset,VLM decides>})"
  echo "════════════════════════════════════════════════════════════"

  fuser -k "${ORCH_PORT}/tcp" 2>/dev/null || true
  wait_port_down $ORCH_PORT 15

  echo "[run] starting orchestrator (curobo + stack-mode) on :$ORCH_PORT ..."
  conda run --no-capture-output -n vlm-orch \
    vlm-orchestrator \
      --port $ORCH_PORT \
      --mode tool_chain \
      --vlm-model YOUR_VLM_MODEL \
      --grasp-seg-mode sam3 \
      --place-seg-mode "${PLACE_SEG_MODE:-molmo_point}" \
      --grasp-topdown-threshold "$GRASP_TOPDOWN_THRESHOLD" \
      --motion-planner curobo \
      --enable-stack-mode \
      $FRONT_CAM_FLAG \
      --log-dir "$ORCH_LOG_DIR" \
      --verbose \
    > "$ORCH_LOG_DIR/orchestrator.log" 2>&1 &
  ORCH_PID=$!
  PIDS+=($ORCH_PID)
  wait_port "orchestrator[$ARM]" $ORCH_PORT 60

  echo "[run] launching robolab eval (tasks=$TASKS, num-runs=$NUM_EPISODES)"
  ( cd "$HOME/robolab" && OMNI_KIT_ACCEPT_EULA=Y \
    .venv/bin/python policies/volo/run.py \
      --headless --policy pi05 --open-loop-horizon 8 \
      --remote-host 127.0.0.1 --remote-port $ORCH_PORT \
      --task-dirs $TASK_DIRS \
      --task $TASKS \
      --num-runs "$NUM_EPISODES" \
      --enable-subtask --enable-gt-state \
      --video-mode all \
      --output-folder-name "$ARM_RUN" ) \
    || echo "[run] ⚠ arm '$ARM' eval exited non-zero (continuing)"

  kill "$ORCH_PID" 2>/dev/null || true
  wait_port_down $ORCH_PORT 15
  echo "[run] arm '$ARM' done → $ORCH_LOG_DIR"
done

echo; echo "[run] all arms done."
echo "[run] build the comparison page with:"
echo "  conda run -n vlm-orch python scripts/build_motion_planner_ab_page.py \\"
echo "    --run-prefix $RUN_NAME --planners $ARMS --tasks $TASKS \\"
echo "    --http-base http://198.51.100.10:8091 --out results/${RUN_NAME}.html"