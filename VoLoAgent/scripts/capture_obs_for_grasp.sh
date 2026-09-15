#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Capture ONE real robolab obs (with depth + camera pose) by running a short
# passthrough eval with GRASP_OBS_DUMP set on the orchestrator. Kills the eval
# as soon as the obs lands. Then scripts/validate_grasp_from_obs.py replays it
# through the grasp tool against a live grasp server.
set -e
VLA_PORT=8000; ORCH_PORT=8001
OBS_DUMP=${OBS_DUMP:-/tmp/grasp_obs.pkl}
TASKS=${TASKS:-RecoverNonFoodInBinTask}
LOGS=/tmp/capture_obs; rm -rf "$LOGS"; mkdir -p "$LOGS"
rm -f "$OBS_DUMP"
for p in $VLA_PORT $ORCH_PORT; do fuser -k ${p}/tcp 2>/dev/null || true; done
sleep 2
PIDS=()
cleanup(){ for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
  for p in $VLA_PORT $ORCH_PORT; do fuser -k ${p}/tcp 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
wait_log(){ local f="$1" pat="$2" t="${3:-600}" s; s=$(date +%s)
  while ! grep -q -- "$pat" "$f" 2>/dev/null; do
    (( $(date +%s)-s > t )) && { echo "❌ timeout: $pat"; return 1; }; sleep 2; done; echo "✓ $pat"; }

echo "[cap] VLA..."
( cd "$HOME/openpi" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
   conda run --no-capture-output -n openpi uv run scripts/serve_policy.py \
     --port $VLA_PORT policy:checkpoint --policy.config=pi05_droid_jointpos \
     --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
) > "$LOGS/vla.log" 2>&1 &
PIDS+=($!)
wait_log "$LOGS/vla.log" "server listening on 0.0.0.0:$VLA_PORT" 600

echo "[cap] orchestrator (passthrough, GRASP_OBS_DUMP=$OBS_DUMP)..."
GRASP_OBS_DUMP="$OBS_DUMP" conda run --no-capture-output -n vlm-orch \
  vlm-orchestrator --env robolab --vla-host 127.0.0.1 --vla-port $VLA_PORT \
    --port $ORCH_PORT --mode passthrough --log-dir "$LOGS" --verbose \
  > "$LOGS/orch.log" 2>&1 &
PIDS+=($!)
sleep 5

echo "[cap] robolab eval (will auto-stop once obs dumped)..."
( cd "$HOME/robolab" && PYTHONPATH="$HOME/robolab" conda run --no-capture-output -n robolab \
    python policies/volo/run.py --headless \
      --remote-host 127.0.0.1 --remote-port $ORCH_PORT --num-runs 1 \
      --enable-gt-state --policy pi05 --task-dirs robovolo \
      --output-folder-name capture_obs_tmp --task $TASKS
) > "$LOGS/eval.log" 2>&1 &
EVAL_PID=$!; PIDS+=($EVAL_PID)

echo "[cap] waiting for obs dump at $OBS_DUMP ..."
for i in $(seq 1 180); do
  [ -f "$OBS_DUMP" ] && { echo "✓ obs captured ($(stat -c%s "$OBS_DUMP") bytes)"; break; }
  sleep 2
done
[ -f "$OBS_DUMP" ] || { echo "❌ no obs dumped"; exit 1; }
echo "[cap] done — obs at $OBS_DUMP"
