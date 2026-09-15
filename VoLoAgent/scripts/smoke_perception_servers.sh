#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Two-phase local smoke test of the perception servers used by tool_chain.
#
# Phase A — Molmo2 path (bf16):
#   grasp_server with --enable-sam2 --enable-molmo --molmo-quantize bf16
#   Sends /v1/chat/completions to the Molmo2 sidecar (point prompts) and
#   /segment to the grasp server (SAM2 from a point).
#
# Phase B — SAM3 path:
#   grasp_server with --enable-sam3
#   Sends /detect_and_segment.
#
# No Isaac Sim, no robolab eval — just verifies the server stack each
# bf16 Molmo2 needs ~17 GB, SAM2 ~1 GB, SAM3 ~10 GB;
# fits comfortably on a 48 GB card with nothing else.
#
# Outputs land in $HOME/vlm-orchestrator/debug_perception/server_smoke/.

set -euo pipefail

ROOT="$HOME/vlm-orchestrator"
OUT="$ROOT/debug_perception/server_smoke"
INPUT="$ROOT/debug_perception/input_frame.png"
mkdir -p "$OUT"

if [ ! -f "$INPUT" ]; then
  echo "[smoke] $INPUT missing — abort"; exit 1
fi

GRASP_PORT=8003
MOLMO_PORT=8122

# Pre-flight: kill anything bound to our ports.
for port in $GRASP_PORT $MOLMO_PORT; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 2

wait_port() {
  local label="$1" port="$2" timeout="${3:-180}"
  local start=$(date +%s)
  while ! python3 -c "import socket; s=socket.socket(); s.settimeout(1); \
                      s.connect(('127.0.0.1', $port)); s.close()" 2>/dev/null; do
    if (( $(date +%s) - start > timeout )); then
      echo "[smoke] ❌ $label port $port never came up after ${timeout}s"
      return 1
    fi
    sleep 2
  done
  echo "[smoke] ✓ $label ready on :$port"
}

# ─── Locate gripper config ───────────────────────────────────────────
GRIPPER_CFG=""
for cand in \
  "$HOME/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml" \
  "$HOME/graspgen/models/checkpoints/graspgen_franka_panda.yml"; do
  if [ -f "$cand" ]; then GRIPPER_CFG="$cand"; break; fi
done
[ -z "$GRIPPER_CFG" ] && { echo "[smoke] gripper cfg not found"; exit 1; }
echo "[smoke] gripper cfg: $GRIPPER_CFG"

# ─── Phase A: Molmo2 path (bf16) ─────────────────────────────────────
echo
echo "═════════════════════════════════════════════"
echo "  Phase A: Molmo2 + SAM2 (bf16)"
echo "═════════════════════════════════════════════"
GRASP_LOG_A="$OUT/grasp_server_A.log"
conda run --no-capture-output -n graspgen \
  python -m vlm_orchestrator.grasp.server \
    --gripper-config "$GRIPPER_CFG" \
    --port $GRASP_PORT \
    --enable-sam2 --sam2-model facebook/sam2.1-hiera-small \
    --enable-molmo --molmo-model allenai/Molmo2-8B \
    --molmo-port $MOLMO_PORT --molmo-quantize bf16 \
  > "$GRASP_LOG_A" 2>&1 &
GRASP_PID=$!

cleanup_A() {
  kill "$GRASP_PID" 2>/dev/null || true
  for port in $GRASP_PORT $MOLMO_PORT; do
    fuser -k "${port}/tcp" 2>/dev/null || true
  done
  sleep 3
}
trap cleanup_A EXIT INT TERM

wait_port "grasp server (A)" $GRASP_PORT 60
wait_port "Molmo2 shim (A)"  $MOLMO_PORT 300

echo
echo "[smoke] running molmo2 + sam2 client tests..."
conda run --no-capture-output -n vlm-orch python - "$INPUT" "$OUT" <<'PYEOF'
import base64, io, json, sys, time
from pathlib import Path

import numpy as np
import PIL.Image
import requests

input_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

img = np.array(PIL.Image.open(input_path).convert("RGB"))
H, W = img.shape[:2]

# 1) Molmo2 /v1/chat/completions via vlm_orchestrator client
from vlm_orchestrator.perception.molmo import point_at, MolmoPointError
results = {"molmo2_points": [], "sam2_segments": []}
for phrase in ["orange", "white bowl", "empty space near the orange"]:
    try:
        t0 = time.time()
        p = point_at(img, phrase, base_url="http://127.0.0.1:8122/v1")
        dt = time.time() - t0
        results["molmo2_points"].append({
            "phrase": phrase, "x_norm": p.x_norm, "y_norm": p.y_norm,
            "latency_s": round(dt, 2),
        })
        print(f"  [molmo2] '{phrase}' → ({p.x_norm:.3f}, {p.y_norm:.3f})  {dt:.1f}s")
    except MolmoPointError as e:
        results["molmo2_points"].append({"phrase": phrase, "error": str(e)})
        print(f"  [molmo2] '{phrase}' FAILED: {e}")

# 2) SAM2 /segment via grasp server
buf = io.BytesIO()
np.savez_compressed(
    buf, image=img,
    point_x=np.float32(0.5), point_y=np.float32(0.5),
)
buf.seek(0)
t0 = time.time()
r = requests.post(
    "http://127.0.0.1:8003/segment",
    data=buf.read(),
    headers={"Content-Type": "application/octet-stream"},
    timeout=60,
)
dt = time.time() - t0
if r.status_code == 200:
    data = dict(np.load(io.BytesIO(r.content), allow_pickle=True))
    mask = data["mask"].astype(bool)
    iou = float(data["iou_score"])
    results["sam2_segments"].append({
        "center_point": [0.5, 0.5],
        "iou": iou, "mask_sum": int(mask.sum()),
        "latency_s": round(dt, 2),
    })
    print(f"  [sam2] center-point → iou={iou:.3f} sum={mask.sum()} {dt:.1f}s")
else:
    print(f"  [sam2] FAILED: {r.status_code} {r.text[:200]}")
    results["sam2_segments"].append({"error": f"{r.status_code} {r.text[:200]}"})

(out_dir / "phase_A_results.json").write_text(json.dumps(results, indent=2))
print(f"[smoke] wrote {out_dir / 'phase_A_results.json'}")
PYEOF

echo "[smoke] tearing down phase A..."
cleanup_A

# ─── Phase B: SAM3 path ──────────────────────────────────────────────
echo
echo "═════════════════════════════════════════════"
echo "  Phase B: SAM3"
echo "═════════════════════════════════════════════"
GRASP_LOG_B="$OUT/grasp_server_B.log"
conda run --no-capture-output -n graspgen \
  python -m vlm_orchestrator.grasp.server \
    --gripper-config "$GRIPPER_CFG" \
    --port $GRASP_PORT \
    --enable-sam3 \
  > "$GRASP_LOG_B" 2>&1 &
GRASP_PID=$!

cleanup_B() {
  kill "$GRASP_PID" 2>/dev/null || true
  fuser -k "${GRASP_PORT}/tcp" 2>/dev/null || true
  sleep 3
}
trap cleanup_B EXIT INT TERM

wait_port "grasp server (B)" $GRASP_PORT 120

echo
echo "[smoke] running sam3 client tests..."
conda run --no-capture-output -n vlm-orch python - "$INPUT" "$OUT" <<'PYEOF'
import io, json, sys, time
from pathlib import Path

import numpy as np
import PIL.Image
import requests

input_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
img = np.array(PIL.Image.open(input_path).convert("RGB"))

results = []
for phrase in ["orange", "white bowl", "rubiks cube"]:
    buf = io.BytesIO()
    np.savez_compressed(
        buf, image=img,
        text_prompt=np.array(phrase),
    )
    buf.seek(0)
    t0 = time.time()
    r = requests.post(
        "http://127.0.0.1:8003/detect_and_segment",
        data=buf.read(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=60,
    )
    dt = time.time() - t0
    if r.status_code == 200:
        data = dict(np.load(io.BytesIO(r.content), allow_pickle=True))
        mask = data["mask"].astype(bool)
        score = float(data["score"])
        box = data["box"].tolist()
        n = int(data["num_detections"])
        results.append({
            "phrase": phrase, "score": score, "box": box,
            "mask_sum": int(mask.sum()),
            "num_detections": n, "latency_s": round(dt, 2),
        })
        print(f"  [sam3] '{phrase}' → score={score:.3f} box={box} sum={mask.sum()} n={n} {dt:.1f}s")
    else:
        results.append({"phrase": phrase, "error": f"{r.status_code} {r.text[:200]}"})
        print(f"  [sam3] '{phrase}' FAILED: {r.status_code} {r.text[:200]}")

(out_dir / "phase_B_results.json").write_text(json.dumps(results, indent=2))
print(f"[smoke] wrote {out_dir / 'phase_B_results.json'}")
PYEOF

echo "[smoke] tearing down phase B..."
cleanup_B
trap - EXIT INT TERM

echo
echo "═════════════════════════════════════════════"
echo "  Smoke complete."
echo "  Results: $OUT/phase_A_results.json"
echo "           $OUT/phase_B_results.json"
echo "  Server logs: $OUT/grasp_server_{A,B}.log"
echo "═════════════════════════════════════════════"
