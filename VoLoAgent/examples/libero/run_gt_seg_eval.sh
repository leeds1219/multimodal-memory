#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the LIBERO eval with GT segmentation for the grasp tool.
#
# Identical to the subgoal+gt+grasp setup used in existing evaluations
# (run_comparison_eval.sh) EXCEPT:
#   --grasp-seg-mode gt_sim   (instead of gdino_sam2)
#
# This replaces GDino+SAM2 with perfect simulator segmentation masks,
# isolating the impact of perception failures on grasp tool success.
#
# Compare results against the existing gdino_sam2 runs in:
#   results/libero/comparison_20260405_230119/subgoal_gt_grasp/
#
# Usage:
#   # Same setup as the existing comparison eval
#   bash examples/libero/run_gt_seg_eval.sh
#
#   # Quick test with 1 trial
#   bash examples/libero/run_gt_seg_eval.sh --num-trials 1
#
#   # Single suite
#   bash examples/libero/run_gt_seg_eval.sh --suites libero_10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Match existing comparison eval setup exactly ───────────────
VLA_PORT="${VLA_PORT:-8002}"
PROXY_PORT=8014
NUM_TRIALS=3            # same as comparison eval
SEED=7                  # same as comparison eval
SUITES_LIST="libero_spatial libero_object libero_goal libero_10"

LIBERO_VENV="${LIBERO_VENV:-$PROJECT_DIR/.libero-venv/bin/python}"
if [ ! -f "$LIBERO_VENV" ]; then
    LIBERO_VENV="$(which python3)"
fi
export PYTHONPATH="${HOME}/openpi/third_party/libero:${PYTHONPATH:-}"

# ── Parse overrides ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-trials) NUM_TRIALS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --suites) SUITES_LIST="$2"; shift 2 ;;
        --vla-port) VLA_PORT="$2"; shift 2 ;;
        --proxy-port) PROXY_PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="results/libero/gt_seg_eval_${TIMESTAMP}"

echo "═══════════════════════════════════════════════════════════"
echo " LIBERO Eval: subgoal + GT monitor + grasp_first + GT_SIM"
echo "═══════════════════════════════════════════════════════════"
echo " Policy server:  port $VLA_PORT (pi0.5 LIBERO)"
echo " Proxy port:     $PROXY_PORT"
echo " Suites:         $SUITES_LIST"
echo " Trials/task:    $NUM_TRIALS"
echo " Seed:           $SEED"
echo " Output:         $LOG_ROOT"
echo ""
echo " Only difference vs existing eval:"
echo "   --grasp-seg-mode gt_sim  (was gdino_sam2)"
echo ""
echo " Compare against:"
echo "   results/libero/comparison_20260405_230119/subgoal_gt_grasp/"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "$LOG_ROOT"

# ── Start proxy ───────────────────────────────────────────────
echo ""
echo "[proxy] Starting subgoal+gt+grasp proxy with GT_SIM seg..."
mkdir -p "$LOG_ROOT/proxy_logs"

$LIBERO_VENV -m vlm_orchestrator.cli \
    --vla-port "$VLA_PORT" \
    --port "$PROXY_PORT" \
    --env libero \
    --mode subgoal \
    --failure-monitor gt \
    --recovery-mode grasp_first \
    --grasp-seg-mode gt_sim \
    --log-dir "$LOG_ROOT/proxy_logs" \
    > "$LOG_ROOT/proxy_stdout.log" 2>&1 &

PROXY_PID=$!
echo "[proxy] PID=$PROXY_PID, waiting for startup..."
sleep 4

if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "[proxy] ERROR: proxy failed to start. Last 20 lines:"
    tail -20 "$LOG_ROOT/proxy_stdout.log"
    exit 1
fi
echo "[proxy] Ready on port $PROXY_PORT"

# ── Trap: cleanup proxy on exit ───────────────────────────────
cleanup() {
    echo ""
    echo "[cleanup] Stopping proxy PID=$PROXY_PID..."
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Run eval ──────────────────────────────────────────────────
for suite in $SUITES_LIST; do
    echo ""
    echo "────────────────────────────────────────────────────────"
    echo " Suite: $suite ($NUM_TRIALS trials/task)"
    echo "────────────────────────────────────────────────────────"

    suite_log="$LOG_ROOT/$suite"
    suite_video="$LOG_ROOT/videos/$suite"
    mkdir -p "$suite_log" "$suite_video"

    $LIBERO_VENV "$PROJECT_DIR/examples/libero/run_eval.py" \
        --host 127.0.0.1 \
        --port "$PROXY_PORT" \
        --task-suite-name "$suite" \
        --num-trials-per-task "$NUM_TRIALS" \
        --log-dir "$suite_log" \
        --video-out-path "$suite_video" \
        --seed "$SEED" \
        --enable-gt-state \
        --enable-depth \
        2>&1 | tee "$suite_log/eval_stdout.log"

    echo "  ✓ $suite complete"
done

# ── Stop proxy ────────────────────────────────────────────────
echo ""
echo "[proxy] Stopping..."
kill "$PROXY_PID" 2>/dev/null || true
wait "$PROXY_PID" 2>/dev/null || true
trap - EXIT

# ── Generate comparison against existing gdino_sam2 results ───
EXISTING_RESULTS="results/libero/comparison_20260405_230119/subgoal_gt_grasp"

echo ""
echo "══════════════════════════════════════════════════════════"
echo " Comparing GT_SIM vs existing GDino+SAM2 results"
echo "══════════════════════════════════════════════════════════"

if [ -d "$EXISTING_RESULTS" ]; then
    $LIBERO_VENV "$PROJECT_DIR/examples/libero/compare_results.py" \
        --mode-a-dir "$EXISTING_RESULTS" \
        --mode-a-name "GDino+SAM2 (existing)" \
        --mode-b-dir "$LOG_ROOT" \
        --mode-b-name "GT Sim Seg (new)" \
        --output "$LOG_ROOT/COMPARISON_vs_gdino.md" \
        2>&1 || echo "(comparison script failed — check results manually)"

    echo ""
    if [ -f "$LOG_ROOT/COMPARISON_vs_gdino.md" ]; then
        cat "$LOG_ROOT/COMPARISON_vs_gdino.md"
    fi
else
    echo "Existing GDino+SAM2 results not found at:"
    echo "  $EXISTING_RESULTS"
    echo "Run run_comparison_eval.sh first, or update the path above."
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Done! Results in: $LOG_ROOT"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo " $LOG_ROOT/COMPARISON_vs_gdino.md  — GT_SIM vs GDino+SAM2"
echo " $LOG_ROOT/<suite>/results.json    — per-suite results"
echo " $LOG_ROOT/proxy_stdout.log        — proxy logs"
echo ""
echo " If GT_SIM >> GDino+SAM2:"
echo "   → Perception failures ARE a bottleneck"
echo "   → Better detection/segmentation would improve recovery"
echo ""
echo " If GT_SIM ≈ GDino+SAM2:"
echo "   → Perception is NOT the bottleneck"
echo "   → Failures are from grasp planning, IK, or execution"
echo "═══════════════════════════════════════════════════════════"
