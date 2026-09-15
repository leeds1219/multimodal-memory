#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Compare grasp tool segmentation modes on LIBERO benchmarks:
#   Mode A: subgoal + GT monitor + grasp_first + GDino/SAM2 (current)
#   Mode B: subgoal + GT monitor + grasp_first + GT segmentation (new)
#
# Both modes use identical VLM subgoal decomposition, GT failure
# monitoring, and grasp-first recovery. The ONLY difference is how
# the grasp tool segments the target object:
#   A: GroundingDINO detection → SAM2 segmentation (visual, noisy)
#   B: Ground-truth segmentation from MuJoCo renderer (perfect)
#
# By comparing A vs B, we isolate the impact of perception failures
# (wrong object detection, bad masks) on overall task success rate.
#
# Supports all LIBERO family benchmarks:
#   --benchmark libero          → libero_spatial, libero_object, libero_goal, libero_10
#   --benchmark libero_plus     → all LIBERO-Plus perturbed tasks
#   --benchmark libero_pro      → LIBERO-PRO generalization tests
#
# Configuration:
#   - Pi0.5 LIBERO model on port 8002 (must be running)
#   - Grasp server on port 8003 (must be running for Mode A)
#   - VLM API key configured (for subgoal decomposition)
#
# Usage:
#   # Basic: original LIBERO, 3 trials per task
#   bash examples/libero/run_gt_seg_comparison.sh
#
#   # Full benchmark: 50 trials per task
#   bash examples/libero/run_gt_seg_comparison.sh --num-trials 50
#
#   # Single suite only
#   bash examples/libero/run_gt_seg_comparison.sh --suites libero_10
#
#   # LIBERO-Plus
#   bash examples/libero/run_gt_seg_comparison.sh --benchmark libero_plus --num-trials 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Default configuration ──────────────────────────────────────
VLA_PORT="${VLA_PORT:-8002}"
GRASP_SERVER_PORT="${GRASP_SERVER_PORT:-8003}"
PROXY_PORT_GDINO=8015
PROXY_PORT_GTSIM=8016
NUM_TRIALS=3
SEED=7
BENCHMARK="libero"
SUITES=""
EXTRA_PROXY_ARGS=""
EXTRA_EVAL_ARGS=""

LIBERO_VENV="${LIBERO_VENV:-$PROJECT_DIR/.libero-venv/bin/python}"
if [ ! -f "$LIBERO_VENV" ]; then
    LIBERO_VENV="$(which python3)"
fi
export PYTHONPATH="${HOME}/openpi/third_party/libero:${PYTHONPATH:-}"

# ── Parse arguments ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-trials) NUM_TRIALS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --suites) SUITES="$2"; shift 2 ;;
        --vla-port) VLA_PORT="$2"; shift 2 ;;
        --grasp-port) GRASP_SERVER_PORT="$2"; shift 2 ;;
        --proxy-port-a) PROXY_PORT_GDINO="$2"; shift 2 ;;
        --proxy-port-b) PROXY_PORT_GTSIM="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="results/libero/gt_seg_comparison_${TIMESTAMP}"

# ── Resolve suites ─────────────────────────────────────────────
if [ -z "$SUITES" ]; then
    case "$BENCHMARK" in
        libero)
            SUITES_LIST="libero_spatial libero_object libero_goal libero_10"
            ;;
        libero_plus|libero_pro)
            # Let run_extended_benchmarks.py handle suite resolution
            SUITES_LIST=""
            ;;
        *)
            echo "Unknown benchmark: $BENCHMARK"
            exit 1
            ;;
    esac
else
    SUITES_LIST="$SUITES"
fi

echo "═══════════════════════════════════════════════════════════════"
echo " GT Segmentation Comparison Eval"
echo "═══════════════════════════════════════════════════════════════"
echo " Benchmark:    $BENCHMARK"
echo " Suites:       ${SUITES_LIST:-'(auto from benchmark)'}"
echo " Trials/task:  $NUM_TRIALS"
echo " Policy:       port $VLA_PORT (pi0.5 LIBERO)"
echo " Grasp server: port $GRASP_SERVER_PORT"
echo " Output:       $LOG_ROOT"
echo ""
echo " Mode A: subgoal + GT monitor + grasp_first + GDino/SAM2"
echo "   Proxy port: $PROXY_PORT_GDINO"
echo ""
echo " Mode B: subgoal + GT monitor + grasp_first + GT sim seg"
echo "   Proxy port: $PROXY_PORT_GTSIM"
echo "═══════════════════════════════════════════════════════════════"

mkdir -p "$LOG_ROOT"

# Save config
cat > "$LOG_ROOT/eval_config.json" <<EOF
{
    "benchmark": "$BENCHMARK",
    "suites": "$SUITES_LIST",
    "num_trials": $NUM_TRIALS,
    "seed": $SEED,
    "vla_port": $VLA_PORT,
    "grasp_server_port": $GRASP_SERVER_PORT,
    "timestamp": "$TIMESTAMP",
    "modes": {
        "A": "subgoal + gt_monitor + grasp_first + gdino_sam2",
        "B": "subgoal + gt_monitor + grasp_first + gt_sim"
    }
}
EOF

# ── Helper: start proxy in background ─────────────────────────
start_proxy() {
    local mode_label=$1
    local port=$2
    local log_dir=$3
    local seg_mode=$4

    echo "[proxy] Starting $mode_label proxy on port $port (seg=$seg_mode)..."
    mkdir -p "$log_dir"

    $LIBERO_VENV -m vlm_orchestrator.cli \
        --vla-port "$VLA_PORT" \
        --port "$port" \
        --env libero \
        --mode subgoal \
        --failure-monitor gt \
        --recovery-mode grasp_first \
        --grasp-seg-mode "$seg_mode" \
        --log-dir "$log_dir/proxy_logs" \
        > "$log_dir/proxy_stdout.log" 2>&1 &

    local pid=$!
    echo "[proxy] PID=$pid, waiting for startup..."
    sleep 4

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[proxy] ERROR: proxy failed to start. Last 20 lines:"
        tail -20 "$log_dir/proxy_stdout.log"
        return 1
    fi

    echo "$pid" > "$log_dir/proxy.pid"
    echo "[proxy] $mode_label proxy ready on port $port (PID=$pid)"
}

stop_proxy() {
    local log_dir=$1
    if [ -f "$log_dir/proxy.pid" ]; then
        local pid=$(cat "$log_dir/proxy.pid")
        echo "[proxy] Stopping PID=$pid..."
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        rm -f "$log_dir/proxy.pid"
    fi
}

# ── Helper: run eval for one mode ─────────────────────────────
run_eval_standard() {
    local mode_name=$1
    local proxy_port=$2
    local log_dir=$3

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo " Running $BENCHMARK eval: $mode_name"
    echo "────────────────────────────────────────────────────────"

    for suite in $SUITES_LIST; do
        echo "  Suite: $suite"
        local suite_log="$log_dir/$suite"
        local suite_video="$log_dir/videos/$suite"
        mkdir -p "$suite_log" "$suite_video"

        $LIBERO_VENV "$PROJECT_DIR/examples/libero/run_eval.py" \
            --host 127.0.0.1 \
            --port "$proxy_port" \
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
}

run_eval_extended() {
    local mode_name=$1
    local proxy_port=$2
    local log_dir=$3

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo " Running $BENCHMARK eval: $mode_name"
    echo "────────────────────────────────────────────────────────"

    $LIBERO_VENV "$PROJECT_DIR/examples/libero/run_extended_benchmarks.py" \
        --benchmark "$BENCHMARK" \
        --proxy-port "$proxy_port" \
        --num-trials "$NUM_TRIALS" \
        --log-dir "$log_dir" \
        --enable-depth \
        --seed "$SEED" \
        2>&1 | tee "$log_dir/eval_stdout.log"
}

run_eval() {
    if [ "$BENCHMARK" = "libero" ]; then
        run_eval_standard "$@"
    else
        run_eval_extended "$@"
    fi
}

# ── Trap: cleanup proxies on exit ─────────────────────────────
cleanup() {
    echo ""
    echo "[cleanup] Stopping proxies..."
    stop_proxy "$LOG_ROOT/mode_a_gdino_sam2"
    stop_proxy "$LOG_ROOT/mode_b_gt_sim"
}
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════
#  MODE A: GDino + SAM2 (current)
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " MODE A: subgoal + GT monitor + grasp_first + GDino/SAM2"
echo "══════════════════════════════════════════════════════════"

start_proxy "mode_a" $PROXY_PORT_GDINO "$LOG_ROOT/mode_a_gdino_sam2" "gdino_sam2"
run_eval "mode_a_gdino_sam2" $PROXY_PORT_GDINO "$LOG_ROOT/mode_a_gdino_sam2"
stop_proxy "$LOG_ROOT/mode_a_gdino_sam2"

# ══════════════════════════════════════════════════════════════
#  MODE B: GT sim segmentation (new)
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " MODE B: subgoal + GT monitor + grasp_first + GT sim seg"
echo "══════════════════════════════════════════════════════════"

start_proxy "mode_b" $PROXY_PORT_GTSIM "$LOG_ROOT/mode_b_gt_sim" "gt_sim"
run_eval "mode_b_gt_sim" $PROXY_PORT_GTSIM "$LOG_ROOT/mode_b_gt_sim"
stop_proxy "$LOG_ROOT/mode_b_gt_sim"

# ══════════════════════════════════════════════════════════════
#  COMPARE
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " Generating comparison report..."
echo "══════════════════════════════════════════════════════════"

$LIBERO_VENV "$PROJECT_DIR/examples/libero/compare_results.py" \
    --mode-a-dir "$LOG_ROOT/mode_a_gdino_sam2" \
    --mode-a-name "GDino+SAM2" \
    --mode-b-dir "$LOG_ROOT/mode_b_gt_sim" \
    --mode-b-name "GT Sim Seg" \
    --output "$LOG_ROOT/COMPARISON.md" \
    2>&1 || echo "(comparison script failed — check results manually)"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Done! Results in: $LOG_ROOT"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo " $LOG_ROOT/COMPARISON.md          — side-by-side comparison"
echo " $LOG_ROOT/mode_a_gdino_sam2/     — GDino+SAM2 results"
echo " $LOG_ROOT/mode_b_gt_sim/         — GT sim seg results"
echo " $LOG_ROOT/eval_config.json       — eval configuration"
echo ""
echo " Key question answered by this comparison:"
echo "   Does wrong object detection / bad segmentation in the grasp"
echo "   tool pipeline contribute to task failures?"
echo ""
echo "   If Mode B >> Mode A:  YES — perception failures are a major"
echo "     bottleneck, and GT segmentation (or better detection)"
echo "     would significantly improve recovery success."
echo ""
echo "   If Mode B ≈ Mode A:  NO — perception is not the bottleneck."
echo "     Failures are from grasp planning, IK, or execution."
echo "═══════════════════════════════════════════════════════════════"
