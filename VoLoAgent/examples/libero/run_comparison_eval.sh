#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a side-by-side comparison of two orchestrator modes on original LIBERO:
#   Mode A: passthrough (baseline, no orchestration)
#   Mode B: subgoal + GT failure monitor + grasp_first recovery
#
# Configuration:
#   - All 4 suites: libero_spatial, libero_object, libero_goal, libero_10
#   - 3 episodes per task
#   - Pi0.5 LIBERO model on port 8002 (already running)
#
# Ports used:
#   8002  - pi0.5 LIBERO policy server (already running)
#   8013  - orchestrator proxy for passthrough
#   8014  - orchestrator proxy for subgoal+gt+grasp
#
# Usage:
#   cd ~/vlm-orchestrator
#   bash examples/libero/run_comparison_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration ──────────────────────────────────────────────
VLA_PORT=8002                       # LIBERO pi0.5 policy server
PROXY_PORT_PASSTHROUGH=8013
PROXY_PORT_SUBGOAL=8014
NUM_TRIALS=3
SEED=7
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="results/libero/comparison_${TIMESTAMP}"
SUITES="libero_spatial libero_object libero_goal libero_10"

LIBERO_VENV="$PROJECT_DIR/.libero-venv/bin/python"
export PYTHONPATH="${HOME}/openpi/third_party/libero:${PYTHONPATH:-}"

echo "═══════════════════════════════════════════════════════════"
echo " LIBERO Comparison Eval: passthrough vs subgoal+gt+grasp"
echo "═══════════════════════════════════════════════════════════"
echo " Policy server:  port $VLA_PORT (pi0.5 LIBERO)"
echo " Suites:         $SUITES"
echo " Trials/task:    $NUM_TRIALS"
echo " Output:         $LOG_ROOT"
echo " Timestamp:      $TIMESTAMP"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "$LOG_ROOT"

# ── Helper: start proxy in background ─────────────────────────
start_proxy() {
    local mode=$1
    local port=$2
    local log_dir=$3
    local extra_args="${4:-}"

    echo "[proxy] Starting $mode proxy on port $port..."
    mkdir -p "$log_dir"

    $LIBERO_VENV -m vlm_orchestrator.cli \
        --vla-port "$VLA_PORT" \
        --port "$port" \
        --env libero \
        --mode "$mode" \
        --log-dir "$log_dir/proxy_logs" \
        $extra_args \
        > "$log_dir/proxy_stdout.log" 2>&1 &

    local pid=$!
    echo "[proxy] PID=$pid, waiting for startup..."
    sleep 3

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[proxy] ERROR: proxy failed to start. Check $log_dir/proxy_stdout.log"
        cat "$log_dir/proxy_stdout.log" | tail -20
        return 1
    fi

    # Wait for health check
    local retries=10
    while [ $retries -gt 0 ]; do
        if curl -s "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
            echo "[proxy] $mode proxy ready on port $port (PID=$pid)"
            echo "$pid" > "$log_dir/proxy.pid"
            return 0
        fi
        sleep 1
        retries=$((retries - 1))
    done
    echo "[proxy] ERROR: proxy health check failed after 10s"
    kill "$pid" 2>/dev/null || true
    return 1
}

# ── Helper: stop proxy ────────────────────────────────────────
stop_proxy() {
    local log_dir=$1
    if [ -f "$log_dir/proxy.pid" ]; then
        local pid=$(cat "$log_dir/proxy.pid")
        echo "[proxy] Stopping proxy PID=$pid..."
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        rm -f "$log_dir/proxy.pid"
    fi
}

# ── Helper: run eval for one mode ─────────────────────────────
run_eval() {
    local mode_name=$1
    local proxy_port=$2
    local log_dir=$3

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo " Running eval: $mode_name"
    echo "────────────────────────────────────────────────────────"

    for suite in $SUITES; do
        echo ""
        echo "  Suite: $suite"
        local suite_log="$log_dir/$suite"
        local suite_video="$log_dir/videos/$suite"
        mkdir -p "$suite_log" "$suite_video"

        local extra_eval_flags=""
        if [ "$mode_name" = "subgoal_gt_grasp" ]; then
            extra_eval_flags="--enable-gt-state --enable-depth"
        fi

        $LIBERO_VENV "$PROJECT_DIR/examples/libero/libero_eval_client.py" \
            --host 127.0.0.1 \
            --port "$proxy_port" \
            --task-suite-name "$suite" \
            --num-trials-per-task "$NUM_TRIALS" \
            --log-dir "$suite_log" \
            --video-out-path "$suite_video" \
            --seed "$SEED" \
            $extra_eval_flags \
            2>&1 | tee "$suite_log/eval_stdout.log"

        echo "  ✓ $suite complete"
    done
}

# ── Trap: cleanup proxies on exit ─────────────────────────────
cleanup() {
    echo ""
    echo "[cleanup] Stopping proxies..."
    stop_proxy "$LOG_ROOT/passthrough"
    stop_proxy "$LOG_ROOT/subgoal_gt_grasp"
}
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════
#  PHASE 1: Passthrough baseline
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " PHASE 1: Passthrough (baseline)"
echo "══════════════════════════════════════════════════════════"

start_proxy passthrough $PROXY_PORT_PASSTHROUGH "$LOG_ROOT/passthrough"
run_eval passthrough $PROXY_PORT_PASSTHROUGH "$LOG_ROOT/passthrough"
stop_proxy "$LOG_ROOT/passthrough"

# ══════════════════════════════════════════════════════════════
#  PHASE 2: Subgoal + GT failure monitor + grasp_first
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " PHASE 2: Subgoal + GT + grasp_first"
echo "══════════════════════════════════════════════════════════"

start_proxy subgoal $PROXY_PORT_SUBGOAL "$LOG_ROOT/subgoal_gt_grasp" \
    "--failure-monitor gt --recovery-mode grasp_first --grasp-seg-mode gdino_sam2"
run_eval subgoal_gt_grasp $PROXY_PORT_SUBGOAL "$LOG_ROOT/subgoal_gt_grasp"
stop_proxy "$LOG_ROOT/subgoal_gt_grasp"

# ══════════════════════════════════════════════════════════════
#  PHASE 3: Compare results
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " Generating comparison report..."
echo "══════════════════════════════════════════════════════════"

$LIBERO_VENV "$PROJECT_DIR/examples/libero/compare_results.py" \
    --mode-a-dir "$LOG_ROOT/passthrough" \
    --mode-a-name "passthrough" \
    --mode-b-dir "$LOG_ROOT/subgoal_gt_grasp" \
    --mode-b-name "subgoal+gt+grasp" \
    --output "$LOG_ROOT/COMPARISON.md" \
    2>&1

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " Done! Results in: $LOG_ROOT"
echo "═══════════════════════════════════════════════════════════"
echo " $LOG_ROOT/COMPARISON.md    — side-by-side comparison"
echo " $LOG_ROOT/passthrough/     — baseline results"
echo " $LOG_ROOT/subgoal_gt_grasp/ — orchestrated results"
echo "═══════════════════════════════════════════════════════════"
