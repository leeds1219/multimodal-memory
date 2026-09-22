#!/usr/bin/env bash
# One "block" of the paper's accumulation protocol, unattended:
#   1. accumulate N episodes (KB writes ON, fresh store) with checkpoints every 50
#   2. frozen evaluation of the M<N> checkpoint on the same task-seed split
# Each stage starts its own server (scripts/server_gemini.sh) and stops it afterwards.
#
#   CUDA_VISIBLE_DEVICES=2 nohup bash scripts/accumulate_block.sh 50 > logs/block50.out 2>&1 &
#
# Env: BLOCK_TAG (store/run name, default acc-<date>), MINEEVOLVE_PORT (9000),
#      BENCHMARK (wooden), LLM (gemini_flash), SKIP_EVAL=1 to stop after stage 1.
# Cost at Gemini 3 Flash list price: ~$0.12 per episode (see docs/reproduction-notes.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
N="${1:?usage: accumulate_block.sh <episodes>}"
TAG="${BLOCK_TAG:-acc-$(date +%Y%m%d)}"
PORT="${MINEEVOLVE_PORT:-9000}"
BENCHMARK="${BENCHMARK:-wooden}"
LLM="${LLM:-gemini_flash}"
STORE="memories/${TAG}"
export PATH="$HERE/scripts:$PATH"   # xvfb-run shim

log() { echo "[block $(date +%H:%M:%S)] $*"; }

# Spending guard (src/mineevolve/util/llm_guard.py): the server PAUSES for review
# when a stage exceeds its estimate (USD_PER_EPISODE x episodes x 1.3), when one
# episode exceeds $0.60 / 80 calls, or when an hour exceeds $3. Resume with
# `python scripts/llm_guard.py resume`, stop with `... abort`.
USD_PER_EPISODE="${USD_PER_EPISODE:-0.12}"
start_server() {  # $1 = store dir, $2 = frozen (0/1), $3 = episodes in this stage
  if pgrep -f "uvicorn ap[p]:app" >/dev/null; then
    log "a MineEvolve server is already running on this box - stop it first"; exit 1
  fi
  if [[ -f logs/llm_guard/PAUSED || -f logs/llm_guard/ABORT ]]; then
    log "logs/llm_guard has a PAUSED/ABORT marker - review it and run scripts/llm_guard.py resume first"; exit 1
  fi
  local cap; cap=$(python -c "print(round($3 * $USD_PER_EPISODE * 1.3, 2))")
  log "guard: session cap \$$cap for $3 episodes"
  MINEEVOLVE_MEMORY_PATH="$1" MINEEVOLVE_KB_FROZEN="$2" MINEEVOLVE_PORT="$PORT" MINEEVOLVE_GUARD_MAX_USD_TOTAL="$cap" \
    nohup bash scripts/server_gemini.sh > "logs/server_${TAG}_frozen$2.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    sleep 5
    grep -q "Uvicorn running" "logs/server_${TAG}_frozen$2.log" 2>/dev/null && { log "server up (pid $SERVER_PID, store $1, frozen=$2)"; return; }
  done
  log "server did not start"; exit 1
}
stop_server() {
  pkill -f "uvicorn ap[p]:app" || true
  sleep 3
}
latest_run_dir() { ls -td logs/eval/*/* | head -1; }

# ---- stage 1: accumulate --------------------------------------------------
# RESUME_DIR=logs/eval/<date>/<time> continues an interrupted stage 1 in place
# (the live store must hold the KB as of the last finished episode).
if [[ -n "${RESUME_DIR:-}" ]]; then
  DONE_EP=$(wc -l < "$RESUME_DIR/runs.jsonl")
  start_server "$STORE" 0 "$((N - DONE_EP))"
  log "stage 1: resuming $RESUME_DIR to $N episodes (store $STORE)"
  xvfb-run -a python -m mineevolve.main benchmark="$BENCHMARK" llm="$LLM" server.port="$PORT" \
    accumulate.episodes="$N" accumulate.kb_store_dir="$STORE" resume_dir="$RESUME_DIR" || log "stage 1 exited non-zero"
  RUN1="$RESUME_DIR"
else
  if [[ -e "$STORE" ]] && [[ -n "$(ls -A "$STORE" 2>/dev/null)" ]]; then
    log "store $STORE already exists and is not empty - refusing to accumulate into it (set BLOCK_TAG)"; exit 1
  fi
  mkdir -p "$STORE"
  start_server "$STORE" 0 "$N"
  log "stage 1: accumulate $N episodes into $STORE"
  xvfb-run -a python -m mineevolve.main benchmark="$BENCHMARK" llm="$LLM" server.port="$PORT" \
    accumulate.episodes="$N" accumulate.kb_store_dir="$STORE" || log "stage 1 exited non-zero"
  RUN1="$(latest_run_dir)"
fi
stop_server
if [[ ! -f "$RUN1/DONE" ]]; then
  log "stage 1 did not finish (no DONE in $RUN1) - resume with resume_dir=$RUN1"; exit 1
fi
log "stage 1 done: $RUN1  ($(wc -l < "$RUN1/runs.jsonl") episodes)"
python scripts/spend.py --since "$(date +%Y-%m-%d)" | tail -1

[[ "${SKIP_EVAL:-0}" == "1" ]] && exit 0

# ---- stage 2: frozen evaluation of M<N> -----------------------------------
CKPT="$RUN1/kb_checkpoints/M${N}"
[[ -d "$CKPT" ]] || { log "no checkpoint $CKPT"; exit 1; }
EVAL_STORE="memories/${TAG}-M${N}-frozen"
mkdir -p "$EVAL_STORE" && cp "$CKPT"/*.json "$EVAL_STORE"/
start_server "$EVAL_STORE" 1 33
log "stage 2: frozen evaluation of $CKPT"
xvfb-run -a python -m mineevolve.main benchmark="$BENCHMARK" llm="$LLM" server.port="$PORT" || log "stage 2 exited non-zero"
RUN2="$(latest_run_dir)"
stop_server
log "stage 2 done: $RUN2"
python scripts/analyze_failures.py "$RUN2" || true
python scripts/spend.py --since "$(date +%Y-%m-%d)" | tail -1
echo "BLOCK DONE $RUN1 $RUN2"
