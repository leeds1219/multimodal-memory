#!/usr/bin/env bash
# One accumulation run (no evaluation stage): N episodes on the fixed split with KB
# writes ON, checkpoints every 50. Two of these run side by side (coordinates off/on).
#
#   CUDA_VISIBLE_DEVICES=2 TAG=off PORT=9000 N=400 bash scripts/accumulate_pair.sh
#   CUDA_VISIBLE_DEVICES=4 TAG=on  PORT=9010 N=400 NEARBY=1 bash scripts/accumulate_pair.sh
#
# Env: TAG (store/log name), PORT, N (episodes), NEARBY=1 to expose block coordinates
# + the approach primitive, BENCHMARK (wooden), LLM (gemini_flash), USD_PER_EPISODE.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
TAG="${TAG:?set TAG}"; PORT="${PORT:-9000}"; N="${N:-400}"
BENCHMARK="${BENCHMARK:-wooden}"; LLM="${LLM:-gemini_flash}"
USD_PER_EPISODE="${USD_PER_EPISODE:-0.078}"     # measured after the prompt compaction
STORE="memories/${TAG}"
export PATH="$HERE/scripts:$PATH"
log() { echo "[$TAG $(date +%H:%M:%S)] $*"; }

if [[ -f logs/llm_guard/${TAG}/PAUSED || -f logs/llm_guard/${TAG}/ABORT ]]; then
  log "guard marker present for $TAG - review first"; exit 1
fi
if [[ -z "${RESUME_DIR:-}" && -d "$STORE" && -n "$(ls -A "$STORE" 2>/dev/null)" ]]; then
  log "store $STORE is not empty - set TAG to something new (or RESUME_DIR=...)"; exit 1
fi
mkdir -p "$STORE" "logs/llm_guard/${TAG}"
CAP=$(python -c "print(round($N * $USD_PER_EPISODE * 1.3, 2))")
log "guard session cap \$$CAP for $N episodes"

MINEEVOLVE_MEMORY_PATH="$STORE" MINEEVOLVE_PORT="$PORT" \
MINEEVOLVE_GUARD_MAX_USD_TOTAL="$CAP" MINEEVOLVE_GUARD_DIR="logs/llm_guard/${TAG}" \
MINEEVOLVE_LLM_LOG="logs/llm_calls_${TAG}.jsonl" \
  nohup bash scripts/server_gemini.sh > "logs/server_${TAG}.log" 2>&1 &
for _ in $(seq 1 60); do sleep 5; grep -q "Uvicorn running" "logs/server_${TAG}.log" && break; done
grep -q "Uvicorn running" "logs/server_${TAG}.log" || { log "server did not start"; exit 1; }
log "server up on :$PORT (store $STORE, nearby=${NEARBY:-0})"

MINEEVOLVE_NEARBY_BLOCKS="${NEARBY:-0}" \
MINEEVOLVE_JVM_OPTS="${MINEEVOLVE_JVM_OPTS:--Dmineevolve.landmarks.exposedOnly=true}" \
MINEEVOLVE_LLM_LOG="logs/llm_calls_${TAG}.jsonl" \
  xvfb-run -a python -m mineevolve.main benchmark="$BENCHMARK" llm="$LLM" server.port="$PORT" \
    accumulate.episodes="$N" accumulate.kb_store_dir="$STORE" ${RESUME_DIR:+resume_dir=$RESUME_DIR} \
    || log "main exited non-zero"
RUN="$(ls -td logs/eval/*/* | head -1)"
pkill -f "uvicorn ap[p]:app --host 0.0.0.0 --port ${PORT}" || true
log "done -> $RUN ($(wc -l < "$RUN/runs.jsonl" 2>/dev/null || echo 0) episodes)"
