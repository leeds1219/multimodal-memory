#!/usr/bin/env bash
# Exit as soon as either accumulation run needs attention, so the caller is notified:
#   guard paused · server gone · main process gone · no new episode for 40 min · finished.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
OFF="${1:?off run dir}"; ON="${2:?on run dir}"
last_off=0; last_on=0; stall_off=0; stall_on=0
while true; do
  for f in logs/llm_guard/*/PAUSED; do
    [ -e "$f" ] && { echo "GUARD PAUSED: $f"; head -3 "$f"; exit 0; }
  done
  n_off=$(wc -l < "$OFF/runs.jsonl" 2>/dev/null || echo 0)
  n_on=$(wc -l < "$ON/runs.jsonl" 2>/dev/null || echo 0)
  [ "$n_off" = "$last_off" ] && stall_off=$((stall_off+1)) || stall_off=0
  [ "$n_on" = "$last_on" ] && stall_on=$((stall_on+1)) || stall_on=0
  last_off=$n_off; last_on=$n_on
  pgrep -f "uvicorn ap[p]:app --host 0.0.0.0 --port 9000" >/dev/null || { echo "OFF server gone (episodes: $n_off/$n_on)"; exit 0; }
  pgrep -f "uvicorn ap[p]:app --host 0.0.0.0 --port 9010" >/dev/null || { echo "ON server gone (episodes: $n_off/$n_on)"; exit 0; }
  [ "$(pgrep -cf 'mineevolve.mai[n]')" -lt 2 ] && { echo "a main process exited (episodes: $n_off/$n_on)"; exit 0; }
  [ "$stall_off" -ge 8 ] && { echo "OFF stalled 40 min at $n_off episodes"; exit 0; }
  [ "$stall_on" -ge 8 ] && { echo "ON stalled 40 min at $n_on episodes"; exit 0; }
  [ "$n_off" -ge 400 ] && [ "$n_on" -ge 400 ] && { echo "both runs finished"; exit 0; }
  sleep 300
done
