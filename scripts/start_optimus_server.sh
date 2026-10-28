#!/usr/bin/env bash
# Optimus-1's server: STEVE-1 actions for every experiment, and the planner / reflector for the Optimus-1 reproduction.
# Usage: start_optimus_server.sh <gpu> <port> <workdir> <ctx_file> <log_file>
set -euo pipefail
GPU=$1; PORT=$2; WD=$3; CTX=$4; LOG=$5
cd "$WD"
export CUDA_VISIBLE_DEVICES=$GPU LLM_CTX_FILE=$CTX METHOD=optimus1
exec /opt/conda/envs/mcagent/bin/uvicorn app:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1
