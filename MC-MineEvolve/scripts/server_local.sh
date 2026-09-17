#!/usr/bin/env bash
# Start the MineEvolve server against the local LLM (scripts/local_llm_server.py)
# instead of a paid API. Same server code as scripts/server.sh; only the LLM
# endpoint differs.
#
#   CUDA_VISIBLE_DEVICES=3 python scripts/local_llm_server.py --model Qwen/Qwen3.5-2B &   # shell 1
#   CUDA_VISIBLE_DEVICES=2 bash scripts/server_local.sh                                   # shell 2
#   xvfb-run -a bash scripts/run_eval.sh wooden local                                     # shell 3
#
# Env overrides: LOCAL_LLM_URL (default http://127.0.0.1:8001/v1), MINEEVOLVE_PORT.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_LLM_URL="${LOCAL_LLM_URL:-http://127.0.0.1:8001/v1}"

until curl -sf "${LOCAL_LLM_URL%/v1}/health" >/dev/null; do
  echo "[server_local] waiting for local LLM at ${LOCAL_LLM_URL} ..." >&2
  sleep 5
done

cd "$HERE"
MINEEVOLVE_LLM_PROVIDER=openai_compat \
MINEEVOLVE_LLM_MODEL=local \
MINEEVOLVE_LLM_BASE_URL="$LOCAL_LLM_URL" \
OPENAI_API_KEY="${OPENAI_API_KEY:-local}" \
  exec bash scripts/server.sh
