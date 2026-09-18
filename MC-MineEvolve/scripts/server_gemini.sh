#!/usr/bin/env bash
# Start the MineEvolve server with Gemini as the planner LLM.
#
#   export GOOGLE_API_KEY=...            # keep it in ~/.bashrc.dosung, never in the repo
#   CUDA_VISIBLE_DEVICES=2 bash scripts/server_gemini.sh          # shell 1
#   xvfb-run -a bash scripts/run_eval.sh wooden gemini_flash      # shell 2
#
# Same server as scripts/server.sh; this only sets the three LLM env vars that
# must change together (provider + model + base_url — server/api.py defaults
# base_url to DashScope, so setting the provider alone sends Gemini requests
# to the wrong host). Gemini 3.x are thinking models: their hidden reasoning
# counts against max_tokens, and the server default of 1536 truncates the plan
# JSON, so this raises MINEEVOLVE_LLM_MAX_TOKENS to 16384 and caps thinking with
# MINEEVOLVE_LLM_REASONING_EFFORT=low (a repair call once thought ~7.8k tokens).
# The knowledge store defaults to memories/<model> so runs with different
# models do not share (and retrieve) each other's induced skills/remedies.
# MINEEVOLVE_KB_FROZEN=1 evaluates with a frozen knowledge base (paper's cold-start
# / checkpoint protocol); start from an empty memories/<model> for a true cold start.
# Override the model with MINEEVOLVE_LLM_MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$HERE/../.env" ]]; then
  set -a; source <(grep -vE '^\s*(#|$)' "$HERE/../.env" | sed -E 's/^export //'); set +a
fi
if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "[server_gemini] GOOGLE_API_KEY is not set" >&2
  exit 1
fi

cd "$HERE"
MINEEVOLVE_LLM_PROVIDER=gemini \
MINEEVOLVE_LLM_MODEL="${MINEEVOLVE_LLM_MODEL:-gemini-3-flash-preview}" \
MINEEVOLVE_LLM_BASE_URL="${MINEEVOLVE_LLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}" \
MINEEVOLVE_LLM_MAX_TOKENS="${MINEEVOLVE_LLM_MAX_TOKENS:-16384}" \
MINEEVOLVE_LLM_REASONING_EFFORT="${MINEEVOLVE_LLM_REASONING_EFFORT:-low}" \
MINEEVOLVE_MEMORY_PATH="${MINEEVOLVE_MEMORY_PATH:-memories/${MINEEVOLVE_LLM_MODEL:-gemini-3-flash-preview}}" \
  exec bash scripts/server.sh
