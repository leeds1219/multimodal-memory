#!/usr/bin/env bash
# Start the MineEvolve FastAPI server (loads STEVE-1 + LLM planner backend).
#
# Default LLM backend is Qwen Plus via DashScope's OpenAI-compatible endpoint.
# Set the matching API key for whichever backend you want to use:
#
#   DASHSCOPE_API_KEY   - Qwen   (DEFAULT)
#   ZHIPUAI_API_KEY     - GLM    (set MINEEVOLVE_LLM_PROVIDER=glm)
#   GOOGLE_API_KEY      - Gemini (set MINEEVOLVE_LLM_PROVIDER=gemini)
#   OPENAI_API_KEY      - GPT    (set MINEEVOLVE_LLM_PROVIDER=openai)
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES        - GPU index for STEVE-1 inference
#   MINEEVOLVE_LLM_PROVIDER     - one of {qwen, glm, gemini, openai, openai_compat}
#   MINEEVOLVE_LLM_MODEL        - model id (e.g. qwen-plus, qwen-flash, glm-4-plus, ...)
#   MINEEVOLVE_LLM_BASE_URL     - override the OpenAI-compatible endpoint
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# API keys live in a git-ignored .env at the monorepo root (or next to this
# sub-project); shell-exported variables take precedence over the file.
for envfile in "$HERE/../.env" "$HERE/.env"; do
  if [[ -f "$envfile" ]]; then
    set -a; # shellcheck disable=SC1090
    source <(grep -vE '^\s*(#|$)' "$envfile" | sed -E 's/^export //')
    set +a
  fi
done

PORT="${MINEEVOLVE_PORT:-9000}"
HOST="${MINEEVOLVE_HOST:-0.0.0.0}"

if [[ -z "${DASHSCOPE_API_KEY:-}" \
   && -z "${OPENAI_API_KEY:-}" \
   && -z "${ZHIPUAI_API_KEY:-}" \
   && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "[warn] No LLM API key in environment; planner calls will fail." >&2
  echo "[hint] Default backend is Qwen: export DASHSCOPE_API_KEY=sk-..." >&2
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  uvicorn app:app --host "${HOST}" --port "${PORT}"
