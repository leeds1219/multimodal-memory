#!/usr/bin/env bash
# Usage: start_mineevolve_server.sh <gpu> <port> <kb_dir> <ctx_file> <log_file>
# Starts the MineEvolve FastAPI server (STEVE-1 + Gemini via the shared layer).
set -euo pipefail
GPU=$1; PORT=$2; KB=$3; CTX=$4; LOG=$5
CK=/home/rag/data/official/NeurIPS24-Optimus-1/checkpoints
mkdir -p "$KB"
cd /home/rag/data/multimodal-memory/MC-MineEvolve
export CUDA_VISIBLE_DEVICES=$GPU LLM_CTX_FILE=$CTX METHOD=mineevolve \
  MINEEVOLVE_LLM_PROVIDER=gemini MINEEVOLVE_LLM_MODEL=see-baseline-repro-configs-llm.yaml \
  MINEEVOLVE_MEMORY_PATH=$KB \
  MINEEVOLVE_VPT_MODEL=$CK/vpt/2x.model MINEEVOLVE_STEVE_WEIGHTS=$CK/steve1/steve1.weights \
  MINEEVOLVE_STEVE_PRIOR=$CK/steve1/steve1_prior.pt MINEEVOLVE_MINECLIP_WEIGHTS=$CK/mineclip/attn.pth
exec /opt/conda/envs/mcagent/bin/uvicorn app:app --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1
