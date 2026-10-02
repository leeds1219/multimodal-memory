#!/usr/bin/env bash
# Rebuild the `mcagent` env (DECISIONS D5) after a container reset.
# The env lives on the data volume so it survives resets; /opt/conda/envs/mcagent
# (hard-coded in chain.py / supervise.py / start_*_server.sh) is a symlink to it.
# Java 8 also comes from conda-forge into the env; Xvfb from apt (re-run after a reset).
#   bash scripts/setup_env.sh        (idempotent)
set -euo pipefail
OFF=/home/rag/data/official
ENV=/home/rag/data/conda_envs/mcagent
LINK=/opt/conda/envs/mcagent
export PIP_CACHE_DIR=/home/rag/data/.cache/pip CONDA_PKGS_DIRS=/home/rag/data/.cache/conda_pkgs

{ command -v Xvfb && ldconfig -p | grep -q libgthread-2.0; } >/dev/null || { apt-get update -qq && apt-get install -y -qq xvfb xauth libgl1 libglu1-mesa libglib2.0-0 >/dev/null; }

if [ ! -x "$ENV/bin/python" ]; then
  /opt/conda/bin/conda create -y -p "$ENV" -c conda-forge python=3.10 openjdk=8
fi
mkdir -p "$(dirname "$LINK")"; [ -e "$LINK" ] || ln -s "$ENV" "$LINK"
PIP="$ENV/bin/python -m pip"

$PIP install -q "setuptools<70" wheel
$PIP install -q torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
$PIP install -q "numpy==1.23.5" "gym==0.23.1" gym3 "transformers==4.46.3" "timm>=0.9.16" accelerate \
  sentencepiece attrdict einops "hydra-core>=1.3.2" omegaconf shortuuid rich opencv-python thefuzz \
  fastapi "uvicorn==0.32.0" "openai>=1.46.0" av attrs pyyaml jsonlines kornia matplotlib scikit-learn \
  dm-tree x-transformers psutil pydantic requests tqdm mock
# MineRL fork (Optimus-1) in develop mode: uses the authors' prebuilt MCP-Reborn jar, no Gradle build.
# READTHEDOCS=1 is REQUIRED: without it minerl/setup.py runs prep_mcp(), whose setup_mcp.sh does
# `rm -rf MCP-Reborn` + a fresh clone + Gradle rebuild, destroying the authors' prebuilt jar.
# If the jar is missing, restore it from the authors' tarball (README: Google Drive
# 1GLy9IpFq5CQOubH7q60UhYCvD6nwU_YG; jar sha256 ebd503dd...).
JAR="$OFF/NeurIPS24-Optimus-1/minerl/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar"
[ -f "$JAR" ] || { echo "missing $JAR - restore MCP-Reborn.tar.gz first" >&2; exit 1; }
$PIP install -q -r "$OFF/NeurIPS24-Optimus-1/minerl/requirements.txt" "numpy==1.23.5"
READTHEDOCS=1 $PIP install -q --no-build-isolation --no-deps -e "$OFF/NeurIPS24-Optimus-1/minerl"
sha256sum "$JAR"
$PIP install -q --no-deps -e "$OFF/NeurIPS24-Optimus-1"
for p in MineCLIP STEVE-1; do [ -f "$OFF/$p/setup.py" ] || [ -f "$OFF/$p/pyproject.toml" ] && $PIP install -q --no-build-isolation --no-deps -e "$OFF/$p"; done

"$ENV/bin/python" - <<'EOF'
import torch, minerl, gym, numpy, optimus1
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| gym", gym.__version__, "| numpy", numpy.__version__)
EOF
"$ENV/bin/java" -version 2>&1 | head -1
