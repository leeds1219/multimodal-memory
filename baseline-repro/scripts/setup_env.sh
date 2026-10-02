#!/usr/bin/env bash
# Rebuild the Optimus-1 reproduction setup from scratch (DECISIONS D5, D39). Idempotent.
#   bash scripts/setup_env.sh
#
# Python packages are pinned to the ORIGINAL env's `pip freeze` (envs/mcagent.freeze.txt,
# copied from the container the published runs used; envs/mcagent.lock.txt = the same
# minus the editable lines). Editable repos are cloned at the commits that freeze lists
# and our patches (baselines/optimus1/*.patch) are applied on top.
#
# The env lives on the data volume (survives container resets); /opt/conda/envs/mcagent,
# hard-coded in chain.py / supervise.py / start_optimus_server.sh, is a symlink to it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OFF=/home/rag/data/official
ENV=${MCAGENT_ENV:-/home/rag/data/conda_envs/mcagent}
LINK=/opt/conda/envs/mcagent
export PIP_CACHE_DIR=/home/rag/data/.cache/pip CONDA_PKGS_DIRS=/home/rag/data/.cache/conda_pkgs

# --- system: Java 8 (MineRL), Xvfb, GL/glib (OpenCV) ----------------------------------
{ command -v java && java -version 2>&1 | grep -q '"1.8' && command -v Xvfb && ldconfig -p | grep -q libgthread-2.0; } >/dev/null || {
  apt-get update -qq
  apt-get install -y -qq openjdk-8-jdk-headless xvfb xauth libgl1 libglu1-mesa libglib2.0-0 git curl unzip >/dev/null; }

# --- code: authors' repos at the pinned commits + our patches ---------------------------
clone() {  # url dir commit
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  [ "$(git -C "$2" rev-parse HEAD)" = "$3" ] || git -C "$2" checkout -q "$3"; }
patch_once() {  # dir patch
  git -C "$1" apply --reverse --check "$2" 2>/dev/null || git -C "$1" apply "$2"; }
clone https://github.com/iLearn-Lab/NeurIPS24-Optimus-1.git "$OFF/NeurIPS24-Optimus-1" 409356cc324a3bf64f31653c6bc0f926b672f19c
clone https://github.com/MineDojo/MineCLIP "$OFF/MineCLIP" e6c06a0245fac63dceb38bc9bd4fecd033dae735
clone https://github.com/Shalev-Lifshitz/STEVE-1.git "$OFF/STEVE-1" 903b244796322f4d0073a8f62c05f51eac3aed52
patch_once "$OFF/NeurIPS24-Optimus-1" "$HERE/baselines/optimus1/optimus1.patch"
patch_once "$OFF/MineCLIP" "$HERE/baselines/optimus1/mineclip_compat.patch"

# --- python env ---------------------------------------------------------------------------
[ -x "$ENV/bin/python" ] || /opt/conda/bin/conda create -y -q -p "$ENV" -c conda-forge python=3.10.21
[ -e "$LINK" ] || { mkdir -p "$(dirname "$LINK")"; ln -s "$ENV" "$LINK"; }
PIP="$ENV/bin/python -m pip"
$PIP install -q -r "$HERE/envs/mcagent.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu128

# --- simulator: the authors' prebuilt MCP-Reborn (never rebuild it) ----------------------
# minerl/setup.py runs prep_mcp() unless READTHEDOCS=1: setup_mcp.sh does `rm -rf MCP-Reborn`,
# a fresh clone and a Gradle build, destroying the prebuilt jar (happened once, D39).
MCP="$OFF/NeurIPS24-Optimus-1/minerl/minerl/MCP-Reborn"
JAR="$MCP/build/libs/mcprec-6.13.jar"
JAR_SHA=ebd503ddd2aa525d53ff19dd987c043ab8708cf0da28188ed0a03f5357fdd095
if [ ! -f "$JAR" ]; then  # authors' tarball, README "download our MCP-Reborn"
  TGZ=$OFF/_downloads/MCP-Reborn.tar.gz; mkdir -p "$(dirname "$TGZ")"
  [ -f "$TGZ" ] || { $PIP install -q gdown; "$ENV/bin/python" -m gdown 1GLy9IpFq5CQOubH7q60UhYCvD6nwU_YG -O "$TGZ"; }
  rm -rf "$MCP"; tar -xzf "$TGZ" -C "$(dirname "$MCP")"
fi
echo "$JAR_SHA  $JAR" | sha256sum -c --quiet

$PIP install -q hatchling  # build backend of optimus1 (pyproject), needed with --no-build-isolation
READTHEDOCS=1 $PIP install -q --no-build-isolation --no-deps -e "$OFF/NeurIPS24-Optimus-1/minerl"
$PIP install -q --no-build-isolation --no-deps -e "$OFF/NeurIPS24-Optimus-1"
$PIP install -q --no-build-isolation --no-deps -e "$OFF/MineCLIP"
$PIP install -q --no-build-isolation --no-deps -e "$OFF/STEVE-1"

# --- checkpoints and memory -----------------------------------------------------------------
if [ ! -d "$OFF/NeurIPS24-Optimus-1/checkpoints/steve1" ]; then  # README "download steve1 checkpoint"
  mkdir -p "$OFF/_downloads"; $PIP install -q gdown
  "$ENV/bin/python" -m gdown 1Mmwqv2juxMuP1xOZYWucnbKopMk0c0DV -O "$OFF/_downloads/optimus1_steve1_ckpt.zip"
  unzip -q -o "$OFF/_downloads/optimus1_steve1_ckpt.zip" -d "$OFF/NeurIPS24-Optimus-1"
fi
if [ ! -d "$OFF/optimus1_full_memory/v1" ]; then  # variant `prebuilt` (D21b): HF MinecraftOptimus/Optimus1_Memory
  echo "missing $OFF/optimus1_full_memory/v1: download HF dataset MinecraftOptimus/Optimus1_Memory," \
       "unpack it there and run baselines/optimus1/repair_memory_json.py (memory_repair_report.txt)" >&2
  exit 1
fi

# --- check ----------------------------------------------------------------------------------
"$ENV/bin/python" - <<'EOF'
import torch, transformers, minerl, gym, numpy, optimus1, steve1, mineclip
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__,
      "| gym", gym.__version__, "| numpy", numpy.__version__)
EOF
java -version 2>&1 | head -1
