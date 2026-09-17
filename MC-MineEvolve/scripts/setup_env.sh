#!/usr/bin/env bash
# Build the `mineevolve` conda env from scratch on a Linux + NVIDIA box.
# No root needed: Java 8 and Xvfb come from conda-forge.
#
#   bash scripts/setup_env.sh              # create/refresh env "mineevolve"
#   ENV_NAME=me2 bash scripts/setup_env.sh # different env name
#   CUDA_TAG=cu118 bash scripts/setup_env.sh
#
# What it installs (and why, since several steps are non-obvious):
#   - python 3.10, openjdk 8 (MineRL's MCP-Reborn build), Xvfb + Mesa (headless software GL;
#     the xvfb-run shim puts $CONDA_PREFIX/lib on LD_LIBRARY_PATH so LWJGL finds it)
#   - torch/torchvision wheels for $CUDA_TAG (default cu121: works with driver >= 530)
#   - gym==0.23.1 built with setuptools<66 (its setup.py breaks on newer setuptools)
#   - MineRL v1.0.2 from GitHub (PyPI only has 0.4.x, which lacks HumanSurvival);
#     this compiles Minecraft via Gradle: ~10-20 min and needs internet
#   - a Java patch on MineRL (patches/, via scripts/patch_minerl.sh): adds the "chat"
#     action verb + allowCommands so /gamerule, /setblock etc. actually work
#   - MineStudio (bundles STEVE-1 + HF weights loader)
#   - libGL / glib from conda-forge: MineStudio hard-pins opencv-python==4.8.0.74,
#     whose wheel needs libGL.so.1 and libgthread-2.0.so.0 that bare servers lack
#   - an `xvfb-run` shim in the env's bin/ (conda's Xvfb package ships none)
#
# Re-running is safe; every step is idempotent.
set -euo pipefail

ENV_NAME="${ENV_NAME:-mineevolve}"
CUDA_TAG="${CUDA_TAG:-cu121}"
MINERL_REF="${MINERL_REF:-v1.0.2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "##### [1/7] conda env '$ENV_NAME' (python 3.10 + openjdk 8 + Xvfb + Mesa software GL) #####"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -q -n "$ENV_NAME" python=3.10 pip
fi
conda install -y -q -n "$ENV_NAME" -c conda-forge openjdk=8 xorg-xvfb-server mesalib libglib libgl libglvnd
conda activate "$ENV_NAME"
java -version 2>&1 | head -1

echo "##### [2/7] torch ($CUDA_TAG) #####"
pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "##### [3/7] core requirements (minus torch / MineStudio, which get their own steps) #####"
REQ_TMP="$(mktemp)"
grep -viE '^(torch|MineStudio)' "$HERE/requirements.txt" > "$REQ_TMP"
pip install -q -r "$REQ_TMP" "numpy>=1.21,<2.0"
rm -f "$REQ_TMP"

echo "##### [4/7] gym 0.23.1 (old setuptools in the build env) #####"
CONSTRAINTS="$(mktemp)"
printf 'setuptools<66\nwheel\n' > "$CONSTRAINTS"
PIP_CONSTRAINT="$CONSTRAINTS" pip install -q "gym==0.23.1"

echo "##### [5/7] MineRL $MINERL_REF from GitHub (Gradle build, be patient) #####"
if ! python -c "import minerl" 2>/dev/null; then
  PIP_CONSTRAINT="$CONSTRAINTS" pip install "git+https://github.com/minerllabs/minerl@${MINERL_REF}"
fi
rm -f "$CONSTRAINTS"

echo "##### [5b/7] patch MineRL: add chat commands + allowCommands, rebuild jar #####"
bash "$HERE/scripts/patch_minerl.sh"

echo "##### [6/7] MineStudio + this package (editable) #####"
pip install -q MineStudio
pip install -q --no-deps -e "$HERE"

echo "##### [7/7] xvfb-run shim #####"
install -m755 "$HERE/scripts/xvfb-run" "$CONDA_PREFIX/bin/xvfb-run"

echo "##### verify #####"
python - <<'PY'
import torch, cv2, numpy, gym, minerl, minestudio, hydra, fastapi, openai
import mineevolve
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival  # what env/custom_env.py needs
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("numpy", numpy.__version__, "cv2", cv2.__version__, "gym", gym.__version__, "minerl", "1.0.x" if hasattr(minerl, "herobraine") else "?")
print("mineevolve import OK")
PY
xvfb-run -a python -c "import os; print('xvfb-run OK, DISPLAY=' + os.environ['DISPLAY'])"
echo "##### DONE: conda activate $ENV_NAME #####"
