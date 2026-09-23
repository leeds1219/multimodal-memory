#!/usr/bin/env bash
# Creates the isolated conda env `optimus3` (py3.11) for Optimus-3 as released.
# Never touches /opt/conda/envs/mcagent. All pip installs use --no-cache-dir.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
R=/home/rag/data/official/Optimus-3
ENV=optimus3
C="$HERE/constraints.txt"
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n $ENV python=3.11 pip
fi
PIP="conda run --no-capture-output -n $ENV pip install --no-cache-dir"
# torch for Blackwell (sm_120)
$PIP --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128 torchvision==0.22.1+cu128
# repo requirements (unpinned upstream; pins from constraints.txt)
# av==11.0.0 (requirements.txt) has no py3.11 wheel -> av from constraints.txt (12.3.0)
grep -v '^av==' "$R/requirements.txt" > /tmp/optimus3_requirements.txt
$PIP -c "$C" --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/optimus3_requirements.txt transformers==4.51.3 av
# LLaMA-Factory: only imported for _register_composite_model; 0.9.3 allows transformers 4.51.3
$PIP -c "$C" --extra-index-url https://download.pytorch.org/whl/cu128 llamafactory==0.9.3
# Optimus-3's MineStudio fork (contains random_ore / workers) + the minecraftoptimus package
$PIP -c "$C" --extra-index-url https://download.pytorch.org/whl/cu128 --no-deps -e "$R/MineStudio"
$PIP -c "$C" --no-deps -e "$R"
$PIP -c "$C" pyyaml pillow
# flash-attn is NOT installed (no sm_120 wheel); attention patched to sdpa (see optimus3.patch)
conda run -n $ENV python -c "import torch,transformers,llamafactory;print(torch.__version__,torch.cuda.get_arch_list(),transformers.__version__,llamafactory.__version__)"
