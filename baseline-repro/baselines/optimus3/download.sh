#!/usr/bin/env bash
# Non-interactive download of Optimus-3 (June-2025 preview) weights + simulator engine.
# PY: any python with huggingface_hub (default: the optimus3 env).
set -euo pipefail
W=/home/rag/data/official/optimus3_weights
PY=${PY:-"conda run --no-capture-output -n optimus3 python"}
mkdir -p $W
free=$(df --output=avail -BG /home/rag/data | tail -1 | tr -dc 0-9)
[ "$free" -gt 60 ] || { echo "only ${free}G free, need >60G (30G floor + ~25G)"; exit 1; }
HF_HUB_DISABLE_PROGRESS_BARS=1 $PY - <<PYEOF
import os, zipfile
from huggingface_hub import snapshot_download, hf_hub_download
W = "$W"
repos = {
    "Optimus-3":            ("iLearn-Lab/Optimus-3", "6168839d8a44c3fab45a31354e683874d14601f3", None),  # preview MLLM (2025-06/07)
    "Optimus-3-ActionHead": ("MinecraftOptimus/Optimus-3-ActionHead", "455e01e30c8e830f420179c93e64f54ce3b30ecc", None),
    "Optimus-3-Task-Router":("MinecraftOptimus/Optimus-3-Task-Router", "ec555ce2d2e769e4cdd6372306084b78b3787bc4", None),
    "sentence-bert-base":   ("efederici/sentence-bert-base", "d34e611b4983aeecfa5beee5f7e2ef1f595a5944", None),
    # MineCLIP text tokenizer (tokenization.py hard-codes a /data7 copy of it)
    "clip-vit-base-patch16-tokenizer": ("openai/clip-vit-base-patch16", None,
        ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]),
}
for d, (rid, rev, pat) in repos.items():
    p = snapshot_download(rid, revision=rev, local_dir=os.path.join(W, d), allow_patterns=pat)
    print("ok", rid, p)
ms = os.path.join(W, "minestudio_dir")
if not os.path.exists(os.path.join(ms, "engine", "build", "libs", "mcprec-6.13.jar")):
    hf_hub_download(repo_id="CraftJarvis/SimulatorEngine", filename="engine.zip", local_dir=ms)
    with zipfile.ZipFile(os.path.join(ms, "engine.zip")) as z:
        z.extractall(ms)
    os.remove(os.path.join(ms, "engine.zip"))
print("engine ok")
PYEOF
du -sh $W/*
