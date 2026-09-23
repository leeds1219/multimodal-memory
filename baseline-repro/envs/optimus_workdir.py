"""Per-chain working directory for Optimus-1 (its paths are cwd-relative).

    <wd>/app.py, checkpoints, src/optimus1/helper   -> symlinks to the official clone
    <wd>/src/optimus1/memories/v1                    -> this chain's own memory (real dir)
    <wd>/imgs, <wd>/api/imgs                         -> server step images / example images
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

OPT = Path("/home/rag/data/official/NeurIPS24-Optimus-1")
FULL_MEMORY = Path("/home/rag/data/official/optimus1_full_memory/v1")


def make_workdir(wd: Path, memory: str) -> Path:
    """memory: 'empty' or 'prebuilt' (the memory shipped in the repo, memories/v1)."""
    wd = Path(wd)
    (wd / "src" / "optimus1").mkdir(parents=True, exist_ok=True)
    for name, target in (("app.py", OPT / "app.py"), ("checkpoints", OPT / "checkpoints"),
                         ("src/optimus1/helper", OPT / "src/optimus1/helper")):
        link = wd / name
        if not link.exists():
            link.symlink_to(target)
    mem = wd / "src" / "optimus1" / "memories" / "v1"
    if not mem.exists():
        if memory == "prebuilt":
            # The authors' full released memory (HF MinecraftOptimus/Optimus1_Memory,
            # DECISIONS D21b). Images are hard-linked (never modified); JSON files
            # are real copies because Optimus-1 rewrites them in place.
            def _copy(src, dst):
                if str(src).endswith(".json"):
                    return shutil.copy2(src, dst)
                return os.link(src, dst)
            shutil.copytree(FULL_MEMORY, mem, copy_function=_copy)
        elif memory == "empty":
            (mem / "reflection" / "img").mkdir(parents=True)
            (mem / "plan" / "success").mkdir(parents=True)
        else:
            raise ValueError(memory)
    (wd / "imgs").mkdir(exist_ok=True)
    (wd / "api" / "imgs").mkdir(parents=True, exist_ok=True)
    return wd


if __name__ == "__main__":
    print(make_workdir(Path(sys.argv[1]), sys.argv[2]))
