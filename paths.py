"""Import paths of the repository. Every entry script starts with

    sys.path.insert(0, str(Path(__file__).resolve().parents[<depth>]))
    import paths  # noqa: F401

after which modules of every code folder import by their file name (`import benchmark`, `from controller import ...`).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Data volume outside the repository: the authors' repos (official/), the conda env, caches. Default: the folder
# that holds this repository; set MM_DATA to use another one.
DATA = Path(os.environ.get("MM_DATA", ROOT.parent))
OFFICIAL = DATA / "official"
CODE_DIRS = ("world", "system1", "system1/tools", "system2", "memory", "experiments", "experiments/optimus1", "analysis", "llm",
             "scripts", "setup")
OPTIMUS1 = OFFICIAL / "NeurIPS24-Optimus-1"  # the authors' code, patched by setup/setup_env.sh

for d in CODE_DIRS:
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)
