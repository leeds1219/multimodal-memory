"""Import paths of the repository. Every entry script starts with

    sys.path.insert(0, str(Path(__file__).resolve().parents[<depth>]))
    import paths  # noqa: F401

after which modules of every code folder import by their file name (`import benchmark`, `from controller import ...`).
The domain's folders (domains/<MM_DOMAIN>/, default minecraft) are on the path too; `import domains` gives the
domain object (domains/__init__.py).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Data volume outside the repository: the authors' repos (official/), the conda env, caches. Default: the folder
# that holds this repository; set MM_DATA to use another one.
DATA = Path(os.environ.get("MM_DATA", ROOT.parent))
OFFICIAL = DATA / "official"
DOMAIN = os.environ.get("MM_DOMAIN", "minecraft")
RESULTS = ROOT / "results"  # results/<method>/<world>/... and results/ours_loop/ (git-ignored)
RUN_DATA = ROOT / "run_data"  # memory stores, chain state, logs, LLM ledger (git-ignored)
CODE_DIRS = ("executor", "designer", "memory", "experiments", "experiments/optimus1", "analysis", "llm", "scripts", "setup",
             f"domains/{DOMAIN}", f"domains/{DOMAIN}/world", f"domains/{DOMAIN}/agent", f"domains/{DOMAIN}/agent/tools")
OPTIMUS1 = OFFICIAL / "NeurIPS24-Optimus-1"  # the authors' code, patched by setup/setup_env.sh

for d in ("",) + CODE_DIRS:
    p = str(ROOT / d)
    if p not in sys.path:
        sys.path.insert(0, p)
