"""Hyperparameters, read once from a config file (the only place they are set): configs/hyperparams.yaml, or
the file $MM_HYPERPARAMS names (e.g. configs/hyperparams_debug.yaml for short debug runs).

    from hyperparams import HP
    HP["loop"]["rollouts"]
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
FILE = Path(os.environ.get("MM_HYPERPARAMS") or ROOT / "configs" / "hyperparams.yaml")
HP = yaml.safe_load((FILE if FILE.is_absolute() else ROOT / FILE).read_text())
