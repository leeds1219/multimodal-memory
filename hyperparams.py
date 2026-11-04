"""Hyperparameters, read once from configs/hyperparams.yaml (the only place they are set):

    from hyperparams import HP
    HP["loop"]["rollouts"]
"""
from pathlib import Path

import yaml

HP = yaml.safe_load((Path(__file__).resolve().parent / "configs" / "hyperparams.yaml").read_text())
