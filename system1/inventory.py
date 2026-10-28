"""Reading MineRL observations."""
from __future__ import annotations


def f(x) -> float:
    import numpy as np
    return float(np.asarray(x))


def inv_counts(obs) -> dict:
    """Item -> count of the agent's inventory (air and empty stacks left out)."""
    inv = obs.get("inventory") or {}
    return {k: int(f(v)) for k, v in inv.items() if f(v) > 0 and k != "air"}
