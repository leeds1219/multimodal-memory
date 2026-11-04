"""Tool: hold raw keys for N steps (the VLM's "keys" command)."""
from __future__ import annotations

import re

KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use")
MAX_STEPS = 100


def run(ctl, cmd: dict, tag: str) -> str:
    """Hold `cmd["keys"]` for `cmd["steps"]` steps; the camera turns once and hotbar keys press once, at the first step."""
    import numpy as np
    ks = [str(k) for k in (cmd.get("keys") or [])]
    cam = cmd.get("camera") or [0, 0]
    try:
        cam = [max(-90.0, min(90.0, float(cam[0]))), max(-180.0, min(180.0, float(cam[1])))]
    except Exception:
        cam = [0.0, 0.0]
    n = max(1, min(int(cmd.get("steps", 1) or 1), MAX_STEPS))
    hot = [k for k in ks if re.fullmatch(r"hotbar\.[1-9]", k)]
    press = {k: 1 for k in ks if k in KEYS}
    ctl.env.can_change_hotbar = True
    try:
        for i in range(n):
            a = ctl.noop()
            for k in press:
                a[k] = np.array(1)
            if i == 0:
                for h in hot:
                    a[h] = np.array(1)
                a["camera"] = np.asarray(cam, dtype=np.float32)
            _, s, done = ctl.step(a, tag)
            if done or ctl.over():
                return "ended"
    finally:
        ctl.env.can_change_hotbar = False
    return "done"
