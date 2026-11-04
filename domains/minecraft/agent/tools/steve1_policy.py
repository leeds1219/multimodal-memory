"""Tool: STEVE-1, a learned low-level policy, runs a short English instruction for N steps (the "steve" command).
Its actions come from the Optimus-1 server (scripts/start_optimus_server.sh), as Optimus-1 drives it."""
from __future__ import annotations

MAX_STEPS = 200


def run(ctl, prompt, n, tag: str) -> str:
    from optimus1.util import ServerAPI
    prompt = str(prompt or "").strip()[:120] or "stand still"
    n = max(1, min(int(n or 20), MAX_STEPS))
    for _ in range(n):
        action = ServerAPI.get_action(ctl.cfg["server"], ctl.obs, prompt, step=ctl.t)
        _, s, done = ctl.step(action, tag)
        if done or ctl.over():
            return "ended"
    return "done"
