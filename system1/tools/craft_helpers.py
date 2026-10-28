"""Tool: Optimus-1's own scripted helpers, equip / craft / smelt (the "equip", "craft", "smelt" commands).
Kept as released: their quirks (e.g. equipping with a full hotbar drops a stack) are part of what an agent meets."""
from __future__ import annotations


def run(ctl, typ: str, item: str, n: int, tag: str) -> str:
    from optimus1.helper import Helper

    class Proxy:  # helper steps go through Controller.step (logged, timed, checked)
        def __getattr__(self, k):
            return getattr(ctl.env, k)

        def step(self, action, *a, **k):
            obs, s, done = ctl.step(action, tag)
            return obs, 0.0, done, ctl.info

    if ctl._helper is None:
        ctl._helper = Helper(Proxy())
    ctl.env.can_change_hotbar = ctl.env.can_open_inventory = True
    try:
        task = f"{typ} {item}"
        ctl._helper.reset(task, None, None, None)
        done, info = ctl._helper.step(task, (item, n))
    except Exception as e:  # time limit or script error
        done, info = False, repr(e)[:120]
    finally:
        ctl.env.can_change_hotbar = ctl.env.can_open_inventory = False
    return "done" if done else f"failed: {info}"
