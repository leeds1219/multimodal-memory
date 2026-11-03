"""Tool: Optimus-1's own scripted helpers, equip / craft / smelt (the "equip", "craft", "smelt" commands).
Kept as released: their quirks (e.g. equipping with a full hotbar drops a stack) are part of what an agent meets.
OPTIMUS_EQUIPFIX=1 (world variant): equip does not drop anything; see equip_nodrop."""
from __future__ import annotations

import os


def equip_nodrop(h, target_item: str):
    """Optimus-1's equip_item without its drop. Released: the item goes to an empty hotbar slot, or to a random one
    when the hotbar is full, where the item there is swapped onto the cursor and dropped when the inventory closes.
    Here: an item already in the hotbar is selected where it is; otherwise it is swapped with the item in the chosen
    hotbar slot, which goes back to where the item came from."""
    from optimus1.helper.slot import SLOT_POS_MAPPING
    import random
    try:
        if not h.info["isGuiOpen"]:
            h.open_inventory_wo_recipe()
        pos_id = h.find_in_inventory(h.get_labels(), target_item)
        h._assert(pos_id, f'missing material: {{"{target_item}": 1}}')
        k = int(pos_id.split("_")[-1])
        if pos_id.startswith("inventory_") and k < 9:  # already in the hotbar
            bar = k
        else:
            _, empty_bar = h.find_empty_box(h.info["plain_inventory"])
            bar = random.choice(empty_bar) if empty_bar else random.choice(range(9))
            slot_pos = SLOT_POS_MAPPING[h.current_gui_type]
            if empty_bar:
                h.pull_item_all(slot_pos, pos_id, f"inventory_{bar}")
            else:
                h.swap_item(slot_pos, pos_id, f"inventory_{bar}")
        h._call_func("inventory")  # close inventory
        h._call_func(f"hotbar.{bar + 1}")
        h._attack_continue()
    except AssertionError as e:
        return False, str(e)
    return True, None


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
        if os.environ.get("OPTIMUS_EQUIPFIX") == "1":
            eh = ctl._helper.equip_helper
            eh.equip_item = lambda target: equip_nodrop(eh, target)
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
