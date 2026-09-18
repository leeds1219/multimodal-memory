"""Per-slot inventory observation (needs the MineEvolve MineRL jar patch).

MineRL's ``FlatInventoryObservation`` aggregates the player's 36 slots into
item counts; a GUI crafting controller needs to know *which slot* holds
what. Our jar patch adds ``"index"`` to every stack in the raw inventory
JSON, and this handler exposes it as an ``(36, 2)`` int array:
column 0 = item id (index into ``MINERL_ITEM_MAP``, 0 = "none"),
column 1 = quantity. Slots 0-8 are the hotbar, 9-35 the main inventory.
"""

from __future__ import annotations

import numpy as np
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.mc import MINERL_ITEM_MAP

N_SLOTS = 36
_ITEM_TO_ID = {name: i for i, name in enumerate(MINERL_ITEM_MAP)}


class SlotInventoryObservation(TranslationHandler):
    def to_string(self) -> str:
        return "inventory_slots"

    def xml_template(self) -> str:
        return "<ObservationFromFullInventory flat=\"false\"/>"

    def __init__(self) -> None:
        super().__init__(spaces.Box(low=0, high=2304, shape=(N_SLOTS, 2), dtype=np.int32))

    def add_to_mission_spec(self, mission_spec):  # same as FlatInventoryObservation
        pass

    def from_hero(self, info):
        out = np.zeros((N_SLOTS, 2), dtype=np.int32)
        for stack in info.get("inventory") or []:
            idx = stack.get("index")
            if idx is None or not (0 <= int(idx) < N_SLOTS):
                continue
            out[int(idx), 0] = _ITEM_TO_ID.get(str(stack.get("type")), 0)
            out[int(idx), 1] = int(stack.get("quantity") or 0)
        return out

    def from_universal(self, obs):
        return np.zeros((N_SLOTS, 2), dtype=np.int32)


def decode_slots(arr) -> dict[int, tuple[str, int]]:
    """``{slot: (item, quantity)}`` for non-empty slots."""
    out: dict[int, tuple[str, int]] = {}
    if arr is None:
        return out
    a = np.asarray(arr)
    for i in range(min(N_SLOTS, a.shape[0])):
        item_id, qty = int(a[i, 0]), int(a[i, 1])
        if qty > 0 and item_id > 0:
            out[i] = (MINERL_ITEM_MAP[item_id], qty)
    return out
