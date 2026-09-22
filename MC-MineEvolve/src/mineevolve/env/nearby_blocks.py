"""Nearest landmark blocks around the agent (needs the MineEvolve MineRL jar patch).

The upstream Adaptor prompt's own example repairs a subgoal "approach oak tree at
(118, 64, 208)", but the released code has no observation that could supply a
tree's coordinates and no executor that walks to one. Our jar patch adds a
``nearby_blocks`` entry to the raw info JSON: for every landmark block type
(``*_log``, ``*_ore``, ``crafting_table``, ``furnace``, ``water``, ``lava``)
within a 33x33x17 box, the 3 nearest positions and the total count::

    {"oak_log": {"nearest": [[x, y, z, dist], ...], "count": 12}, ...}

This handler passes it through as a JSON string (``obs["nearby_blocks"]``);
``decode_nearby`` turns it back into a dict. Whether the planner *sees* it is a
separate switch (``MINEEVOLVE_NEARBY_BLOCKS=1`` in main.py); the ``approach``
primitive reads it directly from the env either way.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Sequence

from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler


class NearbyBlocksObservation(TranslationHandler):
    def to_string(self) -> str:
        return "nearby_blocks"

    def xml_template(self) -> str:
        return ""

    def __init__(self) -> None:
        super().__init__(spaces.Text([1]))

    def add_to_mission_spec(self, mission_spec):
        pass

    def from_hero(self, info):
        return json.dumps(info.get("nearby_blocks") or {})

    def from_universal(self, obs):
        return "{}"


def decode_nearby(obs_value) -> Dict[str, Dict[str, Any]]:
    """``{block: {"nearest": [[x, y, z, dist], ...], "count": n}}`` from the obs field."""
    if isinstance(obs_value, Mapping):
        return dict(obs_value)
    try:
        s = obs_value if isinstance(obs_value, str) else str(obs_value)
        return json.loads(s or "{}")
    except (TypeError, ValueError):
        return {}


def nearest_matching(nearby: Mapping[str, Any], name: str) -> List[float] | None:
    """Nearest [x, y, z, dist] of a block whose id equals or ends with ``name``
    ('oak_log', 'log', 'ore', 'iron_ore', 'crafting_table')."""
    name = str(name).replace("minecraft:", "").strip().lower()
    best = None
    for block, entry in nearby.items():
        if block != name and not block.endswith("_" + name) and not block.endswith(name):
            continue
        for p in entry.get("nearest") or []:
            if best is None or p[3] < best[3]:
                best = list(p)
    return best


def summarize_nearby(nearby: Mapping[str, Any], coords: Sequence[float] | None = None, limit: int = 6) -> str:
    """One line per landmark for a text state, nearest first: 'oak_log x12, nearest (118, 64, 208) 6.3 blocks NE'."""
    rows = []
    for block, entry in nearby.items():
        pts = entry.get("nearest") or []
        if not pts:
            continue
        x, y, z, d = pts[0]
        heading = ""
        if coords is not None:
            dx, dz = x - coords[0], z - coords[2]
            ang = (math.degrees(math.atan2(-dx, dz)) + 360) % 360  # minecraft yaw: 0 = +z (south), 90 = -x (west)
            heading = " " + ["S", "SW", "W", "NW", "N", "NE", "E", "SE"][int((ang + 22.5) // 45) % 8]
        rows.append((d, f"{block} x{entry.get('count', len(pts))}, nearest ({int(x)}, {int(y)}, {int(z)}) {d:.1f} blocks{heading}"))
    rows.sort()
    return "; ".join(r for _, r in rows[:limit])
