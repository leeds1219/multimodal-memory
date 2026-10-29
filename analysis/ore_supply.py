"""Which tasks the hidden ore rule alone can supply, and which need ore generated with the world (natural ore).

The rule (ENVIRONMENT.md, Optimus-1 `random_ore`) places one ore per landing height, and each height is used once
per episode, so the most ore of a kind it can give is the number of landing heights for that kind:
gold lands at y 17-22 (6), redstone at y 9-16 (8), diamond at y 1-11 (11), iron at y 26-40 (15).
A task whose raw-ore need is above that limit can only be done with natural ore, which is a matter of luck within
the time limit. Such tasks are labelled, not removed: summary.py reports success on both kinds separately.
"""
from __future__ import annotations

RULE_MAX = {"gold": 6, "redstone": 8, "diamond": 11, "iron": 15}

# Raw ore each hard task needs (Minecraft 1.16 recipes), before the iron pickaxe.
_RECIPES = {
    "o1_gold_00": {"gold": 1}, "o1_gold_01": {"gold": 3}, "o1_gold_02": {"gold": 3}, "o1_gold_03": {"gold": 2},
    "o1_gold_04": {"gold": 2}, "o1_gold_05": {"gold": 1},
    "o1_redstone_00": {"redstone": 1, "iron": 1}, "o1_redstone_01": {"redstone": 1},
    "o1_redstone_02": {"redstone": 1, "iron": 6}, "o1_redstone_03": {"redstone": 1, "iron": 4},
    "o1_redstone_04": {"redstone": 1}, "o1_redstone_05": {"redstone": 1},
    "o1_diamond_00": {"diamond": 1}, "o1_diamond_01": {"diamond": 3}, "o1_diamond_02": {"diamond": 3},
    "o1_diamond_03": {"diamond": 2}, "o1_diamond_04": {"diamond": 2}, "o1_diamond_05": {"diamond": 1},
    "o1_diamond_06": {"diamond": 1},
    "o1_armor_05": {"diamond": 5}, "o1_armor_06": {"diamond": 8}, "o1_armor_07": {"diamond": 7},
    "o1_armor_08": {"diamond": 4},
    "o1_armor_09": {"gold": 5}, "o1_armor_10": {"gold": 7}, "o1_armor_11": {"gold": 4}, "o1_armor_12": {"gold": 8},
}
# Gold, redstone and diamond ore need an iron pickaxe: 3 more iron.
NEEDS = {t: n | {"iron": n.get("iron", 0) + 3} for t, n in _RECIPES.items()}
# Iron armor and the shield need no iron pickaxe.
NEEDS |= {"o1_armor_00": {"iron": 1}, "o1_armor_01": {"iron": 8}, "o1_armor_02": {"iron": 4},
          "o1_armor_03": {"iron": 7}, "o1_armor_04": {"iron": 5}}


def needs_natural_ore(task: str) -> bool:
    """True when the rule alone can never supply the ore the task needs."""
    return any(n > RULE_MAX.get(ore, n) for ore, n in NEEDS.get(task, {}).items())
