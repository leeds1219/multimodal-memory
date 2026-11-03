"""Minecraft adapter for the loop results page: everything build.py needs that depends on the simulator and agent.

An adapter module has: goal(result) -> dict, episode(path, goal) -> dict, turns(path) -> {turn: record},
frame(path, step) -> keyframe file or None, world(source path) -> int or None.
"""
import glob
import gzip
import json
import re
from pathlib import Path

import yaml

REPRO = Path(__file__).resolve().parents[3]
SUITE = REPRO / "configs" / "suites" / "optimus1"
TARGETS = yaml.safe_load(open(SUITE / "task_targets.yaml"))["targets"]
GRAPHS = json.load(open(SUITE / "craft_graphs.json"))
# the ore a target item is made from: (inventory items counted, label, short label, recipe line that gives the need)
ORES = [("diamond", ["diamond"], "diamond", "diamond", "diamond"),
        ("golden", ["gold_ore", "gold_ingot"], "gold ore/ingot", "gold", "gold_ingot"),
        ("iron", ["iron_ore", "iron_ingot"], "iron ore/ingot", "iron", "iron_ingot"),
        ("redstone", ["redstone"], "redstone", "redstone", "redstone")]


RECIPE = {"golden_sword": ["2 gold ingot", "1 stick"], "golden_axe": ["3 gold ingot", "2 stick"],
          "golden_helmet": ["5 gold ingot"], "golden_boots": ["4 gold ingot"], "diamond_chestplate": ["8 diamond"],
          "diamond_leggings": ["7 diamond"], "diamond_helmet": ["5 diamond"], "diamond_shovel": ["1 diamond", "2 stick"],
          "diamond_hoe": ["2 diamond", "2 stick"]}


def _need(item, line):
    m = re.search(rf"\d+\. {line}: need (\d+)", GRAPHS.get(item, {}).get("release", ""))
    return int(m.group(1)) if m else 1


def goal(res):
    """the task's target item (the success check) and the ore counted as progress"""
    items = TARGETS[res["task"]]["any_of"]
    key = items[0]
    ore = next((o for o in ORES if key.startswith(o[0])), None)
    g = {"label": key.replace("_", " "), "items": items, "unit": "", "unit_short": "", "ore": [], "need": 0, "needs": []}
    if ore:
        g.update(unit=ore[2], unit_short=ore[3], ore=ore[1], need=_need(key, ore[4]))
    # what the item is crafted from (shown on the page only), the stations, the pickaxe the ore needs
    if ore:
        g["needs"] = RECIPE.get(key, [f"{g['need']} {ore[4].replace('_', ' ')}"]) + ["crafting table"] + \
                     ["furnace"] * (ore[0] in ("golden", "iron")) + ["iron pickaxe (to mine the ore)"] * (ore[0] != "iron")
    return g


def episode(ep, g, horizon):
    """progress (ore held) over time, height samples, most ore held, most of the target item held"""
    cnt, ys, last_c, last_y_t, maxc, got = [], [], None, -10**9, 0, 0
    for line in gzip.open(Path(ep) / "trajectory.jsonl.gz", "rt"):
        r = json.loads(line)
        t = r["t"]
        if "inv" in r:
            inv = r["inv"]
            c = sum(inv.get(k, 0) for k in g["ore"])
            maxc = max(maxc, c)
            got = max(got, max(inv.get(k, 0) for k in g["items"]))
            if c != last_c:
                cnt.append([t, c]); last_c = c
        if "pos" in r and t - last_y_t >= 200:
            ys.append([t, round(r["pos"][1], 1)]); last_y_t = t
    return {"count": cnt, "y": ys, "max": maxc, "target": got}


def _cmd(c):
    if not c:
        return ""
    if c.get("type") == "steve":
        return f"steve '{c.get('prompt')}' x{c.get('steps')}"
    if c.get("type") == "keys":
        cam = f" cam [{','.join(map(str, c['camera']))}]" if c.get("camera") else ""
        return f"keys {'+'.join(c.get('keys') or []) or '-'}{cam} x{c.get('steps')}"
    if c.get("type") == "equip":
        return f"equip {c.get('item')}"
    item, n = c.get("item"), c.get("n")
    return f"{c.get('type')} {'' if item is None else item} {'' if n is None else n}".strip()


def turns(ep):
    out = {}
    for x in map(json.loads, open(Path(ep) / "vlm" / "turns.jsonl")):
        out[x["turn"]] = {"turn": x["turn"], "t0": x["t0"], "t1": x["t1"], "cmd": _cmd(x["cmd"]),
                          "thought": x.get("thought", ""), "outcome": x.get("outcome", ""), "effect": x.get("effect", "")}
    return out


def frame(ep, step):
    fs = sorted(glob.glob(str(Path(ep) / "keyframes" / "*.jpg")))
    return min(fs, key=lambda f: abs(int(Path(f).stem) - step)) if fs else None


def world(src):
    m = re.search(r"/nb(\d+)_\d+/", str(src))
    return int(m.group(1)) + 1 if m else None
