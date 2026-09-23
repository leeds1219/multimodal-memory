"""Extend DEPS goal_lib.json to the items our 70 tasks need (DEVIATIONS Stage B).

Same schema as the release: {name: {output: {item: n}, type, precondition, tool}}.
Crafted/smelted items take ingredients from the Minecraft 1.16 recipes (tag
ingredients use the release's generic names: log, planks); mined items take the
vanilla minimum pickaxe. Existing release entries are never modified.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "envs"))
from functional_craft import RecipeBook
from deps_agent import generic_name  # noqa: E402  (same dir)
import yaml

HERE = Path(__file__).resolve().parent
lib = json.loads(Path("/home/rag/data/official/MC-Planner/data/goal_lib.json").read_text())
have = {list(v["output"])[0] for v in lib.values()}
book = RecipeBook()
targets = yaml.safe_load(open(HERE.parents[1] / "configs" / "task_targets.yaml"))["targets"]
MINE_TOOL = {"iron_ore": "stone_pickaxe", "gold_ore": "iron_pickaxe", "redstone": "iron_pickaxe",
             "redstone_ore": "iron_pickaxe", "diamond": "iron_pickaxe", "coal": "wooden_pickaxe",
             "cobblestone": "wooden_pickaxe", "stone": "wooden_pickaxe", "lapis_lazuli": "stone_pickaxe"}
MINE_PLAIN = {"sapling", "apple", "carrot", "leather", "log", "sand", "gravel", "flint", "sugar_cane",
              "wool", "beef", "string", "feather", "egg", "turtle_egg", "scute", "iron_nugget"}

ext = {}

def tag_name(spec):
    spec = spec[0] if isinstance(spec, list) else spec
    name = generic_name((spec.get("item") or spec.get("tag")).split(":")[-1])
    return {"gold_ores": "gold_ore", "iron_ores": "iron_ore", "coals": "coal"}.get(name, name)

def add(item, depth=0):
    item = generic_name(item.lstrip("#").rstrip("s") if item in ("#logs", "#saplings") else item.lstrip("#"))
    if item in have or depth > 6:
        return
    recs = [r for r in book.by_result.get(item, []) if not r["_file"].endswith(("_from_blasting", "_block", "_from_nuggets"))]
    if item in MINE_TOOL or item in MINE_PLAIN or not recs:
        tool = MINE_TOOL.get(item)
        ext[f"mine_{item}"] = {"output": {item: 1}, "type": "mine", "precondition": {}, "tool": {tool: 1} if tool else {}}
        have.add(item)
        return
    r = recs[0]
    need = {}
    if r["_kind"] == "crafting_shaped":
        for row in r["pattern"]:
            for ch in row:
                if ch != " ":
                    n = tag_name(r["key"][ch]); need[n] = need.get(n, 0) + 1
    elif r["_kind"] == "crafting_shapeless":
        for spec in r["ingredients"]:
            n = tag_name(spec); need[n] = need.get(n, 0) + 1
    else:
        need = {tag_name(r["ingredient"]): 1}
    kind = "smelt" if r["_kind"] == "smelting" else "craft"
    tool = {"furnace": 1} if kind == "smelt" else ({"crafting_table": 1} if book.needs_table(r) else {})
    ext[f"{kind}_{item}"] = {"output": {item: r["_count"]}, "type": kind, "precondition": need, "tool": tool}
    have.add(item)
    for n in need:
        add(n, depth + 1)

for uid, t in targets.items():
    for it in t["any_of"]:
        add(it)
(HERE / "goal_lib_ext.json").write_text(json.dumps(ext, indent=1))
print(len(ext), sorted(ext))
