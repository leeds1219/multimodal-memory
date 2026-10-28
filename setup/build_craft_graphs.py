"""Craft graphs: the recipes Optimus-1's planner gets (experiment 4, GT Recipe).

Runs the released KnowledgeGraph.compile(target, 1) -- the <craft graph> text Optimus-1's planner prompt
receives -- for every craft target of the 67 tasks and writes configs/suites/optimus1/craft_graphs.json:
{item: {"release": text as released, "fixed": text with two errors of the released graph corrected}}.

Errors of the released graph (it keeps the first recipe file found and counts an ore once):
  - gold_ingot comes from a gold_block (9 ingots) instead of smelting gold_ore -> "gold_ore: need N"
  - <x>_ore is "need 1" whatever the ingots / gems needed -> ore count = the iron_ingot / gold_ingot /
    diamond count (redstone_ore stays: one ore drops 4-5 redstone)

    /opt/conda/envs/mcagent/bin/python setup/build_craft_graphs.py
"""
import sys
import importlib.util
import json
import re
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OPTIMUS1 as REL  # noqa: E402
CFG = Path(__file__).resolve().parent.parent / "configs/suites/optimus1"


def fix(text: str) -> str:
    need = dict(re.findall(r"\d+\. (\w+): need (\d+)", text))
    text = re.sub(r"gold_block: need (\d+)", r"gold_ore: need \1", text)
    for ore, out in (("iron_ore", "iron_ingot"), ("gold_ore", "gold_ingot"), ("diamond_ore", "diamond")):
        n = need.get(out)
        if n is not None:
            text = re.sub(rf"{ore}: need \d+", f"{ore}: need {n}", text)
    return text


def main():
    spec = importlib.util.spec_from_file_location("graph", REL / "src/optimus1/memories/graph.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    g.KnowledgeGraph._recipe_dir = str(REL / "src/optimus1/helper/recipes")
    kg = g.KnowledgeGraph()
    targets = yaml.safe_load(open(CFG / "task_targets.yaml"))["targets"]
    items = sorted({t["any_of"][0] for t in targets.values()})
    out = {}
    for it in items:
        try:
            rel = kg.compile(it, 1)
        except Exception as e:  # noqa: BLE001 -- a target the released graph cannot compile
            print(f"skip {it}: {e!r}")
            continue
        out[it] = {"release": rel, "fixed": fix(rel)}
    (CFG / "craft_graphs.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} graphs -> {CFG / 'craft_graphs.json'}")


if __name__ == "__main__":
    main()
