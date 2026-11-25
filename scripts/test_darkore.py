"""Test of the dark-ore world rule (setup/build_darkore_jar.py), no LLM: in a sealed stone room underground, each ore is
broken next to the agent with and without a torch beside it, and the inventory gain is recorded.

Two ways to break it: `destroy` (/setblock ... air destroy, the loot table alone) and `mine` (the agent holds attack with
an iron pickaxe, the real path). Run once with the rule jar and once with the plain jar as the control:

    python scripts/test_darkore.py --jar run_data/infra/mcprec-6.13-darkore.jar --out <dir> --gpu 4 --port 28700
    python scripts/test_darkore.py --jar run_data/infra/mcprec-6.13-saveall.jar --out <dir> --gpu 4 --port 28700

Results: <out>/darkore_test.json, one row per (ore, light, way): the drop item's inventory gain.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

TASK = "o1_wooden_00"  # any short task: the script never plays it
DROP = {"coal_ore": "coal", "iron_ore": "iron_ore", "gold_ore": "gold_ore", "diamond_ore": "diamond",
        "redstone_ore": "redstone"}


def scripted(cfg, env, obs, instruction, mon, ep, logger, **kw):
    """Stands in for vlm_agent.run_task inside episode.py."""
    from inventory import inv_counts

    state = {"obs": obs, "t": 0}

    def step(n, **keys):
        for _ in range(n):
            env.cache["ypos"] = {}  # no "Stuck" /kill of the released wrapper
            a = env.noop_action()
            a.update(keys)
            state["obs"], _, _, _ = env.step(a)
            state["t"] += 1
        return inv_counts(state["obs"])

    cmd = env.env.execute_cmd
    loc = obs["location_stats"]
    x, z, y = math.floor(float(loc["xpos"])), math.floor(float(loc["zpos"])), 30
    cmd(f"/fill {x - 3} {y - 1} {z - 3} {x + 3} {y + 4} {z + 3} minecraft:stone hollow")
    cmd(f"/tp @p {x + 0.5} {y} {z + 0.5} 0 0")  # yaw 0: facing +z, where the ore goes, at eye level
    cmd("/replaceitem entity @p weapon.mainhand minecraft:iron_pickaxe")
    step(40)
    rows = []
    for ore, item in DROP.items():
        for lit in (False, True):
            for way in ("destroy", "mine"):
                cmd(f"/setblock {x + 1} {y} {z} " + ("minecraft:torch" if lit else "minecraft:air"))
                cmd(f"/setblock {x} {y + 1} {z + 1} minecraft:{ore}")
                inv0 = step(20)
                if way == "destroy":
                    cmd(f"/setblock {x} {y + 1} {z + 1} minecraft:air destroy")
                    step(5)
                else:
                    step(80, attack=1)
                inv1 = step(30)
                cmd("/kill @e[type=item]")
                rows.append({"ore": ore, "lit": lit, "way": way, "gain": inv1.get(item, 0) - inv0.get(item, 0)})
                logger.info(f"[darkore] {rows[-1]}")
    (Path(ep) / "darkore_test.json").write_text(json.dumps(rows, indent=1))
    return {"success": False, "steps": state["t"]}


def inner() -> int:
    """Run episode.py's main with the scripted agent in place of the Baseline agent."""
    sys.path.insert(0, str(paths.ROOT / "domains" / "minecraft" / "world"))
    sys.path.insert(0, str(paths.ROOT / "domains" / "minecraft" / "agent"))
    import vlm_agent
    vlm_agent.run_task = scripted
    import episode
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    return episode.main()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--inner":
        return inner()
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    import chain
    ch = chain.Chain(argparse.Namespace(method="baseline", world="nolava", save=None, knowledge=None, order="test_darkore",
                                        gpu=a.gpu, port=a.port, retries=0, stall_min=30.0, mock=True, seed_offset=0))
    ch.env_vars.update(OPTIMUS_ISOWORLD="1", OPTIMUS_SAVEALL_JAR=str(Path(a.jar).resolve()))
    popen = subprocess.Popen

    def swap(cmd, *args, **kw):  # chain.run_episode launches episode.py; launch it through this script instead
        cmd = [str(Path(__file__).resolve()) + " --inner" if str(c).endswith("episode.py") else c for c in cmd]
        cmd = [p for c in cmd for p in (c.split(" ") if c.endswith(" --inner") else [c])]
        return popen(cmd, *args, **kw)

    chain.subprocess.Popen = swap
    ep = Path(a.out).resolve()
    ep.mkdir(parents=True, exist_ok=True)
    res = ch.run_episode(TASK, ep)
    f = ep / "darkore_test.json"
    print(json.dumps({"status": res.get("status"), "rows": json.loads(f.read_text()) if f.exists() else None}, indent=1))
    return 0 if f.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
