"""Light check of the dark-ore rule (setup/build_darkore_jar.py) where the agent really mines, no LLM: at
night (the darkore world), torch vs none, in three places built high in the air: a sealed stone room, a 1x1 shaft open to the sky (the agent digs
straight down), and open ground. For each case it reads the light the loot table sees at the ore and its 6 neighbours
(a datapack of predicates "light >= N", the same location_check the jar's loot tables use) and mines the ore with an
iron pickaxe, recording the drop. Mobs are off, as in the night world.

    python scripts/test_darkore_light.py --jar run_data/infra/mcprec-6.13-darkore.jar --out <dir> --gpu 4 --port 28700

Results: <out>/darkore_light.json, one row per (place, time, torch, ore): light at the 7 positions, the drop's gain.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

TASK = "o1_wooden_00"  # any short task: the script never plays it
DROP = {"coal_ore": "coal", "iron_ore": "iron_ore", "gold_ore": "gold_ore", "diamond_ore": "diamond",
        "redstone_ore": "redstone"}
TIMES = {"night": 18000}  # the darkore world is always midnight
Y = 100  # build height: above trees, so "open" really sees the sky


def write_datapack(ep: Path) -> None:
    for lvl in Path(ep).glob("MCP-Reborn/saves/*/level.dat"):
        dp = lvl.parent / "datapacks" / "lighttest"
        (dp / "data" / "lt" / "predicates").mkdir(parents=True, exist_ok=True)
        (dp / "pack.mcmeta").write_text(json.dumps({"pack": {"pack_format": 6, "description": "light probes"}}))
        for n in range(16):
            (dp / "data" / "lt" / "predicates" / f"ge{n}.json").write_text(json.dumps(
                {"condition": "minecraft:location_check", "predicate": {"light": {"light": {"min": n}}}}))


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
    write_datapack(ep)
    cmd("/reload")
    cmd('/datapack enable "file/lighttest"')
    cmd("/gamerule doMobSpawning false")
    cmd("/kill @e[type=!player]")
    cmd("/gamerule sendCommandFeedback true")  # the probes' /say lines must reach client.log
    step(40)
    loc = obs["location_stats"]
    x0, z0 = math.floor(float(loc["xpos"])), math.floor(float(loc["zpos"]))
    log = Path(ep) / "MCP-Reborn" / "logs" / "latest.log"  # chat lands in Minecraft's own log, not client.log

    def light_at(x, y, z, tag) -> int:
        for n in range(16):
            cmd(f"/execute positioned {x} {y} {z} if predicate lt:ge{n} run say LT {tag} {n}")
        step(10)
        text = log.read_text(errors="ignore") if log.exists() else ""
        got = [int(m.group(1)) for m in re.finditer(rf"LT {re.escape(tag)} (\d+)\b", text)]
        return max(got) if got else -1

    # the loot table's 7 positions: the ore and its 6 neighbours (setup/build_darkore_jar.py OFFSETS)
    sides = {"ore": (0, 0, 0), "west": (-1, 0, 0), "east": (1, 0, 0), "below": (0, -1, 0), "above": (0, 1, 0),
             "front": (0, 0, -1), "behind": (0, 0, 1)}

    rows = []
    for pi, place in enumerate(("room", "shaft", "open")):
        x, z, y = x0 + 20 * pi, z0, Y  # agent feet at (x, y, z), facing +z; the ore at (x, y + 1, z + 1)
        cmd(f"/fill {x - 3} {y - 1} {z - 3} {x + 3} {y + 12} {z + 3} minecraft:air")
        if place == "room":
            cmd(f"/fill {x - 2} {y - 1} {z - 2} {x + 2} {y + 3} {z + 2} minecraft:stone hollow")
        elif place == "shaft":  # stone all round, the agent's 1x1 column open to the sky
            cmd(f"/fill {x - 2} {y - 1} {z - 2} {x + 2} {y + 10} {z + 2} minecraft:stone")
            cmd(f"/fill {x} {y} {z} {x} {y + 10} {z} minecraft:air")
        else:  # a stone floor under the sky
            cmd(f"/fill {x - 2} {y - 1} {z - 2} {x + 2} {y - 1} {z + 2} minecraft:stone")
        cmd(f"/tp @p {x + 0.5} {y} {z + 0.5} 0 0")
        cmd("/replaceitem entity @p weapon.mainhand minecraft:iron_pickaxe")
        step(20)
        for time, ticks in TIMES.items():
            cmd(f"/time set {ticks}")
            for torch in (False, True):
                # torch: where the agent stands (the only free block in the shaft), 1 from the ore's air neighbour
                cmd(f"/setblock {x} {y} {z} " + ("minecraft:torch" if torch else "minecraft:air"))
                step(10)
                tag = f"{place}-{time}-{int(torch)}"
                cmd(f"/setblock {x} {y + 1} {z + 1} minecraft:iron_ore")
                step(10)
                lt = {s: light_at(x + dx, y + 1 + dy, z + 1 + dz, f"{tag}-{s}") for s, (dx, dy, dz) in sides.items()}
                for ore, item in DROP.items():
                    cmd(f"/setblock {x} {y + 1} {z + 1} minecraft:{ore}")
                    inv0 = step(20)
                    step(80, attack=1)
                    inv1 = step(30)
                    cmd("/kill @e[type=item]")
                    rows.append({"place": place, "time": time, "torch": torch, "light": lt, "ore": ore,
                                 "gain": inv1.get(item, 0) - inv0.get(item, 0)})
                    logger.info(f"[darkore-light] {rows[-1]}")
    (Path(ep) / "darkore_light.json").write_text(json.dumps(rows, indent=1))
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
    ch = chain.Chain(argparse.Namespace(method="baseline", world="nolava", save=None, knowledge=None,
                                        order="test_darkore_light", gpu=a.gpu, port=a.port, retries=0, stall_min=30.0,
                                        mock=True, seed_offset=0))
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
    f = ep / "darkore_light.json"
    print(json.dumps({"status": res.get("status"), "rows": json.loads(f.read_text()) if f.exists() else None}, indent=1))
    return 0 if f.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
