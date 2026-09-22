"""Local test of the stone-tier executor path (no LLM, no API key).

Spawns on a fixed seed, gives itself a wooden pickaxe + sticks + planks with /give
(TEST ONLY - evaluation forbids /give), digs straight down with a scripted
attack-while-looking-down loop until cobblestone drops appear in the inventory,
then crafts a stone pickaxe and a furnace through the real crafting-table GUI and
checks the stone-tier goal matcher on the resulting inventory.

    xvfb-run -a python scripts/test_stone.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from hydra import compose, initialize_config_dir
from hydra.core.utils import setup_globals


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    setup_globals()
    import mineevolve
    from mineevolve.env import make_env
    from mineevolve.executor.gui_craft import GuiCraftController
    from mineevolve.main import _episode_succeeded

    with initialize_config_dir(config_dir=str(Path(mineevolve.__file__).parent / "conf"), version_base=None):
        cfg = compose("evaluate", overrides=["benchmark=stone"])
    env = make_env(cfg, logger=logging.getLogger("stone-test"))
    env.seed(12345)
    env.reset()
    for c in ("/give @s minecraft:wooden_pickaxe 1", "/give @s minecraft:stick 4", "/give @s minecraft:oak_planks 4"):  # TEST ONLY
        env.execute_cmd(c)
    for _ in range(10):
        env.step(env.action_space.noop())
    out = Path("logs/smoke/stone-test"); out.mkdir(parents=True, exist_ok=True)
    ctl = GuiCraftController(env)
    ok = True

    res = ctl.equip("wooden_pickaxe")
    print(f"{'PASS' if res else 'FAIL'} equip wooden_pickaxe  mainhand={ctl.obs.get('equipped_items', {}).get('mainhand')}")
    ok &= res

    # scripted dig-down: look at the feet and hold attack (what STEVE-1 does for "dig down")
    a = env.action_space.noop(); a["camera"] = np.array([90.0, 0.0])
    env.step(a)
    t0, steps, cobble = time.time(), 0, 0
    while steps < 1500 and cobble < 8:
        a = env.action_space.noop(); a["attack"] = np.array(1)
        obs, _r, done, info = env.step(a); steps += 1
        cobble = int((info.get("inventory") or {}).get("cobblestone", 0))
        if done:
            break
    Image.fromarray(obs["pov"]).save(out / "after_dig.png")
    print(f"{'PASS' if cobble >= 8 else 'FAIL'} dig down -> cobblestone x{cobble} in {steps} steps ({time.time() - t0:.0f}s), y={info.get('coords', [0, 0, 0])[1]:.0f}")
    ok &= cobble >= 8
    print("goal check 'Mine cobblestone with a wooden pickaxe':", _episode_succeeded("Mine cobblestone with a wooden pickaxe", info.get("inventory") or {}))
    print("goal check 'Mine eight cobblestone blocks for a furnace':", _episode_succeeded("Mine eight cobblestone blocks for a furnace", info.get("inventory") or {}))
    print("goal check 'Mine stone slab material from a stone outcrop':", _episode_succeeded("Mine stone slab material from a stone outcrop", info.get("inventory") or {}))

    ctl = GuiCraftController(env)
    for target, n in [("crafting_table", 1), ("stone_pickaxe", 1)]:
        s0 = ctl.steps
        res = ctl.craft(target, n)
        Image.fromarray(ctl.obs["pov"]).save(out / f"after_{target}.png")
        print(f"{'PASS' if res else 'FAIL'} craft {target} x{n}  ({ctl.steps - s0} steps)  inventory={ctl.slots()}")
        ok &= res
    _o, _r, _d, info = env.step(env.action_space.noop())  # refresh the wrapper's inventory view (craft_helper does this too)
    print("goal check 'Craft a stone pickaxe':", _episode_succeeded("Craft a stone pickaxe", info.get("inventory") or {}))
    print("goal check 'Upgrade a wooden pickaxe to a stone pickaxe':", _episode_succeeded("Upgrade a wooden pickaxe to a stone pickaxe", env.info.get("inventory") or {}))
    # furnace needs 8 cobblestone: dig a bit more (3 were used by the pickaxe). The crafted
    # pickaxe sits in the main inventory (slot >= 9) where the auto-pickaxe switch does not
    # look, so the pipeline needs an explicit `equip` subgoal here - same as the planner would.
    res = ctl.equip("stone_pickaxe")
    print(f"{'PASS' if res else 'FAIL'} equip stone_pickaxe  mainhand={ctl.obs.get('equipped_items', {}).get('mainhand')}")
    ok &= res
    for _ in range(20):  # look straight down again (the wrapper limits per-step camera motion)
        a = env.action_space.noop(); a["camera"] = np.array([30.0, 0.0]); _o, _r, _d, info = env.step(a)
        if float(_o.get("location_stats", {}).get("pitch", 0)) >= 85:
            break
    steps, cobble = 0, 0
    while steps < 800 and cobble < 8:
        a = env.action_space.noop(); a["attack"] = np.array(1); _o, _r, _d, info = env.step(a); steps += 1
        cobble = int((info.get("inventory") or {}).get("cobblestone", 0))
    Image.fromarray(_o["pov"]).save(out / "after_second_dig.png")
    print(f"second dig: cobblestone x{cobble} after {steps} steps, y={info.get('coords', [0, 0, 0])[1]:.0f} pitch/yaw={_o.get('location_stats', {}).get('pitch')}/{_o.get('location_stats', {}).get('yaw')} mainhand={_o.get('equipped_items', {}).get('mainhand', {}).get('type')}")
    s0 = ctl.steps
    res = ctl.craft("furnace", 1)
    Image.fromarray(ctl.obs["pov"]).save(out / "after_furnace.png")
    print(f"{'PASS' if res else 'FAIL'} craft furnace x1  ({ctl.steps - s0} steps)  inventory={ctl.slots()}")
    ok &= res
    _o, _r, _d, info = env.step(env.action_space.noop())
    print("goal check 'Craft a furnace':", _episode_succeeded("Craft a furnace", info.get("inventory") or {}))
    env.close()
    print("STONE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
