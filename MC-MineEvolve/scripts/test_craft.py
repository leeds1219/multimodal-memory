"""Local unit test for the GUI crafting controller (no LLM, no API key).

Spawns on a fixed seed, gives itself logs with /give (TEST ONLY - evaluation
forbids /give), then crafts planks -> sticks -> crafting table -> wooden
pickaxe through the real inventory / crafting-table GUIs and asserts the
inventory after each step. Frames are saved under logs/smoke/craft-test/.

    xvfb-run -a python scripts/test_craft.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PIL import Image
from hydra import compose, initialize_config_dir
from hydra.core.utils import setup_globals


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    setup_globals()
    import mineevolve
    from mineevolve.env import make_env
    from mineevolve.executor.gui_craft import GuiCraftController

    with initialize_config_dir(config_dir=str(Path(mineevolve.__file__).parent / "conf"), version_base=None):
        cfg = compose("evaluate", overrides=["benchmark=wooden"])
    env = make_env(cfg, logger=logging.getLogger("craft-test"))
    env.seed(12345)
    env.reset()
    env.execute_cmd("/give @s minecraft:oak_log 4")  # TEST ONLY
    for _ in range(10):
        env.step(env.action_space.noop())
    out = Path("logs/smoke/craft-test"); out.mkdir(parents=True, exist_ok=True)
    ctl = GuiCraftController(env)
    ok = True
    for target, n in [("oak_planks", 12), ("stick", 4), ("crafting_table", 1), ("wooden_pickaxe", 1)]:
        t0, s0 = time.time(), ctl.steps
        res = ctl.craft(target, n)
        Image.fromarray(ctl.obs["pov"]).save(out / f"after_{target}.png")
        print(f"{'PASS' if res else 'FAIL'} craft {target} x{n}  ({ctl.steps - s0} steps, {time.time() - t0:.1f}s)  inventory={ctl.slots()}")
        ok &= res
    env.close()
    print("CRAFT TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
