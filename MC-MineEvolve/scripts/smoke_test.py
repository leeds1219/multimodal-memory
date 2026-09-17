"""End-to-end smoke test that needs NO LLM API key.

Exercises the two non-LLM pieces of MineEvolve:

  1. the MineRL environment (Java + Xvfb + our env spec/wrapper): launch,
     reset, run the reset-time /commands, step;
  2. optionally STEVE-1 inference through the FastAPI server
     (``/chat type=action``) if a server is reachable.

Run from the ``mineevolve`` env, on a headless box wrap with xvfb-run::

    xvfb-run -a python scripts/smoke_test.py                   # env only, random actions
    bash scripts/server.sh &                                   # (in another shell, GPU)
    xvfb-run -a python scripts/smoke_test.py --server http://127.0.0.1:9000

Exit code 0 means every stage that was attempted passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.utils import setup_globals

logger = logging.getLogger("mineevolve.smoke")


def _random_action(env, rng: np.random.Generator) -> dict:
    """noop + a few random movement/attack/camera keys.

    ``action_space.sample()`` is not usable here: the ChatAction's Text space has
    a ``sample()`` signature gym 0.23's Dict does not expect.
    """
    a = env.action_space.noop()
    a["forward"] = np.array(int(rng.random() < 0.7))
    a["jump"] = np.array(int(rng.random() < 0.2))
    a["attack"] = np.array(int(rng.random() < 0.5))
    a["camera"] = np.array(rng.normal(0.0, 5.0, size=2), dtype=np.float32)
    return a


def _inventory(info: dict) -> dict:
    return {k: int(v) for k, v in (info.get("inventory") or {}).items() if int(v) > 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="wooden", help="conf/benchmark/<name>.yaml")
    ap.add_argument("--steps", type=int, default=200, help="env steps to run after reset")
    ap.add_argument("--server", default=None, help="MineEvolve server URL; omit for random actions")
    ap.add_argument("--condition", default="chop a tree", help="STEVE-1 text condition (server mode)")
    ap.add_argument("--out", default="logs/smoke", help="where to drop a POV frame")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # --- config: same tree main.py uses, minus the hydra run dir --------------
    setup_globals()  # registers the ${now:...} resolver used by evaluate.yaml
    import mineevolve

    conf_dir = str(Path(mineevolve.__file__).resolve().parent / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="evaluate", overrides=[f"benchmark={args.benchmark}"])
    logger.info("env=%s biome=%s", cfg.env.name, cfg.env.prefer_biome)

    # --- stage 1: environment ---------------------------------------------------
    from mineevolve.env import make_env
    from mineevolve.main import _safe_pov

    t0 = time.time()
    env = make_env(cfg, logger=logger)
    logger.info("gym.make OK (%.1fs); launching Minecraft, first reset can take a few minutes...", time.time() - t0)
    t0 = time.time()
    obs = env.reset()
    logger.info("reset OK in %.1fs", time.time() - t0)

    pov = obs.get("pov") if isinstance(obs, dict) else None
    if not isinstance(pov, np.ndarray) or pov.ndim != 3:
        logger.error("unexpected obs: keys=%s", list(obs) if isinstance(obs, dict) else type(obs))
        return 1
    logger.info("obs keys=%s pov=%s inventory=%s", sorted(obs), pov.shape, _inventory(env.info))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(pov).save(out / "reset_pov.png")
        logger.info("saved %s", out / "reset_pov.png")
    except Exception as exc:  # pragma: no cover - cosmetic
        logger.warning("could not save POV frame: %s", exc)

    # --- stage 2: step (server STEVE-1 if available, else random) ---------------
    client = None
    if args.server:
        from mineevolve.client import MineEvolveClient

        client = MineEvolveClient(base_url=args.server, timeout=60.0)
        status = client.status()
        logger.info("server status: %s", status)
        client.reset(task_goal=args.condition)

    start_inv = _inventory(env.info)
    rng = np.random.default_rng(0)
    ok_steps = 0
    t0 = time.time()
    for i in range(args.steps):
        if client is not None:
            r = client.action(condition=args.condition, obs={"pov": _safe_pov(obs), "image": None})
            action = r.get("action")
            if action is None:
                logger.error("server returned no action: %s", r.get("error"))
                env.close()
                return 2
        else:
            action = _random_action(env, rng)
        obs, _r, done, info = env.step(action)
        ok_steps += 1
        if (i + 1) % 50 == 0:
            logger.info("step %d  coords=%s inv=%s", i + 1, info.get("coords"), _inventory(info))
        if done:
            logger.info("episode ended at step %d", i + 1)
            break
    dt = time.time() - t0
    logger.info("%d steps in %.1fs (%.1f steps/s)", ok_steps, dt, ok_steps / max(dt, 1e-6))
    logger.info("inventory delta: %s -> %s", start_inv, _inventory(env.info))

    Image.fromarray(obs["pov"]).save(out / "final_pov.png")
    env.close()
    logger.info("SMOKE TEST PASSED (%s)", "server STEVE-1" if client else "random actions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
