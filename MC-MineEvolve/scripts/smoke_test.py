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

Every run is recorded under ``--out`` (default ``logs/smoke/run-<timestamp>``)::

    summary.json        mode, condition, timings, inventory delta, pass/fail
    trajectory.jsonl    one line per step: coords, health, hunger, inventory, action keys
    frames/step_*.png   POV every ``--frame-every`` steps (+ reset / final)

Render it with ``python scripts/plot_smoke.py <run dir>``.

Exit code 0 means every stage that was attempted passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
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


def _pressed(action: dict) -> list[str]:
    """Names of the binary keys held this step (camera / chat are logged separately)."""
    keys = []
    for k, v in action.items():
        if k == "camera" or isinstance(v, str):
            continue
        v = np.asarray(v)
        if v.ndim == 0 and v.dtype.kind in "biu" and int(v) != 0:
            keys.append(k)
    return sorted(keys)


def _step_record(i: int, t: float, action: dict, info: dict, done: bool) -> dict:
    cam = np.asarray(action.get("camera", [0.0, 0.0]), dtype=float).ravel().tolist()
    return {
        "step": i,
        "t": round(t, 3),
        "coords": [float(c) for c in (info.get("coords") or [])],
        "health": info.get("health"),
        "hunger": info.get("hunger"),
        "inventory": _inventory(info),
        "keys": _pressed(action),
        "camera": [round(c, 2) for c in cam],
        "done": bool(done),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="wooden", help="conf/benchmark/<name>.yaml")
    ap.add_argument("--steps", type=int, default=200, help="env steps to run after reset")
    ap.add_argument("--server", default=None, help="MineEvolve server URL; omit for random actions")
    ap.add_argument("--condition", default="chop a tree", help="STEVE-1 text condition (server mode)")
    ap.add_argument("--out", default=None, help="run dir (default logs/smoke/run-<timestamp>)")
    ap.add_argument("--frame-every", type=int, default=10, help="save a POV frame every N steps (0 = only reset/final)")
    ap.add_argument("--seed", type=int, default=None, help="world seed (same mechanism as `seeds` in evaluate.yaml)")
    ap.add_argument("--pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="teleport here after reset (JARVIS-1 close-ended spawn)")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"logs/smoke/run-{datetime.now():%Y%m%d-%H%M%S}"

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

    out = Path(args.out)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "mode": "server STEVE-1" if args.server else "random actions",
        "server": args.server,
        "condition": args.condition if args.server else None,
        "benchmark": args.benchmark,
        "env": cfg.env.name,
        "steps_requested": args.steps,
        "passed": False,
    }

    def _finish(code: int, error: str | None = None) -> int:
        summary["passed"] = code == 0
        summary["error"] = error
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("run dir: %s", out)
        return code

    t0 = time.time()
    env = make_env(cfg, logger=logger)
    logger.info("gym.make OK (%.1fs); launching Minecraft, first reset can take a few minutes...", time.time() - t0)
    t0 = time.time()
    if args.seed is not None:
        env.seed(args.seed)
    summary["seed"] = args.seed
    obs = env.reset()
    if args.pos is not None:
        x, y, z = args.pos
        env.execute_cmd(f"/tp @s {x:.1f} {y:.1f} {z:.1f}")
        env.execute_cmd("/spawnpoint")
        for _ in range(10):
            obs, _r, _d, _i = env.step(env.action_space.noop())
        summary["pos"] = list(args.pos)
    summary["reset_s"] = round(time.time() - t0, 1)
    logger.info("reset OK in %.1fs", summary["reset_s"])

    pov = obs.get("pov") if isinstance(obs, dict) else None
    if not isinstance(pov, np.ndarray) or pov.ndim != 3:
        logger.error("unexpected obs: keys=%s", list(obs) if isinstance(obs, dict) else type(obs))
        return _finish(1, "unexpected obs")
    logger.info("obs keys=%s pov=%s inventory=%s", sorted(obs), pov.shape, _inventory(env.info))
    summary["pov_shape"] = list(pov.shape)

    from PIL import Image

    Image.fromarray(pov).save(frames / "step_0000.png")
    logger.info("saved %s", frames / "step_0000.png")

    # --- stage 2: step (server STEVE-1 if available, else random) ---------------
    client = None
    if args.server:
        from mineevolve.client import MineEvolveClient

        client = MineEvolveClient(base_url=args.server, timeout=60.0)
        status = client.status()
        logger.info("server status: %s", status)
        client.reset(task_goal=args.condition)

    start_inv = _inventory(env.info)
    summary["inventory_start"] = start_inv
    rng = np.random.default_rng(0)
    ok_steps = 0
    t0 = time.time()
    traj = (out / "trajectory.jsonl").open("w")
    traj.write(json.dumps(_step_record(0, 0.0, env.action_space.noop(), env.info, False)) + "\n")
    try:
        for i in range(args.steps):
            if client is not None:
                r = client.action(condition=args.condition, obs={"pov": _safe_pov(obs), "image": None})
                action = r.get("action")
                if action is None:
                    logger.error("server returned no action: %s", r.get("error"))
                    env.close()
                    return _finish(2, f"server returned no action: {r.get('error')}")
            else:
                action = _random_action(env, rng)
            obs, _r, done, info = env.step(action)
            ok_steps += 1
            traj.write(json.dumps(_step_record(i + 1, time.time() - t0, action, info, done)) + "\n")
            if args.frame_every and (i + 1) % args.frame_every == 0:
                Image.fromarray(obs["pov"]).save(frames / f"step_{i + 1:04d}.png")
            if (i + 1) % 50 == 0:
                logger.info("step %d  coords=%s inv=%s", i + 1, info.get("coords"), _inventory(info))
            if done:
                logger.info("episode ended at step %d", i + 1)
                break
    finally:
        traj.close()
    dt = time.time() - t0
    summary.update(steps=ok_steps, step_s=round(dt, 1), steps_per_s=round(ok_steps / max(dt, 1e-6), 2))
    summary["inventory_end"] = _inventory(env.info)
    logger.info("%d steps in %.1fs (%.1f steps/s)", ok_steps, dt, summary["steps_per_s"])
    logger.info("inventory delta: %s -> %s", start_inv, summary["inventory_end"])

    Image.fromarray(obs["pov"]).save(frames / f"step_{ok_steps:04d}.png")
    env.close()
    logger.info("SMOKE TEST PASSED (%s)", summary["mode"])
    return _finish(0)


if __name__ == "__main__":
    sys.exit(main())
