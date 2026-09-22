"""Local test of the `approach` primitive + nearby_blocks observation (no LLM API).

For each JARVIS-1 spawn: reset, print the landmark blocks the jar patch reports,
run `approach log` (deterministic), then let STEVE-1 ("chop a tree") run and count
the steps until the first log is in the inventory. Compare with the paper-condition
baseline where STEVE-1 starts blind (first log median ~850 steps, 23 % never).

    CUDA_VISIBLE_DEVICES=2 bash scripts/server.sh &          # STEVE-1 only, no LLM needed
    xvfb-run -a python scripts/test_approach.py [--no-approach] [--yaws 0,90,180,270]

--no-approach runs the same protocol without the primitive (control). --yaws sets
initial facing directions per spawn (default 0,120,240) so the control is not
lucky about where the tree is.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.utils import setup_globals

SPAWNS = [(19961103, (-79, 64, -512)), (19961103, (-145, 67, -495)), (12345, (195, 73, 812))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:9000")
    ap.add_argument("--no-approach", action="store_true")
    ap.add_argument("--yaws", default="0,120,240")
    ap.add_argument("--max-steps", type=int, default=2400)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    setup_globals()
    import mineevolve
    from mineevolve.client import MineEvolveClient
    from mineevolve.env import make_env
    from mineevolve.env.nearby_blocks import summarize_nearby
    from mineevolve.main import _ApproachScript, _safe_pov

    with initialize_config_dir(config_dir=str(Path(mineevolve.__file__).parent / "conf"), version_base=None):
        cfg = compose("evaluate", overrides=["benchmark=wooden"])
    env = make_env(cfg, logger=logging.getLogger("approach-test"))
    client = MineEvolveClient(base_url=a.server)
    out = Path("logs/smoke/approach-test"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    yaws = [int(v) for v in a.yaws.split(",")]
    for seed, pos in SPAWNS:
        for yaw0 in yaws:
            env.seed(seed)
            obs = env.reset()
            env.execute_cmd(f"/tp @s {pos[0]:.1f} {pos[1]:.1f} {pos[2]:.1f} {yaw0} 0")
            for k in range(60):  # same chunk-load wait as main.py's reset
                obs, _r, _d, info = env.step(env.action_space.noop())
                if k >= 10 and (info.get("nearby_blocks") or {}):
                    break
            nearby = info.get("nearby_blocks") or {}
            summary = summarize_nearby(nearby, info.get("coords"))
            steps_approach, reached, target = 0, None, None
            if not a.no_approach:
                sc = _ApproachScript(env, {"block": "log", "steps": 300})
                while True:
                    act = sc.next(info)
                    if act is None:
                        break
                    obs, _r, done, info = env.step(act); steps_approach += 1
                    if done:
                        break
                reached, target = sc.reached, sc.target
            # STEVE-1 chop until first log
            client.reset(task_goal="Chop an oak log from a tree")
            first_log, steps = None, 0
            while steps < a.max_steps:
                r = client.action(condition="chop a tree", obs={"image": _safe_pov(obs), "pov": _safe_pov(obs)})
                act = r.get("action")
                if act is None:
                    break
                obs, _r, done, info = env.step(act); steps += 1
                if any(k.endswith("_log") for k in (info.get("inventory") or {})):
                    first_log = steps
                    break
                if done:
                    break
            row = dict(seed=seed, pos=list(pos), yaw0=yaw0, approach=not a.no_approach, nearby=summary,
                       approach_steps=steps_approach, reached=reached, target=target,
                       first_log_steps=first_log, total_steps=steps_approach + (first_log or steps))
            rows.append(row)
            print(json.dumps(row))
            with (out / ("with_approach.jsonl" if not a.no_approach else "control.jsonl")).open("a") as fh:
                fh.write(json.dumps(row) + "\n")
    env.close()
    got = [r["total_steps"] for r in rows if r["first_log_steps"]]
    print(f"first log obtained in {len(got)}/{len(rows)} runs; median total steps {int(np.median(got)) if got else None}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
