"""Run ONE MineEvolve episode (client side) in a given evaluation env.

The MineEvolve server (STEVE-1 + LLM + knowledge store) is started separately
by the chain runner and persists across the tasks of one task order.

Glue applied here (every item is in DEVIATIONS.md):
  Env M (MineEvolve's own env)
    * CraftHelper -> functional craft/smelt primitive (upstream helper is a stub)
    * wall-clock subgoal deadlines measured in game time (1 s = 20 ticks)
    * world seeded with ``env.seed(seed)`` before reset
    * episode ends when MineRL reports done (horizon / death) instead of
      repairing and stepping a finished env
    * success = target item(s) in inventory at any step (paper criterion);
      upstream's substring heuristic recorded as ``native_success``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm"))

from common import (EpisodeEnd, EpisodeMonitor, GameTime, SEEDS, TASKS, disk_free_gb,  # noqa: E402
                    llm_summary, write_result)

MINUTE = 1200
ME_CONF = Path("/home/rag/data/multimodal-memory/MC-MineEvolve/src/mineevolve/conf")


def compose_cfg(group: str, port: int, episode_dir: Path):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ME_CONF), version_base=None):
        cfg = compose(config_name="evaluate", overrides=[
            f"benchmark={group}", "llm=gemini_shared", f"server.port={port}",
            "record.video.save=false", f"+artifact_dir={episode_dir / 'artifacts'}",
        ])
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["M", "O"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--order-id", default="order0")
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    if args.env != "M":
        raise SystemExit("Env O glue for MineEvolve lives in mineevolve_episode_envO (not yet)")

    task = TASKS[args.task]
    seed = int(args.seed if args.seed is not None else SEEDS[args.task])
    ep = Path(args.episode_dir)
    ep.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=ep / "client.log",
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("repro.mineevolve")

    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)

    import mineevolve.main as me
    from mineevolve.env import make_env
    from mineevolve.client.server_api import MineEvolveClient
    from functional_craft import FunctionalCraftHelper

    cfg = compose_cfg(task["group"], args.port, ep)
    horizon = int(me._benchmark_cfg(cfg).env.max_minutes) * MINUTE
    mon = EpisodeMonitor(ep, args.task, horizon)

    # ---- env + per-step monitor ------------------------------------------
    env = make_env(cfg, logger=log)
    inner_step = env.env.step

    def monitored_step(action):
        obs, reward, done, info = inner_step(action)
        mon.on_step(action, obs, done, info)
        return obs, reward, done, info

    env.env.step = monitored_step
    # Optimus's MineRL build (DECISIONS D10) adds ``execute_cmd`` to the raw
    # env; stock MineRL 1.0.2 (what Env M expects) has none, so MineEvolve's
    # wrapper sends chat commands as a normal step. Hide it to keep that path.
    env.env.execute_cmd = None
    env.seed(seed)

    # ---- method glue --------------------------------------------------------
    me.CraftHelper = FunctionalCraftHelper
    me.time = GameTime(mon)

    native = {}
    orig_succ = me._episode_succeeded

    def recorded_succ(task_goal, inventory):
        native["success"] = bool(orig_succ(task_goal, inventory))
        return native["success"]

    me._episode_succeeded = recorded_succ

    def guard(fn):
        def wrapped(*a, **k):
            if mon.over:
                raise EpisodeEnd(mon.end_reason)
            return fn(*a, **k)
        return wrapped

    me._run_subgoal = guard(me._run_subgoal)
    me._run_helper_subgoal = guard(me._run_helper_subgoal)
    client = MineEvolveClient(base_url=f"{cfg.server.url}:{cfg.server.port}", timeout=float(cfg.server.timeout))
    client.repair = guard(client.repair)

    status, err = "finished", None
    t0 = time.time()
    try:
        me.run_episode(
            env=env, client=client, task_goal=task["instruction"],
            max_subgoals=int(cfg.runtime.max_subgoals),
            max_steps_per_subgoal=int(cfg.runtime.max_steps_per_subgoal),
            subgoal_timeout_s=int(cfg.runtime.subgoal_timeout_s),
            eta_fail=float(cfg.runtime.eta_fail), recent_window=int(cfg.runtime.recent_window),
            budget_tokens=int(cfg.runtime.budget_tokens), top_k=int(cfg.runtime.top_k),
            artifact_dir=str(ep / "artifacts"), task_id=task["local_id"], run_idx=0,
            evidence_keyframe_interval=int(cfg.record.evidence.keyframe_interval),
        )
        end = "plan_finished"
    except EpisodeEnd as e:
        end = str(e) or "done"
    except Exception as e:  # crash inside the method
        status, end, err = "crashed", "exception", traceback.format_exc()
        log.error(err)
    finally:
        mon.close()
        try:
            env.close()
        except Exception:
            pass

    anomaly = (ep / "ANOMALY").exists()
    result = {
        "env": args.env, "method": os.environ.get("METHOD", "mineevolve"), "task": args.task,
        "group": task["group"], "instruction": task["instruction"], "seed": seed, "order_id": args.order_id,
        "status": "anomaly" if anomaly else status, "end_reason": mon.end_reason or end,
        "success": mon.success_step is not None, "success_step": mon.success_step,
        "native_success": native.get("success"), "steps": mon.steps, "horizon_steps": horizon,
        "wall_time_s": round(time.time() - t0, 1), "final_inventory": mon.last_inventory,
        **llm_summary(ep), "error": err, "disk_free_gb": round(disk_free_gb(), 1),
    }
    write_result(ep, result)
    print(json.dumps({k: result[k] for k in ("task", "status", "success", "steps", "llm_calls", "cost_usd")}))
    return 0 if status != "crashed" else 3


if __name__ == "__main__":
    sys.exit(main())
