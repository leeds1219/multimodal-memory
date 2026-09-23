"""Run ONE Stage B episode (DEPS or JARVIS-1) in Env M or Env O.

STEVE-1 is served by the env's own server (MineEvolve server for Env M,
Optimus-1 server for Env O), started by the chain runner; their LLM parts
are not used by these methods.
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
sys.path.insert(0, str(HERE.parent / "baselines" / "deps"))
sys.path.insert(0, str(HERE.parent / "baselines" / "jarvis1"))

from common import (EpisodeEnd, EpisodeMonitor, SEEDS, TARGETS, TASKS, disk_free_gb,  # noqa: E402
                    is_method_exception, llm_summary, write_result)

STAGEB_CODE = (str(HERE.parent / "baselines"),)


def task_item(uid: str) -> str:
    it = TARGETS[uid]["any_of"][0]
    return {"#logs": "log", "#saplings": "sapling"}.get(it, it)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["deps", "jarvis1"])
    ap.add_argument("--env", required=True, choices=["M", "O"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--order-id", default="order0")
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    task = TASKS[a.task]
    seed = int(a.seed if a.seed is not None else SEEDS[a.task])
    ep = Path(a.episode_dir)
    ep.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=ep / "client.log",
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(f"repro.{a.method}")

    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)

    from gemini_client import GeminiClient
    from stageb_env import make_env

    mon = EpisodeMonitor(ep, a.task, horizon_steps=10**9)  # horizon set by the env
    env = make_env(a.env, task["group"], seed, mon, a.port, logger)
    client = GeminiClient()
    if a.method == "deps":
        from deps_agent import DepsAgent
        agent = DepsAgent(env, client, task_item(a.task), ep / "artifacts")
    else:
        from jarvis_agent import JarvisAgent
        agent = JarvisAgent(env, client, task_item(a.task), ep / "artifacts")

    status, err, end = "finished", None, None
    t0 = time.time()
    try:
        end = agent.run()
    except EpisodeEnd as e:
        end = str(e) or "horizon"
    except Exception as e:
        err = traceback.format_exc()
        logger.error(err)
        frames = traceback.extract_tb(e.__traceback__)
        if frames and frames[-1].filename.startswith(STAGEB_CODE) or is_method_exception(e):
            end = "method_exception"
        else:
            status, end = "crashed", "exception"
    finally:
        try:
            agent.save()
        except Exception:
            pass
        mon.close()
        env.close()

    result = {
        "env": a.env, "method": a.method, "task": a.task, "group": task["group"],
        "instruction": task["instruction"], "seed": seed, "order_id": a.order_id,
        "status": "anomaly" if (ep / "ANOMALY").exists() else status,
        "end_reason": mon.end_reason if mon.over else end,
        "success": mon.success_step is not None, "success_step": mon.success_step,
        "native_success": end == "task_done", "steps": mon.steps, "horizon_steps": env.horizon,
        "wall_time_s": round(time.time() - t0, 1), "final_inventory": mon.last_inventory,
        "replan_rounds": getattr(agent, "replan_rounds", None), **llm_summary(ep), "error": err,
        "disk_free_gb": round(disk_free_gb(), 1),
    }
    write_result(ep, result)
    print(json.dumps({k: result[k] for k in ("task", "status", "success", "steps", "llm_calls", "cost_usd")}))
    return 0 if status != "crashed" else 3


if __name__ == "__main__":
    sys.exit(main())
