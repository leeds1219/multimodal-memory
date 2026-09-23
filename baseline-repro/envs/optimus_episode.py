"""Run ONE Optimus-1 episode (client side) in a given evaluation env.

The Optimus-1 server (app.py: STEVE-1 + Gemini planner/reflector) is started by
the chain runner with the same per-chain working directory, because Optimus-1
resolves memory, recipes, checkpoints and step images relative to the cwd.

Glue applied here (see DEVIATIONS.md):
  * one task per process, task text injected through Hydra (``all_task`` /
    ``evaluate``), the group's own benchmark yaml for horizon / biome;
  * config composed from the absolute conf dir and ``main.__wrapped__(cfg)``
    called directly (Hydra cannot import ``optimus1.conf`` from the editable
    install; no chdir, so the relative paths keep resolving);
  * world seeded with ``env.seed(seed)`` before reset;
  * per-step monitor on the raw MineRL step (paper success criterion);
    Optimus-1's own verdict (all plan steps done) kept as ``native_success``;
  * video / action pickles off (optional recordings);
  * a janitor keeps, per subtask name, only the image Optimus-1 can read back
    (the lowest step, see app.py ``_filter_task_obs``) plus recent files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm"))

from common import EpisodeMonitor, SEEDS, TASKS, disk_free_gb, is_method_exception, llm_summary, write_result  # noqa: E402

MINUTE = 1200
O_GROUP = {"wooden": "wooden", "stone": "stone", "iron": "iron", "gold": "golden",
           "redstone": "redstone", "diamond": "diamond", "armor": "armor"}
_IMG = re.compile(r"^(?P<uuid>[^_]+)_(?P<task>.+)_(?P<step>\d+)\.jpg$")


def janitor(img_dir: Path, stop: threading.Event, keep_recent_s: float = 60.0) -> None:
    """Delete step images Optimus-1 can never read again (infra only)."""
    while not stop.wait(30.0):
        try:
            best: dict[str, tuple[int, Path]] = {}
            files = []
            now = time.time()
            for f in img_dir.iterdir():
                m = _IMG.match(f.name)
                if not m:
                    continue
                task, step = m["task"], int(m["step"])
                files.append((f, task, step))
                if task not in best or step < best[task][0]:
                    best[task] = (step, f)
            keep = {v[1] for v in best.values()}
            for f, _, _ in files:
                if f not in keep and now - f.stat().st_mtime > keep_recent_s:
                    f.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["M", "O"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--order-id", default="order0")
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    task = TASKS[args.task]
    seed = int(args.seed if args.seed is not None else SEEDS[args.task])
    ep = Path(args.episode_dir)
    ep.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)

    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)

    # Optimus-1 registers its envs as ``src.optimus1.env...`` (it is run from
    # its repo root); make that importable while cwd is the chain workdir.
    sys.path.append("/home/rag/data/official/NeurIPS24-Optimus-1")
    import yaml
    import optimus1.main as om

    from cross_glue import env_m_group, env_o_group, optimus_in_M

    genv = env_o_group(task["group"]) if args.env == "O" else env_m_group(task["group"])
    horizon = int(genv["max_minutes"]) * MINUTE
    mon = EpisodeMonitor(ep, args.task, horizon)
    if args.env == "M":
        optimus_in_M(om, mon, task["group"], seed)

    # ---- Env O: seed + per-step monitor on Optimus-1's own env ------------
    orig_make = om.env_make

    def env_make(env_id, cfg, logger):
        env = orig_make(env_id, cfg, logger)
        raw = env.env
        inner_step = raw.step

        def monitored_step(action):
            obs, reward, done, info = inner_step(action)
            mon.on_step(action, obs, done, info)
            return obs, reward, done, info

        raw.step = monitored_step
        env.seed(seed)
        return env

    if args.env == "O":
        om.env_make = env_make

    if os.environ.get("OPTIMUS_GOALFIX") == "1":
        # Variant "goalfix" (DECISIONS D29): Gemini answers <goal inference> with a
        # list ("stone pickaxe, cobblestone, sticks, ..."), the prompt's own example
        # with a single item. Keep the first listed item; nothing else changes.
        import re as _re
        _orig_info = om.get_info_from_plan

        def get_info_from_plan(data):
            goal, visual, env_ = _orig_info(data)
            first = _re.split(r"[,;(/]|\band\b", goal)[0].strip().rstrip(".").strip()
            return (first or goal), visual, env_

        om.get_info_from_plan = get_info_from_plan

    native = {}
    orig_do = om.agent_do

    def agent_do(*a, **k):
        status, steps, plan = orig_do(*a, **k)
        native.update(status=status, steps=steps)
        return status, steps, plan

    om.agent_do = agent_do

    stop = threading.Event()
    img_dir = Path("imgs"); img_dir.mkdir(exist_ok=True)
    threading.Thread(target=janitor, args=(img_dir, stop), daemon=True).start()

    instr = task["instruction"].replace('"', "'")
    overrides = [
        f"benchmark={O_GROUP[task['group']]}",
        f'all_task=[{{id:0,type:{task["type"]},instruction:"{instr}"}}]',
        "evaluate=[0]", "env.times=1", f"server.port={args.port}",
        "record.video.save=false", "record.action.save=false",
    ]
    log_file = open(ep / "client.log", "a")
    sys.stdout = sys.stderr = log_file
    from hydra import compose, initialize_config_dir

    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    with initialize_config_dir(config_dir=str(Path(om.__file__).parent / "conf"), version_base=None):
        cfg = compose(config_name="evaluate", overrides=overrides, return_hydra_config=True)
    (ep / "hydra").mkdir(exist_ok=True)
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = str(ep / "hydra")
    HydraConfig.instance().set_config(cfg)  # Optimus-1's logger reads output_dir from it
    status, err = "finished", None
    t0 = time.time()
    try:
        om.main.__wrapped__(cfg)
    except SystemExit:
        pass
    except BaseException as e:
        err = traceback.format_exc()
        print(err)
        # The JARVIS helpers raise RuntimeError("Timeout!") when the horizon
        # runs out mid-GUI-action (uncaught in optimus1.main as released):
        # that is the episode's normal end, not a crash.
        if mon.steps >= horizon or mon.over or "Timeout!" in str(e):
            mon.end_reason = mon.end_reason or "horizon"
            native.setdefault("status", "failed")
        elif is_method_exception(e):  # released code failed: episode over (D25)
            mon.end_reason = "method_exception"
            native.setdefault("status", "failed")
        else:
            status = "crashed"
    finally:
        mon.close()
        stop.set()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

    if not native:
        status = "crashed" if status == "finished" else status
        err = err or "agent_do did not return (see client.log)"
    llm = llm_summary(ep)
    # Optimus-1 silently falls back to its built-in example plan when the
    # planning request is never made (e.g. retrieve_graph KeyError on the
    # inferred goal): record where the executed plan came from (DECISIONS D20).
    plan_source = "llm" if llm["llm_calls_by_caller"].get("planning") else "example_fallback"
    anomaly = (ep / "ANOMALY").exists()
    result = {
        "env": args.env, "method": os.environ.get("METHOD", "optimus1"), "task": args.task,
        "group": task["group"], "instruction": task["instruction"], "seed": seed, "order_id": args.order_id,
        "status": "budget_stopped" if (ep / "BUDGET_STOP").exists() else ("anomaly" if anomaly else status),
        "end_reason": mon.end_reason or ("plan_finished" if native.get("status") == "success" else "plan_failed_or_timeout"),
        "success": mon.success_step is not None, "success_step": mon.success_step,
        "native_success": (native.get("status") == "success") if native else None,
        "steps": mon.steps, "horizon_steps": horizon, "wall_time_s": round(time.time() - t0, 1),
        "final_inventory": mon.last_inventory, "plan_source": plan_source, **llm, "error": err,
        "disk_free_gb": round(disk_free_gb(), 1),
    }
    write_result(ep, result)
    print(json.dumps({k: result[k] for k in ("task", "status", "success", "steps", "llm_calls", "cost_usd")}))
    return 0 if status != "crashed" else 3


if __name__ == "__main__":
    sys.exit(main())
