"""Run Optimus-3 (Stage C, env "C3") over all 70 tasks, resumably.

No cross-task memory, so order is irrelevant (order0 is used for the path).
  * skip an episode whose result.json has status finished (or crashed_final);
  * a crashed episode (status crashed / no result.json / stalled) is moved aside
    to <seed>.crash<ts> and retried, up to --retries times, then recorded as
    crashed_final;
  * one subprocess per episode (fresh model load + fresh MinecraftSim), its whole
    process group killed if the heartbeat stalls for --stall-min;
  * stops if disk free < 30 GB.
  * --shard i/n runs every n-th task starting at i (several runners can share a GPU).

Example:
  CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/optimus3/bin/python run_all.py --gpu 6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "envs"))
from common import ORDERS, RUNS_ROOT, SEEDS, TASKS, disk_free_gb  # noqa: E402

PY = "/opt/conda/envs/optimus3/bin/python"
DONE = {"finished", "crashed_final"}
MIN_FREE_GB = 30.0


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def run_one(a, task: str, ep: Path) -> dict:
    cmd = [PY, str(HERE / "run_episode.py"), "--task", task, "--seed", str(SEEDS[task]), "--episode-dir", str(ep),
           "--order-id", a.order, "--prompt-form", a.prompt_form, "--biome", a.biome]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu)}
    t0 = time.time()
    reason = None
    with open(ep / "launcher.log", "a") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                             cwd=str(HERE))
        while p.poll() is None:
            time.sleep(15)
            hb = ep / "heartbeat"
            last = hb.stat().st_mtime if hb.exists() else t0
            if time.time() - last > a.stall_min * 60:
                reason = f"stalled {a.stall_min} min"
                log(f"  killing episode: {reason}")
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
                p.wait()
                break
    f = ep / "result.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"status": "crashed", "error": reason or f"exit code {p.returncode}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--order", default="order0")
    ap.add_argument("--tasks", default="", help="comma list; default = all 70 (order0)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stall-min", type=float, default=20.0)
    ap.add_argument("--prompt-form", default="obtain")
    ap.add_argument("--biome", default="mineevolve")
    ap.add_argument("--out", default=str(RUNS_ROOT / "runs" / "C3" / "optimus3"))
    a = ap.parse_args()
    if a.gpu in (0, 1, 2, 3, 4, 5, 7):
        raise SystemExit("GPU policy: only GPU 6 may be used for Optimus-3")
    tasks = a.tasks.split(",") if a.tasks else list(ORDERS[a.order])
    assert all(t in TASKS for t in tasks)
    si, sn = map(int, a.shard.split("/"))
    tasks = tasks[si::sn]
    root = Path(a.out) / a.order
    for idx, task in enumerate(tasks, 1):
        ep = root / task / str(SEEDS[task])
        f = ep / "result.json"
        if f.exists() and json.loads(f.read_text()).get("status") in DONE:
            continue
        res = {}
        for attempt in range(1, a.retries + 2):
            if disk_free_gb() < MIN_FREE_GB:
                log(f"STOP: disk free {disk_free_gb():.1f} GB < {MIN_FREE_GB}")
                return 2
            if ep.exists():
                shutil.move(str(ep), str(ep) + f".crash{int(time.time())}")
            ep.mkdir(parents=True)
            log(f"[{idx}/{len(tasks)}] {task} attempt {attempt}")
            res = run_one(a, task, ep)
            log(f"  -> {res.get('status')} {res.get('end_reason')} success={res.get('success')} "
                f"steps={res.get('steps')} wall={res.get('wall_time_s')}")
            if res.get("status") in DONE:
                break
        else:
            res = {**res, "status": "crashed_final", "task": task, "env": "C3", "method": "optimus3",
                   "group": TASKS[task]["group"], "seed": SEEDS[task], "order_id": a.order, "success": False}
            (ep / "result.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
