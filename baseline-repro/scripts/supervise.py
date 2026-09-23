"""Keep every chain of a run plan running until it finishes.

    nohup python scripts/supervise.py --plan configs/run_plan.yaml &

* GPUs are chosen once (memory.used == 0 and a matmul test) and saved to
  ``state/gpu_assignment.json``; GPUs holding other users' memory are never used.
* A chain that exits 0 is done; exit 2 is a stop condition (disk / LLM cap)
  and stops the supervisor; any other exit is relaunched (resume) up to
  ``max_restarts`` times.
* Once a day a status summary is appended to PROGRESS.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPRO = Path(__file__).resolve().parents[1]
RUNS = Path("/home/rag/data/repro_runs")
PY = "/opt/conda/envs/mcagent/bin/python"


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def idle_gpus(exclude: list[int]) -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    ok = []
    for line in out.strip().splitlines():
        idx, used = (int(x) for x in line.split(","))
        if used != 0 or idx in exclude:
            continue
        r = subprocess.run([PY, "-c", "import torch;x=torch.randn(4096,4096,device='cuda');print(float((x@x).sum()))"],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(idx)}, capture_output=True, text=True)
        if r.returncode == 0:
            ok.append(idx)
    return ok


def chain_name(c: dict) -> str:
    return f"{c['env']}_{c['method']}{'-' + c['variant'] if c.get('variant') else ''}_{c['order']}"


def daily_summary() -> None:
    st = subprocess.run([PY, str(REPRO / "scripts" / "status.py")], capture_output=True, text=True).stdout
    with open(REPRO / "PROGRESS.md", "a") as f:
        f.write(f"\n## {dt.date.today().isoformat()} — daily summary (auto)\n\n```\n{st}```\n")
    subprocess.run(["git", "-C", str(REPRO.parent), "add", "baseline-repro/PROGRESS.md"], capture_output=True)
    subprocess.run(["git", "-C", str(REPRO.parent), "commit", "-q", "-m", f"Daily progress {dt.date.today()}",
                    "-m", "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_019Fpj6emzeuryAB3w5QKYvb"], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--max-restarts", type=int, default=20)
    a = ap.parse_args()
    plan = yaml.safe_load(open(a.plan))
    chains = plan["chains"]
    state_f = RUNS / "state" / "gpu_assignment.json"
    state_f.parent.mkdir(parents=True, exist_ok=True)
    if state_f.exists():
        gpus = json.loads(state_f.read_text())["gpus"]
    else:
        gpus = idle_gpus(plan.get("exclude_gpus", []))[: plan.get("max_gpus", 4)]
        state_f.write_text(json.dumps({"gpus": gpus, "chosen": time.ctime()}))
    if not gpus:
        log("no idle GPU"); return 1
    log(f"GPUs: {gpus}; chains: {len(chains)}")
    procs: dict[str, subprocess.Popen] = {}
    restarts: dict[str, int] = {}
    done: set[str] = set()
    last_day = None
    logs = RUNS / "supervisor"
    logs.mkdir(exist_ok=True)
    while True:
        for i, c in enumerate(chains):
            name = chain_name(c)
            if name in done:
                continue
            p = procs.get(name)
            if p is not None and p.poll() is None:
                continue
            if p is not None:
                rc = p.returncode
                if rc == 0:
                    log(f"{name}: finished"); done.add(name); continue
                if rc == 2:
                    log(f"{name}: STOP condition (exit 2) -> stopping supervisor")
                    for q in procs.values():
                        if q.poll() is None:
                            q.terminate()
                    return 2
                restarts[name] = restarts.get(name, 0) + 1
                log(f"{name}: exited {rc}, restart #{restarts[name]}")
                if restarts[name] > a.max_restarts:
                    log(f"{name}: too many restarts, giving up"); done.add(name); continue
            gpu = gpus[i % len(gpus)]
            port = 9300 + i
            cmd = [PY, str(REPRO / "scripts" / "chain.py"), "--env", c["env"], "--method", c["method"],
                   "--order", c["order"], "--gpu", str(gpu), "--port", str(port)]
            if c.get("variant"):
                cmd += ["--variant", c["variant"]]
            lf = open(logs / f"{name}.log", "a")
            procs[name] = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                                           cwd=str(REPRO))
            log(f"{name}: launched on GPU {gpu} port {port}")
            time.sleep(20)  # stagger Minecraft launches
        if len(done) == len(chains):
            log("all chains done"); daily_summary(); return 0
        today = dt.date.today()
        if last_day is not None and today != last_day:
            daily_summary()
        last_day = today
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
