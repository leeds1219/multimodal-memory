"""Start one condition (a method in a world) on the 32 hard tasks x worlds A, B, C: 15 chains, one per world and
task chunk, spread over the given GPUs. Each chain is experiments/chain.py.

    python experiments/run.py --method baseline --world standard --prefix vt --gpus 0,4 --port 29000 [--dry-run]
    python experiments/run.py --method gt_world_rule --world nolava --prefix ub --gpus 0 --port 29000
    python experiments/run.py --method baseline --world nolava --save 600 --prefix nb --gpus 4 --port 29200
    python experiments/run.py --method ours --world nolava --knowledge pl1 --prefix ev --gpus 0 --port 29400

Results: <runs>/results/<method>/<world>/<order>/...; orders are <prefix><offset>_<chunk> (e.g. vt0_3);
logs go to <runs>/logs/<order>.log. Record the launch in
<runs>/LAUNCHES.md and check nvidia-smi first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import paths  # noqa: E402,F401

from benchmark import RUN_DATA  # noqa: E402
from conditions import add_args, from_args  # noqa: E402

WORLD_OFFSETS = {"A": 0, "B": 1, "C": 4}  # --seed-offset of worlds A, B, C
HARD_CHUNKS = [  # the 32 hard tasks in 5 chunks, the same split for every condition
    "o1_gold_00,o1_gold_05,o1_redstone_04,o1_diamond_03,o1_armor_01,o1_armor_06,o1_armor_11",
    "o1_gold_01,o1_redstone_00,o1_redstone_05,o1_diamond_04,o1_armor_02,o1_armor_07,o1_armor_12",
    "o1_gold_02,o1_redstone_01,o1_diamond_00,o1_diamond_05,o1_armor_03,o1_armor_08",
    "o1_gold_03,o1_redstone_02,o1_diamond_01,o1_diamond_06,o1_armor_04,o1_armor_09",
    "o1_gold_04,o1_redstone_03,o1_diamond_02,o1_armor_00,o1_armor_05,o1_armor_10",
]
PY = "/opt/conda/envs/mcagent/bin/python"


def main() -> int:
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--prefix", required=True, help="order names <prefix><offset>_<chunk>")
    ap.add_argument("--worlds", default="A,B,C")
    ap.add_argument("--chunks", default="0,1,2,3,4")
    ap.add_argument("--gpus", required=True, help="comma list; chains are dealt round-robin")
    ap.add_argument("--port", type=int, required=True, help="first port; each chain uses port and port+1000")
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cond = from_args(a)
    cond_args = ["--method", a.method, "--world", a.world] + (["--save", str(a.save)] if a.save else []) + \
        (["--knowledge", a.knowledge] if a.knowledge else [])
    gpus = [int(g) for g in a.gpus.split(",")]
    logs = RUN_DATA / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"condition {cond.name} -> <runs>/results/{cond.name}/")
    port = a.port
    n = 0
    for w in a.worlds.split(","):
        off = WORLD_OFFSETS[w]
        for k in (int(c) for c in a.chunks.split(",")):
            order = f"{a.prefix}{off}_{k}"
            cmd = [PY, str(REPO / "experiments" / "chain.py"), *cond_args, "--order", order, "--tasks", HARD_CHUNKS[k],
                   "--gpu", str(gpus[n % len(gpus)]), "--port", str(port), "--seed-offset", str(off)] + (["--mock"] if a.mock else [])
            print(" ".join(cmd))
            if not a.dry_run:
                with open(logs / f"{order}.log", "a") as lf:
                    subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                time.sleep(20)  # Minecraft start-up under load: stagger the chains
            port += 10
            n += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
