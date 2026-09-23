"""Project full-run time and cost from smoke-test episodes.

For each (env, method) with smoke results: steps/s, $/step and calls/step are
measured; every one of the 70 tasks is assumed to run to its env's horizon
(upper bound, since failures do) and, as a mid estimate, to 60 % of it.
Chains of one task order run sequentially; chains run in parallel.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "envs"))
from common import RUNS_ROOT, TASKS  # noqa: E402
from cross_glue import env_m_group, env_o_group  # noqa: E402

MIN = 1200


def horizon_steps(env: str) -> int:
    f = env_m_group if env == "M" else env_o_group
    return sum(int(f(t["group"])["max_minutes"]) * MIN for t in TASKS.values())


def smoke_rates():
    acc = defaultdict(lambda: {"steps": 0, "wall": 0.0, "cost": 0.0, "calls": 0, "n": 0})
    for f in (RUNS_ROOT / "runs").glob("*/*/smoke/*/*/result.json"):
        if ".crash" in str(f):
            continue
        r = json.loads(f.read_text())
        if r.get("status") not in ("finished", "anomaly"):
            continue
        env, chain = f.parts[-6], f.parts[-5]
        a = acc[(env, chain)]
        a["steps"] += r["steps"]; a["wall"] += r["wall_time_s"]; a["cost"] += r["cost_usd"]
        a["calls"] += r["llm_calls"]; a["n"] += 1
    return acc


def main() -> int:
    plans = [yaml.safe_load(open(p)) for p in sorted((REPRO / "configs").glob("run_plan_stage*.yaml"))]
    chains = [c for p in plans for c in p["chains"]]
    rates = smoke_rates()
    rows, tot_hi, tot_mid, longest = [], 0.0, 0.0, 0.0
    print(f"{'env':3} {'chain':20} {'n':>2} {'steps/s':>7} {'$/1k steps':>10} {'calls/1k':>8} | "
          f"{'h/chain (max)':>13} {'$ /chain max':>12} {'$ /chain mid':>12}")
    for (env, chain), a in sorted(rates.items()):
        sps = a["steps"] / a["wall"] if a["wall"] else 0
        cps = a["cost"] / a["steps"] if a["steps"] else 0
        kps = a["calls"] / a["steps"] if a["steps"] else 0
        H = horizon_steps(env)
        hours = H / sps / 3600 if sps else float("inf")
        print(f"{env:3} {chain:20} {a['n']:2d} {sps:7.2f} {cps * 1000:10.3f} {kps * 1000:8.2f} | "
              f"{hours:13.1f} {cps * H:12.2f} {cps * H * 0.6:12.2f}")
        n_chains = sum(1 for c in chains if c["env"] == env and
                       c["method"] + (f"-{c['variant']}" if c.get("variant") else "") == chain)
        tot_hi += n_chains * cps * H
        tot_mid += n_chains * cps * H * 0.6
        if n_chains:
            longest = max(longest, hours)
    print(f"\nChains in run plans: {len(chains)}. Projected LLM cost: upper ${tot_hi:.0f}, mid ${tot_mid:.0f}.")
    print(f"Longest chain (all episodes to horizon): {longest:.0f} h wall, if all chains run in parallel.")
    print(f"Horizon steps per 70-task sweep: Env M {horizon_steps('M'):,}, Env O {horizon_steps('O'):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
