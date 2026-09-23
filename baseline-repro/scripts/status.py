"""Progress, success so far and cost so far, per environment / method / order.

    python scripts/status.py            # table
    python scripts/status.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "envs"))
sys.path.insert(0, str(REPRO / "llm"))
from common import ORDERS, RUNS_ROOT, TASKS  # noqa: E402

EXCLUDE_ORDERS = {"smoke", "mocktest"}


def collect():
    rows = defaultdict(lambda: {"done": 0, "success": 0, "cost": 0.0, "calls": 0, "anomaly": 0,
                                "crashed_final": 0, "running": None, "wall_h": 0.0, "steps": 0,
                                "plan_fallback": 0})
    for res in (RUNS_ROOT / "runs").glob("*/*/*/*/*/result.json"):
        env, chain, order, task, seed = res.parts[-6:-1]
        if order in EXCLUDE_ORDERS or ".crash" in seed:
            continue
        try:
            r = json.loads(res.read_text())
        except ValueError:
            continue
        k = (env, chain, order)
        row = rows[k]
        row["done"] += 1
        row["success"] += bool(r.get("success"))
        row["cost"] += r.get("cost_usd") or 0.0
        row["calls"] += r.get("llm_calls") or 0
        row["anomaly"] += r.get("status") == "anomaly"
        row["crashed_final"] += r.get("status") == "crashed_final"
        row["wall_h"] += (r.get("wall_time_s") or 0) / 3600
        row["steps"] += r.get("steps") or 0
        row["plan_fallback"] += r.get("plan_source") == "example_fallback"
    now = time.time()
    for hb in (RUNS_ROOT / "runs").glob("*/*/*/*/*/heartbeat"):
        env, chain, order, task, seed = hb.parts[-6:-1]
        if order in EXCLUDE_ORDERS or (hb.parent / "result.json").exists():
            continue
        if now - hb.stat().st_mtime < 1800:
            rows[(env, chain, order)]["running"] = f"{task} @{hb.read_text().split()[0]}"
    return rows


def ledger():
    from gemini_client import ledger_total
    return ledger_total()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows = collect()
    total_cost = ledger()
    free = shutil.disk_usage("/home/rag/data").free / 1e9
    if a.json:
        print(json.dumps({"rows": {"/".join(k): v for k, v in rows.items()}, "ledger_usd": total_cost,
                          "disk_free_gb": free}, indent=2))
        return 0
    n_tasks = len(TASKS)
    print(f"{'env':3} {'chain':22} {'order':7} {'done':>7} {'succ%':>6} {'cost$':>8} {'calls':>6} "
          f"{'anom':>4} {'crash':>5} {'fallbk':>6} {'wall_h':>7}  running")
    for k in sorted(rows):
        r = rows[k]
        sr = 100 * r["success"] / r["done"] if r["done"] else 0.0
        print(f"{k[0]:3} {k[1]:22} {k[2]:7} {r['done']:3}/{n_tasks:<3} {sr:6.1f} {r['cost']:8.2f} {r['calls']:6d} "
              f"{r['anomaly']:4d} {r['crashed_final']:5d} {r['plan_fallback']:6d} {r['wall_h']:7.1f}  {r['running'] or ''}")
    by_env = defaultdict(float)
    for k, r in rows.items():
        by_env[k[0]] += r["cost"]
    print(f"\nLLM spend (ledger, all runs incl. smoke/debug): ${total_cost:.2f} / cap $3000   "
          f"per env: {dict((k, round(v, 2)) for k, v in by_env.items())}   disk free: {free:.1f} GB")
    stop = RUNS_ROOT / "STOP_DISK"
    if stop.exists():
        print("!! STOP_DISK present:", stop.read_text().strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
