"""Success table of one or more conditions: by task group and by world (A, B, C), with steps and LLM cost.

    python analysis/summary.py baseline/standard gt_world_rule/standard --orders vt,oh
    python analysis/summary.py baseline/nolava_save600 --orders nb --csv

A condition is a folder results/<method>/<world>/. --orders keeps the chains whose name starts with one of the
prefixes (a condition folder can hold several runs, e.g. a rerun). Crashed attempts are left out; an episode that
crashed on every attempt counts as a failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

from benchmark import RESULTS, RUN_DATA, SEEDS  # noqa: E402

GROUPS = ("wooden", "stone", "iron", "gold", "redstone", "diamond", "armor")
WORLDS = {0: "A", 1: "B", 4: "C"}
DONE = ("finished", "anomaly", "crashed_final")


def episodes(cond: str, prefixes: list[str]) -> list[dict]:
    out = []
    for f in sorted((RESULTS / cond).glob("*/*/*/result.json")):
        order, task, seed = f.parts[-4:-1]
        if not seed.isdigit() or (prefixes and not any(order.startswith(p) for p in prefixes)):  # .crash<t>, .budgetstop<t>: set aside
            continue
        r = json.loads(f.read_text())
        if r.get("status") not in DONE:
            continue
        off = int(seed) - SEEDS[task]
        out.append({"order": order, "task": task, "group": r.get("group") or task.split("_")[1],
                    "world": WORLDS.get(off, f"+{off}"), "success": bool(r.get("success")),
                    "steps": r.get("steps") or 0, "cost": r.get("cost_usd") or 0.0})
    return out


def table(rows: dict, cols: list[str]) -> str:
    head = "| condition | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    return "\n".join([head, sep] + [f"| {name} | " + " | ".join(cells.get(c, "") for c in cols) + " |"
                                     for name, cells in rows.items()])


def frac(eps) -> str:
    return f"{sum(e['success'] for e in eps)}/{len(eps)}" if eps else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="+", help="results/<method>/<world>, e.g. baseline/standard")
    ap.add_argument("--orders", default="", help="comma list of chain-name prefixes, one per condition or one for all")
    ap.add_argument("--csv", action="store_true", help="also write <runs>/analysis/summary.csv")
    a = ap.parse_args()
    pre = a.orders.split(",") if a.orders else []
    by_group, by_world, totals, csv = {}, {}, {}, []
    for i, cond in enumerate(a.conditions):
        p = [pre[i]] if len(pre) == len(a.conditions) else pre
        eps = episodes(cond, p)
        g, w = defaultdict(list), defaultdict(list)
        for e in eps:
            g[e["group"]].append(e)
            w[e["world"]].append(e)
            csv.append({"condition": cond, **e})
        by_group[cond] = {k: frac(v) for k, v in g.items()}
        by_world[cond] = {k: frac(v) for k, v in w.items()}
        n = len(eps) or 1
        totals[cond] = {"success": frac(eps), "mean steps": f"{sum(e['steps'] for e in eps) / n:,.0f}",
                        "LLM cost": f"${sum(e['cost'] for e in eps):,.0f}", "$/episode": f"{sum(e['cost'] for e in eps) / n:.2f}"}
    groups = [x for x in GROUPS if any(x in r for r in by_group.values())]
    print(table(totals, ["success", "mean steps", "LLM cost", "$/episode"]), "\n")
    print(table(by_group, groups), "\n")
    print(table(by_world, ["A", "B", "C"]))
    if a.csv:
        out = RUN_DATA / "analysis" / "summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(csv[0]) if csv else []
        out.write_text("\n".join([",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in csv]) + "\n")
        print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
