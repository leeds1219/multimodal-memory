"""Compare our reproduction with the numbers each paper reports.

    python analysis/compare_papers.py   -> analysis/out/compare_groups.csv,
                                           analysis/out/compare_tasks.csv, stdout summary

Group level: our SR = successes / (tasks x task orders) with a Wilson 95 % CI;
a paper value outside that CI is a real difference, not seed noise.
Task level (where a paper reports per-task SR): two-sided exact binomial test
of our k successes in n runs against the paper's SR; p < 0.05 = inconsistent.
Inputs: analysis/our_numbers.json, analysis/paper_numbers.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
G = ["wooden", "stone", "iron", "gold", "redstone", "diamond", "armor"]

# ours (env/chain) -> list of (paper, paper row) references
REFS = {
    "mineevolve": [("mineevolve", "MineEvolve|Gemini-3-Flash")],
    "optimus1": [("mineevolve", "Optimus-1|Gemini-3-Flash"), ("optimus1", "Optimus-1|-"), ("optimus3", "Optimus-1|-")],
    "jarvis1": [("mineevolve", "JARVIS-1|Gemini-3-Flash"), ("jarvis1", "JARVIS-1|-"), ("optimus1", "JARVIS-1|-")],
    "deps": [("mineevolve", "DEPS|Gemini-3-Flash"), ("jarvis1", "DEPS|-"), ("optimus1", "DEPS|-")],
    "optimus3": [("optimus3", "Optimus-3|-")],
}
TASK_REFS = {"optimus1": [("optimus1", "Optimus-1|GPT-4V")], "jarvis1": [("jarvis1", "JARVIS-1|native")]}


def binom_p(k: int, n: int, p: float) -> float:
    """Two-sided exact binomial p-value (sum of outcomes no more likely than k)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    probs = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(q for q in probs if q <= probs[k] * (1 + 1e-9)))


def method_of(chain: str) -> str:
    return chain.split("-")[0]


def main() -> int:
    ours = json.loads((HERE / "our_numbers.json").read_text())
    papers = json.loads((HERE / "paper_numbers.json").read_text())
    OUT.mkdir(exist_ok=True)
    grows, trows = [], []
    for key, o in ours.items():
        env, chain = key.split("/")
        m = method_of(chain)
        for paper, row in REFS.get(m, []):
            pg = (papers.get(paper, {}).get("groups") or {}).get(row)
            if not pg:
                continue
            for g in G:
                og, pv = o["groups"][g], pg.get(g)
                if pv is None or og["n"] == 0:
                    continue
                lo, hi = og["ci95"]
                grows.append([env, chain, paper, row, g, og["sr"], lo, hi, og["n"], pv,
                              round(og["sr"] - pv, 1), "within" if lo <= pv <= hi else ("below" if og["sr"] < pv else "above")])
        for paper, row in TASK_REFS.get(m, []):
            pt = (papers.get(paper, {}).get("tasks") or {}).get(row) or {}
            for uid, t in pt.items():
                if t.get("sr") is None or t.get("match") == "none" or uid not in o["tasks"]:
                    continue
                k, n = o["tasks"][uid]["succ"], o["tasks"][uid]["runs"]
                pv = float(t["sr"]) / (100.0 if float(t["sr"]) > 1 else 1.0)
                trows.append([env, chain, paper, uid, t.get("paper_task", ""), t.get("match"), k, n,
                              round(100 * pv, 1), round(binom_p(k, n, pv), 3)])
    (OUT / "compare_groups.csv").write_text(
        "env,chain,paper,paper_row,group,our_sr,ci_lo,ci_hi,our_n,paper_sr,diff_pp,verdict\n" +
        "\n".join(",".join(map(str, r)) for r in grows) + "\n")
    (OUT / "compare_tasks.csv").write_text(
        "env,chain,paper,task,paper_task,match,our_succ,our_runs,paper_sr,binom_p\n" +
        "\n".join(",".join(map(str, r)) for r in trows) + "\n")
    # summary
    from collections import defaultdict
    s = defaultdict(lambda: [0, 0])
    for r in grows:
        s[(r[0], r[1], r[2])][0] += r[11] == "within"
        s[(r[0], r[1], r[2])][1] += 1
    print("groups within our 95% CI (per ours x paper):")
    for k, v in sorted(s.items()):
        print(f"  {k[0]}/{k[1]:26} vs {k[2]:10}: {v[0]}/{v[1]}")
    t = defaultdict(lambda: [0, 0])
    for r in trows:
        t[(r[0], r[1], r[2])][0] += r[9] >= 0.05
        t[(r[0], r[1], r[2])][1] += 1
    print("tasks consistent with paper SR (binomial p>=0.05):")
    for k, v in sorted(t.items()):
        print(f"  {k[0]}/{k[1]:26} vs {k[2]:10}: {v[0]}/{v[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
