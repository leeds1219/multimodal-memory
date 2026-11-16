"""Rows for the results dashboard (Claude Data artifact): success, steps and cost per stage x condition x level, Ours'
loops and its World Knowledge. Writes <out>/{results,ours_loops,world_knowledge}.json as {rows, meta}, and
<out>/index.html (analysis/results_dashboard.html with the rows in it), the page to publish.

    python analysis/results_dashboard.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "run_data" / "state"
TRANSFER = ("redstone", "diamond", "armor")
# stage -> condition -> state files whose per-level measures add up (transfer: the Baseline's test tasks come from Phase 2)
RUNS = {
    "Phase 1": {"MineEvolve": ["phase1_p1me"], "Optimus-1 memory": ["phase1_p1om"],
                "Ours, previous loop": ["phase1_p1ours"], "Ours": ["phase1_p1ours2"]},
    "Phase 2": {"Baseline": ["phase2_baseline"], "MineEvolve": ["phase2_p1me_c105"],
                "Optimus-1 memory": ["phase2_p1om_c105"]},
    "Transfer": {"Baseline": ["phase2_baseline", "phase2_baseline_train"], "MineEvolve": ["phase2_p1me_c63"],
                 "Optimus-1 memory": ["phase2_p1om_c63"], "Ours": ["phase2_p1ours2_c63"]},
}
LEVELS = ("wooden", "stone", "iron", "gold", "redstone", "diamond", "armor")


def results() -> list:
    rows = []
    for stage, conds in RUNS.items():
        for cond, files in conds.items():
            for level in LEVELS:
                if stage == "Transfer" and level not in TRANSFER:
                    continue
                m = {"runs": 0, "successes": 0, "episodes": 0, "steps": 0, "cost_usd": 0.0}
                for f in files:
                    p = STATE / f"{f}.json"
                    x = json.loads(p.read_text())["measures"].get(level) if p.exists() else None
                    if x:
                        m["runs"] += x.get("units", x["episodes"])
                        for k in ("successes", "episodes", "steps", "cost_usd"):
                            m[k] += x[k]
                if m["runs"]:
                    rows.append({"stage": stage, "condition": cond, "level": level, **m,
                                 "cost_usd": round(m["cost_usd"], 2),
                                 "success_rate": round(m["successes"] / m["runs"], 3)})
    return rows


def loops(store: str) -> list:
    rows = []
    for d in sorted((REPO / "results" / "ours_loop").glob(f"p1-{store}-*/*/loop.json")):
        st, run = json.loads(d.read_text()), d.parent
        exps = st.get("experiments", [])
        verdicts = [e.get("verdict") for e in exps]
        cost = sum(json.loads(l).get("cost_usd") or 0 for l in open(run / "llm" / "calls.jsonl")) \
            if (run / "llm" / "calls.jsonl").exists() else 0
        for f in run.glob("e*/result.json"):
            if ".crash" not in f.parent.name:
                cost += json.loads(f.read_text()).get("cost_usd") or 0
        task, off = run.parent.name.split(f"p1-{store}-")[1].rsplit("-s", 1)
        rows.append({"loop": run.parent.name, "task": task, "world": f"s{off}",
                     "failure_points": "; ".join(f"{fp['milestone'][0]} {fp['milestone'][1]}, {fp['held']} held"
                                                 for fp in st.get("failure_points", [])),
                     "experiments": len(exps), "supported": verdicts.count("Supported"),
                     "refuted": verdicts.count("Refuted"), "not_reached": verdicts.count("Not reached"),
                     "outcome": st.get("done") or "running", "cost_usd": round(cost, 2)})
    return rows


def knowledge(store: str) -> list:
    p = REPO / "run_data" / "memory" / f"wk_{store}.json"
    return [{"statement": e["statement"], "verdict": e["verdict"], "k": e.get("k"), "n": e.get("n"),
             "source": e.get("source")} for e in (json.loads(p.read_text()) if p.exists() else [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default="p1ours2", help="Ours' Phase 1 store")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = {"label": "results_dashboard.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"}
    for name, rows, src in (("results", results(), "run_data/state/phase{1,2}_*.json"),
                            ("ours_loops", loops(a.store), f"results/ours_loop/p1-{a.store}-*/*/loop.json"),
                            ("world_knowledge", knowledge(a.store), f"run_data/memory/wk_{a.store}.json")):
        (out / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": [code, {"label": "Input", "value": src}]}))
        print(name, len(rows))
    data = {n: json.loads((out / f"{n}.json").read_text())["rows"] for n in ("results", "ours_loops", "world_knowledge")}
    data["updated"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    page = (REPO / "analysis" / "results_dashboard.html").read_text().replace("/*DATA*/null", json.dumps(data))
    (out / "index.html").write_text(page)


if __name__ == "__main__":
    main()
