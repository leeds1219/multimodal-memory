"""Per-sub-goal steps of an Optimus-1 run, from the monitors dict that main.py logs at the end of
every episode ({'<subgoal>_<i>': {'SuccessMonitor': 0/1, 'StepMonitor': n}}). Writes
analysis/out/subgoal_steps.csv (one row per sub-goal).

  python analysis/subgoal_steps.py [variant] [order_prefix]
"""
import ast
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_failures import ROOT, messages  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else \
    "optimus1-prebuilt-logfix-memfix-craftfix-tagfix-promptfix-replanfix-escapefix5-isoworld-g38"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "final"
OUT = Path(__file__).parent / "out"


def kind(sub: str) -> str:
    s = sub.lower()
    if s.startswith(("craft", "create")):
        return "craft"
    if s.startswith("smelt") or "smelt" in s:
        return "smelt"
    if s.startswith("equip"):
        return "equip"
    for key, name in (("diamond", "mine diamond"), ("gold", "mine gold"), ("redstone", "mine redstone"),
                      ("iron", "mine iron"), ("coal", "mine coal"), ("cobblestone", "mine cobblestone"),
                      ("stone", "mine cobblestone"), ("log", "chop wood"), ("tree", "chop wood"),
                      ("wood", "chop wood"), ("sand", "dig sand")):
        if key in s:
            return name
    return "other (replan/escape/explore)"


def main():
    rows = []
    for res in sorted(ROOT.glob(f"{VARIANT}/{PREFIX}*/*/*/result.json")):
        d = res.parent
        if "." in d.name or not (d / "client.log").exists():
            continue
        r = json.loads(res.read_text())
        dump = [m for m in messages(d / "client.log") if m.startswith("{'") and "SuccessMonitor" in m]
        if not dump:
            continue
        try:
            mon = ast.literal_eval(re.sub(r"\s+", " ", dump[-1]))
        except (SyntaxError, ValueError):
            continue
        for key, v in mon.items():
            sub, _, idx = key.rpartition("_")
            rows.append({"order": r["order_id"], "task": r["task"], "group": r["group"], "seed": r["seed"],
                         "ep_success": bool(r["success"]), "ep_steps": r["steps"], "horizon": r["horizon_steps"],
                         "idx": int(idx) if idx.isdigit() else -1, "subgoal": sub, "kind": kind(sub),
                         "done": v.get("SuccessMonitor", 0), "steps": v.get("StepMonitor", 0)})
    OUT.mkdir(exist_ok=True)
    with open(OUT / "subgoal_steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "sub-goal rows from", len({(r["order"], r["task"]) for r in rows}), "episodes")


if __name__ == "__main__":
    main()
