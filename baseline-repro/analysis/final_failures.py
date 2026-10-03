"""Failure breakdown of an Optimus-1 run (suite_optimus1): per episode, the subgoal it was on
when it ended, plus flags from client.log (reflector predicaments, escapes, crafting-table GUI
failures). Writes analysis/out/final_failures.{csv,txt}.

  python analysis/final_failures.py [variant] [order_prefix]
"""
import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/rag/data/repro_runs/suite_optimus1/runs/O")
VARIANT = sys.argv[1] if len(sys.argv) > 1 else \
    "optimus1-prebuilt-logfix-memfix-craftfix-tagfix-promptfix-replanfix-escapefix5-isoworld-g38"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "final"
OUT = Path(__file__).parent / "out"

PAPER = {"wooden": 98.60, "stone": 92.35, "iron": 46.69, "gold": 8.51, "diamond": 11.61,
         "redstone": 25.02, "armor": 19.47}  # Table 1
SRC = re.compile(r"\s+[\w.]+\.py:\d+\s*$")
# promptfix v1 rewrote a non-mined goal (D45); v2 rewrites only mined blocks
OVERMATCH = re.compile(r"dig down and mine (?!(cobblestone|stone|coal|coals|diamond|diamonds|redstone|"
                       r"lapis lazuli|gold|(deepslate )?\w+ ore),)")


def messages(log: Path) -> list[str]:
    """Rich console log -> one string per record (continuation lines joined)."""
    out = []
    for line in log.read_text(errors="replace").splitlines():
        if len(line) < 30:
            continue
        head, body = line[:29], SRC.sub("", line[29:]).strip()
        if head.strip():
            out.append(body)
        elif out and body:
            out[-1] += " " + body
    return out


def stage(goal: str, task: str) -> str:
    g, t = goal.lower(), task.lower()
    if t.startswith(("smelt", "cook")) or "ingot" in g or g in ("charcoal", "stone", "glass", "smooth_stone"):
        return "smelt"
    if t.startswith(("craft", "equip")):
        return "craft"
    for key, name in (("log", "wood"), ("wood", "wood"), ("plank", "wood"), ("cobblestone", "cobblestone"),
                      ("coal", "coal"), ("iron_ore", "iron_ore"), ("gold_ore", "gold_ore"),
                      ("diamond", "diamond"), ("redstone", "redstone"), ("sand", "sand"), ("__escape__", "escape")):
        if key in g:
            return name
    return "other:" + g


def episode(d: Path) -> dict:
    r = json.loads((d / "result.json").read_text())
    msgs = messages(d / "client.log") if (d / "client.log").exists() else []
    tasks = [m for m in msgs if m.startswith("Current Task:")]
    last = tasks[-1] if tasks else ""
    m = re.match(r"Current Task: (.*?), Goal: \[?'?([\w:]+)'?,?\s*(\d+)?", last)
    task, goal = (m.group(1), m.group(2)) if m else ("", "")
    text = "\n".join(msgs)
    return {
        "order": r["order_id"], "task": r["task"], "group": r["group"], "seed": r["seed"],
        "instruction": r["instruction"], "success": bool(r["success"]), "status": r["status"],
        "end_reason": r.get("end_reason"), "steps": r.get("steps"), "horizon": r.get("horizon_steps"),
        "last_subgoal": task, "last_goal": goal, "stage": stage(goal, task) if not r["success"] else "",
        "n_subgoals": len(tasks), "replans": text.count("Situation: replan"),
        "in_water": text.count("Predicament: in_water"), "escapes": text.count("escapefix:"),
        "escape_ok": text.count("exited=True"),
        "gui_fail": text.count("could not be opened"), "missing_material": text.count("missing material"),
        "pf_overmatch": int(bool(OVERMATCH.search(last))), "dirt_goal": int("Goal: ['dirt'," in last),
        "llm_calls": r.get("llm_calls"), "cost": r.get("cost_usd"), "cause": "",
        "inventory": {k: v for k, v in (r.get("final_inventory") or {}).items() if v},
    }


def cause(e: dict) -> str:
    """One primary cause per failure, first match wins."""
    st = e["stage"]
    if e["status"] != "finished":
        return "infra"
    if e["pf_overmatch"]:
        return "ours: promptfix v1 rewrote a non-mined goal"
    if e["dirt_goal"]:
        return "ours: replan sub-goal with a dirt goal (pillar/select)"
    if st == "craft" and e["gui_fail"]:
        return "crafting-table GUI never opened (water/cramped)"
    trapped = e["escapes"] >= 2 and e["escape_ok"] == 0 or e["in_water"]
    if st in ("diamond", "gold_ore", "iron_ore", "redstone", "coal"):
        return "ore not found within the horizon" + (" (time lost in a trap earlier)" if trapped else "")
    if trapped:
        return "terrain trap (pit/ravine/water), escape failed"
    if st in ("wood", "cobblestone", "sand"):
        return "slow gathering (wood/cobblestone)"
    if st in ("craft", "smelt"):
        return "craft/smelt failed (materials or GUI)"
    return "other planner sub-goal (" + st + ")"


def main():
    eps = [episode(p.parent) for p in sorted(ROOT.glob(f"{VARIANT}/{PREFIX}*/*/*/result.json"))
           if "." not in p.parent.name]
    for e in eps:
        e["cause"] = cause(e) if not e["success"] else ""
    OUT.mkdir(exist_ok=True)
    with open(OUT / "final_failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0]))
        w.writeheader()
        w.writerows(eps)
    lines = [f"{VARIANT} {PREFIX}*: {len(eps)} episodes"]
    by_g = collections.defaultdict(list)
    for e in eps:
        by_g[e["group"]].append(e)
    lines.append(f"\n{'group':9s} {'ours':>13s} {'paper':>6s}  failures by stage at the end")
    for g in PAPER:
        v = by_g[g]
        s = sum(e["success"] for e in v)
        st = collections.Counter(e["stage"] for e in v if not e["success"])
        lines.append(f"{g:9s} {s:3d}/{len(v):3d} {100*s/len(v):5.1f}% {PAPER[g]:6.1f}  " +
                     ", ".join(f"{k} {n}" for k, n in st.most_common()))
    fail = [e for e in eps if not e["success"]]
    lines.append(f"\nall failures: {len(fail)}")
    lines.append("  end reasons: " + str(collections.Counter(e["end_reason"] for e in fail).most_common()))
    lines.append("  stage: " + str(collections.Counter(e["stage"] for e in fail).most_common()))
    lines.append("\nprimary cause (all groups | wood+stone | iron | gold+diamond+redstone+armor):")
    easy, iron = ("wooden", "stone"), ("iron",)
    for c, n in collections.Counter(e["cause"] for e in fail).most_common():
        sub = [sum(e["cause"] == c and (e["group"] in grp if grp else e["group"] not in easy + iron) for e in fail)
               for grp in (easy, iron, None)]
        lines.append(f"  {n:3d}  {c:55s} {sub[0]:3d} {sub[1]:3d} {sub[2]:3d}")
    for flag in ("replans", "in_water", "escapes", "gui_fail", "missing_material"):
        nf = sum(e[flag] > 0 for e in fail)
        ns = sum(e[flag] > 0 for e in eps if e["success"])
        lines.append(f"  {flag:16s} in {nf:3d}/{len(fail)} failures, {ns:3d}/{len(eps)-len(fail)} successes")
    lines.append("\nper task (succ/n, failure stages):")
    by_t = collections.defaultdict(list)
    for e in eps:
        by_t[e["task"]].append(e)
    for t, v in sorted(by_t.items()):
        s = sum(e["success"] for e in v)
        st = collections.Counter(e["stage"] for e in v if not e["success"])
        lines.append(f"  {t:15s} {v[0]['instruction'][:34]:34s} {s}/{len(v)}  " +
                     ", ".join(f"{k} {n}" for k, n in st.most_common()))
    (OUT / "final_failures.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
