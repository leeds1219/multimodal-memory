"""Classify why each wooden-tier episode failed (no API; reads a run dir).

    python scripts/analyze_failures.py logs/eval/2026-09-19/12-32-24 [more run dirs]

Per episode it reads evidence/task_<id>/run_<n>/*/summary.json in order and derives:
  first_log_step   env step at which any *_log first appeared in the inventory
  max_*            peak counts of logs / planks / sticks / crafting_table seen
  craft_fail       mc_craft/mc_smelt subgoals that failed although the inventory
                   (at that moment) held the recipe's ingredients -> executor bug
  planner_notes    plan-level mistakes we can detect mechanically (plank arithmetic,
                   manual table placing / equipping, chop re-issued with log in hand)
and labels the episode:
  success | no_log (gatherer never got a log) | out_of_time (materials were coming,
  horizon hit) | executor (a helper failed with ingredients present) | planner
  (materials sufficient at some point but the plan never crafted the goal)
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

TOOL_NEED = {  # planks for the head (+2 sticks, +4 planks for the table)
    "wooden_pickaxe": 3, "wooden_axe": 3, "wooden_hoe": 2, "wooden_sword": 2, "wooden_shovel": 1,
}


def inv_count(inv, pred):
    return sum(int(v) for k, v in (inv or {}).items() if pred(k))


def analyze_episode(task, run_dir):
    subs = []
    for f in sorted(glob.glob(os.path.join(run_dir, "[0-9][0-9]_*/summary.json"))):
        s = json.load(open(f))
        subs.append(s)
    if not subs:
        return None
    goal = task.lower()
    steps = 0; first_log = None
    peak = collections.Counter(); craft_fail = []; notes = set(); chop_with_log = 0
    n_move = n_stevei = n_craft = 0
    for s in subs:
        steps += int(s.get("steps") or 0)
        inv = s.get("end_state", {}).get("inventory") or {}
        logs = inv_count(inv, lambda k: k.endswith("_log"))
        if logs and first_log is None:
            first_log = steps
        for key, pred in (("logs", lambda k: k.endswith("_log")), ("planks", lambda k: k.endswith("_planks")),
                          ("stick", lambda k: k == "stick"), ("table", lambda k: k == "crafting_table")):
            peak[key] = max(peak[key], inv_count(inv, pred))
        cond = s.get("condition", "").lower()
        tl = s.get("event_timeline") or []
        helper = any(e.get("event") == "helper_executed" for e in tl)
        if helper:
            n_craft += 1
            if not s["success"]:
                # did the inventory (before this subgoal) hold what the target needed?
                start_inv = s.get("start_state", {}).get("inventory") or {}
                tgt = (s.get("target_item") or "").lower()
                need_planks = TOOL_NEED.get(tgt)
                if tgt == "oak_planks" and inv_count(start_inv, lambda k: k.endswith("_log")) >= 1:
                    craft_fail.append(tgt)
                elif tgt == "stick" and inv_count(start_inv, lambda k: k.endswith("_planks")) >= 2:
                    craft_fail.append(tgt)
                elif tgt == "crafting_table" and inv_count(start_inv, lambda k: k.endswith("_planks")) >= 4:
                    craft_fail.append(tgt)
                elif need_planks and inv_count(start_inv, lambda k: k.endswith("_planks")) >= need_planks and start_inv.get("stick", 0) >= 2 and start_inv.get("crafting_table", 0) >= 1:
                    craft_fail.append(tgt)
                elif need_planks:
                    notes.add("tool craft attempted without enough materials")
        elif "move" in cond[:6] or "turn" in cond[:6] or "rotate" in cond[:8] or "jump" in cond[:5]:
            n_move += 1
        else:
            n_stevei += 1
            if ("chop" in cond or "log" in cond) and logs >= 1 and not s["success"]:
                chop_with_log += 1
        if re.search(r"place (the )?crafting.table|equip (the )?crafting.table", cond):
            notes.add("planner tried to place/equip the crafting table by hand")
    if chop_with_log:
        notes.add(f"chop re-issued {chop_with_log}x with a log already in hand")
    success = any(s["success"] and (s.get("target_item") or "") in goal.replace(" ", "_") for s in subs) or False
    return dict(steps=steps, first_log=first_log, peak=dict(peak), craft_fail=craft_fail, notes=sorted(notes),
                n_sub=len(subs), n_move=n_move, n_stevei=n_stevei, n_craft=n_craft)


def label(r, a, task):
    if r["success"]:
        return "success"
    if a is None:
        return "no_evidence"
    if a["craft_fail"]:
        return "executor"
    if a["peak"].get("logs", 0) == 0:
        return "no_log"
    tool = next((t for t in TOOL_NEED if t.replace("wooden_", "wooden ") in task.lower()), None)
    if tool:
        need = TOOL_NEED[tool] + 2 + 4
        if a["peak"].get("logs", 0) * 4 + a["peak"].get("planks", 0) >= need and a["peak"].get("table", 0) >= 1 and a["peak"].get("stick", 0) >= 2:
            return "planner"      # had everything at some point, never crafted the tool
        return "out_of_time"
    if "sapling" in task.lower():
        return "no_drop"
    return "out_of_time"


def main():
    dirs = sys.argv[1:]
    for d in dirs:
        runs = [json.loads(l) for l in open(os.path.join(d, "runs.jsonl")) if l.strip()]
        calls = collections.Counter()
        cj = os.path.join(d, "llm_calls.jsonl")
        n_calls = sum(1 for l in open(cj) if l.strip()) if os.path.exists(cj) else 0
        labels = collections.Counter(); per_task = collections.defaultdict(list); first_logs = []; notes_all = collections.Counter()
        for r in runs:
            a = analyze_episode(r["task"], os.path.join(d, "evidence", f"task_{r['task_id']}", f"run_{r['run']}"))
            lab = label(r, a, r["task"]); labels[lab] += 1; per_task[r["task_id"]].append(lab)
            if a:
                if a["first_log"]: first_logs.append(a["first_log"])
                for n in a["notes"]: notes_all[n] += 1
                for cf in a["craft_fail"]: notes_all[f"executor craft fail: {cf}"] += 1
        print(f"=== {d}: {sum(r['success'] for r in runs)}/{len(runs)} success, {n_calls} LLM calls ({n_calls/len(runs):.1f}/episode)")
        for k, v in labels.most_common(): print(f"   {k:12} {v}")
        if first_logs:
            fl = sorted(first_logs); print(f"   first log: median {fl[len(fl)//2]} steps, obtained in {len(first_logs)}/{len(runs)} episodes")
        for n, v in notes_all.most_common(8): print(f"   note: {n} x{v}")
        print("   per task:", {t: "".join(l[0].upper() for l in ls) for t, ls in sorted(per_task.items())}, "(S=success N=no_log O=out_of_time P=planner E=executor D=no_drop)")


if __name__ == "__main__":
    main()
