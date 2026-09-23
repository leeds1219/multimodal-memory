"""Planner-only dry run: Optimus3Agent.plan() on all 70 task instructions (no Minecraft).

For every prompt form, saves the raw plan, parsed subgoals/goals and sanity flags:
truncated (512-token cap hit), parse mismatch (#goals != #subgoals), empty plan,
final goal == target item, craft/smelt items without a recipe file (the GUI macro
would fail), steps the released loop routes to the action head that are not
chop/dig/mine.

  CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/optimus3/bin/python plan_dryrun.py --forms obtain,verbatim
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_episode as RE  # noqa: E402  (sets env vars)
from common import TARGETS, TASKS  # noqa: E402

RECIPES = RE.R / "MineStudio/minestudio/assets/recipes"


def route(sub: str) -> str:  # same substring routing as gui_server._step
    if "craft" in sub:
        return "craft"
    if "smelt" in sub:
        return "smelt"
    return "dig" if "dig down" in sub else "policy"


def sanity(uid: str, subgoals, goals) -> dict:
    tgt = TARGETS[uid]
    names = [t.lstrip("#") for t in tgt["any_of"]]
    last = goals[-1]["item"] if goals else None
    flags = {
        "empty": not subgoals,
        "mismatch": len(goals) != len(subgoals),
        "final_is_target": bool(last) and any(n in last or last in n for n in names),
        "no_recipe": [g["item"] for s, g in zip(subgoals, goals)
                      if route(s) in ("craft", "smelt") and not (RECIPES / (g["item"].replace(" ", "_") + ".json")).exists()],
        "policy_steps": [s for s in subgoals if route(s) == "policy"],
        "routes": dict(Counter(route(s) for s in subgoals)),
    }
    flags["odd_policy_steps"] = [s for s in flags["policy_steps"] if not any(v in s for v in ("chop", "mine", "dig", "kill", "collect"))]
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default="obtain,verbatim")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default="/home/rag/data/repro_runs/runs/C3/optimus3/dryrun_plans")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    t0 = time.time()
    agent = RE.load_agent("cuda")
    print(f"loaded in {time.time() - t0:.0f}s, gpu mem {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    uids = a.tasks.split(",") if a.tasks else list(TASKS)
    llm = RE.LLMLogger(agent, None)
    for form in a.forms.split(","):
        llm.path = out / f"llm_calls_{form}.jsonl"
        rows = []
        for uid in uids:
            torch.manual_seed(0)
            text = RE.planner_text(uid, form)
            llm.caller = "plan"
            raw, subgoals, goals = agent.plan(text)
            rec = llm.records[-1]
            row = {"task": uid, "instruction": TASKS[uid]["instruction"], "planner_input": text, "raw": raw,
                   "subgoals": subgoals, "goals": goals, "tokens_in": rec.get("tokens_in"),
                   "tokens_out": rec.get("tokens_out"), "truncated": rec.get("finish_reason") == "length",
                   "latency_s": rec["latency_s"], **sanity(uid, subgoals, goals)}
            rows.append(row)
            print(f"[{form}] {uid:12s} n={len(subgoals):2d} tok={row['tokens_out']:3d} {row['latency_s']:5.1f}s "
                  f"trunc={row['truncated']} final_ok={row['final_is_target']} mism={row['mismatch']} "
                  f"norecipe={row['no_recipe']} | {' / '.join(subgoals)[:200]}", flush=True)
        with open(out / f"plans_{form}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        summ = {"form": form, "n": n, "empty": sum(r["empty"] for r in rows),
                "truncated": sum(r["truncated"] for r in rows), "mismatch": sum(r["mismatch"] for r in rows),
                "final_is_target": sum(r["final_is_target"] for r in rows),
                "with_no_recipe_step": sum(bool(r["no_recipe"]) for r in rows),
                "with_odd_policy_step": sum(bool(r["odd_policy_steps"]) for r in rows),
                "max_tokens_out": max(r["tokens_out"] for r in rows),
                "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 1),
                "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        (out / f"summary_{form}.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
