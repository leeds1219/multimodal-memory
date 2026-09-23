"""Canned LLM responses for the offline (no-API) checks.

Recognises each baseline's prompt family by stable substrings and returns a
response in the exact format that baseline's parser expects. Plans are
hand-written for the few easy wooden tasks used in the offline check.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace


def _text(kw) -> str:
    parts = []
    for m in kw["messages"]:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts += [p.get("text", "") for p in c if p.get("type") == "text"]
    return "\n".join(parts)


def _target(text: str, ctx: dict) -> str:
    instr = (ctx.get("instruction") or "").lower()
    if not instr:
        m = re.search(r"<task>:\s*([^\n]+)", text)
        instr = (m.group(1) if m else "").lower()
    for item in ("wooden pickaxe", "crafting table", "stick", "planks", "wooden axe"):
        if item in instr:
            return item
    return "log"


# Optimus-1 style steps: (task text, goal item, count)
_OPT_STEPS = {
    "log": [("chop trees", "logs", 1)],
    "planks": [("chop trees", "logs", 1), ("craft planks", "planks", 4)],
    "stick": [("chop trees", "logs", 1), ("craft planks", "planks", 4), ("craft sticks", "stick", 4)],
    "crafting table": [("chop trees", "logs", 1), ("craft planks", "planks", 4),
                       ("craft crafting table", "crafting_table", 1)],
    "wooden pickaxe": [("chop trees", "logs", 3), ("craft planks", "planks", 12), ("craft sticks", "stick", 4),
                       ("craft crafting table", "crafting_table", 1), ("craft wooden pickaxe", "wooden_pickaxe", 1)],
}
_OPT_STEPS["wooden axe"] = _OPT_STEPS["wooden pickaxe"][:-1] + [("craft wooden axe", "wooden_axe", 1)]


def _optimus(text: str, ctx: dict) -> str | None:
    if "<visual inference>" in text and "<goal inference>" in text and "craft graph" not in text:
        tgt = _target(text, ctx)
        return (f"<goal inference>: {tgt}\n<visual inference>\nhealth bar: full\nfood bar: full\n"
                f"hotbar: empty\nenvironment: forest")
    if "done, continue, or replan" in text:
        return "Environment: <Forest>\nSituation: <Continue>\nPredicament: <None>"
    steps = _OPT_STEPS[_target(text, ctx)]
    plan = {f"step {i + 1}": {"task": t, "goal": [g, n]} for i, (t, g, n) in enumerate(steps)}
    if "meets <error>" in text:
        return "<replan>: " + json.dumps(plan)
    if "<task planning>" in text or "make a plan" in text:
        return "<task planning>\n" + json.dumps(plan)
    return None


# MineEvolve style subgoals
_ME_STEPS = {
    "log": [("chop a tree", "mine", "stevei", "oak_log", 1)],
    "planks": [("chop a tree", "mine", "stevei", "oak_log", 1), ("craft planks", "craft", "mc_craft", "oak_planks", 4)],
    "crafting table": [("chop a tree", "mine", "stevei", "oak_log", 1),
                       ("craft planks", "craft", "mc_craft", "oak_planks", 4),
                       ("craft crafting table", "craft", "mc_craft", "crafting_table", 1)],
    "wooden pickaxe": [("chop a tree", "mine", "stevei", "oak_log", 3),
                       ("craft planks", "craft", "mc_craft", "oak_planks", 12),
                       ("craft sticks", "craft", "mc_craft", "stick", 4),
                       ("craft crafting table", "craft", "mc_craft", "crafting_table", 1),
                       ("craft wooden pickaxe", "craft", "mc_craft", "wooden_pickaxe", 1)],
}


def _mineevolve(text: str, ctx: dict) -> str | None:
    if "Inducer module of MineEvolve" in text or "Inducer" in text[:400]:
        return json.dumps({"type": "none"})
    if "subgoal_id" in text or "Adaptor" in text[:400] or "Planner" in text[:600]:
        steps = _ME_STEPS.get(_target(text, ctx), _ME_STEPS["log"])
        return json.dumps({
            "plan_id": "p_mock",
            "subgoals": [{"subgoal_id": f"sg_{i + 1:03d}", "condition": c, "task_kind": k, "executor_hint": h,
                          "mode": "move" if h == "stevei" else "stay", "timeout_s": 60,
                          "checks": [{"type": "inv_ge", "item": it, "n": n}], "rationale": "mock"}
                         for i, (c, k, h, it, n) in enumerate(steps)],
            "global_constraints": [],
        })
    return None


def mock_response(kw: dict, ctx: dict):
    text = _text(kw)
    content = _optimus(text, ctx) if ctx.get("method", "").startswith("optimus") else _mineevolve(text, ctx)
    if content is None:
        content = "MOCK: unrecognised prompt"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=None,
    )
