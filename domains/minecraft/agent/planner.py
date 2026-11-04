"""Initial planning (OPTIMUS_VLM_PLAN=1): before the first turn the agent's own model writes the task as an ordered
list of sub-goals, each one item and a count. The current sub-goal (the first one not reached yet) fills the
"Current sub-goal" line of the system prompt every turn; a sub-goal is reached once the inventory has held that
many of the item, or a later sub-goal is reached, and stays reached. The plan is made once and never revised. An
episode started from a save keeps the plan of the episode it was saved from (the save's side carries it), so an
experiment changes only its change.
"""
from __future__ import annotations

import json
import re

PLAN_SYSTEM = """You plan a Minecraft task before playing it. Overall task: {task}
Write the plan as an ordered list of sub-goals, from an empty inventory to the task. Each sub-goal is to have one
item: its Minecraft item id (e.g. oak_log, crafting_table, iron_ingot; for any kind of log or planks write "log" or
"planks") and how many you need to have at once.{extra}
Answer with JSON only: {{"plan": [{{"item": "<item id>", "count": <n>, "why": "<what it is for>"}}, ...]}}"""

PLAN_NOTE = "\n\nYour plan, made before the first turn (reached sub-goals marked [x]):\n{steps}"
FAMILIES = ("log", "planks", "wool")  # generic names: any <kind>_log counts as log


def make(client, task: str, extra: str = "", attempts: int = 2) -> list | None:
    """[{item, count, why}] or None if no answer parses."""
    for _ in range(attempts):
        resp = client.chat.completions.create(model="FROM_CONFIG", max_tokens=800, messages=[
            {"role": "user", "content": PLAN_SYSTEM.format(task=task, extra=extra)}])
        m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
        try:
            plan = json.loads(m.group(0))["plan"] if m else None
        except Exception:
            plan = None
        if isinstance(plan, list):
            plan = [{"item": str(s["item"]).strip().lower(), "count": max(1, int(s.get("count", 1))),
                     "why": str(s.get("why", ""))} for s in plan if isinstance(s, dict) and s.get("item")]
            if plan:
                return plan
    return None


def held(inv: dict, item: str) -> int:
    if item in inv:
        return inv[item]
    if item in FAMILIES:
        return sum(v for k, v in inv.items() if k.endswith("_" + item))
    return 0


def update(plan: list, reached: list, inv: dict) -> list:
    """Indices of the sub-goals reached so far, with those the inventory `inv` reaches now; reaching a sub-goal also
    reaches every one before it (planks used up for sticks were still made)."""
    r = set(reached) | {i for i, s in enumerate(plan) if held(inv, s["item"]) >= s["count"]}
    return list(range(max(r) + 1)) if r else []


def current(plan: list, reached: list) -> int | None:
    return next((i for i in range(len(plan)) if i not in reached), None)


def subgoal_text(plan: list, reached: list) -> str:
    i = current(plan, reached)
    if i is None:
        return "all sub-goals of your plan are reached; finish the task"
    s = plan[i]
    return f"have {s['count']} {s['item']} (sub-goal {i + 1} of {len(plan)}: {s['why']})"


def note(plan: list, reached: list) -> str:
    return PLAN_NOTE.format(steps="\n".join(f"  [{'x' if i in reached else ' '}] {i + 1}. {s['count']} {s['item']}"
                                            for i, s in enumerate(plan)))
