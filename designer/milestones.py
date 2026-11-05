"""The Designer, before the failure point search: the milestones of a task (designer/terms.py "Milestone"), from
the log of the task's first episode and the World Knowledge, with its own knowledge of the domain and no recipe
graph. The loop asks once per task and again whenever World Knowledge grows; code cuts every log at them.
"""
from __future__ import annotations

from llm_json import ask_json
from terms import THOUGHT_NOTE, TURN_ROW_LAYOUT, KNOWLEDGE_LINES, tagged, terms

SYSTEM = """You are the Designer. You decide the milestones of a {domain} task: the items, and how many of each, an agent
must hold on the way to it.

""" + terms("Episode", "Milestone", "Turn row", "Rollout", "World Knowledge") + """

Input:
- <task>: the task, one line.
- <log>: the log of the first episode of this task, one turn row per line:
""" + TURN_ROW_LAYOUT + "\n" + THOUGHT_NOTE + """
- <knowledge>: the World Knowledge, one entry per block (may be empty):
""" + KNOWLEDGE_LINES + """
  Supported: it holds in this world. Refuted: it does not hold (in its rollouts, the agent did the action
  while the situation held, and the result never followed). Where it differs from what you would expect,
  it is right: this world may not work the usual way.

An item is a milestone if the task's item is made from it, directly or through other items, or if it is
a tool needed to get one of those. Decide from your own knowledge of {domain}, corrected by what this
world showed:
- where the game did something other than the agent expected, what it did shows how this world works;
- a Supported entry of <knowledge> can show a step the usual way does not have, a Refuted one that a
  usual step does not work here.
An item still counts if the agent never got it or could not use it yet. When you are not sure, put it
under unsure.

Output:
- milestones: list, in the order the task needs them; each entry:
  - item: string, the item id as the log writes it ("log" or "planks" for any kind of log or planks)
  - count: integer, at least 1, how many the step that uses the item needs at once
  - why: string, what the task needs it for
  - evidence: string, the turns or the <knowledge> entry it rests on, or "usual way" when nothing in
    the input shows it
- unsure: list of strings, the item ids you are not sure about, or [] when there is none

Answer with JSON only:
{{"milestones": [{{"item": "...", "count": <integer>, "why": "...", "evidence": "..."}}, ...], "unsure": ["...", ...]}}"""


def valid(out) -> bool:
    """The answer as Output specifies it."""
    ms = out.get("milestones") if isinstance(out, dict) else None
    return (isinstance(ms, list) and bool(ms) and isinstance(out.get("unsure", []), list)
            and all(isinstance(m, dict) and isinstance(m.get("item"), str) and m["item"].strip()
                    and isinstance(m.get("count"), int) and not isinstance(m["count"], bool) and m["count"] >= 1
                    and isinstance(m.get("why", ""), str) and isinstance(m.get("evidence", ""), str) for m in ms))


def decide(task: str, log_text: str, knowledge: str, events: list, vocab) -> list | None:
    """[{item, count, why, evidence}] in order, or None for an unparsable answer."""
    out = ask_json(SYSTEM.format(domain=vocab.name), tagged(task=task, log=log_text, knowledge=knowledge),
                   events, "milestones")
    if not valid(out):
        return None
    ms = [{"item": str(m["item"]).strip().lower(), "count": m["count"], "why": str(m.get("why", "")), "evidence": str(m.get("evidence", ""))}
          for m in out["milestones"] if isinstance(m, dict) and m.get("item")]
    return ms or None
