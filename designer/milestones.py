"""The Designer, before the failure point search: the milestones of a task (designer/terms.py "Milestone"), from
the log of the task's first episode and the World Knowledge, with its own knowledge of the domain and no recipe
graph. The loop asks once per task and again whenever World Knowledge grows; code cuts every log at them.
"""
from __future__ import annotations

from llm_json import ask_json
from terms import THOUGHT_NOTE, TURN_ROW_LAYOUT, KNOWLEDGE_LINES, tagged, terms

SYSTEM = """You are the Designer. You decide the milestones of a {domain} task: the items an agent must get on the way to it.

""" + terms("Episode", "Milestone", "Turn row", "Rollout", "World Knowledge") + """

Input:
- <task>: the task, one line.
- <log>: the log of the first episode of this task, one turn row per line:
""" + TURN_ROW_LAYOUT + "\n" + THOUGHT_NOTE + """
- <knowledge>: the World Knowledge, one entry per block (may be empty):
""" + KNOWLEDGE_LINES + """
  Supported: it holds in this world. Refuted: it does not hold (in its rollouts, the agent did A while S
  held, and R never followed). Where it differs from what you would expect, it is right: this
  world may not work the usual way.

An item is a milestone if the task's item is made from it, directly or through other items, or if it is
a tool needed to get one of those. Decide from your own knowledge of {domain}, corrected by what this
world showed:
- where the game's result differs from the agent's expectation, the result shows how this world works;
- a Supported item of <knowledge> can show a step the usual way does not have, a Refuted one that a usual
  step does not work here.
An item still counts if the agent never got it or could not use it yet. When you are not sure, put it
under unsure.

Answer with JSON only:
{{"milestones": [{{"item": "...", "why": "<what the task needs it for>", "evidence": "<the turns or the item of
  <knowledge> it rests on, or 'usual way' if nothing in the input shows it>"}}, ...], "unsure": ["..."]}}"""


def decide(task: str, log_text: str, knowledge: str, events: list, vocab) -> list | None:
    """[{item, why, evidence}] in order, or None for an unparsable answer."""
    out = ask_json(SYSTEM.format(domain=vocab.name), tagged(task=task, log=log_text, knowledge=knowledge),
                   events, "milestones")
    if not out or not isinstance(out.get("milestones"), list):
        return None
    ms = [{"item": str(m["item"]).strip().lower(), "why": str(m.get("why", "")), "evidence": str(m.get("evidence", ""))}
          for m in out["milestones"] if isinstance(m, dict) and m.get("item")]
    return ms or None
