"""The Designer, step 1: list every failure point of one timeline (designer/terms.py "Failure point"): the turns whose
turn rows show a mismatch, a missed response, ignored knowledge, or by the Designer's own judgment a better choice.
The loop tests them by turn, earliest first.
"""
from __future__ import annotations

from llm_json import ask_json
from terms import COMMANDS_INPUT, KNOWLEDGE_LINES, MILESTONES_INPUT, PLAN_INPUT, TIMELINE_LOG, tagged, terms

KINDS = ("mismatch", "missed response", "ignored knowledge", "judgment")

HEAD = """You are the Designer. You review one timeline of a failed attempt at a {domain} task, played by an agent that sends one
command per turn, and find its failure points.

""" + terms("Episode", "Attempt", "Timeline", "Milestone", "Turn row", "Segment", "Failure point",
            "Kinds of failure point", "Rollout", "World Knowledge") + """

Input:
- <task>: the task, one line.
""" + COMMANDS_INPUT + "\n" + MILESTONES_INPUT + "\n" + TIMELINE_LOG + """
- <knowledge>: the World Knowledge, one hypothesis per line (may be empty):
""" + KNOWLEDGE_LINES + """
  Supported: it holds. Refuted: it does not hold. The agent was given the Supported ones. Hypotheses tried
  in this attempt outside this timeline are here too."""

WORK = """

Work in this order:
1. Go through the timeline turn by turn. Compare each result with the thought before it, and with what
   <commands> says the command does.
2. List every failure point: every turn whose turn rows show one of the four kinds. Do not choose among
   them or leave one out because it looks small; each will be tested.
3. For each failure point, write what the agent believed or did, and what the log shows instead (for
   judgment: the better choice).
4. A turn a marker line says was tried already: list it only if turn rows after that marker show a
   different cause; cite those rows in evidence and say what is different in differs.
5. Order the list by turn, earliest first.

Answer with JSON only:
{{"failure_points": [{{"turn": <n>, "kind": "mismatch" | "missed response" | "ignored knowledge" | "judgment",
  "decision": "<what the agent decided, and what it believed then>",
  "shown": "<what the log shows instead; for judgment, the better choice>",
  "knowledge": "<the Supported item of <knowledge> it went against, or empty>",
  "evidence": "<the turn rows that show it>",
  "differs": "<for a turn tried already: what the rows after its marker show differently, or empty>"}}, ...]}}"""


def find_system(vocab, plan: bool) -> str:
    """The prompt; the plan input only when the agent planned (no placeholder otherwise)."""
    return HEAD.format(domain=vocab.name) + ("\n" + PLAN_INPUT if plan else "") + WORK.format()


def find(task: str, commands: str, milestones: str, log_text: str, knowledge: str, plan: str | None, events: list,
         vocab) -> list | None:
    """[{turn, kind, decision, shown, knowledge, evidence, differs}] by turn, or None for an unparsable answer."""
    out = ask_json(find_system(vocab, bool(plan)), tagged(task=task, commands=commands, milestones=milestones,
                                                          log=log_text, knowledge=knowledge, plan=plan),
                   events, "find_failure_point")
    if not out or not isinstance(out.get("failure_points"), list):
        return None
    found = {}
    for f in out["failure_points"]:
        if isinstance(f, dict) and isinstance(f.get("turn"), int) and f["turn"] not in found:
            found[f["turn"]] = {"turn": f["turn"], **{k: str(f.get(k, "")) for k in
                                                      ("kind", "decision", "shown", "knowledge", "evidence", "differs")}}
    return [found[t] for t in sorted(found)]
