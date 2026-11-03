"""System 2, step 1: find the failure points in a compressed episode log (the domain's compress(), e.g.
domains/minecraft/episode_log.py): the turns whose decision was a mistake from what the agent could know then, and
caused the failure. System 2 never sees the game, only the logs and the knowledge verified earlier. The domain's words (domains/base.py Vocabulary) fill the prompt.
"""
from __future__ import annotations

import json

from llm_json import ask_json

FIND_SYSTEM = """You review a failed attempt at a {domain} task, made by an agent that sends one command per turn.
You get a compressed log: a timeline, then one line per turn (or per run of turns where nothing changed), with the
start of the agent's own thought.
Find the failure points. A turn is a failure point only if both hold:
- It was a mistake then: from what the agent could know at that turn (the task, its inventory and state, its own
  thoughts and earlier turns), a better choice was clear. Judge it as the agent stood then, not by what happened
  later.
- It mattered: had the agent made that better choice at that turn, from the state it was in then, the task would
  likely have succeeded.
Judge each turn on its own, from the state the agent was in at that turn. A turn counts even when an earlier
failure point also caused the failure: list every turn that passes, not only the first.
A decision that was reasonable then and turned bad only because of a later mistake is not a failure point; that
later mistake is.
When a command had an effect it was not meant to have (for example, an inventory change that the command does
not explain), or something else the agent did not decide hurt the task (an item lost, used up or broken, damage
taken), the failure point is the agent's response to it: the first turn after it where the log showed it and
the agent could still undo or avoid its effect, but did not.
Work in this order:
1. Where did the steps go? Find the periods where the agent made no progress toward the task, and what blocked it
   in each (the log lines and the agent's thoughts show it).
2. What did the agent lack when it was blocked? For each such item, find in the inventory changes every turn
   where it, or what it is made from, was lost or used up, and what the agent did right after.
3. Collect candidates: the turns found in step 2, the turns that brought about each blocked period, and every turn
   inside a blocked period where the agent chose what to do about what it lacked.
4. Test each candidate on its own with both tests above. A turn inside a blocked period, or after an earlier
   failure point, is not only a consequence: if a better choice was clear then and would likely have led to
   success from there, it is a failure point.
5. List every candidate that passes, earliest first.
An experiment will restart the game from the last save before a failure point (turns marked [saved]); the turns
between that save and the failure point are played again, and need not go the same way.
{extra}
Answer with JSON only:
{{"blocked": "<the periods without progress and what blocked each, with turns>",
  "failure_points": [{{"turn": <turn number>, "decision": "<what the agent decided at that turn>",
    "known_then": "<what the agent could see or knew at that turn that made a better choice clear>",
    "better": "<the better choice it could have made then>",
    "reason": "<how this decision led to the failure, with the turns and log lines that show it>"}}, ...]}}"""

FIND_EXTRA_NEXT = """Failure points found before in this attempt, each tried with a change whose prediction was observed.
The log you get is the attempt after those changes; find the other causes of the failure:
{prev}"""


def find_system(vocab, extra: str = "") -> str:
    return FIND_SYSTEM.format(domain=vocab.name, extra=extra)


def find(compressed: str, earlier: list, log: list, vocab) -> list | None:
    """[{turn, decision, reason}], earliest turn first, or None for an unparsable answer. `earlier`: failure points
    whose change was supported earlier in this loop."""
    extra = FIND_EXTRA_NEXT.format(prev="\n".join(json.dumps(e, ensure_ascii=False) for e in earlier)) if earlier else ""
    out = ask_json(find_system(vocab, extra), compressed, log, "find_failure_point")
    if not out or not isinstance(out.get("failure_points"), list):
        return None
    found = {}
    for f in out["failure_points"]:
        if isinstance(f, dict) and isinstance(f.get("turn"), int) and f["turn"] not in found:
            found[f["turn"]] = {"turn": f["turn"], "decision": str(f.get("decision", "")),
                                "known_then": str(f.get("known_then", "")), "better": str(f.get("better", "")),
                                "reason": str(f.get("reason", ""))}
    return [found[t] for t in sorted(found)]
