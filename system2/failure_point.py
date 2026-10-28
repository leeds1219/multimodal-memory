"""System 2, step 1: find the failure point, the turn whose decision caused the failure, in a compressed episode
log (system2/episode_log.py). System 2 never sees the game, only the logs and the knowledge verified earlier.
"""
from __future__ import annotations

import json

from llm_json import ask_json

FIND_SYSTEM = """You review a failed attempt at a Minecraft task, made by an agent that sends one command per turn.
You get a compressed log: one line per turn (or per run of turns where nothing changed).
Find the failure point: the turn where a decision was made that caused the failure. This is often not where the
failure shows. For example, if a later craft fails for lack of logs, the cause is the turn where the agent stopped
collecting logs too early.
The failure point must be a decision that could have been made differently at that turn.
An experiment will restart the game from the last [saved] turn at or before the failure point, and try a change there.
{extra}
Answer with JSON only:
{{"turn": <turn number>, "decision": "<what the agent decided at that turn>",
  "reason": "<why this decision caused the failure, with the turns and log lines that show it>"}}"""

FIND_EXTRA_NEXT = """Failure points found before in this attempt, each tried with a change whose prediction was observed.
The log you get is the attempt after those changes; find a different cause of the failure:
{prev}"""


def find(compressed: str, earlier: list, log: list) -> dict | None:
    """{turn, decision, reason} or None. `earlier`: failure points already closed in this loop."""
    extra = FIND_EXTRA_NEXT.format(prev="\n".join(json.dumps(e, ensure_ascii=False) for e in earlier)) if earlier else ""
    out = ask_json(FIND_SYSTEM.format(extra=extra), compressed, log, "find_failure_point")
    if not out or not isinstance(out.get("turn"), int):
        return None
    return {"turn": out["turn"], "decision": str(out.get("decision", "")), "reason": str(out.get("reason", ""))}
