"""System 2, step 2: propose one hypothesis at a failure point: the change to make there, the result it predicts,
how to try it, and the statement about the world it tests (kept as World Knowledge if Supported).
"""
from __future__ import annotations

import json

from llm_json import ask_json

PROPOSE_SYSTEM = """You improve a Minecraft agent that failed a task. You get the failure point (the turn where a decision
caused the failure), the reasoning for why it caused the failure, the agent's own turns around it as logged
(command, its thought, how it ended, what changed), and the state then.
Propose one hypothesis to test there:
- change: what to do differently from that point; it must be something the agent can do from there.
- prediction: what will be observed in the game if the hypothesis is right. Write it so that the game log
  (heights, position changes, inventory changes, health, deaths, command results) can show it happened or not.
- plan: how the agent should try the change, in a few steps.
- statement: the belief about the world, or about what works in it, that this hypothesis tests. One sentence that
  could be used in other attempts at other tasks, not only at this point.
{extra}
Answer with JSON only:
{{"change": "...", "prediction": "...", "plan": "...", "statement": "..."}}"""

PROPOSE_EXTRA_REFUTED = """Hypotheses already tested at this failure point whose predictions were not observed. Do not repeat
them; use why they failed:
{refuted}"""

FIELDS = ("change", "prediction", "plan", "statement")


def propose(fp: dict, raw: list, state: dict, refuted: list, log: list) -> dict | None:
    """`fp`: {turn, decision, reason}; `raw`: the agent's own turns around it; `state`: task, step, height,
    inventory then; `refuted`: hypotheses already tried at this failure point."""
    extra = PROPOSE_EXTRA_REFUTED.format(refuted="\n".join(json.dumps(r, ensure_ascii=False) for r in refuted)) if refuted else ""
    user = json.dumps({"task": state.get("task"), "failure_point": fp, "state_then": state,
                       "turns_around_it": raw}, ensure_ascii=False)
    out = ask_json(PROPOSE_SYSTEM.format(extra=extra), user, log, "propose")
    if not out or not all(isinstance(out.get(k), str) and out[k].strip() for k in FIELDS):
        return None
    return {k: out[k].strip() for k in FIELDS}
