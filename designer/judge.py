"""The Designer, step 3: in the log of one rollout, the occurrence (designer/terms.py "Occurrence"): the first turn
where the agent did the action while the situation held, and whether the result followed. It gets no other hypotheses, no World Knowledge and no
milestones, so the mark depends on this rollout alone; the loop counts the marks into the verdict and frequency
(verdict()).
"""
from __future__ import annotations

from llm_json import ask_json
from terms import TURN_ROW_LAYOUT, tagged, terms

SYSTEM = """You are the Designer. You read the log of one rollout of an experiment and mark the first time its
hypothesis was put to the test, and what happened then.

""" + terms("Turn row", "Restart point", "Rollout", "Window", "Hypothesis", "Prediction", "Occurrence") + """

Input:
- <commands>: the commands the agent can send, one per entry:
    <command form>   <what it is described to do>
- <hypothesis>: the hypothesis and its parts:
    hypothesis: <in this situation, doing this action leads to this result>
      situation: <...>
      action: <...>
      result: <...>
- <prediction>: when <...>; what <...>
- <log>: this rollout from the restart point to the end of the window, one turn row per line:
""" + TURN_ROW_LAYOUT + """

Work in this order:
1. Go through <log> from the start. Find the occurrence: the first turn where the turn rows before it
   show the situation and the command sent is the action.
2. Say whether the result followed, as how it ended and the changes record it. Read the result by the
   words of <hypothesis> and <prediction>, no stricter and no looser.
3. If no turn shows the situation and the action, say so. Do not give a verdict; whether the task
   succeeded does not matter.

Output:
- occurrence: object, or null when no turn shows the situation and the action:
  - turn: integer, the turn number without "T"
  - situation_shown_by: string, the turn rows that show the situation
  - result_followed: true or false
  - evidence: string, the turn row of the result

Answer with JSON only:
{"occurrence": {"turn": <integer>, "situation_shown_by": "...", "result_followed": <true | false>,
  "evidence": "..."} | null}"""


def occurrence(commands: str, hyp: dict, log_text: str, events: list, attempts: int = 2) -> dict | None:
    """{turn, situation_shown_by, result_followed, evidence}, or None when no turn showed the situation and the
    action; an answer not as Output specifies is asked again, up to `attempts` calls, and counts as none after
    that (events say which)."""
    h, p = hyp["hypothesis"], hyp["prediction"]
    user = tagged(commands=commands, hypothesis=f"hypothesis: {hyp['statement']}\n  situation: {h['situation']}\n"
                  f"  action: {h['action']}\n  result: {h['result']}", prediction=f"when {p['when']}; what {p['what']}",
                  log=log_text)
    for _ in range(attempts):
        out = ask_json(SYSTEM, user, events, "occurrence")
        if valid(out):
            occ = out["occurrence"]
            return None if occ is None else {k: occ[k] for k in ("turn", "situation_shown_by", "result_followed",
                                                                 "evidence")}
    return None


def valid(out) -> bool:
    """The answer as Output specifies it: occurrence null, or an object with an integer turn, a boolean
    result_followed and string fields."""
    if not isinstance(out, dict) or "occurrence" not in out:
        return False
    o = out["occurrence"]
    return o is None or (isinstance(o, dict) and isinstance(o.get("turn"), int) and not isinstance(o["turn"], bool)
                         and isinstance(o.get("result_followed"), bool)
                         and isinstance(o.get("situation_shown_by"), str) and isinstance(o.get("evidence"), str))


def verdict(occurrences: list) -> tuple[str, int, int]:
    """(verdict, k, n) from the rollouts' occurrences (None = none came): Supported when the result followed in at
    least one, Refuted when occurrences came and the result never followed, Not reached when none came; the result
    followed in k of the n rollouts with an occurrence."""
    came = [o for o in occurrences if o]
    k = sum(o["result_followed"] for o in came)
    return ("Supported" if k else "Refuted" if came else "Not reached"), k, len(came)
