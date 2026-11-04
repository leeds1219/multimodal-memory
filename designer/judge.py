"""The Designer, step 3: in the log of one rollout, the occurrence (designer/terms.py "Occurrence"): the first turn
where the agent did A while S held, and whether R followed. It gets no other hypotheses, no World Knowledge and no
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
- <hypothesis>: one field per line:
    situation: <S>
    action: <A>
    result: <R>
- <prediction>: when <...>; what <...>
- <log>: this rollout from the restart point to the end of the window, one turn row per line:
""" + TURN_ROW_LAYOUT + """

Work in this order:
1. Go through <log> from the start. Find the occurrence: the first turn where the turn rows before it
   show S and the command sent is A.
2. Say whether R followed, as how it ended and the changes record it. Read R by the words of
   <hypothesis> and <prediction>, no stricter and no looser.
3. If no turn shows S and A, say so. Do not give a verdict; whether the task succeeded does not matter.

Answer with JSON only:
{"occurrence": {"turn": <n>, "situation_shown_by": "<the turn rows that show S>",
  "result_followed": true | false, "evidence": "<the turn row of the result>"} | null}"""


def occurrence(commands: str, hyp: dict, log_text: str, events: list) -> dict | None:
    """{turn, situation_shown_by, result_followed, evidence}, or None when no turn showed S and A or the answer was
    unparsable (events say which)."""
    h, p = hyp["hypothesis"], hyp["prediction"]
    user = tagged(commands=commands, hypothesis=f"situation: {h['situation']}\naction: {h['action']}\nresult: "
                  f"{h['result']}", prediction=f"when {p['when']}; what {p['what']}", log=log_text)
    out = ask_json(SYSTEM, user, events, "occurrence")
    occ = out.get("occurrence") if out else None
    if isinstance(occ, dict) and isinstance(occ.get("turn"), int):
        return {"turn": occ["turn"], "situation_shown_by": str(occ.get("situation_shown_by", "")),
                "result_followed": occ.get("result_followed") is True, "evidence": str(occ.get("evidence", ""))}
    return None


def verdict(occurrences: list) -> tuple[str, int, int]:
    """(verdict, k, n) from the rollouts' occurrences (None = none came): Supported when R followed in at least one,
    Refuted when occurrences came and R never followed, Not reached when none came; k of n rollouts with an
    occurrence had R."""
    came = [o for o in occurrences if o]
    k = sum(o["result_followed"] for o in came)
    return ("Supported" if k else "Refuted" if came else "Not reached"), k, len(came)
