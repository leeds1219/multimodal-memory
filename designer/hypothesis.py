"""The Designer, step 2: propose one hypothesis at a failure point: the problem the decision there caused later, the
belief about how to avoid it, the change that tests the belief there, the moment
the belief is checked at (the next time the agent needs what the problem was about) and what it predicts then, and
how to make the change. The statement kept as World Knowledge if Supported is the problem and the belief, said
in general; the evidence (turns of this attempt) is kept apart.
The failure point is chosen by the loop (earliest first); the other failure points of the same log are context.
A retry at the same failure point gets the hypotheses tested there, with what their experiments showed, and
proposes a different one (or the same change with a plan the agent can follow).
"""
from __future__ import annotations

import json

from llm_json import ask_json

INPUT = """You improve a {domain} agent that failed a task. You get:
- the failure point: the turn where a decision caused the failure, what the agent could know then, the better
  choice it could have made, and why the decision caused the failure;
- the other failure points found in the same failed attempt, with their turns;
- the agent's own turns from the restart point to just after the failure point, as logged (command, its thought,
  how it ended, what changed);
- the restart point: the last save before the failure point, and the agent's state there.
An experiment will restart the game at the restart point. The turns between it and the failure point are played
again and need not go the same way: something that happened then (for example, an item lost) may not happen again,
or may happen differently."""

PROPOSE_SYSTEM = INPUT + """
Propose a hypothesis and the change that tests it. Work in this order:
1. blocked: the earliest later moment of the failed attempt where the agent was blocked because of the decision at
   the failure point: its turn, and what blocked it then, as the log shows.
2. evidence: the turns and log lines that show how the decision at the failure point led to the block.
3. problem: what goes wrong, said in general: what such a decision causes later (for example, what the agent
   then lacks and why). No turns, no details only this attempt had: it must make sense to an agent in another
   attempt at another task.
4. hypothesis: the belief about how to avoid the problem, as the agent should act on it at such a moment.
5. change: what the agent should do from the restart point on to test the hypothesis. If it answers something that
   happened between the restart point and the failure point, write it as what to do when that happens (when ...,
   do ...), since it may not happen the same way again. It must be something the agent can do from there.
6. check_when: the moment the hypothesis is checked at: the next time the agent needs what the problem was about
   (for example, the next time it tries the action it was blocked at). Describe it by what the agent does or has
   then, not by turn numbers.
7. prediction: what the game log will show at that moment if the hypothesis is right.
8. plan: how the agent should make the change, in a few steps. Only the steps of the change itself: after it, the
   agent goes on with the task on its own.
check_when and prediction: write them so that the game log ({log_shows}) can show them. The experiment has its own turn numbers, so name turns only in
evidence.
{extra}
Answer with JSON only:
{{"blocked": {{"turn": <turn number>, "what": "..."}}, "evidence": "...", "problem": "...", "hypothesis": "...", "change": "...",
  "check_when": "...", "prediction": "...", "plan": "..."}}"""

EXTRA_TRIED = """Hypotheses already tested from this restart point that were not supported, with what the experiment's log
showed. Propose a different hypothesis or a different change, or the same change with a plan the agent can
follow if the log shows it did not make it:
{tried}"""

EXTRA_FAILED = """Hypotheses tested in earlier experiments (at other failure points or in other episodes) that were not
supported, with what the experiment's log showed. Use them to avoid proposing what already did not work:
{failed}"""

FIELDS = ("evidence", "problem", "hypothesis", "change", "check_when", "prediction", "plan")
STATEMENT = "{problem} {hypothesis}"


def _user(fp, others, raw, state, **more) -> str:
    return json.dumps({"task": state.get("task"), "failure_point": fp, "other_failure_points": others,
                       "restart_point": state, "turns_from_restart": raw, **more}, ensure_ascii=False)


def propose_system(vocab, extra: str = "") -> str:
    return PROPOSE_SYSTEM.format(domain=vocab.name, log_shows=vocab.log_shows, extra=extra)


def propose(fp: dict, others: list, raw: list, state: dict, tried: list, log: list, vocab,
            attempts: int = 3, failed: list | None = None) -> dict | None:
    """`fp`: {turn, decision, known_then, better, reason}; `others`: the other failure points of the same log; `raw`: the agent's own
    turns from the restart point to just after fp; `state`: task, turn, step, height, inventory at the restart
    point; `tried`: earlier experiments at fp, each the hypothesis with its verdict, observed and reason;
    `vocab`: the domain's words (domains/base.py); `failed`: hypotheses not supported elsewhere (memory/failed_hypotheses.py).
    An unparsable answer is asked again, up to `attempts` calls; None if none parses."""
    extra = EXTRA_TRIED.format(tried="\n".join(json.dumps({k: t[k] for k in ("hypothesis", "change", "plan", "check_when",
                                                                         "prediction", "verdict", "observed", "reason")},
                                                          ensure_ascii=False) for t in tried)) if tried else ""
    if failed:
        extra += ("\n" if extra else "") + EXTRA_FAILED.format(failed="\n".join(
            json.dumps({k: f[k] for k in ("statement", "change", "verdict", "observed")}, ensure_ascii=False) for f in failed))
    for _ in range(attempts):
        out = ask_json(propose_system(vocab, extra), _user(fp, others, raw, state), log, "propose")
        if out and all(isinstance(out.get(k), str) and out[k].strip() for k in FIELDS):
            h = {k: out[k].strip() for k in FIELDS}
            return {**h, "statement": STATEMENT.format(**h), "blocked": out.get("blocked")}
    return None
