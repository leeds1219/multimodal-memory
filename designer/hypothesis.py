"""The Designer, step 2: one hypothesis at one failure point (designer/terms.py "Hypothesis"): S, A and R in the
words of the log, the change the agent is given, and the prediction (when, what) the rollouts are judged by. A retry
at the same failure point gets that point's experiment record. The statement goes to World Knowledge with its
verdict.
"""
from __future__ import annotations

from llm_json import ask_json
from terms import COMMANDS_INPUT, KNOWLEDGE_LINES, MILESTONES_INPUT, PLAN_INPUT, tagged, terms

HEAD = """You are the Designer. You propose one hypothesis at one failure point of a failed attempt at a {domain}
task, and the experiment that tests it.

""" + terms("Episode", "Attempt", "Timeline", "Milestone", "Turn row", "Failure point", "Kinds of failure point",
            "World Knowledge", "Restart point", "Rollout", "Window", "Hypothesis", "Change", "Prediction",
            "Experiment record") + """

Input:
- <task>: the task, one line.
""" + COMMANDS_INPUT + "\n" + MILESTONES_INPUT + """
- <failure_point>: the failure point to work on, one field per line:
    turn: T<turn>
    kind: <mismatch | missed response | ignored knowledge | judgment>
    decision: <what the agent decided, and what it believed then>
    shown: <what the log shows instead; for judgment, the better choice>
    knowledge: <the Supported item of World Knowledge it went against, or empty>
    evidence: <the turn rows that show it>
- <restart>: the restart point, one line:
    step <step> (after T<turn>), window <window steps> steps, y <height>, inventory: <item> <n>, <item> <n>, ...
- <log>: the timeline the failure point is in, in the same layout as the failure point search: segment
  headings, turn rows and markers for episodes replayed from a save. Two more markers show where the
  restart point and the failure point are:
    -- Restart point: the experiment starts here --
    -- Failure point: T<turn> --
- <knowledge>: the World Knowledge, one entry per block (may be empty):
""" + KNOWLEDGE_LINES + """
- <record>: the experiment record of this failure point, one experiment per block (may be empty):
    hypothesis: <in S, doing A leads to R>
    change: start <...>; do <...>; stop <...>
    prediction: when <...>; what <...>
    verdict: <Refuted | Not reached>
    observed: <what the log of each rollout showed>"""

WORK = """

Write, in this order:
1. hypothesis: S, A and R, filled by the kind of the failure point:
   - mismatch: S is the state at the failure point, A its command, R the unexpected result that came;
     the hypothesis says this result is a rule, not chance;
   - missed response: S is the state after the unexpected result, A the command the agent should have
     sent then, R the result that command should give;
   - ignored knowledge: S is the state at the failure point, A what the knowledge says to do, R what
     doing it in this situation gives;
   - judgment: S is the state at the failure point, A the better choice, R the result it should give.
   Within the window, the agent must be able to meet S and do A. A hypothesis whose test needs longer
   than the window is the wrong hypothesis: pick an S and A that come sooner. It must not
   be in <knowledge> or <record>, unless an observation there shows what is different; then say what is
   different.
2. change: start, what to do, stop, using <commands>. It brings the agent to S and has it do A. The agent
   is given the change, not the hypothesis or the prediction.
3. plan: the few steps of the change only; after it the agent goes on with the task on its own.
4. prediction: when and what. If <record> has a verdict Not reached, pick an S and A the agent will meet
   sooner.

Answer with JSON only:
{{"hypothesis": {{"situation": "...", "action": "...", "result": "..."}},
 "statement": "<one sentence: in S, doing A leads to R>",
 "change": {{"start": "...", "do": "...", "stop": "..."}}, "plan": "...",
 "prediction": {{"when": "...", "what": "..."}},
 "differs_from": "<the item of <knowledge> or <record> it revises and how, or empty>"}}"""


def propose_system(vocab, plan: bool) -> str:
    return HEAD.format(domain=vocab.name) + ("\n" + PLAN_INPUT if plan else "") + WORK.format()


def fp_lines(fp: dict) -> str:
    return "\n".join([f"turn: T{fp['turn']}"] + [f"{k}: {fp.get(k, '')}" for k in
                                                 ("kind", "decision", "shown", "knowledge", "evidence")])


def change_text(c) -> str:
    """The change as one line: start, do, stop."""
    return f"start {c.get('start', '')}; do {c.get('do', '')}; stop {c.get('stop', '')}" if isinstance(c, dict) else str(c)


def record_lines(record: list) -> str:
    return "\n\n".join("\n".join([f"hypothesis: {e['statement']}", f"change: {change_text(e['change'])}",
                                  f"prediction: when {e['prediction']['when']}; what {e['prediction']['what']}",
                                  f"verdict: {e['verdict']}", f"observed: {e['observed']}"]) for e in record)


def propose(task: str, commands: str, milestones: str, fp: dict, restart: str, log_text: str, knowledge: str,
            record: list, plan: str | None, events: list, vocab, attempts: int = 3) -> dict | None:
    """{hypothesis {situation, action, result}, statement, change {start, do, stop}, plan, prediction {when, what},
    differs_from}; an unparsable answer is asked again, up to `attempts` calls; None if none parses."""
    user = tagged(task=task, commands=commands, milestones=milestones, failure_point=fp_lines(fp), restart=restart,
                  log=log_text, knowledge=knowledge, record=record_lines(record), plan=plan)
    for _ in range(attempts):
        out = ask_json(propose_system(vocab, bool(plan)), user, events, "propose")
        try:
            h, c, p = out["hypothesis"], out["change"], out["prediction"]
            hyp = {"hypothesis": {k: str(h[k]) for k in ("situation", "action", "result")},
                   "statement": str(out["statement"]), "change": {k: str(c[k]) for k in ("start", "do", "stop")},
                   "plan": str(out.get("plan", "")), "prediction": {k: str(p[k]) for k in ("when", "what")},
                   "differs_from": str(out.get("differs_from", ""))}
        except (TypeError, KeyError):
            continue
        if hyp["statement"].strip() and all(hyp["hypothesis"].values()):
            return hyp
    return None
