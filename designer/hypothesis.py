"""The Designer, step 2: one hypothesis at one failure point (designer/terms.py "Hypothesis"): its situation, action
and result in the words of the log, the change the agent is given, and the prediction (when, what) the rollouts are
judged by. A retry at the same failure point gets that point's experiment record. The hypothesis sentence goes to
World Knowledge with its verdict (stored under "statement").
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
    knowledge: <the hypothesis of the Supported World Knowledge entry it went against, or empty>
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
    hypothesis: <in this situation, doing this action leads to this result>
    change: start <...>; do <...>; stop <...>
    prediction: when <...>; what <...>
    verdict: <Refuted | Not reached>
    observed: <what the log of each rollout showed>"""

WORK = """

Write, in this order:
1. hypothesis: its situation, action and result, filled by the kind of the failure point:
   - mismatch: the situation is the state at the failure point, the action its command, the result what
     the command did unexpectedly; the hypothesis says this is a rule, not chance;
   - missed response: the situation is the state after the unexpected outcome, the action the command
     the agent should have sent then, the result what that command should give;
   - ignored knowledge: the situation is the state at the failure point, the action what the knowledge
     says to do, the result what doing it in this situation gives;
   - judgment: the situation is the state at the failure point, the action the better choice, the result
     a milestone result: the first milestone in <milestones> that the agent's own choice delayed and the
     better choice reaches within the window, as "reach <item> <count>".
   Within the window, the agent must be able to meet the situation and do the action. A hypothesis whose
   test needs longer than the window is the wrong hypothesis: pick a situation and action that come
   sooner. It must not be in <knowledge> or <record>, unless an observation there shows what is
   different; then say what is different.
2. change: start, what to do, stop, using <commands>. It brings the agent to the situation and has it do
   the action. The agent is given the change, not the hypothesis or the prediction.
3. plan: the few steps of the change only; after it the agent goes on with the task on its own.
4. prediction: when and what. If <record> has a verdict Not reached, pick a situation and action the
   agent will meet sooner.

Output:
- hypothesis: string, one sentence: in this situation, doing this action leads to this result
- situation: string, the situation, in the words of the log
- action: string, the command or short sequence of commands, as <commands> writes them
- result: string, what the turn rows of the action record, or "reach <item> <count>" for judgment
- milestone: string, for judgment the milestone of the result as "<item> <count>", written as in
  <milestones>; "" for the other kinds
- change: object with start (string, a situation, not a turn number), do (string), stop (string)
- plan: string, the few steps of the change
- prediction: object with when (string) and what (string)
- differs_from: string, the <knowledge> entry or <record> experiment it revises and how, or "" when none

Answer with JSON only:
{{"hypothesis": "...", "situation": "...", "action": "...", "result": "...", "milestone": "...",
 "change": {{"start": "...", "do": "...", "stop": "..."}}, "plan": "...",
 "prediction": {{"when": "...", "what": "..."}}, "differs_from": "..."}}"""


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
            record: list, plan: str | None, events: list, vocab, items: list, attempts: int = 3) -> dict | None:
    """{hypothesis {situation, action, result}, statement (the hypothesis sentence), milestone ((item, count) of a
    milestone result, else None), change {start, do, stop}, plan, prediction {when, what}, differs_from}; `items`
    the milestones [(item, count)]. An answer not as Output specifies is asked again, up to `attempts` calls; None
    if none is."""
    user = tagged(task=task, commands=commands, milestones=milestones, failure_point=fp_lines(fp), restart=restart,
                  log=log_text, knowledge=knowledge, record=record_lines(record), plan=plan)
    for _ in range(attempts):
        out = ask_json(propose_system(vocab, bool(plan)), user, events, "propose")
        if valid(out, fp["kind"], items):
            return {"hypothesis": {k: out[k].strip() for k in ("situation", "action", "result")},
                    "statement": out["hypothesis"].strip(), "milestone": milestone_of(out["milestone"], items),
                    "change": {k: out["change"][k].strip() for k in ("start", "do", "stop")},
                    "plan": out["plan"], "prediction": {k: out["prediction"][k].strip() for k in ("when", "what")},
                    "differs_from": out.get("differs_from", "")}
    return None


def milestone_of(text, items: list):
    """(item, count) of the milestone written "<item> <count>", if it is one of `items`; else None."""
    parts = str(text).strip().lower().split()
    m = (parts[0], int(parts[1])) if len(parts) == 2 and parts[1].isdigit() else None
    return m if m in items else None


def valid(out, kind: str, items: list) -> bool:
    """The answer as Output specifies it: non-empty strings, change and prediction objects with all their fields,
    and a milestone of `items` exactly when the kind is judgment."""
    def strs(d, keys):
        return isinstance(d, dict) and all(isinstance(d.get(k), str) and d[k].strip() for k in keys)
    ms = out.get("milestone", "") if isinstance(out, dict) else None
    return (strs(out, ("hypothesis", "situation", "action", "result")) and isinstance(ms, str)
            and (milestone_of(ms, items) is not None if kind == "judgment" else not ms.strip())
            and isinstance(out.get("plan"), str)
            and isinstance(out.get("differs_from", ""), str) and strs(out.get("change"), ("start", "do", "stop"))
            and strs(out.get("prediction"), ("when", "what")))
