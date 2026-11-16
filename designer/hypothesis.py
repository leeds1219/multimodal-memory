"""The Designer, step 2: one hypothesis at one failure point (designer/terms.py "Hypothesis"): the situation and action
under which the agent gains the failure point's item, in the words of the log, the change the agent is given, and the
prediction (when, what) the rollouts are judged by. A retry at the same failure point gets that point's experiment
record; the gains of the item in this timeline and in earlier episodes come as <gains>. The hypothesis sentence goes to
World Knowledge with its verdict (stored under "statement").
"""
from __future__ import annotations

from llm_json import ask_json
from terms import COMMANDS_INPUT, KNOWLEDGE_LINES, MILESTONES_INPUT, PLAN_INPUT, tagged, terms

HEAD = """You are the Designer. You propose one hypothesis at one failure point of a failed attempt at a {domain}
task, and the experiment that tests it.

""" + terms("Episode", "Attempt", "Timeline", "Milestone", "Turn row", "Failure point", "Gain", "World Knowledge", "Restart point", "Rollout", "Window", "Hypothesis", "Change", "Prediction",
            "Experiment record") + """

Input:
- <task>: the task, one line.
""" + COMMANDS_INPUT + "\n" + MILESTONES_INPUT + """
- <failure_point>: the failure point to work on, one field per line:
    milestone: <item> <count>
    held: <how many the agent held at the timeline end>
    segment: <n>, T<first>-T<last>
- <restart>: the restart point, one line:
    step <step> (after T<turn>), window <window steps> steps, y <height>, inventory: <item> <n>, <item> <n>, ...
- <log>: the timeline the failure point is in: segment headings, turn rows and markers for episodes
  replayed from a save. One more marker shows where the restart point is:
    -- Restart point: the experiment starts here --
- <knowledge>: the World Knowledge, one entry per block (may be empty):
""" + KNOWLEDGE_LINES + """
- <record>: the experiment record of this failure point, one experiment per block (may be empty):
    hypothesis: <in this situation, doing this action leads to this result>
    change: start <...>; do <...>; stop <...>
    prediction: when <...>; what <...>
    verdict: <Refuted | Not reached>
    observed: <what the log of each rollout showed>
- <gains>: gains of the failure point's item, in this timeline and in earlier episodes, newest first,
  each with the turn row before it (may be empty):
    == <this timeline | earlier episode of <task>> ==
    <turn row before the gain>
    <turn row of the gain>"""

WORK = """

Write, in this order:
1. hypothesis: the situation and action under which the agent gains the failure point's item, and
   the result "gain <item>". Read <gains> first: what held at the gains (the height, what was held, the
   commands before them), and what did not hold in the turns of <log> after the last gain, where the
   agent went on without one. When <gains> is empty, use <log> and your own knowledge.
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
- situation: string, only the conditions the result depends on, in the words of the log
- action: string, the command or short sequence of commands, each as its type and main argument
- result: string, "gain <item>" with the failure point's item
- change: object with start (string, a situation, not a turn number), do (string), stop (string)
- plan: string, the few steps of the change
- prediction: object with when (string) and what (string)
- differs_from: string, the <knowledge> entry, <record> experiment or <gains> rows it draws on or revises
  and how, or "" when none

Answer with JSON only:
{{"hypothesis": "...", "situation": "...", "action": "...", "result": "...",
 "change": {{"start": "...", "do": "...", "stop": "..."}}, "plan": "...",
 "prediction": {{"when": "...", "what": "..."}}, "differs_from": "..."}}"""


def propose_system(vocab, plan: bool) -> str:
    return HEAD.format(domain=vocab.name) + ("\n" + PLAN_INPUT if plan else "") + WORK.format()


def fp_lines(fp: dict) -> str:
    n, first, last = fp["segment"]
    return "\n".join([f"milestone: {fp['milestone'][0]} {fp['milestone'][1]}", f"held: {fp['held']}",
                      f"segment: {n}, " + (f"T{first}-T{last}" if first is not None else "no turns yet")])


def change_text(c) -> str:
    """The change as one line: start, do, stop."""
    return f"start {c.get('start', '')}; do {c.get('do', '')}; stop {c.get('stop', '')}" if isinstance(c, dict) else str(c)


def record_lines(record: list) -> str:
    return "\n\n".join("\n".join([f"hypothesis: {e['statement']}", f"change: {change_text(e['change'])}",
                                  f"prediction: when {e['prediction']['when']}; what {e['prediction']['what']}",
                                  f"verdict: {e['verdict']}", f"observed: {e['observed']}"]) for e in record)


def propose(task: str, commands: str, milestones: str, fp: dict, restart: str, log_text: str, knowledge: str,
            record: list, plan: str | None, events: list, vocab, gains: str = "", attempts: int = 3) -> dict | None:
    """{hypothesis {situation, action, result}, statement (the hypothesis sentence), change {start, do, stop}, plan,
    prediction {when, what}, differs_from}; `gains` the <gains> rows. An answer not as Output specifies is asked again,
    up to `attempts` calls; None if none is."""
    user = tagged(task=task, commands=commands, milestones=milestones, failure_point=fp_lines(fp), restart=restart,
                  log=log_text, knowledge=knowledge, record=record_lines(record), gains=gains, plan=plan)
    for _ in range(attempts):
        out = ask_json(propose_system(vocab, bool(plan)), user, events, "propose")
        if valid(out):
            return {"hypothesis": {k: out[k].strip() for k in ("situation", "action", "result")},
                    "statement": out["hypothesis"].strip(),
                    "change": {k: out["change"][k].strip() for k in ("start", "do", "stop")},
                    "plan": out["plan"], "prediction": {k: out["prediction"][k].strip() for k in ("when", "what")},
                    "differs_from": out.get("differs_from", "")}
    return None


def valid(out) -> bool:
    """The answer as Output specifies it: non-empty strings, change and prediction objects with all their fields."""
    def strs(d, keys):
        return isinstance(d, dict) and all(isinstance(d.get(k), str) and d[k].strip() for k in keys)
    return (strs(out, ("hypothesis", "situation", "action", "result")) and isinstance(out.get("plan"), str)
            and isinstance(out.get("differs_from", ""), str) and strs(out.get("change"), ("start", "do", "stop"))
            and strs(out.get("prediction"), ("when", "what")))
