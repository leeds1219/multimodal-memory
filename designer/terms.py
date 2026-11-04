"""The terms the Designer's prompts use, each sentence written once (the "Prompt Definitions" doc, word for word).
Every prompt assembles its "Terms used below" from these, so a term means the same in every prompt. {rollouts} and
{window_steps} are filled from configs/hyperparams.yaml.
"""
from __future__ import annotations

from hyperparams import HP

TERMS = {  # each term as the prompts list it, line breaks included
    'Episode': """\
- Episode: one play of the game, from its start or from a save to its end.""",
    'Attempt': """\
- Attempt: everything done on one task: its first episode and every experiment episode replayed from it.""",
    'Timeline': """\
- Timeline: one path through an attempt from step 0: an episode up to a save, then the episode replayed
  from that save, and so on; each step appears once.""",
    'Milestone': """\
- Milestone: getting an item the task needs, or a tool for one, for the first time.""",
    'Turn row': """\
- Turn row: one command of the agent: its steps, height, command, how it ended (with the game's message
  when it failed), changes, and the agent's whole thought.""",
    'Segment': """\
- Segment: the turns from one milestone to the next; the last segment ends where the timeline ends.""",
    'Failure point': """\
- Failure point: a turn whose turn rows show one of the four kinds below. Every one is listed, and each
  is tested by an experiment.""",
    'Kinds of failure point': """\
- Kinds of failure point:
  - mismatch: what the command did contradicts what the agent's thought said it expected: the decision
    rested on a wrong belief about the world.
  - missed response: after a mismatch, the agent goes on as if the command had done what it expected.
  - ignored knowledge: the agent acts against a Supported entry of World Knowledge it was given.
  - judgment: none of the above; only the Designer's own knowledge says a better choice was available
    then.""",
    'Rollout': """\
- Rollout: one play from the restart point with the change. Each hypothesis gets {rollouts} rollouts from
  the same restart point, taken as independent trials.""",
    'World Knowledge': """\
- World Knowledge: what experiments in any attempt found out about this world, as summaries only: each
  hypothesis that was Supported (it holds) or Refuted (it does not hold), with its verdict, its frequency
  (the result followed in k of its n rollouts) and no log. Hypotheses Not reached are not in it. The
  agent is given the Supported ones.""",
    'Restart point': """\
- Restart point: the last save before the failure point; the experiment plays on from there, and
  everything after it is played again.""",
    'Window': """\
- Window: the first {window_steps} steps after the restart point, or fewer if the episode ends first. A
  rollout ends right after its occurrence, or at the end of the window if none comes; the one rollout
  that continues the timeline plays on to the end.""",
    'Hypothesis': """\
- Hypothesis: a rule of this world in three parts, each written in the words the log uses and true
  beyond this attempt.
  situation: a condition turn rows can show: what the inventory holds, what is held or placed, the
  height, how the last command ended. The agent must be able to bring the situation about with its
  commands within the window; a situation it can only wait for by chance does not count.
  action: a command, or a short sequence of commands, from the commands list.
  result: what the turn rows of the action record: how the command ends and what changes in that turn
  (an inventory change, a change of height, a death). Not a later goal that needs more actions after
  the action. The result may depend on chance.
  The rule says the result can follow when the agent does the action while the situation holds; how
  often is measured by the rollouts. No turn numbers, steps, positions or anything only this attempt
  had; a number only when the rule is about that number. If Supported, it becomes World Knowledge.""",
    'Change': """\
- Change: what the agent does to test the hypothesis, with its commands: when it starts (a situation,
  not a turn number), what it does, and when it stops. It brings the agent to the situation and has it
  do the action.""",
    'Prediction': """\
- Prediction: what the log will show if the hypothesis is right.
  When: in each rollout, the first time within the window that the agent does the action while the
  situation holds.
  What: whether the result follows; it must be the game's response, never only that the agent did the
  action.""",
    'Experiment record': """\
- Experiment record: every experiment at one failure point: its hypothesis, change, prediction, verdict
  (Supported, Refuted or Not reached), frequency, and what the log of each rollout showed.""",
    'Occurrence': """\
- Occurrence: the first time in a rollout that the agent does the action while the situation holds. It
  is marked with whether the result followed.""",
}

# The layouts of the inputs, shared by the prompts that get them.
TURN_ROW_LAYOUT = """\
    T<turn> | <first step>-<last step> | y <start>-><end> | <command> | <how it ended> | <changes> | thought: "<the whole thought>"
  <how it ended> is "done" or "failed: <the game's message>"; <changes> lists inventory changes as
  "inv <item> +<n>, <item> -<n>" and other changes (deaths, movement). Turns in a row where nothing
  changed are one line:
    T<first>-T<last> | <first step>-<last step> | y <start>-><end> | <n> turns of <command type>, e.g. <one command> | nothing changed"""
THOUGHT_NOTE = """\
  The thought says what the agent expected the command to do; how it ended and the changes say what the
  game did."""
COMMANDS_INPUT = """\
- <commands>: the commands the agent can send, one per entry:
    <command form>   <what it is described to do>
  This is what the agent knew about its commands; the game may not do exactly this."""
MILESTONES_INPUT = """\
- <milestones>: the milestones of the task, one per line, in order:
    <n>. <item>: <what it is for>"""
TIMELINE_LOG = """\
- <log>: one timeline of this attempt, grouped by segment, with three kinds of lines.
  A segment heading:
    == Segment <n>: T<first>-T<last>, steps <first>-<last> (<count> steps), ended with milestone <item> ==
    (the last segment says "ended at the timeline end" when its milestone never came)
  A turn row:
""" + TURN_ROW_LAYOUT + """
  A marker, where the timeline goes on in an episode replayed from a save:
    -- Step <step>: replayed from a save. Failure point tried: T<turn> (<what happened>). Change given to the agent from here: "<change>". Verdict: <verdict> --
""" + THOUGHT_NOTE
KNOWLEDGE_LINES = """\
    hypothesis: <in this situation, doing this action leads to this result>
      situation: <...>
      action: <...>
      result: <...>
    verdict: <Supported | Refuted> (the result followed in <k> of <n> rollouts)
    from: <the experiments it was tested in>
  Entries are separated by a blank line."""
PLAN_INPUT = """\
- <plan>: the sub-goals the agent wrote before its first turn, one per line, in order:
    <n>. have <count> <item>"""


def fill(text: str) -> str:
    return text.replace("{rollouts}", str(HP["loop"]["rollouts"])).replace("{window_steps}",
                                                                           str(HP["loop"]["window_steps"]))


def term(name: str) -> str:
    """One term as the prompts list it ("- Name: definition", the doc's line breaks), placeholders filled."""
    return fill(TERMS[name])


def terms(*names: str) -> str:
    return "Terms used below:\n" + "\n".join(term(n) for n in names)


def tagged(**parts) -> str:
    """The user message: each input between its tags, in the order given; None parts are left out."""
    return "\n\n".join(f"<{k}>\n{v}\n</{k}>" for k, v in parts.items() if v is not None)
