"""What the method adds to an Executor agent of any domain: the World Knowledge verified so far and, in an experiment
of the learning-from-failure loop (experiments/ours_loop.py), the change to try (start, do, stop) and its plan. The
agent appends prompt_text() to its system prompt; the hypothesis and the prediction are never shown (the agent must
not aim at them).

Read from the environment of the episode:
  OPTIMUS_EXPERIMENT   JSON file {"hypothesis", "knowledge", "from_start"} written by the loop (domains/base.py)
  OPTIMUS_VLM_WK       a World Knowledge store (memory/world_knowledge.py), outside experiments
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import world_knowledge

EXP_NOTE = ("\n\nThis attempt starts from a saved moment of an earlier attempt at this task: the world, your position, "
            "inventory and your last commands are as they were then. In the earlier attempt the task failed. "
            "From here, try this change:\n{change}\nHow to try it: {plan}\n"
            "Keep working on the overall task after trying it.")
EXP_NOTE_START = ("\n\nThis attempt starts from the beginning, in the same world as an earlier attempt at this task. In the "
                  "earlier attempt the task failed. This time, try this change:\n{change}\nHow to try it: {plan}\nKeep working on the overall task after trying it.")


def load() -> dict:
    """{"knowledge": [...] | None, "experiment": hypothesis | None, "from_start": bool | None}"""
    exp = json.loads(Path(os.environ["OPTIMUS_EXPERIMENT"]).read_text()) if os.environ.get("OPTIMUS_EXPERIMENT") else {}
    return {"knowledge": exp.get("knowledge") if exp else world_knowledge.load(os.environ.get("OPTIMUS_VLM_WK")),
            "experiment": exp.get("hypothesis"), "from_start": exp.get("from_start")}


def change_text(c) -> str:
    """The change as the agent is told it: when it starts, what to do, when it stops (old experiments: one string)."""
    return f"When {c['start']}: {c['do']}. Stop when {c['stop']}." if isinstance(c, dict) else str(c)


def prompt_text(x: dict) -> str:
    out = world_knowledge.prompt_text(x.get("knowledge"))
    if x.get("experiment"):
        h = x["experiment"]
        out += (EXP_NOTE_START if x.get("from_start") else EXP_NOTE).format(change=change_text(h["change"]),
                                                                          plan=h["plan"])
    return out
