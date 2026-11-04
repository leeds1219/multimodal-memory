"""The Designer, step 3: judge an experiment. Did the moment the hypothesis is checked at come, and did the prediction
show then, in the experiment's log? Task success does not decide the verdict.
"""
from __future__ import annotations

import json

from llm_json import ask_json

JUDGE_SYSTEM = """You judge an experiment in {domain}. A hypothesis was written before the experiment: a change, the
moment the hypothesis is checked at (check_when), and what it predicts then. The agent then played from a saved
moment before the failure point and tried the change; the log starts at that moment. You get the hypothesis and the compressed log of the experiment.
Decide in this order, using only what the game recorded in the log ({log_shows}):
1. Did the check moment come? Find the first time in the log that shows check_when. If it never came:
   "Not reached".
2. At that first check moment, did what the prediction says the hypothesis leads to happen? Judge what it means,
   not each word: for example, an item the prediction says the agent will have counts if the agent had it then and
   used it. Yes: "Supported". No, or the opposite: "Refuted".
Only the first check moment counts. Whether the task succeeded does not matter.
Answer with JSON only:
{{"verdict": "Supported" | "Refuted" | "Not reached", "check_turn": <turn or null>,
 "observed": "<what the log shows at that turn, and whether and when the agent made the change>",
 "reason": "<why this verdict>"}}"""

VERDICTS = ('Supported', 'Refuted', 'Not reached')


def judge_system(vocab) -> str:
    return JUDGE_SYSTEM.format(domain=vocab.name, log_shows=vocab.log_shows)


def judge(hyp: dict, compressed: str, log: list, vocab) -> dict | None:
    """{verdict, check_turn, observed, reason} or None."""
    user = (f"Hypothesis: {json.dumps({k: hyp[k] for k in ('change', 'check_when', 'prediction')}, ensure_ascii=False)}\n\n"
            f"Experiment log:\n{compressed}")
    out = ask_json(judge_system(vocab), user, log, "judge")
    if not out or out.get("verdict") not in VERDICTS:
        return None
    return {"verdict": out["verdict"], "check_turn": out.get("check_turn"),
            "observed": str(out.get("observed", "")), "reason": str(out.get("reason", ""))}
