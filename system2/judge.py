"""System 2, step 3: judge an experiment. Was the hypothesis' prediction observed in the experiment's log?
Task success does not decide the verdict.
"""
from __future__ import annotations

import json

from llm_json import ask_json

JUDGE_SYSTEM = """You judge an experiment in Minecraft. A hypothesis was written before the experiment, with a prediction.
The agent then played from a saved moment and tried the change. You get the hypothesis and the compressed log of the
experiment.
Decide only whether the prediction was observed in the log. Whether the task succeeded does not matter: the
prediction can be observed in an attempt that still fails, and an attempt can succeed without it.
- "Supported": what the prediction says happened.
- "Refuted": the change was tried and what the prediction says did not happen, or the opposite happened.
- "Not tried": the agent never made the change, so the log cannot show it either way.
Answer with JSON only:
{"verdict": "Supported" | "Refuted" | "Not tried", "observed": "<what the log shows, with turns>",
 "reason": "<why this verdict>"}"""

VERDICTS = ('Supported', 'Refuted', 'Not tried')


def judge(hyp: dict, compressed: str, log: list) -> dict | None:
    """{verdict, observed, reason} or None."""
    user = (f"Hypothesis: {json.dumps({k: hyp[k] for k in ('change', 'prediction', 'plan')}, ensure_ascii=False)}\n\n"
            f"Experiment log:\n{compressed}")
    out = ask_json(JUDGE_SYSTEM, user, log, "judge")
    if not out or out.get("verdict") not in VERDICTS:
        return None
    return {"verdict": out["verdict"], "observed": str(out.get("observed", "")), "reason": str(out.get("reason", ""))}
