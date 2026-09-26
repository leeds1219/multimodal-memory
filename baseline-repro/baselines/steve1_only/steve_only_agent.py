"""STEVE-1-only baseline (no LLM planner): the task instruction is the STEVE-1
text prompt for the whole episode; no crafting helper (as in the MineEvolve
paper's "STEVE-1 only" row). Runs until the target item is obtained or the
horizon ends. Makes no LLM calls."""
from __future__ import annotations

import json
from pathlib import Path


class SteveOnlyAgent:
    replan_rounds = 0

    def __init__(self, env, instruction: str, checker, artifact_dir: Path) -> None:
        self.env, self.prompt, self.checker = env, instruction, checker
        self.dir = Path(artifact_dir); self.dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        (self.dir / "prompt.json").write_text(json.dumps({"prompt": self.prompt}))

    def run(self) -> str:
        self.env.reset()
        stopped, _ = self.env.steve(self.prompt, 10**9, stop=lambda inv: self.checker.satisfied(inv))
        return "task_done" if stopped else "horizon"
