"""World Knowledge (EXPERIMENTS.md 6): statements System 2 judged Supported, kept for later episodes.

Store: <runs>/memory/wk_<name>.json, a list of {"statement", "source", "prediction", "observed"}. The Baseline agent
gets the statements in its system prompt (token wk<name>, or the knowledge of the current loop in an experiment).
"""
from __future__ import annotations

import json
from pathlib import Path

NOTE = ("\n\nWhat earlier attempts found out about this world. Each item was tested: a prediction was written "
        "first and then observed in the game.\n{items}")


def load(path: str | None) -> list | None:
    return json.loads(Path(path).read_text()) if path else None


def add(path: Path, item: dict) -> None:
    items = json.loads(path.read_text()) if path.exists() else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items + [item], indent=1, ensure_ascii=False))


def prompt_text(items) -> str:
    return NOTE.format(items="\n".join(f"- {k['statement']}" for k in items)) if items else ""
