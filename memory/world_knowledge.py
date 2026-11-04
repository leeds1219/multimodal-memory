"""World Knowledge (designer/terms.py "World Knowledge"): hypotheses the Designer's experiments judged Supported or
Refuted, as summaries only, kept for later episodes and attempts.

Store: <runs>/memory/wk_<name>.json, a list of {"statement", "situation", "action", "result", "verdict", "k", "n",
"source"} (verdict Supported or Refuted; R followed in k of n rollouts). The Designer gets every item (lines());
the agent gets the Supported ones in its system prompt (prompt_text(); items written before verdicts were stored
count as Supported).
"""
from __future__ import annotations

import json
from pathlib import Path

NOTE = ("\n\nWhat earlier attempts found out about this world. Each item was tested: a prediction was written "
        "first and then observed in the game.\n{items}")


def load(path: str | Path | None) -> list | None:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else ([] if path else None)


def add(path: Path, item: dict) -> None:
    path = Path(path)
    items = load(path) or []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items + [item], indent=1, ensure_ascii=False))


def supported(items) -> list:
    return [k for k in items or [] if k.get("verdict", "Supported") == "Supported"]


def lines(items) -> str:
    """The Designer's view: one line per item, "[Supported k/n] ..." or "[Refuted 0/n] ..."."""
    return "\n".join(f"[{k.get('verdict', 'Supported')} {k.get('k', '?')}/{k.get('n', '?')}] {k['statement']}"
                     for k in items or [])


def prompt_text(items) -> str:
    """The agent's view: the Supported statements only."""
    s = supported(items)
    return NOTE.format(items="\n".join(f"- {k['statement']}" for k in s)) if s else ""
