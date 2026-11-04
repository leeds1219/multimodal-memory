"""World Knowledge (designer/terms.py "World Knowledge"): hypotheses the Designer's experiments judged Supported or
Refuted, as summaries only, kept for later episodes and attempts.

Store: <runs>/memory/wk_<name>.json, a list of {"statement", "situation", "action", "result", "verdict", "k", "n",
"source", "sources"} (verdict Supported or Refuted; the result followed in k of n rollouts; "statement" is the
hypothesis sentence, shown to the Designer as "hypothesis"). It is one pool shared by every
attempt and task of the store, read and written without waiting for each other: add() takes a file lock, merges and
replaces the file in one rename, so readers never see a half-written file. The same rule tested again (same
situation, action and result) is merged into its item: k and n add up, the verdict follows the summed frequency
(Supported if the result ever followed), and the experiment joins "sources". The Designer gets every item (lines()); the agent
gets the Supported ones in its system prompt (prompt_text(); items written before verdicts were stored count as
Supported).
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

NOTE = ("\n\nWhat earlier attempts found out about this world. Each item was tested: a prediction was written "
        "first and then observed in the game.\n{items}")


def load(path: str | Path | None) -> list | None:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else ([] if path else None)


def _key(k: dict) -> tuple:
    return tuple(" ".join(str(k.get(f, "")).lower().split()) for f in ("situation", "action", "result"))


def add(path: Path, item: dict) -> dict:
    """Add a judged hypothesis, or merge it into the item of the same rule; returns the item as stored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # loops of other tasks may write the same pool at the same time
        items = load(path) or []
        old = next((k for k in items if _key(k) == _key(item)), None)
        if old:
            old["k"], old["n"] = old.get("k", 0) + item.get("k", 0), old.get("n", 0) + item.get("n", 0)
            old["verdict"] = "Supported" if old["k"] > 0 else "Refuted"
            old["sources"] = old.get("sources", [old.get("source")]) + [item.get("source")]
            out = old
        else:
            out = {**item, "sources": [item.get("source")]}
            items.append(out)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False))
        os.replace(tmp, path)
    return out


def supported(items) -> list:
    return [k for k in items or [] if k.get("verdict", "Supported") == "Supported"]


def lines(items) -> str:
    """The Designer's view: every item as a block, the hypothesis first (designer/terms.py KNOWLEDGE_LINES)."""
    return "\n\n".join(
        f"hypothesis: {k.get('statement') or k.get('hypothesis', '')}\n"
        f"  situation: {k.get('situation', '')}\n  action: {k.get('action', '')}\n  result: {k.get('result', '')}\n"
        f"verdict: {k.get('verdict', 'Supported')} (the result followed in {k.get('k', '?')} of {k.get('n', '?')} rollouts)\n"
        f"from: {', '.join(x for x in k.get('sources') or [k.get('source', '')] if x)}"
        for k in items or [])


def prompt_text(items) -> str:
    """The agent's view: the Supported statements only."""
    s = supported(items)
    return NOTE.format(items="\n".join(f"- {k['statement']}" for k in s)) if s else ""
