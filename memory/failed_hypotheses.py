"""Failed hypotheses: hypotheses the Designer judged Refuted or Not reached, kept for later proposals.

Store: <runs>/memory/fh_<name>.json (next to World Knowledge, memory/world_knowledge.py), a list of {"statement",
"change", "plan", "check_when", "prediction", "verdict", "observed", "reason", "source"}. A proposal at any failure
point of any episode gets the latest of them (designer/hypothesis.py), so it can avoid repeating what did not work.
"""
from __future__ import annotations

import json
from pathlib import Path

FIELDS = ("statement", "change", "plan", "check_when", "prediction", "verdict", "observed", "reason")


def load(path: Path) -> list:
    return json.loads(Path(path).read_text()) if Path(path).exists() else []


def add(path: Path, item: dict) -> None:
    path = Path(path)
    items = load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items + [{k: item.get(k, "") for k in FIELDS + ("source",)}], indent=1,
                               ensure_ascii=False))
