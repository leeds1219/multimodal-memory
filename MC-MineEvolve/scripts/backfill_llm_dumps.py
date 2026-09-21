"""Copy the full prompt/response dumps of a run's LLM calls into <run dir>/llm_calls/.

    python scripts/backfill_llm_dumps.py logs/eval/2026-09-19/09-21-55 [more run dirs]

Runs made before the dump path was written into llm_calls.jsonl only have the usage
lines; the dumps still exist under logs/llm_calls/ (one file per call, with the same
`t` timestamp and stage). This matches them by (t, stage) and copies them so the run
dir holds everything needed for offline analysis.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

DUMP_DIR = os.environ.get("MINEEVOLVE_LLM_DUMP_DIR", "logs/llm_calls")


def main() -> int:
    index = {}
    for f in glob.glob(os.path.join(DUMP_DIR, "*.json")):
        try:
            with open(f) as fh:
                head = json.load(fh)
        except (OSError, ValueError):
            continue
        index[(round(float(head.get("t", 0)), 3), head.get("stage"))] = f
    for d in sys.argv[1:]:
        calls = os.path.join(d, "llm_calls.jsonl")
        if not os.path.exists(calls):
            print(f"{d}: no llm_calls.jsonl"); continue
        out = os.path.join(d, "llm_calls"); os.makedirs(out, exist_ok=True)
        n = hit = 0
        for line in open(calls):
            if not line.strip():
                continue
            r = json.loads(line); n += 1
            src = r.get("dump") or index.get((round(float(r["t"]), 3), r.get("stage")))
            if src and os.path.exists(src):
                shutil.copy2(src, os.path.join(out, f"{n:04d}_{r.get('stage')}.json")); hit += 1
        print(f"{d}: {hit}/{n} dumps copied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
