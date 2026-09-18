"""Sum LLM cost across evaluation runs (Gemini 3 Flash list price by default).

    python scripts/spend.py                 # all runs under logs/eval
    python scripts/spend.py --since 2026-09-18
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRICE_IN, PRICE_OUT = 0.75, 3.75  # $/1M tokens, gemini-3-flash-preview (intro pricing)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--root", default="logs/eval")
    a = ap.parse_args()
    total = 0.0; rows = []
    for f in sorted(Path(a.root).glob("*/*/llm_calls.jsonl")):
        if a.since and f.parts[-3] < a.since:
            continue
        pi = po = n = 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line); pi += r.get("prompt_tokens") or 0; po += r.get("completion_tokens") or 0; n += 1
        cost = (pi * PRICE_IN + po * PRICE_OUT) / 1e6; total += cost
        runs = f.parent / "runs.jsonl"
        tasks = sorted({json.loads(l)["task_id"] for l in runs.read_text().splitlines() if l.strip()}) if runs.exists() else []
        succ = sum(json.loads(l)["success"] for l in runs.read_text().splitlines() if l.strip()) if runs.exists() else 0
        nruns = sum(1 for l in runs.read_text().splitlines() if l.strip()) if runs.exists() else 0
        rows.append((f.parts[-3], f.parts[-2], tasks, f"{succ}/{nruns}", n, cost))
    for d, t, tasks, sr, n, c in rows:
        print(f"{d} {t}  tasks={tasks!s:14} {sr:5}  calls={n:3d}  ${c:.3f}")
    print(f"TOTAL ${total:.2f} over {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
