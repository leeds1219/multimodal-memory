"""Token usage + API cost estimate from a ``scripts/local_llm_server.py`` log.

The local server logs one line per completed request::

    ... local_llm_server: prompt=8227 gen=337 tok in 12.0s (28.0 tok/s) finish=stop

Feed that log here to get totals and what the same traffic would cost on the
API backends in ``conf/llm/``::

    python scripts/llm_usage.py llm.log                 # whole log
    python scripts/llm_usage.py llm.log --skip 96       # ignore the first 96 requests
    python scripts/llm_usage.py llm.log --tasks 11      # scale one task to a benchmark
    python scripts/llm_usage.py llm.log --json out.json

Caveats: token counts are the local model's tokenizer (Qwen3.5), API tokenizers
differ by ~10-20 %; prompts truncated by ``--truncate-prompt`` would be longer
on an API model; and a stronger model usually needs *fewer* repair rounds, so
this is closer to an upper bound for the same task horizon.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# USD per 1M tokens (input, output), list prices as of Sep 2026. Update when they move.
PRICES = {
    "qwen-flash": (0.15, 0.47),
    "qwen-plus": (0.40, 1.60),
    "gemini-3.6-flash": (0.75, 3.75),  # intro price to 2026-12-31; 2.5-flash ($0.30/$2.50) returned "not available to new users" for our key
    "gpt-4o": (2.50, 10.00),
}
LINE_RE = re.compile(r"prompt=(\d+) gen=(\d+) tok in ([0-9.]+)s")


def parse(log: Path, skip: int = 0, limit: int | None = None) -> list[dict]:
    """Read either the local server's text log or the server-side
    ``logs/llm_calls.jsonl`` written by the API backends (has a stage per call)."""
    calls = []
    if log.suffix == ".jsonl":
        for line in log.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            calls.append({"prompt": int(r.get("prompt_tokens") or 0), "gen": int(r.get("completion_tokens") or 0),
                          "s": float(r.get("s") or 0), "stage": r.get("stage", "?"), "model": r.get("model", "?"),
                          "finish": r.get("finish")})
    else:
        for m in LINE_RE.finditer(log.read_text(errors="replace")):
            calls.append({"prompt": int(m[1]), "gen": int(m[2]), "s": float(m[3]), "stage": "?"})
    calls = calls[skip:]
    return calls[:limit] if limit else calls


def by_stage(calls: list[dict]) -> dict:
    out: dict = {}
    for c in calls:
        d = out.setdefault(c.get("stage", "?"), {"calls": 0, "prompt": 0, "gen": 0, "truncated": 0})
        d["calls"] += 1; d["prompt"] += c["prompt"]; d["gen"] += c["gen"]
        d["truncated"] += int(c.get("finish") == "length")
    return out


def summarize(calls: list[dict], tasks: int = 1) -> dict:
    n = len(calls)
    p = sum(c["prompt"] for c in calls)
    g = sum(c["gen"] for c in calls)
    t = sum(c["s"] for c in calls)
    cost = {
        m: {
            "per_run_usd": round((p * ci + g * co) / 1e6, 4),
            "per_benchmark_usd": round((p * ci + g * co) / 1e6 * tasks, 3),
        }
        for m, (ci, co) in PRICES.items()
    }
    return {
        "calls": n,
        "prompt_tokens": p,
        "gen_tokens": g,
        "avg_prompt": round(p / n) if n else 0,
        "avg_gen": round(g / n) if n else 0,
        "max_prompt": max((c["prompt"] for c in calls), default=0),
        "llm_time_s": round(t),
        "tok_per_s": round(g / t, 1) if t else 0.0,
        "tasks_scaled": tasks,
        "cost_usd": cost,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--skip", type=int, default=0, help="ignore the first N requests (e.g. an earlier run)")
    ap.add_argument("--limit", type=int, default=None, help="use at most N requests after --skip")
    ap.add_argument("--tasks", type=int, default=1, help="multiply per-run cost by this many tasks")
    ap.add_argument("--json", type=Path, default=None, help="also write the summary here")
    args = ap.parse_args()

    calls = parse(args.log, args.skip, args.limit)
    if not calls:
        sys.exit(f"no 'prompt=... gen=...' lines in {args.log}")
    s = summarize(calls, args.tasks)
    stages = by_stage(calls)
    if set(stages) != {"?"}:
        s["by_stage"] = stages
        print(f"{'stage':14} {'calls':>5} {'prompt tok':>11} {'gen tok':>8} {'truncated':>9}")
        for st, d in sorted(stages.items(), key=lambda kv: -kv[1]["calls"]):
            print(f"{st:14} {d['calls']:5d} {d['prompt']:11,} {d['gen']:8,} {d['truncated']:9d}")
    print(
        f"{s['calls']} calls | prompt {s['prompt_tokens']:,} tok (avg {s['avg_prompt']:,}, max {s['max_prompt']:,})"
        f" | gen {s['gen_tokens']:,} tok (avg {s['avg_gen']}) | {s['llm_time_s']} s of LLM time @ {s['tok_per_s']} tok/s"
    )
    print(f"{'model':18} {'$/1M in':>8} {'$/1M out':>9} {'per run':>9} {'x' + str(args.tasks) + ' tasks':>11}")
    for m, (ci, co) in PRICES.items():
        c = s["cost_usd"][m]
        print(f"{m:18} {ci:8.2f} {co:9.2f} {c['per_run_usd']:9.4f} {c['per_benchmark_usd']:11.3f}")
    if args.json:
        args.json.write_text(json.dumps(s, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
