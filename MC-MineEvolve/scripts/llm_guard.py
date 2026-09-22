"""Operate the server's LLM spending guard (see src/mineevolve/util/llm_guard.py).

    python scripts/llm_guard.py status            # counters + paused/abort state
    python scripts/llm_guard.py resume            # after reviewing: clear PAUSED, restart the windows
    python scripts/llm_guard.py abort             # stop the evaluation at the next LLM call
    python scripts/llm_guard.py pause "reason"    # pause by hand (e.g. before inspecting a run)

Works through the server (--server, default http://127.0.0.1:9000) for status and
through the marker files in --dir (default logs/llm_guard) for everything else, so
it also works while the server is down.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "resume", "abort", "pause"])
    ap.add_argument("reason", nargs="?", default="paused by operator")
    ap.add_argument("--server", default="http://127.0.0.1:9000")
    ap.add_argument("--dir", default="logs/llm_guard")
    a = ap.parse_args()
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    paused, resume, abort = d / "PAUSED", d / "RESUME", d / "ABORT"

    if a.cmd == "status":
        try:
            st = requests.get(f"{a.server}/guard", timeout=10).json()
            print(json.dumps(st, indent=1))
        except Exception as exc:
            print(f"server not reachable ({exc}); marker files only:")
        for f in (paused, resume, abort):
            print(f"  {f.name:7} {'present' if f.exists() else '-'}")
        if paused.exists():
            print(paused.read_text())
        return 0
    if a.cmd == "resume":
        if not paused.exists():
            print("not paused"); return 0
        paused.rename(d / f"PAUSED.{time.strftime('%Y%m%d-%H%M%S')}.resolved")  # keep the record
        resume.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        print("resumed: the next LLM call restarts the episode/hour windows and grants a fresh session allowance")
        return 0
    if a.cmd == "abort":
        abort.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        print("ABORT written: the evaluation exits at its next LLM call (remove logs/llm_guard/ABORT to allow runs again)")
        return 0
    if a.cmd == "pause":
        paused.write_text(json.dumps({"reason": a.reason, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "paused": True}))
        print(f"PAUSED written ({a.reason})")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
