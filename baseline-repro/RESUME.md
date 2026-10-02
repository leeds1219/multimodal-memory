# Resume / check-in guide (for a human or a new Claude session)

Everything runs detached from any Claude session: supervisors → chains → one
process per episode. Nothing here needs a session to keep running.

## Check
    python scripts/status.py                      # progress, success so far, $ so far
    tail /home/rag/data/repro_runs/suite_optimus1/logs/*.log   # one log per chain
    cat /home/rag/data/repro_runs/suite_optimus1/LAUNCHES.md  # what was launched, where
    ls /home/rag/data/repro_runs/STOP_*           # any stop condition?
    tail PROGRESS.md DECISIONS.md

## Spend protection (no action needed)
- Global hard cap $3,000; **tripwires**: soft cap $1,500 total and $40 per hour
  (configs/llm.yaml). Tripping writes `/home/rag/data/repro_runs/STOP_BUDGET`;
  from then on every LLM call fails at $0 and chains stop before new episodes.
- Per-episode guard: 10× (smoke cost/calls per env step) × horizon → `ANOMALY`.
- Retries are bounded everywhere; finished episodes are never rerun; a chain
  refuses to start if the same chain is already running (exit 4).

## If STOP_BUDGET appeared
1. Read it and check `status.py` / the latest `llm/calls.jsonl` of running
   episodes to decide whether the spend was legitimate.
2. If legitimate: raise `soft_cap_usd` / `max_usd_per_hour` in configs/llm.yaml.
3. `python scripts/clear_budget_stops.py --apply` (makes the stopped episodes rerunnable)
4. `rm /home/rag/data/repro_runs/STOP_BUDGET`
5. Relaunch the chains listed as running in
   /home/rag/data/repro_runs/suite_optimus1/LAUNCHES.md with the same command (safe: a running
   chain is detected and never duplicated; a chain resumes at its first unfinished episode).

## If the machine / container restarted
1. `bash scripts/setup_env.sh` (re-creates the /opt/conda/envs/mcagent symlink and the apt
   packages; the env itself is on the data volume and survives).
2. Step 5 above, plus the reaper:
       nohup scripts/reaper_loop.sh >> /home/rag/data/repro_runs/reaper.log 2>&1 &

## Other stop files
- `STOP_DISK`: free disk < 5 GB → free space, delete the file, restart (step 5).

## When all chains are done
    python analysis/analyze.py      # tables (CSV + LaTeX), curves → analysis/out/
