"""After a budget tripwire: make every episode that was stopped by it rerunnable.

An episode dir containing BUDGET_STOP (written by the LLM layer) is moved aside
to <seed>.budgetstop<ts>, whatever its result.json says (older chain code may
have recorded it as crashed_final). Then remove STOP_BUDGET and restart the
supervisors (RESUME.md). Dry run by default; pass --apply.
"""
import sys, time
from pathlib import Path
RUNS = Path("/home/rag/data/repro_runs")
apply = "--apply" in sys.argv
n = 0
for m in (RUNS / "runs").glob("*/*/*/*/*/BUDGET_STOP"):
    ep = m.parent
    if ".budgetstop" in ep.name or ".crash" in ep.name:
        continue
    n += 1
    print(("moving " if apply else "would move ") + str(ep))
    if apply:
        ep.rename(ep.with_name(ep.name + f".budgetstop{int(time.time())}"))
# crashed attempts of the same episodes (older chain code retried them)
for m in (RUNS / "runs").glob("*/*/*/*/*.crash*/BUDGET_STOP"):
    print(("  (attempt dir kept) " if apply else "  attempt: ") + str(m.parent))
print(f"{n} episode(s) {'moved' if apply else 'to move'}; STOP_BUDGET present: {(RUNS / 'STOP_BUDGET').exists()}")
