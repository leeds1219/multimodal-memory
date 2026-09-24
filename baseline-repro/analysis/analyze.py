"""Results tables and curves (Phase 6). Every table is per environment.

    python analysis/analyze.py            # writes analysis/out/*.csv|.tex|.png

Outputs
  success_by_group_{env}.csv/.tex   SR per group, Overall (task-weighted over 70
                                    tasks, MineEvolve paper) and Overall-5
                                    (mean of Iron/Gold/Diamond/Redstone/Armor
                                    groups, Optimus-1 paper); mean ± std over
                                    task orders for memory methods
  cumulative_{env}.csv/.png         running SR vs. task index (per order,
                                    mean ± std across orders)
  memory_overhead_{env}.csv         per snapshot: #entries, per-episode LLM
                                    calls / input tokens / latency
  failure_types_{env}.csv           failure category counts per method
  cost_{env}.csv                    calls, tokens, $ per method
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "envs"))
from common import ORDERS, RUNS_ROOT, TASKS  # noqa: E402

OUT = REPRO / "analysis" / "out"
GROUPS = ["wooden", "stone", "iron", "gold", "redstone", "diamond", "armor"]
HARD5 = ["iron", "gold", "diamond", "redstone", "armor"]
SKIP_ORDERS = {"smoke", "mocktest", "mockfix"}


def load():
    res = []
    for f in (RUNS_ROOT / "runs").glob("*/*/*/*/*/result.json"):
        env, chain, order, task, seed = f.parts[-6:-1]
        if order in SKIP_ORDERS or ".crash" in seed:
            continue
        r = json.loads(f.read_text())
        r.update(env=env, chain=chain, order_id=order, dir=str(f.parent))
        res.append(r)
    return res


# ---------------------------------------------------------------- failure types
def failure_type(r: dict) -> str:
    """Coarse attribution (see PROGRESS for the definitions)."""
    if r.get("success"):
        return "success"
    if r.get("status") == "anomaly":
        return "anomaly"
    if r.get("status") == "crashed_final":
        return "infra_crash"
    end = str(r.get("end_reason") or "")
    if end == "method_exception":
        return "planning/knowledge (method error)"
    if r.get("plan_source") == "example_fallback":
        return "planning/knowledge (no LLM plan)"
    if end == "death_or_done":
        return "situation (death)"
    traj = Path(r["dir"]) / "trajectory.jsonl.gz"
    if end in ("plan_finished", "plan_exhausted", "replan_rounds_exceeded", "plan_failed_or_timeout", "task_done"):
        return "planning/knowledge (plan ended without target)"
    # ran to the horizon: controller if the inventory barely changed, else planning
    inv = r.get("final_inventory") or {}
    return "controller (horizon, little progress)" if sum(inv.values()) < 3 else "controller/planning (horizon)"


def fmt(m, s=None):
    return f"{m:.1f}" if s is None or s == 0 else f"{m:.1f}±{s:.1f}"


def write_table(rows, header, path_csv, path_tex, caption):
    path_csv.write_text("\n".join([",".join(header)] + [",".join(map(str, r)) for r in rows]) + "\n")
    tex = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
           "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    tex += [" & ".join(str(c).replace("±", "$\\pm$").replace("_", "\\_") for c in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path_tex.write_text("\n".join(tex) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = load()
    by_env = defaultdict(list)
    for r in res:
        by_env[r["env"]].append(r)
    for env, rs in sorted(by_env.items()):
        # ---- success by group --------------------------------------------------
        per = defaultdict(lambda: defaultdict(dict))  # chain -> order -> task -> success
        for r in rs:
            per[r["chain"]][r["order_id"]][r["task"]] = bool(r.get("success"))
        rows = []
        for chain, orders in sorted(per.items()):
            vals = defaultdict(list)
            complete = 0
            for order, tasks in orders.items():
                for g in GROUPS:
                    ts = [u for u, t in TASKS.items() if t["group"] == g and u in tasks]
                    if ts:
                        vals[g].append(100 * sum(tasks[u] for u in ts) / len(ts))
                done = [u for u in TASKS if u in tasks]
                if len(done) == len(TASKS):
                    complete += 1
                if done:
                    vals["overall"].append(100 * sum(tasks[u] for u in done) / len(done))
                g5 = [st.mean([100 * tasks[u] for u in TASKS if TASKS[u]["group"] == g and u in tasks] or [0]) for g in HARD5]
                vals["overall5"].append(st.mean(g5))
            row = [chain, f"{complete}/{len(orders)}"]
            for k in GROUPS + ["overall", "overall5"]:
                v = vals.get(k, [])
                row.append(fmt(st.mean(v), st.pstdev(v) if len(v) > 1 else None) if v else "-")
            rows.append(row)
        write_table(rows, ["method", "orders complete"] + GROUPS + ["Overall(70, weighted)", "Overall-5 (Optimus-1 def.)"],
                    OUT / f"success_by_group_{env}.csv", OUT / f"success_by_group_{env}.tex",
                    f"Success rate (\\%) in Env {env}. Mean$\\pm$std over task orders.")

        # ---- cumulative curves ---------------------------------------------------
        cum_rows = []
        curves = {}
        for chain, orders in per.items():
            ks = []
            for order, tasks in orders.items():
                seq = ORDERS.get(order, [])
                acc, n, curve = 0, 0, []
                for u in seq:
                    if u not in tasks:
                        break
                    n += 1; acc += tasks[u]; curve.append(100 * acc / n)
                    cum_rows.append([chain, order, n, u, int(tasks[u]), round(100 * acc / n, 2)])
                ks.append(curve)
            curves[chain] = ks
        (OUT / f"cumulative_{env}.csv").write_text("chain,order,index,task,success,running_sr\n" +
                                                   "\n".join(",".join(map(str, r)) for r in cum_rows) + "\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            for chain, ks in sorted(curves.items()):
                L = min((len(k) for k in ks), default=0)
                if L == 0:
                    continue
                m = [st.mean(k[i] for k in ks) for i in range(L)]
                s = [st.pstdev([k[i] for k in ks]) if len(ks) > 1 else 0 for i in range(L)]
                x = list(range(1, L + 1))
                ax.plot(x, m, label=chain)
                ax.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)], alpha=0.2)
            ax.set_xlabel("task index in order"); ax.set_ylabel("running success rate (%)")
            ax.set_title(f"Env {env}: cumulative success (mean ± std over orders)"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(OUT / f"cumulative_{env}.png", dpi=150); plt.close(fig)
        except Exception as e:
            print("plot skipped:", e)

        # ---- memory overhead -----------------------------------------------------
        mo = []
        for r in rs:
            snap_root = RUNS_ROOT / "memory_snapshots" / env / r["chain"] / r["order_id"]
            idx = ORDERS.get(r["order_id"], []).index(r["task"]) + 1 if r["task"] in ORDERS.get(r["order_id"], []) else None
            n_entries = None
            if idx and snap_root.exists():
                snap = snap_root / f"{idx:02d}_{r['task']}"
                prev = snap_root / (f"{idx - 1:02d}_{ORDERS[r['order_id']][idx - 2]}" if idx > 1 else "00_initial")
                src = prev if prev.exists() else None
                if src is not None:
                    if (src / "skills.json").exists():
                        n_entries = sum(len(json.loads((src / f).read_text() or "[]"))
                                        for f in ("skills.json", "remedies.json") if (src / f).exists())
                    else:
                        n_entries = sum(1 for _ in src.rglob("*.json"))
            mo.append([r["chain"], r["order_id"], idx, r["task"], n_entries, r.get("llm_calls"), r.get("tokens_in"),
                       r.get("llm_latency_s"), round(r.get("cost_usd") or 0, 4)])
        mo.sort(key=lambda x: (x[0], x[1], x[2] or 0))
        (OUT / f"memory_overhead_{env}.csv").write_text(
            "chain,order,index,task,memory_entries_before,llm_calls,tokens_in,llm_latency_s,cost_usd\n" +
            "\n".join(",".join("" if c is None else str(c) for c in row) for row in mo) + "\n")

        # ---- failure types / cost ---------------------------------------------------
        ft = defaultdict(lambda: defaultdict(int))
        cost = defaultdict(lambda: [0, 0, 0, 0.0, 0])
        for r in rs:
            ft[r["chain"]][failure_type(r)] += 1
            c = cost[r["chain"]]
            c[0] += 1; c[1] += r.get("llm_calls") or 0; c[2] += r.get("tokens_in") or 0
            c[3] += r.get("cost_usd") or 0.0; c[4] += r.get("tokens_out_billed") or 0
        cats = sorted({k for d in ft.values() for k in d})
        (OUT / f"failure_types_{env}.csv").write_text("chain," + ",".join(cats) + "\n" + "\n".join(
            chain + "," + ",".join(str(d.get(k, 0)) for k in cats) for chain, d in sorted(ft.items())) + "\n")
        crow = [[ch, c[0], round(c[1] / c[0], 1), round(c[2] / c[0]), round(c[4] / c[0]), round(c[3] / c[0], 4), round(c[3], 2)]
                for ch, c in sorted(cost.items())]
        write_table(crow, ["method", "episodes", "calls/ep", "tok_in/ep", "tok_out/ep", "$/ep", "$ total"],
                    OUT / f"cost_{env}.csv", OUT / f"cost_{env}.tex", f"LLM usage per episode, Env {env}.")
        print(f"Env {env}: {len(rs)} episodes -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
