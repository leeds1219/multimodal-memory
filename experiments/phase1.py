"""Phase 1 (plan 2026-11-11): collect a method's memory on the train tasks of configs/suites/optimus1/phase_split.yaml,
level by level, under one budget of played game steps per level (phase1.horizons_per_train_task x the level's
horizon x its train tasks; configs/hyperparams.yaml). Phase 2 then plays the test tasks with a snapshot of the
memory, frozen.

A method without a loop (mineevolve, optimus1_memory) replays the level's train tasks round-robin, one episode per GPU at a time, every
task with its one seed (seeds.yaml), until the level's played steps reach its budget. An episode is launched only while
the steps played plus the horizons of the episodes in flight stay within the budget, or when none is in flight; so the
last episode of a level may go over by less than one horizon (the actual steps are reported). Memory carries over: one
store for the whole run. After every finished episode the store is copied (frames hard-linked) to
run_data/memory/snapshots/<store stem>/<n>_<order>/, and the state lists the episodes in the order they finished with
the run's measures so far: Phase 2 can take the memory at any episode, i.e. at any budget in steps, tokens or $.

    python experiments/phase1.py --method mineevolve --store p1me --world nolava --plan --gpus 0,1,2,3,4,5,6,7 --port 29100
    python experiments/phase1.py --method optimus1_memory --o1m p1om --world nolava --plan --gpus 0,1,2,3,4,5,6,7 --port 29200

Each episode is one chain.py call (--order p1-<store>-<task>-k<k> --tasks <task>, k = that task's k-th episode; one
order per episode, since chain.py refuses a second live chain on the same order), so results are
results/<condition>/p1-<store>-<task>-k<k>/<task>/<seed>/ and a restarted run resumes from them.
Progress: run_data/state/phase1_<store>.json.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
sys.path.insert(0, str(REPRO / "experiments"))
import paths  # noqa: E402,F401

from benchmark import RESULTS, RUN_DATA, SEEDS, TASKS, env_o_group  # noqa: E402
from conditions import Condition, add_args  # noqa: E402
from hyperparams import HP  # noqa: E402

PY = "/opt/conda/envs/mcagent/bin/python"
MINUTE = 1200  # steps per game minute (chain.py)
SPLIT = REPRO / "configs" / "suites" / "optimus1" / "phase_split.yaml"
STORES = {"mineevolve": ("--store", "OPTIMUS_VLM_ME"),  # method -> its store option and the variable naming the store
          "optimus1_memory": ("--o1m", "OPTIMUS_VLM_O1M")}
DONE = {"finished", "anomaly", "crashed_final"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def horizon(task: str) -> int:
    return int(env_o_group(TASKS[task]["group"])["max_minutes"]) * MINUTE


class Phase1:
    def __init__(self, a, cond: Condition):
        import yaml
        self.a, self.cond = a, cond
        self.levels = yaml.safe_load(SPLIT.read_text())["levels"]
        self.k_hz = HP["phase1"]["horizons_per_train_task"]
        self.budget = {L["name"]: int(self.k_hz * horizon(L["train"][0])) * len(L["train"]) for L in self.levels}
        self.total = sum(self.budget.values())
        self.store = Path(cond.env_vars(RUN_DATA, REPRO)[STORES[cond.method][1]])
        self.snaps = RUN_DATA / "memory" / "snapshots" / self.store.stem
        self.state_f = RUN_DATA / "state" / f"phase1_{a.store}.json"
        self.prefix = f"p1-{a.store}-"
        self.lock = threading.Condition()
        self.steps = {L["name"]: 0 for L in self.levels}  # played steps of finished episodes, per level
        self.episodes = {L["name"]: 0 for L in self.levels}
        self.k = {}  # task -> episodes started so far
        self.ledger = []  # finished episodes in order, each with the run's measures so far and its snapshot
        self._resume()

    # ----------------------------------------------------------------- state
    def _resume(self) -> None:
        level_of = {t: L["name"] for L in self.levels for t in L["train"]}
        root = RESULTS / self.cond.name
        for d in sorted(root.glob(f"{self.prefix}*-k*")) if root.exists() else []:
            m = re.fullmatch(re.escape(self.prefix) + r"(.+)-k(\d+)", d.name)
            if not m or m.group(1) not in level_of:
                continue
            t, k = m.group(1), int(m.group(2))
            self.k[t] = max(self.k.get(t, 0), k)
            res = d / t / str(SEEDS[t]) / "result.json"
            if res.exists():
                r = json.loads(res.read_text())
                if r.get("status") in DONE:
                    self.steps[level_of[t]] += int(r.get("steps") or 0)
                    self.episodes[level_of[t]] += 1
        st = json.loads(self.state_f.read_text()) if self.state_f.exists() else {}
        self.ledger = st.get("ledger", [])

    def measures(self) -> dict:
        """Every compute measure per level, summed over the run's finished episodes (result.json): the budget is
        played steps, the rest is reported beside it."""
        level_of = {t: L["name"] for L in self.levels for t in L["train"]}
        keys = ("steps", "llm_calls", "tokens_in", "tokens_out_billed", "cost_usd", "wall_time_s")
        out = {L["name"]: dict.fromkeys(("episodes", "successes") + keys, 0) for L in self.levels}
        root = RESULTS / self.cond.name
        for res in root.glob(f"{self.prefix}*-k*/*/*/result.json") if root.exists() else []:
            r = json.loads(res.read_text())
            t = res.parent.parent.name
            if t not in level_of or r.get("status") not in DONE:
                continue
            m = out[level_of[t]]
            m["episodes"] += 1
            m["successes"] += bool(r.get("success"))
            for k in keys:
                m[k] += r.get(k) or 0
        out["total"] = {k: sum(m[k] for m in out.values()) for k in out[self.levels[0]["name"]]}
        return out

    def _write_state(self) -> None:
        self.state_f.parent.mkdir(parents=True, exist_ok=True)
        self.state_f.write_text(json.dumps({
            "condition": self.cond.name, "store": str(self.store), "horizons_per_train_task": self.k_hz,
            "budget": self.budget, "steps": self.steps, "episodes": self.episodes,
            "played_total": sum(self.steps.values()), "budget_total": self.total,
            "measures": self.measures(), "ledger": self.ledger,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    def _snapshot(self, label: str, r: dict | None) -> None:  # call with self.lock held
        """Copy the store as it is after this episode; frames are hard-linked (they never change once written)."""
        dst = self.snaps / f"{len(self.ledger) + 1:05d}_{label}"
        if self.store.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            is_dir = self.store.is_dir()  # optimus1_memory: a directory; mineevolve: one file
            with open(self.store / "store.lock" if is_dir else str(self.store) + ".lock", "a") as lf:  # the store's lock
                fcntl.flock(lf, fcntl.LOCK_EX)
                if is_dir:
                    shutil.copytree(self.store, dst, ignore=shutil.ignore_patterns("store.lock"),
                                    copy_function=lambda a, b: os.link(a, b) if a.endswith((".jpg", ".png"))
                                    else shutil.copy2(a, b))
                else:
                    dst.mkdir()
                    shutil.copy2(self.store, dst / self.store.name)
                fcntl.flock(lf, fcntl.LOCK_UN)
        tot = self.measures()["total"]
        self.ledger.append({"n": len(self.ledger) + 1, "episode": label, "status": (r or {}).get("status"),
                            "success": (r or {}).get("success"), "snapshot": str(dst) if self.store.exists() else None,
                            "after": tot})

    # --------------------------------------------------------------- episode
    def _episode(self, task: str, k: int, gpu: int, port: int) -> dict:
        order = f"{self.prefix}{task}-k{k}"
        cmd = [PY, str(REPRO / "experiments" / "chain.py"), "--method", self.cond.method, "--world", self.cond.world,
               STORES[self.cond.method][0], self.a.store, "--order", order, "--tasks", task, "--gpu", str(gpu), "--port", str(port)]
        if self.cond.plan:
            cmd.append("--plan")
        if self.a.mock:
            cmd.append("--mock")
        log(f"[gpu {gpu}] {task} k{k} start")
        out = RUN_DATA / "logs" / f"phase1_{self.a.store}" / f"{order}_{task}.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        res = RESULTS / self.cond.name / order / task / str(SEEDS[task]) / "result.json"
        r = json.loads(res.read_text()) if res.exists() else {"status": "missing", "chain_rc": rc}
        log(f"[gpu {gpu}] {task} k{k} -> {r.get('status')} success={r.get('success')} steps={r.get('steps')} "
            f"cost={r.get('cost_usd')}")
        return r

    def _worker(self, level: str, tasks: list, nxt: list, inflight: dict, gpu: int, port: int) -> None:
        while True:
            with self.lock:
                while True:
                    if self.steps[level] >= self.budget[level] or self.stop:
                        return
                    task = tasks[nxt[0] % len(tasks)]
                    if not inflight or self.steps[level] + sum(inflight.values()) + horizon(task) <= self.budget[level]:
                        break
                    self.lock.wait()
                nxt[0] += 1
                self.k[task] = k = self.k.get(task, 0) + 1
                inflight[(gpu, task, k)] = horizon(task)
            r = self._episode(task, k, gpu, port)
            with self.lock:
                del inflight[(gpu, task, k)]
                if r.get("status") in DONE:
                    self.steps[level] += int(r.get("steps") or 0)
                    self.episodes[level] += 1
                elif r.get("status") == "missing":  # chain.py ran no episode: disk, LLM cap, STOP_BUDGET or an error
                    log(f"STOP: chain.py ran no episode (exit {r.get('chain_rc')}); see run_data/logs/phase1_{self.a.store}/")
                    self.stop = True
                self._snapshot(f"{task}-k{k}", r)
                self._write_state()
                self.lock.notify_all()

    # ------------------------------------------------------------------ main
    def run(self) -> int:
        self.stop = False
        if not self.ledger and sum(self.episodes.values()):  # resumed from a run without a ledger: one entry for it
            with self.lock:
                self._snapshot("resume", None)
        gpus = [int(g) for g in self.a.gpus.split(",")]
        log(f"{self.cond.name}: store {self.store}, budget {self.total} steps "
            f"({self.k_hz} horizons per train task), {len(gpus)} GPUs")
        for L in self.levels:
            name, tasks = L["name"], L["train"]
            if self.a.levels and name not in self.a.levels.split(","):
                continue
            if self.steps[name] >= self.budget[name]:
                log(f"level {name}: done earlier ({self.steps[name]}/{self.budget[name]} steps)")
                continue
            log(f"level {name}: {len(tasks)} train tasks, budget {self.budget[name]} steps "
                f"({self.steps[name]} played before)")
            nxt, inflight = [sum(self.k.get(t, 0) for t in tasks)], {}
            ws = [threading.Thread(target=self._worker, args=(name, tasks, nxt, inflight, g, self.a.port + 10 * i))
                  for i, g in enumerate(gpus[:len(tasks)])]
            for w in ws:
                w.start()
            for w in ws:
                w.join()
            log(f"level {name}: {self.steps[name]}/{self.budget[name]} steps in {self.episodes[name]} episodes")
            if self.stop:
                return 2
        with self.lock:
            self._write_state()
        log(f"done: {sum(self.steps.values())} steps played, budget {self.total}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--gpus", required=True, help="comma list; one episode per GPU at a time")
    ap.add_argument("--port", type=int, required=True, help="first port; worker i uses port + 10 i (and + 1000)")
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls")
    ap.add_argument("--levels", default="", help="comma list of levels to play (tests); default: all, in order")
    ap.add_argument("--report", action="store_true", help="print the run's measures per level and exit")
    a = ap.parse_args()
    if a.method not in STORES or not (a.store or a.o1m) or a.frozen:
        ap.error(f"phase1 needs a method with a memory store ({', '.join(STORES)}), its store name and no --frozen")
    a.store = a.store or a.o1m  # the run's store name, also used for the orders and the state file
    cond = Condition(a.method, a.world, a.save, a.knowledge, a.plan,
                     a.store if a.method == "mineevolve" else None, a.o1m)
    if a.report:
        print(json.dumps(Phase1(a, cond).measures(), indent=1))
        return 0
    return Phase1(a, cond).run()


if __name__ == "__main__":
    sys.exit(main())
