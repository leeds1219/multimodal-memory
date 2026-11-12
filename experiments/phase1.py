"""Phase 1 (plan 2026-11-11): collect a method's memory on the train tasks of configs/suites/optimus1/phase_split.yaml,
level by level. Every train task is played once in each world of phase1.seed_offsets (configs/hyperparams.yaml; the
seeds of worlds A, B, C), with no budget cut while it runs: a method without a loop (mineevolve, optimus1_memory)
plays one episode per (task, world). Ours plays one episode per (task, world) with the World Knowledge so far and
world saves; if it fails, the learning-from-failure loop (experiments/ours_loop.py) runs on it, and its
experiments' <success> segments come from the episodes this run played so far (run_data/state/phase1_<store>_pool.txt).
Phase 2 then plays the test tasks with a snapshot of the memory, frozen.

Memory carries over: one store for the whole run, levels in order (a level starts once the previous one is done).
After every finished episode the store is copied (frames hard-linked) to
run_data/memory/snapshots/<store stem>/<n>_<episode>/, and the state lists the episodes in the order they finished with
the run's measures so far (steps, LLM calls, tokens, $, wall time): the budget is cut afterwards, at any episode, in
any of these measures, and Phase 2 takes the snapshot there.

    python experiments/phase1.py --method mineevolve --store p1me --world nolava --plan --gpus 0,1,2,3,4,5,6,7 --port 29100
    python experiments/phase1.py --method optimus1_memory --o1m p1om --world nolava --plan --gpus 0,1,2,3,4,5,6,7 --port 29200
    python experiments/phase1.py --method ours --knowledge p1wk --world nolava --save 600 --plan --gpus 0,1,2,3 --port 28400

Each episode is one chain.py call (--order p1-<store>-<task>-s<offset> --tasks <task> --seed-offset <offset>; one order
per episode, since chain.py refuses a second live chain on the same order), so results are
results/<condition>/p1-<store>-<task>-s<offset>/<task>/<seed>/ and a restarted run skips the finished ones; Ours'
loop on it is results/ours_loop/p1-<store>-<task>-s<offset>/ (its episodes and the Designer's calls count in the
measures).
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

from benchmark import RESULTS, RUN_DATA, SEEDS  # noqa: E402
from conditions import Condition, add_args  # noqa: E402
from hyperparams import HP  # noqa: E402

PY = "/opt/conda/envs/mcagent/bin/python"
SPLIT = REPRO / "configs" / "suites" / "optimus1" / "phase_split.yaml"
STORES = {"mineevolve": ("--store", "OPTIMUS_VLM_ME"),  # method -> its store option and the variable naming the store
          "optimus1_memory": ("--o1m", "OPTIMUS_VLM_O1M"), "ours": ("--knowledge", "OPTIMUS_VLM_WK")}
DONE = {"finished", "anomaly", "crashed_final"}
KEYS = ("steps", "llm_calls", "tokens_in", "tokens_out_billed", "cost_usd", "wall_time_s")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)



class Phase1:
    def __init__(self, a, cond: Condition):
        import yaml
        self.a, self.cond = a, cond
        self.levels = yaml.safe_load(SPLIT.read_text())["levels"]
        self.offsets = [int(o) for o in HP["phase1"]["seed_offsets"]]
        self.store = Path(cond.env_vars(RUN_DATA, REPRO)[STORES[cond.method][1]])
        self.snaps = RUN_DATA / "memory" / "snapshots" / self.store.stem
        self.state_f = RUN_DATA / "state" / f"phase1_{a.store}.json"
        self.prefix = f"p1-{a.store}-"
        self.pool = RUN_DATA / "state" / f"phase1_{a.store}_pool.txt"  # Ours: episodes the <success> segments come from
        self.lock = threading.Condition()
        self.done = set()  # (task, offset) of finished episodes
        self.ledger = []  # finished episodes in order, each with the run's measures so far and its snapshot
        self._resume()

    # ----------------------------------------------------------------- state
    def _order(self, task: str, off: int) -> str:
        return f"{self.prefix}{task}-s{off}"

    def _result(self, task: str, off: int) -> Path:
        return RESULTS / self.cond.name / self._order(task, off) / task / str(SEEDS[task] + off) / "result.json"

    def _resume(self) -> None:
        for L in self.levels:
            for t in L["train"]:
                for off in self.offsets:
                    res = self._result(t, off)
                    if res.exists() and json.loads(res.read_text()).get("status") in DONE:
                        self.done.add((t, off))
        st = json.loads(self.state_f.read_text()) if self.state_f.exists() else {}
        self.ledger = st.get("ledger", [])

    def _loop_dir(self, task: str, off: int) -> Path:
        return RESULTS / "ours_loop" / self._order(task, off) / f"{task}_{SEEDS[task] + off}"

    def unit(self, task: str, off: int) -> dict | None:
        """The (task, world) so far: its played episodes, whether the task succeeded in one of them, and every
        compute measure over them (for Ours, also the loop's episodes and the Designer's calls); None before its
        first episode finished."""
        res = self._result(task, off)
        r = json.loads(res.read_text()) if res.exists() else {}
        if r.get("status") not in DONE:
            return None
        rs, success = [r], bool(r.get("success"))
        m = dict.fromkeys(KEYS, 0)
        d = self._loop_dir(task, off)
        if d.exists():  # a loop episode from a save counts the steps it played (vlm/summary.json), not the save's
            for f in sorted(d.glob("e*/result.json")):
                if ".crash" not in f.parent.name:
                    x, sm = json.loads(f.read_text()), f.parent / "vlm" / "summary.json"
                    rs.append({**x, "steps": json.loads(sm.read_text())["steps"]} if sm.exists() else x)
            st = json.loads((d / "loop.json").read_text()) if (d / "loop.json").exists() else {}
            success = success or str(st.get("done") or "").startswith("task succeeded")
            calls = d / "llm" / "calls.jsonl"
            for c in (json.loads(l) for l in open(calls)) if calls.exists() else []:
                m["llm_calls"] += 1
                m["tokens_in"] += c.get("tokens_in") or 0
                m["tokens_out_billed"] += c.get("tokens_out_billed") or 0
                m["cost_usd"] += c.get("cost_usd") or 0
                m["wall_time_s"] += c.get("latency_s") or 0
        for x in rs:
            for k in KEYS:
                m[k] += x.get(k) or 0
        return {"episodes": len(rs), "success": success, **m}

    def measures(self) -> dict:
        """Every compute measure per level, summed over the run's (task, world) so far."""
        out = {}
        for L in self.levels:
            m = out[L["name"]] = dict.fromkeys(("units", "successes", "episodes") + KEYS, 0)
            for t in L["train"]:
                for off in self.offsets:
                    u = self.unit(t, off)
                    if u:
                        m["units"] += 1
                        m["successes"] += u["success"]
                        for k in ("episodes",) + KEYS:
                            m[k] += u[k]
        out["total"] = {k: sum(m[k] for m in out.values()) for k in out[self.levels[0]["name"]]}
        return out

    def _write_state(self) -> None:
        self.state_f.parent.mkdir(parents=True, exist_ok=True)
        self.state_f.write_text(json.dumps({
            "condition": self.cond.name, "store": str(self.store), "seed_offsets": self.offsets,
            "episodes_done": len(self.done), "episodes_total": sum(len(L["train"]) for L in self.levels) * len(self.offsets),
            "measures": self.measures(), "ledger": self.ledger,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    def _snapshot(self, label: str, r: dict | None) -> None:  # call with self.lock held
        """Copy the store as it is after this episode; frames are hard-linked (they never change once written)."""
        dst = self.snaps / f"{len(self.ledger) + 1:05d}_{label}"
        if self.store.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            is_dir = self.store.is_dir()  # optimus1_memory: a directory; mineevolve, ours: one file
            lock = self.store / "store.lock" if is_dir else self.store.with_suffix(".lock") \
                if self.cond.method == "ours" else Path(str(self.store) + ".lock")  # the store's own lock
            with open(lock, "a") as lf:
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
        task, off = label.rsplit("-s", 1)
        u = self.unit(task, int(off)) or {}
        self.ledger.append({"n": len(self.ledger) + 1, "episode": label, "status": (r or {}).get("status"),
                            "success": u.get("success"), "episodes": u.get("episodes"),
                            "snapshot": str(dst) if self.store.exists() else None, "after": tot})

    # --------------------------------------------------------------- episode
    def _episode(self, task: str, off: int, gpu: int, port: int) -> dict:
        order = self._order(task, off)
        cmd = [PY, str(REPRO / "experiments" / "chain.py"), "--method", self.cond.method, "--world", self.cond.world,
               STORES[self.cond.method][0], self.a.store, "--order", order, "--tasks", task, "--seed-offset", str(off),
               "--gpu", str(gpu), "--port", str(port)]
        if self.cond.save_every:
            cmd += ["--save", str(self.cond.save_every)]
        if self.cond.plan:
            cmd.append("--plan")
        if self.a.mock:
            cmd.append("--mock")
        log(f"[gpu {gpu}] {task} s{off} start")
        out = RUN_DATA / "logs" / f"phase1_{self.a.store}" / f"{order}.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        res = self._result(task, off)
        r = json.loads(res.read_text()) if res.exists() else {"status": "missing", "chain_rc": rc}
        log(f"[gpu {gpu}] {task} s{off} -> {r.get('status')} success={r.get('success')} steps={r.get('steps')} "
            f"cost={r.get('cost_usd')}")
        if self.cond.method == "ours" and r.get("status") in DONE:
            with open(self.pool, "a") as f:
                f.write(f"{res.parent}\n")
            if not r.get("success") and r.get("status") == "finished":
                self._loop(task, off, res.parent, gpu, port)
        return r

    def _loop(self, task: str, off: int, ep: Path, gpu: int, port: int) -> None:
        """Ours: the learning-from-failure loop on the failed episode, with this run's World Knowledge store."""
        order = self._order(task, off)
        eps = RUN_DATA / "state" / f"phase1_{self.a.store}" / f"{order}.txt"
        eps.parent.mkdir(parents=True, exist_ok=True)
        eps.write_text(f"{ep}\n")
        cmd = [PY, str(REPRO / "experiments" / "ours_loop.py"), "--name", order, "--episodes", str(eps),
               "--world", self.cond.world, "--store", self.a.store, "--gpu", str(gpu), "--port", str(port),
               "--success-pool", str(self.pool)]
        if self.cond.plan:
            cmd.append("--plan")
        if self.a.mock:
            cmd.append("--mock")
        log(f"[gpu {gpu}] {task} s{off} loop start")
        with open(RUN_DATA / "logs" / f"phase1_{self.a.store}" / f"{order}.loop.log", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        st = self._loop_dir(task, off) / "loop.json"
        done = json.loads(st.read_text()).get("done") if st.exists() else None
        u = self.unit(task, off) or {}
        log(f"[gpu {gpu}] {task} s{off} loop -> {done or f'exit {rc}'}; {u.get('episodes')} episodes, "
            f"{u.get('steps')} steps, ${u.get('cost_usd', 0):.2f}")

    def _worker(self, queue: list, gpu: int, port: int) -> None:
        while True:
            with self.lock:
                if self.stop or not queue:
                    return
                task, off = queue.pop(0)
            r = self._episode(task, off, gpu, port)
            with self.lock:
                if r.get("status") in DONE:
                    self.done.add((task, off))
                elif r.get("status") == "missing":  # chain.py ran no episode: disk, LLM cap, STOP_BUDGET or an error
                    log(f"STOP: chain.py ran no episode (exit {r.get('chain_rc')}); see run_data/logs/phase1_{self.a.store}/")
                    self.stop = True
                self._snapshot(f"{task}-s{off}", r)
                self._write_state()

    # ------------------------------------------------------------------ main
    def run(self) -> int:
        self.stop = False
        gpus = [int(g) for g in self.a.gpus.split(",")]
        log(f"{self.cond.name}: store {self.store}, worlds (seed offsets) {self.offsets}, {len(gpus)} workers")
        for L in self.levels:
            name = L["name"]
            if self.a.levels and name not in self.a.levels.split(","):
                continue
            queue = [(t, off) for off in self.offsets for t in L["train"] if (t, off) not in self.done]
            if not queue:
                log(f"level {name}: done earlier")
                continue
            log(f"level {name}: {len(queue)} episodes to play ({len(L['train'])} train tasks x {len(self.offsets)} worlds)")
            ws = [threading.Thread(target=self._worker, args=(queue, g, self.a.port + 10 * i))
                  for i, g in enumerate(gpus[:len(queue)])]
            for w in ws:
                w.start()
            for w in ws:
                w.join()
            m = self.measures()[name]
            log(f"level {name}: {m['units']} (task, world), {m['successes']} successes, {m['episodes']} episodes, "
                f"{m['steps']} steps, ${m['cost_usd']:.2f}")
            if self.stop:
                return 2
        with self.lock:
            self._write_state()
        t = self.measures()["total"]
        log(f"done: {t['units']} (task, world), {t['successes']} successes, {t['episodes']} episodes, {t['steps']} steps, "
            f"${t['cost_usd']:.2f}")
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
    if a.method not in STORES or not (a.store or a.o1m or a.knowledge) or a.frozen:
        ap.error(f"phase1 needs a method with a memory store ({', '.join(STORES)}), its store name and no --frozen")
    if a.method == "ours" and not a.save:
        ap.error("ours needs --save (the loop's restart points)")
    a.store = a.store or a.o1m or a.knowledge  # the run's store name, also used for the orders and the state file
    cond = Condition(a.method, a.world, a.save, a.knowledge, a.plan,
                     a.store if a.method == "mineevolve" else None, a.o1m)
    if a.report:
        print(json.dumps(Phase1(a, cond).measures(), indent=1))
        return 0
    return Phase1(a, cond).run()


if __name__ == "__main__":
    sys.exit(main())
