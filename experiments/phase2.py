"""Phase 2 (plan 2026-11-11): play the test tasks of configs/suites/optimus1/phase_split.yaml, each once in every world of
phase1.seed_offsets (configs/hyperparams.yaml), with a method's Phase 1 memory frozen.

The memory is the Phase 1 snapshot after episode --cut of the ledger (run_data/state/phase1_<store>.json; default: the
last one), copied to the store <store>_c<cut> so a cut is a store of its own and Phase 1's store is never read live.

    python experiments/phase2.py --method baseline --world nolava --plan --gpus 0,1,2,3 --port 29300
    python experiments/phase2.py --method mineevolve --store p1me --world nolava --plan --gpus 0,1,2,3 --port 29400
    python experiments/phase2.py --method optimus1_memory --o1m p1om --world nolava --plan --gpus 0,1,2,3 --port 29500 [--cut 60]

Transfer to unseen levels: --cut at the last episode of a level (the levels ran in order) and --levels the later ones
with --with-train, so their train tasks (never played with that memory) count as unseen too.

Each episode is one chain.py call (--order p2-<run>-<task>-s<offset>), so results are
results/<condition>/p2-<run>-<task>-s<offset>/<task>/<seed>/ and a restarted run skips the finished ones.
Progress: run_data/state/phase2_<run>.json.
"""
from __future__ import annotations

import argparse
import json
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
from phase1 import DONE, PY, SPLIT, STORES, log  # noqa: E402

KEYS = ("steps", "llm_calls", "tokens_in", "tokens_out_billed", "cost_usd", "wall_time_s")


class Phase2:
    def __init__(self, a, cond: Condition):
        import yaml
        self.a, self.cond = a, cond
        self.levels = yaml.safe_load(SPLIT.read_text())["levels"]
        self.offsets = [int(o) for o in HP["phase1"]["seed_offsets"]]
        self.state_f = RUN_DATA / "state" / f"phase2_{a.run}{'_train' if a.train_only else ''}.json"
        self.prefix = f"p2-{a.run}-"
        self.lock = threading.Lock()
        self.stop = False

    def _order(self, task: str, off: int) -> str:
        return f"{self.prefix}{task}-s{off}"

    def _result(self, task: str, off: int) -> Path:
        return RESULTS / self.cond.name / self._order(task, off) / task / str(SEEDS[task] + off) / "result.json"

    def _finished(self, task: str, off: int) -> bool:
        res = self._result(task, off)
        return res.exists() and json.loads(res.read_text()).get("status") in DONE

    def tasks(self, L: dict) -> list:
        return ([] if self.a.train_only else L["test"]) + (L["train"] if self.a.with_train or self.a.train_only else [])

    def measures(self) -> dict:
        """Episodes, successes and every compute measure per level, over the run's finished episodes."""
        out = {}
        for L in self.levels:
            m = out[L["name"]] = dict.fromkeys(("episodes", "successes") + KEYS, 0)
            for t in self.tasks(L):
                for off in self.offsets:
                    if self._finished(t, off):
                        r = json.loads(self._result(t, off).read_text())
                        m["episodes"] += 1
                        m["successes"] += bool(r.get("success"))
                        for k in KEYS:
                            m[k] += r.get(k) or 0
        out["total"] = {k: sum(m[k] for m in out.values()) for k in out[self.levels[0]["name"]]}
        return out

    def _write_state(self) -> None:
        self.state_f.parent.mkdir(parents=True, exist_ok=True)
        self.state_f.write_text(json.dumps({
            "condition": self.cond.name, "memory": self.a.memory, "seed_offsets": self.offsets,
            "measures": self.measures(), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    def _episode(self, task: str, off: int, gpu: int, port: int) -> dict:
        order = self._order(task, off)
        cmd = [PY, str(REPRO / "experiments" / "chain.py"), "--method", self.cond.method, "--world", self.cond.world,
               "--order", order, "--tasks", task, "--seed-offset", str(off), "--gpu", str(gpu), "--port", str(port)]
        if self.cond.method in STORES:
            cmd += [STORES[self.cond.method][0], self.a.cut_store, "--frozen"]
        if self.cond.plan:
            cmd.append("--plan")
        if self.a.mock:
            cmd.append("--mock")
        log(f"[gpu {gpu}] {task} s{off} start")
        out = RUN_DATA / "logs" / f"phase2_{self.a.run}" / f"{order}.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        res = self._result(task, off)
        r = json.loads(res.read_text()) if res.exists() else {"status": "missing", "chain_rc": rc}
        log(f"[gpu {gpu}] {task} s{off} -> {r.get('status')} success={r.get('success')} steps={r.get('steps')} "
            f"cost={r.get('cost_usd')}")
        return r

    def _worker(self, queue: list, gpu: int, port: int) -> None:
        while True:
            with self.lock:
                if self.stop or not queue:
                    return
                task, off = queue.pop(0)
            r = self._episode(task, off, gpu, port)
            with self.lock:
                if r.get("status") == "missing":  # chain.py ran no episode: disk, LLM cap, STOP_BUDGET or an error
                    log(f"STOP: chain.py ran no episode (exit {r.get('chain_rc')}); see run_data/logs/phase2_{self.a.run}/")
                    self.stop = True
                self._write_state()

    def run(self) -> int:
        gpus = [int(g) for g in self.a.gpus.split(",")]
        queue = [(t, off) for L in self.levels if not self.a.levels or L["name"] in self.a.levels.split(",")
                 for off in self.offsets for t in self.tasks(L) if not self._finished(t, off)]
        log(f"{self.cond.name}: memory {self.a.memory}, {len(queue)} episodes to play, {len(gpus)} workers")
        ws = [threading.Thread(target=self._worker, args=(queue, g, self.a.port + 10 * i))
              for i, g in enumerate(gpus[:len(queue)])]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
        self._write_state()
        for name, m in self.measures().items():
            log(f"{name}: {m['successes']}/{m['episodes']} successes, {m['steps']} steps, ${m['cost_usd']:.2f}")
        return 2 if self.stop else 0


def cut_store(a) -> tuple[str, str]:
    """Copy the Phase 1 snapshot after ledger episode a.cut to the store <store>_c<cut>; returns (store name, snapshot)."""
    ledger = json.loads((RUN_DATA / "state" / f"phase1_{a.store}.json").read_text())["ledger"]
    e = ledger[(a.cut or len(ledger)) - 1]
    name = f"{a.store}_c{e['n']}"
    snap = Path(e["snapshot"])
    if a.method == "optimus1_memory":
        dst = RUN_DATA / "memory" / f"o1m_{name}"
        if not dst.exists():
            shutil.copytree(snap, dst)
            (dst / "store.lock").touch()
    else:
        dst = RUN_DATA / "memory" / f"me_{name}.json"
        if not dst.exists():
            shutil.copy2(next(snap.glob("*.json")), dst)
    return name, str(snap)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--gpus", required=True, help="comma list; one episode per GPU at a time")
    ap.add_argument("--port", type=int, required=True, help="first port; worker i uses port + 10 i (and + 1000)")
    ap.add_argument("--cut", type=int, default=0, help="Phase 1 ledger episode whose snapshot is the memory (default: last)")
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls")
    ap.add_argument("--levels", default="", help="comma list of levels to play; default: all")
    ap.add_argument("--with-train", action="store_true", help="also the levels' train tasks (levels the memory never saw)")
    ap.add_argument("--train-only", action="store_true", help="only the levels' train tasks (the test tasks played elsewhere)")
    ap.add_argument("--report", action="store_true", help="print the run's measures per level and exit")
    a = ap.parse_args()
    if a.frozen:
        ap.error("phase2 freezes the memory itself; no --frozen")
    a.store = a.store or a.o1m
    if a.method in STORES:
        if not a.store:
            ap.error(f"method {a.method} needs its Phase 1 store name")
        a.cut_store, a.memory = cut_store(a)
        a.run = a.cut_store + ("_mock" if a.mock else "")
        cond = Condition(a.method, a.world, a.save, a.knowledge, a.plan,
                         a.cut_store if a.method == "mineevolve" else None,
                         a.cut_store if a.method == "optimus1_memory" else None, True)
    else:
        a.memory, a.run = None, a.method + ("_mock" if a.mock else "")
        cond = Condition(a.method, a.world, a.save, a.knowledge, a.plan)
    p = Phase2(a, cond)
    if a.report:
        print(json.dumps(p.measures(), indent=1))
        return 0
    return p.run()


if __name__ == "__main__":
    sys.exit(main())
