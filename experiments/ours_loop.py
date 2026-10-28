"""Ours (EXPERIMENTS.md 6): the learning-from-failure loop.

For each failed episode (run with saves, `experiments/run.py --method baseline --world nolava --save 600`), one loop:
  1. System 2 finds the failure point in the compressed log (system2/failure_point.py, system2/episode_log.py).
  2. System 2 proposes one hypothesis there: change, prediction, plan, statement (system2/hypothesis.py).
  3. Replay world: an episode starts from the last save at or before the failure point (world/replay.py);
     System 1 (system1/vlm_agent.py) gets the hypothesis and the knowledge verified so far in this loop, and plays
     on with the same task, horizon and world rules.
  4. System 2 judges the hypothesis from the experiment's log (system2/judge.py): Supported -> its statement goes
     to World Knowledge (memory/world_knowledge.py, <runs>/memory/wk_<store>.json); Refuted / Not tried -> kept at
     the failure point for the next proposal.
  5. Task check (the episode's success monitor): success -> loop ends. Otherwise Refuted -> new hypothesis at the
     same failure point; Supported -> next failure point, found in the experiment's log.
  Ends on success or after --max-iter experiments.
Nothing from client.log (ore placements) reaches System 2.

    python experiments/ours_loop.py --name pl1 --episodes eps.txt --world nolava --save 600 \
        --gpu 0 --port 28300 --store pl1
eps.txt: one failed episode dir per line (from results/baseline/nolava_save600/). Output:
<runs>/results/ours_loop/<name>/<task>_<seed>/ (loop.json, system2.jsonl, it<k>/ experiment episodes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
import paths  # noqa: E402,F401

import episode_log as ce  # noqa: E402
import failure_point  # noqa: E402
import hypothesis  # noqa: E402
import judge  # noqa: E402
import world_knowledge  # noqa: E402
from benchmark import RESULTS, RUN_DATA, SEEDS, write_ctx  # noqa: E402

RAW_BEFORE, RAW_AFTER = 8, 3  # raw turns given around the failure point


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def turn_rows(ep: Path) -> list[dict]:
    return [json.loads(l) for l in open(ep / "vlm" / "turns.jsonl")]


def checkpoint_for(ep: Path, turn: int) -> tuple[Path, int] | None:
    """Last save of `ep` taken at or before the start of `turn` (saves are taken at turn ends)."""
    rows = {t["turn"]: t for t in turn_rows(ep)}
    if turn not in rows:
        return None
    ok = [s for s in ce.checkpoints(ep) if s <= rows[turn]["t0"]]
    return (ep / "ckpt" / f"s{max(ok):06d}.zip", max(ok)) if ok else None


def state_at(ep: Path, turn: int) -> dict:
    res = json.loads((ep / "result.json").read_text())
    ts, ps, _ = ce._positions(ep)
    rows = {t["turn"]: t for t in turn_rows(ep)}
    t0 = rows[turn]["t0"]
    inv = {}
    import gzip
    for line in gzip.open(ep / "trajectory.jsonl.gz", "rt"):
        d = json.loads(line)
        if d["t"] > t0:
            break
        inv = d.get("inv", inv)
    return {"task": res.get("instruction"), "turn": turn, "step": t0, "of_steps": res.get("horizon_steps"),
            "height_y": round(ce._pos_at(ts, ps, t0)[1]), "inventory": inv}


class Loop:
    def __init__(self, a, ep: Path):
        self.a, self.src = a, Path(ep)
        res = json.loads((self.src / "result.json").read_text())
        self.task, self.seed = res["task"], int(res["seed"])
        self.dir = RESULTS / "ours_loop" / a.name / f"{self.task}_{self.seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_f = self.dir / "loop.json"
        self.st = json.loads(self.state_f.read_text()) if self.state_f.exists() else {
            "source": str(self.src), "task": self.task, "seed": self.seed, "iterations": [], "failure_points": [],
            "knowledge": [], "done": None}
        self.events = []

    def save(self):
        self.state_f.write_text(json.dumps(self.st, indent=1, ensure_ascii=False))
        with open(self.dir / "system2.jsonl", "a") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.events = []

    def chain(self):
        from chain import Chain
        ns = argparse.Namespace(method="baseline", world=self.a.world, save=self.a.save, knowledge=None, order=f"loop_{self.a.name}",
                                gpu=self.a.gpu, port=self.a.port, retries=2, stall_min=60.0, mock=self.a.mock,
                                seed_offset=self.seed - SEEDS[self.task])
        return Chain(ns)

    def run_experiment(self, k: int, ckpt: Path, hyp: dict) -> Path:
        ch = self.chain()
        ep = self.dir / f"it{k}"
        exp_f = self.dir / f"it{k}_experiment.json"
        exp_f.write_text(json.dumps({"hypothesis": hyp, "knowledge": self.st["knowledge"]}, indent=1, ensure_ascii=False))
        ch.env_vars.update(OPTIMUS_LOAD_CKPT=str(ckpt), OPTIMUS_EXPERIMENT=str(exp_f))
        for attempt in range(3):
            if ep.exists():
                ep.rename(ep.with_name(f"{ep.name}.crash{int(time.time())}"))
            ep.mkdir(parents=True)
            try:
                res = ch.run_episode(self.task, ep)
            except Exception as e:
                res = {"status": "crashed", "error": repr(e)}
            log(f"  it{k} attempt {attempt + 1}: {res.get('status')} success={res.get('success')} steps={res.get('steps')}")
            if res.get("status") in ("finished", "anomaly"):
                return ep
        raise RuntimeError(f"experiment it{k} crashed 3 times")

    def run(self):
        if self.st["done"]:
            return
        cur = Path(self.st["iterations"][-1]["episode"]) if self.st["iterations"] and \
            self.st["iterations"][-1]["verdict"] == "Supported" else self.src
        fp = self.st["failure_points"][-1] if self.st["failure_points"] and not self.st["failure_points"][-1].get("closed") else None
        for k in range(len(self.st["iterations"]) + 1, self.a.max_iter + 1):
            if fp is None:  # 2. find a failure point in the latest log
                earlier = [{"decision": f["decision"], "reason": f["reason"]} for f in self.st["failure_points"]]
                found = failure_point.find(ce.compress(cur), earlier, self.events)
                ck = checkpoint_for(cur, found["turn"]) if found else None
                if found and not ck and cur != self.src:  # before the experiment's first save: the checkpoint it started from
                    start = self.st["iterations"][-1]
                    if any(t["turn"] == found["turn"] for t in turn_rows(cur)):
                        ck = (Path(start["checkpoint"]), start["checkpoint_step"])
                if not ck:
                    self.st["done"] = "no usable failure point" + ("" if found else " (unparsable answer)")
                    break
                lo = found["turn"] - RAW_BEFORE
                fp = {**found, "episode": str(cur), "checkpoint": str(ck[0]), "checkpoint_step": ck[1],
                      "state": state_at(cur, found["turn"]),
                      "raw": ce.raw_window(cur, lo, found["turn"] + RAW_AFTER), "refuted": []}
                self.st["failure_points"].append(fp)
                self.save()
                log(f"  failure point T{fp['turn']} (checkpoint step {ck[1]}): {fp['decision'][:120]}")
            hyp = hypothesis.propose({k_: fp[k_] for k_ in ("turn", "decision", "reason")}, fp["raw"], fp["state"],
                                  fp["refuted"], self.events)  # 3.
            if not hyp:
                self.st["done"] = "proposal unparsable"
                break
            log(f"  it{k} hypothesis: {hyp['change'][:120]}")
            ep = self.run_experiment(k, Path(fp["checkpoint"]), hyp)  # 4.
            res = json.loads((ep / "result.json").read_text())
            v = judge.judge(hyp, ce.compress(ep), self.events) or {"verdict": "Not tried", "observed": "",
                                                                     "reason": "judge answer unparsable"}  # 5.
            it = {"k": k, "failure_point_turn": fp["turn"], "checkpoint": fp["checkpoint"],
                  "checkpoint_step": fp["checkpoint_step"], "hypothesis": hyp, **v, "episode": str(ep),
                  "task_success": bool(res.get("success")), "steps": res.get("steps")}
            self.st["iterations"].append(it)
            log(f"  it{k} {v['verdict']}; task {'succeeded' if it['task_success'] else 'failed'}")
            if v["verdict"] == "Supported":
                kn = {"statement": hyp["statement"], "source": f"{self.task}_{self.seed}/it{k}",
                      "prediction": hyp["prediction"], "observed": v["observed"]}
                self.st["knowledge"].append(kn)
                world_knowledge.add(RUN_DATA / "memory" / f"wk_{self.a.store}.json", kn)
                fp["closed"] = True
            else:
                fp["refuted"].append({"change": hyp["change"], "prediction": hyp["prediction"], "verdict": v["verdict"],
                                      "observed": v["observed"], "reason": v["reason"]})
            self.save()
            if it["task_success"]:  # 6.
                self.st["done"] = f"task succeeded at it{k}"
                break
            if v["verdict"] == "Supported":
                cur, fp = ep, None
        self.st["done"] = self.st["done"] or f"max {self.a.max_iter} experiments"
        self.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", required=True, help="file: one failed episode dir per line, in loop order")
    ap.add_argument("--world", default="nolava", help="world of the experiment episodes (the source episodes' world)")
    ap.add_argument("--save", type=int, default=600, help="world saves in the experiment episodes too (the next failure point)")
    ap.add_argument("--store", required=True, help="World Knowledge store memory/wk_<store>.json")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls (System 2 and the agent)")
    a = ap.parse_args()
    if a.mock:
        os.environ["LLM_MOCK"] = "1"
    from conditions import Condition
    os.environ.setdefault("LLM_CONFIG", Condition("baseline", a.world, a.save).env_vars(RUN_DATA, REPRO)["LLM_CONFIG"])
    from gemini_client import CFG
    ctx = RUN_DATA / "state" / "ours_loop" / a.name / "ctx.json"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_CTX_FILE"] = str(ctx)
    for line in Path(a.episodes).read_text().split():
        loop = Loop(a, Path(line))
        write_ctx(ctx, {"env": "O", "method": "ours_system2", "chain": a.name, "order_id": "system2",
                        "task": loop.task, "seed": loop.seed, "episode_dir": str(loop.dir),
                        "run_id": f"ours_loop/{a.name}/{loop.task}_{loop.seed}"})
        stop_budget = Path(CFG["log_root"]) / "STOP_BUDGET"
        if stop_budget.exists():
            log("STOP: STOP_BUDGET present")
            return 2
        log(f"loop {loop.task} seed {loop.seed} from {line}")
        loop.run()
        log(f"  -> {loop.st['done']}; knowledge +{len(loop.st['knowledge'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
