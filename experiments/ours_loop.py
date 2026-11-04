"""Ours (EXPERIMENTS.md 6): the learning-from-failure loop.

For each failed episode (run with saves, `experiments/run.py --method baseline --world nolava --save 600`), one loop:
  1. The Designer lists the failure points in the compressed log (designer/failure_point.py; the log: the domain's compress());
     the loop takes them earliest first. After the first experiment the log is the whole attempt from step 0 (the
     source episode up to the save each experiment was replayed from, then that experiment, and so on), and the Designer
     also gets every experiment so far (failure point, change, verdict, observed); a failure point in an earlier part
     restarts from a save of that part.
  2. The Designer proposes one hypothesis at the failure point, seeing the other failure points of the log: the problem
     the decision caused, the belief about avoiding it, the change that tests it, what shows the change was made, the
     moment it is checked at and the prediction then, plan (designer/hypothesis.py).
  3. Replay world: an episode starts from the last save at or before the failure point, or from the beginning of
     the same world (same seed) when the failure point comes before the first save; The Executor gets the hypothesis
     and the knowledge verified so far in this loop (executor/extras.py), and plays on with the same task, horizon
     and world rules.
  4. The Designer judges the hypothesis from the experiment's log (designer/judge.py): Supported -> its statement (problem
     and belief) goes to World Knowledge (memory/world_knowledge.py, <runs>/memory/wk_<store>.json); Refuted / Not
     reached -> kept at the failure point, and the next proposal there sees it with what its experiment showed; it
     also goes to the failed-hypothesis memory (memory/failed_hypotheses.py, <runs>/memory/fh_<store>.json), which
     every later proposal gets (other failure points, other episodes of the store).
  5. Task check (the episode's success monitor): success -> loop ends. Otherwise Refuted / Not reached
     -> new proposal at the same failure point, or after --tries experiments there the next failure point of the same
     log; Supported -> the failure points are listed again, in the experiment's log. When the candidates of a log are
     used up, the failure points are listed again in the latest experiment's log.
  Ends on success or after --max-iter experiments (0: no limit); also when the latest log has no failure point.
Everything about the environment goes through the domain (domains/base.py; --domain, default minecraft): reading
episodes, saves and states, and running the experiment episodes. --world and --save are Minecraft's run options.

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

import domains  # noqa: E402
import failed_hypotheses  # noqa: E402
import failure_point  # noqa: E402
import hypothesis  # noqa: E402
import judge  # noqa: E402
import world_knowledge  # noqa: E402
from llm_context import write_ctx  # noqa: E402
from paths import RESULTS, RUN_DATA  # noqa: E402

FAILED_MAX = 30  # failed hypotheses of the store (latest first) a proposal gets
RAW_BEFORE, RAW_AFTER = 8, 3  # raw turns given before (or from the restart point, if earlier) and after the failure point


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


class Loop:
    def __init__(self, a, dom, ep: Path):
        self.a, self.dom, self.src = a, dom, Path(ep).resolve()  # episodes run in their own working directory
        res = dom.result(self.src)
        self.task, self.seed = res["task"], int(res["seed"])
        self.dir = RESULTS / "ours_loop" / a.name / f"{self.task}_{self.seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_f = self.dir / "loop.json"
        self.fh_store = RUN_DATA / "memory" / f"fh_{a.store}.json"  # failed hypotheses, shared by the loops of the store
        self.st = json.loads(self.state_f.read_text()) if self.state_f.exists() else {
            "source": str(self.src), "task": self.task, "seed": self.seed, "iterations": [], "failure_points": [],
            "knowledge": [], "done": None,
            "settings": {"max_iter": a.max_iter, "tries": a.tries}}  # read by analysis/loop_ui/build.py
        self.st.setdefault("log", str(self.src))
        self.events = []

    def save(self):
        self.state_f.write_text(json.dumps(self.st, indent=1, ensure_ascii=False))
        with open(self.dir / "system2.jsonl", "a") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.events = []

    def run_experiment(self, k: int, ckpt: str, hyp: dict) -> Path:
        ep = self.dir / f"it{k}"
        exp_f = self.dir / f"it{k}_experiment.json"
        exp_f.write_text(json.dumps({"hypothesis": hyp, "knowledge": self.st["knowledge"], "from_start": not ckpt},
                                    indent=1, ensure_ascii=False))
        opts = {"name": self.a.name, "world": self.a.world, "save": self.a.save, "gpu": self.a.gpu, "port": self.a.port,
                "mock": self.a.mock}
        for attempt in range(3):
            if ep.exists():
                ep.rename(ep.with_name(f"{ep.name}.crash{int(time.time())}"))
            ep.mkdir(parents=True)
            try:
                res = self.dom.run_episode(self.task, self.seed, ep, ckpt, exp_f, opts)
            except Exception as e:
                res = {"status": "crashed", "error": repr(e)}
            log(f"  it{k} attempt {attempt + 1}: {res.get('status')} success={res.get('success')} steps={res.get('steps')}")
            if res.get("status") in ("finished", "anomaly"):
                return ep
        raise RuntimeError(f"experiment it{k} crashed 3 times")

    def next_point(self) -> dict | None:
        """The next failure point to experiment at: the earliest candidate left from the latest log (the source
        episode, or the experiment of the last Supported hypothesis). Candidates are found once per log; when they
        are used up, the failure points are listed again in the latest experiment's log. A log is read as the whole
        attempt from step 0 (parts()), so a failure point can be in an earlier part than the latest experiment."""
        while True:
            fp = self._point_in_log()
            last = self.st["iterations"][-1]["episode"] if self.st["iterations"] else None
            if fp or not last or last == self.st["log"]:
                return fp
            self.st["log"] = last  # candidates used up: list the failure points again, in the latest experiment's log

    def _fp_of(self, it: dict) -> dict:
        return next(f for f in self.st["failure_points"]
                    if f["turn"] == it["failure_point_turn"] and f["checkpoint_step"] == it["checkpoint_step"])

    def parts(self, ep) -> list:
        """The attempt that ends in episode ep, from step 0: [(episode, until_step, note)], each experiment replayed
        from a save of the episode its failure point was in (domain compress(ep, parts))."""
        if Path(ep) == self.src:
            return [(self.src, None, None)]
        it = next(i for i in self.st["iterations"] if i["episode"] == str(ep))
        ck = it["checkpoint_step"]
        turns = self.dom.turns(ep)
        first = f"turn T{turns[0]['turn']}" if turns else "no turns"
        note = (f"Step {ck} ({first}): replayed from a save; the turns below are from experiment {it['k']}, "
                f"which tried: {it['hypothesis']['change']}")
        if not ck:  # replayed from the beginning of the same world: nothing kept from before
            return [(Path(ep), None, note)]
        head = self.parts(it.get("from_episode") or self._fp_of(it)["episode"])
        head[-1] = (head[-1][0], ck, head[-1][2])
        return head + [(Path(ep), None, note)]

    def _point_in_log(self) -> dict | None:
        cur = Path(self.st["log"])
        parts = self.parts(cur)
        if self.st.get("searched") != str(cur):  # 2. find the failure points in the latest log, from step 0
            earlier = [{"failure_point": f"T{it['failure_point_turn']}: {self._fp_of(it)['decision']}",
                        "change": it["hypothesis"]["change"], "verdict": it["verdict"], "observed": it["observed"]}
                       for it in self.st["iterations"]]
            found = failure_point.find(self.dom.compress(cur, parts), earlier, self.events, self.dom.vocabulary)
            self.st.update(searched=str(cur), found=found or [], candidates=list(found or []), unparsable=found is None)
            log(f"  failure point candidates in {cur.name}: " + (", ".join(f"T{f['turn']}" for f in found) if found
                                                                 else "none" if found == [] else "unparsable answer"))
        while self.st["candidates"]:
            found = self.st["candidates"].pop(0)
            where = None  # the part (episode, its turns kept) the turn is in
            for e, until, _ in parts:
                ts = [t for t in self.dom.turns(e) if until is None or t["t0"] < until]
                if any(t["turn"] == found["turn"] for t in ts):
                    where = (e, ts)
            if not where:
                log(f"  T{found['turn']} skipped: not a turn of the log")
                continue
            e, ts = where
            if any(f["episode"] == str(e) and f["turn"] == found["turn"] for f in self.st["failure_points"]):
                log(f"  T{found['turn']} skipped: tried before")
                continue
            ck = self.dom.save_before(e, found["turn"])
            if ck == ("", 0) and e != self.src:  # before the experiment's first save: the checkpoint it started from
                start = next(i for i in reversed(self.st["iterations"]) if i["episode"] == str(e))
                ck = (start["checkpoint"], start["checkpoint_step"])
            restart = next(t["turn"] for t in ts if t["t0"] >= ck[1])  # first turn after the save
            lo = min(restart, found["turn"] - RAW_BEFORE)
            fp = {**found, "episode": str(e), "checkpoint": str(ck[0]), "checkpoint_step": ck[1],
                  "others": [f for f in self.st["found"] if f["turn"] != found["turn"]],
                  "state": self.dom.state_at(e, restart),
                  "raw": self.dom.raw_window(e, lo, min(found["turn"] + RAW_AFTER, ts[-1]["turn"])), "refuted": []}
            self.st["failure_points"].append(fp)
            self.save()
            log(f"  failure point T{fp['turn']} in {e.name} (checkpoint step {ck[1]}): {fp['decision'][:120]}")
            return fp
        return None

    def run(self):
        if self.st["done"]:
            return
        k = len(self.st["iterations"])
        while not self.a.max_iter or k < self.a.max_iter:
            k += 1
            fp = self.st["failure_points"][-1] if self.st["failure_points"] and \
                not self.st["failure_points"][-1].get("closed") else self.next_point()
            if fp is None:
                self.st["done"] = "no failure point left" + (" (unparsable answer)" if self.st.get("unparsable") else "")
                break
            here = {r.get("source") for r in fp["refuted"]}  # shown as tried at this failure point already
            failed = [f for f in failed_hypotheses.load(self.fh_store) if f["source"] not in here][-FAILED_MAX:]
            hyp = hypothesis.propose({k_: fp.get(k_, "") for k_ in ("turn", "decision", "known_then", "better", "reason")}, fp.get("others", []),
                                     fp["raw"], fp["state"], fp["refuted"], self.events, self.dom.vocabulary,
                                     failed=failed)  # 3.
            if not hyp:
                self.st["done"] = "proposal unparsable"
                break
            log(f"  it{k} hypothesis: {hyp['change'][:120]}")
            ep = self.run_experiment(k, fp["checkpoint"], hyp)  # 4.
            res = self.dom.result(ep)
            v = judge.judge(hyp, self.dom.compress(ep), self.events, self.dom.vocabulary) or {"verdict": "Unjudged", "observed": "",
                                                                     "reason": "judge answer unparsable"}  # 5.
            it = {"k": k, "failure_point_turn": fp["turn"], "from_episode": fp["episode"], "checkpoint": fp["checkpoint"],
                  "checkpoint_step": fp["checkpoint_step"], "hypothesis": hyp, **v, "episode": str(ep),
                  "task_success": bool(res.get("success")), "steps": res.get("steps")}
            self.st["iterations"].append(it)
            log(f"  it{k} {v['verdict']}; task {'succeeded' if it['task_success'] else 'failed'}")
            if v["verdict"] == "Supported":
                kn = {"statement": hyp["statement"], "source": f"{self.task}_{self.seed}/it{k}",
                      "prediction": hyp["prediction"], "observed": v["observed"]}
                self.st["knowledge"].append(kn)
                world_knowledge.add(RUN_DATA / "memory" / f"wk_{self.a.store}.json", kn)
                fp["closed"] = "Supported"
                self.st["log"] = str(ep)  # the next failure points come from this experiment's log
            else:
                failed_h = {**hyp, "verdict": v["verdict"], "observed": v["observed"], "reason": v["reason"],
                            "source": f"{self.task}_{self.seed}/it{k}"}
                fp["refuted"].append(failed_h)
                failed_hypotheses.add(self.fh_store, failed_h)
                if len(fp["refuted"]) >= self.a.tries:  # next candidate of the same log
                    fp["closed"] = f"{self.a.tries} experiments without support"
            self.save()
            if it["task_success"]:  # 6.
                self.st["done"] = f"task succeeded at it{k}"
                break
        self.st["done"] = self.st["done"] or f"max {self.a.max_iter} experiments"
        self.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", required=True, help="file: one failed episode dir per line, in loop order")
    ap.add_argument("--domain", default=None, help="domains/<name> (default $MM_DOMAIN, else minecraft)")
    ap.add_argument("--world", default="nolava", help="world of the experiment episodes (the source episodes' world)")
    ap.add_argument("--save", type=int, default=600, help="world saves in the experiment episodes too (the next failure point)")
    ap.add_argument("--store", required=True, help="memory stores: World Knowledge wk_<store>.json, failed hypotheses fh_<store>.json")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-iter", type=int, default=10, help="experiments per episode; 0: no limit")
    ap.add_argument("--tries", type=int, default=3, help="experiments at one failure point before the next candidate")
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls (the Designer and the agent)")
    a = ap.parse_args()
    if a.mock:
        os.environ["LLM_MOCK"] = "1"
    dom = domains.load(a.domain)
    os.environ.setdefault("LLM_CONFIG", dom.llm_config({"world": a.world, "save": a.save}))
    from client import CFG
    ctx = RUN_DATA / "state" / "ours_loop" / a.name / "ctx.json"
    ctx.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_CTX_FILE"] = str(ctx)
    for line in Path(a.episodes).read_text().split():
        loop = Loop(a, dom, Path(line))
        write_ctx(ctx, {"env": dom.env_label, "method": "ours_system2", "chain": a.name, "order_id": "system2",
                        "task": loop.task, "seed": loop.seed, "episode_dir": str(loop.dir),
                        "run_id": f"ours_loop/{a.name}/{loop.task}_{loop.seed}"})
        stop_budget = Path(CFG["log_root"]) / "STOP_BUDGET"
        if stop_budget.exists() and not a.mock:  # mock calls cost nothing (the client skips its tripwire too)
            log("STOP: STOP_BUDGET present")
            return 2
        log(f"loop {loop.task} seed {loop.seed} from {line}")
        loop.run()
        log(f"  -> {loop.st['done']}; knowledge +{len(loop.st['knowledge'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
