"""Ours (EXPERIMENTS.md 6): the learning-from-failure loop. The terms are designer/terms.py's; the hyperparameters
come from configs/hyperparams.yaml (loop.*) only.

For each failed episode (run with saves, `experiments/run.py --method baseline --world nolava --save 600`), one
attempt:
  0. Milestones: the Designer decides the task's milestones from the first episode's log and the World Knowledge
     (designer/milestones.py); again whenever World Knowledge grows. Code cuts every log into segments at them.
  1. Failure points: the Designer lists every failure point of the current timeline (designer/failure_point.py),
     each of a kind (mismatch, missed response, ignored knowledge, judgment). The loop tries them by turn, earliest
     first, skipping turns tried before; after loop.tries_per_failure_point hypotheses at one point, the next one.
  2. Hypothesis: at the failure point, the Designer proposes S, A, R, the change and the prediction, seeing the
     point's experiment record (designer/hypothesis.py).
  3. Rollouts: loop.rollouts episodes from the restart point (the last save before the failure point, or the
     beginning of the same world), each ending loop.window_steps after it; the agent gets the change and the
     Supported World Knowledge (executor/extras.py), never the hypothesis or the prediction.
  4. Occurrence: the Designer marks, in each rollout, the first time the agent did A while S held and whether R
     followed (designer/judge.py); code counts the verdict (Supported if R followed in at least one rollout, Refuted
     if occurrences came and R never followed, Not reached if none came) and the frequency k/n. Supported and
     Refuted go to World Knowledge (memory/world_knowledge.py, <runs>/memory/wk_<store>.json); every experiment goes
     to the failure point's experiment record.
  5. Supported: the timeline goes on with a rollout where R followed, up to its last save, then a continuation
     episode from that save to the episode end, whose agent is also given the hypothesis just Supported; the
     failure points are listed again in the new timeline.
  Ends on task success, after loop.max_experiments hypotheses, or when no failure point is left.
  A rollout is judged at every world save while it plays and ends at the next turn once its occurrence shows.
Everything about the environment goes through the domain (domains/base.py; --domain, default minecraft). --world is
Minecraft's run option.

    python experiments/ours_loop.py --name pl1 --episodes eps.txt --world nolava --gpu 0 --port 28300 --store pl1
eps.txt: one failed episode dir per line (from results/baseline/nolava_save600/). Output:
<runs>/results/ours_loop/<name>/<task>_<seed>/ (loop.json, system2.jsonl, e<k>_r<i>/ rollouts, e<k>_cont/).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
import paths  # noqa: E402,F401

import domains  # noqa: E402
import failure_point  # noqa: E402
import hypothesis  # noqa: E402
import judge  # noqa: E402
import milestones  # noqa: E402
import world_knowledge as wk  # noqa: E402
from extras import change_text  # noqa: E402
from hyperparams import HP  # noqa: E402
from llm_context import write_ctx  # noqa: E402
from paths import RESULTS, RUN_DATA  # noqa: E402

L = HP["loop"]


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


class Loop:
    def __init__(self, a, dom, ep: Path):
        self.a, self.dom, self.src = a, dom, Path(ep).resolve()  # episodes run in their own working directory
        res = dom.result(self.src)
        self.task, self.seed, self.instruction = res["task"], int(res["seed"]), res.get("instruction", res["task"])
        self.horizon = int(res.get("horizon_steps") or 0)
        self.dir = RESULTS / "ours_loop" / a.name / f"{self.task}_{self.seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_f = self.dir / "loop.json"
        self.store = RUN_DATA / "memory" / f"wk_{a.store}.json"
        self.st = json.loads(self.state_f.read_text()) if self.state_f.exists() else {
            "source": str(self.src), "task": self.task, "seed": self.seed, "settings": dict(L), "done": None,
            "milestones": None, "milestones_wk": None,
            "timeline": [{"episode": str(self.src), "until": None, "marker": None, "start_ckpt": "", "start_step": 0}],
            "searched": None, "failure_points": [], "experiments": []}
        self.events = []

    def save(self):
        self.state_f.write_text(json.dumps(self.st, indent=1, ensure_ascii=False))
        with open(self.dir / "system2.jsonl", "a") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.events = []

    # ---------------------------------------------------------------- inputs
    def parts(self) -> list:
        return [(Path(p["episode"]), p["until"], p["marker"]) for p in self.st["timeline"]]

    def items(self) -> list:
        return [m["item"] for m in self.st["milestones"] or []]

    def milestone_lines(self) -> str:
        return "\n".join(f"{i}. {m['item']}: {m['why']}" for i, m in enumerate(self.st["milestones"] or [], 1))

    def knowledge(self) -> list:
        return wk.load(self.store) or []

    def ensure_milestones(self) -> bool:
        """Milestones decided for the World Knowledge as it is now (0.); False when the answer was unparsable."""
        n = len(self.knowledge())
        if self.st["milestones"] is not None and self.st["milestones_wk"] == n:
            return True
        ms = milestones.decide(self.instruction, self.dom.log([(self.src, None, None)]), wk.lines(self.knowledge()),
                               self.events, self.dom.vocabulary)
        if ms is None:
            return False
        self.st.update(milestones=ms, milestones_wk=n)
        log("  milestones: " + ", ".join(self.items()))
        return True

    def part_of(self, turn: int):
        """(index, part, its turns) of the timeline part that holds `turn`, or None."""
        for i, p in enumerate(self.st["timeline"]):
            ts = self.dom.turns(Path(p["episode"]), p["until"])
            if any(t["turn"] == turn for t in ts):
                return i, p, ts
        return None

    # ---------------------------------------------------------------- 1. failure points
    def reopen(self, old: dict, f: dict, sig: str):
        """A failure point listed again. It is tried again, with its experiment record, only if its evidence cites a
        turn of a timeline part added after its last try (the log shows something new there; the stated reason
        alone is not enough) and it has tries left; its hypotheses count against loop.tries_per_failure_point in
        total. Every relisting is kept in old["relisted"]."""
        cited = [int(t) for t in re.findall(r"T(\d+)", f.get("evidence", ""))]
        seen = set(old.get("episodes_at_try", []))
        new = [t for t in cited if (w := self.part_of(t)) and w[1]["episode"] not in seen]
        ok = bool(new) and old["closed"] != "Supported" and len(old["record"]) < L["tries_per_failure_point"]
        old.setdefault("relisted", []).append({"timeline": [x["episode"] for x in self.st["timeline"]], "kind": f["kind"],
                                               "differs": f.get("differs", ""), "evidence": f.get("evidence", ""),
                                               "new_turns_cited": new, "reopened": ok})
        if ok:
            old.update(closed=None, search=sig, kind=f["kind"], decision=f["decision"], shown=f["shown"],
                       knowledge=f["knowledge"], evidence=f["evidence"])
        log(f"  T{f['turn']} tried before: " + (f"reopened (new turns cited: {new})" if ok else
                                               "skipped (no turn of a later part cited, or no tries left)"))

    def search(self) -> bool:
        """List the failure points of the current timeline once; False when the answer was unparsable."""
        sig = json.dumps(self.st["timeline"])
        if self.st["searched"] == sig:
            return True
        found = failure_point.find(self.instruction, self.dom.commands(), self.milestone_lines(),
                                   self.dom.log(self.parts(), self.items()), wk.lines(self.knowledge()),
                                   self.dom.plan(self.src), self.events, self.dom.vocabulary)
        if found is None:
            return False
        self.st["searched"] = sig
        for f in found:
            where = self.part_of(f["turn"])
            if not where:
                log(f"  T{f['turn']} skipped: not a turn of the timeline")
                continue
            i, p, ts = where
            old = next((x for x in self.st["failure_points"] if x["episode"] == p["episode"] and x["turn"] == f["turn"]),
                       None)
            if old:  # tried before: reopened only on evidence from timeline parts added since its last try
                self.reopen(old, f, sig)
                continue
            ck = self.dom.save_before(Path(p["episode"]), f["turn"])
            if ck == ("", 0) or (ck and ck[1] < p["start_step"]):  # before the part's first save: where it started
                ck = (p["start_ckpt"], p["start_step"])
            self.st["failure_points"].append({**f, "episode": p["episode"], "checkpoint": str(ck[0]),
                                              "checkpoint_step": ck[1], "search": sig, "record": [], "closed": None})
        log("  failure points: " + (", ".join(f"T{f['turn']} ({f['kind']})" for f in found) or "none"))
        return True

    def next_point(self) -> dict | None:
        sig = self.st["searched"]
        open_ = [f for f in self.st["failure_points"] if f["search"] == sig and not f["closed"]]
        return min(open_, key=lambda f: f["turn"]) if open_ else None

    # ---------------------------------------------------------------- 3. rollouts
    def run_episode(self, name: str, ckpt: str, exp_f: Path, max_steps: int | None,
                    stop_file: Path | None = None) -> tuple[Path, dict]:
        ep = self.dir / name
        opts = {"name": self.a.name, "world": self.a.world, "save_every": L["save_every"], "gpu": self.a.gpu,
                "port": self.a.port, "mock": self.a.mock, "max_steps": max_steps, "stop_file": stop_file}
        for attempt in range(3):
            if ep.exists():
                ep.rename(ep.with_name(f"{ep.name}.crash{int(time.time())}"))
            ep.mkdir(parents=True)
            try:
                res = self.dom.run_episode(self.task, self.seed, ep, ckpt, exp_f, opts)
            except Exception as e:
                res = {"status": "crashed", "error": repr(e)}
            log(f"  {name} attempt {attempt + 1}: {res.get('status')} success={res.get('success')} steps={res.get('steps')}")
            if res.get("status") in ("finished", "anomaly"):
                return ep, res
        raise RuntimeError(f"episode {name} crashed 3 times")

    def rollout(self, name: str, ckpt: str, exp_f: Path, window: int, hyp: dict) -> tuple[Path, dict, dict | None]:
        """One rollout, judged at every world save while it plays (4.): once its occurrence shows, the loop writes
        the stop file and the episode ends at the next turn; otherwise it ends with the window. Returns the episode,
        its result and the occurrence (judged again on the whole log when none was seen while it played)."""
        stop = self.dir / f"{name}.stop"
        stop.unlink(missing_ok=True)
        done, seen = threading.Event(), {}

        def watch():
            n_saves = 0
            while not done.wait(30):
                ep = self.dir / name
                saves = self.dom.checkpoints(ep) if (ep / "ckpt").exists() else []
                if len(saves) <= n_saves:
                    continue
                n_saves = len(saves)
                try:
                    occ = judge.occurrence(self.dom.commands(), hyp, self.dom.log([(ep, None, None)]), self.events)
                except Exception as e:  # a log read mid-write: try again at the next save
                    self.events.append({"ev": "occurrence_watch_error", "error": repr(e)[:200]})
                    continue
                if occ:
                    seen["occ"] = occ
                    stop.write_text(json.dumps(occ, ensure_ascii=False))
                    return

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            ep, res = self.run_episode(name, ckpt, exp_f, window, stop)
        finally:
            done.set()
            th.join()
        occ = seen.get("occ") or judge.occurrence(self.dom.commands(), hyp, self.dom.log([(ep, None, None)]),
                                                  self.events)
        return ep, res, occ

    def experiment(self, k: int, fp: dict) -> bool:
        """One hypothesis at fp (2.-5.); True when the task succeeded in one of its episodes."""
        p = self.part_of(fp["turn"])[1]
        restart = next(t["turn"] for t in self.dom.turns(Path(fp["episode"]), p["until"])
                       if t["t0"] >= fp["checkpoint_step"])
        window = min(L["window_steps"], max(1, self.horizon - fp["checkpoint_step"])) if self.horizon \
            else L["window_steps"]
        markers = {restart: ["-- Restart point: the experiment starts here --"],
                   fp["turn"]: [f"-- Failure point: T{fp['turn']} --"]}
        if restart == fp["turn"]:
            markers = {restart: markers[restart] + [f"-- Failure point: T{fp['turn']} --"]}
        state = self.dom.state_line(Path(fp["episode"]), fp["checkpoint_step"], window)
        hyp = hypothesis.propose(self.instruction, self.dom.commands(), self.milestone_lines(), fp, state,
                                 self.dom.log(self.parts(), self.items(), markers), wk.lines(self.knowledge()),
                                 fp["record"], self.dom.plan(self.src), self.events, self.dom.vocabulary)
        if not hyp:
            self.st["done"] = "proposal unparsable"
            return False
        log(f"  e{k} at T{fp['turn']}: {hyp['statement'][:120]}")
        exp_f = self.dir / f"e{k}_experiment.json"
        exp_f.write_text(json.dumps({"hypothesis": {"change": hyp["change"], "plan": hyp["plan"]},
                                     "knowledge": wk.supported(self.knowledge()),
                                     "from_start": not fp["checkpoint"]}, indent=1, ensure_ascii=False))
        rollouts, success = [], False
        for i in range(L["rollouts"]):  # 3.
            ep, res, occ = self.rollout(f"e{k}_r{i + 1}", fp["checkpoint"], exp_f, window, hyp)  # 4.
            rollouts.append({"episode": str(ep), "occurrence": occ, "success": bool(res.get("success")),
                             "steps": res.get("steps")})
            success = success or bool(res.get("success"))
        verdict, kk, n = judge.verdict([r["occurrence"] for r in rollouts])
        observed = "; ".join(f"rollout {i + 1}: " + (f"T{o['turn']}, R {'followed' if o['result_followed'] else 'did not follow'}"
                                                    f": {o['evidence']}" if (o := r["occurrence"]) else
                                                    "no occurrence in the window") for i, r in enumerate(rollouts))
        e = {"k": k, "failure_point": {x: fp[x] for x in ("episode", "turn", "kind")}, **hyp, "rollouts": rollouts,
             "verdict": verdict, "frequency": [kk, n], "observed": observed, "continuation": None}
        self.st["experiments"].append(e)
        fp["record"].append({x: e[x] for x in ("statement", "change", "prediction", "verdict", "observed")})
        fp["episodes_at_try"] = [x["episode"] for x in self.st["timeline"]]  # episodes played after this try are new
        log(f"  e{k} {verdict} {kk}/{n}" + ("; task succeeded in a rollout" if success else ""))
        if verdict in ("Supported", "Refuted"):
            wk.add(self.store, {"statement": hyp["statement"], **hyp["hypothesis"], "verdict": verdict, "k": kk, "n": n,
                                "source": f"{self.task}_{self.seed}/e{k}"})
        if verdict == "Supported" and not success:  # 5.
            fp["closed"] = "Supported"
            r = next(x for x in rollouts if x["occurrence"] and x["occurrence"]["result_followed"])
            saves = self.dom.checkpoints(Path(r["episode"]))
            last = saves[-1] if saves else None
            cont_ckpt = str(Path(r["episode"]) / "ckpt" / f"s{last:06d}.zip") if last is not None else fp["checkpoint"]
            cont_f = self.dir / f"e{k}_cont_experiment.json"  # the change, and the knowledge with this hypothesis in it
            cont_f.write_text(json.dumps({**json.loads(exp_f.read_text()), "knowledge": wk.supported(self.knowledge()),
                                          "from_start": not cont_ckpt}, indent=1, ensure_ascii=False))
            cont, res = self.run_episode(f"e{k}_cont", cont_ckpt, cont_f, None)
            e["continuation"] = str(cont)
            success = bool(res.get("success"))
            self.extend_timeline(fp, r["episode"], last, cont, hyp)
        elif len(fp["record"]) >= L["tries_per_failure_point"]:
            fp["closed"] = f"{len(fp['record'])} hypotheses without support"
        return success

    def extend_timeline(self, fp: dict, rollout: str, last: int | None, cont: Path, hyp: dict):
        """The timeline up to the restart point, then the rollout up to its last save, then the continuation."""
        i, p, _ = self.part_of(fp["turn"])
        ck = fp["checkpoint_step"]
        head = self.st["timeline"][:i] + ([{**p, "until": ck}] if ck > p["start_step"] else [])
        marker = (f"-- Step {ck}: replayed from a save. Failure point tried: T{fp['turn']} ({fp['shown'][:150]}). "
                  f"Change given to the agent from here: \"{change_text(hyp['change'])}\". Verdict: Supported --")
        if last is None:  # no save in the rollout: the continuation replays from the restart point itself
            self.st["timeline"] = head + [{"episode": str(cont), "until": None, "marker": marker,
                                           "start_ckpt": fp["checkpoint"], "start_step": ck}]
            return
        self.st["timeline"] = head + [
            {"episode": rollout, "until": last, "marker": marker, "start_ckpt": fp["checkpoint"], "start_step": ck},
            {"episode": str(cont), "until": None, "marker": None,
             "start_ckpt": str(Path(rollout) / "ckpt" / f"s{last:06d}.zip"), "start_step": last}]

    # ---------------------------------------------------------------- the attempt
    def run(self):
        if self.st["done"]:
            return
        while len(self.st["experiments"]) < L["max_experiments"]:
            if not self.ensure_milestones():
                self.st["done"] = "milestones unparsable"
                break
            if not self.search():  # 1.
                self.st["done"] = "failure point answer unparsable"
                break
            self.save()
            fp = self.next_point()
            if fp is None:
                self.st["done"] = "no failure point left"
                break
            succeeded = self.experiment(len(self.st["experiments"]) + 1, fp)
            self.save()
            if self.st["done"]:
                break
            if succeeded:
                self.st["done"] = f"task succeeded at e{len(self.st['experiments'])}"
                break
        self.st["done"] = self.st["done"] or f"max {L['max_experiments']} experiments"
        self.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", required=True, help="file: one failed episode dir per line, in loop order")
    ap.add_argument("--domain", default=None, help="domains/<name> (default $MM_DOMAIN, else minecraft)")
    ap.add_argument("--world", default="nolava", help="world of the experiment episodes (the source episodes' world)")
    ap.add_argument("--store", required=True, help="World Knowledge store memory/wk_<store>.json")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls (the Designer and the agent)")
    a = ap.parse_args()
    if a.mock:
        os.environ["LLM_MOCK"] = "1"
    dom = domains.load(a.domain)
    os.environ.setdefault("LLM_CONFIG", dom.llm_config({"world": a.world, "save_every": L["save_every"]}))
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
        log(f"  -> {loop.st['done']}; experiments {len(loop.st['experiments'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
