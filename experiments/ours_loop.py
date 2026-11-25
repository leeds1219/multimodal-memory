"""Ours (EXPERIMENTS.md 6): the learning-from-failure loop. The terms are designer/terms.py's; the hyperparameters
come from configs/hyperparams.yaml (loop.*) only.

For each failed episode (run with saves, `experiments/run.py --method baseline --world nolava --save 600`), one
attempt:
  0. Milestones: the Designer decides the task's milestones from the first episode's log and the World Knowledge
     (designer/milestones.py); again whenever World Knowledge grows. Code cuts every log into segments at them.
  1. Failure point: code takes the first milestone the current timeline never held and how many of its item it held
     at the end (designer/terms.py "Failure point"); the restart point is the last save before the previous milestone
     was first held (the start of the timeline for the first milestone).
  2. Hypothesis: at the failure point, the Designer proposes the situation and action under which the agent gains
     the item (result "gain <item>"), the change and the prediction (designer/hypothesis.py), seeing the point's
     experiment record and <gains>: the item's gains in this timeline, then, with --success-pool, in the pool's
     episodes outside it, newest first (loop.gain_rows in all, loop.gains_per_episode per episode).
  3. Rollouts: loop.rollouts episodes from the restart point, loop.parallel_rollouts at a time (each on its own port,
     the same GPU), each ending loop.window_steps after it; the agent gets the change and the Supported World
     Knowledge (executor/extras.py), never the hypothesis or the prediction.
  4. Occurrence: the Designer marks, in each rollout, the first time the agent did A while S held and whether R
     followed (designer/judge.py); code checks whether the milestone was held at the occurrence or after it and counts the
     verdict (Supported if R followed in at least one rollout, Refuted
     if occurrences came and R never followed, Not reached if none came) and the frequency k/n. Supported and
     Refuted go to World Knowledge (memory/world_knowledge.py, <runs>/memory/wk_<store>.json); every experiment goes
     to the failure point's experiment record (one per milestone, kept across timelines).
  5. Supported: the timeline goes on with a rollout where R followed (the one furthest along the milestones at its
     last save; fewer steps on a tie), up to its last save, then a continuation
     episode from that save, whose agent is also given the hypothesis just Supported. It plays to the episode end,
     or stops once loop.stall_steps have passed without a new milestone; the failure point is taken again in the
     new timeline, with loop.tries_per_failure_point tries again.
  Ends on task success, after loop.max_experiments hypotheses, when every milestone was held, or after
  loop.tries_per_failure_point hypotheses at one failure point of one timeline without support.
  A rollout is judged at every world save while it plays and ends at the next turn once its occurrence shows.
Everything about the environment goes through the domain (domains/base.py; --domain, default minecraft). --world is
Minecraft's run option.

    python experiments/ours_loop.py --name pl1 --episodes eps.txt --world nolava --gpu 0 --port 28300 --store pl1
        [--plan] [--success-pool pool.txt]
eps.txt: one failed episode dir per line (from results/baseline/nolava_save600/). pool.txt: episode dirs of the run,
one per line; the loop appends its own rollouts and continuations to it (experiments/phase1.py). Output:
<runs>/results/ours_loop/<name>/<task>_<seed>/ (loop.json, system2.jsonl, e<k>_r<i>/ rollouts, e<k>_cont/).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO))
import paths  # noqa: E402,F401

import domains  # noqa: E402
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
            "failure_points": [], "experiments": []}
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
        return [(m["item"], m.get("count", 1)) for m in self.st["milestones"] or []]

    def milestone_lines(self) -> str:
        return "\n".join(f"{i}. {m['item']} {m.get('count', 1)}: {m['why']}" for i, m in enumerate(self.st["milestones"] or [], 1))

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
        log("  milestones: " + ", ".join(f"{i} {n}" for i, n in self.items()))
        return True

    # ---------------------------------------------------------------- 1. the failure point
    def failure_point(self) -> dict | None:
        """The failure point of the current timeline: its entry in st["failure_points"] (one per milestone; its
        experiment record is kept across timelines, its tries start again in a new one). None when every milestone
        was held."""
        sig = json.dumps(self.st["timeline"])
        pr = self.dom.progress(self.parts(), self.items())
        if pr is None:
            return None
        m = list(pr["milestone"])
        fp = next((x for x in self.st["failure_points"] if x["milestone"] == m), None)
        if fp is None:
            fp = {"milestone": m, "record": [], "tries": 0, "timeline": None}
            self.st["failure_points"].append(fp)
        if fp["timeline"] != sig:
            i, (ck, step) = self.restart(pr["before"])
            fp.update(timeline=sig, tries=0, held=pr["held"], segment=list(pr["segment"]), part=i,
                      episode=self.st["timeline"][i]["episode"], checkpoint=str(ck), checkpoint_step=step)
            log(f"  failure point: {m[0]} {m[1]}, {pr['held']} held; restart at step {step}")
        return fp

    def restart(self, before) -> tuple[int, tuple]:
        """(timeline part, (save, step)) of the restart point: the part's last save before `before` (part index, step
        the previous milestone was first held at); the part's own start when it has none. `before` None: the start
        of the timeline."""
        if before is None:
            p = self.st["timeline"][0]
            return 0, (p["start_ckpt"], p["start_step"])
        i, h = before
        p = self.st["timeline"][i]
        ep = Path(p["episode"])
        saves = [x for x in self.dom.checkpoints(ep) if p["start_step"] < x < h and (p["until"] is None or x <= p["until"])]
        x = max(saves, default=None)
        return i, ((ep / "ckpt" / f"s{x:06d}.zip", x) if x is not None else (p["start_ckpt"], p["start_step"]))

    # ---------------------------------------------------------------- 2. <gains>
    def gains(self, item: str) -> str:
        """<gains>: the gains of `item` in the current timeline (the latest ones), then in the pool's episodes outside
        it, newest first; loop.gain_rows gains in all, loop.gains_per_episode per pool episode; "" when none."""
        out, n = [], 0
        tl = [g for p in self.st["timeline"] for g in self.dom.gains(Path(p["episode"]), item, p["until"])]
        if tl:
            tl = tl[-L["gain_rows"]:]
            out.append(["== this timeline =="] + [r for g in tl for r in g if r])
            n = len(tl)
        if self.a.success_pool and Path(self.a.success_pool).exists():
            own = {x["episode"] for x in self.st["timeline"]}
            eps = [Path(e) for e in dict.fromkeys(Path(self.a.success_pool).read_text().split()) if e not in own]
            eps = sorted((e for e in eps if (e / "result.json").exists()),
                         key=lambda e: (e / "result.json").stat().st_mtime, reverse=True)
            for e in eps:
                if n >= L["gain_rows"]:
                    break
                gs = self.dom.gains(e, item)[:min(L["gains_per_episode"], L["gain_rows"] - n)]
                if gs:
                    out.append([f"== earlier episode of {self.dom.result(e)['task']} =="] + [r for g in gs for r in g if r])
                    n += len(gs)
        return "\n\n".join("\n".join(x) for x in out)

    # ---------------------------------------------------------------- 3. rollouts
    def run_episode(self, name: str, ckpt: str, exp_f: Path, max_steps: int | None,
                    stop_file: Path | None = None, slot: int = 0) -> tuple[Path, dict]:
        """One episode on port --port + slot, under a chain order of its own slot, so episodes of different slots can
        play at the same time."""
        ep = self.dir / name
        opts = {"name": f"{self.a.name}_p{slot}", "world": self.a.world, "save_every": L["save_every"],
                "gpu": self.a.gpu, "port": self.a.port + slot, "mock": self.a.mock, "max_steps": max_steps,
                "stop_file": stop_file, "plan": self.a.plan, "env": dict(e.split("=", 1) for e in self.a.env)}
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
                if self.a.success_pool:  # later experiments draw <gains> from it
                    with open(self.a.success_pool, "a") as f:
                        f.write(f"{ep}\n")
                return ep, res
        raise RuntimeError(f"episode {name} crashed 3 times")

    def occurrence(self, ep: Path, hyp: dict, occ: dict | None = None) -> dict | None:
        """The occurrence in rollout ep (judged by the Designer unless given); for a milestone result, whether it
        followed is code's check that the milestone was held at the occurrence or after it."""
        occ = occ or judge.occurrence(self.dom.commands(), hyp, self.dom.log([(ep, None, None)]), self.events)
        if occ and hyp.get("milestone"):
            occ = {**occ, "result_followed_designer": occ.get("result_followed_designer", occ["result_followed"]),
                   "result_followed": self.dom.milestone_held(ep, occ["turn"], tuple(hyp["milestone"]))}
        return occ

    def rollout(self, name: str, ckpt: str, exp_f: Path, window: int, hyp: dict,
                slot: int = 0) -> tuple[Path, dict, dict | None]:
        """One rollout, judged at every world save while it plays (4.): once its occurrence shows (for a milestone
        result, and the milestone is held), the loop writes the stop file and the episode ends at the next turn;
        otherwise it ends with the window. Returns the episode, its result and the occurrence (judged again on the
        whole log when none was seen while it played)."""
        seen = {}

        def check(ep, saves):
            occ = self.occurrence(ep, hyp, seen.get("occ"))
            if occ:
                seen["occ"] = occ
            if occ and hyp.get("milestone") and not occ["result_followed"]:
                return None  # the action came; play on until the milestone or the window end
            return occ

        ep, res = self.watched_episode(name, ckpt, exp_f, window, check, slot)
        occ = self.occurrence(ep, hyp, seen.get("occ"))
        return ep, res, occ

    def watched_episode(self, name: str, ckpt: str, exp_f: Path, max_steps: int | None, check,
                        slot: int = 0) -> tuple[Path, dict]:
        """run_episode, with `check(episode, saves)` called at every new world save while it plays: once it returns
        something, the loop writes it to the stop file and the episode ends at the next turn."""
        stop = self.dir / f"{name}.stop"
        stop.unlink(missing_ok=True)
        done = threading.Event()

        def watch():
            n_saves = 0
            while not done.wait(30):
                ep = self.dir / name
                saves = self.dom.checkpoints(ep) if (ep / "ckpt").exists() else []
                if len(saves) <= n_saves:
                    continue
                n_saves = len(saves)
                try:
                    why = check(ep, saves)
                except Exception as e:  # a log read mid-write: try again at the next save
                    self.events.append({"ev": f"{name}_watch_error", "error": repr(e)[:200]})
                    continue
                if why:
                    stop.write_text(json.dumps(why, ensure_ascii=False))
                    return

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            return self.run_episode(name, ckpt, exp_f, max_steps, stop, slot)
        finally:
            done.set()
            th.join()

    def continuation(self, name: str, ckpt: str, exp_f: Path, before: list, start: int) -> tuple[Path, dict]:
        """The continuation from step `start`, after the timeline parts `before` [(episode, until, marker)]: it plays
        to the episode end, or stops once loop.stall_steps have passed since the last new milestone (or its start)."""
        best = {"reached": self.dom.milestones_reached(before, self.items()), "step": start}

        def check(ep, saves):
            reached = self.dom.milestones_reached(before + [(ep, saves[-1], None)], self.items())
            if reached > best["reached"]:
                best.update(reached=reached, step=saves[-1])
            elif saves[-1] - best["step"] >= L["stall_steps"]:
                return {"stalled": f"no new milestone since step {best['step']}", "milestones_reached": reached}
            return None

        return self.watched_episode(name, ckpt, exp_f, None, check)

    def experiment(self, k: int, fp: dict) -> bool:
        """One hypothesis at fp (2.-5.); True when the task succeeded in one of its episodes."""
        p = self.st["timeline"][fp["part"]]
        ck, item = fp["checkpoint_step"], fp["milestone"][0]
        restart = next((t["turn"] for t in self.dom.turns(Path(fp["episode"]), p["until"]) if t["t0"] >= ck), None)
        window = min(L["window_steps"], max(1, self.horizon - ck)) if self.horizon else L["window_steps"]
        markers = {restart: ["-- Restart point: the experiment starts here --"]} if restart is not None else {}
        state = self.dom.state_line(Path(fp["episode"]), ck, window)
        gains_txt = self.gains(item)
        hyp = hypothesis.propose(self.instruction, self.dom.commands(), self.milestone_lines(), fp, state,
                                 self.dom.log(self.parts(), self.items(), markers), wk.lines(self.knowledge()),
                                 fp["record"], self.dom.plan(self.src), self.events, self.dom.vocabulary, gains_txt)
        if not hyp:
            self.st["done"] = "proposal unparsable"
            return False
        hyp["milestone"] = list(fp["milestone"])  # the result is checked as the failure point's milestone held
        log(f"  e{k} at {item} {fp['milestone'][1]}: {hyp['statement'][:120]}")
        exp_f = self.dir / f"e{k}_experiment.json"
        exp_f.write_text(json.dumps({"hypothesis": {"change": hyp["change"], "plan": hyp["plan"]},
                                     "knowledge": wk.supported(self.knowledge()),
                                     "from_start": not fp["checkpoint"]}, indent=1, ensure_ascii=False))
        slots = queue.Queue()
        for j in range(L["parallel_rollouts"]):
            slots.put(j)

        def one(i):  # 3., 4.
            j = slots.get()
            try:
                ep, res, occ = self.rollout(f"e{k}_r{i + 1}", fp["checkpoint"], exp_f, window, hyp, j)
            finally:
                slots.put(j)
            return {"episode": str(ep), "occurrence": occ, "success": bool(res.get("success")), "steps": res.get("steps")}

        with ThreadPoolExecutor(L["parallel_rollouts"]) as pool:
            rollouts = list(pool.map(one, range(L["rollouts"])))
        success = any(r["success"] for r in rollouts)
        verdict, kk, n = judge.verdict([r["occurrence"] for r in rollouts])
        observed = "; ".join(f"rollout {i + 1}: " + (f"T{o['turn']}, R {'followed' if o['result_followed'] else 'did not follow'}"
                                                    f": {o['evidence']}" if (o := r["occurrence"]) else
                                                    "no occurrence in the window") for i, r in enumerate(rollouts))
        e = {"k": k, "failure_point": {x: fp[x] for x in ("milestone", "held", "segment", "episode", "checkpoint_step")},
             **hyp, "rollouts": rollouts, "verdict": verdict, "frequency": [kk, n], "observed": observed,
             "continuation": None, "gains": gains_txt.count("\n== ") + bool(gains_txt)}
        self.st["experiments"].append(e)
        fp["record"].append({x: e[x] for x in ("statement", "change", "prediction", "verdict", "observed")})
        fp["tries"] += 1
        log(f"  e{k} {verdict} {kk}/{n}" + ("; task succeeded in a rollout" if success else ""))
        if verdict in ("Supported", "Refuted"):
            wk.add(self.store, {"statement": hyp["statement"], **hyp["hypothesis"], "verdict": verdict, "k": kk, "n": n,
                                "source": f"{self.task}_{self.seed}/e{k}"})
        if verdict == "Supported" and not success:  # 5.
            r, last = self.best_rollout(fp, rollouts)
            e["continued_from"] = r["episode"]
            cont_ckpt = str(Path(r["episode"]) / "ckpt" / f"s{last:06d}.zip") if last is not None else fp["checkpoint"]
            cont_f = self.dir / f"e{k}_cont_experiment.json"  # the change, and the knowledge with this hypothesis in it
            cont_f.write_text(json.dumps({**json.loads(exp_f.read_text()), "knowledge": wk.supported(self.knowledge()),
                                          "knowledge_live": str(self.store), "from_start": not cont_ckpt},
                                         indent=1, ensure_ascii=False))
            before = [(Path(p["episode"]), p["until"], p["marker"]) for p in self.head(fp)] + \
                ([(Path(r["episode"]), last, None)] if last is not None else [])
            cont, res = self.continuation(f"e{k}_cont", cont_ckpt, cont_f, before,
                                          last if last is not None else fp["checkpoint_step"])
            e["continuation"] = str(cont)
            stop = self.dir / f"e{k}_cont.stop"
            e["continuation_stalled"] = json.loads(stop.read_text()) if stop.exists() else None
            success = bool(res.get("success"))
            self.extend_timeline(fp, r["episode"], last, cont, hyp)
        return success

    def head(self, fp: dict) -> list:
        """The timeline parts up to fp's restart point."""
        i, ck = fp["part"], fp["checkpoint_step"]
        p = self.st["timeline"][i]
        return self.st["timeline"][:i] + ([{**p, "until": ck}] if ck > p["start_step"] else [])

    def best_rollout(self, fp: dict, rollouts: list) -> tuple[dict, int | None]:
        """(rollout, its last save) the timeline goes on with: of the rollouts where the result followed, the one whose
        last save is furthest along the milestones; on a tie, the one that got there in fewer steps."""
        head = [(Path(p["episode"]), p["until"], p["marker"]) for p in self.head(fp)]
        best = None
        for r in rollouts:
            if not (r["occurrence"] and r["occurrence"]["result_followed"]):
                continue
            saves = self.dom.checkpoints(Path(r["episode"]))
            last = saves[-1] if saves else None
            reached = self.dom.milestones_reached(head + ([(Path(r["episode"]), last, None)] if last else []),
                                                  self.items())
            key = (reached, -(last if last is not None else fp["checkpoint_step"]))
            r["milestones_reached"] = reached
            if best is None or key > best[0]:
                best = (key, r, last)
        return best[1], best[2]

    def extend_timeline(self, fp: dict, rollout: str, last: int | None, cont: Path, hyp: dict):
        """The timeline up to the restart point, then the rollout up to its last save, then the continuation."""
        ck = fp["checkpoint_step"]
        head = self.head(fp)
        marker = (f"-- Step {ck}: replayed from a save. Failure point tried: {fp['milestone'][0]} {fp['milestone'][1]}, "
                  f"{fp['held']} held. "
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
            fp = self.failure_point()  # 1.
            self.save()
            if fp is None:
                self.st["done"] = "every milestone held"
                break
            if fp["tries"] >= L["tries_per_failure_point"]:
                self.st["done"] = f"{fp['tries']} hypotheses without support at {fp['milestone'][0]} {fp['milestone'][1]}"
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
    ap.add_argument("--plan", action="store_true", help="loop episodes from the beginning plan first (from a save: its plan)")
    ap.add_argument("--success-pool", default=None, help="file of episode dirs the <gains> rows come from")
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls (the Designer and the agent)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every loop episode (repeatable)")
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
