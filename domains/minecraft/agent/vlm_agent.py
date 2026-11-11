"""The Executor: the Baseline agent. A VLM does the whole benchmark task, pausing the game every turn.

Each turn gemini gets the overall task, the screen, the game-reported height, position relative to the start,
facing, health and inventory, and how its last 10 commands ended (domains/minecraft/agent/prompts.py). It sends one command that
runs for the number of steps it chose (domains/minecraft/agent/controller.py). No memory, no recipes, no reflection. The episode
ends when the paper's success check (domains/minecraft/world/benchmark.py EpisodeMonitor) is met or the horizon runs out; deaths
respawn the agent, as in Optimus-1.

What an experiment adds (environment variables set from the variant tokens by experiments/chain.py):
  OPTIMUS_VLM_RECIPES         the recipe graph in the prompt (GT Recipe; setup/build_craft_graphs.py)
  OPTIMUS_VLM_OREHINT=1       the hidden ore rule in the prompt (GT World Rule)
  OPTIMUS_VLM_PLAN=1          initial planning: sub-goals written before the first turn (agent/planner.py)
  OPTIMUS_VLM_ME              MineEvolve memory (memory/mineevolve_memory.py)
  OPTIMUS_VLM_O1M             Optimus-1 memory, HDKG + AMEP (memory/optimus1_memory.py); _FROZEN=1: read only
  OPTIMUS_VLM_WK              verified World Knowledge in the prompt (executor/extras.py)
  OPTIMUS_EXPERIMENT          an experiment of experiments/ours_loop.py: the change to try (executor/extras.py)
  OPTIMUS_NOLAVA=1            lava near the agent becomes stone every NOLAVA_EVERY steps (domains/minecraft/world/nolava.py)
  OPTIMUS_CKPT_EVERY=N        the world is saved every N steps (domains/minecraft/world/save.py)
  OPTIMUS_MAX_STEPS=N         the episode ends N steps after its start (tests; the loop's rollouts: the window)
  OPTIMUS_STOP_FILE           the episode ends at the next turn once this file exists (a rollout whose occurrence came)
An episode started from a save (domains/minecraft/world/replay.py) gets `side`: the step count, the VLM's history and start position.

Game commands sent during the episode (nolava, the world save) ride on the chat field of the agent's own action,
one per step: an extra no-op tick would release held keys (Minecraft resets block breaking when attack is released)
and let the world run unrecorded.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import extras
import mineevolve_memory
import optimus1_memory
import planner
import prompts as P
import save
import world_knowledge
from controller import Controller
from hyperparams import HP
from images import img, jpeg
from inventory import inv_counts
from nolava import NOLAVA_EVERY, nolava_cmds
from paths import ROOT

GRAPHS = ROOT / "configs/suites/optimus1/craft_graphs.json"
class EnvBroken(RuntimeError):
    """Minecraft stopped answering (MineRL step timeout -> random observation, then done=True on every step).
    Raised out of the episode so the chain records a crash and retries it, instead of scoring an agent failure."""


def _env_broken(e: BaseException) -> bool:
    return isinstance(e, EnvBroken) or "environment server with done" in str(e)


class _Env:
    """The Optimus-1 env as the agent sees it: the release's per-step ore placement; everything else passes through."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)

    def __getattr__(self, k):
        return getattr(self._env, k)

    def __setattr__(self, k, v):
        setattr(self._env, k, v)

    def step(self, action, *a, **k):
        self._env._only_once = True  # main.py sets this before every STEVE-1 step
        return self._env.step(action)


class VLMAgent(Controller):
    def __init__(self, env, cfg, steps, log, obs, mon, task, opts):
        super().__init__(env, cfg, steps, log)
        self.mon, self.task, self.opts = mon, task, opts
        self.obs = obs
        self.prev = self.state(obs)
        self.start = list(self.prev["pos"])
        self.views, self.last_view = [], None
        self.chat = []  # game commands waiting to ride, one per step, on the chat field of the agent's own action
        self.pending_save = None  # the world save whose /save-all is in self.chat or was just sent

    def step(self, action, tag):
        if self.opts.get("nolava") and self.t % NOLAVA_EVERY == 0:
            self.chat += nolava_cmds(self.prev["pos"][1])
        sent = None
        if self.chat:
            sent = self.chat.pop(0)
            action = {**action, "chat": sent}
        obs, s, done = super().step(action, tag)
        ps = self.pending_save
        if ps:  # the server runs a chat command on the tick after it was sent: zip once level.dat is rewritten
            if sent == save.SAVE_CMD:
                ps["sent_t"] = self.t
            if "sent_t" in ps and save.saved(ps["mcdir"], ps["before"]):
                self.pending_save = None
                save.write(ps["mcdir"], ps["out"], ps["step"], ps["side"])
            elif "sent_t" in ps and self.t - ps["sent_t"] > save.SAVE_WITHIN:
                self.pending_save = None
                self.log(f"  world save at step {ps['step']}: level.dat not rewritten, skipped")
        if isinstance(self.info, dict) and "error" in self.info:  # MineRL: "Failed to take a step ... random observation"
            raise EnvBroken(f"MineRL step failed at step {self.t}: {str(self.info['error'])[:200]}")
        if self.result == "dead":  # respawn and keep going, as Optimus-1 does
            self.result = None
        return obs, s, done

    def check(self, s):
        if self.mon.success_step is not None:
            if self.success_t is None:
                self.success_t = self.t
            return "success"
        return None

    # ------------------------------------------------------------ the VLM
    def system(self):
        extras.refresh(self.opts)  # World Knowledge read live: what other loops added since shows from this turn on
        plan, reached = self.opts.get("plan"), self.opts.get("plan_reached", [])
        subgoal = planner.subgoal_text(plan, reached) if plan else P.NO_SUBGOAL
        out = P.SYSTEM.format(task=self.task, subgoal=subgoal, commands="\n".join([P.CMD_KEYS, P.CMD_STEVE, P.CMD_HELPERS]))
        if plan:
            out += planner.note(plan, reached)
        if self.opts.get("ore_hint"):
            out += {"1": P.ORE_HINT, "2": P.ORE_RULE_CODE}[self.opts["ore_hint"]]
        if self.opts.get("graph"):
            out += P.GRAPH_NOTE.format(graph=self.opts["graph"].strip())
        if self.opts.get("me") is not None:
            out = out.replace(mineevolve_memory.ANSWER_LINE, mineevolve_memory.SUBGOAL_FORMAT) + self.opts["me"].prompt_text()
        if self.opts.get("o1m") is not None:
            out = out.replace(optimus1_memory.ANSWER_LINE, optimus1_memory.REFLECT_FORMAT) + self.opts["o1m"].prompt_text()
        return out + extras.prompt_text(self.opts)

    def ask(self, client, history, n_turn):
        hist = "\n".join(history[-P.N_HISTORY:]) or "(none yet)"
        text = f"State now: {self.describe()}\n\n" + P.TURN.format(history=hist, n=n_turn)
        content = [{"type": "text", "text": text}]
        if self.opts.get("o1m") is not None:  # Optimus-1 memory: past cases at a periodic check
            content += self.opts["o1m"].turn_content(self.t)
        content += [{"type": "text", "text": "Current view:"}, img(jpeg(self.obs["pov"]))]
        if self.last_view is not None:
            content += [{"type": "text", "text": "View at your previous decision:"}, img(jpeg(self.last_view))]
        for label, pov in self.views:
            content += [{"type": "text", "text": f"Look-around view, {label}:"}, img(jpeg(pov))]
        self.views = []
        self.last_view = self.obs["pov"].copy()
        msgs = [{"role": "system", "content": self.system()}, {"role": "user", "content": content}]
        resp = client.chat.completions.create(model="FROM_CONFIG", messages=msgs, max_tokens=HP["executor"]["max_tokens"])
        txt = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        cmd = d.get("command") if isinstance(d.get("command"), dict) else None
        return cmd or {"type": "keys", "keys": [], "steps": 5}, d.get("thought", ""), txt


def subgoal_of(raw: str):
    """MineEvolve: the "subgoal" field of the VLM's JSON answer (None if missing or unparsable)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        return json.loads(m.group(0)).get("subgoal") if m else None
    except Exception:
        return None


def craft_graph(target: str | None) -> str | None:
    kind = os.environ.get("OPTIMUS_VLM_RECIPES")
    if not kind or not target:
        return None
    return json.loads(GRAPHS.read_text())[target][kind]


def run_task(cfg, env, obs, instruction, mon, ep: Path, logger, target: str | None = None,
             task_id: str | None = None, mcdir: Path | None = None, side: dict | None = None) -> dict:
    from client import LLMClient
    remaining = max(1, int(env.timeout) - int(env.num_steps))
    if os.environ.get("OPTIMUS_MAX_STEPS"):  # tests; the loop's rollouts (the window)
        remaining = min(remaining, int(os.environ["OPTIMUS_MAX_STEPS"]))
    opts = {"graph": craft_graph(target), "me": None, "o1m": None, "plan": None, "plan_reached": [],
            "ore_hint": os.environ.get("OPTIMUS_VLM_OREHINT"), "nolava": os.environ.get("OPTIMUS_NOLAVA") == "1",
            **extras.load()}  # World Knowledge, the experiment's change (executor/extras.py)
    save_every = int(os.environ.get("OPTIMUS_CKPT_EVERY") or 0)
    out = ep / "vlm"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    lines = []
    agent = VLMAgent(_Env(env), cfg, remaining, lambda m: lines.append(m), obs, mon, instruction, opts)
    me = None
    if os.environ.get("OPTIMUS_VLM_ME"):
        from client import context
        ctx = context()
        me = opts["me"] = mineevolve_memory.MineEvolve(os.environ["OPTIMUS_VLM_ME"], instruction, task_id, ctx.get("seed"),
                                                       ctx.get("order_id", ""))
        me.start(agent.prev, inv_counts(agent.obs))
    o1m = None
    if os.environ.get("OPTIMUS_VLM_O1M"):
        from client import context
        ctx = context()
        o1m = opts["o1m"] = optimus1_memory.Optimus1Memory(os.environ["OPTIMUS_VLM_O1M"], instruction, task_id, target,
                                                           ctx.get("seed"), ctx.get("order_id", ""))
    client = LLMClient()
    history, n, t0 = [], 0, time.time()
    step0 = int(side["step"]) if side else 0  # episode step of the save this episode starts from
    if side:
        history, n = list(side["history"]), int(side["turn"])
        agent.start = list(side["start"])
    if side and side.get("plan"):  # from a save: the plan of the episode the save is from, as far as it was reached
        opts["plan"], opts["plan_reached"] = side["plan"], list(side.get("plan_reached", []))
    elif not side and os.environ.get("OPTIMUS_VLM_PLAN") == "1":
        extra = (P.GRAPH_NOTE.format(graph=opts["graph"].strip()) if opts.get("graph") else "") + \
            world_knowledge.prompt_text(opts.get("knowledge"))
        opts["plan"] = planner.make(client, instruction, extra)
        if opts["plan"]:
            opts["plan_reached"] = planner.update(opts["plan"], [], inv_counts(agent.obs))
        (out / "plan.json").write_text(json.dumps({"plan": opts["plan"]}, indent=1))
    next_save = (step0 // save_every + 1) * save_every if save_every else None
    stop_f = os.environ.get("OPTIMUS_STOP_FILE")
    try:
        with open(out / "turns.jsonl", "a") as tf:
            while not agent.over():
                if stop_f and Path(stop_f).exists():  # the loop saw this rollout's occurrence: nothing more to show
                    agent.stop = "occurrence_seen"
                    break
                n += 1
                s0, t_start, inv0 = dict(agent.prev), agent.t, inv_counts(agent.obs)
                cmd, thought, raw = agent.ask(client, history, n)
                if me is not None:
                    frame = jpeg(agent.obs["pov"], (448, 252), 70)
                    me.before_turn(subgoal_of(raw), n, agent.prev, inv0, agent.t)
                if o1m is not None:
                    o1m.after_answer(raw, optimus1_memory.frame_bytes(agent.obs["pov"]), agent.prev, inv0, agent.t)
                try:
                    outcome = agent.execute(cmd, f"t{n}")
                except Exception as e:  # malformed command fields, helper errors
                    if _env_broken(e):
                        raise EnvBroken(f"env broke at turn {n}, step {agent.t}: {e!r}"[:300]) from e
                    outcome = f"invalid command: {repr(e)[:80]}"
                    if "Timeout!" in repr(e):  # the release's helpers raise this when the horizon runs out
                        agent.stop = agent.stop or "env_ended"
                inv1 = inv_counts(agent.obs)
                if opts["plan"]:
                    opts["plan_reached"] = planner.update(opts["plan"], opts["plan_reached"], inv1)
                eff = agent.effect(s0, agent.prev, inv0, inv1, agent.rows[t_start:agent.t])
                history.append(f"{n}. {json.dumps(cmd)} -> {outcome}, {agent.t - t_start} steps; {eff}")
                tf.write(json.dumps({"turn": n, "t0": step0 + t_start, "t1": step0 + agent.t, "cmd": cmd, "thought": thought,
                                     "outcome": outcome, "effect": eff, "raw": raw[:600],
                                     **({"subgoal": planner.current(opts["plan"], opts["plan_reached"])} if opts["plan"] else {})})
                         + "\n")
                tf.flush()
                if me is not None:
                    typ = cmd.get("type") if isinstance(cmd, dict) else None
                    failed = f"{typ} {outcome}" if typ in ("craft", "smelt", "equip") and str(outcome).startswith("failed") else None
                    seg = agent.rows[t_start:agent.t]
                    me.after_turn({"turn": n, "cmd": json.dumps(cmd), "outcome": outcome, "effect": eff}, frame, seg,
                                  agent.prev, inv1, agent.t, any(r.get("died") for r in seg), failed)
                if o1m is not None:
                    o1m.after_turn(cmd, outcome, inv0, inv1)
                agent.guard_turn(t_start, s0, inv0, inv1)
                if next_save is not None and step0 + agent.t >= next_save and not agent.over():
                    sv = {"turn": n, "history": history, "start": agent.start, "pos": agent.prev["pos"],
                          "ore_map": save.ore_map_to_json(env.ORE_MAP), "spawn": mon.first_pos,
                          "deaths": agent.deaths, "inventory": inv1,
                          "plan": opts["plan"], "plan_reached": opts["plan_reached"]}
                    agent.pending_save = dict(mcdir=mcdir, out=ep / "ckpt", step=step0 + agent.t, side=sv,
                                              before=save.level_mtime(mcdir))
                    agent.chat.append(save.SAVE_CMD)
                    next_save = ((step0 + agent.t) // save_every + 1) * save_every
                if n % 10 == 0 or agent.over():
                    (out / "frames" / f"{agent.t:06d}.jpg").write_bytes(jpeg(agent.obs["pov"], (256, 144), 70))
    except BaseException as e:
        if me is not None and "Timeout!" not in repr(e):  # a crashed attempt is retried, so undo its store writes
            me.abort()
        raise
    if me is not None:
        me.finish(agent.prev, inv_counts(agent.obs), agent.t)
        with open(out / "mineevolve.jsonl", "w") as f:
            for e in me.events:
                f.write(json.dumps(e) + "\n")
    if o1m is not None:  # written after the episode; a crashed attempt never gets here, so it writes nothing
        o1m.finish(agent.result == "success", agent.prev, inv_counts(agent.obs), agent.t)
        with open(out / "optimus1_memory.jsonl", "w") as f:
            for e in o1m.events:
                f.write(json.dumps(e) + "\n")
    summ = {"success": agent.result == "success", "success_t": agent.success_t, "stop": agent.stop,
            "steps": agent.t, "turns": n, "vlm_calls": agent.calls, "deaths": agent.deaths,
            "recipes": os.environ.get("OPTIMUS_VLM_RECIPES"),
            "ore_hint": opts["ore_hint"], "wall_s": round(time.time() - t0, 1), "log": lines[-5:],
            "start_step": step0, "nolava": opts["nolava"], "knowledge": len(opts["knowledge"] or []),
            "experiment": opts["experiment"],
            "plan": len(opts["plan"]) if opts["plan"] else None, "plan_reached": len(opts["plan_reached"])}
    if me is not None:
        ev = me.events
        summ["mineevolve"] = {"executions": sum(e["ev"] == "feedback" for e in ev),
                              "succeeded": sum(e["ev"] == "feedback" and e["y"] == 1 for e in ev),
                              "repairs": sum(e["ev"] == "repair" for e in ev),
                              "accepted": sum(r["accepted"] for e in ev if e["ev"] == "ingest" for r in e["report"]),
                              "rejected": sum(not r["accepted"] for e in ev if e["ev"] == "ingest" for r in e["report"]),
                              "frozen": me.frozen}
    if o1m is not None:
        ev = o1m.events
        summ["optimus1_memory"] = {"experiences": sum(e["ev"] == "experience" for e in ev),
                                   "graph_events": sum(e["ev"] in ("made", "missing", "gathered") for e in ev),  # _learn logs the event's own name
                                   "case_retrievals": sum(e["ev"] == "reflect_retrieval" and bool(e["ids"]) for e in ev),
                                   "target_known": ev[0]["target_known"], "frozen": o1m.frozen}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    logger.info(f"[vlmtask] {instruction!r}: {summ}")
    return summ
