"""System 1: the Baseline agent. A VLM does the whole benchmark task, pausing the game every turn.

Each turn gemini gets the overall task, the screen, the game-reported height, position relative to the start,
facing, health and inventory, and how its last 10 commands ended (system1/prompts.py). It sends one command that
runs for the number of steps it chose (system1/controller.py). No memory, no recipes, no reflection. The episode
ends when the paper's success check (world/benchmark.py EpisodeMonitor) is met or the horizon runs out; deaths
respawn the agent, as in Optimus-1.

What an experiment adds (environment variables set from the variant tokens by experiments/chain.py):
  OPTIMUS_VLM_RECIPES         the recipe graph in the prompt (GT Recipe; setup/build_craft_graphs.py)
  OPTIMUS_VLM_OREHINT=1       the hidden ore rule in the prompt (GT World Rule)
  OPTIMUS_VLM_ME              MineEvolve memory (memory/mineevolve_memory.py)
  OPTIMUS_VLM_WK              verified World Knowledge in the prompt (memory/world_knowledge.py)
  OPTIMUS_EXPERIMENT          an experiment of experiments/ours_loop.py: the change to try, in the prompt
  OPTIMUS_NOLAVA=1            lava near the agent becomes stone every NOLAVA_EVERY steps (world change)
  OPTIMUS_CKPT_EVERY=N        the world is saved every N steps (world/save.py)
  OPTIMUS_MAX_STEPS=N         tests only: the episode ends after N steps
An episode started from a save (world/replay.py) gets `side`: the step count, the VLM's history and start position.

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

import mineevolve_memory
import prompts as P
import save
import world_knowledge
from controller import Controller
from images import img, jpeg
from inventory import inv_counts

GRAPHS = Path(__file__).resolve().parents[1] / "configs/suites/optimus1/craft_graphs.json"
NOLAVA_EVERY, NOLAVA_R = 10, 12  # (2R+1)^3 = 15625 blocks, under /fill's 32768
# The fill is run by an invisible marker armor stand at the agent: a /fill that finds no lava fails, and a failure
# run by the player prints a red "No blocks were filled" on its screen; run by the stand it prints nothing.
# /fill refuses the whole box when a corner is below y 0, so the box is centred at max(y, R): near the bottom it is
# y 0 to 2R, which still holds the agent.


def nolava_cmds(y: float) -> tuple[str, str, str]:
    r = NOLAVA_R
    return ('/execute at @p run summon minecraft:armor_stand ~ ~ ~ {Invisible:1b,Marker:1b,NoGravity:1b,Silent:1b,Tags:["nolava"]}',
            f"/execute as @e[tag=nolava] at @s positioned ~ {max(int(y), r)} ~ run fill ~-{r} ~-{r} ~-{r} ~{r} ~{r} ~{r} "
            "minecraft:stone replace minecraft:lava",
            "/kill @e[tag=nolava]")


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
        out = P.SYSTEM.format(task=self.task, subgoal=P.NO_SUBGOAL, commands="\n".join([P.CMD_KEYS, P.CMD_STEVE, P.CMD_HELPERS]))
        if self.opts.get("ore_hint"):
            out += {"1": P.ORE_HINT, "2": P.ORE_RULE_CODE}[self.opts["ore_hint"]]
        if self.opts.get("graph"):
            out += P.GRAPH_NOTE.format(graph=self.opts["graph"].strip())
        if self.opts.get("me") is not None:
            out = out.replace(mineevolve_memory.ANSWER_LINE, mineevolve_memory.SUBGOAL_FORMAT) + self.opts["me"].prompt_text()
        out += world_knowledge.prompt_text(self.opts.get("knowledge"))
        if self.opts.get("experiment"):
            h = self.opts["experiment"]
            out += P.EXP_NOTE.format(change=h["change"], prediction=h["prediction"], plan=h["plan"])
        return out

    def ask(self, client, history, n_turn):
        hist = "\n".join(history[-P.N_HISTORY:]) or "(none yet)"
        text = f"State now: {self.describe()}\n\n" + P.TURN.format(history=hist, n=n_turn)
        content = [{"type": "text", "text": text}, {"type": "text", "text": "Current view:"}, img(jpeg(self.obs["pov"]))]
        if self.last_view is not None:
            content += [{"type": "text", "text": "View at your previous decision:"}, img(jpeg(self.last_view))]
        for label, pov in self.views:
            content += [{"type": "text", "text": f"Look-around view, {label}:"}, img(jpeg(pov))]
        self.views = []
        self.last_view = self.obs["pov"].copy()
        msgs = [{"role": "system", "content": self.system()}, {"role": "user", "content": content}]
        resp = client.chat.completions.create(model="FROM_CONFIG", messages=msgs, max_tokens=500)
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
    from gemini_client import GeminiClient
    remaining = max(1, int(env.timeout) - int(env.num_steps))
    if os.environ.get("OPTIMUS_MAX_STEPS"):  # tests only: end the episode after this many steps
        remaining = min(remaining, int(os.environ["OPTIMUS_MAX_STEPS"]))
    exp = json.loads(Path(os.environ["OPTIMUS_EXPERIMENT"]).read_text()) if os.environ.get("OPTIMUS_EXPERIMENT") else {}
    opts = {"graph": craft_graph(target), "me": None,
            "ore_hint": os.environ.get("OPTIMUS_VLM_OREHINT"), "nolava": os.environ.get("OPTIMUS_NOLAVA") == "1",
            "knowledge": exp.get("knowledge") if exp else world_knowledge.load(os.environ.get("OPTIMUS_VLM_WK")),
            "experiment": exp.get("hypothesis")}
    save_every = int(os.environ.get("OPTIMUS_CKPT_EVERY") or 0)
    out = ep / "vlm"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    lines = []
    agent = VLMAgent(_Env(env), cfg, remaining, lambda m: lines.append(m), obs, mon, instruction, opts)
    me = None
    if os.environ.get("OPTIMUS_VLM_ME"):
        from gemini_client import context
        ctx = context()
        me = opts["me"] = mineevolve_memory.MineEvolve(os.environ["OPTIMUS_VLM_ME"], instruction, task_id, ctx.get("seed"),
                                                       ctx.get("order_id", ""))
        me.start(agent.prev, inv_counts(agent.obs))
    client = GeminiClient()
    history, n, t0 = [], 0, time.time()
    step0 = int(side["step"]) if side else 0  # episode step of the save this episode starts from
    if side:
        history, n = list(side["history"]), int(side["turn"])
        agent.start = list(side["start"])
    next_save = (step0 // save_every + 1) * save_every if save_every else None
    try:
        with open(out / "turns.jsonl", "a") as tf:
            while not agent.over():
                n += 1
                s0, t_start, inv0 = dict(agent.prev), agent.t, inv_counts(agent.obs)
                cmd, thought, raw = agent.ask(client, history, n)
                if me is not None:
                    frame = jpeg(agent.obs["pov"], (448, 252), 70)
                    me.before_turn(subgoal_of(raw), n, agent.prev, inv0, agent.t)
                try:
                    outcome = agent.execute(cmd, f"t{n}")
                except Exception as e:  # malformed command fields, helper errors
                    if _env_broken(e):
                        raise EnvBroken(f"env broke at turn {n}, step {agent.t}: {e!r}"[:300]) from e
                    outcome = f"invalid command: {repr(e)[:80]}"
                    if "Timeout!" in repr(e):  # the release's helpers raise this when the horizon runs out
                        agent.stop = agent.stop or "env_ended"
                inv1 = inv_counts(agent.obs)
                eff = agent.effect(s0, agent.prev, inv0, inv1, agent.rows[t_start:agent.t])
                history.append(f"{n}. {json.dumps(cmd)} -> {outcome}, {agent.t - t_start} steps; {eff}")
                tf.write(json.dumps({"turn": n, "t0": step0 + t_start, "t1": step0 + agent.t, "cmd": cmd, "thought": thought,
                                     "outcome": outcome, "effect": eff, "raw": raw[:600]}) + "\n")
                tf.flush()
                if me is not None:
                    typ = cmd.get("type") if isinstance(cmd, dict) else None
                    failed = f"{typ} {outcome}" if typ in ("craft", "smelt", "equip") and str(outcome).startswith("failed") else None
                    seg = agent.rows[t_start:agent.t]
                    me.after_turn({"turn": n, "cmd": json.dumps(cmd), "outcome": outcome, "effect": eff}, frame, seg,
                                  agent.prev, inv1, agent.t, any(r.get("died") for r in seg), failed)
                agent.guard_turn(t_start, s0, inv0, inv1)
                if next_save is not None and step0 + agent.t >= next_save and not agent.over():
                    sv = {"turn": n, "history": history, "start": agent.start, "pos": agent.prev["pos"],
                          "ore_map": save.ore_map_to_json(env.ORE_MAP), "spawn": mon.first_pos,
                          "deaths": agent.deaths, "inventory": inv1}
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
    summ = {"success": agent.result == "success", "success_t": agent.success_t, "stop": agent.stop,
            "steps": agent.t, "turns": n, "vlm_calls": agent.calls, "deaths": agent.deaths,
            "recipes": os.environ.get("OPTIMUS_VLM_RECIPES"),
            "ore_hint": opts["ore_hint"], "wall_s": round(time.time() - t0, 1), "log": lines[-5:],
            "start_step": step0, "nolava": opts["nolava"], "knowledge": len(opts["knowledge"] or []),
            "experiment": opts["experiment"]}
    if me is not None:
        ev = me.events
        summ["mineevolve"] = {"executions": sum(e["ev"] == "feedback" for e in ev),
                              "succeeded": sum(e["ev"] == "feedback" and e["y"] == 1 for e in ev),
                              "repairs": sum(e["ev"] == "repair" for e in ev),
                              "accepted": sum(r["accepted"] for e in ev if e["ev"] == "ingest" for r in e["report"]),
                              "rejected": sum(not r["accepted"] for e in ev if e["ev"] == "ingest" for r in e["report"]),
                              "frozen": me.frozen}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    logger.info(f"[vlmtask] {instruction!r}: {summ}")
    return summ
