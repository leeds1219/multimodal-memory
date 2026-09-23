"""Run ONE Optimus-3 (as released) episode in its own MineStudio simulator (env "C3").

The released code has no automated rollout; its only live-episode driver is the
interactive GUI server (``gui_server.py``). This harness reproduces that
server's loop headless:

  * ``/reset``  -> ``MinecraftSim(obs_size=(128,128), preferred_spawn_biome, seed,
                   callbacks=[CommandsCallback(<8 gamerules>)])`` + Craft/Smelt/Equip
                   workers (gui_server.py:261-311);
  * ``/send_text task=planning``  -> ``Optimus3Agent.plan(text)`` (gui_server.py:461-466);
  * ``/send_text task=action`` in a loop -> ``_step`` per tick, ``model.reset(subgoal)``
                   when a new subgoal starts, next subgoal on ``check_inventory``
                   (gui_server.py:314-386, 471-503).  ``released_step`` below is
                   ``_step`` copied verbatim minus the GUI plumbing (pause flag,
                   frame pump, ``step_hook``).

Evaluation glue (DEVIATIONS.md "Optimus-3 (Stage C, own models)"):
  * world seed from configs/seeds.yaml (new MinecraftSim per episode); python,
    numpy, torch seeded with it (random_ore and the VAE prior are stochastic);
  * a MinecraftCallback sees EVERY env step (also inside the craft/smelt GUI
    macros): shared EpisodeMonitor (paper success criterion, trajectory,
    keyframes) and a BaseException that ends the episode on success / horizon;
  * planner input converted to the planner's own form (DECISIONS C1);
  * every MLLM call logged to llm_calls.jsonl;
  * released loop has no end: it stops when the plan is exhausted (as the GUI
    answers "success") -> end_reason plan_exhausted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE.parents[1]
sys.path.insert(0, str(REPRO / "envs"))

from common import EpisodeMonitor, SEEDS, TASKS, TARGETS, disk_free_gb, load_yaml, write_result  # noqa: E402

R = Path("/home/rag/data/official/Optimus-3")
W = Path("/home/rag/data/official/optimus3_weights")
os.environ.setdefault("MINESTUDIO_DIR", str(W / "minestudio_dir"))
os.environ.setdefault("OPTIMUS3_SBERT_DIR", str(W / "sentence-bert-base"))
os.environ.setdefault("OPTIMUS3_MINECLIP_TOKENIZER", str(W / "clip-vit-base-patch16-tokenizer"))
os.environ.setdefault("OPTIMUS3_ATTN", "sdpa")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

MINUTE = 1200
# Same 8 reset commands as gui_server.py:278-289 and MineEvolve conf/evaluate.yaml
RESET_COMMANDS = [
    "/gamerule sendCommandFeedback false",
    "/gamerule commandBlockOutput false",
    "/gamerule keepInventory true",
    "/effect give @a night_vision 99999 250 true",
    "/gamerule doDaylightCycle false",
    "/time set 0",
    "/gamerule doImmediateRespawn true",
    "/spawnpoint",
]
# MineEvolve benchmark yamls (prefer_biome), DECISIONS C2
MINEEVOLVE_BIOME = {"wooden": "forest", "stone": "plains", "iron": "plains", "gold": "plains",
                    "redstone": "plains", "diamond": "plains", "armor": "plains"}
# Method code: an exception whose innermost frame is here = the released algorithm failed (D25).
METHOD_DIRS = (str(R / "src") + "/", str(R / "MineStudio/minestudio/models") + "/")


# ----------------------------------------------------------------------------
# Planner input (DECISIONS C1)
# ----------------------------------------------------------------------------
PLANNER_ITEM = {"#saplings": "oak_sapling", "#logs": "logs"}


def planner_text(uid: str, form: str) -> str:
    """Text given to ``Optimus3Agent.plan`` (which wraps it as "How to <text> from scratch?")."""
    instr = TASKS[uid]["instruction"]
    if form == "verbatim":
        return instr[0].lower() + instr[1:]
    spec = TARGETS[uid]
    item = PLANNER_ITEM.get(spec["any_of"][0], spec["any_of"][0])
    if form == "obtain":  # the planner's own benchmark form: "How to obtain 1 pink_dye from scratch?"
        return f"obtain {int(spec['count'])} {item}"
    if form == "get":  # README GUI form "get a xxx"
        return f"get {int(spec['count'])} {item.replace('_', ' ')}"
    raise ValueError(form)


# ----------------------------------------------------------------------------
# MLLM call logging (no change to the calls themselves)
# ----------------------------------------------------------------------------
class LLMLogger:
    def __init__(self, agent, path: Path | None):
        self.agent, self.path, self.records = agent, path, []
        self._last = {}
        tok = agent.processor.tokenizer
        eos = agent.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.caller = "?"
        orig_gen = agent.model.generate

        def generate(*a, **k):
            t = time.time()
            out = orig_gen(*a, **k)
            n_in = int(k["input_ids"].shape[1])
            new = out[0, n_in:].tolist()
            self._last = {"tokens_in": n_in, "tokens_out": len(new), "gen_latency_s": round(time.time() - t, 3),
                          "max_new_tokens": k.get("max_new_tokens"),
                          "finish_reason": "stop" if any(i in self.eos for i in new) else "length"}
            return out

        agent.model.generate = generate
        orig_g = agent._generate

        def _generate(messages, max_new_tokens=2048, task_type="plan", skip_special_tokens=True):
            t = time.time()
            prompt = agent.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            out = orig_g(messages, max_new_tokens=max_new_tokens, task_type=task_type,
                         skip_special_tokens=skip_special_tokens)
            rec = {"i": len(self.records), "caller": self.caller, "task_type": task_type, "prompt": prompt,
                   "response": out[0], "latency_s": round(time.time() - t, 3), **self._last,
                   "time": time.time(), "env_step": self.step_fn() if self.step_fn else None}
            self.records.append(rec)
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            return out

        agent._generate = _generate
        self.step_fn = None
        del tok

    def summary(self) -> dict:
        r = self.records
        return {"llm_calls": len(r), "tokens_in": sum(x.get("tokens_in", 0) for x in r),
                "tokens_out": sum(x.get("tokens_out", 0) for x in r),
                "llm_latency_s": round(sum(x["latency_s"] for x in r), 1),
                "llm_truncated": sum(1 for x in r if x.get("finish_reason") == "length"),
                "llm_calls_by_caller": {c: sum(1 for x in r if x["caller"] == c) for c in {x["caller"] for x in r}},
                "cost_usd": 0.0}


def load_agent(device: str = "cuda"):
    from minecraftoptimus.model.agent.optimus3 import Optimus3Agent
    return Optimus3Agent(str(W / "Optimus-3-ActionHead"), str(W / "Optimus-3"), str(W / "Optimus-3-Task-Router"),
                         device=device)


# ----------------------------------------------------------------------------
# Per-step monitor as a MineStudio callback (sees macro steps too)
# ----------------------------------------------------------------------------
class EpisodeStop(BaseException):
    """Ends the episode from inside any env step. BaseException so the workers'
    ``except AssertionError`` / ``except Exception`` never swallow it."""


def _inventory(info) -> dict:
    out = {}
    for _, v in (info.get("inventory") or {}).items():
        t = str(v.get("type", "none")).replace("minecraft:", "")
        if t in ("none", "air"):
            continue
        out[t] = out.get(t, 0) + int(v.get("quantity", 0))
    return out


def make_monitor_callback(mon: EpisodeMonitor, horizon: int):
    from minestudio.simulator.callbacks import MinecraftCallback

    class StepMonitor(MinecraftCallback):
        def __init__(self):
            super().__init__()
            self.action = None
            self.active = False  # set after reset

        def before_step(self, sim, action):
            self.action = action
            return action

        def after_step(self, sim, obs, reward, terminated, truncated, info):
            if not self.active:
                return obs, reward, terminated, truncated, info
            o = {"pov": info.get("pov"), "inventory": _inventory(info),
                 "location_stats": info.get("location_stats"), "life_stats": info.get("life_stats")}
            mon.on_step(self.action or {}, o, bool(terminated), info)
            if mon.success_step is not None:
                mon.end_reason = "success"
                raise EpisodeStop("success")
            if mon.steps >= horizon:
                mon.end_reason = "horizon"
                raise EpisodeStop("horizon")
            if mon.over:
                raise EpisodeStop(mon.end_reason or "done")
            return obs, reward, terminated, truncated, info

    return StepMonitor()


# ----------------------------------------------------------------------------
# gui_server._step, verbatim minus GUI plumbing (gui_server.py:314-386)
# ----------------------------------------------------------------------------
STATE = {"look_down_once": False}


def released_step(env, agent, obs, task, goal, helper):
    import numpy as np
    from minecraftoptimus.model.agent.optimus3 import check_inventory

    if "craft" in task:
        result, _ = helper["craft"].crafting(goal["item"], goal["count"])
        action = env.env.noop_action()

        pickaxe = env.find_best_pickaxe()
        if pickaxe:
            helper["equip"].equip_item(pickaxe)
        obs, reward, terminated, truncated, info = env.step(action)

    elif "smelt" in task:
        result, _ = helper["smelt"].smelting(goal["item"], goal["count"])
        obs, reward, terminated, truncated, info = env.step(env.env.noop_action())
    else:
        env._only_once = True
        action, memory = agent.get_action(obs, task)
        action = env.agent_action_to_env_action(action)
        action["drop"] = np.array(0)
        action["inventory"] = np.array(0)
        action["use"] = np.array(0)
        for i in range(9):
            action[f"hotbar.{i + 1}"] = np.array(0)

        if "dig down" in task:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)
            if not STATE["look_down_once"]:
                pickaxe = env.find_best_pickaxe()
                helper["equip"].equip_item(pickaxe)
                helper["craft"]._look_down()
                STATE["look_down_once"] = True
            action["attack"] = np.array(1)

        if action["attack"] > 0:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)

        obs, reward, terminated, truncated, info = env.step(action)

    check, count = check_inventory(info["inventory"], goal["item"], goal["count"])
    if check:
        STATE["look_down_once"] = False
    return obs, info, check


class MethodFailure(Exception):
    pass


def is_method_exception(exc: BaseException) -> bool:
    frames = traceback.extract_tb(exc.__traceback__)
    return bool(frames) and frames[-1].filename.startswith(METHOD_DIRS)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--order-id", default="order0")
    ap.add_argument("--prompt-form", default="obtain", choices=["obtain", "verbatim", "get"])
    ap.add_argument("--biome", default="mineevolve", choices=["mineevolve", "released"],
                    help="mineevolve: group biome from MineEvolve yamls; released: always forest (gui_server)")
    args = ap.parse_args()

    uid = args.task
    task = TASKS[uid]
    group = task["group"]
    seed = int(args.seed if args.seed is not None else SEEDS[uid])
    horizon = int(load_yaml("tasks.yaml")["horizon_minutes"][group]) * MINUTE
    biome = MINEEVOLVE_BIOME[group] if args.biome == "mineevolve" else "forest"
    ep = Path(args.episode_dir)
    ep.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    import numpy as np
    import torch

    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)

    status, err, end_reason = "finished", None, None
    plan_raw, subgoals, goals, sg_done, native_success = None, [], [], 0, None
    mon = None
    env = None
    llm = None
    t_loop = None
    ptext = planner_text(uid, args.prompt_form)
    events = open(ep / "events.jsonl", "a")

    def event(**kw):
        kw.setdefault("step", mon.steps if mon else 0)
        kw["t"] = round(time.time() - t_start, 1)
        events.write(json.dumps(kw) + "\n"); events.flush()

    try:
        t0 = time.time()
        agent = load_agent("cuda")
        event(kind="model_loaded", load_s=round(time.time() - t0, 1))
        llm = LLMLogger(agent, ep / "llm_calls.jsonl")

        # --- planning (gui_server /send_text task=planning) ---
        llm.caller = "plan"
        plan_raw, subgoals, goals = agent.plan(ptext)
        (ep / "plan.json").write_text(json.dumps({"planner_input": ptext, "raw": plan_raw, "subgoals": subgoals,
                                                  "goals": goals}, indent=2))
        event(kind="plan", n_subgoals=len(subgoals), n_goals=len(goals))

        # --- env (gui_server /reset) ---
        from minestudio.models import CraftWorker, EquipWorker, SmeltWorker
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks.commands import CommandsCallback

        mon = EpisodeMonitor(ep, uid, horizon)
        cb = make_monitor_callback(mon, horizon)
        llm.step_fn = lambda: mon.steps
        random.seed(seed)
        env = MinecraftSim(obs_size=(128, 128), preferred_spawn_biome=biome, seed=seed, inventory={},
                           callbacks=[CommandsCallback(RESET_COMMANDS), cb])
        obs, info = env.reset()
        helper = {"craft": CraftWorker(env), "smelt": SmeltWorker(env), "equip": EquipWorker(env)}
        cb.active = True
        event(kind="env_ready", reset_s=round(time.time() - t0, 1),
              pos=[float(np.asarray(info["location_stats"][k])) for k in ("xpos", "ypos", "zpos")]
              if "location_stats" in info else None)

        # --- action loop (gui_server /send_text task=action, repeated) ---
        t_loop = time.time()
        i = 0
        agent.task = None
        STATE["look_down_once"] = False
        while i < len(subgoals):
            if agent.task is None:
                llm.caller = "action_reset"
                agent.reset(subgoals[i])
                event(kind="subgoal_start", i=i, subgoal=subgoals[i],
                      goal=goals[i] if i < len(goals) else None,
                      task_token=llm.records[-1]["response"] if llm.records else None)
            if i >= len(goals):  # released server: IndexError on every /send_text tick, forever
                raise MethodFailure(f"no parsed goal for subgoal {i} ({len(goals)} goals, {len(subgoals)} subgoals)")
            obs, info, check = released_step(env, agent, obs, subgoals[i], goals[i], helper)
            if check:
                event(kind="subgoal_done", i=i)
                i += 1
                sg_done = i
                agent.task = None
        native_success = True if subgoals else None  # GUI answers "success" once the plan is exhausted
        end_reason = "plan_exhausted" if subgoals else "plan_empty"
        event(kind="plan_exhausted")
    except EpisodeStop as e:
        end_reason = mon.end_reason or str(e)
        native_success = False if subgoals else None
    except BaseException as e:  # noqa: BLE001
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        if mon is not None and (mon.steps >= horizon or mon.success_step is not None):
            end_reason = "success" if mon.success_step is not None else "horizon"
        elif isinstance(e, MethodFailure) or is_method_exception(e):
            end_reason = "method_exception"  # released algorithm failed (D25)
        else:
            status = "crashed"
        native_success = False if subgoals else None
    finally:
        if mon is not None:
            mon.close()
        if env is not None:
            try:
                env.close()
            except BaseException:  # noqa: BLE001
                pass
        events.close()

    steps = mon.steps if mon else 0
    loop_s = (time.time() - t_loop) if t_loop else None
    result = {
        "env": "C3", "method": "optimus3", "task": uid, "group": group, "instruction": task["instruction"],
        "planner_input": ptext, "prompt_form": args.prompt_form, "biome": biome,
        "seed": seed, "order_id": args.order_id, "status": status, "end_reason": end_reason,
        "success": bool(mon and mon.success_step is not None), "success_step": mon.success_step if mon else None,
        "native_success": native_success, "steps": steps, "horizon_steps": horizon,
        "wall_time_s": round(time.time() - t_start, 1),
        "steps_per_s": round(steps / loop_s, 2) if loop_s and steps else None,
        "final_inventory": mon.last_inventory if mon else {},
        "n_subgoals": len(subgoals), "subgoals_completed": sg_done, "subgoals": subgoals,
        **(llm.summary() if llm else {"llm_calls": 0}),
        "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "error": err, "disk_free_gb": round(disk_free_gb(), 1),
    }
    write_result(ep, result)
    print(json.dumps({k: result[k] for k in ("task", "status", "end_reason", "success", "steps", "wall_time_s")}))
    return 0 if status != "crashed" else 3


if __name__ == "__main__":
    sys.exit(main())
