"""Interface glue that runs a method inside the *other* method's environment.

An environment = MineRL env spec (biome, horizon, observations) + env-level
behaviour (reset commands, ore spawning, auto-pickaxe, kill heuristics) +
craft/smelt/equip primitive. The method's own agent loop, prompts, memory and
STEVE-1 wrapper are untouched.

* ``optimus_in_M``: Optimus-1's main loop and CustomEnvWrapper on Env M:
  MineEvolve's env spec (+ per-slot inventory, DECISIONS D19) with Env M's
  group biome / horizon; Env M ore bands on every env step; no ``/kill``
  heuristics and no Optimus ore rule; chat commands are env steps (as in
  MineEvolve's wrapper); craft/smelt/equip through the functional primitive,
  reporting failures in Optimus-1's own ``missing material: {...}`` format.
  Optimus-1's pickaxe auto-select is kept: it is the same rule as Env M's
  (best pickaxe in hotbar when y < 70).

* ``mineevolve_in_O``: MineEvolve's main loop and wrapper on Env O: the
  Optimus-1 env spec with Env O's group biome / horizon, wrapped by
  Optimus-1's CustomEnvWrapper (its ore rule per STEVE-1 step, auto-pickaxe,
  ``/kill`` heuristics), with MineEvolve's wrapper on top (its own ore /
  pickaxe logic disabled so nothing runs twice) and MineEvolve's craft/smelt
  requests executed by Optimus-1's scripted GUI helpers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np

from functional_craft import CraftRequest, FunctionalCraftHelper, RecipeBook

logger = logging.getLogger("baseline_repro.cross_glue")
MINUTE = 1200
ME_BENCH = "/home/rag/data/multimodal-memory/MC-MineEvolve/src/mineevolve/conf/benchmark"
O_BENCH = "/home/rag/data/official/NeurIPS24-Optimus-1/src/optimus1/conf/benchmark"
O_GROUP = {"wooden": "wooden", "stone": "stone", "iron": "iron", "gold": "golden",
           "redstone": "redstone", "diamond": "diamond", "armor": "armor"}


def _bench(path: str, group: str) -> dict:
    import yaml
    return yaml.safe_load(open(f"{path}/{group}.yaml"))


def env_m_group(group: str) -> dict:
    return _bench(ME_BENCH, group)["env"]


def env_o_group(group: str) -> dict:
    return _bench(O_BENCH, O_GROUP[group])["env"]


# =============================================================================
# Optimus-1 in Env M
# =============================================================================

class FunctionalOptimusHelper:
    """Optimus-1 ``Helper`` API on top of the functional craft primitive."""

    def __init__(self, env, inventory_fn, chat_fn, noop_fn) -> None:
        self.env = env
        self.fc = FunctionalCraftHelper(env, execute_cmd=chat_fn, inventory=inventory_fn, noop_step=noop_fn)
        self._inv = inventory_fn
        self.steps = 0

    def reset(self, task, pbar, task_id, logger_):
        self.steps = 0

    def get_task_steps(self, task: str):
        return self.steps

    def _missing(self, target: str) -> str:
        err = self.fc.last_error
        if "crafting_table not in inventory" in err:
            need = {"crafting_table": 1}
        elif "furnace not in inventory" in err:
            need = {"furnace": 1}
        elif "no fuel" in err:
            need = {"coal": 2}  # Optimus-1 smelt helper reports 2 of its fuel type
        else:
            need = self._one_craft_ingredients(target)
        return "missing material: " + json.dumps(need)

    def _one_craft_ingredients(self, target: str) -> Dict[str, int]:
        book = self.fc.book
        for t in book.resolve(target):
            for r in book.by_result.get(t, []):
                need: Dict[str, int] = {}
                if r["_kind"] == "crafting_shaped":
                    for row in r["pattern"]:
                        for ch in row:
                            if ch == " ":
                                continue
                            spec = r["key"][ch]
                            spec = spec[0] if isinstance(spec, list) else spec
                            name = (spec.get("item") or spec.get("tag")).split(":")[-1]
                            need[name] = need.get(name, 0) + 1
                elif r["_kind"] == "crafting_shapeless":
                    for spec in r["ingredients"]:
                        spec = spec[0] if isinstance(spec, list) else spec
                        name = (spec.get("item") or spec.get("tag")).split(":")[-1]
                        need[name] = need.get(name, 0) + 1
                else:
                    spec = r["ingredient"]
                    spec = spec[0] if isinstance(spec, list) else spec
                    need[(spec.get("item") or spec.get("tag")).split(":")[-1]] = 1
                return need
        return {target: 1}

    def step(self, task: str, goal):
        before = self.env.num_steps if hasattr(self.env, "num_steps") else 0
        target, num = str(goal[0]), int(goal[1])
        if "equip" in task:
            # Env M has no equip primitive; holding the item is what counts
            # (MineEvolve helper semantics), pickaxes are auto-selected.
            ok = self._inv().get(target, 0) > 0
            return (True, None) if ok else (False, f'missing material: {{"{target}": 1}}')
        kind = "mc_smelt" if "smelt" in task else "mc_craft" if "craft" in task else None
        if kind is None:
            return False, "not support"
        ok = self.fc.execute(CraftRequest(kind, target, num))
        self.steps = len(self.fc.commands)
        return (True, None) if ok else (False, self._missing(target))


def optimus_in_M(om, mon, group: str, seed: int) -> int:
    """Patch ``optimus1.main`` so its episode runs in Env M. Returns horizon."""
    import optimus1.env.wrapper as ow
    from mineevolve.env import custom_env as me_spec
    from mineevolve.env.wrapper import DynamicOreSpawnMixin
    from optimus1.env.plain_inventory import PlainInventoryObservation

    genv = env_m_group(group)
    horizon = int(genv["max_minutes"]) * MINUTE

    from minerl.herobraine.hero import handlers as mh

    # Observation-only additions Optimus-1's code reads (per-slot inventory,
    # GUI flag); they do not change the world. HumanSurvival has no IsGuiOpen.
    _orig_obs = me_spec.MineEvolveBaseSpec.create_observables
    me_spec.MineEvolveBaseSpec.create_observables = lambda self: list(_orig_obs(self)) + [
        PlainInventoryObservation(), mh.IsGuiOpen()]

    def register_custom_env(cfg):
        me_spec.register_mineevolve_env(env_name=str(cfg["env"]["name"]), prefer_biome=str(genv["prefer_biome"]),
                                        max_minutes=int(genv["max_minutes"]))

    om.register_custom_env = register_custom_env

    class _OreM(DynamicOreSpawnMixin):
        def __init__(self, chat):
            self.execute_cmd = chat
            self._reset_ore_state()

    class OptimusWrapperEnvM(ow.CustomEnvWrapper):
        """Optimus-1 wrapper step with Env M's env-level behaviour."""

        def step(self, action, goal=None, prompt=None):
            # --- same as CustomEnvWrapper.step up to the env step ---
            if not self.can_change_hotbar:
                for i in range(9):
                    action[f"hotbar.{i+1}"] = np.array(0)
                action["use"] = np.array(0)
                action["inventory"] = np.array(0)
                hotbar = self.find_best_pickaxe()
                if hotbar:
                    action[hotbar] = np.array(1)
            if not self.can_open_inventory:
                action["inventory"] = np.array(0)
            action["drop"] = np.array(0)
            observation, reward, done, info = self.env.step(action)
            if goal is not None and goal[0] != self.cache["task"]:
                self.task_checker_mod.reset(observation["inventory"])
                self.cache["task"] = goal[0]
            self.record_mod.step(observation, prompt, action)
            self.status_mod.step(observation, action)
            info.update(self.status_mod.get_status())
            ypos = self.status_mod.get_height()
            # --- Env M: ore bands every step; no /kill heuristics, no O ore rule ---
            self._ore_m._maybe_spawn_ore(self.env, int(ypos))
            try:
                self._current_task_finish = self.task_checker_mod.step(observation["inventory"], goal)
            except Exception as e:
                print("Error ", e)
                self._current_task_finish = True
            if self._current_task_finish:
                self.cache["task"] = ""
            info["isGuiOpen"] = observation["isGuiOpen"]
            self.cache["info"] = info
            return observation, reward, done, info

        def reset(self):
            self._ore_m._reset_ore_state()
            return super().reset()

    def env_make(env_id, cfg, logger_):
        import gym
        raw = gym.make(env_id)
        inner_step = raw.step

        def monitored_step(action):
            obs, reward, done, info = inner_step(action)
            mon.on_step(action, obs, done, info)
            return obs, reward, done, info

        def chat(cmd: str):  # Env M: a chat command is an ordinary env step
            a = raw.action_space.noop()
            a["chat"] = cmd
            return raw.step(a)

        raw.step = monitored_step
        raw.execute_cmd = chat
        # Optimus-1's own entry point wraps every env in BasaltTimeoutWrapper
        # (provides env.timeout / env.num_steps that its main loop reads).
        env = OptimusWrapperEnvM(ow.BasaltTimeoutWrapper(raw), cfg, logger_)
        env._ore_m = _OreM(chat)
        env._ore_m._reset_ore_state()
        env.seed(seed)
        return env

    om.env_make = env_make

    def make_helper(env):
        return FunctionalOptimusHelper(
            env, inventory_fn=lambda: dict(mon.last_inventory),
            chat_fn=lambda c: env.env.execute_cmd(c),
            noop_fn=lambda: env.env.step(env.env.action_space.noop()),
        )

    om.Helper = make_helper
    return horizon


# =============================================================================
# MineEvolve in Env O
# =============================================================================

class OptimusGuiCraftAdapter:
    """MineEvolve ``CraftHelper`` API executed by Optimus-1's GUI helpers."""

    o_env = None  # set by mineevolve_in_O
    m_env = None
    mon = None

    def __init__(self, env: Any) -> None:
        from optimus1.helper import Helper
        self.helper = Helper(self.o_env)
        self.last_error = ""

    def execute(self, req) -> bool:
        from rich.progress import Progress

        kind = str(req.kind)
        target, n = str(req.target), max(1, int(req.quantity))
        if kind in ("place", "use"):
            ok = self.mon.last_inventory.get(target, 0) > 0 or any(
                k.endswith(target) for k in self.mon.last_inventory)
            return ok
        task = {"mc_craft": f"craft {target}", "mc_smelt": f"smelt {target}"}.get(kind)
        if task is None:
            return False
        o = self.o_env
        o.can_change_hotbar = o.can_open_inventory = True  # as optimus1.main.agent_do
        try:
            with Progress(disable=True) as pbar:
                tid = pbar.add_task("helper", total=1)
                self.helper.reset(task, pbar, tid, logger)
                done, info = self.helper.step(task, (target, n))
        finally:
            o.can_change_hotbar = o.can_open_inventory = False
        self.last_error = "" if done else str(info)
        # the GUI helper stepped the inner env: refresh MineEvolve's view
        m = self.m_env
        if self.mon.last_obs is not None:
            m.status_mod.step(self.mon.last_obs, {})
            m._latest_info = {**(m._latest_info or {}), **m.status_mod.get_status()}
            m.cache_obs = self.mon.last_obs
        logger.info("gui craft %s %s x%d -> %s %s", kind, target, n, done, info)
        return bool(done)


def mineevolve_in_O(me, mon, group: str, seed: int, cfg_m, log) -> tuple:
    """Build MineEvolve's env on Env O. Returns (env, horizon)."""
    import gym
    from omegaconf import OmegaConf
    from mineevolve.env.wrapper import MineEvolveEnvWrapper
    from optimus1.env import register_custom_env
    from optimus1.env.wrapper import CustomEnvWrapper

    import os
    import sys
    from pathlib import Path
    # Optimus-1's GUI helpers read src/optimus1/helper/{recipes,tag_items.json}
    # relative to the cwd: run from a directory that provides that path.
    wd = Path("/home/rag/data/repro_runs/state/_optimus_helper_cwd")
    (wd / "src" / "optimus1").mkdir(parents=True, exist_ok=True)
    link = wd / "src" / "optimus1" / "helper"
    if not link.exists():
        link.symlink_to("/home/rag/data/official/NeurIPS24-Optimus-1/src/optimus1/helper")
    os.chdir(wd)
    if "/home/rag/data/official/NeurIPS24-Optimus-1" not in sys.path:
        sys.path.append("/home/rag/data/official/NeurIPS24-Optimus-1")  # env ids are src.optimus1...
    genv = env_o_group(group)
    horizon = int(genv["max_minutes"]) * MINUTE
    cfg_o = OmegaConf.create({
        "env": {"name": f"ReproO_{group}-v0", "prefer_biome": genv["prefer_biome"], "initial_inventory": [],
                "max_minutes": genv["max_minutes"], "times": 1},
        "record": {"video": {"save": False, "sub_task": False, "path": "videos", "name": "x"},
                   "action": {"save": False, "sub_task": False}},
        "commands": list(OmegaConf.to_container(cfg_m.commands)),  # identical in both envs
        "version": "v1", "type": "headless",
    })
    register_custom_env(cfg_o)
    raw = gym.make(cfg_o.env.name)
    inner_step = raw.step

    def monitored_step(action):
        try:
            obs, reward, done, info = inner_step(action)
        except Exception as exc:  # Minecraft died: infra crash (common.EnvDead)
            from common import EnvDead
            raise EnvDead(repr(exc)) from exc
        mon.on_step(action, obs, done, info)
        return obs, reward, done, info

    raw.step = monitored_step
    o_env = CustomEnvWrapper(raw, cfg_o, log)

    # MineEvolve's wrapper on top; Env O's wrapper already issues the reset
    # commands and handles ore / pickaxe / kill, so switch MineEvolve's off.
    cfg_top = OmegaConf.merge(cfg_m, {"commands": []})

    class MineEvolveOnO(MineEvolveEnvWrapper):
        _subgoal_for_o: Optional[tuple] = None

        def _maybe_spawn_ore(self, env, ypos):
            return None

        def _select_best_pickaxe_slot(self, ypos, plain_inventory):
            return None

        def step(self, action, subgoal=None):
            # Optimus-1 spawns ore once per STEVE-1 step and uses the current
            # goal for its iron-ore kill rule; mirror optimus1.main.agent_do.
            self.env._only_once = True
            self.env._repro_goal = tuple(subgoal) if subgoal else None
            return super().step(action, subgoal=subgoal)

    o_step = o_env.step

    def o_step_with_goal(action, goal=None, prompt=None):
        return o_step(action, goal if goal is not None else getattr(o_env, "_repro_goal", None), prompt)

    o_env.step = o_step_with_goal
    m_env = MineEvolveOnO(o_env, cfg=cfg_top, logger=log)
    raw.seed(seed)

    OptimusGuiCraftAdapter.o_env, OptimusGuiCraftAdapter.m_env, OptimusGuiCraftAdapter.mon = o_env, m_env, mon
    me.CraftHelper = OptimusGuiCraftAdapter
    return m_env, horizon
