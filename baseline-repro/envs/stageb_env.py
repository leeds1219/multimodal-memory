"""Environment interface for the Stage B ports (JARVIS-1, DEPS).

Neither method ships an environment we can run, so each port talks to the
evaluation env through this small API, implemented once per env with exactly
the pieces that env's native method uses:

  Env M: MineEvolve's env spec (+ per-slot inventory, D19) and wrapper (ore
         bands, auto-pickaxe), functional craft/smelt primitive, STEVE-1 via
         the MineEvolve server's action route (cond_scale 4.0, PNG frames,
         state reset when the prompt text changes).
  Env O: Optimus-1's env spec and CustomEnvWrapper (ore per STEVE-1 step,
         auto-pickaxe, /kill heuristics), Optimus-1's GUI craft/smelt/equip
         helpers, STEVE-1 via the Optimus-1 server's action route
         (cond_scale 6.0, JPEG frames, state reset per episode).

API
    env.reset() -> obs
    env.inventory() -> {item: count}
    env.steve(prompt, max_steps, stop=lambda inv: bool) -> (stopped, steps)
    env.craft(item, n) / env.smelt(item, n) / env.equip(item) -> (ok, info)
    env.over -> bool (horizon reached or died)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from common import EpisodeEnd, EpisodeMonitor

MIN = 1200
log = logging.getLogger("baseline_repro.stageb_env")


class _Base:
    horizon: int
    mon: EpisodeMonitor

    @property
    def over(self) -> bool:
        return self.mon.over or self.mon.steps >= self.horizon

    def inventory(self) -> Dict[str, int]:
        return dict(self.mon.last_inventory)

    def _check(self) -> None:
        if self.over:
            raise EpisodeEnd(self.mon.end_reason or "horizon")


# =============================================================================
class EnvM(_Base):
    def __init__(self, group: str, seed: int, mon: EpisodeMonitor, port: int, logger) -> None:
        from hydra import compose, initialize_config_dir
        from mineevolve.client.server_api import MineEvolveClient
        from mineevolve.env import custom_env as me_spec
        from mineevolve.env import make_env
        from mineevolve.main import _safe_pov
        from optimus1.env.plain_inventory import PlainInventoryObservation
        from functional_craft import CraftRequest, FunctionalCraftHelper

        conf = "/home/rag/data/multimodal-memory/MC-MineEvolve/src/mineevolve/conf"
        with initialize_config_dir(config_dir=conf, version_base=None):
            cfg = compose(config_name="evaluate", overrides=[f"benchmark={group}", f"server.port={port}",
                                                            "record.video.save=false", "record.evidence.save=false"])
        cfg_env = cfg.benchmark.env if "benchmark" in cfg else cfg.env
        self.horizon = int(cfg_env.max_minutes) * MIN
        self.biome = str(cfg_env.prefer_biome)
        self.mon = mon
        mon.horizon = self.horizon
        _orig = me_spec.MineEvolveBaseSpec.create_observables
        me_spec.MineEvolveBaseSpec.create_observables = lambda s: list(_orig(s)) + [PlainInventoryObservation()]
        self.env = make_env(cfg, logger=logger)
        inner = self.env.env.step

        def monitored(action):
            obs, r, done, info = inner(action)
            mon.on_step(action, obs, done, info)
            return obs, r, done, info

        self.env.env.step = monitored
        self.env.env.execute_cmd = None  # stock-MineRL chat path (see mineevolve_episode.py)
        self.env.seed(seed)
        self.client = MineEvolveClient(base_url=f"{cfg.server.url}:{port}", timeout=float(cfg.server.timeout))
        self._pov = _safe_pov
        self._Req, self._FC = CraftRequest, FunctionalCraftHelper
        self.obs = None

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def steve(self, prompt: str, max_steps: int, stop: Optional[Callable[[Dict[str, int]], bool]] = None) -> Tuple[bool, int]:
        n = 0
        while n < max_steps:
            self._check()
            r = self.client.action(condition=prompt, obs={"image": self._pov(self.obs), "pov": self._pov(self.obs)})
            action = r.get("action")
            if action is None:
                raise RuntimeError(f"STEVE-1 server returned no action: {r.get('error')}")
            self.obs, _, done, _ = self.env.step(action, subgoal=None)
            n += 1
            if stop is not None and stop(self.inventory()):
                return True, n
            if done:
                self._check()
        return False, n

    def noop(self, n: int) -> None:
        for _ in range(max(0, int(n))):
            self._check()
            self.obs, _, _, _ = self.env.step(self.env.env.action_space.noop(), subgoal=None)

    def _helper(self, kind: str, item: str, n: int) -> Tuple[bool, Optional[str]]:
        self._check()
        h = self._FC(self.env)
        ok = h.execute(self._Req(kind, item, n))
        self.obs = self.env.cache_obs
        return ok, (None if ok else h.last_error)

    def craft(self, item: str, n: int = 1):
        return self._helper("mc_craft", item, n)

    def smelt(self, item: str, n: int = 1):
        return self._helper("mc_smelt", item, n)

    def equip(self, item: str):
        # Env M has no equip primitive (MineEvolve helper semantics: holding the
        # item is enough; pickaxes are auto-selected below y=70).
        ok = self.inventory().get(item, 0) > 0
        return ok, (None if ok else f"no {item} in inventory")

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# =============================================================================
class EnvO(_Base):
    def __init__(self, group: str, seed: int, mon: EpisodeMonitor, port: int, logger) -> None:
        import os
        import gym
        from omegaconf import OmegaConf
        sys.path.append("/home/rag/data/official/NeurIPS24-Optimus-1")
        from optimus1.env import register_custom_env
        from optimus1.env.wrapper import CustomEnvWrapper
        from optimus1.helper import Helper
        from optimus1.util import ServerAPI
        from cross_glue import env_o_group

        wd = Path("/home/rag/data/repro_runs/state/_optimus_helper_cwd")
        (wd / "src" / "optimus1").mkdir(parents=True, exist_ok=True)
        if not (wd / "src" / "optimus1" / "helper").exists():
            (wd / "src" / "optimus1" / "helper").symlink_to("/home/rag/data/official/NeurIPS24-Optimus-1/src/optimus1/helper")
        os.chdir(wd)
        genv = env_o_group(group)
        self.horizon = int(genv["max_minutes"]) * MIN
        self.biome = str(genv["prefer_biome"])
        self.mon = mon
        mon.horizon = self.horizon
        me_cmds = OmegaConf.load("/home/rag/data/multimodal-memory/MC-MineEvolve/src/mineevolve/conf/evaluate.yaml").commands
        cfg = OmegaConf.create({
            "env": {"name": f"ReproO_{group}-v0", "prefer_biome": genv["prefer_biome"], "initial_inventory": [],
                    "max_minutes": genv["max_minutes"], "times": 1},
            "record": {"video": {"save": False, "sub_task": False, "path": "videos", "name": "x"},
                       "action": {"save": False, "sub_task": False}},
            "commands": list(me_cmds), "version": "v1", "type": "headless",
        })
        register_custom_env(cfg)
        raw = gym.make(cfg.env.name)
        inner = raw.step

        def monitored(action):
            obs, r, done, info = inner(action)
            mon.on_step(action, obs, done, info)
            return obs, r, done, info

        raw.step = monitored
        self.env = CustomEnvWrapper(raw, cfg, logger)
        raw.seed(seed)
        self._Helper = Helper  # built after reset: its constructor steps the env (as in optimus1.main)
        self.helper = None
        self.server = OmegaConf.create({"url": "http://127.0.0.1", "port": port, "timeout": 2000})
        self.api = ServerAPI
        self.obs = None

    def reset(self):
        t = self.api.reset(self.server)
        self.obs = self.env.reset()
        t.join()
        self.helper = self._Helper(self.env)
        return self.obs

    def steve(self, prompt, max_steps, stop=None, goal=None):
        n = 0
        while n < max_steps:
            self._check()
            self.env._only_once = True  # Optimus-1 ore rule: once per STEVE-1 step
            action = self.api.get_action(self.server, self.obs, prompt, step=self.env.num_steps)
            self.obs, _, done, _ = self.env.step(action, goal)
            n += 1
            if stop is not None and stop(self.inventory()):
                return True, n
            if done:
                self._check()
        return False, n

    def noop(self, n: int) -> None:
        for _ in range(max(0, int(n))):
            self._check()
            self.obs, _, _, _ = self.env.step(self.env.env.noop_action(), None)

    def _helper(self, task: str, goal: tuple):
        from rich.progress import Progress
        self._check()
        self.env.can_change_hotbar = self.env.can_open_inventory = True
        try:
            with Progress(disable=True) as pbar:
                tid = pbar.add_task("helper", total=1)
                self.helper.reset(task, pbar, tid, log)
                ok, info = self.helper.step(task, goal)
        except RuntimeError as e:
            if "Timeout" in str(e):
                raise EpisodeEnd("horizon")
            raise
        finally:
            self.env.can_change_hotbar = self.env.can_open_inventory = False
        if self.mon.last_obs is not None:
            self.obs = self.mon.last_obs
        return bool(ok), info

    def craft(self, item, n=1):
        return self._helper(f"craft {item}", (item, n))

    def smelt(self, item, n=1):
        return self._helper(f"smelt {item}", (item, n))

    def equip(self, item):
        return self._helper(f"equip {item}", (item, 1))

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_env(env: str, group: str, seed: int, mon: EpisodeMonitor, port: int, logger) -> _Base:
    return (EnvM if env == "M" else EnvO)(group, seed, mon, port, logger)
