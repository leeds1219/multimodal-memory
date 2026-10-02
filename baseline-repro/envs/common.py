"""Shared episode plumbing used by every method in both environments.

* configs (tasks, targets, seeds, orders) loaded once;
* ``TargetChecker``: the paper success criterion (target items in inventory);
* ``EpisodeMonitor``: wraps an env's ``step`` to count env steps, record a
  compact trajectory, check success every step, and flag the end of the
  episode (MineRL ``done`` = horizon or death);
* ``GameTime``: stands in for the ``time`` module where a method measures
  timeouts in seconds, so that 1 s = 20 env ticks (1200 steps = 1 minute);
* episode context file for the shared LLM layer;
* disk guard.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time as _time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import yaml

REPRO = Path(__file__).resolve().parents[1]
# REPRO_SUITE selects the task suite: unset = the 70 MCU tasks (configs/*.yaml);
# "optimus1" = the Optimus-1 paper's own tasks (configs/suites/optimus1/). Each
# suite writes to its own runs root so results never mix.
SUITE = os.environ.get("REPRO_SUITE", "")
CONFIGS = REPRO / "configs" / "suites" / SUITE if SUITE else REPRO / "configs"
RUNS_ROOT = Path("/home/rag/data/repro_runs") / (f"suite_{SUITE}" if SUITE else "")
MIN_FREE_GB = 5.0


def load_yaml(name: str) -> dict:
    return yaml.safe_load(open(CONFIGS / name))


TASKS = {t["uid"]: t for t in load_yaml("tasks.yaml")["tasks"]}
TARGETS = load_yaml("task_targets.yaml")["targets"]
SEEDS = load_yaml("seeds.yaml")["seeds"]
ORDERS = load_yaml("task_orders.yaml")["orders"]


class EnvDead(BaseException):
    """The Minecraft instance died (socket timeout / stepping a closed env).
    BaseException so a method's ``except Exception`` cannot treat it as an
    ordinary subgoal failure; the episode is recorded as an infra crash."""


class EpisodeEnd(BaseException):
    """Raised to leave a method's loop once the env reported done.

    Derives from BaseException so the methods' own ``except Exception``
    handlers do not swallow it."""


def disk_free_gb(path: str | os.PathLike = "/home/rag/data") -> float:
    return shutil.disk_usage(path).free / 1e9


def write_ctx(ctx_file: str | os.PathLike, ctx: Mapping[str, Any]) -> None:
    tmp = Path(str(ctx_file) + ".tmp")
    tmp.write_text(json.dumps(dict(ctx)))
    os.replace(tmp, ctx_file)


# ----------------------------------------------------------------------------
# Success criterion
# ----------------------------------------------------------------------------

class TargetChecker:
    def __init__(self, task_uid: str) -> None:
        from functional_craft import RecipeBook  # same tag data as the craft primitive

        spec = TARGETS[task_uid]
        self.count = int(spec["count"])
        self.items: list[str] = []
        self.suffixes: list[str] = []
        tags = RecipeBook.tags_only()
        for it in spec["any_of"]:
            if it == "#saplings":
                self.suffixes.append("_sapling")
            elif it.startswith("#"):
                self.items += tags.get(it[1:], [])
            else:
                self.items.append(it)

    def amount(self, inventory: Mapping[str, Any]) -> int:
        n = 0
        for k, v in (inventory or {}).items():
            k = str(k).replace("minecraft:", "")
            if k in self.items or any(k.endswith(s) for s in self.suffixes):
                n += int(v)
        return n

    def satisfied(self, inventory: Mapping[str, Any]) -> bool:
        return self.amount(inventory) >= self.count


# ----------------------------------------------------------------------------
# Per-step monitor
# ----------------------------------------------------------------------------

def _inventory_of(obs: Mapping[str, Any]) -> Dict[str, int]:
    inv = obs.get("inventory") if isinstance(obs, Mapping) else None
    if not isinstance(inv, Mapping):
        return {}
    out = {}
    for k, v in inv.items():
        try:
            n = int(np.asarray(v).sum())
        except (TypeError, ValueError):
            continue
        if n > 0:
            out[str(k)] = n
    return out


def _compact_action(action: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (action or {}).items():
        if k == "camera":
            cam = np.asarray(v, dtype=float).reshape(-1).tolist()
            if any(abs(c) > 1e-6 for c in cam):
                out[k] = [round(c, 2) for c in cam]
        elif k == "chat":
            if v:
                out[k] = str(v)
        else:
            try:
                if int(np.asarray(v).sum()) != 0:
                    out[k] = 1
            except (TypeError, ValueError):
                pass
    return out


class EpisodeMonitor:
    """Counts steps, records a gzip JSONL trajectory, checks success."""

    KEYFRAME_EVERY = 100

    def __init__(self, episode_dir: Path, task_uid: str, horizon_steps: int) -> None:
        self.dir = Path(episode_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checker = TargetChecker(task_uid)
        self.horizon = int(horizon_steps)
        self.steps = 0
        self.success_step: Optional[int] = None
        self.over = False
        self.end_reason: Optional[str] = None
        self.last_inventory: Dict[str, int] = {}
        self.last_pos: Optional[list] = None
        self.first_pos: Optional[list] = None  # spawn point (escapefix go_to_land)
        self.last_obs: Any = None
        self.deaths = 0
        self._traj = gzip.open(self.dir / "trajectory.jsonl.gz", "wt")
        self._t0 = _time.time()

    def on_step(self, action: Mapping[str, Any], obs: Mapping[str, Any], done: bool, info: Mapping[str, Any] | None = None) -> None:
        self.steps += 1
        self.last_obs = obs
        if isinstance(obs, Mapping) and "pov" in obs and (self.steps == 1 or self.steps % self.KEYFRAME_EVERY == 0 or done):
            try:
                from PIL import Image
                img = Image.fromarray(np.asarray(obs["pov"], dtype=np.uint8))
                if self.steps == 1:  # full-res first frame, for the same-seed fairness check
                    img.save(self.dir / "first_frame.png")
                kf = self.dir / "keyframes"
                kf.mkdir(exist_ok=True)  # low-res keyframes for visual failure analysis
                img.resize((320, 180)).save(kf / f"{self.steps:06d}.jpg", quality=70)
            except Exception:
                pass
        inv = _inventory_of(obs)
        rec: Dict[str, Any] = {"t": self.steps}
        a = _compact_action(action)
        if a:
            rec["a"] = a
        if inv != self.last_inventory:
            rec["inv"] = inv
            self.last_inventory = inv
        loc = obs.get("location_stats") if isinstance(obs, Mapping) else None
        if isinstance(loc, Mapping):
            pos = [round(float(np.asarray(loc.get(k, 0))), 1) for k in ("xpos", "ypos", "zpos")]
            if self.first_pos is None:
                self.first_pos = pos
            if pos != self.last_pos:
                rec["pos"] = pos
                self.last_pos = pos
        life = obs.get("life_stats") if isinstance(obs, Mapping) else None
        if isinstance(life, Mapping) and self.steps % 20 == 0:
            rec["hp"] = round(float(np.asarray(life.get("life", 0))), 1)
        if self.success_step is None and self.checker.satisfied(inv):
            self.success_step = self.steps
            rec["success"] = True
        if done:
            self.over = True
            self.end_reason = "horizon" if self.steps >= self.horizon else "death_or_done"
            rec["done"] = self.end_reason
        self._traj.write(json.dumps(rec) + "\n")
        if self.steps % 100 == 0 or done:
            (self.dir / "heartbeat").write_text(f"{self.steps} {_time.time():.0f}\n")

    def close(self) -> None:
        try:
            self._traj.close()
        except Exception:
            pass

    @property
    def wall_time(self) -> float:
        return _time.time() - self._t0


class GameTime:
    """``time``-module stand-in: monotonic()/time() advance with env steps."""

    TICKS_PER_S = 20.0

    def __init__(self, monitor: EpisodeMonitor) -> None:
        self._m = monitor

    def monotonic(self) -> float:
        return self._m.steps / self.TICKS_PER_S

    def __getattr__(self, name: str):
        return getattr(_time, name)


# ----------------------------------------------------------------------------
# LLM usage summary for an episode
# ----------------------------------------------------------------------------

def llm_summary(episode_dir: Path) -> Dict[str, Any]:
    f = Path(episode_dir) / "llm" / "calls.jsonl"
    s = {"llm_calls": 0, "tokens_in": 0, "tokens_out_billed": 0, "cost_usd": 0.0, "llm_errors": 0,
         "llm_truncated": 0, "llm_latency_s": 0.0, "llm_calls_by_caller": {}}
    if not f.exists():
        return s
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        s["llm_calls"] += 1
        s["tokens_in"] += r.get("tokens_in") or 0
        s["tokens_out_billed"] += r.get("tokens_out_billed") or 0
        s["cost_usd"] += r.get("cost_usd") or 0.0
        s["llm_latency_s"] += r.get("latency_s") or 0.0
        s["llm_errors"] += 1 if r.get("error") else 0
        s["llm_truncated"] += 1 if r.get("finish_reason") == "length" else 0
        fn = str(r.get("caller") or "?").split(":")[-1]
        s["llm_calls_by_caller"][fn] = s["llm_calls_by_caller"].get(fn, 0) + 1
    s["cost_usd"] = round(s["cost_usd"], 5)
    s["llm_latency_s"] = round(s["llm_latency_s"], 1)
    return s


def write_result(episode_dir: Path, result: Mapping[str, Any]) -> None:
    tmp = Path(episode_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(dict(result), indent=2))
    os.replace(tmp, Path(episode_dir) / "result.json")


METHOD_CODE_DIRS = ("/home/rag/data/official/NeurIPS24-Optimus-1/src/optimus1/",
                    "/home/rag/data/multimodal-memory/MC-MineEvolve/src/mineevolve/")


def is_method_exception(exc: BaseException) -> bool:
    """True if the innermost frame of the traceback is in the method's own code
    (the released algorithm failed), False for env / server / glue errors."""
    import traceback as _tb
    frames = _tb.extract_tb(exc.__traceback__)
    return bool(frames) and frames[-1].filename.startswith(METHOD_CODE_DIRS)

