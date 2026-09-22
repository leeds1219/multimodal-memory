"""Hydra entry-point: client-side Algorithm 1 main loop.

Run with::

    python -m mineevolve.main benchmark=iron llm=qwen_plus

The server (FastAPI loading STEVE-1 + planner + 4 modules) must already be
running, e.g. via ``scripts/server.sh``. This client process owns the
MineRL environment and orchestrates the per-episode lifecycle in
Algorithm 1 of the paper.
"""

from __future__ import annotations

import logging
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .client import LLMGuardAbort, MineEvolveClient


class EnvCrashed(RuntimeError):
    """Minecraft died under us (socket timeout, MineRL terminated the instance).

    Raised out of the subgoal runners so run_episode aborts and main() relaunches
    the env; treating it as an ordinary subgoal failure sent the repair loop into
    an unbounded LLM-calling spin (episode 49 of the 2026-09-21 block: 7,100 calls,
    ~$41, in 6.5 h while the env was dead).
    """
from .executor import CraftHelper, CraftRequest
from .monitors import StepMonitor, SuccessMonitor
from .util.evidence import SubgoalEvidenceRecorder, make_subgoal_evidence_dir
from .util.image import encode_pov_to_base64
from .util.items import inventory_satisfies
from .util.logger import info_panel, print_results, setup_logging


logger = logging.getLogger("mineevolve.main")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _get_evaluate_tasks(cfg: DictConfig) -> List[Tuple[int, str, str]]:
    """Resolve a benchmark yaml's ``evaluate`` list into [(id, type, instr)]."""

    benchmark_cfg = _benchmark_cfg(cfg)
    all_tasks = list(benchmark_cfg.all_task)
    id_map = {int(t["id"]): t for t in all_tasks}
    selected = list(benchmark_cfg.get("evaluate") or [])
    if not selected:
        return [(int(t["id"]), str(t["type"]), str(t["instruction"])) for t in all_tasks]
    return [
        (int(i), str(id_map[int(i)]["type"]), str(id_map[int(i)]["instruction"]))
        for i in selected
        if int(i) in id_map
    ]


def _benchmark_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.benchmark if "benchmark" in cfg else cfg


def _state_snapshot(env_info: Mapping[str, Any], task_goal: str) -> Dict[str, Any]:
    state = {
        "task_goal": task_goal,
        "inventory": dict(env_info.get("inventory") or {}),
        "coords": list(env_info.get("coords") or [0, 64, 0]),
        "ypos": int(env_info.get("ypos") or 64),
        "health": float(env_info.get("health") or 20.0),
        "hunger": float(env_info.get("hunger") or 20.0),
        "isGuiOpen": bool(env_info.get("isGuiOpen") or False),
    }
    if os.environ.get("MINEEVOLVE_NEARBY_BLOCKS") == "1":
        # RECONSTRUCTION (off by default): the coordinates behind the upstream prompt
        # example "approach oak tree at (118, 64, 208)". Paper-condition runs keep it off.
        from .env.nearby_blocks import summarize_nearby

        state["nearby_blocks"] = summarize_nearby(env_info.get("nearby_blocks") or {}, state["coords"])
    return state


# ----------------------------------------------------------------------
# Subgoal execution loop
# ----------------------------------------------------------------------


def _move_script(env, params: Mapping[str, Any]):
    """Actions for the `move` executor primitive (see util/vocab.py).

    Turn by yaw/pitch in <= 30 deg camera steps, then walk forward with sprint,
    alternating jump so the agent climbs 1-block ledges. Deterministic; no policy.
    """
    yaw = float(np.clip(float(params.get("yaw_deg", 0) or 0), -180, 180))
    pitch = float(np.clip(float(params.get("pitch_deg", 0) or 0), -45, 45))
    steps = int(np.clip(int(params.get("steps", 40) or 0), 0, 400))
    jump = bool(params.get("jump", True))
    noop = env.action_space.noop()

    def cam(dp: float, dy: float):
        a = dict(noop)
        a["camera"] = np.array([dp, dy], dtype=np.float32)
        return a

    while abs(yaw) > 1e-3 or abs(pitch) > 1e-3:
        dy = float(np.clip(yaw, -30, 30)); dp = float(np.clip(pitch, -30, 30))
        yaw -= dy; pitch -= dp
        yield cam(dp, dy)
    for i in range(steps):
        a = dict(noop)
        a["forward"] = np.array(1); a["sprint"] = np.array(1)
        a["jump"] = np.array(int(jump and i % 2 == 0))
        yield a


class _ApproachScript:
    """Closed-loop `approach` executor primitive: walk to the nearest landmark block.

    Reads the nearest matching block from ``info["nearby_blocks"]`` every step,
    turns toward it (<= 30 deg per step), walks forward with sprint and jumps every
    other step; done when within ``stop_dist`` blocks horizontally. Deterministic;
    no policy. Reconstructs the executor behind the upstream prompt example
    "approach oak tree at (118, 64, 208)" (see env/nearby_blocks.py).
    """

    def __init__(self, env, params: Mapping[str, Any]):
        self.env = env
        self.block = str(params.get("block") or params.get("target") or "log")
        self.stop_dist = float(params.get("stop_dist", 2.0) or 2.0)
        self.max_steps = int(np.clip(int(params.get("steps", 300) or 300), 1, 600))
        self.i = 0
        self.reached = False
        self.target = None

    def next(self, info: Mapping[str, Any]):
        from .env.nearby_blocks import nearest_matching

        if self.i >= self.max_steps or self.reached:
            return None
        self.i += 1
        noop = self.env.action_space.noop()
        tgt = nearest_matching(info.get("nearby_blocks") or {}, self.block)
        if tgt is None:
            if self.target is None:
                return None  # nothing of that kind within the landmark box
            tgt = self.target  # keep the last known position while the block is out of the box
        self.target = tgt
        x, y, z = (info.get("coords") or [0, 64, 0])[:3]
        dx, dz = tgt[0] + 0.5 - x, tgt[2] + 0.5 - z
        dist = float(np.hypot(dx, dz))
        if dist <= self.stop_dist:
            self.reached = True
            return None
        want_yaw = float(np.degrees(np.arctan2(-dx, dz)))          # minecraft: yaw 0 = +z, 90 = -x
        d_yaw = (want_yaw - float(info.get("yaw", 0.0)) + 180.0) % 360.0 - 180.0
        want_pitch = float(np.degrees(np.arctan2(-(tgt[1] - y), dist)))  # look at the block (+ = down)
        d_pitch = float(np.clip(want_pitch, -60, 30) - float(info.get("pitch", 0.0)))
        a = dict(noop)
        a["camera"] = np.array([float(np.clip(d_pitch, -30, 30)), float(np.clip(d_yaw, -30, 30))], dtype=np.float32)
        if abs(d_yaw) < 45:  # walk once roughly facing the target
            a["forward"] = np.array(1); a["sprint"] = np.array(1)
            a["jump"] = np.array(int(self.i % 2 == 0))
        return a


def _ypos_check_ok(subgoal: Mapping[str, Any], info: Mapping[str, Any]) -> bool:
    """True if the subgoal carries a ypos_le / ypos_ge check and the agent satisfies it."""
    checks = [c for c in (subgoal.get("checks") or []) if isinstance(c, Mapping) and c.get("type") in ("ypos_le", "ypos_ge")]
    if not checks:
        return False
    y = info.get("ypos")
    if y is None:
        coords = info.get("coords") or []
        y = coords[1] if len(coords) == 3 else None
    if y is None:
        return False
    return all((float(y) <= float(c.get("n", 0))) if c["type"] == "ypos_le" else (float(y) >= float(c.get("n", 0))) for c in checks)


def _moved_threshold(subgoal: Mapping[str, Any]) -> float | None:
    for c in subgoal.get("checks") or []:
        if isinstance(c, Mapping) and c.get("type") == "moved":
            return float(c.get("n") or 2)
    return None


def _run_subgoal(
    env,
    client: MineEvolveClient,
    subgoal: Mapping[str, Any],
    task_goal: str,
    max_steps: int,
    default_timeout_s: int = 60,
    coord_window_size: int = 256,
    artifact_dir: str | None = None,
    task_id: int | None = None,
    run_idx: int = 0,
    evidence_keyframe_interval: int = 40,
) -> Dict[str, Any]:
    """Execute one subgoal under the STEVE-1 policy until success / timeout."""

    target = None
    for c in subgoal.get("checks") or []:
        if isinstance(c, Mapping) and c.get("type") == "inv_ge":
            target = (str(c.get("item") or ""), int(c.get("n") or 1))
            break
    if target is None:
        target = (str(subgoal.get("condition") or ""), 1)

    coord_history: deque = deque(maxlen=coord_window_size)
    obs = env.cache_obs if hasattr(env, "cache_obs") else None
    if obs is None:
        try:
            obs = env.reset()
        except Exception:
            obs = {}

    start_inv = dict((env.info or {}).get("inventory") or {})
    start_coords = list((env.info or {}).get("coords") or [0, 64, 0])
    start_state = _state_snapshot(env.info or {}, task_goal)
    evidence_dir = make_subgoal_evidence_dir(
        artifact_dir=artifact_dir,
        task_id=task_id,
        run_idx=run_idx,
        subgoal_id=str(subgoal.get("subgoal_id") or ""),
        condition=str(subgoal.get("condition") or ""),
    )
    evidence = (
        SubgoalEvidenceRecorder(
            root=evidence_dir,
            task_goal=task_goal,
            subgoal_id=str(subgoal.get("subgoal_id") or ""),
            condition=str(subgoal.get("condition") or ""),
            target_item=target[0] if target else None,
            keyframe_interval=evidence_keyframe_interval,
        )
        if evidence_dir is not None
        else None
    )

    timeout_s = max(int(subgoal.get("timeout_s") or 0), int(default_timeout_s))
    deadline = time.monotonic() + max(1, timeout_s)

    success = False
    timed_out = False
    died = False
    steps = 0
    executor_hint = str(subgoal.get("executor_hint") or "stevei").strip().lower()
    script = _move_script(env, subgoal.get("params") or {}) if executor_hint == "move" else None
    approach = _ApproachScript(env, subgoal.get("params") or {}) if executor_hint == "approach" else None
    if script is not None or approach is not None:
        logger.info("%s primitive: params=%s", executor_hint, dict(subgoal.get("params") or {}))

    while steps < max_steps:
        if time.monotonic() > deadline:
            timed_out = True
            break

        if approach is not None:
            action = approach.next(env.info or {})
            if action is None:
                break  # reached / no target / step cap; success = reached (set below)
        elif script is not None:
            action = next(script, None)
            if action is None:
                break  # script finished; success decided by the `moved` check below
        else:
            # Ask server for next STEVE-1 action conditioned on the subgoal text.
            try:
                r = client.action(
                    condition=str(subgoal.get("condition") or task_goal),
                    obs={"image": _safe_pov(obs), "pov": _safe_pov(obs)},
                )
            except Exception as exc:
                logger.warning("server.action failed: %s", exc)
                break

            action = r.get("action")
            if action is None:
                logger.warning("server returned empty action: %s", r.get("error"))
                break

        try:
            obs, _reward, done, info = env.step(
                action,
                subgoal=target,
            )
        except Exception as exc:
            logger.exception("env.step crashed: %s", exc)
            raise EnvCrashed(str(exc)) from exc

        steps += 1
        coords = list(info.get("coords") or [0, 64, 0])
        coord_history.append(coords)
        if evidence is not None:
            evidence.record_step(
                step=steps,
                action=action,
                obs=obs if isinstance(obs, Mapping) else {},
                info=info,
                reward=_reward,
                done=done,
                current_subgoal_done=bool(env.current_subgoal_done),
            )

        if env.current_subgoal_done or _ypos_check_ok(subgoal, info):
            success = True
            break
        if done:
            died = True
            break

    end_info = env.info or {}
    end_inv = dict(end_info.get("inventory") or {})
    end_coords = list(end_info.get("coords") or [0, 64, 0])

    if approach is not None and not success and not died:
        success = bool(approach.reached)
        logger.info("approach %s: %s (target %s, %d steps)", approach.block, "reached" if success else "not reached", approach.target, approach.i)
    moved_n = _moved_threshold(subgoal)
    if moved_n is not None and not success and not died:
        dist_xz = float(np.hypot(end_coords[0] - start_coords[0], end_coords[2] - start_coords[2]))
        success = dist_xz >= moved_n
        logger.info("moved check: %.1f blocks (need >= %.1f) -> %s", dist_xz, moved_n, success)

    delta_v = _diff_int_dict(start_inv, end_inv)
    delta_s = {
        "coords_start": start_coords,
        "coords_end": end_coords,
    }
    end_state = _state_snapshot(end_info, task_goal)
    if evidence is not None:
        delta_s["evidence"] = evidence.finalize(
            success=success,
            timed_out=timed_out,
            died=died,
            start_state=start_state,
            end_state=end_state,
            delta_v=delta_v,
            delta_s={k: v for k, v in delta_s.items() if k != "evidence"},
        )

    return {
        "success": success,
        "timed_out": timed_out,
        "died": died,
        "steps": steps,
        "delta_v": delta_v,
        "delta_s": delta_s,
        "coords_history": list(coord_history),
        "end_state": end_state,
        "target_item": target[0] if target else None,
        "subgoal_id": str(subgoal.get("subgoal_id") or ""),
        "condition": str(subgoal.get("condition") or ""),
    }


def _item_in_text(text: str) -> str | None:
    """Longest vanilla item id mentioned in a sentence ('equip the crafting table' -> crafting_table)."""
    from minerl.herobraine.hero.mc import ALL_ITEMS

    t = " " + re.sub(r"[^a-z0-9_ ]", " ", str(text).lower()).replace("_", " ") + " "
    t = re.sub(r"\s+", " ", t)
    best = None
    for item in ALL_ITEMS:
        words = " " + item.replace("_", " ") + " "
        if words in t and (best is None or len(item) > len(best)):
            best = item
    return best


def _run_helper_subgoal(
    env,
    subgoal: Mapping[str, Any],
    task_goal: str,
    artifact_dir: str | None = None,
    task_id: int | None = None,
    run_idx: int = 0,
) -> Dict[str, Any]:
    """Execute one non-STEVE helper subgoal such as mc_craft or mc_smelt."""

    target = None
    params = subgoal.get("params") or {}
    for c in subgoal.get("checks") or []:
        if isinstance(c, Mapping) and c.get("type") == "inv_ge":
            target = (str(c.get("item") or ""), int(c.get("n") or 1))
            break
    if target is None and isinstance(params, Mapping) and params.get("item"):
        target = (str(params["item"]), int(params.get("n") or 1))
    if target is None:
        # equip / place / use subgoals usually carry no inv_ge check: read the item off the sentence
        item = _item_in_text(subgoal.get("condition") or "")
        target = (item, 1) if item else (str(subgoal.get("condition") or ""), 1)

    start_inv = dict((env.info or {}).get("inventory") or {})
    start_coords = list((env.info or {}).get("coords") or [0, 64, 0])
    start_state = _state_snapshot(env.info or {}, task_goal)

    hint = str(subgoal.get("executor_hint") or "").strip().lower()
    helper = CraftHelper(env)
    success = helper.execute(
        CraftRequest(
            kind=hint,
            target=target[0],
            quantity=target[1],
            timeout_s=float(subgoal.get("timeout_s") or 10),
        )
    )
    helper_steps = int(getattr(helper, "last_steps", 0) or 0)
    helper_died = bool(getattr(helper, "episode_ended", False))
    helper_error = str(getattr(helper, "last_error", "") or "")
    if helper_error:
        logger.warning("helper %s: %s", hint, helper_error)

    end_info = env.info or {}
    end_inv = dict(end_info.get("inventory") or {})
    end_coords = list(end_info.get("coords") or start_coords)
    delta_v = _diff_int_dict(start_inv, end_inv)
    delta_s = {
        "coords_start": start_coords,
        "coords_end": end_coords,
        "state_start": start_state,
    }
    end_state = _state_snapshot(end_info, task_goal)
    if helper_error and os.environ.get("MINEEVOLVE_EXECUTOR_ERRORS") == "1":
        # EXTENSION (off by default, not in the paper): what a player would read off the
        # GUI ("not enough planks") is added to the state the Inducer/Adaptor prompts
        # render. The paper's Monitor only emits a failure *type*.
        end_state["last_executor_error"] = helper_error
    evidence_dir = make_subgoal_evidence_dir(
        artifact_dir=artifact_dir,
        task_id=task_id,
        run_idx=run_idx,
        subgoal_id=str(subgoal.get("subgoal_id") or ""),
        condition=str(subgoal.get("condition") or ""),
    )
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "task_goal": task_goal,
            "subgoal_id": str(subgoal.get("subgoal_id") or ""),
            "condition": str(subgoal.get("condition") or ""),
            "target_item": target[0] if target else None,
            "success": bool(success),
            "timed_out": False,
            "died": False,
            "steps": helper_steps,
            "start_state": start_state,
            "end_state": end_state,
            "delta_v": delta_v,
            "delta_s": {k: v for k, v in delta_s.items() if k != "evidence"},
            "event_timeline": [
                {
                    "step": 0,
                    "event": "helper_executed",
                    "executor_hint": hint,
                    "inventory_diff": delta_v,
                    "inventory": end_inv,
                    "coords": end_coords,
                }
            ],
        }
        summary_path = evidence_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        summary["summary_json"] = str(summary_path)
        delta_s["evidence"] = summary

    return {
        "success": bool(success),
        "timed_out": False,
        "died": helper_died,  # env done during the GUI script -> the episode loop stops
        "steps": helper_steps,
        "delta_v": delta_v,
        "delta_s": delta_s,
        "coords_history": [start_coords, end_coords],
        "end_state": end_state,
        "target_item": target[0] if target else None,
        "subgoal_id": str(subgoal.get("subgoal_id") or ""),
        "condition": str(subgoal.get("condition") or ""),
    }


def _safe_pov(obs: Any) -> Any:
    """Wire format for the POV sent to the server each step.

    A base64 PNG is ~50x smaller than ``pov.tolist()`` (6.6 MB of JSON for a
    640x360 frame), which otherwise dominates the per-step latency.
    """
    if isinstance(obs, Mapping):
        pov = obs.get("pov")
        if isinstance(pov, np.ndarray):
            return encode_pov_to_base64(pov) or pov.tolist()
    return None


def _diff_int_dict(prev: Mapping[str, int], curr: Mapping[str, int]) -> Dict[str, int]:
    keys = set(prev) | set(curr)
    out: Dict[str, int] = {}
    for k in keys:
        delta = int(curr.get(k, 0)) - int(prev.get(k, 0))
        if delta != 0:
            out[str(k)] = delta
    return out


# ----------------------------------------------------------------------
# Episode loop (Algorithm 1)
# ----------------------------------------------------------------------


def run_episode(
    env,
    client: MineEvolveClient,
    task_goal: str,
    max_subgoals: int = 12,
    max_episode_wall_s: float = 1800.0,
    max_steps_per_subgoal: int = 1200,
    subgoal_timeout_s: int = 60,
    eta_fail: float = 0.5,
    recent_window: int = 4,
    budget_tokens: int = 512,
    top_k: int = 16,
    artifact_dir: str | None = None,
    task_id: int | None = None,
    run_idx: int = 0,
    evidence_keyframe_interval: int = 40,
    seed: int | None = None,
) -> Tuple[bool, int]:
    """Algorithm 1: planner -> executor -> Monitor -> Inducer -> Curator -> Adaptor."""

    pos = None
    if isinstance(seed, Mapping):
        seed, pos = seed.get("seed"), seed.get("pos")
    if seed is not None:
        # MineRL sends the seed with the next mission and forgets it after reset,
        # so it must be set before every episode (paper: fixed task-seed split).
        env.seed(int(seed))
        logger.info("world seed %s pos %s for task %s run %d", seed, pos, task_id, run_idx + 1)
    obs = env.reset()
    if pos is not None:
        # JARVIS-1-style close-ended spawn: fixed seed + fixed player position.
        x, y, z = (float(v) for v in pos)
        # yaw 0 / pitch 0: a fixed spawn must also fix the view; without it the agent
        # keeps the previous episode's camera (often looking straight up at leaves)
        env.execute_cmd(f"/tp @s {x:.1f} {y:.1f} {z:.1f} 0 0")
        env.execute_cmd("/spawnpoint")
        # Let the chunks around the new position load before the first frame: the client
        # needs ~5-15 ticks after a teleport, during which the POV shows an unloaded world
        # and the landmark observation is empty (measured with scripts/test_approach.py).
        for k in range(60):
            obs, _r, _d, _i = env.step(env.action_space.noop())
            if k >= 10 and (_i.get("nearby_blocks") or {}):
                break
        logger.info("chunks loaded after %d ticks (%d landmark kinds)", k + 1, len(_i.get("nearby_blocks") or {}))
    env_info = env.info or {}
    state = _state_snapshot(env_info, task_goal)

    client.reset(task_goal=task_goal)
    plan_resp = client.plan(state=state, budget_tokens=budget_tokens, top_k=top_k)
    plan = plan_resp.get("plan") or {}
    _save_plan_artifact(
        artifact_dir=artifact_dir,
        task_goal=task_goal,
        task_id=task_id,
        run_idx=run_idx,
        stage="initial",
        plan=plan,
        extra={
            "retrieved_skill_ids": plan_resp.get("retrieved_skill_ids", []),
            "retrieved_remedy_ids": plan_resp.get("retrieved_remedy_ids", []),
        },
    )
    subgoals: List[Dict[str, Any]] = list(plan.get("subgoals") or [])

    if not subgoals:
        logger.warning("Planner returned no subgoals for task %s", task_goal)
        return False, 0

    consecutive_failures = 0
    total_steps = 0
    n_repairs = 0
    zero_step_subgoals = 0   # subgoals that consumed no env step at all
    t_episode = time.time()
    i = 0
    while i < min(len(subgoals), max_subgoals):
        # Backstops against a dead env slipping through as "subgoal failures" (each
        # of which costs 2-3 LLM calls): no healthy episode has 5 zero-step subgoals in
        # a row, and none lasts longer than max_episode_wall_s (paper horizon 2 min,
        # observed max ~10 min incl. LLM latency).
        if zero_step_subgoals >= 5:
            raise EnvCrashed(f"{zero_step_subgoals} consecutive subgoals consumed 0 env steps")
        if time.time() - t_episode > max_episode_wall_s:
            raise EnvCrashed(f"episode wall clock exceeded {max_episode_wall_s}s")
        sg = subgoals[i]
        executor_hint = str(sg.get("executor_hint") or "stevei").strip().lower()
        if executor_hint in {"mc_craft", "mc_smelt", "place", "use", "equip"}:
            try:
                result = _run_helper_subgoal(
                    env=env,
                    subgoal=sg,
                    task_goal=task_goal,
                    artifact_dir=artifact_dir,
                    task_id=task_id,
                    run_idx=run_idx,
                )
            except EnvCrashed:
                raise
            except Exception as exc:
                if "done=True" in str(exc) or "timed out" in str(exc):
                    raise EnvCrashed(str(exc)) from exc
                raise
        else:
            result = _run_subgoal(
                env=env,
                client=client,
                subgoal=sg,
                task_goal=task_goal,
                max_steps=max_steps_per_subgoal,
                default_timeout_s=subgoal_timeout_s,
                artifact_dir=artifact_dir,
                task_id=task_id,
                run_idx=run_idx,
                evidence_keyframe_interval=evidence_keyframe_interval,
            )
        total_steps += int(result["steps"])
        zero_step_subgoals = zero_step_subgoals + 1 if int(result["steps"]) == 0 and not result["success"] else 0

        # Push typed feedback to server (Monitor stage 1)
        try:
            client.monitor(
                subgoal=[result["condition"], 1],
                target_item=result["target_item"],
                coords_history=result["coords_history"],
                delta_v=result["delta_v"],
                delta_s=result["delta_s"],
                gui_open=bool(result["end_state"].get("isGuiOpen", False)),
                success=int(bool(result["success"])),
                timed_out=bool(result["timed_out"]),
                died=bool(result["died"]),
                p_goal=0.0,
                subgoal_id=result["subgoal_id"],
                coords_now=result["end_state"].get("coords", []),
            )
        except Exception as exc:
            logger.warning("client.monitor failed: %s", exc)

        # Inducer + Curator (stage 2/3 of Algorithm 1, plus Figure 3 deadlock)
        try:
            client.induce(
                eta_fail=eta_fail,
                recent_window=recent_window,
                state=result["end_state"],
            )
        except Exception as exc:
            logger.warning("client.induce failed: %s", exc)

        # Algorithm 1 line 16: "if task goal g is completed then return success" is
        # checked after EVERY subgoal, not only when the loop ends. Otherwise an
        # item picked up during a `move` (or a failed subgoal) leaves the loop
        # repairing an already-solved task until the horizon.
        if _episode_succeeded(task_goal=task_goal, inventory=dict((env.info or {}).get("inventory") or {})):
            logger.info("task goal satisfied after %s; ending episode", result["subgoal_id"])
            client.advance(success=True)
            break

        if result["success"]:
            consecutive_failures = 0
            client.advance(success=True)
            i += 1
            continue

        # env.step returned done=True (death or the benchmark's max_minutes horizon):
        # the env cannot be stepped again without a reset, so every further subgoal
        # would end on step 1 and the repair loop below would spin forever.
        if result["died"]:
            logger.warning("Environment episode ended (done=True) during %s; stopping", result["subgoal_id"])
            client.advance(success=False)
            break

        # Failure or stagnation: ask Adaptor to repair the suffix
        consecutive_failures += 1
        try:
            repair_resp = client.repair(
                state=result["end_state"],
                recent_window=recent_window,
                eta_fail=eta_fail,
                budget_tokens=budget_tokens,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("client.repair failed: %s", exc)
            repair_resp = {"repaired": False}

        if repair_resp.get("repaired"):
            n_repairs += 1
            new_plan = repair_resp.get("plan") or {}
            _save_plan_artifact(
                artifact_dir=artifact_dir,
                task_goal=task_goal,
                task_id=task_id,
                run_idx=run_idx,
                stage=f"repair{n_repairs:02d}_i{i}",  # numbered: repeated repairs at the same i no longer overwrite
                plan=new_plan,
                extra={
                    "active_remedies_used": repair_resp.get("active_remedies_used", []),
                    "deadlock_signal": repair_resp.get("deadlock_signal"),
                },
            )
            new_subgoals = list(new_plan.get("subgoals") or [])
            if new_subgoals:
                subgoals = new_subgoals
                continue  # do not advance i; retry from the same logical position
        else:
            logger.warning("Adaptor did not repair (%s); advancing past %s", repair_resp.get("reason", "?"), sg.get("subgoal_id"))

        # No repair; advance to next subgoal to avoid infinite retry on the same step
        client.advance(success=False)
        i += 1
        if consecutive_failures >= 3:
            logger.warning("Aborting episode after %d consecutive failures", consecutive_failures)
            break

    end_inv = dict((env.info or {}).get("inventory") or {})
    success = _episode_succeeded(task_goal=task_goal, inventory=end_inv)
    return success, total_steps


def _save_plan_artifact(
    artifact_dir: str | None,
    task_goal: str,
    task_id: int | None,
    run_idx: int,
    stage: str,
    plan: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    if not artifact_dir:
        return
    out_dir = Path(artifact_dir) / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_goal = "".join(c if c.isalnum() or c in "_-" else "_" for c in task_goal)[:60]
    task_part = f"task_{task_id}" if task_id is not None else "task_unknown"
    path = out_dir / f"{task_part}_run_{run_idx + 1}_{stage}_{safe_goal}.json"
    payload = {
        "task_id": task_id,
        "run_idx": run_idx,
        "task_goal": task_goal,
        "stage": stage,
        "subgoal_count": len(list(plan.get("subgoals") or [])) if isinstance(plan, Mapping) else 0,
        "plan": dict(plan or {}),
        "extra": dict(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _goal_object(task_goal: str) -> str:
    """The item phrase a task asks for: 'Craft a wooden pickaxe' -> 'wooden pickaxe',
    'Smelt iron ore into iron ingot' -> 'iron ingot', 'Kill a cow to obtain leather' -> 'leather'."""
    g = re.sub(r"[^a-z0-9 ]", " ", str(task_goal).lower())
    g = re.sub(r"\s+", " ", g).strip()
    for marker in (" to obtain ", " into ", " to mine ", " to gather ", " to get ", " to craft ", " to make "):
        if marker in g:
            g = g.split(marker, 1)[1]
            break
    else:
        g = re.sub(r"^(craft|mine|collect|chop|punch|smelt|dig|kill|upgrade|trade|wash|repair|gather|obtain|make|get)\s+", "", g)
        if " to " in g and g.startswith(("upgrade", "a wooden", "a stone", "an iron")):
            g = g.split(" to ", 1)[1]
    g = re.split(r" (from|for|with|at|in|using|without|by|on) ", g)[0]
    g = re.sub(r"^(down and |down to )?(mine |craft |collect )?", "", g)          # "dig down and mine a diamond"
    g = re.sub(r"^(a|an|the|some|eight|three|two|one|\d+) ", "", g).strip()
    g = re.sub(r" (blocks?|material|items?|ores?)$", lambda m: "" if m.group(1).startswith(("block", "material", "item")) else m.group(0), g)
    return g.strip()


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _goal_quantity(task_goal: str) -> int:
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", str(task_goal).lower())
    if not m:
        return 1
    w = m.group(1)
    return int(w) if w.isdigit() else _NUM_WORDS.get(w, 1)


def _episode_succeeded(task_goal: str, inventory: Mapping[str, int]) -> bool:
    """Success if the inventory holds the item (and quantity) the task asks for."""

    need = _goal_quantity(task_goal)
    obj = _goal_object(task_goal)
    # Task 8 of the stone tier, "Mine stone slab material from a stone outcrop", asks for
    # the *material* (what mining stone yields), not a crafted slab. Literal reading.
    if obj == "stone slab":
        return inventory_satisfies(inventory, "cobblestone", need) or inventory_satisfies(inventory, "stone", need)
    obj_words = obj.split()
    # wood-gathering goals ("chop an oak log", "punch a tree to gather wood"): any log counts.
    # Only when the *object* is wood/log - "wooden sword" also contains "wood" (upstream's
    # substring rule scored a sword task as solved by holding a log).
    if obj_words and obj_words[-1] in ("log", "logs", "wood") and any(
        inventory_satisfies(inventory, target, need) for target in ("log", "oak_log", "wood")
    ):
        return True
    if not obj_words:
        return False
    for k, count in inventory.items():
        if int(count) < need:
            continue
        name = str(k).lower().replace("minecraft:", "").replace("_", " ").strip()
        words = name.split()
        head, obj_head = words[-1], obj_words[-1]
        same_head = head == obj_head or head.rstrip("s") == obj_head.rstrip("s")
        if not same_head:
            continue
        # every qualifier in the goal must appear in the item (wooden pickaxe != stone pickaxe);
        # extra qualifiers on the item are fine (oak sapling for "sapling")
        if all(any(q == w or q.rstrip("s") == w.rstrip("s") for w in words) for q in obj_words[:-1]):
            return True
    return False


# ----------------------------------------------------------------------
# Hydra entry
# ----------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    setup_logging()
    benchmark_cfg = _benchmark_cfg(cfg)
    info_panel(f"MineEvolve: {benchmark_cfg.env.name}  |  llm={cfg.llm.provider}/{cfg.llm.model}")

    # Lazy imports so the module can be examined without a MineRL install
    from .env import make_env

    client = MineEvolveClient(
        base_url=f"{cfg.server.url}:{cfg.server.port}",
        timeout=float(cfg.server.timeout),
    )

    env = make_env(cfg, logger=logger)

    tasks = _get_evaluate_tasks(cfg)
    success_mon = SuccessMonitor()
    step_mon = StepMonitor()
    # `seeds: [..]` -> one run per world seed (reproducible); otherwise `env.times`
    # runs on random worlds, as upstream did.
    seeds = [
        {"seed": int(x["seed"]), "pos": [float(v) for v in x["pos"]]} if isinstance(x, Mapping) and "pos" in x
        else int(x["seed"] if isinstance(x, Mapping) else x)
        for x in (OmegaConf.to_container(OmegaConf.select(cfg, "seeds")) if OmegaConf.select(cfg, "seeds") is not None else [])
    ]
    runs = [(i, s) for i, s in enumerate(seeds)] or [(i, None) for i in range(int(benchmark_cfg.env.get("times") or 1))]
    try:
        from hydra.core.hydra_config import HydraConfig

        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # not launched through hydra
        run_dir = None
    resume_dir = OmegaConf.select(cfg, "resume_dir")
    if resume_dir:
        run_dir = Path(str(resume_dir))
        if not run_dir.is_dir():
            raise SystemExit(f"resume_dir does not exist: {run_dir}")
    runs_log = run_dir / "runs.jsonl" if run_dir else None
    # Every run is self-contained: plans/ and evidence/ go under the hydra run dir
    # (logs/eval/<date>/<time>/) unless artifact_dir is set explicitly, and the
    # server-side LLM call log for this run is copied there at the end.
    artifact_dir = str(OmegaConf.select(cfg, "artifact_dir") or run_dir or ".")
    llm_log = Path(os.environ.get("MINEEVOLVE_LLM_LOG", "logs/llm_calls.jsonl"))
    t_eval = time.time()

    # Episode sequence: tasks x seeds in fixed order (the task-seed split). In
    # accumulation mode the sequence is repeated until accumulate.episodes episodes
    # have run; each repetition is a "pass" with its own evidence/ plans/ subdir.
    sequence = [(task_id, task_type, instruction, run_idx, seed)
                for task_id, task_type, instruction in tasks for run_idx, seed in runs]
    n_accumulate = int(OmegaConf.select(cfg, "accumulate.episodes") or 0)
    n_episodes = n_accumulate if n_accumulate > 0 else len(sequence)
    checkpoint_every = int(OmegaConf.select(cfg, "accumulate.kb_checkpoint_every") or 50)
    kb_store_dir = Path(str(OmegaConf.select(cfg, "accumulate.kb_store_dir") or "memories/run"))
    done_episodes = 0
    if runs_log is not None and runs_log.exists():
        done_episodes = sum(1 for line in runs_log.read_text().splitlines() if line.strip())
    if done_episodes and not resume_dir:
        raise SystemExit(f"{runs_log} already has {done_episodes} episodes; pass resume_dir=... to continue it")
    if resume_dir:
        logger.info("resuming %s at episode %d/%d", run_dir, done_episodes + 1, n_episodes)
        if n_accumulate > 0 and kb_store_dir.is_dir():
            n_sk = sum(len(json.loads((kb_store_dir / f).read_text())) for f in ("skills.json", "remedies.json") if (kb_store_dir / f).exists())
            logger.info("live KB store %s holds %d entries", kb_store_dir, n_sk)
            if done_episodes >= checkpoint_every and n_sk == 0:
                logger.warning("live KB store is EMPTY after %d episodes - restore kb_checkpoints/ into %s before resuming", done_episodes, kb_store_dir)

    def _checkpoint_kb(n_done: int) -> None:
        if run_dir is None:
            return
        dst = run_dir / "kb_checkpoints" / f"M{n_done}"
        if not kb_store_dir.is_dir():
            logger.warning("KB checkpoint M%d skipped: store dir %s not found", n_done, kb_store_dir)
            return
        dst.mkdir(parents=True, exist_ok=True)
        import shutil
        for name in ("skills.json", "remedies.json"):
            if (kb_store_dir / name).exists():
                shutil.copy2(kb_store_dir / name, dst / name)
        logger.info("KB checkpoint M%d -> %s", n_done, dst)

    episodes_on_instance = 0
    recycle_every = int(OmegaConf.select(cfg, "env_recycle_every") or 8)

    def _fresh_env(reason: str):
        nonlocal env, episodes_on_instance
        logger.warning("relaunching Minecraft (%s)", reason)
        try:
            env.close()
        except Exception as exc:  # the old instance may already be dead
            logger.warning("env.close() failed: %s", exc)
        env = make_env(cfg, logger=logger)
        episodes_on_instance = 0

    for ep_idx in range(done_episodes, n_episodes):
        task_id, task_type, instruction, run_idx, seed = sequence[ep_idx % len(sequence)]
        pass_no = ep_idx // len(sequence) + 1
        ep_artifact_dir = str(Path(artifact_dir) / f"pass{pass_no}") if n_accumulate > 0 else artifact_dir
        seed_note = f"  seed={seed}" if seed is not None else ""
        info_panel(f"episode {ep_idx + 1}/{n_episodes} (pass {pass_no})  task {task_id} ({task_type})  run {run_idx + 1}/{len(runs)}{seed_note}: {instruction}")
        t_run = time.time()
        if episodes_on_instance >= recycle_every:
            _fresh_env(f"{episodes_on_instance} episodes on this instance")
        episode_kwargs = dict(
                client=client,
                task_goal=instruction,
                max_subgoals=int(cfg.runtime.get("max_subgoals", 12)),
                max_episode_wall_s=float(cfg.runtime.get("max_episode_wall_s", 1800)),
                max_steps_per_subgoal=int(cfg.runtime.get("max_steps_per_subgoal", 1200)),
                subgoal_timeout_s=int(cfg.runtime.get("subgoal_timeout_s", 60)),
                eta_fail=float(cfg.runtime.get("eta_fail", 0.5)),
                recent_window=int(cfg.runtime.get("recent_window", 4)),
                budget_tokens=int(cfg.runtime.get("budget_tokens", 512)),
                top_k=int(cfg.runtime.get("top_k", 16)),
                artifact_dir=ep_artifact_dir,
                task_id=task_id,
                run_idx=run_idx,
                evidence_keyframe_interval=int(
                    OmegaConf.select(cfg, "record.evidence.keyframe_interval") or 40
                ),
                seed=seed,
            )
        success, steps = False, 0
        for attempt in range(2):
            try:
                success, steps = run_episode(env=env, **episode_kwargs)
                episodes_on_instance += 1
                break
            except LLMGuardAbort as exc:
                logger.error("LLM guard aborted by operator (%s); stopping after %d episodes", exc, ep_idx)
                if run_dir is not None:
                    (run_dir / "ABORTED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
                try:
                    env.close()
                finally:
                    os._exit(3)
            except Exception as exc:
                # a dead / hung Minecraft (socket timeout, EnvCrashed) is an infrastructure
                # failure, not an agent result: relaunch once and retry the episode
                logger.exception("run_episode failed (attempt %d): %s", attempt + 1, exc)
                if attempt == 0:
                    _fresh_env("episode failed")
                else:
                    logger.error("episode %d failed twice; recorded as failure", ep_idx + 1)
        success_mon.record(instruction, success)
        step_mon.record(instruction, steps)
        if runs_log is not None:
            with runs_log.open("a") as fh:
                row = {
                    "task_id": task_id, "task": instruction, "run": run_idx + 1,
                    "seed": seed if not isinstance(seed, Mapping) else seed["seed"],
                    "pos": seed["pos"] if isinstance(seed, Mapping) else None,
                    "success": bool(success), "steps": int(steps), "wall_s": round(time.time() - t_run, 1),
                    "llm": f"{cfg.llm.provider}/{cfg.llm.model}",
                    "episode": ep_idx + 1, "pass": pass_no,
                }
                fh.write(json.dumps(row) + "\n")
            if n_accumulate > 0:
                # per-pass copy so scripts/analyze_failures.py works on <run dir>/pass<k>
                Path(ep_artifact_dir).mkdir(parents=True, exist_ok=True)
                with (Path(ep_artifact_dir) / "runs.jsonl").open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
        if n_accumulate > 0 and ((ep_idx + 1) % checkpoint_every == 0 or ep_idx + 1 == n_episodes):
            _checkpoint_kb(ep_idx + 1)
        save_video = getattr(env, "save_video", None)
        if callable(save_video):
            status = "success" if success else "failed"
            thread = save_video(instruction, status)
            if thread is not None:
                thread.join(timeout=30.0)

    if run_dir is not None and llm_log.exists():
        # copy this run's LLM calls (by timestamp) and their prompt/response dumps
        try:
            kept = []
            for line in llm_log.read_text().splitlines():
                if line.strip() and json.loads(line).get("t", 0) >= t_eval - 1:
                    kept.append(line)
            with (run_dir / "llm_calls.jsonl").open("a" if resume_dir else "w") as fh:
                fh.write("\n".join(kept) + ("\n" if kept else ""))
            dump_dir = run_dir / "llm_calls"
            for line in kept:
                src = json.loads(line).get("dump")
                if src and Path(src).exists():
                    dump_dir.mkdir(exist_ok=True)
                    (dump_dir / Path(src).name).write_bytes(Path(src).read_bytes())
            logger.info("run dir: %s (%d LLM calls)", run_dir, len(kept))
        except Exception as exc:
            logger.warning("could not copy LLM log into the run dir: %s", exc)

    print_results(
        title=f"Results: {benchmark_cfg.env.name}",
        tasks=[t[2] for t in tasks],
        success=success_mon,
        step=step_mon,
    )

    OmegaConf.save(config=cfg, f="resolved_config.yaml")
    if run_dir is not None:
        (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    # MineRL leaves non-daemon threads / a zombie launchClient behind; the interpreter
    # then hangs at exit (a 33-episode run sat "running" for 10 h after printing its
    # results). Close what we can and exit hard.
    try:
        env.close()
    except Exception as exc:
        logger.warning("env.close() at exit failed: %s", exc)
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
