#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO-Mem evaluation client.

Evaluates pi0.5 on LIBERO-Mem's 10 memory-intensive tasks with sequential
subgoal tracking.  Connects to the VLM-orchestrator proxy (or directly to
the VLA server in passthrough mode).

Key differences from standard LIBERO eval:
  - Metric is **subgoal completion rate** (not just binary success).
    For a task with N sequential subgoals, completing K gives K/N.
  - Episode lengths scale with subgoal count (200 steps per subgoal, min 300).
  - Calls ``env.env._check_success(inc=True)`` every step to advance the
    internal subgoal timer (required by Mem's Sequence goal checking).
  - Reports both binary success AND tiered (subgoal) completion rate.
  - Strips numeric prefix from task language (grab_language_from_filename
    produces "1 pick up the bowl..." instead of "pick up the bowl...").

Usage::

    # Passthrough mode (direct VLA):
    python run_mem_eval.py --port 8001 --mode passthrough

    # Orchestrated mode:
    python run_mem_eval.py --port 8001 --mode orchestrated --enable-gt-state
"""

import argparse
import collections
import json
import logging
import math
import os
import pathlib
import re
import sys
import time

import numpy as np
import torch

# Patch torch.load for LIBERO compatibility
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

logger = logging.getLogger("libero_mem_eval")

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # 7-dim: 6 zeros + gripper close

# Steps per subgoal (generous upper bound)
STEPS_PER_SUBGOAL = 250
MIN_STEPS = 300

# Number of subgoals per task (from BDDL analysis)
TASK_SUBGOAL_COUNTS = {
    "KITCHEN_SCENE1_1": 1,
    "KITCHEN_SCENE1_2": 1,
    "KITCHEN_SCENE1_3": 3,
    "KITCHEN_SCENE1_4": 3,
    "KITCHEN_SCENE1_5": 5,
    "KITCHEN_SCENE1_6": 7,
    "KITCHEN_SCENE1_7": 3,
    "KITCHEN_SCENE1_8": 4,
    "KITCHEN_SCENE1_9": 2,
    "KITCHEN_SCENE1_10": 2,
}


def get_max_steps_for_task(task_name: str) -> int:
    """Determine max steps based on task's subgoal count."""
    for prefix, n_subgoals in TASK_SUBGOAL_COUNTS.items():
        if task_name.startswith(prefix):
            return max(STEPS_PER_SUBGOAL * n_subgoals, MIN_STEPS)
    return 600  # fallback


def get_num_subgoals(task_name: str) -> int:
    """Get the number of subgoals for a task."""
    for prefix, n_subgoals in TASK_SUBGOAL_COUNTS.items():
        if task_name.startswith(prefix):
            return n_subgoals
    return 1


def clean_language(language: str) -> str:
    """Strip numeric prefix from LIBERO-Mem task language.
    
    grab_language_from_filename produces "1 pick up the bowl..." from
    "KITCHEN_SCENE1_1_pick_up_the_bowl_..." — we strip the leading digits.
    """
    return re.sub(r"^\d+\s+", "", language)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → axis-angle (3,)."""
    w = float(quat[3])
    w = max(-1.0, min(1.0, w))
    den = math.sqrt(1.0 - w * w)
    if den < 1e-6:
        return np.zeros(3, dtype=quat.dtype)
    angle = 2.0 * math.acos(w)
    axis = np.array(quat[:3], dtype=quat.dtype) / den
    return axis * angle


def _robosuite_depth_to_metres(
    depth_norm: np.ndarray,
    near: float = 0.01,
    far: float = 50.0,
) -> np.ndarray:
    """Convert MuJoCo normalised [0,1] depth buffer to metres.

    Mirrors ``libero_eval_client._robosuite_depth_to_metres``.  Uses the
    standard MuJoCo zbuffer formula::

        depth_m = near / (1 - d * (1 - near / far))
    """
    d = np.clip(depth_norm, 0.0, 0.9999)
    return (near / (1.0 - d * (1.0 - near / far))).astype(np.float32)


def build_wire_obs(
    obs: dict,
    task_description: str,
    resize_size: int = 224,
    ground_truth_done: bool = False,
    camera_K=None,
    camera_extrinsic=None,
    depth_near: float = 0.01,
    depth_far: float = 50.0,
) -> dict:
    """Build the wire-format observation dict for the proxy.

    Wire-format parity with ``libero_eval_client.build_wire_obs`` so the
    same orchestrator / grasp-tool / failure-detector code paths apply.
    """
    import cv2

    # 180° rotation — matches pi0.5's LIBERO training distribution.
    # Y-flip alone (just OpenGL→OpenCV) is NOT enough: pi0.5 was trained
    # on images that are also X-mirrored, so omitting the second flip
    # makes the policy behave as if the scene is left/right reversed.
    raw_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    image_224 = cv2.resize(raw_image, (resize_size, resize_size))
    wrist_224 = cv2.resize(wrist_image, (resize_size, resize_size))

    # State: eef_pos(3) + axisangle(3) + gripper_qpos(2) = 8 dims
    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ])

    # Gripper width: mean of two finger joint positions, normalised so
    # higher = more closed (matches what the failure detector expects).
    # robosuite Panda gripper range is [0, 0.04].
    raw_grip = float(np.mean(obs["robot0_gripper_qpos"]))
    grip_normalised = np.array([1.0 - min(raw_grip / 0.04, 1.0)],
                               dtype=np.float64)

    wire = {
        "observation/image": image_224,
        "observation/wrist_image": wrist_224,
        "observation/state": state.astype(np.float32),
        "prompt": str(task_description),
        # Full-res for VLM
        "observation/image_raw": raw_image,
        # EE state for proxy + failure detector
        "observation/ee_pos": obs["robot0_eef_pos"].astype(np.float64),
        "observation/gripper_position": grip_normalised,
        # GT success flag (logging only)
        "ground_truth_done": bool(ground_truth_done),
    }

    # EE quaternion for grasp tool orientation planning.
    # robosuite stores as (x,y,z,w); the grasp tool expects (w,x,y,z).
    eef_quat_xyzw = obs.get("robot0_eef_quat")
    if eef_quat_xyzw is not None:
        q = np.asarray(eef_quat_xyzw, dtype=np.float64).flatten()
        wire["observation/ee_quat"] = np.array([q[3], q[0], q[1], q[2]])

    # Depth + camera matrices (grasp tool inputs).
    depth = obs.get("agentview_depth")
    if depth is not None:
        # robosuite depth is (H, W, 1) in [0,1] normalised range, flipped.
        # Squeeze, flip, then convert normalised values to metres.
        depth_squeezed = np.squeeze(depth)[::-1, ::-1].astype(np.float32)
        depth_m = _robosuite_depth_to_metres(
            depth_squeezed, near=depth_near, far=depth_far,
        )
        wire["observation/depth_agentview"] = np.ascontiguousarray(depth_m)

    if camera_K is not None:
        wire["observation/camera_K"] = camera_K.flatten().astype(np.float64)
    if camera_extrinsic is not None:
        wire["observation/camera_extrinsic"] = (
            camera_extrinsic.flatten().astype(np.float64)
        )

    # ── GT segmentation support (grasp_tool GT_SIM mode) ──
    # When `build_wire_obs._gt_seg_provider` has been attached, render
    # the full segmentation buffer so the grasp tool can extract a
    # perfect mask for any target object — no GDino/SAM needed.
    gt_seg_provider = getattr(build_wire_obs, "_gt_seg_provider", None)
    if gt_seg_provider is not None:
        try:
            sim = gt_seg_provider._env.sim
            seg = sim.render(
                camera_name="agentview",
                height=raw_image.shape[0],
                width=raw_image.shape[1],
                segmentation=True,
            )
            # seg: (H, W, 2) — channel 0 = objtype, channel 1 = objid.
            # Convert geom_ids → body_ids via model.geom_bodyid.
            geom_ids = seg[::-1, :, 1].astype(np.int32)
            geom_bodyid = sim.model.geom_bodyid
            body_ids = np.where(
                geom_ids >= 0,
                geom_bodyid[np.clip(geom_ids, 0, len(geom_bodyid) - 1)],
                -1,
            ).astype(np.int32)
            wire["gt_seg/body_ids"] = body_ids
            wire["gt_seg/obj_body_id"] = gt_seg_provider._obj_body_id
        except Exception:
            # Silent — failures don't block the eval
            pass

    return wire


def parse_args():
    p = argparse.ArgumentParser(description="LIBERO-Mem evaluation")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--num-trials-per-task", type=int, default=20,
                    help="Rollouts per task (paper uses 20)")
    p.add_argument("--seed", type=int, default=42)
    # Outputs land in the orchestrator's per-episode log dir
    # (advertised via response["orchestrator_episode_log_dir"]).
    # --video-out-path is an explicit fallback path used ONLY when the
    # orchestrator runs without --log-dir.  Default None → no fallback,
    # video saving is skipped with a warning instead of silently
    # creating a results/libero_mem tree.
    p.add_argument("--log-dir", default=None,
                    help="Unused — kept for argparse compatibility.")
    p.add_argument("--video-out-path", default=None,
                    help="Fallback dir for videos when the orchestrator "
                         "does not expose its episode log dir.  Default: "
                         "skip video saving in that case.")
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--num-steps-wait", type=int, default=10,
                    help="Warmup steps before policy takes over")
    p.add_argument("--enable-gt-state", action="store_true")
    p.add_argument("--enable-depth", action="store_true")
    p.add_argument("--task-ids", type=str, default="",
                    help="Comma-separated task indices (0-9), empty=all")
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from libero.libero import benchmark as libero_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy

    np.random.seed(args.seed)

    # Load benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_mem"]()
    num_tasks = task_suite.n_tasks

    logger.info(f"LIBERO-Mem: {num_tasks} tasks")
    logger.info(f"Connecting to proxy at {args.host}:{args.port}")

    # Per-episode dirs are created lazily as videos initialise — no
    # upfront mkdir needed.  Eval outputs live under the orchestrator's
    # episode_log_dir; the legacy --log-dir / --video-out-path trees
    # are only mkdir'd when their fallback paths actually need to be
    # written to.

    # Connect to proxy
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Parse task IDs
    if args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    else:
        task_ids = list(range(num_tasks))

    # ── Evaluation loop ──
    all_results = []
    overall_stats = {
        "total_episodes": 0,
        "total_successes": 0,
        "total_subgoal_completion": 0.0,
        "total_subgoals_completed": 0,
        "total_subgoals_possible": 0,
    }

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        language = clean_language(task.language)
        n_subgoals = get_num_subgoals(task.name)
        max_steps = get_max_steps_for_task(task.name)

        bddl_file = task_suite.get_task_bddl_file_path(task_id)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Task T{task_id+1}: {language}")
        logger.info(f"  Subgoals: {n_subgoals}, Max steps: {max_steps}")
        logger.info(f"  BDDL: {bddl_file}")

        env_kwargs = dict(
            bddl_file_name=bddl_file,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
            horizon=max_steps + 100,  # ensure robosuite horizon exceeds our step limit
        )
        if args.enable_depth:
            env_kwargs["camera_depths"] = True
        env = OffScreenRenderEnv(**env_kwargs)
        env.seed(args.seed)

        # Extract camera matrices once per env (mirrors libero_eval_client).
        # Without these, the grasp tool silently degrades — point cloud
        # back-projection has no intrinsics, cam_to_world is unknown.
        camera_K = None
        camera_extrinsic = None
        depth_near = 0.01
        depth_far = 50.0
        if args.enable_depth:
            try:
                from robosuite.utils.camera_utils import (
                    get_camera_intrinsic_matrix,
                    get_camera_extrinsic_matrix,
                )
                camera_K = get_camera_intrinsic_matrix(
                    env.env.sim, "agentview",
                    LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION,
                )
                camera_extrinsic = get_camera_extrinsic_matrix(
                    env.env.sim, "agentview",
                )
                sim = env.env.sim
                extent = sim.model.stat.extent
                depth_near = float(sim.model.vis.map.znear * extent)
                depth_far = float(sim.model.vis.map.zfar * extent)
                logger.info(
                    f"  Camera K + extrinsic extracted, "
                    f"depth clip near={depth_near:.4f} far={depth_far:.4f}"
                )
            except Exception as e:
                logger.warning(f"  Failed to extract camera matrices: {e}")

        # GT state exporter
        gt_exporter = None
        if args.enable_gt_state:
            try:
                from vlm_orchestrator.aux_benchmarks.libero_gt import LiberoGTStateExporter
                gt_exporter = LiberoGTStateExporter(env)
            except Exception as e:
                logger.warning(f"  GT state exporter failed: {e}")

        # GT segmentation provider (for grasp_tool GT_SIM mode).
        # Attached as a closure variable on build_wire_obs so the
        # render code inside build_wire_obs can reach it without a
        # signature change.  Mirrors libero_eval_client.py:472–488.
        build_wire_obs._gt_seg_provider = None
        if args.enable_gt_state:
            try:
                from vlm_orchestrator.perception.gt_segmentation import (
                    RobosuiteSegProvider,
                )
                build_wire_obs._gt_seg_provider = RobosuiteSegProvider(
                    env,
                    camera_name="agentview",
                    height=LIBERO_ENV_RESOLUTION,
                    width=LIBERO_ENV_RESOLUTION,
                )
                logger.info(
                    "  GT segmentation provider initialised "
                    f"({len(build_wire_obs._gt_seg_provider._obj_body_id)} bodies)"
                )
            except Exception as e:
                logger.debug(f"  GT seg provider init failed: {e}")

        task_results = {
            "task_id": task_id,
            "task_name": task.name,
            "language": language,
            "n_subgoals": n_subgoals,
            "max_steps": max_steps,
            "episodes": [],
        }

        for ep_idx in range(args.num_trials_per_task):
            # Pick init state (cycle if fewer than num_trials)
            init_idx = ep_idx % len(init_states)

            logger.info(f"  Episode {ep_idx+1}/{args.num_trials_per_task} "
                         f"(init_state={init_idx})")

            # Reset environment
            env.reset()
            obs = env.set_init_state(init_states[init_idx])

            # Reset subgoal tracking in the Mem environment
            env.env.reset_subgoal_progress()

            action_plan = collections.deque()
            last_response = {}

            ep_meta = {
                "task_id": task_id,
                "episode_idx": ep_idx,
                "init_state_idx": init_idx,
                "start_time": time.time(),
                "orchestrator_subgoals": None,
                "orchestrator_instructions": [],
                "orchestrator_failures": [],
                "episode_log_dir": None,
            }

            # Video writers — raw + annotated.  Lazily initialised
            # once the orchestrator advertises its episode_log_dir
            # so videos colocate with rewrites.jsonl / metadata.json.
            raw_writer = None
            annotated_writer = None
            video_initialised = False
            # Pending frames captured before the orchestrator reveals
            # the episode_log_dir (so we don't lose the warm-up window).
            pending_raw: list = []
            pending_annot: list = []  # tuples of (raw_full, step, response)

            t = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                # Warmup steps
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    # Still need to advance subgoal timer during warmup
                    env.env._check_success(inc=True)
                    t += 1
                    continue

                # Build observation
                wire_obs = build_wire_obs(
                    obs, language, args.resize_size,
                    ground_truth_done=done,
                    camera_K=camera_K,
                    camera_extrinsic=camera_extrinsic,
                    depth_near=depth_near,
                    depth_far=depth_far,
                )

                # Episode-id marker so the proxy detects new episodes
                # and rotates its per-episode log dir.  Without this
                # all episodes write into the first episode's dir.
                wire_obs["__episode_id"] = f"{task_id}_{ep_idx}"

                if gt_exporter is not None:
                    try:
                        wire_obs["gt_state"] = gt_exporter.export(obs)
                    except Exception:
                        pass

                # Capture per-step frames.  The annotated writer needs
                # the response, so we buffer until the action plan
                # query gives us one.
                raw_frame = wire_obs["observation/image"]
                raw_full = wire_obs.get("observation/image_raw", raw_frame)

                if not action_plan:
                    # Query proxy
                    response = client.infer(wire_obs)
                    last_response = response

                    # Capture orchestrator metadata for the per-episode log.
                    if "orchestrator_subgoals" in response:
                        ep_meta["orchestrator_subgoals"] = (
                            response["orchestrator_subgoals"]
                        )
                    if "orchestrator_instruction" in response:
                        ep_meta["orchestrator_instructions"].append(
                            response["orchestrator_instruction"]
                        )
                    if "orchestrator_failure" in response:
                        ep_meta["orchestrator_failures"].append({
                            "step": t,
                            "failure": response["orchestrator_failure"],
                        })
                    if (ep_meta["episode_log_dir"] is None
                            and "orchestrator_episode_log_dir" in response):
                        ep_meta["episode_log_dir"] = (
                            response["orchestrator_episode_log_dir"]
                        )

                    # Lazy video-writer init once the episode_log_dir
                    # is known.  Use the orchestrator's per-episode dir;
                    # fall back to args.video_out_path ONLY if the user
                    # explicitly set it.  Otherwise skip video saving
                    # so we never create a results/libero_mem tree by
                    # accident.
                    if (not video_initialised
                            and args.save_video and not args.no_video):
                        if ep_meta["episode_log_dir"]:
                            ep_dir = pathlib.Path(ep_meta["episode_log_dir"])
                        elif args.video_out_path:
                            ep_dir = (pathlib.Path(args.video_out_path)
                                      / f"T{task_id+1}_ep{ep_idx}")
                        else:
                            ep_dir = None

                        if ep_dir is None:
                            logger.warning(
                                "    Skipping video save: orchestrator "
                                "did not advertise an episode_log_dir "
                                "and --video-out-path was not set.  "
                                "Launch the proxy with --log-dir or "
                                "pass --video-out-path explicitly."
                            )
                            video_initialised = True   # don't try again
                            pending_raw.clear()
                            pending_annot.clear()
                        else:
                            ep_dir.mkdir(parents=True, exist_ok=True)
                            ep_meta["episode_log_dir"] = str(ep_dir)
                            from vlm_orchestrator.utils.video_annotator import (
                                EpisodeVideoWriter, RawVideoWriter,
                            )
                            raw_writer = RawVideoWriter(
                                str(ep_dir / "rollout_raw.mp4"), fps=10,
                            )
                            annotated_writer = EpisodeVideoWriter(
                                str(ep_dir / "rollout_annotated.mp4"),
                                fps=10,
                            )
                            video_initialised = True
                            for f in pending_raw:
                                raw_writer.add_frame(f)
                            for raw_full_b, step_b, resp_b in pending_annot:
                                annotated_writer.add_frame(
                                    raw_full_b, step=step_b, response=resp_b,
                                )
                            pending_raw.clear()
                            pending_annot.clear()
                            logger.info(f"    Videos → {ep_dir}")

                    # Handle flush
                    if response.get("orchestrator_flush_actions"):
                        action_plan.clear()

                    action_chunk = response["actions"]
                    action_plan.extend(action_chunk[:args.replan_steps])

                # Record the frame.
                if args.save_video and not args.no_video:
                    if raw_writer is not None:
                        raw_writer.add_frame(raw_frame)
                        annotated_writer.add_frame(
                            raw_full, step=t, response=last_response,
                        )
                    else:
                        pending_raw.append(raw_frame)
                        pending_annot.append((raw_full, t, last_response))

                # Execute action
                action = action_plan.popleft()
                try:
                    obs, reward, done, info = env.step(action.tolist())
                except ValueError as e:
                    # "executing action in terminated episode" — robosuite
                    # horizon exceeded
                    if "terminated" in str(e):
                        logger.debug(f"    Robosuite horizon reached at step {t}")
                        break
                    raise

                # CRITICAL: Advance the Mem subgoal counter AND capture
                # the actual sequence-goal success signal.
                #
                # LIBERO-Mem's `_check_success` only mutates internal
                # counters (`_sub_goal_live_time`, `_satisfied_subgoals`)
                # when called with `inc=True`.  The base env's step()
                # calls `_check_success()` without inc, so the `done` it
                # returns never fires on Mem sequence-goal tasks.  We OR
                # the `inc=True` return into `done` to terminate early
                # on full task success instead of running to max_steps.
                mem_done = bool(env.env._check_success(inc=True))
                if mem_done:
                    done = True

                if done:
                    break

                t += 1

            # ── Episode finished ──
            # Get subgoal completion
            satisfied = env.env.get_satisfied_subgoals(language)
            n_completed = len(satisfied)
            tiered_success = n_completed / n_subgoals if n_subgoals > 0 else 0.0

            ep_meta.update({
                "end_time": time.time(),
                "num_steps": t,
                "success": bool(done),
                "n_subgoals_completed": n_completed,
                "n_subgoals_total": n_subgoals,
                "tiered_success": tiered_success,
                "satisfied_subgoals": satisfied,
                "duration_s": time.time() - ep_meta["start_time"],
            })
            task_results["episodes"].append(ep_meta)

            # Update overall stats
            overall_stats["total_episodes"] += 1
            if done:
                overall_stats["total_successes"] += 1
            overall_stats["total_subgoal_completion"] += tiered_success
            overall_stats["total_subgoals_completed"] += n_completed
            overall_stats["total_subgoals_possible"] += n_subgoals

            # Close raw + annotated video writers.
            if raw_writer is not None:
                try:
                    raw_writer.close()
                except Exception as e:
                    logger.debug(f"    Raw video close failed: {e}")
            if annotated_writer is not None:
                try:
                    annotated_writer.close(success=bool(done))
                except Exception as e:
                    logger.debug(f"    Annotated video close failed: {e}")

            # Per-episode eval-side metadata next to the video.
            if ep_meta["episode_log_dir"]:
                try:
                    ep_dir = pathlib.Path(ep_meta["episode_log_dir"])
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    with open(ep_dir / "eval_metadata.json", "w") as f:
                        json.dump(ep_meta, f, indent=2, default=str)
                except Exception as e:
                    logger.debug(f"    eval_metadata save failed: {e}")

            status = "SUCCESS" if done else "FAILURE"
            logger.info(
                f"    {status} | steps={t} | "
                f"subgoals={n_completed}/{n_subgoals} "
                f"({tiered_success:.1%})"
            )

        # Task summary
        eps = task_results["episodes"]
        n_success = sum(1 for e in eps if e["success"])
        avg_tiered = np.mean([e["tiered_success"] for e in eps])
        logger.info(
            f"  Task T{task_id+1} summary: "
            f"success={n_success}/{len(eps)} ({100*n_success/len(eps):.1f}%), "
            f"avg_subgoal_rate={avg_tiered:.1%}"
        )

        all_results.append(task_results)
        env.close()

    # ── Final summary ──
    n_ep = overall_stats["total_episodes"]
    n_succ = overall_stats["total_successes"]
    avg_scr = overall_stats["total_subgoal_completion"] / max(n_ep, 1)
    total_sg_done = overall_stats["total_subgoals_completed"]
    total_sg_poss = overall_stats["total_subgoals_possible"]

    logger.info(f"\n{'='*70}")
    logger.info(f"FINAL RESULTS: LIBERO-Mem")
    logger.info(f"  Episodes: {n_ep}")
    logger.info(f"  Binary success: {n_succ}/{n_ep} ({100*n_succ/n_ep:.1f}%)")
    logger.info(f"  Subgoal completion rate: {avg_scr:.1%}")
    logger.info(f"  Total subgoals: {total_sg_done}/{total_sg_poss}")
    logger.info(f"{'='*70}")

    # Per-task breakdown
    logger.info("\nPer-task breakdown:")
    logger.info(f"{'Task':<6} {'Language':<55} {'SGs':>3} {'Success':>10} {'SCR':>8}")
    logger.info("-" * 90)
    for tr in all_results:
        eps = tr["episodes"]
        n_s = sum(1 for e in eps if e["success"])
        scr = np.mean([e["tiered_success"] for e in eps])
        logger.info(
            f"T{tr['task_id']+1:<4d} "
            f"{tr['language'][:55]:<55} "
            f"{tr['n_subgoals']:>3d} "
            f"{n_s:>3d}/{len(eps):<3d} "
            f"({100*n_s/len(eps):5.1f}%) "
            f"{scr:>6.1%}"
        )

    # No aggregate file is written here — per-episode results live in
    # eval_metadata.json next to each episode's videos under the
    # orchestrator's per-episode dir.  Roll up across episodes / runs
    # post-hoc when needed.


if __name__ == "__main__":
    main()
