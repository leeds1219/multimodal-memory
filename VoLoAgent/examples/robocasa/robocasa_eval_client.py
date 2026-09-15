#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoboCasa evaluation client for VLM-orchestrator.

Runs RoboCasa kitchen manipulation tasks, routing observations through
the orchestrator proxy (or directly to a VLA server).  Sends extra
observation keys for the proxy's failure signal detector and optional
GT state exporter.

Wire format (matches --env libero proxy mode):
  observation/image:            224×224 RGB (agentview_left)
  observation/wrist_image:      224×224 RGB (eye_in_hand)
  observation/state:            16D proprioceptive state
  observation/ee_pos:           3D EE position (for failure detector)
  observation/gripper_position: 1D gripper width (for failure detector)
  prompt:                       language string

Usage::

    # Terminal 1: RoboCasa pi0 policy server
    cd ~/robocasa-openpi
    uv run scripts/serve_policy.py --port 8002 policy:checkpoint \\
        --policy.config=pi0_robocasa_pretrain_human300 \\
        --policy.dir=<checkpoint>

    # Terminal 2: VLM orchestrator proxy
    vlm-orchestrator --env libero --mode subgoal \\
        --vla-port 8002 --port 8019

    # Terminal 3: This script (orchestrated)
    python examples/robocasa/robocasa_eval_client.py \\
        --port 8019 --task-set atomic_seen --split pretrain --num-trials 10

    # Or passthrough (direct to VLA):
    python examples/robocasa/robocasa_eval_client.py \\
        --port 8002 --task-set atomic_seen --split pretrain --num-trials 10
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import time
from datetime import datetime

import numpy as np

logger = logging.getLogger("robocasa_eval")


def parse_args():
    p = argparse.ArgumentParser(description="RoboCasa eval through orchestrator")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000,
                    help="Port of VLA server or orchestrator proxy")
    p.add_argument("--task-set", nargs="+", default=["atomic_seen"],
                    help="Task sets to evaluate (e.g. atomic_seen composite_seen)")
    p.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    p.add_argument("--num-trials", type=int, default=50,
                    help="Number of rollouts per task")
    p.add_argument("--max-tasks", type=int, default=0,
                    help="Max tasks to evaluate (0=all)")
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--save-video", action="store_true", default=False)
    p.add_argument("--enable-depth", action="store_true", default=False,
                    help="Enable depth rendering for grasp tool support")
    p.add_argument("--send-gt-state", action="store_true", default=False,
                    help="Send GT state for proxy's GT failure detector")
    return p.parse_args()


def _get_robosuite_env(env):
    """Unwrap a gymnasium-wrapped RoboCasa env to the raw robosuite env.

    Caches on the function object to avoid repeated traversal.
    """
    if not hasattr(_get_robosuite_env, "_cached"):
        inner = env
        while hasattr(inner, "env") and inner.env is not inner:
            inner = inner.env
        if hasattr(inner, "unwrapped"):
            inner = inner.unwrapped
        _get_robosuite_env._cached = inner
    return _get_robosuite_env._cached


def build_wire_obs(
    obs: dict,
    task_lang: str,
    resize_size: int,
    env=None,
    send_gt_state: bool = False,
    enable_depth: bool = False,
) -> dict:
    """Convert RoboCasa gym obs to orchestrator wire format.

    Args:
        obs: Raw gymnasium observation dict from RoboCasa.
        task_lang: Language instruction for this task.
        resize_size: Image resize target (224 for pi0).
        env: RoboCasa gym env (for extracting raw robosuite state).
        send_gt_state: If True, extract and include gt_state dict.
        enable_depth: If True, forward depth image for grasp tool.

    Returns:
        Dict ready to send via websocket to proxy/VLA.
    """
    from openpi_client import image_tools

    # Images — RoboCasa gym wrapper provides video.* keys
    img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
    wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])

    img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    # Proprioceptive state (16D for RoboCasa)
    state = np.concatenate([
        obs["state.end_effector_position_relative"],
        obs["state.end_effector_rotation_relative"],
        obs["state.base_position"],
        obs["state.base_rotation"],
        obs["state.gripper_qpos"],
    ], axis=0)

    wire = {
        "observation/image": img_resized,
        "observation/wrist_image": wrist_resized,
        "observation/state": state.astype(np.float32),
        "prompt": task_lang,
    }

    # Extra keys for the proxy's failure signal detector.
    # EE position: Use the relative position (already in state),
    # or extract absolute from the underlying robosuite env.
    wire["observation/ee_pos"] = obs[
        "state.end_effector_position_relative"
    ].astype(np.float64)
    wire["observation/gripper_position"] = np.mean(
        obs["state.gripper_qpos"]
    ).astype(np.float64)

    # Full-res images for VLM (proxy strips _raw before forwarding to VLA)
    wire["observation/image_raw"] = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, 256, 256)
    )
    wire["observation/wrist_image_raw"] = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 256, 256)
    )

    # GT state for GT failure detector (optional)
    if send_gt_state and env is not None:
        try:
            from vlm_orchestrator.aux_benchmarks.robocasa_gt import RoboCasaGTStateExporter
            if not hasattr(build_wire_obs, "_gt_exporter"):
                build_wire_obs._gt_exporter = RoboCasaGTStateExporter(env)
            wire["gt_state"] = build_wire_obs._gt_exporter.export(obs)
        except Exception as e:
            logger.debug(f"GT state export failed: {e}")

    # Depth image + camera matrices for grasp tool (optional)
    if enable_depth:
        # RoboCasa gym wrapper provides depth via video.* keys when
        # camera_depths=True on the underlying robosuite env.
        depth_key = "video.robot0_agentview_left_depth"
        depth = obs.get(depth_key)
        if depth is not None:
            depth_norm = np.squeeze(depth).astype(np.float32)
            # MuJoCo returns [0,1] normalised depth; convert to metres
            # using the scene's actual near/far clip planes.
            near, far = 0.01, 50.0  # MuJoCo defaults
            if env is not None:
                try:
                    rs = _get_robosuite_env(env)
                    extent = rs.sim.model.stat.extent
                    near = float(rs.sim.model.vis.map.znear * extent)
                    far = float(rs.sim.model.vis.map.zfar * extent)
                except Exception:
                    pass
            d = np.clip(depth_norm, 0.0, 0.9999)
            depth_m = (near / (1.0 - d * (1.0 - near / far))).astype(np.float32)
            wire["observation/depth_agentview"] = np.ascontiguousarray(depth_m)
        else:
            logger.warning(
                "enable_depth=True but depth key '%s' not in obs. "
                "Ensure camera_depths=True was passed to gym.make().",
                depth_key,
            )

        # Camera intrinsics and extrinsics for 3D point-cloud construction.
        # Navigate to the raw robosuite env to access the MuJoCo sim.
        if env is not None:
            try:
                from robosuite.utils.camera_utils import (
                    get_camera_extrinsic_matrix,
                    get_camera_intrinsic_matrix,
                )
                rs_env = _get_robosuite_env(env)
                cam_name = "robot0_agentview_left"
                cam_h = rs_env.camera_heights[0] if hasattr(rs_env, "camera_heights") else 256
                cam_w = rs_env.camera_widths[0] if hasattr(rs_env, "camera_widths") else 256
                K = get_camera_intrinsic_matrix(rs_env.sim, cam_name, cam_h, cam_w)
                T_cam = get_camera_extrinsic_matrix(rs_env.sim, cam_name)
                wire["observation/camera_K"] = K.flatten().astype(np.float64)
                wire["observation/camera_extrinsic"] = T_cam.flatten().astype(np.float64)
            except Exception as e:
                logger.debug(f"Camera matrix extraction failed: {e}")

    # GT segmentation for grasp tool GT_SIM mode (optional).
    # Render the full body-ID segmentation buffer so the proxy's
    # grasp tool can extract per-object masks without GDino/SAM.
    # (Outside enable_depth block — GT seg is useful independently.)
    if env is not None and send_gt_state:
        try:
            rs_env = _get_robosuite_env(env)
            seg_h = rs_env.camera_heights[0] if hasattr(rs_env, "camera_heights") else 256
            seg_w = rs_env.camera_widths[0] if hasattr(rs_env, "camera_widths") else 256
            seg = rs_env.sim.render(
                camera_name="robot0_agentview_left",
                height=seg_h, width=seg_w,
                segmentation=True,
            )
            # seg: (H, W, 2) — channel 0 = objtype, channel 1 = geom_id.
            # Convert geom_ids → body_ids via model.geom_bodyid.
            geom_ids = seg[:, :, 1].astype(np.int32)
            geom_bodyid = rs_env.sim.model.geom_bodyid
            body_ids = np.where(
                geom_ids >= 0,
                geom_bodyid[np.clip(geom_ids, 0, len(geom_bodyid) - 1)],
                -1,
            ).astype(np.int32)
            wire["gt_seg/body_ids"] = body_ids
            # Build name→body_id mapping from scene objects
            if not hasattr(build_wire_obs, "_obj_body_id"):
                obj_map = {}
                for obj in getattr(rs_env, "objects", []):
                    name = obj.name if hasattr(obj, "name") else str(obj)
                    try:
                        bid = rs_env.sim.model.body_name2id(name + "_main")
                        obj_map[name] = bid
                    except Exception:
                        try:
                            bid = rs_env.sim.model.body_name2id(name)
                            obj_map[name] = bid
                        except Exception:
                            pass
                build_wire_obs._obj_body_id = obj_map
            wire["gt_seg/obj_body_id"] = build_wire_obs._obj_body_id
        except Exception as e:
            logger.debug(f"GT seg render failed: {e}")

    return wire


def eval_task(
    env_name: str,
    split: str,
    num_trials: int,
    resize_size: int,
    replan_steps: int,
    host: str,
    port: int,
    seed: int,
    log_dir: pathlib.Path | None,
    save_video: bool,
    send_gt_state: bool,
    enable_depth: bool = False,
) -> dict:
    """Evaluate one RoboCasa task.

    Returns:
        Dict with task results.
    """
    import gymnasium as gym
    import robocasa  # noqa: F401 — registers envs
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    from openpi_client import websocket_client_policy

    task_horizon = get_task_horizon(env_name)
    horizon = int(task_horizon * 1.5)  # extra time (policy may be slow)

    logger.info(f"Task: {env_name} | horizon={horizon} (base={task_horizon}) | "
                f"trials={num_trials}")

    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    env = gym.make(
        f"robocasa/{env_name}", split=split, seed=seed,
        camera_depths=enable_depth,
    )

    task_successes = 0
    task_results = []

    for ep in range(num_trials):
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()

        # Reset caches for new episode
        if hasattr(build_wire_obs, "_gt_exporter"):
            del build_wire_obs._gt_exporter
        if hasattr(build_wire_obs, "_obj_body_id"):
            del build_wire_obs._obj_body_id
        if hasattr(_get_robosuite_env, "_cached"):
            del _get_robosuite_env._cached

        success = False
        replay_images = []
        ep_start = time.time()

        for t in range(horizon):
            wire_obs = build_wire_obs(
                obs, task_lang, resize_size,
                env=env if (send_gt_state or enable_depth) else None,
                send_gt_state=send_gt_state,
                enable_depth=enable_depth,
            )
            # Episode marker for proxy episode boundary detection.
            wire_obs["__episode_id"] = f"{env_name}_{ep}"
            # Sim step counter — lets the orchestrator gate VLM cadence
            # in step units rather than chunk units (pi05=8, robocasa
            # replan_steps=5 → without this, proxy fallback misestimates
            # by ~60%).
            wire_obs["__step"] = t

            if save_video and t % 2 == 0:
                replay_images.append(wire_obs["observation/image"])

            if not action_plan:
                response = client.infer(wire_obs)
                action_chunk = response["actions"]

                # Handle orchestrator flush signal
                if response.get("orchestrator_flush_actions"):
                    action_plan.clear()

                n_use = min(len(action_chunk), replan_steps)
                action_plan.extend(action_chunk[:n_use])

            action = action_plan.popleft()
            # Pad to 12D if needed (grasp tool returns 7D, some models
            # return 7D; convert_action expects up to 12D for
            # [ee_pos(3), ee_rot(3), grip(1), base(4), ctrl_mode(1)]).
            if len(action) < 12:
                action = np.concatenate([action, np.zeros(12 - len(action))])
            action = convert_action(action)
            obs, reward, done, truncated, info = env.step(action)
            success = info.get("success", False)

            if success:
                task_successes += 1
                break

        ep_time = time.time() - ep_start
        task_results.append({
            "episode": ep,
            "success": success,
            "steps": t + 1,
            "duration_s": round(ep_time, 1),
            "instruction": task_lang,
        })

        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"  [{ep+1}/{num_trials}] {status} | "
                     f"steps={t+1} | {ep_time:.1f}s | {task_lang[:60]}")

        # Save video
        if save_video and log_dir and replay_images:
            try:
                import imageio
                suffix = "success" if success else "failure"
                vid_path = log_dir / f"{env_name}_ep{ep}_{suffix}.mp4"
                imageio.mimwrite(str(vid_path), replay_images, fps=20)
            except Exception as e:
                logger.debug(f"Video save failed: {e}")

    env.close()

    sr = task_successes / max(num_trials, 1)
    logger.info(f"  → {env_name}: {task_successes}/{num_trials} ({sr*100:.1f}%)")

    return {
        "task": env_name,
        "split": split,
        "successes": task_successes,
        "total": num_trials,
        "success_rate": sr,
        "episodes": task_results,
    }


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    np.random.seed(args.seed)

    # Resolve log directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path(
        args.log_dir or f"results/robocasa/eval_{ts}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve task list from task sets
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    all_tasks = []
    for ts_name in args.task_set:
        if ts_name not in TASK_SET_REGISTRY:
            logger.error(f"Unknown task set: {ts_name}. "
                         f"Available: {list(TASK_SET_REGISTRY.keys())}")
            return
        all_tasks.extend(TASK_SET_REGISTRY[ts_name])

    if args.max_tasks > 0:
        all_tasks = all_tasks[:args.max_tasks]

    logger.info(f"Evaluating {len(all_tasks)} tasks × {args.num_trials} trials "
                f"on port {args.port}")
    logger.info(f"Tasks: {all_tasks}")

    # Run evaluations
    all_results = []
    total_succ = 0
    total_eps = 0

    for task_name in all_tasks:
        try:
            result = eval_task(
                env_name=task_name,
                split=args.split,
                num_trials=args.num_trials,
                resize_size=args.resize_size,
                replan_steps=args.replan_steps,
                host=args.host,
                port=args.port,
                seed=args.seed,
                log_dir=log_dir if args.save_video else None,
                save_video=args.save_video,
                send_gt_state=args.send_gt_state,
                enable_depth=args.enable_depth,
            )
            all_results.append(result)
            total_succ += result["successes"]
            total_eps += result["total"]
        except Exception as e:
            logger.error(f"Task {task_name} failed: {e}")
            all_results.append({
                "task": task_name,
                "error": str(e),
                "successes": 0,
                "total": args.num_trials,
                "success_rate": 0.0,
            })
            total_eps += args.num_trials

    # Final summary
    overall_sr = total_succ / max(total_eps, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS: RoboCasa ({args.split})")
    logger.info(f"  Task sets: {args.task_set}")
    logger.info(f"  Total: {total_succ}/{total_eps} ({overall_sr*100:.1f}%)")
    for r in all_results:
        sr = r["success_rate"] * 100
        logger.info(f"  {r['task']}: {r.get('successes',0)}/{r.get('total',0)} "
                     f"({sr:.1f}%)")
    logger.info(f"{'='*60}")

    # Save results
    results_path = log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "benchmark": "robocasa",
            "split": args.split,
            "task_sets": args.task_set,
            "port": args.port,
            "total_successes": total_succ,
            "total_episodes": total_eps,
            "success_rate": overall_sr,
            "tasks": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Generate markdown report
    report = generate_report(args, all_results, total_succ, total_eps)
    report_path = log_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")


def generate_report(args, results, total_succ, total_eps):
    """Generate a markdown report."""
    overall_sr = total_succ / max(total_eps, 1) * 100
    lines = [
        f"# RoboCasa Evaluation Report",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Split**: {args.split}",
        f"**Task sets**: {', '.join(args.task_set)}",
        f"**Port**: {args.port}",
        f"**Trials/task**: {args.num_trials}",
        f"",
        f"## Overall Results",
        f"",
        f"**Success rate: {total_succ}/{total_eps} ({overall_sr:.1f}%)**",
        f"",
        f"## Per-Task Results",
        f"",
        f"| Task | Successes | Total | Rate |",
        f"|------|-----------|-------|------|",
    ]
    for r in results:
        sr = r["success_rate"] * 100
        lines.append(
            f"| {r['task']} | {r.get('successes',0)} | {r.get('total',0)} | "
            f"{sr:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
