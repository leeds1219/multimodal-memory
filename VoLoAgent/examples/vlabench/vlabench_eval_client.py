#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLABench evaluation client for VLM-orchestrator.

Runs VLABench manipulation tasks, routing observations through the
orchestrator proxy (or directly to a VLA server).

Wire format sent to VLA / proxy:
  observation/image:            224×224 RGB (front camera, cam idx 2)
  observation/second_image:     224×224 RGB (right camera, cam idx 0)
  observation/wrist_image:      224×224 RGB (wrist camera, cam idx 3)
  observation/state:            8D [ee_pos_offset(3), ee_euler(3), gripper(1), pad(1)]
  prompt:                       language string

  Extra keys for orchestrator failure signals:
  observation/ee_pos:           3D EE position (world frame)
  observation/gripper_position: 1D gripper state

Usage::

    # Terminal 1: VLABench pi0.5 policy server
    cd ~/VLABench/third_party/openpi && git checkout pi05
    PYTHONPATH=src ~/openpi/.venv/bin/python scripts/serve_policy.py \\
        --port 8002 policy:checkpoint \\
        --policy.config=pi05_ft_vlabench_primitive \\
        --policy.dir=~/vlabench-checkpoints/pi05-primitive-10task

    # Terminal 2: VLM orchestrator proxy
    vlm-orchestrator --env libero --mode subgoal \\
        --vla-port 8002 --port 8019

    # Terminal 3: This script (orchestrated)
    python examples/vlabench/vlabench_eval_client.py \\
        --port 8019 --eval-track track_1_in_distribution --n-episode 5

    # Or passthrough (direct to VLA):
    python examples/vlabench/vlabench_eval_client.py \\
        --port 8002 --eval-track track_1_in_distribution --n-episode 5
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import pathlib
import sys
import time
import traceback
from datetime import datetime

import numpy as np

# Ensure the vlm_orchestrator package is importable when running this
# script directly. The vlabench conda env (unlike .libero-venv) does
# not have vlm-orchestrator installed editably, and Python only adds
# the script's own directory to sys.path; without this, the annotated-
# video import (vlm_orchestrator.utils.video_annotator) fails silently.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger("vlabench_eval")


def parse_args():
    p = argparse.ArgumentParser(description="VLABench eval through orchestrator")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000,
                    help="Port of VLA server or orchestrator proxy")
    p.add_argument("--tasks", nargs="+", default=None,
                    help="Specific task names (e.g. select_fruit select_toy)")
    p.add_argument("--eval-track", default="track_1_in_distribution",
                    choices=[
                        "track_1_in_distribution",
                        "track_2_cross_category",
                        "track_3_common_sense",
                        "track_4_semantic_instruction",
                        "track_6_unseen_texture",
                    ])
    p.add_argument("--n-episode", type=int, default=50,
                    help="Number of episodes per task")
    p.add_argument("--max-tasks", type=int, default=0,
                    help="Max tasks to evaluate (0=all)")
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200,
                    help="Max steps per episode (default from task_config or 200)")
    p.add_argument("--num-steps-wait", type=int, default=10,
                    help="Steps to wait for objects to settle (handled by VLABench)")
    p.add_argument("--enable-depth", action="store_true", default=False,
                    help="Enable depth rendering for grasp tool support (~10%% overhead)")
    p.add_argument("--send-gt-state", action="store_true", default=False,
                    help="Send GT state for proxy's GT failure detector")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--no-video", action="store_true", default=False,
                    help="Skip saving rollout videos (videos are on by default)")
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument(
        "--composite-config",
        default=None,
        help="Path to JSON produced by scripts/vlabench_probe_composite_seeds.py. "
             "When set, each task uses its pre-validated seed for ALL episodes "
             "(deterministic init across runs, no physics-error retry loop). "
             "Tasks absent from the JSON or with seed=null are skipped.",
    )
    return p.parse_args()


def quaternion_to_euler(quat):
    """Convert quaternion [w, x, y, z] to euler [rx, ry, rz]."""
    from scipy.spatial.transform import Rotation as R
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
    return r.as_euler('xyz', degrees=False)


def euler_to_quaternion_wxyz(euler):
    """Convert euler [rx, ry, rz] to quaternion [w, x, y, z]."""
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('xyz', euler)
    q = r.as_quat()  # returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])


def build_wire_obs(
    observation: dict,
    resize_size: int,
    enable_depth: bool = False,
    send_gt_state: bool = False,
    env=None,
    task_slug: str | None = None,
) -> dict:
    """Convert VLABench observation to wire format for VLA server.

    The VLABench env provides:
      - observation["rgb"]: list of 4 camera images [right, left, front, wrist]
        at 480×480 resolution
      - observation["depth"]: list of 4 depth images (always present)
      - observation["instrinsic"]: list of 4 intrinsic matrices (3×3)
      - observation["extrinsic"]: list of 4 extrinsic matrices (4×4)
      - observation["ee_state"]: [ee_pos(3), ee_quat(4), gripper_state(1)]
      - observation["instruction"]: language prompt
      - observation["robot_frame"]: robot base position (3D)

    Maps to VLABench policy input format:
      - observation/image:        front camera (idx 2)
      - observation/second_image: right camera (idx 0)
      - observation/wrist_image:  wrist camera (idx 3)
      - observation/state:        [pos-robot_frame, euler, gripper, 0] (8D, padded)
      - prompt:                   instruction string
    """
    from openpi_client import image_tools

    # Extract cameras: right(0), left(1), front(2), wrist(3)
    right_img, _left_img, front_img, wrist_img = observation["rgb"]

    # Resize to 224×224
    front_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(front_img, resize_size, resize_size)
    )
    right_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(right_img, resize_size, resize_size)
    )
    wrist_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    # EE state
    ee_state = observation["ee_state"]
    pos = ee_state[:3].copy()
    quat = ee_state[3:7]
    gripper_state = ee_state[-1]
    robot_frame = observation["robot_frame"]

    # Policy input state: pos offset from robot frame, euler angles, gripper
    ee_euler = quaternion_to_euler(quat)
    pos_offset = pos - robot_frame
    state = np.concatenate([
        pos_offset,
        ee_euler,
        np.array(gripper_state).reshape(-1),
    ]).astype(np.float32)

    wire = {
        "observation/image": front_resized,
        "observation/second_image": right_resized,
        "observation/wrist_image": wrist_resized,
        "observation/state": state,
        "prompt": observation["instruction"],
    }

    # Extra keys for orchestrator failure signal detector
    wire["observation/ee_pos"] = pos.astype(np.float64)
    wire["observation/gripper_position"] = np.float64(gripper_state)
    # EE quaternion (w,x,y,z) for grasp tool orientation planning
    wire["observation/ee_quat"] = quat.astype(np.float64)

    # Stable task slug. VLABench prompts vary per episode (different
    # target object), so we hand the orchestrator a stable name to use
    # for episode-dir naming. Without this, three eps of select_drink
    # would land in three separate "Take_out_the_*" dirs.
    if task_slug:
        wire["__task_slug"] = task_slug

    # ── Wide third-person camera for VLM consumption ──
    # VLABench's "front" camera (idx 2 in observation/rgb) is a tight
    # top-down workspace close-up — bad for VLM scene understanding,
    # since the robot, table, and surrounding context are all cropped
    # out. The "right" / "left" cameras (idx 0 / idx 1) are wide
    # third-person views with the full scene in frame. We pack idx 0
    # ("right") as observation/front_image_left so that launching the
    # orchestrator with --use-front-camera routes the VLM here. The
    # policy still receives observation/image (the trained idx 2
    # view) and the grasp tool still receives observation/image_for_grasp
    # (idx 2 with matching camera_K). LIBERO/robolab don't pack this
    # key with the same content, so they're unaffected.
    wire["observation/front_image_left"] = np.ascontiguousarray(
        np.asarray(right_img, dtype=np.uint8)
    )

    # Depth image for grasp tool (front camera = idx 2).
    # VLABench always renders depth (not gated by require_pcd).
    if enable_depth:
        depth_images = observation.get("depth")
        if depth_images is not None and len(depth_images) > 2:
            front_depth = depth_images[2]  # front camera depth
            if front_depth is not None:
                depth_squeezed = np.ascontiguousarray(
                    np.squeeze(front_depth).astype(np.float32)
                )
                wire["observation/depth_agentview"] = depth_squeezed

        # ── Grasp-tool-specific image + depth, native 480x480 ──
        # VLABench's dm_control physics.render() returns images already
        # in y-down image convention (verified empirically on saved
        # rollout_raw.mp4 frames — the table is at the bottom of the
        # array, matching OpenCV / camera_K's expectation). So unlike
        # LIBERO (which uses robosuite's IMAGE_CONVENTION="opengl",
        # rendering y-up and requiring [::-1] flips), VLABench needs
        # NO mirror operation here.
        #
        # Why the dedicated _for_grasp keys at all? Two reasons:
        # (1) Resolution: observation/image is 224x224 (resize-with-pad
        #     for the policy), but camera_K / camera_extrinsic / depth
        #     are calibrated/rendered at native 480x480. Feeding the
        #     224 image into GDino+SAM2 then back-projecting with the
        #     480 camera_K produced spatial misalignment between mask
        #     and depth pointcloud.
        # (2) The grasp tool's _extract_image, in GraspEnvMode.LIBERO
        #     branch, applies an unconditional [:, ::-1] X-flip (a
        #     LIBERO-specific un-mirror). VLABench needs that flip
        #     skipped. The new wire keys give the grasp tool a path
        #     that bypasses the LIBERO-specific mirror.
        # LIBERO/robolab clients don't pack these keys → grasp tool
        # falls through to its existing branches unchanged.
        wire["observation/image_for_grasp"] = np.ascontiguousarray(
            np.asarray(front_img, dtype=np.uint8)
        )
        if depth_images is not None and len(depth_images) > 2:
            fd = depth_images[2]
            if fd is not None:
                wire["observation/depth_for_grasp"] = np.ascontiguousarray(
                    np.squeeze(fd).astype(np.float32)
                )

        # Camera intrinsics and extrinsics for 3D point-cloud construction.
        # VLABench always renders these (front camera = idx 2).
        intrinsics = observation.get("instrinsic")  # VLABench typo: "instrinsic"
        extrinsics = observation.get("extrinsic")
        if intrinsics is not None and len(intrinsics) > 2:
            K = np.asarray(intrinsics[2], dtype=np.float64)  # front camera
            wire["observation/camera_K"] = K.flatten()
        if extrinsics is not None and len(extrinsics) > 2:
            T = np.asarray(extrinsics[2], dtype=np.float64)  # front camera
            wire["observation/camera_extrinsic"] = T.flatten()

    # GT state for GT failure detector (optional)
    if send_gt_state and env is not None:
        try:
            from vlm_orchestrator.aux_benchmarks.vlabench_gt import VLABenchGTStateExporter
            if not hasattr(build_wire_obs, "_gt_exporter"):
                build_wire_obs._gt_exporter = VLABenchGTStateExporter(env)
            wire["gt_state"] = build_wire_obs._gt_exporter.export(observation)
        except Exception as e:
            logger.debug(f"GT state export failed: {e}")

        # GT segmentation for grasp tool GT_SIM mode.
        # VLABench always renders segmentation (front camera = idx 2).
        # dm_control seg: (H, W, 2) — channel 0 = geom type, channel 1 = geom id.
        # The grasp tool expects BODY ids, so we remap via geom_bodyid.
        # dm_control seg: (H, W, 2) — channel 0 = object ID, channel 1 = object type.
        seg_images = observation.get("segmentation")
        if seg_images is not None and len(seg_images) > 2:
            front_seg = np.asarray(seg_images[2], dtype=np.int32)
            if front_seg.ndim == 3 and front_seg.shape[2] >= 2:
                geom_ids = front_seg[:, :, 0]  # channel 0 = geom ID in dm_control
            elif front_seg.ndim == 2:
                geom_ids = front_seg
            else:
                geom_ids = None
            if geom_ids is not None:
                # Map geom ids → body ids.  Negative ids (background) stay -1.
                geom_bodyid = env.physics.model.geom_bodyid
                body_ids = np.where(
                    geom_ids >= 0,
                    geom_bodyid[np.clip(geom_ids, 0, len(geom_bodyid) - 1)],
                    -1,
                ).astype(np.int32)
                wire["gt_seg/body_ids"] = body_ids
            # Build name→body_id mapping from dm_control physics
            if not hasattr(build_wire_obs, "_obj_body_id"):
                obj_map = {}
                try:
                    physics = env.physics
                    for i in range(physics.model.nbody):
                        name = physics.model.id2name(i, "body")
                        if name and name != "world":
                            obj_map[name] = i
                except Exception:
                    pass
                build_wire_obs._obj_body_id = obj_map
            wire["gt_seg/obj_body_id"] = build_wire_obs._obj_body_id

    return wire


def decode_action(raw_action: np.ndarray, robot_frame: np.ndarray):
    """Convert policy output action to VLABench format.

    Args:
        raw_action: 7D action [pos_offset(3), euler(3), gripper(1)]
        robot_frame: Robot base position (3D) to add back to EE target

    Returns:
        (target_pos, target_euler, gripper_state)
    """
    target_pos = raw_action[:3].copy()
    target_euler = raw_action[3:6]
    gripper = raw_action[-1]

    # Undo position offset: model outputs pos relative to robot frame
    target_pos += robot_frame

    # Binarize gripper: >= 0.1 → open (0.04, 0.04), < 0.1 → closed (0, 0)
    if gripper >= 0.1:
        gripper_state = np.ones(2) * 0.04
    else:
        gripper_state = np.zeros(2)

    return target_pos, target_euler, gripper_state


def run_single_episode(
    env,
    client,
    robot_frame: np.ndarray,
    max_steps: int,
    replan_steps: int,
    resize_size: int,
    task_name: str,
    episode_idx: int,
    enable_depth: bool = False,
    send_gt_state: bool = False,
    tolerance: float = 1e-2,
    max_substeps: int = 1,
    episode_metadata: dict | None = None,
) -> dict:
    """Run a single evaluation episode.

    Follows the reference evaluator logic from VLABench/evaluation/evaluator/base.py.
    """
    action_plan = collections.deque()
    success = False
    last_action = None
    step = 0
    grasp_tool_active = False  # persists across iterations for multi-step grasp
    # Snapshot of EE pose at the start of a grasp-tool invocation.
    # Grasp-tool actions are OSC_POSE-normalised EE-deltas planned as a
    # continuous trajectory; we accumulate them onto this fixed origin
    # rather than onto the drifting current pose, so per-step IK
    # convergence error doesn't compound across the chunk. Reset to
    # None whenever grasp_tool_active transitions back to False.
    grasp_chunk_pos_origin: np.ndarray | None = None
    grasp_chunk_euler_origin: np.ndarray | None = None
    grasp_chunk_pos_cumsum: np.ndarray = np.zeros(3)
    grasp_chunk_euler_cumsum: np.ndarray = np.zeros(3)
    frames: list[np.ndarray] = []  # 2x2 tiled multi-view frames for video
    # Per-step inputs for annotated video (only used when running
    # orchestrated; passthrough rollouts get only rollout_raw.mp4).
    front_full_frames: list[np.ndarray] = []
    wrist_full_frames: list[np.ndarray] = []
    responses_per_step: list[dict] = []
    last_response: dict = {}
    # Capture the original task instruction (the per-episode random
    # target prompt — e.g. "Please select the painting of style
    # ukiyo-e") on the first step so it can be saved alongside the
    # task family slug in eval_metadata.json.
    original_instruction: str | None = None

    while step < max_steps:
        observation = env.get_observation(require_pcd=False)
        # Lock instruction at step 0. Some VLABench tasks (e.g.
        # density_qa, weight_qa) define a LIST of alternative phrasings
        # in get_instruction() and randomly return one per call. Calling
        # it every step would flip the prompt mid-episode, which fools
        # the orchestrator's prompt-changed boundary detector into
        # rotating episode logs every step. Lock once on step 0 so the
        # policy and orchestrator both see a single stable instruction.
        if step == 0:
            original_instruction = env.task.get_instruction()
        observation["instruction"] = original_instruction
        observation["robot_frame"] = robot_frame
        # Capture all 4 cameras tiled 2x2 (top=[right,left], bottom=[front,
        # wrist]), matching VLABench's reference save_video at
        # evaluation/evaluator/base.py:207. Each cam is 480x480 → tiled
        # frame is 960x960.
        rgb = observation.get("rgb")
        if rgb is not None and len(rgb) >= 4:
            arr = np.asarray(rgb, dtype=np.uint8)
            tiled = np.vstack([
                np.hstack(arr[:2]),
                np.hstack(arr[2:4]),
            ])
            frames.append(tiled)
            # Annotated video uses the wide third-person right camera
            # (idx 0), matching what observation/front_image_left exposes
            # to the VLM. This makes the annotation overlay sit on the
            # same scene the VLM is reasoning about, instead of the
            # tight idx-2 top-down view the policy consumes.
            front_full_frames.append(arr[0].copy())
            wrist_full_frames.append(arr[3].copy())
            responses_per_step.append(dict(last_response))

        ee_state = observation["ee_state"]
        if last_action is None:
            last_action = np.concatenate([
                ee_state[:3], quaternion_to_euler(ee_state[3:7])
            ])
        observation["last_action"] = last_action

        # Reset caches per episode (task/env may change between episodes)
        if step == 0:
            if hasattr(build_wire_obs, "_gt_exporter"):
                del build_wire_obs._gt_exporter
            if hasattr(build_wire_obs, "_obj_body_id"):
                del build_wire_obs._obj_body_id

        # Build wire observation and infer
        wire_obs = build_wire_obs(
            observation, resize_size,
            enable_depth=enable_depth,
            send_gt_state=send_gt_state,
            env=env if send_gt_state else None,
            task_slug=task_name,
        )
        # Episode marker for proxy episode boundary detection.
        wire_obs["__episode_id"] = f"{task_name}_{episode_idx}"
        # Sim step counter — lets the orchestrator gate VLM cadence in
        # step units rather than chunk units (pi05=8, vlabench
        # replan_steps=5 → without this, proxy fallback misestimates
        # by ~60%).
        wire_obs["__step"] = step

        if not action_plan:
            response = client.infer(wire_obs)
            last_response = response
            action_chunk = response["actions"]

            if episode_metadata is not None:
                if (episode_metadata.get("episode_log_dir") is None
                        and "orchestrator_episode_log_dir" in response):
                    episode_metadata["episode_log_dir"] = (
                        response["orchestrator_episode_log_dir"]
                    )
                if "orchestrator_subgoals" in response:
                    episode_metadata["orchestrator_subgoals"] = (
                        response["orchestrator_subgoals"]
                    )
                if "orchestrator_failure" in response:
                    episode_metadata.setdefault(
                        "orchestrator_failures", []
                    ).append({
                        "step": step,
                        "failure": response["orchestrator_failure"],
                    })

            # Orchestrator may request flushing action queue
            if response.get("orchestrator_flush_actions"):
                action_plan.clear()

            # Check if grasp tool is producing these actions (EE-deltas,
            # not EE-targets like the VLA policy outputs).
            grasp_info = response.get("orchestrator_grasp_tool", {})
            grasp_tool_active = grasp_info.get("active", False)

            n_use = min(len(action_chunk), replan_steps)
            action_plan.extend(action_chunk[:n_use])

        raw_action = action_plan.popleft()

        if grasp_tool_active:
            # Grasp tool actions are OSC_POSE-NORMALIZED EE-deltas
            # ([-1, 1] → ±OSC_POSE_POS_SCALE m / ±OSC_POSE_ORI_SCALE rad
            # per env step).  LIBERO's OSC_POSE controller scales these
            # internally and absorbs per-step error elastically;
            # VLABench's joint-pos env IKs each target rigidly with no
            # compliance.  Naively applying delta-from-current EE pose
            # at every step compounds IK convergence error across the
            # chunk — the robot drifts behind the planned trajectory
            # and never reaches the final grasp pose.
            #
            # Fix: snapshot the EE pose at the start of a grasp
            # invocation and accumulate scaled deltas onto that fixed
            # origin.  IK then targets the planned trajectory point
            # regardless of where the robot has drifted to, giving
            # the controller a stable reference to converge against.
            from vlm_orchestrator.grasp.tool import (
                OSC_POSE_POS_SCALE, OSC_POSE_ORI_SCALE,
            )
            if grasp_chunk_pos_origin is None:
                # Start of a new grasp invocation — snapshot reference.
                grasp_chunk_pos_origin = ee_state[:3].copy().astype(np.float64)
                grasp_chunk_euler_origin = quaternion_to_euler(
                    ee_state[3:7]
                ).astype(np.float64)
                grasp_chunk_pos_cumsum = np.zeros(3, dtype=np.float64)
                grasp_chunk_euler_cumsum = np.zeros(3, dtype=np.float64)
            grasp_chunk_pos_cumsum = (
                grasp_chunk_pos_cumsum
                + raw_action[:3] * OSC_POSE_POS_SCALE
            )
            grasp_chunk_euler_cumsum = (
                grasp_chunk_euler_cumsum
                + raw_action[3:6] * OSC_POSE_ORI_SCALE
            )
            target_pos = grasp_chunk_pos_origin + grasp_chunk_pos_cumsum
            target_euler = (
                grasp_chunk_euler_origin + grasp_chunk_euler_cumsum
            )
            # Grasp tool uses LIBERO convention: -1=open, +1=closed.
            # VLABench convention: 0.04=open, 0=closed.
            gripper_val = raw_action[6] if len(raw_action) > 6 else -1.0
            gripper_state = np.ones(2) * 0.04 if gripper_val < 0.0 else np.zeros(2)
        else:
            # Grasp invocation ended (or never started this chunk).
            # Reset snapshot so the next invocation starts fresh.
            if grasp_chunk_pos_origin is not None:
                grasp_chunk_pos_origin = None
                grasp_chunk_euler_origin = None
            target_pos, target_euler, gripper_state = decode_action(
                raw_action, robot_frame
            )

        last_action = np.concatenate([target_pos, target_euler])

        # Convert EE target to joint positions via IK
        quat = euler_to_quaternion_wxyz(target_euler)
        _, joint_action = env.robot.get_qpos_from_ee_pos(
            physics=env.physics, pos=target_pos, quat=quat
        )
        action = np.concatenate([joint_action, gripper_state])

        # Step environment (with substep convergence check).
        #
        # While the grasp tool is active, allow many more physics
        # substeps per action so the joint-position IK target has
        # time to converge smoothly. Each grasp step can request up
        # to ~5cm EE motion (LIBERO_MAX_POS_DELTA = 0.05 m), which
        # in joint space is a large discontinuous setpoint move; with
        # only 1 physics step (the default for policy actions),
        # MuJoCo sets joints discontinuously and the arm oscillates
        # wildly off the planned trajectory. LIBERO's OSC_POSE
        # controller hides this complexity internally; VLABench's
        # joint-position env does not.
        active_substeps = 50 if grasp_tool_active else max_substeps
        for _sub in range(active_substeps):
            timestep = env.step(action)
            if timestep.last():
                success = True
                break
            current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
            if (np.max(current_qpos - np.array(action)[:7]) < tolerance and
                    np.min(current_qpos - np.array(action)[:7]) > -tolerance):
                break

        if success:
            break
        step += 1

    # Get extra metrics
    try:
        intention_score = env.get_intention_score(threshold=0.2)
    except Exception:
        intention_score = 0.0
    try:
        progress_score = env.get_task_progress()
    except Exception:
        progress_score = 0.0

    return {
        "success": success,
        "steps": step + 1,
        "intention_score": intention_score,
        "progress_score": progress_score,
        "frames": frames,
        "front_full_frames": front_full_frames,
        "wrist_full_frames": wrist_full_frames,
        "responses_per_step": responses_per_step,
        "instruction": original_instruction,
    }


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    np.random.seed(args.seed)

    # Set VLABENCH_ROOT if not already set
    vlabench_root = os.getenv("VLABENCH_ROOT")
    if not vlabench_root:
        vlabench_root = str(pathlib.Path.home() / "VLABench" / "VLABench")
        os.environ["VLABENCH_ROOT"] = vlabench_root

    # Set MUJOCO_GL for headless rendering
    os.environ["MUJOCO_GL"] = "egl"

    # Import VLABench (must be after setting env vars)
    from VLABench.envs import load_env, TASK_CONFIG
    from VLABench.configs import name2config
    from VLABench.utils.utils import find_key_by_value
    import VLABench.tasks   # register tasks
    import VLABench.robots  # register robots
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    # Resolve log directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path(args.log_dir or f"results/vlabench/eval_{ts}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load episode configs from track. Explicit --tasks takes precedence
    # over --eval-track so composite task names (which have no track JSON)
    # can be specified directly without loading a primitive-track config.
    episode_configs = None
    if args.tasks:
        tasks = args.tasks
    elif args.eval_track:
        track_path = os.path.join(
            vlabench_root, "configs/evaluation/tracks",
            f"{args.eval_track}.json"
        )
        with open(track_path) as f:
            episode_configs = json.load(f)
        tasks = list(episode_configs.keys())
    else:
        tasks = ["select_fruit"]

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    # Composite-config mode: each task's init is determined by a single
    # pre-validated seed (produced by scripts/vlabench_probe_composite_seeds.py).
    # Tasks absent from the JSON or with seed=null are hard-skipped — we
    # never want eval numbers polluted by retry loops on tasks the probe
    # already determined have no stable random_init.
    composite_config: dict | None = None
    if args.composite_config:
        with open(args.composite_config) as f:
            composite_config = json.load(f)
        cc_tasks = composite_config.get("tasks", {})
        kept, skipped_no_entry, skipped_null = [], [], []
        for t in tasks:
            entry = cc_tasks.get(t)
            if entry is None:
                skipped_no_entry.append(t)
            elif entry.get("seed") is None:
                skipped_null.append(t)
            else:
                kept.append(t)
        if skipped_no_entry:
            logger.warning(
                f"  composite-config: {len(skipped_no_entry)} task(s) "
                f"not in JSON, skipping: {sorted(skipped_no_entry)}"
            )
        if skipped_null:
            logger.warning(
                f"  composite-config: {len(skipped_null)} task(s) have "
                f"seed=null (probe failed), skipping: {sorted(skipped_null)}"
            )
        tasks = kept
        if not tasks:
            logger.error(
                "  composite-config: no tasks remain after filtering. "
                "Either re-run the probe or check --tasks selection."
            )
            return

    logger.info(f"Evaluating {len(tasks)} tasks × {args.n_episode} episodes "
                f"on port {args.port}")
    logger.info(f"Tasks: {tasks}")

    # Connect to policy server / proxy
    client = WebsocketClientPolicy(args.host, args.port)

    all_results = []
    total_succ = 0
    total_eps = 0

    for task_name in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task_name}")
        logger.info(f"{'='*60}")

        task_successes = 0
        task_results = []
        task_intention_scores = []
        task_progress_scores = []

        # Get per-task max episode length from task_config
        task_series = find_key_by_value(name2config, task_name)
        task_cfg = TASK_CONFIG.get(task_series, {})
        max_episode_length = task_cfg.get("evaluation", {}).get(
            "max_episode_length", args.max_steps
        )

        # Get episode configs for this task
        task_episodes = None
        if episode_configs and task_name in episode_configs:
            task_episodes = episode_configs[task_name]

        for ep in range(args.n_episode):
            try:
                ep_config = None
                if task_episodes and ep < len(task_episodes):
                    ep_config = task_episodes[ep]

                # Environment construction. Three sources of init,
                # in priority order:
                #   1. composite_config: pre-validated seed (deterministic
                #      across all eps of the task; same seed fed to both
                #      passthrough and orchestrated runs for fair compare).
                #   2. ep_config (track JSON): deterministic per-episode
                #      configs from VLABench's primitive eval tracks.
                #   3. random_init with seed=42+ep, plus a 4-attempt
                #      retry loop on PhysicsError (legacy fallback —
                #      only fires when neither composite-config nor
                #      track JSON applies).
                from dm_control.rl.control import PhysicsError
                import random as _random

                env = None
                if composite_config is not None:
                    # Composite-config mode. Single seed, single attempt
                    # — probe already validated reset+5-step stability.
                    seed = composite_config["tasks"][task_name]["seed"]
                    np.random.seed(seed)
                    _random.seed(seed)
                    env = load_env(
                        task_name,
                        random_init=True,
                        run_mode="eval",
                    )
                    env.reset()
                else:
                    MAX_RESET_RETRIES = 4
                    for retry in range(MAX_RESET_RETRIES):
                        try:
                            if ep_config is not None:
                                env = load_env(
                                    task_name,
                                    episode_config=ep_config,
                                    random_init=False,
                                    run_mode="eval",
                                )
                            else:
                                seed = 42 + ep + retry * 997
                                np.random.seed(seed)
                                _random.seed(seed)
                                env = load_env(
                                    task_name,
                                    random_init=True,
                                    run_mode="eval",
                                )
                            env.reset()
                            break
                        except PhysicsError as pe:
                            logger.warning(
                                f"  reset PhysicsError "
                                f"(attempt {retry+1}/{MAX_RESET_RETRIES}): {pe}"
                            )
                            try:
                                if env is not None:
                                    env.close()
                            except Exception:
                                pass
                            env = None
                            if ep_config is not None:
                                # Deterministic config can't be reseeded.
                                raise
                            if retry == MAX_RESET_RETRIES - 1:
                                raise
                robot_frame = env.get_robot_frame_position()

                episode_metadata: dict = {
                    "task_slug": task_name,
                    "episode_idx": ep,
                    "instruction": None,  # set after run_single_episode
                    "start_time": time.time(),
                    "orchestrator_subgoals": None,
                    "orchestrator_failures": [],
                    "episode_log_dir": None,
                }

                ep_start = time.time()
                result = run_single_episode(
                    env=env,
                    client=client,
                    robot_frame=robot_frame,
                    max_steps=max_episode_length,
                    replan_steps=args.replan_steps,
                    resize_size=args.resize_size,
                    task_name=task_name,
                    episode_idx=ep,
                    enable_depth=args.enable_depth,
                    send_gt_state=args.send_gt_state,
                    episode_metadata=episode_metadata,
                )
                ep_time = time.time() - ep_start

                if result["success"]:
                    task_successes += 1

                task_intention_scores.append(result["intention_score"])
                task_progress_scores.append(result["progress_score"])

                task_results.append({
                    "episode": ep,
                    "success": result["success"],
                    "steps": result["steps"],
                    "duration_s": round(ep_time, 1),
                    "intention_score": result["intention_score"],
                    "progress_score": result["progress_score"],
                })

                # Per-episode eval_metadata.json for build_libero_eval.py.
                # Use orchestrator's advertised dir when present (mirrors
                # the LIBERO client); otherwise fall back to log_dir/<task>/
                # episode_<N>/.
                episode_metadata["end_time"] = time.time()
                episode_metadata["num_steps"] = result["steps"]
                episode_metadata["success"] = result["success"]
                episode_metadata["duration_s"] = ep_time
                episode_metadata["intention_score"] = result["intention_score"]
                episode_metadata["progress_score"] = result["progress_score"]
                episode_metadata["instruction"] = result.get("instruction")

                if episode_metadata["episode_log_dir"]:
                    ep_dir = pathlib.Path(episode_metadata["episode_log_dir"])
                else:
                    ep_dir = log_dir / task_name / f"episode_{ep}"
                try:
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    # Strip frames out of metadata payload (they belong in mp4)
                    meta_to_save = {
                        k: v for k, v in episode_metadata.items()
                        if k != "frames"
                    }
                    with open(ep_dir / "eval_metadata.json", "w") as f:
                        json.dump(meta_to_save, f, indent=2, default=str)
                except Exception as e:
                    logger.debug(f"  eval_metadata save failed: {e}")

                if not args.no_video and result.get("frames"):
                    try:
                        import imageio
                        video_path = ep_dir / "rollout_raw.mp4"
                        imageio.mimsave(
                            str(video_path),
                            result["frames"],
                            fps=args.video_fps,
                        )
                    except Exception as e:
                        logger.warning(f"  video save failed: {e}")

                # Annotated video: only when running orchestrated (some
                # response in this episode carried orchestrator_* keys).
                # Skip for passthrough — the annotation overlay would be
                # empty and just duplicate rollout_raw.mp4.
                ran_orchestrated = (
                    episode_metadata.get("episode_log_dir") is not None
                    or any(
                        any(
                            k.startswith("orchestrator_")
                            for k in (r or {}).keys()
                        )
                        for r in result.get("responses_per_step", [])
                    )
                )
                logger.info(
                    f"  annot debug: ran_orchestrated={ran_orchestrated} "
                    f"front_frames={len(result.get('front_full_frames', []))} "
                    f"responses={len(result.get('responses_per_step', []))} "
                    f"no_video={args.no_video}"
                )
                if (not args.no_video
                        and ran_orchestrated
                        and result.get("front_full_frames")):
                    annot_path = ep_dir / "rollout_annotated.mp4"
                    logger.info(f"  writing annotated video → {annot_path}")
                    try:
                        from vlm_orchestrator.utils.video_annotator import (
                            EpisodeVideoWriter,
                        )
                        writer = EpisodeVideoWriter(
                            path=str(annot_path),
                            fps=args.video_fps,
                        )
                        n = len(result["front_full_frames"])
                        for i in range(n):
                            writer.add_frame(
                                raw_image=result["front_full_frames"][i],
                                step=i,
                                response=result["responses_per_step"][i],
                                raw_wrist_image=(
                                    result["wrist_full_frames"][i]
                                    if i < len(result["wrist_full_frames"])
                                    else None
                                ),
                            )
                        writer.close(success=result["success"])
                        logger.info(
                            f"  annotated video done "
                            f"(exists={annot_path.exists()}, "
                            f"size={annot_path.stat().st_size if annot_path.exists() else 0})"
                        )
                    except Exception as e:
                        import traceback as _tb
                        logger.error(
                            f"  annotated video save FAILED: "
                            f"{type(e).__name__}: {e}\n{_tb.format_exc()}"
                        )

                status = "SUCCESS" if result["success"] else "FAILURE"
                logger.info(
                    f"  [{ep+1}/{args.n_episode}] {status} | "
                    f"steps={result['steps']}/{max_episode_length} | "
                    f"{ep_time:.1f}s | "
                    f"progress={result['progress_score']:.2f}"
                )

                env.close()

            except Exception as e:
                logger.error(f"  Episode {ep} error: {e}")
                traceback.print_exc()
                task_results.append({
                    "episode": ep,
                    "success": False,
                    "error": str(e),
                })
                try:
                    env.close()
                except Exception:
                    pass
                # Write a stub eval_metadata.json so the aggregator counts
                # this episode in the denominator (otherwise SR is inflated
                # by skipping it).
                try:
                    ep_dir = log_dir / task_name / f"episode_{ep}"
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    with open(ep_dir / "eval_metadata.json", "w") as f:
                        json.dump({
                            "task_slug": task_name,
                            "episode_idx": ep,
                            "success": False,
                            "num_steps": 0,
                            "duration_s": 0.0,
                            "error": str(e),
                        }, f, indent=2, default=str)
                except Exception:
                    pass

        total_succ += task_successes
        total_eps += args.n_episode

        sr = task_successes / max(args.n_episode, 1)
        avg_progress = np.mean(task_progress_scores) if task_progress_scores else 0
        logger.info(
            f"  → {task_name}: {task_successes}/{args.n_episode} "
            f"({sr*100:.1f}%) | avg_progress={avg_progress:.3f}"
        )

        all_results.append({
            "task": task_name,
            "successes": task_successes,
            "total": args.n_episode,
            "success_rate": sr,
            "avg_intention_score": float(np.mean(task_intention_scores)) if task_intention_scores else 0,
            "avg_progress_score": float(avg_progress),
            "episodes": task_results,
        })

    # Final summary
    overall_sr = total_succ / max(total_eps, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS: VLABench ({args.eval_track or 'custom'})")
    logger.info(f"  Total: {total_succ}/{total_eps} ({overall_sr*100:.1f}%)")
    for r in all_results:
        logger.info(
            f"  {r['task']}: {r['successes']}/{r['total']} "
            f"({r['success_rate']*100:.1f}%) "
            f"progress={r['avg_progress_score']:.3f}"
        )
    logger.info(f"{'='*60}")

    # Save results
    results_data = {
        "benchmark": "vlabench",
        "eval_track": args.eval_track,
        "port": args.port,
        "total_successes": total_succ,
        "total_episodes": total_eps,
        "success_rate": overall_sr,
        "tasks": all_results,
    }
    results_path = log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Generate report
    report = generate_report(args, all_results, total_succ, total_eps)
    report_path = log_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")


def generate_report(args, results, total_succ, total_eps):
    overall_sr = total_succ / max(total_eps, 1) * 100
    lines = [
        f"# VLABench Evaluation Report",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Track**: {args.eval_track or 'custom'}",
        f"**Port**: {args.port}",
        f"**Episodes/task**: {args.n_episode}",
        f"",
        f"## Overall Results",
        f"",
        f"**Success rate: {total_succ}/{total_eps} ({overall_sr:.1f}%)**",
        f"",
        f"## Per-Task Results",
        f"",
        f"| Task | Successes | Total | Rate | Progress |",
        f"|------|-----------|-------|------|----------|",
    ]
    for r in results:
        sr = r["success_rate"] * 100
        prog = r["avg_progress_score"]
        lines.append(
            f"| {r['task']} | {r['successes']} | {r['total']} | "
            f"{sr:.1f}% | {prog:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
