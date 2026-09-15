# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa ground-truth state exporter.

Runs inside the RoboCasa eval client process and extracts per-step
ground-truth state from the RoboCasa environment, packaging it into a
``gt_state`` dict compatible with the proxy's :class:`GTFailureDetector`.

RoboCasa is built on robosuite, so the MuJoCo sim state access is
similar to LIBERO.  The main differences:

- RoboCasa uses a PandaMobile robot (Franka + mobile base).
- Tasks are defined in Python classes (not BDDL files).
- Success checking uses ``env._check_success()`` (binary).
- Object tracking uses ``env.objects`` and ``env.fixtures``.

Usage::

    from vlm_orchestrator.aux_benchmarks.robocasa_gt import RoboCasaGTStateExporter

    exporter = RoboCasaGTStateExporter(env)
    # After each env.step():
    gt_state = exporter.export(obs)
    wire_obs["gt_state"] = gt_state
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RoboCasaGTStateExporter:
    """Extracts ground-truth state from a RoboCasa environment.

    Args:
        env: A RoboCasa gymnasium env instance.  We navigate to the
            underlying robosuite env via ``env.env`` or ``env.unwrapped``.
    """

    def __init__(self, env):
        # Navigate to the raw robosuite env
        self._gym_env = env
        inner = env
        # Unwrap gymnasium wrappers
        while hasattr(inner, "env") and inner.env is not inner:
            inner = inner.env
        # If we have an unwrapped attribute, use it
        if hasattr(inner, "unwrapped"):
            inner = inner.unwrapped
        self._env = inner

        # Discover scene objects
        self._scene_objects: list[str] = []
        self._obj_body_ids: dict[str, int] = {}

        # RoboCasa env objects
        if hasattr(self._env, "objects"):
            for obj in self._env.objects:
                name = obj.name if hasattr(obj, "name") else str(obj)
                self._scene_objects.append(name)
        if hasattr(self._env, "fixtures"):
            for fix in self._env.fixtures:
                name = fix.name if hasattr(fix, "name") else str(fix)
                self._scene_objects.append(name)

        # Try to get body IDs for position tracking
        if hasattr(self._env, "sim"):
            sim = self._env.sim
            for name in self._scene_objects:
                try:
                    body_name = name + "_main"  # robosuite convention
                    body_id = sim.model.body_name2id(body_name)
                    self._obj_body_ids[name] = body_id
                except Exception:
                    try:
                        body_id = sim.model.body_name2id(name)
                        self._obj_body_ids[name] = body_id
                    except Exception:
                        pass

        logger.info(f"[RoboCasaGT] Initialized: "
                     f"{len(self._scene_objects)} scene objects, "
                     f"{len(self._obj_body_ids)} tracked")

    def _detect_grasped_object(self) -> str | None:
        """Detect which object the robot is currently grasping."""
        env = self._env
        if not hasattr(env, "_check_grasp"):
            return None

        for obj_name in self._scene_objects:
            try:
                if env._check_grasp(
                    gripper=env.robots[0].gripper,
                    object_geoms=[g for g in env.sim.model.geom_names
                                  if obj_name in g],
                ):
                    return obj_name
            except Exception:
                continue
        return None

    def export(self, obs: dict | None = None) -> dict:
        """Export the current ground-truth state.

        Returns a dict structured for the GTFailureDetector.
        """
        env = self._env

        # Robot state
        grasped = self._detect_grasped_object()

        # EE position
        ee_pos = np.zeros(3)
        if hasattr(env, "sim"):
            try:
                site_id = env.sim.model.site_name2id("gripper0_grip_site")
                ee_pos = env.sim.data.site_xpos[site_id].copy()
            except Exception:
                try:
                    ee_pos = env.robots[0].controller.ee_pos.copy()
                except Exception:
                    pass

        # Gripper width
        gripper_width = 0.0
        if hasattr(env, "sim"):
            try:
                qpos_addr = env.sim.model.get_joint_qpos_addr(
                    "gripper0_finger_joint1"
                )
                gripper_width = float(env.sim.data.qpos[qpos_addr])
            except Exception:
                pass

        robot = {
            "grasped_object": grasped,
            "gripper_width": gripper_width,
            "ee_pos": ee_pos.tolist(),
        }

        # Object states
        objects = {}
        if hasattr(env, "sim"):
            sim = env.sim
            for obj_name, body_id in self._obj_body_ids.items():
                try:
                    pos = sim.data.body_xpos[body_id].copy()
                    quat = sim.data.body_xquat[body_id].copy()
                    objects[obj_name] = {
                        "pos": pos.tolist(),
                        "quat": quat.tolist(),
                    }
                except Exception:
                    pass

        # Success check
        success = False
        try:
            success = bool(env._check_success())
        except Exception:
            pass

        # Build subtask dict
        # RoboCasa doesn't have BDDL predicates, so we use binary success.
        subtask = {
            "score": 1.0 if success else 0.0,
            "conditions": [{
                "condition_idx": 0,
                "predicate": "task_success",
                "object": "",
                "target": None,
                "satisfied": success,
                "info": f"task_success={success}",
            }],
            "object_completed": {},
            "all_subtask_conditions": {},
        }

        return {
            "robot": robot,
            "objects": objects,
            "scene_objects": self._scene_objects,
            "subtask": subtask,
        }
