# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLABench ground-truth state exporter.

Runs inside the VLABench eval client process and extracts per-step
ground-truth state, packaging it into a ``gt_state`` dict compatible
with the proxy's :class:`GTFailureDetector`.

VLABench uses dm_control / MuJoCo 3.2 with structured Condition objects
(contain, on, above, grasped, button_pressed, etc.).

Usage::

    from vlm_orchestrator.aux_benchmarks.vlabench_gt import VLABenchGTStateExporter

    exporter = VLABenchGTStateExporter(env)
    gt_state = exporter.export(obs)
    wire_obs["gt_state"] = gt_state
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VLABenchGTStateExporter:
    """Extracts ground-truth state from a VLABench environment.

    Args:
        env: A VLABench env instance (VLABenchEnv or subclass).
    """

    def __init__(self, env):
        self._env = env

        # Discover scene objects from task
        self._scene_objects: list[str] = []
        self._conditions: list = []

        # VLABench task has a list of conditions
        task = getattr(env, "task", None)
        if task is not None:
            # Task conditions
            conds = getattr(task, "conditions", [])
            self._conditions = conds

            # Scene objects from components
            components = getattr(task, "components", {})
            if isinstance(components, dict):
                self._scene_objects = list(components.keys())
            elif isinstance(components, (list, tuple)):
                self._scene_objects = [
                    getattr(c, "name", str(c)) for c in components
                ]

        logger.info(f"[VLABenchGT] Initialized: "
                     f"{len(self._scene_objects)} objects, "
                     f"{len(self._conditions)} conditions")

    def _get_ee_state(self) -> tuple[np.ndarray, float]:
        """Get EE position and gripper state."""
        env = self._env
        ee_pos = np.zeros(3)
        gripper = 0.0

        if hasattr(env, "robot"):
            robot = env.robot
            physics = getattr(env, "physics", None)
            if physics is not None:
                try:
                    ee_state = robot.get_ee_state(physics)
                    ee_pos = ee_state[:3]
                    gripper = float(ee_state[-1])
                except Exception:
                    pass

        return ee_pos, gripper

    def _detect_grasped(self) -> str | None:
        """Detect grasped object (if any)."""
        env = self._env
        physics = getattr(env, "physics", None)
        if physics is None:
            return None

        # Check each condition for IsGraspedCondition
        for cond in self._conditions:
            cond_type = type(cond).__name__
            if "grasp" in cond_type.lower():
                try:
                    if cond.is_satisfied(physics):
                        obj = getattr(cond, "entity", None)
                        if obj is not None:
                            return getattr(obj, "name", str(obj))
                except Exception:
                    pass
        return None

    def _eval_conditions(self) -> list[dict]:
        """Evaluate all task conditions."""
        env = self._env
        physics = getattr(env, "physics", None)
        result = []

        for i, cond in enumerate(self._conditions):
            satisfied = False
            cond_type = type(cond).__name__

            if physics is not None:
                try:
                    satisfied = bool(cond.is_satisfied(physics))
                except Exception:
                    pass

            # Extract object/target names
            obj_name = ""
            target_name = None
            if hasattr(cond, "entity"):
                obj_name = getattr(cond.entity, "name", str(cond.entity))
            if hasattr(cond, "container"):
                target_name = getattr(cond.container, "name",
                                      str(cond.container))
            elif hasattr(cond, "target"):
                target_name = getattr(cond.target, "name",
                                      str(cond.target))

            result.append({
                "condition_idx": i,
                "predicate": cond_type,
                "object": obj_name,
                "target": target_name,
                "satisfied": satisfied,
                "info": f"{cond_type}({obj_name}"
                        + (f", {target_name})" if target_name else ")"),
            })

        return result

    def _get_object_states(self) -> dict:
        """Get positions of scene objects."""
        env = self._env
        physics = getattr(env, "physics", None)
        objects = {}

        if physics is None:
            return objects

        for name in self._scene_objects:
            try:
                # Try getting body position from physics
                body_id = physics.model.name2id(name, "body")
                pos = physics.data.xpos[body_id].copy()
                quat = physics.data.xquat[body_id].copy()
                objects[name] = {
                    "pos": pos.tolist(),
                    "quat": quat.tolist(),
                }
            except Exception:
                pass

        return objects

    def export(self, obs: dict | None = None) -> dict:
        """Export the current ground-truth state.

        Returns:
            Dict compatible with GTFailureDetector.
        """
        ee_pos, gripper = self._get_ee_state()
        grasped = self._detect_grasped()
        conditions = self._eval_conditions()
        objects = self._get_object_states()

        n_satisfied = sum(1 for c in conditions if c["satisfied"])
        score = n_satisfied / max(len(conditions), 1)

        robot = {
            "grasped_object": grasped,
            "gripper_width": gripper,
            "ee_pos": ee_pos.tolist(),
        }

        subtask = {
            "score": score,
            "conditions": conditions,
            "object_completed": {
                c["object"]: c["satisfied"]
                for c in conditions if c["object"]
            },
            "all_subtask_conditions": {
                str(i): c["satisfied"]
                for i, c in enumerate(conditions)
            },
        }

        return {
            "robot": robot,
            "objects": objects,
            "scene_objects": self._scene_objects,
            "subtask": subtask,
        }
