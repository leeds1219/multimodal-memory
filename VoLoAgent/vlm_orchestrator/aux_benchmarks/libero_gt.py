# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO ground-truth state exporter.

Runs **inside the LIBERO eval client process** (not the proxy) and
extracts per-step ground-truth state from the LIBERO environment,
packaging it into a ``gt_state`` dict compatible with the proxy's
:class:`GTFailureDetector`.

LIBERO's ground-truth information comes from three sources:

1. **BDDL goal predicates** — parsed from the ``.bddl`` file and
   evaluated each step via ``env._check_success()`` and
   ``env._eval_predicate()``.  Each predicate is a ``(fn, obj1, obj2)``
   or ``(fn, obj)`` tuple (e.g. ``("In", "alphabet_soup_1",
   "basket_1_contain_region")``).

2. **MuJoCo sim state** — object body positions from
   ``sim.data.body_xpos``, contact detection from
   ``env.check_contact()``, and grasp checking from
   ``env._check_grasp()``.

3. **``obj_of_interest``** — BDDL declares which objects are task-
   relevant (e.g. ``alphabet_soup_1``, ``tomato_sauce_1``, ``basket_1``).

The exported ``gt_state`` dict is structured to mirror robolab's format
so that the existing :class:`GTFailureDetector` can consume it with
minimal adaptation.

Usage in the LIBERO eval client::

    from vlm_orchestrator.aux_benchmarks.libero_gt import LiberoGTStateExporter

    exporter = LiberoGTStateExporter(env)
    # After each env.step():
    gt_state = exporter.export(obs)
    wire_obs["gt_state"] = gt_state
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LiberoGTStateExporter:
    """Extracts ground-truth state from a LIBERO environment.

    Args:
        env: A LIBERO ``ControlEnv`` or ``OffScreenRenderEnv`` instance.
            Must have ``env.env`` pointing to the underlying
            ``BDDLBaseDomain`` subclass.
    """

    def __init__(self, env):
        # Navigate to the actual robosuite env (OffScreenRenderEnv → ControlEnv → BDDLBaseDomain)
        self._raw_env = env
        inner = env
        while hasattr(inner, "env") and inner.env is not inner:
            inner = inner.env
        self._env = inner

        # Extract BDDL problem info
        self._parsed_problem = getattr(self._env, "parsed_problem", {})
        self._goal_state = self._parsed_problem.get("goal_state", [])
        self._obj_of_interest = self._parsed_problem.get("obj_of_interest", [])

        # Scene objects: all manipulable objects + fixtures
        self._scene_objects: list[str] = []
        objects_dict = getattr(self._env, "objects_dict", {})
        fixtures_dict = getattr(self._env, "fixtures_dict", {})
        self._scene_objects = (
            list(objects_dict.keys()) + list(fixtures_dict.keys())
        )

        # Object body IDs (for position extraction)
        self._obj_body_id = getattr(self._env, "obj_body_id", {})

        # Cache goal predicates as structured dicts for the detector
        self._goal_conditions = self._parse_goal_conditions()

        logger.info(
            f"[LiberoGT] Initialized: "
            f"scene_objects={self._scene_objects}, "
            f"obj_of_interest={self._obj_of_interest}, "
            f"goal_conditions={len(self._goal_conditions)}"
        )

    def _parse_goal_conditions(self) -> list[dict]:
        """Parse BDDL goal predicates into structured condition dicts.

        Each BDDL goal predicate looks like::

            ["In", "alphabet_soup_1", "basket_1_contain_region"]
            ["On", "white_mug_1", "plate_1_contain_region"]
            ["Close", "microwave_1"]

        We convert these into condition dicts compatible with the
        GT failure detector's ``conditions`` format.
        """
        conditions = []
        for i, state in enumerate(self._goal_state):
            cond = {
                "condition_idx": i,
                "predicate": state[0] if len(state) > 0 else "unknown",
                "satisfied": False,
            }
            if len(state) == 3:
                cond["object"] = state[1]
                cond["target"] = state[2]
                cond["info"] = f"{state[0]}({state[1]}, {state[2]})"
            elif len(state) == 2:
                cond["object"] = state[1]
                cond["target"] = None
                cond["info"] = f"{state[0]}({state[1]})"
            else:
                cond["object"] = ""
                cond["info"] = str(state)
            conditions.append(cond)
        return conditions

    def export(self, obs: dict | None = None) -> dict:
        """Export the current ground-truth state.

        Returns a dict structured like robolab's ``gt_state``::

            {
                "robot": {
                    "grasped_object": "alphabet_soup_1" | None,
                    "gripper_width": 0.04,
                    "ee_pos": [x, y, z],
                },
                "objects": {
                    "alphabet_soup_1": {"pos": [x, y, z], "quat": [w, x, y, z]},
                    ...
                },
                "scene_objects": ["alphabet_soup_1", "basket_1", ...],
                "subtask": {
                    "score": 0.5,
                    "conditions": [...],           # per-goal-predicate status
                    "object_completed": {...},      # per-object completion
                    "all_subtask_conditions": {...}, # for regression detection
                },
            }
        """
        env = self._env
        sim = env.sim

        # ── Robot state ──
        grasped = self._detect_grasped_object()
        gripper_qpos = sim.data.qpos[
            sim.model.get_joint_qpos_addr("gripper0_finger_joint1")
        ] if "gripper0_finger_joint1" in [
            sim.model.joint_id2name(i) for i in range(sim.model.njnt)
        ] else 0.0

        ee_pos = sim.data.site_xpos[
            sim.model.site_name2id("gripper0_grip_site")
        ].copy() if "gripper0_grip_site" in [
            sim.model.site_id2name(i) for i in range(sim.model.nsite)
        ] else np.zeros(3)

        robot = {
            "grasped_object": grasped,
            "gripper_width": float(gripper_qpos),
            "ee_pos": ee_pos.tolist(),
        }

        # ── Object states ──
        objects = {}
        for obj_name in self._scene_objects:
            if obj_name in self._obj_body_id:
                body_id = self._obj_body_id[obj_name]
                pos = sim.data.body_xpos[body_id].copy()
                quat = sim.data.body_xquat[body_id].copy()
                objects[obj_name] = {
                    "pos": pos.tolist(),
                    "quat": quat.tolist(),
                }

        # ── Evaluate goal predicates ──
        conditions = []
        score = 0.0
        n_goals = max(len(self._goal_state), 1)
        object_completed: dict[str, bool] = {}
        all_subtask_conditions: dict[str, bool] = {}

        for i, state in enumerate(self._goal_state):
            try:
                satisfied = env._eval_predicate(state)
            except Exception:
                satisfied = False

            cond_key = f"goal_{i}"
            all_subtask_conditions[cond_key] = bool(satisfied)

            if satisfied:
                score += 1.0 / n_goals

            # Build condition entry
            cond = dict(self._goal_conditions[i]) if i < len(self._goal_conditions) else {}
            cond["satisfied"] = bool(satisfied)
            conditions.append(cond)

            # Track per-object completion
            obj = cond.get("object", "")
            if obj:
                if obj not in object_completed:
                    object_completed[obj] = bool(satisfied)
                else:
                    # AND: object is complete only if ALL its conditions are met
                    object_completed[obj] = object_completed[obj] and bool(satisfied)

        # ── Synthesize CSM-style per-object conditions ──
        # The robolab GT detector expects per-object condition tables:
        #   condition_idx 0 = grabbed
        #   condition_idx 3 = in_container (goal satisfied)
        # We synthesize these from LIBERO's grasped_object + BDDL predicates.
        csm_conditions = []
        for obj_name in self._obj_of_interest:
            # Skip containers / regions — only track manipulable objects
            if "_region" in obj_name or obj_name in (
                getattr(env, "fixtures_dict", {}).keys()
            ):
                continue

            is_grabbed = (grasped == obj_name)
            is_goal_done = object_completed.get(obj_name, False)

            csm_conditions.append({
                "object": obj_name,
                "condition_idx": 0,  # grabbed
                "satisfied": is_grabbed,
                "info": f"grabbed({obj_name})",
            })
            csm_conditions.append({
                "object": obj_name,
                "condition_idx": 3,  # in_container / goal met
                "satisfied": is_goal_done,
                "info": f"goal_satisfied({obj_name})",
            })

        # Merge BDDL conditions + CSM conditions
        merged_conditions = conditions + csm_conditions

        subtask = {
            "score": score,
            "conditions": merged_conditions,
            "object_completed": object_completed,
            "all_subtask_conditions": all_subtask_conditions,
        }

        return {
            "robot": robot,
            "objects": objects,
            "scene_objects": list(self._scene_objects),
            "subtask": subtask,
        }

    def _detect_grasped_object(self) -> str | None:
        """Detect which object (if any) the gripper is grasping.

        Uses robosuite's ``_check_grasp()`` for each object of interest,
        which tests contact between the gripper finger pads and object
        geoms.
        """
        env = self._env
        if not hasattr(env, "robots") or not env.robots:
            return None

        gripper = env.robots[0].gripper

        # Check obj_of_interest first (most likely targets)
        for obj_name in self._obj_of_interest:
            obj = self._get_object(obj_name)
            if obj is not None:
                try:
                    if env._check_grasp(gripper=gripper, object_geoms=obj):
                        return obj_name
                except Exception:
                    pass

        # Check all manipulable objects
        objects_dict = getattr(env, "objects_dict", {})
        for obj_name, obj in objects_dict.items():
            if obj_name in self._obj_of_interest:
                continue  # already checked
            try:
                if env._check_grasp(gripper=gripper, object_geoms=obj):
                    return obj_name
            except Exception:
                pass

        return None

    def _get_object(self, name: str):
        """Get a robosuite object by name (from objects or fixtures)."""
        env = self._env
        obj = getattr(env, "objects_dict", {}).get(name)
        if obj is None:
            obj = getattr(env, "fixtures_dict", {}).get(name)
        return obj
