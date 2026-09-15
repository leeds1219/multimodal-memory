# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replayable live-VLM fixtures for debugging System-2 GT metrics.

These tests intentionally sit one layer below long RoboLab runs.  Each case
uses a real/cached VLM JSON response as the control-log input and verifies
that the passive GT metric detectors classify the decision against structured
GT.  Refresh mode records new real responses; normal test runs replay the
committed cassette file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vlm_orchestrator.gt_metrics import GTMetricsManager
from vlm_orchestrator.strategies.base import SessionState
from vlm_orchestrator.vlm import chat_create, parse_json


CASSETTE_PATH = Path(__file__).parent / "fixtures" / "system2_vlm_cassettes.json"
DEFAULT_MODEL = "YOUR_VLM_MODEL"
DEFAULT_BASE_URL = "https://YOUR_VLM_ENDPOINT/v1"


@dataclass(frozen=True)
class VLMMetricCase:
    case_id: str
    family: str
    metric_types: set[str]
    gt_state: dict[str, Any]
    subgoals: list[str]
    expected_metric: str
    expected_observed: str
    visual_setup: str
    decision_to_express: str
    prompt_kind: str = "scene_success_check"
    scene_kind: str = "blocks"
    subgoal_idx: int = 0
    raw_override: str | None = None


def _obj(pos=(0.0, 0.0, 0.026), vel=None):
    return {
        "pos": np.array(pos, dtype=np.float32),
        "quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "vel": np.array(vel or [0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def _state(case: VLMMetricCase) -> SessionState:
    state = SessionState()
    state.episode_id = abs(hash(case.case_id)) % 100000
    state.infer_count = 10
    state.episode_step = 80
    state.subgoals = list(case.subgoals)
    state.current_subgoal_idx = case.subgoal_idx
    return state


def _stack_subtask_checks(
    relations: list[tuple[str, str]],
    *,
    satisfied: list[bool] | None = None,
) -> dict[str, Any]:
    satisfied = satisfied or [False] * len(relations)
    checks: dict[str, Any] = {}
    for idx, ((bottom, top), done) in enumerate(zip(relations, satisfied, strict=True)):
        checks[f"subtask_{idx}"] = {
            "subtask_idx": idx,
            "name": f"stack_{top}_on_{bottom}",
            "logical": "all",
            "satisfied": done,
            "conditions": [
                {
                    "object": top,
                    "predicate": "stacked",
                    "target_objects": [bottom, top],
                    "info": (
                        f"stacked(objects=['{bottom}', '{top}'], "
                        "order=bottom_to_top)"
                    ),
                    "satisfied": done,
                }
            ],
        }
    return checks


def _gt(
    *,
    scene_objects: list[str],
    condition_object: str,
    condition_target_objects: list[str] | None = None,
    placed: bool = False,
    score: float = 0.0,
    all_subtask_conditions: dict[str, bool] | None = None,
    all_subtask_checks: dict[str, Any] | None = None,
    object_completed: dict[str, bool] | None = None,
    task: dict[str, Any] | None = None,
    out_of_scene: str | None = None,
) -> dict[str, Any]:
    objects = {
        obj: _obj((-0.1 + idx * 0.08, 0.0, -0.08 if obj == out_of_scene else 0.026))
        for idx, obj in enumerate(scene_objects)
    }
    all_subtask_conditions = (
        all_subtask_conditions
        if all_subtask_conditions is not None
        else {"subtask_0": placed}
    )
    if object_completed is None:
        object_completed = {condition_object: placed}
    gt = {
        "robot": {
            "grasped_object": None,
            "objects_in_contact": [],
            "gripper_width": 0.04,
            "ee_pos": [0.3, 0.0, 0.4],
        },
        "objects": objects,
        "scene_objects": scene_objects,
        "subtask": {
            "score": score,
            "conditions": [
                {
                    "object": condition_object,
                    "condition_idx": 0,
                    "predicate": "Grabbed",
                    "satisfied": False,
                    "info": f"grabbed({condition_object})",
                },
                {
                    "object": condition_object,
                    "condition_idx": 3,
                    "predicate": "Terminal",
                    "target": (
                        condition_target_objects[0]
                        if condition_target_objects else "target"
                    ),
                    "satisfied": placed,
                    "info": "terminal_condition",
                    "target_objects": (
                        condition_target_objects or [condition_object]
                    ),
                },
            ],
            "current_index": 0,
            "object_completed": object_completed,
            "all_subtask_conditions": all_subtask_conditions,
            "all_subtask_checks": all_subtask_checks or {
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": placed,
                    "conditions": [
                        {
                            "object": condition_object,
                            "predicate": "terminal",
                            "info": "terminal_condition",
                            "target_objects": (
                                condition_target_objects or [condition_object]
                            ),
                            "satisfied": placed,
                        }
                    ],
                }
            },
        },
    }
    if task is not None:
        gt["task"] = task
    return gt


def _task_checks(
    *,
    success_satisfied: bool,
    invariant_satisfied: bool = True,
) -> dict[str, Any]:
    return {
        "success_satisfied": success_satisfied and invariant_satisfied,
        "success_checks": [
            {
                "name": "target_object_in_destination",
                "type": "success",
                "predicate": "object_in_container",
                "objects": ["tuna_can"],
                "reference": "container_b03",
                "satisfied": success_satisfied,
            }
        ],
        "invariant_checks": [
            {
                "name": "fruit_remains_in_bowl",
                "type": "invariant",
                "predicate": "object_in_container",
                "objects": ["lemon_02"],
                "reference": "serving_bowl",
                "satisfied": invariant_satisfied,
            }
        ],
    }


def _sort_task() -> dict[str, Any]:
    return {
        "success_checks": [
            {
                "name": "food_items_in_bowl",
                "type": "success",
                "predicate": "object_in_container",
                "objects": ["lime01"],
                "reference": "serving_bowl",
                "satisfied": False,
            },
            {
                "name": "nonfood_items_in_bin",
                "type": "success",
                "predicate": "object_in_container",
                "objects": ["blue_block"],
                "reference": "bin_a02",
                "satisfied": False,
            },
        ],
        "invariant_checks": [],
    }


def _outside_task() -> dict[str, Any]:
    return {
        "success_checks": [
            {
                "name": "tuna_can_removed_from_bowl",
                "type": "success",
                "predicate": "object_outside_of",
                "objects": ["tuna_can"],
                "reference": "serving_bowl",
                "satisfied": False,
            }
        ],
        "invariant_checks": [],
    }


def _scene_cases() -> list[VLMMetricCase]:
    base = {
        "scene_objects": ["red_block", "blue_block", "green_block"],
        "condition_object": "blue_block",
        "condition_target_objects": ["red_block", "blue_block"],
    }
    return [
        VLMMetricCase(
            "scene_confirm_complete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=True, score=1.0),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa",
            "confirmed_complete",
            "blue block is stably on red block",
            "Say the current step is complete because the blue block is on the red block.",
        ),
        VLMMetricCase(
            "scene_aligned_incomplete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=False),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa",
            "aligned_incomplete",
            "blue block remains on the table beside red block",
            "Say the step is still in progress and the robot should continue.",
        ),
        VLMMetricCase(
            "scene_false_complete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=False),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa_failure",
            "false_complete",
            "blue block is beside red block, not on top",
            "Incorrectly claim the blue block is already stacked and mark complete.",
        ),
        VLMMetricCase(
            "scene_missed_complete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=True, score=1.0),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa_failure",
            "missed_complete",
            "blue block is already on red block",
            "Incorrectly say the robot still needs to place the blue block.",
        ),
        VLMMetricCase(
            "scene_unchecked_parse_failure",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=False),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa",
            "unchecked_parse_failure",
            "camera view is ambiguous",
            "Return a malformed non-JSON progress answer.",
            raw_override="The block looks maybe done, but I cannot be sure.",
        ),
        VLMMetricCase(
            "scene_cube_wording_complete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=True, score=1.0),
            ["Pick the blue cube and stack it on the red cube"],
            "vlm_scene_qa",
            "confirmed_complete",
            "blue cube is stably on red cube",
            "Say the cube stack step is complete.",
        ),
        VLMMetricCase(
            "scene_transient_contact_false_complete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=False),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa_failure",
            "false_complete",
            "blue block is merely touching the red block and is not stable",
            "Incorrectly call the touching blocks stacked and complete.",
        ),
        VLMMetricCase(
            "scene_failure_but_incomplete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=False),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa",
            "aligned_incomplete",
            "robot picked the wrong area; blue is not stacked",
            "Say this is a failure requiring replan, not a complete step.",
        ),
        VLMMetricCase(
            "scene_next_action_false_complete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=False),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa_failure",
            "false_complete",
            "blue block is on table, red block is separate",
            "Use action next even though the step is not done.",
        ),
        VLMMetricCase(
            "scene_done_bool_complete",
            "scene_qa",
            {"scene_qa"},
            _gt(**base, placed=True, score=1.0),
            ["Pick the blue block and stack it on the red block"],
            "vlm_scene_qa",
            "confirmed_complete",
            "blue block is already in final stacked state",
            "Use a done boolean to mark the step complete.",
        ),
    ]


def _target_cases() -> list[VLMMetricCase]:
    return [
        VLMMetricCase(
            "target_blue_block",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["red_block", "blue_block"],
                condition_object="blue_block",
                condition_target_objects=["red_block", "blue_block"],
            ),
            ["Pick the blue block and stack it on the red block"],
            "vlm_grasp_target_qa",
            "target_aligned",
            "blue and red blocks are visible",
            "Choose the blue block as the grasp target.",
        ),
        VLMMetricCase(
            "target_green_cube_alias",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["red_block", "green_block"],
                condition_object="green_block",
                condition_target_objects=["red_block", "green_block"],
            ),
            ["Pick the green cube and stack it on the red cube"],
            "vlm_grasp_target_qa",
            "target_aligned",
            "green cube and red cube are visible",
            "Choose the green cube as the grasp target.",
        ),
        VLMMetricCase(
            "target_support_object_mismatch",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["red_block", "blue_block"],
                condition_object="blue_block",
                condition_target_objects=["red_block", "blue_block"],
            ),
            ["Pick the blue block and stack it on the red block"],
            "vlm_grasp_target_mismatch",
            "target_mismatch",
            "blue is the moving object and red is the support",
            "Incorrectly choose the red support block as the grasp target.",
        ),
        VLMMetricCase(
            "target_missing",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["red_block", "blue_block"],
                condition_object="blue_block",
                condition_target_objects=["red_block", "blue_block"],
            ),
            ["Pick the blue block and stack it on the red block"],
            "vlm_grasp_target_mismatch",
            "missing_grasp_target",
            "blue and red blocks are visible",
            "Call the grasp tool but omit the grasp_target field.",
        ),
        VLMMetricCase(
            "target_target_object_field",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["red_block", "blue_block"],
                condition_object="blue_block",
                condition_target_objects=["red_block", "blue_block"],
            ),
            ["Pick the blue cube and stack it on the red cube"],
            "vlm_grasp_target_qa",
            "target_aligned",
            "blue cube is the moving object",
            "Use target_object rather than grasp_target for blue cube.",
        ),
        VLMMetricCase(
            "target_lime_alias",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["container_b03", "lime01", "table"],
                condition_object="lime01",
                object_completed={"lime01": False},
            ),
            ["Pick the lime from the grey bin and place it on the table"],
            "vlm_grasp_target_qa",
            "target_aligned",
            "a green lime is in a grey bin",
            "Choose the green lime as the grasp target.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "target_blue_tin_can_alias",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["serving_bowl", "tuna_can", "lemon_02"],
                condition_object="tuna_can",
                object_completed={"tuna_can": False},
            ),
            ["Pick the tuna can from the bowl and place it on the table"],
            "vlm_grasp_target_qa",
            "target_aligned",
            "a blue tuna can is misplaced in the bowl",
            "Choose the blue tin can as the grasp target.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "target_ambiguous_can",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["serving_bowl", "tuna_can", "soup_can"],
                condition_object="tuna_can",
                object_completed={"tuna_can": False},
            ),
            ["Pick the tuna can from the bowl and place it on the table"],
            "vlm_grasp_target_mismatch",
            "target_mismatch",
            "two cans are visible",
            "Choose only a generic tin can target, making the target ambiguous.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "target_uses_entry_subgoal_over_stale_gt",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["red_block", "blue_block", "green_block"],
                condition_object="green_block",
                condition_target_objects=["blue_block", "green_block"],
                object_completed={"blue_block": False, "green_block": False},
            ),
            ["Pick the blue cube and stack it on the red cube"],
            "vlm_grasp_target_qa",
            "target_aligned",
            "GT current condition has advanced, but the log entry is for blue",
            "Choose the blue cube as the grasp target for the entry subgoal.",
        ),
        VLMMetricCase(
            "target_rubiks_toy_cube_alias",
            "target_qa",
            {"target_qa"},
            _gt(
                scene_objects=["serving_bowl", "rubiks_cube", "table"],
                condition_object="rubiks_cube",
                object_completed={"rubiks_cube": False},
            ),
            ["Pick the Rubik's cube from the bowl and place it on the table"],
            "vlm_grasp_target_qa",
            "target_aligned",
            "a multicolored Rubik's cube is visible",
            "Choose the toy cube as the grasp target.",
            scene_kind="recover",
        ),
    ]


def _plan_cases() -> list[VLMMetricCase]:
    stack_gt = _gt(
        scene_objects=["red_block", "blue_block", "green_block"],
        condition_object="blue_block",
        all_subtask_conditions={"subtask_0": False, "subtask_1": False},
        all_subtask_checks=_stack_subtask_checks([
            ("red_block", "blue_block"),
            ("blue_block", "green_block"),
        ]),
    )
    sort_gt = _gt(
        scene_objects=["serving_bowl", "bin_a02", "lime01", "blue_block"],
        condition_object="lime01",
        object_completed={"lime01": False, "blue_block": False},
        task=_sort_task(),
    )
    recover_gt = _gt(
        scene_objects=["serving_bowl", "tuna_can", "lemon_02", "table"],
        condition_object="tuna_can",
        object_completed={"tuna_can": False},
        task=_outside_task(),
    )
    return [
        VLMMetricCase(
            "plan_stack_aligned",
            "plan_qa",
            {"plan_qa"},
            stack_gt,
            ["Stack blue on red", "Stack green on blue"],
            "vlm_plan_qa",
            "plan_aligned",
            "three colored blocks on the table",
            "Return subgoals stacking blue on red, then green on blue.",
            prompt_kind="plan",
        ),
        VLMMetricCase(
            "plan_stack_cube_words",
            "plan_qa",
            {"plan_qa"},
            stack_gt,
            ["Pick the blue cube and stack it on the red cube", "Pick the green cube and stack it on the blue cube"],
            "vlm_plan_qa",
            "plan_aligned",
            "three cubes are visible",
            "Return cube-worded subgoals for blue-on-red and green-on-blue.",
            prompt_kind="plan",
        ),
        VLMMetricCase(
            "plan_wrong_stack_order",
            "plan_qa",
            {"plan_qa"},
            stack_gt,
            ["Pick the green block and stack it on the blue block", "Pick the blue block and stack it on the red block"],
            "vlm_plan_qa",
            "plan_aligned",
            "three colored blocks on the table",
            "Return the stack subgoals in the wrong order. The recorded VLM response corrected this to the GT order.",
            prompt_kind="plan",
        ),
        VLMMetricCase(
            "plan_color_only_stack",
            "plan_qa",
            {"plan_qa"},
            stack_gt,
            ["Stack blue on red", "Stack green on blue"],
            "vlm_plan_qa",
            "plan_aligned",
            "blocks are referred to by color only",
            "Return concise color-only stack subgoals.",
            prompt_kind="plan",
        ),
        VLMMetricCase(
            "plan_sort_aligned",
            "plan_qa",
            {"plan_qa"},
            sort_gt,
            ["Pick up the blue cube and place it in the grey bin", "Pick up the green lime and place it in the white bowl"],
            "vlm_plan_qa",
            "plan_aligned",
            "sorting scene with food bowl and nonfood bin",
            "Return one subgoal for blue cube to bin and one for green lime to bowl.",
            prompt_kind="plan",
            scene_kind="sort",
        ),
        VLMMetricCase(
            "plan_sort_wrong_container",
            "plan_qa",
            {"plan_qa"},
            sort_gt,
            ["Pick up the blue cube and place it in the white bowl", "Pick up the green lime and place it in the grey bin"],
            "vlm_plan_qa",
            "plan_aligned",
            "sorting scene with two containers",
            "Swap the destinations for the cube and lime. The recorded VLM response corrected this to the GT destinations.",
            prompt_kind="plan",
            scene_kind="sort",
        ),
        VLMMetricCase(
            "plan_source_destination_container",
            "plan_qa",
            {"plan_qa"},
            _gt(
                scene_objects=["container_b03", "tuna_can", "corn_can"],
                condition_object="tuna_can",
                object_completed={"tuna_can": False},
                task={
                    "success_checks": [
                        {
                            "name": "tuna_can_moved_to_can_bin",
                            "type": "success",
                            "predicate": "object_in_container",
                            "objects": ["tuna_can"],
                            "reference": "container_b03",
                            "satisfied": False,
                        }
                    ],
                    "invariant_checks": [],
                },
            ),
            ["Pick the tuna can from the white plate and place it into the grey tray"],
            "vlm_plan_qa",
            "plan_aligned",
            "a tuna can needs to be moved into the tray",
            "Mention both source and destination containers.",
            prompt_kind="plan",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "plan_blue_can_alias",
            "plan_qa",
            {"plan_qa"},
            _gt(
                scene_objects=["container_b03", "tuna_can", "corn_can"],
                condition_object="tuna_can",
                object_completed={"tuna_can": False},
                task={
                    "success_checks": [
                        {
                            "name": "tuna_can_moved_to_can_bin",
                            "type": "success",
                            "predicate": "object_in_container",
                            "objects": ["tuna_can"],
                            "reference": "container_b03",
                            "satisfied": False,
                        }
                    ],
                    "invariant_checks": [],
                },
            ),
            ["Pick the blue can from the white plate and place it into the grey tray"],
            "vlm_plan_qa",
            "plan_aligned",
            "the tuna can appears as a blue can",
            "Use blue can wording for the tuna can destination plan.",
            prompt_kind="plan",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "plan_recover_remove",
            "plan_qa",
            {"plan_qa"},
            recover_gt,
            ["Pick the blue can from the white bowl and place it on the table"],
            "vlm_plan_qa",
            "plan_aligned",
            "a blue can does not belong in the bowl",
            "Remove the blue can from the white bowl.",
            prompt_kind="plan",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "plan_unparseable_unchecked",
            "plan_qa",
            {"plan_qa"},
            stack_gt,
            ["Move the first block to the support block"],
            "vlm_plan_qa",
            "unchecked",
            "the plan uses vague object references",
            "Return an intentionally vague unparseable subgoal.",
            prompt_kind="plan",
        ),
    ]


def _task_cases() -> list[VLMMetricCase]:
    scene_objects = ["serving_bowl", "container_b03", "tuna_can", "lemon_02"]
    return [
        VLMMetricCase(
            "task_aligned_incomplete",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=False)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_task_success_qa",
            "aligned_task_incomplete",
            "target can has not reached destination",
            "Say the task is still in progress.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_confirmed_success",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=True)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_task_success_qa",
            "confirmed_task_success",
            "all success and invariant checks are satisfied",
            "Say the entire task is complete.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_false_success",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=False)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_task_success_qa_failure",
            "false_task_success",
            "the target can remains misplaced",
            "Incorrectly claim the whole task is complete.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_missed_success",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=True)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_task_success_qa_failure",
            "missed_task_success",
            "the task is already complete",
            "Incorrectly say the task remains incomplete.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_invariant_satisfied",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=True, invariant_satisfied=True)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_invariant_qa",
            "invariants_satisfied_when_claimed_complete",
            "target complete and collateral fruit remains in bowl",
            "Claim task complete while all invariants hold.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_invariant_missed",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=True, invariant_satisfied=False)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_invariant_qa_failure",
            "missed_invariant_violation",
            "target is done but a collateral fruit invariant is violated",
            "Claim task complete despite the invariant violation.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_intermediate_complete_not_task",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=False)),
            ["Pick the tuna can", "Place the tuna can in the tray"],
            "vlm_task_success_qa",
            "aligned_task_incomplete",
            "first subgoal is complete but task is not",
            "Mark only the current subgoal complete, not the whole task.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_overall_phrase_false_success",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=False)),
            ["Pick the tuna can", "Place the tuna can in the tray"],
            "vlm_task_success_qa_failure",
            "false_task_success",
            "the task remains incomplete",
            "Use explicit wording that the overall task is complete.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_done_bool_success",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=True)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_task_success_qa",
            "confirmed_task_success",
            "all checks are satisfied",
            "Use a done boolean to mark the task complete.",
            scene_kind="recover",
        ),
        VLMMetricCase(
            "task_success_check_false_due_invariant",
            "task_qa",
            {"task_qa"},
            _gt(scene_objects=scene_objects, condition_object="tuna_can", task=_task_checks(success_satisfied=True, invariant_satisfied=False)),
            ["Pick the tuna can and place it in the grey tray"],
            "vlm_task_success_qa_failure",
            "false_task_success",
            "success check is true but invariant check fails",
            "Claim the task is complete while ignoring the invariant.",
            scene_kind="recover",
        ),
    ]


def _failure_cases() -> list[VLMMetricCase]:
    def case(
        case_id: str,
        target: str,
        subgoal: str,
        expected_metric: str,
        expected_observed: str,
        decision: str,
        scene_objects: list[str] | None = None,
    ) -> VLMMetricCase:
        scene_objects = scene_objects or ["red_block", "blue_block", "green_block"]
        return VLMMetricCase(
            case_id,
            "failure_qa",
            {"failure_qa"},
            _gt(
                scene_objects=scene_objects,
                condition_object=target,
                object_completed={target: False},
                out_of_scene=target,
            ),
            [subgoal],
            expected_metric,
            expected_observed,
            f"{target} has fallen below the table plane",
            decision,
            scene_kind="recover" if "can" in target or "lime" in target else "blocks",
        )

    return [
        case("failure_blue_detected", "blue_block", "Pick the blue block and stack it on the red block", "vlm_failure_qa", "failure_detected", "Detect that the blue block is out of scene and request replan."),
        case("failure_blue_missed", "blue_block", "Pick the blue block and stack it on the red block", "vlm_failure_qa", "failure_detected", "Miss the out-of-scene blue block and say to continue. The recorded VLM response still explicitly detected the failure."),
        case("failure_stack_bottom_detected", "red_block", "Pick the blue block and stack it on the red block", "vlm_failure_qa", "failure_detected", "Detect the red support block is gone and request replan."),
        case("failure_stack_top_missed", "green_block", "Pick the green block and stack it on the blue block", "vlm_failure_missed", "missed_blocking_failure", "Miss that the green block is out of scene."),
        case("failure_tuna_detected", "tuna_can", "Pick the tuna can from the bowl and place it on the table", "vlm_failure_qa", "failure_detected", "Detect that the tuna can is out of scene.", ["serving_bowl", "tuna_can", "table"]),
        case("failure_tuna_missed", "tuna_can", "Pick the tuna can from the bowl and place it on the table", "vlm_failure_missed", "missed_blocking_failure", "Say the robot should continue even though the can is gone.", ["serving_bowl", "tuna_can", "table"]),
        case("failure_lime_detected", "lime01", "Pick the lime from the bin and place it in the bowl", "vlm_failure_qa", "failure_detected", "Detect that the lime is no longer reachable.", ["serving_bowl", "bin_a02", "lime01"]),
        case("failure_lime_missed", "lime01", "Pick the lime from the bin and place it in the bowl", "vlm_failure_qa", "failure_detected", "Miss the lost lime and continue. The recorded VLM response still explicitly detected the failure.", ["serving_bowl", "bin_a02", "lime01"]),
        case("failure_rubiks_detected", "rubiks_cube", "Pick the Rubik's cube and place it on the shelf", "vlm_failure_qa", "failure_detected", "Detect that the Rubik's cube fell away.", ["rubiks_cube", "shelf"]),
        case("failure_rubiks_missed", "rubiks_cube", "Pick the Rubik's cube and place it on the shelf", "vlm_failure_missed", "missed_blocking_failure", "Miss that the Rubik's cube is out of scene.", ["rubiks_cube", "shelf"]),
    ]


CASES = [
    *_scene_cases(),
    *_target_cases(),
    *_plan_cases(),
    *_task_cases(),
    *_failure_cases(),
]


def _load_cassettes() -> dict[str, Any]:
    if not CASSETTE_PATH.exists():
        return {"schema": 1, "responses": {}}
    return json.loads(CASSETTE_PATH.read_text())


def _write_cassettes(cassettes: dict[str, Any]) -> None:
    CASSETTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASSETTE_PATH.write_text(json.dumps(cassettes, indent=2, sort_keys=True) + "\n")


def _image_for_case(case: VLMMetricCase) -> Image.Image:
    image = Image.new("RGB", (320, 220), (238, 238, 228))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 150, 320, 220], fill=(205, 195, 178))
    if case.scene_kind == "sort":
        draw.ellipse([30, 35, 125, 130], outline=(240, 240, 240), width=6)
        draw.rectangle([195, 42, 295, 135], outline=(120, 120, 120), width=6)
        draw.ellipse([72, 86, 96, 110], fill=(30, 170, 50))
        draw.rectangle([230, 80, 260, 110], fill=(40, 70, 220))
    elif case.scene_kind == "recover":
        draw.ellipse([40, 45, 145, 145], outline=(245, 245, 245), width=6)
        draw.rectangle([210, 55, 290, 130], outline=(130, 130, 130), width=6)
        draw.rectangle([78, 85, 110, 112], fill=(60, 110, 210))
        draw.ellipse([225, 86, 252, 113], fill=(50, 180, 55))
    else:
        draw.rectangle([78, 85, 118, 125], fill=(220, 40, 35))
        draw.rectangle([132, 85, 172, 125], fill=(45, 90, 230))
        draw.rectangle([186, 85, 226, 125], fill=(40, 165, 65))
    return image


def _image_b64_and_hash(image: Image.Image) -> tuple[str, str]:
    buf = BytesIO()
    image.save(buf, format="PNG")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha1(raw).hexdigest()


def _dotenv_vlm_key() -> str:
    for key in ("VLM_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(key):
            return str(os.environ[key])
    return ""


def _prompt_for_case(case: VLMMetricCase) -> str:
    return (
        "Return ONLY one JSON object for this robot VLM fixture.\n"
        f"Prompt kind: {case.prompt_kind}\n"
        f"Current subgoal(s): {case.subgoals}\n"
        f"Visual setup: {case.visual_setup}\n"
        f"Decision to express: {case.decision_to_express}\n"
        "For scene/task/failure checks use fields: status, action, reason, "
        "optional done, optional grasp_target.\n"
        "For plan checks use fields: ordered and subgoals.\n"
        "Use varied natural wording in reason, but preserve the requested "
        "semantic decision exactly."
    )


def _call_live_vlm(case: VLMMetricCase) -> dict[str, Any]:
    api_key = _dotenv_vlm_key()
    if not api_key:
        raise RuntimeError(
            "SYSTEM2_VLM_REFRESH=1 requires VLM_API_KEY or OPENAI_API_KEY."
        )
    from openai import OpenAI

    model = os.environ.get("VLM_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("VLM_BASE_URL", DEFAULT_BASE_URL)
    image_b64, image_hash = _image_b64_and_hash(_image_for_case(case))
    prompt = _prompt_for_case(case)
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = chat_create(
        client,
        model=model,
        temperature=0.0,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate compact JSON decisions for robot VLM "
                    "metric tests. Never include markdown."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            },
        ],
    )
    raw = response.choices[0].message.content or ""
    return {
        "raw": raw,
        "model": model,
        "base_url": base_url,
        "prompt_hash": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
        "image_hash": image_hash,
    }


def _raw_response_for(case: VLMMetricCase) -> str:
    if case.raw_override is not None:
        return case.raw_override
    cassettes = _load_cassettes()
    refresh = os.environ.get("SYSTEM2_VLM_REFRESH") == "1"
    if refresh:
        response = _call_live_vlm(case)
        cassettes.setdefault("responses", {})[case.case_id] = response
        _write_cassettes(cassettes)
        return str(response["raw"])
    response = cassettes.get("responses", {}).get(case.case_id)
    if not response:
        raise AssertionError(
            f"Missing System-2 VLM cassette for {case.case_id}. "
            "Run with SYSTEM2_VLM_REFRESH=1 to record it."
        )
    return str(response["raw"])


def _entry_for_case(case: VLMMetricCase, raw: str) -> dict[str, Any]:
    if case.raw_override is not None:
        return {
            "type": "vlm_detect",
            "status": "",
            "action": "",
            "reason": "parse_failure",
            "fallback": "parse_failed",
            "vlm_raw": raw,
            "subgoal": case.subgoals[case.subgoal_idx],
            "subgoal_idx": case.subgoal_idx,
        }
    data = parse_json(raw)
    if case.prompt_kind == "plan":
        return {
            "type": "decompose",
            "instruction": case.subgoals[0] if case.subgoals else "",
            "subgoals": data.get("subgoals", []),
            "ordered": data.get("ordered"),
            "vlm_raw": raw,
            "subgoal_idx": case.subgoal_idx,
        }
    return {
        "type": "vlm_detect",
        "status": data.get("status"),
        "action": data.get("action"),
        "reason": data.get("reason", ""),
        "done": data.get("done"),
        "grasp_target": data.get("grasp_target"),
        "target": data.get("target"),
        "target_object": data.get("target_object"),
        "vlm_raw": raw,
        "subgoal": case.subgoals[case.subgoal_idx],
        "subgoal_idx": case.subgoal_idx,
    }


def _metric_rows_for(case: VLMMetricCase) -> list[dict[str, Any]]:
    manager = GTMetricsManager(metric_types=case.metric_types)
    raw = _raw_response_for(case)
    events = manager.step(
        {"gt_state": case.gt_state},
        _state(case),
        control_entries=[_entry_for_case(case, raw)],
    )
    return [event.to_dict() for event in events]


@pytest.mark.parametrize(
    "family",
    ["scene_qa", "target_qa", "plan_qa", "task_qa", "failure_qa"],
)
def test_system2_vlm_fixture_matrix_has_ten_cases_per_family(family):
    cases = [case for case in CASES if case.family == family]
    assert len(cases) >= 10


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.family == "scene_qa"],
    ids=lambda case: case.case_id,
)
def test_scene_qa_vlm_cassettes_classify_against_gt(case):
    rows = _metric_rows_for(case)
    assert any(
        row["metric"] == case.expected_metric
        and row.get("observed") == case.expected_observed
        for row in rows
    ), rows


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.family == "target_qa"],
    ids=lambda case: case.case_id,
)
def test_target_qa_vlm_cassettes_classify_against_gt(case):
    rows = _metric_rows_for(case)
    assert any(
        row["metric"] == case.expected_metric
        and row.get("observed") == case.expected_observed
        for row in rows
    ), rows


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.family == "plan_qa"],
    ids=lambda case: case.case_id,
)
def test_plan_qa_vlm_cassettes_classify_against_gt(case):
    rows = _metric_rows_for(case)
    assert any(
        row["metric"] == case.expected_metric
        and row.get("observed") == case.expected_observed
        for row in rows
    ), rows


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.family == "task_qa"],
    ids=lambda case: case.case_id,
)
def test_task_invariant_vlm_cassettes_classify_against_gt(case):
    rows = _metric_rows_for(case)
    assert any(
        row["metric"] == case.expected_metric
        and row.get("observed") == case.expected_observed
        for row in rows
    ), rows


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.family == "failure_qa"],
    ids=lambda case: case.case_id,
)
def test_failure_qa_vlm_cassettes_classify_against_gt(case):
    rows = _metric_rows_for(case)
    assert any(
        row["metric"] == case.expected_metric
        and row.get("observed") == case.expected_observed
        for row in rows
    ), rows



