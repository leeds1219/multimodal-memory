# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for passive ground-truth metrics detectors."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from vlm_orchestrator.strategies.base import SessionState


def _obj(pos=(0.0, 0.0, 0.02), vel=None):
    return {
        "pos": np.array(pos, dtype=np.float32),
        "quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "vel": np.array(vel or [0, 0, 0, 0, 0, 0], dtype=np.float32),
    }


def _gt_state(
    *,
    grasped=None,
    grabbed=False,
    placed=False,
    score=0.0,
    scene_objects=None,
    all_subtask_conditions=None,
    object_completed=None,
    current_index=0,
    condition_object="blue_block",
    condition_info=None,
    condition_target_objects=None,
    all_subtask_checks=None,
    task=None,
):
    scene_objects = scene_objects or ["blue_block", "red_bin", "red_block"]
    object_completed = object_completed or {"blue_block": placed}
    all_subtask_conditions = (
        all_subtask_conditions
        if all_subtask_conditions is not None
        else {"subtask_0": placed}
    )
    gt = {
        "robot": {
            "grasped_object": grasped,
            "objects_in_contact": [grasped] if grasped else [],
            "gripper_width": 0.04,
            "ee_pos": [0.3, 0.0, 0.4],
        },
        "objects": {
            "blue_block": _obj(),
            "red_block": _obj((0.2, 0.0, 0.02)),
            "red_bin": _obj((0.5, 0.0, 0.02)),
        },
        "scene_objects": scene_objects,
        "subtask": {
            "score": score,
            "conditions": [
                {
                    "object": condition_object,
                    "condition_idx": 0,
                    "predicate": "Grabbed",
                    "satisfied": grabbed,
                    "info": f"grabbed({condition_object})",
                },
                {
                    "object": condition_object,
                    "condition_idx": 3,
                    "predicate": "In",
                    "target": "red_bin",
                    "satisfied": placed,
                    "info": condition_info or (
                        f"object_in_container({condition_object}, red_bin)"
                    ),
                    "target_objects": condition_target_objects or [
                        condition_object,
                    ],
                },
            ],
            "current_index": current_index,
            "object_completed": object_completed,
            "all_subtask_conditions": all_subtask_conditions,
            "all_subtask_checks": all_subtask_checks or {
                key: {
                    "subtask_idx": idx,
                    "satisfied": satisfied,
                    "conditions": [
                        {
                            "object": condition_object,
                            "predicate": "stacked",
                            "info": condition_info or (
                                "stacked(objects=['red_block', "
                                f"'{condition_object}'], "
                                "order=bottom_to_top)"
                            ),
                            "target_objects": (
                                condition_target_objects
                                or ["red_block", condition_object]
                            ),
                            "satisfied": satisfied,
                        }
                    ],
                }
                for idx, (key, satisfied) in enumerate(
                    all_subtask_conditions.items()
                )
            },
        },
    }
    if task is not None:
        gt["task"] = task
    return gt


def _state(step=0, subgoals=None):
    state = SessionState()
    state.episode_id = 1
    state.infer_count = step
    state.episode_step = step * 8
    state.subgoals = subgoals or [
        "Pick up the blue block and place it in the red bin"
    ]
    state.current_subgoal_idx = 0
    return state


def test_vlm_perception_mismatch_logs_color_confusion():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"perception"})
    state = _state(step=1)
    obs = {"gt_state": _gt_state()}

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "subgoals": [
                    "Pick up the purple block and place it in the red bin"
                ],
                "subgoal_idx": 0,
            }
        ],
    )

    assert len(events) == 1
    event = events[0]
    assert event.metric == "vlm_perception_mismatch"
    assert event.category == "perception"
    assert event.subject_object == "blue_block"
    assert event.expected == "blue block"
    assert event.observed == "purple block"
    assert event.evidence["source_log_type"] == "decompose"


def test_vlm_perception_mismatch_ignores_valid_support_blocks():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"perception"})
    state = _state(
        step=1,
        subgoals=["Pick the blue block and stack it on the red block"],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=["blue_block", "red_block", "green_block"],
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": (
                    "Stack the blue block on the red block, and stack "
                    "the green block on the blue block"
                ),
                "subgoals": [
                    "Pick the blue block and stack it on the red block",
                    "Pick the green block and stack it on the blue block",
                ],
            }
        ],
    )

    assert [e for e in events if e.metric == "vlm_perception_mismatch"] == []


def test_vlm_completion_mismatch_logs_false_complete_against_gt():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"perception", "gt_state"})
    state = _state(
        step=20,
        subgoals=["Pick the blue block and stack it on the red block"],
    )
    obs = {
        "gt_state": _gt_state(
            placed=False,
            score=0.0,
            scene_objects=["blue_block", "red_block", "green_block"],
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detection",
                "status": "complete",
                "action": "next",
                "reason": "The blue block is on the red block.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    mismatch = [e for e in events if e.metric == "vlm_completion_mismatch"]
    checks = [e for e in events if e.metric == "gt_state_check"]

    assert len(mismatch) == 1
    assert mismatch[0].subject_object == "blue_block"
    assert mismatch[0].observed == "vlm_complete_but_gt_incomplete"
    assert mismatch[0].evidence["source_log_type"] == "vlm_detection"
    assert mismatch[0].evidence["current_subtask_satisfied"] is False
    assert len(checks) == 1
    assert checks[0].evidence["target_objects"][0] == "blue_block"


def test_vlm_scene_qa_logs_vlm_result_gt_state_and_comparison():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa"})
    state = _state(
        step=10,
        subgoals=["Pick the blue block and stack it on the red block"],
    )
    obs = {"gt_state": _gt_state(placed=False, score=0.0)}

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "continue",
                "reason": "The robot is moving the blue block.",
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    event = events[0]
    assert event.metric == "vlm_scene_qa"
    assert event.observed == "aligned_incomplete"
    assert event.evidence["vlm"]["status"] == "in_progress"
    assert event.evidence["gt_predicate_satisfied"] is False
    assert event.evidence["matched_predicate"]["satisfied"] is False
    assert "blue_block" in event.evidence["object_states"]
    assert event.decision_id
    assert event.decision_kind == "scene_success_check"
    assert event.episode_step == 80


def test_vlm_scene_qa_logs_failure_for_false_complete():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa"})
    state = _state(
        step=20,
        subgoals=["Pick the blue block and stack it on the red block"],
    )
    obs = {"gt_state": _gt_state(placed=False, score=0.0)}

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detection",
                "status": "complete",
                "action": "next",
                "reason": "The blue block is stacked on the red block.",
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    event = events[0]
    assert event.metric == "vlm_scene_qa_failure"
    assert event.observed == "false_complete"
    assert event.causal_role == "perception_error"
    assert event.evidence["vlm_claims_complete"] is True
    assert event.evidence["gt_predicate_satisfied"] is False


def test_vlm_response_format_logs_parse_failure():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"format_qa"})
    state = _state(
        step=20,
        subgoals=["Pick the blue block and stack it on the red block"],
    )

    events = manager.step(
        {},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "continue",
                "reason": "parse_failure",
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
                "vlm_raw": (
                    '{"status":"in_progress","action":"continue"}\n'
                    "extra explanation"
                ),
            }
        ],
    )

    assert len(events) == 1
    event = events[0]
    assert event.metric == "vlm_response_parse_failure"
    assert event.category == "format_qa"
    assert event.observed == "parse_failure"
    assert event.causal_role == "response_format_error"
    assert event.decision_kind == "scene_success_check"
    assert event.evidence["parse_failed"] is True
    assert "extra explanation" in event.evidence["raw"]


def test_vlm_scene_qa_marks_parse_failure_unchecked():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa"})
    state = _state(
        step=20,
        subgoals=["Pick the blue block and stack it on the red block"],
    )

    events = manager.step(
        {"gt_state": _gt_state(placed=False)},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "continue",
                "reason": "parse_failure",
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
                "vlm_raw": '{"status":"in_progress"}\nextra text',
            }
        ],
    )

    scene_events = [e for e in events if e.category == "scene_qa"]
    assert len(scene_events) == 1
    event = scene_events[0]
    assert event.metric == "vlm_scene_qa"
    assert event.observed == "unchecked_parse_failure"
    assert event.causal_role == "visual_state_unchecked"
    assert event.evidence["parse_failed"] is True


def test_vlm_scene_qa_uses_manipulated_object_for_container_subgoal():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa"})
    state = _state(
        step=20,
        subgoals=[
            "Place the green lime into the white bowl",
            "Place the blue cube into the grey bin",
        ],
    )
    gt = _gt_state(
        scene_objects=[
            "serving_bowl",
            "bin_a02",
            "lime01",
            "blue_block",
        ],
        condition_object="lime01",
        object_completed={"lime01": True, "blue_block": False},
        all_subtask_conditions={"subtask_0": False},
        placed=True,
        task={
            "success_satisfied": False,
            "success_checks": [
                {
                    "name": "all_food_items_in_serving_bowl",
                    "predicate": "object_in_container",
                    "objects": ["lime01"],
                    "reference": "serving_bowl",
                    "satisfied": True,
                },
                {
                    "name": "all_nonfood_items_in_bin",
                    "predicate": "object_in_container",
                    "objects": ["blue_block"],
                    "reference": "bin_a02",
                    "satisfied": False,
                },
            ],
            "invariant_checks": [],
        },
    )
    gt["subtask"]["conditions"].extend([
        {
            "object": "blue_block",
            "condition_idx": 0,
            "predicate": None,
            "satisfied": False,
            "info": "",
        },
        {
            "object": "blue_block",
            "condition_idx": 3,
            "predicate": None,
            "satisfied": False,
            "info": "",
        },
    ])

    events = manager.step(
        {"gt_state": gt},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "complete",
                "action": "next",
                "reason": "The green lime is now inside the white bowl.",
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
            }
        ],
    )

    scene_events = [e for e in events if e.category == "scene_qa"]
    assert len(scene_events) == 1
    assert scene_events[0].metric == "vlm_scene_qa"
    assert scene_events[0].observed == "confirmed_complete"
    assert scene_events[0].subject_object == "lime01"
    assert scene_events[0].evidence["target_objects"] == ["lime01"]


def test_vlm_task_invariant_qa_flags_false_task_success_and_invariant_miss():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa", "gt_state"})
    state = _state(
        step=170,
        subgoals=[
            "Pick the Rubik's cube from the bowl and place it on the table"
        ],
    )
    obs = {
        "gt_state": _gt_state(
            placed=True,
            score=1.0,
            scene_objects=[
                "serving_bowl",
                "orange_01",
                "lemon_02",
                "rubiks_cube",
                "table",
            ],
            object_completed={"rubiks_cube": True},
            all_subtask_conditions={"subtask_0": False},
            condition_object="rubiks_cube",
            task={
                "success_satisfied": False,
                "success_checks": [
                    {
                        "name": "rubiks_cube_removed_from_bowl",
                        "predicate": "object_outside_of",
                        "objects": ["rubiks_cube"],
                        "reference": "serving_bowl",
                        "satisfied": True,
                    },
                ],
                "invariant_checks": [
                    {
                        "name": "orange_remains_in_bowl",
                        "predicate": "object_in_container",
                        "objects": ["orange_01"],
                        "reference": "serving_bowl",
                        "satisfied": True,
                    },
                    {
                        "name": "lemon_remains_in_bowl",
                        "predicate": "object_in_container",
                        "objects": ["lemon_02"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                ],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "complete",
                "action": "next",
                "reason": (
                    "The Rubik's cube has been removed from the bowl and "
                    "is now on the table. The task is complete."
                ),
                "subgoal": state.subgoals[0],
            }
        ],
    )

    task_failures = [
        e for e in events if e.metric == "vlm_task_success_qa_failure"
    ]
    invariant_failures = [
        e for e in events if e.metric == "vlm_invariant_qa_failure"
    ]
    task_checks = [e for e in events if e.metric == "gt_task_state_check"]

    assert len(task_failures) == 1
    assert task_failures[0].observed == "false_task_success"
    assert task_failures[0].evidence["task_success_satisfied"] is False
    assert task_failures[0].evidence["failed_invariants"] == [
        "lemon_remains_in_bowl"
    ]
    assert len(invariant_failures) == 1
    assert invariant_failures[0].subject_object == "lemon_02"
    assert invariant_failures[0].observed == "missed_invariant_violation"
    assert invariant_failures[0].expected == (
        "object_in_container(lemon_02, serving_bowl)"
    )
    assert len(task_checks) == 1
    assert task_checks[0].observed == "task_incomplete"
    assert task_checks[0].evidence["failed_invariants"] == [
        "lemon_remains_in_bowl"
    ]


def test_vlm_task_qa_does_not_promote_intermediate_subgoal_complete_to_task_success():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa", "gt_state"})
    state = _state(
        step=20,
        subgoals=[
            "Pick up the blue cube and place it in the grey bin",
            "Pick up the green lime and place it in the white bowl",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            placed=True,
            score=0.5,
            scene_objects=[
                "serving_bowl",
                "bin_a02",
                "lime01",
                "blue_block",
            ],
            object_completed={"blue_block": True, "lime01": False},
            all_subtask_conditions={"subtask_0": False},
            condition_object="blue_block",
            task={
                "success_satisfied": False,
                "success_checks": [
                    {
                        "name": "all_food_items_in_serving_bowl",
                        "predicate": "object_in_container",
                        "objects": ["lime01"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                    {
                        "name": "all_nonfood_items_in_bin",
                        "predicate": "object_in_container",
                        "objects": ["blue_block"],
                        "reference": "bin_a02",
                        "satisfied": True,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "complete",
                "action": "next",
                "reason": "The blue cube subgoal has been completed.",
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
            }
        ],
    )

    task_events = [
        e for e in events
        if e.metric in {"vlm_task_success_qa", "vlm_task_success_qa_failure"}
    ]
    assert len(task_events) == 1
    assert task_events[0].metric == "vlm_task_success_qa"
    assert task_events[0].observed == "aligned_task_incomplete"
    assert task_events[0].evidence["vlm_claims_subgoal_complete"] is True
    assert task_events[0].evidence["vlm_claims_task_complete"] is False
    assert [e for e in events if e.metric.startswith("vlm_invariant_qa")] == []


def test_vlm_task_qa_does_not_match_negated_task_complete_phrase():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"task_qa"})
    state = _state(
        step=20,
        subgoals=[
            "Pick up the blue cube and place it in the grey bin",
            "Pick up the green lime and place it in the white bowl",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            placed=True,
            score=0.5,
            scene_objects=[
                "serving_bowl",
                "bin_a02",
                "lime01",
                "blue_block",
            ],
            object_completed={"blue_block": True, "lime01": False},
            all_subtask_conditions={"subtask_0": False},
            condition_object="blue_block",
            task={
                "success_satisfied": False,
                "success_checks": [
                    {
                        "name": "all_food_items_in_serving_bowl",
                        "predicate": "object_in_container",
                        "objects": ["lime01"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                    {
                        "name": "all_nonfood_items_in_bin",
                        "predicate": "object_in_container",
                        "objects": ["blue_block"],
                        "reference": "bin_a02",
                        "satisfied": True,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "complete",
                "action": "next",
                "reason": (
                    "The blue cube subgoal is complete, but not all "
                    "required subgoals are complete yet."
                ),
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
            }
        ],
    )

    task_events = [
        e for e in events
        if e.metric in {"vlm_task_success_qa", "vlm_task_success_qa_failure"}
    ]
    assert len(task_events) == 1
    assert task_events[0].metric == "vlm_task_success_qa"
    assert task_events[0].observed == "aligned_task_incomplete"
    assert task_events[0].evidence["vlm_claims_subgoal_complete"] is True
    assert task_events[0].evidence["vlm_claims_task_complete"] is False


def test_vlm_task_invariant_qa_skips_without_structured_gt_task_checks():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa", "gt_state"})
    state = _state(
        step=20,
        subgoals=[
            "Pick the Rubik's cube from the bowl and place it on the table"
        ],
    )

    events = manager.step(
        {"gt_state": _gt_state(placed=True, score=1.0)},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "complete",
                "action": "next",
                "reason": "The task is complete.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert [
        e for e in events
        if e.metric in {
            "vlm_task_success_qa",
            "vlm_task_success_qa_failure",
            "vlm_invariant_qa",
            "vlm_invariant_qa_failure",
            "gt_task_state_check",
        }
    ] == []


def test_task_qa_not_duplicated_when_selected_with_perception():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"perception", "task_qa"})
    state = _state(
        step=170,
        subgoals=[
            "Pick the Rubik's cube from the bowl and place it on the table"
        ],
    )
    obs = {
        "gt_state": _gt_state(
            placed=True,
            score=1.0,
            scene_objects=[
                "serving_bowl",
                "lemon_02",
                "rubiks_cube",
                "table",
            ],
            object_completed={"rubiks_cube": True},
            all_subtask_conditions={"subtask_0": False},
            condition_object="rubiks_cube",
            task={
                "success_satisfied": False,
                "success_checks": [
                    {
                        "name": "rubiks_cube_removed_from_bowl",
                        "predicate": "object_outside_of",
                        "objects": ["rubiks_cube"],
                        "reference": "serving_bowl",
                        "satisfied": True,
                    },
                ],
                "invariant_checks": [
                    {
                        "name": "lemon_remains_in_bowl",
                        "predicate": "object_in_container",
                        "objects": ["lemon_02"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                ],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "complete",
                "action": "next",
                "reason": "The task is complete.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert [
        e.metric for e in events
        if e.metric == "vlm_task_success_qa_failure"
    ] == ["vlm_task_success_qa_failure"]
    assert [
        e.metric for e in events
        if e.metric == "vlm_invariant_qa_failure"
    ] == ["vlm_invariant_qa_failure"]


def test_vlm_scene_qa_prefers_completed_subtask_over_snapshot_conditions():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa", "gt_state"})
    state = _state(
        step=170,
        subgoals=["Pick up the blue block and place it in the red bin"],
    )

    events = manager.step(
        {
            "gt_state": _gt_state(
                placed=True,
                score=1.0,
                object_completed={"blue_block": True},
                all_subtask_conditions={"subtask_0": False},
            )
        },
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "continue",
                "reason": "The robot still needs to place the blue block.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    scene_events = [e for e in events if e.category == "scene_qa"]
    assert len(scene_events) == 1
    assert scene_events[0].metric == "vlm_scene_qa_failure"
    assert scene_events[0].observed == "missed_complete"
    assert scene_events[0].evidence["current_subtask_satisfied"] is True


def _out_of_scene_stack_gt():
    gt = _gt_state(
        placed=False,
        scene_objects=["green_block", "red_block"],
        object_completed={
            "green_block": False,
            "red_block": False,
        },
        all_subtask_conditions={"subtask_0": False},
        condition_object="green_block",
        condition_info=(
            "stacked(objects=['red_block', 'green_block'], "
            "order=bottom_to_top)"
        ),
        condition_target_objects=["red_block", "green_block"],
    )
    gt["objects"]["green_block"] = _obj((0.3, 0.0, 0.02))
    gt["objects"]["red_block"] = _obj((0.2, 0.0, -0.65))
    return gt


def test_vlm_failure_state_qa_flags_missed_out_of_scene_target():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"failure_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the green block and stack it on the red block"],
    )

    events = manager.step(
        {"gt_state": _out_of_scene_stack_gt()},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "continue",
                "reason": "The robot should keep trying.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    event = events[0]
    assert event.metric == "vlm_failure_missed"
    assert event.observed == "missed_blocking_failure"
    assert event.causal_role == "failure_detection_error"
    assert event.subject_object == "red_block"
    failure = event.evidence["blocking_failures"][0]
    assert failure["failure_type"] == "object_out_of_scene"
    assert failure["object"] == "red_block"
    assert failure["pos"][2] < -0.05


def test_vlm_failure_state_qa_accepts_detected_out_of_scene_failure():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"failure_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the green block and stack it on the red block"],
    )

    events = manager.step(
        {"gt_state": _out_of_scene_stack_gt()},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "failure",
                "action": "replan",
                "reason": "The red block has fallen off the table.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    event = events[0]
    assert event.metric == "vlm_failure_qa"
    assert event.observed == "failure_detected"
    assert event.causal_role == "failure_detection_audit"
    assert event.evidence["vlm_claims_failure"] is True


def test_vlm_completion_mismatch_uses_source_subgoal_idx_after_advance():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"perception", "gt_state"})
    state = _state(
        step=20,
        subgoals=[
            "Pick the blue block and stack it on the red block",
            "Pick the green block and stack it on the blue block",
        ],
    )
    state.current_subgoal_idx = 1
    obs = {
        "gt_state": _gt_state(
            score=0.5,
            scene_objects=["blue_block", "red_block", "green_block"],
            object_completed={
                "red_block": True,
                "blue_block": True,
                "green_block": False,
            },
            all_subtask_conditions={
                "subtask_0": True,
                "subtask_1": False,
            },
            current_index=1,
            condition_object="green_block",
            condition_target_objects=[
                "red_block", "blue_block", "green_block",
            ],
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detection",
                "status": "complete",
                "action": "next",
                "reason": "The blue block is on the red block.",
                "subgoal_idx": 0,
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert [e for e in events if e.metric == "vlm_completion_mismatch"] == []
    checks = [e for e in events if e.metric == "gt_state_check"]
    assert len(checks) == 1
    assert checks[0].subgoal_idx == 0
    assert checks[0].subject_object == "blue_block"
    assert checks[0].evidence["current_subtask_key"] == "subtask_0"
    assert checks[0].evidence["current_subtask_satisfied"] is True


def test_vlm_completion_mismatch_suppressed_when_gt_satisfied():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"perception"})
    state = _state(
        step=20,
        subgoals=["Pick the blue block and stack it on the red block"],
    )
    obs = {"gt_state": _gt_state(placed=True, score=1.0)}

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detection",
                "status": "complete",
                "action": "next",
                "reason": "The blue block is on the red block.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert [e for e in events if e.metric == "vlm_completion_mismatch"] == []


def test_vlm_scene_qa_logs_missed_complete_when_gt_satisfied():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"scene_qa"})
    state = _state(
        step=30,
        subgoals=["Pick the blue block and stack it on the red block"],
    )
    obs = {"gt_state": _gt_state(placed=True, score=1.0)}

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "continue",
                "reason": "The robot still needs to place the block.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_scene_qa_failure"
    assert events[0].observed == "missed_complete"


def test_vlm_grasp_target_qa_accepts_current_target():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the blue block and stack it on the red block"],
    )

    events = manager.step(
        {"gt_state": _gt_state(placed=False)},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "failure",
                "action": "grasp_tool",
                "grasp_target": "blue block",
                "reason": "Use the grasp tool on the blue block.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_qa"
    assert events[0].observed == "target_aligned"
    assert events[0].evidence["observed_target"] == "blue block"


def test_vlm_grasp_target_qa_reads_target_from_detection_raw_response():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the blue block and stack it on the red block"],
    )

    events = manager.step(
        {"gt_state": _gt_state(placed=False)},
        state,
        control_entries=[
            {
                "type": "vlm_detection",
                "status": "in_progress",
                "action": "grasp_tool",
                "reason": "Use the grasp tool on the blue block.",
                "subgoal_idx": 0,
                "vlm_raw": json.dumps({
                    "status": "in_progress",
                    "action": "grasp_tool",
                    "reason": "Use the grasp tool on the blue block.",
                    "grasp_target": "blue block",
                }),
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_qa"
    assert events[0].observed == "target_aligned"
    assert events[0].evidence["observed_target"] == "blue block"


def test_vlm_grasp_target_qa_flags_wrong_target():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the blue block and stack it on the red block"],
    )

    events = manager.step(
        {
            "gt_state": _gt_state(
                scene_objects=["blue_block", "red_block", "yellow_block"],
                placed=False,
            )
        },
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "failure",
                "action": "grasp_tool",
                "grasp_target": "yellow block",
                "reason": "Use the grasp tool on the yellow block.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_mismatch"
    assert events[0].observed == "target_mismatch"


def test_vlm_grasp_target_qa_rejects_support_object_for_stack_subgoal():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the blue block and stack it on the red block"],
    )

    events = manager.step(
        {
            "gt_state": _gt_state(
                scene_objects=["blue_block", "red_block"],
                placed=False,
            )
        },
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "failure",
                "action": "grasp_tool",
                "grasp_target": "red block",
                "reason": "Use the grasp tool on the red support block.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_mismatch"
    assert events[0].observed == "target_mismatch"
    assert events[0].evidence["expected_targets"] == ["blue_block"]


def test_vlm_grasp_target_qa_accepts_cube_alias_for_block_target():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the green block and stack it on the red block"],
    )

    events = manager.step(
        {
            "gt_state": _gt_state(
                scene_objects=["green_block", "red_block"],
                condition_object="green_block",
                object_completed={"green_block": False},
                placed=False,
            )
        },
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "failure",
                "action": "grasp_tool",
                "grasp_target": "green cube",
                "reason": "Use the grasp tool on the green cube.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_qa"
    assert events[0].observed == "target_aligned"


def test_vlm_grasp_target_qa_accepts_common_sense_object_alias():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=72,
        subgoals=["Pick the lime from the grey bin and place it on the table"],
    )

    events = manager.step(
        {
            "gt_state": _gt_state(
                scene_objects=["container_b03", "lime01", "table"],
                condition_object="lime01",
                object_completed={"lime01": False},
                all_subtask_conditions={"subtask_0": False},
                placed=False,
            )
        },
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "grasp_tool",
                "grasp_target": "green lime",
                "reason": "Use the grasp tool on the green lime.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_qa"
    assert events[0].observed == "target_aligned"
    assert events[0].evidence["expected_targets"] == ["lime01"]


def test_vlm_grasp_target_qa_dedupes_detect_and_detection_rows():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=40,
        subgoals=["Pick the blue block and stack it on the red block"],
    )
    raw = json.dumps({
        "status": "in_progress",
        "action": "grasp_tool",
        "reason": "Use the grasp tool on the blue block.",
        "grasp_target": "blue block",
    })

    events = manager.step(
        {"gt_state": _gt_state(placed=False)},
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "grasp_tool",
                "grasp_target": "blue block",
                "reason": "Use the grasp tool on the blue block.",
                "subgoal": state.subgoals[0],
                "vlm_raw": raw,
            },
            {
                "type": "vlm_detection",
                "status": "in_progress",
                "action": "grasp_tool",
                "reason": "Use the grasp tool on the blue block.",
                "subgoal_idx": 0,
                "vlm_raw": raw,
            },
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_qa"
    assert events[0].observed == "target_aligned"


def test_vlm_grasp_target_qa_accepts_unique_can_category_alias():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=30,
        subgoals=["Pick the tuna can from the bowl and place it on the table"],
    )

    events = manager.step(
        {
            "gt_state": _gt_state(
                scene_objects=[
                    "serving_bowl",
                    "orange_01",
                    "lemon_02",
                    "tuna_can",
                ],
                condition_object="tuna_can",
                object_completed={"tuna_can": False},
                all_subtask_conditions={"subtask_0": False},
                placed=False,
            )
        },
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "grasp_tool",
                "grasp_target": "blue tin can",
                "reason": "Use the grasp tool on the blue tin can.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_qa"
    assert events[0].observed == "target_aligned"
    assert events[0].evidence["observed_scene_target"] == "tuna_can"


def test_vlm_grasp_target_qa_rejects_ambiguous_can_category_alias():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"target_qa"})
    state = _state(
        step=30,
        subgoals=["Pick the tuna can from the bowl and place it on the table"],
    )

    events = manager.step(
        {
            "gt_state": _gt_state(
                scene_objects=[
                    "serving_bowl",
                    "tuna_can",
                    "soup_can",
                ],
                condition_object="tuna_can",
                object_completed={"tuna_can": False},
                all_subtask_conditions={"subtask_0": False},
                placed=False,
            )
        },
        state,
        control_entries=[
            {
                "type": "vlm_detect",
                "status": "in_progress",
                "action": "grasp_tool",
                "grasp_target": "tin can",
                "reason": "Use the grasp tool on the tin can.",
                "subgoal": state.subgoals[0],
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_grasp_target_mismatch"
    assert events[0].observed == "target_mismatch"
    assert events[0].evidence["observed_scene_target"] is None


def _stack_checks():
    return {
        "subtask_0": {
            "subtask_idx": 0,
            "satisfied": False,
            "conditions": [
                {
                    "object": "blue_block",
                    "predicate": "stacked",
                    "info": "stacked(objects=['red_block', 'blue_block'], order=bottom_to_top)",
                    "target_objects": ["red_block", "blue_block"],
                    "satisfied": False,
                }
            ],
        },
        "subtask_1": {
            "subtask_idx": 1,
            "satisfied": False,
            "conditions": [
                {
                    "object": "green_block",
                    "predicate": "stacked",
                    "info": "stacked(objects=['red_block', 'blue_block', 'green_block'], order=bottom_to_top)",
                    "target_objects": [
                        "red_block", "blue_block", "green_block",
                    ],
                    "satisfied": False,
                }
            ],
        },
    }


def test_vlm_plan_alignment_accepts_color_stack_decompose():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the blue block and stack it on the red block",
            "Pick the green block and stack it on the blue block",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=["red_block", "blue_block", "green_block"],
            all_subtask_conditions={"subtask_0": False, "subtask_1": False},
            all_subtask_checks=_stack_checks(),
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": (
                    "Stack the blue block on the red block, then "
                    "stack the green block on the blue block"
                ),
                "subgoals": state.subgoals,
                "ordered": True,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].decision_kind == "plan"


def test_vlm_plan_alignment_accepts_color_only_stack_phrasing():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Stack blue on red",
            "Stack green on blue",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=["red_block", "blue_block", "green_block"],
            all_subtask_conditions={"subtask_0": False, "subtask_1": False},
            all_subtask_checks=_stack_checks(),
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Stack blue on red, then green on blue.",
                "subgoals": state.subgoals,
                "ordered": True,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["parsed_relations"] == [
        {"bottom": "red_block", "top": "blue_block"},
        {"bottom": "blue_block", "top": "green_block"},
    ]


def test_vlm_plan_alignment_accepts_cube_worded_stack_subgoals():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the blue cube and stack it on the red cube",
            "Pick the green cube and stack it on the blue cube",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=["red_block", "blue_block", "green_block"],
            all_subtask_conditions={"subtask_0": False, "subtask_1": False},
            all_subtask_checks=_stack_checks(),
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": (
                    "Stack the blue cube on the red cube, then "
                    "stack the green cube on the blue cube"
                ),
                "subgoals": state.subgoals,
                "ordered": True,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["parsed_relations"] == [
        {"bottom": "red_block", "top": "blue_block"},
        {"bottom": "blue_block", "top": "green_block"},
    ]


def test_vlm_plan_alignment_flags_wrong_color_stack_order():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(step=1)
    obs = {
        "gt_state": _gt_state(
            scene_objects=["red_block", "blue_block", "green_block"],
            all_subtask_conditions={"subtask_0": False, "subtask_1": False},
            all_subtask_checks=_stack_checks(),
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Stack blocks.",
                "subgoals": [
                    "Pick the green block and stack it on the blue block",
                    "Pick the blue block and stack it on the red block",
                ],
                "ordered": True,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_mismatch"
    assert events[0].observed == "plan_mismatch"


def _sort_task_checks():
    return {
        "success_checks": [
            {
                "name": "all_food_items_in_serving_bowl",
                "type": "success",
                "predicate": "object_in_container",
                "objects": ["lime01", "lemon_02", "lychee01"],
                "reference": "serving_bowl",
                "satisfied": False,
            },
            {
                "name": "all_nonfood_items_in_bin",
                "type": "success",
                "predicate": "object_in_container",
                "objects": ["blue_block", "rubiks_cube", "crabbypenholder"],
                "reference": "bin_a02",
                "satisfied": False,
            },
        ],
        "invariant_checks": [
            {
                "name": "food_exemplars_remain_in_serving_bowl",
                "type": "invariant",
                "predicate": "object_in_container",
                "objects": ["lemon_02", "lychee01"],
                "reference": "serving_bowl",
                "satisfied": True,
            },
            {
                "name": "nonfood_exemplars_remain_in_bin",
                "type": "invariant",
                "predicate": "object_in_container",
                "objects": ["rubiks_cube", "crabbypenholder"],
                "reference": "bin_a02",
                "satisfied": True,
            },
        ],
    }


def test_vlm_plan_alignment_accepts_sort_container_plan():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick up the blue cube and place it in the grey bin",
            "Pick up the green lime and place it in the white bowl",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "bin_a02",
                "lime01",
                "lemon_02",
                "lychee01",
                "blue_block",
                "rubiks_cube",
                "crabbypenholder",
            ],
            object_completed={"lime01": False, "blue_block": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task=_sort_task_checks(),
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Sort food and non-food items.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["expected_relation_type"] == "container"
    assert events[0].evidence["parsed_relations"] == [
        {"object": "blue_block", "container": "bin_a02"},
        {"object": "lime01", "container": "serving_bowl"},
    ]


def test_vlm_plan_alignment_uses_subtask_container_checks_without_task_checks():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the yellow mustard bottle and place it in bin_a06",
            "Pick the coffee can and place it in bin_b03",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "bin_a06",
                "bin_b03",
                "mustard",
                "coffee_can",
                "sugar_box",
            ],
            object_completed={"mustard": False, "coffee_can": False},
            all_subtask_conditions={
                "subtask_0": False,
                "subtask_1": False,
            },
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [
                        {
                            "object": "mustard",
                            "predicate": "object_grabbed",
                            "info": "object_grabbed(object=mustard)",
                            "target_objects": ["mustard"],
                            "satisfied": False,
                        },
                        {
                            "object": "mustard",
                            "predicate": "object_in_container",
                            "info": (
                                "object_in_container(object=mustard, "
                                "container=bin_a06, tolerance=0.01)"
                            ),
                            "target_objects": ["mustard"],
                            "satisfied": False,
                        },
                    ],
                },
                "subtask_1": {
                    "subtask_idx": 1,
                    "satisfied": False,
                    "conditions": [
                        {
                            "object": "coffee_can",
                            "predicate": "object_in_container",
                            "info": (
                                "object_in_container(object=coffee_can, "
                                "container=bin_b03, tolerance=0.01)"
                            ),
                            "target_objects": ["coffee_can"],
                            "satisfied": False,
                        },
                    ],
                },
            },
            task=None,
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Sort yellow and blue objects.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["expected_relation_type"] == "container"
    assert events[0].evidence["expected_relations"] == [
        {
            "object": "mustard",
            "container": "bin_a06",
            "predicate": (
                "object_in_container(object=mustard, "
                "container=bin_a06, tolerance=0.01)"
            ),
        },
        {
            "object": "coffee_can",
            "container": "bin_b03",
            "predicate": (
                "object_in_container(object=coffee_can, "
                "container=bin_b03, tolerance=0.01)"
            ),
        },
    ]


def _food_packing_by_color_gt_state():
    return {
        "robot": {
            "grasped_object": None,
            "objects_in_contact": [],
            "gripper_width": 0.04,
            "ee_pos": [0.3, 0.0, 0.4],
        },
        "objects": {
            "bin_a06": _obj((0.71, 0.37, 0.003)),
            "bin_b03": _obj((0.65, -0.39, 0.003)),
            "mustard": _obj((0.41, 0.15, 0.098)),
            "coffee_can": _obj((0.32, 0.10, 0.073)),
            "sugar_box": _obj((0.59, 0.12, 0.091)),
        },
        "scene_objects": [
            "bin_a06",
            "bin_b03",
            "mustard",
            "coffee_can",
            "sugar_box",
        ],
        "subtask": {
            "score": 0.0,
            "conditions": [],
            "current_index": 0,
            "object_completed": {
                "mustard": False,
                "coffee_can": False,
            },
            "all_subtask_conditions": {
                "subtask_0": False,
                "subtask_1": False,
            },
            "all_subtask_checks": {
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [
                        {
                            "object": "mustard",
                            "predicate": "object_in_container",
                            "info": (
                                "object_in_container(object=mustard, "
                                "container=bin_a06, tolerance=0.01)"
                            ),
                            "target_objects": ["mustard"],
                            "satisfied": False,
                        },
                    ],
                },
                "subtask_1": {
                    "subtask_idx": 1,
                    "satisfied": False,
                    "conditions": [
                        {
                            "object": "coffee_can",
                            "predicate": "object_in_container",
                            "info": (
                                "object_in_container(object=coffee_can, "
                                "container=bin_b03, tolerance=0.01)"
                            ),
                            "target_objects": ["coffee_can"],
                            "satisfied": False,
                        },
                    ],
                },
            },
        },
    }


def test_vlm_plan_alignment_accepts_natural_sort_container_aliases():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the yellow mustard bottle and place it in the right container",
            "Pick the blue coffee can and place it in the left container",
        ],
    )

    events = manager.step(
        {"gt_state": _food_packing_by_color_gt_state()},
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": (
                    "Pack yellow objects in right container and blue object "
                    "in the left container"
                ),
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["parsed_relations"] == [
        {"object": "mustard", "container": "bin_a06"},
        {"object": "coffee_can", "container": "bin_b03"},
    ]


def test_vlm_plan_alignment_flags_extra_sort_object_with_aliases():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the yellow mustard bottle and place it in the right container",
            "Pick the yellow Domino sugar box and place it in the right container",
            "Pick the blue container (blue lid jar) and place it in the left container",
        ],
    )

    events = manager.step(
        {"gt_state": _food_packing_by_color_gt_state()},
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": (
                    "Pack yellow objects in right container and blue object "
                    "in the left container"
                ),
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_mismatch"
    assert events[0].observed == "plan_mismatch"
    assert events[0].evidence["unchecked_reason"] is None
    assert events[0].evidence["parsed_relations"] == [
        {"object": "mustard", "container": "bin_a06"},
        {"object": "sugar_box", "container": "bin_a06"},
        {"object": "coffee_can", "container": "bin_b03"},
    ]


def test_vlm_plan_alignment_flags_wrong_sort_container():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick up the blue cube and place it in the white bowl",
            "Pick up the green lime and place it in the grey bin",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "bin_a02",
                "lime01",
                "lemon_02",
                "lychee01",
                "blue_block",
                "rubiks_cube",
                "crabbypenholder",
            ],
            object_completed={"lime01": False, "blue_block": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task=_sort_task_checks(),
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Sort food and non-food items.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_mismatch"
    assert events[0].observed == "plan_mismatch"
    assert events[0].evidence["expected_relation_type"] == "container"


def test_vlm_plan_alignment_parses_source_and_destination_container_plan():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the tuna can from the white plate and place it into the grey tray"
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "container_b03",
                "tuna_can",
                "corn_can",
                "spam_can",
            ],
            object_completed={"tuna_can": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task={
                "success_checks": [
                    {
                        "name": "tuna_can_moved_to_can_bin",
                        "type": "success",
                        "predicate": "object_in_container",
                        "objects": ["tuna_can"],
                        "reference": "container_b03",
                        "satisfied": False,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Move the misplaced can.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["parsed_relations"] == [
        {"object": "tuna_can", "container": "container_b03"}
    ]


def test_vlm_plan_alignment_resolves_blue_can_alias():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the blue can from the white plate and place it into the grey tray"
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "container_b03",
                "tuna_can",
                "corn_can",
                "spam_can",
            ],
            object_completed={"tuna_can": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task={
                "success_checks": [
                    {
                        "name": "tuna_can_moved_to_can_bin",
                        "type": "success",
                        "predicate": "object_in_container",
                        "objects": ["tuna_can"],
                        "reference": "container_b03",
                        "satisfied": False,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Move the misplaced can.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["parsed_relations"] == [
        {"object": "tuna_can", "container": "container_b03"}
    ]


def test_vlm_plan_alignment_ignores_trailing_spatial_modifiers():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Place the spam can from the grey tray into the grey tray "
            "upright beside the corn can"
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=["container_b03", "spam_can", "corn_can"],
            object_completed={"spam_can": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task={
                "success_checks": [
                    {
                        "name": "spam_can_moved_to_can_bin",
                        "type": "success",
                        "predicate": "object_in_container",
                        "objects": ["spam_can"],
                        "reference": "container_b03",
                        "satisfied": False,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Clean up the tray.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["parsed_relations"] == [
        {"object": "spam_can", "container": "container_b03"}
    ]


def test_vlm_plan_alignment_accepts_recover_remove_plan():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=["Pick the blue can from the white bowl and place it on the table"],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "orange_01",
                "lemon_02",
                "tuna_can",
                "table",
            ],
            object_completed={"tuna_can": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task={
                "success_checks": [
                    {
                        "name": "tuna_can_removed_from_bowl",
                        "type": "success",
                        "predicate": "object_outside_of",
                        "objects": ["tuna_can"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Remove what does not belong.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "plan_aligned"
    assert events[0].evidence["expected_relation_type"] == "outside_container"
    assert events[0].evidence["parsed_relations"] == [
        {"object": "tuna_can", "container": "serving_bowl"}
    ]


def test_vlm_plan_alignment_flags_wrong_recover_remove_object():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=["Pick the orange from the white bowl and place it on the table"],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "orange_01",
                "lemon_02",
                "tuna_can",
                "table",
            ],
            object_completed={"tuna_can": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task={
                "success_checks": [
                    {
                        "name": "tuna_can_removed_from_bowl",
                        "type": "success",
                        "predicate": "object_outside_of",
                        "objects": ["tuna_can"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Remove what does not belong.",
                "subgoals": state.subgoals,
                "ordered": False,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_mismatch"
    assert events[0].observed == "plan_mismatch"
    assert events[0].evidence["expected_relation_type"] == "outside_container"


def test_vlm_plan_alignment_does_not_ignore_wrong_object_prelude():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the lemon from the white bowl and place it on the table",
            "Pick the blue can from the white bowl and place it on the table",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "orange_01",
                "lemon_02",
                "tuna_can",
                "table",
            ],
            object_completed={"tuna_can": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task={
                "success_checks": [
                    {
                        "name": "tuna_can_removed_from_bowl",
                        "type": "success",
                        "predicate": "object_outside_of",
                        "objects": ["tuna_can"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Remove what does not belong.",
                "subgoals": [
                    "Grab the lemon",
                    "Pick the tuna can from the white bowl and place it on the table",
                ],
                "ordered": True,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "unchecked"
    assert events[0].evidence["expected_relation_type"] == "outside_container"
    assert events[0].evidence["ignored_plan_preludes"] == []


def test_vlm_plan_alignment_keeps_all_ignored_preludes_unchecked():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(
        step=1,
        subgoals=[
            "Pick the blue can from the white bowl and place it on the table",
        ],
    )
    obs = {
        "gt_state": _gt_state(
            scene_objects=[
                "serving_bowl",
                "orange_01",
                "lemon_02",
                "tuna_can",
                "table",
            ],
            object_completed={"tuna_can": False},
            all_subtask_checks={
                "subtask_0": {
                    "subtask_idx": 0,
                    "satisfied": False,
                    "conditions": [],
                },
            },
            task={
                "success_checks": [
                    {
                        "name": "tuna_can_removed_from_bowl",
                        "type": "success",
                        "predicate": "object_outside_of",
                        "objects": ["tuna_can"],
                        "reference": "serving_bowl",
                        "satisfied": False,
                    },
                ],
                "invariant_checks": [],
            },
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Remove what does not belong.",
                "subgoals": [
                    "Grasp the tuna can",
                    "Release it on the table",
                ],
                "ordered": True,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "unchecked"
    assert events[0].evidence["expected_relation_type"] == "outside_container"
    assert events[0].evidence["unchecked_reason"] == (
        "no_parseable_vlm_outside_container_relation"
    )


def test_vlm_plan_alignment_marks_unparseable_as_unchecked():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"plan_qa"})
    state = _state(step=1)
    obs = {
        "gt_state": _gt_state(
            scene_objects=["red_block", "blue_block", "green_block"],
            all_subtask_checks=_stack_checks(),
        )
    }

    events = manager.step(
        obs,
        state,
        control_entries=[
            {
                "type": "decompose",
                "instruction": "Stack blocks.",
                "subgoals": ["Move the first block to the support block"],
                "ordered": True,
            }
        ],
    )

    assert len(events) == 1
    assert events[0].metric == "vlm_plan_qa"
    assert events[0].observed == "unchecked"
    assert (
        events[0].evidence["unchecked_reason"]
        == "unparseable_vlm_stack_relation"
    )


def test_placement_outcome_logs_release_without_terminal_condition():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(
        metric_types={"placement"},
        placement_confirm_steps=2,
    )
    state = _state()

    # First observe a legitimate grasp of the target.
    state.infer_count = 1
    assert manager.step(
        {"gt_state": _gt_state(grasped="blue_block", grabbed=True)},
        state,
        control_entries=[],
    ) == []

    # Then release it without satisfying the terminal placement predicate.
    state.infer_count = 2
    assert manager.step(
        {"gt_state": _gt_state(grasped=None, grabbed=False, placed=False)},
        state,
        control_entries=[],
    ) == []

    state.infer_count = 3
    events = manager.step(
        {"gt_state": _gt_state(grasped=None, grabbed=False, placed=False)},
        state,
        control_entries=[],
    )

    assert len(events) == 1
    event = events[0]
    assert event.metric == "placement_failed"
    assert event.category == "placement"
    assert event.subject_object == "blue_block"
    assert event.expected == "object_in_container(blue_block, red_bin)"
    assert event.observed == "released_without_terminal_condition"
    assert event.evidence["release_step"] == 2


def test_tool_causality_links_grasp_to_prior_failure_and_outcome():
    from vlm_orchestrator.gt_metrics import GTMetricsManager

    manager = GTMetricsManager(metric_types={"tool_causality"})
    state = _state()

    state.infer_count = 5
    manager.step(
        {"gt_state": _gt_state(score=0.0)},
        state,
        control_entries=[
            {
                "type": "gt_failure_detected",
                "failure_type": "object_dropped",
                "reason": "blue_block dropped",
                "target_objects": ["blue_block"],
                "subgoal_idx": 0,
            }
        ],
    )

    state.infer_count = 6
    escalation_events = manager.step(
        {"gt_state": _gt_state(score=0.0)},
        state,
        control_entries=[
            {
                "type": "gt_grasp_escalation",
                "target_object": "blue_block",
                "gt_failure_type": "object_dropped",
                "subgoal_idx": 0,
            }
        ],
    )

    assert len(escalation_events) == 1
    escalation = escalation_events[0]
    assert escalation.metric == "grasp_tool_invocation"
    assert escalation.causal_role == "recovery_attempt_after_failure"
    assert escalation.parent_event_id is not None
    assert escalation.evidence["prior_failure_type"] == "object_dropped"

    state.infer_count = 9
    outcome_events = manager.step(
        {"gt_state": _gt_state(score=1.0, placed=True)},
        state,
        control_entries=[
            {
                "type": "grasp_tool_done",
                "target_object": "blue_block",
                "confidence": 0.91,
            }
        ],
    )

    assert len(outcome_events) == 1
    outcome = outcome_events[0]
    assert outcome.metric == "grasp_tool_outcome"
    assert outcome.causal_role == "corrective_success"
    assert outcome.parent_event_id == escalation.event_id
    assert outcome.evidence["score_delta"] == 1.0


def test_session_state_metrics_drain_to_jsonl():
    from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
    from vlm_orchestrator.gt_metrics import GTMetricEvent

    with tempfile.TemporaryDirectory() as tmp:
        proxy = OrchestratorProxy(ProxyConfig(log_dir=tmp))
        state = _state(step=12)
        proxy._rotate_episode_log(1, "Pick up the blue block", state=state)

        event = GTMetricEvent(
            metric="placement_failed",
            category="placement",
            step_count=12,
            subgoal_idx=0,
            subject_object="blue_block",
            expected="in red bin",
            observed="on table",
        )
        state.log_metric(event)
        proxy._drain_metric_entries(state)

        metrics_path = (
            Path(tmp)
            / "Pick_up_the_blue_block"
            / "episode_1"
            / "metrics.jsonl"
        )
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["type"] == "gt_metric"
    assert rows[0]["metric"] == "placement_failed"
    assert rows[0]["subject_object"] == "blue_block"
