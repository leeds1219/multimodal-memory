# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GT failure detection system.

Tests the GTFailureDetector, subgoal→object mapping, and integration
with SubgoalBaseStrategy.
"""

import numpy as np
import pytest

from vlm_orchestrator.failure_handlers.gt_detector import (
    GTFailureDetector,
    GTFailureResult,
    GTFailureType,
    map_subgoal_to_objects,
)


# ── Helpers ──────────────────────────────────────────────────────────

def make_obj(pos=(0.3, 0.1, 0.02), disp=0.0, z_lift=0.0, max_lift=0.0, lifted=False):
    return {
        "pos": np.array(pos, dtype=np.float32),
        "quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "vel": np.zeros(6, dtype=np.float32),
        "displacement": np.float32(disp),
        "z_lift": np.float32(z_lift),
        "max_z_lift": np.float32(max_lift),
        "lifted": lifted,
    }


def make_gt_state(
    objects=None, grasped=None, gripper_width=0.0,
    conditions=None, score=0.0, scene_objects=None,
    object_completed=None, all_subtask_conditions=None,
    objects_in_contact=None,
):
    return {
        "objects": objects or {},
        "robot": {
            "ee_pos": np.array([0.3, 0.0, 0.4]),
            "ee_quat": np.array([1, 0, 0, 0]),
            "gripper_width": np.float32(gripper_width),
            "grasped_object": grasped,
            "objects_in_contact": objects_in_contact or [],
        },
        "subtask": {
            "completed": 0, "total": 1, "score": np.float32(score),
            "info": "",
            "conditions": conditions or [],
            "object_completed": object_completed or {},
            "all_subtask_conditions": all_subtask_conditions or {},
        },
        "scene_objects": scene_objects or ["red_hammer", "right_bin"],
    }


# ── Mapping tests ────────────────────────────────────────────────────

class TestSubgoalMapping:
    def test_tool_organization(self):
        t, c = map_subgoal_to_objects(
            "Pick up the red hammer and place it in the right bin",
            ["left_bin", "right_bin", "red_hammer", "husky_hammer",
             "cordless_drill", "spring_clamp", "table"],
        )
        assert "red_hammer" in t
        assert c == "right_bin"

    def test_spoons_in_pot(self):
        t, c = map_subgoal_to_objects(
            "Put serving spoons in the pot",
            ["green_serving_spoon", "red_serving_spoon", "ladle", "pot"],
        )
        assert c == "pot"
        # "serving spoon" matches the spoon objects
        assert len(t) >= 1

    def test_marker_in_mug(self):
        t, c = map_subgoal_to_objects(
            "Pick up the marker and put it in the mug",
            ["marker", "red_mug", "table"],
        )
        assert "marker" in t
        assert c == "red_mug"

    def test_fruits_on_plate(self):
        t, c = map_subgoal_to_objects(
            "Pick up the apple and place it on the plate",
            ["red_apple", "banana", "white_plate"],
        )
        assert c == "white_plate"

    def test_gt_conditions_override(self):
        """GT conditions should take priority over instruction parsing."""
        t, c = map_subgoal_to_objects(
            "Some vague instruction",
            ["obj_a", "obj_b", "container_x"],
            gt_conditions=[
                {"object": "obj_a", "satisfied": False},
                {"object": "obj_b", "satisfied": True},
            ],
        )
        # Unsatisfied conditions yield targets
        assert "obj_a" in t


# ── Detector tests ───────────────────────────────────────────────────

class TestGTFailureDetector:
    def setup_method(self):
        self.det = GTFailureDetector()
        self.det.set_subgoal(
            "Put red hammer in bin",
            scene_objects=["red_hammer", "right_bin", "cordless_drill"],
        )

    def _warmup(self, gt_state, n=10):
        for _ in range(n):
            self.det.update(gt_state)

    def test_wrong_object_picked(self):
        gt = make_gt_state(
            objects={"red_hammer": make_obj(), "cordless_drill": make_obj()},
            conditions=[{"object": "red_hammer", "satisfied": False}],
        )
        self._warmup(gt)
        gt["robot"]["grasped_object"] = "cordless_drill"
        gt["robot"]["gripper_width"] = np.float32(0.6)
        r = self.det.update(gt)
        assert r.failure_type == GTFailureType.WRONG_OBJECT_PICKED
        assert "cordless_drill" in r.reason
        assert r.grasped_object == "cordless_drill"
        assert not r.grasped_is_correct

    def test_wrong_object_suppressed_by_contact_list(self):
        """Primary contact is non-target, but target is in contact list.

        When the gripper touches both blocks, grasped_object reports
        one (non-target) but objects_in_contact includes both.  If the
        target appears in the contact list, suppress wrong_object_picked.
        """
        gt = make_gt_state(
            objects={
                "red_block": make_obj(),
                "blue_block": make_obj(),
            },
            grasped="red_block",
            gripper_width=0.6,
            # Both blocks in contact with gripper
            objects_in_contact=["red_block", "blue_block"],
            conditions=[
                {"object": "blue_block", "condition_idx": 0, "satisfied": False},
                {"object": "red_block", "condition_idx": 0, "satisfied": False},
            ],
        )
        self.det.set_subgoal(
            instruction="pick up the blue block",
            scene_objects=["red_block", "blue_block"],
            gt_conditions=gt["subtask"]["conditions"],
        )
        self._warmup(gt)
        r = self.det.update(gt)
        assert r.failure_type != GTFailureType.WRONG_OBJECT_PICKED, (
            f"False positive: reported {r.failure_type} with reason: {r.reason}"
        )

    def test_wrong_object_suppressed_by_csm_grabbed(self):
        """Primary contact is non-target, but CSM says target is grabbed.

        Fallback check: even if objects_in_contact is empty/missing,
        the CSM grabbed condition (condition 0) for the target is True.
        """
        gt = make_gt_state(
            objects={
                "red_block": make_obj(),
                "blue_block": make_obj(),
            },
            grasped="red_block",
            gripper_width=0.6,
            # No contact list (or empty) — test CSM fallback
            objects_in_contact=[],
            conditions=[
                # blue_block (target) is grabbed per CSM
                {"object": "blue_block", "condition_idx": 0, "satisfied": True},
                {"object": "blue_block", "condition_idx": 1, "satisfied": False},
                {"object": "blue_block", "condition_idx": 2, "satisfied": False},
                {"object": "blue_block", "condition_idx": 3, "satisfied": False},
                {"object": "red_block", "condition_idx": 0, "satisfied": False},
            ],
        )
        self.det.set_subgoal(
            instruction="pick up the blue block",
            scene_objects=["red_block", "blue_block"],
            gt_conditions=gt["subtask"]["conditions"],
        )
        self._warmup(gt)
        r = self.det.update(gt)
        assert r.failure_type != GTFailureType.WRONG_OBJECT_PICKED, (
            f"False positive: reported {r.failure_type} with reason: {r.reason}"
        )

    def test_object_dropped(self):
        """Dropped = CSM condition 0 (grabbed) was True, now False,
        and condition 3 (in_container) still False."""
        # Step 1: object is grabbed (condition 0 = True)
        gt = make_gt_state(
            objects={"red_hammer": make_obj(
                (0.3, 0.1, 0.15), disp=0.1, z_lift=0.13,
                max_lift=0.13, lifted=True,
            )},
            grasped="red_hammer", gripper_width=0.6,
            conditions=[
                {"object": "red_hammer", "condition_idx": 0, "satisfied": True},
                {"object": "red_hammer", "condition_idx": 1, "satisfied": True},
                {"object": "red_hammer", "condition_idx": 2, "satisfied": False},
                {"object": "red_hammer", "condition_idx": 3, "satisfied": False},
            ],
        )
        self._warmup(gt)

        # Step 2: object released — condition 0 becomes False
        gt["robot"]["grasped_object"] = None
        gt["robot"]["gripper_width"] = np.float32(0.0)
        gt["subtask"]["conditions"] = [
            {"object": "red_hammer", "condition_idx": 0, "satisfied": False},
            {"object": "red_hammer", "condition_idx": 1, "satisfied": False},
            {"object": "red_hammer", "condition_idx": 2, "satisfied": False},
            {"object": "red_hammer", "condition_idx": 3, "satisfied": False},
        ]
        r = self.det.update(gt)
        assert r.failure_type == GTFailureType.OBJECT_DROPPED

    def test_no_progress(self):
        """Score doesn't improve for NO_PROGRESS_PATIENCE_CHUNKS chunks → failure."""
        self.det.NO_PROGRESS_PATIENCE_CHUNKS = 5
        gt = make_gt_state(
            objects={"red_hammer": make_obj()},
            conditions=[
                {"object": "red_hammer", "condition_idx": 0, "satisfied": False},
            ],
        )
        for _ in range(20):
            r = self.det.update(gt)
        assert r.failure_type == GTFailureType.NO_PROGRESS

    def test_subgoal_complete(self):
        gt = make_gt_state(
            objects={"red_hammer": make_obj((0.5, 0, 0.03), disp=0.2)},
            score=1.0,
            object_completed={"red_hammer": True},
        )
        # Need COMPLETION_CONFIRM_CHUNKS consecutive frames for confirmation
        found_complete = False
        for _ in range(self.det.COMPLETION_CONFIRM_CHUNKS + 1):
            r = self.det.update(gt)
            if r.failure_type == GTFailureType.SUBGOAL_COMPLETE:
                found_complete = True
                break
        assert found_complete, (
            f"Expected SUBGOAL_COMPLETE within "
            f"{self.det.COMPLETION_CONFIRM_CHUNKS + 1} steps"
        )

    def test_in_progress_during_normal_operation(self):
        """Correct object grasped and moving → IN_PROGRESS."""
        gt = make_gt_state(
            objects={"red_hammer": make_obj(
                (0.3, 0.1, 0.15), disp=0.1, z_lift=0.13,
                max_lift=0.13, lifted=True,
            )},
            grasped="red_hammer", gripper_width=0.6,
            conditions=[{"object": "red_hammer", "satisfied": False}],
        )
        self._warmup(gt)
        r = self.det.update(gt)
        assert r.failure_type == GTFailureType.IN_PROGRESS

    def test_reset_clears_state(self):
        gt = make_gt_state(
            objects={"red_hammer": make_obj()},
            conditions=[{"object": "red_hammer", "satisfied": False}],
        )
        self._warmup(gt)
        self.det.reset()
        assert self.det._chunks_on_subgoal == 0
        assert self.det._target_objects == []

    def test_set_subgoal_resets_tracking(self):
        gt = make_gt_state(
            objects={"red_hammer": make_obj()},
            conditions=[{"object": "red_hammer", "satisfied": False}],
        )
        self._warmup(gt)
        self.det.set_subgoal(
            "Put husky hammer in bin",
            scene_objects=["husky_hammer", "right_bin"],
        )
        assert self.det._chunks_on_subgoal == 0
        assert "husky_hammer" in self.det._target_objects


# ── Regression tests ─────────────────────────────────────────────────

class TestSubtaskRegression:
    """Tests for SUBTASK_REGRESSION detection."""

    def setup_method(self):
        self.det = GTFailureDetector()
        # Use a small confirm count for tests
        self.det.REGRESSION_CONFIRM_CHUNKS = 3

    def _confirm(self, gt, n=None):
        """Feed gt_state enough times to confirm conditions."""
        n = n or self.det.REGRESSION_CONFIRM_CHUNKS + 1
        for _ in range(n):
            self.det.update(gt)

    def test_basic_regression(self):
        """Subtask condition satisfied then unsatisfied → SUBTASK_REGRESSION."""
        self.det.set_subgoal(
            "Pick up the blue block and place it on top of the red block",
            scene_objects=["red_block", "blue_block", "green_block"],
        )
        # subtask_0 (blue on red) becomes True — feed enough steps to confirm
        gt = make_gt_state(
            objects={"blue_block": make_obj(), "red_block": make_obj()},
            score=0.5,
            all_subtask_conditions={"subtask_0": True, "subtask_1": False, "subtask_2": False},
            scene_objects=["red_block", "blue_block", "green_block"],
        )
        self._confirm(gt)
        assert self.det._globally_completed.get("subtask_0") is True

        # Advance to subgoal 2
        self.det.set_subgoal(
            "Pick up the green block and place it on top of the blue block",
            scene_objects=["red_block", "blue_block", "green_block"],
        )

        # subtask_0 regresses (blue fell off red)
        gt2 = make_gt_state(
            objects={"blue_block": make_obj(), "red_block": make_obj()},
            score=0.0,
            all_subtask_conditions={"subtask_0": False, "subtask_1": False, "subtask_2": False},
            scene_objects=["red_block", "blue_block", "green_block"],
        )
        r = self.det.update(gt2)
        assert r.failure_type == GTFailureType.SUBTASK_REGRESSION
        assert "subtask_0" in r.reason
        assert "replan" in r.suggested_actions

    def test_no_regression_on_first_satisfied(self):
        """Condition going from False → True should not trigger regression."""
        self.det.set_subgoal(
            "Pick up the red hammer and place it in the bin",
            scene_objects=["red_hammer", "right_bin"],
        )
        gt = make_gt_state(
            all_subtask_conditions={"subtask_0": False},
            scene_objects=["red_hammer", "right_bin"],
        )
        r = self.det.update(gt)
        assert r.failure_type != GTFailureType.SUBTASK_REGRESSION

        gt["subtask"]["all_subtask_conditions"] = {"subtask_0": True}
        r = self.det.update(gt)
        assert r.failure_type != GTFailureType.SUBTASK_REGRESSION

    def test_regression_fires_during_cooldown(self):
        """Regression should be detected even during failure cooldown."""
        self.det.set_subgoal(
            "Pick up the blue block and stack it",
            scene_objects=["blue_block", "red_block"],
        )
        # subtask_0 satisfied — confirm it
        gt = make_gt_state(
            score=0.5,
            all_subtask_conditions={"subtask_0": True, "subtask_1": False},
            scene_objects=["blue_block", "red_block"],
        )
        self._confirm(gt)

        # Advance subgoal, then acknowledge (start cooldown)
        self.det.set_subgoal(
            "Pick up the red block",
            scene_objects=["blue_block", "red_block"],
        )
        self.det.acknowledge()
        assert self.det._cooldown_remaining > 0

        # Regression during cooldown
        gt2 = make_gt_state(
            score=0.0,
            all_subtask_conditions={"subtask_0": False, "subtask_1": False},
            scene_objects=["blue_block", "red_block"],
        )
        r = self.det.update(gt2)
        assert r.failure_type == GTFailureType.SUBTASK_REGRESSION

    def test_regression_does_not_fire_twice(self):
        """Same regression should only fire once (snapshot updates)."""
        self.det.set_subgoal(
            "task A", scene_objects=["obj_a"],
        )
        # subtask_0 satisfied — confirm it
        gt = make_gt_state(
            score=0.5,
            all_subtask_conditions={"subtask_0": True},
            scene_objects=["obj_a"],
        )
        self._confirm(gt)

        # Advance
        self.det.set_subgoal("task B", scene_objects=["obj_a", "obj_b"])

        # Regress
        gt2 = make_gt_state(
            score=0.0,
            all_subtask_conditions={"subtask_0": False},
            scene_objects=["obj_a", "obj_b"],
        )
        r = self.det.update(gt2)
        assert r.failure_type == GTFailureType.SUBTASK_REGRESSION

        # Second update with same state — should NOT fire again
        r = self.det.update(gt2)
        assert r.failure_type != GTFailureType.SUBTASK_REGRESSION

    def test_reset_clears_global_tracking(self):
        """Full reset should clear global completion tracking."""
        self.det.set_subgoal("task", scene_objects=["obj_a"])
        gt = make_gt_state(
            score=0.5,
            all_subtask_conditions={"subtask_0": True},
            scene_objects=["obj_a"],
        )
        self._confirm(gt)
        assert self.det._globally_completed.get("subtask_0") is True

        self.det.reset()
        assert self.det._globally_completed == {}

    def test_set_subgoal_preserves_global_tracking(self):
        """set_subgoal should NOT clear global completion tracking."""
        self.det.set_subgoal("task A", scene_objects=["obj_a"])
        gt = make_gt_state(
            score=0.5,
            all_subtask_conditions={"subtask_0": True},
            scene_objects=["obj_a"],
        )
        self._confirm(gt)

        self.det.set_subgoal("task B", scene_objects=["obj_a", "obj_b"])
        assert self.det._globally_completed.get("subtask_0") is True

    def test_multiple_subtasks_regress(self):
        """Multiple subtask conditions regressing at once."""
        self.det.set_subgoal("stack blocks", scene_objects=["a", "b", "c"])
        gt = make_gt_state(
            score=1.0,
            all_subtask_conditions={"subtask_0": True, "subtask_1": True, "subtask_2": False},
            scene_objects=["a", "b", "c"],
        )
        self._confirm(gt)

        self.det.set_subgoal("finish c", scene_objects=["a", "b", "c"])
        gt2 = make_gt_state(
            score=0.0,
            all_subtask_conditions={"subtask_0": False, "subtask_1": False, "subtask_2": False},
            scene_objects=["a", "b", "c"],
        )
        r = self.det.update(gt2)
        assert r.failure_type == GTFailureType.SUBTASK_REGRESSION
        assert "subtask_0" in r.reason
        assert "subtask_1" in r.reason

    def test_transient_flicker_does_not_trigger(self):
        """A brief True→False flicker should NOT trigger regression."""
        self.det.set_subgoal("task", scene_objects=["a"])
        # Condition True for only 1 step (below confirm threshold)
        gt_true = make_gt_state(
            all_subtask_conditions={"subtask_0": True},
            scene_objects=["a"],
        )
        self.det.update(gt_true)
        # Immediately False
        gt_false = make_gt_state(
            all_subtask_conditions={"subtask_0": False},
            scene_objects=["a"],
        )
        r = self.det.update(gt_false)
        assert r.failure_type != GTFailureType.SUBTASK_REGRESSION

    def test_is_failure_property_for_regression(self):
        r = GTFailureResult(failure_type=GTFailureType.SUBTASK_REGRESSION)
        assert r.is_failure is True


# ── Result tests ─────────────────────────────────────────────────────

class TestGTFailureResult:
    def test_is_failure_property(self):
        r = GTFailureResult(failure_type=GTFailureType.WRONG_OBJECT_PICKED)
        assert r.is_failure is True

        r = GTFailureResult(failure_type=GTFailureType.IN_PROGRESS)
        assert r.is_failure is False

        r = GTFailureResult(failure_type=GTFailureType.SUBGOAL_COMPLETE)
        assert r.is_failure is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
