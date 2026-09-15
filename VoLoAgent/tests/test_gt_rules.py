# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the stateless GT-rules module.

These tests exercise each ``check_*`` function on synthetic ``gt_state``
snapshots — no Isaac sim needed.  The synthetic shapes mirror what
robolab's ``GTStateExporter`` emits for ``pick_and_place`` composites.
"""

from __future__ import annotations

from vlm_orchestrator.diagnostics.metrics1_gt_rules import (
    COMPLETION_CONFIRM_CHUNKS,
    REGRESSION_CONFIRM_CHUNKS,
    STUCK_FIRE_CHUNKS,
    WRONG_TARGET_PLACE_CONFIRM_CHUNKS,
    Aspect1EventType,
    RuleState,
    build_obj_conds,
    check_object_complete,
    check_object_regression,
    check_recoveries,
    check_stuck,
    check_wrong_object_picked,
    check_wrong_target_place,
    current_subtask_targets,
    update_completed_set,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic gt_state builders
# ──────────────────────────────────────────────────────────────────────

def _ladder(obj: str, *, grabbed=False, above=False, dropped=False, in_cont=False):
    """Build the 4 standard pick_and_place condition rows for one object."""
    return [
        {"object": obj, "condition_idx": 0, "satisfied": grabbed, "info": ""},
        {"object": obj, "condition_idx": 1, "satisfied": above, "info": ""},
        {"object": obj, "condition_idx": 2, "satisfied": dropped, "info": ""},
        {"object": obj, "condition_idx": 3, "satisfied": in_cont, "info": ""},
    ]


def _gt(
    *,
    score: float = 0.0,
    conditions: list[dict] | None = None,
    object_completed: dict[str, bool] | None = None,
    all_subtask_conditions: dict[str, bool] | None = None,
    grasped: str | None = None,
    objects_in_contact: list[str] | None = None,
    completed: int = 0,
):
    return {
        "subtask": {
            "score": score,
            "conditions": conditions or [],
            "object_completed": object_completed or {},
            "all_subtask_conditions": all_subtask_conditions or {},
            "completed": completed,
        },
        "robot": {
            "grasped_object": grasped,
            "objects_in_contact": objects_in_contact or [],
        },
        "objects": {},
    }


# ──────────────────────────────────────────────────────────────────────
# Helpers: build_obj_conds + update_completed_set
# ──────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_build_obj_conds_basic(self):
        gt = _gt(conditions=_ladder("red_block", grabbed=True, dropped=True))
        obj_conds = build_obj_conds(gt)
        assert obj_conds == {
            "red_block": {0: True, 1: False, 2: True, 3: False}
        }

    def test_build_obj_conds_empty(self):
        gt = _gt()
        assert build_obj_conds(gt) == {}

    def test_build_obj_conds_multi_object(self):
        gt = _gt(conditions=_ladder("red", grabbed=True) + _ladder("blue", in_cont=True))
        obj_conds = build_obj_conds(gt)
        assert obj_conds["red"][0] is True
        assert obj_conds["blue"][3] is True

    def test_update_completed_set_filters_by_target(self):
        state = RuleState()
        gt = _gt(object_completed={"red": True, "blue": True, "lemon": True})
        update_completed_set(state, gt, target_set={"red", "blue"})
        assert state.completed_objects == {"red", "blue"}  # lemon excluded

    def test_update_completed_set_only_True_entries(self):
        state = RuleState()
        gt = _gt(object_completed={"red": True, "blue": False})
        update_completed_set(state, gt, target_set={"red", "blue"})
        assert state.completed_objects == {"red"}


# ──────────────────────────────────────────────────────────────────────
# check_object_complete
# ──────────────────────────────────────────────────────────────────────

class TestSubgoalComplete:

    def test_per_object_fires_after_streak(self):
        """Per-object emission: each target fires its own SUBGOAL_COMPLETE
        once its streak crosses ``COMPLETION_CONFIRM_CHUNKS``."""
        state = RuleState()
        gt = _gt(score=1.0, object_completed={"red": True, "blue": True})
        targets = ["red", "blue"]
        # Streak builds for COMPLETION_CONFIRM_CHUNKS - 1 ticks: no fire
        for _ in range(COMPLETION_CONFIRM_CHUNKS - 1):
            res = check_object_complete(state, gt, targets)
            assert res is None
        # Tick that crosses the threshold: fires for the first target
        # (sorted alphabetically — "blue" before "red")
        res = check_object_complete(state, gt, targets)
        assert res is not None
        assert res.event_type == Aspect1EventType.OBJECT_COMPLETE
        assert res.target_objects == ["blue"]
        assert "blue" in state.emitted_completions
        # Next tick fires for the second target ("red")
        res = check_object_complete(state, gt, targets)
        assert res is not None
        assert res.target_objects == ["red"]
        assert "red" in state.emitted_completions
        # Subsequent ticks: no more fires (already emitted)
        for _ in range(5):
            res = check_object_complete(state, gt, targets)
            assert res is None

    def test_only_completed_targets_fire(self):
        """Targets whose object_completed is False never fire."""
        state = RuleState()
        gt = _gt(object_completed={"red": True, "blue": False})
        targets = ["red", "blue"]
        # Build streak to cross threshold for "red" (which is True)
        results = []
        for _ in range(COMPLETION_CONFIRM_CHUNKS + 5):
            res = check_object_complete(state, gt, targets)
            if res is not None:
                results.append(res)
        # Exactly one fire — for "red"
        assert len(results) == 1
        assert results[0].target_objects == ["red"]
        # "blue" never reached the streak threshold
        assert "blue" not in state.emitted_completions
        assert state.completion_streak.get("blue", 0) == 0

    def test_streak_resets_when_target_uncompletes(self):
        """If object_completed is False, no fire and streak stays at 0."""
        state = RuleState()
        targets = ["red"]
        gt_undone = _gt(object_completed={"red": False})
        for _ in range(COMPLETION_CONFIRM_CHUNKS * 5):
            res = check_object_complete(state, gt_undone, targets)
            assert res is None
        assert state.completion_streak.get("red", 0) == 0
        assert "red" not in state.emitted_completions

    def test_no_fire_with_empty_targets(self):
        state = RuleState()
        gt = _gt(object_completed={"red": True})
        res = check_object_complete(state, gt, [])
        assert res is None

    def test_lh_a01_per_phase_emission(self):
        """LH_A01 episode-13 repro: even if the bottom cube never
        completes, the top + middle cubes should still fire individual
        SUBGOAL_COMPLETE events as their phases finish.

        Previously the rule waited for ALL three cubes to complete and
        stayed silent the whole episode.
        """
        state = RuleState()
        targets = ["rubiks_cube_top", "rubiks_cube_middle", "rubiks_cube_bottom"]

        # Phase 1 finishes: top is unstacked
        gt_phase1 = _gt(object_completed={
            "rubiks_cube_top": True,
            "rubiks_cube_middle": False,
            "rubiks_cube_bottom": False,
        })
        # Build streak for "top"
        results = []
        for _ in range(COMPLETION_CONFIRM_CHUNKS + 1):
            res = check_object_complete(state, gt_phase1, targets)
            if res is not None:
                results.append(res)
        assert len(results) == 1
        assert results[0].target_objects == ["rubiks_cube_top"]

        # Phase 2 finishes: middle also unstacked
        gt_phase2 = _gt(object_completed={
            "rubiks_cube_top": True,
            "rubiks_cube_middle": True,
            "rubiks_cube_bottom": False,
        })
        results = []
        for _ in range(COMPLETION_CONFIRM_CHUNKS + 1):
            res = check_object_complete(state, gt_phase2, targets)
            if res is not None:
                results.append(res)
        assert len(results) == 1
        assert results[0].target_objects == ["rubiks_cube_middle"]

        # Bottom never completes — no further fires for the rest of the
        # episode.
        gt_phase3_stuck = gt_phase2
        for _ in range(COMPLETION_CONFIRM_CHUNKS * 5):
            res = check_object_complete(state, gt_phase3_stuck, targets)
            assert res is None

        # Summary: 2 events fired, "bottom" never completed
        assert state.emitted_completions == {"rubiks_cube_top", "rubiks_cube_middle"}

    def test_fires_immediately_on_first_true(self):
        """With COMPLETION_CONFIRM_CHUNKS=1, OBJECT_COMPLETE fires on
        the very first tick where object_completed[obj] is True — no
        artificial logger-side stability delay (the CSM's own
        completion check already gates on stable physics state).
        """
        state = RuleState()
        targets = ["red_block"]
        gt = _gt(object_completed={"red_block": True})
        res = check_object_complete(state, gt, targets)
        assert res is not None
        assert res.event_type == Aspect1EventType.OBJECT_COMPLETE
        assert res.target_objects == ["red_block"]


# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# check_wrong_object_picked
# ──────────────────────────────────────────────────────────────────────

class TestWrongObjectPicked:

    def test_fires_on_non_target(self):
        state = RuleState()
        gt = _gt(grasped="apple", objects_in_contact=["apple"])
        res = check_wrong_object_picked(state, gt, ["banana"], set(), {})
        assert res is not None
        assert res.event_type == Aspect1EventType.WRONG_OBJECT_PICKED
        assert res.grasped_object == "apple"
        assert state.outstanding_wrong_object == "apple"

    def test_no_fire_if_grasped_is_target(self):
        state = RuleState()
        gt = _gt(grasped="banana", objects_in_contact=["banana"])
        res = check_wrong_object_picked(state, gt, ["banana"], set(), {})
        assert res is None

    def test_no_fire_if_target_in_contact_list(self):
        """Suppression 1: target also in contact → grasped is just primary contact."""
        state = RuleState()
        gt = _gt(grasped="apple", objects_in_contact=["apple", "banana"])
        res = check_wrong_object_picked(state, gt, ["banana"], set(), {})
        assert res is None

    def test_fires_even_if_target_csm_cond0_latched(self):
        """CSM ``cond_idx 0`` is latched True after the target was
        grasped at any point.  The rule must NOT use it as a current-
        grasp suppression — only ``objects_in_contact`` (current-state)
        suppresses.  Otherwise, after the agent grasps the target
        once and releases, every subsequent wrong-object grasp would
        be silently suppressed.
        """
        state = RuleState()
        # Agent currently holds apple (the wrong thing).  The target
        # banana is NOT in contact right now (released earlier).
        # CSM cond 0 for banana is latched True from the earlier grasp.
        gt = _gt(grasped="apple", objects_in_contact=["apple"])
        obj_conds = {"banana": {0: True}}
        res = check_wrong_object_picked(state, gt, ["banana"], set(), obj_conds)
        assert res is not None
        assert res.event_type == Aspect1EventType.WRONG_OBJECT_PICKED
        assert res.grasped_object == "apple"

    def test_completed_targets_not_considered(self):
        state = RuleState()
        gt = _gt(grasped="apple", objects_in_contact=["apple", "banana"])
        # banana is completed → suppression 1 should still apply on remaining targets
        res = check_wrong_object_picked(state, gt, ["banana", "lemon"], {"banana"}, {})
        # only lemon is remaining, lemon not in contact → fires
        assert res is not None
        assert "lemon" in res.reason

    def test_no_fire_with_no_grasped(self):
        state = RuleState()
        gt = _gt(grasped=None)
        res = check_wrong_object_picked(state, gt, ["banana"], set(), {})
        assert res is None

    def test_dedup_same_wrong_object(self):
        """Same wrong object grasped on subsequent ticks shouldn't re-fire."""
        state = RuleState()
        gt = _gt(grasped="apple", objects_in_contact=["apple"])
        first = check_wrong_object_picked(state, gt, ["banana"], set(), {})
        assert first is not None  # first fire
        # Subsequent calls with the same wrong-grasp don't re-fire
        for _ in range(10):
            res = check_wrong_object_picked(state, gt, ["banana"], set(), {})
            assert res is None
        assert state.outstanding_wrong_object == "apple"


# ──────────────────────────────────────────────────────────────────────
# check_wrong_target_place — the new rule replacing OBJECT_DROPPED
# ──────────────────────────────────────────────────────────────────────

class TestWrongTargetPlace:

    def test_fires_after_streak_when_released_outside_target(self):
        state = RuleState()
        targets = ["red"]
        # red has been released and is NOT in container — stable state
        obj_conds = {"red": {0: False, 1: True, 2: True, 3: False}}
        gt = _gt(conditions=_ladder("red", above=True, dropped=True))
        for i in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS - 1):
            res = check_wrong_target_place(state, gt, targets, set(), obj_conds)
            assert res is None
            assert state.released_streak["red"] == i + 1
        res = check_wrong_target_place(state, gt, targets, set(), obj_conds)
        assert res is not None
        assert res.event_type == Aspect1EventType.WRONG_TARGET_PLACE
        # Streak resets after fire (avoids re-firing every tick)
        assert state.released_streak["red"] == 0

    def test_no_fire_for_transient_release(self):
        """Object released for fewer than streak chunks → no fire."""
        state = RuleState()
        targets = ["red"]
        obj_conds = {"red": {0: False, 1: True, 2: True, 3: False}}
        gt = _gt()
        # Release for streak-1 chunks, then re-grab
        for _ in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS - 1):
            check_wrong_target_place(state, gt, targets, set(), obj_conds)
        # Re-grab
        obj_conds = {"red": {0: True, 1: True, 2: False, 3: False}}
        res = check_wrong_target_place(state, gt, targets, set(), obj_conds)
        assert res is None
        # Streak reset
        assert state.released_streak["red"] == 0

    def test_no_fire_when_in_container(self):
        """Object dropped INTO target → cond 3 True → no fire."""
        state = RuleState()
        targets = ["red"]
        obj_conds = {"red": {0: False, 1: True, 2: True, 3: True}}
        gt = _gt()
        for _ in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS * 2):
            res = check_wrong_target_place(state, gt, targets, set(), obj_conds)
            assert res is None

    def test_no_fire_for_completed_objects(self):
        state = RuleState()
        targets = ["red"]
        # Completed targets are skipped even if their cond happens to look like a wrong-place
        obj_conds = {"red": {0: False, 2: True, 3: False}}
        gt = _gt()
        for _ in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS * 2):
            res = check_wrong_target_place(
                state, gt, targets, completed_objects={"red"}, obj_conds=obj_conds,
            )
            assert res is None

    def test_no_fire_when_still_grabbed(self):
        """Object currently held by gripper → not stably released → no fire.

        Uses ``robot.grasped_object`` (current state) rather than CSM
        cond 0 (which is latched once True and stays True even after
        the agent releases).
        """
        state = RuleState()
        targets = ["red"]
        # Cond 2 (released) latched True — the agent did release at
        # some point — but cond 3 (in_container) False.  The CSM table
        # is sticky, so cond 2 stays True even while the agent has
        # re-grasped the object.  Only ``grasped_object`` reflects the
        # current-tick reality.
        obj_conds = {"red": {0: True, 2: True, 3: False}}
        gt = _gt(grasped="red")
        for _ in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS * 2):
            res = check_wrong_target_place(state, gt, targets, set(), obj_conds)
            assert res is None

    def test_independent_streaks_per_object(self):
        """Each target object has its own streak; one firing doesn't reset another."""
        state = RuleState()
        targets = ["red", "blue"]
        # red is wrong-placed, blue is in container
        obj_conds = {
            "red": {0: False, 2: True, 3: False},
            "blue": {0: False, 2: True, 3: True},
        }
        gt = _gt()
        for _ in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS):
            res = check_wrong_target_place(state, gt, targets, set(), obj_conds)
        # Should fire for red, not blue
        assert res is not None
        assert "red" in res.reason


# ──────────────────────────────────────────────────────────────────────
# current_subtask_targets helper
# ──────────────────────────────────────────────────────────────────────

class TestCurrentSubtaskTargets:

    def test_extracts_unique_objects(self):
        gt = _gt(conditions=_ladder("red", grabbed=True) + _ladder("blue"))
        assert current_subtask_targets(gt) == ["red", "blue"]

    def test_dedups(self):
        # 4 cond rows for one object → only one entry returned.
        gt = _gt(conditions=_ladder("red"))
        assert current_subtask_targets(gt) == ["red"]

    def test_empty_when_no_conditions(self):
        gt = _gt()
        assert current_subtask_targets(gt) == []


# ──────────────────────────────────────────────────────────────────────
# OBJECT_REGRESSION
# ──────────────────────────────────────────────────────────────────────

class TestObjectRegression:

    def test_fire_after_streak(self):
        """Previously emitted-complete object's flag flips False
        for ``REGRESSION_CONFIRM_CHUNKS`` consecutive ticks → fires."""
        state = RuleState()
        state.emitted_completions.add("red")
        state.completed_objects.add("red")
        gt = _gt(object_completed={"red": False})
        for i in range(REGRESSION_CONFIRM_CHUNKS - 1):
            res = check_object_regression(state, gt)
            assert res is None
        res = check_object_regression(state, gt)
        assert res is not None
        assert res.event_type == Aspect1EventType.OBJECT_REGRESSION
        assert "red" in res.reason
        # Removed from completed_objects + emitted_completions so it
        # can re-emit SUBGOAL_COMPLETE if it recovers.
        assert "red" not in state.completed_objects
        assert "red" not in state.emitted_completions
        assert "red" in state.outstanding_object_regression

    def test_no_fire_if_not_emitted(self):
        """Object that never reached SUBGOAL_COMPLETE doesn't trigger
        OBJECT_REGRESSION."""
        state = RuleState()
        gt = _gt(object_completed={"red": False})
        for _ in range(REGRESSION_CONFIRM_CHUNKS * 2):
            assert check_object_regression(state, gt) is None

    def test_no_fire_if_still_complete(self):
        """Streak resets when object_completed is True again."""
        state = RuleState()
        state.emitted_completions.add("red")
        # Two ticks False, then True → streak should reset.
        for tick in range(REGRESSION_CONFIRM_CHUNKS - 1):
            check_object_regression(state, _gt(object_completed={"red": False}))
        # Recovery flip → reset
        check_object_regression(state, _gt(object_completed={"red": True}))
        assert state.object_regression_streak.get("red", 0) == 0

    def test_recovery_after_regression(self):
        """After OBJECT_REGRESSION fires, returning to target triggers
        a RECOVERY event."""
        state = RuleState()
        state.emitted_completions.add("red")
        # Trigger regression
        for _ in range(REGRESSION_CONFIRM_CHUNKS):
            check_object_regression(state, _gt(object_completed={"red": False}))
        assert "red" in state.outstanding_object_regression
        # Now the object is back at target — recovery should fire.
        res = check_recoveries(
            state, _gt(object_completed={"red": True}), obj_conds={},
        )
        assert res is not None
        assert res.event_type == Aspect1EventType.RECOVERY
        assert "OBJECT_REGRESSION recovered" in res.reason
        assert "red" not in state.outstanding_object_regression

    def test_no_fire_when_object_is_current_target(self):
        """Cross-subtask re-handling: an object placed by an earlier
        subtask gets lifted off when a later subtask's target is to
        move it elsewhere.  The exporter shows object_completed=False
        for the prior subtask's predicate, but the object is in the
        active subtask's target list — that is intentional movement,
        not regression."""
        state = RuleState()
        state.emitted_completions.add("red")
        state.completed_objects.add("red")
        gt = _gt(object_completed={"red": False})
        # Even after many ticks, regression must not fire while
        # "red" is in the current subtask's target list.
        for _ in range(REGRESSION_CONFIRM_CHUNKS * 3):
            res = check_object_regression(
                state, gt, current_targets=["red"],
            )
            assert res is None
        # Streak stays at 0 — not even accumulating ticks.
        assert state.object_regression_streak.get("red", 0) == 0
        assert "red" not in state.outstanding_object_regression
        # As soon as "red" stops being a current target (e.g. SSM
        # advanced past the re-handling subtask) and is still off
        # target, normal regression detection resumes.
        for _ in range(REGRESSION_CONFIRM_CHUNKS - 1):
            assert check_object_regression(state, gt) is None
        res = check_object_regression(state, gt)
        assert res is not None
        assert res.event_type == Aspect1EventType.OBJECT_REGRESSION


# ──────────────────────────────────────────────────────────────────────
# STUCK
# ──────────────────────────────────────────────────────────────────────

class TestStuck:

    def test_fires_after_no_progress_streak(self):
        """No score change, no completions → STUCK fires after
        ``STUCK_FIRE_CHUNKS`` chunks."""
        state = RuleState()
        gt = _gt(score=0.0, object_completed={"a": False, "b": False})
        for _ in range(STUCK_FIRE_CHUNKS - 1):
            res = check_stuck(state, gt)
            assert res is None
        res = check_stuck(state, gt)
        assert res is not None
        assert res.event_type == Aspect1EventType.STUCK
        assert state.outstanding_stuck is True
        # Streak resets to 0 after firing — re-fires every N more.
        assert state.stuck_streak == 0

    def test_re_fires_periodically_while_stuck(self):
        """Sparse heartbeat: STUCK re-fires every ``STUCK_FIRE_CHUNKS``
        chunks, not every chunk."""
        state = RuleState()
        gt = _gt(score=0.0, object_completed={"a": False})
        # First fire
        for _ in range(STUCK_FIRE_CHUNKS):
            check_stuck(state, gt)
        assert state.outstanding_stuck is True
        # Next STUCK_FIRE_CHUNKS - 1 ticks: no fire
        for _ in range(STUCK_FIRE_CHUNKS - 1):
            res = check_stuck(state, gt)
            assert res is None
        # The next tick fires again
        res = check_stuck(state, gt)
        assert res is not None
        assert res.event_type == Aspect1EventType.STUCK

    def test_no_fire_on_score_progress(self):
        """Score increasing → progress → no STUCK fires."""
        state = RuleState()
        # Climb score steadily — should never fire stuck.
        for i in range(STUCK_FIRE_CHUNKS * 3):
            gt = _gt(score=0.001 * (i + 1))
            res = check_stuck(state, gt)
            assert res is None
        assert state.outstanding_stuck is False

    def test_no_fire_on_completion_progress(self):
        """A new object_completed flipping True counts as progress."""
        state = RuleState()
        gt_no_progress = _gt(score=0.0, object_completed={"a": False, "b": False})
        for _ in range(STUCK_FIRE_CHUNKS - 1):
            check_stuck(state, gt_no_progress)
        # Just before fire, "a" completes
        gt_progress = _gt(score=0.0, object_completed={"a": True, "b": False})
        res = check_stuck(state, gt_progress)
        assert res is None
        assert state.stuck_streak == 0
        assert state.last_progress_completions == 1

    def test_other_event_fired_resets_streak(self):
        """When the caller signals another failure fired this tick,
        the stuck timer resets — the policy is doing something even
        if wrong, not stuck."""
        state = RuleState()
        gt = _gt(score=0.0, object_completed={"a": False})
        for _ in range(STUCK_FIRE_CHUNKS - 1):
            check_stuck(state, gt)
        assert state.stuck_streak == STUCK_FIRE_CHUNKS - 1
        # A failure event fired — reset the timer.
        res = check_stuck(state, gt, other_event_fired=True)
        assert res is None
        assert state.stuck_streak == 0
        # Subsequent ticks need a full new streak.
        for _ in range(STUCK_FIRE_CHUNKS - 1):
            assert check_stuck(state, gt) is None
        res = check_stuck(state, gt)
        assert res is not None and res.event_type == Aspect1EventType.STUCK

    def test_recovery_on_progress(self):
        """After STUCK fires, observed progress emits RECOVERY."""
        state = RuleState()
        gt = _gt(score=0.0, object_completed={"a": False})
        for _ in range(STUCK_FIRE_CHUNKS):
            check_stuck(state, gt)
        assert state.outstanding_stuck is True
        # Score climbs — recovery fires.
        gt_progress = _gt(score=0.5, object_completed={"a": False})
        res = check_stuck(state, gt_progress)
        assert res is not None
        assert res.event_type == Aspect1EventType.RECOVERY
        assert "STUCK recovered" in res.reason
        assert state.outstanding_stuck is False

    def test_no_recovery_if_never_fired(self):
        """Progress without a prior STUCK doesn't emit a RECOVERY."""
        state = RuleState()
        gt = _gt(score=0.5)
        res = check_stuck(state, gt)
        assert res is None
        assert state.outstanding_stuck is False

    def test_no_fire_when_holding_target(self):
        """Holding a current-subtask target counts as progress, even
        if score / completions are flat — the policy is mid-pick."""
        state = RuleState()
        # Score and completions never change, but the gripper has held
        # the target the whole time.  STUCK must not fire.
        for _ in range(STUCK_FIRE_CHUNKS * 2):
            gt = _gt(
                score=0.0,
                object_completed={"red_block": False},
                grasped="red_block",
            )
            res = check_stuck(state, gt, target_objects=["red_block"])
            assert res is None
        assert state.outstanding_stuck is False
        assert state.stuck_streak == 0

    def test_holding_non_target_does_not_count_as_progress(self):
        """Holding an object that is NOT a current target doesn't
        suppress STUCK — that's the wrong-object-picked case."""
        state = RuleState()
        gt = _gt(
            score=0.0,
            object_completed={"red_block": False},
            grasped="distractor",
        )
        for _ in range(STUCK_FIRE_CHUNKS - 1):
            assert check_stuck(state, gt, target_objects=["red_block"]) is None
        res = check_stuck(state, gt, target_objects=["red_block"])
        assert res is not None and res.event_type == Aspect1EventType.STUCK

    def test_recovery_via_holding_target(self):
        """Outstanding STUCK + gripper newly grasps a target → RECOVERY
        fires even though the score formula hasn't moved yet."""
        state = RuleState()
        gt_idle = _gt(score=0.0, object_completed={"red_block": False})
        for _ in range(STUCK_FIRE_CHUNKS):
            check_stuck(state, gt_idle, target_objects=["red_block"])
        assert state.outstanding_stuck is True
        gt_holding = _gt(
            score=0.0,
            object_completed={"red_block": False},
            grasped="red_block",
        )
        res = check_stuck(state, gt_holding, target_objects=["red_block"])
        assert res is not None
        assert res.event_type == Aspect1EventType.RECOVERY
        assert "holds target" in res.reason
        assert state.outstanding_stuck is False


# ──────────────────────────────────────────────────────────────────────
# Integration: a happy-path P&P trajectory across all rules
# ──────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_happy_path_no_failures_then_complete(self):
        """A successful single-object P&P should produce no failures, then SUBGOAL_COMPLETE."""
        state = RuleState()
        targets = ["red"]

        # Phase 1: warming up (nothing happening)
        for _ in range(5):
            gt = _gt(score=0.0, conditions=_ladder("red"))
            obj_conds = build_obj_conds(gt)
            assert check_wrong_object_picked(state, gt, targets, set(), obj_conds) is None
            assert check_wrong_target_place(state, gt, targets, set(), obj_conds) is None
            assert check_object_complete(state, gt, targets) is None
            assert check_object_regression(state, gt) is None

        # Phase 2: red gets grabbed, score climbs to 0.25
        for _ in range(3):
            gt = _gt(
                score=0.25,
                conditions=_ladder("red", grabbed=True),
                grasped="red", objects_in_contact=["red"],
            )
            obj_conds = build_obj_conds(gt)
            assert check_wrong_object_picked(state, gt, targets, set(), obj_conds) is None
            assert check_wrong_target_place(state, gt, targets, set(), obj_conds) is None

        # Phase 3: red lands in container, score 1.0, object_completed[red] = True
        for _ in range(COMPLETION_CONFIRM_CHUNKS):
            gt = _gt(
                score=1.0,
                conditions=_ladder("red", above=True, dropped=True, in_cont=True),
                object_completed={"red": True},
                all_subtask_conditions={"subtask_0": True},
            )
            res = check_object_complete(state, gt, targets)
            obj_conds = build_obj_conds(gt)
            # Wrong-target-place doesn't fire (in_container is True)
            assert check_wrong_target_place(state, gt, targets, state.completed_objects, obj_conds) is None
        # Final assertion: subgoal complete fired in the loop above
        assert res is not None
        assert res.event_type == Aspect1EventType.OBJECT_COMPLETE


# ──────────────────────────────────────────────────────────────────────
# check_recoveries
# ──────────────────────────────────────────────────────────────────────

class TestRecoveries:

    def test_no_outstanding_no_recovery(self):
        """Empty outstanding state → no recovery event."""
        state = RuleState()
        gt = _gt()
        assert check_recoveries(state, gt, {}) is None

    def test_wrong_object_recovery_requires_target_grasp(self):
        """Releasing the wrong object alone is NOT a recovery — the
        agent must actually grab a target.  Strict semantic kicks in
        when ``target_objects`` is provided to ``check_recoveries``.
        """
        state = RuleState()
        # Fire a wrong-object failure
        gt_wrong = _gt(grasped="apple", objects_in_contact=["apple"])
        check_wrong_object_picked(state, gt_wrong, ["banana"], set(), {})
        assert state.outstanding_wrong_object == "apple"
        # Release the wrong object — NOT a recovery (still hasn't
        # grabbed a target).
        gt_released = _gt(grasped=None)
        res = check_recoveries(state, gt_released, {}, target_objects=["banana"])
        assert res is None
        assert state.outstanding_wrong_object == "apple"
        # Grab another wrong object — still not a recovery.
        gt_other_wrong = _gt(grasped="lemon", objects_in_contact=["lemon"])
        res = check_recoveries(state, gt_other_wrong, {}, target_objects=["banana"])
        assert res is None
        # Now grab the target — recovery fires.
        gt_correct = _gt(grasped="banana", objects_in_contact=["banana"])
        res = check_recoveries(state, gt_correct, {}, target_objects=["banana"])
        assert res is not None
        assert res.event_type == Aspect1EventType.RECOVERY
        assert "WRONG_OBJECT_PICKED" in res.reason
        assert "banana" in res.reason
        assert state.outstanding_wrong_object is None

    def test_wrong_object_picked_multiple_then_recovery(self):
        """Multiple distinct wrong-grasps produce a queue of
        WRONG_OBJECT_PICKED events; a single RECOVERY fires when the
        agent finally grabs a target."""
        state = RuleState()
        targets = ["banana"]
        # 1. Grab apple — fires WRONG.
        gt = _gt(grasped="apple", objects_in_contact=["apple"])
        res = check_wrong_object_picked(state, gt, targets, set(), {})
        assert res is not None and res.event_type == Aspect1EventType.WRONG_OBJECT_PICKED
        assert state.outstanding_wrong_object == "apple"
        # 2. Release — no recovery (not holding target).
        gt = _gt(grasped=None)
        assert check_recoveries(state, gt, {}, target_objects=targets) is None
        assert state.outstanding_wrong_object == "apple"
        # 3. Grab lemon — different wrong object, fires WRONG again.
        gt = _gt(grasped="lemon", objects_in_contact=["lemon"])
        res = check_wrong_object_picked(state, gt, targets, set(), {})
        assert res is not None and res.event_type == Aspect1EventType.WRONG_OBJECT_PICKED
        assert state.outstanding_wrong_object == "lemon"
        # 4. Finally grab banana (target) — RECOVERY fires.
        gt = _gt(grasped="banana", objects_in_contact=["banana"])
        res = check_recoveries(state, gt, {}, target_objects=targets)
        assert res is not None and res.event_type == Aspect1EventType.RECOVERY
        assert state.outstanding_wrong_object is None

    def test_wrong_object_recovery_loose_fallback(self):
        """When ``target_objects`` is omitted, falls back to the loose
        "no longer holds wrong object" semantic for backward
        compatibility.
        """
        state = RuleState()
        gt_wrong = _gt(grasped="apple", objects_in_contact=["apple"])
        check_wrong_object_picked(state, gt_wrong, ["banana"], set(), {})
        # Release alone counts under loose semantic.
        gt_released = _gt(grasped=None)
        res = check_recoveries(state, gt_released, {})  # no target_objects
        assert res is not None
        assert res.event_type == Aspect1EventType.RECOVERY
        assert state.outstanding_wrong_object is None

    def test_wrong_target_place_recovery_on_in_container(self):
        """Misplaced object → reaches target container → RECOVERY fires."""
        state = RuleState()
        # Fire a WRONG_TARGET_PLACE for "red"
        obj_conds_misplaced = {"red": {0: False, 1: True, 2: True, 3: False}}
        gt = _gt()
        for _ in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS):
            check_wrong_target_place(state, gt, ["red"], set(), obj_conds_misplaced)
        assert "red" in state.outstanding_wrong_target_place
        # Now red is in the target container
        obj_conds_done = {"red": {0: False, 2: True, 3: True}}
        res = check_recoveries(state, gt, obj_conds_done)
        assert res is not None
        assert res.event_type == Aspect1EventType.RECOVERY
        assert "WRONG_TARGET_PLACE" in res.reason
        assert "red" in res.reason
        assert "red" not in state.outstanding_wrong_target_place

    def test_recovery_priority_order(self):
        """When multiple outstanding failures resolve same tick, the
        function returns one per call (wrong-object first, then
        wrong-target-place, then regression keys, then object regression).
        """
        state = RuleState()
        # Set up two outstanding failures
        state.outstanding_wrong_object = "apple"
        state.outstanding_wrong_target_place.add("red")
        gt = _gt(grasped=None)  # wrong-object recovers
        obj_conds = {"red": {3: True}}  # wrong-target-place recovers
        # First call: wrong-object recovery (priority 1)
        res = check_recoveries(state, gt, obj_conds)
        assert res is not None
        assert "WRONG_OBJECT_PICKED" in res.reason
        # Second call: wrong-target-place recovery (priority 2)
        res = check_recoveries(state, gt, obj_conds)
        assert res is not None
        assert "WRONG_TARGET_PLACE" in res.reason
