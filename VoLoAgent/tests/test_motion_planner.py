# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the motion-planner abstraction (Part 2, step 2.1).

Guarantees:
* :class:`LinearInterpPlanner` reproduces
  :func:`vlm_orchestrator.grasp.ik.interpolate_joints` exactly (no behavior
  change vs. the historical path).
* The factory resolves ``linear`` and rejects unknown kinds.
* The result dataclass exposes loud-fail semantics (``success=False`` +
  ``waypoints=None``) as required by the "no silent lossy fallback" rule.
"""

import numpy as np
import pytest

from vlm_orchestrator.grasp.ik import interpolate_joints
from vlm_orchestrator.motion import (
    LinearInterpPlanner,
    MotionPlanner,
    MotionPlanResult,
    build_motion_planner,
)


def test_linear_planner_matches_interpolate_joints():
    planner = LinearInterpPlanner()
    q_start = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    q_end = np.array([0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3])

    for n in (2, 3, 8, 16, 40):
        result = planner.plan_segment(q_start, q_end, n_steps=n)
        ref = interpolate_joints(q_start, q_end, n)
        assert result.success is True
        assert result.label == "linear"
        assert len(result.waypoints) == len(ref)
        for a, b in zip(result.waypoints, ref):
            assert np.allclose(a, b)


def test_linear_planner_endpoints_exact():
    planner = LinearInterpPlanner()
    q_start = np.zeros(7)
    q_end = np.ones(7)
    wp = planner.plan_segment(q_start, q_end, n_steps=10).waypoints
    assert np.allclose(wp[0], q_start)
    assert np.allclose(wp[-1], q_end)


def test_linear_planner_degenerate_steps():
    # num_steps < 2 → single waypoint at q_end (matches interpolate_joints).
    planner = LinearInterpPlanner()
    q_start = np.zeros(7)
    q_end = np.arange(7, dtype=float)
    res = planner.plan_segment(q_start, q_end, n_steps=1)
    assert len(res.waypoints) == 1
    assert np.allclose(res.waypoints[0], q_end)


def test_linear_planner_ignores_scene_pc_and_phase():
    planner = LinearInterpPlanner()
    q_start = np.zeros(7)
    q_end = np.ones(7)
    scene = np.random.rand(100, 3)
    res = planner.plan_segment(
        q_start, q_end, n_steps=5, scene_pc=scene, phase="approach",
    )
    ref = interpolate_joints(q_start, q_end, 5)
    assert res.success
    for a, b in zip(res.waypoints, ref):
        assert np.allclose(a, b)


def test_factory_linear():
    p = build_motion_planner("linear")
    assert isinstance(p, LinearInterpPlanner)
    assert isinstance(p, MotionPlanner)  # runtime_checkable protocol
    assert p.name == "linear"


def test_factory_default_is_linear():
    assert isinstance(build_motion_planner(), LinearInterpPlanner)


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown motion planner"):
        build_motion_planner("nonsense")


class _FailingPlanner:
    """Planner that always fails — exercises the loud-fail path."""

    name = "always_fail"

    def plan_segment(self, q_start, q_end, *, n_steps, scene_pc=None, phase=None):
        return MotionPlanResult.failed("always_fail", "test-induced failure")


def test_grasp_executor_plan_segment_records_label():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    exec_ = GraspToolExecutor()
    exec_._grasp_log = {}
    wp = exec_._plan_segment(
        np.zeros(7), np.ones(7), 5, phase="approach",
    )
    assert len(wp) == 5
    assert exec_._grasp_log["ik_motion_planner_approach"] == "linear"


def test_grasp_executor_plan_segment_fails_loudly():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    exec_ = GraspToolExecutor(motion_planner=_FailingPlanner())
    exec_._grasp_log = {}
    with pytest.raises(RuntimeError, match="Motion planning failed"):
        exec_._plan_segment(np.zeros(7), np.ones(7), 5, phase="approach")
    # label still recorded before the raise, for observability
    assert exec_._grasp_log["ik_motion_planner_approach"] == "always_fail"


def test_place_executor_plan_segment_fails_loudly():
    from vlm_orchestrator.place.tool import PlaceToolExecutor

    exec_ = PlaceToolExecutor(motion_planner=_FailingPlanner())
    exec_._place_log = {}
    with pytest.raises(RuntimeError, match="Motion planning failed"):
        exec_._plan_segment(np.zeros(7), np.ones(7), 5, phase="final")
    assert exec_._place_log["motion_planner_final"] == "always_fail"


def test_curobo_remote_planner_success(monkeypatch):
    """CuroboRemotePlanner returns ok result when the client succeeds."""
    from vlm_orchestrator.motion.curobo_client import CuroboRemotePlanner

    planner = CuroboRemotePlanner.__new__(CuroboRemotePlanner)  # skip __init__

    class FakeClient:
        def plan_motion(self, q_start, q_end, *, n_steps=None, scene_pc=None, disable_fingers=False):
            return np.zeros((n_steps or 5, 7), dtype=np.float32), True

    planner._client = FakeClient()
    res = planner.plan_segment(
        np.zeros(7), np.ones(7), n_steps=8, phase="approach",
    )
    assert res.success is True
    assert res.label == "curobo"
    assert len(res.waypoints) == 8


def test_curobo_disables_fingers_on_approach_only():
    """The approach segment must disable finger collisions (reach to table);
    other phases keep full collision."""
    from vlm_orchestrator.motion.curobo_client import CuroboRemotePlanner

    planner = CuroboRemotePlanner.__new__(CuroboRemotePlanner)
    seen = {}

    class RecordingClient:
        def plan_motion(self, q_start, q_end, *, n_steps=None, scene_pc=None,
                        disable_fingers=False):
            seen[n_steps] = disable_fingers  # keyed per call
            return np.zeros((n_steps or 5, 7), dtype=np.float32), True

    planner._client = RecordingClient()
    # Fingers are only disabled on approach WHEN a scene cloud is present
    # (collision-free path). Plain-cuRobo (scene_pc=None) keeps full model.
    sc = np.zeros((4, 3), dtype=np.float32)
    planner.plan_segment(np.zeros(7), np.ones(7), n_steps=1, phase="approach",
                         scene_pc=sc)
    planner.plan_segment(np.zeros(7), np.ones(7), n_steps=2, phase="final",
                         scene_pc=sc)
    planner.plan_segment(np.zeros(7), np.ones(7), n_steps=3, phase="retreat",
                         scene_pc=sc)
    planner.plan_segment(np.zeros(7), np.ones(7), n_steps=4, phase="approach",
                         scene_pc=None)
    assert seen[1] is True   # approach + scene → fingers disabled
    assert seen[2] is False  # final → full collision
    assert seen[3] is False  # retreat → full collision
    assert seen[4] is False  # approach, no scene (plain cuRobo) → full model


def test_curobo_remote_planner_fails_loudly(monkeypatch):
    """Client failure → failed result with curobo_failed label (no fallback)."""
    from vlm_orchestrator.motion.curobo_client import CuroboRemotePlanner

    planner = CuroboRemotePlanner.__new__(CuroboRemotePlanner)

    class FailClient:
        def plan_motion(self, q_start, q_end, *, n_steps=None, scene_pc=None, disable_fingers=False):
            return None, False

    planner._client = FailClient()
    res = planner.plan_segment(np.zeros(7), np.ones(7), n_steps=8, phase="final")
    assert res.success is False
    assert res.waypoints is None
    assert res.label == "curobo_failed"


def test_curobo_remote_planner_handles_client_exception():
    from vlm_orchestrator.motion.curobo_client import CuroboRemotePlanner

    planner = CuroboRemotePlanner.__new__(CuroboRemotePlanner)

    class RaisingClient:
        def plan_motion(self, *a, **k):
            raise RuntimeError("connection refused")

    planner._client = RaisingClient()
    res = planner.plan_segment(np.zeros(7), np.ones(7), n_steps=8, phase="lift")
    assert res.success is False
    assert res.label == "curobo_failed"
    assert "connection refused" in res.detail


def test_result_ok_and_failed_helpers():
    wp = [np.zeros(7), np.ones(7)]
    ok = MotionPlanResult.ok(wp, "linear")
    assert ok.success is True
    assert ok.waypoints is wp
    assert ok.label == "linear"

    bad = MotionPlanResult.failed("curobo_failed", "no collision-free path")
    assert bad.success is False
    assert bad.waypoints is None
    assert bad.label == "curobo_failed"
    assert "collision-free" in bad.detail


# ---------------------------------------------------------------------------
# _build_obstacle_cloud (grasp tool): scene-minus-target obstacle for
# collision-aware planning.  Skips work for the linear planner; masks the
# target object out; subsamples to the cap; None on empty.
# ---------------------------------------------------------------------------


def test_build_obstacle_cloud_skips_for_linear():
    from vlm_orchestrator.grasp.tool import _build_obstacle_cloud
    from vlm_orchestrator.motion import LinearInterpPlanner

    pc = np.random.rand(100, 3).astype(np.float32)
    keep = np.ones(100, dtype=bool)
    # linear planner → no cloud built (free cost)
    assert _build_obstacle_cloud(pc, keep, LinearInterpPlanner()) is None


def test_build_obstacle_cloud_masks_and_returns_for_curobo(monkeypatch):
    import vlm_orchestrator.grasp.tool as gt
    from vlm_orchestrator.grasp.tool import _build_obstacle_cloud

    # Collision-free path is off by default (plain-cuRobo); enable for this test.
    monkeypatch.setattr(gt, "COLLISION_FREE_PLANNING", True)

    class FakeCurobo:
        name = "curobo"

    pc = np.arange(30, dtype=np.float32).reshape(10, 3)
    # obstacle mask keeps rows 0..4 (the "not-target" complement)
    keep = np.array([True] * 5 + [False] * 5)
    out = _build_obstacle_cloud(pc, keep, FakeCurobo())
    assert out is not None
    assert out.shape == (5, 3)
    assert np.allclose(out, pc[:5])


def test_build_obstacle_cloud_subsamples_to_cap(monkeypatch):
    import vlm_orchestrator.grasp.tool as gt
    from vlm_orchestrator.grasp.tool import _build_obstacle_cloud, MAX_OBSTACLE_POINTS

    monkeypatch.setattr(gt, "COLLISION_FREE_PLANNING", True)

    class FakeCurobo:
        name = "curobo"

    n = MAX_OBSTACLE_POINTS + 500
    pc = np.random.rand(n, 3).astype(np.float32)
    keep = np.ones(n, dtype=bool)
    out = _build_obstacle_cloud(pc, keep, FakeCurobo())
    assert out.shape == (MAX_OBSTACLE_POINTS, 3)


def test_build_obstacle_cloud_none_when_empty():
    from vlm_orchestrator.grasp.tool import _build_obstacle_cloud

    class FakeCurobo:
        name = "curobo"

    pc = np.random.rand(10, 3).astype(np.float32)
    keep = np.zeros(10, dtype=bool)  # nothing is an obstacle
    assert _build_obstacle_cloud(pc, keep, FakeCurobo()) is None
