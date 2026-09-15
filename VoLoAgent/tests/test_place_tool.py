# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the place-with-tool pipeline.

Covers:
* Destination resolver (gt_sim path + raycast Z-adjustment).
* PerceptionFailure on missing inputs / bad depth.
* PlaceToolExecutor end-to-end with target_point_3d_world.
* Failure-reason propagation when gt_sim has no gt_state.
* Mock executor lifecycle (proxy-bypass simulation).
"""

from __future__ import annotations

import numpy as np
import pytest

from vlm_orchestrator.grasp.camera import CameraIntrinsics
from vlm_orchestrator.place import (
    DestinationSpec,
    PlacePhase,
    PlaceSegMode,
    PlaceToolExecutor,
)
from vlm_orchestrator.place.destination import (
    PerceptionFailure,
    PointToPlace2D,
    point_to_place_2d,
    raycast_2d_to_3d,
)
from vlm_orchestrator.place.mock import MockPlaceToolExecutor
from vlm_orchestrator.strategies.base import SessionState


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480,
    )


def _cam_to_world() -> np.ndarray:
    """Camera 1m above the table, looking straight down."""
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 1.0]
    T[:3, :3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    return T


def _gt_state() -> dict:
    return {
        "objects": {
            "red_bowl": {
                "pos": [0.5, 0.0, 0.05],
                "aabb_lower": [0.45, -0.05, 0.0],
                "aabb_upper": [0.55, 0.05, 0.10],
            },
        },
    }


def _fake_obs(*, gripper_closed: bool = True, with_depth: bool = True) -> dict:
    joints = np.array([0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.7], dtype=np.float64)
    obs = {
        "observation/joint_position": joints,
        "observation/gripper_position": np.array(
            [0.005 if gripper_closed else 0.08],
        ),
        "observation/ee_pos": np.array([0.4, 0.0, 0.5], dtype=np.float64),
        "observation/exterior_image_1_left": np.zeros(
            (480, 640, 3), dtype=np.uint8,
        ),
    }
    if with_depth:
        obs["observation/depth_external"] = np.full(
            (480, 640), 0.95, dtype=np.float32,
        )
        obs["observation/camera_pos"] = np.array([0.5, 0.0, 1.0])
        obs["observation/camera_quat"] = np.array([0.0, 1.0, 0.0, 0.0])
    return obs


# ----------------------------------------------------------------------
# Destination resolver — gt_sim
# ----------------------------------------------------------------------


def test_gt_sim_projects_object_to_image_centre():
    pt = point_to_place_2d(
        image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        instruction="put the cube in the red bowl",
        target_phrase="red_bowl",
        seg_mode="gt_sim",
        gt_state=_gt_state(),
        intrinsics=_intrinsics(),
        cam_to_world=_cam_to_world(),
    )
    assert pt.source == "gt_sim"
    assert abs(pt.x_norm - 0.5) < 0.05
    assert abs(pt.y_norm - 0.5) < 0.05


def test_gt_sim_missing_object_raises():
    with pytest.raises(PerceptionFailure):
        point_to_place_2d(
            image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            instruction="...",
            target_phrase="nonexistent_thing",
            seg_mode="gt_sim",
            gt_state=_gt_state(),
            intrinsics=_intrinsics(),
            cam_to_world=_cam_to_world(),
        )


def test_gt_sim_without_gt_state_raises():
    with pytest.raises(PerceptionFailure):
        point_to_place_2d(
            image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            instruction="...",
            target_phrase="red_bowl",
            seg_mode="gt_sim",
            gt_state=None,
            intrinsics=_intrinsics(),
            cam_to_world=_cam_to_world(),
        )


# ----------------------------------------------------------------------
# Raycast Z-adjustment
# ----------------------------------------------------------------------


def test_raycast_pre_resolved_3d_short_circuit():
    """Pre-resolved 3D returns the OBJECT centre = anchor + half_h + clearance.
    No flange offset (that's now applied in the trajectory planner)."""
    pt = PointToPlace2D(x_norm=0.5, y_norm=0.5, confidence=1.0, source="vlm")
    resolved = raycast_2d_to_3d(
        pt,
        depth=np.zeros((480, 640)),
        intrinsics=_intrinsics(),
        cam_to_world=_cam_to_world(),
        relation="in",
        held_object_height_m=0.04,
        pre_resolved_3d=np.array([0.5, 0.0, 0.05]),
    )
    expected_z = 0.05 + 0.04 / 2.0 + 0.01   # anchor + half_h + clearance
    assert abs(resolved.target_world[2] - expected_z) < 1e-6


def test_raycast_from_depth_on_relation():
    """OBJECT-centre Z = surface + held_h/2 + clearance.  No GRIPPER_DEPTH_M."""
    pt = PointToPlace2D(x_norm=0.5, y_norm=0.5, confidence=1.0, source="vlm")
    depth = np.full((480, 640), 0.95, dtype=np.float32)  # surface at world z=0.05
    resolved = raycast_2d_to_3d(
        pt, depth, _intrinsics(), _cam_to_world(),
        relation="on",
        held_object_height_m=0.04,
    )
    expected_z = 0.05 + 0.02 + 0.01
    assert abs(resolved.target_world[2] - expected_z) < 1e-3


def test_raycast_unknown_held_height_just_clearance():
    """With no held_h, OBJECT-centre Z = surface + clearance (no half-height)."""
    pt = PointToPlace2D(x_norm=0.5, y_norm=0.5, confidence=1.0, source="vlm")
    depth = np.full((480, 640), 0.95, dtype=np.float32)
    resolved = raycast_2d_to_3d(
        pt, depth, _intrinsics(), _cam_to_world(),
        relation="on",
        held_object_height_m=None,
    )
    expected_z = 0.05 + 0.01
    assert abs(resolved.target_world[2] - expected_z) < 1e-3


def test_topdown_rotation_with_yaw_makes_z_world_neg_z():
    """The top-down rotation builder should put panda_hand Z = world −Z
    while preserving the current yaw about world Z."""
    from vlm_orchestrator.place.tool import _topdown_rotation_with_yaw

    # Current EE has X aligned with world +X, Y aligned with world +Y,
    # Z aligned with world +Z (no rotation).  After top-down enforcement,
    # X stays aligned with world +X, Z flips to −Z.
    R = _topdown_rotation_with_yaw(np.eye(3))
    assert np.allclose(R[:, 2], [0.0, 0.0, -1.0])
    assert np.allclose(R[:, 0], [1.0, 0.0, 0.0])
    # Right-handed: Y = Z × X = (0,0,-1) × (1,0,0) = (0,-1,0)
    assert np.allclose(R[:, 1], [0.0, -1.0, 0.0])


def test_topdown_rotation_preserves_yaw():
    """Yaw of 90° about world Z should carry forward into the top-down rotation."""
    from vlm_orchestrator.place.tool import _topdown_rotation_with_yaw

    yaw_90 = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    R = _topdown_rotation_with_yaw(yaw_90)
    assert np.allclose(R[:, 2], [0.0, 0.0, -1.0])
    # X should still be world (0, 1, 0) (yaw 90°)
    assert np.allclose(R[:, 0], [0.0, 1.0, 0.0])


def test_topdown_rotation_falls_back_when_x_is_vertical():
    """If current X-axis is nearly vertical, fall back to world-X for yaw."""
    from vlm_orchestrator.place.tool import _topdown_rotation_with_yaw

    # Pathological rotation where col 0 = world +Z (gripper sideways).
    R_weird = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    R = _topdown_rotation_with_yaw(R_weird)
    assert np.allclose(R[:, 2], [0.0, 0.0, -1.0])
    assert np.allclose(R[:, 0], [1.0, 0.0, 0.0])  # fallback yaw


def test_raycast_invalid_depth_raises():
    pt = PointToPlace2D(x_norm=0.5, y_norm=0.5, confidence=1.0, source="vlm")
    with pytest.raises(PerceptionFailure):
        raycast_2d_to_3d(
            pt,
            depth=np.zeros((480, 640), dtype=np.float32),
            intrinsics=_intrinsics(),
            cam_to_world=_cam_to_world(),
            relation="in",
        )


# ----------------------------------------------------------------------
# PlaceToolExecutor — end to end with explicit 3D target
# ----------------------------------------------------------------------


def test_executor_runs_to_done_with_3d_target():
    state = SessionState()
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3)
    exec_.start(
        DestinationSpec(
            target_point_3d_world=(0.5, 0.0, 0.40),
            relation="on",
        ),
        _fake_obs(), state,
    )
    assert exec_.phase == PlacePhase.ENSURE_CLOSED

    n = 0
    while exec_.is_active and n < 200:
        resp = exec_.step(_fake_obs(), state)
        assert resp["actions"].shape == (8, 8)
        state.episode_step += 8
        n += 1

    assert exec_.phase == PlacePhase.DONE, (
        f"expected DONE, got {exec_.phase} (failure={exec_.failure_reason!r})"
    )
    assert exec_.failure_reason == ""


def test_held_object_depth_uses_robotiq_in_robolab(monkeypatch):
    """In ROBOLAB (clean-IK) mode the held-object depth must be the measured
    Robotiq fingertip depth (0.1311), not the Panda GraspGen 0.1034 — else the
    object descends ~0.0277 m too low and hits the table."""
    import vlm_orchestrator.place.tool as pt
    from vlm_orchestrator.place.destination import (
        GRIPPER_DEPTH_M as GD_PANDA,
        ROBOTIQ_GRIPPER_DEPTH_M as GD_ROBOTIQ,
    )

    monkeypatch.setattr(pt, "PLACE_ROBOLAB_CLEAN_IK", True)
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3)
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), SessionState(),
    )
    assert abs(exec_._object_in_flange[2] - GD_ROBOTIQ) < 1e-9
    assert exec_._object_in_flange[2] != GD_PANDA


def test_held_object_depth_uses_panda_when_not_clean_ik(monkeypatch):
    """LIBERO/panda path (clean-IK off) keeps the GraspGen Panda depth."""
    import vlm_orchestrator.place.tool as pt
    from vlm_orchestrator.place.destination import GRIPPER_DEPTH_M as GD_PANDA

    monkeypatch.setattr(pt, "PLACE_ROBOLAB_CLEAN_IK", False)
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3)
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), SessionState(),
    )
    assert abs(exec_._object_in_flange[2] - GD_PANDA) < 1e-9


def test_held_object_depth_env_override(monkeypatch):
    """PLACE_GRIPPER_DEPTH_M env var overrides the selected depth (audit sweep)."""
    import vlm_orchestrator.place.tool as pt

    monkeypatch.setattr(pt, "PLACE_ROBOLAB_CLEAN_IK", True)
    monkeypatch.setenv("PLACE_GRIPPER_DEPTH_M", "0.155")
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3)
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), SessionState(),
    )
    assert abs(exec_._object_in_flange[2] - 0.155) < 1e-9


def test_executor_uses_injected_motion_planner():
    """A custom motion planner is invoked for every joint-space segment
    and its label is recorded in the place log."""
    from vlm_orchestrator.motion import LinearInterpPlanner, MotionPlanResult

    class SpyPlanner(LinearInterpPlanner):
        name = "spy"

        def __init__(self):
            self.calls = []

        def plan_segment(self, q_start, q_end, *, n_steps,
                         scene_pc=None, phase=None):
            self.calls.append(phase)
            wp = super().plan_segment(
                q_start, q_end, n_steps=n_steps,
                scene_pc=scene_pc, phase=phase,
            ).waypoints
            return MotionPlanResult.ok(wp, self.name)

    spy = SpyPlanner()
    state = SessionState()
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3, motion_planner=spy)
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state,
    )
    n = 0
    while exec_.is_active and n < 200:
        exec_.step(_fake_obs(), state)
        state.episode_step += 8
        n += 1

    assert exec_.phase == PlacePhase.DONE
    # approach + final (+ retreat) segments should have been planned by the spy
    assert "approach" in spy.calls
    assert "final" in spy.calls
    assert exec_._place_log.get("motion_planner_approach") == "spy"


def test_executor_default_motion_planner_is_linear():
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3)
    assert exec_._motion_planner.name == "linear"


def test_linear_planner_attach_detach_are_noops():
    """The collision-unaware linear planner must expose attach/detach as
    no-ops returning False (so place can call them uniformly)."""
    from vlm_orchestrator.motion import LinearInterpPlanner

    p = LinearInterpPlanner()
    assert p.attach_object(None, None) is False
    assert p.detach_object() is False


def test_collision_aware_place_attaches_held_object_and_detaches(monkeypatch):
    """A collision-aware planner: place builds a scene cloud, attaches the
    grasp-time held object, passes scene_pc to the approach segment, and
    always detaches afterwards."""
    import numpy as np
    import vlm_orchestrator.place.tool as pt
    from vlm_orchestrator.motion import LinearInterpPlanner, MotionPlanResult

    # Collision-free path is off by default (plain-cuRobo); enable for this test.
    monkeypatch.setattr(pt, "COLLISION_FREE_PLANNING", True)

    class CollisionSpyPlanner(LinearInterpPlanner):
        name = "curobo"  # non-linear → collision-aware code paths fire

        def __init__(self):
            self.attached = []
            self.detached = 0
            self.approach_scene_pc = "unset"

        def plan_segment(self, q_start, q_end, *, n_steps,
                         scene_pc=None, phase=None):
            if phase == "approach":
                self.approach_scene_pc = scene_pc
            wp = super().plan_segment(
                q_start, q_end, n_steps=n_steps, scene_pc=None, phase=phase,
            ).waypoints
            return MotionPlanResult.ok(wp, self.name)

        def attach_object(self, obj_pc, q_hold, *, num_spheres=4):
            self.attached.append((np.asarray(obj_pc).shape, num_spheres))
            return True

        def detach_object(self):
            self.detached += 1
            return True

    spy = CollisionSpyPlanner()
    state = SessionState()
    # Simulate a prior grasp: stash held-object cloud + FK EE pose.
    from vlm_orchestrator.grasp.ik import forward_kinematics
    q_grasp = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.6, 0.78])
    state.last_grasped_object_pc_world = (
        np.random.default_rng(0).normal(0.5, 0.02, (500, 3)).astype(np.float32)
    )
    state.last_grasped_ee_pose = forward_kinematics(q_grasp)

    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3, motion_planner=spy)
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state,
    )
    n = 0
    while exec_.is_active and n < 200:
        exec_.step(_fake_obs(), state)
        state.episode_step += 8
        n += 1

    assert exec_.phase == PlacePhase.DONE
    # Held object was attached exactly once, with the sphere cap.
    assert len(spy.attached) == 1
    assert spy.attached[0][1] == 4
    # And detached afterwards (no leaked attachment).
    assert spy.detached >= 1
    # The approach segment received a non-None scene cloud.
    assert spy.approach_scene_pc is not None
    assert exec_._place_log.get("place_attach_path") == "attached_grasp_cloud"


def test_collision_aware_place_without_prior_grasp_skips_attach(monkeypatch):
    """If no grasp cloud is stashed, place still plans (collision-aware for the
    scene) but does not attach anything."""
    import vlm_orchestrator.place.tool as pt
    from vlm_orchestrator.motion import LinearInterpPlanner, MotionPlanResult

    monkeypatch.setattr(pt, "COLLISION_FREE_PLANNING", True)

    class CollisionSpyPlanner(LinearInterpPlanner):
        name = "curobo"

        def __init__(self):
            self.attached = 0

        def plan_segment(self, q_start, q_end, *, n_steps,
                         scene_pc=None, phase=None):
            wp = super().plan_segment(
                q_start, q_end, n_steps=n_steps, scene_pc=None, phase=phase,
            ).waypoints
            return MotionPlanResult.ok(wp, self.name)

        def attach_object(self, obj_pc, q_hold, *, num_spheres=4):
            self.attached += 1
            return True

        def detach_object(self):
            return True

    spy = CollisionSpyPlanner()
    state = SessionState()  # no last_grasped_* set
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3, motion_planner=spy)
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state,
    )
    n = 0
    while exec_.is_active and n < 200:
        exec_.step(_fake_obs(), state)
        state.episode_step += 8
        n += 1

    assert exec_.phase == PlacePhase.DONE
    assert spy.attached == 0
    assert exec_._place_log.get("place_attach_path") == "none_no_grasp_cloud"


def test_executor_close_hold_when_gripper_open():
    """ENSURE_CLOSED must hold the close command for at least one chunk."""
    state = SessionState()
    exec_ = PlaceToolExecutor()
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(gripper_closed=False), state,
    )
    held_chunks = 0
    while exec_.phase == PlacePhase.ENSURE_CLOSED and held_chunks < 20:
        resp = exec_.step(_fake_obs(gripper_closed=False), state)
        # Close gripper should be commanded
        assert resp["actions"][0, -1] == 1.0  # GRIPPER_CLOSE
        held_chunks += 1
        state.episode_step += 8
    assert held_chunks >= 1
    assert exec_.phase != PlacePhase.ENSURE_CLOSED


def test_executor_fails_loudly_on_gt_sim_without_gt_state():
    """seg_mode='gt_sim' with no gt_state in obs → FAILED + perception_no_target."""
    state = SessionState()
    exec_ = PlaceToolExecutor(seg_mode="gt_sim")
    exec_.start(
        DestinationSpec(target_object="bowl", relation="in"),
        _fake_obs(), state,
    )
    n = 0
    while exec_.is_active and n < 20:
        exec_.step(_fake_obs(), state)
        state.episode_step += 8
        n += 1
    assert exec_.phase == PlacePhase.FAILED
    assert exec_.failure_reason == "perception_no_target"


# ----------------------------------------------------------------------
# DestinationSpec validation
# ----------------------------------------------------------------------


def test_destination_spec_requires_exactly_one_target():
    with pytest.raises(ValueError):
        DestinationSpec().validate()
    with pytest.raises(ValueError):
        DestinationSpec(
            target_object="x", target_point_2d=(0.5, 0.5),
        ).validate()
    DestinationSpec(target_object="x").validate()  # OK


# ----------------------------------------------------------------------
# Mock executor — same surface as real executor
# ----------------------------------------------------------------------


def test_mock_executor_runs_to_done():
    state = SessionState()
    exec_ = MockPlaceToolExecutor()
    exec_.start(
        DestinationSpec(target_object="anywhere", relation="in"),
        _fake_obs(), state,
    )
    n = 0
    while exec_.is_active and n < 30:
        resp = exec_.step(_fake_obs(), state)
        assert resp["actions"].shape == (8, 8)
        n += 1
    assert exec_.phase == PlacePhase.DONE


# ----------------------------------------------------------------------
# Stack / no-stack placement semantics (Part #1)
# ----------------------------------------------------------------------


def test_stack_default_disabled_ignores_arg():
    """Master switch OFF → stack arg ignored, historical grasp-consistent
    placement (self._stack stays True regardless of the arg)."""
    state = SessionState()
    exec_ = PlaceToolExecutor(seg_mode=PlaceSegMode.SAM3)
    assert exec_._stack_mode_enabled is False
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state, stack=False,
    )
    # Arg was False but master switch off → must stay True (today's path).
    assert exec_._stack is True
    assert exec_._place_log["stack"] is True
    assert exec_._place_log["stack_mode_enabled"] is False


def test_stack_enabled_honours_false_arg():
    """Master switch ON + stack=False → top-down placement selected."""
    state = SessionState()
    exec_ = PlaceToolExecutor(
        seg_mode=PlaceSegMode.SAM3, stack_mode_enabled=True,
    )
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state, stack=False,
    )
    assert exec_._stack is False
    assert exec_._place_log["stack"] is False
    assert exec_._place_log["stack_mode_enabled"] is True


def test_stack_enabled_honours_true_arg():
    """Master switch ON + stack=True → grasp-consistent placement."""
    state = SessionState()
    exec_ = PlaceToolExecutor(
        seg_mode=PlaceSegMode.SAM3, stack_mode_enabled=True,
    )
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state, stack=True,
    )
    assert exec_._stack is True
    assert exec_._place_log["stack"] is True


def test_stack_default_arg_is_false():
    """start() default stack=False → plain top-down release for ordinary
    pick-and-place when the caller omits the arg (with the master switch on).
    Stacking must be an explicit opt-in."""
    state = SessionState()
    exec_ = PlaceToolExecutor(
        seg_mode=PlaceSegMode.SAM3, stack_mode_enabled=True,
    )
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state,
    )
    assert exec_._stack is False


def test_stack_switch_off_forces_historical_grasp_consistent():
    """When the master switch is OFF, ``stack`` is ignored and the historical
    grasp-consistent placement (stack semantics == True) is used regardless of
    the arg."""
    state = SessionState()
    exec_ = PlaceToolExecutor(
        seg_mode=PlaceSegMode.SAM3, stack_mode_enabled=False,
    )
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state, stack=False,
    )
    assert exec_._stack is True


@pytest.mark.parametrize(
    "stack,expected_first_label",
    [
        (True, "current_ee"),
        (False, "topdown_yaw"),
    ],
)
def test_stack_rot_candidates_selection(monkeypatch, stack, expected_first_label):
    """_plan_trajectory tries current_ee-first when stack=True and top-down
    only when stack=False. Fake IK to converge on the FIRST candidate so the
    accepted rotation label reflects the ordering the branch produced."""
    import vlm_orchestrator.grasp.ik as _ik

    call_order: list[str] = []

    # inverse_kinematics_multistart returns (q, converged, strategy).
    # Converge immediately so the first candidate is accepted; record how
    # many times it's called to infer which candidate ran first.
    # Patched at the source module — _plan_trajectory imports it lazily.
    def _fake_ik(T_target, q_seed, **kw):
        call_order.append("ik")
        return np.zeros(7), True, "primary"

    monkeypatch.setattr(_ik, "inverse_kinematics_multistart", _fake_ik)

    state = SessionState()
    exec_ = PlaceToolExecutor(
        seg_mode=PlaceSegMode.SAM3, stack_mode_enabled=True,
    )
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state, stack=stack,
    )
    # Resolve target directly to skip perception; supply approach axis.
    exec_._resolved_target_world = np.array([0.5, 0.0, 0.40])
    exec_._approach_axis_world = np.array([0.0, 0.0, -1.0])
    exec_._plan_trajectory(_fake_obs())

    # First candidate converges → its label is recorded.
    assert exec_._place_log.get("R_target_label") == expected_first_label
    if stack:
        # current_ee + topdown available; only the first was tried (pre+place).
        assert len(call_order) == 2
    else:
        # single candidate → pre+place = 2 IK calls, top-down only.
        assert len(call_order) == 2


@pytest.mark.parametrize("clean_ik", [True, False])
def test_clean_ik_relabels_target(monkeypatch, clean_ik):
    """Clean-IK path must feed IK the base_link-relabelled target (T_hand @ C)
    with T_flange_to_tcp=C; legacy path feeds the raw panda_hand target with
    tcp=None. Pins the frame-relabel wiring so it can't silently regress."""
    import vlm_orchestrator.grasp.ik as _ik
    import vlm_orchestrator.place.tool as _pt

    monkeypatch.setattr(_pt, "PLACE_ROBOLAB_CLEAN_IK", clean_ik)
    C = np.linalg.inv(_ik._T_FLANGE_HAND) @ _ik.T_JOINT7_TO_ROBOTIQ_BASE

    captured: list[dict] = []

    def _fake_ik(T_target, q_seed, **kw):
        captured.append({"T": np.asarray(T_target).copy(),
                         "tcp": kw.get("T_flange_to_tcp")})
        return np.zeros(7), True, "primary"

    monkeypatch.setattr(_ik, "inverse_kinematics_multistart", _fake_ik)

    state = SessionState()
    exec_ = PlaceToolExecutor(
        seg_mode=PlaceSegMode.SAM3, stack_mode_enabled=False,
    )
    exec_.start(
        DestinationSpec(target_point_3d_world=(0.5, 0.0, 0.40), relation="on"),
        _fake_obs(), state, stack=False,
    )
    exec_._resolved_target_world = np.array([0.5, 0.0, 0.40])
    exec_._approach_axis_world = np.array([0.0, 0.0, -1.0])
    exec_._plan_trajectory(_fake_obs())

    assert exec_._place_log.get("ik_clean_robolab") is clean_ik
    assert len(captured) == 2  # pre-place + place
    for c in captured:
        if clean_ik:
            assert c["tcp"] is not None
            np.testing.assert_allclose(c["tcp"], C, atol=1e-12)
        else:
            assert c["tcp"] is None
    # In the clean path, IK target should be the panda_hand target @ C:
    # recovering T_hand = T_ik @ inv(C) must be a valid SE(3) pose.
    if clean_ik:
        T_hand = captured[1]["T"] @ np.linalg.inv(C)
        # rotation block orthonormal
        Rblk = T_hand[:3, :3]
        np.testing.assert_allclose(Rblk @ Rblk.T, np.eye(3), atol=1e-9)
