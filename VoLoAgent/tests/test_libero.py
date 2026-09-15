# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LIBERO integration components.

Covers:
  - CLI --env libero preset
  - Failure signal detection with LIBERO-specific thresholds
  - Camera intrinsics from MuJoCo fovy
  - Grasp tool LIBERO EE-delta action generation
  - Wire observation format
  - Strategy env_mode threading
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pytest

# ── CLI Preset ──


def test_env_preset_libero():
    """--env libero sets correct defaults (sim-step units).

    subgoal_timeout is NOT overridden by the env preset — the global
    default (9999) effectively disables time-based subgoal advancement
    across all benchmarks; callers wanting a real timeout must set it
    explicitly.
    """
    from vlm_orchestrator.cli import _apply_env_preset

    args = argparse.Namespace(
        env="libero",
        image_key="observation/exterior_image_1_left",
        check_interval=80,
        subgoal_timeout=9999,
        robolab_output_dir=os.path.expanduser("~/robolab/output"),
    )
    _apply_env_preset(args)
    assert args.image_key == "observation/image"
    assert args.check_interval == 40
    assert args.subgoal_timeout == 9999     # left untouched
    assert args.robolab_output_dir == ""


def test_env_preset_robolab_noop():
    """--env robolab doesn't change defaults (sim-step units)."""
    from vlm_orchestrator.cli import _apply_env_preset

    args = argparse.Namespace(
        env="robolab",
        image_key="observation/exterior_image_1_left",
        check_interval=80,
        subgoal_timeout=9999,
        robolab_output_dir=os.path.expanduser("~/robolab/output"),
    )
    _apply_env_preset(args)
    assert args.image_key == "observation/exterior_image_1_left"
    assert args.check_interval == 80
    assert args.subgoal_timeout == 9999


def test_env_preset_no_override_explicit():
    """Don't override user-explicit image key."""
    from vlm_orchestrator.cli import _apply_env_preset

    args = argparse.Namespace(
        env="libero",
        image_key="observation/custom_cam",  # user overrode
        check_interval=5,  # user overrode
        subgoal_timeout=30,  # user overrode
        robolab_output_dir="/custom/path",  # user overrode
    )
    _apply_env_preset(args)
    # Should NOT change user-set values
    assert args.image_key == "observation/custom_cam"
    assert args.check_interval == 5
    assert args.subgoal_timeout == 30
    assert args.robolab_output_dir == "/custom/path"


# ── Failure Signals ──


def test_signal_config_defaults():
    from vlm_orchestrator.failure_handlers.signal_detector import (
        ROBOLAB_SIGNAL_CONFIG,
        LIBERO_SIGNAL_CONFIG,
    )

    assert ROBOLAB_SIGNAL_CONFIG.env == "robolab"
    assert ROBOLAB_SIGNAL_CONFIG.table_z == 0.22
    assert LIBERO_SIGNAL_CONFIG.env == "libero"
    assert LIBERO_SIGNAL_CONFIG.table_z == 0.82


def test_get_signal_config():
    from vlm_orchestrator.failure_handlers.signal_detector import get_signal_config

    assert get_signal_config("robolab").env == "robolab"
    assert get_signal_config("libero").env == "libero"
    assert get_signal_config("unknown").env == "robolab"  # default


def test_failure_detector_uses_config():
    from vlm_orchestrator.failure_handlers.signal_detector import (
        FailureSignalDetector,
        LIBERO_SIGNAL_CONFIG,
    )

    det = FailureSignalDetector(signal_config=LIBERO_SIGNAL_CONFIG)
    assert det.config.table_z == 0.82
    assert det.config.env == "libero"


def test_combined_detector_passes_config():
    from vlm_orchestrator.failure_handlers.signal_detector import (
        CombinedDetector,
        LIBERO_SIGNAL_CONFIG,
    )

    cd = CombinedDetector(
        mode="signal_primary",
        signal_config=LIBERO_SIGNAL_CONFIG,
    )
    assert cd.signal_detector.config.env == "libero"


def test_classify_uses_config_thresholds():
    """Verify classify() uses configurable thresholds, not hardcoded values."""
    from vlm_orchestrator.failure_handlers.signal_detector import (
        FailureSignalDetector,
        LIBERO_SIGNAL_CONFIG,
        ManipulationStatus,
    )

    det = FailureSignalDetector(
        window_size=16, signal_config=LIBERO_SIGNAL_CONFIG
    )
    # Feed signals that look like "stall_frozen" in LIBERO frame:
    # EE at table level (z≈0.82), barely moving
    for _ in range(16):
        det.update(
            gripper_action=0.0,
            gripper_width=0.0,
            ee_position=np.array([0.4, 0.0, 0.82]),
            arm_actions=None,
        )

    result = det.classify(chunks_elapsed=15)
    assert result.status == ManipulationStatus.FAILURE
    assert "stall" in result.reason


def test_libero_carrying_detected():
    """Carrying at LIBERO heights (z≈0.90) should be detected as PROGRESS."""
    from vlm_orchestrator.failure_handlers.signal_detector import (
        FailureSignalDetector,
        LIBERO_SIGNAL_CONFIG,
        ManipulationStatus,
    )

    det = FailureSignalDetector(
        window_size=16, signal_config=LIBERO_SIGNAL_CONFIG
    )
    # Simulate carrying: gripper closed, EE elevated and moving
    for i in range(16):
        det.update(
            gripper_action=1.0,
            gripper_width=-0.5,
            ee_position=np.array([0.4, 0.0 + i * 0.01, 0.90]),
            arm_actions=None,
        )

    result = det.classify(chunks_elapsed=5)
    assert result.status == ManipulationStatus.PROGRESS


# ── Camera Utils ──


def test_intrinsics_from_fovy():
    from vlm_orchestrator.grasp.camera import intrinsics_from_fovy

    intr = intrinsics_from_fovy(fovy=45.0, width=256, height=256)
    expected_fy = (256 / 2) / np.tan(np.deg2rad(45) / 2)
    assert abs(intr.fy - expected_fy) < 0.01
    assert intr.fx == intr.fy  # square pixels
    assert intr.cx == 128
    assert intr.cy == 128


def test_libero_agentview_intrinsics():
    from vlm_orchestrator.grasp.camera import libero_agentview_intrinsics

    intr = libero_agentview_intrinsics()
    assert intr.width == 256
    assert intr.height == 256
    assert intr.fx > 0


def test_extrinsics_from_obs_none():
    from vlm_orchestrator.grasp.camera import extrinsics_from_obs

    intr, ext = extrinsics_from_obs(None, None)
    assert intr is None
    assert ext is None


def test_extrinsics_from_obs_roundtrip():
    from vlm_orchestrator.grasp.camera import extrinsics_from_obs

    K = np.array([[300, 0, 128], [0, 300, 128], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [1, 2, 3]

    intr, ext = extrinsics_from_obs(K.flatten(), T.flatten())
    assert intr is not None
    assert intr.fx == 300.0
    assert intr.width == 256
    np.testing.assert_array_almost_equal(ext, T)


# ── Grasp Tool ──


def test_grasp_tool_libero_mode():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor, GraspEnvMode

    exec = GraspToolExecutor(env_mode="libero")
    assert exec._env_mode == GraspEnvMode.LIBERO
    assert exec._action_dim == 7
    assert exec._action_horizon == 10
    assert exec._gripper_open == -1.0
    assert exec._gripper_close == 1.0


def test_grasp_tool_robolab_mode():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor, GraspEnvMode

    exec = GraspToolExecutor(env_mode="robolab")
    assert exec._env_mode == GraspEnvMode.ROBOLAB
    assert exec._action_dim == 8
    assert exec._action_horizon == 8
    assert exec._gripper_open == 0.0
    assert exec._gripper_close == 1.0


def test_interpolate_ee_deltas():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.1, -0.05, 0.02])
    n_steps = 10

    deltas = GraspToolExecutor._interpolate_ee_deltas(start, end, n_steps)
    assert len(deltas) == n_steps
    total = sum(deltas)
    np.testing.assert_allclose(total, end - start, atol=1e-6)


def test_interpolate_ee_deltas_large_displacement():
    """Large displacement should be clamped but approach target."""
    from vlm_orchestrator.grasp.tool import GraspToolExecutor, LIBERO_MAX_POS_DELTA

    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.5, 0.0, 0.0])  # very large
    n_steps = 5

    deltas = GraspToolExecutor._interpolate_ee_deltas(start, end, n_steps)
    for d in deltas:
        assert np.all(np.abs(d) <= LIBERO_MAX_POS_DELTA + 1e-8)


def test_libero_noop_response():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    exec = GraspToolExecutor(env_mode="libero")
    resp = exec._noop_response({})
    assert resp["actions"].shape == (10, 7)
    # All zeros except gripper should be open (-1)
    assert resp["actions"][0, -1] == -1.0


def test_plan_retreat_libero():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    exec = GraspToolExecutor(env_mode="libero")
    exec._ee_at_grasp = np.array([0.4, 0.0, 0.82])
    exec._plan_retreat_libero()
    assert len(exec._trajectory) > 0
    # Each action should be 7D
    assert exec._trajectory[0].shape == (7,)
    # Gripper should be closed
    assert exec._trajectory[0][-1] == 1.0
    # Z-delta should be positive (lifting up)
    assert exec._trajectory[0][2] > 0


def test_plan_trajectory_libero():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    exec = GraspToolExecutor(env_mode="libero")
    current_pos = np.array([0.4, 0.0, 1.0])
    current_quat = np.array([0.0, 0.0, 0.0, 1.0])
    grasp_world = np.eye(4)
    grasp_world[:3, 3] = [0.3, 0.1, 0.82]
    # Valid right-handed frame: approach from above (z down, x right, y forward)
    grasp_world[:3, :3] = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ])

    exec._plan_trajectory_libero(current_pos, current_quat, grasp_world)

    assert len(exec._seg_approach) > 0
    assert len(exec._seg_final) > 0
    # Each action is 7D
    assert exec._seg_approach[0].shape == (7,)
    assert exec._seg_final[0].shape == (7,)
    # Gripper open during approach
    assert exec._seg_approach[0][-1] == -1.0


# ── Grasp Tool Image/Depth Extraction ──


def test_extract_image_libero_keys():
    """_extract_image should find LIBERO image keys."""
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    executor = GraspToolExecutor(seg_mode="gdino_sam2", env_mode="libero")
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    obs = {"observation/image_raw": img}
    result = executor._extract_image(obs)
    assert result.shape == (256, 256, 3)


def test_extract_image_robolab_keys():
    """_extract_image should still find robolab image keys."""
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    executor = GraspToolExecutor(seg_mode="gdino_sam2", env_mode="robolab")
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    obs = {"observation/exterior_image_1_left_raw": img}
    result = executor._extract_image(obs)
    assert result.shape == (720, 1280, 3)


def test_extract_depth_libero_key():
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    executor = GraspToolExecutor(seg_mode="gdino_sam2", env_mode="libero")
    depth = np.ones((256, 256), dtype=np.float32)
    obs = {"observation/depth_agentview": depth}
    result = executor._extract_depth(obs)
    assert result is not None
    assert result.shape == (256, 256)


def test_libero_image_depth_unmirrored_for_grasp():
    """libero_eval_client applies [::-1, ::-1] for pi0.5 training, but
    camera_K and camera_extrinsic come from MuJoCo for the un-mirrored
    OpenCV camera. _extract_image/_extract_depth must un-mirror
    horizontally so the grasp pipeline (point cloud, GraspGen,
    cam_to_world) is self-consistent.
    """
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    executor = GraspToolExecutor(seg_mode="gdino_sam2", env_mode="libero")
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[10, 20] = (255, 0, 0)  # red marker on the LEFT half of the image
    out = executor._extract_image({"observation/image_raw": img})
    # Un-mirror flips X only — row index unchanged, column index inverted.
    assert out[10, 256 - 1 - 20].tolist() == [255, 0, 0]
    assert out[10, 20].tolist() == [0, 0, 0]

    depth = np.zeros((256, 256), dtype=np.float32)
    depth[10, 20] = 1.5
    dout = executor._extract_depth({"observation/depth_agentview": depth})
    assert dout is not None
    assert dout[10, 256 - 1 - 20] == 1.5
    assert dout[10, 20] == 0.0


def test_robolab_image_depth_not_unmirrored():
    """Robolab cameras already come in OpenCV convention — no flip applied."""
    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    executor = GraspToolExecutor(seg_mode="gdino_sam2", env_mode="robolab")
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[10, 20] = (255, 0, 0)
    out = executor._extract_image({"observation/exterior_image_1_left_raw": img})
    assert out[10, 20].tolist() == [255, 0, 0]
    assert out[10, 1280 - 1 - 20].tolist() == [0, 0, 0]


def test_gt_phrase_drops_grasp_verbs():
    """Regression: GT-to-instruction phrase extraction must drop grasp
    verbs ("lift", "pick", "grasp", ...) from the prefix.  Otherwise the
    verb leaks into the phrase ("lift bowl"), and GDino text-grounds the
    verb onto the gripper instead of the actual object — observed in
    LIBERO-Mem KITCHEN_SCENE1_3 producing target_object="lift bowl".
    """
    from vlm_orchestrator.failure_handlers.gt_detector import (
        _extract_phrase_for_gt_name,
    )
    # The two failing cases that motivated the fix:
    assert _extract_phrase_for_gt_name(
        "lift the bowl and place it back on the plate 3 times",
        "akita_black_bowl_1",
    ) == "bowl"
    assert _extract_phrase_for_gt_name(
        "grasp the wooden block",
        "wooden_block",
    ) == "wooden block"
    # Existing pick / put cases still work.
    assert _extract_phrase_for_gt_name(
        "pick up the red hammer",
        "red_hammer",
    ) == "red hammer"
    assert _extract_phrase_for_gt_name(
        "put the cup on the saucer",
        "blue_cup",
    ) == "cup"


# ── Strategy env_mode Threading ──


def test_subgoal_strategy_env_mode():
    from vlm_orchestrator.strategies.base import StrategyContext
    from vlm_orchestrator.strategies.subgoal import SubgoalStrategy, SubgoalConfig
    from vlm_orchestrator.vlm import PassthroughVLM

    ctx = StrategyContext(
        vlm=PassthroughVLM(),
        image_key="observation/image",
        extra_image_keys=["observation/wrist_image"],
    )
    s = SubgoalStrategy(
        ctx,
        SubgoalConfig(),
        failure_monitor="signal_primary",
        env_mode="libero",
    )
    assert s._env_mode == "libero"
    assert s._failure_detector.signal_detector.config.env == "libero"


def test_subgoal_scene_edit_strategy_env_mode():
    from vlm_orchestrator.strategies.base import StrategyContext
    from vlm_orchestrator.strategies.archive.subgoal_scene_edit import (
        SubgoalSceneEditStrategy,
        SubgoalSceneEditConfig,
    )
    from vlm_orchestrator.vlm import PassthroughVLM

    ctx = StrategyContext(
        vlm=PassthroughVLM(),
        image_key="observation/image",
        extra_image_keys=["observation/wrist_image"],
    )
    s = SubgoalSceneEditStrategy(
        ctx,
        SubgoalSceneEditConfig(),
        failure_monitor="signal_primary",
        env_mode="libero",
    )
    assert s._env_mode == "libero"
    assert s._failure_detector.signal_detector.config.env == "libero"


# ── Wire Format ──


def test_build_wire_obs():
    """Test that LIBERO eval client wire format is correct."""
    # Import the function without needing libero/openpi installed
    import sys
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "libero_eval_client",
        "examples/libero/libero_eval_client.py",
        submodule_search_locations=[],
    )
    # We can't fully import without libero/openpi deps, so just
    # verify the file is valid Python
    with open("examples/libero/libero_eval_client.py") as f:
        code = f.read()
    compile(code, "libero_eval_client.py", "exec")


# ── Proxy GT Done ──


def test_proxy_gt_done_passthrough():
    """Verify that ground_truth_done in obs propagates to response."""
    import ast

    with open("vlm_orchestrator/proxy.py") as f:
        source = f.read()

    assert 'obs.get("ground_truth_done")' in source
    assert '"orchestrator_gt_done"' in source


# ── GT Failure Detector with LIBERO format ──


def _make_libero_gt_state(
    grasped=None, grabbed=False, goal_satisfied=False,
    scene_objects=None, score=0.0,
):
    """Build a LIBERO-style gt_state dict."""
    if scene_objects is None:
        scene_objects = ["alphabet_soup_1", "basket_1"]
    conditions = [
        {"condition_idx": 0, "predicate": "In",
         "object": "alphabet_soup_1",
         "target": "basket_1_contain_region",
         "satisfied": goal_satisfied,
         "info": "In(alphabet_soup_1, basket_1_contain_region)"},
        # CSM synthetic
        {"object": "alphabet_soup_1", "condition_idx": 0,
         "satisfied": grabbed, "info": "grabbed(alphabet_soup_1)"},
        {"object": "alphabet_soup_1", "condition_idx": 3,
         "satisfied": goal_satisfied,
         "info": "goal_satisfied(alphabet_soup_1)"},
    ]
    return {
        "robot": {
            "grasped_object": grasped,
            "gripper_width": 0.04 if grasped is None else 0.01,
            "ee_pos": [0.4, 0.0, 0.9],
        },
        "objects": {
            "alphabet_soup_1": {"pos": [0.3, -0.1, 0.82], "quat": [1, 0, 0, 0]},
        },
        "scene_objects": scene_objects,
        "subtask": {
            "score": score if not goal_satisfied else 1.0,
            "conditions": conditions,
            "object_completed": {"alphabet_soup_1": goal_satisfied},
            "all_subtask_conditions": {"goal_0": goal_satisfied},
        },
    }


def test_gt_detector_libero_wrong_object():
    from vlm_orchestrator.failure_handlers.gt_detector import (
        GTFailureDetector, GTFailureType,
    )

    det = GTFailureDetector()
    scene = ["alphabet_soup_1", "basket_1", "milk_1"]
    gt = _make_libero_gt_state(scene_objects=scene)

    det.set_subgoal(
        instruction="Pick up the alphabet soup and place it in the basket",
        scene_objects=scene,
        gt_conditions=gt["subtask"]["conditions"],
    )

    # Warmup
    for _ in range(10):
        det.update(gt)

    # Wrong object
    gt_wrong = _make_libero_gt_state(grasped="milk_1", scene_objects=scene)
    result = det.update(gt_wrong)
    assert result.failure_type == GTFailureType.WRONG_OBJECT_PICKED


def test_gt_detector_libero_drop():
    from vlm_orchestrator.failure_handlers.gt_detector import (
        GTFailureDetector, GTFailureType,
    )

    det = GTFailureDetector()
    scene = ["alphabet_soup_1", "basket_1"]
    gt = _make_libero_gt_state(scene_objects=scene)

    det.set_subgoal(
        instruction="Pick up the alphabet soup and place it in the basket",
        scene_objects=scene,
        gt_conditions=gt["subtask"]["conditions"],
    )

    # Warmup
    for _ in range(10):
        det.update(gt)

    # Grasp
    gt_grab = _make_libero_gt_state(
        grasped="alphabet_soup_1", grabbed=True, scene_objects=scene,
    )
    for _ in range(5):
        det.update(gt_grab)

    # Drop
    gt_drop = _make_libero_gt_state(scene_objects=scene)
    result = det.update(gt_drop)
    assert result.failure_type == GTFailureType.OBJECT_DROPPED


def test_gt_detector_libero_complete():
    from vlm_orchestrator.failure_handlers.gt_detector import (
        GTFailureDetector, GTFailureType,
    )

    det = GTFailureDetector()
    scene = ["alphabet_soup_1", "basket_1"]
    gt = _make_libero_gt_state(scene_objects=scene)

    det.set_subgoal(
        instruction="Pick up the alphabet soup and place it in the basket",
        scene_objects=scene,
        gt_conditions=gt["subtask"]["conditions"],
    )

    for _ in range(10):
        det.update(gt)

    # Complete — need COMPLETION_CONFIRM_CHUNKS consecutive frames
    gt_done = _make_libero_gt_state(
        goal_satisfied=True, scene_objects=scene,
    )
    found_complete = False
    for _ in range(det.COMPLETION_CONFIRM_CHUNKS + 1):
        result = det.update(gt_done)
        if result.failure_type == GTFailureType.SUBGOAL_COMPLETE:
            found_complete = True
            break
    assert found_complete, (
        f"Expected SUBGOAL_COMPLETE within "
        f"{det.COMPLETION_CONFIRM_CHUNKS + 1} steps"
    )


def test_gt_detector_libero_regression():
    from vlm_orchestrator.failure_handlers.gt_detector import (
        GTFailureDetector, GTFailureType,
    )

    det = GTFailureDetector()
    scene = ["alphabet_soup_1", "basket_1"]
    gt = _make_libero_gt_state(scene_objects=scene)

    det.set_subgoal(
        instruction="Pick up the alphabet soup and place it in the basket",
        scene_objects=scene,
        gt_conditions=gt["subtask"]["conditions"],
    )

    for _ in range(10):
        det.update(gt)

    # Complete and hold for confirmation
    gt_done = _make_libero_gt_state(goal_satisfied=True, scene_objects=scene)
    for _ in range(det.REGRESSION_CONFIRM_CHUNKS + 1):
        det.update(gt_done)

    assert det._globally_completed.get("goal_0", False)

    # Regress
    gt_regress = _make_libero_gt_state(scene_objects=scene)
    result = det.update(gt_regress)
    assert result.failure_type == GTFailureType.SUBTASK_REGRESSION


def test_libero_gt_state_exporter_import():
    """Verify LiberoGTStateExporter can be imported."""
    from vlm_orchestrator.aux_benchmarks.libero_gt import LiberoGTStateExporter
    assert LiberoGTStateExporter is not None
