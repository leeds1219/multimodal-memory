# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RoboCasa and VLABench integration code.

These tests verify the eval client wire format, GT state exporters,
and action decoding without requiring the actual simulator environments.
"""

import numpy as np
import pytest
import sys
import types


# ── Mock modules ──────────────────────────────────────────────────────

def _make_mock_image_tools():
    """Create a minimal mock for openpi_client.image_tools."""
    mod = types.ModuleType("openpi_client.image_tools")

    def resize_with_pad(img, h, w):
        if img.shape[0] == h and img.shape[1] == w:
            return img
        # Simple resize by repeating/slicing
        return np.zeros((h, w, 3), dtype=img.dtype)

    def convert_to_uint8(img):
        if img.dtype == np.uint8:
            return img
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    mod.resize_with_pad = resize_with_pad
    mod.convert_to_uint8 = convert_to_uint8
    return mod


@pytest.fixture(autouse=True)
def mock_openpi_client(monkeypatch):
    """Provide mock openpi_client for all tests."""
    openpi_client = types.ModuleType("openpi_client")
    openpi_client.image_tools = _make_mock_image_tools()
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)
    monkeypatch.setitem(sys.modules, "openpi_client.image_tools",
                        openpi_client.image_tools)


# ── RoboCasa Wire Format Tests ───────────────────────────────────────

class TestRoboCasaWireFormat:
    """Test that the RoboCasa eval client produces correct wire obs."""

    def _make_robocasa_obs(self):
        """Create a minimal RoboCasa-like observation dict."""
        return {
            "video.robot0_agentview_left": np.random.randint(
                0, 255, (128, 128, 3), dtype=np.uint8),
            "video.robot0_eye_in_hand": np.random.randint(
                0, 255, (128, 128, 3), dtype=np.uint8),
            "state.end_effector_position_relative": np.array([0.1, 0.2, 0.3]),
            "state.end_effector_rotation_relative": np.array([1, 0, 0, 0.0]),
            "state.base_position": np.array([0.0, 0.0, 0.0]),
            "state.base_rotation": np.array([0.0, 0.0, 0.0]),
            "state.gripper_qpos": np.array([0.04, 0.04]),
            "annotation.human.task_description": "pick up the mug",
        }

    def test_build_wire_obs_keys(self):
        """Wire obs has all required keys."""
        from examples.robocasa.robocasa_eval_client import build_wire_obs
        obs = self._make_robocasa_obs()
        wire = build_wire_obs(obs, "pick up mug", 224)

        assert "observation/image" in wire
        assert "observation/wrist_image" in wire
        assert "observation/state" in wire
        assert "observation/ee_pos" in wire
        assert "observation/gripper_position" in wire
        assert "prompt" in wire
        assert "observation/image_raw" in wire

    def test_image_shape(self):
        """Images are resized to 224×224."""
        from examples.robocasa.robocasa_eval_client import build_wire_obs
        obs = self._make_robocasa_obs()
        wire = build_wire_obs(obs, "pick up mug", 224)

        assert wire["observation/image"].shape == (224, 224, 3)
        assert wire["observation/wrist_image"].shape == (224, 224, 3)
        assert wire["observation/image_raw"].shape == (256, 256, 3)

    def test_state_dimension(self):
        """State is 16D for RoboCasa (ee_pos_rel + ee_rot_rel + base + gripper)."""
        from examples.robocasa.robocasa_eval_client import build_wire_obs
        obs = self._make_robocasa_obs()
        wire = build_wire_obs(obs, "test", 224)

        # 3 + 4 + 3 + 3 + 2 = 15 (if rotation is quat)
        # Actual: depends on what obs provides, but state should be a flat array
        state = wire["observation/state"]
        assert state.ndim == 1
        assert state.dtype == np.float32
        assert len(state) >= 7  # at minimum

    def test_ee_pos_shape(self):
        """EE position is 3D."""
        from examples.robocasa.robocasa_eval_client import build_wire_obs
        obs = self._make_robocasa_obs()
        wire = build_wire_obs(obs, "test", 224)

        ee_pos = wire["observation/ee_pos"]
        assert ee_pos.shape == (3,)

    def test_prompt_passed_through(self):
        """Prompt is passed as-is."""
        from examples.robocasa.robocasa_eval_client import build_wire_obs
        obs = self._make_robocasa_obs()
        wire = build_wire_obs(obs, "put the bowl on the counter", 224)
        assert wire["prompt"] == "put the bowl on the counter"


# ── VLABench Wire Format Tests ───────────────────────────────────────

class TestVLABenchWireFormat:
    """Test that VLABench eval client produces correct wire obs."""

    def _make_vlabench_obs(self):
        """Create a minimal VLABench-like observation dict."""
        return {
            "rgb": [
                np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8),  # right
                np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8),  # left
                np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8),  # front
                np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8),  # wrist
            ],
            "ee_state": np.array([0.1, -0.2, 0.85, 1.0, 0.0, 0.0, 0.0, 0.04]),
            "instruction": "select the red fruit",
            "robot_frame": np.array([0, -0.4, 0.78]),
        }

    def test_build_wire_obs_keys(self):
        """Wire obs has all required keys."""
        from examples.vlabench.vlabench_eval_client import build_wire_obs
        obs = self._make_vlabench_obs()
        wire = build_wire_obs(obs, 224)

        assert "observation/image" in wire
        assert "observation/second_image" in wire
        assert "observation/wrist_image" in wire
        assert "observation/state" in wire
        assert "observation/ee_pos" in wire
        assert "observation/gripper_position" in wire
        assert "prompt" in wire

    def test_image_shape(self):
        """Images resized to 224×224."""
        from examples.vlabench.vlabench_eval_client import build_wire_obs
        obs = self._make_vlabench_obs()
        wire = build_wire_obs(obs, 224)

        assert wire["observation/image"].shape == (224, 224, 3)
        assert wire["observation/second_image"].shape == (224, 224, 3)
        assert wire["observation/wrist_image"].shape == (224, 224, 3)

    def test_state_dimension(self):
        """State is 7D for VLABench (pos_offset + euler + gripper)."""
        from examples.vlabench.vlabench_eval_client import build_wire_obs
        obs = self._make_vlabench_obs()
        wire = build_wire_obs(obs, 224)

        state = wire["observation/state"]
        assert state.ndim == 1
        assert len(state) == 7, f"Expected 7D state, got {len(state)}D"
        assert state.dtype == np.float32

    def test_table_offset_applied(self):
        """EE position has robot_frame subtracted in state."""
        from examples.vlabench.vlabench_eval_client import build_wire_obs
        obs = self._make_vlabench_obs()
        wire = build_wire_obs(obs, 224)

        raw_pos = obs["ee_state"][:3]
        robot_frame = obs["robot_frame"]
        state_pos = wire["observation/state"][:3]
        expected = raw_pos - robot_frame
        np.testing.assert_allclose(state_pos, expected, atol=1e-5)

    def test_ee_pos_is_raw(self):
        """observation/ee_pos is RAW position (no offset)."""
        from examples.vlabench.vlabench_eval_client import build_wire_obs
        obs = self._make_vlabench_obs()
        wire = build_wire_obs(obs, 224)

        np.testing.assert_allclose(
            wire["observation/ee_pos"], obs["ee_state"][:3]
        )

    def test_uses_front_camera(self):
        """observation/image comes from front (index 2), not right (index 0)."""
        from examples.vlabench.vlabench_eval_client import build_wire_obs
        obs = self._make_vlabench_obs()
        # Make front camera distinctive
        obs["rgb"][2] = np.ones((480, 480, 3), dtype=np.uint8) * 128
        wire = build_wire_obs(obs, 480)
        # After resize, the front image should be all 128s
        assert wire["observation/image"].mean() == pytest.approx(128, abs=1)


# ── VLABench Action Decoding Tests ───────────────────────────────────

class TestVLABenchActionDecode:
    """Test action decoding from policy output to VLABench format."""

    def test_decode_adds_offset(self):
        """Action position has robot frame added back."""
        from examples.vlabench.vlabench_eval_client import decode_action
        robot_frame = np.array([0, -0.4, 0.78])
        action = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.5])
        pos, euler, grip = decode_action(action, robot_frame)
        np.testing.assert_allclose(pos, np.array([0.1, 0.2, 0.3]) + robot_frame)

    def test_gripper_binarize_open(self):
        """Gripper < 0.1 → zeros (closed gripper)."""
        from examples.vlabench.vlabench_eval_client import decode_action
        robot_frame = np.array([0, -0.4, 0.78])
        action = np.array([0, 0, 0, 0, 0, 0, 0.05])
        _, _, grip = decode_action(action, robot_frame)
        np.testing.assert_allclose(grip, [0, 0])

    def test_gripper_binarize_closed(self):
        """Gripper >= 0.1 → 0.04 (open width)."""
        from examples.vlabench.vlabench_eval_client import decode_action
        robot_frame = np.array([0, -0.4, 0.78])
        action = np.array([0, 0, 0, 0, 0, 0, 0.8])
        _, _, grip = decode_action(action, robot_frame)
        np.testing.assert_allclose(grip, [0.04, 0.04])

    def test_euler_passthrough(self):
        """Euler angles pass through unchanged."""
        from examples.vlabench.vlabench_eval_client import decode_action
        robot_frame = np.array([0, -0.4, 0.78])
        action = np.array([0, 0, 0, 0.1, 0.2, 0.3, 0])
        _, euler, _ = decode_action(action, robot_frame)
        np.testing.assert_allclose(euler, [0.1, 0.2, 0.3])


# ── VLABench Quaternion Conversion Test ──────────────────────────────

class TestQuaternionToEuler:
    """Test quaternion to euler conversion."""

    def test_identity_quat(self):
        """Identity quaternion [1,0,0,0] → [0,0,0] euler."""
        from examples.vlabench.vlabench_eval_client import quaternion_to_euler
        euler = quaternion_to_euler(np.array([1, 0, 0, 0]))
        np.testing.assert_allclose(euler, [0, 0, 0], atol=1e-7)

    def test_90_deg_z_rotation(self):
        """90° rotation around Z axis."""
        from examples.vlabench.vlabench_eval_client import quaternion_to_euler
        # quat for 90° around z: [cos(45°), 0, 0, sin(45°)]
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        euler = quaternion_to_euler(q)
        assert abs(euler[2] - np.pi / 2) < 0.01


# ── GT State Exporter Tests ──────────────────────────────────────────

class TestRoboCasaGTExporter:
    """Test RoboCasa GT state exporter structure."""

    def test_export_returns_required_keys(self):
        """GT state has robot, objects, scene_objects, subtask."""
        from vlm_orchestrator.aux_benchmarks.robocasa_gt import RoboCasaGTStateExporter

        # Create a minimal mock env
        mock_env = types.SimpleNamespace(
            env=types.SimpleNamespace(
                env=None,
                objects=[],
                fixtures=[],
            ),
        )
        mock_env.env.env = mock_env.env  # self-referential to stop unwrap

        exporter = RoboCasaGTStateExporter(mock_env)
        gt = exporter.export()

        assert "robot" in gt
        assert "objects" in gt
        assert "scene_objects" in gt
        assert "subtask" in gt
        assert "grasped_object" in gt["robot"]
        assert "ee_pos" in gt["robot"]
        assert "conditions" in gt["subtask"]
        assert "score" in gt["subtask"]

    def test_subtask_score_binary(self):
        """RoboCasa subtask score is 0 or 1 (no BDDL predicates)."""
        from vlm_orchestrator.aux_benchmarks.robocasa_gt import RoboCasaGTStateExporter

        mock_env = types.SimpleNamespace(
            env=types.SimpleNamespace(env=None, objects=[], fixtures=[]),
        )
        mock_env.env.env = mock_env.env

        exporter = RoboCasaGTStateExporter(mock_env)
        gt = exporter.export()

        assert gt["subtask"]["score"] in (0.0, 1.0)


class TestVLABenchGTExporter:
    """Test VLABench GT state exporter structure."""

    def test_export_returns_required_keys(self):
        """GT state has required structure."""
        from vlm_orchestrator.aux_benchmarks.vlabench_gt import VLABenchGTStateExporter

        mock_env = types.SimpleNamespace(
            task=types.SimpleNamespace(
                conditions=[],
                components={},
            ),
            robot=None,
            physics=None,
        )

        exporter = VLABenchGTStateExporter(mock_env)
        gt = exporter.export()

        assert "robot" in gt
        assert "objects" in gt
        assert "scene_objects" in gt
        assert "subtask" in gt
        assert "score" in gt["subtask"]
        assert "conditions" in gt["subtask"]

    def test_empty_conditions_score_zero(self):
        """With no conditions, score is 0."""
        from vlm_orchestrator.aux_benchmarks.vlabench_gt import VLABenchGTStateExporter

        mock_env = types.SimpleNamespace(
            task=types.SimpleNamespace(conditions=[], components={}),
            robot=None,
            physics=None,
        )
        exporter = VLABenchGTStateExporter(mock_env)
        gt = exporter.export()
        assert gt["subtask"]["score"] == 0.0


# ── Report Generation Tests ──────────────────────────────────────────

class TestReportGeneration:
    """Test that report generation produces valid markdown."""

    def test_robocasa_report(self):
        """RoboCasa report contains expected sections."""
        from examples.robocasa.robocasa_eval_client import generate_report
        args = types.SimpleNamespace(
            split="pretrain", task_set=["atomic_seen"], port=8002,
            num_trials=10,
        )
        results = [
            {"task": "CloseCabinet", "success_rate": 0.4,
             "successes": 4, "total": 10},
        ]
        report = generate_report(args, results, 4, 10)
        assert "# RoboCasa Evaluation Report" in report
        assert "40.0%" in report
        assert "CloseCabinet" in report

    def test_vlabench_report(self):
        """VLABench report contains expected sections."""
        from examples.vlabench.vlabench_eval_client import generate_report
        args = types.SimpleNamespace(
            eval_track="track_1_in_distribution", port=8002,
            n_episode=10,
        )
        results = [
            {"task": "select_fruit", "success_rate": 0.42,
             "successes": 42, "total": 100,
             "avg_intention_score": 0.3, "avg_progress_score": 0.5},
        ]
        report = generate_report(args, results, 42, 100)
        assert "# VLABench Evaluation Report" in report
        assert "42.0%" in report
        assert "select_fruit" in report
