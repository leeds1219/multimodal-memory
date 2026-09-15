# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the env-agnostic per-step recorder and LeRobot conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vlm_orchestrator.utils.lerobot_converter import (
    ENV_CONFIGS,
    DROID_SIM_CONFIG,
    LIBERO_CONFIG,
    ROBOCASA_CONFIG,
    VLABENCH_CONFIG,
    EnvConfig,
    StepLevelRecorder,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_wire_obs(cfg: EnvConfig, step: int = 0) -> dict:
    """Build a fake wire-format observation dict for the given env."""
    wire: dict = {}
    for _lerobot_name, wire_key in cfg.image_keys.items():
        wire[wire_key] = np.random.randint(
            0, 255, (cfg.image_size, cfg.image_size, 3), dtype=np.uint8
        )
    wire[cfg.state_key] = np.random.rand(cfg.state_dim).astype(np.float32)
    wire["prompt"] = "do the task"
    wire["gt_state"] = {"subtask": {"score": step / 30.0}}
    return wire


def _run_episode(
    tmp_path: Path,
    cfg: EnvConfig,
    n_steps: int = 20,
    success: bool = True,
    inject_recovery: bool = False,
) -> Path:
    """Record a fake episode and return the episode directory."""
    recorder = StepLevelRecorder(str(tmp_path), env_config=cfg)
    recorder.begin_episode(0, "test task")

    for i in range(n_steps):
        wire = _make_wire_obs(cfg, step=i)
        action = np.random.rand(cfg.action_dim).astype(np.float32)

        # Simulate grasp_tool being active for middle steps
        source = "policy"
        recovery = False
        if inject_recovery and 8 <= i < 14:
            source = "grasp_tool"
            recovery = True

        recorder.record_step(
            wire_obs=wire,
            action=action,
            action_source=source,
            is_recovery=recovery,
            gt_score=i / n_steps,
        )

    recorder.end_episode(success=success)
    return tmp_path / "episode_000000"


# ── Tests: LIBERO (default) ─────────────────────────────────────────


class TestStepLevelRecorderLibero:
    def test_basic_episode(self, tmp_path):
        ep_dir = _run_episode(tmp_path, LIBERO_CONFIG, n_steps=30)
        assert ep_dir.exists()
        assert (ep_dir / "data.npz").exists()
        assert (ep_dir / "metadata.json").exists()

        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["num_frames"] == 30
        assert meta["success"] is True
        assert meta["env"] == "libero"
        assert meta["state_dim"] == 8
        assert meta["action_dim"] == 7
        assert set(meta["image_keys"]) == {"image", "wrist_image"}

        data = np.load(str(ep_dir / "data.npz"))
        assert data["actions"].shape == (30, 7)
        assert data["states"].shape == (30, 8)
        assert data["images_image"].shape == (30, 256, 256, 3)
        assert data["images_wrist_image"].shape == (30, 256, 256, 3)

    def test_mixed_sources(self, tmp_path):
        ep_dir = _run_episode(
            tmp_path, LIBERO_CONFIG, n_steps=20, inject_recovery=True
        )
        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)

        assert meta["has_recovery"] is True
        assert meta["num_recovery_frames"] == 6  # steps 8-13
        assert meta["action_sources"].count("grasp_tool") == 6
        assert meta["action_sources"].count("policy") == 14

    def test_short_episode_skipped(self, tmp_path):
        recorder = StepLevelRecorder(str(tmp_path), env_config=LIBERO_CONFIG)
        recorder.begin_episode(0, "task")

        for i in range(3):  # too short (< 5)
            wire = _make_wire_obs(LIBERO_CONFIG)
            recorder.record_step(wire_obs=wire, action=np.zeros(7))

        recorder.end_episode(success=False)
        assert not (tmp_path / "episode_000000").exists()

    def test_auto_end_on_begin(self, tmp_path):
        recorder = StepLevelRecorder(str(tmp_path), env_config=LIBERO_CONFIG)
        recorder.begin_episode(0, "task 1")

        for i in range(10):
            wire = _make_wire_obs(LIBERO_CONFIG)
            recorder.record_step(wire_obs=wire, action=np.zeros(7))

        # Start new episode without ending — should auto-end previous
        recorder.begin_episode(1, "task 2")
        assert (tmp_path / "episode_000000").exists()

    def test_missing_image_filled_with_zeros(self, tmp_path):
        recorder = StepLevelRecorder(str(tmp_path), env_config=LIBERO_CONFIG)
        recorder.begin_episode(0, "task")

        for i in range(10):
            wire = _make_wire_obs(LIBERO_CONFIG)
            # Remove wrist image for some frames
            if i % 2 == 0:
                del wire["observation/wrist_image"]
            recorder.record_step(wire_obs=wire, action=np.zeros(7))

        recorder.end_episode(success=True)

        data = np.load(str(tmp_path / "episode_000000" / "data.npz"))
        assert data["images_wrist_image"].shape == (10, 256, 256, 3)
        # Even frames should be zeros (missing wrist)
        assert data["images_wrist_image"][0].sum() == 0
        # Odd frames should have data
        assert data["images_wrist_image"][1].sum() > 0


# ── Tests: RoboCasa ──────────────────────────────────────────────────


class TestStepLevelRecorderRoboCasa:
    def test_robocasa_episode(self, tmp_path):
        ep_dir = _run_episode(tmp_path, ROBOCASA_CONFIG, n_steps=15)

        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["env"] == "robocasa"
        assert meta["state_dim"] == 16
        assert meta["action_dim"] == 7

        data = np.load(str(ep_dir / "data.npz"))
        assert data["states"].shape == (15, 16)
        assert data["actions"].shape == (15, 7)
        assert "images_image" in data
        assert "images_wrist_image" in data


# ── Tests: VLABench (3 cameras) ─────────────────────────────────────


class TestStepLevelRecorderVLABench:
    def test_vlabench_three_cameras(self, tmp_path):
        ep_dir = _run_episode(tmp_path, VLABENCH_CONFIG, n_steps=12)

        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["env"] == "vlabench"
        assert set(meta["image_keys"]) == {
            "image", "second_image", "wrist_image"
        }

        data = np.load(str(ep_dir / "data.npz"))
        assert data["images_image"].shape == (12, 256, 256, 3)
        assert data["images_second_image"].shape == (12, 256, 256, 3)
        assert data["images_wrist_image"].shape == (12, 256, 256, 3)
        assert data["states"].shape == (12, 8)


# ── Tests: DROID Sim ─────────────────────────────────────────────────


class TestStepLevelRecorderDroidSim:
    def test_droid_sim_different_keys(self, tmp_path):
        """DROID uses different wire keys and action dim."""
        ep_dir = _run_episode(tmp_path, DROID_SIM_CONFIG, n_steps=10)

        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["env"] == "droid_sim"
        assert meta["action_dim"] == 8
        assert set(meta["image_keys"]) == {
            "exterior_image_1_left", "wrist_image_left"
        }

        data = np.load(str(ep_dir / "data.npz"))
        assert data["actions"].shape == (10, 8)
        assert "images_exterior_image_1_left" in data
        assert "images_wrist_image_left" in data


# ── Tests: Response-based auto-detection ─────────────────────────────


class TestAutoDetection:
    def test_action_source_from_response(self, tmp_path):
        """action_source auto-detected from orchestrator_grasp_tool."""
        recorder = StepLevelRecorder(str(tmp_path), env_config=LIBERO_CONFIG)
        recorder.begin_episode(0, "task")

        for i in range(10):
            wire = _make_wire_obs(LIBERO_CONFIG)
            # Simulate proxy response with grasp tool active
            response = {}
            if 3 <= i < 7:
                response = {
                    "orchestrator_grasp_tool": {"active": True},
                    "orchestrator_failure": "wrong_object",
                }
            recorder.record_step(
                wire_obs=wire, action=np.zeros(7), response=response
            )

        recorder.end_episode(success=True)

        with open(tmp_path / "episode_000000" / "metadata.json") as f:
            meta = json.load(f)
        assert meta["action_sources"][3] == "grasp_tool"
        assert meta["action_sources"][7] == "policy"
        assert meta["has_recovery"] is True
        assert meta["num_recovery_frames"] == 4

    def test_gt_score_from_wire_obs(self, tmp_path):
        """gt_score auto-detected from wire_obs['gt_state']."""
        recorder = StepLevelRecorder(str(tmp_path), env_config=LIBERO_CONFIG)
        recorder.begin_episode(0, "task")

        for i in range(10):
            wire = _make_wire_obs(LIBERO_CONFIG, step=i)
            # gt_state is in the wire_obs (set by _make_wire_obs)
            recorder.record_step(wire_obs=wire, action=np.zeros(7))

        recorder.end_episode(success=True)

        data = np.load(str(tmp_path / "episode_000000" / "data.npz"))
        # Scores should increase (step / 30.0)
        assert data["gt_scores"][0] < data["gt_scores"][9]


# ── Tests: ENV_CONFIGS registry ──────────────────────────────────────


class TestEnvConfigs:
    def test_all_configs_registered(self):
        assert set(ENV_CONFIGS.keys()) == {
            "libero", "robocasa", "vlabench", "droid_sim"
        }

    def test_each_config_has_image_keys(self):
        for name, cfg in ENV_CONFIGS.items():
            assert len(cfg.image_keys) >= 2, f"{name} needs at least 2 cameras"
            assert cfg.state_dim > 0
            assert cfg.action_dim > 0
            assert cfg.fps > 0
