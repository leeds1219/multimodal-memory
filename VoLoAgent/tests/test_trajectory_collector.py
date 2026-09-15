# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trajectory collector."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_orchestrator.utils.trajectory_collector import (
    ActionSource,
    CollectorConfig,
    Segment,
    SegmentLabel,
    TrajectoryCollector,
    compute_reward_weights,
    filter_corrective_segments,
    filter_successful_episodes,
    load_episode,
)


# ======================================================================
# Segment labeling
# ======================================================================


class TestSegmentLabeling:
    def test_corrective_success(self):
        seg = Segment(
            start_step=0, end_step=10,
            source=ActionSource.GRASP_TOOL,
            score_before=0.0, score_after=0.5,
            is_recovery=True,
        )
        assert seg.label == SegmentLabel.CORRECTIVE_SUCCESS
        assert seg.delta_score == pytest.approx(0.5)

    def test_corrective_failure(self):
        seg = Segment(
            start_step=0, end_step=5,
            source=ActionSource.GRASP_TOOL,
            score_before=0.5, score_after=0.3,
            is_recovery=True,
        )
        assert seg.label == SegmentLabel.CORRECTIVE_FAILURE

    def test_corrective_neutral(self):
        seg = Segment(
            start_step=0, end_step=5,
            source=ActionSource.GRASP_TOOL,
            score_before=0.5, score_after=0.5,
            is_recovery=True,
        )
        assert seg.label == SegmentLabel.CORRECTIVE_NEUTRAL

    def test_policy_success(self):
        seg = Segment(
            start_step=0, end_step=20,
            source=ActionSource.POLICY,
            score_before=0.0, score_after=0.25,
        )
        assert seg.label == SegmentLabel.POLICY_SUCCESS

    def test_policy_failure(self):
        seg = Segment(
            start_step=0, end_step=20,
            source=ActionSource.POLICY,
            score_before=0.5, score_after=0.2,
        )
        assert seg.label == SegmentLabel.POLICY_FAILURE

    def test_to_dict(self):
        seg = Segment(
            start_step=5, end_step=15,
            source=ActionSource.GRASP_TOOL,
            instruction="pick up the red block",
            score_before=0.0, score_after=0.5,
            is_recovery=True,
            failure_type="object_dropped",
        )
        d = seg.to_dict()
        assert d["start_step"] == 5
        assert d["end_step"] == 15
        assert d["source"] == "grasp_tool"
        assert d["label"] == "corrective_success"
        assert d["delta_score"] == pytest.approx(0.5)
        assert d["is_recovery"] is True


# ======================================================================
# Collector lifecycle
# ======================================================================


def _make_obs(
    step: int = 0,
    image_shape: tuple = (224, 224, 3),
    gt_score: float = 0.0,
    grasped: str | None = None,
) -> dict:
    """Build a fake observation dict."""
    return {
        "observation/image": np.random.randint(
            0, 255, image_shape, dtype=np.uint8,
        ),
        "observation/ee_pos": np.array([0.3 + step * 0.01, 0.0, 0.4]),
        "observation/gripper_position": np.array([0.5]),
        "gt_state": {
            "robot": {
                "grasped_object": grasped,
                "gripper_width": 0.04,
            },
            "subtask": {
                "score": gt_score,
                "all_subtask_conditions": {
                    "goal_0": gt_score >= 0.5,
                },
            },
        },
    }


def _make_actions(horizon: int = 10, dim: int = 7) -> np.ndarray:
    """Build a fake action chunk."""
    return np.random.randn(horizon, dim).astype(np.float64)


class TestCollectorLifecycle:
    def test_basic_episode(self, tmp_path):
        config = CollectorConfig(output_dir=str(tmp_path))
        collector = TrajectoryCollector(config)

        collector.begin_episode(1, "pick up the red block")

        # Record 20 policy steps with increasing score
        for i in range(20):
            score = i * 0.05
            obs = _make_obs(step=i, gt_score=score)
            actions = _make_actions()
            collector.record_step(
                obs=obs, actions=actions,
                prompt="pick up the red block",
                action_source="policy",
                gt_state=obs["gt_state"],
                metadata={"subgoal_idx": 0},
            )

        collector.end_episode(success=True, final_score=0.95)

        # Verify files were created
        ep_dir = tmp_path / "episodes" / "episode_000001"
        assert ep_dir.exists()
        assert (ep_dir / "steps.npz").exists()
        assert (ep_dir / "metadata.json").exists()
        assert (ep_dir / "strings.json").exists()

        # Verify metadata
        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["episode_id"] == 1
        assert meta["success"] is True
        assert meta["num_steps"] == 20
        assert meta["final_score"] == pytest.approx(0.95)

        # Verify steps data
        steps = dict(np.load(str(ep_dir / "steps.npz")))
        assert steps["actions"].shape[0] == 20
        assert steps["gt_scores"].shape[0] == 20

    def test_segment_tracking_source_switch(self, tmp_path):
        """Test that segments are created at action source transitions."""
        config = CollectorConfig(output_dir=str(tmp_path))
        collector = TrajectoryCollector(config)

        collector.begin_episode(1, "pick up block")

        # 10 policy steps (score 0)
        for i in range(10):
            obs = _make_obs(step=i, gt_score=0.0)
            collector.record_step(
                obs=obs, actions=_make_actions(),
                prompt="pick up block",
                action_source="policy",
                gt_state=obs["gt_state"],
            )

        # 5 grasp tool steps (score goes to 0.5)
        for i in range(5):
            score = 0.1 * (i + 1)
            obs = _make_obs(step=10 + i, gt_score=score)
            collector.record_step(
                obs=obs, actions=_make_actions(),
                prompt="pick up block",
                action_source="grasp_tool",
                gt_state=obs["gt_state"],
                metadata={"is_recovery": True},
            )

        # 5 more policy steps (score stays at 0.5)
        for i in range(5):
            obs = _make_obs(step=15 + i, gt_score=0.5)
            collector.record_step(
                obs=obs, actions=_make_actions(),
                prompt="pick up block",
                action_source="policy",
                gt_state=obs["gt_state"],
            )

        collector.end_episode(success=True, final_score=0.5)

        # Check segments
        ep_dir = tmp_path / "episodes" / "episode_000001"
        with open(ep_dir / "metadata.json") as f:
            meta = json.load(f)

        segments = meta["segments"]
        assert len(segments) == 3

        # Segment 0: policy (score 0.0 → 0.0)
        assert segments[0]["source"] == "policy"
        assert segments[0]["label"] == "policy_neutral"

        # Segment 1: grasp_tool (score 0.0 → 0.5)
        assert segments[1]["source"] == "grasp_tool"
        assert segments[1]["label"] == "corrective_success"
        assert segments[1]["is_recovery"] is True

        # Segment 2: policy (score 0.5 → 0.5)
        assert segments[2]["source"] == "policy"

    def test_short_episode_skipped(self, tmp_path):
        config = CollectorConfig(
            output_dir=str(tmp_path), min_episode_steps=10,
        )
        collector = TrajectoryCollector(config)

        collector.begin_episode(1, "task")
        for i in range(5):
            obs = _make_obs(step=i)
            collector.record_step(
                obs=obs, actions=_make_actions(),
                prompt="task", gt_state=obs["gt_state"],
            )
        collector.end_episode(success=False, final_score=0.0)

        # Episode too short — should NOT be saved
        ep_dir = tmp_path / "episodes" / "episode_000001"
        assert not ep_dir.exists()

    def test_multiple_episodes(self, tmp_path):
        config = CollectorConfig(
            output_dir=str(tmp_path), min_episode_steps=5,
        )
        collector = TrajectoryCollector(config)

        for ep_id in range(1, 4):
            collector.begin_episode(ep_id, f"task_{ep_id}")
            for i in range(10):
                obs = _make_obs(step=i, gt_score=i * 0.1)
                collector.record_step(
                    obs=obs, actions=_make_actions(),
                    prompt=f"task_{ep_id}", gt_state=obs["gt_state"],
                )
            collector.end_episode(
                success=ep_id % 2 == 0,
                final_score=0.9 if ep_id % 2 == 0 else 0.3,
            )

        assert collector.get_summary()["total_episodes"] == 3

    def test_manifest(self, tmp_path):
        config = CollectorConfig(
            output_dir=str(tmp_path), min_episode_steps=5,
        )
        collector = TrajectoryCollector(config)

        for ep_id in [1, 2]:
            collector.begin_episode(ep_id, "test task")
            for i in range(10):
                obs = _make_obs(step=i, gt_score=float(ep_id == 2))
                collector.record_step(
                    obs=obs, actions=_make_actions(),
                    prompt="test task", gt_state=obs["gt_state"],
                )
            collector.end_episode(
                success=ep_id == 2, final_score=float(ep_id == 2),
            )

        collector.write_manifest()

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["total_episodes"] == 2
        assert manifest["successful_episodes"] == 1


# ======================================================================
# Filtering utilities
# ======================================================================


class TestFiltering:
    def _create_test_episode(
        self, ep_dir: Path, score_profile: list[float],
        sources: list[str] | None = None,
    ):
        """Create a test episode with the given score profile."""
        n = len(score_profile)
        if sources is None:
            sources = ["policy"] * n

        ep_dir.mkdir(parents=True, exist_ok=True)

        actions = np.random.randn(n, 10, 7).astype(np.float64)
        gt_scores = np.array(score_profile, dtype=np.float32)

        np.savez_compressed(
            str(ep_dir / "steps.npz"),
            actions=actions,
            gt_scores=gt_scores,
            ee_positions=np.zeros((n, 3)),
            gripper_positions=np.zeros(n),
            timestamps=np.arange(n, dtype=np.float64),
            subgoal_indices=np.zeros(n, dtype=np.int32),
            is_recovery=np.array(
                [s == "grasp_tool" for s in sources], dtype=bool,
            ),
        )

        strings = {
            "prompts": ["test"] * n,
            "action_sources": sources,
            "gt_grasped_objects": [""] * n,
        }
        with open(ep_dir / "strings.json", "w") as f:
            json.dump(strings, f)

        # Build segments from source transitions
        segments = []
        seg_start = 0
        for i in range(1, n):
            if sources[i] != sources[i - 1]:
                segments.append({
                    "start_step": seg_start,
                    "end_step": i - 1,
                    "source": sources[seg_start],
                    "score_before": score_profile[seg_start],
                    "score_after": score_profile[i - 1],
                    "delta_score": score_profile[i - 1] - score_profile[seg_start],
                    "label": "corrective_success" if (
                        sources[seg_start] == "grasp_tool"
                        and score_profile[i - 1] > score_profile[seg_start]
                    ) else "policy_neutral",
                    "is_recovery": sources[seg_start] == "grasp_tool",
                    "instruction": "test",
                    "failure_type": "",
                })
                seg_start = i
        # Final segment
        segments.append({
            "start_step": seg_start,
            "end_step": n - 1,
            "source": sources[seg_start],
            "score_before": score_profile[seg_start],
            "score_after": score_profile[-1],
            "delta_score": score_profile[-1] - score_profile[seg_start],
            "label": "policy_neutral",
            "is_recovery": sources[seg_start] == "grasp_tool",
            "instruction": "test",
            "failure_type": "",
        })

        meta = {
            "episode_id": 1,
            "task_instruction": "test task",
            "num_steps": n,
            "success": score_profile[-1] >= 0.5,
            "final_score": score_profile[-1],
            "segments": segments,
            "num_recoveries": sum(
                1 for s in segments if s["is_recovery"]
            ),
            "num_recovery_successes": 0,
            "recovery_success_rate": 0.0,
            "duration_s": 10.0,
            "num_policy_steps": sum(1 for s in sources if s == "policy"),
            "num_grasp_tool_steps": sum(
                1 for s in sources if s == "grasp_tool"
            ),
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

    def test_load_episode(self, tmp_path):
        ep_dir = tmp_path / "episodes" / "episode_000001"
        self._create_test_episode(
            ep_dir, [0.0, 0.0, 0.5, 0.5, 1.0],
        )
        data = load_episode(ep_dir)
        assert "metadata" in data
        assert "steps" in data
        assert data["steps"]["gt_scores"].shape == (5,)

    def test_filter_corrective_segments(self, tmp_path):
        ep_dir = tmp_path / "episodes" / "episode_000001"
        sources = (
            ["policy"] * 5
            + ["grasp_tool"] * 5
            + ["policy"] * 5
        )
        scores = (
            [0.0] * 5
            + [0.0, 0.1, 0.2, 0.3, 0.5]  # grasp tool → score increases
            + [0.5] * 5
        )
        self._create_test_episode(ep_dir, scores, sources)

        corrective = filter_corrective_segments(ep_dir, min_score_delta=0.01)
        assert len(corrective) >= 1
        assert corrective[0]["source"] == "grasp_tool"

    def test_filter_successful_episodes(self, tmp_path):
        episodes_dir = tmp_path / "episodes"

        # Successful episode
        ep1 = episodes_dir / "episode_000001"
        self._create_test_episode(ep1, [0.0, 0.5, 1.0])

        # Failed episode
        ep2 = episodes_dir / "episode_000002"
        self._create_test_episode(ep2, [0.0, 0.1, 0.2])

        successful = filter_successful_episodes(tmp_path, min_score=0.5)
        assert len(successful) == 1
        assert successful[0].name == "episode_000001"


# ======================================================================
# Reward weights
# ======================================================================


class TestRewardWeights:
    def test_basic_weights(self, tmp_path):
        ep_dir = tmp_path / "episodes" / "episode_000001"
        # Score increases at step 5
        scores = [0.0] * 5 + [0.5] * 5
        # Create a minimal episode structure
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(ep_dir / "steps.npz"),
            gt_scores=np.array(scores, dtype=np.float32),
            actions=np.zeros((10, 10, 7)),
            ee_positions=np.zeros((10, 3)),
            gripper_positions=np.zeros(10),
            timestamps=np.arange(10, dtype=np.float64),
            subgoal_indices=np.zeros(10, dtype=np.int32),
            is_recovery=np.zeros(10, dtype=bool),
        )
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump({"episode_id": 1, "task_instruction": "test",
                        "num_steps": 10, "success": True,
                        "final_score": 0.5, "segments": [],
                        "num_recoveries": 0, "num_recovery_successes": 0,
                        "recovery_success_rate": 0.0,
                        "duration_s": 5.0,
                        "num_policy_steps": 10,
                        "num_grasp_tool_steps": 0}, f)

        weights = compute_reward_weights(ep_dir, lookahead=5, gamma=0.99)
        assert len(weights) == 10

        # Steps right before the score jump should have high weights
        # Steps after the score plateaus should have zero weights
        assert weights[0] > 0  # can see future improvement
        assert weights[9] == 0.0  # no future improvement

    def test_empty_episode(self, tmp_path):
        ep_dir = tmp_path / "episodes" / "episode_000001"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(ep_dir / "steps.npz"),
            gt_scores=np.array([], dtype=np.float32),
        )
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump({"episode_id": 1, "task_instruction": "",
                        "num_steps": 0, "success": False,
                        "final_score": 0, "segments": [],
                        "num_recoveries": 0, "num_recovery_successes": 0,
                        "recovery_success_rate": 0.0,
                        "duration_s": 0,
                        "num_policy_steps": 0,
                        "num_grasp_tool_steps": 0}, f)
        weights = compute_reward_weights(ep_dir)
        assert len(weights) == 0
