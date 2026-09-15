# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory collector for DAgger-style fine-tuning data generation.

Records per-step observations, actions, and ground-truth labels during
orchestrated evaluation runs.  Designed to integrate into the proxy loop
with minimal overhead — stores data in memory during the episode, then
flushes to disk at episode boundaries.

The collector is **source-aware**: it tracks whether each action came
from the VLA policy or the grasp tool, and records GT state for offline
labeling (score deltas, grasp success, failure events).

Usage::

    collector = TrajectoryCollector(output_dir="results/training_data")

    # In strategy._on_new_episode():
    collector.begin_episode(episode_id=1, task_instruction="pick up ...")

    # In strategy._on_step() / proxy loop:
    collector.record_step(
        obs=obs, actions=response["actions"],
        prompt=current_prompt, action_source="policy",
        gt_state=obs.get("gt_state"),
        metadata={"subgoal_idx": 0, "is_recovery": False},
    )

    # On episode end:
    collector.end_episode(success=True, final_score=1.0)

Collected data is written as:
    <output_dir>/
        episodes/
            episode_000001/
                steps.npz        — compressed numpy arrays
                metadata.json    — episode-level info + segment labels
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Segment labeling
# ======================================================================

class ActionSource(str, Enum):
    """Where an action came from."""
    POLICY = "policy"
    GRASP_TOOL = "grasp_tool"


class SegmentLabel(str, Enum):
    """Quality label for a trajectory segment."""
    CORRECTIVE_SUCCESS = "corrective_success"   # Grasp tool → score increased
    CORRECTIVE_NEUTRAL = "corrective_neutral"   # Grasp tool → score unchanged
    CORRECTIVE_FAILURE = "corrective_failure"   # Grasp tool → score decreased
    POLICY_SUCCESS = "policy_success"           # Policy → score increased
    POLICY_NEUTRAL = "policy_neutral"           # Policy → no change
    POLICY_FAILURE = "policy_failure"           # Policy → score decreased


@dataclass
class Segment:
    """A contiguous trajectory segment with uniform action source."""
    start_step: int
    end_step: int = -1  # filled on segment close
    source: str = "policy"
    instruction: str = ""
    score_before: float = 0.0
    score_after: float = 0.0
    failure_type: str = ""  # what triggered this segment (if recovery)
    is_recovery: bool = False

    @property
    def delta_score(self) -> float:
        return self.score_after - self.score_before

    @property
    def label(self) -> SegmentLabel:
        """Auto-label based on source and score change."""
        delta = self.delta_score
        is_grasp = self.source == ActionSource.GRASP_TOOL

        if is_grasp:
            if delta > 0.01:
                return SegmentLabel.CORRECTIVE_SUCCESS
            elif delta < -0.01:
                return SegmentLabel.CORRECTIVE_FAILURE
            else:
                return SegmentLabel.CORRECTIVE_NEUTRAL
        else:
            if delta > 0.01:
                return SegmentLabel.POLICY_SUCCESS
            elif delta < -0.01:
                return SegmentLabel.POLICY_FAILURE
            else:
                return SegmentLabel.POLICY_NEUTRAL

    def to_dict(self) -> dict:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "source": self.source,
            "instruction": self.instruction,
            "score_before": round(self.score_before, 4),
            "score_after": round(self.score_after, 4),
            "delta_score": round(self.delta_score, 4),
            "label": self.label.value,
            "failure_type": self.failure_type,
            "is_recovery": self.is_recovery,
        }


# ======================================================================
# Per-step record (kept in memory, flushed at episode end)
# ======================================================================

@dataclass
class StepRecord:
    """Lightweight per-step data. Images stored as references until flush."""
    step: int
    timestamp: float

    # Observation data
    image_primary: np.ndarray | None = None     # (H, W, 3) uint8
    image_wrist: np.ndarray | None = None       # (H, W, 3) uint8, optional
    ee_pos: np.ndarray | None = None            # (3,) float64
    gripper_pos: float = 0.0

    # Action
    actions: np.ndarray | None = None           # (horizon, action_dim) float64

    # Instruction
    prompt: str = ""
    original_prompt: str = ""

    # Source + context
    action_source: str = "policy"
    subgoal_idx: int = 0
    is_recovery: bool = False
    failure_type: str = ""

    # Ground truth
    gt_score: float = 0.0
    gt_grasped: str | None = None
    gt_conditions: dict = field(default_factory=dict)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class CollectorConfig:
    """Configuration for the trajectory collector."""
    output_dir: str = "results/training_data"

    # What to save
    save_raw_images: bool = False       # Save full-res images (large!)
    save_wrist_images: bool = True      # Save wrist camera
    image_format: str = "uint8"         # "uint8" (raw) or "jpg" (compressed)

    # Filtering
    min_episode_steps: int = 10         # Skip tiny episodes
    min_score_for_success: float = 0.5  # Minimum GT score to label episode "success"

    # Memory management
    max_steps_in_memory: int = 2000     # Flush warning threshold


# ======================================================================
# Main collector
# ======================================================================

class TrajectoryCollector:
    """Records trajectory data during orchestrated evaluation runs.

    Thread-safety: NOT thread-safe.  Designed to be called from the
    single-threaded proxy loop (strategy.process() runs in one thread
    at a time via asyncio.to_thread).
    """

    def __init__(self, config: CollectorConfig | None = None):
        self.config = config or CollectorConfig()
        self._output_dir = Path(self.config.output_dir)
        self._episodes_dir = self._output_dir / "episodes"
        self._episodes_dir.mkdir(parents=True, exist_ok=True)

        # Current episode state
        self._episode_id: int = 0
        self._task_instruction: str = ""
        self._steps: list[StepRecord] = []
        self._segments: list[Segment] = []
        self._current_segment: Segment | None = None
        self._episode_active: bool = False
        self._episode_start_time: float = 0.0

        # Counters for the manifest
        self._total_episodes: int = 0
        self._total_steps: int = 0
        self._total_recovery_steps: int = 0

        # Recovery tracking
        self._num_recoveries: int = 0
        self._num_recovery_successes: int = 0

        logger.info(
            f"TrajectoryCollector initialized: output_dir={self._output_dir}"
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(
        self,
        episode_id: int,
        task_instruction: str,
    ) -> None:
        """Start recording a new episode.

        If a previous episode was not properly ended, flushes it first.
        """
        if self._episode_active:
            logger.warning(
                f"begin_episode({episode_id}) called while episode "
                f"{self._episode_id} is still active — auto-ending it"
            )
            self.end_episode(success=False, final_score=0.0)

        self._episode_id = episode_id
        self._task_instruction = task_instruction
        self._steps = []
        self._segments = []
        self._current_segment = None
        self._episode_active = True
        self._episode_start_time = time.time()
        self._num_recoveries = 0
        self._num_recovery_successes = 0

        logger.debug(
            f"[Collector] Episode {episode_id} started: "
            f"{task_instruction[:60]}"
        )

    def record_step(
        self,
        obs: dict,
        actions: np.ndarray,
        prompt: str,
        action_source: str = "policy",
        gt_state: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Record one step of data.

        Called once per proxy process() call (= once per action chunk).

        Parameters
        ----------
        obs : dict
            Raw observation dict from the eval client.
        actions : np.ndarray
            Action chunk, shape (horizon, action_dim).
        prompt : str
            Instruction currently sent to the VLA.
        action_source : str
            ``"policy"`` or ``"grasp_tool"``.
        gt_state : dict, optional
            Ground-truth state from sim (scores, grasped object, etc.).
        metadata : dict, optional
            Extra fields: subgoal_idx, is_recovery, failure_type, etc.
        """
        if not self._episode_active:
            return

        metadata = metadata or {}
        step_idx = len(self._steps)

        # Extract GT info
        gt_score = 0.0
        gt_grasped = None
        gt_conditions = {}
        if gt_state is not None:
            subtask = gt_state.get("subtask", {})
            gt_score = float(subtask.get("score", 0.0))
            gt_grasped = gt_state.get("robot", {}).get("grasped_object")
            gt_conditions = subtask.get("all_subtask_conditions", {})

        # Extract observation data
        image_primary = self._extract_image(obs)
        image_wrist = self._extract_wrist_image(obs) if self.config.save_wrist_images else None
        ee_pos = self._extract_ee_pos(obs)
        gripper_pos = self._extract_gripper(obs)

        record = StepRecord(
            step=step_idx,
            timestamp=time.time(),
            image_primary=image_primary,
            image_wrist=image_wrist,
            ee_pos=ee_pos,
            gripper_pos=gripper_pos,
            actions=np.asarray(actions, dtype=np.float64) if actions is not None else None,
            prompt=prompt,
            original_prompt=self._task_instruction,
            action_source=action_source,
            subgoal_idx=metadata.get("subgoal_idx", 0),
            is_recovery=metadata.get("is_recovery", False),
            failure_type=metadata.get("failure_type", ""),
            gt_score=gt_score,
            gt_grasped=gt_grasped,
            gt_conditions=dict(gt_conditions),
        )
        self._steps.append(record)

        # Track segments (contiguous runs of same action source)
        self._update_segments(record)

        if len(self._steps) % 100 == 0:
            logger.debug(
                f"[Collector] Episode {self._episode_id}: "
                f"{len(self._steps)} steps, score={gt_score:.3f}"
            )

    def end_episode(
        self,
        success: bool,
        final_score: float,
    ) -> None:
        """Finalize and flush the current episode to disk.

        Parameters
        ----------
        success : bool
            Whether the task was completed (all goals satisfied).
        final_score : float
            Final GT score at episode end.
        """
        if not self._episode_active:
            return

        self._episode_active = False

        # Close current segment
        if self._current_segment is not None:
            self._close_segment(len(self._steps) - 1)

        n_steps = len(self._steps)

        if n_steps < self.config.min_episode_steps:
            logger.debug(
                f"[Collector] Episode {self._episode_id} too short "
                f"({n_steps} steps), skipping"
            )
            self._steps = []
            return

        # Compute segment labels
        segment_dicts = [s.to_dict() for s in self._segments]

        # Count recovery statistics
        recovery_segments = [
            s for s in self._segments if s.is_recovery
        ]
        self._num_recoveries = len(recovery_segments)
        self._num_recovery_successes = sum(
            1 for s in recovery_segments
            if s.label == SegmentLabel.CORRECTIVE_SUCCESS
        )

        # Build episode metadata
        episode_meta = {
            "episode_id": self._episode_id,
            "task_instruction": self._task_instruction,
            "num_steps": n_steps,
            "success": success,
            "final_score": round(final_score, 4),
            "duration_s": round(time.time() - self._episode_start_time, 2),
            "num_recoveries": self._num_recoveries,
            "num_recovery_successes": self._num_recovery_successes,
            "recovery_success_rate": (
                round(self._num_recovery_successes / max(self._num_recoveries, 1), 3)
            ),
            "segments": segment_dicts,
            "num_policy_steps": sum(
                1 for s in self._steps
                if s.action_source == ActionSource.POLICY
            ),
            "num_grasp_tool_steps": sum(
                1 for s in self._steps
                if s.action_source == ActionSource.GRASP_TOOL
            ),
        }

        # Flush to disk
        try:
            self._flush_episode(episode_meta)
            self._total_episodes += 1
            self._total_steps += n_steps
            self._total_recovery_steps += episode_meta["num_grasp_tool_steps"]
        except Exception as e:
            logger.error(
                f"[Collector] Failed to flush episode "
                f"{self._episode_id}: {e}",
                exc_info=True,
            )

        # Log summary
        labels = [s.label.value for s in self._segments]
        logger.info(
            f"[Collector] Episode {self._episode_id} saved: "
            f"{n_steps} steps, score={final_score:.3f}, "
            f"success={success}, "
            f"recoveries={self._num_recoveries}, "
            f"segments={labels}"
        )

        # Free memory
        self._steps = []
        self._segments = []
        self._current_segment = None

    # ------------------------------------------------------------------
    # Segment tracking
    # ------------------------------------------------------------------

    def _update_segments(self, record: StepRecord) -> None:
        """Track action-source transitions to build segments."""
        source = record.action_source

        if self._current_segment is None:
            # First step — start a new segment
            self._current_segment = Segment(
                start_step=record.step,
                source=source,
                instruction=record.prompt,
                score_before=record.gt_score,
                is_recovery=record.is_recovery,
                failure_type=record.failure_type,
            )
            return

        # Check for source transition
        if source != self._current_segment.source:
            # Close current segment
            self._close_segment(record.step - 1)

            # Start new segment
            self._current_segment = Segment(
                start_step=record.step,
                source=source,
                instruction=record.prompt,
                score_before=record.gt_score,
                is_recovery=record.is_recovery,
                failure_type=record.failure_type,
            )

    def _close_segment(self, end_step: int) -> None:
        """Close the current segment with the score at end_step."""
        seg = self._current_segment
        if seg is None:
            return

        seg.end_step = end_step

        # Get the score at the end of this segment
        if end_step < len(self._steps):
            seg.score_after = self._steps[end_step].gt_score
        elif self._steps:
            seg.score_after = self._steps[-1].gt_score

        self._segments.append(seg)
        self._current_segment = None

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _flush_episode(self, metadata: dict) -> None:
        """Write episode data to disk as compressed numpy + JSON."""
        ep_dir = self._episodes_dir / f"episode_{self._episode_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        n = len(self._steps)

        # --- Collect arrays ---
        # Actions: (N, horizon, action_dim) — variable horizon padded
        max_horizon = max(
            (s.actions.shape[0] for s in self._steps if s.actions is not None),
            default=1,
        )
        action_dim = max(
            (s.actions.shape[1] for s in self._steps if s.actions is not None and s.actions.ndim == 2),
            default=7,
        )

        actions = np.zeros((n, max_horizon, action_dim), dtype=np.float64)
        gt_scores = np.zeros(n, dtype=np.float32)
        ee_positions = np.zeros((n, 3), dtype=np.float64)
        gripper_positions = np.zeros(n, dtype=np.float32)
        timestamps = np.zeros(n, dtype=np.float64)

        prompts = []
        action_sources = []
        subgoal_indices = []
        is_recovery_flags = []
        gt_grasped_objects = []

        for i, rec in enumerate(self._steps):
            if rec.actions is not None:
                h = min(rec.actions.shape[0], max_horizon)
                if rec.actions.ndim == 2:
                    d = min(rec.actions.shape[1], action_dim)
                    actions[i, :h, :d] = rec.actions[:h, :d]
                elif rec.actions.ndim == 1:
                    actions[i, 0, :min(len(rec.actions), action_dim)] = rec.actions[:action_dim]

            gt_scores[i] = rec.gt_score
            if rec.ee_pos is not None:
                ee_positions[i] = rec.ee_pos[:3]
            gripper_positions[i] = rec.gripper_pos
            timestamps[i] = rec.timestamp

            prompts.append(rec.prompt)
            action_sources.append(rec.action_source)
            subgoal_indices.append(rec.subgoal_idx)
            is_recovery_flags.append(rec.is_recovery)
            gt_grasped_objects.append(rec.gt_grasped or "")

        # --- Save numpy arrays (compressed) ---
        np.savez_compressed(
            str(ep_dir / "steps.npz"),
            actions=actions,
            gt_scores=gt_scores,
            ee_positions=ee_positions,
            gripper_positions=gripper_positions,
            timestamps=timestamps,
            subgoal_indices=np.array(subgoal_indices, dtype=np.int32),
            is_recovery=np.array(is_recovery_flags, dtype=bool),
        )

        # --- Save string arrays as JSON (compact) ---
        strings_data = {
            "prompts": prompts,
            "action_sources": action_sources,
            "gt_grasped_objects": gt_grasped_objects,
        }
        with open(ep_dir / "strings.json", "w") as f:
            json.dump(strings_data, f)

        # --- Save images (optional, can be large) ---
        # Save as separate npz to allow lazy loading
        images_to_save = {}
        primary_images = []
        has_primary = False
        for rec in self._steps:
            if rec.image_primary is not None:
                primary_images.append(rec.image_primary)
                has_primary = True
            else:
                # Placeholder — zero-sized marker
                primary_images.append(np.zeros((1, 1, 3), dtype=np.uint8))

        if has_primary:
            # Stack into (N, H, W, 3) — all must be same shape
            # If shapes vary (shouldn't normally), skip image saving
            shapes = set(
                img.shape for img in primary_images
                if img.shape[0] > 1
            )
            if len(shapes) == 1:
                images_to_save["image_primary"] = np.stack(
                    [img for img in primary_images if img.shape[0] > 1]
                )

        if self.config.save_wrist_images:
            wrist_images = [
                rec.image_wrist for rec in self._steps
                if rec.image_wrist is not None
            ]
            if wrist_images:
                wrist_shapes = set(img.shape for img in wrist_images)
                if len(wrist_shapes) == 1:
                    images_to_save["image_wrist"] = np.stack(wrist_images)

        if images_to_save:
            np.savez_compressed(
                str(ep_dir / "images.npz"),
                **images_to_save,
            )

        # --- Save episode metadata ---
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Compute disk size
        total_bytes = sum(
            f.stat().st_size for f in ep_dir.iterdir() if f.is_file()
        )
        logger.debug(
            f"[Collector] Flushed episode {self._episode_id}: "
            f"{n} steps, {total_bytes / 1024:.0f} KB"
        )

    # ------------------------------------------------------------------
    # Observation extraction (mirrors grasp_tool.py helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray | None:
        """Extract the primary camera image."""
        for key in (
            "observation/image",
            "observation/image_raw",
            "observation/exterior_image_1_left",
            "observation/exterior_image_1_left_raw",
        ):
            img = obs.get(key)
            if img is not None and hasattr(img, "shape"):
                return np.asarray(img, dtype=np.uint8)
        return None

    @staticmethod
    def _extract_wrist_image(obs: dict) -> np.ndarray | None:
        """Extract the wrist camera image."""
        for key in (
            "observation/wrist_image_left",
            "observation/wrist_image_left_raw",
        ):
            img = obs.get(key)
            if img is not None and hasattr(img, "shape"):
                return np.asarray(img, dtype=np.uint8)
        return None

    @staticmethod
    def _extract_ee_pos(obs: dict) -> np.ndarray | None:
        p = obs.get("observation/ee_pos")
        if p is None:
            return None
        return np.asarray(p, dtype=np.float64).flatten()[:3]

    @staticmethod
    def _extract_gripper(obs: dict) -> float:
        g = obs.get("observation/gripper_position")
        if g is None:
            return 0.0
        return float(np.asarray(g).flatten()[0])

    # ------------------------------------------------------------------
    # Summary / manifest
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return collection statistics."""
        return {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "total_recovery_steps": self._total_recovery_steps,
            "output_dir": str(self._output_dir),
        }

    def write_manifest(self) -> None:
        """Write a manifest.json summarizing all collected data."""
        manifest_path = self._output_dir / "manifest.json"

        # Scan episode directories for metadata
        episode_summaries = []
        for ep_dir in sorted(self._episodes_dir.iterdir()):
            meta_path = ep_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                episode_summaries.append({
                    "episode_id": meta["episode_id"],
                    "task": meta["task_instruction"],
                    "success": meta["success"],
                    "score": meta["final_score"],
                    "steps": meta["num_steps"],
                    "recoveries": meta["num_recoveries"],
                    "recovery_successes": meta.get("num_recovery_successes", 0),
                })

        # Compute aggregate stats
        successes = [e for e in episode_summaries if e["success"]]
        with_recovery = [e for e in episode_summaries if e["recoveries"] > 0]

        manifest = {
            "total_episodes": len(episode_summaries),
            "successful_episodes": len(successes),
            "success_rate": (
                round(len(successes) / max(len(episode_summaries), 1), 3)
            ),
            "episodes_with_recovery": len(with_recovery),
            "total_steps": sum(e["steps"] for e in episode_summaries),
            "total_recoveries": sum(
                e["recoveries"] for e in episode_summaries
            ),
            "episodes": episode_summaries,
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"[Collector] Manifest written: {manifest_path} "
            f"({len(episode_summaries)} episodes)"
        )


# ======================================================================
# Dataset filtering utilities (for offline export)
# ======================================================================


def load_episode(episode_dir: str | Path) -> dict:
    """Load a collected episode from disk.

    Returns a dict with keys: metadata, steps (npz), strings, images.
    """
    ep_dir = Path(episode_dir)

    with open(ep_dir / "metadata.json") as f:
        metadata = json.load(f)

    steps_path = ep_dir / "steps.npz"
    steps = dict(np.load(str(steps_path))) if steps_path.exists() else {}

    strings_path = ep_dir / "strings.json"
    if strings_path.exists():
        with open(strings_path) as f:
            strings = json.load(f)
    else:
        strings = {}

    images_path = ep_dir / "images.npz"
    images = dict(np.load(str(images_path))) if images_path.exists() else {}

    return {
        "metadata": metadata,
        "steps": steps,
        "strings": strings,
        "images": images,
    }


def filter_corrective_segments(
    episode_dir: str | Path,
    min_score_delta: float = 0.01,
) -> list[dict]:
    """Extract successful corrective segments from an episode.

    Returns a list of segment dicts where the grasp tool led to a
    score increase — these are the "expert corrections" for DAgger.
    """
    data = load_episode(episode_dir)
    metadata = data["metadata"]
    segments = metadata.get("segments", [])

    corrective = []
    for seg in segments:
        if (
            seg.get("source") == ActionSource.GRASP_TOOL
            and seg.get("delta_score", 0) >= min_score_delta
        ):
            corrective.append(seg)

    return corrective


def filter_successful_episodes(
    output_dir: str | Path,
    min_score: float = 0.5,
) -> list[Path]:
    """Return paths to episodes where the task succeeded.

    These contain both policy and grasp-tool steps — the full
    trajectory that achieved the goal.
    """
    episodes_dir = Path(output_dir) / "episodes"
    successful = []

    for ep_dir in sorted(episodes_dir.iterdir()):
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("success", False) or meta.get("final_score", 0) >= min_score:
            successful.append(ep_dir)

    return successful


def compute_reward_weights(
    episode_dir: str | Path,
    lookahead: int = 5,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute per-step reward weights based on GT score changes.

    For Approach B (Reward-Weighted Regression): each step gets a
    weight proportional to the discounted future score improvement.

    Parameters
    ----------
    lookahead : int
        Number of future steps to consider for score change.
    gamma : float
        Discount factor for future score improvements.

    Returns
    -------
    weights : ndarray, shape (num_steps,)
        Per-step weights ≥ 0.
    """
    data = load_episode(episode_dir)
    scores = data["steps"].get("gt_scores", np.array([]))

    if len(scores) == 0:
        return np.array([])

    n = len(scores)
    weights = np.zeros(n, dtype=np.float64)

    for t in range(n):
        # Discounted future score improvement
        future_value = 0.0
        for k in range(1, min(lookahead + 1, n - t)):
            delta = max(0.0, scores[t + k] - scores[t])
            future_value += (gamma ** k) * delta
        weights[t] = future_value

    # Normalize to [0, 1]
    max_w = weights.max()
    if max_w > 0:
        weights /= max_w

    return weights
