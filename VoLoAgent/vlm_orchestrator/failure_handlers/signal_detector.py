# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-based failure signal detector for robot manipulation.

Detects manipulation failures (drops, stalls, failed grasps) using only
action/EE signals available through the proxy, without requiring VLM
inference or ground-truth object poses.

Signals used:
  - Gripper action commands (open/close from VLA output)
  - Gripper finger joint positions (actual gripper state from observations)
  - End-effector position (z-height, speed)
  - Arm joint action magnitude (movement speed)

Supports two simulation backends:
  - **robolab** (IsaacLab/PhysX): joint-position actions, table z≈0.22
  - **LIBERO** (MuJoCo/robosuite): EE-delta actions, table z≈0.82

Per-environment thresholds are encapsulated in :class:`SignalConfig`.

Usage:
    detector = FailureSignalDetector()
    for each step:
        detector.update(gripper_action, gripper_width, ee_position, arm_actions)
    result = detector.classify()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np


# ======================================================================
# Per-environment signal configuration
# ======================================================================

@dataclass
class SignalConfig:
    """Per-environment thresholds for failure detection.

    All z-height thresholds are in world-frame metres and must be
    calibrated for each simulator's table height and robot mounting.

    The default values are calibrated for **robolab** (IsaacLab/PhysX)
    from 89 manipulation episodes across 6 tasks.
    """

    # ── Stall detection ──
    stall_z_range: float = 0.03      # EE z-range below which = barely moving
    stall_speed: float = 0.002       # EE speed below which = frozen

    # ── Height thresholds (z in world frame) ──
    table_z: float = 0.22            # EE z when gripper is at table level
    carry_z: float = 0.22            # EE z when carrying an object
    drop_z: float = 0.28             # EE z above which an open → likely drop
    place_z: float = 0.25            # EE z below which with open → placed

    # ── Gripper thresholds ──
    grip_closed_threshold: float = 0.5   # gripper_action > this → "closed"
    grip_never_closed_width: float = -0.05  # grip_width_min above this → never really closed

    # ── Timing (chunks before detection kicks in) ──
    min_chunks_stall: int = 10       # stall_frozen
    min_chunks_fumble: int = 15      # fumble (need enough grip transitions)
    min_chunks_never_gripped: int = 30  # never gripped (robot needs approach time)

    # ── Fumble thresholds ──
    fumble_transitions_high: int = 6  # transitions for fumble-at-table
    fumble_transitions_low: int = 4   # transitions for general fumble

    # ── Env identifier (informational) ──
    env: str = "robolab"


# Pre-built configs

ROBOLAB_SIGNAL_CONFIG = SignalConfig()  # defaults are robolab values

LIBERO_SIGNAL_CONFIG = SignalConfig(
    # LIBERO (robosuite) with Panda on RethinkMount.
    # Table surface is at z≈0.81 in world frame.
    # EE at rest is z≈1.05.  Typical carrying height: z≈0.88-0.95.
    # These are initial estimates — calibrate from real LIBERO runs.
    stall_z_range=0.03,
    stall_speed=0.002,
    table_z=0.82,
    carry_z=0.85,
    drop_z=0.92,
    place_z=0.84,
    grip_closed_threshold=0.5,
    grip_never_closed_width=-0.05,
    min_chunks_stall=10,
    min_chunks_fumble=15,
    min_chunks_never_gripped=30,
    fumble_transitions_high=6,
    fumble_transitions_low=4,
    env="libero",
)


def get_signal_config(env: str = "robolab") -> SignalConfig:
    """Return the :class:`SignalConfig` for the given environment."""
    if env == "libero":
        return LIBERO_SIGNAL_CONFIG
    return ROBOLAB_SIGNAL_CONFIG


class ManipulationStatus(str, Enum):
    DONE = "done"
    PROGRESS = "not_done_progress"
    FAILURE = "not_done_failure"


@dataclass
class SignalFeatures:
    """Computed features from action/observation history."""
    n_grip_transitions: int = 0
    grip_final: float = 0.0
    grip_width_min: float = 0.0
    grip_width_final: float = 0.0
    max_consecutive_closed: int = 0
    ee_z_final: float = 0.0
    ee_z_max: float = 0.0
    ee_z_range: float = 0.0
    mean_ee_speed: float = 0.0
    mean_arm_speed: float = 0.0
    descent_before_open: float = 0.0
    ee_z_at_last_open: float = 0.0
    descending_before_open: float = 0.0


@dataclass
class ClassificationResult:
    status: ManipulationStatus
    confidence: float
    reason: str
    features: SignalFeatures


class FailureSignalDetector:
    """Detects manipulation failures from action/observation signals.

    Call ``update()`` each step with current observations, then
    ``classify()`` to get the current manipulation status.

    Thresholds are environment-specific and controlled by a
    :class:`SignalConfig`.  Pass ``signal_config`` to the constructor
    to override the default (robolab).

    Robolab thresholds were calibrated on 89 manipulation episodes across
    6 tasks (banana, apple, blocks, cubes, pumpkin, yogurt) — see
    ``results/failure_eval/REPORT.md`` for details.
    """

    def __init__(
        self,
        window_size: int = 16,
        signal_config: SignalConfig | None = None,
    ):
        self.window_size = window_size
        self.config = signal_config or ROBOLAB_SIGNAL_CONFIG

        # Rolling history buffers
        self._gripper_actions: deque[float] = deque(maxlen=window_size)
        self._gripper_widths: deque[float] = deque(maxlen=window_size)
        self._ee_positions: deque[np.ndarray] = deque(maxlen=window_size)
        self._arm_actions: deque[np.ndarray] = deque(maxlen=window_size)
        self._prev_joint_pos: np.ndarray | None = None

    def reset(self) -> None:
        """Clear all history buffers."""
        self._gripper_actions.clear()
        self._gripper_widths.clear()
        self._ee_positions.clear()
        self._arm_actions.clear()
        self._prev_joint_pos: np.ndarray | None = None

    def update(
        self,
        gripper_action: float,
        gripper_width: float,
        ee_position: np.ndarray,
        arm_actions: np.ndarray | None = None,
    ) -> None:
        """Record one timestep of observations.

        Args:
            gripper_action: VLA gripper command (0=open, 1=close).
            gripper_width: Actual gripper finger width (sum of two finger
                joint positions; negative = closed, ~0 = open).
            ee_position: End-effector position [x, y, z].
            arm_actions: Arm joint actions (first 6 dims of action vector).
        """
        self._gripper_actions.append(float(gripper_action))
        self._gripper_widths.append(float(gripper_width))
        self._ee_positions.append(np.asarray(ee_position, dtype=np.float32))
        if arm_actions is not None:
            self._arm_actions.append(np.asarray(arm_actions, dtype=np.float32))

    def update_from_obs(self, obs: dict) -> None:
        """Record one timestep directly from the proxy observation dict.

        Extracts signals from standard observation keys:
          - ``observation/gripper_position``: scalar 0=open, 1=closed.
            Used as both the grip action (>0.5 = closed) and, via
            negation, as a gripper-width proxy (more negative = more
            closed, matching offline eval semantics).
          - ``observation/ee_pos``: [x, y, z] end-effector position.
          - ``observation/joint_position``: 7-dim arm joint positions.
            Differences between consecutive steps serve as arm actions.

        Gracefully skips the step if required keys are missing (e.g.
        when the robolab change to forward ``ee_pos`` has not yet been
        deployed).
        """
        ee_pos = obs.get("observation/ee_pos")
        if ee_pos is None:
            return  # can't run detector without EE position

        grip_pos = obs.get("observation/gripper_position")
        if grip_pos is None:
            return

        # Scalar extraction (arrays from msgpack may be 0-d or 1-d)
        grip_val = float(np.asarray(grip_pos).flat[0])

        # Map gripper_position (0=open, 1=closed) → gripper_width
        # Offline eval used raw finger joint sum: ~0 when open, negative
        # when closed (e.g. -1.57).  gripper_position = joint / (π/4)
        # so width = -grip_val * (π/4) gives the same sign convention.
        gripper_width = -grip_val * (np.pi / 4)

        # Arm speed: use joint position deltas as pseudo-actions
        arm_actions = None
        joint_pos = obs.get("observation/joint_position")
        if joint_pos is not None:
            joint_pos = np.asarray(joint_pos, dtype=np.float32).flatten()
            if hasattr(self, "_prev_joint_pos") and self._prev_joint_pos is not None:
                arm_actions = joint_pos[:7] - self._prev_joint_pos[:7]
            self._prev_joint_pos = joint_pos.copy()

        self.update(
            gripper_action=grip_val,
            gripper_width=gripper_width,
            ee_position=np.asarray(ee_pos, dtype=np.float32).flatten()[:3],
            arm_actions=arm_actions,
        )

    def compute_features(self) -> SignalFeatures:
        """Compute signal features from current history window."""
        if len(self._gripper_actions) < 2:
            return SignalFeatures()

        ga = np.array(self._gripper_actions)
        gw = np.array(self._gripper_widths)
        ee = np.stack(list(self._ee_positions))
        ee_z = ee[:, 2]

        grip_binary = (ga > 0.5).astype(int)
        transitions = np.abs(np.diff(grip_binary))

        # Grip transitions
        n_transitions = int(transitions.sum())

        # Max consecutive closed
        max_consec = 0
        run = 0
        for g in grip_binary:
            if g == 1:
                run += 1
                max_consec = max(max_consec, run)
            else:
                run = 0

        # EE speed
        ee_deltas = np.diff(ee, axis=0)
        ee_speed = np.linalg.norm(ee_deltas, axis=1)
        mean_ee_speed = float(ee_speed.mean()) if len(ee_speed) > 0 else 0.0

        # Arm speed
        mean_arm_speed = 0.0
        if len(self._arm_actions) > 1:
            aa = np.stack(list(self._arm_actions))
            arm_deltas = np.diff(aa, axis=0)
            mean_arm_speed = float(np.linalg.norm(arm_deltas, axis=1).mean())

        # Descent before last grip open
        close_to_open = np.where(np.diff(grip_binary) == -1)[0]
        descent_before_open = 0.0
        ee_z_at_last_open = 0.0
        descending_frac = 0.0

        if len(close_to_open) > 0:
            last_open_idx = close_to_open[-1]
            ee_z_at_last_open = float(ee_z[last_open_idx])
            pre_open = ee_z[max(0, last_open_idx - 15): last_open_idx + 1]
            if len(pre_open) > 1:
                descent_before_open = float(pre_open[0] - pre_open[-1])
                descending_frac = float(np.mean(np.diff(pre_open) < 0))

        return SignalFeatures(
            n_grip_transitions=n_transitions,
            grip_final=float(ga[-1]),
            grip_width_min=float(gw.min()),
            grip_width_final=float(gw[-1]),
            max_consecutive_closed=max_consec,
            ee_z_final=float(ee_z[-1]),
            ee_z_max=float(ee_z.max()),
            ee_z_range=float(ee_z.max() - ee_z.min()),
            mean_ee_speed=mean_ee_speed,
            mean_arm_speed=mean_arm_speed,
            descent_before_open=descent_before_open,
            ee_z_at_last_open=ee_z_at_last_open,
            descending_before_open=descending_frac,
        )

    def classify(self, chunks_elapsed: int = 999) -> ClassificationResult:
        """Classify the current manipulation status.

        Parameters
        ----------
        chunks_elapsed : int
            How many chunks have been executed on the current subgoal.
            Used to gate certain failure types that need more observation
            time (e.g. "never_gripped" shouldn't fire while the robot
            is still approaching the object).

        Returns a 3-way label (done / not_done_progress / not_done_failure)
        with a confidence score and human-readable reason string.

        Thresholds are drawn from ``self.config`` (:class:`SignalConfig`).
        Robolab defaults were calibrated on 89 episodes:
            - Failure F1: 0.750  (P=0.677, R=0.840)
            - Progress F1: 0.877  (P=0.842, R=0.914)
            - Done F1: 0.735  (P=0.900, R=0.621)
        """
        f = self.compute_features()
        c = self.config

        # === STALL DETECTION ===

        # Stall: robot barely moving vertically and slowly overall
        if (chunks_elapsed >= c.min_chunks_stall
                and f.ee_z_range < c.stall_z_range
                and f.mean_ee_speed < c.stall_speed):
            return ClassificationResult(
                ManipulationStatus.FAILURE, 0.9,
                "stall_frozen: EE barely moving (z_range={:.3f}, speed={:.4f})".format(
                    f.ee_z_range, f.mean_ee_speed),
                f,
            )

        # Stall: never gripped anything
        if (chunks_elapsed >= c.min_chunks_never_gripped
                and f.n_grip_transitions == 0 and f.max_consecutive_closed == 0
                and f.grip_width_min > c.grip_never_closed_width):
            return ClassificationResult(
                ManipulationStatus.FAILURE, 0.8,
                "stall_never_gripped: no grip transitions, gripper never closed",
                f,
            )

        # Stall: excessive oscillation at table level (fumbling)
        if (chunks_elapsed >= c.min_chunks_fumble
                and f.n_grip_transitions >= c.fumble_transitions_high
                and f.ee_z_final < c.table_z):
            return ClassificationResult(
                ManipulationStatus.FAILURE, 0.85,
                "fumble: {} grip transitions at low height ({:.3f})".format(
                    f.n_grip_transitions, f.ee_z_final),
                f,
            )

        # === PROGRESS: carrying object ===

        if (f.grip_final > c.grip_closed_threshold
                and f.ee_z_final > c.carry_z
                and f.ee_z_range > c.stall_z_range):
            return ClassificationResult(
                ManipulationStatus.PROGRESS, 0.85,
                "carrying: gripper closed, EE elevated ({:.3f}), moving".format(
                    f.ee_z_final),
                f,
            )

        # Gripper closed, elevated but not moving → stuck mid-air
        if (f.grip_final > c.grip_closed_threshold
                and f.ee_z_final > c.carry_z
                and f.ee_z_range <= c.stall_z_range):
            if f.mean_ee_speed < c.stall_speed:
                return ClassificationResult(
                    ManipulationStatus.FAILURE, 0.7,
                    "stuck_midair: gripper closed, elevated but frozen",
                    f,
                )
            return ClassificationResult(
                ManipulationStatus.PROGRESS, 0.6,
                "slow_carry: gripper closed, elevated, slow movement",
                f,
            )

        # === FUMBLE ===

        if (chunks_elapsed >= c.min_chunks_fumble
                and f.n_grip_transitions >= c.fumble_transitions_low):
            return ClassificationResult(
                ManipulationStatus.FAILURE, 0.75,
                "fumble: {} grip transitions (repeated open/close)".format(
                    f.n_grip_transitions),
                f,
            )

        # === GRIPPER OPEN: done vs drop ===

        # Descended before opening → placement
        if (f.grip_final < c.grip_closed_threshold
                and f.descent_before_open > 0.005
                and f.descending_before_open > 0.4):
            return ClassificationResult(
                ManipulationStatus.DONE, 0.7,
                "descended_placed: EE descended before opening (descent={:.3f})".format(
                    f.descent_before_open),
                f,
            )

        # Gripper opened while EE was high → transport drop
        if (f.grip_final < c.grip_closed_threshold
                and f.ee_z_at_last_open > c.drop_z):
            return ClassificationResult(
                ManipulationStatus.FAILURE, 0.65,
                "open_high: gripper opened at height {:.3f} (likely drop)".format(
                    f.ee_z_at_last_open),
                f,
            )

        # Gripper open after sustained close → likely completed placement
        if f.grip_final < c.grip_closed_threshold and f.max_consecutive_closed > 20:
            return ClassificationResult(
                ManipulationStatus.DONE, 0.5,
                "open_was_closed: gripper open after {} consecutive closed steps".format(
                    f.max_consecutive_closed),
                f,
            )

        # Gripper open, low EE → placed low
        if f.grip_final < c.grip_closed_threshold and f.ee_z_final < c.place_z:
            return ClassificationResult(
                ManipulationStatus.DONE, 0.4,
                "open_low: gripper open, EE low ({:.3f})".format(f.ee_z_final),
                f,
            )

        # Gripper closed, approaching
        if f.grip_final > c.grip_closed_threshold:
            return ClassificationResult(
                ManipulationStatus.PROGRESS, 0.5,
                "approaching: gripper closed, EE at {:.3f}".format(f.ee_z_final),
                f,
            )

        # Default
        return ClassificationResult(
            ManipulationStatus.PROGRESS, 0.3,
            "default: no strong signal detected",
            f,
        )


class CombinedDetector:
    """Combines signal-based and VLM-based failure detection.

    Three combination modes:

    ``"signal_primary"`` (default):
        Signal predictions are used as-is. VLM is never consulted.
        Best overall accuracy (79.8%) and efficiency (no VLM calls).

    ``"union_failure"``:
        If *either* signal or VLM detects failure, report failure.
        Best with 2-image VLM. Maximizes failure recall (96%) at cost
        of some false alarms. Use with limited episode budgets.

    ``"intersect_video"``:
        Signal detects failure candidates; 6-frame video VLM confirms.
        Best failure F1 (0.784) with highest precision (0.769).
        Use when false recoveries are expensive. Requires extracting
        6 frames from the last 80 steps when signal flags failure.

    From 89-case evaluation:

    +-----------------------+--------+--------+--------+---------+
    | Strategy              |  Acc   | Fail-P | Fail-R | Fail-F1 |
    +=======================+========+========+========+=========+
    | signal_primary        | 79.8%  | 0.677  | 0.840  |  0.750  |
    | union_failure (2img)  | 77.5%  | 0.615  | 0.960  |  0.750  |
    | intersect_video       | 78.7%  | 0.769  | 0.800  |  0.784  |
    | vlm_only (reference)  | 58.4%  | 0.500  | 0.440  |  0.468  |
    +-----------------------+--------+--------+--------+---------+
    """

    def __init__(
        self,
        mode: str = "signal_primary",
        window_size: int = 16,
        signal_config: SignalConfig | None = None,
    ):
        assert mode in ("signal_primary", "union_failure", "intersect_video")
        self.mode = mode
        self.signal_detector = FailureSignalDetector(
            window_size=window_size, signal_config=signal_config,
        )

    def reset(self) -> None:
        self.signal_detector.reset()

    def update(self, *args, **kwargs) -> None:
        """Pass-through to signal detector's update."""
        self.signal_detector.update(*args, **kwargs)

    def update_from_obs(self, obs: dict) -> None:
        """Pass-through to signal detector's update_from_obs."""
        self.signal_detector.update_from_obs(obs)

    @property
    def needs_vlm_check(self) -> bool:
        """Whether the current mode requires a VLM call this step.

        For ``intersect_video``, returns ``True`` only when the signal
        detector currently predicts failure — allowing the caller to
        skip VLM calls when signals look clean.
        """
        if self.mode == "signal_primary":
            return False
        if self.mode == "union_failure":
            return True  # always need VLM to catch signal-missed failures
        if self.mode == "intersect_video":
            result = self.signal_detector.classify()
            return result.status == ManipulationStatus.FAILURE
        return False

    def classify(
        self,
        vlm_prediction: str | None = None,
        chunks_elapsed: int = 999,
    ) -> ClassificationResult:
        """Classify current status, optionally incorporating VLM prediction.

        Args:
            vlm_prediction: VLM's 3-way prediction string, if available.
                One of "done", "not_done_progress", "not_done_failure",
                or None.
            chunks_elapsed: How many chunks on the current subgoal.
                Passed through to the signal detector for per-type gating.

        Returns:
            ClassificationResult with final status, confidence, and reason.
        """
        sig_result = self.signal_detector.classify(chunks_elapsed=chunks_elapsed)

        if vlm_prediction is None or self.mode == "signal_primary":
            return sig_result

        # Union mode: either detecting failure → failure
        if self.mode == "union_failure":
            if (sig_result.status == ManipulationStatus.FAILURE
                    or vlm_prediction == "not_done_failure"):
                if sig_result.status == ManipulationStatus.FAILURE:
                    return sig_result
                else:
                    return ClassificationResult(
                        ManipulationStatus.FAILURE,
                        0.5,
                        f"vlm_detected_failure: signal said "
                        f"{sig_result.status.value} ({sig_result.reason}), "
                        f"VLM overrode to failure",
                        sig_result.features,
                    )
            return sig_result

        # Intersect-video mode: signal detects, video confirms
        if self.mode == "intersect_video":
            if sig_result.status == ManipulationStatus.FAILURE:
                if vlm_prediction == "not_done_failure":
                    # Both agree → confirmed failure
                    return ClassificationResult(
                        ManipulationStatus.FAILURE,
                        min(sig_result.confidence + 0.1, 1.0),
                        f"confirmed_failure: signal ({sig_result.reason}) "
                        f"AND video VLM both detect failure",
                        sig_result.features,
                    )
                elif sig_result.confidence >= 0.8:
                    # High-confidence signal overrides video disagreement
                    return sig_result
                else:
                    # Low-confidence signal, video disagrees → trust video
                    vlm_status = ManipulationStatus(vlm_prediction)
                    return ClassificationResult(
                        vlm_status,
                        0.5,
                        f"signal_overridden: signal said failure "
                        f"({sig_result.reason}, conf={sig_result.confidence:.2f}) "
                        f"but video VLM said {vlm_prediction}",
                        sig_result.features,
                    )
            # Signal doesn't detect failure → use signal prediction
            return sig_result

        return sig_result
