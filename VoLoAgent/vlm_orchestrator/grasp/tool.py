# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grasp-with-tool executor: planned grasping that bypasses the VLA.

Orchestrates the full pipeline:
  1. Detect the target object (GroundingDINO, already in proxy)
  2. Segment it (SAM2 via grasp server, or sim ground-truth mask)
  3. Build a point cloud from depth + intrinsics
  4. Predict a grasp pose (GraspGen via grasp server)
  5. Plan a trajectory (IK + interpolation, Phase 1)
  6. Serve action chunks that the proxy injects in place of VLA actions
  7. Verify grasp success and hand back to VLA

The executor is a state machine driven by ``step()`` calls from the proxy.
Each ``step()`` returns a response dict identical in format to what the VLA
would return — the eval client cannot tell the difference.

Usage (inside proxy.py)::

    executor = GraspToolExecutor()  # uses GRASP_SERVER_HOST/PORT env vars
    executor.start(target_object="red block", obs=obs, state=state)

    # In the proxy main loop:
    if state.grasp_tool_active:
        response = state.grasp_tool_executor.step(obs, state)
    else:
        # normal VLA path
        ...
"""

from __future__ import annotations

import logging
import math
import os
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from vlm_orchestrator.motion import MotionPlanner

from vlm_orchestrator.grasp.camera import (
    CameraIntrinsics,
    depth_to_pointcloud,
    extrinsics_from_obs,
    front_camera_intrinsics,
    intrinsics_from_fovy,
    libero_agentview_intrinsics,
    overshoulder_left_intrinsics,
    pose_opengl_to_opencv,
)
from vlm_orchestrator.grasp.client import GraspClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment modes
# ---------------------------------------------------------------------------

class GraspEnvMode(str, Enum):
    """Which simulator backend the grasp tool targets."""
    ROBOLAB = "robolab"  # Joint-position actions (7 joints + 1 gripper = 8D)
    LIBERO = "libero"    # EE-delta actions (6 EE delta + 1 gripper = 7D)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Action chunk size — must match Pi0DroidJointposClient.open_loop_horizon
# For robolab: 8.  For LIBERO: 5 (replan interval), but we use the policy's
# action_horizon of 10 for compatibility with the proxy's chunk serving.
ACTION_HORIZON = 8

# LIBERO uses a different action horizon (replan every 5 steps, policy
# produces 10-step chunks).  The proxy serves whatever horizon we give it.
LIBERO_ACTION_HORIZON = 10

# Action dimensions: robolab = 8 (7 joints + gripper), LIBERO = 7 (6 EE + gripper)
ACTION_DIM = 8
LIBERO_ACTION_DIM = 7

# Gripper values
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0
# LIBERO gripper: -1 = open, +1 = close (OSC_POSE convention)
LIBERO_GRIPPER_OPEN = -1.0
LIBERO_GRIPPER_CLOSE = 1.0

# NOTE: there is no gripper-pos-based grasp verify step any more.
#
# We tried both directions of comparison + several thresholds on the
# DROID gripper and the signal turned out to be unreliable: the
# actuator doesn't reach a stable settled state inside any reasonable
# CLOSE_HOLD_STEPS window, so the same observed gripper_pos can mean
# "blocked at object" or "still mid-close" depending on scene timing
# and object width.  On real robots we don't have GT contact either,
# so the only robust held-vs-empty signal is visual.
#
# The grasp tool now closes → retreats → reports DONE unconditionally;
# the tool_chain strategy's next VLM cycle inspects the post-retreat
# scene and decides "continue + place" (held) or "continue + grasp"
# (retry).  ``_last_commanded_gripper`` tracks the most recent gripper
# command so DONE / FAILED no-op chunks keep commanding the same
# thing instead of echoing the observation (which would binarize to
# OPEN and drop a held object).

# Trajectory interpolation: steps between waypoints
INTERP_STEPS_APPROACH = 40   # ~5 chunks — slow approach to pre-grasp
INTERP_STEPS_FINAL = 16      # ~2 chunks — final approach to grasp
INTERP_STEPS_RETREAT = 24    # ~3 chunks — lift after grasping

# Pre-grasp offset: back off along the grasp approach axis (metres)
PRE_GRASP_OFFSET = 0.08

# ── Flange→fingertip depths (metres), for the post-move fingertip-error audit ──
# GraspGen returns a pose whose origin is backed off the fingertip *contact
# point* by GRASPGEN_GRIPPER_DEPTH_M along −approach (Franka Panda gripper;
# graspgen_franka_panda.yml).  So GraspGen's PREDICTED fingertip contact =
# grasp_world_pos + GRASPGEN_GRIPPER_DEPTH_M * approach_axis.
GRASPGEN_GRIPPER_DEPTH_M = 0.1034
# The DROID robot mounts a Robotiq 2F-85.  observation/ee_pos reports its
# base_link flange.  The robot's ACTUAL fingertip (pad grasp-surface center) =
# ee_pos + ROBOTIQ_FLANGE_TO_FINGERTIP_M * approach_axis(ee_quat).
#
# ⚠ This is MEASURED from the sim, not the datasheet.  The Robotiq datasheet
# quotes ~162.8 mm flange→fingertip, but the flattened Isaac USD's pad mesh
# grasp-surface center sits at 131.1 mm from base_link (distal tip-END at
# 150.2 mm) — calibrated from the pad-mesh AABB via robolab droid.py's
# fingertip_frame FrameTransformer.  Using the datasheet 162.8 mm over-
# estimates the sim geometry by 31.7 mm, which showed up as a rock-constant
# 31.7 mm gap between the true-GT pad and the model-based estimate, and as a
# ~32 mm fingertip OVER-shoot when used to size the depth correction below.
ROBOTIQ_FLANGE_TO_FINGERTIP_M = 0.1311

# ── Robotiq flange-depth correction (metres) ──────────────────────────────
# GraspGen outputs a *panda_hand* flange pose: its origin sits
# GRASPGEN_GRIPPER_DEPTH_M (0.1034) behind the intended fingertip contact
# point.  That is CORRECT for a Franka panda_hand (what GraspGen was trained
# on, and what LIBERO uses).  But the DROID/robolab robot mounts a Robotiq
# 2F-85 whose MEASURED flange→pad-center is 0.1311 m, so placing the *Robotiq*
# base_link on GraspGen's panda-flange target puts the Robotiq FINGERTIPS
# (0.1311 − 0.1034) = 0.0277 m too deep along the approach axis.  To land the
# Robotiq fingertips on the contact point, shift the IK target BACK along the
# approach axis by this difference.  We do NOT touch GraspGen's gripper_depth
# (that is baked into the trained contact regression); the correction is a
# downstream, gripper-specific frame shift, applied only in ROBOLAB env mode.
#
# ⚠ CALIBRATION HISTORY: the first version used the datasheet 0.1628 m →
# 0.0594 m correction, which OVER-shot: the true-GT pad audit then showed a
# consistent ~−38 mm fingertip UNDERSHOOT (too shallow) because the flange was
# pulled back 31.7 mm too far.  The measured 0.1311 m (sim pad geometry) gives
# the correct 0.0277 m shift.  Env-var override lets the audit sweep it
# (0 = disabled).
ROBOTIQ_DEPTH_CORRECTION_M = ROBOTIQ_FLANGE_TO_FINGERTIP_M - GRASPGEN_GRIPPER_DEPTH_M
_robotiq_corr_env = os.environ.get("GRASP_ROBOTIQ_DEPTH_CORRECTION_M")
if _robotiq_corr_env is not None:
    ROBOTIQ_DEPTH_CORRECTION_M = float(_robotiq_corr_env)

# ── Clean-model IK toggle (ROBOLAB) ───────────────────────────────────────
# When True (default), the ROBOLAB grasp path targets the Robotiq **base_link**
# frame directly, using the empirically-measured joint7→base_link mount
# transform (ik.T_JOINT7_TO_ROBOTIQ_BASE).  This makes forward_kinematics
# output the true sim-controlled frame, so:
#   * the per-grasp ``fk_vs_ee`` correction collapses to ~0 (base == world),
#   * the panda-hand approximation (FLANGE_Z_M=0.107 + HAND_YAW_RAD=-π/2 +
#     its orientation-dependent ~5–15 mm residual leak) is retired,
#   * the depth correction stays as the single explicit fingertip shift.
# Set GRASP_ROBOLAB_CLEAN_IK=0 to fall back to the legacy panda-hand path
# (kept for A/B verification / rollback).
GRASP_ROBOLAB_CLEAN_IK = (
    os.environ.get("GRASP_ROBOLAB_CLEAN_IK", "1") not in ("0", "false", "False")
)

# Retreat offset: lift straight up in world frame (metres)
RETREAT_HEIGHT = 0.10

# Safe height: lift arm to this z before perception (clears camera view)
SAFE_HEIGHT_Z = 0.45

# Settle delay: wait this many control steps before starting perception,
# so a recently dropped/placed object has time to land on the table and
# gt_state reflects its resting position.  At 15 Hz, 10 steps ≈ 0.67s.
SETTLE_STEPS = 10

# ── TESTING/DIAGNOSTICS: final-waypoint settle hold ──────────────────
# The robolab grasp path plays an OPEN-LOOP joint-space trajectory whose last
# waypoint is q_grasp.  Setting GRASP_SETTLE_HOLD_STEPS > 0 re-commands q_grasp
# for that many extra sim steps BEFORE measuring, so we can read the settled
# pose.  Default 0 preserves production behaviour exactly; diagnostics knob only.
#
# ⚠ DIAGNOSIS UPDATE: the undershoot is NOT ramp-tracking lag (a settle-hold
# sweep of 0/20/60/120 steps showed it does NOT decay — it slightly worsens).
# Ground-truth joint logging (post_move_joint_err_rad) shows the settled joints
# sit 1.3–2.3° SHORT of commanded q_grasp on the gravity-loaded joints (j2/j4/j6)
# → it is a joint-space P-controller STEADY-STATE error (err ≈ τ_gravity/Kp,
# Kp=400, no integral term), which maps to ~20–26 mm Cartesian undershoot.
# Re-commanding the same setpoint (settle hold) cannot fix a steady-state error;
# the fix is INTEGRAL action — see GRASP_FINAL_INTEGRAL_* below.
try:
    GRASP_SETTLE_HOLD_STEPS = int(os.environ.get("GRASP_SETTLE_HOLD_STEPS", "0"))
except (ValueError, TypeError):
    GRASP_SETTLE_HOLD_STEPS = 0

# ── Final-approach integral correction (ROBOLAB) ─────────────────────
# Fixes the joint-space P-controller steady-state undershoot (see above) by
# accumulating the joint error (q_grasp − q_measured) over a few sim steps and
# ADDING it to the commanded joints, so the controller is driven past q_grasp
# until the settled pose reaches it.  This is a discrete integral term applied
# in the orchestrator (we cannot change the sim controller's gains).
#
#   q_cmd(k+1) = q_grasp + Ki * Σ (q_grasp − q_measured(k))
#
# Ki<1 for stability (the sim controller itself has a fast response, so we
# under-relax).  Runs for GRASP_FINAL_INTEGRAL_STEPS sim steps or until the max
# joint error falls below GRASP_FINAL_INTEGRAL_TOL_RAD.  Gripper stays OPEN
# (same as FINAL_APPROACH).  Default ON for ROBOLAB; set
# GRASP_FINAL_INTEGRAL=0 to disable (falls back to the open-loop endpoint).
GRASP_FINAL_INTEGRAL = (
    os.environ.get("GRASP_FINAL_INTEGRAL", "1") not in ("0", "false", "False")
)
try:
    GRASP_FINAL_INTEGRAL_STEPS = int(
        os.environ.get("GRASP_FINAL_INTEGRAL_STEPS", "80")
    )
except (ValueError, TypeError):
    GRASP_FINAL_INTEGRAL_STEPS = 80
try:
    GRASP_FINAL_INTEGRAL_KI = float(
        os.environ.get("GRASP_FINAL_INTEGRAL_KI", "1.0")
    )
except (ValueError, TypeError):
    GRASP_FINAL_INTEGRAL_KI = 1.0
try:
    GRASP_FINAL_INTEGRAL_TOL_RAD = float(
        os.environ.get("GRASP_FINAL_INTEGRAL_TOL_RAD", "0.003")  # ~0.17°
    )
except (ValueError, TypeError):
    GRASP_FINAL_INTEGRAL_TOL_RAD = 0.003
# Safety clamp: never push the integral command more than this past q_grasp on
# any joint (prevents runaway if a joint is mechanically blocked, e.g. contact).
try:
    GRASP_FINAL_INTEGRAL_MAX_RAD = float(
        os.environ.get("GRASP_FINAL_INTEGRAL_MAX_RAD", "0.15")  # ~8.6°
    )
except (ValueError, TypeError):
    GRASP_FINAL_INTEGRAL_MAX_RAD = 0.15
# Anti-windup: if the max joint error stops improving for this many consecutive
# integral updates, the arm has hit a mechanical constraint (contact with the
# object/table) — pushing harder just winds up the accumulator and drives the
# pose WORSE (observed: a contact grasp wound to 10° and ended 34 mm off).  Stop
# and keep the BEST pose seen instead of the wound-up one.
try:
    GRASP_FINAL_INTEGRAL_STALL_UPDATES = int(
        os.environ.get("GRASP_FINAL_INTEGRAL_STALL_UPDATES", "3")
    )
except (ValueError, TypeError):
    GRASP_FINAL_INTEGRAL_STALL_UPDATES = 3

# Known-good joint configuration for perception: arm retracted near rest
# pose so the over-shoulder camera has a clear view of the workspace.
# EE lands at approximately [0.39, 0, 0.46] — above the table, out of
# the camera's line of sight.
PERCEPTION_JOINTS = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])

# ── LIBERO EE-delta trajectory constants ──
# OSC_POSE limits: position delta capped at ±0.05 m per step,
# orientation delta at ±0.5 rad per step.
LIBERO_MAX_POS_DELTA = 0.05     # m — full OSC_POSE per-step limit
LIBERO_MAX_ORI_DELTA = 0.5      # rad — full OSC_POSE per-step limit
# Step counts sized for closed-loop motion: must satisfy
#   max_segment_distance / LIBERO_MAX_POS_DELTA  <=  N_steps
#   max_orientation_delta / LIBERO_MAX_ORI_DELTA <=  N_steps
# but no larger, since extra steps just slow the grasp pipeline down.
# With MAX_ORI_DELTA = 0.5, worst-case π rad rotation needs ≥ 7 steps;
# typical libero distances: approach ≤ 30 cm → ≥ 6, final ≤ 6 cm → ≥ 2,
# retreat ≤ 8 cm → ≥ 2.
LIBERO_INTERP_APPROACH = 8      # steps for approach phase
LIBERO_INTERP_FINAL = 3         # steps for final approach
LIBERO_INTERP_RETREAT = 4       # steps for retreat
LIBERO_RETREAT_HEIGHT = 0.08    # m to lift after grasping
LIBERO_PRE_GRASP_OFFSET = 0.06  # m above grasp pose

# Perception-lift parameters for LIBERO. Without this lift, the robot arm
# itself sits between the agentview camera and the workspace, occluding the
# target object. A small upward + backward EE motion before perception keeps
# the arm out of the camera's line of sight. (The agentview camera is fixed
# above and in front of the robot base — pulling the EE backward toward the
# base + upward clears the FOV.)
LIBERO_PERCEPTION_LIFT_DZ = 0.25    # m up from current EE
LIBERO_PERCEPTION_LIFT_DX = -0.12   # m back toward robot base (world x)
LIBERO_INTERP_PERCEPTION_LIFT = 6   # steps (≥ Δ/MAX_DELTA per axis)

# Axis-convention correction: GraspGen's grasp frame uses
#   +X = between fingers (open direction), +Y = perpendicular, +Z = approach.
# Standard Franka panda_hand (LIBERO's EE body) uses
#   +X = perpendicular, +Y = between fingers,         +Z = approach.
# So commanding the EE to GraspGen's grasp_rot directly puts the gripper
# 90° off about its approach axis.  Compose grasp_rot @ Rz(yaw) before
# computing the world-frame delta so the EE's "between-fingers" axis
# (panda_hand +Y) ends up aligned with GraspGen's "between-fingers"
# axis (grasp +X).  In robolab this offset was absorbed by HAND_YAW_RAD
# in the IK; LIBERO has no IK in this path, so apply it explicitly.
LIBERO_GRASPGEN_TO_EE_YAW_RAD = np.pi / 2

# Closed-loop phase-done threshold: when EE is within this distance of
# the current segment's target, advance to the next phase.  Coarsened
# from 15 mm to 25 mm — with closed-loop replanning the next phase
# corrects any residual error, so we don't need to converge tightly
# at every segment boundary.
LIBERO_PHASE_DONE_M = 0.025         # 25 mm

# OSC_POSE scaling: robosuite OSC_POSE expects actions in [-1, 1] which
# it scales internally to ±OSC_POSE_POS_SCALE m for position and
# ±OSC_POSE_ORI_SCALE rad for orientation per env step (default robosuite
# config).  Our planner computes motion in metres / radians, so we divide
# by these scales before emitting actions.  Without this conversion,
# sending raw metres (e.g. 0.04) is interpreted as a unit-fraction and
# produces ~2 mm of motion per step.
OSC_POSE_POS_SCALE = 0.05       # m per unit input action
OSC_POSE_ORI_SCALE = 0.5        # rad per unit input action

# Interpolation steps for lift-to-safe
INTERP_STEPS_LIFT = 24  # ~3 chunks

# Cap on obstacle-cloud size fed to the collision-aware motion planner.
# The server meshes the cloud, so a few thousand points is plenty; more
# just slows the mesh build.  Random-subsampled when exceeded.
MAX_OBSTACLE_POINTS = 4096

# Master switch for the collision-free (scene-aware) planning path.  When
# False (default), cuRobo plans joint→joint in an EMPTY world — i.e. "plain
# cuRobo", identical trajectories to the linear planner's endpoints but with
# cuRobo's smoothing/optimization, and NO scene point cloud, NO finger-link
# disabling, NO attached-object.  Set GRASP_COLLISION_FREE=1 to re-enable the
# scene-aware obstacle avoidance (currently under debugging — see
# workspace/curobo_finger_fix_status.md).
COLLISION_FREE_PLANNING = os.environ.get("GRASP_COLLISION_FREE", "0") == "1"

# How many sim steps to hold the gripper closed before checking success.
# Picked to match the prior "1 chunk" hold for both robolab (chunk=8)
# and LIBERO (chunk=10) — 10 sim steps ≈ 0.7 s at 15 Hz — enough for
# the Panda gripper to fully close on the object.  Step-based since
# multi-VLA support; chunk-size invariant.
CLOSE_HOLD_STEPS = 10

# How many sim steps to hold gripper open before lifting (release a held
# object).  ≈ 0.7 s at 15 Hz — enough for fingers to open.
RELEASE_HOLD_STEPS = 10


class GraspPhase(str, Enum):
    """State machine phases for the grasp executor."""
    IDLE = "idle"
    RELEASING = "releasing"           # open gripper to drop held object
    SETTLING = "settling"             # wait for object to settle
    LIFTING = "lifting"              # lift arm to clear camera view
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    APPROACHING = "approaching"        # moving to pre-grasp
    FINAL_APPROACH = "final_approach"  # pre-grasp → grasp
    CLOSING = "closing"               # close gripper + hold
    RETREATING = "retreating"         # lift with object
    SETTLE_HOLD = "settle_hold"       # TESTING: hold final waypoint to let ramp-lag decay
    INTEGRAL_SETTLE = "integral_settle"  # ROBOLAB: integral correction of P-controller undershoot
    MEASURING = "measuring"           # one-shot: capture post-move metrics, then DONE
    DONE = "done"
    FAILED = "failed"


class GraspSegMode(str, Enum):
    """Segmentation backend for the grasp pipeline."""
    GDINO_SAM2 = "gdino_sam2"    # Two-stage: GDino detect → SAM2 segment
    SAM3 = "sam3"                 # Single-stage: SAM3 detect+segment
    MOLMO_SAM2 = "molmo_sam2"    # Two-stage: Molmo2 point → SAM2 segment
    VLM_SAM2 = "vlm_sam2"        # Two-stage: orchestrator VLM point → SAM2 segment
    GT_SIM = "gt_sim"            # Ground-truth mask from simulator


def _build_obstacle_cloud(pc_world, keep_mask, motion_planner):
    """Build a subsampled world-frame obstacle cloud for collision planning.

    Returns ``None`` when the active planner is not collision-aware (linear),
    so free-space planners pay zero cost.  Otherwise returns the masked,
    downsampled ``(M, 3)`` cloud (``keep_mask`` selects obstacle points, i.e.
    scene MINUS the target object for grasp).  Returns ``None`` if empty.
    """
    # Only collision-aware planners consume a scene cloud.  Linear ignores it,
    # so skip the (small) masking/subsample cost entirely.  Also skipped when
    # the collision-free path is disabled (plain-cuRobo mode).
    if not COLLISION_FREE_PLANNING:
        return None
    if getattr(motion_planner, "name", "linear") == "linear":
        return None
    pts = np.asarray(pc_world)[np.asarray(keep_mask, dtype=bool)]
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] > MAX_OBSTACLE_POINTS:
        idx = np.random.choice(pts.shape[0], MAX_OBSTACLE_POINTS, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


class GraspToolExecutor:
    """Drives a planned grasp sequence, producing VLA-compatible action chunks.

    Supports two environment backends:

    ``robolab`` (default)
        Joint-position actions (7 joints + 1 gripper = 8D).
        Uses ``franka_ik.py`` for inverse kinematics.

    ``libero``
        EE-delta actions (6 Cartesian delta + 1 gripper = 7D).
        Uses direct Cartesian interpolation (no IK needed) to produce
        OSC_POSE-compatible actions.

    Set ``env_mode`` in the constructor or via the ``--env`` CLI flag.
    """

    def __init__(
        self,
        grasp_server_url: str | None = None,
        intrinsics: CameraIntrinsics | None = None,
        seg_mode: str | GraspSegMode = GraspSegMode.GDINO_SAM2,
        env_mode: str | GraspEnvMode = GraspEnvMode.ROBOLAB,
        use_front_camera: bool = False,
        topdown_threshold: float | None = None,
        motion_planner: "MotionPlanner | None" = None,
        stack_mode_enabled: bool = False,
    ):
        self._grasp_client = GraspClient(url=grasp_server_url)
        self._seg_mode = GraspSegMode(seg_mode)
        self._env_mode = GraspEnvMode(env_mode)
        self._use_front_camera = use_front_camera
        # Motion planner for joint-space trajectory segments.  Defaults to
        # straight linear interpolation (historical behaviour); a
        # collision-aware planner can be injected without changing pose
        # selection / IK.
        if motion_planner is None:
            from vlm_orchestrator.motion import LinearInterpPlanner
            motion_planner = LinearInterpPlanner()
        self._motion_planner = motion_planner
        # Strict top-down filter (dot-product cutoff in [0, 1]).  Tighter
        # values reject Contact-GraspNet candidates whose approach axis
        # tilts away from gravity.  ~0.85 ≈ within 32° of vertical.
        # `None` falls through to the server's default (0.3, permissive).
        self._topdown_threshold = topdown_threshold

        # Master switch for stack/no-stack grasp coupling (CLI
        # --enable-stack-mode).  When ON, a per-grasp ``stack`` arg controls
        # top-down filtering: stack=True keeps the top-down filter (object is
        # held in a controlled pose for precise stacking); stack=False DISABLES
        # the filter entirely (drop-from-above placement, so grasp orientation
        # is irrelevant → keep all GraspGen candidates → higher grasp success).
        # When OFF the ``stack`` arg is ignored and the fixed ``topdown_threshold``
        # applies to every grasp — zero behaviour change.
        self._stack_mode_enabled = bool(stack_mode_enabled)
        # Per-grasp stack intent, set by ``start(stack=...)``.  Default True
        # preserves the historical top-down-filtered behaviour.
        self._stack_this_grasp: bool = True

        # Set env-appropriate defaults
        if self._env_mode == GraspEnvMode.LIBERO:
            self._intrinsics = intrinsics or libero_agentview_intrinsics()
            self._action_horizon = LIBERO_ACTION_HORIZON
            self._action_dim = LIBERO_ACTION_DIM
            self._gripper_open = LIBERO_GRIPPER_OPEN
            self._gripper_close = LIBERO_GRIPPER_CLOSE
        else:
            self._intrinsics = intrinsics or overshoulder_left_intrinsics()
            self._action_horizon = ACTION_HORIZON
            self._action_dim = ACTION_DIM
            self._gripper_open = GRIPPER_OPEN
            self._gripper_close = GRIPPER_CLOSE

        # ---- per-grasp state ----
        self._phase: GraspPhase = GraspPhase.IDLE
        self._target_object: str = ""
        self._status_message: str = ""

        # Track the last gripper command we sent.  Used by
        # ``_noop_response`` (which fires when phase is DONE / FAILED
        # / unexpected) to keep the gripper in the state we last
        # commanded — otherwise echoing the observation through the
        # eval client's binarizer ``action[-1] > 0.5`` would flip a
        # held-but-only-partially-closed gripper to OPEN and drop the
        # object.  Set every time we tile / build an action.
        self._last_commanded_gripper: float = self._gripper_open
        self._last_obs: dict | None = None  # stash obs for calibration

        # Perception results
        self._grasp_pose_camera: np.ndarray | None = None  # 4×4 in camera frame
        self._grasp_confidence: float = 0.0
        self._intrinsics_used: CameraIntrinsics | None = None
        self._grasp_log: dict = {}
        # World-frame obstacle cloud (scene minus target) for collision-aware
        # motion planning; None for linear planner or when no cloud is built.
        self._obstacle_pc_world: np.ndarray | None = None
        # Target-object cloud (world frame) captured for collision-aware PLACE
        # (handed to the place tool via SessionState).  None for linear planner.
        self._target_pc_world: np.ndarray | None = None

        # Trajectory (list of (action_dim,) arrays)
        self._trajectory: list[np.ndarray] = []
        self._traj_cursor: int = 0

        # Trajectory segments (loaded per-phase)
        self._seg_approach: list[np.ndarray] = []
        self._seg_final: list[np.ndarray] = []
        self._q_at_grasp: np.ndarray | None = None

        # LIBERO closed-loop motion-control state.  Populated by
        # _plan_trajectory_libero / _plan_perception_lift_libero;
        # consumed by _step_trajectory_libero_closed_loop.
        self._libero_pre_grasp_pos: np.ndarray | None = None
        self._libero_grasp_pos: np.ndarray | None = None
        self._libero_seg_target_pos: np.ndarray | None = None
        # Target rotation matrix (3×3, world-frame) for closed-loop
        # orientation tracking.  None means "hold current orientation"
        # (e.g. perception lift, retreat).
        self._libero_seg_target_rot: np.ndarray | None = None
        # Open-loop fallback orientation delta (only used when target_rot
        # is None and we still want a small fixed rotation per step).
        self._libero_seg_ori_per_step_norm: np.ndarray = np.zeros(3)
        self._libero_seg_remaining_steps_estimate: int = 0

        # LIBERO-specific: EE pose at grasp (for retreat planning)
        self._ee_at_grasp: np.ndarray | None = None

        # Calibrated tool offset
        self._T_flange_to_tcp: np.ndarray | None = None

        # Sim step at which the close-gripper hold was entered.  Hold
        # ends when ``state.episode_step - _step_at_close_start
        # >= CLOSE_HOLD_STEPS``.  Step-delta gating keeps the hold
        # duration consistent across VLAs (was per-chunk previously,
        # which drifted with chunk size).
        self._step_at_close_start: int = -CLOSE_HOLD_STEPS

        # Sim step at which the release-gripper hold was entered.
        self._step_at_release_start: int = -RELEASE_HOLD_STEPS
        # Sim step at which the (testing-only) settle hold was entered.
        self._step_at_settle_start: int = 0
        self._skip_lift_after_release: bool = True

        # Final-approach integral correction state (ROBOLAB).  Accumulates the
        # joint error to drive out the P-controller steady-state undershoot.
        self._step_at_integral_start: int = 0
        self._integral_accum: np.ndarray | None = None  # (7,) accumulated error
        self._integral_best_err: float = np.inf   # best (min) max-err seen
        self._integral_stall_updates: int = 0      # consecutive non-improving updates

        # Settle counter (wait for object to land before perception)
        self._settle_remaining: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> GraspPhase:
        return self._phase

    @property
    def is_active(self) -> bool:
        return self._phase not in (GraspPhase.IDLE, GraspPhase.DONE, GraspPhase.FAILED)

    @property
    def status_message(self) -> str:
        return self._status_message

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(
        self,
        target_object: str,
        obs: dict,
        state,
        stack: bool = True,
    ) -> None:
        """Kick off a planned grasp for *target_object*.

        ``stack`` is advisory and only honoured when the executor was
        constructed with ``stack_mode_enabled=True``.  ``stack=False`` means
        the object will be dropped from above at place time, so its grasp
        orientation does not matter — the top-down filter is disabled to keep
        all GraspGen candidates (higher grasp success on awkward geometries).
        ``stack=True`` (default) keeps the top-down filter for a controlled
        hold suitable for precise stacking.

        This runs the perception pipeline **synchronously** (detection →
        segmentation → depth → grasp prediction).  It blocks the proxy
        loop for a few seconds but that's fine because the robot is
        effectively paused during perception anyway.

        After this returns, the executor is in APPROACHING phase and
        subsequent ``step()`` calls will serve the planned trajectory.
        """
        self._target_object = target_object
        self._stack_this_grasp = bool(stack)
        self._status_message = f"Lifting arm for '{target_object}'..."
        self._last_obs = obs
        logger.info(f"grasp_with_tool: start for '{target_object}'")
        if self._stack_mode_enabled:
            logger.info(
                f"  stack-mode ON: stack={self._stack_this_grasp} → "
                f"top-down filter {'ENABLED' if self._stack_this_grasp else 'DISABLED'}"
            )

        # Get HITL handle for debug visualization (may be None)
        hitl = getattr(state, "_hitl", None)
        if hitl is None:
            hitl = getattr(state, "hitl", None)

        # By default, lift the arm to a known perception pose before
        # grasping — this clears the camera view and gives a reliable
        # point cloud.  In HITL mode the operator can toggle this off
        # (skip the lift) when the arm is already out of the way.
        skip_lift = False
        if hitl is not None:
            skip_lift = getattr(hitl, "skip_pre_grasp_lift", False)

        try:
            # ---- Release gripper if holding something ----
            # Unconditionally release first (~RELEASE_HOLD_STEPS sim
            # steps with the gripper opened).  Cheap insurance against
            # the previous grasp tool leaving the gripper closed on
            # an object we want to drop before approaching the new
            # target.  This also keeps grasp-tool behavior
            # deterministic without relying on gripper_pos as a
            # held/empty signal (which is unreliable on robolab's
            # DROID gripper and unavailable on real robots).
            gripper_pos = self._extract_gripper(obs)
            logger.info(
                f"  Starting grasp (current gripper_pos={gripper_pos:.3f}); "
                f"releasing first"
            )
            self._step_at_release_start = state.episode_step
            self._phase = GraspPhase.RELEASING
            self._status_message = "Releasing held object..."
            # Stash skip_lift so _step_releasing can resume correctly
            self._skip_lift_after_release = skip_lift
            return

            if self._env_mode == GraspEnvMode.LIBERO:
                # LIBERO: no joint-space IK available, but the agentview
                # camera is FIXED in front of and above the robot — the
                # arm itself sits between the camera and the workspace
                # and blocks the target object. Use OSC_POSE EE-delta
                # to lift the arm up and back before perception, so
                # GDino / SAM see the unobstructed scene.
                ee_pos_now = self._libero_extract_ee_pos(obs)
                if ee_pos_now is None:
                    logger.warning(
                        "  LIBERO perception lift skipped — no ee_pos in obs"
                    )
                    self._start_perception(obs, state)
                    return
                self._plan_perception_lift_libero(ee_pos_now)
                if self._trajectory:
                    self._phase = GraspPhase.LIFTING
                    self._traj_cursor = 0
                    self._status_message = (
                        f"LIBERO: lifting arm out of camera view "
                        f"for '{target_object}'..."
                    )
                    logger.info(
                        f"  LIBERO perception lift: "
                        f"{len(self._trajectory)} EE-delta waypoints"
                    )
                    return
                logger.info(
                    "  LIBERO perception lift produced no waypoints "
                    "— perceiving from current pose"
                )
                self._start_perception(obs, state)
                return

            if skip_lift:
                # ---- Operator chose to skip the lift ----
                logger.info(
                    "  skip_pre_grasp_lift is ON — "
                    "perceiving from current arm position"
                )
                self._start_perception(obs, state)
                return

            # ---- 0. Move arm to perception pose first ----
            # Always move to a known retracted joint config so the
            # over-shoulder camera has a clear, unoccluded view.
            current_joints = self._extract_joints(obs)
            self._plan_lift_to_safe(current_joints)

            if self._trajectory:
                self._phase = GraspPhase.LIFTING
                self._traj_cursor = 0
                self._status_message = (
                    f"Moving arm to perception pose for '{target_object}'..."
                )
                logger.info(
                    f"  Moving to perception pose: "
                    f"{len(self._trajectory)} waypoints "
                    f"({len(self._trajectory) // self._action_horizon} chunks)"
                )
                # Perception will start after lift completes
                # (triggered in _advance_phase_after_trajectory)
                return
            else:
                # Already at perception pose — go straight to perception
                logger.info("  Arm already at perception pose, skipping lift")
                self._start_perception(obs, state)
        except Exception as e:
            logger.error(f"grasp_with_tool lift failed: {e}", exc_info=True)
            self._phase = GraspPhase.FAILED
            self._status_message = f"Failed: {e}"

    def _start_perception(self, obs: dict, state) -> None:
        """Run the perception pipeline (detection → segmentation → grasp).

        Called either after the lift completes or directly if the arm is
        already at a safe height.
        """
        target_object = self._target_object
        self._phase = GraspPhase.PERCEIVING
        self._status_message = f"Detecting '{target_object}'..."
        self._last_obs = obs

        # Allow debug visualization to stash images on the state.
        # For LIBERO the grasp tool internally un-mirrors images
        # (``_extract_image`` applies ``[:, ::-1]``) but the main eval
        # video shows the un-un-mirrored frame, so flip the PIP back
        # before stashing so it visually matches.
        from vlm_orchestrator.grasp import debug as dbg
        dbg.set_state(
            state,
            pip_x_flip=(self._env_mode == GraspEnvMode.LIBERO),
        )

        hitl = getattr(state, "_hitl", None)
        if hitl is None:
            hitl = getattr(state, "hitl", None)

        try:
            from vlm_orchestrator.grasp import debug as dbg

            # ---- 1. Get current robot state from obs ----
            if self._env_mode == GraspEnvMode.LIBERO:
                # LIBERO: we don't have joint positions, only EE state
                current_joints = None
                current_gripper = self._extract_gripper(obs)
                current_ee_pos = self._extract_ee_pos(obs)
                current_ee_quat = self._extract_ee_quat(obs)
            else:
                current_joints = self._extract_joints(obs)
                current_gripper = self._extract_gripper(obs)
                current_ee_pos = self._extract_ee_pos(obs)
                current_ee_quat = None

            # ---- 2. Get depth + build point cloud ----
            depth = self._extract_depth(obs)
            if depth is None:
                raise RuntimeError(
                    "No depth data in observation. Ensure camera is "
                    "configured with data_types=['rgb', 'depth'] and "
                    "depth is forwarded through the eval client."
                )
            # Select intrinsics matching the active camera.
            # Prefer the runtime camera_K matrix from obs (Isaac Lab
            # packs this at the sensor's actual render resolution),
            # falling back to static defaults only if unavailable.
            #
            # IMPORTANT: when using the front camera, we must NOT fall
            # back to the external camera's K matrix — the two cameras
            # have different resolutions and focal lengths.  Pairing
            # external-cam intrinsics with front-cam depth produces a
            # wrong point cloud.
            _sfx = "_front" if self._use_front_camera else ""
            # NOTE: do NOT use ``a or b`` — if a is a numpy array,
            # Python tries bool(array) which raises ValueError.
            obs_K = obs.get(f"observation/camera_K{_sfx}")
            if obs_K is None and not self._use_front_camera:
                # Only fall back to the non-suffixed key when we are
                # actually using the external camera.  For front camera,
                # skip straight to the static front_camera_intrinsics().
                obs_K = obs.get("observation/camera_K")
            if obs_K is not None:
                intrinsics = self._intrinsics_from_K(
                    np.asarray(obs_K, dtype=np.float64), depth,
                )
                logger.info(
                    f"  Using intrinsics from obs camera_K{_sfx}: "
                    f"fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
                    f"cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}, "
                    f"{intrinsics.width}x{intrinsics.height}"
                )
            elif self._use_front_camera and obs.get("observation/depth_front") is not None:
                intrinsics = front_camera_intrinsics()
                logger.info("  Using static front_camera_intrinsics() for grasp planning")
            else:
                intrinsics = self._intrinsics  # env default
            point_cloud = depth_to_pointcloud(depth, intrinsics)
            logger.info(
                f"  Point cloud: {point_cloud.shape[0]} points from "
                f"{np.squeeze(depth).shape[0]}×{np.squeeze(depth).shape[1]} depth"
            )

            # ---- 3+4. Detect and segment target object ----
            image = self._extract_image(obs)

            if self._seg_mode == GraspSegMode.GT_SIM:
                # ---- GT_SIM: use ground-truth mask from simulator ----
                # Debug: check what gt_seg keys are available
                gt_seg_keys = [k for k in obs if k.startswith("gt_seg")]
                logger.info(
                    f"  GT_SIM: available gt_seg keys = {gt_seg_keys}"
                )
                mask, bbox, detection_score = self._get_gt_mask(
                    obs, target_object, image.shape[:2],
                )
                logger.info(
                    f"  GT_SIM segmentation: '{target_object}' — "
                    f"{mask.sum()} pixels "
                    f"({mask.sum() * 100 / mask.size:.1f}% of image), "
                    f"mask shape={mask.shape}, bbox={bbox}"
                )
                if mask.sum() == 0:
                    raise RuntimeError(
                        f"GT_SIM mask for '{target_object}' has 0 pixels. "
                        f"gt_seg keys in obs: {gt_seg_keys}. "
                        f"The object may be occluded or gt_seg data may "
                        f"not be forwarded by the eval client."
                    )
            elif self._seg_mode == GraspSegMode.SAM3:
                # ---- SAM3: unified detect + segment in one call ----
                logger.info(f"  Using SAM3 for '{target_object}'")
                mask, detection_score, bbox = (
                    self._grasp_client.detect_and_segment(
                        image, target_object,
                    )
                )
                bbox = tuple(bbox)
                logger.info(
                    f"  SAM3 detection: '{target_object}' at {bbox} "
                    f"(score={detection_score:.2f}, "
                    f"mask={mask.sum()} px)"
                )
            elif self._seg_mode == GraspSegMode.MOLMO_SAM2:
                # ---- Molmo + SAM2: pointing → point-prompt segmentation ----
                # Molmo gives a single normalised pixel; SAM2 grows it
                # into an object mask.  No GDino needed.
                logger.info(f"  Using Molmo + SAM2 for '{target_object}'")
                mask, detection_score, bbox = self._molmo_sam2_segment(
                    image, target_object,
                )
                logger.info(
                    f"  Molmo+SAM2: '{target_object}' at {bbox} "
                    f"(iou={detection_score:.2f}, "
                    f"mask={mask.sum()} px)"
                )
            elif self._seg_mode == GraspSegMode.VLM_SAM2:
                # ---- VLM + SAM2: orchestrator VLM points → SAM2 segments ----
                # Mirrors MOLMO_SAM2 but uses the orchestrator's main VLM
                # (Claude / GPT / etc.) for the pointing step instead of
                # Molmo2.  Lets us run "single-VLM perception" (same model
                # picks grasp point AND place point) for the A3 perception
                # ablation.
                logger.info(f"  Using VLM + SAM2 for '{target_object}'")
                mask, detection_score, bbox = self._vlm_sam2_segment(
                    image, target_object,
                )
                logger.info(
                    f"  VLM+SAM2: '{target_object}' at {bbox} "
                    f"(iou={detection_score:.2f}, "
                    f"mask={mask.sum()} px)"
                )
            else:
                # ---- GDino + SAM2: two-stage detect → segment ----
                bbox, detection_score = self._detect_object(
                    image, target_object,
                )
                logger.info(
                    f"  Detection: '{target_object}' at {bbox} "
                    f"(score={detection_score:.2f})"
                )
                mask = self._segment_object(image, bbox)
                logger.info(
                    f"  Mask: {mask.sum()} pixels "
                    f"({mask.sum()*100/mask.size:.1f}% of image)"
                )

            # 🔍 Debug: detection result
            dbg.vis_detection(image, bbox, target_object, detection_score,
                              hitl=hitl)

            # 🎭 Debug: mask overlay
            dbg.vis_mask(image, mask, bbox, hitl=hitl)

            # 📏 Debug: depth visualization (bbox center as initial grasp estimate)
            bbox_center_uv = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            dbg.vis_depth(depth, bbox, hitl=hitl, grasp_uv=bbox_center_uv)

            # ---- 5. Predict grasp pose ----
            self._status_message = f"Predicting grasp for '{target_object}'..."
            self._phase = GraspPhase.PLANNING

            # Compute gravity vector in camera frame for top-down filtering
            cam_to_world = self._get_camera_to_world(obs)

            # ── Diagnostic: camera transform sanity check ──
            cam_pos_world = cam_to_world[:3, 3]
            cam_z_axis = cam_to_world[:3, 2]  # camera Z in world
            logger.info(
                f"  cam_to_world: pos={cam_pos_world}, "
                f"Z-axis(viewing)={cam_z_axis}"
            )
            # Sanity: camera should be above the table (~0.3-1.5m Z)
            # and its Z-axis should roughly point forward/down
            if cam_pos_world[2] < 0 or cam_pos_world[2] > 3.0:
                logger.warning(
                    f"  ⚠ Camera Z position {cam_pos_world[2]:.3f} "
                    f"looks wrong (expected 0.3-1.5)"
                )

            # ── Diagnostic: point cloud sanity ──
            pc_world = (cam_to_world[:3, :3] @ point_cloud.T).T + cam_to_world[:3, 3]
            logger.info(
                f"  Point cloud (world frame): "
                f"X=[{pc_world[:,0].min():.3f}, {pc_world[:,0].max():.3f}] "
                f"Y=[{pc_world[:,1].min():.3f}, {pc_world[:,1].max():.3f}] "
                f"Z=[{pc_world[:,2].min():.3f}, {pc_world[:,2].max():.3f}]"
            )
            # Masked points only
            mask_flat = mask.flatten()
            depth_sq = np.squeeze(depth)
            h_d, w_d = depth_sq.shape[:2]
            v_grid, u_grid = np.mgrid[0:h_d, 0:w_d]
            mask_resized = mask
            if mask.shape != (h_d, w_d):
                import cv2 as _cv2
                _m8 = (_cv2.resize(mask.astype(np.uint8)*255, (w_d, h_d),
                        interpolation=_cv2.INTER_NEAREST) > 127)
                mask_resized = _m8
            masked_z = depth_sq[mask_resized]
            if masked_z.size > 0:
                # Back-project mask center to world
                mask_ys, mask_xs = np.where(mask_resized)
                cu, cv = int(mask_xs.mean()), int(mask_ys.mean())
                cz = float(depth_sq[cv, cu])
                cx_cam = (cu - intrinsics.cx) / intrinsics.fx * cz
                cy_cam = (cv - intrinsics.cy) / intrinsics.fy * cz
                center_cam = np.array([cx_cam, cy_cam, cz])
                center_world = cam_to_world[:3, :3] @ center_cam + cam_to_world[:3, 3]
                logger.info(
                    f"  Mask center (cam): {center_cam} → "
                    f"(world): {center_world}, "
                    f"depth range=[{masked_z.min():.3f}, {masked_z.max():.3f}]"
                )

            R_world_to_cam = cam_to_world[:3, :3].T
            gravity_world = np.array([0.0, 0.0, -1.0])  # Z-up world
            gravity_cam = R_world_to_cam @ gravity_world
            gravity_cam = gravity_cam / np.linalg.norm(gravity_cam)
            logger.info(f"  Gravity in camera frame: {gravity_cam}")

            t0 = time.time()
            # ── Pre-flight check: verify mask/point-cloud overlap ──
            # Reproduce the server's projection locally so we can
            # diagnose mismatches before hitting a 422.  The projection is
            # in DEPTH/intrinsics resolution (that's what point_cloud was
            # built from); the mask is in RGB resolution.  Mirror the server:
            # project into intrinsics-res pixels, then scale to mask coords via
            # ``* mask_dim / intrinsics_dim`` (handles front-cam depth≠RGB).
            _h_proj, _w_proj = intrinsics.height, intrinsics.width
            _fx, _fy = intrinsics.fx, intrinsics.fy
            _cx, _cy = intrinsics.cx, intrinsics.cy
            _x, _y, _z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
            _u = (_fx * _x / _z + _cx).astype(int)
            _v = (_fy * _y / _z + _cy).astype(int)
            _in_bounds = (
                (_u >= 0) & (_u < _w_proj) & (_v >= 0) & (_v < _h_proj)
            )
            _mask_h, _mask_w = mask.shape[:2]
            _u_mask = (_u * _mask_w / _w_proj).astype(int)
            _v_mask = (_v * _mask_h / _h_proj).astype(int)
            _keep = np.zeros(len(point_cloud), dtype=bool)
            _keep[_in_bounds] = mask[
                _v_mask[_in_bounds], _u_mask[_in_bounds]
            ]
            logger.info(
                f"  Pre-flight: pc={point_cloud.shape[0]} pts, "
                f"in_bounds={int(_in_bounds.sum())}, "
                f"overlap={int(_keep.sum())}, "
                f"mask_px={int(mask.sum())}, "
                f"mask_shape={mask.shape}, "
                f"proj_hw=({_h_proj},{_w_proj}), "
                f"fx={_fx:.1f}, fy={_fy:.1f}, "
                f"cx={_cx:.1f}, cy={_cy:.1f}, "
                f"pc_z=[{point_cloud[:,2].min():.3f},{point_cloud[:,2].max():.3f}], "
                f"u=[{_u.min()},{_u.max()}], v=[{_v.min()},{_v.max()}]"
            )
            if int(_keep.sum()) == 0:
                raise RuntimeError(
                    f"Pre-flight: 0 overlap between point cloud "
                    f"({point_cloud.shape[0]} pts, "
                    f"u=[{_u.min()},{_u.max()}], "
                    f"v=[{_v.min()},{_v.max()}]) and mask "
                    f"({int(mask.sum())} px at {mask.shape}). "
                    f"Camera intrinsics or mask may be wrong."
                )

            # ---- Collision-obstacle cloud (world frame) ----
            # For collision-aware motion planning we want the SCENE minus
            # the target object (the object we're reaching for must not be
            # an obstacle for the final descent).  ``_keep`` marks target
            # points; the complement is everything else (other objects,
            # table, containers).  Stored in WORLD frame; converted to base
            # frame in ``_plan_trajectory`` via the same fk_vs_ee offset used
            # for the grasp target.  ``None`` if the planner is linear.
            self._obstacle_pc_world = _build_obstacle_cloud(
                pc_world, ~_keep, self._motion_planner,
            )

            # ---- Held-object cloud (world frame) for collision-aware PLACE ----
            # Capture the TARGET object's own points (``_keep``) so the place
            # tool can attach them to the gripper as a cuRobo attached-object.
            # Only when a collision-aware planner is active (linear ignores it).
            # Subsampled to the same cap as the obstacle cloud.
            self._target_pc_world = None
            if COLLISION_FREE_PLANNING and getattr(self._motion_planner, "name", "linear") != "linear":
                tgt = np.asarray(pc_world)[np.asarray(_keep, dtype=bool)]
                if tgt.shape[0] > MAX_OBSTACLE_POINTS:
                    idx = np.random.choice(
                        tgt.shape[0], MAX_OBSTACLE_POINTS, replace=False,
                    )
                    tgt = tgt[idx]
                if tgt.shape[0] > 0:
                    self._target_pc_world = tgt.astype(np.float32)

            # Stack coupling: when stack-mode is ON and this grasp will end in
            # a drop-from-above (stack=False), disable the top-down filter so
            # GraspGen returns its best grasp at any approach angle.  Otherwise
            # leave the filter on (None → server default / configured threshold).
            enable_td_filter = None
            if self._stack_mode_enabled and not self._stack_this_grasp:
                enable_td_filter = False
            # ``image_hw`` MUST be the resolution the projection is in — i.e.
            # the DEPTH/intrinsics resolution — NOT the RGB resolution.  The
            # server scales projected pixels to mask space via
            # ``u_mask = u * mask_w / w``; ``u,v`` come from the depth-res
            # intrinsics, so ``w,h`` must be depth-res too.  When the front
            # camera has a different depth vs RGB resolution (e.g. depth
            # 864×480, RGB/mask 1280×720) passing the RGB size mis-scales the
            # projection and yields 0 mask/point-cloud overlap.  For the
            # exterior camera (RGB==depth res) this is identical to image.shape.
            grasp_pose, confidence = self._grasp_client.compute_grasp(
                point_cloud=point_cloud,
                mask=mask,
                image_hw=(intrinsics.height, intrinsics.width),
                focal_length_px=intrinsics.fx,
                fy_px=intrinsics.fy,
                cx_px=intrinsics.cx,
                cy_px=intrinsics.cy,
                topdown_gravity=gravity_cam,
                topdown_threshold=self._topdown_threshold,
                enable_topdown_filter=enable_td_filter,
            )
            logger.info(
                f"  GraspGen: confidence={confidence:.3f} "
                f"({time.time() - t0:.2f}s)"
            )
            self._grasp_pose_camera = grasp_pose
            self._grasp_confidence = confidence

            # 🤏 Debug: grasp pose projected on image
            dbg.vis_grasp_pose(image, grasp_pose, intrinsics,
                               confidence, hitl=hitl)

            # ---- 6. Convert grasp to world frame ----
            # cam_to_world already computed above for gravity
            grasp_pose_world = cam_to_world @ grasp_pose


            self._intrinsics_used = intrinsics

            approach_world = grasp_pose_world[:3, 2]
            logger.info(
                f"  GraspGen approach (world): [{approach_world[0]:.2f}, "
                f"{approach_world[1]:.2f}, {approach_world[2]:.2f}]"
            )

            # ── Diagnostic: compare grasp position to GT object position ──
            gt_state = obs.get("gt_state", {})
            gt_objects = gt_state.get("objects", {})
            gt_obj = gt_objects.get(target_object) or gt_objects.get(
                target_object.replace(" ", "_")
            )
            gt_pos_str = ""
            if gt_obj is not None:
                gt_p = np.array(gt_obj["pos"])
                dist = np.linalg.norm(grasp_pose_world[:3, 3] - gt_p)
                gt_pos_str = (
                    f"  GT '{target_object}' pos={gt_p}, "
                    f"dist_to_grasp={dist:.4f}m"
                )

            logger.info(
                f"  Grasp (world): pos={grasp_pose_world[:3, 3]}, "
                f"approach={grasp_pose_world[:3, 2]}"
            )
            if gt_pos_str:
                logger.info(gt_pos_str)

            self._grasp_log = {
                "target_object": self._target_object,
                "confidence": float(confidence),
                "grasp_camera_pos": grasp_pose[:3, 3].tolist(),
                "grasp_camera_x_axis": grasp_pose[:3, 0].tolist(),
                "grasp_camera_z_depth": float(grasp_pose[2, 3]),
                "grasp_world_pos": grasp_pose_world[:3, 3].tolist(),
                "grasp_world_approach": grasp_pose_world[:3, 2].tolist(),
                "grasp_world_x_axis": grasp_pose_world[:3, 0].tolist(),
                "cam_to_world_pos": cam_to_world[:3, 3].tolist(),
                "cam_to_world_rot": cam_to_world[:3, :3].tolist(),
            }

            # ---- 7. Plan trajectory ----
            self._status_message = "Planning trajectory..."
            if self._env_mode == GraspEnvMode.LIBERO:
                self._plan_trajectory_libero(
                    current_ee_pos, current_ee_quat,
                    grasp_pose_world,
                )
            else:
                self._plan_trajectory(
                    current_joints, current_gripper,
                    grasp_pose_world, current_ee_pos,
                )
            logger.info(
                f"  Trajectory: {len(self._trajectory)} waypoints "
                f"({len(self._trajectory) // self._action_horizon} chunks)"
            )

            # 🌍 Debug: world-frame grasp + IK result
            dbg.vis_world_grasp(
                image, grasp_pose_world, current_ee_pos,
                q_pre=self._seg_approach[-1][:7] if self._seg_approach else None,
                q_grasp=self._q_at_grasp,
                hitl=hitl,
            )

            # ---- Hand held-object geometry to the place tool ----
            # Store the target-object cloud (world frame) + the FK EE transform
            # at the grasp config on SessionState so a subsequent place can
            # attach the carried object to the gripper for collision-aware
            # planning.  Only meaningful for ROBOLAB + a collision-aware
            # planner; harmless otherwise (place guards on planner support).
            try:
                if (self._env_mode == GraspEnvMode.ROBOLAB
                        and self._target_pc_world is not None
                        and self._q_at_grasp is not None):
                    from vlm_orchestrator.grasp.ik import forward_kinematics
                    state.last_grasped_object_pc_world = self._target_pc_world
                    state.last_grasped_ee_pose = forward_kinematics(
                        self._q_at_grasp
                    )
                else:
                    state.last_grasped_object_pc_world = None
                    state.last_grasped_ee_pose = None
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Failed to stash held-object cloud for place: {e}"
                )
                state.last_grasped_object_pc_world = None
                state.last_grasped_ee_pose = None

            # ---- Ready to execute ----
            self._phase = GraspPhase.APPROACHING
            self._traj_cursor = 0
            self._status_message = "Approaching pre-grasp..."

        except Exception as e:
            logger.error(f"grasp_with_tool perception failed: {e}", exc_info=True)
            self._phase = GraspPhase.FAILED
            self._status_message = f"Failed: {e}"

    # ------------------------------------------------------------------
    # Step (called once per proxy loop iteration = one action chunk)
    # ------------------------------------------------------------------

    def step(self, obs: dict, state) -> dict:
        """Produce the next action chunk.

        Returns a dict compatible with the VLA response format:
        ``{"actions": np.ndarray shape (horizon, action_dim)}``.
        """
        if self._phase == GraspPhase.RELEASING:
            return self._step_releasing(obs, state)

        if self._phase == GraspPhase.SETTLING:
            return self._step_settling(obs, state)

        if self._phase == GraspPhase.CLOSING:
            return self._step_closing(obs, state)

        if self._phase in (GraspPhase.LIFTING,
                           GraspPhase.APPROACHING,
                           GraspPhase.FINAL_APPROACH,
                           GraspPhase.RETREATING):
            return self._step_trajectory(obs, state)

        if self._phase == GraspPhase.SETTLE_HOLD:
            return self._step_settle_hold(obs, state)

        if self._phase == GraspPhase.INTEGRAL_SETTLE:
            return self._step_integral_settle(obs, state)

        if self._phase == GraspPhase.MEASURING:
            return self._step_measuring(obs, state)

        if self._phase in (GraspPhase.DONE, GraspPhase.FAILED):
            # Shouldn't be called but return a no-op
            return self._noop_response(obs)

        # Fallback
        logger.warning(f"grasp_tool step() in unexpected phase: {self._phase}")
        return self._noop_response(obs)

    def _step_releasing(self, obs: dict, state) -> dict:
        """Hold position with gripper open to drop a held object."""
        # Build hold-position + open-gripper action
        if self._env_mode == GraspEnvMode.LIBERO:
            action = np.zeros(self._action_dim)
            action[-1] = self._gripper_open
        else:
            joints = self._extract_joints(obs)
            action = np.concatenate([joints, [GRIPPER_OPEN]])
        actions = np.tile(action, (self._action_horizon, 1))
        self._last_commanded_gripper = self._gripper_open

        # Step-delta gating: hold for at least RELEASE_HOLD_STEPS sim
        # steps after entering this phase, regardless of chunk size.
        if (state.episode_step - self._step_at_release_start
                >= RELEASE_HOLD_STEPS):
            logger.info("  Release complete → resuming grasp pipeline")
            # Resume the normal start() flow after release
            self._resume_after_release(obs, state)

        return {"actions": actions}

    def _resume_after_release(self, obs: dict, state) -> None:
        """Continue the grasp pipeline after the gripper release."""
        try:
            if self._env_mode == GraspEnvMode.LIBERO:
                # Same rationale as start(): the agentview camera is fixed
                # but the arm is still in its FOV — lift before perceiving.
                ee_pos_now = self._libero_extract_ee_pos(obs)
                if ee_pos_now is None:
                    logger.warning(
                        "  LIBERO perception lift skipped — no ee_pos in obs"
                    )
                    self._start_perception(obs, state)
                    return
                self._plan_perception_lift_libero(ee_pos_now)
                if self._trajectory:
                    self._phase = GraspPhase.LIFTING
                    self._traj_cursor = 0
                    self._status_message = (
                        f"LIBERO: lifting arm out of camera view "
                        f"for '{self._target_object}'..."
                    )
                    logger.info(
                        f"  LIBERO perception lift (post-release): "
                        f"{len(self._trajectory)} EE-delta waypoints"
                    )
                    return
                logger.info(
                    "  LIBERO perception lift produced no waypoints "
                    "— perceiving from current pose"
                )
                self._start_perception(obs, state)
                return

            if self._skip_lift_after_release:
                logger.info(
                    "  skip_pre_grasp_lift is ON — "
                    "perceiving from current arm position"
                )
                self._start_perception(obs, state)
                return

            current_joints = self._extract_joints(obs)
            self._plan_lift_to_safe(current_joints)

            if self._trajectory:
                self._phase = GraspPhase.LIFTING
                self._traj_cursor = 0
                self._status_message = (
                    f"Moving arm to perception pose for "
                    f"'{self._target_object}'..."
                )
                logger.info(
                    f"  Moving to perception pose: "
                    f"{len(self._trajectory)} waypoints"
                )
                return
            else:
                logger.info("  Arm already at perception pose, skipping lift")
                self._start_perception(obs, state)
        except Exception as e:
            logger.error(f"grasp_with_tool resume failed: {e}", exc_info=True)
            self._phase = GraspPhase.FAILED
            self._status_message = f"Failed: {e}"

    def _step_settling(self, obs: dict, state) -> dict:
        """Hold position while waiting for the object to settle."""
        self._settle_remaining -= 1
        if self._settle_remaining <= 0:
            logger.info("  Settle complete → starting perception with fresh obs")
            try:
                self._start_perception(obs, state)
            except Exception as e:
                logger.error(f"Perception after settle failed: {e}", exc_info=True)
                self._phase = GraspPhase.FAILED
                self._status_message = f"Failed: {e}"
        return self._noop_response(obs)

    def _step_trajectory(self, obs: dict, state) -> dict:
        """Serve the next action_horizon waypoints from the planned trajectory."""
        # ── LIBERO closed-loop override for LIFTING / APPROACHING / FINAL_APPROACH ──
        # The pre-computed trajectory assumes perfect tracking; OSC_POSE has
        # impedance lag and won't reach commanded delta exactly, so error
        # accumulates open-loop.  Recompute the per-step delta each chunk
        # from the latest current_ee_pos toward the segment target.
        if (
            self._env_mode == GraspEnvMode.LIBERO
            and self._phase in (
                GraspPhase.LIFTING,
                GraspPhase.APPROACHING,
                GraspPhase.FINAL_APPROACH,
            )
            and self._libero_seg_target_pos is not None
        ):
            return self._step_trajectory_libero_closed_loop(obs, state)

        chunk = []
        for _ in range(self._action_horizon):
            if self._traj_cursor < len(self._trajectory):
                chunk.append(self._trajectory[self._traj_cursor])
                self._traj_cursor += 1
            else:
                # Repeat last waypoint to hold position
                chunk.append(self._trajectory[-1])

        actions = np.array(chunk, dtype=np.float64)

        # Check if we've finished the current trajectory segment
        if self._traj_cursor >= len(self._trajectory):
            self._advance_phase_after_trajectory(obs, state)

        return {"actions": actions}

    def _step_trajectory_libero_closed_loop(self, obs, state) -> dict:
        """Closed-loop chunk for LIBERO LIFTING / APPROACHING / FINAL_APPROACH.

        Position: per-chunk recompute of delta from current EE pos to
        segment target, capped at the per-step Cartesian limit and
        scaled to OSC_POSE input units.

        Orientation: if a target rotation is set, recompute the per-step
        rotation delta from the current EE quat — this is what keeps us
        from over-shooting the orientation when LIBERO_INTERP_APPROACH
        is small.  OSC_POSE_DELTA's ``goal_ori = R_delta @ R_current``
        makes a constant non-zero delta keep rotating past the target,
        so the cached fallback is only safe when the segment really
        wants no orientation change.

        Phase advances when EE is within ``LIBERO_PHASE_DONE_M`` of the
        target — prevents over-shoot at the segment boundary.
        """
        from scipy.spatial.transform import Rotation as _R
        cur = self._extract_ee_pos(obs)
        target = self._libero_seg_target_pos
        delta_m = target - cur
        dist = float(np.linalg.norm(delta_m))

        # Per-step delta in metres, capped at LIBERO_MAX_POS_DELTA per axis.
        per_step_m = np.clip(
            delta_m / max(self._libero_seg_remaining_steps_estimate, 1),
            -LIBERO_MAX_POS_DELTA, LIBERO_MAX_POS_DELTA,
        )
        per_step_norm = per_step_m / OSC_POSE_POS_SCALE

        # Closed-loop orientation when a target_rot is set; otherwise
        # fall back to the cached open-loop delta (zero for retreat /
        # perception lift, set at plan time for approach).
        if self._libero_seg_target_rot is not None:
            cur_quat = self._extract_ee_quat(obs)
            if cur_quat is not None:
                cur_rot = _R.from_quat([
                    cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0],
                ]).as_matrix()
                R_delta_world = (
                    self._libero_seg_target_rot @ cur_rot.T
                )
                rot_delta = _R.from_matrix(R_delta_world).as_rotvec()
                per_step_rad = np.clip(
                    rot_delta / max(
                        self._libero_seg_remaining_steps_estimate, 1,
                    ),
                    -LIBERO_MAX_ORI_DELTA, LIBERO_MAX_ORI_DELTA,
                )
                ori_norm = per_step_rad / OSC_POSE_ORI_SCALE
            else:
                ori_norm = self._libero_seg_ori_per_step_norm
        else:
            ori_norm = self._libero_seg_ori_per_step_norm
        gripper = (
            self._gripper_close
            if self._phase == GraspPhase.FINAL_APPROACH
            else self._gripper_open
        )

        action = np.concatenate([per_step_norm, ori_norm, [gripper]])
        actions = np.tile(action, (self._action_horizon, 1))

        # Decrement the step-budget estimate so per-step delta grows
        # smaller as we approach the target (graceful slowdown).
        self._libero_seg_remaining_steps_estimate = max(
            1, self._libero_seg_remaining_steps_estimate - 1,
        )

        # Phase-done check: advance once we're close enough.
        if dist <= LIBERO_PHASE_DONE_M:
            self._advance_phase_after_trajectory(obs, state)

        return {"actions": actions}

    def _step_closing(self, obs: dict, state) -> dict:
        """Hold position with gripper closed and check for grasp success."""
        if self._env_mode == GraspEnvMode.LIBERO:
            # LIBERO: send zero EE-delta + close gripper
            hold_action = np.zeros(self._action_dim)
            hold_action[-1] = self._gripper_close
        else:
            # robolab: hold current joint positions + close gripper
            current_joints = self._extract_joints(obs)
            hold_action = np.concatenate([current_joints, [self._gripper_close]])
        actions = np.tile(hold_action, (self._action_horizon, 1))
        # We're commanding CLOSE this chunk — record it so post-DONE
        # / post-FAILED no-op chunks keep commanding the same thing.
        self._last_commanded_gripper = self._gripper_close

        # Step-delta gating: hold for at least CLOSE_HOLD_STEPS sim
        # steps after entering this phase, regardless of chunk size.
        if (state.episode_step - self._step_at_close_start
                >= CLOSE_HOLD_STEPS):
            # No verify-from-gripper_pos: the DROID gripper's
            # observed joint position isn't a reliable success
            # signal — the actuator doesn't settle inside a
            # reasonable hold window, and the same value can mean
            # "blocked by held object" or "still travelling toward
            # closed" depending on scene geometry / timing.  On
            # real robots we don't have GT contact either, so the
            # only robust signal is visual: after we close + lift,
            # the VLM looks at the post-retreat scene and decides
            # whether the object is in the gripper.
            #
            # So always proceed mechanically: close → retreat →
            # DONE.  The tool_chain strategy's next VLM cycle will
            # see the lifted arm + scene and emit either
            # "continue + place" (object held) or "continue + grasp"
            # (retry).
            gripper_pos = self._extract_gripper(obs)
            logger.info(
                f"  Grasp closing complete (pos={gripper_pos:.3f}); "
                f"retreating — VLM will judge held-vs-empty visually"
            )
            self._status_message = "Closed; retreating..."
            self._plan_retreat(obs)
            self._phase = GraspPhase.RETREATING
            self._traj_cursor = 0

        return {"actions": actions}

    def _step_settle_hold(self, obs: dict, state) -> dict:
        """TESTING-ONLY: re-command the final grasp waypoint for extra sim steps.

        Only entered when ``GRASP_SETTLE_HOLD_STEPS > 0``.  The production path
        (default 0) never uses this phase.  Holding the same open-gripper joint
        target lets the PD controller's ramp-tracking lag decay so the
        subsequent MEASURING snapshot reads the SETTLED undershoot rather than
        a mid-settle value.  Gripper stays OPEN here (same as FINAL_APPROACH)
        so the measurement reflects arm position only, not finger closure.
        """
        last_action = self._seg_final[-1] if self._seg_final else self._trajectory[-1]
        actions = np.tile(last_action, (self._action_horizon, 1))
        if (state.episode_step - self._step_at_settle_start
                >= GRASP_SETTLE_HOLD_STEPS):
            logger.info(
                f"  Settle hold complete "
                f"({state.episode_step - self._step_at_settle_start} steps) "
                f"→ measuring settled post-move pose"
            )
            self._phase = GraspPhase.MEASURING
        return {"actions": actions}

    def _step_integral_settle(self, obs: dict, state) -> dict:
        """Integral correction of the joint-space P-controller undershoot.

        The sim's joint controller settles ~1–2° short of ``q_grasp`` on the
        gravity-loaded joints (steady-state error err ≈ τ_gravity/Kp, no
        integral term).  Here we close that loop in the orchestrator: each step
        we read the settled joints, accumulate the error toward q_grasp, and
        command ``q_grasp + Ki·Σerr`` so the controller is driven onto the
        target.  Gripper stays OPEN (same as FINAL_APPROACH) — we correct arm
        position before closing on the object.

        Terminates when the max joint error < TOL or after STEPS sim steps.
        Falls through to MEASURING (which then closes) on completion.
        """
        q_grasp = self._q_at_grasp
        try:
            q_now = self._extract_joints(obs)[:7]
        except Exception:
            # No joints available — cannot close the loop; command endpoint.
            last = self._seg_final[-1] if self._seg_final else self._trajectory[-1]
            self._phase = GraspPhase.MEASURING
            return {"actions": np.tile(last, (self._action_horizon, 1))}

        err = q_grasp[:7] - q_now
        max_err = float(np.max(np.abs(err)))
        elapsed = state.episode_step - self._step_at_integral_start

        # Anti-windup stall detection: track whether the error is still
        # improving.  If it plateaus (contact), stop before winding up.
        if max_err < self._integral_best_err - 1e-4:
            self._integral_best_err = max_err
            self._integral_stall_updates = 0
        else:
            self._integral_stall_updates += 1
        stalled = self._integral_stall_updates >= GRASP_FINAL_INTEGRAL_STALL_UPDATES

        # Done: settled onto target, stalled against a constraint, or budget out.
        if (
            max_err < GRASP_FINAL_INTEGRAL_TOL_RAD
            or stalled
            or elapsed >= GRASP_FINAL_INTEGRAL_STEPS
        ):
            reason = (
                "converged" if max_err < GRASP_FINAL_INTEGRAL_TOL_RAD
                else "stalled (contact)" if stalled
                else "step-budget"
            )
            self._grasp_log["integral_settle_steps"] = int(elapsed)
            self._grasp_log["integral_settle_final_err_rad"] = float(max_err)
            self._grasp_log["integral_settle_reason"] = reason
            self._grasp_log["integral_settle_accum_rad"] = (
                self._integral_accum.tolist()
                if self._integral_accum is not None else None
            )
            logger.info(
                f"  Integral settle done [{reason}] ({elapsed} steps): "
                f"max joint err {np.degrees(max_err):.2f}° "
                f"(accum max {np.degrees(np.max(np.abs(self._integral_accum))):.2f}°) "
                f"→ measuring"
            )
            self._phase = GraspPhase.MEASURING
            last = self._seg_final[-1] if self._seg_final else self._trajectory[-1]
            return {"actions": np.tile(last, (self._action_horizon, 1))}

        # Conditional integration (anti-windup): only accumulate while the
        # correction is not saturated at the safety clamp, so a blocked joint
        # can't wind the accumulator unbounded.
        prev_accum = self._integral_accum.copy()
        self._integral_accum = self._integral_accum + err
        correction = GRASP_FINAL_INTEGRAL_KI * self._integral_accum
        saturated = np.abs(correction) >= GRASP_FINAL_INTEGRAL_MAX_RAD
        # Roll back accumulation on any joint whose command is saturated.
        self._integral_accum[saturated] = prev_accum[saturated]
        correction = np.clip(
            GRASP_FINAL_INTEGRAL_KI * self._integral_accum,
            -GRASP_FINAL_INTEGRAL_MAX_RAD, GRASP_FINAL_INTEGRAL_MAX_RAD,
        )
        q_cmd = q_grasp[:7] + correction
        action = np.concatenate([q_cmd, [self._gripper_open]])
        return {"actions": np.tile(action, (self._action_horizon, 1))}

    def _step_measuring(self, obs: dict, state) -> dict:
        """Hold at the final grasp pose for one chunk while logging post-move metrics.

        Called exactly once after FINAL_APPROACH ends.  The obs here reflects
        the robot state after the last trajectory chunk has executed, so
        position/orientation metrics are accurate.  Transitions to CLOSING
        so the grasp tool closes the gripper on the object itself rather
        than handing an open-gripper-at-grasp-pose to the VLA.
        """
        self._do_post_move_log(obs, state)
        self._phase = GraspPhase.CLOSING
        self._step_at_close_start = state.episode_step
        self._status_message = "Closing gripper..."
        last_action = self._seg_final[-1] if self._seg_final else self._trajectory[-1]
        return {"actions": np.tile(last_action, (self._action_horizon, 1))}

    def _do_post_move_log(self, obs: dict, state) -> None:
        """Capture post-move position/orientation metrics and save debug images."""
        try:
            from vlm_orchestrator.grasp import debug as dbg
            if (self._grasp_pose_camera is None
                    or self._intrinsics_used is None):
                return
            image = self._extract_image(obs)
            ee_pos = self._extract_ee_pos(obs)
            cam_to_world = self._get_camera_to_world(obs)
            hitl = getattr(state, "_hitl", None) or getattr(state, "hitl", None)

            joints = None
            try:
                joints = self._extract_joints(obs)
            except Exception:
                pass

            # ---- Joint-space undershoot diagnostic ----
            # Compare the settled joints to the commanded grasp config
            # (self._q_at_grasp).  A non-zero per-joint delta here is the
            # SOURCE of the Cartesian undershoot: the sim's joint P-controller
            # settles short of the setpoint (steady-state err ≈ τ_gravity/Kp,
            # no integral term).  Distinguishes controller undershoot (this is
            # non-zero) from a planning error (this ≈ 0, error is elsewhere).
            if joints is not None and self._q_at_grasp is not None:
                q_err = np.asarray(joints, dtype=float)[:7] - self._q_at_grasp[:7]
                self._grasp_log["post_move_joint_err_rad"] = q_err.tolist()
                self._grasp_log["post_move_joint_err_max_rad"] = float(
                    np.max(np.abs(q_err))
                )

            # ---- Position error ----
            grasp_world = (
                cam_to_world[:3, :3] @ self._grasp_pose_camera[:3, 3]
                + cam_to_world[:3, 3]
            )
            delta = ee_pos - grasp_world
            self._grasp_log["post_move_settle_hold_steps"] = GRASP_SETTLE_HOLD_STEPS
            self._grasp_log["post_move_ee_pos"] = ee_pos.tolist()
            self._grasp_log["post_move_grasp_world_pos"] = grasp_world.tolist()
            self._grasp_log["post_move_delta_xyz"] = delta.tolist()
            self._grasp_log["post_move_total_err_m"] = float(np.linalg.norm(delta))

            # ---- Fingertip-placement audit (log-only; model-based) ----
            # Does the robot's ACTUAL fingertip land where GraspGen PREDICTED
            # the fingertip contact point?  This isolates the panda-vs-Robotiq
            # flange→fingertip depth mismatch (0.1034 m vs 0.1628 m) that the
            # grasp stack silently assumes away (it is tuned for a Franka hand
            # but DROID mounts a Robotiq 2F-85).
            #
            # ⚠ MODEL-BASED, not ground truth.  Both fingertips are
            # extrapolated along an approach axis using spec depths, NOT read
            # from the sim's actual finger bodies.  It therefore CANNOT catch a
            # wrong gripper mount, finger flex, or a bad ee_quat — only the
            # depth-constant mismatch (which is the specific hypothesis here).
            # For ground truth, read left/right_inner_finger body_pos_w in
            # robolab and add a fingertip_pos obs term instead.
            #
            # Approach axis:
            #   * GraspGen fingertip uses the LOGGED grasp approach (world +Z of
            #     the grasp frame) — same axis GraspGen backed its origin along.
            #   * Robot fingertip uses the CURRENT ee_quat's approach axis, so
            #     it reflects where the gripper actually ended up pointing.
            gg_approach = np.asarray(
                self._grasp_log.get("grasp_world_approach"), dtype=np.float64
            ) if self._grasp_log.get("grasp_world_approach") is not None else None
            ee_quat = self._extract_ee_quat(obs)
            if gg_approach is not None and ee_quat is not None:
                from scipy.spatial.transform import Rotation as _Rft
                # ee_quat is (w, x, y, z) → scipy wants (x, y, z, w)
                R_ee = _Rft.from_quat(
                    [ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]
                ).as_matrix()
                self._grasp_log["post_move_ee_quat"] = np.asarray(
                    ee_quat, dtype=np.float64
                ).tolist()
                # ⚠ The DROID ee_quat is the Robotiq *base_link* frame, whose
                # axes are rotated relative to the gripper's approach direction
                # (EEF_OFFSET_ROT=(0.5,-0.5,0.5,-0.5) maps base_link +Z → eef
                # +X).  The real approach axis (eef_frame +Z) equals −Y of
                # base_link, i.e. −R_ee[:,1] — NOT +Z.  Using +Z here (the
                # earlier bug) put the "robot fingertip" 90° off, producing a
                # spurious ~160 mm lateral error.  We pick the base_link column
                # best aligned with GraspGen's approach so the audit is robust
                # even if the offset convention changes.
                cand = {
                    "+x": R_ee[:, 0], "-x": -R_ee[:, 0],
                    "+y": R_ee[:, 1], "-y": -R_ee[:, 1],
                    "+z": R_ee[:, 2], "-z": -R_ee[:, 2],
                }
                best = max(cand.items(), key=lambda kv: float(np.dot(kv[1], gg_approach)))
                ee_approach = best[1]
                self._grasp_log["post_move_ee_approach_axis"] = best[0]

                p_fingertip_graspgen = (
                    grasp_world + GRASPGEN_GRIPPER_DEPTH_M * gg_approach
                )
                p_fingertip_robot = (
                    ee_pos + ROBOTIQ_FLANGE_TO_FINGERTIP_M * ee_approach
                )
                fingertip_delta = p_fingertip_robot - p_fingertip_graspgen
                # Split the error along vs perpendicular to GraspGen's approach:
                # err_along isolates the depth-constant mismatch; err_lateral is
                # perception / IK / registration error.
                err_along = float(np.dot(fingertip_delta, gg_approach))
                err_lateral = float(
                    np.linalg.norm(fingertip_delta - err_along * gg_approach)
                )
                self._grasp_log["post_move_fingertip_graspgen_pos"] = (
                    p_fingertip_graspgen.tolist()
                )
                self._grasp_log["post_move_fingertip_robot_pos"] = (
                    p_fingertip_robot.tolist()
                )
                self._grasp_log["post_move_fingertip_err_m"] = float(
                    np.linalg.norm(fingertip_delta)
                )
                self._grasp_log["post_move_fingertip_err_along_m"] = err_along
                self._grasp_log["post_move_fingertip_err_lateral_m"] = err_lateral
                self._grasp_log["post_move_fingertip_model_based"] = True
                # Outcome-level: did the fingertip reach the target object?
                gt_objs = (obs.get("gt_state", {}) or {}).get("objects", {})
                gt_obj = gt_objs.get(self._target_object) or gt_objs.get(
                    self._target_object.replace(" ", "_")
                )
                if gt_obj is not None and gt_obj.get("pos") is not None:
                    d_obj = float(np.linalg.norm(
                        p_fingertip_robot - np.asarray(gt_obj["pos"],
                                                       dtype=np.float64)
                    ))
                    self._grasp_log["post_move_fingertip_to_object_m"] = d_obj
                logger.info(
                    f"  Fingertip audit (model-based): "
                    f"err={self._grasp_log['post_move_fingertip_err_m']*1e3:.1f}mm "
                    f"(along={err_along*1e3:+.1f}mm, "
                    f"lateral={err_lateral*1e3:.1f}mm)"
                )

                # ── TRUE GROUND-TRUTH fingertip (if robolab forwards it) ──
                # observation/fingertip_pos = the Robotiq pad grasp-surface
                # CENTER, read from robolab's ``fingertip_frame``
                # FrameTransformer (base_link + a calibrated +0.1311 m offset
                # along the approach axis, measured from the pad-mesh AABB in
                # the flattened Isaac USD).  This is the ACTUAL fingertip — not
                # the knuckle body (whose origin sits ~2 mm from base_link) and
                # not extrapolated from the datasheet 162.8 mm constant.
                #
                # The clean metric: ACTUAL fingertip (sim pad) vs GraspGen's
                # PREDICTED fingertip (the contact point GraspGen regressed,
                # = grasp_world + gripper_depth·approach).  No object center,
                # no flange offset assumption — a direct predicted-vs-actual
                # comparison of the point that physically grasps.
                ft_gt = obs.get("observation/fingertip_pos")
                if ft_gt is not None:
                    p_ft_actual = np.asarray(
                        ft_gt, dtype=np.float64
                    ).flatten()[:3]
                    d_ft = p_ft_actual - p_fingertip_graspgen
                    ft_along = float(np.dot(d_ft, gg_approach))
                    ft_lat = float(
                        np.linalg.norm(d_ft - ft_along * gg_approach)
                    )
                    self._grasp_log["post_move_fingertip_gt_pos"] = (
                        p_ft_actual.tolist()
                    )
                    self._grasp_log["post_move_fingertip_gt_err_m"] = float(
                        np.linalg.norm(d_ft)
                    )
                    self._grasp_log["post_move_fingertip_gt_err_along_m"] = (
                        ft_along
                    )
                    self._grasp_log["post_move_fingertip_gt_err_lateral_m"] = (
                        ft_lat
                    )
                    # Cross-check: actual sim fingertip vs the model-based
                    # (flange + datasheet-constant) estimate.  A large gap here
                    # flags that ROBOTIQ_FLANGE_TO_FINGERTIP_M (162.8 mm) does
                    # not match the sim pad geometry (measured ~131 mm center).
                    self._grasp_log["post_move_fingertip_gt_vs_model_m"] = float(
                        np.linalg.norm(p_ft_actual - p_fingertip_robot)
                    )
                    logger.info(
                        f"  Fingertip audit (TRUE GROUND-TRUTH pad): "
                        f"err={self._grasp_log['post_move_fingertip_gt_err_m']*1e3:.1f}mm "
                        f"(along={ft_along*1e3:+.1f}mm, lateral={ft_lat*1e3:.1f}mm); "
                        f"gt-vs-model="
                        f"{self._grasp_log['post_move_fingertip_gt_vs_model_m']*1e3:.1f}mm"
                    )

            # ---- Orientation error (FK-based, per axis) ----
            axis_errs: dict[str, float] = {}
            rot_err_fk = None
            R_planned_world = None   # predicted grasp rotation (world), for overlay
            R_fk_world = None        # actual EE rotation (world), for overlay
            if joints is not None:
                from vlm_orchestrator.grasp.ik import forward_kinematics as _fk
                T_final = _fk(joints)
                self._grasp_log["post_move_fk_pos"] = T_final[:3, 3].tolist()
                self._grasp_log["post_move_fk_vs_ee_delta"] = (
                    T_final[:3, 3] - ee_pos
                ).tolist()

                # Convert GraspGen-frame planned rotation to the panda_hand
                # convention so per-axis errors are computed against the
                # SAME labelled axes the FK / EE quat reports.  Skipping
                # this would put a built-in 90°/180° offset into every
                # per-axis error.
                grasp_pose_camera_for_err = self._grasp_pose_camera[:3, :3]
                if (
                    self._env_mode == GraspEnvMode.LIBERO
                    and LIBERO_GRASPGEN_TO_EE_YAW_RAD != 0.0
                ):
                    from scipy.spatial.transform import Rotation as _R
                    R_corr_3 = _R.from_euler(
                        "z", LIBERO_GRASPGEN_TO_EE_YAW_RAD,
                    ).as_matrix()
                    grasp_pose_camera_for_err = (
                        self._grasp_pose_camera[:3, :3] @ R_corr_3
                    )
                R_planned = cam_to_world[:3, :3] @ grasp_pose_camera_for_err
                R_fk = T_final[:3, :3]
                R_planned_world = R_planned  # capture for the debug overlay
                R_fk_world = R_fk

                for i, name in enumerate(["x_open", "y", "z_approach"]):
                    pred_ax = R_planned[:, i]
                    fk_ax = R_fk[:, i]
                    err_deg = float(np.degrees(np.arccos(
                        np.clip(abs(np.dot(pred_ax, fk_ax)), 0.0, 1.0)
                    )))
                    self._grasp_log[f"post_move_pred_{name}_axis"] = pred_ax.tolist()
                    self._grasp_log[f"post_move_fk_{name}_axis"] = fk_ax.tolist()
                    self._grasp_log[f"post_move_{name}_err_deg"] = err_deg
                    axis_errs[name] = err_deg

                R_rel_fk = R_planned.T @ R_fk
                tr_fk = float(np.clip(np.trace(R_rel_fk), -1.0, 3.0))
                rot_err_fk = float(
                    np.degrees(np.arccos(np.clip((tr_fk - 1) / 2, -1.0, 1.0)))
                )
                self._grasp_log["post_move_fk_rot_err_deg"] = rot_err_fk
                self._grasp_log["post_move_fk_approach_err_deg"] = axis_errs["z_approach"]
                logger.info(
                    f"  Post-move FK rot_err={rot_err_fk:.1f}°  "
                    f"X(open)={axis_errs['x_open']:.1f}°  "
                    f"Y={axis_errs['y']:.1f}°  "
                    f"Z(approach)={axis_errs['z_approach']:.1f}°"
                )

            # ---- Debug image (uses pre-computed axis_errs) ----
            # Pass raw GraspGen pose (vis_post_move_grasp draws the
            # gripper outline along col 0, which is correct for the
            # GraspGen frame).  Pass the *inverse* of the
            # GraspGen→EE yaw — vis re-expresses the actual EE quat
            # in GraspGen labelling (X=open, Y=perp) so bright + dim
            # arrows compare like-coloured regardless of the
            # panda_hand vs GraspGen convention swap.
            ee_axis_yaw = (
                -LIBERO_GRASPGEN_TO_EE_YAW_RAD
                if self._env_mode == GraspEnvMode.LIBERO
                else 0.0
            )
            # Fingertip GT-vs-predicted overlay data (world frame).  Prefer the
            # true sim pad (post_move_fingertip_gt_pos); fall back to the
            # model-based estimate when robolab didn't forward fingertip_pos.
            _ft_gt_world = self._grasp_log.get("post_move_fingertip_gt_pos")
            if _ft_gt_world is None:
                _ft_gt_world = self._grasp_log.get("post_move_fingertip_robot_pos")
            _ft_pred_world = self._grasp_log.get("post_move_fingertip_graspgen_pos")
            # Use GT errors when available, else model-based.
            _ft_errs = None
            if self._grasp_log.get("post_move_fingertip_gt_err_m") is not None:
                _ft_errs = {
                    "err_m": self._grasp_log["post_move_fingertip_gt_err_m"],
                    "along_m": self._grasp_log["post_move_fingertip_gt_err_along_m"],
                    "lateral_m": self._grasp_log["post_move_fingertip_gt_err_lateral_m"],
                }
            elif self._grasp_log.get("post_move_fingertip_err_m") is not None:
                _ft_errs = {
                    "err_m": self._grasp_log["post_move_fingertip_err_m"],
                    "along_m": self._grasp_log["post_move_fingertip_err_along_m"],
                    "lateral_m": self._grasp_log["post_move_fingertip_err_lateral_m"],
                }
            dbg.vis_post_move_grasp(
                image, self._grasp_pose_camera, self._intrinsics_used,
                self._grasp_confidence,
                ee_pos_world=ee_pos,
                ee_quat_wxyz=self._extract_ee_quat(obs),
                cam_to_world=cam_to_world,
                q_joints=joints,
                axis_errs=axis_errs,
                hitl=hitl,
                ee_axis_yaw_to_graspgen_rad=ee_axis_yaw,
                fingertip_gt_world=(
                    np.asarray(_ft_gt_world) if _ft_gt_world is not None else None
                ),
                fingertip_pred_world=(
                    np.asarray(_ft_pred_world) if _ft_pred_world is not None else None
                ),
                fingertip_errs=_ft_errs,
                R_pred_world=R_planned_world,
                R_actual_world=R_fk_world,
            )
            dbg.save_grasp_log(self._grasp_log)
        except Exception as e:
            logger.warning(f"  [debug] Post-move log failed: {e}")

    def _advance_phase_after_trajectory(self, obs, state):
        """Called when the current trajectory segment is fully consumed."""
        if self._phase == GraspPhase.LIFTING:
            # Lift complete → start perception with fresh obs
            logger.info("  Lift complete → starting perception")
            self._last_obs = obs  # use the latest obs after lift
            try:
                self._start_perception(obs, state)
            except Exception as e:
                logger.error(f"Perception after lift failed: {e}", exc_info=True)
                self._phase = GraspPhase.FAILED
                self._status_message = f"Failed: {e}"
            return

        elif self._phase == GraspPhase.APPROACHING:
            # Load the final-approach segment
            logger.info("  Approach complete → final approach")
            self._phase = GraspPhase.FINAL_APPROACH
            self._status_message = "Final approach to grasp pose..."
            self._trajectory = self._seg_final
            self._traj_cursor = 0
            # LIBERO closed-loop: switch target to grasp pos, hold orientation
            if (self._env_mode == GraspEnvMode.LIBERO
                    and self._libero_grasp_pos is not None):
                self._libero_seg_target_pos = self._libero_grasp_pos.copy()
                self._libero_seg_ori_per_step_norm = np.zeros(3)
                self._libero_seg_remaining_steps_estimate = LIBERO_INTERP_FINAL

        elif self._phase == GraspPhase.FINAL_APPROACH:
            # Defer measurement by one step so the obs reflects the arm AFTER
            # the last trajectory chunk executes, not while it's being served.
            if (
                GRASP_FINAL_INTEGRAL
                and self._env_mode == GraspEnvMode.ROBOLAB
                and self._q_at_grasp is not None
            ):
                # Integral correction of the joint-space P-controller undershoot
                # before closing.  Drives the settled joints onto q_grasp.
                logger.info(
                    "  Final approach complete → INTEGRAL_SETTLE "
                    f"(Ki={GRASP_FINAL_INTEGRAL_KI}, "
                    f"≤{GRASP_FINAL_INTEGRAL_STEPS} steps)"
                )
                self._phase = GraspPhase.INTEGRAL_SETTLE
                self._step_at_integral_start = state.episode_step
                self._integral_accum = np.zeros(7, dtype=np.float64)
                self._integral_best_err = np.inf
                self._integral_stall_updates = 0
            elif GRASP_SETTLE_HOLD_STEPS > 0:
                # TESTING: hold the final waypoint for extra sim steps.
                logger.info(
                    f"  Final approach complete → SETTLE_HOLD "
                    f"({GRASP_SETTLE_HOLD_STEPS} steps) before measuring"
                )
                self._phase = GraspPhase.SETTLE_HOLD
                self._step_at_settle_start = state.episode_step
            else:
                logger.info(
                    "  Final approach complete → measuring post-move pose"
                )
                self._phase = GraspPhase.MEASURING

        elif self._phase == GraspPhase.RETREATING:
            logger.info("  Retreat complete → DONE")
            self._phase = GraspPhase.DONE
            self._status_message = "Grasp complete — handing back to VLA"

    # ------------------------------------------------------------------
    # Trajectory planning (IK + motion-planned joint-space segments)
    # ------------------------------------------------------------------

    def _plan_segment(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        n_steps: int,
        *,
        phase: str,
        scene_pc: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Plan one joint-space segment via ``self._motion_planner``.

        Records the planner label under ``ik_motion_planner_<phase>`` in the
        grasp log for post-hoc observability.  On planner failure raises
        ``RuntimeError`` (no silent fallback to linear) — the caller's
        existing ``FAILED`` handling surfaces it.
        """
        result = self._motion_planner.plan_segment(
            q_start, q_end, n_steps=n_steps, scene_pc=scene_pc, phase=phase,
        )
        self._grasp_log[f"ik_motion_planner_{phase}"] = result.label
        if not result.success or result.waypoints is None:
            raise RuntimeError(
                f"Motion planning failed for {phase} segment "
                f"(planner={result.label}): {result.detail}"
            )
        return result.waypoints

    def _plan_trajectory(
        self,
        current_joints: np.ndarray,
        current_gripper: float,
        grasp_pose_world: np.ndarray,
        current_ee_pos: np.ndarray,
    ) -> None:
        """Build approach + final-approach trajectory using IK.

        Computes:
        1. Pre-grasp pose: offset along negative approach axis
        2. IK for pre-grasp → joint targets
        3. IK for grasp pose → joint targets
        4. Interpolate: current joints → pre-grasp joints (APPROACHING)
        5. Interpolate: pre-grasp joints → grasp joints (FINAL_APPROACH)

        The two segments are stored separately so the state machine can
        transition between them.
        """
        from vlm_orchestrator.grasp.ik import (
            forward_kinematics,
            forward_kinematics_robolab,
            inverse_kinematics_multistart,
            _T_FLANGE_HAND,
            T_JOINT7_TO_ROBOTIQ_BASE,
        )

        # Clean-model ROBOLAB path: target the Robotiq base_link frame directly.
        # ``clean_ik`` gates the whole change; ``_ik_tcp`` is the fixed relabel
        # from the panda-hand frame (what forward_kinematics/IK natively use) to
        # the Robotiq base_link frame, so passing it as T_flange_to_tcp makes the
        # solver operate in the true controlled frame.  ``_fk_ee`` is the FK that
        # outputs that same frame, used for post-solve verification / logging.
        clean_ik = (
            GRASP_ROBOLAB_CLEAN_IK
            and self._env_mode == GraspEnvMode.ROBOLAB
        )
        _ik_tcp = None
        _fk_ee = forward_kinematics
        _C_rot = None  # axis-convention relabel (grasp/panda_hand → base_link)
        if clean_ik:
            _C = np.linalg.inv(_T_FLANGE_HAND) @ T_JOINT7_TO_ROBOTIQ_BASE
            _ik_tcp = _C
            _C_rot = _C[:3, :3].copy()
            _fk_ee = forward_kinematics_robolab

        # ---- GraspGen output frame ----
        # GraspGen returns a FLANGE (panda_hand) pose, NOT a fingertip
        # pose.  Inside GraspGen's ``build_6d_grasp()`` (action_decoder.py),
        # the grasp origin is computed as:
        #
        #     grasp_translation = contact_pt - gripper_depth * approach_dir
        #
        # where gripper_depth = 0.1034 m (see graspgen_franka_panda.yml).
        # This places the grasp frame origin 0.1034 m behind the fingertip
        # contact point — i.e. at the panda_hand (flange) location.
        #
        # Therefore we must NOT apply an additional TCP offset in IK.
        # Passing T_flange_to_tcp to the IK solver would subtract another
        # 0.1034 m, resulting in the arm being placed ~10.34 cm too far
        # from the object (double offset).
        self._T_flange_to_tcp = None  # GraspGen already accounts for gripper depth

        # Log observed EE + FK residual.  In the clean ROBOLAB path _fk_ee
        # outputs the true Robotiq base_link frame, so fk_vs_ee is ~0 (proving
        # base == world); in the legacy path it's the ~18 mm panda-model offset.
        ee_pos_obs = self._extract_ee_pos(self._last_obs or {})
        T_fk_now = _fk_ee(current_joints)
        fk_vs_ee = T_fk_now[:3, 3] - ee_pos_obs  # base_frame_pos - world_frame_pos
        logger.info(
            f"  FK ee-frame ({'Robotiq base_link' if clean_ik else 'panda_hand'}): "
            f"{T_fk_now[:3, 3]}, sim ee_pos: {ee_pos_obs}, "
            f"fk_vs_ee: {fk_vs_ee} (|{np.linalg.norm(fk_vs_ee)*1e3:.2f}mm|)"
        )
        self._grasp_log["perception_fk_pos"] = T_fk_now[:3, 3].tolist()
        self._grasp_log["perception_ee_pos"] = ee_pos_obs.tolist()
        self._grasp_log["perception_fk_vs_ee_delta"] = fk_vs_ee.tolist()
        self._grasp_log["ik_clean_robolab"] = bool(clean_ik)

        # ── Mount-transform calibration data (joint7 → sim-controlled frame) ──
        # Log q + full sim EE pose (pos AND quat) at the SAME instant so we can
        # solve T_joint7_to_baselink = inv(FK_joint7(q)) @ T_base_simEE offline.
        # This is what lets us replace the panda-hand assumptions
        # (FLANGE_Z_M=0.107, HAND_YAW_RAD=-π/2) with the true Robotiq mount.
        try:
            ee_quat_obs = self._extract_ee_quat(self._last_obs or {})
            if ee_quat_obs is not None:
                self._grasp_log["calib_joints"] = np.asarray(
                    current_joints, dtype=float
                ).tolist()
                self._grasp_log["calib_sim_ee_pos"] = np.asarray(
                    ee_pos_obs, dtype=float
                ).tolist()
                self._grasp_log["calib_sim_ee_quat_wxyz"] = np.asarray(
                    ee_quat_obs, dtype=float
                ).flatten().tolist()
        except Exception as _e:  # noqa: BLE001
            logger.debug(f"  mount-calib logging skipped: {_e}")

        # ---- Convert grasp target from world frame to robot base frame ----
        # Clean ROBOLAB path: robot base == world (proven by fk_vs_ee ≈ 0), and
        # IK targets the Robotiq base_link frame via _ik_tcp, so the target is
        # just grasp_pose_world.  Legacy path: forward_kinematics/IK are in the
        # panda-hand base frame; fk_vs_ee shifts world→base (also absorbing the
        # ~18 mm panda-model error).
        grasp_pose_base = grasp_pose_world.copy()
        if not clean_ik:
            grasp_pose_base[:3, 3] += fk_vs_ee

        # ---- Robotiq flange-depth correction (ROBOLAB / DROID only) ----
        # GraspGen's pose targets a panda_hand flange (0.1034 m behind the
        # fingertip contact).  The DROID robot's Robotiq flange→fingertip is
        # 0.1628 m, so without correction the Robotiq fingertips land ~0.0594 m
        # too deep.  Shift the IK target BACK along the grasp approach axis
        # (grasp frame +Z) by ROBOTIQ_DEPTH_CORRECTION_M so the *fingertips*
        # (not the flange) reach the contact point.  LIBERO (panda_hand) keeps
        # GraspGen's native frame — no shift.  This propagates to the pre-grasp
        # (derived from grasp_pose_base below), so the whole trajectory targets
        # the corrected fingertip pose.
        if (
            self._env_mode == GraspEnvMode.ROBOLAB
            and abs(ROBOTIQ_DEPTH_CORRECTION_M) > 1e-9
        ):
            corr_approach = grasp_pose_base[:3, 2]  # grasp frame +Z = approach
            grasp_pose_base[:3, 3] -= corr_approach * ROBOTIQ_DEPTH_CORRECTION_M
            self._grasp_log["robotiq_depth_correction_m"] = float(
                ROBOTIQ_DEPTH_CORRECTION_M
            )
            logger.info(
                f"  Robotiq flange-depth correction: shifted IK target "
                f"{ROBOTIQ_DEPTH_CORRECTION_M*1e3:+.1f}mm back along approach "
                f"(panda-flange → Robotiq-fingertip)"
            )

        # ---- Axis-convention relabel to the Robotiq base_link frame ----
        # (clean ROBOLAB path only.)  GraspGen's grasp frame and the Robotiq
        # base_link frame differ by a fixed rotation (_C_rot ≈ +90° about Y →
        # 120° geodesic).  The IK solver targets base_link (via _ik_tcp), so the
        # TARGET orientation must be expressed in base_link too: R_target =
        # R_grasp @ _C_rot.  The depth shift above is applied FIRST, using the
        # grasp-frame approach axis (grasp +Z), so it stays physically correct;
        # only the orientation is relabelled here.  Position is unchanged.
        # (In the legacy path this 120° relabel was absorbed by the panda-hand
        # HAND_YAW_RAD=-π/2 assumption inside forward_kinematics.)
        # Capture the PHYSICAL approach axis (grasp frame +Z, in world/base
        # coords) BEFORE any orientation relabel, so the pre-grasp backoff below
        # moves along the true approach direction regardless of frame.
        physical_approach = grasp_pose_base[:3, 2].copy()
        if clean_ik and _C_rot is not None:
            grasp_pose_base[:3, :3] = grasp_pose_base[:3, :3] @ _C_rot

        # ---- Obstacle cloud in robot base frame (collision-aware planners) ----
        # Same fk_vs_ee shift applied to the grasp target, so obstacles stay
        # registered with the IK targets.  Passed ONLY to the 'approach' phase:
        #   * 'final' = the last ~8 cm descent to the grasp pose; must not treat
        #     the just-subtracted target's neighbourhood as an obstacle.
        #   * 'lift'/'retreat' happen AFTER the grasp, when the object is in the
        #     gripper and the perception cloud is stale (object has moved), so
        #     feeding it would create phantom obstacles.  Those stay free-space.
        obstacle_pc_base = None
        if self._obstacle_pc_world is not None:
            obstacle_pc_base = self._obstacle_pc_world + fk_vs_ee
            self._grasp_log["motion_obstacle_pts"] = int(len(obstacle_pc_base))

        # ---- Compute pre-grasp pose ----
        # Back off along the PHYSICAL approach axis (grasp frame +Z in world
        # coords).  In the clean path grasp_pose_base's z-column is the
        # relabelled base_link z, so we use the pre-captured physical_approach.
        pre_grasp = grasp_pose_base.copy()
        approach_dir = physical_approach  # true grasp approach (world/base)
        pre_grasp[:3, 3] -= approach_dir * PRE_GRASP_OFFSET

        # ---- IK with multi-start ladder ----
        # Strategies tried in order: primary seed, wrist flip (±π on joint 6),
        # canonical reset (PERCEPTION_JOINTS), then wide random seeds.  Each
        # attempt must pass IK's internal pos+rot tolerances AND FK re-verification
        # (``accept_rot_deg`` guards against the ~180° _pose_error singularity
        # false-convergence that motivated the original topdown fallback).
        q_pre, conv_pre, strat_pre = inverse_kinematics_multistart(
            pre_grasp, current_joints, q_canonical=PERCEPTION_JOINTS,
            T_flange_to_tcp=_ik_tcp,
        )
        self._grasp_log["ik_pre_strategy"] = strat_pre
        if not conv_pre:
            raise RuntimeError(
                f"IK for pre-grasp failed (strategy={strat_pre}) — target likely "
                f"unreachable. pre_grasp pos={pre_grasp[:3,3].tolist()}, "
                f"reach={float(np.linalg.norm(pre_grasp[:3,3])):.3f}m, "
                f"approach={grasp_pose_base[:3,2].tolist()}"
            )
        logger.info(f"  IK for pre-grasp: strategy={strat_pre}")

        # For grasp IK, enforce branch consistency: reject solutions far from
        # q_pre in joint space.  Pre-grasp and grasp are only ~8cm apart in
        # Cartesian; the correct IK solutions must be joint-space neighbours.
        # Without this guard, a wide random seed can find a valid grasp config
        # in a different IK branch, making linear joint-space interpolation
        # between q_pre and q_grasp swing the arm through garbage.
        q_grasp, conv_grasp, strat_grasp = inverse_kinematics_multistart(
            grasp_pose_base, q_pre,
            q_canonical=None,               # no canonical; stay close to q_pre
            max_joint_dist_rad=1.0,         # reject far-branch solutions
            reference_q=q_pre,
            T_flange_to_tcp=_ik_tcp,
        )
        self._grasp_log["ik_grasp_strategy"] = strat_grasp
        if not conv_grasp:
            raise RuntimeError(
                f"IK for grasp failed (strategy={strat_grasp}) — no valid "
                f"solution within joint-space distance of q_pre. "
                f"grasp pos={grasp_pose_base[:3,3].tolist()}"
            )
        logger.info(f"  IK for grasp: strategy={strat_grasp}")
        used_topdown_fallback = False

        T_fk_pre = _fk_ee(q_pre)
        T_fk_grasp = _fk_ee(q_grasp)
        self._grasp_log["ik_pre_converged"] = conv_pre
        self._grasp_log["ik_grasp_converged"] = conv_grasp
        self._grasp_log["ik_used_topdown_fallback"] = used_topdown_fallback
        self._grasp_log["ik_pre_target_pos"] = pre_grasp[:3, 3].tolist()
        self._grasp_log["ik_grasp_target_pos"] = grasp_pose_base[:3, 3].tolist()
        self._grasp_log["ik_pre_fk_pos"] = T_fk_pre[:3, 3].tolist()
        self._grasp_log["ik_grasp_fk_pos"] = T_fk_grasp[:3, 3].tolist()
        self._grasp_log["ik_pre_pos_error"] = (T_fk_pre[:3, 3] - pre_grasp[:3, 3]).tolist()
        self._grasp_log["ik_grasp_pos_error"] = (T_fk_grasp[:3, 3] - grasp_pose_base[:3, 3]).tolist()

        # ---- Build trajectory segments ----
        # Segment 1: approach (current → pre-grasp, gripper open)
        approach_joints = self._plan_segment(
            current_joints, q_pre, INTERP_STEPS_APPROACH, phase="approach",
            scene_pc=obstacle_pc_base,
        )
        self._seg_approach = [
            np.concatenate([q, [GRIPPER_OPEN]]) for q in approach_joints
        ]

        # Segment 2: final approach (pre-grasp → grasp, gripper open)
        final_joints = self._plan_segment(
            q_pre, q_grasp, INTERP_STEPS_FINAL, phase="final",
        )
        self._seg_final = [
            np.concatenate([q, [GRIPPER_OPEN]]) for q in final_joints
        ]

        # Store grasp joint config for retreat planning
        self._q_at_grasp = q_grasp.copy()

        # Load the first segment
        self._trajectory = self._seg_approach
        logger.info(
            f"  Trajectory planned: approach={len(self._seg_approach)} "
            f"final={len(self._seg_final)} waypoints"
        )

    def _plan_lift_to_safe(self, current_joints: np.ndarray) -> None:
        """Plan a trajectory to move the arm to a known perception pose.

        Always moves to ``PERCEPTION_JOINTS`` — a retracted rest
        configuration where the over-shoulder camera has a clear,
        unoccluded view of the workspace.  Simply checking EE height
        is insufficient: the arm can be high but still blocking the
        camera (e.g. after dropping an object mid-transport).
        """
        from vlm_orchestrator.grasp.ik import forward_kinematics

        T_ee = forward_kinematics(current_joints)
        current_z = T_ee[2, 3]

        # Check if already very close to perception pose
        joint_dist = np.linalg.norm(current_joints - PERCEPTION_JOINTS)
        if joint_dist < 0.1:
            logger.info(
                f"  Already at perception pose (joint dist={joint_dist:.3f}), "
                f"skipping lift"
            )
            self._trajectory = []
            return

        # Move from current joints directly to perception joints.
        lift_joints = self._plan_segment(
            current_joints, PERCEPTION_JOINTS, INTERP_STEPS_LIFT, phase="lift",
        )
        self._trajectory = [
            np.concatenate([q, [GRIPPER_OPEN]]) for q in lift_joints
        ]
        logger.info(
            f"  Lift plan: EE z={current_z:.3f}m, joint_dist={joint_dist:.2f} "
            f"→ perception pose, {len(self._trajectory)} waypoints"
        )

    def _plan_retreat(self, obs: dict) -> None:
        """Plan a retreat trajectory (lift up with gripper closed)."""
        if self._env_mode == GraspEnvMode.LIBERO:
            self._plan_retreat_libero()
            return

        from vlm_orchestrator.grasp.ik import (
            forward_kinematics,
            inverse_kinematics_multistart,
        )

        q_now = self._q_at_grasp if self._q_at_grasp is not None else self._extract_joints(obs)

        # Current panda_hand pose (no TCP offset — matches GraspGen frame)
        T_ee = forward_kinematics(q_now)

        # Retreat: move up in world z
        T_retreat = T_ee.copy()
        T_retreat[2, 3] += RETREAT_HEIGHT

        # Retreat IK with branch consistency — we're holding an object, so
        # accepting a far-branch solution would swing the arm through bad
        # configs and likely drop the object.  Fall through to best-effort
        # on failure (retreat is non-critical; a short lift is still useful).
        q_retreat, conv, strat = inverse_kinematics_multistart(
            T_retreat, q_now,
            q_canonical=None,
            max_joint_dist_rad=1.0,
            reference_q=q_now,
        )
        if not conv:
            logger.warning(
                f"  IK for retreat did not converge ({strat}) — using best-effort"
            )
        else:
            logger.info(f"  IK for retreat: strategy={strat}")

        retreat_joints = self._plan_segment(
            q_now, q_retreat, INTERP_STEPS_RETREAT, phase="retreat",
        )
        self._trajectory = [
            np.concatenate([q, [GRIPPER_CLOSE]]) for q in retreat_joints
        ]
        self._traj_cursor = 0

    # ------------------------------------------------------------------
    # Perception helpers
    # ------------------------------------------------------------------

    def _get_gt_mask(
        self,
        obs: dict,
        target_object: str,
        image_hw: tuple[int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
        """Get a segmentation mask for the target object using GT info.

        Pipeline:
          1. If the seg buffer is available, extract the pixel-perfect mask
             directly from ``body_ids == matched_id``.  This is the ideal
             path — frame-accurate and doesn't require SAM2.
          2. If the seg buffer is unavailable or matching failed, fall back
             to SAM2 prompted with the gt_state 3D→2D projection.
          3. Bbox fallback if SAM2 is also unavailable.

        Returns ``(mask, bbox, score)``.
        """
        h, w = image_hw
        _sfx = "_front" if self._use_front_camera else ""

        # ── Step 1: try pixel-perfect mask from seg buffer ──
        body_ids = obs.get("gt_seg/body_ids")
        if body_ids is None:
            body_ids = obs.get(f"gt_seg/instance_ids{_sfx}")

        if body_ids is not None:
            body_ids_arr = np.asarray(body_ids, dtype=np.int32)
            result = self._match_centroid_to_gt(
                obs, target_object, body_ids_arr, h, w,
            )
            if result is not None:
                matched_id, cu, cv = result
                # Extract pixel-perfect mask from seg buffer
                seg_mask = (body_ids_arr == matched_id)
                seg_h, seg_w = seg_mask.shape[:2]
                # Resize to RGB image resolution if needed
                if (seg_h, seg_w) != (h, w):
                    import cv2 as _cv2
                    seg_mask = (
                        _cv2.resize(
                            seg_mask.astype(np.uint8) * 255,
                            (w, h),
                            interpolation=_cv2.INTER_NEAREST,
                        ) > 127
                    )
                bbox = self._mask_to_bbox(seg_mask)
                logger.info(
                    f"  [GT_SIM] pixel-perfect mask from seg buffer "
                    f"(id={matched_id}): {seg_mask.sum()} px, bbox={bbox}"
                )
                return seg_mask, bbox, 1.0
            logger.info(
                "  [GT_SIM] seg buffer available but centroid matching "
                "failed — falling through to SAM2/bbox"
            )

        # ── Step 2: no seg-buffer mask — locate center via gt_state ──
        center_uv, center_src = self._project_gt_position(
            obs, target_object, h, w,
        )

        if center_uv is None:
            raise RuntimeError(
                f"GT_SIM: could not locate '{target_object}' — "
                f"no seg buffer match and no gt_state available."
            )

        u, v = int(round(center_uv[0])), int(round(center_uv[1]))
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        logger.info(
            f"  [GT_SIM] center=({u},{v}) src={center_src}"
        )

        # ── Step 3: SAM2 with point prompt (no hardcoded box) ──
        # Let SAM2 determine the object extent from the point alone,
        # just as gdino_sam2 mode lets GDino supply the real bbox.
        try:
            image = self._extract_image(obs)
            mask, iou = self._grasp_client.segment(
                image, u / w, v / h,
            )
            logger.info(
                f"  GT_SIM → SAM2 (point-only): iou={iou:.3f}, "
                f"{mask.sum()} px"
            )
            if mask.sum() >= 50:
                bbox = self._mask_to_bbox(mask)
                return mask, bbox, 1.0
            logger.warning(f"  SAM2 mask too small ({mask.sum()} px)")
        except Exception as e:
            logger.warning(f"  SAM2 failed ({e})")

        # ── Step 4: bbox fallback (no SAM2 available at all) ──
        # Conservative small radius — only reached when SAM2 is down.
        fallback_radius = 20
        bbox = (max(0, u - fallback_radius), max(0, v - fallback_radius),
                min(w, u + fallback_radius), min(h, v + fallback_radius))
        mask = self._bbox_to_mask(bbox, (h, w))
        logger.info(
            f"  GT_SIM bbox fallback (no SAM2): ({u},{v}), "
            f"radius={fallback_radius}, {mask.sum()} px"
        )
        return mask, bbox, 1.0

    def _find_object_center(
        self,
        obs: dict,
        target_object: str,
        h: int,
        w: int,
    ) -> tuple[tuple[float, float] | None, str | None]:
        """Find the best 2D center pixel for the target object.

        Priority:
          A) Seg buffer centroid — frame-accurate (same render as image)
          B) gt_state 3D projection — may lag 1 frame

        Returns ``((u, v), source_description)`` or ``(None, None)``.
        """
        _sfx = "_front" if self._use_front_camera else ""

        # ── A) Seg buffer centroid ──
        body_ids = obs.get("gt_seg/body_ids")
        if body_ids is None:
            body_ids = obs.get(f"gt_seg/instance_ids{_sfx}")

        if body_ids is not None:
            body_ids = np.asarray(body_ids, dtype=np.int32)
            result = self._match_centroid_to_gt(
                obs, target_object, body_ids, h, w,
            )
            if result is not None:
                matched_id, cu, cv = result
                seg_h, seg_w = body_ids.shape[:2]
                center_uv = (cu * w / seg_w, cv * h / seg_h)
                return center_uv, f"seg_centroid(id={matched_id})"
            logger.info("  [GT_SIM] seg centroid matching failed")

        # ── B) gt_state 3D projection ──
        return self._project_gt_position(obs, target_object, h, w)

    def _project_gt_position(
        self,
        obs: dict,
        target_object: str,
        h: int,
        w: int,
    ) -> tuple[tuple[float, float] | None, str | None]:
        """Project the gt_state 3D position to a 2D pixel.

        Returns ``((u, v), "gt_projection")`` or ``(None, None)``.
        May be ~30-50 px off if the object is still settling.
        """
        gt_state = obs.get("gt_state")
        if gt_state is None:
            return None, None
        objects = gt_state.get("objects", {})

        obj_data = None
        target_us = target_object.replace(" ", "_")
        target_sp = target_object.replace("_", " ")
        for variant in (target_object, target_us, target_sp):
            if variant in objects:
                obj_data = objects[variant]
                break
        if obj_data is None:
            target_norm = target_us.lower()
            for name, data in objects.items():
                if target_norm in name.lower().replace(" ", "_"):
                    obj_data = data
                    break
        if obj_data is None:
            return None, None

        obj_pos = np.array(obj_data["pos"])
        try:
            cam_to_world = self._get_camera_to_world(obs)
        except Exception:
            return None, None
        world_to_cam = np.linalg.inv(cam_to_world)
        pos_cam = world_to_cam[:3, :3] @ obj_pos + world_to_cam[:3, 3]
        if pos_cam[2] <= 0:
            return None, None

        _sfx = "_front" if self._use_front_camera else ""
        obs_K = obs.get(f"observation/camera_K{_sfx}")
        if obs_K is None:
            obs_K = obs.get("observation/camera_K")
        if obs_K is not None:
            K = np.asarray(obs_K).reshape(3, 3)
        else:
            K = np.array([
                [self._intrinsics.fx, 0, self._intrinsics.cx],
                [0, self._intrinsics.fy, self._intrinsics.cy],
                [0, 0, 1],
            ])

        px = K @ pos_cam
        u = float(px[0] / px[2])
        v = float(px[1] / px[2])
        logger.info(
            f"  [GT_SIM] gt_projection: '{target_object}' "
            f"pos={obj_pos} → pixel ({u:.0f},{v:.0f})"
        )
        return (u, v), "gt_projection"

    def _match_centroid_to_gt(
        self,
        obs: dict,
        target_object: str,
        body_ids: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> tuple[int, float, float] | None:
        """Match a seg-buffer instance to the target via centroid proximity.

        The seg buffer is rendered in the SAME frame as the RGB image,
        so its pixel data is perfectly synchronised.  ``gt_state``
        positions may lag by a frame (object still settling / falling),
        so we only use them for **coarse** matching: project gt position
        → pixel, then pick the seg-buffer instance whose centroid is
        nearest to that projection.

        Returns ``(instance_id, centroid_u, centroid_v)`` in **seg-buffer
        coordinates**, or *None* if no plausible candidate is found.
        """
        # ── 1. Project gt_state position → approximate pixel ──
        gt_state = obs.get("gt_state", {})
        objects = gt_state.get("objects", {})

        obj_data = None
        target_us = target_object.replace(" ", "_")
        target_sp = target_object.replace("_", " ")
        for variant in (target_object, target_us, target_sp):
            if variant in objects:
                obj_data = objects[variant]
                break
        if obj_data is None:
            target_norm = target_us.lower()
            for name, data in objects.items():
                if target_norm in name.lower().replace(" ", "_"):
                    obj_data = data
                    break
        if obj_data is None:
            return None

        obj_pos = np.array(obj_data["pos"])
        try:
            cam_to_world = self._get_camera_to_world(obs)
        except Exception:
            return None
        world_to_cam = np.linalg.inv(cam_to_world)
        pos_cam = world_to_cam[:3, :3] @ obj_pos + world_to_cam[:3, 3]
        if pos_cam[2] <= 0:
            return None

        _sfx = "_front" if self._use_front_camera else ""
        obs_K = obs.get(f"observation/camera_K{_sfx}")
        if obs_K is None:
            obs_K = obs.get("observation/camera_K")
        if obs_K is not None:
            K = np.asarray(obs_K).reshape(3, 3)
        else:
            K = np.array([
                [self._intrinsics.fx, 0, self._intrinsics.cx],
                [0, self._intrinsics.fy, self._intrinsics.cy],
                [0, 0, 1],
            ])

        px = K @ pos_cam
        proj_u = px[0] / px[2]
        proj_v = px[1] / px[2]

        # ── 2. Compute centroids of all small-object IDs in seg buffer ──
        seg_h, seg_w = body_ids.shape[:2]
        total_px = seg_h * seg_w
        # Scale projection if seg buffer resolution differs from image
        scale_u = seg_w / img_w
        scale_v = seg_h / img_h
        proj_u_seg = proj_u * scale_u
        proj_v_seg = proj_v * scale_v

        unique_ids = np.unique(body_ids)
        candidates = []  # (id, centroid_u, centroid_v, npx, dist)
        for uid in unique_ids:
            uid = int(uid)
            if uid == 0:
                continue  # background
            id_mask = (body_ids == uid)
            npx = int(id_mask.sum())
            frac = npx / total_px
            # Skip IDs that are too large (table/robot: >3%) or too
            # small (<10 px: noise)
            if npx < 10 or frac > 0.03:
                continue
            ys, xs = np.where(id_mask)
            cu = float(xs.mean())
            cv = float(ys.mean())
            dist = math.sqrt((cu - proj_u_seg) ** 2 + (cv - proj_v_seg) ** 2)
            candidates.append((uid, cu, cv, npx, dist))

        if not candidates:
            logger.info(
                f"  [GT_SIM] centroid match: no small-object IDs in seg buffer"
            )
            return None

        # ── 3. Pick nearest centroid to projected position ──
        candidates.sort(key=lambda c: c[4])  # sort by distance
        best_id, best_cu, best_cv, best_npx, best_dist = candidates[0]

        # Reject if too far (>150 px — generous for 1-frame lag)
        max_dist = 150
        if best_dist > max_dist:
            logger.info(
                f"  [GT_SIM] centroid match: nearest ID={best_id} is "
                f"{best_dist:.0f}px away (>{max_dist}), rejecting. "
                f"Candidates: {[(c[0], c[3], f'{c[4]:.0f}px') for c in candidates[:5]]}"
            )
            return None

        logger.info(
            f"  [GT_SIM] centroid match: '{target_object}' → ID={best_id}, "
            f"centroid=({best_cu:.0f},{best_cv:.0f}), {best_npx}px, "
            f"dist={best_dist:.0f}px from projection ({proj_u_seg:.0f},{proj_v_seg:.0f}). "
            f"Others: {[(c[0], c[3], f'{c[4]:.0f}px') for c in candidates[1:4]]}"
        )
        return best_id, best_cu, best_cv

    @staticmethod
    def _resolve_gt_body_id(
        target_object: str,
        obj_body_id: dict,
    ) -> int | None:
        """Resolve target object name → MuJoCo body ID.

        Uses the mapping provided by the eval client's GT seg provider.
        Supports exact match, _main suffix, and fuzzy substring match.
        Normalises spaces ↔ underscores so ``"green block"`` matches
        ``"green_block"`` and vice-versa.
        """
        # Normalise: try both space and underscore variants
        target_us = target_object.replace(" ", "_")
        target_sp = target_object.replace("_", " ")

        # Exact match (original, underscore, space)
        for variant in (target_object, target_us, target_sp):
            if variant in obj_body_id:
                return obj_body_id[variant]

        # _main suffix (robosuite convention)
        for variant in (target_object, target_us, target_sp):
            with_main = f"{variant}_main"
            if with_main in obj_body_id:
                return obj_body_id[with_main]

        # Fuzzy substring match (normalise both sides to underscores)
        target_norm = target_us.lower()
        candidates = [
            (name, bid) for name, bid in obj_body_id.items()
            if target_norm in name.lower().replace(" ", "_")
        ]
        if candidates:
            # Take shortest name (most specific)
            candidates.sort(key=lambda x: len(x[0]))
            name, bid = candidates[0]
            logger.info(
                f"  GT_SIM: fuzzy body ID match "
                f"'{target_object}' → '{name}' (body_id={bid})"
            )
            return bid

        logger.warning(
            f"  GT_SIM: could not resolve '{target_object}' to body_id. "
            f"Available: {list(obj_body_id.keys())[:15]}"
        )
        return None

    @staticmethod
    def _mask_to_bbox(
        mask: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """Compute tight bounding box from a boolean mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return (0, 0, 0, 0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(x1), int(y1), int(x2) + 1, int(y2) + 1)

    def _detect_object(
        self, image: np.ndarray, target_object: str,
    ) -> tuple[tuple[int, int, int, int], float]:
        """Detect the target object using GroundingDINO via the grasp server.

        Returns ``((x1, y1, x2, y2), score)`` in pixel coordinates.

        Raises :class:`RuntimeError` if the grasp server's ``/detect``
        endpoint is not loaded — no silent fallback to a local
        :class:`GroundingDINODetector` (the "no silent lossy
        fallbacks" design rule).  A separate in-process detector would hide
        server-config bugs and load a duplicate ~1 GB model into the
        orchestrator process.  The operator must enable GDino on the
        grasp server explicitly.
        """
        import requests as _requests
        prompt = target_object.strip()
        if not prompt.endswith("."):
            prompt += "."

        try:
            detections = self._grasp_client.detect(image, prompt)
        except _requests.HTTPError as e:
            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    body = resp.json()
                    if isinstance(body, dict) and "detail" in body:
                        detail = str(body["detail"])
                except Exception:
                    try:
                        detail = resp.text[:200]
                    except Exception:
                        pass
            hint = (
                " (start the grasp server with --enable-gdino)"
                if resp is not None and resp.status_code == 501
                else ""
            )
            raise RuntimeError(
                f"GDino detect failed for '{target_object}': "
                f"{detail or e}{hint}"
            ) from e

        if not detections:
            raise RuntimeError(
                f"GroundingDINO found no detections for '{target_object}'"
            )

        # Take the highest-confidence detection
        best = max(detections, key=lambda d: d["score"])
        box = best["box"]  # [x1, y1, x2, y2] in pixel coords
        pixel_bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        return pixel_bbox, best["score"]

    def _molmo_sam2_segment(
        self,
        image: np.ndarray,
        target_object: str,
    ) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
        """Molmo2 pointing → SAM2 point-prompt segmentation.

        Returns ``(mask_HxW_bool, iou_score, pixel_bbox)``.  The bbox is
        the tight bounding box of the SAM2 mask (used downstream for
        debug visualisation; the actual grasp planning uses the mask
        directly).

        Reads MOLMO_BASE_URL / MOLMO_MODEL / MOLMO_API_KEY env vars
        (same convention as place tool's molmo_point seg-mode).
        """
        import os
        from vlm_orchestrator.perception.molmo import (
            DEFAULT_BASE_URL, DEFAULT_MODEL, MolmoPointError, point_at,
        )

        base_url = os.environ.get("MOLMO_BASE_URL", DEFAULT_BASE_URL)
        model = os.environ.get("MOLMO_MODEL", DEFAULT_MODEL)
        api_key = os.environ.get("MOLMO_API_KEY") or None

        try:
            mp = point_at(
                image_rgb=image,
                target_phrase=target_object,
                base_url=base_url,
                model=model,
                api_key=api_key,
            )
        except MolmoPointError as e:
            raise RuntimeError(
                f"Molmo pointing failed for {target_object!r}: {e}"
            ) from e

        # Hand the normalised point to SAM2 (no box prompt — let SAM2
        # grow the mask from the single point).
        try:
            mask, iou = self._grasp_client.segment(
                image, mp.x_norm, mp.y_norm, box=None,
            )
        except Exception as e:
            raise RuntimeError(
                f"SAM2 segmentation unavailable for Molmo point "
                f"({mp.x_norm:.3f}, {mp.y_norm:.3f}): {e}"
            ) from e

        logger.info(
            f"  Molmo+SAM2: point=({mp.x_norm:.3f}, {mp.y_norm:.3f}) "
            f"iou={iou:.3f}, mask_px={int(mask.sum())}"
        )
        if mask.sum() < 50:
            raise RuntimeError(
                f"SAM2 from Molmo point returned degenerate mask "
                f"({int(mask.sum())} px, iou={iou:.3f}) for "
                f"{target_object!r} at "
                f"({mp.x_norm:.3f}, {mp.y_norm:.3f}). "
                f"Molmo likely pointed at the wrong object — retry "
                f"with a more specific phrase."
            )

        # Tight bbox from the mask for downstream debug overlays.
        ys, xs = np.where(mask)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        return mask, float(iou), (x1, y1, x2, y2)

    def _vlm_sam2_segment(
        self,
        image: np.ndarray,
        target_object: str,
    ) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
        """Orchestrator VLM pointing → SAM2 point-prompt segmentation.

        Same shape as :meth:`_molmo_sam2_segment` but uses the
        orchestrator's main VLM (Claude / Bedrock / GPT — any
        OpenAI-compatible chat backend) for the pointing step.

        Reads VLM_BASE_URL / VLM_MODEL / VLM_API_KEY (or
        OPENAI_API_KEY) env vars.  Defaults to the placeholder VLM
        endpoint/model; set these env vars (or the CLI flags) to point at
        your OpenAI-compatible VLM for the primary subgoal+replan_tools
        configuration.
        """
        import json
        import os
        from openai import OpenAI
        from vlm_orchestrator.vlm import encode_image_b64, parse_json

        base_url = os.environ.get(
            "VLM_BASE_URL", "https://YOUR_VLM_ENDPOINT/v1",
        )
        model = os.environ.get(
            "VLM_MODEL", "YOUR_VLM_MODEL",
        )
        api_key = (
            os.environ.get("VLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "seg_mode='vlm_sam2' requires VLM_API_KEY or "
                "OPENAI_API_KEY in the environment."
            )

        system_prompt = (
            "You are a robot's spatial-reasoning assistant.  Given a "
            "camera image and a target object phrase, return the single "
            "2D pixel that lies on the target object (ideally near the "
            "centre of its grasp surface).\n\n"
            "Output ONLY a JSON object with three fields:\n"
            '  "x_norm":   float in [0, 1]  (0 = left edge, 1 = right edge)\n'
            '  "y_norm":   float in [0, 1]  (0 = top edge, 1 = bottom edge)\n'
            '  "rationale": one short sentence explaining the chosen pixel\n\n'
            "If the target is not visible, still output your best guess "
            "and explain low confidence in the rationale — DO NOT return "
            "an obviously off-image coordinate as a hedge."
        )
        user_text = (
            f'Target object to grasp: "{target_object}"\n\n'
            "Return ONLY the JSON object — no markdown, no extra text."
        )
        image_b64 = encode_image_b64(image)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            },
        ]

        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=300,
                messages=messages,
            )
        except Exception as e:
            raise RuntimeError(
                f"VLM pointing call failed for {target_object!r}: {e}"
            ) from e

        raw = response.choices[0].message.content or ""
        try:
            data = parse_json(raw)
            x_norm = float(data["x_norm"])
            y_norm = float(data["y_norm"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"VLM pointing returned invalid JSON for "
                f"{target_object!r}: {e}\nraw response: {raw!r}"
            ) from e

        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            raise RuntimeError(
                f"VLM pointing returned out-of-image coords "
                f"x_norm={x_norm}, y_norm={y_norm} for "
                f"{target_object!r}; refusing silent clip "
                f"(no-silent-fallback design rule)."
            )

        # Hand the normalised point to SAM2 (no box prompt).
        try:
            mask, iou = self._grasp_client.segment(
                image, x_norm, y_norm, box=None,
            )
        except Exception as e:
            raise RuntimeError(
                f"SAM2 segmentation unavailable for VLM point "
                f"({x_norm:.3f}, {y_norm:.3f}): {e}"
            ) from e

        logger.info(
            f"  VLM+SAM2: point=({x_norm:.3f}, {y_norm:.3f}) "
            f"iou={iou:.3f}, mask_px={int(mask.sum())}"
        )
        if mask.sum() < 50:
            raise RuntimeError(
                f"SAM2 from VLM point returned degenerate mask "
                f"({int(mask.sum())} px, iou={iou:.3f}) for "
                f"{target_object!r} at ({x_norm:.3f}, {y_norm:.3f}). "
                f"VLM likely pointed at the wrong object — retry "
                f"with a more specific phrase."
            )

        # Tight bbox from the mask for downstream debug overlays.
        ys, xs = np.where(mask)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        return mask, float(iou), (x1, y1, x2, y2)

    def _segment_object(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Segment the object inside *bbox* using SAM2 (via grasp server).

        Raises ``RuntimeError`` if SAM2 is unavailable or returns a degenerate
        mask.  The former "fall back to rectangular bbox mask" was removed
        because it silently poisoned the point cloud with background pixels
        (the bbox corners include table / adjacent objects) — grasp poses
        computed from that mask average across the real target and its
        surroundings.  If SAM2 can't produce a precise mask, fail the grasp
        and let the caller escalate (retry, VLM replan, human-in-the-loop).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 / w   # normalised center
        cy = (y1 + y2) / 2.0 / h

        try:
            mask, iou = self._grasp_client.segment(image, cx, cy, box=bbox)
        except Exception as e:
            raise RuntimeError(
                f"SAM2 segmentation unavailable ({e}). The grasp server "
                f"needs SAM2 loaded; bbox-rectangle fallback is not accurate "
                f"enough for grasp pose estimation."
            ) from e

        logger.info(f"  SAM2 segmentation: iou={iou:.3f}, mask_px={mask.sum()}")
        if mask.sum() < 50:
            raise RuntimeError(
                f"SAM2 returned degenerate mask ({int(mask.sum())} px, "
                f"iou={iou:.3f}) for bbox={bbox}. Object likely misdetected "
                f"or occluded; aborting grasp rather than fabricating a mask."
            )
        return mask

    @staticmethod
    def _bbox_to_mask(
        bbox: tuple[int, int, int, int],
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        """Create a boolean mask from a bounding box (fallback)."""
        h, w = image_hw
        mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        mask[y1:y2, x1:x2] = True
        return mask

    # ------------------------------------------------------------------
    # Observation extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_joints(obs: dict) -> np.ndarray:
        """Extract 7-DOF joint positions from obs dict.

        Returns zeros if joint_position is not available (e.g. LIBERO,
        where the eval client doesn't always send it).
        """
        j = obs.get("observation/joint_position")
        if j is None:
            # LIBERO doesn't require joint positions for EE-delta control.
            # Return zeros so callers that check presence don't crash.
            raise RuntimeError(
                "observation/joint_position not in obs. "
                "For LIBERO, use env_mode='libero' (EE-delta trajectory)."
            )
        return np.asarray(j, dtype=np.float64).flatten()[:7]

    @staticmethod
    def _extract_gripper(obs: dict) -> float:
        """Extract scalar gripper position.

        Returns the normalized DROID gripper position:

          - 0.0 = fully OPEN  (fingers wide apart, max gap)
          - 1.0 = fully CLOSED (fingers together, zero gap)

        Computed by ``robolab/robots/droid.py:gripper_pos`` as
        ``finger_joint / (pi/4)`` (the raw revolute joint angle
        divided by its closed-state target).  This matches the
        COMMAND convention used by ``BinaryJointPositionActionCfg``
        for the DROID gripper (open=0.0, close=pi/4).  Both
        ``GRIPPER_OPEN`` and ``GRIPPER_CLOSE`` constants in this
        module use the same scale (0=open, 1=close) for sending
        actions.

        Earlier docstrings reversed the convention; we caught and
        corrected it on the RecoverNonFoodInBinTask run (2026-05-11)
        after observing the gripper visually held the cube at
        ``gripper_pos=0.000`` (fingers blocked at open, never
        moved) while empty-air closes reached ``gripper_pos ≈
        0.05`` (fingers traveled a bit before stopping).
        """
        g = obs.get("observation/gripper_position")
        if g is None:
            return 0.0
        return float(np.asarray(g).flatten()[0])

    @staticmethod
    def _intrinsics_from_K(
        K: np.ndarray, depth: np.ndarray,
    ) -> "CameraIntrinsics":
        """Build ``CameraIntrinsics`` from a 3×3 intrinsic matrix.

        Isaac Lab's ``intrinsic_matrices`` are computed at the camera's
        actual render resolution.  We use the depth image dimensions
        (guaranteed to match) rather than inferring from the principal
        point, because ``cx, cy`` may not be exactly ``w/2, h/2`` for
        off-centre principal points.
        """
        K = K.reshape(3, 3)
        h, w = np.squeeze(depth).shape[:2]
        return CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=w,
            height=h,
        )

    @staticmethod
    def _extract_ee_pos(obs: dict) -> np.ndarray:
        """Extract 3-D end-effector position."""
        p = obs.get("observation/ee_pos")
        if p is None:
            return np.zeros(3)
        return np.asarray(p, dtype=np.float64).flatten()[:3]

    @staticmethod
    def _extract_ee_quat(obs: dict) -> np.ndarray | None:
        """Extract end-effector quaternion (w, x, y, z).

        Returns ``None`` when the observation doesn't contain the key,
        so callers can distinguish "not available" from a real identity
        quaternion.
        """
        q = obs.get("observation/ee_quat")
        if q is None:
            return None
        return np.asarray(q, dtype=np.float64).flatten()[:4]

    def _extract_image(self, obs: dict) -> np.ndarray:
        """Extract the primary RGB image (H, W, 3) uint8.

        Uses the front camera when ``use_front_camera=True`` and front
        depth is available.  Otherwise uses the exterior / agentview
        camera.

        For LIBERO: ``libero_eval_client`` applies ``[::-1, ::-1]`` to
        ``agentview_image`` and ``agentview_depth``.  The first ``[::-1]``
        converts MuJoCo's OpenGL framebuffer (origin bottom-left, Y up)
        to OpenCV convention (origin top-left, Y down) — needed because
        ``robosuite/macros.py`` ships with ``IMAGE_CONVENTION = "opengl"``.
        The *second* ``[::-1]`` is an extra X-mirror applied to match
        pi0.5's training distribution.  ``camera_K`` and
        ``camera_extrinsic`` come from MuJoCo for the *un-mirrored*
        OpenCV camera, so we must un-mirror image and depth here
        (``[:, ::-1]``) to keep the grasp pipeline self-consistent.
        Without this, the point cloud's X coordinates are flipped about
        the optical axis and ``cam_to_world @ grasp_pose_camera``
        produces a world target reflected from the actual object — the
        robot reaches the (wrong) target precisely while missing the
        object.

        For VLABench: the eval client packs raw MuJoCo y-up imagery in
        ``observation/image`` (matching what pi05-VLABench was trained
        on) AND a y-flipped OpenCV-convention copy in
        ``observation/image_for_grasp`` for the grasp tool. We prefer
        the latter when present and apply NO further mirror — it's
        already in the convention ``camera_K`` / ``camera_extrinsic``
        expect. LIBERO and robolab clients don't pack
        ``observation/image_for_grasp``, so this branch is a no-op for
        them.
        """
        # VLABench: dedicated OpenCV-convention image already y-flipped
        # by the eval client. No mirror manipulation needed.
        img = obs.get("observation/image_for_grasp")
        if img is not None and hasattr(img, "shape"):
            return np.asarray(img, dtype=np.uint8)
        # Front camera — only when explicitly requested AND depth is co-registered
        if self._use_front_camera and obs.get("observation/depth_front") is not None:
            for key in (
                "observation/front_image_left_raw",
                "observation/front_image_left",
            ):
                img = obs.get(key)
                if img is not None and hasattr(img, "shape"):
                    return np.asarray(img, dtype=np.uint8)
            # LOUD FAIL (no silent lossy fallbacks (design rule)).  Front camera
            # was requested and front DEPTH is present, but no front RGB.
            # Silently falling back to the EXTERIOR image here segments a
            # DIFFERENT camera than the depth/intrinsics used to build the
            # point cloud → cross-camera mask/point-cloud mismatch.  The eval
            # client must forward the front RGB (robolab: egocentric_mirrored
            # _camera → observation/front_image_left_raw in _orchestrator_keys).
            raise RuntimeError(
                "Front camera requested (use_front_camera=True) and "
                "observation/depth_front is present, but no front RGB image "
                "(observation/front_image_left[_raw]).  Refusing to fall back "
                "to the exterior image — that would segment a different camera "
                "than the depth was captured from.  Ensure the eval client "
                "forwards the front RGB (robolab: egocentric_mirrored_camera)."
            )
        # Exterior camera (robolab)
        for key in (
            "observation/exterior_image_1_left_raw",
            "observation/exterior_image_1_left",
        ):
            img = obs.get(key)
            if img is not None and hasattr(img, "shape"):
                return np.asarray(img, dtype=np.uint8)
        # LIBERO agentview
        for key in (
            "observation/image_raw",
            "observation/image",
        ):
            img = obs.get(key)
            if img is not None and hasattr(img, "shape"):
                arr = np.asarray(img, dtype=np.uint8)
                if self._env_mode == GraspEnvMode.LIBERO:
                    arr = np.ascontiguousarray(arr[:, ::-1])
                return arr
        raise RuntimeError("No RGB image found in observation")

    def _extract_depth(self, obs: dict) -> np.ndarray | None:
        """Extract depth image (H, W) float32 in metres, or None.

        Matches the camera selected by ``_extract_image``: front camera
        when ``use_front_camera=True``, exterior/agentview otherwise.

        For LIBERO: un-mirrors depth horizontally to match the un-mirrored
        image (see ``_extract_image`` docstring).

        For VLABench: the eval client packs ``observation/depth_for_grasp``
        already y-flipped to OpenCV convention. We prefer it when
        present with no further mirroring. LIBERO/robolab don't pack
        this key — branch is a no-op for them.
        """
        # VLABench: dedicated OpenCV-convention depth.
        d = obs.get("observation/depth_for_grasp")
        if d is not None:
            return np.asarray(d, dtype=np.float32)
        if self._use_front_camera:
            d = obs.get("observation/depth_front")
            if d is not None:
                return np.asarray(d, dtype=np.float32)
        # Exterior camera (robolab)
        d = obs.get("observation/depth_external")
        if d is not None:
            return np.asarray(d, dtype=np.float32)
        # LIBERO agentview depth
        d = obs.get("observation/depth_agentview")
        if d is not None:
            arr = np.asarray(d, dtype=np.float32)
            if self._env_mode == GraspEnvMode.LIBERO:
                arr = np.ascontiguousarray(arr[:, ::-1])
            return arr
        # Last resort: front depth (even when not explicitly requested,
        # some setups only have front depth)
        d = obs.get("observation/depth_front")
        if d is not None:
            return np.asarray(d, dtype=np.float32)
        return None

    def _get_camera_to_world(self, obs: dict) -> np.ndarray:
        """Get the camera-to-world transform (4×4, OpenCV convention).

        Reads runtime camera pose from obs — ``observation/camera_extrinsic``
        (LIBERO/RoboCasa) or ``camera_pos``/``camera_quat`` (robolab, optionally
        with ``_front`` suffix when the front camera is active).

        Raises ``RuntimeError`` if no runtime pose is available.  The former
        "static hardcoded fallback" was removed because it silently substituted
        a baked-in camera pose (OverShoulderLeft or EgocentricMirrored) when
        the obs didn't carry one — producing decimetre-scale errors in all
        downstream 3D grasp targets whenever the sim's camera was moved or a
        new camera configuration was added.
        """
        # ── LIBERO / RoboCasa: extrinsic matrix packed by eval client ──
        cam_ext = obs.get("observation/camera_extrinsic")
        if cam_ext is not None:
            T = np.asarray(cam_ext, dtype=np.float64).reshape(4, 4)
            # robosuite's get_camera_extrinsic_matrix returns camera-to-world
            # (camera pose in world frame, with axis correction to OpenCV).
            # This is already what we need — no inversion required.
            return T

        # ── robolab: runtime camera pose (OpenGL convention) ──
        _sfx = "_front" if self._use_front_camera else ""
        cam_pos = obs.get(f"observation/camera_pos{_sfx}")
        cam_quat = obs.get(f"observation/camera_quat{_sfx}")

        if cam_pos is not None and cam_quat is not None:
            pos = np.asarray(cam_pos).flatten()[:3]
            quat = np.asarray(cam_quat).flatten()[:4]
            logger.info(
                f"  Camera pose from obs: pos={pos}, "
                f"quat(wxyz)={quat}, sfx='{_sfx}'"
            )
            return pose_opengl_to_opencv(pos, quat)

        raise RuntimeError(
            f"No camera pose in obs (looked for observation/camera_extrinsic, "
            f"observation/camera_pos{_sfx}, observation/camera_quat{_sfx}). "
            f"The eval client must forward camera extrinsics for accurate "
            f"3D grasp-target computation."
        )

    def _noop_response(self, obs: dict) -> dict:
        """Return a hold-position action chunk.

        Critical: the gripper element of the action must be the
        last COMMANDED gripper state, not the observed one.  The
        eval client (pi0_family.py:140-143) binarizes
        ``action[-1] > 0.5`` for the robolab gripper; if we echo the
        observed gripper (in [0, 1] where 0=open, 1=closed) we flip
        any partial closure (< 0.5) to OPEN — and the partial
        closure is exactly what a successful grasp looks like
        (fingers blocked by the object).  Echoing the observation
        therefore DROPS the held object on every DONE / FAILED /
        unexpected-phase chunk.

        ``self._last_commanded_gripper`` is updated everywhere we
        emit an action so it reflects what we want the gripper to
        be doing right now.
        """
        if self._env_mode == GraspEnvMode.LIBERO:
            action = np.zeros(self._action_dim)
            action[-1] = self._last_commanded_gripper
            actions = np.tile(action, (self._action_horizon, 1))
            return {"actions": actions}
        try:
            joints = self._extract_joints(obs)
        except RuntimeError:
            joints = np.zeros(7)
        action = np.concatenate([joints, [self._last_commanded_gripper]])
        actions = np.tile(action, (self._action_horizon, 1))
        return {"actions": actions}

    # ------------------------------------------------------------------
    # LIBERO EE-delta trajectory planning
    # ------------------------------------------------------------------

    def _plan_trajectory_libero(
        self,
        current_ee_pos: np.ndarray,
        current_ee_quat: np.ndarray | None,
        grasp_pose_world: np.ndarray,
    ) -> None:
        """Build approach + final-approach trajectory using Cartesian interpolation.

        Instead of IK, directly produce OSC_POSE-compatible EE-delta actions:
        ``[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`` (7D).

        The delta is in world frame, scaled to the OSC_POSE input range
        (position max ±0.05 m, orientation max ±0.5 rad per step).

        Phases:
          1. Approach: current EE → pre-grasp position (with orientation)
          2. Final: pre-grasp → grasp position (hold orientation)
          3. (Close gripper — handled by _step_closing)
          4. Retreat: grasp → lift up (handled by _plan_retreat_libero)
        """
        from scipy.spatial.transform import Rotation as R

        grasp_pos = grasp_pose_world[:3, 3]
        approach_dir = grasp_pose_world[:3, 2]  # z-axis of grasp frame

        # Pre-grasp: offset along negative approach direction
        pre_grasp_pos = grasp_pos - approach_dir * LIBERO_PRE_GRASP_OFFSET

        # Compute total orientation delta (current → grasp frame).
        # Grasp frame rotation from the planner.
        grasp_rot = R.from_matrix(grasp_pose_world[:3, :3])

        # Convert GraspGen axis convention (X=open) → panda_hand convention
        # (Y=open) by post-rotating about the grasp's own Z (approach) axis.
        # See LIBERO_GRASPGEN_TO_EE_YAW_RAD docstring above.
        if LIBERO_GRASPGEN_TO_EE_YAW_RAD != 0.0:
            grasp_rot = grasp_rot * R.from_euler(
                "z", LIBERO_GRASPGEN_TO_EE_YAW_RAD,
            )

        if current_ee_quat is not None:
            # current_ee_quat is (w, x, y, z) → scipy expects (x, y, z, w)
            current_rot = R.from_quat([
                current_ee_quat[1], current_ee_quat[2],
                current_ee_quat[3], current_ee_quat[0],
            ])
        else:
            # No orientation data — assume identity; orientation deltas
            # will still be computed but may be inaccurate.
            current_rot = R.identity()

        # Total rotation delta: current → grasp, as axis-angle (radians).
        # OSC_POSE applies orientation delta as `goal = R_delta @ R_current`
        # (left-multiplication, see robosuite control_utils.set_goal_orientation),
        # so delta must be expressed in the WORLD frame:
        #     R_delta_world = R_grasp @ R_current.inv()
        rot_delta = (grasp_rot * current_rot.inv()).as_rotvec()
        # Spread orientation change across approach phase only
        n_approach = LIBERO_INTERP_APPROACH
        ori_per_step_rad = rot_delta / max(n_approach, 1)
        # Clamp per-step rotation to LIBERO_MAX_ORI_DELTA radians
        ori_per_step_rad = np.clip(
            ori_per_step_rad, -LIBERO_MAX_ORI_DELTA, LIBERO_MAX_ORI_DELTA,
        )
        # Convert radians → OSC_POSE normalised input units
        ori_per_step = ori_per_step_rad / OSC_POSE_ORI_SCALE

        # Closed-loop state for APPROACHING.  Target rotation is the
        # corrected grasp_rot — closed-loop step recomputes per-step
        # rotation each chunk so we don't overshoot.  ori_per_step is
        # kept as an open-loop fallback for the case where ee_quat is
        # unavailable.
        self._libero_pre_grasp_pos = pre_grasp_pos.copy()
        self._libero_grasp_pos = grasp_pos.copy()
        self._libero_seg_target_pos = pre_grasp_pos.copy()
        self._libero_seg_target_rot = grasp_rot.as_matrix()
        self._libero_seg_ori_per_step_norm = (
            ori_per_step_rad / OSC_POSE_ORI_SCALE
        )
        self._libero_seg_remaining_steps_estimate = n_approach

        # ── Segment 1: approach (current → pre-grasp + rotate to grasp ori) ──
        # Open-loop fallback (closed-loop step is preferred at runtime).
        approach_deltas = self._interpolate_ee_deltas(
            current_ee_pos, pre_grasp_pos, n_approach,
        )
        self._seg_approach = [
            np.concatenate([
                d / OSC_POSE_POS_SCALE,        # m → unit-fraction
                ori_per_step,
                [self._gripper_open],
            ])
            for d in approach_deltas
        ]

        # ── Segment 2: final approach (pre-grasp → grasp, hold orientation) ──
        final_deltas = self._interpolate_ee_deltas(
            pre_grasp_pos, grasp_pos, LIBERO_INTERP_FINAL,
        )
        self._seg_final = [
            np.concatenate([
                d / OSC_POSE_POS_SCALE,
                np.zeros(3),
                [self._gripper_open],
            ])
            for d in final_deltas
        ]

        # Store grasp position for retreat planning
        self._ee_at_grasp = grasp_pos.copy()

        # Load the first segment
        self._trajectory = self._seg_approach
        logger.info(
            f"  LIBERO trajectory planned: approach={len(self._seg_approach)} "
            f"final={len(self._seg_final)} waypoints (EE-delta), "
            f"ori_delta={np.linalg.norm(rot_delta):.2f} rad"
        )

    def _plan_retreat_libero(self) -> None:
        """Plan a LIBERO retreat: lift straight up with gripper closed."""
        # Lift: move up in world z over several steps. Compute in metres,
        # cap at the per-step Cartesian limit, then convert to OSC_POSE
        # normalised input units (m / OSC_POSE_POS_SCALE).
        up_delta_per_step_m = LIBERO_RETREAT_HEIGHT / LIBERO_INTERP_RETREAT
        up_delta_per_step_m = min(up_delta_per_step_m, LIBERO_MAX_POS_DELTA)
        up_delta_per_step = up_delta_per_step_m / OSC_POSE_POS_SCALE

        self._trajectory = []
        for _ in range(LIBERO_INTERP_RETREAT):
            action = np.zeros(self._action_dim)
            action[2] = up_delta_per_step    # Δz (up), normalised
            action[-1] = self._gripper_close
            self._trajectory.append(action)
        self._traj_cursor = 0

    @staticmethod
    def _libero_extract_ee_pos(obs: dict) -> np.ndarray | None:
        """Best-effort EE position read for LIBERO obs.

        LIBERO eval clients pack ``observation/ee_pos``; older clients
        only ship ``observation/state`` whose first 3 dims are EE xyz.
        Returns None when neither is available so the caller can fall
        back to perception without lifting.
        """
        ee_pos = obs.get("observation/ee_pos")
        if ee_pos is None:
            state_vec = obs.get("observation/state")
            if state_vec is not None and len(state_vec) >= 3:
                ee_pos = np.asarray(state_vec[:3], dtype=np.float64)
        if ee_pos is None:
            return None
        return np.asarray(ee_pos, dtype=np.float64).flatten()[:3]

    def _plan_perception_lift_libero(
        self, current_ee_pos: np.ndarray,
    ) -> None:
        """Plan an EE-delta lift that clears the arm from the agentview FOV.

        Sets the closed-loop target so ``_step_trajectory_libero_closed_loop``
        recomputes the per-step delta each chunk from the latest current
        EE pos (open-loop impedance lag would leave the arm short of the
        target).  The pre-computed open-loop trajectory is also populated
        as a backup / step-budget reference.
        """
        target_pos = current_ee_pos + np.array([
            LIBERO_PERCEPTION_LIFT_DX,
            0.0,
            LIBERO_PERCEPTION_LIFT_DZ,
        ])
        # Closed-loop target for LIFTING phase: hold orientation,
        # only translate.  target_rot=None tells the closed-loop step
        # to use the zero-delta open-loop fallback for orientation.
        self._libero_seg_target_pos = target_pos.copy()
        self._libero_seg_target_rot = None
        self._libero_seg_ori_per_step_norm = np.zeros(3)
        self._libero_seg_remaining_steps_estimate = LIBERO_INTERP_PERCEPTION_LIFT

        # Open-loop fallback (used only when closed-loop is disabled).
        deltas = self._interpolate_ee_deltas(
            current_ee_pos, target_pos, LIBERO_INTERP_PERCEPTION_LIFT,
        )
        self._trajectory = []
        for d in deltas:
            action = np.zeros(self._action_dim)
            action[:3] = d / OSC_POSE_POS_SCALE     # m → unit-fraction
            action[-1] = self._gripper_open
            self._trajectory.append(action)
        self._traj_cursor = 0

    @staticmethod
    def _interpolate_ee_deltas(
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        n_steps: int,
    ) -> list[np.ndarray]:
        """Produce n_steps EE position deltas to move from start to end.

        Each delta is capped at ``LIBERO_MAX_POS_DELTA`` per axis.
        If the total displacement requires more than n_steps at max speed,
        the move is truncated (robot won't reach target — increase n_steps).
        """
        total_delta = end_pos - start_pos
        per_step = total_delta / max(n_steps, 1)

        # Clamp to OSC_POSE limits
        per_step = np.clip(per_step, -LIBERO_MAX_POS_DELTA, LIBERO_MAX_POS_DELTA)

        deltas = []
        remaining = total_delta.copy()
        for _ in range(n_steps):
            step = np.clip(remaining / max(1, n_steps - len(deltas)),
                           -LIBERO_MAX_POS_DELTA, LIBERO_MAX_POS_DELTA)
            deltas.append(step.copy())
            remaining -= step
        return deltas

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the executor to idle state."""
        self._phase = GraspPhase.IDLE
        self._target_object = ""
        self._status_message = ""
        self._last_obs = None
        self._grasp_pose_camera = None
        self._grasp_confidence = 0.0
        self._intrinsics_used = None
        self._grasp_log = {}
        self._trajectory = []
        self._traj_cursor = 0
        self._seg_approach = []
        self._seg_final = []
        self._q_at_grasp = None
        self._ee_at_grasp = None
        self._T_flange_to_tcp = None
        self._libero_pre_grasp_pos = None
        self._libero_grasp_pos = None
        self._libero_seg_target_pos = None
        self._libero_seg_target_rot = None
        self._libero_seg_ori_per_step_norm = np.zeros(3)
        self._libero_seg_remaining_steps_estimate = 0
        # Reset to a value far in the past so that, on next phase entry,
        # the (episode_step - start) delta starts cleanly from zero.
        self._step_at_close_start = -CLOSE_HOLD_STEPS
        self._step_at_release_start = -RELEASE_HOLD_STEPS
        self._skip_lift_after_release = True
