# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Place-with-tool executor: planned placement that bypasses the VLA.

Mirrors :mod:`vlm_orchestrator.grasp.tool` one-to-one in shape: a state
machine driven by ``step()`` calls from the proxy.  Each ``step()``
returns ``{"actions": np.ndarray}`` identical to a VLA response.

State machine
-------------

    IDLE
      ↓ start(destination, obs, state)
    ENSURE_CLOSED   if gripper open: command close N chunks; else pass through
      ↓
    LIFTING         IK to PERCEPTION_JOINTS (only if EE below SAFE_HEIGHT_Z)
      ↓
    PERCEIVING      destination grounding → 2D pixel → 3D world point
      ↓
    PLANNING        IK + waypoints (pre-place, place, retreat)
      ↓
    APPROACHING     current → pre-place (gripper closed)
      ↓
    FINAL_APPROACH  pre-place → place (gripper closed)
      ↓
    MEASURING       log EE-vs-target delta
      ↓
    RELEASING       open gripper, hold N chunks
      ↓
    SETTLING        wait for object to land
      ↓
    RETREATING      lift EE up, gripper open
      ↓
    DONE | FAILED

The tool reports ``DONE`` once ``RETREATING`` completes.  Task-success
determination (did the object actually land in the target?) is upstream
— the strategy / failure detector handles it.

Failure reasons
---------------

``PlacePhase.FAILED`` ships with a short ``failure_reason`` string:

* ``"perception_no_target"`` — destination grounding produced nothing.
* ``"ik_unreachable"``       — multi-start IK exhausted seeds.
* ``"motion_blocked"``       — measured joint position lagged commanded.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from vlm_orchestrator.motion import MotionPlanner

from vlm_orchestrator.grasp.camera import (
    CameraIntrinsics,
    front_camera_intrinsics,
    overshoulder_left_intrinsics,
    pose_opengl_to_opencv,
)
from vlm_orchestrator.place.client import PlaceClient
from vlm_orchestrator.place.destination import (
    PerceptionFailure,
    PointToPlace2D,
    point_to_place_2d,
    raycast_2d_to_3d,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (mirrors grasp/tool.py shape; robolab-only — no LIBERO branch)
# ---------------------------------------------------------------------------

ACTION_HORIZON = 8
ACTION_DIM = 8
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0

# Below this gripper_position counts as "already closed" (empty or holding).
PLACE_GRIPPER_CLOSED_THRESHOLD = 0.06

# Approach geometry (m)
PLACE_PRE_OFFSET = 0.08            # pre-place height above the release pose
PLACE_RETREAT_HEIGHT = 0.12        # retreat lift after release

# Drop-from-safe-height model.  We do NOT try to know the held object's
# size — that's privileged GT info on robolab, real-world unavailable.
# Instead, use an empirical strategy: position the held-object
# centroid this far above the destination surface, open the gripper, let
# gravity finish the placement.  The finger-tip ends up
# ``PLACE_RELEASE_HEIGHT_M`` above the surface (since
# object_in_flange = [0, 0, GRIPPER_DEPTH_M] places the centroid at
# finger-tip depth under our top-down-grasp assumption).
#
# We try the heights in PLACE_RELEASE_HEIGHTS_M in order; the highest
# value that's IK-feasible wins.  Going from high to low gives tall
# objects clearance where the workspace allows, and gracefully falls
# back at workspace-edge placements (where the higher pre-place pose is
# past Franka's reach).  Locally verified 2026-05-11: with a fixed
# 0.10 the white bowl at X≈0.83 was IK-unreachable; falling back to
# 0.07 or 0.05 puts the pre-place back in range.
PLACE_RELEASE_HEIGHTS_M = (0.10, 0.07, 0.05)
PLACE_RELEASE_HEIGHT_M = PLACE_RELEASE_HEIGHTS_M[0]   # primary / default
PLACE_RELEASE_HEIGHT_MIN_M = PLACE_RELEASE_HEIGHTS_M[-1]  # last fallback

# Trajectory interpolation step counts (joint waypoints per segment).
INTERP_STEPS_LIFT = 24          # ~3 chunks of 8 — same as grasp lift
INTERP_STEPS_APPROACH = 40      # current → pre-place
INTERP_STEPS_FINAL = 16         # pre-place → place
INTERP_STEPS_RETREAT = 24       # release pose → retreat

# Hold timings (sim steps, gated via state.episode_step)
CLOSE_HOLD_STEPS = 10           # ENSURE_CLOSED hold when starting open
RELEASE_HOLD_STEPS = 10         # RELEASING hold
SETTLE_STEPS = 20               # SETTLING wait

# Collision-aware place: cap on the scene obstacle cloud fed to cuRobo
# (mirrors grasp/tool.py::MAX_OBSTACLE_POINTS).  Only the APPROACH segment
# is collision-aware; the final descent + retreat stay free-space (the
# near-target cloud is stale / would collide with the very container we're
# reaching into).
MAX_OBSTACLE_POINTS = 4096
# Master switch for the collision-free (scene-aware) placement path.  When
# False (default), cuRobo plans joint→joint in an EMPTY world ("plain cuRobo")
# with NO scene cloud and NO attached held object.  Set GRASP_COLLISION_FREE=1
# to re-enable scene-aware avoidance (kept aligned with grasp/tool.py's flag).
COLLISION_FREE_PLANNING = os.environ.get("GRASP_COLLISION_FREE", "0") == "1"
# Number of spheres used to approximate the carried object (franka.yml caps
# the ``attached_object`` link at 4 sphere slots).
PLACE_ATTACH_NUM_SPHERES = 4

SAFE_HEIGHT_Z = 0.45
PERCEPTION_JOINTS = np.array(
    [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
)

# IK tolerances passed to inverse_kinematics_multistart for placement.
# The default sub-mm pos / 0.3° rot thresholds in grasp/ik.py are too
# tight for placement: the controller's tracking lag is ~5 cm anyway,
# so insisting on 0.5 mm IK accuracy just causes the multi-start ladder
# to exhaust all seeds and report failure on perfectly usable 2 mm / 1°
# solutions.  Relax to a few mm / a degree.
PLACE_IK_POS_TOL = 3e-3         # 3 mm position tolerance
PLACE_IK_ROT_TOL = 2e-2         # ~1.15° rotation tolerance

# ---------------------------------------------------------------------------
# Clean-model ROBOLAB IK (mirrors grasp/tool.py GRASP_ROBOLAB_CLEAN_IK)
# ---------------------------------------------------------------------------
# When ON (default), place IK targets the true Robotiq base_link frame via
# ``forward_kinematics_robolab`` + the fixed relabel
# ``C = inv(_T_FLANGE_HAND) @ T_JOINT7_TO_ROBOTIQ_BASE`` (passed as
# ``T_flange_to_tcp``).  This is the SAME exact-kinematics path grasp uses and
# it eliminates the legacy per-call ``fk_vs_ee`` panda-model-error patch (a
# crude ~18 mm rotation-dependent translation fudge that only approximated the
# real 120°-geodesic mount offset).
#
# Set PLACE_ROBOLAB_CLEAN_IK=0 to fall back to the legacy panda-hand +
# fk_vs_ee path (rollback / A-B comparison).  Place is robolab-only (no LIBERO
# branch), so no env-mode gating is needed — only the flag.
PLACE_ROBOLAB_CLEAN_IK = (
    os.environ.get("PLACE_ROBOLAB_CLEAN_IK", "1") not in ("0", "false", "False")
)


def _topdown_rotation_with_yaw(R_current: np.ndarray) -> np.ndarray:
    """Build a top-down panda_hand rotation that preserves the current
    EE's yaw about world Z.

    Returns a 3×3 rotation R_target such that:
      - panda_hand Z (col 2) = world −Z (gripper points straight down)
      - panda_hand X (col 0) projects onto the world XY plane along the
        same direction as the current EE's X-axis (preserves yaw — no
        wrist-spin)
      - panda_hand Y = Z × X

    If the current EE's X-axis is nearly vertical (gripper flat), we
    fall back to world-X for the new X to avoid an undefined yaw.
    """
    rx = R_current[:, 0].astype(np.float64).copy()
    rx[2] = 0.0
    n = float(np.linalg.norm(rx))
    if n < 1e-3:
        rx = np.array([1.0, 0.0, 0.0])
    else:
        rx = rx / n
    rz = np.array([0.0, 0.0, -1.0])
    ry = np.cross(rz, rx)
    return np.column_stack([rx, ry, rz])


# ---------------------------------------------------------------------------
# Phase + seg-mode enums
# ---------------------------------------------------------------------------

class PlacePhase(str, Enum):
    IDLE = "idle"
    ENSURE_CLOSED = "ensure_closed"
    LIFTING = "lifting"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    APPROACHING = "approaching"
    FINAL_APPROACH = "final_approach"
    MEASURING = "measuring"
    RELEASING = "releasing"
    SETTLING = "settling"
    RETREATING = "retreating"
    DONE = "done"
    FAILED = "failed"


class PlaceSegMode(str, Enum):
    """Destination-grounding backend (mirrors --place-seg-mode)."""
    GT_SIM = "gt_sim"
    SAM3 = "sam3"
    GDINO_SAM2 = "gdino_sam2"
    VLM_POINT = "vlm_point"
    MOLMO_POINT = "molmo_point"


# ---------------------------------------------------------------------------
# DestinationSpec
# ---------------------------------------------------------------------------

# Spatial-preposition first-words found in freeform destination phrases.
# Used by DestinationSpec.describe() to avoid double-prepending the
# ``relation`` field onto a target_object that already begins with one
# of these (the VLM emits phrases like "in the white bowl",
# "empty space next to the orange", "on the wire rack").
_DESTINATION_PREPOSITIONS = frozenset({
    "in", "on", "above", "below", "near", "next", "behind",
    "between", "atop", "beside", "empty", "around", "over", "under",
    "left", "right", "inside",
})


@dataclass
class DestinationSpec:
    """Where to place — exactly one of the three fields must be set."""
    target_object: str | None = None
    target_point_2d: tuple[float, float] | None = None
    target_point_3d_world: tuple[float, float, float] | None = None

    relation: Literal["in", "on", "on_top_of"] = "in"
    approach_axis_world: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def describe(self) -> str:
        """Short human-readable label (used for logs / HITL display).

        For ``target_object``: when the phrase already begins with a
        spatial preposition (typical for freeform ``place_destination``
        from the VLM, e.g. ``"in the white bowl"`` or ``"empty space
        next to the orange"``), do not double-prepend ``self.relation``
        — that produces ``"in in the white bowl"``.  Bare nouns (e.g.
        ``"the red bowl"``) still get the relation prepended.
        """
        if self.target_object:
            first = self.target_object.split(None, 1)[0].lower() \
                if self.target_object.strip() else ""
            if first in _DESTINATION_PREPOSITIONS:
                return self.target_object
            return f"{self.relation} {self.target_object}"
        if self.target_point_3d_world is not None:
            return f"{self.relation} {self.target_point_3d_world}"
        if self.target_point_2d is not None:
            return f"{self.relation} pixel{self.target_point_2d}"
        return "<unspecified>"

    def validate(self) -> None:
        n = sum(
            1 for v in (
                self.target_object,
                self.target_point_2d,
                self.target_point_3d_world,
            )
            if v is not None
        )
        if n != 1:
            raise ValueError(
                f"DestinationSpec: exactly one of "
                f"target_object / target_point_2d / target_point_3d_world "
                f"must be set (got {n})."
            )


# ---------------------------------------------------------------------------
# PlaceToolExecutor
# ---------------------------------------------------------------------------

class PlaceToolExecutor:
    """Drives a planned placement, producing VLA-compatible action chunks."""

    def __init__(
        self,
        place_server_url: str | None = None,
        intrinsics: CameraIntrinsics | None = None,
        seg_mode: str | PlaceSegMode = PlaceSegMode.SAM3,
        use_front_camera: bool = False,
        vlm=None,
        motion_planner: "MotionPlanner | None" = None,
        stack_mode_enabled: bool = False,
    ):
        self._client = PlaceClient(url=place_server_url)
        self._seg_mode = PlaceSegMode(seg_mode)
        self._use_front_camera = use_front_camera
        # Master switch (CLI --enable-stack-mode). When False, the per-call
        # ``stack`` arg is ignored and placement uses the historical
        # grasp-consistent rotation candidates (current_ee first, topdown
        # fallback) — zero behaviour change. When True, ``stack`` is honoured.
        self._stack_mode_enabled = bool(stack_mode_enabled)
        # Per-call advisory arg, set in start(). Default True == preserve the
        # held object's grasp orientation (stacking); False == force top-down.
        self._stack: bool = True
        self._intrinsics = intrinsics or overshoulder_left_intrinsics()
        self._vlm = vlm
        # Motion planner for joint-space trajectory segments (default:
        # straight linear interpolation — historical behaviour).
        if motion_planner is None:
            from vlm_orchestrator.motion import LinearInterpPlanner
            motion_planner = LinearInterpPlanner()
        self._motion_planner = motion_planner

        # Per-placement state
        self._phase: PlacePhase = PlacePhase.IDLE
        self._destination: DestinationSpec | None = None
        self._held_object_hint: str | None = None
        self._status_message: str = ""
        self._failure_reason: str = ""
        self._instruction: str = ""

        # Resolved destination (filled during PERCEIVING)
        self._resolved_target_world: np.ndarray | None = None
        self._approach_axis_world: np.ndarray | None = None
        self._point_2d: PointToPlace2D | None = None
        self._intrinsics_used: CameraIntrinsics | None = None
        self._cam_to_world_used: np.ndarray | None = None

        # Object centroid expressed in the **flange** frame.  Set in
        # start() to ``[0, 0, GRIPPER_DEPTH_M]`` (GraspGen finger-tip
        # convention).  Structural assumption: the grasp tool's
        # ``--grasp-topdown-threshold`` keeps the gripper near-vertical
        # at pickup, so the held object sits straight below the flange.
        self._object_in_flange: np.ndarray | None = None

        # Trajectory (joint-space, robolab)
        self._trajectory: list[np.ndarray] = []
        self._traj_cursor: int = 0
        self._seg_approach: list[np.ndarray] = []
        self._seg_final: list[np.ndarray] = []
        self._q_at_release: np.ndarray | None = None

        # Hold timing
        self._step_at_close_start: int = -CLOSE_HOLD_STEPS
        self._step_at_release_start: int = -RELEASE_HOLD_STEPS
        self._settle_remaining: int = 0

        # Diagnostic accumulator
        self._place_log: dict = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> PlacePhase:
        return self._phase

    @property
    def is_active(self) -> bool:
        return self._phase not in (
            PlacePhase.IDLE, PlacePhase.DONE, PlacePhase.FAILED,
        )

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    # ------------------------------------------------------------------
    # Reset (for reuse across activations)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._phase = PlacePhase.IDLE
        self._destination = None
        self._held_object_hint = None
        self._status_message = ""
        self._failure_reason = ""
        self._resolved_target_world = None
        self._approach_axis_world = None
        self._point_2d = None
        self._intrinsics_used = None
        self._cam_to_world_used = None
        self._object_in_flange = None
        self._trajectory = []
        self._traj_cursor = 0
        self._seg_approach = []
        self._seg_final = []
        self._q_at_release = None
        self._step_at_close_start = -CLOSE_HOLD_STEPS
        self._step_at_release_start = -RELEASE_HOLD_STEPS
        self._settle_remaining = 0
        self._place_log = {}

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(
        self,
        destination: DestinationSpec,
        obs: dict,
        state,
        *,
        held_object_hint: str | None = None,
        instruction: str = "",
        stack: bool = False,
    ) -> None:
        """Kick off a planned placement.

        ``held_object_hint`` is advisory — used for the destination
        raycast height calculation and for logs.  Never validated
        against the actual gripper contents.

        ``stack`` is advisory and only honoured when the executor was
        constructed with ``stack_mode_enabled=True`` (CLI
        ``--enable-stack-mode``).  ``stack=False`` (default) forces a
        purely top-down release rotation, which makes the vertical descent
        axis consistent with the release orientation — the right choice for
        ordinary pick-and-place (into a bowl/bin, onto an open surface).
        ``stack=True`` preserves the held object's grasp orientation at
        release (current_ee rotation first, top-down fallback) — needed
        when setting an object ON TOP of another (stacking, nesting a lid).
        When the master switch is OFF, ``stack`` is ignored and the
        historical grasp-consistent behaviour (== ``stack=True``) is used
        regardless.

        Transitions from IDLE → ENSURE_CLOSED.  No motion is produced
        by ``start()``; the next ``step()`` call serves the first chunk.
        """
        # Honour the advisory stack arg only when the master switch is on.
        # OFF → historical grasp-consistent placement (stack semantics == True).
        self._stack = bool(stack) if self._stack_mode_enabled else True
        destination.validate()
        self._destination = destination
        self._held_object_hint = held_object_hint
        self._instruction = instruction
        self._status_message = (
            f"Place: ensure-closed before {destination.describe()}"
        )
        self._failure_reason = ""

        # Held-object offset in the flange frame.
        #
        # By design, the grasp tool's ``--grasp-topdown-threshold`` (0.85
        # by default) only accepts near-vertical grasps, so by the time
        # the place tool runs the gripper is approximately pointing
        # straight down and the held object sits directly below the
        # flange at the finger-tip depth.  We therefore use the GraspGen
        # convention ``[0, 0, GRIPPER_DEPTH_M]`` as a structural
        # assumption — not a fallback — and ``held_object_hint`` is
        # log-only (it surfaces in debug output and post-hoc analysis).
        #
        # If the top-down filter is relaxed or removed, this assumption
        # breaks and placement accuracy suffers along whatever axis the
        # grasp tilted.  Don't reach for ``gt_state`` to "fix" that —
        # privileged sim info doesn't generalise to real robots; revisit
        # the grasp filter or plumb the offset from grasp-time perception.
        # ROBOLAB mounts a Robotiq 2F-85 (fingertip ROBOTIQ_GRIPPER_DEPTH_M
        # below the flange), NOT the Panda hand GraspGen assumes.  The place IK
        # runs in the true Robotiq base_link frame (PLACE_ROBOLAB_CLEAN_IK), so
        # the held-object depth must be the Robotiq value or the object descends
        # ~0.0277 m too low (GRIPPER_HIT_TABLE).  LIBERO/panda path keeps 0.1034.
        # Env override PLACE_GRIPPER_DEPTH_M for audit sweeps.
        from vlm_orchestrator.place.destination import (
            GRIPPER_DEPTH_M as _GD_PANDA,
            ROBOTIQ_GRIPPER_DEPTH_M as _GD_ROBOTIQ,
        )
        _gd = _GD_ROBOTIQ if PLACE_ROBOLAB_CLEAN_IK else _GD_PANDA
        _gd_env = os.environ.get("PLACE_GRIPPER_DEPTH_M")
        if _gd_env is not None:
            _gd = float(_gd_env)
        self._object_in_flange = np.array([0.0, 0.0, _gd])
        logger.info(
            "  place held-object depth: %.4f m (%s)",
            _gd,
            "robotiq/robolab" if PLACE_ROBOLAB_CLEAN_IK else "panda/libero",
        )

        # Wire debug stash to this state
        from vlm_orchestrator.place import debug as _dbg
        _dbg.set_state(state)

        self._phase = PlacePhase.ENSURE_CLOSED
        self._step_at_close_start = state.episode_step
        logger.info(
            f"place_with_tool: start → ENSURE_CLOSED  "
            f"destination={destination.describe()!r}  "
            f"held_hint={held_object_hint!r}  "
            f"seg_mode={self._seg_mode.value}"
        )

        # Seed log
        self._place_log = {
            "destination": destination.describe(),
            "held_object_hint": held_object_hint,
            "seg_mode": self._seg_mode.value,
            "instruction": instruction,
            "started_at": time.time(),
            "stack_mode_enabled": self._stack_mode_enabled,
            "stack": self._stack,
        }

    # ------------------------------------------------------------------
    # Step (called once per proxy loop iteration)
    # ------------------------------------------------------------------

    def step(self, obs: dict, state) -> dict:
        """Produce the next action chunk."""
        try:
            if self._phase == PlacePhase.ENSURE_CLOSED:
                return self._step_ensure_closed(obs, state)
            if self._phase == PlacePhase.LIFTING:
                return self._step_lifting(obs, state)
            if self._phase == PlacePhase.PERCEIVING:
                return self._step_perceiving(obs, state)
            if self._phase == PlacePhase.PLANNING:
                # Planning is synchronous inside PERCEIVING currently;
                # if we ever split it, this branch is the dispatch.
                return self._noop_response(obs)
            if self._phase in (
                PlacePhase.APPROACHING,
                PlacePhase.FINAL_APPROACH,
                PlacePhase.RETREATING,
            ):
                return self._step_trajectory(obs, state)
            if self._phase == PlacePhase.MEASURING:
                return self._step_measuring(obs, state)
            if self._phase == PlacePhase.RELEASING:
                return self._step_releasing(obs, state)
            if self._phase == PlacePhase.SETTLING:
                return self._step_settling(obs, state)
            if self._phase in (PlacePhase.DONE, PlacePhase.FAILED):
                return self._noop_response(obs)
        except PerceptionFailure as e:
            self._fail("perception_no_target", str(e))
            return self._noop_response(obs)
        except Exception as e:
            logger.error(f"place tool step() crashed: {e}", exc_info=True)
            self._fail("ik_unreachable" if "IK" in str(e) else "motion_blocked", str(e))
            return self._noop_response(obs)

        logger.warning(f"place tool step() in unexpected phase: {self._phase}")
        return self._noop_response(obs)

    # ------------------------------------------------------------------
    # Phase steppers
    # ------------------------------------------------------------------

    def _step_ensure_closed(self, obs: dict, state) -> dict:
        """If gripper open, command close for N chunks; otherwise advance.

        Either way we always advance once ``CLOSE_HOLD_STEPS`` have passed
        (or immediately when the gripper is already closed).
        """
        gripper_pos = float(self._extract_gripper(obs))
        already_closed = gripper_pos < PLACE_GRIPPER_CLOSED_THRESHOLD

        # Build hold-position + close-gripper action
        joints = self._extract_joints(obs)
        action = np.concatenate([joints, [GRIPPER_CLOSE]])
        actions = np.tile(action, (ACTION_HORIZON, 1))

        if already_closed:
            logger.info(
                f"  ENSURE_CLOSED: gripper already closed "
                f"(pos={gripper_pos:.3f}) → advance to PERCEIVING"
            )
            self._advance_to_perceiving(obs)
            return {"actions": actions}

        if (state.episode_step - self._step_at_close_start
                >= CLOSE_HOLD_STEPS):
            logger.info(
                f"  ENSURE_CLOSED: close-hold elapsed "
                f"(pos={gripper_pos:.3f}) → advance to PERCEIVING"
            )
            self._advance_to_perceiving(obs)
        else:
            self._status_message = (
                f"Closing gripper (pos={gripper_pos:.3f})"
            )

        return {"actions": actions}

    def _advance_to_perceiving(self, obs: dict) -> None:
        """Skip to PERCEIVING from the current EE pose.

        We deliberately do **not** lift to a canonical ``PERCEPTION_JOINTS``
        config like the grasp tool does.  That lift resets the gripper
        orientation to whatever FK at PERCEPTION_JOINTS produces (a 45°
        tilt for our Franka), destroying the near-vertical orientation
        the strict-grasp filter (``--grasp-topdown-threshold``) gave us.

        We don't lift to a dedicated perception pose — the destination
        is perceived from wherever the EE happens to be after the grasp:
        the over-shoulder camera usually has a clean
        view of the destination from any post-grasp pose, and our
        downstream IK / multi-start handles cramped joint configs.

        If the destination is genuinely occluded by the arm in some
        future workload, the place tool fails loudly with
        ``failure_reason='perception_no_target'`` rather than silently
        producing a bad target.
        """
        from vlm_orchestrator.grasp.ik import forward_kinematics

        current_joints = self._extract_joints(obs)
        T_ee = forward_kinematics(current_joints)
        current_z = float(T_ee[2, 3])
        logger.info(
            f"  PERCEIVING from current pose (z={current_z:.3f}, "
            f"no canonical-joints prelift)"
        )
        self._trajectory = []
        self._phase = PlacePhase.PERCEIVING
        self._status_message = "Place: detecting destination..."

    def _step_lifting(self, obs: dict, state) -> dict:
        """Serve the next lift chunk."""
        return self._step_trajectory(obs, state)

    def _step_perceiving(self, obs: dict, state) -> dict:
        """Resolve the destination → 3D world point → plan trajectory."""
        from vlm_orchestrator.place import debug as _dbg

        dest = self._destination
        assert dest is not None

        # Pre-resolved 3D path: route through raycast_2d_to_3d so the
        # clearance Z-lift is applied consistently with the 2D paths.
        # raycast returns the OBJECT centroid target — translation to a
        # flange target happens in _plan_trajectory.
        if dest.target_point_3d_world is not None:
            stub_pt = PointToPlace2D(
                x_norm=0.5, y_norm=0.5, confidence=1.0,
                source="explicit_3d",
                rationale=(
                    f"target_point_3d_world={list(dest.target_point_3d_world)}"
                ),
            )
            resolved = raycast_2d_to_3d(
                stub_pt,
                depth=np.zeros((1, 1)),  # unused on pre_resolved_3d path
                intrinsics=self._intrinsics,
                cam_to_world=np.eye(4),
                relation=dest.relation,
                held_object_height_m=None,
                clearance_m=PLACE_RELEASE_HEIGHT_M,
                approach_axis_world=dest.approach_axis_world,
                pre_resolved_3d=np.asarray(
                    dest.target_point_3d_world, dtype=np.float64,
                ),
            )
            self._resolved_target_world = resolved.target_world
            self._approach_axis_world = resolved.approach_axis_world
            self._point_2d = stub_pt
            self._place_log["object_target_world"] = (
                resolved.target_world.tolist()
            )
            self._place_log["point_2d_source"] = "explicit_3d"
            self._place_log["object_in_flange"] = self._object_in_flange.tolist()
            return self._after_perception_plan(obs, state)

        # Stage 1: language → 2D pixel
        image = self._extract_image(obs)
        intrinsics = self._intrinsics_for_obs(obs)
        cam_to_world = self._get_camera_to_world(obs)
        self._intrinsics_used = intrinsics
        self._cam_to_world_used = cam_to_world

        if dest.target_point_2d is not None:
            pt = PointToPlace2D(
                x_norm=float(dest.target_point_2d[0]),
                y_norm=float(dest.target_point_2d[1]),
                confidence=1.0,
                source="hitl_click",
                rationale="caller-provided 2D pixel",
            )
        else:
            target_phrase = dest.target_object or ""
            pt = point_to_place_2d(
                image_rgb=image,
                instruction=self._instruction,
                target_phrase=target_phrase,
                seg_mode=self._seg_mode.value,
                grasp_client=self._client,
                vlm=self._vlm,
                gt_state=obs.get("gt_state"),
                intrinsics=intrinsics,
                cam_to_world=cam_to_world,
            )

        self._point_2d = pt
        self._place_log["point_2d_source"] = pt.source
        self._place_log["point_2d"] = [pt.x_norm, pt.y_norm]
        if pt.rationale:
            self._place_log["point_2d_rationale"] = pt.rationale

        # Debug overlay for the chosen 2D pixel
        hitl = getattr(state, "_hitl", None) or getattr(state, "hitl", None)
        _dbg.vis_destination_2d(
            image, (pt.x_norm, pt.y_norm),
            source=pt.source,
            target_phrase=dest.target_object or "<2D-click>",
            confidence=pt.confidence,
            rationale=pt.rationale,
            hitl=hitl,
        )

        # Stage 2: 2D pixel → 3D world point
        depth = self._extract_depth(obs)
        if depth is None:
            raise PerceptionFailure(
                "No depth in obs — cannot raycast.  Ensure --enable-depth on "
                "the eval client."
            )
        resolved = raycast_2d_to_3d(
            pt, depth, intrinsics, cam_to_world,
            relation=dest.relation,
            held_object_height_m=None,
            clearance_m=PLACE_RELEASE_HEIGHT_M,
            approach_axis_world=dest.approach_axis_world,
        )
        self._resolved_target_world = resolved.target_world
        self._approach_axis_world = resolved.approach_axis_world
        self._place_log["object_target_world"] = resolved.target_world.tolist()
        self._place_log["release_height_m"] = PLACE_RELEASE_HEIGHT_M
        self._place_log["relation"] = resolved.relation
        self._place_log["object_in_flange"] = self._object_in_flange.tolist()

        _dbg.vis_target_world(
            image, resolved.target_world, cam_to_world, intrinsics,
            relation=resolved.relation,
            held_object_height_m=None,
            hitl=hitl,
        )

        return self._after_perception_plan(obs, state)

    def _after_perception_plan(self, obs: dict, state) -> dict:
        """Plan APPROACHING + FINAL_APPROACH segments and switch phase."""
        self._phase = PlacePhase.PLANNING
        self._status_message = "Place: planning trajectory..."
        self._plan_trajectory(obs, state)
        # Load segment 1
        self._trajectory = self._seg_approach
        self._traj_cursor = 0
        self._phase = PlacePhase.APPROACHING
        self._status_message = (
            f"Approaching pre-place "
            f"({len(self._seg_approach)} approach + "
            f"{len(self._seg_final)} final waypoints)"
        )
        # First chunk
        return self._step_trajectory(obs, state)

    def _step_trajectory(self, obs: dict, state) -> dict:
        """Serve the next ACTION_HORIZON waypoints."""
        chunk: list[np.ndarray] = []
        for _ in range(ACTION_HORIZON):
            if self._traj_cursor < len(self._trajectory):
                chunk.append(self._trajectory[self._traj_cursor])
                self._traj_cursor += 1
            else:
                # Hold last waypoint
                chunk.append(self._trajectory[-1] if self._trajectory else self._hold_action(obs))
        actions = np.array(chunk, dtype=np.float64)
        if self._traj_cursor >= len(self._trajectory):
            self._advance_phase_after_trajectory(obs, state)
        return {"actions": actions}

    def _advance_phase_after_trajectory(self, obs: dict, state) -> None:
        if self._phase == PlacePhase.LIFTING:
            logger.info("  Lift complete → PERCEIVING")
            self._phase = PlacePhase.PERCEIVING
            self._status_message = "Place: detecting destination..."
            return
        if self._phase == PlacePhase.APPROACHING:
            logger.info("  Approach complete → FINAL_APPROACH")
            self._phase = PlacePhase.FINAL_APPROACH
            self._status_message = "Final approach to release pose..."
            self._trajectory = self._seg_final
            self._traj_cursor = 0
            return
        if self._phase == PlacePhase.FINAL_APPROACH:
            logger.info("  Final approach complete → MEASURING")
            self._phase = PlacePhase.MEASURING
            self._status_message = "Measuring post-move pose..."
            return
        if self._phase == PlacePhase.RETREATING:
            logger.info("  Retreat complete → DONE")
            self._phase = PlacePhase.DONE
            self._status_message = "Place complete — handing back to VLA"
            self._save_place_log()
            return

    def _step_measuring(self, obs: dict, state) -> dict:
        """Log post-move EE-vs-target delta, then transition to RELEASING."""
        self._do_post_move_log(obs, state)
        self._phase = PlacePhase.RELEASING
        self._step_at_release_start = state.episode_step
        self._status_message = "Releasing held object..."
        # Last commanded waypoint, but with gripper OPEN
        last = (
            self._seg_final[-1] if self._seg_final
            else (self._trajectory[-1] if self._trajectory else self._hold_action(obs))
        )
        last_open = last.copy()
        last_open[-1] = GRIPPER_OPEN
        return {"actions": np.tile(last_open, (ACTION_HORIZON, 1))}

    def _step_releasing(self, obs: dict, state) -> dict:
        """Hold position with gripper open."""
        joints = self._extract_joints(obs)
        action = np.concatenate([joints, [GRIPPER_OPEN]])
        actions = np.tile(action, (ACTION_HORIZON, 1))
        if (state.episode_step - self._step_at_release_start
                >= RELEASE_HOLD_STEPS):
            logger.info("  Release complete → SETTLING")
            self._phase = PlacePhase.SETTLING
            self._settle_remaining = SETTLE_STEPS
            self._status_message = "Settling (waiting for object to land)..."
        return {"actions": actions}

    def _step_settling(self, obs: dict, state) -> dict:
        """Hold open while the object lands."""
        joints = self._extract_joints(obs)
        action = np.concatenate([joints, [GRIPPER_OPEN]])
        actions = np.tile(action, (ACTION_HORIZON, 1))
        # Settle is sim-step-counted: decrement once per chunk's worth of steps.
        self._settle_remaining -= ACTION_HORIZON
        if self._settle_remaining <= 0:
            logger.info("  Settle complete → planning retreat")
            self._plan_retreat(obs)
            self._traj_cursor = 0
            self._phase = PlacePhase.RETREATING
            self._status_message = "Retreating..."
        return {"actions": actions}

    # ------------------------------------------------------------------
    # Trajectory planning
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

        Records the planner label under ``motion_planner_<phase>`` in the
        place log.  Raises ``RuntimeError`` on planner failure (no silent
        fallback to linear).
        """
        result = self._motion_planner.plan_segment(
            q_start, q_end, n_steps=n_steps, scene_pc=scene_pc, phase=phase,
        )
        self._place_log[f"motion_planner_{phase}"] = result.label
        if not result.success or result.waypoints is None:
            raise RuntimeError(
                f"Motion planning failed for {phase} segment "
                f"(planner={result.label}): {result.detail}"
            )
        return result.waypoints

    def _build_scene_obstacle_cloud(self, obs: dict) -> np.ndarray | None:
        """Full-scene obstacle cloud (base frame) for collision-aware approach.

        Unlike grasp — which subtracts the *target* object — place keeps the
        whole scene: the destination container and other objects are all real
        obstacles the arm must avoid on the way in.  The CARRIED object is NOT
        in this cloud (it's a fresh depth capture of the scene in front of the
        camera); it's handled separately via cuRobo attached-object spheres.

        Returns ``None`` for a collision-unaware planner (linear) or when depth
        is unavailable, so free-space planning pays zero cost.
        """
        if not COLLISION_FREE_PLANNING:
            return None
        if getattr(self._motion_planner, "name", "linear") == "linear":
            return None
        depth = self._extract_depth(obs)
        if depth is None:
            return None
        try:
            from vlm_orchestrator.grasp.camera import depth_to_pointcloud
            intrinsics = self._intrinsics_used or self._intrinsics
            cam_to_world = self._cam_to_world_used
            if cam_to_world is None:
                cam_to_world = self._get_camera_to_world(obs)
            if cam_to_world is None:
                return None
            pc_cam = depth_to_pointcloud(depth, intrinsics)
            if pc_cam.shape[0] == 0:
                return None
            # camera → world (base == world in the clean ROBOLAB path)
            pc_h = np.hstack([pc_cam, np.ones((pc_cam.shape[0], 1), np.float32)])
            pc_world = (cam_to_world @ pc_h.T).T[:, :3]
            if pc_world.shape[0] > MAX_OBSTACLE_POINTS:
                idx = np.random.choice(
                    pc_world.shape[0], MAX_OBSTACLE_POINTS, replace=False,
                )
                pc_world = pc_world[idx]
            return pc_world.astype(np.float32)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"place scene-cloud build failed: {e}")
            return None

    def _attach_held_object(self, state, current_joints: np.ndarray) -> bool:
        """Attach the carried object to the gripper on the cuRobo planner.

        Uses the grasp-time target cloud + FK EE pose stashed on SessionState
        by the grasp executor.  The object was captured on the table (grasp EE
        pose); it is now held rigidly, so we transform it grasp-EE → current-EE
        to place it correctly in the gripper at the current config.

        Returns True if an attach happened (planner supports it + geometry was
        available).  Logs the taken path in place_log.  Never raises.
        """
        planner = self._motion_planner
        if not hasattr(planner, "attach_object"):
            return False
        obj_pc_world = getattr(state, "last_grasped_object_pc_world", None)
        grasp_ee = getattr(state, "last_grasped_ee_pose", None)
        if obj_pc_world is None or grasp_ee is None:
            self._place_log["place_attach_path"] = "none_no_grasp_cloud"
            return False
        try:
            from vlm_orchestrator.grasp.ik import forward_kinematics
            T_now = forward_kinematics(current_joints)      # current EE (base)
            # rigid held-object transform: grasp-EE frame → current-EE frame
            T_rel = T_now @ np.linalg.inv(grasp_ee)
            pc = np.asarray(obj_pc_world, dtype=np.float32)
            pc_h = np.hstack([pc, np.ones((pc.shape[0], 1), np.float32)])
            pc_held = (T_rel @ pc_h.T).T[:, :3].astype(np.float32)
            ok = planner.attach_object(
                pc_held, current_joints[:7],
                num_spheres=PLACE_ATTACH_NUM_SPHERES,
            )
            self._place_log["place_attach_path"] = (
                "attached_grasp_cloud" if ok else "attach_failed"
            )
            self._place_log["place_attach_pts"] = int(len(pc_held))
            if ok:
                logger.info(
                    f"  Place: attached held object ({len(pc_held)} pts) "
                    f"to gripper for collision-aware approach"
                )
            else:
                logger.warning(
                    "  Place: attach_object returned False — approach will "
                    "plan WITHOUT the carried object attached"
                )
            return ok
        except Exception as e:  # noqa: BLE001
            logger.warning(f"place attach_held_object failed: {e}")
            self._place_log["place_attach_path"] = "attach_error"
            return False

    def _plan_trajectory(self, obs: dict, state=None) -> None:
        """Plan APPROACHING + FINAL_APPROACH segments to the release pose.

        Design notes:

        1. **Try a small list of placement rotations.**  IK multi-start
           only varies *seeds*, so when the rotation in T_target is
           infeasible at a perfectly reachable XY (e.g. wrist drifted
           between grasp and place), every seed strategy fails
           identically — exactly the pathology observed in smoke-2
           (episode 2: 28 nearly-identical IK rejections at the same
           target).  We try top-down-with-yaw first (matches our
           structural assumption + most reachable), then current EE
           rotation as a fallback.  The accepted label is logged for
           post-hoc debugging.
        2. **Translate the object-centre target into a flange target
           using the gripper-frame offset.**  ``object_in_flange`` is
           the structural assumption ``[0, 0, GRIPPER_DEPTH_M]`` set in
           ``start()`` (relies on the grasp tool's top-down filter
           keeping the gripper near-vertical at pickup).
        """
        from vlm_orchestrator.grasp.ik import (
            forward_kinematics,
            forward_kinematics_robolab,
            inverse_kinematics_multistart,
            _T_FLANGE_HAND,
            T_JOINT7_TO_ROBOTIQ_BASE,
        )

        current_joints = self._extract_joints(obs)
        object_target_world = self._resolved_target_world
        approach = self._approach_axis_world
        if object_target_world is None or approach is None:
            raise PerceptionFailure(
                "Trajectory planning called before perception resolved"
            )

        # ── Clean-model ROBOLAB IK path (mirrors grasp/tool.py) ──────────────
        # ``clean_ik`` targets the true Robotiq base_link frame directly.
        #   _ik_tcp = C = inv(_T_FLANGE_HAND) @ T_JOINT7_TO_ROBOTIQ_BASE is the
        #   fixed relabel from the panda_hand frame (what forward_kinematics/IK
        #   natively use) to base_link.  Passing it as T_flange_to_tcp makes the
        #   solver operate in the true controlled frame.  _fk_ee is the FK that
        #   outputs that same frame.
        # All place geometry below (object_in_flange, R_target, flange target)
        # stays in the panda_hand/GraspGen convention; we relabel the FINAL 4×4
        # flange target to base_link with @C right before calling IK.
        clean_ik = PLACE_ROBOLAB_CLEAN_IK
        _ik_tcp = None
        _fk_ee = forward_kinematics
        _C = None
        if clean_ik:
            _C = np.linalg.inv(_T_FLANGE_HAND) @ T_JOINT7_TO_ROBOTIQ_BASE
            _ik_tcp = _C
            _fk_ee = forward_kinematics_robolab
        self._place_log["ik_clean_robolab"] = bool(clean_ik)

        # ALL placement geometry (object_in_flange, R_current/R_target, the
        # flange target) is built in the panda_hand / GraspGen convention where
        # +Z is the approach axis, so T_fk_now MUST stay panda_hand FK.  Only
        # the final 4×4 IK target is relabelled to base_link (via @_C) right at
        # the solver boundary — see the ``_relabel`` closure below.
        ee_pos_obs = self._extract_ee_pos(obs)
        T_fk_now = forward_kinematics(current_joints)

        # World ↔ base offset.  Clean path: base == world (proven by the mount
        # calibration; the panda-model error is absorbed exactly by _C at the
        # IK boundary), so no shift.  Legacy path: fk_vs_ee is the crude
        # panda-model-error patch that also carries world→base.
        fk_vs_ee = (
            np.zeros(3) if clean_ik else (T_fk_now[:3, 3] - ee_pos_obs)
        )
        self._place_log["perception_fk_vs_ee_delta"] = (
            (T_fk_now[:3, 3] - ee_pos_obs).tolist()
        )

        def _relabel(T_hand: np.ndarray) -> np.ndarray:
            """Map a panda_hand-frame 4×4 target to the IK solver's frame.

            Clean path: base_link target = T_hand @ _C, solved against
            ``forward_kinematics_robolab`` (== forward_kinematics @ _C) with
            ``T_flange_to_tcp=_C``, so IK recovers the SAME q that would place
            the panda_hand flange at T_hand — minus the legacy fk_vs_ee patch.
            Legacy path: identity (solver is native panda_hand FK).
            """
            return T_hand @ _C if clean_ik else T_hand

        # Object centre in flange frame.  Always set in start() to the
        # GraspGen finger-tip convention; see start() for the
        # top-down-grasp assumption it relies on.
        object_in_flange = self._object_in_flange

        # Candidate placement orientations, in priority order.  IK
        # multi-start only varies *seeds* — the rotation in T_target is
        # fixed per call — so a single fixed R_target can be infeasible
        # at a perfectly reachable XY and yield identical failures across
        # every seed strategy.  Try a small list of rotation candidates
        # so we recover from those infeasible-rotation cases.
        #
        # 1. Current EE rotation — reuse the grasp quaternion at place
        #    time, preserving the held object's orientation.  Works for
        #    the common case where
        #    the top-down grasp filter kept the gripper near-vertical
        #    and the position is well inside the workspace.
        # 2. Top-down with the current EE's yaw — fallback for the rare
        #    wrist-yaw-branch case where the current rotation is
        #    infeasible at the target XY (Franka wrist limits at large
        #    X or low Z).
        R_current = T_fk_now[:3, :3]
        R_topdown = _topdown_rotation_with_yaw(R_current)
        if self._stack:
            # Stacking / grasp-consistent placement (historical default):
            # preserve the held object's orientation, top-down as fallback.
            rot_candidates = [
                (R_current, "current_ee"),
                (R_topdown, "topdown_yaw"),
            ]
        else:
            # Simple pick-and-place: force a top-down release. This makes
            # the release rotation consistent with the vertical (0,0,-1)
            # descent axis and keeps object_in_flange=[0,0,d] exactly valid.
            rot_candidates = [
                (R_topdown, "topdown_yaw"),
            ]

        last_err: str = ""
        chosen_label: str | None = None
        chosen_height: float = PLACE_RELEASE_HEIGHT_M
        q_pre = q_place = None
        strat_pre = strat_place = ""
        T_target_base = T_pre = None
        flange_target_pos_world = None
        adjusted_target_world = object_target_world

        # ``object_target_world`` was raycast with PLACE_RELEASE_HEIGHT_M
        # (the highest candidate).  For each lower candidate the target
        # just shifts down by ``(primary - candidate)`` along world +Z;
        # we don't re-raycast.
        primary_height = PLACE_RELEASE_HEIGHT_M
        for release_height in PLACE_RELEASE_HEIGHTS_M:
            dz = release_height - primary_height
            adjusted_target_world = object_target_world.copy()
            adjusted_target_world[2] += dz

            for R_target, rot_label in rot_candidates:
                # Flange position needed so the held object lands at the
                # target: object_world = flange_world + R_target @ object_in_flange
                # ⇒  flange_world = object_world − R_target @ object_in_flange
                ftpw = adjusted_target_world - R_target @ object_in_flange
                ftpb = ftpw + fk_vs_ee
                T_tb = np.eye(4)
                T_tb[:3, :3] = R_target
                T_tb[:3, 3] = ftpb
                # Pre-place: back off along the WORLD-frame approach axis
                # (top-down → +Z) so the EE descends straight down.
                T_p = T_tb.copy()
                T_p[:3, 3] = T_tb[:3, 3] - approach * PLACE_PRE_OFFSET

                q_pre_c, conv_pre, strat_pre_c = inverse_kinematics_multistart(
                    _relabel(T_p), current_joints,
                    q_canonical=PERCEPTION_JOINTS,
                    pos_tol=PLACE_IK_POS_TOL,
                    rot_tol=PLACE_IK_ROT_TOL,
                    T_flange_to_tcp=_ik_tcp,
                )
                if not conv_pre:
                    last_err = (
                        f"height={release_height:.3f} rot={rot_label}: "
                        f"pre-place IK failed (strategy={strat_pre_c}); "
                        f"pre_world={(T_p[:3, 3] - fk_vs_ee).tolist()}"
                    )
                    logger.info(
                        f"  Place IK try [h={release_height:.2f} "
                        f"rot={rot_label}] pre: FAIL"
                    )
                    continue

                q_place_c, conv_place, strat_place_c = (
                    inverse_kinematics_multistart(
                        _relabel(T_tb), q_pre_c,
                        q_canonical=None,
                        max_joint_dist_rad=1.0,
                        reference_q=q_pre_c,
                        pos_tol=PLACE_IK_POS_TOL,
                        rot_tol=PLACE_IK_ROT_TOL,
                        T_flange_to_tcp=_ik_tcp,
                    )
                )
                if not conv_place:
                    last_err = (
                        f"height={release_height:.3f} rot={rot_label}: "
                        f"place IK failed (strategy={strat_place_c}); "
                        f"pre-place was reachable but final pose is not"
                    )
                    logger.info(
                        f"  Place IK try [h={release_height:.2f} "
                        f"rot={rot_label}] pre OK, place FAIL"
                    )
                    continue

                # Both converged — accept this (height, rotation).
                chosen_label = rot_label
                chosen_height = release_height
                q_pre, q_place = q_pre_c, q_place_c
                strat_pre, strat_place = strat_pre_c, strat_place_c
                T_target_base, T_pre = T_tb, T_p
                flange_target_pos_world = ftpw
                logger.info(
                    f"  Place IK accepted at release_height="
                    f"{release_height:.2f} rotation={rot_label} "
                    f"(pre={strat_pre_c}, place={strat_place_c})"
                )
                break

            if chosen_label is not None:
                break

        if chosen_label is None:
            raise RuntimeError(
                "IK for place failed across all release-height × rotation "
                f"candidates ({list(PLACE_RELEASE_HEIGHTS_M)} × "
                f"{[lbl for _, lbl in rot_candidates]}); "
                f"last error: {last_err}"
            )

        # If we fell back to a lower height, update the resolved target
        # so downstream code (logs, retreat planning) sees the actual
        # chosen Z.
        self._resolved_target_world = adjusted_target_world
        self._place_log["release_height_m"] = chosen_height
        self._place_log["release_height_fallback"] = (
            chosen_height < primary_height
        )

        self._place_log["flange_target_world"] = (
            flange_target_pos_world.tolist()
        )
        self._place_log["R_target_world"] = T_target_base[:3, :3].tolist()
        self._place_log["R_target_label"] = chosen_label
        self._place_log["object_in_flange_used"] = object_in_flange.tolist()
        self._place_log["ik_pre_strategy"] = strat_pre
        self._place_log["ik_place_strategy"] = strat_place

        # ── Collision-aware approach (cuRobo only) ───────────────────────────
        # Build the scene obstacle cloud and attach the carried object to the
        # gripper so the approach avoids the scene without treating the held
        # object as a fixed world obstacle.  Both are no-ops on the linear
        # planner / when geometry is unavailable.  The FINAL descent + retreat
        # stay free-space (near-target cloud is stale / self-colliding).
        scene_pc = None
        attached = False
        if state is not None:
            scene_pc = self._build_scene_obstacle_cloud(obs)
            if scene_pc is not None:
                self._place_log["motion_obstacle_pts"] = int(len(scene_pc))
                attached = self._attach_held_object(state, current_joints)

        try:
            approach_joints = self._plan_segment(
                current_joints, q_pre, INTERP_STEPS_APPROACH, phase="approach",
                scene_pc=scene_pc,
            )
        finally:
            # Always detach so the attached spheres don't leak into the next
            # plan (final descent, retreat, or a later grasp/place).
            if attached and hasattr(self._motion_planner, "detach_object"):
                self._motion_planner.detach_object()
        self._seg_approach = [
            np.concatenate([q, [GRIPPER_CLOSE]]) for q in approach_joints
        ]

        final_joints = self._plan_segment(
            q_pre, q_place, INTERP_STEPS_FINAL, phase="final",
        )
        self._seg_final = [
            np.concatenate([q, [GRIPPER_CLOSE]]) for q in final_joints
        ]
        self._q_at_release = q_place.copy()

        T_fk_place = forward_kinematics(q_place)
        self._place_log["ik_pre_target_pos"] = T_pre[:3, 3].tolist()
        self._place_log["ik_place_target_pos"] = T_target_base[:3, 3].tolist()
        self._place_log["ik_place_fk_pos"] = T_fk_place[:3, 3].tolist()
        self._place_log["ik_place_pos_error"] = (
            T_fk_place[:3, 3] - T_target_base[:3, 3]
        ).tolist()
        logger.info(
            f"  Place trajectory: approach={len(self._seg_approach)} + "
            f"final={len(self._seg_final)} waypoints "
            f"(IK pre={strat_pre}, place={strat_place})"
        )

    def _plan_retreat(self, obs: dict) -> None:
        """Lift straight up after release."""
        from vlm_orchestrator.grasp.ik import (
            forward_kinematics,
            inverse_kinematics_multistart,
        )

        q_now = (
            self._q_at_release
            if self._q_at_release is not None
            else self._extract_joints(obs)
        )
        T_now = forward_kinematics(q_now)
        T_retreat = T_now.copy()
        T_retreat[2, 3] += PLACE_RETREAT_HEIGHT

        q_retreat, conv, strat = inverse_kinematics_multistart(
            T_retreat, q_now,
            q_canonical=None,
            max_joint_dist_rad=1.0,
            reference_q=q_now,
            pos_tol=PLACE_IK_POS_TOL,
            rot_tol=PLACE_IK_ROT_TOL,
        )
        if not conv:
            logger.warning(
                f"  IK for retreat did not converge ({strat}) — best-effort"
            )
        self._place_log["ik_retreat_strategy"] = strat

        retreat_joints = self._plan_segment(
            q_now, q_retreat, INTERP_STEPS_RETREAT, phase="retreat",
        )
        # Gripper stays open during retreat
        self._trajectory = [
            np.concatenate([q, [GRIPPER_OPEN]]) for q in retreat_joints
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _do_post_move_log(self, obs: dict, state) -> None:
        try:
            from vlm_orchestrator.place import debug as _dbg
            from vlm_orchestrator.grasp.ik import (
                forward_kinematics as _fk,
            )
            if (
                self._resolved_target_world is None
                or self._intrinsics_used is None
                or self._cam_to_world_used is None
                or self._object_in_flange is None
            ):
                return
            image = self._extract_image(obs)
            ee_pos = self._extract_ee_pos(obs)
            # The held object's expected world position is the flange
            # position plus the object's offset in the current flange
            # frame.  Compare *that* against the resolved object-centre
            # target — not the flange itself, otherwise the delta is
            # off by ~GRIPPER_DEPTH_M (~10 cm) systematically.
            joints = self._extract_joints(obs)
            R_flange = _fk(joints)[:3, :3]
            object_pos_est = ee_pos + R_flange @ self._object_in_flange
            self._place_log["post_move_ee_pos"] = ee_pos.tolist()
            self._place_log["post_move_object_pos_est"] = (
                object_pos_est.tolist()
            )
            delta = object_pos_est - self._resolved_target_world
            self._place_log["post_move_delta_xyz"] = delta.tolist()
            err_m = float(np.linalg.norm(delta))
            self._place_log["post_move_err_m"] = err_m

            hitl = (
                getattr(state, "_hitl", None)
                or getattr(state, "hitl", None)
            )
            # Chosen release rotation from planning (panda_hand frame, same
            # convention as R_flange), if available.
            _R_target = self._place_log.get("R_target_world")
            _R_target = (
                np.asarray(_R_target, dtype=np.float64)
                if _R_target is not None
                else None
            )
            _dbg.vis_post_place(
                image, self._resolved_target_world, ee_pos,
                self._cam_to_world_used, self._intrinsics_used,
                hitl=hitl,
                object_pos_world=object_pos_est,
                R_target_world=_R_target,
                R_actual_world=R_flange,
            )
            logger.info(
                f"  Post-move: object err={err_m * 100:.1f}cm "
                f"(target={self._resolved_target_world.tolist()}, "
                f"object_est={object_pos_est.tolist()}, "
                f"ee={ee_pos.tolist()})"
            )
        except Exception as e:
            logger.warning(f"  [debug] Post-move log failed: {e}")

    def _save_place_log(self) -> None:
        try:
            from vlm_orchestrator.place import debug as _dbg
            self._place_log["finished_at"] = time.time()
            self._place_log["status"] = "done"
            _dbg.save_place_log(self._place_log)
        except Exception as e:
            logger.warning(f"  [debug] save_place_log failed: {e}")

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _fail(self, reason: str, message: str) -> None:
        """Set FAILED phase with a known reason string."""
        logger.warning(f"  place tool FAILED ({reason}): {message}")
        self._phase = PlacePhase.FAILED
        self._failure_reason = reason
        self._status_message = f"Place failed [{reason}]: {message}"
        self._place_log["status"] = "failed"
        self._place_log["failure_reason"] = reason
        self._place_log["failure_message"] = message
        try:
            from vlm_orchestrator.place import debug as _dbg
            self._place_log["finished_at"] = time.time()
            _dbg.save_place_log(self._place_log)
        except Exception:
            pass

    def _noop_response(self, obs: dict) -> dict:
        """Hold-position chunk with gripper at its current state."""
        return {"actions": np.tile(self._hold_action(obs), (ACTION_HORIZON, 1))}

    def _hold_action(self, obs: dict) -> np.ndarray:
        joints = self._extract_joints(obs)
        gripper_pos = float(self._extract_gripper(obs))
        gripper_cmd = (
            GRIPPER_OPEN
            if self._phase in (
                PlacePhase.RELEASING,
                PlacePhase.SETTLING,
                PlacePhase.RETREATING,
                PlacePhase.DONE,
            )
            else GRIPPER_CLOSE
        )
        # If we don't have a clear policy (very early), echo current state.
        if self._phase == PlacePhase.IDLE:
            gripper_cmd = (
                GRIPPER_CLOSE
                if gripper_pos < PLACE_GRIPPER_CLOSED_THRESHOLD
                else GRIPPER_OPEN
            )
        return np.concatenate([joints, [gripper_cmd]])

    # ------------------------------------------------------------------
    # Obs extraction (mirrors grasp/tool.py — robolab joint-space only)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_joints(obs: dict) -> np.ndarray:
        v = obs.get("observation/joint_position")
        if v is None:
            raise RuntimeError(
                "obs missing observation/joint_position (place tool requires "
                "robolab joint-space inputs)"
            )
        return np.asarray(v, dtype=np.float64).reshape(-1)[:7]

    @staticmethod
    def _extract_gripper(obs: dict) -> float:
        v = obs.get("observation/gripper_position")
        if v is None:
            raise RuntimeError("obs missing observation/gripper_position")
        return float(np.asarray(v).reshape(-1)[0])

    @staticmethod
    def _extract_ee_pos(obs: dict) -> np.ndarray:
        v = obs.get("observation/ee_pos")
        if v is None:
            raise RuntimeError("obs missing observation/ee_pos")
        return np.asarray(v, dtype=np.float64).reshape(3)

    def _extract_image(self, obs: dict) -> np.ndarray:
        key = (
            "observation/front_image_left"
            if self._use_front_camera
            else "observation/exterior_image_1_left"
        )
        raw = obs.get(key + "_raw")
        if raw is not None:
            return np.asarray(raw)
        img = obs.get(key)
        if img is None:
            raise RuntimeError(f"obs missing {key}")
        return np.asarray(img)

    def _extract_depth(self, obs: dict) -> np.ndarray | None:
        if self._use_front_camera:
            d = obs.get("observation/depth_front")
            if d is not None:
                return np.asarray(d)
        d = obs.get("observation/depth_external")
        if d is not None:
            return np.asarray(d)
        return None

    def _intrinsics_for_obs(self, obs: dict) -> CameraIntrinsics:
        """Prefer per-frame camera_K from obs; fall back to static defaults."""
        sfx = "_front" if self._use_front_camera else ""
        K = obs.get(f"observation/camera_K{sfx}")
        if K is None and not self._use_front_camera:
            K = obs.get("observation/camera_K")
        depth = self._extract_depth(obs)
        if K is not None and depth is not None:
            d2 = np.squeeze(depth)
            h, w = d2.shape[:2]
            K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
            return CameraIntrinsics(
                fx=float(K_arr[0, 0]), fy=float(K_arr[1, 1]),
                cx=float(K_arr[0, 2]), cy=float(K_arr[1, 2]),
                width=w, height=h,
            )
        if self._use_front_camera:
            return front_camera_intrinsics()
        return self._intrinsics

    def _get_camera_to_world(self, obs: dict) -> np.ndarray:
        """Get camera-to-world (4×4, OpenCV).  Mirror of grasp tool's helper.

        Reads runtime pose from obs only — never substitutes a baked-in
        fallback (the "no silent lossy fallbacks" design rule rule).
        """
        cam_ext = obs.get("observation/camera_extrinsic")
        if cam_ext is not None:
            return np.asarray(cam_ext, dtype=np.float64).reshape(4, 4)
        sfx = "_front" if self._use_front_camera else ""
        cam_pos = obs.get(f"observation/camera_pos{sfx}")
        cam_quat = obs.get(f"observation/camera_quat{sfx}")
        if cam_pos is not None and cam_quat is not None:
            pos = np.asarray(cam_pos).flatten()[:3]
            quat = np.asarray(cam_quat).flatten()[:4]
            return pose_opengl_to_opencv(pos, quat)
        raise PerceptionFailure(
            f"No camera pose in obs (looked for observation/camera_extrinsic, "
            f"observation/camera_pos{sfx}, observation/camera_quat{sfx}). "
            f"The eval client must forward camera extrinsics for placement."
        )

    # NOTE: previously a `_held_object_height_m(obs)` helper read the held
    # object's AABB extent from gt_state to compute a per-object half-
    # height lift.  Removed — gt_state is sim-only privileged info, and
    # the half-height framing itself doesn't generalise either (no real
    # robot has AABBs).  Replaced by ``PLACE_RELEASE_HEIGHT_M``: a fixed
    # finger-tip clearance at release, agnostic to object size.  Uses an
    # empirical "drop from safe height, gravity finishes the placement"
    # strategy.
