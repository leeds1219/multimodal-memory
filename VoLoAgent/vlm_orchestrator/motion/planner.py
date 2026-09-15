# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-planner interface + default linear-interpolation implementation.

Design
------
Grasp / place tools plan joint-space trajectory *segments* between two IK
solutions.  A :class:`MotionPlanner` turns a ``(q_start, q_end)`` pair plus a
step budget into a list of intermediate joint configurations.

* :class:`LinearInterpPlanner` (default) is a thin wrapper over
  :func:`vlm_orchestrator.grasp.ik.interpolate_joints` — byte-for-byte the
  historical behaviour.  It never fails.
* Future collision-aware planners (e.g. cuRobo) implement the same protocol.
  They may fail (unreachable / no collision-free path).  Per the repo's
  "no silent lossy fallback" rule they must return a *failed*
  :class:`MotionPlanResult` (``success=False``, ``waypoints=None``) with a
  descriptive ``label`` so the caller can log it and decide — never quietly
  substitute a straight line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class MotionPlanResult:
    """Result of planning one joint-space trajectory segment.

    Attributes
    ----------
    waypoints:
        List of joint configurations (each a ``(7,)`` array) including both
        endpoints, or ``None`` when planning failed.
    label:
        Planner-specific strategy label, propagated to ``grasp_log`` /
        ``place_log`` for post-hoc observability (e.g. ``"linear"``,
        ``"curobo"``, ``"curobo_failed"``).
    success:
        ``True`` iff ``waypoints`` is a usable trajectory.
    detail:
        Optional human-readable reason (populated on failure).
    """

    waypoints: Optional[list[np.ndarray]]
    label: str
    success: bool
    detail: str = ""

    @classmethod
    def ok(cls, waypoints: list[np.ndarray], label: str) -> "MotionPlanResult":
        return cls(waypoints=waypoints, label=label, success=True)

    @classmethod
    def failed(cls, label: str, detail: str) -> "MotionPlanResult":
        return cls(waypoints=None, label=label, success=False, detail=detail)


@runtime_checkable
class MotionPlanner(Protocol):
    """Plans a single joint-space trajectory segment.

    Implementations must be side-effect-free and thread-safe enough to be
    called once per tool segment.
    """

    #: Short stable name, used in logs and as the default result label.
    name: str

    def plan_segment(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        *,
        n_steps: int,
        scene_pc: Optional[np.ndarray] = None,
        phase: Optional[str] = None,
    ) -> MotionPlanResult:
        """Plan ``q_start`` → ``q_end`` in ``n_steps`` waypoints.

        Parameters
        ----------
        q_start, q_end:
            ``(7,)`` joint configurations (arm only; gripper handled by caller).
        n_steps:
            Requested number of waypoints (including endpoints).  Planners may
            treat this as advisory but should honour the endpoints.
        scene_pc:
            Optional ``(N, 3)`` scene point cloud (world/base frame) for
            collision-aware planners.  Ignored by the linear planner.
        phase:
            Optional label for the segment (``"approach"`` / ``"final"`` /
            ``"lift"`` / ``"retreat"``) for logging.
        """
        ...

    def attach_object(
        self, obj_pc: np.ndarray, q_hold: np.ndarray, *, num_spheres: int = 4,
    ) -> bool:
        """Attach a held-object cloud to the gripper link for collision-aware
        planning (PLACE path).  Collision-unaware planners are a no-op and
        return ``False``.  Never raises."""
        ...

    def detach_object(self) -> bool:
        """Detach a previously-attached held object.  No-op / ``False`` on
        planners that don't support attachment.  Never raises."""
        ...


class LinearInterpPlanner:
    """Straight-line joint-space interpolation (historical default).

    Reproduces :func:`vlm_orchestrator.grasp.ik.interpolate_joints` exactly.
    Never fails.
    """

    name = "linear"

    def plan_segment(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        *,
        n_steps: int,
        scene_pc: Optional[np.ndarray] = None,
        phase: Optional[str] = None,
    ) -> MotionPlanResult:
        # Imported lazily to avoid a hard import cycle
        # (ik.py has no dependency on this module, but keep the seam clean).
        from vlm_orchestrator.grasp.ik import interpolate_joints

        waypoints = interpolate_joints(q_start, q_end, n_steps)
        return MotionPlanResult.ok(waypoints, self.name)

    def attach_object(
        self, obj_pc: np.ndarray, q_hold: np.ndarray, *, num_spheres: int = 4,
    ) -> bool:
        # Linear interpolation is collision-unaware — nothing to attach.
        return False

    def detach_object(self) -> bool:
        return False


def build_motion_planner(kind: str = "linear", **kwargs) -> MotionPlanner:
    """Factory: construct a motion planner by name.

    Parameters
    ----------
    kind:
        ``"linear"`` (default) → :class:`LinearInterpPlanner`.
        ``"curobo"`` → collision-aware cuRobo remote planner (added later).
    **kwargs:
        Forwarded to the concrete planner constructor (e.g. cuRobo server
        host/port).  Ignored by the linear planner.
    """
    if kind == "linear":
        return LinearInterpPlanner()
    if kind == "curobo":
        # Deferred import — cuRobo client lives in a submodule added in a
        # later step so the linear path carries no extra dependency.
        from vlm_orchestrator.motion.curobo_client import CuroboRemotePlanner

        return CuroboRemotePlanner(**kwargs)
    raise ValueError(
        f"Unknown motion planner {kind!r}; expected 'linear' or 'curobo'"
    )
