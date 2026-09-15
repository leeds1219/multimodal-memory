# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side cuRobo motion planner.

Implements the :class:`~vlm_orchestrator.motion.planner.MotionPlanner`
protocol by delegating each segment to the grasp server's ``/plan_motion``
endpoint (cuRobo ``plan_cspace``).  The server must be started with
``--enable-curobo``.

Loud-fail semantics (per the design rules): on any planning failure this returns a
*failed* :class:`MotionPlanResult` (``success=False``, ``waypoints=None``).
The tool layer records the label and raises — it does NOT silently fall
back to linear interpolation.  A future ``--curobo-on-fail linear`` policy
would be an explicit, logged decision made by the caller, not here.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from vlm_orchestrator.motion.planner import MotionPlanResult

logger = logging.getLogger(__name__)


class CuroboRemotePlanner:
    """Motion planner backed by the grasp server's cuRobo endpoint."""

    name = "curobo"

    def __init__(self, grasp_server_url: str | None = None):
        # Reuse the grasp client (same server host/port resolution).
        from vlm_orchestrator.grasp.client import GraspClient

        self._client = GraspClient(url=grasp_server_url)

    def plan_segment(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        *,
        n_steps: int,
        scene_pc: Optional[np.ndarray] = None,
        phase: Optional[str] = None,
    ) -> MotionPlanResult:
        # The APPROACH segment brings the gripper down toward an on-table
        # object; its fingers extend ~10 cm below panda_hand and would trip the
        # table collision check at the pre-grasp goal.  Disable the finger/hand
        # collision links for this segment only (exactly cuRobo's own
        # plan_grasp behaviour).  Other phases keep full collision.  Only
        # relevant when a scene cloud is present (collision-free path); in
        # plain-cuRobo mode (empty world) there is nothing to collide with, so
        # keep the full model untouched.
        disable_fingers = (phase == "approach") and (scene_pc is not None)
        try:
            waypoints, ok = self._client.plan_motion(
                q_start, q_end, n_steps=n_steps, scene_pc=scene_pc,
                disable_fingers=disable_fingers,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"cuRobo plan_segment ({phase}) raised {type(e).__name__}: {e}"
            )
            return MotionPlanResult.failed(
                "curobo_failed", f"{type(e).__name__}: {e}",
            )

        if not ok or waypoints is None or len(waypoints) == 0:
            return MotionPlanResult.failed(
                "curobo_failed",
                f"planner returned no trajectory for {phase} segment",
            )

        return MotionPlanResult.ok(
            [np.asarray(w) for w in waypoints], self.name,
        )

    # -- Attached-object collision (PLACE path) --

    def attach_object(
        self,
        obj_pc: np.ndarray,
        q_hold: np.ndarray,
        *,
        num_spheres: int = 4,
    ) -> bool:
        """Attach a held-object cloud to the gripper link on the server.

        Returns True on success, False on any failure (never raises).
        """
        try:
            return self._client.attach_object(
                obj_pc, q_hold, num_spheres=num_spheres,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"cuRobo attach_object raised {type(e).__name__}: {e}"
            )
            return False

    def detach_object(self) -> bool:
        """Detach the held object on the server (safe if none attached)."""
        try:
            return self._client.detach_object()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"cuRobo detach_object raised {type(e).__name__}: {e}"
            )
            return False
