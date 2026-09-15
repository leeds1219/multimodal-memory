# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-planning abstraction for grasp / place trajectory generation.

The grasp and place tools generate joint-space trajectories between IK
solutions (current → pre-grasp → grasp, etc.).  Historically every segment
was a straight linear interpolation in joint space
(:func:`vlm_orchestrator.grasp.ik.interpolate_joints`).

This package introduces a small :class:`MotionPlanner` seam so the path
*between* two joint configurations can be swapped (e.g. for a collision-aware
planner) without touching pose selection / IK.  The default
:class:`LinearInterpPlanner` reproduces the historical behaviour exactly.
"""

from vlm_orchestrator.motion.planner import (
    LinearInterpPlanner,
    MotionPlanner,
    MotionPlanResult,
    build_motion_planner,
)

__all__ = [
    "MotionPlanner",
    "MotionPlanResult",
    "LinearInterpPlanner",
    "build_motion_planner",
]
