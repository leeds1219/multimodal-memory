# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Place-with-tool recovery pipeline.

Submodules
----------
tool         – PlaceToolExecutor: planned placement that bypasses the VLA.
client       – HTTP client wrapper (reuses GraspClient endpoints).
destination  – Destination grounding (gt_sim / sam3 / gdino_sam2 / vlm_point).
debug        – Debug visualization for the place pipeline.
"""

from vlm_orchestrator.place.tool import (
    DestinationSpec,
    PlacePhase,
    PlaceSegMode,
    PlaceToolExecutor,
)

__all__ = [
    "DestinationSpec",
    "PlacePhase",
    "PlaceSegMode",
    "PlaceToolExecutor",
]
