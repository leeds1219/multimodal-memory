# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive ground-truth metric detectors for evaluation logging.

Public submodules:
  - ``events``           — GTMetricEvent dataclass + JSON serialization
  - ``manager``          — GTMetricsManager (runs detectors per step)
  - ``detectors``        — individual passive detectors
                           (VLMSceneQADetector, VLMGraspTargetQADetector, ...)
  - ``target_resolver``  — TargetResolver: string + alias + optional
                           VLM-grounded matcher for ``vlm_grasp_target_*``
                           events. Used by both the dashboard build and
                           the offline ``scripts/resolve_target_mismatches.py``
                           reprocessor. See module docstring for the
                           live-detector ↔ offline-resolver split.
"""

from vlm_orchestrator.gt_metrics.events import GTMetricEvent
from vlm_orchestrator.gt_metrics.manager import GTMetricsManager
from vlm_orchestrator.gt_metrics.target_resolver import (
    ResolveResult,
    TargetResolver,
)

__all__ = [
    "GTMetricEvent",
    "GTMetricsManager",
    "ResolveResult",
    "TargetResolver",
]
