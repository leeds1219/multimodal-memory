# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manager for passive ground-truth evaluation metrics."""

from __future__ import annotations

from typing import Any

from vlm_orchestrator.gt_metrics.detectors import (
    GTMetricDetector,
    GTStateCheckDetector,
    GTTaskStateCheckDetector,
    MetricContext,
    PlacementOutcomeDetector,
    VLMFailureStateQADetector,
    VLMGraspTargetQADetector,
    VLMPlanAlignmentDetector,
    VLMResponseFormatDetector,
    VLMTaskInvariantQADetector,
    ToolCausalityDetector,
    VLMCompletionMismatchDetector,
    VLMPerceptionMismatchDetector,
    VLMSceneQADetector,
    _gt_state,
)
from vlm_orchestrator.gt_metrics.events import GTMetricEvent


_ALL_METRIC_TYPES = {
    "perception", "placement", "tool_causality", "gt_state", "scene_qa",
    "target_qa", "plan_qa", "format_qa", "failure_qa", "task_qa",
}


class GTMetricsManager:
    """Run passive GT metric detectors over each proxy step."""

    def __init__(
        self,
        *,
        metric_types: set[str] | list[str] | tuple[str, ...] | None = None,
        placement_confirm_steps: int = 5,
        attribution_window_steps: int = 10,
    ) -> None:
        selected = set(metric_types) if metric_types is not None else set(_ALL_METRIC_TYPES)
        if "all" in selected:
            selected = set(_ALL_METRIC_TYPES)
        unknown = selected - _ALL_METRIC_TYPES
        if unknown:
            raise ValueError(
                f"Unknown GT metric type(s): {sorted(unknown)}. "
                f"Valid types: {sorted(_ALL_METRIC_TYPES)}"
            )

        detectors: list[GTMetricDetector] = []
        adds_task_qa = "scene_qa" in selected or "perception" in selected
        if adds_task_qa:
            detectors.append(VLMResponseFormatDetector())
            detectors.append(VLMSceneQADetector())
            detectors.append(VLMTaskInvariantQADetector())
        elif "format_qa" in selected:
            detectors.append(VLMResponseFormatDetector())
        if "task_qa" in selected and not adds_task_qa:
            detectors.append(VLMTaskInvariantQADetector())
        if "failure_qa" in selected or "scene_qa" in selected:
            detectors.append(VLMFailureStateQADetector())
        if "target_qa" in selected or "scene_qa" in selected:
            detectors.append(VLMGraspTargetQADetector())
        if "plan_qa" in selected:
            detectors.append(VLMPlanAlignmentDetector())
        if "perception" in selected:
            detectors.append(VLMPerceptionMismatchDetector())
            detectors.append(VLMCompletionMismatchDetector())
        if "placement" in selected:
            detectors.append(PlacementOutcomeDetector(
                confirm_steps=placement_confirm_steps,
            ))
        if "tool_causality" in selected:
            detectors.append(ToolCausalityDetector(
                attribution_window_steps=attribution_window_steps,
            ))
        if "gt_state" in selected:
            detectors.append(GTStateCheckDetector())
            detectors.append(GTTaskStateCheckDetector())
        self.detectors = detectors
        self._last_episode_id: int | None = None

    def reset(self) -> None:
        for detector in self.detectors:
            detector.reset()
        self._last_episode_id = None

    def step(
        self,
        obs: dict[str, Any],
        state: Any,
        *,
        control_entries: list[dict[str, Any]] | None = None,
    ) -> list[GTMetricEvent]:
        episode_id = int(getattr(state, "episode_id", 0) or 0)
        if self._last_episode_id is None:
            self._last_episode_id = episode_id
        elif episode_id != self._last_episode_id:
            self.reset()
            self._last_episode_id = episode_id

        ctx = MetricContext(
            obs=obs,
            gt_state=_gt_state(obs),
            state=state,
            control_entries=list(control_entries or []),
            episode_id=episode_id,
            step_count=int(getattr(state, "infer_count", 0) or 0),
            episode_step=int(getattr(state, "episode_step", 0) or 0),
            subgoal_idx=int(getattr(state, "current_subgoal_idx", 0) or 0),
        )

        events: list[GTMetricEvent] = []
        for detector in self.detectors:
            events.extend(detector.detect(ctx))
        return events
