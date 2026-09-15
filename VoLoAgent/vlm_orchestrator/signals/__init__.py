# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instruction-quality metrics for the VLM orchestrator.

Each metric answers: "How ambiguous is this instruction for the policy?"

Available metrics:

- **td** (Trajectory Disagreement):
  Probe the VLA N times, measure how much the sampled action trajectories
  disagree. Core metric used in compare-and-pick.

- **action_variance**:
  Simpler version of TD — just the mean variance of action chunks over N probes.

- **direction_clustering**:
  K-means on first-step joint deltas; measures whether the policy produces
  distinct movement "modes" (e.g. reach left vs reach right).

- **trajectory_modes**:
  K-means + DBSCAN on 8-step trajectory windows; detects multi-modal
  trajectory distributions.

- **linguistic**:
  Pure text analysis — counts grounding words (colors, spatial refs, quantities),
  computes a vagueness score. No VLA or VLM needed.

- **vlm_entropy**:
  Ask a VLM to generate K diverse interpretations of the instruction,
  cluster them, compute Shannon entropy. High entropy = ambiguous.

All metrics implement a common interface::

    def compute(instruction, *, obs=None, task=None, **kwargs) -> MetricResult

Use :func:`get_metric` to instantiate by name, or :func:`run_metrics` to
run several at once over a set of tasks.
"""

from vlm_orchestrator.signals.registry import get_metric, list_metrics
from vlm_orchestrator.signals.base import MetricResult

__all__ = ["get_metric", "list_metrics", "MetricResult"]
