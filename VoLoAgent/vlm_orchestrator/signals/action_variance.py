# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action Variance metric.

Simpler version of TD — probes the VLA N times and computes the mean
variance of the returned action chunks.  Also reports first-step variance,
spread, and gripper agreement.

Corresponds to the original Metric 1 (Behavioral Vagueness).
"""

from __future__ import annotations

import numpy as np

from vlm_orchestrator.signals.base import Metric, MetricResult


class ActionVariance(Metric):
    """Mean action variance over N policy probes."""

    name = "action_variance"
    needs_policy = True
    needs_obs = True

    def __init__(
        self,
        n_samples: int = 30,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        self.n_samples = n_samples
        self._host = host
        self._port = port
        self._prober = None

    def setup(self, **kwargs):
        from vlm_orchestrator.signals.policy_prober import PolicyProber

        self._prober = PolicyProber(host=self._host, port=self._port)

    def teardown(self):
        if self._prober is not None:
            self._prober.close()
            self._prober = None

    def compute(self, instruction, *, obs=None, **kwargs) -> MetricResult:
        if obs is None:
            raise ValueError("ActionVariance metric requires an observation dict (obs=)")
        if self._prober is None:
            self.setup()

        probe_obs = dict(obs)
        probe_obs["prompt"] = instruction
        result = self._prober.probe(probe_obs, n_samples=self.n_samples)

        actions = result.all_actions  # (N, horizon, dim)

        # First-step variance
        first_step_var = float(np.mean(np.var(actions[:, 0, :], axis=0)))

        # Gripper agreement at step 0
        gripper_actions = actions[:, 0, -1]
        gripper_binary = (gripper_actions > 0.5).astype(float)
        gripper_agreement = max(np.mean(gripper_binary), 1 - np.mean(gripper_binary))

        return MetricResult(
            name=self.name,
            score=result.mean_action_var,
            detail={
                "first_step_var": first_step_var,
                "action_spread": result.action_spread,
                "gripper_agreement": float(gripper_agreement),
                "traj_disagreement": result.traj_disagreement,
                "n_samples": self.n_samples,
            },
        )
