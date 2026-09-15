# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory Disagreement (TD) metric.

Probes the VLA policy N times with the same observation and measures
how much the resulting action trajectories disagree.  This is the core
metric used in the adaptive compare-and-pick pipeline.

Higher TD → the policy is confused → the instruction is ambiguous.
"""

from __future__ import annotations

from vlm_orchestrator.signals.base import Metric, MetricResult


class TrajectoryDisagreement(Metric):
    """Trajectory disagreement via policy probing."""

    name = "td"
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
            raise ValueError("TD metric requires an observation dict (obs=)")
        if self._prober is None:
            self.setup()

        probe_obs = dict(obs)
        probe_obs["prompt"] = instruction

        result = self._prober.probe(probe_obs, n_samples=self.n_samples)

        return MetricResult(
            name=self.name,
            score=result.traj_disagreement,
            detail={
                "mean_action_var": result.mean_action_var,
                "action_spread": result.action_spread,
                "n_samples": self.n_samples,
            },
        )

    def compare(self, original, rewritten, *, obs=None, **kwargs) -> MetricResult:
        """Optimised compare that reuses the prober connection."""
        if obs is None:
            raise ValueError("TD metric requires an observation dict (obs=)")
        if self._prober is None:
            self.setup()

        from vlm_orchestrator.signals.policy_prober import compare_and_pick

        orig_obs = dict(obs)
        orig_obs["prompt"] = original
        orig_result = self._prober.probe(orig_obs, n_samples=self.n_samples)

        if rewritten == original:
            return MetricResult(
                name=self.name,
                score=orig_result.traj_disagreement,
                score_original=orig_result.traj_disagreement,
                score_rewritten=orig_result.traj_disagreement,
                improvement=0.0,
                detail={"identical": True},
            )

        rw_obs = dict(obs)
        rw_obs["prompt"] = rewritten
        rw_result = self._prober.probe(rw_obs, n_samples=self.n_samples)

        cmp = compare_and_pick(orig_result, rw_result)

        return MetricResult(
            name=self.name,
            score=rw_result.traj_disagreement,
            score_original=orig_result.traj_disagreement,
            score_rewritten=rw_result.traj_disagreement,
            improvement=cmp.improvement,
            detail={
                "decision": cmp.winner,
                "orig_var": orig_result.mean_action_var,
                "rw_var": rw_result.mean_action_var,
            },
        )
