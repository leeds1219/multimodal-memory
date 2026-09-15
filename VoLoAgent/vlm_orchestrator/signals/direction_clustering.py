# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direction Clustering metric.

Probes the VLA N times, extracts first-step joint deltas, runs K-means
clustering, and measures disagreement (1 − largest cluster fraction).

High disagreement → the policy reaches in different directions → ambiguous.

Requires sklearn.
"""

from __future__ import annotations

import numpy as np

from vlm_orchestrator.signals.base import Metric, MetricResult


def _to_unit(vecs: np.ndarray) -> np.ndarray:
    """Normalise rows to unit length."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-10)


def _direction_clustering(actions: np.ndarray, initial_joints: np.ndarray | None = None) -> dict:
    """Compute direction clustering on first-step joint deltas.

    Parameters
    ----------
    actions : ndarray, shape (N, horizon, 8)
    initial_joints : ndarray, shape (7,), optional
        If provided, deltas are computed relative to initial joints.
        Otherwise, raw first-step positions are used directly.

    Returns
    -------
    dict with disagreement, silhouette, cluster sizes for k=2,3.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if initial_joints is not None:
        first_deltas = actions[:, 0, :7] - initial_joints[np.newaxis, :]
    else:
        first_deltas = actions[:, 0, :7]

    unit_dirs = _to_unit(first_deltas)
    N = unit_dirs.shape[0]

    # Pairwise cosine consensus
    cos_mat = unit_dirs @ unit_dirs.T
    triu = np.triu_indices(N, k=1)
    consensus = float(cos_mat[triu].mean())

    results = {
        "direction_consensus": consensus,
        "direction_disagreement": 1.0 - consensus,
        "min_cosine_sim": float(cos_mat[triu].min()),
    }

    for k in [2, 3]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(unit_dirs)
        sizes = np.bincount(labels, minlength=k)
        largest_frac = sizes.max() / N
        n_nonempty = np.sum(sizes > 0)
        sil = silhouette_score(unit_dirs, labels) if n_nonempty >= 2 else -1.0

        results[f"k{k}_disagreement"] = float(1.0 - largest_frac)
        results[f"k{k}_largest_frac"] = float(largest_frac)
        results[f"k{k}_silhouette"] = float(sil)
        results[f"k{k}_cluster_sizes"] = sizes.tolist()

    return results


class DirectionClustering(Metric):
    """First-step direction clustering metric."""

    name = "direction_clustering"
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
            raise ValueError("DirectionClustering metric requires obs=")
        if self._prober is None:
            self.setup()

        probe_obs = dict(obs)
        probe_obs["prompt"] = instruction
        result = self._prober.probe(probe_obs, n_samples=self.n_samples)

        initial_joints = obs.get("observation/joint_position")
        dc = _direction_clustering(result.all_actions, initial_joints)

        return MetricResult(
            name=self.name,
            score=dc["direction_disagreement"],
            detail=dc,
        )
