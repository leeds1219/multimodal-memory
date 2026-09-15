# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory Modes metric.

Probes the VLA N times, extracts 8-step trajectory windows, runs
DBSCAN + K-means to detect distinct trajectory modes, and computes
mode entropy.

High mode count / entropy → the policy has multiple "strategies" → ambiguous.

Requires sklearn.
"""

from __future__ import annotations

import numpy as np

from vlm_orchestrator.signals.base import Metric, MetricResult


def _trajectory_modes(actions: np.ndarray) -> dict:
    """Detect trajectory modes in N sampled action chunks.

    Parameters
    ----------
    actions : ndarray, shape (N, horizon, 8)

    Returns
    -------
    dict with mode count, entropy, DBSCAN and K-means details.
    """
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    traj = actions[:, :8, :]  # first 8 steps
    N = traj.shape[0]
    flat = traj.reshape(N, -1)

    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)

    results: dict = {}

    # DBSCAN: auto-detect modes
    best_db = None
    for eps in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        db = DBSCAN(eps=eps, min_samples=3)
        labels = db.fit_predict(flat_scaled)
        n_clusters = len(set(labels) - {-1})
        n_noise = int(np.sum(labels == -1))
        noise_frac = n_noise / N
        if n_clusters >= 1 and noise_frac < 0.5:
            if best_db is None or n_clusters > 1:
                best_db = {
                    "eps": eps,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_frac": float(noise_frac),
                }
            if n_clusters >= 2:
                break

    if best_db is None:
        best_db = {"eps": -1, "n_clusters": 1, "n_noise": 0, "noise_frac": 0.0}
    results["dbscan_modes"] = best_db["n_clusters"]
    results["dbscan_eps"] = best_db["eps"]
    results["dbscan_noise_frac"] = best_db.get("noise_frac", 0.0)

    # K-means k=1..5 — pick best by silhouette
    sil_scores = {}
    for k in range(1, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(flat_scaled)
        sil_scores[k] = float(silhouette_score(flat_scaled, labels)) if k >= 2 else 0.0

    best_k = max(range(2, 6), key=lambda k: sil_scores[k])
    results["kmeans_best_k"] = best_k
    results["kmeans_best_silhouette"] = sil_scores[best_k]

    # Decision: real split if k=2 silhouette > 0.25
    n_modes = best_k if sil_scores[2] > 0.25 else 1
    results["trajectory_modes"] = n_modes

    # Mode entropy for best k
    km_best = KMeans(n_clusters=n_modes, n_init=10, random_state=42)
    labels_best = km_best.fit_predict(flat_scaled)
    sizes = np.bincount(labels_best, minlength=n_modes)
    probs = sizes / sizes.sum()
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_ent = float(np.log2(n_modes)) if n_modes > 1 else 0.0

    results["mode_entropy"] = entropy
    results["mode_max_entropy"] = max_ent
    results["mode_normalized_entropy"] = entropy / max_ent if max_ent > 0 else 0.0
    results["mode_cluster_sizes"] = sizes.tolist()

    return results


class TrajectoryModes(Metric):
    """Trajectory mode detection metric."""

    name = "trajectory_modes"
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
            raise ValueError("TrajectoryModes metric requires obs=")
        if self._prober is None:
            self.setup()

        probe_obs = dict(obs)
        probe_obs["prompt"] = instruction
        result = self._prober.probe(probe_obs, n_samples=self.n_samples)

        tm = _trajectory_modes(result.all_actions)

        return MetricResult(
            name=self.name,
            score=tm["mode_entropy"],
            detail=tm,
        )
