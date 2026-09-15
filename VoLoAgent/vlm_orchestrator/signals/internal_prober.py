# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prober that extracts and analyses model-internal activations.

Unlike :class:`~vlm_orchestrator.signals.policy_prober.PolicyProber` (which only sees
output actions), this sends ``__diagnose__`` requests that return velocity
field vectors, action-expert hidden-state norms, and prefix embeddings at
every denoising step.

Requires the instrumented OpenPI server (see the ``__diagnose__`` protocol
added to ``websocket_policy_server.py``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vlm_orchestrator.utils import codec

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass
class VelocityFieldAnalysis:
    """Analysis of the velocity field across N denoising trajectories."""

    mean_norm_profile: list[float]
    """Mean ‖v_t‖ at each denoising step, averaged over N samples."""
    late_early_ratio: float
    """mean(‖v_t‖ for last 1/3 steps) / mean(‖v_t‖ for first 1/3 steps).
    Lower = faster convergence = more confident."""
    agreement_per_step: list[float]
    """Cross-sample cosine agreement of v_t at each step."""
    late_agreement: float
    """Mean cosine agreement at the last 1/3 of steps.  Higher = more confident."""

    def to_dict(self) -> dict:
        return {
            "mean_norm_profile": self.mean_norm_profile,
            "late_early_ratio": round(self.late_early_ratio, 6),
            "agreement_per_step": [round(a, 4) for a in self.agreement_per_step],
            "late_agreement": round(self.late_agreement, 4),
        }


@dataclass
class ConvergenceAnalysis:
    """How quickly the denoising trajectory converges."""

    delta_norm_profile: list[float]
    """Mean ‖x_{t+1} − x_t‖ at each step, averaged over N samples."""
    convergence_ratio: float
    """Last-step delta / first-step delta.  Lower = faster convergence."""
    final_spread: float
    """Mean L2 distance of final predictions from their centroid."""

    def to_dict(self) -> dict:
        return {
            "delta_norm_profile": [round(d, 6) for d in self.delta_norm_profile],
            "convergence_ratio": round(self.convergence_ratio, 6),
            "final_spread": round(self.final_spread, 6),
        }


@dataclass
class HiddenStateAnalysis:
    """Analysis of action-expert hidden states."""

    hidden_norm_profile: list[float]
    """Mean hidden-state norm per denoising step."""
    final_hidden_agreement: float
    """Cross-sample cosine agreement of hidden states at the final step."""

    def to_dict(self) -> dict:
        return {
            "hidden_norm_profile": [round(h, 4) for h in self.hidden_norm_profile],
            "final_hidden_agreement": round(self.final_hidden_agreement, 4),
        }


@dataclass
class InternalProbeResult:
    """Full diagnostic result from an internal probe."""

    n_samples: int
    elapsed_s: float
    velocity: VelocityFieldAnalysis
    convergence: ConvergenceAnalysis
    hidden: HiddenStateAnalysis
    self_consistency: dict | None = None
    """Flow-matching loss on the model's own prediction (if requested)."""
    prefix_embedding: np.ndarray | None = field(default=None, repr=False)
    actions: np.ndarray | None = field(default=None, repr=False)
    """Stacked output actions [N, H, D] for downstream use."""

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "n_samples": self.n_samples,
            "elapsed_s": round(self.elapsed_s, 2),
            "velocity": self.velocity.to_dict(),
            "convergence": self.convergence.to_dict(),
            "hidden": self.hidden.to_dict(),
        }
        if self.self_consistency is not None:
            d["self_consistency"] = self.self_consistency
        return d

    @property
    def confidence_scores(self) -> dict[str, float]:
        """Composite confidence scores, each in [0, 1], higher = more confident."""
        scores = {
            "velocity_convergence": 1.0 - min(self.velocity.late_early_ratio, 1.0),
            "velocity_agreement": self.velocity.late_agreement,
            "trajectory_convergence": 1.0 - min(self.convergence.convergence_ratio, 1.0),
            "hidden_agreement": self.hidden.final_hidden_agreement,
        }
        if self.self_consistency is not None:
            ml = self.self_consistency.get("midrange_loss", self.self_consistency.get("mean_loss", 1.0))
            scores["self_consistency"] = 1.0 / (1.0 + ml)
        return {k: round(v, 4) for k, v in scores.items()}


# ------------------------------------------------------------------
# Pure-numpy analysis helpers
# ------------------------------------------------------------------

def _pairwise_cosine_mean(vectors: np.ndarray) -> float:
    """Mean pairwise cosine similarity of rows in *vectors* [N, D]."""
    N = vectors.shape[0]
    if N < 2:
        return 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-8)
    cos = unit @ unit.T
    return float(cos[np.triu_indices(N, k=1)].mean())


def analyse_velocity(v_norms: np.ndarray, v_vecs: np.ndarray) -> VelocityFieldAnalysis:
    """
    Parameters
    ----------
    v_norms : [N, num_steps]
    v_vecs  : [N, num_steps, action_horizon, action_dim]
    """
    N, S = v_norms.shape
    mean_profile = v_norms.mean(axis=0).tolist()
    n_third = max(1, S // 3)
    late_v = float(v_norms[:, -n_third:].mean())
    early_v = float(v_norms[:, :n_third].mean())
    ratio = late_v / max(early_v, 1e-8)

    agreement = []
    for s in range(S):
        flat = v_vecs[:, s].reshape(N, -1)
        agreement.append(_pairwise_cosine_mean(flat))

    late_agree = float(np.mean(agreement[-n_third:]))
    return VelocityFieldAnalysis(
        mean_norm_profile=mean_profile,
        late_early_ratio=ratio,
        agreement_per_step=agreement,
        late_agreement=late_agree,
    )


def analyse_convergence(x_traj: np.ndarray) -> ConvergenceAnalysis:
    """
    Parameters
    ----------
    x_traj : [N, num_steps+1, action_horizon, action_dim]
    """
    N = x_traj.shape[0]
    deltas = np.diff(x_traj, axis=1)                       # [N, S, H, D]
    delta_norms = np.linalg.norm(
        deltas.reshape(N, deltas.shape[1], -1), axis=-1,
    )                                                        # [N, S]
    profile = delta_norms.mean(axis=0).tolist()              # [S]
    ratio = profile[-1] / max(profile[0], 1e-8)

    finals = x_traj[:, -1].reshape(N, -1)                   # [N, H*D]
    centroid = finals.mean(axis=0)
    spread = float(np.linalg.norm(finals - centroid, axis=1).mean())

    return ConvergenceAnalysis(
        delta_norm_profile=profile,
        convergence_ratio=ratio,
        final_spread=spread,
    )


def analyse_hidden(h_norms: np.ndarray, v_vecs: np.ndarray) -> HiddenStateAnalysis:
    """
    Parameters
    ----------
    h_norms : [N, num_steps]  – mean hidden-state norms per step.
    v_vecs  : [N, num_steps, action_horizon, action_dim]  – used for
              final-step cross-sample agreement (hidden states are not
              transferred in full to save bandwidth; we approximate via
              the last-step v_t vectors which are a linear projection of
              the hidden states).
    """
    profile = h_norms.mean(axis=0).tolist()
    N = v_vecs.shape[0]
    final_flat = v_vecs[:, -1].reshape(N, -1)
    agree = _pairwise_cosine_mean(final_flat)
    return HiddenStateAnalysis(
        hidden_norm_profile=profile,
        final_hidden_agreement=agree,
    )


# ------------------------------------------------------------------
# Main prober class
# ------------------------------------------------------------------

class InternalProber:
    """Extracts model-internal activations from the VLA policy server.

    Requires the server to support the ``__diagnose__`` and
    ``__self_consistency__`` message flags (see the instrumented
    ``websocket_policy_server.py``).

    Usage::

        with InternalProber("127.0.0.1", 8000) as prober:
            result = prober.probe(obs_dict, n_samples=10)
            print(result.confidence_scores)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        import websockets.sync.client as ws_client

        uri = f"ws://{host}:{port}"
        logger.info("InternalProber connecting to %s", uri)
        self._ws = ws_client.connect(uri, compression=None, max_size=None)
        self._packer = codec.Packer()
        self._metadata = codec.unpackb(self._ws.recv())
        logger.info("InternalProber connected – metadata: %s", self._metadata)

    # ----- core probes -------------------------------------------------

    def probe(
        self,
        obs_dict: dict,
        n_samples: int = 10,
        *,
        self_consistency: bool = True,
        consistency_probes: int = 5,
    ) -> InternalProbeResult:
        """Send *n_samples* diagnostic requests and return analysed internals.

        If *self_consistency* is True, also computes the flow-matching
        self-consistency loss on the mean predicted action (costs one extra
        round-trip).
        """
        diag_obs = dict(obs_dict)
        diag_obs["__diagnose__"] = True
        packed = self._packer.pack(diag_obs)

        all_v_norms: list[np.ndarray] = []
        all_v_vecs: list[np.ndarray] = []
        all_x_traj: list[np.ndarray] = []
        all_h_norms: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        prefix_embs: list[np.ndarray] = []

        t0 = time.time()
        for i in range(n_samples):
            self._ws.send(packed)
            raw = self._ws.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Server error on probe {i}: {raw[:300]}")
            resp = codec.unpackb(raw)
            d = resp["diagnostics"]
            all_v_norms.append(d["velocity_norms"])
            all_v_vecs.append(d["velocity_vectors"])
            all_x_traj.append(d["x_trajectory"])
            all_h_norms.append(d["suffix_hidden_norms"])
            all_actions.append(resp["actions"])
            prefix_embs.append(d["prefix_embedding"])

        elapsed = time.time() - t0
        v_norms = np.stack(all_v_norms).squeeze()    # [N, steps]
        v_vecs = np.stack(all_v_vecs).squeeze()      # [N, steps, H, D]
        x_traj = np.stack(all_x_traj).squeeze()      # [N, steps+1, H, D]
        h_norms = np.stack(all_h_norms).squeeze()    # [N, steps]
        actions = np.stack(all_actions)               # [N, H, D]
        prefix_emb = np.stack(prefix_embs).squeeze()  # [N, emb]

        vel = analyse_velocity(v_norms, v_vecs)
        conv = analyse_convergence(x_traj)
        hid = analyse_hidden(h_norms, v_vecs)

        sc: dict | None = None
        if self_consistency:
            # Use model-space actions from x_trajectory (32-dim, before
            # output transforms) — the model's action_in_proj expects the
            # raw 32-dim representation, not the 8-dim physical actions.
            mean_action = x_traj[:, -1].mean(axis=0)  # [H, 32]
            sc = self._self_consistency(obs_dict, mean_action, consistency_probes)
            elapsed = time.time() - t0  # include consistency time

        return InternalProbeResult(
            n_samples=n_samples,
            elapsed_s=elapsed,
            velocity=vel,
            convergence=conv,
            hidden=hid,
            self_consistency=sc,
            prefix_embedding=prefix_emb.mean(axis=0),
            actions=actions,
        )

    def _self_consistency(self, obs_dict: dict, actions: np.ndarray, n_probes: int) -> dict:
        msg = dict(obs_dict)
        msg["__self_consistency__"] = True
        msg["__actions__"] = actions
        msg["__n_probes__"] = n_probes
        self._ws.send(self._packer.pack(msg))
        raw = self._ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"Server error on self-consistency: {raw[:300]}")
        return codec.unpackb(raw)

    # ----- lifecycle ---------------------------------------------------

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
