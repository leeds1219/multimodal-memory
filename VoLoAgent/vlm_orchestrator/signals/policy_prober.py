# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive compare-and-pick rewrite gating for the VLM orchestrator.

Core idea:
  VLA policies with stochastic flow-matching (e.g. Pi0.5) produce different action
  chunks for the same observation on every forward pass.  When the instruction is
  vague, the policy's uncertainty is higher.

  Previous approach (fixed-threshold raw variance) conflates "stochastic but
  confident" (high variance, succeeds) with "genuinely confused" (high variance,
  fails).  Trajectory disagreement fixes this by measuring directional consensus
  rather than raw spread.

  The **compare-and-pick** strategy eliminates fixed thresholds entirely:
    1. Probe the policy N times with the original instruction → traj_disagreement.
    2. Call VLM to rewrite the instruction.
    3. Probe the policy N times with the rewritten instruction → traj_disagreement.
    4. Pick whichever instruction has LOWER traj_disagreement (more consensus).
    5. If the rewrite didn't improve, keep the original.

  Trajectory disagreement (traj_disagreement_8):
    - Truncate action chunks to first 8 steps (open_loop_horizon for Pi0.5).
    - Flatten to (N, 64), standardise per-feature, compute pairwise cosine sims.
    - traj_disagreement = 1 − mean(upper-triangle cosine similarities).
    - 3.3× separation on the key false-positive case vs 2.0× for raw variance.

  Reference: ~/robolab/analysis/vagueness_metrics/clustering/analyze_clusters.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from vlm_orchestrator.utils import codec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Probe result container
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Outcome of probing the policy N times with a single observation."""

    instruction: str
    n_samples: int
    elapsed_s: float

    # Variance metrics (matching metric1_behavioral.py conventions)
    mean_action_var: float      # mean variance across all dims & timesteps
    first_step_var: float       # variance of just the first predicted step
    action_spread: float        # mean L2 distance from centroid
    gripper_agreement: float    # fraction agreeing on gripper open/close

    # Trajectory disagreement (compare-and-pick metric)
    traj_disagreement: float = 0.0  # 0=full consensus, 1=max disagreement

    # Raw data for downstream analysis
    all_actions: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialisable summary (excludes bulky ``all_actions``)."""
        return {
            "instruction": self.instruction,
            "n_samples": self.n_samples,
            "elapsed_s": round(self.elapsed_s, 2),
            "mean_action_var": self.mean_action_var,
            "first_step_var": self.first_step_var,
            "action_spread": self.action_spread,
            "gripper_agreement": self.gripper_agreement,
            "traj_disagreement": round(self.traj_disagreement, 6),
        }


# ---------------------------------------------------------------------------
#  Trajectory disagreement (pure numpy, no I/O)
# ---------------------------------------------------------------------------

def compute_traj_disagreement(all_actions: np.ndarray, horizon: int = 8) -> float:
    """Compute trajectory disagreement from an (N, T, action_dim) array.

    Measures whether the N sampled action chunks agree on *where the arm goes*
    (goal direction), not just whether raw values are similar.

    Steps:
      1. Truncate to first *horizon* steps, joints only (dims 0-6, drop gripper).
      2. Compute step-to-step deltas → cumulative sum → endpoint direction.
      3. L2-normalise each endpoint to a unit direction vector.
      4. Pairwise cosine similarities of unit directions.
      5. traj_consensus = mean of upper triangle.
      6. Return traj_disagreement = 1 − traj_consensus.

    A value near 0 means all samples head in the same direction (confident).
    A value near 0.5+ means samples disagree on goal (confused).

    Reference: ~/robolab/analysis/vagueness_metrics/clustering/analyze_clusters.py
    """
    n = all_actions.shape[0]
    if n < 2:
        return 0.0

    # 1. Truncate to first `horizon` steps, joints only (exclude gripper)
    trunc = all_actions[:, :horizon, :7]         # (N, horizon, 7)

    # 2. Step-to-step deltas → cumulative → endpoint direction
    deltas = np.diff(trunc, axis=1)              # (N, horizon-1, 7)
    cum = np.cumsum(deltas, axis=1)              # (N, horizon-1, 7)
    endpoint = cum[:, -1, :]                     # (N, 7) cumulative delta

    # 3. L2-normalise to unit direction vectors
    norms = np.linalg.norm(endpoint, axis=1, keepdims=True)

    # If a sample has near-zero movement, it contributes noise.
    # Filter out near-stationary samples.
    valid = (norms.squeeze() > 1e-8)
    if valid.sum() < 2:
        return 0.0

    endpoint = endpoint[valid]
    norms = norms[valid]
    unit = endpoint / norms                      # (N', 7) unit vectors

    # 4. Pairwise cosine similarities
    cos_sim = unit @ unit.T                      # (N', N')

    # 5. Mean of upper triangle (exclude diagonal)
    n_valid = unit.shape[0]
    triu_indices = np.triu_indices(n_valid, k=1)
    traj_consensus = float(cos_sim[triu_indices].mean())

    # 6. Disagreement
    return 1.0 - traj_consensus


# ---------------------------------------------------------------------------
#  Compare-and-pick decision (no threshold needed)
# ---------------------------------------------------------------------------

@dataclass
class CompareResult:
    """Outcome of comparing original vs rewritten probes."""

    winner: str                    # "original" or "rewritten"
    original: ProbeResult
    rewritten: ProbeResult
    winning_instruction: str
    improvement: float             # positive means rewrite was better

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "winning_instruction": self.winning_instruction,
            "improvement": round(self.improvement, 6),
            "original": self.original.to_dict(),
            "rewritten": self.rewritten.to_dict(),
        }


def compare_and_pick(
    original_result: ProbeResult,
    rewritten_result: ProbeResult,
) -> CompareResult:
    """Pick the instruction with lower trajectory disagreement.

    Returns a :class:`CompareResult` with the winner. If the rewrite did not
    strictly improve (lower disagreement), the original wins — we prefer the
    user's wording when it's a wash.
    """
    orig_td = original_result.traj_disagreement
    rewr_td = rewritten_result.traj_disagreement
    improvement = orig_td - rewr_td  # positive = rewrite is better

    if improvement > 0:
        winner = "rewritten"
        winning_instruction = rewritten_result.instruction
    else:
        winner = "original"
        winning_instruction = original_result.instruction

    return CompareResult(
        winner=winner,
        original=original_result,
        rewritten=rewritten_result,
        winning_instruction=winning_instruction,
        improvement=improvement,
    )


# ---------------------------------------------------------------------------
#  Variance computation (pure numpy, no I/O) — kept for backward compat
# ---------------------------------------------------------------------------

def compute_variance_metrics(all_actions: np.ndarray) -> dict:
    """Compute variance metrics from an (N, horizon, action_dim) array.

    Returns dict with keys: mean_action_var, first_step_var, action_spread,
    gripper_agreement.
    """
    n = all_actions.shape[0]
    if n < 2:
        return {
            "mean_action_var": 0.0,
            "first_step_var": 0.0,
            "action_spread": 0.0,
            "gripper_agreement": 1.0,
        }

    # Mean variance across all dimensions and timesteps
    var_per_dim = np.var(all_actions, axis=0)          # (horizon, dim)
    mean_action_var = float(np.mean(var_per_dim))

    # Variance of just the first step
    first_step_var = float(np.mean(np.var(all_actions[:, 0, :], axis=0)))

    # Mean L2 spread (distance from centroid)
    centroid = np.mean(all_actions, axis=0)             # (horizon, dim)
    dists = np.sqrt(np.sum((all_actions - centroid[None]) ** 2, axis=(1, 2)))
    action_spread = float(np.mean(dists))

    # Gripper agreement: last dim is gripper action
    gripper_actions = all_actions[:, 0, -1]
    gripper_binary = (gripper_actions > 0.5).astype(float)
    gripper_agreement = float(max(np.mean(gripper_binary), 1 - np.mean(gripper_binary)))

    return {
        "mean_action_var": mean_action_var,
        "first_step_var": first_step_var,
        "action_spread": action_spread,
        "gripper_agreement": gripper_agreement,
    }


# ---------------------------------------------------------------------------
#  Policy prober (websocket I/O)
# ---------------------------------------------------------------------------

class PolicyProber:
    """Opens a temporary connection to the VLA policy server and probes it.

    Designed for two usage patterns:
      1. **Persistent** – create once, call :meth:`probe` many times (pipeline).
      2. **One-shot** – call :func:`probe_once` which connects, probes, disconnects.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        import websockets.sync.client as ws_client

        self._host = host
        self._port = port
        uri = f"ws://{host}:{port}"
        logger.info(f"PolicyProber connecting to {uri}")
        self._ws = ws_client.connect(uri, compression=None, max_size=None)
        self._packer = codec.Packer()

        # Consume mandatory server metadata
        meta_raw = self._ws.recv()
        self._metadata = codec.unpackb(meta_raw)
        logger.info(f"PolicyProber connected – server metadata: {self._metadata}")

    # ------------------------------------------------------------------ #
    #  Core probe
    # ------------------------------------------------------------------ #

    def probe(self, obs_dict: dict, n_samples: int = 20) -> ProbeResult:
        """Send *obs_dict* to the policy *n_samples* times and return metrics."""
        instruction = obs_dict.get("prompt", "<unknown>")
        logger.info(
            f"Probing policy {n_samples}× for instruction: {instruction!r}"
        )

        packed = self._packer.pack(obs_dict)
        all_actions: list[np.ndarray] = []
        t0 = time.time()

        for i in range(n_samples):
            self._ws.send(packed)
            resp_raw = self._ws.recv()
            if isinstance(resp_raw, str):
                raise RuntimeError(f"Policy server error on probe {i}: {resp_raw[:300]}")
            resp = codec.unpackb(resp_raw)
            actions = resp["actions"]          # (horizon, action_dim)
            all_actions.append(actions)

        elapsed = time.time() - t0
        stacked = np.array(all_actions)        # (N, horizon, dim)
        metrics = compute_variance_metrics(stacked)
        traj_disag = compute_traj_disagreement(stacked)

        result = ProbeResult(
            instruction=instruction,
            n_samples=n_samples,
            elapsed_s=elapsed,
            all_actions=stacked,
            traj_disagreement=traj_disag,
            **metrics,
        )
        logger.info(
            f"Probe complete in {elapsed:.1f}s — "
            f"mean_var={metrics['mean_action_var']:.6f}, "
            f"traj_disagreement={traj_disag:.4f}, "
            f"spread={metrics['action_spread']:.4f}"
        )
        return result

    # ------------------------------------------------------------------ #

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
#  Convenience: one-shot probe (connect → probe → disconnect)
# ---------------------------------------------------------------------------

def probe_once(
    obs_dict: dict,
    n_samples: int = 20,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ProbeResult:
    """Open a temporary connection, probe, close. For use inside the proxy."""
    with PolicyProber(host, port) as prober:
        return prober.probe(obs_dict, n_samples)
