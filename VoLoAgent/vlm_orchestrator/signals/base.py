# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for instruction-quality metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MetricResult:
    """Standard output from any instruction-quality metric.

    Every metric must populate *score* (lower = clearer instruction)
    and *name*.  Everything else is optional detail.
    """

    name: str
    score: float
    detail: dict[str, Any] = field(default_factory=dict)

    # Optional: populated when comparing two instructions
    score_original: float | None = None
    score_rewritten: float | None = None
    improvement: float | None = None  # original - rewritten (positive = rewrite helped)

    def to_dict(self) -> dict:
        d = {"metric": self.name, "score": round(self.score, 6)}
        if self.score_original is not None:
            d["score_original"] = round(self.score_original, 6)
        if self.score_rewritten is not None:
            d["score_rewritten"] = round(self.score_rewritten, 6)
        if self.improvement is not None:
            d["improvement"] = round(self.improvement, 6)
        if self.detail:
            d["detail"] = self.detail
        return d


class Metric(ABC):
    """Base class for all instruction-quality metrics.

    Subclasses must implement :meth:`compute`.  They may optionally
    implement :meth:`setup` (called once) and :meth:`teardown`.
    """

    name: str = "base"

    # What resources does this metric need?
    needs_policy: bool = False  # requires a VLA policy server
    needs_vlm: bool = False  # requires a VLM API
    needs_obs: bool = False  # requires an observation dict (images + state)

    def setup(self, **kwargs) -> None:
        """One-time initialisation (e.g. connect to policy server)."""
        pass

    def teardown(self) -> None:
        """Release resources."""
        pass

    @abstractmethod
    def compute(
        self,
        instruction: str,
        *,
        obs: dict | None = None,
        task: str | None = None,
        image: np.ndarray | None = None,
        **kwargs,
    ) -> MetricResult:
        """Score a single instruction.

        Parameters
        ----------
        instruction : str
            The instruction to evaluate.
        obs : dict, optional
            Full policy observation dict (keys like ``observation/exterior_image_1_left``).
            Required if :attr:`needs_obs` or :attr:`needs_policy` is True.
        task : str, optional
            Task class name (e.g. ``"BananaInBowlTask"``).
        image : np.ndarray, optional
            Scene image for VLM-based metrics. If not provided, extracted from *obs*.
        """
        ...

    def compare(
        self,
        original: str,
        rewritten: str,
        *,
        obs: dict | None = None,
        task: str | None = None,
        image: np.ndarray | None = None,
        **kwargs,
    ) -> MetricResult:
        """Score both instructions and return improvement.

        Default implementation calls :meth:`compute` twice.  Subclasses
        may override for efficiency (e.g. reuse the same policy connection).
        """
        r_orig = self.compute(original, obs=obs, task=task, image=image, **kwargs)
        # Build obs with rewritten prompt for the rewritten call
        rw_obs = None
        if obs is not None:
            rw_obs = dict(obs)
            rw_obs["prompt"] = rewritten
        r_rw = self.compute(rewritten, obs=rw_obs, task=task, image=image, **kwargs)

        return MetricResult(
            name=self.name,
            score=r_rw.score,
            score_original=r_orig.score,
            score_rewritten=r_rw.score,
            improvement=r_orig.score - r_rw.score,
            detail={
                "original_detail": r_orig.detail,
                "rewritten_detail": r_rw.detail,
            },
        )

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *args):
        self.teardown()
