# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric registry."""

from __future__ import annotations

from vlm_orchestrator.signals.base import Metric, MetricResult

# ── Registry ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[Metric]] = {}


def _auto_register():
    """Populate the registry on first access."""
    if _REGISTRY:
        return

    from vlm_orchestrator.signals.td import TrajectoryDisagreement
    from vlm_orchestrator.signals.action_variance import ActionVariance
    from vlm_orchestrator.signals.direction_clustering import DirectionClustering
    from vlm_orchestrator.signals.trajectory_modes import TrajectoryModes
    from vlm_orchestrator.signals.linguistic import Linguistic
    from vlm_orchestrator.signals.vlm_entropy import VLMEntropy

    for cls in [
        TrajectoryDisagreement,
        ActionVariance,
        DirectionClustering,
        TrajectoryModes,
        Linguistic,
        VLMEntropy,
    ]:
        _REGISTRY[cls.name] = cls


def list_metrics() -> list[str]:
    """Return names of all available metrics."""
    _auto_register()
    return sorted(_REGISTRY.keys())


def get_metric(name: str, **kwargs) -> Metric:
    """Instantiate a metric by name.

    Parameters
    ----------
    name : str
        One of :func:`list_metrics`.
    **kwargs
        Passed to the metric constructor (e.g. ``n_samples=50``).

    Examples
    --------
    >>> m = get_metric("td", n_samples=30, host="127.0.0.1", port=8000)
    >>> m = get_metric("linguistic")
    """
    _auto_register()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list_metrics()}"
        )
    return _REGISTRY[name](**kwargs)
