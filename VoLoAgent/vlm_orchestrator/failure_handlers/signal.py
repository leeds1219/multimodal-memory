# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signal-based failure handler (thin wrapper).

Wraps the existing signal-based failure detection and recovery logic in
:class:`SubgoalBaseStrategy` behind the :class:`FailureHandler`
interface.  The actual detection and recovery code remains in
``subgoal_base.py`` for now — this handler delegates to it.

Future cleanup: move the signal detection/recovery logic into this
module so that ``subgoal_base.py`` only calls
``self._failure_handler.step()``.
"""

from __future__ import annotations

import logging
from typing import Any

from vlm_orchestrator.failure_handlers.signal_detector import CombinedDetector

from .base import (
    ACTION_CONTINUE,
    FailureHandler,
    HandlerResult,
    STATUS_IN_PROGRESS,
)

logger = logging.getLogger(__name__)


class SignalFailureHandler(FailureHandler):
    """Signal-based failure detection handler.

    This is a **thin wrapper** — the actual signal accumulation,
    classification, and recovery logic still lives in
    ``subgoal_base.py`` (``_handle_failure``,
    ``_recover_with_instruction``, etc.).

    For now, it holds the :class:`CombinedDetector` instance and
    exposes it so ``subgoal_base.py`` can continue using the legacy
    code path.

    Parameters
    ----------
    mode:
        Signal detection mode (``signal_primary``, ``union_failure``,
        or ``intersect_video``).
    env_mode:
        Simulation environment for signal thresholds.
    """

    def __init__(
        self,
        mode: str = "signal_primary",
        env_mode: str = "robolab",
    ):
        from vlm_orchestrator.failure_handlers.signal_detector import get_signal_config
        signal_config = get_signal_config(env_mode)
        self.detector = CombinedDetector(
            mode=mode, window_size=80,
            signal_config=signal_config,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(self, obs: dict, state: Any) -> None:
        self.detector.reset()

    def on_subgoal_advanced(self, obs: dict, state: Any, idx: int) -> None:
        self.detector.reset()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def step(self, obs: dict, state: Any) -> HandlerResult | None:
        """Signal detection runs every step via the legacy path.

        Returns ``None`` — the legacy signal accumulation and
        classification in ``subgoal_base.py`` handles everything
        directly.  This stub exists for interface consistency.

        TODO: migrate signal detection logic here.
        """
        # Legacy path handles signal detection in subgoal_base.py
        return None
