# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth failure handler (thin wrapper).

Wraps the existing GT failure detection and recovery logic in
:class:`SubgoalBaseStrategy` behind the :class:`FailureHandler`
interface.  The actual detection and recovery code remains in
``subgoal_base.py`` for now — this handler delegates to it.

Future cleanup: move the GT detection/recovery logic into this
module so that ``subgoal_base.py`` only calls
``self._failure_handler.step()``.
"""

from __future__ import annotations

import logging
from typing import Any

from vlm_orchestrator.failure_handlers.gt_detector import (
    GTFailureDetector,
    GTFailureType,
)

from .base import (
    ACTION_CONTINUE,
    ACTION_GRASP,
    ACTION_NEXT,
    ACTION_REPLAN,
    FailureHandler,
    HandlerResult,
    STATUS_COMPLETE,
    STATUS_FAILURE,
    STATUS_IN_PROGRESS,
)

logger = logging.getLogger(__name__)


class GTFailureHandler(FailureHandler):
    """GT-based failure detection handler.

    This is a **thin wrapper** — the actual GT detection and recovery
    logic still lives in ``subgoal_base.py`` (``_tick_gt_detection``,
    ``_handle_gt_failure``, etc.).  This handler exists so that the
    handler interface is consistent and the code can be migrated
    incrementally.

    For now, it holds the :class:`GTFailureDetector` instance and
    exposes it so ``subgoal_base.py`` can continue using the legacy
    code path.

    Parameters
    ----------
    enabled_failure_types:
        Subset of GT failure types to monitor (or ``None`` for all).
    """

    def __init__(
        self,
        enabled_failure_types: set[str] | None = None,
    ):
        self.detector = GTFailureDetector(
            enabled_failure_types=enabled_failure_types,
        )
        self._all_done: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(self, obs: dict, state: Any) -> None:
        self.detector.reset()
        self._all_done = False

    def on_subgoal_advanced(self, obs: dict, state: Any, idx: int) -> None:
        # GT detector is reconfigured by _advance_subgoal in
        # subgoal_base.py (needs gt_state, scene_objects, conditions).
        pass

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def step(self, obs: dict, state: Any) -> HandlerResult | None:
        """GT detection runs every step via the legacy path.

        Returns ``None`` — the legacy ``_tick_gt_detection()`` in
        ``subgoal_base.py`` handles everything directly.  This stub
        exists for interface consistency.

        TODO: migrate ``_tick_gt_detection`` logic here.
        """
        # Legacy path handles GT detection in subgoal_base.py
        return None
