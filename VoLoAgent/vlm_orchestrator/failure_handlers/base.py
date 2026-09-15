# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base interface for failure detection and recovery handlers.

Every handler implements three lifecycle hooks:

* :meth:`on_episode_start`     — reset state for a new episode.
* :meth:`on_subgoal_advanced`  — reset per-subgoal state.
* :meth:`step`                 — run one detection cycle (returns
  :class:`HandlerResult` or ``None``).

The strategy layer (:class:`SubgoalBaseStrategy`) calls these hooks and
**executes** the returned action — handlers never mutate strategy state
directly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ======================================================================
# Status vocabulary (observation — what happened?)
# ======================================================================

STATUS_COMPLETE = "complete"
STATUS_FAILURE = "failure"
STATUS_IN_PROGRESS = "in_progress"

VALID_STATUSES = {STATUS_COMPLETE, STATUS_FAILURE, STATUS_IN_PROGRESS}

# ======================================================================
# Action vocabulary (decision — what to do?)
# ======================================================================

ACTION_NEXT = "next"          # advance to next subgoal
ACTION_REPLAN = "replan"      # replan remaining subtasks from current scene
ACTION_CONTINUE = "continue"  # keep working with current instruction
ACTION_GRASP = "grasp_tool"   # activate grasp tool for target object
ACTION_PLACE = "place_tool"   # activate place tool at a destination

VALID_ACTIONS = {
    ACTION_NEXT, ACTION_REPLAN, ACTION_CONTINUE, ACTION_GRASP, ACTION_PLACE,
}


# ======================================================================
# Handler result
# ======================================================================

@dataclass
class HandlerResult:
    """Result of one detection cycle.

    ``status`` is the diagnosis (what happened), ``action`` is the
    prescription (what to do).  The strategy executes the action;
    the status is logged as metadata.
    """

    status: str
    """One of ``complete``, ``failure``, ``in_progress``."""

    action: str
    """One of ``next``, ``replan``, ``continue``, ``grasp_tool``."""

    reason: str = ""
    """Human-readable explanation (for logging / debugging)."""

    confidence: float = 1.0
    """Detection confidence (always 1.0 for GT)."""

    # -- Action-specific data ------------------------------------------

    instruction: str | None = None
    """New instruction for the VLA (used by ``continue`` with refinement
    or signal-based retry)."""

    grasp_target: str | None = None
    """Target object name for ``grasp_tool`` action."""

    place_destination: str | None = None
    """Freeform destination phrase for ``place_tool`` action, suitable
    for the VLM-pointing perception layer (Molmo2 / Claude vlm_point).

    Examples:
      - "in the white bowl"            (containment)
      - "empty space next to the orange"  (spatial relation)
      - "empty space on the table"     (open surface)
      - "on the wire rack shelf"       (surface object)

    When set, takes precedence over the legacy ``place_target`` +
    ``place_relation`` pair.  See the place-tool prompt rules in
    ``failure_handlers/vlm.py`` (mirrored from
    ``strategies/tool_chain_prompts.py``).
    """

    place_target: str | None = None
    """Legacy: destination noun phrase for ``place_tool`` (e.g. "red
    bowl").  Combined with ``place_relation`` into a single freeform
    phrase if ``place_destination`` is not set.  New code should emit
    ``place_destination`` directly."""

    place_relation: str = "in"
    """Legacy: spatial relation for ``place_tool``:
    ``in`` / ``on`` / ``on_top_of``.  Used only when
    ``place_destination`` is not set."""

    place_held_object: str | None = None
    """Advisory hint for ``place_tool``: what we expect to be holding."""

    place_stack: bool = False
    """Release-orientation control for ``place_tool``.  Only honoured when
    the strategy was built with ``stack_mode_enabled=True`` (CLI
    ``--enable-stack-mode``); otherwise ignored.

    ``False`` (default) → plain top-down release (ordinary pick-and-place:
    into a bowl/bin, onto an open surface, beside another object).
    ``True`` → preserve the held object's grasp orientation at release
    (setting an object ON TOP of another: stacking a block, nesting a lid).
    """

    # -- Extra context (for logging) -----------------------------------

    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra context for logging / HITL display."""

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {self.status!r}, "
                f"expected one of {VALID_STATUSES}"
            )
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action {self.action!r}, "
                f"expected one of {VALID_ACTIONS}"
            )


# ======================================================================
# Abstract handler
# ======================================================================

class FailureHandler(ABC):
    """Base class for failure detection and recovery handlers."""

    @abstractmethod
    def on_episode_start(self, obs: dict, state: Any) -> None:
        """Reset handler state for a new episode."""

    @abstractmethod
    def on_subgoal_advanced(self, obs: dict, state: Any, idx: int) -> None:
        """Reset per-subgoal state after advancing to subgoal *idx*."""

    @abstractmethod
    def step(self, obs: dict, state: Any) -> HandlerResult | None:
        """Run one detection cycle.

        Returns :class:`HandlerResult` if detection produced a
        verdict (complete / failure), or ``None`` if nothing to
        report this step (in-progress, or not time to check yet).
        """
