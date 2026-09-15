# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock placement tool — returns canned success chunks without IK / sim.

Use this in CI tests that exercise the SessionState lifecycle (proxy
bypass, debug-image queue, place_tool_done event) without depending on
GraspGen, the grasp server, or a running sim.

The mock skips ENSURE_CLOSED bookkeeping and runs through a tiny phase
trajectory:

    IDLE → PERCEIVING (1 chunk) → APPROACHING (1) → FINAL_APPROACH (1)
         → MEASURING (1) → RELEASING (1) → SETTLING (1)
         → RETREATING (1) → DONE

Each chunk is a hold-position action with the appropriate gripper
command (closed before RELEASING, open after).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from vlm_orchestrator.place.tool import (
    ACTION_DIM,
    ACTION_HORIZON,
    DestinationSpec,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PlacePhase,
)

logger = logging.getLogger(__name__)


class MockPlaceToolExecutor:
    """Drop-in stand-in for :class:`PlaceToolExecutor` in tests."""

    _PHASE_SEQUENCE = (
        PlacePhase.PERCEIVING,
        PlacePhase.APPROACHING,
        PlacePhase.FINAL_APPROACH,
        PlacePhase.MEASURING,
        PlacePhase.RELEASING,
        PlacePhase.SETTLING,
        PlacePhase.RETREATING,
        PlacePhase.DONE,
    )

    def __init__(self, *args, **kwargs):
        # Accept any constructor args (drop-in compat).
        self._phase: PlacePhase = PlacePhase.IDLE
        self._cursor: int = 0
        self._destination: DestinationSpec | None = None
        self._held_object_hint: str | None = None
        self._status_message: str = ""
        self._failure_reason: str = ""
        self._place_log: dict = {}

    @property
    def phase(self) -> PlacePhase:
        return self._phase

    @property
    def is_active(self) -> bool:
        return self._phase not in (
            PlacePhase.IDLE, PlacePhase.DONE, PlacePhase.FAILED,
        )

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    def reset(self) -> None:
        self._phase = PlacePhase.IDLE
        self._cursor = 0
        self._destination = None
        self._held_object_hint = None
        self._status_message = ""
        self._failure_reason = ""
        self._place_log = {}

    def start(
        self,
        destination: DestinationSpec,
        obs: dict,
        state,
        *,
        held_object_hint: str | None = None,
        instruction: str = "",
    ) -> None:
        destination.validate()
        self._destination = destination
        self._held_object_hint = held_object_hint
        self._cursor = 0
        self._phase = PlacePhase.ENSURE_CLOSED
        self._status_message = (
            f"[mock] Place: {destination.describe()}"
        )
        self._place_log = {
            "destination": destination.describe(),
            "held_object_hint": held_object_hint,
            "started_at": time.time(),
            "mock": True,
        }

    def step(self, obs: dict, state) -> dict:
        if self._phase == PlacePhase.ENSURE_CLOSED:
            self._phase = self._PHASE_SEQUENCE[0]
            self._cursor = 0

        if self._phase in (PlacePhase.DONE, PlacePhase.FAILED):
            return {"actions": self._hold_action(closed=False)}

        actions = self._hold_action(
            closed=self._phase
            in (
                PlacePhase.PERCEIVING,
                PlacePhase.APPROACHING,
                PlacePhase.FINAL_APPROACH,
                PlacePhase.MEASURING,
            ),
        )

        # Advance one phase per chunk.
        next_idx = self._PHASE_SEQUENCE.index(self._phase) + 1
        if next_idx >= len(self._PHASE_SEQUENCE):
            self._phase = PlacePhase.DONE
        else:
            self._phase = self._PHASE_SEQUENCE[next_idx]
        self._status_message = f"[mock] {self._phase.value}"
        return {"actions": actions}

    @staticmethod
    def _hold_action(*, closed: bool) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[-1] = GRIPPER_CLOSE if closed else GRIPPER_OPEN
        return np.tile(action, (ACTION_HORIZON, 1))
