# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passthrough strategy — forwards observations unchanged."""

from __future__ import annotations

from .base import OrchestrationStrategy, SessionState


class PassthroughStrategy(OrchestrationStrategy):
    """No-op: never modifies the instruction."""

    def process(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        return obs, state
