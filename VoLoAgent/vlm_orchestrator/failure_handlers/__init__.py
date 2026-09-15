# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure detection and recovery handlers.

Each handler encapsulates a detection→decision pipeline:

* :class:`VLMFailureHandler`    — periodic VLM check (status + action).
* :class:`GTFailureHandler`     — per-step ground-truth state detection.
* :class:`SignalFailureHandler`  — per-step action/EE signal detection.

All handlers implement :class:`FailureHandler` and return
:class:`HandlerResult` from :meth:`step`.
"""

from .base import FailureHandler, HandlerResult
from .gt import GTFailureHandler
from .signal import SignalFailureHandler
from .vlm import VLMFailureHandler

__all__ = [
    "FailureHandler",
    "HandlerResult",
    "GTFailureHandler",
    "SignalFailureHandler",
    "VLMFailureHandler",
]
