# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM backend clients.

- ``api`` — OpenAI-compatible HTTP client (works with any vision-capable
  model served over an OpenAI-compatible endpoint).  Uses the configured
  ``--vlm-base-url`` / ``--vlm-model``.  Top-level imports — ``from vlm_orchestrator.vlm import
  VLMBackend, encode_image_b64, parse_json`` — resolve here.
"""

from vlm_orchestrator.vlm.api import *  # noqa: F401, F403 — top-level re-export
