# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision-perception clients and model loaders.

- ``molmo`` — Molmo2 pointing client (talks to a local OpenAI-compatible HTTP
  server, e.g. ``vlm_orchestrator/utils/molmo2_hf_server.py``).
- ``gdino`` — GroundingDINO open-vocab detector (loaded in-process by the
  grasp server).
- ``sam2`` — SAM2 point-prompt segmentation (loaded in-process by the grasp
  server).
- ``sam3`` — SAM3 phrase-prompt detection + segmentation (loaded in-process
  by the grasp server).
- ``gt_segmentation`` — Ground-truth segmentation providers (per simulator).
  Used by the grasp / place tools when ``--{grasp,place}-seg-mode gt_sim``.
"""

__all__ = ["molmo", "gdino", "sam2", "sam3", "gt_segmentation"]
