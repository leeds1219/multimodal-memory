# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 point-prompt segmentation, loaded in-process by the grasp server.

Wraps ``sam2.sam2_image_predictor.SAM2ImagePredictor`` so the grasp server
keeps a single predictor instance for the lifetime of the process.

Usage:
    from vlm_orchestrator.perception import sam2

    sam2.load("facebook/sam2.1-hiera-small")
    mask, iou = sam2.segment_from_point(image, point_x, point_y, box=box)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_predictor = None


def load(model_id: str = "facebook/sam2.1-hiera-small") -> None:
    """Load SAM2 predictor (idempotent — re-calling replaces the global)."""
    global _predictor

    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SAM2 model '{model_id}' on {device}")
    t0 = time.time()
    _predictor = SAM2ImagePredictor.from_pretrained(
        model_id, device=device, mask_threshold=0.15,
    )
    logger.info(f"SAM2 loaded in {time.time() - t0:.1f}s")


def is_loaded() -> bool:
    return _predictor is not None


def segment_from_point(
    image: np.ndarray,
    point_x: float,
    point_y: float,
    box: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Run SAM2 segmentation.

    Args:
        image: ``(H, W, 3)`` uint8 RGB.
        point_x: normalised x ∈ [0, 1].
        point_y: normalised y ∈ [0, 1].
        box: optional ``(4,)`` array ``[x1, y1, x2, y2]`` in **pixel** coords.

    Returns:
        ``(mask_HxW_bool, iou_score)``.
    """
    import torch

    assert _predictor is not None, "SAM2 not loaded — call sam2.load() first"

    h, w = image.shape[:2]
    px = int(point_x * w)
    py = int(point_y * h)

    with torch.inference_mode():
        _predictor.set_image(image)
        masks, scores, _ = _predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            box=box,
            multimask_output=True,
        )

    best = int(np.argmax(scores))
    mask = masks[best].astype(bool)
    iou = float(scores[best])
    return mask, iou
