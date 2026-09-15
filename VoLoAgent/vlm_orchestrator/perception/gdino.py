# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GroundingDINO-based object detector for scene editing.

Runs text-prompted object detection locally on GPU, providing much more
accurate bounding boxes than asking a VLM for pixel coordinates.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class GroundingDINODetector:
    """Text-prompted object detector using GroundingDINO (runs on local GPU).

    Compared to VLM bbox queries:
      - Much faster (~50ms vs ~5s)
      - Much more spatially accurate
      - No API calls needed
      - Uses ~0.65 GB GPU memory
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        text_threshold: float = 0.15,
        box_threshold: float = 0.25,
    ):
        self.model_id = model_id
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        logger.info(f"Loading GroundingDINO model: {self.model_id}")
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        elapsed = time.time() - t0
        logger.info(f"GroundingDINO loaded in {elapsed:.1f}s")

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Run text-prompted object detection.

        Args:
            image: RGB uint8 array (H, W, 3).
            text_prompt: Text description of the target object.
                         Must end with a period, e.g. "green block."
            score_threshold: Minimum confidence score. Defaults to
                ``self.box_threshold``.

        Returns:
            List of detections, sorted by score descending. Each is a dict::

                {
                    "score": float,
                    "label": str,
                    "box": [x1, y1, x2, y2],  # pixel coords
                }
        """
        import torch

        self._ensure_loaded()
        threshold = score_threshold or self.box_threshold

        # Ensure prompt ends with period (GroundingDINO convention)
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."

        pil_image = Image.fromarray(image.astype(np.uint8))
        h, w = image.shape[:2]

        inputs = self._processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self._model(**inputs)
        elapsed = time.time() - t0

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )

        detections = []
        for score, label, box in zip(
            results[0]["scores"],
            results[0]["labels"],
            results[0]["boxes"],
        ):
            s = score.item()
            if s < threshold:
                continue
            b = box.tolist()
            detections.append({
                "score": s,
                "label": label,
                "box": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
            })

        # Sort by score descending
        detections.sort(key=lambda d: d["score"], reverse=True)

        logger.debug(
            f"GroundingDINO: {len(detections)} detections for '{text_prompt}' "
            f"on {w}x{h} image ({elapsed*1000:.0f}ms)"
        )
        return detections

    def detect_best(
        self,
        image: np.ndarray,
        text_prompt: str,
        score_threshold: float | None = None,
        color_filter: str | None = None,
    ) -> dict | None:
        """Return the single best detection, optionally filtered by color.

        Args:
            image: RGB uint8 array (H, W, 3).
            text_prompt: Target description.
            score_threshold: Minimum confidence.
            color_filter: If set (e.g. "green"), picks the detection whose
                box region has the highest dominance of that color channel.

        Returns:
            Best detection dict, or None if nothing found.
        """
        detections = self.detect(image, text_prompt, score_threshold)
        if not detections:
            return None

        if color_filter and len(detections) > 1:
            return self._pick_by_color(image, detections, color_filter)

        return detections[0]

    @staticmethod
    def _pick_by_color(
        image: np.ndarray,
        detections: list[dict],
        color: str,
    ) -> dict:
        """Among detections, pick the one whose box region best matches the
        named color.

        Supports: "red", "green", "blue", "yellow".
        """
        color_idx = {"red": 0, "green": 1, "blue": 2}.get(color.lower())

        best = detections[0]
        best_score = -999.0

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            # Clamp and extract region
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            region = image[y1:y2, x1:x2].astype(np.float32)
            mean_rgb = region.mean(axis=(0, 1))

            if color.lower() == "yellow":
                # Yellow = high R + high G, low B
                score = (mean_rgb[0] + mean_rgb[1]) / 2 - mean_rgb[2]
            elif color_idx is not None:
                # Dominance of the target channel over others
                others = [mean_rgb[i] for i in range(3) if i != color_idx]
                score = mean_rgb[color_idx] - max(others)
            else:
                # Unknown color, just use detection score
                score = det["score"]

            if score > best_score:
                best_score = score
                best = det

        logger.debug(
            f"Color filter '{color}': picked box {best['box']} "
            f"(color_score={best_score:.1f})"
        )
        return best
