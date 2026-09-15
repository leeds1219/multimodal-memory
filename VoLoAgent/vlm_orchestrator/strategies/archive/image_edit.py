# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene image editing for visual policy guidance.

Provides functions to highlight target objects and dim distractors
in 224x224 policy input images based on VLM-detected bounding boxes.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image

from vlm_orchestrator.vlm import encode_image_b64, parse_json

logger = logging.getLogger(__name__)


# ======================================================================
# Bounding box dataclass
# ======================================================================

class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = max(0, int(x1))
        self.y1 = max(0, int(y1))
        self.x2 = int(x2)
        self.y2 = int(y2)

    def scale(self, src_h: int, src_w: int, dst_h: int, dst_w: int) -> "BBox":
        """Scale bounding box from (src_h, src_w) image to (dst_h, dst_w).

        Simple linear scaling — does NOT account for letterbox padding.
        Use :meth:`scale_with_pad` for images resized via ``resize_with_pad``.
        """
        sx = dst_w / src_w
        sy = dst_h / src_h
        return BBox(
            x1=int(self.x1 * sx),
            y1=int(self.y1 * sy),
            x2=int(self.x2 * sx),
            y2=int(self.y2 * sy),
        )

    def scale_with_pad(
        self, src_h: int, src_w: int, dst_h: int, dst_w: int
    ) -> "BBox":
        """Scale bbox from raw (src_h, src_w) to a letterboxed (dst_h, dst_w) image.

        Matches the ``resize_with_pad`` transform used by the eval client:
        the raw image is uniformly scaled to fit within (dst_h, dst_w),
        then centered with black padding.

        Returns a new BBox in the padded destination coordinate system.
        """
        ratio = max(src_w / dst_w, src_h / dst_h)
        resized_w = int(src_w / ratio)
        resized_h = int(src_h / ratio)
        pad_x = max(0, int((dst_w - resized_w) / 2))
        pad_y = max(0, int((dst_h - resized_h) / 2))
        s = 1.0 / ratio
        return BBox(
            x1=int(self.x1 * s + pad_x),
            y1=int(self.y1 * s + pad_y),
            x2=int(self.x2 * s + pad_x),
            y2=int(self.y2 * s + pad_y),
        )

    def clamp(self, h: int, w: int) -> "BBox":
        """Clamp coordinates to image dimensions."""
        return BBox(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(0, min(self.x2, w)),
            y2=max(0, min(self.y2, h)),
        )

    def pad(self, px: int, h: int, w: int) -> "BBox":
        """Expand the box by `px` pixels on each side, clamped to image dims."""
        return BBox(
            x1=max(0, self.x1 - px),
            y1=max(0, self.y1 - px),
            x2=min(w, self.x2 + px),
            y2=min(h, self.y2 + px),
        )

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    def __repr__(self) -> str:
        return f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


# ======================================================================
# Image editing functions
# ======================================================================

def edit_highlight(image: np.ndarray, bbox: BBox, border_px: int = 3,
                   color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw a bright colored border around the target bounding box.

    Args:
        image: (H, W, 3) uint8 RGB array.
        bbox: Target bounding box in image coordinates.
        border_px: Border thickness in pixels.
        color: RGB tuple for the border color.

    Returns:
        Edited image (copy).
    """
    img = image.copy()
    h, w = img.shape[:2]
    bb = bbox.clamp(h, w)

    # Top border
    y_top = max(0, bb.y1 - border_px)
    img[y_top:bb.y1, bb.x1:bb.x2] = color
    # Bottom border
    y_bot = min(h, bb.y2 + border_px)
    img[bb.y2:y_bot, bb.x1:bb.x2] = color
    # Left border
    x_left = max(0, bb.x1 - border_px)
    img[y_top:y_bot, x_left:bb.x1] = color
    # Right border
    x_right = min(w, bb.x2 + border_px)
    img[y_top:y_bot, bb.x2:x_right] = color

    return img


def edit_dim(image: np.ndarray, bbox: BBox,
             dim_factor: float = 0.3) -> np.ndarray:
    """Darken everything outside the target bounding box.

    Args:
        image: (H, W, 3) uint8 RGB array.
        bbox: Target bounding box in image coordinates.
        dim_factor: Multiply outside pixel values by this (0.3 = 30% brightness).

    Returns:
        Edited image (copy).
    """
    h, w = img_shape = image.shape[:2]
    bb = bbox.clamp(h, w)

    # Dim the whole image, then paste the original region back
    img = (image.astype(np.float32) * dim_factor).clip(0, 255).astype(np.uint8)
    img[bb.y1:bb.y2, bb.x1:bb.x2] = image[bb.y1:bb.y2, bb.x1:bb.x2]

    return img


def edit_both(image: np.ndarray, bbox: BBox,
              dim_factor: float = 0.3, border_px: int = 3,
              color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Dim distractors AND highlight the target with a border.

    Args:
        image: (H, W, 3) uint8 RGB array.
        bbox: Target bounding box in image coordinates.
        dim_factor: Brightness factor for outside region.
        border_px: Border thickness.
        color: RGB border color.

    Returns:
        Edited image (copy).
    """
    img = edit_dim(image, bbox, dim_factor=dim_factor)
    img = edit_highlight(img, bbox, border_px=border_px, color=color)
    return img


def apply_edit(image: np.ndarray, bbox: BBox | None,
               mode: str) -> np.ndarray:
    """Apply the specified edit mode.

    Args:
        image: (H, W, 3) uint8 RGB array.
        bbox: Target bounding box, or None if object not visible.
        mode: One of "none", "highlight", "dim", "both".

    Returns:
        Edited image (or original copy if mode is "none" or bbox is None).
    """
    if mode == "none" or bbox is None:
        return image.copy()
    if mode == "highlight":
        return edit_highlight(image, bbox)
    if mode == "dim":
        return edit_dim(image, bbox)
    if mode == "both":
        return edit_both(image, bbox)
    raise ValueError(f"Unknown edit mode: {mode!r}")


# ======================================================================
# VLM bounding box querying
# ======================================================================

BBOX_QUERY_PROMPT = """\
Look at this robot workspace image. Identify the green block (a small \
solid-colored wooden block, NOT a Rubik's cube and NOT the green \
dinosaur/lizard figurine). 
Return the bounding box as JSON: {{"x1": <left>, "y1": <top>, "x2": \
<right>, "y2": <bottom>}} in pixel coordinates relative to the image \
dimensions.
If the green block is not visible, return {{"visible": false}}.\
"""

WRIST_BBOX_QUERY_PROMPT = """\
Look at this robot wrist camera image. Identify the green block (a \
small solid-colored wooden block, NOT a Rubik's cube and NOT the green \
dinosaur/lizard figurine).
Return the bounding box as JSON: {{"x1": <left>, "y1": <top>, "x2": \
<right>, "y2": <bottom>}} in pixel coordinates relative to the image \
dimensions.
If the green block is not visible in this view, return {{"visible": false}}.\
"""


class BBoxQueryClient:
    """Queries a VLM for object bounding boxes in scene images."""

    def __init__(
        self,
        model: str = "YOUR_VLM_MODEL",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 300,
    ):
        import openai
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = openai.OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query_bbox(
        self,
        image: np.ndarray,
        prompt: str = BBOX_QUERY_PROMPT,
    ) -> tuple[BBox | None, dict]:
        """Query the VLM for a bounding box.

        Args:
            image: RGB uint8 array (H, W, 3).
            prompt: The query prompt to use.

        Returns:
            (bbox, raw_response_dict): bbox is None if object not visible.
        """
        image_b64 = encode_image_b64(image)
        h, w = image.shape[:2]

        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]

        t0 = time.time()
        try:
            from vlm_orchestrator.vlm import chat_create
            response = chat_create(
                self.client,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": "You are a precise visual object detector. Output only JSON."},
                    {"role": "user", "content": user_content},
                ],
            )
            raw_text = response.choices[0].message.content.strip()
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            logger.warning(f"VLM bbox query failed ({elapsed:.1f}s): {e}")
            return None, {"error": str(e), "elapsed_s": elapsed}

        logger.debug(f"VLM bbox response ({elapsed:.1f}s): {raw_text}")

        try:
            data = parse_json(raw_text)
        except ValueError as e:
            logger.warning(f"Cannot parse VLM bbox response: {e}")
            return None, {"raw": raw_text, "error": str(e), "elapsed_s": elapsed}

        response_info = {
            "raw": raw_text,
            "parsed": data,
            "elapsed_s": elapsed,
            "image_dims": {"h": h, "w": w},
        }

        # Check if object is not visible
        if data.get("visible") is False or data.get("visible") == "false":
            logger.info(f"VLM says object not visible ({elapsed:.1f}s)")
            return None, response_info

        # Extract bounding box
        try:
            bbox = BBox(
                x1=int(data["x1"]),
                y1=int(data["y1"]),
                x2=int(data["x2"]),
                y2=int(data["y2"]),
            )
            # Sanity check
            if bbox.area() < 4:
                logger.warning(f"VLM returned tiny bbox {bbox}, treating as not visible")
                return None, response_info
            bbox = bbox.clamp(h, w)
            logger.info(f"VLM bbox: {bbox} on {w}x{h} image ({elapsed:.1f}s)")
            return bbox, response_info
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Cannot extract bbox from VLM response: {e}")
            return None, response_info


def save_debug_image(image: np.ndarray, path: str) -> None:
    """Save an image array to disk for visual inspection."""
    pil = Image.fromarray(image.astype(np.uint8))
    pil.save(path)
    logger.debug(f"Saved debug image: {path}")
