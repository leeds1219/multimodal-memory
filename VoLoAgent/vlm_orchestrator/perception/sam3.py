# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3-based object detector+segmenter for the grasp pipeline.

SAM3 (Segment Anything Model 3) from Meta unifies detection and segmentation
in a single model with native text-prompt support.  Unlike the GDino+SAM2
two-stage pipeline, SAM3 performs text-prompted detection AND per-instance
segmentation in one forward pass.

Model:  ``facebook/sam3``  (gated — requires HuggingFace approval)
API:    Native ``sam3`` package (``build_sam3_image_model``).

Input:  RGB image (H, W, 3) uint8 + text prompt (str)
Output: list of dicts, each with ``score``, ``box`` [x1,y1,x2,y2],
        ``mask`` (H, W) bool

Usage::

    detector = SAM3Detector()
    results = detector.detect_and_segment(image, "green wooden block")
    # results = [{"score": 0.92, "box": [x1,y1,x2,y2], "mask": ndarray, "label": "..."}]

    # Or use the drop-in replacements for the GDino+SAM2 interfaces:
    detections = detector.detect(image, "green block.")      # GDino-compatible
    mask, iou = detector.segment_at_box(image, "green block", bbox=(x1,y1,x2,y2))

Runs on the grasp server (``grasp_server.py``) alongside GraspGen.
Requires ``sam3>=0.1.0`` package (uses native API, no transformers dependency).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# SAM3 image preprocessing constants (from model_builder / training config)
_SAM3_IMG_SIZE = 1008
_SAM3_MEAN = (0.5, 0.5, 0.5)
_SAM3_STD = (0.5, 0.5, 0.5)


def _preprocess_image(image: np.ndarray, device: str) -> tuple[torch.Tensor, int, int]:
    """Convert RGB uint8 (H,W,3) → normalised (1,3,1008,1008) tensor.

    Resizes with letterbox padding (aspect-ratio preserved, padded with 0).
    Returns ``(tensor, orig_h, orig_w)``.
    """
    h, w = image.shape[:2]

    # HWC uint8 → CHW float [0,1]
    img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # Resize longest side to _SAM3_IMG_SIZE, pad shorter side
    scale = _SAM3_IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    img_t = F.interpolate(
        img_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False,
    ).squeeze(0)  # (3, new_h, new_w)

    # Pad to (3, 1008, 1008)
    pad_h = _SAM3_IMG_SIZE - new_h
    pad_w = _SAM3_IMG_SIZE - new_w
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), value=0.0)  # (3, 1008, 1008)

    # Normalize: (x - mean) / std
    mean = torch.tensor(_SAM3_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_SAM3_STD, dtype=torch.float32).view(3, 1, 1)
    img_t = (img_t - mean) / std

    return img_t.unsqueeze(0).to(device), h, w  # (1, 3, 1008, 1008)


def _build_batched_datapoint(
    img_tensor: torch.Tensor,
    text_prompt: str,
    device: str,
):
    """Construct the ``BatchedDatapoint`` that ``Sam3Image.forward()`` expects.

    This is the minimal construction needed for a single-image, single-text,
    no-box-prompt inference call.  We build tensors directly rather than going
    through ``convert_my_tensors`` (which expects list-of-tensor inputs from
    the data loader).
    """
    from sam3.model.data_misc import (
        BatchedDatapoint,
        BatchedFindTarget,
        BatchedInferenceMetadata,
        FindStage,
    )

    dev = torch.device(device)

    # One find stage: one query (the text prompt) referencing one image
    find_stage = FindStage(
        img_ids=torch.tensor([0], dtype=torch.long, device=dev),
        text_ids=torch.tensor([0], dtype=torch.long, device=dev),
        input_boxes=torch.zeros(0, 1, 4, device=dev),          # no box prompts
        input_boxes_mask=torch.zeros(1, 0, dtype=torch.bool, device=dev),
        input_boxes_label=torch.zeros(0, 1, dtype=torch.long, device=dev),
        input_points=torch.zeros(0, 1, 2, device=dev),         # no point prompts
        input_points_mask=torch.zeros(1, 0, dtype=torch.bool, device=dev),
        object_ids=None,
    )

    # Dummy targets (not used in eval mode but required by forward signature)
    find_target = BatchedFindTarget(
        num_boxes=torch.tensor([0], dtype=torch.long),
        boxes=torch.zeros(0, 4),
        boxes_padded=torch.zeros(1, 0, 4),
        is_exhaustive=torch.tensor([True], dtype=torch.bool),
        segments=None,
        semantic_segments=None,
        is_valid_segment=None,
        repeated_boxes=torch.zeros(0, 4),
        object_ids=torch.zeros(0, dtype=torch.long),
        object_ids_padded=torch.zeros(1, 0, dtype=torch.long),
    )

    # Dummy metadata
    find_metadata = BatchedInferenceMetadata(
        coco_image_id=torch.tensor([0], dtype=torch.long),
        original_size=torch.tensor([[_SAM3_IMG_SIZE, _SAM3_IMG_SIZE]], dtype=torch.long),
        object_id=torch.tensor([0], dtype=torch.long),
        frame_index=torch.tensor([0], dtype=torch.long),
        original_image_id=torch.tensor([0], dtype=torch.long),
        original_category_id=torch.tensor([0], dtype=torch.int),
        is_conditioning_only=[False],
    )

    batch = BatchedDatapoint(
        img_batch=img_tensor,             # (1, 3, 1008, 1008)
        find_text_batch=[text_prompt],     # list of text prompts
        find_inputs=[find_stage],
        find_targets=[find_target],
        find_metadatas=[find_metadata],
    )

    return batch


class SAM3Detector:
    """Text-prompted object detection + segmentation using SAM3.

    Replaces the two-stage GDino + SAM2 pipeline with a single model.

    GPU memory: ~3–4 GB (ViT-L backbone, 1008×1008 input).
    Inference:  ~200–400 ms per image on A100/H100.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str | None = None,
        score_threshold: float = 0.15,
        mask_threshold: float = 0.5,
    ):
        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        import sam3 as _sam3_pkg
        from sam3 import build_sam3_image_model

        # Resolve the BPE tokenizer vocabulary path.
        # The default resolution in model_builder.py has a packaging bug
        # (looks for ``../assets/`` relative to model_builder.py instead
        # of inside the sam3 package).  We fix it here by searching a few
        # likely locations.
        bpe_path = None
        pkg_dir = os.path.dirname(_sam3_pkg.__file__)
        for candidate in [
            os.path.join(pkg_dir, "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(pkg_dir, "..", "assets", "bpe_simple_vocab_16e6.txt.gz"),
        ]:
            if os.path.isfile(candidate):
                bpe_path = candidate
                break
        if bpe_path is not None:
            logger.info(f"Using BPE vocab: {bpe_path}")

        logger.info(f"Loading SAM3 model: {self.model_id}")
        t0 = time.time()

        kwargs = dict(
            device=self.device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=True,
        )
        if bpe_path is not None:
            kwargs["bpe_path"] = bpe_path
        self._model = build_sam3_image_model(**kwargs)
        elapsed = time.time() - t0
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(
            f"SAM3 loaded in {elapsed:.1f}s on {self.device} "
            f"(params={n_params:.0f}M)"
        )

    # ------------------------------------------------------------------
    # Primary API: unified detect + segment
    # ------------------------------------------------------------------

    def detect_and_segment(
        self,
        image: np.ndarray,
        text_prompt: str,
        score_threshold: float | None = None,
        mask_threshold: float | None = None,
    ) -> list[dict]:
        """Detect and segment all instances matching *text_prompt*.

        Args:
            image: RGB uint8 array (H, W, 3).
            text_prompt: Text description of the target object.
            score_threshold: Minimum confidence score for detections.
            mask_threshold: Threshold for binarising predicted masks.

        Returns:
            List of dicts sorted by score descending::

                [
                    {
                        "score": float,
                        "label": str,
                        "box": [x1, y1, x2, y2],  # pixel coords, int
                        "mask": np.ndarray,         # (H, W) bool
                    },
                    ...
                ]
        """
        self._ensure_loaded()

        threshold = score_threshold or self.score_threshold
        m_threshold = mask_threshold or self.mask_threshold
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        img_tensor, _, _ = _preprocess_image(image, self.device)

        # Build input datapoint
        batch = _build_batched_datapoint(img_tensor, text_prompt, self.device)

        # Forward pass
        t0 = time.time()
        with torch.no_grad():
            output = self._model(batch)

        # Extract last-stage output.
        # SAM3Output is initialised with IterMode.LAST_STEP_PER_STAGE,
        # so output[0] gives the final decoder step of the first (only) stage.
        last_out = output[0]  # dict with pred_logits, pred_boxes, pred_masks, …
        elapsed = time.time() - t0

        # --- Post-process ---
        detections = []

        # Scores: sigmoid of logits × presence logit
        pred_logits = last_out["pred_logits"]  # (1, N, 1) or (1, N, C)
        scores = pred_logits.sigmoid()  # (1, N, C)
        if "presence_logit_dec" in last_out:
            presence = last_out["presence_logit_dec"].sigmoid().unsqueeze(-1)  # (1, N, 1)
            scores = scores * presence
        scores = scores.squeeze(0).max(dim=-1).values  # (N,)

        # Boxes: cxcywh normalised → xyxy pixel
        pred_boxes = last_out.get("pred_boxes")  # (1, N, 4) normalised cxcywh
        if pred_boxes is not None:
            pred_boxes = pred_boxes.squeeze(0)  # (N, 4)

        # Masks: (1, N, H_mask, W_mask) logits
        pred_masks = last_out.get("pred_masks")  # may be None if segmentation disabled

        # Log top-k scores before filtering (diagnostic)
        topk_k = min(10, scores.numel())
        topk_scores, topk_idx = scores.topk(topk_k)
        logger.info(
            f"SAM3 raw scores for '{text_prompt}': "
            f"top-{topk_k} = {[f'{s:.3f}' for s in topk_scores.tolist()]}, "
            f"threshold={threshold}"
        )

        # Filter by threshold
        keep = scores > threshold
        keep_idx = keep.nonzero(as_tuple=True)[0]

        for i in keep_idx:
            s = float(scores[i])
            idx = int(i)

            # Box: convert cxcywh normalised → xyxy pixel coords
            if pred_boxes is not None:
                cx, cy, bw, bh = pred_boxes[idx].tolist()
                # Boxes are normalised to [0, 1] relative to the padded 1008×1008 image
                # We need to map back to original image coords
                scale = _SAM3_IMG_SIZE / max(orig_h, orig_w)
                x1 = int((cx - bw / 2) * _SAM3_IMG_SIZE / scale)
                y1 = int((cy - bh / 2) * _SAM3_IMG_SIZE / scale)
                x2 = int((cx + bw / 2) * _SAM3_IMG_SIZE / scale)
                y2 = int((cy + bh / 2) * _SAM3_IMG_SIZE / scale)
                # Clamp to image bounds
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
                box = [x1, y1, x2, y2]
            else:
                box = [0, 0, orig_w, orig_h]

            # Mask: resize from model resolution to original image size
            if pred_masks is not None:
                mask_logits = pred_masks[0, idx]  # (H_mask, W_mask)
                # Resize to padded 1008×1008 first, then crop to original aspect
                mask_resized = F.interpolate(
                    mask_logits.unsqueeze(0).unsqueeze(0).float(),
                    size=(_SAM3_IMG_SIZE, _SAM3_IMG_SIZE),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()  # (1008, 1008)
                # Crop the letterboxed region
                scale = _SAM3_IMG_SIZE / max(orig_h, orig_w)
                crop_h = int(orig_h * scale + 0.5)
                crop_w = int(orig_w * scale + 0.5)
                mask_cropped = mask_resized[:crop_h, :crop_w]
                # Resize to original image size
                mask_orig = F.interpolate(
                    mask_cropped.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                mask = (mask_orig > m_threshold).cpu().numpy()
            else:
                mask = np.ones((orig_h, orig_w), dtype=bool)

            detections.append({
                "score": s,
                "label": text_prompt,
                "box": box,
                "mask": mask,
            })

        # Sort by score descending
        detections.sort(key=lambda d: d["score"], reverse=True)

        logger.debug(
            f"SAM3: {len(detections)} detections for '{text_prompt}' "
            f"on {orig_w}×{orig_h} image ({elapsed * 1000:.0f}ms)"
        )
        return detections

    # ------------------------------------------------------------------
    # GDino-compatible interface (drop-in replacement)
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Run text-prompted detection (GDino-compatible interface).

        Returns list of dicts with ``score``, ``label``, ``box`` keys
        (masks omitted for compatibility with existing GDino callers).

        Args:
            image: RGB uint8 array (H, W, 3).
            text_prompt: Text description. Should end with period for
                GDino compat, but SAM3 doesn't require it.
            score_threshold: Minimum confidence.

        Returns:
            Same format as ``GroundingDINODetector.detect()``.
        """
        # Strip trailing period (GDino convention, not needed for SAM3)
        prompt = text_prompt.rstrip(".")

        results = self.detect_and_segment(
            image, prompt, score_threshold=score_threshold,
        )

        # Return GDino-compatible format (without masks)
        return [
            {"score": r["score"], "label": r["label"], "box": r["box"]}
            for r in results
        ]

    def detect_best(
        self,
        image: np.ndarray,
        text_prompt: str,
        score_threshold: float | None = None,
        color_filter: str | None = None,
    ) -> dict | None:
        """Return the single best detection.

        Compatible with ``GroundingDINODetector.detect_best()``.
        """
        detections = self.detect(image, text_prompt, score_threshold)
        if not detections:
            return None

        if color_filter and len(detections) > 1:
            return _pick_by_color(image, detections, color_filter)

        return detections[0]

    # ------------------------------------------------------------------
    # SAM2-compatible segmentation interface
    # ------------------------------------------------------------------

    def segment_at_box(
        self,
        image: np.ndarray,
        text_prompt: str,
        bbox: tuple[int, int, int, int] | None = None,
        score_threshold: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Segment the target object, optionally constrained to *bbox*.

        Returns ``(mask, score)`` — compatible with the SAM2 ``segment()``
        return format used in ``grasp_client.py``.

        If *bbox* is provided, picks the detection with highest overlap.
        Otherwise picks the highest-scoring detection.
        """
        results = self.detect_and_segment(
            image, text_prompt, score_threshold=score_threshold,
        )
        if not results:
            # Fallback: return empty mask
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0

        if bbox is not None:
            # Pick detection with highest IoU to the given bbox
            best = _pick_by_bbox_overlap(results, bbox)
        else:
            best = results[0]  # highest score

        return best["mask"], best["score"]


# ------------------------------------------------------------------
# Utilities (shared with GroundingDINODetector)
# ------------------------------------------------------------------


def _pick_by_color(
    image: np.ndarray,
    detections: list[dict],
    color: str,
) -> dict:
    """Among detections, pick the one whose box region best matches the
    named color."""
    color_idx = {"red": 0, "green": 1, "blue": 2}.get(color.lower())

    best = detections[0]
    best_score = -999.0

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = image[y1:y2, x1:x2].astype(np.float32)
        mean_rgb = region.mean(axis=(0, 1))

        if color.lower() == "yellow":
            score = (mean_rgb[0] + mean_rgb[1]) / 2 - mean_rgb[2]
        elif color_idx is not None:
            others = [mean_rgb[i] for i in range(3) if i != color_idx]
            score = mean_rgb[color_idx] - max(others)
        else:
            score = det["score"]

        if score > best_score:
            best_score = score
            best = det

    return best


def _pick_by_bbox_overlap(
    detections: list[dict],
    target_bbox: tuple[int, int, int, int],
) -> dict:
    """Pick the detection with highest IoU to *target_bbox*."""
    tx1, ty1, tx2, ty2 = target_bbox
    t_area = max(0, tx2 - tx1) * max(0, ty2 - ty1)

    best = detections[0]
    best_iou = -1.0

    for det in detections:
        dx1, dy1, dx2, dy2 = det["box"]
        inter_x1 = max(tx1, dx1)
        inter_y1 = max(ty1, dy1)
        inter_x2 = min(tx2, dx2)
        inter_y2 = min(ty2, dy2)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        d_area = max(0, dx2 - dx1) * max(0, dy2 - dy1)
        union = t_area + d_area - inter
        iou = inter / union if union > 0 else 0.0

        if iou > best_iou:
            best_iou = iou
            best = det

    return best
