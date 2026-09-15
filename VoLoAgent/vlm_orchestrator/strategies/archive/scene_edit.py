# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-edit strategy — visual highlighting for policy guidance.

Detects the target object in the scene (via GroundingDINO or VLM),
then edits the policy input images to highlight the target and/or
dim distractors. The instruction is passed through unchanged.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

from vlm_orchestrator.strategies.archive.image_edit import (
    BBox,
    BBoxQueryClient,
    apply_edit,
    save_debug_image,
    BBOX_QUERY_PROMPT,
    WRIST_BBOX_QUERY_PROMPT,
)
from ..base import OrchestrationStrategy, SessionState, StrategyContext

logger = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================

@dataclass
class SceneEditConfig:
    """Configuration for the scene-edit strategy."""

    edit_mode: str = "highlight"
    """Image edit mode: 'none', 'highlight', 'dim', 'both'."""

    requery_interval: int = 40
    """Re-query for updated bounding box every N sim steps.  Gated by
    step delta against ``state.episode_step``; chunk-size invariant
    across VLAs (pi05=8 → 5 chunks, gr00t=10 → 4 chunks, etc.)."""

    edit_wrist: bool = True
    """If False, skip detection and editing on the wrist camera."""

    fixed_instruction: str | None = None
    """If set, override the client instruction with this fixed string."""

    # --- Detection backend ---
    detector: str = "gdino"
    """Detection backend: 'gdino' (GroundingDINO, local GPU) or 'vlm'."""

    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    """GroundingDINO model ID (only used when detector='gdino')."""

    gdino_prompt: str = "small green wooden block."
    """Text prompt for GroundingDINO detection."""

    gdino_color_filter: str | None = "green"
    """If set, pick the detection whose region best matches this color."""

    gdino_score_threshold: float = 0.20
    """Minimum detection confidence for GroundingDINO."""

    # --- VLM backend (used when detector='vlm') ---
    vlm_model: str = "YOUR_VLM_MODEL"
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_temperature: float = 0.0

    save_debug_images: bool = True
    """Save sample edited images for visual inspection."""

    debug_image_dir: str | None = None
    """Directory for debug images. If None, uses log_dir/debug_images."""


# ======================================================================
# Extended session state fields for scene edit
# ======================================================================

@dataclass
class SceneEditState:
    """Extra state tracked by the SceneEditStrategy, stored on SessionState."""

    exterior_bbox: BBox | None = None
    wrist_bbox: BBox | None = None
    # Sim step at which the last bbox re-query happened.  Cadence is
    # gated by step delta against ``state.episode_step``; chunk-size
    # invariant across VLAs.
    step_at_last_query: int = 0
    vlm_responses: list[dict] = field(default_factory=list)
    # Episode chunk counter retained for log/annotation purposes
    # (records how many VLA inferences occurred in this episode).
    episode_chunk_count: int = 0
    first_frame_saved: bool = False


# ======================================================================
# Strategy
# ======================================================================

class SceneEditStrategy(OrchestrationStrategy):
    """Edit policy input images to highlight targets / dim distractors.

    Supports two detection backends:

    * ``"gdino"`` — GroundingDINO: fast (~50ms), accurate, local GPU.
    * ``"vlm"``   — VLM bbox query: slower (~5s), less accurate spatially.

    On the first frame of each episode, detects the target object.
    Every ``requery_interval`` action chunks, re-detects (object may have
    moved or been picked up). Applies the configured image edit to both
    camera views before forwarding to the policy server.
    """

    def __init__(self, ctx: StrategyContext, config: SceneEditConfig):
        super().__init__(ctx)
        self.config = config
        self._bbox_client: BBoxQueryClient | None = None
        self._gdino_detector = None
        self._edit_states: dict[int, SceneEditState] = {}

    @property
    def bbox_client(self) -> BBoxQueryClient:
        if self._bbox_client is None:
            self._bbox_client = BBoxQueryClient(
                model=self.config.vlm_model,
                base_url=self.config.vlm_base_url,
                api_key=self.config.vlm_api_key,
                temperature=self.config.vlm_temperature,
            )
        return self._bbox_client

    @property
    def gdino_detector(self):
        if self._gdino_detector is None:
            from vlm_orchestrator.perception.gdino import GroundingDINODetector
            self._gdino_detector = GroundingDINODetector(
                model_id=self.config.gdino_model,
                box_threshold=self.config.gdino_score_threshold,
            )
        return self._gdino_detector

    def _get_edit_state(self, state: SessionState) -> SceneEditState:
        """Get or create the SceneEditState for the current episode."""
        eid = state.episode_id
        if eid not in self._edit_states:
            self._edit_states[eid] = SceneEditState()
        return self._edit_states[eid]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        current_prompt = self.ctx.get_prompt(obs)
        is_new = self._is_new_episode(obs, state)

        if is_new:
            state.original_instruction = current_prompt
            # Clean up old edit states
            old_ids = [k for k in self._edit_states if k < state.episode_id - 1]
            for k in old_ids:
                del self._edit_states[k]

        edit_state = self._get_edit_state(state)

        # Override instruction if configured
        if self.config.fixed_instruction:
            obs = self.ctx.set_prompt(obs, self.config.fixed_instruction)
            state.rewritten_instruction = self.config.fixed_instruction

        # Skip edits if mode is none
        if self.config.edit_mode == "none":
            return obs, state

        # Query VLM for bounding box on first frame or at requery interval
        should_query = (
            is_new
            or state.episode_step - edit_state.step_at_last_query
                >= self.config.requery_interval
        )

        if should_query:
            obs, edit_state = self._query_and_update_bbox(obs, state, edit_state, is_new)
            edit_state.step_at_last_query = state.episode_step

        # Apply image edits
        obs = self._apply_edits(obs, state, edit_state)

        edit_state.episode_chunk_count += 1
        return obs, state

    # ------------------------------------------------------------------
    # Bounding box detection (dispatches to gdino or vlm backend)
    # ------------------------------------------------------------------

    def _query_and_update_bbox(
        self,
        obs: dict,
        state: SessionState,
        edit_state: SceneEditState,
        is_new_episode: bool,
    ) -> tuple[dict, SceneEditState]:
        """Detect target object bbox via the configured backend."""
        if self.config.detector == "gdino":
            return self._query_gdino(obs, state, edit_state)
        else:
            return self._query_vlm(obs, state, edit_state)

    # ------------------------------------------------------------------
    # GroundingDINO backend
    # ------------------------------------------------------------------

    def _query_gdino(
        self,
        obs: dict,
        state: SessionState,
        edit_state: SceneEditState,
    ) -> tuple[dict, SceneEditState]:
        """Detect target using GroundingDINO on the raw image."""

        # Detect on exterior camera (prefer high-res raw image)
        ext_image = self.ctx.get_vlm_image(obs)
        if ext_image is not None:
            t0 = time.time()
            det = self.gdino_detector.detect_best(
                ext_image,
                text_prompt=self.config.gdino_prompt,
                score_threshold=self.config.gdino_score_threshold,
                color_filter=self.config.gdino_color_filter,
            )
            elapsed = time.time() - t0

            if det is not None:
                b = det["box"]
                bbox = BBox(b[0], b[1], b[2], b[3])
                edit_state.exterior_bbox = bbox
                logger.info(
                    f"  GDino exterior: {bbox} score={det['score']:.3f} "
                    f"({elapsed*1000:.0f}ms)"
                )
            else:
                edit_state.exterior_bbox = None
                logger.info(f"  GDino exterior: not found ({elapsed*1000:.0f}ms)")

            state.log({
                "type": "scene_edit_bbox",
                "camera": "exterior",
                "detector": "gdino",
                "bbox": bbox.to_dict() if det else None,
                "score": det["score"] if det else None,
                "elapsed_s": elapsed,
                "visible": det is not None,
            })
        else:
            logger.warning("  No exterior image for GDino detection")

        # Detect on wrist camera (skip if edit_wrist is disabled)
        if not self.config.edit_wrist:
            edit_state.wrist_bbox = None
            return obs, edit_state

        wrist_image = None
        extra_images = self.ctx.get_vlm_extra_images(obs)
        if extra_images and len(extra_images) > 0:
            wrist_image = extra_images[0]

        if wrist_image is not None and edit_state.exterior_bbox is not None:
            t0 = time.time()
            wrist_det = self.gdino_detector.detect_best(
                wrist_image,
                text_prompt=self.config.gdino_prompt,
                score_threshold=self.config.gdino_score_threshold,
                color_filter=self.config.gdino_color_filter,
            )
            elapsed = time.time() - t0

            if wrist_det is not None:
                b = wrist_det["box"]
                edit_state.wrist_bbox = BBox(b[0], b[1], b[2], b[3])
            else:
                edit_state.wrist_bbox = None

            state.log({
                "type": "scene_edit_bbox",
                "camera": "wrist",
                "detector": "gdino",
                "bbox": edit_state.wrist_bbox.to_dict() if wrist_det else None,
                "elapsed_s": elapsed,
                "visible": wrist_det is not None,
            })
        else:
            edit_state.wrist_bbox = None

        return obs, edit_state

    # ------------------------------------------------------------------
    # VLM backend
    # ------------------------------------------------------------------

    def _query_vlm(
        self,
        obs: dict,
        state: SessionState,
        edit_state: SceneEditState,
    ) -> tuple[dict, SceneEditState]:
        """Detect target using VLM bbox query on the raw image."""

        # Query exterior camera
        ext_image = self.ctx.get_vlm_image(obs)
        if ext_image is not None:
            bbox, response_info = self.bbox_client.query_bbox(
                ext_image, prompt=BBOX_QUERY_PROMPT
            )
            edit_state.exterior_bbox = bbox
            response_info["camera"] = "exterior"
            response_info["episode_id"] = state.episode_id
            response_info["chunk"] = edit_state.episode_chunk_count
            edit_state.vlm_responses.append(response_info)

            state.log({
                "type": "scene_edit_bbox",
                "camera": "exterior",
                "detector": "vlm",
                "bbox": bbox.to_dict() if bbox else None,
                "elapsed_s": response_info.get("elapsed_s"),
                "visible": bbox is not None,
            })

            logger.info(
                f"  VLM exterior bbox: {bbox}" if bbox
                else "  VLM exterior: target not visible"
            )
        else:
            logger.warning("  No exterior image available for VLM bbox query")

        # Query wrist camera (only if we found the target in exterior)
        if not self.config.edit_wrist:
            edit_state.wrist_bbox = None
            return obs, edit_state

        wrist_image = None
        extra_images = self.ctx.get_vlm_extra_images(obs)
        if extra_images and len(extra_images) > 0:
            wrist_image = extra_images[0]

        if wrist_image is not None and edit_state.exterior_bbox is not None:
            wrist_bbox, wrist_info = self.bbox_client.query_bbox(
                wrist_image, prompt=WRIST_BBOX_QUERY_PROMPT
            )
            edit_state.wrist_bbox = wrist_bbox
            wrist_info["camera"] = "wrist"
            wrist_info["episode_id"] = state.episode_id
            wrist_info["chunk"] = edit_state.episode_chunk_count
            edit_state.vlm_responses.append(wrist_info)

            state.log({
                "type": "scene_edit_bbox",
                "camera": "wrist",
                "detector": "vlm",
                "bbox": wrist_bbox.to_dict() if wrist_bbox else None,
                "elapsed_s": wrist_info.get("elapsed_s"),
                "visible": wrist_bbox is not None,
            })
        else:
            edit_state.wrist_bbox = None

        return obs, edit_state

    # ------------------------------------------------------------------
    # Apply image edits
    # ------------------------------------------------------------------

    def _apply_edits(
        self,
        obs: dict,
        state: SessionState,
        edit_state: SceneEditState,
    ) -> dict:
        """Apply image edits to the observation's camera images."""
        obs = dict(obs)  # shallow copy

        # Edit exterior image
        ext_image = obs.get(self.ctx.image_key)
        if isinstance(ext_image, np.ndarray) and edit_state.exterior_bbox is not None:
            # Scale bbox from VLM (raw) image dims to policy (padded) image dims.
            # The policy image is created by resize_with_pad: the raw image is
            # uniformly scaled to fit 224×224 and centered with black bars.
            vlm_img = self.ctx.get_vlm_image(obs)
            if vlm_img is not None and vlm_img.shape != ext_image.shape:
                bbox = edit_state.exterior_bbox.scale_with_pad(
                    vlm_img.shape[0], vlm_img.shape[1],
                    ext_image.shape[0], ext_image.shape[1],
                )
            else:
                bbox = edit_state.exterior_bbox

            edited = apply_edit(ext_image, bbox, self.config.edit_mode)
            obs[self.ctx.image_key] = edited

            # Save debug image on first frame
            if not edit_state.first_frame_saved and self.config.save_debug_images:
                self._save_debug(
                    ext_image, edited, bbox,
                    f"ep{state.episode_id}_ext",
                    edit_state,
                )

        # Edit wrist image
        for i, key in enumerate(self.ctx.extra_image_keys):
            wrist_image = obs.get(key)
            if isinstance(wrist_image, np.ndarray) and edit_state.wrist_bbox is not None:
                # Scale wrist bbox similarly
                vlm_wrist = None
                vlm_extras = self.ctx.get_vlm_extra_images(obs)
                if vlm_extras and len(vlm_extras) > i:
                    vlm_wrist = vlm_extras[i]

                if vlm_wrist is not None and vlm_wrist.shape != wrist_image.shape:
                    wrist_bbox = edit_state.wrist_bbox.scale_with_pad(
                        vlm_wrist.shape[0], vlm_wrist.shape[1],
                        wrist_image.shape[0], wrist_image.shape[1],
                    )
                else:
                    wrist_bbox = edit_state.wrist_bbox

                edited_wrist = apply_edit(wrist_image, wrist_bbox, self.config.edit_mode)
                obs[key] = edited_wrist

                if not edit_state.first_frame_saved and self.config.save_debug_images:
                    self._save_debug(
                        wrist_image, edited_wrist, wrist_bbox,
                        f"ep{state.episode_id}_wrist",
                        edit_state,
                    )

        edit_state.first_frame_saved = True
        return obs

    # ------------------------------------------------------------------
    # Debug image saving
    # ------------------------------------------------------------------

    def _save_debug(
        self,
        original: np.ndarray,
        edited: np.ndarray,
        bbox: BBox,
        prefix: str,
        edit_state: SceneEditState,
    ) -> None:
        """Save original and edited images side by side for inspection."""
        debug_dir = self.config.debug_image_dir
        if not debug_dir:
            return

        os.makedirs(debug_dir, exist_ok=True)
        save_debug_image(original, os.path.join(debug_dir, f"{prefix}_original.png"))
        save_debug_image(edited, os.path.join(debug_dir, f"{prefix}_{self.config.edit_mode}.png"))

        # Also save a side-by-side comparison
        h, w = original.shape[:2]
        comparison = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w + 4:] = edited
        comparison[:, w:w + 4] = 128  # grey separator
        save_debug_image(comparison, os.path.join(debug_dir, f"{prefix}_comparison.png"))
