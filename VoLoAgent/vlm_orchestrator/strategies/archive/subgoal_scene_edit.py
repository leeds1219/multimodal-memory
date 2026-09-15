# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subgoal + Scene-edit strategy — decomposes tasks, detects targets
with GroundingDINO, and edits policy images to highlight them.

Extends :class:`SubgoalBaseStrategy` with:
* Per-subgoal target identification (VLM) and detection (GDino).
* Per-frame image editing (highlight / dim) around the detected bbox.
* VLM completion checks using the highlighted image.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

from vlm_orchestrator.strategies.archive.image_edit import (
    BBox, apply_edit, save_debug_image,
)
from ..base import OrchestrationStrategy, SessionState, StrategyContext
from ..subgoal_base import (
    CHECK_DONE_SHARED_PROMPT,
    DECOMPOSE_SHARED_PROMPT,
    MAX_RECYCLES,
    SubgoalBaseStrategy,
    parse_json,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Prompts
# ======================================================================

DECOMPOSE_SYSTEM_PROMPT = DECOMPOSE_SHARED_PROMPT + """

For each subgoal, also specify the PRIMARY target object to focus on \
(a short noun phrase the robot's object detector can look for in the \
scene).

Output ONLY valid JSON (no markdown):
{
  "subgoals": [
    {"instruction": "Pick up the yellow banana and place it in the bowl",
     "target_object": "yellow banana"},
    {"instruction": "Pick up the Rubik's cube and place it in the bowl",
     "target_object": "Rubik's cube"}
  ],
  "ordered": true
}\
"""

IDENTIFY_TARGET_PROMPT = """\
You are helping a robot arm. Given a manipulation subgoal, identify the \
best short text prompt to find the target object in the scene using an \
object detector.

The prompt should be a short noun phrase describing the object's visual \
appearance (colour, shape, type). Examples:
- "yellow banana"
- "small green wooden block"
- "red Rubik's cube"
- "grey rectangular bin"

Subgoal: "{subgoal}"

Output ONLY valid JSON:
{{"gdino_prompt": "short object description", \
"color_filter": "dominant_color_or_null"}}\
"""

CHECK_DONE_SYSTEM_PROMPT = CHECK_DONE_SHARED_PROMPT


# ======================================================================
# Config
# ======================================================================


@dataclass
class SubgoalSceneEditConfig:
    """Config for the combined subgoal + scene-edit strategy."""

    # --- Scene edit ---
    edit_mode: str = "highlight"
    """Image edit mode: 'highlight', 'dim', 'both'."""

    edit_wrist: bool = False
    """Whether to also edit wrist camera images."""

    requery_interval: int = 40
    """Re-detect target bbox every N sim steps.  Gated by step delta
    against ``state.episode_step``; chunk-size invariant across VLAs."""

    # --- GroundingDINO ---
    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    gdino_score_threshold: float = 0.20

    # --- Subgoal --- (sim steps; see SubgoalConfig)
    check_interval: int = 40
    # Effectively-disabled timeout — advancement gated on VLM/GT
    # completion checks instead.  Match SubgoalConfig default.
    subgoal_timeout: int = 9999

    # --- VLM ---
    vlm_model: str = "YOUR_VLM_MODEL"
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 300

    # --- Debug ---
    save_debug_images: bool = True
    debug_image_dir: str | None = None


# ======================================================================
# Per-subgoal edit state
# ======================================================================


@dataclass
class SubgoalEditState:
    """Tracking state for the current subgoal's scene editing."""
    gdino_prompt: str = ""
    color_filter: str | None = None
    exterior_bbox: BBox | None = None
    wrist_bbox: BBox | None = None
    chunks_since_detect: int = 0


# ======================================================================
# Strategy
# ======================================================================


class SubgoalSceneEditStrategy(SubgoalBaseStrategy):
    """Subgoal decomposition + GroundingDINO scene editing.

    Optionally wraps an *initial strategy* (rewrite / adaptive) that runs
    before decomposition.
    """

    def __init__(
        self,
        ctx: StrategyContext,
        config: SubgoalSceneEditConfig,
        initial_strategy: OrchestrationStrategy | None = None,
        failure_monitor: str | None = None,
        recovery_mode: str = "template",
        hitl_state=None,
        grasp_seg_mode: str = "gdino_sam2",
        place_seg_mode: str | None = None,
        env_mode: str = "robolab",
        collect_trajectories: str | None = None,
        gt_failure_types: set[str] | None = None,
        grasp_topdown_threshold: float | None = None,
        motion_planner: str = "linear",
    ):
        super().__init__(
            ctx,
            initial_strategy=initial_strategy,
            failure_monitor=failure_monitor,
            recovery_mode=recovery_mode,
            hitl_state=hitl_state,
            grasp_seg_mode=grasp_seg_mode,
            place_seg_mode=place_seg_mode,
            env_mode=env_mode,
            collect_trajectories=collect_trajectories,
            gt_failure_types=gt_failure_types,
            grasp_topdown_threshold=grasp_topdown_threshold,
            motion_planner=motion_planner,
        )
        self.config = config
        self._gdino = None
        self._edit_state = SubgoalEditState()

    @property
    def gdino(self):
        if self._gdino is None:
            from vlm_orchestrator.perception.gdino import GroundingDINODetector
            self._gdino = GroundingDINODetector(
                model_id=self.config.gdino_model,
                box_threshold=self.config.gdino_score_threshold,
            )
        return self._gdino

    # ------------------------------------------------------------------
    # Decomposition (with target objects)
    # ------------------------------------------------------------------

    def _decompose_and_setup(self, obs, state, instruction, image):
        extra = self.ctx.get_vlm_extra_images(obs)
        user_content = self._build_image_message(
            f'Scene instruction: "{instruction}"', image, extra,
        )

        t0 = time.time()
        try:
            raw = self._vlm_call(DECOMPOSE_SYSTEM_PROMPT, user_content)
            elapsed = time.time() - t0
            logger.info(f"  Decompose VLM call: {elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"  Decompose failed: {e}")
            state.subgoals = [instruction]
            state._subgoal_targets = [
                {"gdino_prompt": "", "color_filter": None}
            ]
            state.rewritten_instruction = instruction
            state.log({
                "type": "decompose",
                "instruction": instruction,
                "subgoals": [instruction],
                "targets": [{"gdino_prompt": "", "color_filter": None}],
                "fallback": "vlm_call_failed",
                "error": str(e),
            })
            return

        logger.debug(f"  Decompose raw: {raw[:400]}")

        parse_error: str | None = None
        try:
            data = parse_json(raw)
            subgoals_raw = data.get("subgoals", [])
            if not subgoals_raw:
                raise ValueError("empty subgoals")

            results = self._parse_subgoals(subgoals_raw)

            # Fill in missing gdino prompts
            for r in results:
                if not r["gdino_prompt"] or r["gdino_prompt"] == ".":
                    info = self._identify_target(r["instruction"])
                    r["gdino_prompt"] = info.get(
                        "gdino_prompt", "object."
                    )
                    r["color_filter"] = info.get("color_filter")

        except (ValueError, KeyError) as e:
            logger.warning(f"  Cannot parse decomposition: {e}")
            results = [{
                "instruction": instruction,
                "gdino_prompt": "",
                "color_filter": None,
            }]
            parse_error = str(e)

        state.subgoals = [r["instruction"] for r in results]
        state._subgoal_targets = [
            {"gdino_prompt": r.get("gdino_prompt", ""),
             "color_filter": r.get("color_filter")}
            for r in results
        ]

        log_entry = {
            "type": "decompose",
            "instruction": instruction,
            "subgoals": list(state.subgoals),
            "targets": list(state._subgoal_targets),
            "vlm_latency_s": round(elapsed, 2),
        }
        if parse_error is not None:
            log_entry["fallback"] = "parse_failed"
            log_entry["error"] = parse_error
        state.log(log_entry)

        if state.subgoals:
            state.rewritten_instruction = state.subgoals[0]
            obs = self.ctx.set_prompt(obs, state.subgoals[0])
            logger.info(
                f"Episode {state.episode_id}: "
                f"{len(state.subgoals)} subgoals"
            )
            for i, r in enumerate(results):
                logger.info(
                    f"  [{i + 1}] {r['instruction']} "
                    f"| detect: '{r.get('gdino_prompt', '?')}'"
                )

            self._setup_subgoal_detection(state, 0)
            self._detect_target(obs)
        else:
            state.rewritten_instruction = instruction

    # ------------------------------------------------------------------
    # VLM check (binary, with highlighted images)
    # ------------------------------------------------------------------

    def _check_subgoal(self, obs, state, image):
        current_sg = state.subgoals[state.current_subgoal_idx]
        total = len(state.subgoals)

        text = (
            f'Step [{state.current_subgoal_idx + 1}/{total}]: '
            f'"{current_sg}"\nIs this step done?'
        )

        # Use highlighted image for NOW so VLM sees same cues as policy
        now_image = self._apply_edit_to_image(image)
        extra = self.ctx.get_vlm_extra_images(obs)
        primary_label, extra_labels = self.ctx.vlm_camera_labels
        user_content = self._build_check_message(
            text, state.initial_image, now_image, extra,
            initial_extra_images=state.initial_extra_images,
            primary_label=primary_label,
            extra_labels=extra_labels,
        )

        t0 = time.time()
        try:
            raw = self._vlm_call(CHECK_DONE_SYSTEM_PROMPT, user_content)
            elapsed = time.time() - t0
        except Exception as e:
            logger.warning(f"  Check VLM call failed: {e}")
            return obs, state

        logger.debug(f"  Check raw ({elapsed:.1f}s): {raw[:200]}")

        done = False
        try:
            data = parse_json(raw)
            done = bool(data.get("done", False))
        except ValueError:
            pass

        state.log({
            "type": "check",
            "done": done,
            "subgoal_idx": state.current_subgoal_idx,
            "instruction": current_sg,
            "vlm_latency_s": round(elapsed, 2),
        })

        state.vlm_check_result = {
            "done": done,
            "action": "next" if done else "continue",
        }
        state.vlm_check_infer_step = state.infer_count

        if done and state.current_subgoal_idx < total - 1:
            logger.info(f"  ✓ Subgoal DONE: \"{current_sg}\"")
            self._advance_subgoal(obs, state)
            state.log({
                "type": "subgoal_advance",
                "subgoal_idx": state.current_subgoal_idx,
                "instruction": state.rewritten_instruction,
            })
        elif done:
            # Last subgoal done — try recycling with the RAW image
            # (not highlighted) so VLM can clearly see remaining objects.
            if self._recycle_count < MAX_RECYCLES:
                if self._recycle(obs, state, image):
                    return obs, state
            logger.info("  ✓ Last subgoal DONE, continuing execution")
        else:
            logger.debug("  Check: NOT DONE")

        return obs, state

    # ------------------------------------------------------------------
    # Recycling (with target objects + GDino)
    # ------------------------------------------------------------------

    def _apply_recycle_subgoals(self, obs, state, data):
        subgoals_raw = data["subgoals"]
        results = self._parse_subgoals(subgoals_raw)

        # Fill in missing gdino prompts
        for r in results:
            if not r["gdino_prompt"] or r["gdino_prompt"] == ".":
                info = self._identify_target(r["instruction"])
                r["gdino_prompt"] = info.get("gdino_prompt", "object.")
                r["color_filter"] = info.get("color_filter")

        state.subgoals = [r["instruction"] for r in results]
        state._subgoal_targets = [
            {"gdino_prompt": r.get("gdino_prompt", ""),
             "color_filter": r.get("color_filter")}
            for r in results
        ]
        state.current_subgoal_idx = 0
        state.rewritten_instruction = state.subgoals[0]
        # Detection reset, GT config, flush, and subclass hooks
        # (_on_subgoal_advanced → _setup_subgoal_detection +
        # _detect_target) are handled by _recycle() after this
        # method returns True.

        logger.info(f"  Recycle: {len(state.subgoals)} new subgoals")
        for i, r in enumerate(results):
            logger.info(
                f"    [{i + 1}] {r['instruction']} "
                f"| detect: '{r.get('gdino_prompt', '?')}'"
            )

        state.log({
            "type": "recycle",
            "recycle_count": self._recycle_count,
            "new_subgoals": [r["instruction"] for r in results],
        })
        return True

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _on_frame(self, obs, state):
        self._edit_state.chunks_since_detect += 1
        if self._edit_state.chunks_since_detect >= self.config.requery_interval:
            self._detect_target(obs)
            self._edit_state.chunks_since_detect = 0

    def _on_subgoal_advanced(self, obs, state, idx):
        self._setup_subgoal_detection(state, idx)
        self._detect_target(obs)

    def _post_process(self, obs, state):
        return self._apply_edits(obs, state)

    # ------------------------------------------------------------------
    # GDino detection + subgoal setup
    # ------------------------------------------------------------------

    def _setup_subgoal_detection(self, state, idx):
        """Configure GDino for the target of subgoal *idx*."""
        targets = getattr(state, '_subgoal_targets', [])
        if idx < len(targets):
            t = targets[idx]
            self._edit_state = SubgoalEditState(
                gdino_prompt=t.get("gdino_prompt", ""),
                color_filter=t.get("color_filter"),
            )
        else:
            self._edit_state = SubgoalEditState()
        logger.info(
            f"  Detection target: '{self._edit_state.gdino_prompt}' "
            f"color={self._edit_state.color_filter}"
        )

    def _detect_target(self, obs):
        """Run GroundingDINO to find the target object."""
        prompt = self._edit_state.gdino_prompt
        if not prompt or prompt == ".":
            self._edit_state.exterior_bbox = None
            return

        ext_image = self.ctx.get_vlm_image(obs)
        if ext_image is not None:
            t0 = time.time()
            det = self.gdino.detect_best(
                ext_image, text_prompt=prompt,
                score_threshold=self.config.gdino_score_threshold,
                color_filter=self._edit_state.color_filter,
            )
            elapsed = time.time() - t0
            if det is not None:
                b = det["box"]
                self._edit_state.exterior_bbox = BBox(
                    b[0], b[1], b[2], b[3],
                )
                logger.debug(
                    f"  GDino ext: {self._edit_state.exterior_bbox} "
                    f"score={det['score']:.3f} ({elapsed * 1000:.0f}ms)"
                )
            else:
                self._edit_state.exterior_bbox = None
                logger.debug(
                    f"  GDino ext: not found ({elapsed * 1000:.0f}ms)"
                )

        if self.config.edit_wrist:
            extras = self.ctx.get_vlm_extra_images(obs)
            if extras:
                det = self.gdino.detect_best(
                    extras[0], text_prompt=prompt,
                    score_threshold=self.config.gdino_score_threshold,
                    color_filter=self._edit_state.color_filter,
                )
                if det:
                    b = det["box"]
                    self._edit_state.wrist_bbox = BBox(
                        b[0], b[1], b[2], b[3],
                    )
                else:
                    self._edit_state.wrist_bbox = None
        else:
            self._edit_state.wrist_bbox = None

    # ------------------------------------------------------------------
    # Scene editing
    # ------------------------------------------------------------------

    def _apply_edit_to_image(self, image):
        """Apply scene edit to a single high-res image for VLM input."""
        if self._edit_state.exterior_bbox is None:
            return image
        return apply_edit(
            image, self._edit_state.exterior_bbox, self.config.edit_mode,
        )

    def _apply_edits(self, obs, state):
        """Edit policy images with highlight/dim around detected target."""
        obs = dict(obs)
        state.frame_edited = False
        state.edit_bbox_dict = None
        state.edit_mode = "none"

        # Exterior camera
        ext_img = obs.get(self.ctx.image_key)
        if (isinstance(ext_img, np.ndarray)
                and self._edit_state.exterior_bbox is not None):
            vlm_img = self.ctx.get_vlm_image(obs)
            if (vlm_img is not None
                    and vlm_img.shape != ext_img.shape):
                bbox = self._edit_state.exterior_bbox.scale_with_pad(
                    vlm_img.shape[0], vlm_img.shape[1],
                    ext_img.shape[0], ext_img.shape[1],
                )
            else:
                bbox = self._edit_state.exterior_bbox

            obs[self.ctx.image_key] = apply_edit(
                ext_img, bbox, self.config.edit_mode,
            )
            state.frame_edited = True
            state.edit_bbox_dict = bbox.to_dict()
            state.edit_mode = self.config.edit_mode

        # Wrist camera
        if (self.config.edit_wrist
                and self._edit_state.wrist_bbox is not None):
            for key in self.ctx.extra_image_keys:
                wrist_img = obs.get(key)
                if isinstance(wrist_img, np.ndarray):
                    vlm_wrist = obs.get(key + "_raw")
                    if (isinstance(vlm_wrist, np.ndarray)
                            and vlm_wrist.shape != wrist_img.shape):
                        wbbox = self._edit_state.wrist_bbox.scale_with_pad(
                            vlm_wrist.shape[0], vlm_wrist.shape[1],
                            wrist_img.shape[0], wrist_img.shape[1],
                        )
                    else:
                        wbbox = self._edit_state.wrist_bbox
                    obs[key] = apply_edit(
                        wrist_img, wbbox, self.config.edit_mode,
                    )

        # Save debug images on first frame of each subgoal
        if (self.config.save_debug_images
                and self.config.debug_image_dir
                and self._edit_state.chunks_since_detect == 0):
            self._save_debug(obs, state)

        return obs, state

    def _save_debug(self, obs, state):
        debug_dir = self.config.debug_image_dir
        os.makedirs(debug_dir, exist_ok=True)
        prefix = (
            f"ep{state.episode_id}_sg{state.current_subgoal_idx}"
        )
        vlm_img = self.ctx.get_vlm_image(obs)
        if (vlm_img is not None
                and self._edit_state.exterior_bbox is not None):
            edited = apply_edit(
                vlm_img, self._edit_state.exterior_bbox,
                self.config.edit_mode,
            )
            h, w = vlm_img.shape[:2]
            comparison = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
            comparison[:, :w] = vlm_img
            comparison[:, w + 4:] = edited
            comparison[:, w:w + 4] = 128
            save_debug_image(
                comparison,
                os.path.join(debug_dir, f"{prefix}_comparison.png"),
            )

    # ------------------------------------------------------------------
    # Target identification helpers
    # ------------------------------------------------------------------

    def _identify_target(self, subgoal):
        """Ask VLM to produce a GDino prompt for a subgoal's target."""
        prompt = IDENTIFY_TARGET_PROMPT.format(subgoal=subgoal)
        try:
            raw = self._vlm_call(
                "You are a helpful assistant.",
                [{"type": "text", "text": prompt}],
            )
            data = parse_json(raw)
            gdino_prompt = data.get("gdino_prompt", "object") + "."
            color = data.get("color_filter")
            if color == "null" or color is None:
                color = None
            return {"gdino_prompt": gdino_prompt, "color_filter": color}
        except Exception as e:
            logger.warning(f"  Identify target failed: {e}")
            return {"gdino_prompt": "object.", "color_filter": None}

    @staticmethod
    def _guess_color(target_str: str) -> str | None:
        """Guess dominant color from target description."""
        colors = [
            "red", "green", "blue", "yellow", "orange", "white",
            "black", "grey", "gray", "pink", "brown", "purple",
        ]
        target_lower = target_str.lower()
        for c in colors:
            if c in target_lower:
                return c
        return None

    @staticmethod
    def _parse_subgoals(subgoals_raw: list) -> list[dict]:
        """Parse subgoal entries (string or dict) into uniform dicts."""
        results = []
        for sg in subgoals_raw:
            if isinstance(sg, str):
                results.append({
                    "instruction": sg,
                    "gdino_prompt": "",
                    "color_filter": None,
                })
            elif isinstance(sg, dict):
                instr = sg.get("instruction", str(sg))
                target = sg.get("target_object", "")
                results.append({
                    "instruction": instr,
                    "gdino_prompt": (target + ".") if target else "",
                    "color_filter": (
                        SubgoalSceneEditStrategy._guess_color(target)
                        if target else None
                    ),
                })
            else:
                results.append({
                    "instruction": str(sg),
                    "gdino_prompt": "",
                    "color_filter": None,
                })
        return results
