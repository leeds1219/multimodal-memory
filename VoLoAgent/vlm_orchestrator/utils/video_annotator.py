# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Annotated video writer for LIBERO (and other) evaluation episodes.

Composites orchestrator metadata (current subgoal, failure events,
scene-edit bounding boxes, grasp debug overlays) on top of raw
environment frames to produce annotated replay videos.

The annotation layout mirrors robolab's viewport style::

    ┌────────────────────────────────────────┐
    │  ┌──────────────────────┐  ┌────────┐ │
    │  │                      │  │ grasp  │ │
    │  │   main camera view   │  │ debug  │ │
    │  │   (with bbox overlay)│  │ PIP    │ │
    │  │                      │  └────────┘ │
    │  └──────────────────────┘              │
    │  step: 123  |  subgoal 2/3            │
    │  ▶ Pick up the tomato sauce …          │
    │  ⚠ GT:object_dropped (step 80)        │
    └────────────────────────────────────────┘

Usage::

    from vlm_orchestrator.utils.video_annotator import EpisodeVideoWriter

    writer = EpisodeVideoWriter(
        path="results/episode_0_annotated.mp4",
        fps=10,
        width=512,
        height=512,
    )
    # After each proxy response:
    writer.add_frame(raw_image, step=t, response=response)
    # At end of episode:
    writer.close(success=True)
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import cv2
import imageio
import numpy as np

logger = logging.getLogger(__name__)

# ── Colours (BGR for cv2) ──
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (0, 0, 220)
YELLOW = (0, 220, 220)
CYAN = (220, 200, 0)
ORANGE = (0, 140, 255)
PANEL_BG = (40, 40, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN


def _put_text(img, text, x, y, scale=0.5, color=WHITE, thickness=1, font=FONT):
    """Put anti-aliased text on image."""
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _wrap_text(text: str, max_chars: int = 60) -> list[str]:
    """Wrap text to fit within max_chars per line."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 > max_chars:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines


class EpisodeVideoWriter:
    """Accumulates annotated frames and writes an mp4 on close.

    Parameters
    ----------
    path : str
        Output .mp4 file path.
    fps : int
        Frames per second.
    canvas_w, canvas_h : int
        Output video resolution.  The main camera image is scaled to
        fit the left portion; the right column holds grasp debug PIP.
    """

    PAUSE_DURATION_S = 1.5
    PAUSE_TEXT_SCALE = 1.0

    def __init__(
        self,
        path: str,
        fps: int = 10,
        canvas_w: int = 640,
        canvas_h: int = 480,
    ):
        self.path = path
        self.fps = fps
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self._pause_n_frames = max(1, round(self.PAUSE_DURATION_S * fps))

        self._frames: list[np.ndarray] = []

        # Persistent state across frames
        self._current_instruction: str | None = None
        self._subgoals: list[str] | None = None
        self._subgoal_idx: int = 0
        self._num_subgoals: int = 0
        self._last_failure: dict | None = None
        self._failure_step: int = -1
        self._failure_display_until: int = -1  # show failure for N frames
        self._grasp_active: bool = False
        self._grasp_phase: str = ""
        self._grasp_target: str = ""

        # Pause-trigger tracking (separate from normal-flow state so pause
        # detection doesn't disturb existing change-detection logic).
        self._pt_last_vlm_check_step: int = -1
        self._pt_last_failure_step: int = -1
        self._pt_last_subgoal_idx: int = -1
        self._pt_last_grasp_jpg: bytes | None = None
        self._pt_last_subgoals: tuple | None = None

    def add_frame(
        self,
        raw_image: np.ndarray,
        step: int,
        response: dict[str, Any] | None = None,
        *,
        raw_wrist_image: np.ndarray | None = None,
    ) -> None:
        """Add one annotated frame.

        Parameters
        ----------
        raw_image : np.ndarray
            RGB image from the main camera (any resolution).
        step : int
            Current environment step number.
        response : dict
            The full response dict from the orchestrator proxy, containing
            ``orchestrator_*`` metadata keys.
        raw_wrist_image : np.ndarray, optional
            If provided, shown as a small PIP in the bottom-right.
        """
        if response is None:
            response = {}

        # ── Emit pause-on-event holds before the regular frame ──
        pause_events = self._collect_pause_events(response)
        for ev_type, ev_data in pause_events:
            paused = self._build_paused_frame(
                raw_image, response, ev_type, ev_data, step,
            )
            for _ in range(self._pause_n_frames):
                self._frames.append(paused.copy())

        # The orchestrator builds a queue of debug images during
        # PERCEIVING (detection → mask → depth → grasp-pose, all
        # captured in one synchronous call), then drips them out
        # across the subsequent LIFTING / APPROACHING chunks
        # (proxy.py:710-755 holds each image for ~2 chunks before
        # advancing to the next).  Per-chunk, the response always
        # carries ``orchestrator_grasp_debug_jpg`` while ANY image
        # is still being shown.  We want detection → mask → depth →
        # grasp-pose to play back-to-back without any robot-camera
        # frames interleaved.
        #
        # Suppress whenever a debug-jpg is present in the response —
        # the "image is currently being shown" signal — not only on
        # the chunks where the jpg changes.  This covers both the
        # changeover chunks (a fresh pause fires) AND the hold
        # chunks (same jpg as before, no pause but still no robot
        # motion wanted).  Once the queue empties, the jpg is gone
        # from the response and motion resumes.
        debug_jpg_active = bool(response.get("orchestrator_grasp_debug_jpg"))
        suppress_regular_frame = debug_jpg_active

        # ── Update persistent state from response ──
        if "orchestrator_instruction" in response:
            self._current_instruction = response["orchestrator_instruction"]
        if "orchestrator_subgoals" in response:
            self._subgoals = response["orchestrator_subgoals"]
            self._num_subgoals = len(self._subgoals)
        if "orchestrator_subgoal_idx" in response:
            self._subgoal_idx = response["orchestrator_subgoal_idx"]
        if "orchestrator_failure" in response:
            self._last_failure = response["orchestrator_failure"]
            self._failure_step = step
            self._failure_display_until = step + 50  # show for ~50 frames
        if "orchestrator_grasp_tool" in response:
            gt = response["orchestrator_grasp_tool"]
            self._grasp_active = gt.get("active", False)
            self._grasp_phase = gt.get("phase", "")
            self._grasp_target = gt.get("target", "")
        elif self._grasp_active:
            # Grasp tool no longer in response → deactivated
            self._grasp_active = False

        # ── Build canvas ──
        canvas = np.full((self.canvas_h, self.canvas_w, 3), PANEL_BG, dtype=np.uint8)

        # Layout: main image takes ~75% width, info panel takes bottom
        main_w = self.canvas_w
        main_h = self.canvas_h - 100  # reserve 100px at bottom for text
        pip_size = 0

        # Grasp debug PIP in top-right corner
        grasp_debug_img = None
        if "orchestrator_grasp_debug_jpg" in response:
            try:
                jpg_bytes = response["orchestrator_grasp_debug_jpg"]
                grasp_debug_img = cv2.imdecode(
                    np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                grasp_debug_img = cv2.cvtColor(grasp_debug_img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        if grasp_debug_img is not None:
            pip_size = min(main_h // 3, main_w // 3)
            main_w = self.canvas_w - pip_size - 4

        # ── Draw main camera image ──
        img_rgb = raw_image
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

        # Scale to fit main area
        h, w = img_rgb.shape[:2]
        scale = min(main_w / w, main_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Draw scene-edit bounding boxes on the scaled image
        if response.get("orchestrator_frame_edited") and "orchestrator_edit_bbox" in response:
            self._draw_bboxes(img_scaled, response["orchestrator_edit_bbox"], scale)

        # Center the main image
        x_off = (main_w - new_w) // 2
        y_off = (main_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img_scaled

        # ── Draw grasp debug PIP ──
        if grasp_debug_img is not None and pip_size > 0:
            pip_resized = cv2.resize(grasp_debug_img, (pip_size, pip_size))
            px = self.canvas_w - pip_size - 2
            py = 2
            canvas[py:py + pip_size, px:px + pip_size] = pip_resized
            # Label
            label = response.get("orchestrator_grasp_debug_label", "grasp debug")
            _put_text(canvas, label[:30], px, py + pip_size + 14,
                      scale=0.35, color=CYAN, thickness=1)

        # ── Draw wrist PIP (bottom-right of main area) ──
        if raw_wrist_image is not None and pip_size == 0:
            wrist_pip = min(main_h // 4, 80)
            wrist_resized = cv2.resize(raw_wrist_image, (wrist_pip, wrist_pip))
            wx = main_w - wrist_pip - 4
            wy = main_h - wrist_pip - 4
            canvas[wy:wy + wrist_pip, wx:wx + wrist_pip] = wrist_resized
            cv2.rectangle(canvas, (wx - 1, wy - 1),
                          (wx + wrist_pip, wy + wrist_pip), WHITE, 1)

        # ── Draw info panel at bottom ──
        panel_y = self.canvas_h - 98
        cv2.rectangle(canvas, (0, panel_y), (self.canvas_w, self.canvas_h), PANEL_BG, -1)
        # Top line separator
        cv2.line(canvas, (0, panel_y), (self.canvas_w, panel_y), (80, 80, 80), 1)

        text_x = 8
        text_y = panel_y + 16

        # Step counter + subgoal index
        step_text = f"step: {step}"
        if self._num_subgoals > 0:
            step_text += f"  |  subgoal {self._subgoal_idx + 1}/{self._num_subgoals}"
        if self._grasp_active:
            step_text += f"  |  GRASP [{self._grasp_phase}]"
        _put_text(canvas, step_text, text_x, text_y, scale=0.45, color=(180, 180, 180))
        text_y += 20

        # Current instruction
        if self._current_instruction:
            instr_lines = _wrap_text(self._current_instruction, max_chars=80)
            for line in instr_lines[:2]:  # max 2 lines
                _put_text(canvas, f"> {line}", text_x, text_y, scale=0.45, color=GREEN)
                text_y += 18

        # Failure event
        if self._last_failure and step <= self._failure_display_until:
            reason = self._last_failure
            if isinstance(reason, dict):
                reason = reason.get("reason", str(reason))
            reason_str = str(reason)[:90]
            gt_type = ""
            if isinstance(self._last_failure, dict):
                gt_type = self._last_failure.get("gt_failure_type", "")

            color = RED
            icon = "!!"
            if "regression" in gt_type:
                color = ORANGE
                icon = "<<"
            elif "dropped" in gt_type:
                color = YELLOW
                icon = "vv"
            elif "wrong" in gt_type:
                color = RED
                icon = "XX"
            elif "complete" in gt_type:
                color = GREEN
                icon = "OK"

            _put_text(canvas, f"[{icon}] {reason_str}", text_x, text_y,
                      scale=0.38, color=color)
            text_y += 16

        # Success/failure flash at episode end (handled in close())

        # Convert to RGB for imageio.  Skip the robot-camera frame between
        # back-to-back grasp-debug pauses when the tool is still in
        # pre-motion phases (see comment near pause_events above).
        if not suppress_regular_frame:
            self._frames.append(canvas)

    def _collect_pause_events(self, response: dict) -> list[tuple[str, Any]]:
        """Detect newly-fired reasoning events worth pausing on."""
        events: list[tuple[str, Any]] = []

        # Decomposition: subgoal list first appears or changes (recycle).
        subgoals = response.get("orchestrator_subgoals")
        if subgoals:
            sg_tuple = tuple(subgoals)
            if sg_tuple != self._pt_last_subgoals:
                ev = {
                    "subgoals": list(subgoals),
                    "instruction": response.get("orchestrator_original_instruction"),
                    "is_recycle": self._pt_last_subgoals is not None,
                }
                events.append(("decompose", ev))
                self._pt_last_subgoals = sg_tuple

        check = response.get("orchestrator_vlm_check")
        check_step = response.get("orchestrator_vlm_check_step", -1)
        if check and check_step != self._pt_last_vlm_check_step:
            self._pt_last_vlm_check_step = check_step
            events.append(("vlm_check", check))

        failure = response.get("orchestrator_failure")
        failure_step = response.get("orchestrator_failure_step", -1)
        if failure and failure_step != self._pt_last_failure_step:
            self._pt_last_failure_step = failure_step
            events.append(("failure", failure))

        sg_idx = response.get("orchestrator_subgoal_idx")
        if sg_idx is not None and sg_idx != self._pt_last_subgoal_idx:
            # Skip the very first appearance (initial idx, not an "advance").
            if self._pt_last_subgoal_idx != -1:
                events.append(("subgoal", {
                    "idx": sg_idx,
                    "subgoals": response.get("orchestrator_subgoals"),
                }))
            self._pt_last_subgoal_idx = sg_idx

        jpg_bytes = response.get("orchestrator_grasp_debug_jpg")
        if jpg_bytes is not None and jpg_bytes != self._pt_last_grasp_jpg:
            self._pt_last_grasp_jpg = jpg_bytes
            events.append(("grasp", {
                "jpg": jpg_bytes,
                "label": response.get("orchestrator_grasp_debug_label", ""),
            }))

        return events

    def _build_paused_frame(
        self,
        raw_image: np.ndarray,
        response: dict,
        ev_type: str,
        ev_data: Any,
        step: int,
    ) -> np.ndarray:
        """Build a single paused frame for a reasoning event.

        For grasp events the main camera area is replaced with the grasp
        debug image at full size; otherwise the held scene is kept.
        Text overlays are rendered at PAUSE_TEXT_SCALE×.
        """
        canvas = np.full(
            (self.canvas_h, self.canvas_w, 3), PANEL_BG, dtype=np.uint8,
        )

        # Reserve a taller info panel for the enlarged event text.
        panel_h = int(100 * self.PAUSE_TEXT_SCALE)
        main_h = self.canvas_h - panel_h
        main_w = self.canvas_w

        # Choose what to put in the main area.
        if ev_type == "grasp":
            jpg_bytes = ev_data.get("jpg")
            try:
                main_img = cv2.imdecode(
                    np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR,
                )
                main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
            except Exception:
                main_img = raw_image
        else:
            main_img = raw_image
        if main_img.ndim == 2:
            main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2RGB)

        # Letterbox-fit into the main area.
        h, w = main_img.shape[:2]
        s = min(main_w / w, main_h / h)
        new_w, new_h = int(w * s), int(h * s)
        scaled = cv2.resize(main_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x_off = (main_w - new_w) // 2
        y_off = (main_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled

        # Pause badge in the top-left of the main area.
        badge = self._pause_badge(ev_type)
        _put_text(
            canvas, badge, 8, 22,
            scale=0.7, color=(0, 220, 255), thickness=2,
        )

        # Info panel: enlarged step/instruction/event text.
        panel_y = self.canvas_h - panel_h
        cv2.rectangle(
            canvas, (0, panel_y), (self.canvas_w, self.canvas_h),
            PANEL_BG, -1,
        )
        cv2.line(canvas, (0, panel_y), (self.canvas_w, panel_y), (80, 80, 80), 1)

        big_scale = 0.45 * self.PAUSE_TEXT_SCALE
        big_thick = max(1, round(self.PAUSE_TEXT_SCALE))
        line_h = int(20 * self.PAUSE_TEXT_SCALE)
        text_x = 8
        text_y = panel_y + int(18 * self.PAUSE_TEXT_SCALE)

        step_text = f"step: {step}"
        if self._num_subgoals > 0:
            step_text += f"  |  subgoal {self._subgoal_idx + 1}/{self._num_subgoals}"
        _put_text(
            canvas, step_text, text_x, text_y,
            scale=big_scale, color=(180, 180, 180), thickness=big_thick,
        )
        text_y += line_h

        # Wrap each event text line at a generous width given the bigger font.
        max_chars = max(20, int(80 / self.PAUSE_TEXT_SCALE))
        for line in self._pause_event_lines(ev_type, ev_data):
            for wrapped in _wrap_text(line, max_chars=max_chars):
                if text_y + line_h > self.canvas_h:
                    break
                _put_text(
                    canvas, wrapped, text_x, text_y,
                    scale=big_scale, color=YELLOW, thickness=big_thick,
                )
                text_y += line_h

        return canvas

    @staticmethod
    def _pause_badge(ev_type: str) -> str:
        return {
            "decompose": "[||] PAUSED -- DECOMPOSE",
            "vlm_check": "[||] PAUSED -- VLM CHECK",
            "failure": "[||] PAUSED -- FAILURE",
            "subgoal": "[||] PAUSED -- SUBGOAL ADVANCE",
            "grasp": "[||] PAUSED -- GRASP TOOL",
        }.get(ev_type, "[||] PAUSED")

    @staticmethod
    def _pause_event_lines(ev_type: str, ev_data: Any) -> list[str]:
        if ev_type == "decompose":
            sgs = ev_data.get("subgoals") or []
            tag = "RECYCLE" if ev_data.get("is_recycle") else "DECOMPOSE"
            out = [f"{tag} -> {len(sgs)} subgoals:"]
            for i, sg in enumerate(sgs, 1):
                out.append(f"  [{i}] {sg}")
            return out
        if ev_type == "vlm_check":
            done = ev_data.get("done", False)
            action = ev_data.get("action", "continue")
            if done:
                lines = ["VLM check: DONE -> next subgoal"]
            elif action == "refine":
                lines = [f'VLM check: REFINE -> "{ev_data.get("instruction", "")}"']
            elif action == "next_goal":
                lines = [f'Next goal: "{ev_data.get("instruction", "")}"']
            else:
                lines = ["VLM check: NOT DONE"]
            status = ev_data.get("status")
            if status and status != action:
                lines.append(f"status: {status}")
            reason = ev_data.get("reason")
            if reason:
                lines.append(f"why: {reason}")
            return lines
        if ev_type == "failure":
            reason = ev_data.get("reason", "unknown") if isinstance(ev_data, dict) else str(ev_data)
            out = [f"FAILURE: {reason}"]
            if isinstance(ev_data, dict):
                conf = ev_data.get("confidence")
                mode = ev_data.get("recovery_mode")
                if conf is not None or mode:
                    out.append(f"conf={conf}  recovery={mode}")
                rationale = ev_data.get("rationale")
                if rationale:
                    out.append(f"why: {rationale}")
            return out
        if ev_type == "subgoal":
            sg_list = ev_data.get("subgoals") or []
            idx = ev_data["idx"]
            out = [f"SUBGOAL ADVANCE -> {idx + 1}/{len(sg_list)}"]
            if sg_list and 0 <= idx < len(sg_list):
                out.append(f"now: {sg_list[idx]}")
            return out
        if ev_type == "grasp":
            label = ev_data.get("label") or "grasp"
            return [f"GRASP TOOL: {label}"]
        return []

    def _draw_bboxes(self, img: np.ndarray, bbox_dict: dict, scale: float):
        """Draw scene-edit bounding boxes on the scaled image."""
        try:
            for obj_name, bbox_info in bbox_dict.items():
                if isinstance(bbox_info, (list, tuple)) and len(bbox_info) == 4:
                    x1, y1, x2, y2 = [int(v * scale) for v in bbox_info]
                elif isinstance(bbox_info, dict):
                    x1 = int(bbox_info.get("x1", 0) * scale)
                    y1 = int(bbox_info.get("y1", 0) * scale)
                    x2 = int(bbox_info.get("x2", 0) * scale)
                    y2 = int(bbox_info.get("y2", 0) * scale)
                else:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
                label = str(obj_name)[:20]
                _put_text(img, label, x1, max(y1 - 5, 12),
                          scale=0.35, color=GREEN, thickness=1)
        except Exception as e:
            logger.debug(f"Failed to draw bboxes: {e}")

    def close(self, success: bool | None = None) -> str | None:
        """Finalize and write the annotated video.

        Parameters
        ----------
        success : bool, optional
            If provided, the last few frames get a SUCCESS/FAILURE banner.

        Returns
        -------
        str or None
            Path to the written video, or None if no frames were collected.
        """
        if not self._frames:
            logger.warning("No frames to write")
            return None

        # Add success/failure banner to last 5 frames
        if success is not None:
            banner_text = "SUCCESS" if success else "FAILURE"
            banner_color = GREEN if success else RED
            n_banner = min(5, len(self._frames))
            for i in range(len(self._frames) - n_banner, len(self._frames)):
                frame = self._frames[i]
                # Semi-transparent overlay
                overlay = frame.copy()
                h, w = frame.shape[:2]
                cv2.rectangle(overlay, (w // 4, h // 3), (3 * w // 4, 2 * h // 3),
                              BLACK, -1)
                frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                # Text
                (tw, th), _ = cv2.getTextSize(banner_text, FONT, 1.5, 3)
                tx = (w - tw) // 2
                ty = h // 2 + th // 2
                _put_text(frame, banner_text, tx, ty, scale=1.5,
                          color=banner_color, thickness=3)

        try:
            imageio.mimwrite(
                self.path,
                self._frames,
                fps=self.fps,
                codec="libx264",
                output_params=["-crf", "23", "-preset", "fast"],
            )
            logger.info(f"Annotated video saved: {self.path} ({len(self._frames)} frames)")
            return self.path
        except Exception as e:
            # Fallback without codec params
            try:
                imageio.mimwrite(self.path, self._frames, fps=self.fps)
                logger.info(f"Annotated video saved (fallback): {self.path}")
                return self.path
            except Exception as e2:
                logger.error(f"Failed to write annotated video: {e2}")
                return None
        finally:
            self._frames.clear()


class RawVideoWriter:
    """Simple raw replay video writer (no annotations)."""

    def __init__(self, path: str, fps: int = 10):
        self.path = path
        self.fps = fps
        self._frames: list[np.ndarray] = []

    def add_frame(self, image: np.ndarray) -> None:
        self._frames.append(np.asarray(image))

    def close(self) -> str | None:
        if not self._frames:
            return None
        try:
            imageio.mimwrite(self.path, self._frames, fps=self.fps)
            logger.info(f"Raw video saved: {self.path} ({len(self._frames)} frames)")
            return self.path
        except Exception as e:
            logger.error(f"Failed to write raw video: {e}")
            return None
        finally:
            self._frames.clear()
