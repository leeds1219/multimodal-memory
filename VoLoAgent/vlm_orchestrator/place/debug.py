# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug visualization for the place-with-tool pipeline.

Mirrors ``vlm_orchestrator/grasp/debug.py``: saves annotated images and a
per-attempt ``place_log.json`` to disk, and stashes the images on the
SessionState so the proxy can forward them to the eval client as
picture-in-picture in the annotated video.

Save location resolution (in order):
1. ``state.episode_log_dir / "debug_place"`` when a SessionState with an
   ``episode_log_dir`` is set (the proxy populates this under
   ``$LOG_DIR/<task_slug>/episode_<N>/``).  Artefacts persist alongside
   ``rewrites.jsonl`` and survive a container exit on a shared filesystem.
2. ``~/vlm-orchestrator/debug_place/`` (legacy local fallback for unit
   tests / interactive runs where no per-episode log dir is wired).

Saved artefacts (all timestamped):

1. ``HHMMSS_1_destination.jpg`` — destination-detection result (red dot
   for the chosen 2D pixel + label naming the source mode).
2. ``HHMMSS_2_target_world.jpg`` — projected 3D target onto the image
   plane after raycast / GT-sim resolution.
3. ``HHMMSS_3_post_place.jpg`` — post-release scene with EE delta.
4. ``HHMMSS_place_log.json`` — accumulated diagnostics for the run.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from vlm_orchestrator.viz import gripper_overlay as _ov

logger = logging.getLogger(__name__)

# Legacy local default — used only when no per-episode log dir is wired.
_DEFAULT_DEBUG_DIR = Path(os.path.expanduser("~/vlm-orchestrator/debug_place"))

# Module-level handle to the current SessionState (set by the executor
# before running the pipeline so vis helpers can stash images).
_current_state = None
_pip_x_flip: bool = False


def save_place_log(data: dict) -> None:
    """Write per-place diagnostic data to a JSON file under debug_place/.

    Call once at the end of a placement attempt with all accumulated values.
    The file is timestamped to match the other debug images from the same run.
    """
    d = _ensure_dir()
    path = d / f"{_timestamp()}_place_log.json"
    import json
    with open(path, "w") as f:
        json.dump(
            data, f, indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
    logger.info(f"  [debug] Place log saved: {path}")


def set_state(state, *, pip_x_flip: bool = False) -> None:
    """Set the session state + PIP-flip flag for debug image stashing."""
    global _current_state, _pip_x_flip
    _current_state = state
    _pip_x_flip = bool(pip_x_flip)


def _stash(image: np.ndarray, label: str) -> None:
    """Append *image* + *label* to the place_debug_queue on SessionState."""
    if _current_state is not None:
        img = image
        if _pip_x_flip:
            img = np.ascontiguousarray(img[:, ::-1])
        else:
            img = img.copy()
        _current_state.place_debug_queue.append((img, label))


def _ensure_dir():
    """Resolve the debug-dir based on the current SessionState.

    Prefers ``state.episode_log_dir / debug_place`` (under the
    user-specified ``--log-dir``) so artefacts persist across workflow
    exits.  Falls back to ``~/vlm-orchestrator/debug_place`` only when
    no episode log dir is wired (unit tests, ad-hoc runs).
    """
    ep_dir = getattr(_current_state, "episode_log_dir", None)
    if ep_dir:
        d = Path(ep_dir) / "debug_place"
    else:
        d = _DEFAULT_DEBUG_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _timestamp() -> str:
    return time.strftime("%H%M%S")


_DISK_MIN_LONG_SIDE = 768


def _imwrite_upscaled(path, image_rgb: np.ndarray) -> None:
    """Upscale (nearest-neighbour) and write *image_rgb* to *path* as JPG."""
    h, w = image_rgb.shape[:2]
    long_side = max(h, w)
    if long_side < _DISK_MIN_LONG_SIDE:
        scale = int(np.ceil(_DISK_MIN_LONG_SIDE / long_side))
        out = cv2.resize(
            image_rgb, (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        out = image_rgb
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _font_metrics(image: np.ndarray) -> tuple[float, int, int]:
    """Return ``(font_scale, thickness, line_height)`` sized for the image."""
    h = image.shape[0]
    ref_h = 480.0
    scale = max(0.30, min(0.9, 0.8 * h / ref_h))
    thickness = max(1, int(round(2 * h / ref_h)))
    line_h = int(round(28 * h / ref_h))
    return scale, thickness, line_h


# ------------------------------------------------------------------
# 1. Destination detection result (2D pixel + source mode)
# ------------------------------------------------------------------

def vis_destination_2d(
    image: np.ndarray,
    point_2d_norm: tuple[float, float],
    *,
    source: str,
    target_phrase: str,
    confidence: float = 1.0,
    rationale: str | None = None,
    hitl=None,
) -> np.ndarray:
    """Draw a red dot at the chosen 2D pixel + label naming the source mode.

    ``point_2d_norm`` is normalized image coords ``(x, y)`` in ``[0, 1]``.
    """
    d = _ensure_dir()
    vis = image.copy()
    h, w = vis.shape[:2]
    px = int(round(point_2d_norm[0] * w))
    py = int(round(point_2d_norm[1] * h))
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))

    cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.circle(vis, (px, py), 14, (0, 0, 255), 2)
    cv2.circle(vis, (px, py), 4, (255, 255, 255), -1)

    fscale, fthick, line_h = _font_metrics(vis)
    label = f"[{source}] {target_phrase} ({confidence:.2f})"
    cv2.putText(
        vis, label, (10, line_h),
        cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), fthick,
    )
    if rationale:
        # Wrap rationale to fit the image width
        max_chars = max(20, int(w / max(8, fscale * 16)))
        for i, line in enumerate(_wrap(rationale, max_chars)[:3]):
            cv2.putText(
                vis, line, (10, line_h * (2 + i)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.30, fscale * 0.8),
                (255, 255, 0), max(1, fthick - 1),
            )

    path = d / f"{_timestamp()}_1_destination.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] Destination saved: {path}  ({label})")

    _stash(vis, f"Place dest: {label}")

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"📍 {label}")

    return vis


# ------------------------------------------------------------------
# 2. Target world point projected back onto the image
# ------------------------------------------------------------------

def vis_target_world(
    image: np.ndarray,
    target_world: np.ndarray,
    cam_to_world: np.ndarray,
    intrinsics,
    *,
    relation: str = "in",
    held_object_height_m: float | None = None,
    hitl=None,
) -> np.ndarray:
    """Project the resolved 3D target back onto the image for sanity check."""
    d = _ensure_dir()
    vis = image.copy()
    h, w = vis.shape[:2]

    world_to_cam = np.linalg.inv(cam_to_world)
    target_cam = world_to_cam[:3, :3] @ target_world + world_to_cam[:3, 3]
    if target_cam[2] > 0:
        u = int(intrinsics.fx * target_cam[0] / target_cam[2] + intrinsics.cx)
        v = int(intrinsics.fy * target_cam[1] / target_cam[2] + intrinsics.cy)
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        cv2.drawMarker(vis, (u, v), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 28, 3)
        cv2.circle(vis, (u, v), 18, (0, 255, 0), 2)

    fscale, fthick, line_h = _font_metrics(vis)
    extra = ""
    if held_object_height_m is not None:
        extra = f"  held_h={held_object_height_m * 100:.1f}cm"
    label = (
        f"target world=[{target_world[0]:.3f}, {target_world[1]:.3f}, "
        f"{target_world[2]:.3f}]  rel={relation}{extra}"
    )
    cv2.putText(
        vis, label, (10, line_h),
        cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 255, 0), fthick,
    )

    path = d / f"{_timestamp()}_2_target_world.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] Target world saved: {path}")

    _stash(vis, f"Place target: {label}")

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"🌍 {label}")

    return vis


# ------------------------------------------------------------------
# 3. Post-release scene with EE delta
# ------------------------------------------------------------------

def vis_post_place(
    image: np.ndarray,
    target_world: np.ndarray,
    ee_pos_world: np.ndarray,
    cam_to_world: np.ndarray,
    intrinsics,
    *,
    hitl=None,
    object_pos_world: np.ndarray | None = None,
    R_target_world: np.ndarray | None = None,
    R_actual_world: np.ndarray | None = None,
) -> np.ndarray:
    """Annotate the after-release scene with the target vs actual placement.

    Uses the shared gripper-overlay primitives so the look matches the grasp
    tool's post-move card:

      * TARGET  (predicted, cyan): where we aimed to release the held object —
        ``target_world`` with the chosen release rotation ``R_target_world``.
      * ACTUAL  (yellow): where the object actually ended up —
        ``object_pos_world`` (held-object centroid estimate; falls back to the
        raw EE position) with the final EE rotation ``R_actual_world``.

    The headline metric is the OBJECT landing error (object vs target), which
    is the physically meaningful quantity for placement — not a flange/EE
    frame.  A parallel-jaw claw is drawn at each point when the corresponding
    rotation is supplied.
    """
    d = _ensure_dir()
    vis = image.copy()
    h, w = vis.shape[:2]

    # Actual placement point = held-object centroid when available, else EE.
    actual_world = (
        np.asarray(object_pos_world)[:3]
        if object_pos_world is not None
        else np.asarray(ee_pos_world)[:3]
    )
    target_world = np.asarray(target_world)[:3]

    world_to_cam = np.linalg.inv(cam_to_world)

    def project(pt_world):
        pt_cam = world_to_cam[:3, :3] @ np.asarray(pt_world)[:3] + world_to_cam[:3, 3]
        if pt_cam[2] <= 0:
            return None
        u = int(intrinsics.fx * pt_cam[0] / pt_cam[2] + intrinsics.cx)
        v = int(intrinsics.fy * pt_cam[1] / pt_cam[2] + intrinsics.cy)
        return (max(0, min(w - 1, u)), max(0, min(h - 1, v)))

    fscale, fthick, line_h = _font_metrics(vis)

    # ── TARGET release pose: cyan gripper + centre dot ──
    t_uv = project(target_world)
    if t_uv is not None:
        if R_target_world is not None:
            _ov.draw_gripper(vis, project, target_world, R_target_world,
                             _ov.C_PRED, max(2, fthick))
        cv2.circle(vis, t_uv, max(3, fthick + 1), _ov.C_PRED, -1)

    # ── ACTUAL placement: yellow gripper + centre dot ──
    a_uv = project(actual_world)
    if a_uv is not None:
        if R_actual_world is not None:
            _ov.draw_gripper(vis, project, actual_world, R_actual_world,
                             _ov.C_ACTUAL, max(2, fthick))
        cv2.circle(vis, a_uv, max(3, fthick + 1), _ov.C_ACTUAL, -1)
        if t_uv is not None:
            cv2.line(vis, t_uv, a_uv, _ov.C_ACTUAL, max(1, fthick - 1))

    # ── Info panel ──
    delta = actual_world - target_world
    err_m = float(np.linalg.norm(delta))
    lateral = float(np.linalg.norm(delta[:2]))
    rows = [
        ("big", ("object landing err", f"{err_m * 100:.1f} cm")),
        ("kv", ("lateral / vertical",
                f"{lateral * 100:.1f} / {delta[2] * 100:+.1f} cm")),
    ]
    if R_target_world is not None and R_actual_world is not None:
        R_rel = np.asarray(R_target_world).T @ np.asarray(R_actual_world)
        tr = float(np.clip(np.trace(R_rel), -1.0, 3.0))
        rot_deg = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))))
        rows.append(("kv", ("release rotation err", f"{rot_deg:.0f} deg")))
    _ov.draw_info_panel(
        vis, fscale, fthick, line_h,
        title="PLACE ACCURACY  (after release)", rows=rows,
        legend=("target", "actual"),
    )

    path = d / f"{_timestamp()}_3_post_place.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] Post-place saved: {path}  (err={err_m * 100:.2f}cm)")

    _stash(vis, f"Post-place err={err_m * 100:.1f}cm")

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"🎯 object err={err_m * 100:.1f}cm")

    return vis


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _wrap(text: str, max_chars: int) -> list[str]:
    """Greedy word-wrap helper for rationale text overlays."""
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return lines


_VIS_PAUSE = 1.5


def _push_to_hitl(hitl, image_rgb: np.ndarray, status: str):
    """Push an image + status to the HITL web UI with a short pause."""
    try:
        hitl.update_image(image_rgb)
        hitl.update_status(status_message=status)
        time.sleep(_VIS_PAUSE)
    except Exception as e:
        logger.debug(f"  [debug] Could not push to HITL: {e}")
