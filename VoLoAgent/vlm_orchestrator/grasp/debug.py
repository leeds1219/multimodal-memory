# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Debug visualization for the grasp-with-tool pipeline.

Saves annotated images to disk at each stage and optionally pushes
them to the HITL web UI so the operator can inspect:

1. GDino detection (bbox overlay)
2. Mask (overlay on RGB)
3. Point cloud stats + masked region
4. Predicted grasp pose (projected axes on image)
5. Grasp pose in world frame + IK target

Images are also stashed on the ``SessionState`` so the proxy can
forward them to the eval client for picture-in-picture rendering
in the annotated video.

Save location resolution (in order, mirrors place/debug.py):
1. ``state.episode_log_dir / "debug_grasp"`` when a SessionState with an
   ``episode_log_dir`` is set (the proxy populates this under
   ``$LOG_DIR/<task_slug>/episode_<N>/``).  Artefacts persist alongside
   ``rewrites.jsonl`` and survive a container exit on a shared filesystem.
2. ``~/vlm-orchestrator/debug_grasp/`` (legacy local fallback for unit
   tests / interactive runs where no per-episode log dir is wired).
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
_DEFAULT_DEBUG_DIR = Path(os.path.expanduser("~/vlm-orchestrator/debug_grasp"))

# Module-level handle to the current SessionState (set by the executor
# before running the pipeline so vis helpers can stash images).
_current_state = None

# When True, ``_stash`` X-flips the image before pushing it into the
# session's grasp_debug_queue.  The grasp pipeline reads images via
# ``GraspToolExecutor._extract_image`` which un-mirrors LIBERO frames
# (``[:, ::-1]``) so K + extrinsic line up; the main video, however,
# shows the eval-client's 180°-rotated frame.  Re-applying the X-flip
# only for the PIP keeps it in the same orientation as the surrounding
# video.  Set per-session via ``set_state``.
_pip_x_flip: bool = False


def save_grasp_log(data: dict) -> None:
    """Write per-grasp diagnostic data to a JSON file under debug_grasp/.

    Call once at the end of a grasp attempt with all accumulated values.
    The file is timestamped to match the other debug images from the same run.
    """
    d = _ensure_dir()
    path = d / f"{_timestamp()}_grasp_log.json"
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    logger.info(f"  [debug] Grasp log saved: {path}")


def set_state(state, *, pip_x_flip: bool = False) -> None:
    """Set the session state + PIP-flip flag for debug image stashing.

    ``pip_x_flip`` should be ``True`` for environments where the grasp
    tool operates in a different image convention than the main eval
    video (LIBERO un-mirrors internally; eval video shows the rotated
    frame).  When ``True``, ``_stash`` re-applies an X-flip so the PIP
    matches the surrounding video.
    """
    global _current_state, _pip_x_flip
    _current_state = state
    _pip_x_flip = bool(pip_x_flip)


def _stash(image: np.ndarray, label: str) -> None:
    """Append *image* + *label* to the debug queue on SessionState."""
    if _current_state is not None:
        img = image
        if _pip_x_flip:
            img = np.ascontiguousarray(img[:, ::-1])
        else:
            img = img.copy()
        _current_state.grasp_debug_queue.append((img, label))


def _ensure_dir():
    """Resolve the debug-dir based on the current SessionState.

    Prefers ``state.episode_log_dir / debug_grasp`` (under the
    user-specified ``--log-dir``) so artefacts persist across workflow
    exits.  Falls back to ``~/vlm-orchestrator/debug_grasp`` only when
    no episode log dir is wired (unit tests, ad-hoc runs).
    """
    ep_dir = getattr(_current_state, "episode_log_dir", None)
    if ep_dir:
        d = Path(ep_dir) / "debug_grasp"
    else:
        d = _DEFAULT_DEBUG_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _timestamp() -> str:
    return time.strftime("%H%M%S")


# Disk-write upscale: small renders (LIBERO 256²) become unreadable.
# Upscale to at least this many pixels on the long side before imwrite
# so the saved JPG is legible.  Projection / annotation math runs at
# the native resolution; this only affects the saved file.
_DISK_MIN_LONG_SIDE = 768


def _imwrite_upscaled(path, image_rgb: np.ndarray) -> None:
    """Upscale (nearest-neighbour) and write *image_rgb* to *path* as JPG.

    Nearest-neighbour keeps drawn lines and text crisp.  Native-size
    images larger than ``_DISK_MIN_LONG_SIDE`` are written as-is.
    """
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
    """Return ``(font_scale, thickness, line_height)`` sized for the image.

    Robolab images are typically 1024+; LIBERO renders at 256². Without
    this scaling the text overflows the LIBERO frame and is unreadable.
    Reference: 0.8 scale + 2 thick at 480 px image height.
    """
    h = image.shape[0]
    ref_h = 480.0
    scale = max(0.30, min(0.9, 0.8 * h / ref_h))
    thickness = max(1, int(round(2 * h / ref_h)))
    line_h = int(round(28 * h / ref_h))
    return scale, thickness, line_h


# ------------------------------------------------------------------
# 1. Detection result
# ------------------------------------------------------------------

def vis_detection(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_object: str,
    score: float,
    hitl=None,
) -> np.ndarray:
    """Draw bounding box on image. Returns annotated copy."""
    d = _ensure_dir()
    vis = image.copy()
    x1, y1, x2, y2 = bbox

    # Draw bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Label
    fscale, fthick, _ = _font_metrics(vis)
    label = f"{target_object} ({score:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)
    cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(vis, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), fthick)

    # Bbox center crosshair
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    path = d / f"{_timestamp()}_1_detection.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] Detection saved: {path}")

    _stash(vis, f"Detection: {label}")

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"🔍 Detection: {label}")

    return vis


# ------------------------------------------------------------------
# 2. Mask overlay
# ------------------------------------------------------------------

def vis_mask(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    hitl=None,
) -> np.ndarray:
    """Overlay mask on image in semi-transparent green."""
    d = _ensure_dir()
    vis = image.copy()

    # Resize mask to image size if needed
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        mask_resized = mask.astype(bool)

    # Green overlay on masked pixels
    overlay = vis.copy()
    overlay[mask_resized] = (
        overlay[mask_resized] * 0.5 + np.array([0, 200, 0]) * 0.5
    ).astype(np.uint8)
    vis = overlay

    # Draw bbox outline
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Stats
    fscale, fthick, line_h = _font_metrics(vis)
    label = f"Mask: {mask_resized.sum():,} px ({mask_resized.sum()*100/(h*w):.1f}%)"
    cv2.putText(vis, label, (10, line_h),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), fthick)

    path = d / f"{_timestamp()}_2_mask.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] Mask saved: {path}")

    _stash(vis, f"Mask: {label}")

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"🎭 {label}")

    return vis


# ------------------------------------------------------------------
# 3. Depth visualization
# ------------------------------------------------------------------

def vis_depth(
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    hitl=None,
    grasp_uv: tuple[int, int] | None = None,
) -> np.ndarray:
    """Visualize depth map with bbox region highlighted.

    Args:
        depth: (H, W) or (H, W, 1) depth in metres.
        bbox: (x1, y1, x2, y2) bounding box.
        hitl: HITL handle (optional) for pushing to web UI.
        grasp_uv: (u, v) pixel position of the predicted grasp point (optional).
    """
    d = _ensure_dir()
    depth_2d = np.squeeze(depth).astype(np.float32)

    # Normalize for visualization — filter out inf/nan/zero before percentiles
    valid = depth_2d[(depth_2d > 0) & np.isfinite(depth_2d)]
    if len(valid) == 0:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.percentile(valid, [2, 98])

    depth_norm = np.clip((depth_2d - vmin) / (vmax - vmin + 1e-6), 0, 1)
    depth_color = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO,
    )
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

    # Black-out invalid pixels (inf / zero / nan) so the scene stands out
    invalid = ~((depth_2d > 0) & np.isfinite(depth_2d))
    depth_color[invalid] = 0

    # Draw bbox
    x1, y1, x2, y2 = bbox
    cv2.rectangle(depth_color, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Mark the grasp point if provided
    if grasp_uv is not None:
        gu, gv = int(grasp_uv[0]), int(grasp_uv[1])
        cv2.drawMarker(depth_color, (gu, gv), (255, 0, 255),
                        cv2.MARKER_CROSS, 30, 3)
        cv2.circle(depth_color, (gu, gv), 12, (255, 0, 255), 2)
        # Show depth at grasp point
        h, w = depth_2d.shape
        if 0 <= gv < h and 0 <= gu < w:
            gd = depth_2d[gv, gu]
            fscale, fthick, _ = _font_metrics(depth_color)
            cv2.putText(depth_color, f"grasp depth={gd:.3f}m",
                        (gu + 15, gv - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.30, fscale * 0.75), (255, 0, 255), fthick)

    # Stats in bbox region
    region = depth_2d[y1:y2, x1:x2]
    region_valid = region[(region > 0) & np.isfinite(region)]
    if len(region_valid) > 0:
        label = (f"Depth in bbox: {region_valid.mean():.3f}m "
                 f"(min={region_valid.min():.3f}, max={region_valid.max():.3f})")
    else:
        label = "Depth in bbox: NO VALID PIXELS"
    fscale, fthick, line_h = _font_metrics(depth_color)
    cv2.putText(depth_color, label, (10, line_h),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), fthick)

    path = d / f"{_timestamp()}_3_depth.jpg"
    _imwrite_upscaled(path, depth_color)
    logger.info(f"  [debug] Depth saved: {path}  ({label})")

    _stash(depth_color, f"Depth: {label}")

    if hitl is not None:
        _push_to_hitl(hitl, depth_color, f"📏 {label}")

    return depth_color


# ------------------------------------------------------------------
# 4. Grasp pose projected onto image
# ------------------------------------------------------------------

def vis_grasp_pose(
    image: np.ndarray,
    grasp_pose_camera: np.ndarray,
    intrinsics,
    confidence: float,
    hitl=None,
) -> np.ndarray:
    """Project grasp pose axes onto the image.

    Red=X (finger opening), Green=Y, Blue=Z (approach direction).
    """
    d = _ensure_dir()
    vis = image.copy()
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    origin = grasp_pose_camera[:3, 3]
    axis_len = 0.05  # 5cm axes

    # Project origin
    def project(pt3d):
        if pt3d[2] <= 0:
            return None
        u = int(fx * pt3d[0] / pt3d[2] + cx)
        v = int(fy * pt3d[1] / pt3d[2] + cy)
        return (u, v)

    o = project(origin)
    if o is None:
        logger.warning("  [debug] Grasp origin behind camera, can't visualize")
        return vis

    # Draw axes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # R=X, G=Y, B=Z
    labels = ["X(open)", "Y", "Z(approach)"]
    for i in range(3):
        tip = origin + grasp_pose_camera[:3, i] * axis_len
        t = project(tip)
        if t is not None:
            cv2.arrowedLine(vis, o, t, colors[i], 3, tipLength=0.3)

    # Draw origin point
    cv2.circle(vis, o, 8, (255, 255, 0), -1)
    cv2.circle(vis, o, 8, (0, 0, 0), 2)

    # Label
    fscale, fthick, line_h = _font_metrics(vis)
    label = f"Grasp conf={confidence:.3f}  z={origin[2]:.3f}m"
    cv2.putText(vis, label, (10, line_h),
                cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 0), fthick)

    # Gripper outline (simplified: two parallel lines along X)
    gripper_width = 0.04  # ~half the finger span
    for sign in [-1, 1]:
        finger_base = origin + grasp_pose_camera[:3, 0] * gripper_width * sign
        finger_tip = (finger_base +
                      grasp_pose_camera[:3, 2] * 0.06)  # 6cm deep
        fb = project(finger_base)
        ft = project(finger_tip)
        if fb and ft:
            cv2.line(vis, fb, ft, (255, 255, 0), 2)

    path = d / f"{_timestamp()}_4_grasp_pose.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] Grasp pose saved: {path}")

    _stash(vis, f"Grasp: {label}")

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"🤏 {label}")

    return vis


# ------------------------------------------------------------------
# 5. Grasp pose in world frame + IK result
# ------------------------------------------------------------------

def vis_world_grasp(
    image: np.ndarray,
    grasp_pose_world: np.ndarray,
    current_ee_pos: np.ndarray,
    q_pre: np.ndarray | None = None,
    q_grasp: np.ndarray | None = None,
    hitl=None,
) -> np.ndarray:
    """Annotate image with world-frame grasp info."""
    d = _ensure_dir()
    vis = image.copy()

    gp = grasp_pose_world[:3, 3]
    lines = [
        f"Grasp world: [{gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}]",
        f"Current EE:  [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]",
        f"Distance:    {np.linalg.norm(gp - current_ee_pos):.3f}m",
    ]
    if q_pre is not None:
        lines.append(f"IK pre-grasp: joints=[{', '.join(f'{v:.2f}' for v in q_pre[:4])}...]")
    if q_grasp is not None:
        lines.append(f"IK grasp:     joints=[{', '.join(f'{v:.2f}' for v in q_grasp[:4])}...]")

    fscale, fthick, line_h = _font_metrics(vis)
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, line_h + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), fthick)

    path = d / f"{_timestamp()}_5_world_grasp.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] World grasp saved: {path}")

    label = f"World grasp: [{gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}]"
    _stash(vis, label)

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"🌍 Target: [{gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}]")

    return vis


# ------------------------------------------------------------------
# 6. Post-move: actual robot pose with predicted grasp overlay
# ------------------------------------------------------------------

def vis_post_move_grasp(
    image: np.ndarray,
    grasp_pose_camera: np.ndarray,
    intrinsics,
    confidence: float,
    ee_pos_world: np.ndarray | None = None,
    ee_quat_wxyz: np.ndarray | None = None,
    cam_to_world: np.ndarray | None = None,
    q_joints: np.ndarray | None = None,
    axis_errs: dict | None = None,
    hitl=None,
    ee_axis_yaw_to_graspgen_rad: float = 0.0,
    fingertip_gt_world: np.ndarray | None = None,
    fingertip_pred_world: np.ndarray | None = None,
    fingertip_errs: dict | None = None,
    R_pred_world: np.ndarray | None = None,
    R_actual_world: np.ndarray | None = None,
) -> np.ndarray:
    """Project predicted vs actual grasp pose (at the FINGERTIP) after moving.

    Everything is drawn at the **fingertip** — the point that physically
    grasps — not the flange/wrist.  Two coincident triads:

      * PREDICTED (bright): GraspGen's fingertip + ``R_pred_world`` axes,
        anchored at ``fingertip_pred_world``.
      * ACTUAL (dim): the robot's true fingertip + ``R_actual_world`` axes,
        anchored at ``fingertip_gt_world``.

    Axis colours: Red=X (finger opening), Green=Y, Blue=Z (approach).

    ``R_pred_world`` and ``R_actual_world`` are passed in already expressed in
    the SAME labelling convention (computed once in ``tool.py`` as ``R_planned``
    / ``R_fk``), so same-coloured arrows are directly comparable — when the
    grasp is accurate the dim arrow lies on top of the bright arrow of the same
    colour.  (The old code re-derived each triad from a different raw frame,
    which is why the colours looked ~90° off even at 0° error.)
    """
    d = _ensure_dir()
    vis = image.copy()
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    origin = grasp_pose_camera[:3, 3]

    def project(pt3d):
        if pt3d[2] <= 0:
            return None
        u = int(fx * pt3d[0] / pt3d[2] + cx)
        v = int(fy * pt3d[1] / pt3d[2] + cy)
        return (u, v)

    fscale, fthick, line_h = _font_metrics(vis)

    label = f"Post-move conf={confidence:.3f}  z={origin[2]:.3f}m"

    if cam_to_world is not None:
        world_to_cam = np.linalg.inv(cam_to_world)

        def _proj_world(pw):
            pc = world_to_cam[:3, :3] @ np.asarray(pw)[:3] + world_to_cam[:3, 3]
            return project(pc)

        # ── PREDICTED grasp: cyan gripper + centre dot ──
        pp = None
        if fingertip_pred_world is not None:
            pp = _proj_world(fingertip_pred_world)
            if pp is not None:
                if R_pred_world is not None:
                    _ov.draw_gripper(vis, _proj_world, fingertip_pred_world,
                                     R_pred_world, _ov.C_PRED, max(2, fthick))
                cv2.circle(vis, pp, max(3, fthick + 1), _ov.C_PRED, -1)

        # ── ACTUAL grasp: yellow gripper + centre dot ──
        if fingertip_gt_world is not None:
            pg = _proj_world(fingertip_gt_world)
            if pg is not None:
                if R_actual_world is not None:
                    _ov.draw_gripper(vis, _proj_world, fingertip_gt_world,
                                     R_actual_world, _ov.C_ACTUAL,
                                     max(2, fthick))
                cv2.circle(vis, pg, max(3, fthick + 1), _ov.C_ACTUAL, -1)
                if pp is not None:
                    cv2.line(vis, pp, pg, _ov.C_ACTUAL, max(1, fthick - 1))

        # ── Clean info panel (top-left) ───────────────────────────────────
        rows: list = []
        if fingertip_errs:
            ft3 = fingertip_errs.get("err_m", float("nan")) * 1000
            al = fingertip_errs.get("along_m", float("nan")) * 1000
            lat = fingertip_errs.get("lateral_m", float("nan")) * 1000
            rows.append(("big", ("fingertip err", f"{ft3:.1f} mm")))
            rows.append(("kv", ("along / lateral",
                                f"{al:+.1f} / {lat:.1f} mm")))
        if axis_errs:
            xe = axis_errs.get("x_open", float("nan"))
            ye = axis_errs.get("y", float("nan"))
            ze = axis_errs.get("z_approach", float("nan"))
            rows.append(("kv", ("rotation X/Y/Z",
                                f"{xe:.0f} / {ye:.0f} / {ze:.0f} deg")))
        rows.append(("kv", ("confidence / depth",
                            f"{confidence:.2f} / {float(origin[2]):.3f} m")))
        _ov.draw_info_panel(
            vis, fscale, fthick, line_h,
            title="GRASP ACCURACY  (after move)", rows=rows,
        )

    path = d / f"{_timestamp()}_6_post_move_grasp.jpg"
    _imwrite_upscaled(path, vis)
    logger.info(f"  [debug] Post-move grasp saved: {path}")

    _stash(vis, f"Post-move: {label}")

    if hitl is not None:
        _push_to_hitl(hitl, vis, f"🤖 Post-move: {label}")

    return vis


# ------------------------------------------------------------------
# Helper: push image to HITL UI
# ------------------------------------------------------------------

_VIS_PAUSE = 1.5  # seconds — enough for the HITL browser to render each step


def _push_to_hitl(hitl, image_rgb: np.ndarray, status: str):
    """Push an image + status to the HITL web UI with a short pause.

    A brief pause is inserted so the operator can see each
    visualization step before it's replaced by the next one.
    """
    try:
        hitl.update_image(image_rgb)
        hitl.update_status(status_message=status)
        time.sleep(_VIS_PAUSE)
    except Exception as e:
        logger.debug(f"  [debug] Could not push to HITL: {e}")
