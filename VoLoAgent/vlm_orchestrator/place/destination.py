# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Destination grounding for the place tool.

Stages:
  1. ``point_to_place_2d`` — language → normalized 2D pixel.  Backend
     selected by ``--place-seg-mode``: ``gt_sim`` / ``sam3`` /
     ``gdino_sam2`` / ``vlm_point``.  Each backend returns a
     :class:`PointToPlace2D`.
  2. ``raycast_2d_to_3d`` — 2D pixel → 3D world point with a
     per-relation Z adjustment.

No silent fallback between backends: the user picks one mode and
gets that mode's behavior.  If the chosen mode produces nothing the
caller raises ``PerceptionFailure`` and the executor records
``failure_reason='perception_no_target'``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import requests

from vlm_orchestrator.grasp.camera import CameraIntrinsics
from vlm_orchestrator.grasp.client import GraspClient

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _server_error_detail(exc: Exception) -> str:
    """Extract the FastAPI ``detail`` field from an HTTP error response.

    The grasp server returns helpful messages like
    ``"GroundingDINO not loaded — start server with --enable-gdino"``
    in the JSON body when an endpoint requires a backend that wasn't
    enabled.  ``requests.raise_for_status()`` only surfaces the status
    code + URL; pull out the body so the place tool's failure message
    is actionable.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return ""
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
    except Exception:
        try:
            return resp.text[:200]
        except Exception:
            return ""
    return ""


# ----------------------------------------------------------------------
# Failure type
# ----------------------------------------------------------------------

class PerceptionFailure(RuntimeError):
    """Raised when destination grounding fails.

    The caller should set ``PlacePhase.FAILED`` with
    ``failure_reason='perception_no_target'`` and surface this
    exception's message in the diagnostic log.
    """


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------

PointSource = Literal[
    "vlm", "molmo_point", "sam3_centroid", "gdino_sam2_centroid",
    "hitl_click", "gt_sim", "explicit_3d",
]
RelationT = Literal["in", "on", "on_top_of"]


# ----------------------------------------------------------------------
# Geometry constants
# ----------------------------------------------------------------------

# Distance from the panda flange (panda_hand origin) to the fingertip
# contact plane along the gripper's approach axis.  Mirrors GraspGen's
# graspgen_franka_panda.yml `gripper_depth` and the constant referenced
# in vlm_orchestrator/grasp/tool.py: GraspGen returns flange poses, so
# the held object's bottom typically sits at flange.z − gripper_depth −
# (held_object_height / 2).  We need this offset to compute a flange
# Z that puts the *held object* (not the flange) on the surface.
#
# NOTE: this Panda value is only correct for the LIBERO/panda_hand path.
# ROBOLAB mounts a Robotiq 2F-85 whose MEASURED flange→fingertip is
# ROBOTIQ_GRIPPER_DEPTH_M below — the place tool selects that in robolab
# mode (see PlaceToolExecutor.start).  Unlike the grasp path there is NO
# GraspGen pose to keep consistent with here, so we reference the Robotiq
# fingertip depth directly rather than correcting a panda-flange target.
GRIPPER_DEPTH_M = 0.1034

# Robotiq 2F-85 (DROID/robolab) flange→fingertip pad-center distance, MEASURED
# from the sim USD (matches ROBOTIQ_FLANGE_TO_FINGERTIP_M in grasp/tool.py).
# Using the Panda 0.1034 here would position the held object ~0.0277 m too low
# on robolab → the object + fingers dip toward the table (GRIPPER_HIT_TABLE).
ROBOTIQ_GRIPPER_DEPTH_M = 0.1311


@dataclass
class PointToPlace2D:
    """Output of stage 1 (language → 2D pixel)."""
    x_norm: float
    y_norm: float
    confidence: float
    source: PointSource
    rationale: str | None = None
    debug_jpg: bytes | None = None


@dataclass
class ResolvedDestination:
    """Output of stage 2 (2D pixel → 3D world point)."""
    target_world: np.ndarray            # (3,) target for held-object centre
    approach_axis_world: np.ndarray     # (3,) unit vector
    point_2d: PointToPlace2D            # for logging / debug overlay
    relation: RelationT
    held_object_height_m: float | None  # what we used for Z adjust
    clearance_m: float


# ----------------------------------------------------------------------
# Stage 1 — language → 2D pixel
# ----------------------------------------------------------------------

def point_to_place_2d(
    image_rgb: np.ndarray,
    instruction: str,
    target_phrase: str,
    *,
    seg_mode: Literal["gt_sim", "sam3", "gdino_sam2", "vlm_point", "molmo_point"],
    grasp_client: GraspClient | None = None,
    vlm=None,                                           # VLMBackend
    gt_state: dict | None = None,
    intrinsics: CameraIntrinsics | None = None,
    cam_to_world: np.ndarray | None = None,
) -> PointToPlace2D:
    """Pick a normalized 2D pixel for the destination.

    ``seg_mode`` controls dispatch:

    * ``gt_sim``      — read ``gt_state.objects[target].pos`` and project
      to image plane via ``intrinsics`` + ``cam_to_world``.
    * ``sam3``        — SAM3 detect+segment via grasp server.
    * ``gdino_sam2``  — GroundingDINO bbox → SAM2 mask, centroid.
    * ``vlm_point``   — orchestrator main VLM (Claude / Bedrock) pointing.
    * ``molmo_point`` — Molmo2 vLLM server pointing (see vision/molmo.py).

    No silent fallback to other modes — raises :class:`PerceptionFailure`
    if the chosen mode produces nothing.
    """
    if seg_mode == "gt_sim":
        return _gt_sim_point(
            target_phrase=target_phrase,
            gt_state=gt_state,
            image_hw=image_rgb.shape[:2],
            intrinsics=intrinsics,
            cam_to_world=cam_to_world,
        )
    if seg_mode in ("sam3", "gdino_sam2"):
        if grasp_client is None:
            raise PerceptionFailure(
                f"seg_mode={seg_mode!r} requires grasp_client (None given)"
            )
        return _seg_centroid_point(
            image_rgb=image_rgb,
            target_phrase=target_phrase,
            seg_mode=seg_mode,
            grasp_client=grasp_client,
        )
    if seg_mode == "vlm_point":
        if vlm is None:
            raise PerceptionFailure(
                "seg_mode='vlm_point' requires a VLMBackend (None given)"
            )
        return _vlm_point(
            image_rgb=image_rgb,
            instruction=instruction,
            target_phrase=target_phrase,
            vlm=vlm,
        )
    if seg_mode == "molmo_point":
        return _molmo_point(
            image_rgb=image_rgb,
            target_phrase=target_phrase,
        )
    raise PerceptionFailure(
        f"Unknown place seg_mode {seg_mode!r}"
    )


# ----------------------------------------------------------------------
# molmo_point backend
# ----------------------------------------------------------------------

def _molmo_point(
    *,
    image_rgb: np.ndarray,
    target_phrase: str,
) -> PointToPlace2D:
    """Use Molmo2 (via local vLLM server) to point at the destination.

    Reads ``MOLMO_BASE_URL`` / ``MOLMO_MODEL`` env vars (mirrors how
    VLM_API_KEY / OPENAI_API_KEY override the Claude path) so the
    same place tool wiring works against either a local Molmo2 server
    or a remote endpoint.
    """
    import os
    from vlm_orchestrator.perception.molmo import (
        DEFAULT_BASE_URL, DEFAULT_MODEL, MolmoPointError, point_at,
    )
    base_url = os.environ.get("MOLMO_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("MOLMO_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("MOLMO_API_KEY") or None
    try:
        pt = point_at(
            image_rgb=image_rgb,
            target_phrase=target_phrase,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )
    except MolmoPointError as e:
        raise PerceptionFailure(
            f"Molmo pointing failed for {target_phrase!r}: {e}"
        ) from e
    return PointToPlace2D(
        x_norm=pt.x_norm,
        y_norm=pt.y_norm,
        confidence=1.0,
        source="molmo_point",
        rationale=f"Molmo2 {model}; raw[:120]={pt.raw_text[:120]!r}",
    )


# ----------------------------------------------------------------------
# Stage 2 — 2D pixel → 3D world point
# ----------------------------------------------------------------------

def raycast_2d_to_3d(
    pt: PointToPlace2D,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    cam_to_world: np.ndarray,
    *,
    relation: RelationT,
    held_object_height_m: float | None = None,
    clearance_m: float = 0.01,
    pre_resolved_3d: np.ndarray | None = None,
    approach_axis_world: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> ResolvedDestination:
    """Convert a 2D pixel pick to a 3D **object-centre** target in world frame.

    The returned ``target_world`` is where we want the held object's
    centre to end up — *not* the flange position.  Translating the
    object-centre target to a flange target is the trajectory planner's
    job, because it depends on (a) where the held object sits in the
    gripper (rotation-invariant gripper-frame offset, captured at
    place-tool start) and (b) the chosen placement orientation (we
    enforce top-down).

    Object-centre target Z = depth-at-pixel + half_h + clearance, so the
    held object's bottom rests just above the surface that the depth ray
    intersected.  ``in`` and ``on``/``on_top_of`` use the same formula —
    depth-at-pixel returns the closest visible surface (the rim for "in"
    containers, the top for "on" surfaces) and we drop the object onto
    that surface in both cases.

    ``pre_resolved_3d`` short-circuits the raycast — used when the caller
    already has a 3D point (e.g. ``DestinationSpec.target_point_3d_world``).
    The provided point is treated as the surface anchor and lifted by
    ``half_h + clearance`` for the object centre.

    Raises :class:`PerceptionFailure` if depth at the chosen pixel is
    invalid (zero / NaN / inf) — no median-fill silent fallback.
    """
    approach_axis = np.asarray(approach_axis_world, dtype=np.float64)
    approach_axis = approach_axis / max(float(np.linalg.norm(approach_axis)), 1e-9)

    held_half_h = (
        held_object_height_m / 2.0
        if held_object_height_m is not None
        else 0.0
    )

    if pre_resolved_3d is not None:
        object_target = np.asarray(pre_resolved_3d, dtype=np.float64).copy()
        object_target[2] += held_half_h + clearance_m
        return ResolvedDestination(
            target_world=object_target,
            approach_axis_world=approach_axis,
            point_2d=pt,
            relation=relation,
            held_object_height_m=held_object_height_m,
            clearance_m=clearance_m,
        )

    depth_2d = np.squeeze(depth).astype(np.float64)
    h, w = depth_2d.shape[:2]
    px = int(round(pt.x_norm * w))
    py = int(round(pt.y_norm * h))
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))

    z = depth_2d[py, px]
    if not np.isfinite(z) or z <= 0:
        raise PerceptionFailure(
            f"Invalid depth {z} at picked pixel ({px}, {py}); "
            f"refusing to substitute a fallback (the no-silent-lossy-fallback design rule)."
        )

    # Back-project pixel to camera-frame point at the measured depth.
    x_cam = (px - intrinsics.cx) / intrinsics.fx * z
    y_cam = (py - intrinsics.cy) / intrinsics.fy * z
    pt_cam = np.array([x_cam, y_cam, z], dtype=np.float64)
    surface_world = cam_to_world[:3, :3] @ pt_cam + cam_to_world[:3, 3]

    # Object-centre target = surface + half-height + clearance.
    object_target_world = surface_world.copy()
    object_target_world[2] += held_half_h + clearance_m

    return ResolvedDestination(
        target_world=object_target_world,
        approach_axis_world=approach_axis,
        point_2d=pt,
        relation=relation,
        held_object_height_m=held_object_height_m,
        clearance_m=clearance_m,
    )


# ----------------------------------------------------------------------
# gt_sim backend
# ----------------------------------------------------------------------

def _gt_sim_point(
    *,
    target_phrase: str,
    gt_state: dict | None,
    image_hw: tuple[int, int],
    intrinsics: CameraIntrinsics | None,
    cam_to_world: np.ndarray | None,
) -> PointToPlace2D:
    """Read ``gt_state.objects[target].pos`` and project to image."""
    if gt_state is None:
        raise PerceptionFailure(
            "seg_mode='gt_sim' requires gt_state in obs "
            "(pass --enable-gt-state to the eval client)"
        )
    if intrinsics is None or cam_to_world is None:
        raise PerceptionFailure(
            "seg_mode='gt_sim' requires intrinsics + cam_to_world"
        )

    objects = gt_state.get("objects", {}) or {}
    obj = _lookup_gt_object(objects, target_phrase)
    if obj is None:
        raise PerceptionFailure(
            f"GT_SIM: object {target_phrase!r} not in gt_state.objects "
            f"(known: {sorted(objects.keys())})"
        )

    pos_world = np.asarray(obj["pos"], dtype=np.float64).reshape(3)

    # Project world → camera → pixel
    world_to_cam = np.linalg.inv(cam_to_world)
    pt_cam = world_to_cam[:3, :3] @ pos_world + world_to_cam[:3, 3]
    if pt_cam[2] <= 0:
        raise PerceptionFailure(
            f"GT_SIM: object {target_phrase!r} at world={pos_world} "
            f"projects behind camera (cam_z={pt_cam[2]:.3f})"
        )
    u = intrinsics.fx * pt_cam[0] / pt_cam[2] + intrinsics.cx
    v = intrinsics.fy * pt_cam[1] / pt_cam[2] + intrinsics.cy
    h, w = image_hw
    if not (0 <= u < w and 0 <= v < h):
        raise PerceptionFailure(
            f"GT_SIM: object {target_phrase!r} projects outside image "
            f"(u={u:.1f} v={v:.1f}, image {w}x{h})"
        )

    return PointToPlace2D(
        x_norm=float(u / w),
        y_norm=float(v / h),
        confidence=1.0,
        source="gt_sim",
        rationale=f"gt_state.objects[{target_phrase!r}].pos={pos_world.tolist()}",
    )


def _lookup_gt_object(objects: dict, name: str) -> dict | None:
    """Find an object in gt_state.objects by name, with simple aliases."""
    if name in objects:
        return objects[name]
    snake = name.replace(" ", "_")
    if snake in objects:
        return objects[snake]
    lower = name.lower()
    for k, v in objects.items():
        if k.lower() == lower:
            return v
    return None


# ----------------------------------------------------------------------
# sam3 / gdino_sam2 backend (stub — implemented in step 7)
# ----------------------------------------------------------------------

def _seg_centroid_point(
    *,
    image_rgb: np.ndarray,
    target_phrase: str,
    seg_mode: Literal["sam3", "gdino_sam2"],
    grasp_client: GraspClient,
) -> PointToPlace2D:
    """Detect the destination via SAM3 or GDino+SAM2 and return mask centroid.

    Routes through the existing grasp server's endpoints — same calls
    grasp uses for object detection.  No new server surface area.

    Both modes return the **mask centroid** as a normalized 2D pixel.
    For ``in`` / ``on`` destinations on tabletop containers the centroid
    is a reasonable proxy for "centre of the container's open top".
    """
    if not target_phrase or not target_phrase.strip():
        raise PerceptionFailure(
            f"seg_mode={seg_mode!r} requires a non-empty target_phrase"
        )

    h, w = image_rgb.shape[:2]

    if seg_mode == "sam3":
        # SAM3 single-call detect+segment.
        try:
            mask, score, box = grasp_client.detect_and_segment(
                image_rgb, target_phrase,
            )
        except requests.HTTPError as e:
            detail = _server_error_detail(e)
            hint = (
                " (start the grasp server with --enable-sam3)"
                if e.response is not None and e.response.status_code == 501
                else ""
            )
            raise PerceptionFailure(
                f"SAM3 detect_and_segment failed for {target_phrase!r}: "
                f"{detail or e}{hint}"
            ) from e
        except Exception as e:
            raise PerceptionFailure(
                f"SAM3 detect_and_segment failed for {target_phrase!r}: {e}"
            ) from e
        source: PointSource = "sam3_centroid"
    else:
        # GDino + SAM2 two-stage.  No silent local-GDino fallback —
        # the grasp tool has one for historical reasons but it hides
        # server-config bugs (the "no silent lossy fallbacks" design rule).
        # Fail loudly; the operator must enable GDino on the grasp server.
        try:
            detections = grasp_client.detect(image_rgb, target_phrase)
        except requests.HTTPError as e:
            detail = _server_error_detail(e)
            hint = (
                " (start the grasp server with --enable-gdino --enable-sam2)"
                if e.response is not None and e.response.status_code == 501
                else ""
            )
            raise PerceptionFailure(
                f"GDino detect failed for {target_phrase!r}: "
                f"{detail or e}{hint}"
            ) from e
        except Exception as e:
            raise PerceptionFailure(
                f"GDino detect failed for {target_phrase!r}: {e}"
            ) from e

        if not detections:
            raise PerceptionFailure(
                f"GDino: no detections for {target_phrase!r}"
            )

        # Best-scoring detection
        best = max(detections, key=lambda d: d["score"])
        x1, y1, x2, y2 = best["box"]
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        try:
            mask, _iou = grasp_client.segment(
                image_rgb,
                point_x=float(cx_px / w),
                point_y=float(cy_px / h),
                box=(int(x1), int(y1), int(x2), int(y2)),
            )
        except requests.HTTPError as e:
            detail = _server_error_detail(e)
            hint = (
                " (start the grasp server with --enable-sam2)"
                if e.response is not None and e.response.status_code == 501
                else ""
            )
            raise PerceptionFailure(
                f"SAM2 segment failed for {target_phrase!r}: "
                f"{detail or e}{hint}"
            ) from e
        except Exception as e:
            raise PerceptionFailure(
                f"SAM2 segment failed for {target_phrase!r}: {e}"
            ) from e
        score = float(best["score"])
        box = (int(x1), int(y1), int(x2), int(y2))
        source = "gdino_sam2_centroid"

    if mask is None or mask.sum() == 0:
        raise PerceptionFailure(
            f"{seg_mode}: produced an empty mask for {target_phrase!r} "
            f"(box={box}, score={score:.3f})"
        )

    # Mask centroid (normalised image coords).  Resize-aware so server
    # masks at a different resolution still map to the source image.
    mh, mw = mask.shape[:2]
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        raise PerceptionFailure(
            f"{seg_mode}: mask had no foreground pixels for {target_phrase!r}"
        )
    x_norm = float(xs.mean() / mw)
    y_norm = float(ys.mean() / mh)
    return PointToPlace2D(
        x_norm=x_norm,
        y_norm=y_norm,
        confidence=float(score),
        source=source,
        rationale=(
            f"{seg_mode} centroid of {target_phrase!r} "
            f"(box={box}, score={score:.3f}, "
            f"mask_px={int(mask.sum())})"
        ),
    )


# ----------------------------------------------------------------------
# vlm_point backend (stub — implemented in step 8)
# ----------------------------------------------------------------------

_VLM_POINT_SYSTEM_PROMPT = """\
You are a robot's spatial-reasoning assistant.  Given a camera image,
the overall task instruction, and a destination description, return the
single 2D pixel where the robot should release the object it's currently
holding so the placement satisfies the destination description.

Output ONLY a JSON object with three fields:
  "x_norm":   float in [0, 1]  (0 = left edge, 1 = right edge)
  "y_norm":   float in [0, 1]  (0 = top edge, 1 = bottom edge)
  "rationale": one short sentence explaining the chosen pixel

Coordinate convention: (0, 0) is the top-left of the image, (1, 1) the
bottom-right.  Pick the pixel that corresponds to where the held object
should END UP — the centre of the container's open top (for "in"), the
top surface of the target (for "on" / "on_top_of"), etc.

If the destination is not visible in the image, still output your best
guess and explain in the rationale that confidence is low — DO NOT
return an obviously off-image coordinate as a hedge.
"""


def _vlm_point(
    *,
    image_rgb: np.ndarray,
    instruction: str,
    target_phrase: str,
    vlm,
) -> PointToPlace2D:
    """Ask the VLM for a normalized 2D pixel where to place.

    Uses ``vlm.client`` directly via :func:`chat_create` — same pattern
    as ``next_goal.py``.  Parses the response as JSON; raises
    :class:`PerceptionFailure` on parse error or out-of-image coordinates.
    No silent fallback to a default pixel.
    """
    from vlm_orchestrator.vlm import chat_create, encode_image_b64, parse_json

    if not hasattr(vlm, "client") or not hasattr(vlm, "model"):
        raise PerceptionFailure(
            "seg_mode='vlm_point' requires a VLM backend with a `client` "
            "and `model` attribute (e.g. OpenAIVLM); got "
            f"{type(vlm).__name__}"
        )

    image_b64 = encode_image_b64(image_rgb)
    user_text = (
        f'Task instruction: "{instruction}"\n'
        f'Destination phrase: "{target_phrase}"\n\n'
        f"Return ONLY the JSON object — no markdown, no extra text."
    )
    messages = [
        {"role": "system", "content": _VLM_POINT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
            ],
        },
    ]
    try:
        response = chat_create(
            vlm.client,
            model=getattr(vlm, "model", None) or "YOUR_VLM_MODEL",
            temperature=getattr(vlm, "temperature", 0.0),
            max_tokens=300,
            messages=messages,
        )
    except Exception as e:
        raise PerceptionFailure(
            f"VLM pointing call failed for {target_phrase!r}: {e}"
        ) from e

    raw = ""
    try:
        raw = response.choices[0].message.content or ""
        data = parse_json(raw)
    except Exception as e:
        raise PerceptionFailure(
            f"VLM pointing returned invalid JSON for {target_phrase!r}: "
            f"{e}\nraw response: {raw!r}"
        ) from e

    try:
        x_norm = float(data["x_norm"])
        y_norm = float(data["y_norm"])
    except Exception as e:
        raise PerceptionFailure(
            f"VLM pointing JSON missing x_norm / y_norm for "
            f"{target_phrase!r}: {data!r}"
        ) from e

    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        raise PerceptionFailure(
            f"VLM pointing returned out-of-image coords "
            f"x_norm={x_norm}, y_norm={y_norm} for {target_phrase!r}; "
            f"refusing silent clip (the no-silent-lossy-fallback design rule)."
        )

    rationale = str(data.get("rationale", "")).strip() or None
    return PointToPlace2D(
        x_norm=x_norm,
        y_norm=y_norm,
        confidence=1.0,
        source="vlm",
        rationale=rationale,
    )
