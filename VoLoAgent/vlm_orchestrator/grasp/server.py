# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP server for grasp prediction (and optional segmentation).

Runs in a separate process (typically in the ``tool_graspgen`` conda env)
and exposes GPU-accelerated grasp generation over HTTP so the
orchestrator proxy can call it without importing heavy ML dependencies.

Endpoints
---------
POST /compute_grasp
    Generate a collision-free grasp pose for a masked object.
    Request:  multipart/form-data  OR  application/octet-stream (.npz)
              Fields: point_cloud (Nx3 float32), mask (HxW bool),
                      image (JPEG bytes), focal_length_px (float)
    Response: application/octet-stream (.npz)
              Arrays: grasp_pose (4x4 float64), grasp_confidence (scalar)

POST /segment
    Segment the object at a 2-D point (requires SAM2).
    Request:  multipart/form-data
              Fields: image (JPEG bytes), point_x (float), point_y (float)
    Response: application/octet-stream (.npz)
              Arrays: mask (HxW bool), iou_score (scalar)

GET /health
    Returns JSON ``{"status": "ok", "models": [...]}``.

Usage
-----
::

    conda activate tool_graspgen
    python -m vlm_orchestrator.grasp.server \\
        --port 8003 \\
        --gripper-config /path/to/graspgen_franka_panda.yml

    # Optional: also load SAM2 for segmentation
    python -m vlm_orchestrator.grasp.server \\
        --port 8003 \\
        --gripper-config /path/to/graspgen_franka_panda.yml \\
        --enable-sam2
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import subprocess
import time
from typing import Optional

import numpy as np

# FastAPI types imported at module level so that `from __future__ import
# annotations` (PEP 563 deferred evaluation) doesn't break FastAPI's
# annotation-based parameter resolution in closures.
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.requests import Request
    from fastapi.responses import Response, JSONResponse
except ImportError:
    # The server file may be parsed in envs without fastapi installed
    # (e.g. for type-checking).  Provide stubs to avoid import errors.
    FastAPI = Request = HTTPException = Response = JSONResponse = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded heavy deps (only imported when server actually starts)
# ---------------------------------------------------------------------------

_graspgen_sampler = None
_graspgen_cfg = None
_sam3_detector = None
_gdino_detector = None


def _load_graspgen(gripper_config: str):
    """Load GraspGen model (called once at startup)."""
    global _graspgen_sampler, _graspgen_cfg

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    logger.info(f"Loading GraspGen with config: {gripper_config}")
    t0 = time.time()
    _graspgen_cfg = load_grasp_cfg(gripper_config)
    _graspgen_sampler = GraspGenSampler(_graspgen_cfg)
    logger.info(f"GraspGen loaded in {time.time() - t0:.1f}s")


def _load_gdino(model_id: str = "IDEA-Research/grounding-dino-tiny"):
    """Load GroundingDINO model (called once at startup, optional).

    Runs text-prompted object detection on GPU (~0.65 GB VRAM).
    Used by the ``/detect`` endpoint so the orchestrator does not need
    ``transformers`` or a GPU.
    """
    global _gdino_detector

    from vlm_orchestrator.perception.gdino import GroundingDINODetector

    logger.info(f"Loading GroundingDINO model: {model_id}")
    t0 = time.time()
    _gdino_detector = GroundingDINODetector(model_id=model_id)
    _gdino_detector._ensure_loaded()  # force weight download now
    logger.info(f"GroundingDINO loaded in {time.time() - t0:.1f}s")


_KNOWN_COLORS = {
    "red", "blue", "green", "yellow", "black", "white",
    "orange", "pink", "purple", "brown", "grey", "gray",
}


def _extract_color(text: str) -> str | None:
    """Extract color word from a text prompt, if any."""
    for word in text.lower().split():
        if word in _KNOWN_COLORS:
            return word
    return None


def _prompt_variants(text_prompt: str) -> list[str]:
    """Generate fallback prompt variants for SAM3 detection.

    SAM3 was trained primarily on COCO/LVIS category names (plain nouns).
    Attribute+noun queries like "black hammer" often fail because the
    model hasn't seen that exact text during training.  We try:
      1. Just the last noun (e.g. "hammer")
      2. With article (e.g. "a hammer")
      3. Original with trailing period (GDino convention)
    """
    words = text_prompt.strip().split()
    variants = []
    if len(words) > 1:
        # Just the noun (last word)
        noun = words[-1]
        variants.append(noun)
        # With article
        variants.append(f"a {noun}")
    # With trailing period (some models expect this)
    if not text_prompt.endswith("."):
        variants.append(f"{text_prompt}.")
    # Deduplicate while preserving order
    seen = {text_prompt}
    return [v for v in variants if v not in seen and not seen.add(v)]


def _load_sam3(model_id: str = "facebook/sam3"):
    """Load SAM3 model (called once at startup, optional).

    SAM3 unifies detection + segmentation — it replaces both GDino and SAM2
    with a single text-prompted model.  Requires ``sam3>=0.1.0`` package
    (uses native API, no transformers version requirement).

    The model is gated on HuggingFace.  Requires prior access approval at
    https://huggingface.co/facebook/sam3.
    """
    global _sam3_detector

    try:
        from vlm_orchestrator.perception.sam3 import SAM3Detector
    except ModuleNotFoundError:
        from sam3_detector import SAM3Detector

    logger.info(f"Loading SAM3 model '{model_id}'")
    t0 = time.time()
    _sam3_detector = SAM3Detector(model_id=model_id)
    _sam3_detector._ensure_loaded()  # force load now, not lazily
    logger.info(f"SAM3 loaded in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Grasp computation core (extracted from SpaceTools GraspGeneratorTool)
# ---------------------------------------------------------------------------

# Top-down filtering: prefer grasps whose z-axis aligns with gravity.
# For IsaacLab Z-up worlds the gravity vector in camera frame depends on
# camera orientation.  We use a permissive default; callers can override.
_TOPDOWN_GRAVITY_VECTOR = np.array([0, 0.7, 0.3])
_TOPDOWN_GRAVITY_VECTOR = _TOPDOWN_GRAVITY_VECTOR / np.linalg.norm(
    _TOPDOWN_GRAVITY_VECTOR
)
_TOPDOWN_SCORE_THRESHOLD = 0.3


def _project_points(
    xyz: np.ndarray,
    image_size: tuple[int, int],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project 3-D points to pixel coordinates (u, v)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.stack([u, v], axis=1)


def _filter_topdown_grasps(
    grasp_poses: np.ndarray,
    grasp_confidences: np.ndarray,
    gravity_vector: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep grasps whose z-axis is roughly aligned with *gravity_vector*."""
    grasp_z_axes = grasp_poses[:, :3, 2]
    topdown_scores = np.dot(grasp_z_axes, gravity_vector)
    keep = topdown_scores >= threshold
    logger.info(
        f"Top-down filtering: {len(grasp_poses)} → {keep.sum()} grasps "
        f"(threshold={threshold:.2f}, "
        f"range=[{topdown_scores.min():.3f}, {topdown_scores.max():.3f}])"
    )
    return grasp_poses[keep], grasp_confidences[keep]


def compute_grasp_core(
    point_cloud: np.ndarray,
    mask: np.ndarray,
    image_hw: tuple[int, int],
    focal_length_px: float,
    *,
    fy_px: float | None = None,
    cx_px: float | None = None,
    cy_px: float | None = None,
    grasp_threshold: float = -1.0,
    num_grasps: int = 200,
    collision_threshold: float = 0.01,
    max_scene_points: int = 8192,
    enable_topdown_filter: bool = True,
    topdown_gravity: np.ndarray | None = None,
    topdown_threshold: float = _TOPDOWN_SCORE_THRESHOLD,
) -> tuple[np.ndarray, float]:
    """Run GraspGen inference and return ``(best_grasp_4x4, confidence)``.

    This is the server-side core.  Mirrors the logic in
    ``SpaceTools-Toolshed/toolshed/tools/grasp_generator.py``
    (``GraspGeneratorTool.compute_grasp``).
    """
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.robot import get_gripper_info

    try:
        from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
    except ImportError:
        filter_colliding_grasps = None

    assert _graspgen_sampler is not None, "GraspGen not loaded"

    pts = point_cloud.astype(np.float32)
    mask_array = mask.astype(bool)
    h, w = image_hw
    fx = focal_length_px
    fy = fy_px if fy_px is not None else fx
    cx = cx_px if cx_px is not None else w / 2.0
    cy = cy_px if cy_px is not None else h / 2.0

    # ---- mask → object points ----
    # Project 3D points to image pixels, then scale to mask resolution.
    # The mask may be lower-res than the image (e.g. 256×256 bbox mask).
    mask_h, mask_w = mask_array.shape[:2]
    pixels = _project_points(pts, (h, w), fx, fy, cx, cy)
    # Scale to mask coords
    u_mask = (pixels[:, 0] * mask_w / w).astype(int)
    v_mask = (pixels[:, 1] * mask_h / h).astype(int)
    in_bounds = (
        (u_mask >= 0) & (u_mask < mask_w)
        & (v_mask >= 0) & (v_mask < mask_h)
    )
    keep = np.zeros(len(pts), dtype=bool)
    keep[in_bounds] = mask_array[v_mask[in_bounds], u_mask[in_bounds]]
    obj_pts = pts[keep]
    if len(obj_pts) == 0:
        # Diagnostics for debugging
        n_pts = len(pts)
        n_in_bounds = int(in_bounds.sum())
        n_mask_true = int(mask_array.sum())
        z_range = (float(pts[:, 2].min()), float(pts[:, 2].max())) if n_pts > 0 else (0, 0)
        u_range = (float(pixels[:, 0].min()), float(pixels[:, 0].max())) if n_pts > 0 else (0, 0)
        v_range = (float(pixels[:, 1].min()), float(pixels[:, 1].max())) if n_pts > 0 else (0, 0)
        logger.error(
            f"0 object points! scene_pts={n_pts}, in_bounds={n_in_bounds}, "
            f"mask_true_px={n_mask_true}, mask_shape={mask_array.shape}, "
            f"image_hw=({h},{w}), fx={fx:.1f}, fy={fy:.1f}, "
            f"cx={cx:.1f}, cy={cy:.1f}, "
            f"z_range={z_range}, u_range={u_range}, v_range={v_range}"
        )
        raise RuntimeError(
            f"Mask produced 0 object points from {n_pts} scene points "
            f"(in_bounds={n_in_bounds}, mask_px={n_mask_true}, "
            f"mask={mask_array.shape}, image=({h},{w}), "
            f"z=[{z_range[0]:.3f},{z_range[1]:.3f}], "
            f"u=[{u_range[0]:.1f},{u_range[1]:.1f}], "
            f"v=[{v_range[0]:.1f},{v_range[1]:.1f}])"
        )

    # ---- Pad object points if too few (PointNet needs ≥64 points) ----
    MIN_OBJ_POINTS = 64
    if len(obj_pts) < MIN_OBJ_POINTS:
        logger.warning(
            f"Only {len(obj_pts)} object points — padding to {MIN_OBJ_POINTS} "
            f"(PointNet CUDA kernels may crash with too few points)"
        )
        pad_idx = np.random.choice(len(obj_pts), MIN_OBJ_POINTS - len(obj_pts), replace=True)
        obj_pts = np.concatenate([obj_pts, obj_pts[pad_idx]], axis=0)

    logger.info(
        f"  GraspGen input: {len(obj_pts)} object pts, "
        f"{len(pts)} scene pts, mask_px={int(mask_array.sum())}"
    )

    # ---- GraspGen inference ----
    grasps, grasp_conf = GraspGenSampler.run_inference(
        obj_pts,
        _graspgen_sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=100,
    )
    if len(grasps) == 0:
        raise RuntimeError("GraspGen produced zero grasps.")

    grasps_np = grasps.cpu().numpy()
    conf_np = grasp_conf.cpu().numpy()
    grasps_np[:, 3, 3] = 1

    # ---- top-down filtering ----
    if enable_topdown_filter:
        gvec = (
            topdown_gravity
            if topdown_gravity is not None
            else _TOPDOWN_GRAVITY_VECTOR
        )
        grasps_np, conf_np = _filter_topdown_grasps(
            grasps_np, conf_np, gvec, topdown_threshold,
        )
        if len(grasps_np) == 0:
            raise RuntimeError(
                "Top-down filtering removed all grasps."
            )

    # ---- collision filtering (optional — depends on grasp_gen version) ----
    if filter_colliding_grasps is not None:
        gripper_info = get_gripper_info(_graspgen_cfg.data.gripper_name)
        if len(pts) > max_scene_points:
            idx = np.random.choice(
                len(pts), max_scene_points, replace=False,
            )
            scene_pc = pts[idx]
        else:
            scene_pc = pts

        logger.info(
            f"Collision filter: {len(grasps_np)} grasps, "
            f"{len(scene_pc)} scene pts, threshold={collision_threshold}"
        )
        collision_free = filter_colliding_grasps(
            scene_pc=scene_pc,
            grasp_poses=grasps_np,
            gripper_collision_mesh=gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
        )
        coll_free_grasps = grasps_np[collision_free]
        coll_free_conf = conf_np[collision_free]

        if len(coll_free_grasps) == 0:
            logger.warning(
                f"All {len(grasps_np)} grasps collide — "
                f"falling back to best unfiltered grasp."
            )
            coll_free_grasps = grasps_np
            coll_free_conf = conf_np
    else:
        logger.info(
            "Collision filtering unavailable — "
            "selecting best grasp without collision check"
        )
        coll_free_grasps = grasps_np
        coll_free_conf = conf_np

    # ---- best grasp ----
    best_idx = int(np.argmax(coll_free_conf))
    best_grasp = coll_free_grasps[best_idx]  # 4×4
    best_conf = float(coll_free_conf[best_idx])
    logger.info(
        f"Best grasp: confidence={best_conf:.3f}, "
        f"{len(coll_free_grasps)}/{len(grasps_np)} "
        f"{'collision-free' if filter_colliding_grasps else 'unfiltered'}"
    )
    return best_grasp, best_conf


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def build_app(
    gripper_config: str,
    enable_sam2: bool = False,
    sam2_model: str = "facebook/sam2.1-hiera-small",
    enable_sam3: bool = False,
    sam3_model: str = "facebook/sam3",
    enable_gdino: bool = False,
    gdino_model: str = "IDEA-Research/grounding-dino-tiny",
    topdown_gravity: Optional[np.ndarray] = None,
    enable_curobo: bool = False,
    curobo_robot_config: str = "franka.yml",
    curobo_scene_model: str = "collision_test.yml",
):
    """Create the FastAPI app and load models."""

    # ---- load models at import time (before first request) ----
    _load_graspgen(gripper_config)
    if enable_sam2:
        from vlm_orchestrator.perception import sam2 as _sam2_mod
        _sam2_mod.load(sam2_model)
    if enable_gdino:
        _load_gdino(gdino_model)
    if enable_sam3:
        _load_sam3(sam3_model)
    if enable_curobo:
        from vlm_orchestrator.grasp.curobo_planner import load_curobo
        load_curobo(
            robot_config=curobo_robot_config,
            scene_model=curobo_scene_model,
        )

    app = FastAPI(title="Grasp Prediction Server")

    # ---- helpers ----
    def _npz_response(**arrays) -> Response:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="application/octet-stream",
        )

    def _load_npz(body: bytes) -> dict[str, np.ndarray]:
        return dict(np.load(io.BytesIO(body), allow_pickle=True))

    # ---- endpoints ----

    @app.get("/health")
    async def health():
        from vlm_orchestrator.perception import sam2 as _sam2_mod
        models = ["graspgen"]
        if _gdino_detector is not None:
            models.append("gdino")
        if _sam2_mod.is_loaded():
            models.append("sam2")
        if _sam3_detector is not None:
            models.append("sam3")
        # Report cuRobo so orchestrators defaulting to --motion-planner curobo
        # can probe capability at startup instead of loud-failing mid-episode.
        from vlm_orchestrator.grasp.curobo_planner import get_curobo
        if get_curobo() is not None:
            models.append("curobo")
        return {"status": "ok", "models": models}

    @app.post("/compute_grasp")
    async def compute_grasp(request: Request):
        """Compute a collision-free grasp pose.

        Expects an ``.npz`` body with arrays:
          - ``point_cloud``  (N, 3) float32
          - ``mask``         (H, W) bool / uint8
          - ``image_hw``     (2,) int — [height, width]
          - ``focal_length_px`` (scalar) float
        Optional:
          - ``topdown_gravity``  (3,) float — gravity vector for filtering
        """
        try:
            body = await request.body()
            data = _load_npz(body)

            point_cloud = data["point_cloud"]
            mask = data["mask"]
            image_hw = tuple(int(x) for x in data["image_hw"])
            focal_length_px = float(data["focal_length_px"])
            fy_px = float(data["fy_px"]) if "fy_px" in data else None
            cx_px = float(data["cx_px"]) if "cx_px" in data else None
            cy_px = float(data["cy_px"]) if "cy_px" in data else None

            td_grav = (
                data["topdown_gravity"]
                if "topdown_gravity" in data
                else topdown_gravity
            )
            td_threshold = (
                float(data["topdown_threshold"])
                if "topdown_threshold" in data
                else _TOPDOWN_SCORE_THRESHOLD
            )
            # Optional master toggle: when the caller sends
            # enable_topdown_filter=False (stack=False placements — drop from
            # above, grasp angle irrelevant), skip the top-down filter and
            # return GraspGen's best grasp at any approach angle.  Defaults on.
            enable_td_filter = (
                bool(data["enable_topdown_filter"])
                if "enable_topdown_filter" in data
                else True
            )

            t0 = time.time()
            grasp_pose, confidence = compute_grasp_core(
                point_cloud,
                mask,
                image_hw,
                focal_length_px,
                fy_px=fy_px,
                cx_px=cx_px,
                cy_px=cy_px,
                topdown_gravity=td_grav,
                topdown_threshold=td_threshold,
                enable_topdown_filter=enable_td_filter,
            )
            elapsed = time.time() - t0
            logger.info(f"/compute_grasp completed in {elapsed:.2f}s")

            return _npz_response(
                grasp_pose=grasp_pose,
                grasp_confidence=np.array(confidence),
            )

        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"/compute_grasp error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/plan_motion")
    async def plan_motion(request: Request):
        """Plan a collision-free joint-space trajectory (cuRobo).

        Expects an ``.npz`` body with arrays:
          - ``q_start``  (7,) float32 — start arm joint config
          - ``q_end``    (7,) float32 — goal arm joint config (from caller IK)
        Optional:
          - ``n_steps``   (scalar) int — resample the path to this many waypoints
          - ``scene_pc``  (N, 3) float32 — base-frame scene point cloud obstacle

        Returns ``.npz`` with ``waypoints`` (M, 7) and ``success`` (bool).
        Responds 422 (not 500) with ``success=False`` on planning failure so
        the client can surface a named planner label without a silent fallback.
        """
        from vlm_orchestrator.grasp.curobo_planner import get_curobo

        planner = get_curobo()
        if planner is None:
            raise HTTPException(
                status_code=503,
                detail="cuRobo planner not loaded (start server with "
                       "--enable-curobo)",
            )
        try:
            data = _load_npz(await request.body())
            q_start = data["q_start"]
            q_end = data["q_end"]
            n_steps = int(data["n_steps"]) if "n_steps" in data else None
            scene_pc = data["scene_pc"] if "scene_pc" in data else None
            # ``disable_fingers`` (scalar bool, optional) → disable the
            # gripper finger/hand collision links for THIS plan so a top-down
            # pre-grasp near the table isn't rejected (fingers reach ~10cm
            # below panda_hand).  Matches cuRobo's own plan_grasp approach.
            disable_links = None
            if "disable_fingers" in data and bool(data["disable_fingers"]):
                from vlm_orchestrator.grasp.curobo_planner import (
                    GRASP_APPROACH_DISABLE_LINKS,
                )
                disable_links = GRASP_APPROACH_DISABLE_LINKS

            t0 = time.time()
            waypoints, ok = planner.plan_cspace(
                q_start, q_end, n_steps=n_steps, scene_pc=scene_pc,
                disable_links=disable_links,
            )
            logger.info(
                f"/plan_motion {'ok' if ok else 'FAILED'} "
                f"in {time.time() - t0:.3f}s "
                f"({0 if waypoints is None else len(waypoints)} waypoints)"
            )
            if not ok or waypoints is None:
                return _npz_response(
                    waypoints=np.zeros((0, 7), dtype=np.float32),
                    success=np.array(False),
                )
            return _npz_response(
                waypoints=np.asarray(waypoints, dtype=np.float32),
                success=np.array(True),
            )
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=f"missing array: {e}",
            )
        except Exception as e:
            logger.error(f"/plan_motion error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/attach_object")
    async def attach_object(request: Request):
        """Attach a held-object cloud to the gripper link (cuRobo).

        Expects an ``.npz`` body with arrays:
          - ``obj_pc``  (N, 3) float32 — held-object cloud in robot base frame
          - ``q_hold``  (7,) float32 — arm config at which the object is held
        Optional:
          - ``num_spheres``  (scalar) int — sphere-hull resolution (≤4)

        Returns ``.npz`` with ``success`` (bool).  Used before a collision-
        aware PLACE approach so the carried object moves with the hand instead
        of acting as a fixed world obstacle.
        """
        from vlm_orchestrator.grasp.curobo_planner import get_curobo

        planner = get_curobo()
        if planner is None:
            raise HTTPException(
                status_code=503,
                detail="cuRobo planner not loaded (start server with "
                       "--enable-curobo)",
            )
        try:
            data = _load_npz(await request.body())
            obj_pc = data["obj_pc"]
            q_hold = data["q_hold"]
            num_spheres = (
                int(data["num_spheres"]) if "num_spheres" in data else 4
            )
            ok = planner.attach_object(
                obj_pc, q_hold, num_spheres=num_spheres,
            )
            logger.info(
                f"/attach_object {'ok' if ok else 'FAILED'} "
                f"({0 if obj_pc is None else len(obj_pc)} pts)"
            )
            return _npz_response(success=np.array(bool(ok)))
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=f"missing array: {e}",
            )
        except Exception as e:
            logger.error(f"/attach_object error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/detach_object")
    async def detach_object(request: Request):
        """Detach the held object and re-enable disabled world obstacles.

        No body required.  Safe to call even if nothing was attached.
        """
        from vlm_orchestrator.grasp.curobo_planner import get_curobo

        planner = get_curobo()
        if planner is None:
            raise HTTPException(
                status_code=503,
                detail="cuRobo planner not loaded (start server with "
                       "--enable-curobo)",
            )
        try:
            planner.detach_object()
            return _npz_response(success=np.array(True))
        except Exception as e:
            logger.error(f"/detach_object error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/segment")
    async def segment(request: Request):
        """Segment object at a normalised 2-D point (and optional box).

        Expects ``.npz`` body with:
          - ``image``    (H, W, 3) uint8
          - ``point_x``  (scalar) float, normalised [0, 1]
          - ``point_y``  (scalar) float, normalised [0, 1]
        Optional:
          - ``box``      (4,) int [x1, y1, x2, y2] in pixel coords
        """
        from vlm_orchestrator.perception import sam2 as _sam2_mod
        if not _sam2_mod.is_loaded():
            raise HTTPException(
                status_code=501,
                detail="SAM2 not loaded (start server with --enable-sam2)",
            )
        try:
            body = await request.body()
            data = _load_npz(body)

            image = data["image"]
            point_x = float(data["point_x"])
            point_y = float(data["point_y"])
            box = data.get("box", None)
            if box is not None:
                box = np.asarray(box, dtype=np.float32)

            t0 = time.time()
            mask, iou = _sam2_mod.segment_from_point(
                image, point_x, point_y, box=box,
            )
            elapsed = time.time() - t0
            logger.info(
                f"/segment completed in {elapsed:.2f}s  "
                f"(iou={iou:.3f}, mask_sum={mask.sum()})"
            )

            return _npz_response(
                mask=mask.astype(np.uint8),
                iou_score=np.array(iou),
            )

        except Exception as e:
            logger.error(f"/segment error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/detect")
    async def detect(request: Request):
        """Detect objects using GroundingDINO (text-prompted).

        Runs GroundingDINO on the **grasp server's** GPU so the
        orchestrator container doesn't need ``transformers`` or CUDA.

        Expects ``.npz`` body with:
          - ``image``         (H, W, 3) uint8 RGB
          - ``text_prompt``   (scalar) bytes/str — object description
                              (should end with ".", e.g. "red block.")

        Returns ``.npz`` with:
          - ``boxes``          (N, 4) int — [x1, y1, x2, y2] pixel coords
          - ``scores``         (N,) float — confidence scores
          - ``labels``         (N,) str — detected labels
          - ``num_detections`` (scalar) int
        """
        if _gdino_detector is None:
            raise HTTPException(
                status_code=501,
                detail="GroundingDINO not loaded (start server with --enable-gdino)",
            )
        try:
            body = await request.body()
            data = _load_npz(body)

            image = data["image"]
            text_prompt = data["text_prompt"]
            if isinstance(text_prompt, (bytes, np.bytes_)):
                text_prompt = text_prompt.decode("utf-8")
            elif isinstance(text_prompt, np.ndarray):
                text_prompt = str(text_prompt)

            t0 = time.time()
            detections = _gdino_detector.detect(image, text_prompt)
            elapsed = time.time() - t0

            if not detections:
                logger.info(
                    f"/detect: no detections for '{text_prompt}' "
                    f"({elapsed:.2f}s)"
                )
                return _npz_response(
                    boxes=np.zeros((0, 4), dtype=np.int32),
                    scores=np.zeros((0,), dtype=np.float32),
                    labels=np.array([], dtype=object),
                    num_detections=np.array(0),
                )

            boxes = np.array([d["box"] for d in detections], dtype=np.int32)
            scores = np.array([d["score"] for d in detections], dtype=np.float32)
            labels = np.array([d["label"] for d in detections], dtype=object)

            logger.info(
                f"/detect completed in {elapsed:.2f}s  "
                f"({len(detections)} detections for '{text_prompt}', "
                f"best={scores[0]:.3f})"
            )

            return _npz_response(
                boxes=boxes,
                scores=scores,
                labels=labels,
                num_detections=np.array(len(detections)),
            )

        except Exception as e:
            logger.error(f"/detect error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/detect_and_segment")
    async def detect_and_segment(request: Request):
        """Detect and segment an object using SAM3 (text-prompted).

        Replaces the two-step GDino detect → SAM2 segment pipeline with
        a single SAM3 call.

        Expects ``.npz`` body with:
          - ``image``         (H, W, 3) uint8 RGB
          - ``text_prompt``   (scalar) bytes/str — object description

        Returns ``.npz`` with:
          - ``mask``          (H, W) uint8 — best detection mask
          - ``score``         (scalar) float — detection confidence
          - ``box``           (4,) int — [x1, y1, x2, y2] pixel coords
          - ``num_detections`` (scalar) int — total detections found
        """
        if _sam3_detector is None:
            raise HTTPException(
                status_code=501,
                detail="SAM3 not loaded (start server with --enable-sam3)",
            )
        try:
            body = await request.body()
            data = _load_npz(body)

            image = data["image"]
            text_prompt = data["text_prompt"]
            # numpy stores strings as byte arrays
            if isinstance(text_prompt, (bytes, np.bytes_)):
                text_prompt = text_prompt.decode("utf-8")
            elif isinstance(text_prompt, np.ndarray):
                text_prompt = str(text_prompt)

            t0 = time.time()
            results = _sam3_detector.detect_and_segment(image, text_prompt)
            used_variant = False

            # If no detections, retry with prompt variants.
            # SAM3 was trained on COCO-style category names (plain nouns).
            # Attribute+noun queries like "black hammer" often score below
            # threshold, so we try progressively simpler prompts.
            if not results:
                variants = _prompt_variants(text_prompt)
                for variant in variants:
                    logger.info(
                        f"  SAM3 retry with variant: '{variant}' "
                        f"(original: '{text_prompt}')"
                    )
                    results = _sam3_detector.detect_and_segment(
                        image, variant,
                    )
                    if results:
                        logger.info(
                            f"  SAM3 variant '{variant}' found "
                            f"{len(results)} detection(s)"
                        )
                        used_variant = True
                        break

            elapsed = time.time() - t0

            if not results:
                raise RuntimeError(
                    f"SAM3 found no detections for '{text_prompt}' "
                    f"(also tried: {_prompt_variants(text_prompt)})"
                )

            # If we fell back to a generic noun and the original prompt
            # had a color, filter by color to pick the right instance.
            best = results[0]
            if used_variant and len(results) > 1:
                color = _extract_color(text_prompt)
                if color:
                    best = _sam3_detector.detect_best(
                        image, text_prompt,
                        color_filter=color,
                    ) or results[0]
                    if best not in results:
                        # detect_best returns GDino-format (no mask);
                        # match by box overlap to get the full result
                        from vlm_orchestrator.perception.sam3 import (
                            _pick_by_bbox_overlap,
                        )
                        best = _pick_by_bbox_overlap(results, best["box"])
                    logger.info(
                        f"  Color filter '{color}' selected box "
                        f"{best['box']} (score={best['score']:.3f})"
                    )
            logger.info(
                f"/detect_and_segment completed in {elapsed:.2f}s  "
                f"(score={best['score']:.3f}, "
                f"mask_sum={best['mask'].sum()}, "
                f"n_detections={len(results)})"
            )

            return _npz_response(
                mask=best["mask"].astype(np.uint8),
                score=np.array(best["score"]),
                box=np.array(best["box"], dtype=np.int32),
                num_detections=np.array(len(results)),
            )

        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(
                f"/detect_and_segment error: {e}", exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="HTTP server for grasp prediction (GraspGen + optional SAM2)",
    )
    parser.add_argument(
        "--port", type=int, default=8003,
        help="Port to listen on (default: 8003)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--gripper-config", required=True,
        help="Path to GraspGen gripper YAML config "
             "(e.g. graspgen_franka_panda.yml)",
    )
    parser.add_argument(
        "--enable-sam2", action="store_true",
        help="Also load SAM2 for /segment endpoint",
    )
    parser.add_argument(
        "--sam2-model", default="facebook/sam2.1-hiera-small",
        help="SAM2 model ID (default: facebook/sam2.1-hiera-small)",
    )
    parser.add_argument(
        "--enable-gdino", action="store_true",
        help="Also load GroundingDINO for /detect endpoint "
             "(text-prompted bounding-box detection). "
             "Requires transformers package. ~0.65 GB VRAM.",
    )
    parser.add_argument(
        "--gdino-model", default="IDEA-Research/grounding-dino-tiny",
        help="GroundingDINO model ID (default: IDEA-Research/grounding-dino-tiny)",
    )
    parser.add_argument(
        "--enable-sam3", action="store_true",
        help="Also load SAM3 for /detect_and_segment endpoint "
             "(text-prompted detection + segmentation in one model). "
             "Replaces GDino + SAM2 pipeline. "
             "Requires sam3 package and transformers >= 5.0. "
             "Model is gated — needs HuggingFace approval for facebook/sam3.",
    )
    parser.add_argument(
        "--sam3-model", default="facebook/sam3",
        help="SAM3 model ID (default: facebook/sam3)",
    )
    parser.add_argument(
        "--enable-molmo", action="store_true",
        help="Also launch a Molmo2 HF-transformers HTTP shim as a "
             "subprocess (managed by this server's process lifetime). "
             "The shim runs in conda env `molmo-env` because Molmo2 "
             "needs transformers >= 4.55 + huggingface_hub >= 0.30, "
             "which the SAM3 deps installed in this env are not "
             "compatible with.  Listens on --molmo-port and exposes "
             "an OpenAI-compatible /v1/chat/completions endpoint that "
             "vlm_orchestrator/perception/molmo.py talks to.",
    )
    parser.add_argument(
        "--molmo-model", default="allenai/Molmo2-8B",
        help="Molmo2 HF model ID for the spawned shim (default: "
             "allenai/Molmo2-8B).",
    )
    parser.add_argument(
        "--molmo-port", type=int, default=8122,
        help="Port the spawned Molmo2 shim listens on (default: 8122).",
    )
    parser.add_argument(
        "--molmo-env", default="molmo-env",
        help="Conda env that hosts the Molmo2 HF shim (default: "
             "molmo-env; needs transformers >= 4.55 + accelerate + "
             "tensorflow-cpu + bitsandbytes for the HF dynamic-module "
             "check and optional 4-bit quant).",
    )
    parser.add_argument(
        "--molmo-quantize", default="bf16",
        choices=["bf16", "int8", "int4"],
        help="Molmo2 bnb quantization passed to the sidecar.  bf16 "
             "(default) needs ~17 GB VRAM; int4 needs ~5 GB.  Use int4 "
             "when sharing GPU with Isaac Sim + grasp server on a "
             "single 48 GB card.",
    )
    parser.add_argument(
        "--enable-curobo", action="store_true",
        help="Load the cuRobo collision-aware motion planner and expose "
             "the /plan_motion endpoint. Requires cuRobo + cuda-core[cu12] "
             "installed in this env (see workspace/curobo_spike_findings.md).",
    )
    parser.add_argument(
        "--curobo-robot-config", default="franka.yml",
        help="cuRobo robot config name (default: franka.yml).",
    )
    parser.add_argument(
        "--curobo-scene-model", default="collision_test.yml",
        help="cuRobo default scene/collision model (default: "
             "collision_test.yml); replaced per-request when scene_pc is sent.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = build_app(
        gripper_config=args.gripper_config,
        enable_sam2=args.enable_sam2,
        sam2_model=args.sam2_model,
        enable_gdino=args.enable_gdino,
        gdino_model=args.gdino_model,
        enable_sam3=args.enable_sam3,
        sam3_model=args.sam3_model,
        enable_curobo=args.enable_curobo,
        curobo_robot_config=args.curobo_robot_config,
        curobo_scene_model=args.curobo_scene_model,
    )

    # ---- Optional Molmo2 sidecar via conda run -n <molmo-env> ----
    molmo_proc = None
    if args.enable_molmo:
        molmo_proc = _spawn_molmo_sidecar(
            env_name=args.molmo_env,
            model=args.molmo_model,
            port=args.molmo_port,
            quantize=args.molmo_quantize,
        )

    import uvicorn

    logger.info(
        f"Grasp server listening on {args.host}:{args.port}"
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if molmo_proc is not None:
            logger.info(f"Stopping Molmo2 sidecar (pid={molmo_proc.pid})")
            molmo_proc.terminate()
            try:
                molmo_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                molmo_proc.kill()


def _spawn_molmo_sidecar(env_name: str, model: str, port: int, quantize: str = "bf16"):
    """Fork an OpenAI-compatible Molmo2 shim in another conda env.

    The shim (``vlm_orchestrator/utils/molmo2_hf_server.py``) needs newer
    transformers + huggingface_hub than the grasp-server env, so we run
    it in a separate env via ``conda run``.  Its process is supervised
    by this grasp-server process — when we exit, the shim exits too
    (via the ``finally`` clause in main()).
    """
    import atexit
    import signal
    import os as _os

    # Sibling under vlm_orchestrator/utils/.  grasp/server.py is at
    # vlm_orchestrator/grasp/server.py, so go up one level (to
    # vlm_orchestrator/) then over to utils/.
    script = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "utils", "molmo2_hf_server.py",
    )
    if not _os.path.isfile(script):
        raise FileNotFoundError(
            f"Molmo2 sidecar script not found: {script}"
        )

    cmd = [
        "conda", "run", "--no-capture-output", "-n", env_name,
        "python", script,
        "--model", model,
        "--port", str(port),
        "--quantize", quantize,
    ]
    logger.info(
        f"Spawning Molmo2 sidecar in conda env '{env_name}' on port "
        f"{port} (model={model}, quantize={quantize}): {' '.join(cmd)}"
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # New process group so SIGINT to grasp server doesn't double-kill
        preexec_fn=_os.setsid,
    )

    # Forward sidecar stdout to our log via a daemon thread so failures
    # are visible without filling memory.
    import threading
    def _pipe():
        for line in iter(proc.stdout.readline, b""):
            logger.info(f"[molmo_sidecar] {line.decode(errors='replace').rstrip()}")
    threading.Thread(target=_pipe, daemon=True).start()

    # atexit fallback — terminate sidecar on abnormal exit.
    def _kill():
        if proc.poll() is None:
            try:
                _os.killpg(_os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    atexit.register(_kill)

    return proc


if __name__ == "__main__":
    main()
