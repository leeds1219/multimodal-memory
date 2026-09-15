# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: mask ↔ point-cloud overlap must be correct when the front
camera's DEPTH resolution differs from its RGB/mask resolution.

Bug (2026-07): the grasp tool passed ``image_hw`` = RGB resolution
(1280×720) to the server, but the point cloud is projected in DEPTH
resolution (864×480 for robolab's front camera).  The server scales
projected pixels to mask space via ``u_mask = u * mask_w / w``; with the
wrong ``w`` the scale factor is off AND the projected ``v`` (max 479)
never reaches mask rows below 480, so a target at v≈597 in the 720-row
mask got 0 overlap → every front-camera grasp failed pre-flight.

Fix: pass ``image_hw`` = intrinsics/depth resolution (the frame the
projection is actually in).  For the exterior camera (RGB==depth) it's
unchanged.

This test reproduces the server's masking math (``_project_points`` +
scale-to-mask) directly, so it needs no server / GPU.
"""
import numpy as np


def _server_overlap(point_cloud, mask, image_hw, fx, fy, cx, cy):
    """Mirror grasp/server.py::compute_grasp_core's mask→object-points step."""
    h, w = image_hw
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    mask_h, mask_w = mask.shape[:2]
    u_mask = (u * mask_w / w).astype(int)
    v_mask = (v * mask_h / h).astype(int)
    in_bounds = (
        (u_mask >= 0) & (u_mask < mask_w) & (v_mask >= 0) & (v_mask < mask_h)
    )
    keep = np.zeros(len(point_cloud), dtype=bool)
    keep[in_bounds] = mask[v_mask[in_bounds], u_mask[in_bounds]]
    return int(keep.sum())


def _make_front_scene():
    """Front camera: depth/intrinsics 864×480, RGB/mask 1280×720.

    Put a small object patch in the LOWER part of the frame (v > 480 in
    RGB space) — exactly where the old bug dropped it.
    """
    # Front intrinsics (matches vlm_orchestrator/grasp/camera.py values).
    fx, fy, cx, cy = 989.549, 753.434, 432.0, 240.0
    depth_w, depth_h = 864, 480
    rgb_w, rgb_h = 1280, 720

    # A cluster of 3D points that project to the lower-centre of the image
    # (v near the bottom of the depth frame → bottom of the RGB frame).
    # Pick depth pixels around (u=610, v=430) in DEPTH space.
    us_d = np.array([600, 610, 620, 605, 615], dtype=float)
    vs_d = np.array([420, 430, 440, 435, 425], dtype=float)
    z = np.full(us_d.shape, 0.8)
    x = (us_d - cx) * z / fx
    y = (vs_d - cy) * z / fy
    pc = np.stack([x, y, z], axis=1).astype(np.float32)

    # RGB-resolution mask: mark the corresponding region.  Those depth
    # pixels map to RGB via the resolution ratio: u_rgb = u_d * 1280/864,
    # v_rgb = v_d * 720/480.
    mask = np.zeros((rgb_h, rgb_w), dtype=bool)
    u_rgb = (us_d * rgb_w / depth_w).astype(int)
    v_rgb = (vs_d * rgb_h / depth_h).astype(int)
    for uu, vv in zip(u_rgb, v_rgb):
        mask[vv - 6:vv + 6, uu - 6:uu + 6] = True

    return pc, mask, (fx, fy, cx, cy), (depth_h, depth_w), (rgb_h, rgb_w)


def test_front_camera_overlap_fixed_with_depth_res():
    """Passing image_hw = DEPTH resolution → overlap > 0 (the fix)."""
    pc, mask, (fx, fy, cx, cy), depth_hw, _rgb_hw = _make_front_scene()
    overlap = _server_overlap(pc, mask, depth_hw, fx, fy, cx, cy)
    assert overlap > 0, "depth-res image_hw must recover mask/pc overlap"
    assert overlap == len(pc)


def test_front_camera_overlap_broken_with_rgb_res():
    """Passing image_hw = RGB resolution → 0 overlap (reproduces the bug).

    Guards against a regression that reverts image_hw to image.shape.
    """
    pc, mask, (fx, fy, cx, cy), _depth_hw, rgb_hw = _make_front_scene()
    overlap = _server_overlap(pc, mask, rgb_hw, fx, fy, cx, cy)
    assert overlap == 0, (
        "RGB-res image_hw is the OLD bug; if this is non-zero the scaling "
        "changed and the test scene needs revisiting"
    )


def test_exterior_camera_unaffected():
    """When depth res == RGB res (exterior camera), both give the same
    (correct) overlap — the fix is a no-op there."""
    fx, fy, cx, cy = 500.0, 500.0, 640.0, 360.0
    hw = (720, 1280)  # depth == RGB
    # Points projecting to (u=640, v=360) centre.
    z = np.full(5, 1.0)
    us = np.array([630, 640, 650, 635, 645], dtype=float)
    vs = np.array([350, 360, 370, 355, 365], dtype=float)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pc = np.stack([x, y, z], axis=1).astype(np.float32)
    mask = np.zeros(hw, dtype=bool)
    mask[340:380, 620:660] = True
    overlap = _server_overlap(pc, mask, hw, fx, fy, cx, cy)
    assert overlap == len(pc)
