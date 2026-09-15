# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera intrinsics / extrinsics utilities for the grasp pipeline.

Supports two backends:

**IsaacLab (robolab)**
    Converts ``PinholeCameraCfg`` parameters (focal_length, aperture) to
    standard ``(fx, fy, cx, cy)`` pixel-space intrinsics.  Handles
    OpenGL ↔ OpenCV frame conversions (IsaacLab cameras use OpenGL: X-right,
    Y-up, Z-backward).

**robosuite (LIBERO)**
    Converts MuJoCo ``cam_fovy`` and resolution to pixel-space intrinsics.
    Extrinsics are obtained via ``robosuite.utils.camera_utils``.

The grasp pipeline (GraspGen, point-cloud projection) uses OpenCV convention:
    X-right, Y-down, Z-forward
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# OpenGL → OpenCV: flip Y and Z axes.
# Multiply a pose in OpenGL convention on the right to get OpenCV.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
OPENCV_TO_OPENGL = OPENGL_TO_OPENCV  # self-inverse


@dataclass
class CameraIntrinsics:
    """Pixel-space camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)


def intrinsics_from_pinhole_cfg(
    *,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    width: int,
    height: int,
) -> CameraIntrinsics:
    """Compute pixel-space intrinsics from IsaacLab PinholeCameraCfg params.

    The physical focal length (mm) and aperture (mm) map to pixels via::

        fx = focal_length / horizontal_aperture * width
        fy = focal_length / vertical_aperture  * height

    Principal point is assumed at image centre.

    These parameters come from the ``spawn`` field of a ``CameraCfg``, e.g.
    ``OverShoulderLeftCameraCfg`` in ``robolab/variations/camera.py``::

        spawn = PinholeCameraCfg(
            focal_length=2.1,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        )
        # height=720, width=1280
    """
    fx = focal_length / horizontal_aperture * width
    fy = focal_length / vertical_aperture * height
    cx = width / 2.0
    cy = height / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


# Convenience presets for the cameras configured in robolab
# ---------------------------------------------------------

def overshoulder_left_intrinsics() -> CameraIntrinsics:
    """Intrinsics for ``OverShoulderLeftCameraCfg`` (external_cam).

    From ``robolab/variations/camera.py``:
      focal_length=2.1, horizontal_aperture=5.376, vertical_aperture=3.024,
      width=1280, height=720
    """
    return intrinsics_from_pinhole_cfg(
        focal_length=2.1,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
        width=1280,
        height=720,
    )


def droid_wrist_intrinsics() -> CameraIntrinsics:
    """Intrinsics for ``DroidCfg.wrist_cam``.

    From ``robolab/robots/droid.py``:
      focal_length=2.8, horizontal_aperture=5.376, vertical_aperture=3.024,
      width=1280, height=720
    """
    return intrinsics_from_pinhole_cfg(
        focal_length=2.8,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
        width=1280,
        height=720,
    )


def front_camera_intrinsics() -> CameraIntrinsics:
    """Intrinsics for ``EgocentricMirroredCameraCfg`` (front/top-down camera).

    From ``robolab/variations/camera.py``:
      focal_length=24.0, horizontal_aperture=20.955, vertical_aperture=15.29,
      width=864, height=480
    """
    return intrinsics_from_pinhole_cfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.29,
        width=864,
        height=480,
    )


# Extrinsics helpers
# ------------------

def pose_opengl_to_opencv(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert an IsaacLab camera pose (OpenGL convention) to a 4×4 OpenCV
    camera-to-world matrix.

    Args:
        pos: (3,) translation in world frame.
        quat_wxyz: (4,) quaternion in (w, x, y, z) order.

    Returns:
        (4, 4) camera-to-world transform in OpenCV convention.
    """
    T_opengl = _pose_to_matrix(pos, quat_wxyz)
    # The camera's local axes are in OpenGL convention; convert to OpenCV
    # by flipping Y and Z in the camera's own frame.
    T_opencv = T_opengl @ OPENGL_TO_OPENCV
    return T_opencv


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    max_depth: float = 10.0,
) -> np.ndarray:
    """Convert a depth image to an (N, 3) point cloud in camera frame (OpenCV).

    Args:
        depth: (H, W) float depth in metres (from IsaacLab camera).
        intrinsics: Camera intrinsics.
        max_depth: Discard points beyond this distance.

    Returns:
        (N, 3) float32 array of XYZ points in OpenCV camera frame.
    """
    h, w = depth.shape[:2]
    v, u = np.mgrid[0:h, 0:w]
    z = np.squeeze(depth).astype(np.float32)  # (H,W,1) → (H,W)

    x = (u - intrinsics.cx) / intrinsics.fx * z
    y = (v - intrinsics.cy) / intrinsics.fy * z

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter invalid points
    valid = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 2] > 0)
        & (pts[:, 2] < max_depth)
    )
    return pts[valid].astype(np.float32)


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------


def _quat_to_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _pose_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from position + quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_rotation(quat_wxyz)
    T[:3, 3] = pos
    return T


# ==================================================================
# robosuite / LIBERO camera utilities
# ==================================================================

def intrinsics_from_fovy(
    *,
    fovy: float,
    width: int,
    height: int,
) -> CameraIntrinsics:
    """Compute pixel-space intrinsics from MuJoCo field-of-view-Y.

    MuJoCo defines cameras by vertical field of view (``cam_fovy`` in
    degrees).  The focal length in pixels is::

        fy = (height / 2) / tan(fovy / 2)
        fx = fy  (square pixels assumed)

    This matches robosuite's
    ``robosuite.utils.camera_utils.get_camera_intrinsic_matrix``.

    Args:
        fovy: Vertical field of view in **degrees**.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def libero_agentview_intrinsics(
    width: int = 256,
    height: int = 256,
    fovy: float = 45.0,
) -> CameraIntrinsics:
    """Intrinsics for LIBERO's ``agentview`` camera.

    The default MuJoCo camera fovy is 45° (robosuite default).  Override
    if LIBERO or the specific task changes it.  Resolution defaults to
    256×256 (LIBERO training resolution).
    """
    return intrinsics_from_fovy(fovy=fovy, width=width, height=height)


def extrinsics_from_obs(
    camera_K: np.ndarray | None = None,
    camera_extrinsic: np.ndarray | None = None,
) -> tuple[CameraIntrinsics | None, np.ndarray | None]:
    """Reconstruct intrinsics and extrinsics from flattened obs arrays.

    The LIBERO eval client packs camera matrices as flattened 1-D arrays
    in the wire observation:
      - ``observation/camera_K``: flattened 3×3 intrinsic matrix
      - ``observation/camera_extrinsic``: flattened 4×4 camera-to-world

    Returns ``(CameraIntrinsics, cam_to_world_4x4)`` or ``(None, None)``
    if the obs keys are missing.
    """
    if camera_K is None or camera_extrinsic is None:
        return None, None

    K = np.asarray(camera_K, dtype=np.float64).reshape(3, 3)
    T = np.asarray(camera_extrinsic, dtype=np.float64).reshape(4, 4)

    # Infer resolution from the principal point (assumes image centre)
    width = int(round(K[0, 2] * 2))
    height = int(round(K[1, 2] * 2))

    intrinsics = CameraIntrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        width=width,
        height=height,
    )
    return intrinsics, T
