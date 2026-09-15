# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward and inverse kinematics for the Franka Emika Panda arm.

Implements FK using the **URDF joint-origin transforms** (unambiguous,
unlike DH parameters which depend on convention).  Provides numerical
Jacobian and damped-least-squares IK for trajectory planning.

No heavy dependencies — pure numpy.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# URDF joint-origin transforms (from panda URDF)
# ======================================================================
# Each entry is (xyz, rpy) defining the fixed transform from the parent
# link frame to the joint axis.  All joints are revolute about z.
#
# panda_link{i-1} --[origin_i]--> joint_i --[Rz(q_i)]--> panda_link{i}

_JOINT_ORIGINS = [
    # (xyz,                    rpy)
    ([0.0, 0.0, 0.333],       [0.0, 0.0, 0.0]),           # joint 1
    ([0.0, 0.0, 0.0],         [-np.pi / 2, 0.0, 0.0]),    # joint 2
    ([0.0, -0.316, 0.0],      [np.pi / 2, 0.0, 0.0]),     # joint 3
    ([0.0825, 0.0, 0.0],      [np.pi / 2, 0.0, 0.0]),     # joint 4
    ([-0.0825, 0.384, 0.0],   [-np.pi / 2, 0.0, 0.0]),    # joint 5
    ([0.0, 0.0, 0.0],         [np.pi / 2, 0.0, 0.0]),     # joint 6
    ([0.088, 0.0, 0.0],       [np.pi / 2, 0.0, 0.0]),     # joint 7
]

# ----------------------------------------------------------------------
# End-of-arm mount transform (joint7 → controlled EE frame)
# ----------------------------------------------------------------------
# The FK chain above ends at the joint-7 link.  The fixed transform from
# there to the frame the *controller* actually commands depends on what is
# bolted on:
#
#   * LIBERO / bare Franka  → panda_hand flange.  Standard URDF:
#       joint8   (fixed): xyz=[0,0,0.107], rpy=[0,0,0]
#       hand_joint(fixed): xyz=[0,0,0],     rpy=[0,0,-π/4]
#     Historically this repo used FLANGE_Z_M=0.107 + HAND_YAW_RAD=-π/2 as an
#     *approximation* of the DROID Robotiq mount too — and then patched the
#     residual error downstream (per-grasp ``fk_vs_ee`` for the ~18 mm
#     translation error, plus the depth correction).  That worked but was
#     impossible to reason about (the "-π/2 yaw" is a Z-rotation, whereas the
#     real Robotiq mount is a +90° rotation about Y — 120° apart).
#
#   * ROBOLAB / DROID Robotiq → Robotiq base_link.  Measured empirically
#     (``scripts/calibrate_mount_transform.py``): solve
#         T_joint7→ee = inv(FK_joint7(q)) @ pose(sim_ee_pos, sim_ee_quat)
#     which comes out EXACTLY constant across poses (translation std = 0,
#     rotation dev = 0°) and reproduces the sim ee_pos/ee_quat to <1e-3 mm.
#     Using it directly makes forward_kinematics_robolab() output the true
#     controlled frame, so ``fk_vs_ee`` collapses to ~0 and the panda-model
#     patches disappear.  See docs / workspace note "Mount-transform
#     calibration" for the derivation.
#
# LIBERO keeps the panda-hand path (FLANGE_Z_M / HAND_YAW_RAD) unchanged.

# Standard Franka panda-hand params (LIBERO / bare-Franka path).
FLANGE_Z_M: float = 0.107      # joint8 Z offset (metres)
HAND_YAW_RAD: float = -np.pi / 2  # panda_hand_joint Z rotation (radians)

_FLANGE_XYZ = [0.0, 0.0, FLANGE_Z_M]  # backward compat alias
_HAND_YAW = HAND_YAW_RAD               # backward compat alias

# DROID Robotiq mount (ROBOLAB path).  Measured; see note above.  Rotation is
# a pure +90° about Y; translation is 0.12517 m along joint7 +Z.
T_JOINT7_TO_ROBOTIQ_BASE: np.ndarray = np.array(
    [
        [0.0,  0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0,  0.0, 0.0, 0.1251742],
        [0.0,  0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Joint limits (rad) — from Franka spec
JOINT_LIMITS_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
JOINT_LIMITS_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)
JOINT_MID = (JOINT_LIMITS_LOWER + JOINT_LIMITS_UPPER) / 2.0


# ======================================================================
# Rotation / transform primitives
# ======================================================================

def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)


def _make_transform(xyz, rpy) -> np.ndarray:
    """Build a 4×4 homogeneous transform from xyz translation + rpy rotation."""
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rotation(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = xyz
    return T


def _rotz_4x4(theta: float) -> np.ndarray:
    """4×4 rotation about z."""
    T = np.eye(4)
    T[:3, :3] = _Rz(theta)
    return T


# Pre-compute the fixed origin transforms (constant)
_ORIGIN_TRANSFORMS = [_make_transform(xyz, rpy) for xyz, rpy in _JOINT_ORIGINS]

def recompute_hand_transform() -> None:
    """Recompute the flange→hand fixed transform after changing HAND_YAW_RAD or FLANGE_Z_M.

    Call this if you override the module-level calibration constants before
    using FK or IK::

        import vlm_orchestrator.grasp.ik as franka_ik
        franka_ik.HAND_YAW_RAD = 0.0
        franka_ik.FLANGE_Z_M = 0.107
        franka_ik.recompute_hand_transform()
    """
    global _T_FLANGE_HAND
    _T_FLANGE_HAND = _build_flange_hand(FLANGE_Z_M, HAND_YAW_RAD)


def _build_flange_hand(flange_z: float, hand_yaw: float) -> np.ndarray:
    T_flange = np.eye(4)
    T_flange[2, 3] = flange_z
    T_hand = np.eye(4)
    T_hand[:3, :3] = _Rz(hand_yaw)
    return T_flange @ T_hand


_T_FLANGE_HAND = _build_flange_hand(FLANGE_Z_M, HAND_YAW_RAD)


# ======================================================================
# Forward Kinematics
# ======================================================================

def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Compute the 4×4 flange (panda_hand) pose in the robot base frame.

    Args:
        q: (7,) joint angles in radians.

    Returns:
        (4, 4) T_base_hand.
    """
    T = np.eye(4)
    for i in range(7):
        T = T @ _ORIGIN_TRANSFORMS[i] @ _rotz_4x4(q[i])
    T = T @ _T_FLANGE_HAND
    return T


def forward_kinematics_joint7(q: np.ndarray) -> np.ndarray:
    """FK to the *joint-7 link* frame, WITHOUT any hand/flange transform.

    This is the pure Franka arm chain (joints 1–7).  Used to empirically
    calibrate the fixed transform from joint7 to whatever end frame the sim
    actually controls (e.g. the Robotiq base_link), independent of the
    panda-hand assumptions baked into ``_T_FLANGE_HAND``.

    Args:
        q: (7,) joint angles in radians.

    Returns:
        (4, 4) T_base_joint7.
    """
    T = np.eye(4)
    for i in range(7):
        T = T @ _ORIGIN_TRANSFORMS[i] @ _rotz_4x4(q[i])
    return T


def forward_kinematics_robolab(q: np.ndarray) -> np.ndarray:
    """FK to the DROID Robotiq **base_link** — the frame the sim controls.

    Uses the empirically-measured joint7→base_link mount transform
    (``T_JOINT7_TO_ROBOTIQ_BASE``) instead of the panda-hand approximation.
    Output matches the sim's reported ee_pos/ee_quat exactly, so no
    downstream ``fk_vs_ee`` correction is needed.

    Args:
        q: (7,) joint angles in radians.

    Returns:
        (4, 4) T_base_robotiqbase.
    """
    return forward_kinematics_joint7(q) @ T_JOINT7_TO_ROBOTIQ_BASE


def forward_kinematics_with_tcp(
    q: np.ndarray, T_flange_to_tcp: np.ndarray,
) -> np.ndarray:
    """FK including a tool-center-point offset beyond the hand.

    Args:
        q: (7,) joint angles.
        T_flange_to_tcp: (4, 4) fixed transform from hand to TCP.

    Returns:
        (4, 4) T_base_tcp.
    """
    return forward_kinematics(q) @ T_flange_to_tcp


# ======================================================================
# Jacobian (numerical, central finite-difference)
# ======================================================================

def jacobian(
    q: np.ndarray,
    T_flange_to_tcp: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """6×7 geometric Jacobian.

    Rows 0–2: linear velocity (dx, dy, dz).
    Rows 3–5: angular velocity (wx, wy, wz).
    """
    fk = (
        (lambda qi: forward_kinematics_with_tcp(qi, T_flange_to_tcp))
        if T_flange_to_tcp is not None
        else forward_kinematics
    )

    J = np.zeros((6, 7))
    for i in range(7):
        q_p = q.copy(); q_p[i] += eps
        q_m = q.copy(); q_m[i] -= eps
        Tp = fk(q_p)
        Tm = fk(q_m)

        # Linear part
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)

        # Angular part (from skew-symmetric part of dR)
        dR = Tp[:3, :3] @ Tm[:3, :3].T
        J[3, i] = (dR[2, 1] - dR[1, 2]) / (2 * eps)
        J[4, i] = (dR[0, 2] - dR[2, 0]) / (2 * eps)
        J[5, i] = (dR[1, 0] - dR[0, 1]) / (2 * eps)

    return J


# ======================================================================
# Inverse Kinematics (Damped Least Squares)
# ======================================================================

def _pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """6-D pose error: [dx, dy, dz, ex, ey, ez]."""
    dp = T_target[:3, 3] - T_current[:3, 3]

    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    trace = np.clip(np.trace(R_err), -1.0, 3.0)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if abs(angle) < 1e-8:
        return np.concatenate([dp, np.zeros(3)])

    # Near 180°: skew = R[2,1]-R[1,2] etc. is ALWAYS ZERO for any 180° rotation
    # (since R = 2*n*n^T - I is symmetric), so the standard formula breaks down.
    # Extract the rotation axis from the positive-semidefinite part (R+I)/2 = n*n^T.
    if abs(angle - np.pi) < 0.05:
        M = (R_err + np.eye(3)) / 2.0  # = n * n^T
        norms_sq = np.array([M[0, 0], M[1, 1], M[2, 2]])
        i = int(np.argmax(norms_sq))
        ni = np.sqrt(max(norms_sq[i], 0.0))
        if ni < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = M[:, i] / ni
            axis = axis / (np.linalg.norm(axis) + 1e-10)
        return np.concatenate([dp, axis * angle])

    skew = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    axis = skew / (2 * np.sin(angle) + 1e-10)
    return np.concatenate([dp, axis * angle])


def inverse_kinematics(
    T_target: np.ndarray,
    q_init: np.ndarray,
    T_flange_to_tcp: np.ndarray | None = None,
    *,
    max_iters: int = 500,
    pos_tol: float = 5e-4,
    rot_tol: float = 5e-3,
    damping: float = 0.005,
    null_space_gain: float = 0.5,
) -> tuple[np.ndarray, bool]:
    """Solve IK via damped least squares with null-space joint centering.

    Returns ``(q_solution, converged)``.
    """
    fk = (
        (lambda qi: forward_kinematics_with_tcp(qi, T_flange_to_tcp))
        if T_flange_to_tcp is not None
        else forward_kinematics
    )

    q = q_init.copy().astype(np.float64)

    for it in range(max_iters):
        T_cur = fk(q)
        err = _pose_error(T_cur, T_target)

        pos_err = np.linalg.norm(err[:3])
        rot_err = np.linalg.norm(err[3:])

        if pos_err < pos_tol and rot_err < rot_tol:
            logger.debug(
                f"IK converged in {it + 1} iters: "
                f"pos={pos_err:.6f}m rot={rot_err:.6f}rad"
            )
            return q, True

        J = jacobian(q, T_flange_to_tcp)
        JJT = J @ J.T + (damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Null-space: push toward joint midpoints
        J_pinv_J = J.T @ np.linalg.solve(JJT, J)
        null_proj = np.eye(7) - J_pinv_J
        dq += null_proj @ (null_space_gain * (JOINT_MID - q))

        # Clamp step size
        norm = np.linalg.norm(dq)
        if norm > 0.2:
            dq *= 0.2 / norm

        q += dq
        q = np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    pos_err = np.linalg.norm(err[:3])
    rot_err = np.linalg.norm(err[3:])
    logger.warning(
        f"IK did NOT converge after {max_iters} iters: "
        f"pos={pos_err:.5f}m rot={rot_err:.5f}rad"
    )
    return q, False


# ======================================================================
# Multi-start IK (escape local minima)
# ======================================================================

def _fk_rot_err_deg(
    q: np.ndarray, T_target: np.ndarray,
    T_flange_to_tcp: np.ndarray | None = None,
) -> float:
    """Rotation angle (deg) between FK(q) and T_target.

    ``T_flange_to_tcp`` must match whatever frame ``T_target`` is expressed in
    (e.g. the Robotiq base_link relabel for the ROBOLAB path); otherwise the
    acceptance check compares mismatched frames.
    """
    fk = forward_kinematics(q) if T_flange_to_tcp is None else \
        forward_kinematics_with_tcp(q, T_flange_to_tcp)
    R_rel = T_target[:3, :3] @ fk[:3, :3].T
    tr = float(np.clip(np.trace(R_rel), -1.0, 3.0))
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))))


def _fk_pos_err_m(
    q: np.ndarray, T_target: np.ndarray,
    T_flange_to_tcp: np.ndarray | None = None,
) -> float:
    """Cartesian distance (m) between FK(q) and T_target."""
    fk = forward_kinematics(q) if T_flange_to_tcp is None else \
        forward_kinematics_with_tcp(q, T_flange_to_tcp)
    return float(np.linalg.norm(fk[:3, 3] - T_target[:3, 3]))


def inverse_kinematics_multistart(
    T_target: np.ndarray,
    q_primary: np.ndarray,
    q_canonical: np.ndarray | None = None,
    *,
    accept_rot_deg: float = 2.0,
    accept_pos_m: float = 5e-3,
    max_joint_dist_rad: float | None = None,
    reference_q: np.ndarray | None = None,
    random_seeds: int = 3,
    rng: np.random.Generator | None = None,
    **ik_kwargs,
) -> tuple[np.ndarray, bool, str]:
    """IK with a ladder of seed strategies to escape local minima.

    Strategies, tried in order, stopping at the first that satisfies all
    acceptance criteria:

      1. ``q_primary`` (caller's seed)
      2. ``q_primary`` with wrist flip (±π on joint 6) — cheap branch jump
      3. ``q_canonical`` (if provided) — canonical well-conditioned reset
      4. Up to ``random_seeds`` random draws from ``JOINT_MID ± 1.5`` rad

    Acceptance criteria:
      - IK reports ``converged=True`` (internal pos+rot tolerances)
      - FK rotation error < ``accept_rot_deg`` (guards against the ~180°
        ``_pose_error`` singularity which can false-positive convergence)
      - FK position error < ``accept_pos_m``
      - If ``max_joint_dist_rad`` is set: ``||q - ref||_inf <= max_joint_dist_rad``
        where ``ref`` is ``reference_q`` if provided else ``q_primary``.  Enforces
        branch consistency — rejects solutions that jump to a far IK branch
        (which would make joint-space trajectory interpolation pass through
        garbage configurations).

    Returns ``(q_best, converged, strategy_label)``.  When no attempt passes,
    ``converged=False`` is returned with the best-effort joint config and the
    label suffixed with ``(best-effort)``.  Callers should treat this as failure.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ref = reference_q if reference_q is not None else q_primary
    # The acceptance FK checks must use the same tool frame as the IK solve so
    # they compare like-for-like (e.g. ROBOLAB passes the Robotiq base_link
    # relabel via T_flange_to_tcp).
    _tcp = ik_kwargs.get("T_flange_to_tcp")

    def joint_dist(q: np.ndarray) -> float:
        return float(np.max(np.abs(q - ref)))

    def try_seed(q_seed: np.ndarray, label: str):
        q_seed = np.clip(q_seed, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
        q, conv = inverse_kinematics(T_target, q_seed, **ik_kwargs)
        return (
            q,
            conv,
            _fk_rot_err_deg(q, T_target, _tcp),
            _fk_pos_err_m(q, T_target, _tcp),
            joint_dist(q),
            label,
        )

    def acceptable(res) -> bool:
        _, conv, rot_deg, pos_m, jd, _ = res
        if not conv or rot_deg >= accept_rot_deg or pos_m >= accept_pos_m:
            return False
        if max_joint_dist_rad is not None and jd > max_joint_dist_rad:
            return False
        return True

    attempts: list = []
    res = try_seed(q_primary, "primary")
    attempts.append(res)
    if acceptable(res):
        return res[0], True, res[5]

    for delta, lbl in [(np.pi, "wrist+π"), (-np.pi, "wrist-π")]:
        q_seed = q_primary.copy()
        q_seed[6] = q_primary[6] + delta
        res = try_seed(q_seed, lbl)
        attempts.append(res)
        if acceptable(res):
            return res[0], True, res[5]

    if q_canonical is not None:
        res = try_seed(q_canonical, "canonical")
        attempts.append(res)
        if acceptable(res):
            return res[0], True, res[5]

    for i in range(random_seeds):
        q_seed = JOINT_MID + rng.uniform(-1.5, 1.5, 7)
        res = try_seed(q_seed, f"random{i}")
        attempts.append(res)
        if acceptable(res):
            return res[0], True, res[5]

    # No attempt met acceptance criteria — return a best-effort result
    # and signal failure with converged=False.  Caller should abort.
    def score(a):
        # prefer (converged > not), then small pos err, then small rot err
        return (not a[1], a[3], a[2])
    best = min(attempts, key=score)
    return best[0], False, f"{best[5]}(best-effort)"


# ======================================================================
# Trajectory helpers
# ======================================================================

def interpolate_joints(
    q_start: np.ndarray,
    q_end: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    """Linear interpolation in joint space (includes both endpoints)."""
    if num_steps < 2:
        return [q_end.copy()]
    return [
        q_start * (1 - a) + q_end * a
        for a in np.linspace(0.0, 1.0, num_steps)
    ]


# ======================================================================
# Calibration helper
# ======================================================================

def calibrate_flange_to_tcp(
    q_current: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Compute T_flange_to_tcp from observed EE pose.

    ``FK(q) @ T_flange_to_tcp ≈ T_ee_observed``
    """
    from vlm_orchestrator.grasp.camera import _pose_to_matrix

    T_fk = forward_kinematics(q_current)
    T_ee = _pose_to_matrix(ee_pos, ee_quat_wxyz)
    return np.linalg.inv(T_fk) @ T_ee
