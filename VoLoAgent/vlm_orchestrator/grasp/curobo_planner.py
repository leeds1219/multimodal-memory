# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side cuRobo motion planner (loaded only inside the grasp server).

Wraps cuRobo v0.8.0 ``MotionPlanner.plan_cspace`` (joint→joint) so the
orchestrator can request a collision-free trajectory between two joint
configurations it already solved with its own IK.  The scene point cloud
(built for GraspGen) is optionally injected as a mesh obstacle.

This module imports cuRobo lazily and is only touched when the server is
started with ``--enable-curobo``.  cuRobo + ``cuda-core[cu12]`` must be
installed in the grasp-server conda env (see
``workspace/curobo_spike_findings.md``).

Design (matches the "no silent lossy fallback" design rule):
* ``plan_cspace`` returns ``(waypoints, True)`` on success or
  ``(None, False)`` on failure.  The caller decides what to do — the
  orchestrator surfaces the failure via a named planner label; it never
  silently substitutes a straight line here.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Number of Franka arm joints the orchestrator's IK produces / consumes.
_ARM_DOF = 7

# Voxel size (m) used when meshing the obstacle point cloud via
# ``Mesh.from_pointcloud``.  Smooths depth noise into a solid surface and sets
# the obstacle inflation.  1 cm is a good default for our tabletop scenes;
# override via ``GRASP_SCENE_PC_VOXEL_PITCH_M`` for tuning.
_SCENE_PC_VOXEL_PITCH_M = float(
    os.environ.get("GRASP_SCENE_PC_VOXEL_PITCH_M", "0.01")
)

# Franka gripper links whose collision spheres are disabled during the
# grasp/place APPROACH segment (the gripper reaches down toward an on-table
# object; its fingers extend ~10 cm below panda_hand and would otherwise trip
# the table collision check).  These are exactly ``franka.yml``'s
# ``grasp_contact_link_names`` (minus ``attached_object``, which the place path
# manages via the AttachmentManager).  Matches cuRobo's own ``plan_grasp``.
GRASP_APPROACH_DISABLE_LINKS = [
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
]


class CuroboMotionPlanner:
    """Thin wrapper over cuRobo ``MotionPlanner`` for joint→joint planning."""

    def __init__(
        self,
        robot_config: str = "franka.yml",
        scene_model: str = "collision_test.yml",
        warmup_iterations: int = 5,
    ):
        import torch  # noqa: F401  (ensures torch present before cuRobo)
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self._torch = __import__("torch")
        logger.info(
            f"Loading cuRobo MotionPlanner (robot={robot_config}, "
            f"scene={scene_model})..."
        )
        # Pre-allocate a collision cache so runtime ``update_world`` calls
        # (scene point cloud → mesh, and the attached-object cuboid) have
        # somewhere to load into.  Without this cuRobo raises "Mesh cache not
        # initialized" and the scene update is silently dropped, leaving the
        # planner collision-UNAWARE.  ``mesh`` covers the scene obstacle
        # cloud; ``cuboid`` covers the attached held-object box + headroom.
        cfg = MotionPlannerCfg.create(
            robot=robot_config, scene_model=scene_model,
            collision_cache={"mesh": 4, "cuboid": 16},
        )
        self._planner = MotionPlanner(cfg)
        self._planner.warmup(
            enable_graph=True, num_warmup_iterations=warmup_iterations,
        )
        # cuRobo's active joint ordering (arm joints, in planner order).
        self._joint_names = list(self._planner.joint_names)
        logger.info(
            f"cuRobo MotionPlanner ready (joints={self._joint_names})"
        )

    # ------------------------------------------------------------------

    def _to_joint_state(self, q_arm: np.ndarray):
        """Build a cuRobo ``JointState`` from a (7,) arm config.

        Only the arm joints are set; cuRobo fills the rest from the robot
        model defaults.  ``joint_names`` is passed so ordering is explicit.
        """
        from curobo.types import JointState

        t = self._torch.as_tensor(
            np.asarray(q_arm, dtype=np.float32)[None, :],
            device="cuda",
        )
        return JointState.from_position(
            t, joint_names=self._joint_names[:len(q_arm)],
        )

    def update_scene(self, scene_pc: Optional[np.ndarray]) -> None:
        """Replace the collision world with a mesh built from ``scene_pc``.

        ``scene_pc`` is an ``(N, 3)`` point cloud in the robot base frame.

        Passing ``None`` / empty **clears the world to empty** (free space).
        This matters: the planner is built with ``scene_model="collision_test.yml"``
        which loads a default table + box.  In the ROBOLAB path the runtime
        depth cloud is the authoritative scene — the default test obstacles
        must NOT leak through and block a grasp/place when the caller has
        (legitimately) no obstacle cloud to send (e.g. after support-plane
        removal empties the cloud, or on a linear→curobo first call).  Leaving
        the default scene in place silently blocked on-table approaches
        (observed: a pre-grasp at z≤0.12 collided with the default table top
        at z=0).  Failures here are logged and swallowed — planning then
        proceeds against whatever world is currently loaded; this is NOT a
        silent *pose* fallback.
        """
        if scene_pc is None or len(scene_pc) == 0:
            # No obstacle cloud → clear the (default) world to empty so stale
            # test obstacles don't block the plan.  Also evict our cached mesh.
            try:
                self._evict_mesh_cache_entry("scene_pc")
                if hasattr(self._planner, "clear_scene_cache"):
                    self._planner.clear_scene_cache()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"cuRobo clear-scene failed ({type(e).__name__}: {e}); "
                    f"planning against existing world"
                )
            return
        try:
            from curobo.scene import Mesh, Scene

            # Build the obstacle mesh with cuRobo's own voxelized-surface
            # extractor (``Mesh.from_pointcloud``), NOT ``PointCloud.get_mesh``.
            # ``PointCloud.get_mesh`` routes through ``get_trimesh_mesh`` (a
            # marching-cubes / trimesh path) which in v0.8.0 emits a malformed
            # mesh whose faces index past the vertex array — it crashed sphere
            # fitting / warp upload with ``IndexError: index N out of bounds``.
            # ``Mesh.from_pointcloud`` voxelizes the cloud and stamps quad faces
            # on occupied/empty voxel boundaries — a watertight mesh with no
            # scikit-image / marching-cubes dependency.  ``pitch`` is the voxel
            # size: it both smooths depth noise and sets the obstacle inflation,
            # so keep it modest (1 cm) — large enough to fuse noisy points into
            # a solid surface, small enough not to bloat obstacles into the
            # gripper's approach corridor.
            mesh = Mesh.from_pointcloud(
                np.asarray(scene_pc, dtype=np.float32),
                pitch=_SCENE_PC_VOXEL_PITCH_M,
                name="scene_pc",
            )

            # cuRobo caches warp meshes BY NAME and never evicts on reload
            # (``MeshData._load_mesh_into_cache`` logs "Mesh already in cache,
            # reusing existing instance" and returns the STALE geometry).
            # ``update_world`` → ``load_from_scene_cfg`` → ``clear()`` only
            # clears the env slot, not the warp cache (``clear_warp_cache``
            # defaults False).  So every scene update after the first would
            # silently reuse the FIRST cloud's geometry — the planner goes
            # collision-stale after one call.  Evict our reused "scene_pc"
            # key before reloading so the new cloud actually takes effect.
            self._evict_mesh_cache_entry("scene_pc")

            scene = Scene(mesh=[mesh])
            self._planner.update_world(scene)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"cuRobo scene update from point cloud failed "
                f"({type(e).__name__}: {e}); planning against existing world"
            )

    def _evict_mesh_cache_entry(self, name: str) -> None:
        """Remove ``name`` from cuRobo's warp mesh cache so a re-``update_world``
        with the same mesh name loads FRESH geometry instead of silently
        reusing the stale cached instance.

        The cache lives at
        ``planner.scene_collision_checker.data.meshes.wp_cache`` (a
        ``Dict[str, WarpMeshCache]``).  Best-effort: any missing attribute or
        key is a no-op (first call has nothing to evict).
        """
        try:
            scc = getattr(self._planner, "scene_collision_checker", None)
            if scc is None:
                return
            meshes = getattr(getattr(scc, "data", None), "meshes", None)
            wp_cache = getattr(meshes, "wp_cache", None)
            if isinstance(wp_cache, dict):
                wp_cache.pop(name, None)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"_evict_mesh_cache_entry({name}) no-op: {e}")

    def _disable_link_collisions(self, links: Optional[list[str]]) -> list[str]:
        """Disable collision spheres for ``links`` (best-effort).

        Returns the list of links actually toggled off so the caller can
        re-enable exactly those in a ``finally``.  Uses cuRobo's own
        ``MotionPlanner.disable_link_collision`` (the same call ``plan_grasp``
        makes).  Any missing method / bad link name is a logged no-op — we
        never want a failed toggle to abort planning.
        """
        if not links:
            return []
        fn = getattr(self._planner, "disable_link_collision", None)
        if fn is None:
            logger.debug("disable_link_collision unavailable; skipping toggle")
            return []
        try:
            fn(links)
            logger.info(
                f"  cuRobo: disabled collision for links {links} "
                f"(grasp-approach reach — matches plan_grasp)"
            )
            return list(links)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"cuRobo disable_link_collision({links}) failed "
                f"({type(e).__name__}: {e}); planning with full collision"
            )
            return []

    def _enable_link_collisions(self, links: list[str]) -> None:
        """Re-enable collision spheres for ``links`` (restore after a plan)."""
        if not links:
            return
        fn = getattr(self._planner, "enable_link_collision", None)
        if fn is None:
            return
        try:
            fn(links)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"cuRobo enable_link_collision({links}) failed "
                f"({type(e).__name__}: {e}); collision state may be degraded"
            )

    # ------------------------------------------------------------------
    # Attached-object collision (for the PLACE path)
    # ------------------------------------------------------------------
    # When the robot carries an object, that object must be treated as part
    # of the robot (attached to the gripper link) rather than as a fixed
    # world obstacle in front of the hand.  cuRobo's AttachmentManager fits
    # a few spheres to the held-object geometry and rigidly attaches them to
    # the ``attached_object`` link so they move with the hand each planning
    # iteration while still being collision-checked against the world (so we
    # don't ram the carried object into a bin rim).
    #
    # NOTE (franka.yml): the ``attached_object`` link has only 4 sphere
    # slots (``extra_collision_spheres: attached_object: 4``).  Fitting more
    # than 4 raises in AttachmentManager.update — so num_spheres is capped
    # at 4 here.  A coarse 4-sphere hull is adequate for our small objects.

    _MAX_ATTACH_SPHERES = 4

    def _attachment_manager(self):
        """Return cuRobo's AttachmentManager for this planner.

        The ``MotionPlanner.attachment_manager`` property in curobo v0.8.0
        delegates to ``trajopt_solver.attachment_manager``, but ``TrajOptSolver``
        wraps ``SolverCore`` as ``.core`` and does NOT re-expose that attribute
        — so the property raises ``AttributeError`` on a live server.  The
        manager actually lives at ``trajopt_solver.core.attachment_manager``.
        Try the documented property first, then fall through to the real
        location so we're robust to either layout.
        """
        planner = self._planner
        for getter in (
            lambda: planner.attachment_manager,
            lambda: planner.trajopt_solver.core.attachment_manager,
            lambda: planner.trajopt_solver.attachment_manager,
        ):
            try:
                am = getter()
                if am is not None:
                    return am
            except AttributeError:
                continue
        raise AttributeError(
            "cuRobo AttachmentManager not reachable via "
            "MotionPlanner.attachment_manager or "
            "trajopt_solver.core.attachment_manager"
        )

    def attach_object(
        self,
        obj_pc_base: np.ndarray,
        q_hold: np.ndarray,
        *,
        num_spheres: int = _MAX_ATTACH_SPHERES,
    ) -> bool:
        """Attach a held-object point cloud to the gripper link.

        ``obj_pc_base`` is an ``(N, 3)`` cloud of the CARRIED object in the
        robot base frame (== world in the clean ROBOLAB path).  ``q_hold`` is
        the (7,) arm config at which the object is held (the current config
        at place-approach start).  The cloud is expressed in the base/world
        frame, so ``world_objects_pose_offset`` is identity and cuRobo maps
        the sphere centres world→link via FK at ``q_hold``.

        Geometry: we approximate the held object by its **axis-aligned
        bounding box** (a ``Cuboid`` obstacle) rather than the raw cloud.
        Two reasons:
          1. cuRobo v0.8.0's ``Mesh.from_pointcloud`` emits a malformed
             voxel mesh (faces index past the vertex array), which crashes
             trimesh during sphere fitting — a bounding-box proxy sidesteps
             that path entirely.
          2. The Franka ``attached_object`` link has only 4 sphere slots
             (``franka.yml``), so fine surface geometry cannot be captured
             anyway; a conservative box is the right fidelity for 4 spheres
             and matches cuRobo's own attach-object test fixtures.
        This is a documented, loud approximation — NOT a silent lossy
        fallback: the box is always an over-approximation (never smaller
        than the object), so collisions are conservative.

        Returns True on success.  On failure logs and returns False (the
        caller decides whether to proceed collision-unaware or abort — this
        is NOT a silent pose fallback).
        """
        if obj_pc_base is None or len(obj_pc_base) == 0:
            logger.warning("attach_object: empty cloud; nothing attached")
            return False
        try:
            from curobo._src.geom.types import Cuboid
            from curobo.types import JointState, Pose

            n = min(int(num_spheres), self._MAX_ATTACH_SPHERES)
            pts = np.asarray(obj_pc_base, dtype=np.float32)

            # Axis-aligned bounding box of the held-object cloud (world/base
            # frame).  Clamp each dimension to a small minimum so a near-flat
            # object still yields a non-degenerate box.
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
            center = (lo + hi) / 2.0
            dims = np.maximum(hi - lo, 0.01)
            obj = Cuboid(
                name="attached_object",
                pose=[
                    float(center[0]), float(center[1]), float(center[2]),
                    1.0, 0.0, 0.0, 0.0,
                ],
                dims=[float(dims[0]), float(dims[1]), float(dims[2])],
            )
            mesh = obj

            q = self._torch.as_tensor(
                np.asarray(q_hold, dtype=np.float32)[None, :], device="cuda",
            )
            js = JointState.from_position(
                q, joint_names=self._joint_names[:len(q_hold)],
            )
            # The mesh points (and hence the fitted sphere centres) are in the
            # world/base frame.  Passing an IDENTITY world pose makes the
            # AttachmentManager compute obj→link = ee_pose(q_hold)⁻¹, i.e. it
            # maps those world-frame centres into the gripper-link frame.  With
            # None it would instead assume the centres are already link-local
            # (wrong here — they'd land at the arm base).
            identity = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

            am = self._attachment_manager()
            am.attach(
                js,
                [mesh],
                link_name="attached_object",
                num_spheres=n,
                world_objects_pose_offset=identity,
            )
            self._planner.enable_link_collision(["attached_object"])
            logger.info(
                f"attach_object: attached {len(pts)}-pt held object as an "
                f"AABB box (dims={dims.round(3).tolist()} m) → {n} spheres "
                f"on 'attached_object' link"
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"attach_object failed ({type(e).__name__}: {e}); "
                f"nothing attached"
            )
            return False

    def detach_object(self) -> None:
        """Detach the held object and re-enable any disabled world obstacles.

        Safe to call even if nothing was attached (no-op)."""
        try:
            self._planner.disable_link_collision(["attached_object"])
            self._attachment_manager().detach(link_name="attached_object")
            logger.info("detach_object: detached 'attached_object' link")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"detach_object failed ({type(e).__name__}: {e})"
            )

    def plan_cspace(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        *,
        n_steps: Optional[int] = None,
        scene_pc: Optional[np.ndarray] = None,
        disable_links: Optional[list[str]] = None,
    ) -> tuple[Optional[np.ndarray], bool]:
        """Plan a collision-free joint trajectory ``q_start`` → ``q_end``.

        Returns ``(waypoints, True)`` where ``waypoints`` is
        ``(M, arm_dof)``, or ``(None, False)`` on failure.  If ``n_steps``
        is given, the cuRobo interpolated path is resampled to exactly
        ``n_steps`` waypoints (linear index resample) so it matches the
        orchestrator's fixed per-segment chunk cadence.

        ``disable_links`` temporarily removes those links' collision spheres
        for the duration of this plan (restored in ``finally``).  This is
        exactly what cuRobo's own ``MotionPlanner.plan_grasp`` does for the
        approach segment: the gripper's finger/hand links
        (``grasp_contact_link_names`` in ``franka.yml``:
        ``panda_hand``/``panda_leftfinger``/``panda_rightfinger``) extend ~10 cm
        below ``panda_hand``, so a top-down pre-grasp 8 cm above an on-table
        object puts the fingertip spheres AT the table → the goal state is
        genuinely "in collision" and cuRobo rejects the plan before optimizing.
        Disabling the finger links (arm links still checked) lets the gripper
        reach down to the object while real obstacles (bins, walls, other
        objects) are still avoided.  Loud + documented — NOT a lossy fallback.
        """
        q_start = np.asarray(q_start, dtype=np.float32)
        q_end = np.asarray(q_end, dtype=np.float32)

        self.update_scene(scene_pc)

        js_start = self._to_joint_state(q_start)
        js_goal = self._to_joint_state(q_end)

        toggled = self._disable_link_collisions(disable_links)
        try:
            result = self._planner.plan_cspace(js_goal, js_start)
        finally:
            self._enable_link_collisions(toggled)
        if result is None or not bool(result.success.any()):
            # Optional debug dump of the exact failing problem so it can be
            # replayed offline (GRASP_DUMP_FAILED_PLANS=<dir>).  Off by default.
            _dump_dir = os.environ.get("GRASP_DUMP_FAILED_PLANS")
            if _dump_dir:
                try:
                    os.makedirs(_dump_dir, exist_ok=True)
                    import time as _t
                    fn = os.path.join(_dump_dir, f"failed_{int(_t.time()*1000)}.npz")
                    np.savez(
                        fn,
                        q_start=q_start, q_end=q_end,
                        scene_pc=(scene_pc if scene_pc is not None
                                  else np.zeros((0, 3), np.float32)),
                        disable_links=np.array(disable_links or [], dtype=object),
                    )
                    logger.info(f"  dumped failed plan → {fn}")
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"failed-plan dump no-op: {e}")
            return None, False

        interp = result.get_interpolated_plan()
        # position may carry leading batch dims → flatten to (M, dof)
        pos = interp.position.reshape(-1, interp.position.shape[-1])
        pos = pos.detach().cpu().numpy()

        # Keep only the arm joints (first _ARM_DOF columns / matching count).
        arm_dof = min(_ARM_DOF, pos.shape[-1], len(q_end))
        pos = pos[:, :arm_dof]

        if n_steps is not None and n_steps >= 2 and len(pos) != n_steps:
            pos = _resample_path(pos, n_steps)

        return pos, True


def _resample_path(path: np.ndarray, n_steps: int) -> np.ndarray:
    """Resample an ``(M, dof)`` path to exactly ``n_steps`` rows by linear
    interpolation over the normalized path index.  Endpoints preserved."""
    m = len(path)
    if m == n_steps:
        return path
    src = np.linspace(0.0, 1.0, m)
    dst = np.linspace(0.0, 1.0, n_steps)
    out = np.empty((n_steps, path.shape[1]), dtype=path.dtype)
    for j in range(path.shape[1]):
        out[:, j] = np.interp(dst, src, path[:, j])
    return out


# Module-level singleton (loaded once at server startup).
_curobo_planner: Optional[CuroboMotionPlanner] = None


def load_curobo(
    robot_config: str = "franka.yml",
    scene_model: str = "collision_test.yml",
) -> None:
    global _curobo_planner
    if _curobo_planner is None:
        _curobo_planner = CuroboMotionPlanner(
            robot_config=robot_config, scene_model=scene_model,
        )


def get_curobo() -> Optional[CuroboMotionPlanner]:
    return _curobo_planner
