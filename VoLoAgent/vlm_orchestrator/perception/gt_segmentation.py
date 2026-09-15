# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth segmentation providers for each simulator backend.

Each provider renders a per-pixel segmentation from the simulator's
internal state, bypassing vision-based detection (GDino) and
segmentation (SAM2/SAM3) entirely.  The result is a perfect binary mask
for a named target object, suitable for the grasp tool's point-cloud
masking step.

Supported simulators
~~~~~~~~~~~~~~~~~~~~

- **MuJoCo / robosuite** (LIBERO, RoboCasa):
  ``sim.render(camera_name, H, W, segmentation=True)`` returns a
  ``(H, W, 2)`` array where ``[:,:,1]`` contains per-pixel body IDs.
  Map target object name → ``env.obj_body_id[name]`` → binary mask.

- **dm_control / MuJoCo** (VLABench):
  ``physics.render(H, W, camera_id=N, segmentation=True)`` returns a
  ``(H, W, 2)`` array with the same format.

- **Isaac Lab / PhysX** (DROID Sim):
  Add ``"instance_segmentation_fast"`` to the camera sensor's
  ``data_types``.  The annotator produces a ``(H, W)`` integer array
  of instance IDs that can be mapped to object names via the USD stage.

Usage in eval clients
~~~~~~~~~~~~~~~~~~~~~

Each eval client creates the appropriate provider and calls
``render_mask()`` to get a binary mask for a target object::

    # LIBERO example:
    from vlm_orchestrator.perception.gt_segmentation import RobosuiteSegProvider

    seg = RobosuiteSegProvider(env, camera_name="agentview")
    mask = seg.render_mask("alphabet_soup_1")  # (H, W) bool
    wire_obs["gt_seg/mask"] = mask
    wire_obs["gt_seg/target"] = "alphabet_soup_1"

The proxy's grasp tool (``GraspSegMode.GT_SIM``) reads these keys
instead of running GDino + SAM2.

Wire format
~~~~~~~~~~~

When GT segmentation is active, eval clients add two keys to the wire
observation dict:

- ``gt_seg/mask``: ``np.ndarray`` shape ``(H, W)``, dtype ``bool`` —
  binary mask of the target object.
- ``gt_seg/target``: ``str`` — name of the segmented object (for
  verification / logging).

These keys are stripped by the proxy before forwarding to the VLA
policy server (the policy never sees them).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Abstract base
# ======================================================================


class GTSegProvider(ABC):
    """Base class for GT segmentation providers."""

    @abstractmethod
    def render_mask(self, target_object: str) -> np.ndarray:
        """Render a binary mask for *target_object*.

        Returns
        -------
        mask : ndarray, shape (H, W), dtype bool
            True where the target object is visible.

        Raises
        ------
        ValueError
            If *target_object* is not found in the scene.
        RuntimeError
            If the simulator cannot render segmentation.
        """

    @abstractmethod
    def available_objects(self) -> list[str]:
        """Return names of all objects that can be segmented."""

    def render_mask_for_obs(
        self,
        target_object: str,
        wire_obs: dict,
    ) -> dict:
        """Render and inject GT segmentation into a wire obs dict.

        Convenience method: renders the mask and adds the ``gt_seg/*``
        keys that the proxy's grasp tool expects.

        Returns the modified wire_obs (also modifies in-place).
        """
        mask = self.render_mask(target_object)
        wire_obs["gt_seg/mask"] = mask
        wire_obs["gt_seg/target"] = target_object
        return wire_obs


# ======================================================================
# MuJoCo / robosuite (LIBERO, RoboCasa)
# ======================================================================


class RobosuiteSegProvider(GTSegProvider):
    """GT segmentation from robosuite's MuJoCo renderer.

    Works with LIBERO and RoboCasa environments that use robosuite.

    ``sim.render(camera_name, H, W, segmentation=True)`` returns a
    ``(H, W, 2)`` int32 array:

    - Channel 0: object type (5 = geom, -1 for background)
    - Channel 1: object ID within that type (geom_id, -1 for background)

    Since each MuJoCo object can have multiple geoms but shares a
    single body, callers should convert geom IDs → body IDs via
    ``sim.model.geom_bodyid[geom_id]`` for body-level matching.

    Parameters
    ----------
    env :
        A robosuite / LIBERO / RoboCasa environment instance.  The
        provider navigates wrappers to find the inner env with ``sim``.
    camera_name : str
        MuJoCo camera to render from (e.g. ``"agentview"``).
    height, width : int
        Resolution of the segmentation render.  Should match the
        resolution of the RGB/depth images used by the grasp tool.
    """

    def __init__(
        self,
        env,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ):
        # Navigate through wrappers to the robosuite env
        inner = env
        while hasattr(inner, "env") and inner.env is not inner:
            inner = inner.env
        self._env = inner
        self._camera_name = camera_name
        self._height = height
        self._width = width

        # Build name → body_id mapping
        self._obj_body_id: dict[str, int] = {}

        # robosuite envs have obj_body_id
        obj_body_id = getattr(self._env, "obj_body_id", {})
        self._obj_body_id.update(obj_body_id)

        # Also map via sim.model for objects not in obj_body_id
        sim = getattr(self._env, "sim", None)
        if sim is not None:
            for i in range(sim.model.nbody):
                name = sim.model.body_id2name(i)
                if name and name not in self._obj_body_id:
                    self._obj_body_id[name] = i

        logger.info(
            f"[RobosuiteSeg] Initialized: camera={camera_name}, "
            f"res={height}x{width}, "
            f"{len(self._obj_body_id)} bodies mapped"
        )

    def render_mask(self, target_object: str) -> np.ndarray:
        sim = getattr(self._env, "sim", None)
        if sim is None:
            raise RuntimeError(
                "Environment has no 'sim' attribute — cannot render "
                "segmentation."
            )

        # Resolve target object → body ID
        body_id = self._resolve_body_id(target_object)

        # Render segmentation
        seg = sim.render(
            camera_name=self._camera_name,
            height=self._height,
            width=self._width,
            segmentation=True,
        )
        # seg shape: (H, W, 2) — channel 0 = objtype, channel 1 = geom_id.
        # robosuite renders images upside-down (origin at bottom-left);
        # flip to match the RGB images.
        seg = seg[::-1]

        # Convert geom IDs → body IDs via model.geom_bodyid
        geom_ids = seg[:, :, 1].astype(np.int32)
        geom_bodyid = sim.model.geom_bodyid
        body_ids = np.where(
            geom_ids >= 0,
            geom_bodyid[np.clip(geom_ids, 0, len(geom_bodyid) - 1)],
            -1,
        ).astype(np.int32)
        mask = (body_ids == body_id)

        n_pixels = int(mask.sum())
        if n_pixels == 0:
            logger.warning(
                f"[RobosuiteSeg] Object '{target_object}' (body_id={body_id}) "
                f"has 0 visible pixels from camera '{self._camera_name}'. "
                f"Object may be occluded or out of view."
            )

        logger.debug(
            f"[RobosuiteSeg] '{target_object}' → body_id={body_id}, "
            f"{n_pixels} pixels ({n_pixels * 100 / mask.size:.1f}%)"
        )
        return mask

    def available_objects(self) -> list[str]:
        # Return objects of interest first, then all bodies
        parsed = getattr(self._env, "parsed_problem", None)
        if isinstance(parsed, dict):
            obj_of_interest = parsed.get("obj_of_interest", [])
        else:
            obj_of_interest = []
        all_names = list(self._obj_body_id.keys())
        # Deduplicate while preserving order
        seen = set()
        result = []
        for name in list(obj_of_interest) + all_names:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def _resolve_body_id(self, target_object: str) -> int:
        """Resolve a target object name to a MuJoCo body ID.

        Tries exact match first, then partial match (object names in
        robosuite often have ``_main`` suffix or numeric suffixes).
        """
        # Exact match
        if target_object in self._obj_body_id:
            return self._obj_body_id[target_object]

        # Try with _main suffix (robosuite body naming convention)
        with_main = f"{target_object}_main"
        if with_main in self._obj_body_id:
            return self._obj_body_id[with_main]

        # Partial match: target_object is a substring of a body name
        candidates = [
            (name, bid) for name, bid in self._obj_body_id.items()
            if target_object.lower() in name.lower()
        ]
        if len(candidates) == 1:
            name, bid = candidates[0]
            logger.info(
                f"[RobosuiteSeg] Fuzzy match: '{target_object}' → "
                f"'{name}' (body_id={bid})"
            )
            return bid
        if len(candidates) > 1:
            # Take the shortest name (most specific match)
            candidates.sort(key=lambda x: len(x[0]))
            name, bid = candidates[0]
            logger.info(
                f"[RobosuiteSeg] Multiple fuzzy matches for "
                f"'{target_object}', using '{name}' (body_id={bid})"
            )
            return bid

        raise ValueError(
            f"Object '{target_object}' not found in scene. "
            f"Available: {list(self._obj_body_id.keys())[:20]}"
        )


# ======================================================================
# dm_control / MuJoCo (VLABench)
# ======================================================================


class DMControlSegProvider(GTSegProvider):
    """GT segmentation from dm_control's MuJoCo renderer.

    Works with VLABench environments that use dm_control.

    ``physics.render(H, W, camera_id=N, segmentation=True)`` returns a
    ``(H, W, 2)`` int32 array:
    channel 0 = object ID (geom_id for geoms), channel 1 = object type
    (mjtObj enum, e.g. 5 = geom).  Note: dm_control's channel order
    is *swapped* compared to robosuite.  We use
    ``physics.model.geom_bodyid`` to convert geom IDs → body IDs.

    Parameters
    ----------
    env :
        A VLABench environment with a ``physics`` attribute.
    camera_id : int
        MuJoCo camera index to render from.  VLABench default cameras:
        0=right, 1=left, 2=front, 3=wrist.  Front (2) is typically best
        for grasp planning.
    height, width : int
        Segmentation render resolution.
    """

    def __init__(
        self,
        env,
        camera_id: int = 2,
        height: int = 256,
        width: int = 256,
    ):
        self._env = env
        self._camera_id = camera_id
        self._height = height
        self._width = width

        # Build name → body_id mapping from the physics model
        self._name_to_body_id: dict[str, int] = {}
        physics = getattr(env, "physics", None)
        if physics is not None:
            model = physics.model
            for i in range(model.nbody):
                name = model.id2name(i, "body")
                if name:
                    self._name_to_body_id[name] = i

        # Also map geom → body for geom-level matching
        self._geom_to_body: dict[int, int] = {}
        if physics is not None:
            for i in range(physics.model.ngeom):
                self._geom_to_body[i] = int(physics.model.geom_bodyid[i])

        logger.info(
            f"[DMControlSeg] Initialized: camera_id={camera_id}, "
            f"res={height}x{width}, "
            f"{len(self._name_to_body_id)} bodies mapped"
        )

    def render_mask(self, target_object: str) -> np.ndarray:
        physics = getattr(self._env, "physics", None)
        if physics is None:
            raise RuntimeError(
                "Environment has no 'physics' attribute — cannot render "
                "segmentation."
            )

        body_id = self._resolve_body_id(target_object)

        # dm_control segmentation render
        seg = physics.render(
            height=self._height,
            width=self._width,
            camera_id=self._camera_id,
            segmentation=True,
        )
        # seg shape: (H, W, 2) — channel 0 = geom_id, channel 1 = type_id
        # We need to map geom_ids → body_ids
        geom_ids = seg[:, :, 0]

        # Build mask: pixel belongs to target if its geom's body matches
        mask = np.zeros((self._height, self._width), dtype=bool)
        unique_geoms = np.unique(geom_ids)
        for gid in unique_geoms:
            if gid < 0:
                continue  # background
            if gid in self._geom_to_body:
                if self._geom_to_body[gid] == body_id:
                    mask |= (geom_ids == gid)

        n_pixels = int(mask.sum())
        logger.debug(
            f"[DMControlSeg] '{target_object}' → body_id={body_id}, "
            f"{n_pixels} pixels ({n_pixels * 100 / mask.size:.1f}%)"
        )
        return mask

    def available_objects(self) -> list[str]:
        return list(self._name_to_body_id.keys())

    def _resolve_body_id(self, target_object: str) -> int:
        if target_object in self._name_to_body_id:
            return self._name_to_body_id[target_object]

        # Fuzzy match
        candidates = [
            (name, bid) for name, bid in self._name_to_body_id.items()
            if target_object.lower() in name.lower()
        ]
        if candidates:
            candidates.sort(key=lambda x: len(x[0]))
            name, bid = candidates[0]
            logger.info(
                f"[DMControlSeg] Fuzzy match: '{target_object}' → "
                f"'{name}' (body_id={bid})"
            )
            return bid

        raise ValueError(
            f"Object '{target_object}' not found. "
            f"Available: {list(self._name_to_body_id.keys())[:20]}"
        )


# ======================================================================
# Isaac Lab / PhysX (DROID Sim)
# ======================================================================


class IsaacLabSegProvider(GTSegProvider):
    """GT segmentation from Isaac Lab's ``instance_segmentation_fast``.

    Isaac Lab cameras can produce per-pixel instance IDs via the
    ``instance_segmentation_fast`` annotator.  The eval client must
    enable this in the camera config::

        camera_cfg.data_types = ["rgb", "depth", "instance_segmentation_fast"]

    The provider then reads the segmentation output and maps USD prim
    paths to instance IDs for mask extraction.

    Parameters
    ----------
    env :
        An Isaac Lab environment with ``env.scene["camera"]`` or
        equivalent camera sensor.
    camera_name : str
        Name of the camera sensor in the scene.
    """

    def __init__(
        self,
        env,
        camera_name: str = "camera",
    ):
        self._env = env
        self._camera_name = camera_name

        # Build prim_path → instance_id mapping
        # This is populated lazily on first render_mask() call
        # because Isaac Lab's annotators may not be ready at init time.
        self._prim_to_instance: dict[str, int] | None = None
        self._instance_to_name: dict[int, str] = {}

        logger.info(
            f"[IsaacLabSeg] Initialized: camera={camera_name}"
        )

    def _ensure_mapping(self) -> None:
        """Lazily build the prim_path → instance_id mapping."""
        if self._prim_to_instance is not None:
            return

        self._prim_to_instance = {}
        try:
            camera = self._env.scene[self._camera_name]
            # Isaac Lab provides id_to_labels mapping from the annotator
            info = camera.data.info
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            id_to_labels = info.get(
                "instance_segmentation_fast", {}
            ).get("idToLabels", {})

            for instance_id_str, label_info in id_to_labels.items():
                instance_id = int(instance_id_str)
                prim_path = label_info.get(
                    "prim_path", label_info.get("name", "")
                )
                # Extract short name from prim path
                # e.g. "/World/envs/env_0/red_block" → "red_block"
                short_name = prim_path.rsplit("/", 1)[-1] if prim_path else ""
                self._prim_to_instance[prim_path] = instance_id
                if short_name:
                    self._prim_to_instance[short_name] = instance_id
                self._instance_to_name[instance_id] = short_name or prim_path

            logger.info(
                f"[IsaacLabSeg] Mapped {len(self._prim_to_instance)} "
                f"prims to instance IDs"
            )
        except Exception as e:
            logger.warning(
                f"[IsaacLabSeg] Failed to build instance mapping: {e}. "
                f"Make sure 'instance_segmentation_fast' is in camera "
                f"data_types."
            )
            self._prim_to_instance = {}

    def render_mask(self, target_object: str) -> np.ndarray:
        self._ensure_mapping()

        # Resolve target → instance ID
        instance_id = self._resolve_instance_id(target_object)

        # Read segmentation output from camera
        try:
            camera = self._env.scene[self._camera_name]
            seg_data = camera.data.output["instance_segmentation_fast"]
            # Shape: (N, H, W, 1) or (H, W, 1) — squeeze env dim if batched
            seg_arr = seg_data.cpu().numpy() if hasattr(seg_data, "cpu") else np.asarray(seg_data)
            if seg_arr.ndim == 4:
                seg_arr = seg_arr[0]  # take first env
            if seg_arr.ndim == 3:
                seg_arr = seg_arr[:, :, 0]  # squeeze channel
            # seg_arr: (H, W) int — per-pixel instance IDs
        except Exception as e:
            raise RuntimeError(
                f"Failed to read instance_segmentation_fast from camera "
                f"'{self._camera_name}': {e}. Ensure the camera config "
                f"includes 'instance_segmentation_fast' in data_types."
            ) from e

        mask = (seg_arr == instance_id)

        n_pixels = int(mask.sum())
        logger.debug(
            f"[IsaacLabSeg] '{target_object}' → instance_id={instance_id}, "
            f"{n_pixels} pixels"
        )
        return mask

    def available_objects(self) -> list[str]:
        self._ensure_mapping()
        return list(set(self._instance_to_name.values()))

    def _resolve_instance_id(self, target_object: str) -> int:
        if target_object in self._prim_to_instance:
            return self._prim_to_instance[target_object]

        # Fuzzy match on prim paths and short names
        candidates = [
            (name, iid) for name, iid in self._prim_to_instance.items()
            if target_object.lower() in name.lower()
        ]
        if candidates:
            candidates.sort(key=lambda x: len(x[0]))
            name, iid = candidates[0]
            logger.info(
                f"[IsaacLabSeg] Fuzzy match: '{target_object}' → "
                f"'{name}' (instance_id={iid})"
            )
            return iid

        raise ValueError(
            f"Object '{target_object}' not found in instance map. "
            f"Available: {list(self._prim_to_instance.keys())[:20]}"
        )


# ======================================================================
# Factory
# ======================================================================


def create_seg_provider(
    env,
    env_type: str,
    **kwargs,
) -> GTSegProvider:
    """Create the appropriate GT segmentation provider.

    Parameters
    ----------
    env :
        The simulator environment instance.
    env_type : str
        One of ``"libero"``, ``"robocasa"``, ``"vlabench"``, ``"droid_sim"``.
    **kwargs :
        Extra arguments forwarded to the provider constructor
        (e.g. ``camera_name``, ``camera_id``, ``height``, ``width``).
    """
    if env_type in ("libero", "robocasa"):
        return RobosuiteSegProvider(env, **kwargs)
    elif env_type == "vlabench":
        return DMControlSegProvider(env, **kwargs)
    elif env_type == "droid_sim":
        return IsaacLabSegProvider(env, **kwargs)
    else:
        raise ValueError(
            f"Unknown env_type '{env_type}'. "
            f"Supported: libero, robocasa, vlabench, droid_sim"
        )
