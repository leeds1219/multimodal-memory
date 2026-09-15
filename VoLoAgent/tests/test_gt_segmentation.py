# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ground-truth segmentation providers and grasp tool integration."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_orchestrator.perception.gt_segmentation import (
    DMControlSegProvider,
    GTSegProvider,
    IsaacLabSegProvider,
    RobosuiteSegProvider,
    create_seg_provider,
)
from vlm_orchestrator.grasp.tool import GraspSegMode, GraspToolExecutor


# ======================================================================
# Mock simulators
# ======================================================================


class MockMujocoModel:
    """Minimal MuJoCo model mock for robosuite / dm_control."""

    def __init__(self, bodies: dict[str, int], geoms: dict[int, int] | None = None):
        """
        bodies: {name: body_id}
        geoms: {geom_id: body_id}
        """
        self._bodies = bodies
        self._id_to_name = {v: k for k, v in bodies.items()}
        self.nbody = max(bodies.values()) + 1 if bodies else 0
        # geom → body mapping
        self._geoms = geoms or {}
        self.ngeom = max(self._geoms.keys()) + 1 if self._geoms else 0
        self.geom_bodyid = np.zeros(max(self.ngeom, 1), dtype=int)
        for gid, bid in self._geoms.items():
            self.geom_bodyid[gid] = bid

    def body_id2name(self, body_id):
        return self._id_to_name.get(body_id, "")

    def body_name2id(self, name):
        if name in self._bodies:
            return self._bodies[name]
        raise KeyError(f"body '{name}' not found")

    def id2name(self, idx, entity_type):
        if entity_type == "body":
            return self._id_to_name.get(idx, "")
        return ""

    def get_joint_qpos_addr(self, name):
        return 0

    def joint_id2name(self, idx):
        return ""

    def site_name2id(self, name):
        return 0

    def site_id2name(self, idx):
        return ""


class MockSim:
    """Minimal robosuite sim mock."""

    def __init__(self, bodies: dict[str, int], seg_image: np.ndarray):
        self.model = MockMujocoModel(bodies)
        self._seg_image = seg_image
        self.data = type("Data", (), {
            "body_xpos": np.zeros((self.model.nbody, 3)),
            "body_xquat": np.zeros((self.model.nbody, 4)),
            "qpos": np.zeros(20),
            "site_xpos": np.zeros((1, 3)),
        })()

    def render(self, camera_name, height, width, segmentation=False):
        if segmentation:
            return self._seg_image
        return np.zeros((height, width, 3), dtype=np.uint8)


class MockRobosuiteEnv:
    """Minimal robosuite env mock."""

    def __init__(self, bodies: dict[str, int], seg_image: np.ndarray):
        self.sim = MockSim(bodies, seg_image)
        self.obj_body_id = bodies


class MockPhysics:
    """Minimal dm_control physics mock."""

    def __init__(
        self,
        bodies: dict[str, int],
        geoms: dict[int, int],
        seg_image: np.ndarray,
    ):
        self.model = MockMujocoModel(bodies, geoms)
        self._seg_image = seg_image
        self.data = type("Data", (), {
            "xpos": np.zeros((self.model.nbody, 3)),
            "xquat": np.zeros((self.model.nbody, 4)),
        })()

    def render(self, height, width, camera_id=0, segmentation=False):
        if segmentation:
            return self._seg_image
        return np.zeros((height, width, 3), dtype=np.uint8)


class MockDMControlEnv:
    """Minimal VLABench env mock."""

    def __init__(self, physics: MockPhysics):
        self.physics = physics


# ======================================================================
# Tests: RobosuiteSegProvider (LIBERO, RoboCasa)
# ======================================================================


class TestRobosuiteSegProvider:
    def _make_env(self, target_body_id: int = 5):
        """Create a mock env with a known segmentation image.

        The real robosuite renderer returns geom IDs in channel 1.
        ``render_mask`` converts geom IDs → body IDs via
        ``model.geom_bodyid``.  We use geom_ids = 10 (→ body 5)
        and 12 (→ body 7) to match the body IDs.
        """
        h, w = 256, 256
        bodies = {
            "red_block": 5,
            "blue_cup": 7,
            "table": 2,
        }
        # geom_id → body_id mapping
        geoms = {
            10: target_body_id,  # red_block geom
            12: 7,               # blue_cup geom
            3: 2,                # table geom
        }

        # Segmentation image: (H, W, 2) — channel 1 = geom_id
        # Place target object in a 50x50 pixel region
        seg = np.zeros((h, w, 2), dtype=np.int32)
        seg[:, :, 1] = -1  # background
        seg[100:150, 80:130, 1] = 10  # red_block geom_id
        seg[50:70, 200:220, 1] = 12   # blue_cup geom_id
        # robosuite renders upside down, so flip
        seg = seg[::-1]

        env = MockRobosuiteEnv(bodies, seg)
        # Set up geom→body mapping on the model
        env.sim.model = MockMujocoModel(bodies, geoms)
        return env

    def test_exact_match(self):
        env = self._make_env(target_body_id=5)
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)

        mask = seg.render_mask("red_block")
        assert mask.shape == (256, 256)
        assert mask.dtype == bool
        assert mask.sum() == 50 * 50  # 50x50 region

    def test_fuzzy_match(self):
        env = self._make_env(target_body_id=5)
        # Add a body with _main suffix
        env.sim.model._bodies["red_block_main"] = 5
        env.sim.model._id_to_name[5] = "red_block_main"
        env.obj_body_id["red_block_main"] = 5

        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)
        mask = seg.render_mask("red_block")
        assert mask.sum() == 50 * 50

    def test_object_not_found(self):
        env = self._make_env()
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)

        with pytest.raises(ValueError, match="not found"):
            seg.render_mask("nonexistent_object")

    def test_available_objects(self):
        env = self._make_env()
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)
        objects = seg.available_objects()
        assert "red_block" in objects
        assert "blue_cup" in objects

    def test_render_mask_for_obs(self):
        env = self._make_env()
        seg = RobosuiteSegProvider(env, camera_name="agentview", height=256, width=256)

        wire_obs = {"observation/image": np.zeros((256, 256, 3))}
        seg.render_mask_for_obs("red_block", wire_obs)

        assert "gt_seg/mask" in wire_obs
        assert "gt_seg/target" in wire_obs
        assert wire_obs["gt_seg/target"] == "red_block"
        assert wire_obs["gt_seg/mask"].sum() == 50 * 50


# ======================================================================
# Tests: DMControlSegProvider (VLABench)
# ======================================================================


class TestDMControlSegProvider:
    def _make_env(self):
        h, w = 256, 256
        bodies = {"target_ball": 3, "container": 4, "world": 0}
        # geom 10, 11 belong to body 3 (target_ball)
        # geom 20 belongs to body 4 (container)
        geoms = {10: 3, 11: 3, 20: 4}

        # Segmentation: channel 0 = geom_id
        seg = np.full((h, w, 2), -1, dtype=np.int32)
        seg[60:100, 60:100, 0] = 10   # geom 10 → body 3
        seg[100:120, 60:80, 0] = 11   # geom 11 → body 3
        seg[150:200, 150:200, 0] = 20  # geom 20 → body 4

        physics = MockPhysics(bodies, geoms, seg)
        return MockDMControlEnv(physics)

    def test_multi_geom_body(self):
        env = self._make_env()
        seg = DMControlSegProvider(env, camera_id=2, height=256, width=256)

        mask = seg.render_mask("target_ball")
        assert mask.shape == (256, 256)
        # Two geom regions: 40*40 + 20*20 = 1600 + 400 = 2000
        assert mask.sum() == 2000

    def test_container(self):
        env = self._make_env()
        seg = DMControlSegProvider(env, camera_id=2, height=256, width=256)

        mask = seg.render_mask("container")
        assert mask.sum() == 50 * 50  # 50x50 region

    def test_fuzzy_match(self):
        env = self._make_env()
        seg = DMControlSegProvider(env, camera_id=2, height=256, width=256)
        # "ball" should fuzzy-match "target_ball"
        mask = seg.render_mask("ball")
        assert mask.sum() == 2000


# ======================================================================
# Tests: Factory
# ======================================================================


class TestFactory:
    def test_create_libero(self):
        env = MockRobosuiteEnv({"obj": 1}, np.zeros((256, 256, 2), dtype=np.int32))
        provider = create_seg_provider(env, "libero", camera_name="agentview")
        assert isinstance(provider, RobosuiteSegProvider)

    def test_create_robocasa(self):
        env = MockRobosuiteEnv({"obj": 1}, np.zeros((256, 256, 2), dtype=np.int32))
        provider = create_seg_provider(env, "robocasa", camera_name="agentview")
        assert isinstance(provider, RobosuiteSegProvider)

    def test_create_vlabench(self):
        physics = MockPhysics({"obj": 1}, {}, np.zeros((256, 256, 2), dtype=np.int32))
        env = MockDMControlEnv(physics)
        provider = create_seg_provider(env, "vlabench", camera_id=2)
        assert isinstance(provider, DMControlSegProvider)

    def test_unknown_env(self):
        with pytest.raises(ValueError, match="Unknown env_type"):
            create_seg_provider(None, "unknown_sim")


# ======================================================================
# Tests: GraspTool GT_SIM integration
# ======================================================================


class TestGraspToolGTSIM:
    def test_seg_mode_enum(self):
        assert GraspSegMode.GT_SIM == "gt_sim"
        assert GraspSegMode("gt_sim") == GraspSegMode.GT_SIM

    def test_get_gt_mask_from_obs(self):
        """Test _get_gt_mask when eval client provides body_ids buffer.

        Centroid matching projects gt_state position → pixel, then
        picks the seg-buffer instance whose centroid is nearest.
        The identity camera (extrinsic=I) projects (0.3, 0.0, 0.1)
        to pixel (1500, 500) which clips to image bounds, but the
        centroid of body_id=5 at (105, 75) should still be matched
        as the closest candidate.
        """
        executor = GraspToolExecutor(
            seg_mode=GraspSegMode.GT_SIM,
            env_mode="libero",
        )

        # Simulate body_ids seg buffer with known object positions
        body_ids = np.full((256, 256), -1, dtype=np.int32)
        body_ids[50:100, 80:130] = 5  # red_block body_id

        obs = {
            "gt_seg/body_ids": body_ids,
            "gt_seg/obj_body_id": {"red_block": 5},
            "gt_state": {
                "objects": {
                    "red_block": {"pos": [0.3, 0.0, 0.1]},
                },
                "subtask": {"score": 0.5},
            },
            # Camera info for centroid projection
            "observation/camera_extrinsic": np.eye(4).tolist(),
            "observation/camera_K": np.array([
                [500, 0, 128],
                [0, 500, 128],
                [0, 0, 1],
            ]).tolist(),
        }

        result_mask, bbox, score = executor._get_gt_mask(
            obs, "red_block", (256, 256)
        )

        assert result_mask.shape == (256, 256)
        assert result_mask.sum() > 0  # pixels matched
        assert score == 1.0

    def test_get_gt_mask_no_data_raises(self):
        """Test that _get_gt_mask raises when no seg buffer or gt_state."""
        executor = GraspToolExecutor(
            seg_mode=GraspSegMode.GT_SIM,
            env_mode="libero",
        )
        obs = {}  # no gt_seg, no gt_state

        with pytest.raises(RuntimeError, match="could not locate"):
            executor._get_gt_mask(obs, "red_block", (256, 256))

    def test_mask_to_bbox(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True
        bbox = GraspToolExecutor._mask_to_bbox(mask)
        assert bbox == (30, 20, 60, 40)

    def test_mask_to_bbox_empty(self):
        mask = np.zeros((100, 100), dtype=bool)
        bbox = GraspToolExecutor._mask_to_bbox(mask)
        assert bbox == (0, 0, 0, 0)

    def test_get_gt_mask_from_body_ids_buffer(self):
        """Test _get_gt_mask when eval client sends raw body-ID seg image.

        Centroid matching requires gt_state + camera info to project
        3D position → pixel, then matches nearest seg-buffer instance.
        """
        executor = GraspToolExecutor(
            seg_mode=GraspSegMode.GT_SIM,
            env_mode="libero",
        )

        # Simulate raw segmentation buffer from eval client
        body_ids = np.full((256, 256), -1, dtype=np.int32)
        body_ids[40:80, 60:120] = 5   # red_block body
        body_ids[150:180, 100:140] = 7  # blue_cup body

        obs = {
            "gt_seg/body_ids": body_ids,
            "gt_seg/obj_body_id": {
                "red_block": 5,
                "blue_cup": 7,
                "table": 2,
            },
            "gt_state": {
                "objects": {
                    "red_block": {"pos": [0.3, 0.0, 0.1]},
                    "blue_cup": {"pos": [0.1, 0.1, 0.1]},
                },
                "subtask": {"score": 0.5},
            },
            "observation/camera_extrinsic": np.eye(4).tolist(),
            "observation/camera_K": np.array([
                [500, 0, 128],
                [0, 500, 128],
                [0, 0, 1],
            ]).tolist(),
        }

        mask, bbox, score = executor._get_gt_mask(obs, "red_block", (256, 256))
        assert mask.shape == (256, 256)
        assert mask.sum() > 0  # pixels matched
        assert score == 1.0

    def test_get_gt_mask_body_ids_fuzzy_match(self):
        """Fuzzy name matching with body-ID buffer."""
        executor = GraspToolExecutor(
            seg_mode=GraspSegMode.GT_SIM,
            env_mode="libero",
        )

        body_ids = np.full((256, 256), -1, dtype=np.int32)
        body_ids[50:100, 50:100] = 5

        obs = {
            "gt_seg/body_ids": body_ids,
            "gt_seg/obj_body_id": {"alphabet_soup_1_main": 5},
            "gt_state": {
                "objects": {
                    "alphabet_soup_1_main": {"pos": [0.1, 0.0, 0.1]},
                },
                "subtask": {"score": 0.5},
            },
            "observation/camera_extrinsic": np.eye(4).tolist(),
            "observation/camera_K": np.array([
                [500, 0, 128],
                [0, 500, 128],
                [0, 0, 1],
            ]).tolist(),
        }

        # "alphabet_soup" should fuzzy-match "alphabet_soup_1_main"
        mask, bbox, score = executor._get_gt_mask(
            obs, "alphabet_soup", (256, 256)
        )
        assert mask.sum() > 0

    def test_resolve_gt_body_id(self):
        mapping = {"red_block": 5, "red_block_main": 5, "blue_cup": 7}
        assert GraspToolExecutor._resolve_gt_body_id("red_block", mapping) == 5
        assert GraspToolExecutor._resolve_gt_body_id("blue_cup", mapping) == 7
        # Fuzzy
        assert GraspToolExecutor._resolve_gt_body_id("blue", mapping) == 7
        # Not found
        assert GraspToolExecutor._resolve_gt_body_id("nonexistent", mapping) is None
