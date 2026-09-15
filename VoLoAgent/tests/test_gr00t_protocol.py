# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vlm_orchestrator.protocols.gr00t_zmq.

Validates translation in both directions, plus frontend/backend
round-trips against thread-local mock ZMQ servers.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import zmq

from vlm_orchestrator.protocols.gr00t_zmq import (
    GR00T_RESOLUTION,
    Gr00tZmqBackend,
    Gr00tZmqBackendConnection,
    _gr00t_pack,
    _gr00t_unpack,
    canonical_action_to_gr00t,
    canonical_to_gr00t,
    gr00t_action_to_canonical,
    gr00t_to_canonical,
    quat_to_euler_xyz,
)


# ──────────────────────────────────────────────────────────────────────
# Translation primitives
# ──────────────────────────────────────────────────────────────────────


def test_quat_to_euler_identity():
    e = quat_to_euler_xyz(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(e, [0.0, 0.0, 0.0])


def test_quat_to_euler_90_yaw():
    s = np.sqrt(0.5)
    e = quat_to_euler_xyz(np.array([s, 0.0, 0.0, s]))
    np.testing.assert_array_almost_equal(e, [0.0, 0.0, np.pi / 2])


# ──────────────────────────────────────────────────────────────────────
# canonical → gr00t (used by Backend)
# ──────────────────────────────────────────────────────────────────────


def _make_canonical(prompt: str = "pick up the block", with_raw: bool = True) -> dict:
    obs = {
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "observation/ee_pos": np.array([0.4, 0.0, 0.3], dtype=np.float32),
        "observation/ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "prompt": prompt,
    }
    if with_raw:
        obs["observation/exterior_image_1_left_raw"] = (
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        obs["observation/wrist_image_left_raw"] = (
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
    else:
        obs["observation/exterior_image_1_left"] = (
            np.zeros((224, 224, 3), dtype=np.uint8)
        )
        obs["observation/wrist_image_left"] = (
            np.zeros((224, 224, 3), dtype=np.uint8)
        )
    return obs


def test_canonical_to_gr00t_shapes():
    out = canonical_to_gr00t(_make_canonical())
    assert out["video.exterior_image_1_left"].shape == (1, 1, *GR00T_RESOLUTION, 3)
    assert out["video.wrist_image_left"].shape == (1, 1, *GR00T_RESOLUTION, 3)
    assert out["state.joint_position"].shape == (1, 1, 7)
    assert out["state.gripper_position"].shape == (1, 1, 1)
    assert out["state.eef_position"].shape == (1, 1, 3)
    assert out["state.eef_rotation"].shape == (1, 1, 3)
    assert out["annotation.language.language_instruction"] == ["pick up the block"]
    assert out["annotation.language.language_instruction_2"] == ["pick up the block"]
    assert out["annotation.language.language_instruction_3"] == ["pick up the block"]


def test_canonical_to_gr00t_dtypes():
    out = canonical_to_gr00t(_make_canonical())
    assert out["state.joint_position"].dtype == np.float32
    assert out["state.gripper_position"].dtype == np.float32


def test_canonical_to_gr00t_falls_back_to_policy_image():
    out = canonical_to_gr00t(_make_canonical(with_raw=False))
    assert out["video.exterior_image_1_left"].shape == (1, 1, *GR00T_RESOLUTION, 3)


# ──────────────────────────────────────────────────────────────────────
# gr00t → canonical (used by Frontend)
# ──────────────────────────────────────────────────────────────────────


def _make_gr00t_obs(prompt: str = "pick up the block") -> dict:
    return {
        "video.exterior_image_1_left": np.zeros(
            (1, 1, *GR00T_RESOLUTION, 3), dtype=np.uint8,
        ),
        "video.wrist_image_left": np.zeros(
            (1, 1, *GR00T_RESOLUTION, 3), dtype=np.uint8,
        ),
        "state.joint_position": np.arange(7, dtype=np.float32)[None, None, :],
        "state.gripper_position": np.array([0.5], dtype=np.float32)[None, None, :],
        "state.eef_position": np.array(
            [0.4, 0.0, 0.3], dtype=np.float32,
        )[None, None, :],
        "state.eef_rotation": np.zeros((1, 1, 3), dtype=np.float32),
        "annotation.language.language_instruction": [prompt],
        "annotation.language.language_instruction_2": [prompt],
        "annotation.language.language_instruction_3": [prompt],
    }


def test_gr00t_to_canonical_shapes():
    out = gr00t_to_canonical(_make_gr00t_obs())
    assert out["observation/exterior_image_1_left"].shape == (*GR00T_RESOLUTION, 3)
    assert out["observation/wrist_image_left"].shape == (*GR00T_RESOLUTION, 3)
    assert out["observation/joint_position"].shape == (7,)
    assert out["observation/gripper_position"].shape == (1,)
    assert out["observation/ee_pos"].shape == (3,)
    assert out["observation/ee_quat"].shape == (4,)
    assert out["prompt"] == "pick up the block"


def test_gr00t_to_canonical_provides_both_image_keys():
    """Strategies prefer ``_raw``; provide both for safety."""
    out = gr00t_to_canonical(_make_gr00t_obs())
    assert "observation/exterior_image_1_left" in out
    assert "observation/exterior_image_1_left_raw" in out
    np.testing.assert_array_equal(
        out["observation/exterior_image_1_left"],
        out["observation/exterior_image_1_left_raw"],
    )


def test_round_trip_gr00t_canonical_gr00t_preserves_state():
    """gr00t → canonical → gr00t round-trip preserves numeric state."""
    original = _make_gr00t_obs()
    canonical = gr00t_to_canonical(original)
    # Add an image suffix the canonical_to_gr00t needs
    out = canonical_to_gr00t(canonical)
    np.testing.assert_array_almost_equal(
        out["state.joint_position"], original["state.joint_position"],
    )
    np.testing.assert_array_almost_equal(
        out["state.gripper_position"], original["state.gripper_position"],
    )
    np.testing.assert_array_almost_equal(
        out["state.eef_position"], original["state.eef_position"],
    )


# ──────────────────────────────────────────────────────────────────────
# Action translation
# ──────────────────────────────────────────────────────────────────────


def test_gr00t_action_to_canonical_concat_shape():
    action_dict = {
        "action.joint_position": np.ones((1, 10, 7), dtype=np.float32),
        "action.gripper_position": np.zeros((1, 10, 1), dtype=np.float32),
    }
    out = gr00t_action_to_canonical(action_dict)
    assert out["actions"].shape == (10, 8)


def test_canonical_action_to_gr00t_split_shape():
    canonical = {"actions": np.zeros((10, 8), dtype=np.float32)}
    out = canonical_action_to_gr00t(canonical)
    assert out["action.joint_position"].shape == (1, 10, 7)
    assert out["action.gripper_position"].shape == (1, 10, 1)


def test_canonical_action_to_gr00t_pads_short_chunks():
    """Grasp-tool chunks (8 actions) get padded to gr00t client's
    minimum (10) by repeating the last action — eval client never
    reads past the end of a chunk."""
    actions = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    out = canonical_action_to_gr00t({"actions": actions})
    assert out["action.joint_position"].shape == (1, 10, 7)
    assert out["action.gripper_position"].shape == (1, 10, 1)
    # First 8 rows preserved
    np.testing.assert_array_equal(
        out["action.joint_position"][0, :8], actions[:, :7],
    )
    # Last 2 rows are repeats of action[7]
    np.testing.assert_array_equal(
        out["action.joint_position"][0, 8], actions[7, :7],
    )
    np.testing.assert_array_equal(
        out["action.joint_position"][0, 9], actions[7, :7],
    )


def test_canonical_action_to_gr00t_passes_long_chunks_through():
    """gr00t native 32-step chunks aren't padded (they already exceed 10)."""
    actions = np.zeros((32, 8), dtype=np.float32)
    out = canonical_action_to_gr00t({"actions": actions})
    assert out["action.joint_position"].shape == (1, 32, 7)
    assert out["action.gripper_position"].shape == (1, 32, 1)


def test_action_round_trip():
    """canonical → gr00t → canonical preserves values."""
    canonical = {"actions": np.random.RandomState(42).randn(10, 8).astype(np.float32)}
    grok = canonical_action_to_gr00t(canonical)
    back = gr00t_action_to_canonical(grok)
    np.testing.assert_array_almost_equal(back["actions"], canonical["actions"])


# ──────────────────────────────────────────────────────────────────────
# Backend round-trip with mock GR00T ZMQ server
# ──────────────────────────────────────────────────────────────────────


class _MockGR00TServer:
    """Thread-local ZMQ REP server returning a fixed action chunk."""

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self._stop = threading.Event()
        self.received_obs: dict | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.socket.close(linger=0)
        finally:
            self.context.term()

    def _run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))
            if self.socket not in socks:
                continue
            request = _gr00t_unpack(self.socket.recv())
            if request.get("endpoint") == "ping":
                self.socket.send(_gr00t_pack({"ok": True}))
                continue
            self.received_obs = request["data"]["observation"]
            response = (
                {
                    "action.joint_position": np.zeros(
                        (1, 10, 7), dtype=np.float32,
                    ),
                    "action.gripper_position": np.zeros(
                        (1, 10, 1), dtype=np.float32,
                    ),
                },
                {"info": "ok"},
            )
            self.socket.send(_gr00t_pack(response))


@pytest.fixture
def mock_gr00t_server():
    server = _MockGR00TServer()
    server.start()
    time.sleep(0.05)
    yield server
    server.stop()


def test_backend_round_trip(mock_gr00t_server):
    import asyncio

    async def _go():
        backend = Gr00tZmqBackend("127.0.0.1", mock_gr00t_server.port)
        conn = await backend.connect()
        try:
            obs = _make_canonical()
            obs["__step"] = 5
            obs["__episode_id"] = "task_0"
            response = await conn.infer(obs)
            assert response["actions"].shape == (10, 8)
            recv = mock_gr00t_server.received_obs
            assert "video.exterior_image_1_left" in recv
            assert "annotation.language.language_instruction" in recv
            assert "observation/joint_position" not in recv
            assert "__step" not in recv
            assert "__episode_id" not in recv
        finally:
            await conn.close()

    asyncio.run(_go())


# ──────────────────────────────────────────────────────────────────────
# Frontend round-trip: simulate a --policy gr00t eval client connecting
# ──────────────────────────────────────────────────────────────────────


def test_frontend_translates_obs_and_action(mock_gr00t_server):
    """Spin up Gr00tZmqFrontend, post a gr00t-format request, verify
    the frontend hands canonical obs to the on_session callback and
    sends back a properly-formatted gr00t action."""
    import asyncio

    from vlm_orchestrator.protocols.gr00t_zmq import Gr00tZmqFrontend

    received_canonical: list[dict] = []

    async def _on_session(session):
        # First call: receive the obs (already canonical)
        canonical = await session.recv_obs()
        received_canonical.append(canonical)
        # Send back a canonical action
        await session.send_action({
            "actions": np.full((10, 8), 0.42, dtype=np.float32),
        })
        # Don't loop — single round-trip for test

    # Run frontend in a thread, post one request from a ZMQ REQ client.
    frontend = Gr00tZmqFrontend()
    # Reuse a free random port via a temp socket bind.
    tmp_ctx = zmq.Context()
    tmp_sock = tmp_ctx.socket(zmq.REP)
    fe_port = tmp_sock.bind_to_random_port("tcp://127.0.0.1")
    tmp_sock.close()
    tmp_ctx.term()

    server_done = threading.Event()

    def _run_frontend():
        async def _go():
            await frontend.serve("127.0.0.1", fe_port, _on_session)
        try:
            asyncio.run(_go())
        except Exception:
            pass
        server_done.set()

    thread = threading.Thread(target=_run_frontend, daemon=True)
    thread.start()
    time.sleep(0.1)  # let the server bind

    # Pretend to be a gr00t eval client.
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{fe_port}")
    request = {
        "endpoint": "get_action",
        "data": {
            "observation": _make_gr00t_obs("test prompt"),
            "options": None,
        },
    }
    sock.send(_gr00t_pack(request))
    response_raw = sock.recv()
    response = _gr00t_unpack(response_raw)

    sock.close(linger=0)
    ctx.term()

    # Validate translation went both ways.
    assert len(received_canonical) == 1
    canonical = received_canonical[0]
    assert canonical["prompt"] == "test prompt"
    assert canonical["observation/joint_position"].shape == (7,)

    # Response is gr00t (action_dict, info) tuple
    assert isinstance(response, (list, tuple))
    action_dict = response[0]
    assert action_dict["action.joint_position"].shape == (1, 10, 7)
    assert (action_dict["action.joint_position"] == 0.42).all()
    assert action_dict["action.gripper_position"].shape == (1, 10, 1)


def test_frontend_extracts_extras_from_request():
    """A ``--policy gr00t`` eval client that forwards depth /
    camera_pose / gt_state via the top-level ``__extras`` field has
    those fields show up in the orchestrator's canonical obs."""
    import asyncio

    from vlm_orchestrator.protocols.gr00t_zmq import Gr00tZmqFrontend

    received_canonical: list[dict] = []

    async def _on_session(session):
        canonical = await session.recv_obs()
        received_canonical.append(canonical)
        await session.send_action({
            "actions": np.zeros((10, 8), dtype=np.float32),
        })

    frontend = Gr00tZmqFrontend()
    tmp_ctx = zmq.Context()
    tmp_sock = tmp_ctx.socket(zmq.REP)
    fe_port = tmp_sock.bind_to_random_port("tcp://127.0.0.1")
    tmp_sock.close()
    tmp_ctx.term()

    def _run_frontend():
        async def _go():
            await frontend.serve("127.0.0.1", fe_port, _on_session)
        try:
            asyncio.run(_go())
        except Exception:
            pass

    thread = threading.Thread(target=_run_frontend, daemon=True)
    thread.start()
    time.sleep(0.1)

    full_res_image = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_image = np.ones((480, 640), dtype=np.float32) * 0.7
    camera_K = np.eye(3, dtype=np.float32)
    gt_state = {"objects": {"banana": {"position": [0.4, 0.0, 0.3]}}}

    request = {
        "endpoint": "get_action",
        "data": {
            "observation": _make_gr00t_obs("test prompt"),
            "options": None,
        },
        "__extras": {
            "observation/exterior_image_1_left_raw": full_res_image,
            "observation/wrist_image_left_raw": full_res_image,
            "observation/depth_external": depth_image,
            "observation/camera_K": camera_K,
            "observation/camera_pos": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "observation/camera_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "gt_state": gt_state,
        },
    }
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{fe_port}")
    sock.send(_gr00t_pack(request))
    sock.recv()  # action response, ignored
    sock.close(linger=0)
    ctx.term()

    assert len(received_canonical) == 1
    canonical = received_canonical[0]
    # Extras override the frontend's default ``_raw`` (which was the
    # 180×320 model copy because gr00t's request lacks full-res).
    assert canonical["observation/exterior_image_1_left_raw"].shape == (480, 640, 3)
    assert canonical["observation/wrist_image_left_raw"].shape == (480, 640, 3)
    # Depth / camera / gt_state surface in canonical for grasp tool /
    # failure handlers.
    np.testing.assert_array_equal(
        canonical["observation/depth_external"], depth_image,
    )
    assert canonical["observation/camera_K"].shape == (3, 3)
    assert canonical["gt_state"] == gt_state
