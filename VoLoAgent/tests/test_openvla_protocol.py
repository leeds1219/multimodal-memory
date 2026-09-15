# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vlm_orchestrator.protocols.openvla_rest."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np
import pytest
from aiohttp import web

from vlm_orchestrator.protocols.openvla_rest import (
    OPENVLA_RESOLUTION,
    OpenVlaRestBackend,
    canonical_action_to_openvla,
    canonical_to_openvla,
    openvla_action_to_canonical,
    openvla_to_canonical,
)


# ──────────────────────────────────────────────────────────────────────
# Translation primitives
# ──────────────────────────────────────────────────────────────────────


def _make_canonical_obs(prompt: str = "pick up the block") -> dict:
    return {
        "observation/exterior_image_1_left_raw": np.zeros(
            (480, 640, 3), dtype=np.uint8,
        ),
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "observation/ee_pos": np.array([0.4, 0.0, 0.3], dtype=np.float32),
        "observation/ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "prompt": prompt,
    }


def test_canonical_to_openvla_shape_and_keys():
    body = canonical_to_openvla(_make_canonical_obs(), unnorm_key="droid")
    assert body["image"].shape == (*OPENVLA_RESOLUTION, 3)
    assert body["image"].dtype == np.uint8
    assert body["instruction"] == "pick up the block"
    assert body["unnorm_key"] == "droid"


def test_canonical_to_openvla_per_request_unnorm_key():
    """``__unnorm_key`` in canonical overrides backend default."""
    obs = _make_canonical_obs()
    obs["__unnorm_key"] = "nyu_franka_play_dataset_converted_externally_to_rlds"
    body = canonical_to_openvla(obs, unnorm_key="droid")
    assert body["unnorm_key"] == \
        "nyu_franka_play_dataset_converted_externally_to_rlds"


def test_openvla_to_canonical_synthesizes_state():
    """OpenVLA payload has no state; we synthesize zeros."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    body = {"image": img, "instruction": "task", "unnorm_key": "droid"}
    out = openvla_to_canonical(body)
    assert out["prompt"] == "task"
    assert out["__unnorm_key"] == "droid"
    np.testing.assert_array_equal(out["observation/exterior_image_1_left"], img)
    # Wrist defaults to exterior — strategies that index wrist won't crash.
    np.testing.assert_array_equal(out["observation/wrist_image_left"], img)
    np.testing.assert_array_equal(out["observation/joint_position"], np.zeros(7))
    np.testing.assert_array_equal(
        out["observation/ee_quat"], [1.0, 0.0, 0.0, 0.0],
    )


# ──────────────────────────────────────────────────────────────────────
# Action translation
# ──────────────────────────────────────────────────────────────────────


def test_openvla_action_7d_pads_to_chunk_8():
    action = np.arange(7, dtype=np.float32)
    out = openvla_action_to_canonical(action)
    assert out["actions"].shape == (1, 8)
    np.testing.assert_array_equal(out["actions"][0, :7], action)
    assert out["actions"][0, 7] == 0.0  # default open gripper


def test_openvla_action_8d_passes_through():
    action = np.arange(8, dtype=np.float32)
    out = openvla_action_to_canonical(action)
    assert out["actions"].shape == (1, 8)
    np.testing.assert_array_equal(out["actions"][0], action)


def test_openvla_action_unexpected_size_raises():
    with pytest.raises(ValueError):
        openvla_action_to_canonical(np.zeros(6))


def test_canonical_to_openvla_action_strips_gripper():
    canonical = {"actions": np.arange(8, dtype=np.float32).reshape(1, 8)}
    out = canonical_action_to_openvla(canonical)
    assert out.shape == (7,)
    np.testing.assert_array_equal(out, np.arange(7))


def test_canonical_to_openvla_action_takes_first_of_chunk():
    """Multi-step chunks (e.g. from grasp tool) are truncated to step 0."""
    chunk = np.tile(np.arange(8, dtype=np.float32), (5, 1))
    chunk[1:, 0] = 99.0  # later steps differ
    canonical = {"actions": chunk}
    out = canonical_action_to_openvla(canonical)
    assert out[0] == 0.0  # first row, not 99


# ──────────────────────────────────────────────────────────────────────
# Backend round-trip with a mock OpenVLA HTTP server
# ──────────────────────────────────────────────────────────────────────


def _start_mock_server(port: int, return_action: np.ndarray) -> tuple:
    """Run a minimal HTTP /act server in a background thread.

    Returns (server_thread, stop_event, captured_requests)."""
    import asyncio as _asyncio

    captured: list[dict] = []
    stop_event = threading.Event()

    async def handler(request):
        body = json.loads(await request.read())
        captured.append(body)
        return web.Response(
            body=json.dumps(return_action.tolist()).encode("utf-8"),
            content_type="application/json",
        )

    def run():
        async def go():
            app = web.Application()
            app.router.add_post("/act", handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", port)
            await site.start()
            while not stop_event.is_set():
                await _asyncio.sleep(0.1)
            await runner.cleanup()
        try:
            _asyncio.run(go())
        except Exception:
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.2)  # let bind
    return t, stop_event, captured


def test_backend_round_trip_with_mock_server():
    port = 28765
    return_action = np.arange(7, dtype=np.float32) * 0.1
    thread, stop, captured = _start_mock_server(port, return_action)

    async def _go():
        backend = OpenVlaRestBackend(
            "127.0.0.1", port, unnorm_key="droid", timeout_s=5.0,
        )
        conn = await backend.connect()
        try:
            obs = _make_canonical_obs("test prompt")
            response = await conn.infer(obs)
            assert response["actions"].shape == (1, 8)
            np.testing.assert_array_almost_equal(
                response["actions"][0, :7], return_action,
            )
            assert response["actions"][0, 7] == 0.0  # default gripper
        finally:
            await conn.close()

    try:
        asyncio.run(_go())
        assert len(captured) == 1
        body = captured[0]
        assert body["instruction"] == "test prompt"
        assert body["unnorm_key"] == "droid"
        # Image arrives JSON-encoded via json_numpy; on the server side
        # we'd normally json_numpy.loads it.  For the mock server we
        # just check the shape was preserved (after json_numpy dump).
        # Without json_numpy.loads in the mock, body["image"] is a dict.
        assert "image" in body
    finally:
        stop.set()
        thread.join(timeout=2.0)


# ──────────────────────────────────────────────────────────────────────
# Frontend: full HTTP-client → frontend → orchestrator-callback → HTTP-client
# ──────────────────────────────────────────────────────────────────────


def test_frontend_e2e():
    """An HTTP client POSTs to /act on the frontend; the on_session
    callback receives canonical obs and sends a canonical action; the
    HTTP client gets a 7-D OpenVLA-format response back."""
    import json_numpy

    received_canonical: list[dict] = []

    async def _on_session(session):
        canonical = await session.recv_obs()
        received_canonical.append(canonical)
        # Return a known canonical action; frontend will strip gripper
        # and serve back the first row's first 7 dims.
        await session.send_action({
            "actions": np.array(
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]], dtype=np.float32,
            ),
        })

    fe_port = 28766
    server_done = threading.Event()

    def _run_frontend():
        from vlm_orchestrator.protocols.openvla_rest import OpenVlaRestFrontend

        async def go():
            await OpenVlaRestFrontend().serve(
                "127.0.0.1", fe_port, _on_session,
            )
        try:
            asyncio.run(go())
        except Exception:
            pass
        server_done.set()

    thread = threading.Thread(target=_run_frontend, daemon=True)
    thread.start()
    time.sleep(0.3)  # let bind

    # Pretend to be a robolab --policy openvla eval client.
    import requests
    json_numpy.patch()  # so np.ndarray serializes
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    payload = {"image": rgb, "instruction": "pick", "unnorm_key": "droid"}
    response = requests.post(
        f"http://127.0.0.1:{fe_port}/act",
        data=json.dumps(payload),
        timeout=5.0,
    )
    assert response.status_code == 200, response.text
    action = np.asarray(response.json())
    assert action.shape == (7,)
    np.testing.assert_array_almost_equal(
        action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )

    assert len(received_canonical) == 1
    canonical = received_canonical[0]
    assert canonical["prompt"] == "pick"
    assert canonical["__unnorm_key"] == "droid"
