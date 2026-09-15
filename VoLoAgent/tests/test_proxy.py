# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: spin up a mock VLA server, the orchestrator proxy, and a client."""

import threading
import time

import numpy as np
import websockets.sync.client as ws_client
import websockets.sync.server as ws_server_sync

from vlm_orchestrator.utils import codec
from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
from vlm_orchestrator.vlm import VLMBackend


class MockVLM(VLMBackend):
    """Test VLM that uppercases the instruction."""

    def rewrite_instruction(
        self, instruction: str, image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        return instruction.upper()


# --- Mock VLA Server ---

def mock_vla_handler(ws):
    packer = codec.Packer()
    # Send metadata
    ws.send(packer.pack({"model": "mock-vla", "version": "test"}))
    # Echo loop
    while True:
        try:
            raw = ws.recv()
            obs = codec.unpackb(raw)
            response = {
                "actions": np.zeros((8, 8), dtype=np.float32),
                "received_prompt": obs.get("prompt", ""),
                "received_keys": sorted(obs.keys()),
            }
            ws.send(packer.pack(response))
        except Exception:
            break


def start_mock_vla(port: int) -> ws_server_sync.WebSocketServer:
    server = ws_server_sync.serve(
        mock_vla_handler, "127.0.0.1", port, compression=None, max_size=None
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_passthrough():
    """Passthrough mode should not modify the instruction."""
    vla_port = 19876
    proxy_port = 19877

    vla_server = start_mock_vla(vla_port)

    from vlm_orchestrator.vlm import PassthroughVLM

    config = ProxyConfig(
        vla_host="127.0.0.1",
        vla_port=vla_port,
        host="127.0.0.1",
        port=proxy_port,
        vlm=PassthroughVLM(),
    )
    proxy = OrchestratorProxy(config)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    time.sleep(0.5)

    packer = codec.Packer()
    client = ws_client.connect(
        f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
    )

    # Receive metadata
    metadata = codec.unpackb(client.recv())
    assert metadata["model"] == "mock-vla"
    assert metadata["orchestrator"] is True

    # Send observation
    obs = {
        "prompt": "pick up the banana",
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
    }
    client.send(packer.pack(obs))
    response = codec.unpackb(client.recv())

    # In passthrough, prompt should be unchanged
    assert response["received_prompt"] == "pick up the banana"

    client.close()
    vla_server.shutdown()


def test_rewrite():
    """Rewrite mode should modify the instruction via VLM."""
    vla_port = 19878
    proxy_port = 19879

    vla_server = start_mock_vla(vla_port)

    config = ProxyConfig(
        vla_host="127.0.0.1",
        vla_port=vla_port,
        host="127.0.0.1",
        port=proxy_port,
        vlm=MockVLM(),
    )
    proxy = OrchestratorProxy(config)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    time.sleep(0.5)

    packer = codec.Packer()
    client = ws_client.connect(
        f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
    )

    # Receive metadata
    metadata = codec.unpackb(client.recv())

    # First infer - should trigger rewrite
    obs = {
        "prompt": "pick up the banana",
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
    }
    client.send(packer.pack(obs))
    response = codec.unpackb(client.recv())
    assert response["received_prompt"] == "PICK UP THE BANANA"
    assert response["orchestrator_instruction"] == "PICK UP THE BANANA"

    # Second infer with same prompt - should use cached rewrite, not call VLM again
    client.send(packer.pack(obs))
    response = codec.unpackb(client.recv())
    assert response["received_prompt"] == "PICK UP THE BANANA"

    # New episode (different prompt) - should trigger new rewrite
    obs2 = dict(obs)
    obs2["prompt"] = "stack the bowls"
    client.send(packer.pack(obs2))
    response = codec.unpackb(client.recv())
    assert response["received_prompt"] == "STACK THE BOWLS"

    client.close()
    vla_server.shutdown()


def test_cosmos3_passthrough_strips_orchestrator_only_keys():
    """Cosmos3 protocol: the stitched observation/image + proprio + prompt
    reach the policy server, but orchestrator-only keys (depth, camera pose,
    gt_state, *_raw) are stripped before forwarding.

    Cosmos3's policy server is openpi-ws, so the proxy does NO obs/action
    translation (unlike dreamzero). It must still strip the proxy-only keys
    the robolab Cosmos3Client forwards via _orchestrator_keys().
    """
    vla_port = 19884
    proxy_port = 19885

    vla_server = start_mock_vla(vla_port)

    from vlm_orchestrator.vlm import PassthroughVLM

    config = ProxyConfig(
        vla_host="127.0.0.1",
        vla_port=vla_port,
        host="127.0.0.1",
        port=proxy_port,
        vlm=PassthroughVLM(),
        client_protocol="cosmos3",
    )
    proxy = OrchestratorProxy(config)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    time.sleep(0.5)

    packer = codec.Packer()
    client = ws_client.connect(
        f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
    )
    codec.unpackb(client.recv())  # metadata

    # A cosmos3-shaped obs: stitched single image + proprio, plus the
    # orchestrator-only keys the robolab client forwards verbatim.
    obs = {
        "prompt": "put the banana in the bowl",
        "observation/image": np.zeros((540, 640, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        # orchestrator-only (must NOT reach the policy server):
        "observation/image_raw": np.zeros((540, 640, 3), dtype=np.uint8),
        "observation/depth_external": np.zeros((720, 1280, 1), dtype=np.float32),
        "observation/camera_pos": np.zeros(3, dtype=np.float32),
        "observation/camera_K": np.zeros((3, 3), dtype=np.float32),
        "gt_state": {"step": 1, "subtask": {"score": 0.0}},
    }
    client.send(packer.pack(obs))
    response = codec.unpackb(client.recv())

    received = set(response["received_keys"])
    # Prompt passes through unchanged (passthrough mode).
    assert response["received_prompt"] == "put the banana in the bowl"
    # Policy-relevant keys forwarded.
    assert "observation/image" in received
    assert "observation/joint_position" in received
    assert "observation/gripper_position" in received
    assert "prompt" in received
    # Orchestrator-only keys stripped.
    for stripped in (
        "observation/image_raw",
        "observation/depth_external",
        "observation/camera_pos",
        "observation/camera_K",
        "gt_state",
    ):
        assert stripped not in received, f"{stripped} leaked to policy server"

    client.close()
    vla_server.shutdown()


if __name__ == "__main__":
    test_passthrough()
    print("PASS: test_passthrough")
    test_rewrite()
    print("PASS: test_rewrite")
    test_cosmos3_passthrough_strips_orchestrator_only_keys()
    print("PASS: test_cosmos3_passthrough_strips_orchestrator_only_keys")
    print("All tests passed.")
