#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for multi-episode log saving.

Verifies the fix for the bug where only episode_1/ was ever created because
_current_episode_id reset to 0 per websocket connection.

The test:
1. Starts a mock VLA server.
2. Starts the OrchestratorProxy with a temp log_dir.
3. Makes 3 **separate** websocket connections (simulating 3 robolab episodes).
   Each connection sends a few observations with the same prompt, then disconnects.
4. Asserts that 3 separate episode directories (episode_1, episode_2, episode_3)
   were created, each with correct metadata.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time

import numpy as np
import websockets.sync.client as ws_client
import websockets.sync.server as ws_server_sync

from vlm_orchestrator.utils import codec
from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig


# --- Mock VLA Server (same as test_proxy.py) ---

def mock_vla_handler(ws):
    packer = codec.Packer()
    ws.send(packer.pack({"model": "mock-vla", "version": "test"}))
    while True:
        try:
            raw = ws.recv()
            obs = codec.unpackb(raw)
            response = {
                "actions": np.zeros((8, 8), dtype=np.float32),
                "received_prompt": obs.get("prompt", ""),
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


def make_obs(prompt: str) -> dict:
    """Build a minimal observation dict."""
    return {
        "prompt": prompt,
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
    }


def run_one_episode(proxy_port: int, prompt: str, n_steps: int = 3):
    """Simulate one robolab episode: connect, send n_steps observations, disconnect."""
    packer = codec.Packer()
    client = ws_client.connect(
        f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
    )
    # Receive metadata
    metadata = codec.unpackb(client.recv())
    assert metadata.get("orchestrator") is True, f"Missing orchestrator flag: {metadata}"

    obs = make_obs(prompt)
    for _ in range(n_steps):
        client.send(packer.pack(obs))
        response = codec.unpackb(client.recv())
        assert response["received_prompt"] == prompt

    client.close()
    # Give proxy time to finalize
    time.sleep(0.3)


def run_shared_connection_episodes(proxy_port: int, prompt: str, n_episodes: int, n_steps: int = 2):
    """Simulate the lh-on-main runner: ONE persistent connection, multiple
    same-task episodes distinguished only by the client-supplied
    ``__episode_id`` marker (no prompt change, no disconnect between episodes).
    """
    packer = codec.Packer()
    client = ws_client.connect(
        f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
    )
    metadata = codec.unpackb(client.recv())
    assert metadata.get("orchestrator") is True, f"Missing orchestrator flag: {metadata}"

    for ep_idx in range(n_episodes):
        obs = make_obs(prompt)
        obs["__episode_id"] = ep_idx  # the marker our base_client fix forwards
        for _ in range(n_steps):
            client.send(packer.pack(obs))
            response = codec.unpackb(client.recv())
            assert response["received_prompt"] == prompt

    client.close()
    time.sleep(0.3)


def test_multi_episode_shared_connection_episode_id_marker():
    """Same task, SHARED connection, episodes told apart only by __episode_id.

    This is the lh-on-main regression: the runner reuses one client/connection
    across every episode and never changes the prompt, so the proxy's
    infer_count==0 and prompt-change heuristics both fail. Only the
    client-forwarded ``__episode_id`` marker (base_client._eval_episode_idx)
    lets the proxy rotate per-episode dirs.
    """
    vla_port = 19897
    proxy_port = 19898
    log_dir = tempfile.mkdtemp(prefix="episode_log_test_")

    try:
        vla_server = start_mock_vla(vla_port)

        config = ProxyConfig(
            vla_host="127.0.0.1",
            vla_port=vla_port,
            host="127.0.0.1",
            port=proxy_port,
            log_dir=log_dir,
        )
        proxy = OrchestratorProxy(config)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        time.sleep(0.5)

        # 3 episodes of the SAME task over ONE connection.
        run_shared_connection_episodes(proxy_port, "pick up the banana", n_episodes=3, n_steps=2)

        banana_dir = os.path.join(log_dir, "pick_up_the_banana")
        assert os.path.isdir(banana_dir), f"Missing task dir: {os.listdir(log_dir)}"
        eps = sorted(os.listdir(banana_dir))
        for expected in ("episode_1", "episode_2", "episode_3"):
            assert expected in eps, (
                f"Shared-connection same-task episodes did not rotate: "
                f"expected {expected}, got {eps}"
            )

        print("\n  PASS: Shared-connection same-task episodes rotate via __episode_id!")
        return True

    finally:
        try:
            vla_server.shutdown()
        except Exception:
            pass
        shutil.rmtree(log_dir, ignore_errors=True)


def test_multi_episode_separate_connections():
    """3 separate connections (episodes) should create episode_1/, episode_2/, episode_3/."""
    vla_port = 19990
    proxy_port = 19991
    log_dir = tempfile.mkdtemp(prefix="episode_log_test_")

    try:
        # Start mock VLA
        vla_server = start_mock_vla(vla_port)

        # Start proxy with log_dir
        config = ProxyConfig(
            vla_host="127.0.0.1",
            vla_port=vla_port,
            host="127.0.0.1",
            port=proxy_port,
            log_dir=log_dir,
        )
        proxy = OrchestratorProxy(config)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        time.sleep(0.5)

        prompt = "pick up the banana"
        task_slug = "pick_up_the_banana"

        # Simulate 3 separate robolab episodes (separate connections!)
        for ep in range(3):
            print(f"  Running episode {ep + 1}...")
            run_one_episode(proxy_port, prompt, n_steps=3)

        # Check that 3 separate episode directories were created
        task_dir = os.path.join(log_dir, task_slug)
        assert os.path.isdir(task_dir), f"Task dir not found: {task_dir}"

        contents = sorted(os.listdir(task_dir))
        print(f"  Task dir contents: {contents}")

        for ep_id in [1, 2, 3]:
            ep_dir = os.path.join(task_dir, f"episode_{ep_id}")
            assert os.path.isdir(ep_dir), (
                f"episode_{ep_id}/ not found! Contents: {contents}"
            )

            # Check metadata.json
            meta_path = os.path.join(ep_dir, "metadata.json")
            assert os.path.exists(meta_path), f"metadata.json not found in {ep_dir}"
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["episode_id"] == ep_id, (
                f"Expected episode_id={ep_id}, got {meta['episode_id']}"
            )
            assert meta["original_instruction"] == prompt
            assert "end_timestamp" in meta, "metadata not finalized"
            assert meta["infer_count"] == 3, (
                f"Expected 3 inferences, got {meta['infer_count']}"
            )
            print(f"  episode_{ep_id}/metadata.json: OK (episode_id={meta['episode_id']}, infer_count={meta['infer_count']})")

        print("\n  PASS: All 3 episodes logged to separate directories!")
        return True

    finally:
        # Cleanup
        try:
            vla_server.shutdown()
        except Exception:
            pass
        shutil.rmtree(log_dir, ignore_errors=True)


def test_multi_episode_different_tasks():
    """Episodes with different prompts should create separate task dirs."""
    vla_port = 19992
    proxy_port = 19993
    log_dir = tempfile.mkdtemp(prefix="episode_log_test_")

    try:
        vla_server = start_mock_vla(vla_port)

        config = ProxyConfig(
            vla_host="127.0.0.1",
            vla_port=vla_port,
            host="127.0.0.1",
            port=proxy_port,
            log_dir=log_dir,
        )
        proxy = OrchestratorProxy(config)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        time.sleep(0.5)

        # Episode 1 & 2: same task
        run_one_episode(proxy_port, "pick up the banana", n_steps=2)
        run_one_episode(proxy_port, "pick up the banana", n_steps=2)
        # Episode 3: different task
        run_one_episode(proxy_port, "stack the bowls", n_steps=2)

        # Check directory structure
        all_dirs = []
        for root, dirs, files in os.walk(log_dir):
            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), log_dir)
                all_dirs.append(rel)
        print(f"  All dirs under log_dir: {sorted(all_dirs)}")

        # banana task should have episode_1 and episode_2
        banana_dir = os.path.join(log_dir, "pick_up_the_banana")
        assert os.path.isdir(banana_dir)
        banana_eps = sorted(os.listdir(banana_dir))
        assert "episode_1" in banana_eps, f"Missing episode_1 in banana: {banana_eps}"
        assert "episode_2" in banana_eps, f"Missing episode_2 in banana: {banana_eps}"

        # bowls task should have episode_3 (global counter continues!)
        bowls_dir = os.path.join(log_dir, "stack_the_bowls")
        assert os.path.isdir(bowls_dir)
        bowls_eps = sorted(os.listdir(bowls_dir))
        assert "episode_3" in bowls_eps, f"Expected episode_3 in bowls: {bowls_eps}"

        print("\n  PASS: Different tasks get separate dirs with correct global episode IDs!")
        return True

    finally:
        try:
            vla_server.shutdown()
        except Exception:
            pass
        shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Multi-episode separate connections (same task)")
    print("=" * 60)
    test_multi_episode_separate_connections()

    print()
    print("=" * 60)
    print("Test 2: Multi-episode different tasks")
    print("=" * 60)
    test_multi_episode_different_tasks()

    print()
    print("=" * 60)
    print("Test 3: Shared-connection same-task episodes (__episode_id marker)")
    print("=" * 60)
    test_multi_episode_shared_connection_episode_id_marker()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
