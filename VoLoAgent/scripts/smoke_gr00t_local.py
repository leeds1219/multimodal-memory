# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the GR00T protocol pipeline locally.

Spins up:
  1. A mock GR00T ZMQ server (returns zero-action chunks)
  2. The orchestrator with --frontend gr00t-zmq --backend gr00t-zmq, in
     passthrough mode
  3. A fake "--policy gr00t" eval client (ZMQ REQ) that sends one
     observation and verifies the response shape.

If this passes, the wire path is correct.  Real GR00T testing needs
the actual server (see scripts/setup_gr00t_local.md).

Usage:
    python scripts/smoke_gr00t_local.py
"""

import asyncio
import logging
import threading
import time
from contextlib import contextmanager

import numpy as np
import zmq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("smoke")


# ── 1. Mock GR00T server ──────────────────────────────────────────────


def run_mock_gr00t_server(port: int, stop_event: threading.Event):
    from vlm_orchestrator.protocols.gr00t_zmq import _gr00t_pack, _gr00t_unpack

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{port}")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    log.info(f"[mock-gr00t] listening on tcp://127.0.0.1:{port}")
    while not stop_event.is_set():
        socks = dict(poller.poll(timeout=200))
        if sock not in socks:
            continue
        req = _gr00t_unpack(sock.recv())
        if req.get("endpoint") == "ping":
            sock.send(_gr00t_pack({"ok": True}))
            continue
        log.info(f"[mock-gr00t] got obs with keys {list(req['data']['observation'])[:3]}…")
        action_dict = {
            "action.joint_position": np.zeros((1, 10, 7), dtype=np.float32),
            "action.gripper_position": np.zeros((1, 10, 1), dtype=np.float32),
        }
        sock.send(_gr00t_pack((action_dict, {})))
    sock.close(linger=0)
    ctx.term()


# ── 2. Orchestrator (in-process) ──────────────────────────────────────


def run_orchestrator(listen_port: int, gr00t_port: int, stop_event: threading.Event):
    from vlm_orchestrator.protocols.gr00t_zmq import (
        Gr00tZmqBackend, Gr00tZmqFrontend,
    )
    from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
    from vlm_orchestrator.strategies.passthrough import PassthroughStrategy
    from vlm_orchestrator.strategies.base import StrategyContext
    from vlm_orchestrator.vlm import PassthroughVLM

    ctx = StrategyContext(vlm=PassthroughVLM())
    cfg = ProxyConfig(
        host="127.0.0.1", port=listen_port,
        vla_host="127.0.0.1", vla_port=gr00t_port,
        strategy=PassthroughStrategy(ctx),
        frontend=Gr00tZmqFrontend(),
        backend=Gr00tZmqBackend("127.0.0.1", gr00t_port),
    )
    proxy = OrchestratorProxy(cfg)

    async def _run():
        # Race the orchestrator against the stop event so we can shut down.
        task = asyncio.create_task(proxy._run())
        while not stop_event.is_set():
            await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


# ── 3. Fake gr00t eval client ─────────────────────────────────────────


def run_eval_client(orch_port: int):
    from vlm_orchestrator.protocols.gr00t_zmq import (
        GR00T_RESOLUTION, _gr00t_pack, _gr00t_unpack,
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{orch_port}")

    obs = {
        "video.exterior_image_1_left": np.zeros(
            (1, 1, *GR00T_RESOLUTION, 3), dtype=np.uint8,
        ),
        "video.wrist_image_left": np.zeros(
            (1, 1, *GR00T_RESOLUTION, 3), dtype=np.uint8,
        ),
        "state.joint_position": np.zeros((1, 1, 7), dtype=np.float32),
        "state.gripper_position": np.zeros((1, 1, 1), dtype=np.float32),
        "state.eef_position": np.zeros((1, 1, 3), dtype=np.float32),
        "state.eef_rotation": np.zeros((1, 1, 3), dtype=np.float32),
        "annotation.language.language_instruction": ["pick up the block"],
        "annotation.language.language_instruction_2": ["pick up the block"],
        "annotation.language.language_instruction_3": ["pick up the block"],
        "__episode_id": "smoke_0",
        "__step": 0,
    }
    request = {
        "endpoint": "get_action",
        "data": {"observation": obs, "options": None},
    }
    log.info(f"[client] sending obs to tcp://127.0.0.1:{orch_port}")
    sock.send(_gr00t_pack(request))
    response = _gr00t_unpack(sock.recv())
    sock.close(linger=0)
    ctx.term()

    assert isinstance(response, (list, tuple)), f"bad response: {type(response)}"
    action_dict = response[0]
    joint = action_dict["action.joint_position"]
    grip = action_dict["action.gripper_position"]
    log.info(f"[client] got joint shape {joint.shape}, gripper shape {grip.shape}")
    assert joint.shape == (1, 10, 7)
    assert grip.shape == (1, 10, 1)
    log.info("[client] ✅ wire path OK")


# ── Orchestration ─────────────────────────────────────────────────────


def main():
    gr00t_port = 25555
    orch_port = 28001
    stop = threading.Event()
    threads = [
        threading.Thread(target=run_mock_gr00t_server, args=(gr00t_port, stop), daemon=True),
        threading.Thread(target=run_orchestrator, args=(orch_port, gr00t_port, stop), daemon=True),
    ]
    for t in threads:
        t.start()
    time.sleep(1.0)  # let both bind
    try:
        run_eval_client(orch_port)
        log.info("✅ smoke test passed")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=3.0)


if __name__ == "__main__":
    main()
