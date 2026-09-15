# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local smoke test for the OpenVLA protocol pipeline.

Spins up:
  1. A mock OpenVLA HTTP server (returns a fixed 7-D action)
  2. The orchestrator with --frontend openvla-rest --backend openvla-rest,
     in passthrough mode
  3. A fake "--policy openvla" eval client that POSTs once and verifies
     the response shape.

Usage:
    python scripts/smoke_openvla_local.py
"""

import asyncio
import json
import logging
import threading
import time

import json_numpy
import numpy as np
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("smoke")


# ── 1. Mock OpenVLA server ────────────────────────────────────────────


def run_mock_openvla_server(port: int, stop_event: threading.Event):
    captured: list[dict] = []

    async def handler(request):
        body = json.loads(await request.read())
        captured.append(body)
        log.info(
            f"[mock-openvla] got POST /act with keys "
            f"{sorted(body.keys())}, instruction={body.get('instruction')!r}"
        )
        # Return a known 7-D action so the client side can verify.
        action_7d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        return web.Response(
            body=json.dumps(action_7d).encode("utf-8"),
            content_type="application/json",
        )

    async def go():
        app = web.Application()
        app.router.add_post("/act", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        log.info(f"[mock-openvla] listening on http://127.0.0.1:{port}")
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        await runner.cleanup()

    asyncio.run(go())


# ── 2. Orchestrator (in-process) ──────────────────────────────────────


def run_orchestrator(listen_port: int, openvla_port: int, stop_event: threading.Event):
    from vlm_orchestrator.protocols.openvla_rest import (
        OpenVlaRestBackend, OpenVlaRestFrontend,
    )
    from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
    from vlm_orchestrator.strategies.passthrough import PassthroughStrategy
    from vlm_orchestrator.strategies.base import StrategyContext
    from vlm_orchestrator.vlm import PassthroughVLM

    ctx = StrategyContext(vlm=PassthroughVLM())
    cfg = ProxyConfig(
        host="127.0.0.1", port=listen_port,
        vla_host="127.0.0.1", vla_port=openvla_port,
        strategy=PassthroughStrategy(ctx),
        frontend=OpenVlaRestFrontend(),
        backend=OpenVlaRestBackend(
            "127.0.0.1", openvla_port, unnorm_key="droid",
        ),
    )
    proxy = OrchestratorProxy(cfg)

    async def _run():
        task = asyncio.create_task(proxy._run())
        while not stop_event.is_set():
            await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


# ── 3. Fake eval client ───────────────────────────────────────────────


def run_eval_client(orch_port: int):
    import requests
    json_numpy.patch()  # so np.ndarray serializes
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    payload = {
        "image": rgb,
        "instruction": "pick up the red block",
        "unnorm_key": "droid",
    }
    log.info(f"[client] POSTing to http://127.0.0.1:{orch_port}/act")
    resp = requests.post(
        f"http://127.0.0.1:{orch_port}/act",
        data=json.dumps(payload),
        timeout=10.0,
    )
    resp.raise_for_status()
    action = np.asarray(resp.json())
    log.info(f"[client] got action shape={action.shape}, vals={action[:3]}…")
    assert action.shape == (7,)
    np.testing.assert_array_almost_equal(
        action, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    log.info("[client] ✅ wire path OK")


# ── Orchestration ─────────────────────────────────────────────────────


def main():
    openvla_port = 28765
    orch_port = 28766
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=run_mock_openvla_server, args=(openvla_port, stop),
            daemon=True,
        ),
        threading.Thread(
            target=run_orchestrator, args=(orch_port, openvla_port, stop),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
    time.sleep(1.0)  # let servers bind
    try:
        run_eval_client(orch_port)
        log.info("✅ smoke test passed")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=3.0)


if __name__ == "__main__":
    main()
