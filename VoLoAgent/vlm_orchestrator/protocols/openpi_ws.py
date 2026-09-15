# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenPI WebSocket protocol: native for pi0 / pi05 / paligemma.

Canonical schema = native schema, so this module is essentially a
pass-through wrapper around WebSocket + msgpack-numpy.
"""

from __future__ import annotations

import asyncio
import functools
import http
import logging
from typing import Awaitable, Callable

import websockets.asyncio.server as ws_server
import websockets.sync.client as ws_client

from vlm_orchestrator.utils import codec

from .base import Backend, BackendConnection, Frontend, FrontendSession

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Frontend
# ──────────────────────────────────────────────────────────────────────


class OpenpiWsSession(FrontendSession):
    def __init__(self, ws):
        self._ws = ws
        self._packer = codec.Packer()

    async def send_metadata(self, metadata: dict) -> None:
        await self._ws.send(self._packer.pack(metadata))

    async def recv_obs(self) -> dict | None:
        try:
            raw = await self._ws.recv()
        except Exception:
            return None
        return codec.unpackb(raw)

    async def send_action(self, canonical_action: dict) -> None:
        await self._ws.send(self._packer.pack(canonical_action))


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class OpenpiWsFrontend(Frontend):
    async def serve(
        self,
        host: str,
        port: int,
        on_session: Callable[[FrontendSession], Awaitable[None]],
    ) -> None:
        async def _ws_handler(ws):
            session = OpenpiWsSession(ws)
            await on_session(session)

        async with ws_server.serve(
            _ws_handler, host, port,
            compression=None, max_size=None,
            process_request=_health_check,
            ping_interval=60, ping_timeout=120,
        ) as server:
            logger.info(f"OpenpiWsFrontend listening on {host}:{port}")
            await server.serve_forever()


# ──────────────────────────────────────────────────────────────────────
# Backend
# ──────────────────────────────────────────────────────────────────────


class OpenpiWsBackendConnection(BackendConnection):
    """Wraps a sync ``websockets.sync.client`` connection.  We use the
    sync API because the upstream VLA blocks during JIT compilation
    (~30–60 s on first inference) and won't respond to async pings.
    """

    def __init__(self, vla_ws):
        self._ws = vla_ws
        self._packer = codec.Packer()

    async def recv_metadata(self) -> dict:
        raw = await asyncio.to_thread(self._ws.recv)
        return codec.unpackb(raw)

    async def infer(self, canonical_obs: dict) -> dict:
        await asyncio.to_thread(self._ws.send, self._packer.pack(canonical_obs))
        raw = await asyncio.to_thread(self._ws.recv)
        return codec.unpackb(raw)

    async def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


class OpenpiWsBackend(Backend):
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

    async def connect(self) -> BackendConnection:
        uri = f"ws://{self._host}:{self._port}"
        # Disable pings on the VLA connection.  The VLA server blocks
        # its event loop during JAX JIT compilation (30–60 s on first
        # inference) and cannot respond to pings in that window.
        ws = await asyncio.to_thread(
            functools.partial(
                ws_client.connect, uri,
                compression=None, max_size=None, ping_interval=None,
            )
        )
        logger.info(f"OpenpiWsBackend connected to {uri}")
        return OpenpiWsBackendConnection(ws)
