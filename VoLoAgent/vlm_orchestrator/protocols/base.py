# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract Frontend / Backend interfaces for protocol modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable


class FrontendSession(ABC):
    """One logical session with an eval client.

    For long-lived connections (WebSocket): one session per connection.
    For request/response transports (ZMQ REQ/REP): one session for the
    server's lifetime, since there's no notion of a connection.
    """

    @abstractmethod
    async def send_metadata(self, metadata: dict) -> None:
        """Send the initial metadata frame, if the protocol expects one.

        WebSocket+openpi expects a metadata frame before the first obs.
        ZMQ+gr00t has no metadata frame; implementations may no-op.
        """

    @abstractmethod
    async def recv_obs(self) -> dict | None:
        """Receive the next observation, translated to canonical openpi schema.

        Returns ``None`` when the session has ended (client disconnect,
        EOF, etc.).  Proxy-specific keys like ``__step`` and
        ``__episode_id`` should be passed through unchanged in the
        canonical dict so the orchestrator can extract them.
        """

    @abstractmethod
    async def send_action(self, canonical_action: dict) -> None:
        """Send a canonical openpi action back, translating to native protocol.

        ``canonical_action`` is a dict containing at least ``"actions"``
        of shape ``[N, 8]``, plus any orchestrator-injected keys.
        """


class Frontend(ABC):
    """Listens for eval-client connections in a specific protocol."""

    @abstractmethod
    async def serve(
        self,
        host: str,
        port: int,
        on_session: Callable[[FrontendSession], Awaitable[None]],
    ) -> None:
        """Bind to ``host:port`` and accept connections.

        For each new session, call ``on_session(session)`` and wait for
        it to finish.  Concurrent sessions are protocol-specific:
        WebSocket allows many; ZMQ REP allows one at a time.
        """


class BackendConnection(ABC):
    """One open connection to a VLA server."""

    @abstractmethod
    async def recv_metadata(self) -> dict:
        """Read the VLA server's metadata, or synthesize a minimal one
        for protocols without a metadata frame.
        """

    @abstractmethod
    async def infer(self, canonical_obs: dict) -> dict:
        """Send a canonical openpi observation, return a canonical action dict.

        The implementation is responsible for translating to/from the
        VLA's native schema.
        """

    @abstractmethod
    async def close(self) -> None: ...


class Backend(ABC):
    """Connects to a VLA server in a specific protocol."""

    @abstractmethod
    async def connect(self) -> BackendConnection:
        """Open a fresh connection to the VLA server.

        Called once per eval-client session.  The orchestrator uses one
        backend connection per frontend session.
        """
