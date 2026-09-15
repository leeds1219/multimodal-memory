# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-based IPC backend for cross-machine VLA serving.

When the orchestrator and the VLA policy server run on different machines
that cannot open TCP sockets to each other but do share a filesystem (e.g.
a shared network mount), we use a shared directory as the transport instead
of WebSocket.

Wire format per call (orchestrator writes obs, server writes act):

    {dir}/server_config.msgpack          one-time, server-side, on startup
    {dir}/obs_{seq:010d}.msgpack         per-call, orchestrator-side
    {dir}/act_{seq:010d}.msgpack         per-call, server-side

Both writes are atomic via `.tmp` + `os.rename`. The orchestrator
deletes the act file after consuming it; the server deletes the obs
file after consuming it. Files are msgpack-packed by the same codec
the WebSocket backend uses, so the only difference is the transport.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from vlm_orchestrator.utils import codec

from .base import Backend, BackendConnection

logger = logging.getLogger(__name__)


class FileIPCBackendConnection(BackendConnection):
    """One file-IPC session. Holds a monotonic sequence counter for the
    obs/act file naming. Polling cadence is configurable.
    """

    def __init__(
        self,
        ipc_dir: Path,
        *,
        poll_interval_s: float = 0.05,
        server_config_timeout_s: float = 1800.0,
        infer_timeout_s: float = 600.0,
    ):
        self._ipc_dir = ipc_dir
        self._poll = poll_interval_s
        self._cfg_timeout = server_config_timeout_s
        self._infer_timeout = infer_timeout_s
        self._packer = codec.Packer()
        self._seq = 0

    async def recv_metadata(self) -> dict:
        cfg_path = self._ipc_dir / "server_config.msgpack"
        deadline = time.time() + self._cfg_timeout
        while not cfg_path.is_file():
            if time.time() > deadline:
                raise TimeoutError(
                    f"server_config.msgpack never appeared in "
                    f"{self._ipc_dir} (waited {self._cfg_timeout}s)"
                )
            await asyncio.sleep(self._poll)
        raw = cfg_path.read_bytes()
        meta = codec.unpackb(raw)
        logger.info(
            f"FileIPCBackend: server_config read "
            f"({len(raw)} bytes): {list(meta)}"
        )
        return meta

    async def infer(self, canonical_obs: dict) -> dict:
        seq = self._seq
        self._seq += 1
        obs_tmp = self._ipc_dir / f"obs_{seq:010d}.tmp"
        obs_path = self._ipc_dir / f"obs_{seq:010d}.msgpack"
        act_path = self._ipc_dir / f"act_{seq:010d}.msgpack"
        err_path = self._ipc_dir / f"err_{seq:010d}.txt"

        # Write obs atomically: write to .tmp, then rename.
        payload = self._packer.pack(canonical_obs)
        obs_tmp.write_bytes(payload)
        os.rename(obs_tmp, obs_path)

        # Poll for the matching act file (or an err marker).
        deadline = time.time() + self._infer_timeout
        while True:
            if act_path.is_file():
                raw = act_path.read_bytes()
                try:
                    act_path.unlink()
                except FileNotFoundError:
                    pass
                return codec.unpackb(raw)
            if err_path.is_file():
                err_text = err_path.read_text(errors="replace")[:2000]
                try:
                    err_path.unlink()
                except FileNotFoundError:
                    pass
                raise RuntimeError(
                    f"FileIPCBackend: server reported error on seq={seq}:\n"
                    f"{err_text}"
                )
            if time.time() > deadline:
                raise TimeoutError(
                    f"FileIPCBackend: no action file after "
                    f"{self._infer_timeout}s for seq={seq}"
                )
            await asyncio.sleep(self._poll)

    async def close(self) -> None:
        # Nothing to close — the directory persists.
        pass


class FileIPCBackend(Backend):
    """File-IPC backend.

    Uses `ipc_dir` (must exist) as the transport. The server should be
    started independently (e.g., on a different machine that shares
    the filesystem mount) before the orchestrator connects.
    """

    def __init__(self, ipc_dir: str, *, poll_interval_s: float = 0.05):
        self._ipc_dir = Path(ipc_dir)
        self._ipc_dir.mkdir(parents=True, exist_ok=True)
        self._poll = poll_interval_s

    async def connect(self) -> BackendConnection:
        logger.info(f"FileIPCBackend.connect() — dir={self._ipc_dir}")
        return FileIPCBackendConnection(
            self._ipc_dir, poll_interval_s=self._poll
        )
