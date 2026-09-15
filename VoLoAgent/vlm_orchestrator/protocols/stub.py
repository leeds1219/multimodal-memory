# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process backend stub for ``--mode tool_chain``.

tool_chain bypasses the VLA — strategies emit their own action chunks
(hold-position chunks or tool-generated trajectories).  But
``proxy.py`` unconditionally connects to a backend and reads a
metadata frame before any strategy code runs (the openpi WebSocket
protocol always sends one frame on connect).

Rather than launch a real VLA (waste of GPU: pi0.5 weights load
+ idle JAX process) or a separate stub-server process (extra step for
local runners), this backend synthesizes the metadata frame in-process.

``infer()`` raises if anything ever forwards an obs to the VLA — which
tool_chain shouldn't.  The exception is intentionally loud so a routing
bug surfaces immediately rather than silently no-op'ing.
"""

from __future__ import annotations

from .base import Backend, BackendConnection

# Same shape robolab's pi05 client expects.  ``action_horizon`` sizes
# the chunk buffer; ``action_dim`` is the robot DoF + gripper; the rest
# are informational.  Robolab doesn't tightly validate these — pi0.5's
# real values come from the policy weights, but tool_chain doesn't
# consume any of them.
DEFAULT_METADATA = {
    "model": "tool-chain-stub-backend",
    "action_horizon": 8,
    "action_dim": 8,
    "control_frequency_hz": 20.0,
    "orchestrator_notice": (
        "In-process stub backend.  --mode tool_chain bypasses the VLA; "
        "this backend exists only to satisfy proxy.py's metadata "
        "handshake."
    ),
}


class StubBackendConnection(BackendConnection):
    async def recv_metadata(self) -> dict:
        return dict(DEFAULT_METADATA)

    async def infer(self, canonical_obs: dict) -> dict:
        raise RuntimeError(
            "StubBackendConnection.infer was called — tool_chain mode "
            "should never forward observations to the VLA.  Something "
            "in proxy.py's routing logic is wrong."
        )

    async def close(self) -> None:
        return


class StubBackend(Backend):
    """Backend that synthesizes openpi metadata in-process; never connects
    to a real VLA.  Used by ``--mode tool_chain``."""

    async def connect(self) -> BackendConnection:
        return StubBackendConnection()
