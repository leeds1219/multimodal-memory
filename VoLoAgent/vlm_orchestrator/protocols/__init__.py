# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol modules: pluggable frontend / backend transports.

Each VLA backend has its own wire protocol and observation schema.  The
orchestrator bridges between them via two pluggable layers:

  * **Frontend** — handles the eval-client-facing connection.  Listens
    on a port in the eval client's native protocol, translates each
    incoming observation into a *canonical openpi schema* before
    handing it to the strategy layer, and translates outgoing actions
    back to the native protocol.

  * **Backend** — handles the VLA-server-facing connection.  Receives
    canonical openpi observations from the strategy layer, translates
    to the VLA's native schema, sends, and translates the action
    response back to canonical.

Strategies in ``vlm_orchestrator/strategies/`` and failure handlers in
``vlm_orchestrator/failure_handlers/`` only ever see the canonical
schema (``observation/exterior_image_1_left``, ``prompt``,
``observation/joint_position``, …).  Adding a new VLA = add one more
protocol module; no changes to strategies.

Available protocols:

  ``openpi_ws``   — WebSocket + msgpack-numpy + openpi schema.  Native
                    for pi0 / pi0_fast / pi05 / paligemma.  No
                    translation needed (canonical = native).
  ``gr00t_zmq``   — ZMQ REQ/REP + msgpack + gr00t schema.  Native for
                    NVIDIA Isaac-GR00T.  Translates schema in both
                    directions.
"""
