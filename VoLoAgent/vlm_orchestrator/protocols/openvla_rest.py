# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenVLA REST protocol.

OpenVLA's deploy.py runs an HTTP server that accepts POST /act with a
JSON body and returns a JSON-encoded numpy action vector.  The wire
format (matches ``robolab/.../openvla.py:60-101``) is:

  Request body:
      {"image": <H, W, 3 uint8>, "instruction": <str>, "unnorm_key": <str>}
  (encoded with ``json_numpy`` so np.ndarray fields serialize.)

  Response body:
      <list of 7 floats>  (joint deltas only, no gripper)

The robolab client appends a default gripper (0.0 = open) when the model
returns 7-D and binarizes the gripper dim if 8-D.  We do the same here:
on the way in (action → canonical chunk), pad to 8-D; on the way out
(canonical chunk → action), drop the gripper and return the first row
because OpenVLA is single-step.

OpenVLA single-step is significantly faster than chunked policies but
means each eval-client step incurs one HTTP round-trip.  No native
support for action chunking — strategies that expect chunks (grasp tool
generates 8-step chunks) will see those chunks truncated to 1 action
when forwarded to a ``--policy openvla`` eval client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Awaitable, Callable

import aiohttp
import numpy as np
from aiohttp import web
from PIL import Image

# json_numpy patches the stdlib json module to handle np.ndarray.  The
# OpenVLA server expects this exact format on the wire (see
# ``robolab/.../openvla.py``).  We patch on import so subsequent
# json.dumps / json.loads handle arrays transparently.
import json_numpy as _json_numpy  # noqa: F401
_json_numpy.patch()

from .base import Backend, BackendConnection, Frontend, FrontendSession

logger = logging.getLogger(__name__)

# OpenVLA's training data is rendered at 256×256.
OPENVLA_RESOLUTION = (256, 256)


# ──────────────────────────────────────────────────────────────────────
# Image resize (aspect-preserving with zero padding)
# ──────────────────────────────────────────────────────────────────────


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[-3:-1] == (height, width):
        return image
    cur_h, cur_w = image.shape[-3:-1]
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    pil = Image.fromarray(image)
    resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    out = Image.new(resized.mode, (width, height), 0)
    out.paste(resized, (max(0, (width - new_w) // 2),
                        max(0, (height - new_h) // 2)))
    return np.asarray(out)


def _resolve_image_canonical(canonical: dict, key: str) -> np.ndarray:
    raw = canonical.get(key + "_raw")
    if raw is not None:
        return raw
    img = canonical.get(key)
    if img is None:
        raise KeyError(f"Missing image: tried {key + '_raw'!r} and {key!r}")
    return img


# ──────────────────────────────────────────────────────────────────────
# Schema translation
# ──────────────────────────────────────────────────────────────────────


def canonical_to_openvla(
    canonical: dict, unnorm_key: str = "droid",
) -> dict:
    """Canonical openpi schema → OpenVLA HTTP body."""
    img = _resolve_image_canonical(canonical, "observation/exterior_image_1_left")
    img_resized = _resize_with_pad(img, *OPENVLA_RESOLUTION)
    instruction = str(canonical.get("prompt", ""))
    return {
        "image": img_resized,
        "instruction": instruction,
        # Caller may override unnorm_key per-request via ``__unnorm_key``;
        # otherwise use the backend's configured default.
        "unnorm_key": str(canonical.get("__unnorm_key", unnorm_key)),
    }


def openvla_to_canonical(body: dict) -> dict:
    """OpenVLA HTTP body → canonical openpi schema.

    OpenVLA only sends image + instruction + unnorm_key, so we synthesize
    zero state arrays for fields strategies/handlers may reference.
    Fields supplied as zeros: joint_position, gripper_position, ee_pos,
    ee_quat (identity).
    """
    img = np.asarray(body["image"])
    instruction = body.get("instruction", "")
    unnorm_key = body.get("unnorm_key", "")
    return {
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_1_left_raw": img,
        # OpenVLA has no wrist camera in its standard payload — provide
        # the same exterior view as a stand-in so strategies that index
        # ``observation/wrist_image_left`` don't KeyError.
        "observation/wrist_image_left": img,
        "observation/wrist_image_left_raw": img,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "observation/ee_pos": np.zeros(3, dtype=np.float32),
        "observation/ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "prompt": instruction,
        "__unnorm_key": unnorm_key,
    }


def openvla_action_to_canonical(action: np.ndarray) -> dict:
    """OpenVLA single-step action [7] (or [8]) → canonical chunk [1, 8]."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size == 7:
        # Append default closed-gripper (0.0); robolab eval client
        # binarizes anyway, so the value here is mostly a placeholder.
        action = np.concatenate([action, np.zeros(1, dtype=np.float32)])
    if action.size != 8:
        raise ValueError(
            f"Unexpected OpenVLA action size: {action.size} (want 7 or 8)"
        )
    # Wrap as a 1-step "chunk" so the orchestrator's response-attachment
    # logic and downstream code see the standard ``[N, 8]`` shape.
    return {"actions": action.reshape(1, 8)}


def canonical_action_to_openvla(canonical_action: dict) -> np.ndarray:
    """Canonical ``{"actions": [N, 8]}`` → OpenVLA single-step [7].

    OpenVLA is single-step.  Strategies/grasp-tool that generate longer
    chunks get truncated to the first action.  Drop the gripper since
    OpenVLA's response convention is the 7-D arm action; the eval
    client appends and binarizes downstream.
    """
    actions = np.asarray(canonical_action["actions"])  # [N, 8]
    return actions[0, :7].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Frontend (eval client uses --policy openvla and POSTs here)
# ──────────────────────────────────────────────────────────────────────


class OpenVlaRestSession(FrontendSession):
    """One logical session for an HTTP-based eval client.

    HTTP /act is request/response (no long-lived connection state), so
    the orchestrator treats the entire HTTP server's lifetime as one
    session.  Per-eval-client state (episode tracking) keys off
    ``__episode_id`` in the request body when present.
    """

    def __init__(self):
        self._req_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._resp_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def send_metadata(self, metadata: dict) -> None:
        # OpenVLA has no metadata frame.  No-op.
        return

    async def recv_obs(self) -> dict | None:
        item = await self._req_queue.get()
        if item is None:
            return None
        return item  # canonical obs

    async def send_action(self, canonical_action: dict) -> None:
        await self._resp_queue.put(canonical_action)

    # Used internally by the HTTP handler to drive the session.
    async def _handle_post_act(self, body: dict) -> np.ndarray:
        canonical = openvla_to_canonical(body)
        await self._req_queue.put(canonical)
        canonical_action = await self._resp_queue.get()
        return canonical_action_to_openvla(canonical_action)


class OpenVlaRestFrontend(Frontend):
    async def serve(
        self,
        host: str,
        port: int,
        on_session: Callable[[FrontendSession], Awaitable[None]],
    ) -> None:
        session = OpenVlaRestSession()

        async def _act_handler(request: web.Request) -> web.Response:
            raw = await request.read()
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception as e:
                return web.Response(status=400, text=f"bad json: {e}")
            try:
                action_7d = await session._handle_post_act(body)
            except Exception as e:
                logger.exception(f"OpenVLA frontend session error: {e}")
                return web.Response(status=500, text=f"server error: {e}")
            return web.Response(
                body=json.dumps(action_7d.tolist()).encode("utf-8"),
                content_type="application/json",
            )

        app = web.Application()
        app.router.add_post("/act", _act_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info(f"OpenVlaRestFrontend listening on http://{host}:{port}")
        try:
            # Run the orchestrator's session loop; HTTP server stays up
            # in the background until the session ends or is cancelled.
            await on_session(session)
        finally:
            # Signal session end to any pending recv_obs.
            await session._req_queue.put(None)
            await runner.cleanup()


# ──────────────────────────────────────────────────────────────────────
# Backend (orchestrator forwards to upstream OpenVLA deploy.py)
# ──────────────────────────────────────────────────────────────────────


class OpenVlaRestBackendConnection(BackendConnection):
    def __init__(self, host: str, port: int, unnorm_key: str, timeout_s: float):
        self._url = f"http://{host}:{port}/act"
        self._unnorm_key = unnorm_key
        self._timeout_s = timeout_s
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout_s),
            )
        return self._session

    async def recv_metadata(self) -> dict:
        # OpenVLA has no metadata frame.
        return {"backend": "openvla", "url": self._url, "unnorm_key": self._unnorm_key}

    async def infer(self, canonical_obs: dict) -> dict:
        body = canonical_to_openvla(canonical_obs, self._unnorm_key)
        sess = await self._ensure_session()
        # Use json_numpy-patched json.dumps so np.ndarray serializes
        # correctly, but explicitly set Content-Type so FastAPI parses
        # the body as JSON (vs. treating raw ``data=`` bytes as text).
        async with sess.post(
            self._url,
            data=json.dumps(body),
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            response_json = json.loads(await resp.text())
        action = np.asarray(response_json)
        return openvla_action_to_canonical(action)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class OpenVlaRestBackend(Backend):
    def __init__(
        self,
        host: str,
        port: int,
        unnorm_key: str = "droid",
        timeout_s: float = 30.0,
    ):
        self._host = host
        self._port = port
        self._unnorm_key = unnorm_key
        self._timeout_s = timeout_s

    async def connect(self) -> BackendConnection:
        return OpenVlaRestBackendConnection(
            self._host, self._port, self._unnorm_key, self._timeout_s,
        )
