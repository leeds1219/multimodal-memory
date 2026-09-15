# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T ZMQ protocol: native for NVIDIA Isaac-GR00T-n16-droid.

Translates between the canonical openpi schema (what strategies see)
and GR00T's native ``video.../state.../annotation....`` schema in both
directions.

Wire details (matches ``robolab/.../gr00t.py:_extract_observation``):

* Eval client request format::

    {"endpoint": "get_action",
     "data": {"observation": <gr00t obs>, "options": None}}

* Server response format::

    (action_dict, info_dict)  # tuple

* Image resolution: 180 × 320 (wide aspect).
* Rotation: client sends euler XYZ in ``state.eef_rotation``; we still
  fill ``observation/ee_quat`` on the canonical side from a synthetic
  ``[1, 0, 0, 0]`` since no strategy actually consumes it (verified
  via grep).  The original euler is stashed and forwarded verbatim
  on the backend side.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from typing import Awaitable, Callable

import msgpack
import numpy as np
import zmq
from PIL import Image

from .base import Backend, BackendConnection, Frontend, FrontendSession

logger = logging.getLogger(__name__)

# GR00T expects (H, W) = (180, 320) images.
GR00T_RESOLUTION = (180, 320)


# ──────────────────────────────────────────────────────────────────────
# msgpack-numpy serialization (matches GR00T server's _MsgSerializer)
# ──────────────────────────────────────────────────────────────────────


def _encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    # Numpy scalar types (np.float32, np.int64, np.bool_, …) — msgpack
    # has no native handler.  Reachable when nested in ``__extras``
    # (e.g. ``gt_state`` carries scalars deep in its structure).
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _decode_numpy(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _gr00t_pack(data) -> bytes:
    return msgpack.packb(data, default=_encode_numpy)


def _gr00t_unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode_numpy)


# ──────────────────────────────────────────────────────────────────────
# Quaternion (w, x, y, z) ↔ Euler (roll, pitch, yaw) XYZ
# (matches gr00t.py:quat_to_euler_xyz)
# ──────────────────────────────────────────────────────────────────────


def quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1)


# ──────────────────────────────────────────────────────────────────────
# eef_9d (xyz + 6D rotation) for GR00T-N1.7 DROID
#
# N1.7's ``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`` modality requires
# ``state.eef_9d``.  N1.6's ``OXE_DROID`` does not, but tolerates the
# extra key — verified empirically with scripts/probe_gr00t_n17.py
# ([5/5] passes).  So we always emit it; one code path serves both
# server versions.
#
# The post-multiply by ``DROID_EEF_ROTATION_CORRECT`` matches the OXE
# DROID training pipeline frame convention (TFG); without it the
# rot6d the model sees is in the wrong frame.  Verbatim from
# Isaac-GR00T-public/examples/DROID/main_gr00t.py @ n1.7-release.
# ──────────────────────────────────────────────────────────────────────
_DROID_EEF_ROTATION_CORRECT = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def compute_eef_9d(ee_xyz: np.ndarray, ee_quat_wxyz: np.ndarray) -> np.ndarray:
    """Build N1.7 DROID's ``state.eef_9d`` from (XYZ, quat-WXYZ).

    Output is ``[x, y, z, r00, r01, r02, r10, r11, r12]`` (xyz + first
    two rows of the DROID-corrected rotation matrix flattened, i.e. rot6d).

    Going quat → matrix directly (skipping Euler) avoids the silent
    intrinsic-vs-extrinsic mismatch between our hand-rolled
    ``quat_to_euler_xyz`` (extrinsic XYZ) and scipy's
    ``from_euler("XYZ", ...)`` (intrinsic XYZ).

    The (xyz, quat) inputs MUST already be in the robot base frame —
    DROID training data places eef pose relative to the Franka base.
    Use ``world_to_base_pose()`` if your source data is world-frame.
    """
    # Lazy import: scipy is not a declared dependency of vlm-orchestrator,
    # but it ships with grasp/* deps and the eval envs always have it.
    from scipy.spatial.transform import Rotation
    xyz = np.asarray(ee_xyz, dtype=np.float64).reshape(3)
    q = np.asarray(ee_quat_wxyz, dtype=np.float64).reshape(4)
    # IsaacLab / robolab convention is [w, x, y, z]; scipy wants [x, y, z, w].
    rot_mat = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() @ _DROID_EEF_ROTATION_CORRECT
    rot6d = rot_mat[:2, :].reshape(6)
    return np.concatenate([xyz, rot6d]).astype(np.float32)


def world_to_base_pose(
    pos_world: np.ndarray,
    quat_world: np.ndarray,
    base_pos_world: np.ndarray,
    base_quat_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express (pos_world, quat_world) in the robot base frame.

    All quaternions are [w, x, y, z] (IsaacLab convention).  Returns
    (pos_base, quat_base) as (3,) and (4,) float32 arrays.

    DROID training data has eef pose in the Franka base frame
    (panda_link0).  IsaacLab's ``ee_pos`` / ``ee_quat`` are in the
    world frame.  For typical IsaacLab Franka tasks the robot base
    has identity orientation but is translated above the table, so
    failing to subtract the base position pushes z out of training
    distribution by ~table-height.
    """
    from scipy.spatial.transform import Rotation
    bq = np.asarray(base_quat_world, dtype=np.float64).reshape(4)
    bp = np.asarray(base_pos_world, dtype=np.float64).reshape(3)
    pw = np.asarray(pos_world, dtype=np.float64).reshape(3)
    qw = np.asarray(quat_world, dtype=np.float64).reshape(4)

    # scipy uses [x, y, z, w]; we use [w, x, y, z].
    R_base = Rotation.from_quat([bq[1], bq[2], bq[3], bq[0]])
    R_ee = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])

    pos_base = R_base.inv().apply(pw - bp)
    R_ee_base = R_base.inv() * R_ee
    q_xyzw = R_ee_base.as_quat()
    quat_base_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    return pos_base.astype(np.float32), quat_base_wxyz


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


# ──────────────────────────────────────────────────────────────────────
# Schema translation
# ──────────────────────────────────────────────────────────────────────


def _resolve_image_canonical(canonical: dict, key: str) -> np.ndarray:
    """Prefer ``<key>_raw``; fall back to ``<key>``."""
    raw = canonical.get(key + "_raw")
    if raw is not None:
        return raw
    img = canonical.get(key)
    if img is None:
        raise KeyError(f"Missing image: tried {key + '_raw'!r} and {key!r}")
    return img


def canonical_to_gr00t(canonical: dict) -> dict:
    """Canonical openpi schema → GR00T native observation."""
    ext = _resolve_image_canonical(canonical, "observation/exterior_image_1_left")
    wrist = _resolve_image_canonical(canonical, "observation/wrist_image_left")
    ext_resized = _resize_with_pad(ext, *GR00T_RESOLUTION)
    wrist_resized = _resize_with_pad(wrist, *GR00T_RESOLUTION)

    joint_pos = np.asarray(canonical["observation/joint_position"]).astype(np.float32)
    gripper_pos = np.asarray(canonical["observation/gripper_position"]).astype(np.float32)
    ee_pos = np.asarray(canonical["observation/ee_pos"]).astype(np.float32)
    ee_quat = np.asarray(canonical["observation/ee_quat"]).astype(np.float32)
    # ee_pos / ee_quat in canonical are in the eval client's WORLD frame
    # (e.g. IsaacLab world).  DROID-trained VLAs expect proprio in the
    # robot's BASE frame.  Use ``observation/base_pos`` /
    # ``observation/base_quat`` (forwarded by an updated robolab gr00t
    # client) to translate.  Falls back to identity when the eval client
    # didn't forward them — i.e. world == base, matching legacy behavior.
    base_pos = canonical.get("observation/base_pos")
    base_quat = canonical.get("observation/base_quat")
    if base_pos is not None and base_quat is not None:
        ee_pos_base, ee_quat_base = world_to_base_pose(
            ee_pos, ee_quat,
            np.asarray(base_pos, dtype=np.float32),
            np.asarray(base_quat, dtype=np.float32),
        )
    else:
        ee_pos_base, ee_quat_base = ee_pos, ee_quat
    eef_euler_base = quat_to_euler_xyz(ee_quat_base).astype(np.float32)
    # N1.7 DROID requires state.eef_9d; N1.6 tolerates the extra key
    # (verified via scripts/probe_gr00t_n17.py [5/5]).  Always emit.
    # Compute directly from the quaternion to avoid Euler convention
    # ambiguity — see compute_eef_9d's docstring.
    eef_9d = compute_eef_9d(ee_pos_base, ee_quat_base)

    instruction = str(canonical.get("prompt", ""))
    return {
        "video.exterior_image_1_left": ext_resized[None, None, ...],
        "video.wrist_image_left": wrist_resized[None, None, ...],
        "state.joint_position": joint_pos[None, None, ...],
        "state.gripper_position": gripper_pos[None, None, ...],
        # ee pose fields go to the gr00t server in BASE frame (DROID
        # training-distribution); canonical's observation/ee_pos remains
        # world-frame for other consumers (grasp tool, failure handlers).
        "state.eef_position": ee_pos_base[None, None, ...],
        "state.eef_rotation": eef_euler_base[None, None, ...],
        "state.eef_9d": eef_9d[None, None, ...],
        "annotation.language.language_instruction": [instruction],
        "annotation.language.language_instruction_2": [instruction],
        "annotation.language.language_instruction_3": [instruction],
    }


def gr00t_to_canonical(gr00t_obs: dict) -> dict:
    """GR00T native observation → canonical openpi schema.

    Used by the Frontend when translating incoming requests from a
    ``--policy gr00t`` eval client.  Strategies see canonical keys.
    """
    # Drop (B, T) dims: gr00t images are [1, 1, H, W, C].
    ext = np.asarray(gr00t_obs["video.exterior_image_1_left"])[0, 0]
    wrist = np.asarray(gr00t_obs["video.wrist_image_left"])[0, 0]
    joint_pos = np.asarray(gr00t_obs["state.joint_position"])[0, 0]
    gripper_pos = np.asarray(gr00t_obs["state.gripper_position"])[0, 0]
    ee_pos = np.asarray(gr00t_obs["state.eef_position"])[0, 0]
    # We don't reconstruct ee_quat from euler — no canonical strategy
    # consumes it.  Provide a neutral identity quaternion as a stub.
    instruction_field = gr00t_obs.get(
        "annotation.language.language_instruction", [""],
    )
    instruction = (
        instruction_field[0] if isinstance(instruction_field, (list, tuple))
        else str(instruction_field)
    )
    return {
        # Both _raw and the standard key — strategies prefer _raw, but
        # gr00t images already arrive at 180×320, which is fine for VLM.
        "observation/exterior_image_1_left": ext,
        "observation/exterior_image_1_left_raw": ext,
        "observation/wrist_image_left": wrist,
        "observation/wrist_image_left_raw": wrist,
        "observation/joint_position": np.asarray(joint_pos, dtype=np.float32),
        "observation/gripper_position": np.asarray(gripper_pos, dtype=np.float32),
        "observation/ee_pos": np.asarray(ee_pos, dtype=np.float32),
        "observation/ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "prompt": instruction,
    }


# Minimum chunk length the frontend serves.  Must be >= the gr00t eval
# client's ``open_loop_horizon`` (default 10 in
# ``robolab/.../gr00t.py:172``) so the client never tries to read past
# the end of a chunk.  Native gr00t-N1.6 returns 32-step chunks, which
# is already above this; the constraint binds when the orchestrator's
# *grasp tool* returns chunks (default size 8, sized for pi05's
# horizon).
GR00T_MIN_CHUNK_LEN = 10


def canonical_action_to_gr00t(canonical_action: dict, min_chunk_len: int = GR00T_MIN_CHUNK_LEN) -> dict:
    """``{"actions": [N, 8]}`` → ``{"action.joint_position": [1, N, 7], "action.gripper_position": [1, N, 1]}``.

    Pads chunks shorter than ``min_chunk_len`` by repeating the last
    action (hold-position semantics).  This keeps the wire-format
    consistent for any eval client expecting at least ``min_chunk_len``
    actions per request, regardless of which orchestrator-internal
    component (VLA backend / grasp tool) produced the chunk.
    """
    actions = np.asarray(canonical_action["actions"])  # [N, 8]
    if actions.shape[0] < min_chunk_len:
        last = actions[-1:]  # keep dim, shape [1, 8]
        pad = np.tile(last, (min_chunk_len - actions.shape[0], 1))
        actions = np.concatenate([actions, pad], axis=0)
    joint = actions[:, :7].astype(np.float32)[None, ...]   # [1, N, 7]
    gripper = actions[:, 7:8].astype(np.float32)[None, ...]  # [1, N, 1]
    return {
        "action.joint_position": joint,
        "action.gripper_position": gripper,
    }


def gr00t_action_to_canonical(action_dict: dict) -> dict:
    """``action_dict`` (gr00t) → ``{"actions": [N, 8]}`` (canonical)."""
    joint = np.asarray(action_dict["action.joint_position"])[0]    # [N, 7]
    gripper = np.asarray(action_dict["action.gripper_position"])[0]  # [N, 1]
    actions = np.concatenate([joint, gripper], axis=1).astype(np.float32)
    return {"actions": actions}


# ──────────────────────────────────────────────────────────────────────
# Frontend (eval client uses --policy gr00t and connects here)
# ──────────────────────────────────────────────────────────────────────


class Gr00tZmqSession(FrontendSession):
    """One ``zmq.REP`` server lifetime = one logical session.

    GR00T's REP protocol is single-shot request/response with no
    long-lived connection state, so a single REP socket serves all
    eval-client requests sequentially.  The orchestrator's per-session
    bookkeeping (episode tracking) keys off ``__episode_id`` in the
    obs, which the frontend extracts and forwards in the canonical
    dict.
    """

    def __init__(self, socket):
        self._socket = socket

    async def send_metadata(self, metadata: dict) -> None:
        # GR00T has no metadata frame — eval client doesn't read one.
        # No-op.
        return

    async def recv_obs(self) -> dict | None:
        # ZMQ REP blocks until a request arrives.  Run the recv in a
        # thread so the asyncio event loop stays unblocked.
        while True:
            try:
                raw = await asyncio.to_thread(self._socket.recv)
            except zmq.error.ContextTerminated:
                return None
            try:
                request = _gr00t_unpack(raw)
            except Exception as e:
                logger.warning(f"Failed to unpack ZMQ request: {e}")
                await asyncio.to_thread(
                    self._socket.send, _gr00t_pack({"error": str(e)})
                )
                continue
            # Handle ping requests transparently — don't surface them
            # to the orchestrator.
            if isinstance(request, dict) and request.get("endpoint") == "ping":
                await asyncio.to_thread(
                    self._socket.send, _gr00t_pack({"ok": True})
                )
                continue
            # ``end_episode`` is the eval-client's signal that a single
            # episode has ended (mirrors the pi05 WS disconnect).  ACK
            # immediately so the eval client unblocks, then surface a
            # sentinel obs so the proxy can flush per-episode metadata
            # + create video symlinks without invoking the strategy.
            if (
                isinstance(request, dict)
                and request.get("endpoint") == "end_episode"
            ):
                await asyncio.to_thread(
                    self._socket.send, _gr00t_pack({"ok": True})
                )
                ep_data = request.get("data", {}) or {}
                return {
                    "__finalize_only": True,
                    "__episode_id": ep_data.get("episode_id"),
                }
            if not isinstance(request, dict):
                logger.warning(f"Unexpected request type: {type(request)}")
                continue
            data = request.get("data", {})
            native_obs = data.get("observation", {})
            canonical = gr00t_to_canonical(native_obs)
            # Forward proxy keys (sent inside the observation dict by
            # an updated gr00t eval client) into the canonical dict.
            for proxy_key in ("__step", "__episode_id"):
                if proxy_key in native_obs:
                    canonical[proxy_key] = native_obs[proxy_key]
            # Top-level ``__extras`` dict carries fields the gr00t
            # server doesn't accept but strategies / failure handlers
            # / grasp tool need: depth, camera pose, gt_state,
            # full-res images, etc.  Sent in canonical openpi schema
            # already (e.g. ``observation/depth_external``,
            # ``gt_state``).  Merge into canonical, overriding the
            # frontend's defaults (e.g. ``_raw`` images that
            # ``gr00t_to_canonical`` set to the 180×320 model copy
            # because gr00t doesn't send a separate full-res view).
            extras = request.get("__extras", {})
            if isinstance(extras, dict) and extras:
                canonical.update(extras)
            return canonical

    async def send_action(self, canonical_action: dict) -> None:
        gr00t_action = canonical_action_to_gr00t(canonical_action)
        # Strip orchestrator-only fields (server_timing, orchestrator_*)
        # before sending — gr00t schema doesn't carry them, but the
        # eval client ignores unknown info_dict fields.
        info: dict = {}
        for k, v in canonical_action.items():
            if k.startswith("orchestrator_") or k == "server_timing":
                info[k] = v
        response = (gr00t_action, info)
        await asyncio.to_thread(self._socket.send, _gr00t_pack(response))


class Gr00tZmqFrontend(Frontend):
    async def serve(
        self,
        host: str,
        port: int,
        on_session: Callable[[FrontendSession], Awaitable[None]],
    ) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(f"tcp://{host}:{port}")
        logger.info(f"Gr00tZmqFrontend listening on tcp://{host}:{port}")
        try:
            session = Gr00tZmqSession(socket)
            await on_session(session)
        finally:
            socket.close(linger=0)
            ctx.term()


# ──────────────────────────────────────────────────────────────────────
# Backend (orchestrator forwards to upstream gr00t server)
# ──────────────────────────────────────────────────────────────────────


class Gr00tZmqBackendConnection(BackendConnection):
    def __init__(self, host: str, port: int, api_token: str | None = None):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._uri = f"tcp://{host}:{port}"
        self._socket.connect(self._uri)
        self._api_token = api_token
        # ZMQ REQ is not thread-safe; serialize calls from the event loop.
        self._lock = threading.Lock()

    async def recv_metadata(self) -> dict:
        # GR00T has no metadata frame.  Return a minimal synthesized one.
        return {"backend": "gr00t", "uri": self._uri}

    async def infer(self, canonical_obs: dict) -> dict:
        # Drop orchestrator-only / proxy-only keys before translation.
        cleaned = {
            k: v for k, v in canonical_obs.items() if not k.startswith("__")
        }
        cleaned.pop("server_timing", None)
        gr00t_obs = canonical_to_gr00t(cleaned)
        request = {
            "endpoint": "get_action",
            "data": {"observation": gr00t_obs, "options": None},
        }
        if self._api_token:
            request["api_token"] = self._api_token

        def _zmq_round_trip():
            with self._lock:
                self._socket.send(_gr00t_pack(request))
                return self._socket.recv()

        message = await asyncio.to_thread(_zmq_round_trip)
        if message == b"ERROR":
            raise RuntimeError(
                "GR00T server reported a generic error — check server logs"
            )
        response = _gr00t_unpack(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        if not isinstance(response, (list, tuple)) or len(response) < 1:
            raise RuntimeError(f"Unexpected GR00T response: {response!r}")
        action_dict = response[0]
        return gr00t_action_to_canonical(action_dict)

    async def close(self) -> None:
        try:
            self._socket.close(linger=0)
        finally:
            self._context.term()


class Gr00tZmqBackend(Backend):
    def __init__(self, host: str, port: int, api_token: str | None = None):
        self._host = host
        self._port = port
        self._api_token = api_token

    async def connect(self) -> BackendConnection:
        return Gr00tZmqBackendConnection(
            self._host, self._port, self._api_token,
        )
