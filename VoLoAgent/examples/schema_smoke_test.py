#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal eval-client smoke test for the orchestrator's robolab schema.

Pure client. Connects to a running ``vlm-orchestrator``, sends a few
synthetic observations matching the robolab schema documented in
``docs/instructions/eval-client-contract.md``, and asserts each response carries an
``actions`` array of the right shape. Optionally includes the
grasp-tool keys (depth + intrinsics + extrinsic + EE quaternion) when
``--grasp`` is passed.

Prerequisite: an orchestrator (and its upstream VLA, real or mocked)
must already be running. The VLA setup is the orchestrator's concern,
not this script's. If you want a fully self-contained offline test
without a real policy server, run the existing ``tests/test_proxy.py``
suite instead — it spawns a mock VLA in-process.

Usage
-----

Same machine::

    # Terminal A: a real VLA on :8000, then the orchestrator on :8001:
    vlm-orchestrator --mode passthrough \\
        --vla-host 127.0.0.1 --vla-port 8000 --port 8001

    # Terminal B:
    python examples/schema_smoke_test.py \\
        --orch-host 127.0.0.1 --orch-port 8001 --steps 3 --grasp

Cross-machine (NVIDIA-network reachability)::

    # Other machine — orch host is 198.51.100.10:
    python examples/schema_smoke_test.py \\
        --orch-host 198.51.100.10 --orch-port 8001 --steps 3

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import websockets.sync.client as ws_client

from vlm_orchestrator.utils import codec

logger = logging.getLogger("schema_smoke")


def _make_wire_obs(
    step: int, episode_id: str, prompt: str, with_grasp_keys: bool,
) -> dict:
    rng = np.random.default_rng(step)
    ext = (rng.integers(0, 256, size=(224, 224, 3))).astype(np.uint8)
    wrist = (rng.integers(0, 256, size=(224, 224, 3))).astype(np.uint8)
    joint_position = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float64)
    gripper_position = np.array([0.0], dtype=np.float64)        # 0=open, 1=closed
    wire = {
        "prompt": prompt,
        "observation/exterior_image_1_left": ext,
        "observation/wrist_image_left":      wrist,
        "observation/joint_position":        joint_position,    # (7,) radians
        "observation/gripper_position":      gripper_position,  # (1,) scalar
        "__episode_id": episode_id,
        "__step":       int(step),
    }
    if with_grasp_keys:
        # Synthetic depth + camera matrices — values are nonsensical but
        # shapes / dtypes match the contract. Use real calibration for
        # actual grasp-recovery testing.
        depth = rng.uniform(0.3, 2.0, size=(224, 224)).astype(np.float32)
        K = np.array(
            [[300.0, 0.0, 112.0],
             [0.0, 300.0, 112.0],
             [0.0,   0.0,   1.0]], dtype=np.float64,
        )
        T_cam2world = np.eye(4, dtype=np.float64)
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # (w,x,y,z)
        wire["observation/depth_exterior_image_1_left"] = depth
        wire["observation/camera_K"] = K.flatten()
        wire["observation/camera_extrinsic"] = T_cam2world.flatten()
        wire["observation/ee_quat"] = ee_quat
    return wire


def run_client(host: str, port: int, steps: int, grasp: bool) -> bool:
    url = f"ws://{host}:{port}"
    logger.info(f"connecting to {url} ...")
    packer = codec.Packer()
    try:
        ws = ws_client.connect(url, compression=None, max_size=None)
    except Exception as e:
        logger.error(f"connect failed: {e}")
        return False

    try:
        metadata = codec.unpackb(ws.recv())
        if not metadata.get("orchestrator"):
            logger.error(
                f"[FAIL] metadata missing 'orchestrator' marker: {metadata!r}"
            )
            return False
        logger.info(f"metadata OK: keys={sorted(metadata)[:6]}…")

        episode_id = "smoke_ep_0"
        for t in range(steps):
            wire = _make_wire_obs(
                step=t, episode_id=episode_id,
                prompt="pick up the red block",
                with_grasp_keys=grasp,
            )
            ws.send(packer.pack(wire))
            resp = codec.unpackb(ws.recv())
            actions = resp.get("actions")
            if actions is None:
                logger.error(f"[FAIL step {t}] response missing 'actions': "
                             f"keys={sorted(resp)[:8]}…")
                return False
            if actions.ndim != 2 or actions.shape[1] != 8:
                logger.error(
                    f"[FAIL step {t}] unexpected actions shape: "
                    f"{actions.shape} (want (H, 8))"
                )
                return False
            flush = bool(resp.get("orchestrator_flush_actions", False))
            instr = resp.get("orchestrator_instruction", "")
            logger.info(
                f"step {t}: actions={actions.shape}, "
                f"flush={flush}, instruction={instr!r}"
            )

        ws.send(packer.pack({
            "__finalize_only": True,
            "__episode_id": episode_id,
        }))
        time.sleep(0.2)  # let the proxy process the finalize sentinel

    finally:
        try:
            ws.close()
        except Exception:
            pass

    logger.info("[PASS] all assertions held")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--orch-host", default="127.0.0.1")
    parser.add_argument("--orch-port", type=int, default=8001)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--grasp", action="store_true",
                        help="Include depth + intrinsics + extrinsic + ee_quat")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
    )
    ok = run_client(args.orch_host, args.orch_port, args.steps, args.grasp)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
