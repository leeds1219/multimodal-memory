<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Eval-client I/O contract (robolab schema)

> **Audience:** anyone writing a real-robot driver that connects to
> `vlm-orchestrator` and looks like robolab on the wire. The robot
> driver replaces the *simulator* portion of robolab; everything below
> the WebSocket layer (msgpack frames, observation keys, action shape)
> stays identical so the orchestrator doesn't know whether it's
> talking to sim or hardware.
>
> **Scope:** `--env robolab` preset only.
>
> **Out of scope (per current plan):** signal-based failure monitor
> (`--failure-monitor signal_primary` / `union_failure` /
> `intersect_video`) — its extra key (`observation/ee_pos`) is not
> required. Note that `observation/gripper_position` *is* required
> always (the openpi droid policy needs it as a policy input), even
> though it's also used as a signal-failure feature.

---

## 1. Wire transport

- **Protocol:** WebSocket. URL: `ws://<orch-ip>:<orch-port>/`. No
  subpath, no auth, no TLS (NVIDIA-internal only).
- **Codec:** msgpack-numpy via `vlm_orchestrator/codec.py`. Each frame
  is one msgpack-encoded dict; numpy arrays survive round-trip with
  `dtype` and `shape` preserved.
- **Health check:** the server answers HTTP `GET /healthz` with
  `200 OK` (`protocols/openpi_ws.py:49`). Useful for liveness probes.
- **Keepalive:** server pings every 60 s, times out after 120 s
  (`protocols/openpi_ws.py:70`). Driver should keep its WS client
  responsive to pings.

## 2. Session lifecycle

```
client                                        orchestrator
  │                                                 │
  │ ── WebSocket connect ─────────────────────────▶ │
  │                                                 │  opens upstream
  │                                                 │  VLA connection
  │ ◀──────────── metadata frame ────────────────── │  (one dict)
  │                                                 │
  │ ── obs frame #0 ──────────────────────────────▶ │
  │ ◀───────────── action response #0 ───────────── │
  │ ── obs frame #1 ──────────────────────────────▶ │
  │ ◀───────────── action response #1 ───────────── │
  │     ...                                         │
  │ ── { "__finalize_only": true, ... } ──────────▶ │  (optional, end of episode)
  │ ── close ─────────────────────────────────────▶ │
```

- **Metadata frame.** Sent once by the server, immediately after
  connect. It's the upstream VLA's metadata dict with one extra key:
  `"orchestrator": True`. The driver can ignore the contents — it just
  needs to read and discard one frame before sending the first obs
  (`proxy.py:287–293`).
- **Strict alternation.** After metadata, the driver must send exactly
  one obs and wait for exactly one action response. No batching, no
  pipelining.
- **New episode.** Either change `prompt`, or send a new
  `__episode_id`. The orchestrator detects the boundary, finalizes the
  previous episode's logs, rotates output directories, and resets its
  per-session strategy state (`proxy.py:382–448`).
- **End of episode.** Optional. Send `{"__finalize_only": True,
  "__episode_id": <same as last>}` to make the orchestrator flush
  metadata + video symlinks for the just-ended episode without
  expecting an action back (`proxy.py:314`). If you skip this, finalize
  happens on the *next* episode boundary or on session close.
- **Reconnect.** A dropped connection is harmless: open a new WS, read
  the metadata frame, continue. Episode counters in the orchestrator
  process keep incrementing across reconnects.

## 3. Observation schema (per inference)

All keys live in a single flat dict. Numeric arrays are numpy. Robolab
uses 224×224 policy images — match that.

### Required

| Key | Type | Shape / dtype | Notes |
|---|---|---|---|
| `prompt` | `str` | — | Natural-language task. Same string within an episode unless you intentionally change it. |
| `observation/exterior_image_1_left` | ndarray | `(224, 224, 3)` `uint8`, **RGB** HWC | Scene camera. Policy input. (`cli.py:367`) |
| `observation/wrist_image_left` | ndarray | `(224, 224, 3)` `uint8`, **RGB** HWC | Wrist camera. Policy input. (`cli.py:617`) |
| `observation/joint_position` | ndarray | `(7,)` `float64` | Arm joint angles in **radians**. Required by the openpi droid policy (`droid_policy.py:15, 40`) — it concatenates this with `gripper_position` to form the policy state vector internally. |
| `observation/gripper_position` | ndarray | `(1,)` `float64` | Scalar in `[0, 1]` (0 = open, 1 = closed). Required by the droid policy (`droid_policy.py:16, 36`). |

### Strongly recommended

| Key | Type | Notes |
|---|---|---|
| `__episode_id` | hashable (e.g. `str`) | Stable per-episode marker. Triggers a new-episode boundary even when `prompt` is unchanged (e.g. trial 2 of the same task). (`proxy.py:345`) |
| `__step` | `int` | Real-robot/sim step counter. Lets the orchestrator gate VLM check cadence in step units. Without it, the proxy estimates `infer_count * 8` which is correct for robolab anyway, but explicit is safer. (`proxy.py:360`) |
| `__task_slug` | `str` | Stable per-task slug used for output directory naming. Useful if `prompt` varies within a task (e.g. paraphrased instructions). (`proxy.py:352`) |

### Required for `--recovery-mode replan_grasp` / `grasp` / `vlm_grasp` (and future grasp/manipulation tools)

| Key | Type | Notes |
|---|---|---|
| `observation/depth_<cam>` | ndarray | `(H, W)` `float32`, **metres**. Use depth from the same camera as your scene image. The grasp tool indexes by suffix (`depth_exterior_image_1_left`, etc.). |
| `observation/camera_K` | ndarray | `(9,)` `float64`, row-major flatten of the 3×3 OpenCV intrinsic matrix `K`. |
| `observation/camera_extrinsic` | ndarray | `(16,)` `float64`, row-major flatten of the 4×4 **camera-to-world** matrix in **OpenCV** convention (X-right, Y-down, Z-forward). If your driver natively uses OpenGL convention, convert with `vlm_orchestrator.grasp.camera.pose_opengl_to_opencv` before sending. |
| `observation/ee_quat` | ndarray | `(4,)` `float64`, EE orientation as `(w, x, y, z)`. Used for grasp orientation planning. |

> **No silent fallback.** If you skip these keys but launch with a
> grasp recovery mode, the grasp tool will raise rather than substitute
> a hardcoded camera pose (per the "no silent lossy fallbacks" design rule). Fail loud, not subtly wrong.

### Optional (logging only)

| Key | Notes |
|---|---|
| `gt_state` | Sim-only ground-truth state machine for GT-based failure monitor. **Skip on real robot.** |
| `ground_truth_done` | Sim success flag. Skip. |

## 4. Action response (per inference)

Server returns one msgpack dict. The fields split into two groups:
fields the driver **must** consume for correct behavior, and fields
that are informational / for logging.

### 4a. Must consume (driver implementation MUST support)

| Key | Type | Notes |
|---|---|---|
| `actions` | ndarray | `(horizon, 8)`, dtype usually `float32`/`float64`. Each row is `[arm_joint_pos(7 in radians), gripper(1)]`. Gripper convention `0=open, 1=closed` — same as obs. The horizon depends on the policy: openpi pi05 returns 8, other configs may return more (e.g. 15). The driver should not hard-code a horizon — replan no more frequently than the chunk you got back. |
| `orchestrator_flush_actions` | `bool`, optional | When `True`, drop any cached chunk and use the fresh `actions` from this same response. The orchestrator sets this when it rewrites the instruction (subgoal advance, grasp handoff). Skipping this lets stale actions execute across an instruction-change boundary. (`proxy.py:569`) |

### 4b. Informational (safe to ignore; useful for logging / co-located outputs)

| Key | Type | Notes |
|---|---|---|
| `orchestrator_instruction` | `str`, optional | Current rewritten subgoal sent to the VLA. Useful in the driver's per-step log. |
| `orchestrator_original_instruction` | `str`, optional | Original task instruction. |
| `orchestrator_subgoals` | `list[str]`, optional | Full decomposition produced at episode start. |
| `orchestrator_subgoal_idx` | `int`, optional | Current index into `orchestrator_subgoals`. |
| `orchestrator_episode_log_dir` | `str`, optional | Path under `--log-dir/...` where the orchestrator saves this episode's logs. The driver may colocate its own video / hdf5 there but isn't required to. |
| `orchestrator_grasp_tool` | `dict`, optional | `{active, phase, target}` while a grasp recovery is mid-execution. |
| `orchestrator_use_front_camera` | `bool`, optional | Echoes `--use-front-camera`. |

Other `orchestrator_*` keys may appear over time; treat unknown keys
as informational.

## 5. Action-chunk replay loop

Robolab consumes the full 8-step chunk before re-inferring. The
canonical client loop is:

```
action_plan = []
while not episode_done:
    if not action_plan:
        response = ws.send_then_recv(wire_obs)
        if response.get("orchestrator_flush_actions"):
            action_plan.clear()
        action_plan = list(response["actions"])
    cmd = action_plan.pop(0)
    execute_on_robot(cmd)
```

Two real-robot considerations on top of this:

- **Shorter replan window.** Sim runs at MuJoCo speed; real-robot
  control loops are slower per step but more sensitive to stale
  actions. You may want to consume only the first N (e.g. 4) of the
  8 actions before re-inferring, to stay reactive. Reducing N below
  the chunk length never breaks the protocol — the orchestrator
  doesn't care.
- **Flush takes precedence.** Even if you've only consumed 1 of 8
  actions, when `orchestrator_flush_actions=True` arrives, drop the
  rest and re-infer immediately.

## 6. Coordinate / unit conventions (cheat sheet)

| Quantity | Units / convention |
|---|---|
| Joint angles | **radians** |
| Gripper (obs and action) | scalar, **0 = open, 1 = closed** |
| Images | `uint8`, HWC, **RGB** (not BGR), 224×224 |
| Depth | `float32`, **metres**, same H×W as the matching color image |
| Camera intrinsics `K` | OpenCV 3×3, units of pixels |
| Camera extrinsic | 4×4 **camera-to-world**, OpenCV convention (X-right, Y-down, Z-forward) |
| EE quaternion | `(w, x, y, z)` |

If your driver natively produces any of these in a different
convention (BGR, OpenGL camera, gripper inverted), convert *before*
packing the wire dict. Do not push the conversion into the
orchestrator.

## 7. Reference code

The **canonical simulator client** is `~/robolab/policies/volo/run.py`
(in the RoboLab repository). It already speaks the schema described above; the
real-robot driver should mirror its observation construction and
action-replay loop, swapping the IsaacLab obs source for live
hardware.

Within *this* repo, three files are useful when in doubt:

- `vlm_orchestrator/proxy.py` — server-side `_handle_session`
  (lines ~263–660). Source of truth for which obs keys are read and
  which `orchestrator_*` response fields mean what. If the doc and
  the code disagree, the code wins.
- `vlm_orchestrator/grasp/camera.py` — `pose_opengl_to_opencv` plus
  the camera-frame convention notes. Only relevant if you wire grasp
  recovery; ignore otherwise.
- `examples/schema_smoke_test.py` — minimal stand-in for a robot
  driver that exercises every required key on the wire. Run it
  against a live orchestrator (real or mocked VLA) to verify the
  schema independently of robot hardware. See §9.

## 8. Minimal client skeleton

This is the smallest end-to-end driver shape, including the keys
needed for `--recovery-mode replan_grasp` (depth, intrinsics,
extrinsic, EE quaternion). Replace `read_obs_from_robot()` and
`apply_action_on_robot()` with the real hardware glue.

```python
import numpy as np
from openpi_client import websocket_client_policy

ORCH_HOST = "198.51.100.10"
ORCH_PORT = 8001

# ── Camera calibration for the scene camera (set once at startup) ──
# K is 3×3 OpenCV intrinsics in pixels.
# T_cam2world is 4×4 camera-to-world in OpenCV convention
#   (X-right, Y-down, Z-forward). If your driver natively gives an
#   OpenGL-style pose, convert it before flattening:
#       from vlm_orchestrator.grasp.camera import pose_opengl_to_opencv
#       T_cam2world = pose_opengl_to_opencv(T_cam2world_gl)
CAMERA_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
CAMERA_T_CAM2WORLD = np.eye(4, dtype=np.float64)   # fill from calibration


def make_wire_obs(
    prompt, ext_rgb, wrist_rgb, joints_rad, gripper,
    step, episode_id,
    depth_m=None, ee_quat_wxyz=None,
):
    """Build one observation frame for the orchestrator.

    Required (always):
        prompt, ext_rgb, wrist_rgb, joints_rad, gripper.

    Required for grasp recovery and any future manipulation tools
    that consume 3D scene geometry (--recovery-mode replan_grasp /
    grasp / vlm_grasp today, more later):
        depth_m       — float32 (H, W) metres, same camera as ext_rgb
        ee_quat_wxyz  — float64 (4,) (w, x, y, z)
        plus CAMERA_K + CAMERA_T_CAM2WORLD attached below.

    These can be omitted only when running a mode that doesn't use
    them at all (e.g. passthrough, or subgoal without grasp recovery).
    A real-robot driver should plan to provide them by default.
    """
    wire = {
        "prompt": prompt,
        "observation/exterior_image_1_left": ext_rgb.astype(np.uint8),       # (224,224,3) RGB
        "observation/wrist_image_left":      wrist_rgb.astype(np.uint8),     # (224,224,3) RGB
        "observation/joint_position":        joints_rad.astype(np.float64),  # (7,) radians
        "observation/gripper_position": np.array([gripper], dtype=np.float64),  # (1,) 0=open, 1=closed
        "__episode_id": episode_id,
        "__step":       int(step),
    }
    if depth_m is not None:
        # Key suffix must match the scene image's suffix so the grasp
        # tool can pair them up.
        wire["observation/depth_exterior_image_1_left"] = (
            np.ascontiguousarray(depth_m, dtype=np.float32)
        )
        wire["observation/camera_K"] = CAMERA_K.flatten().astype(np.float64)
        wire["observation/camera_extrinsic"] = (
            CAMERA_T_CAM2WORLD.flatten().astype(np.float64)
        )
    if ee_quat_wxyz is not None:
        wire["observation/ee_quat"] = np.asarray(ee_quat_wxyz, dtype=np.float64)
    return wire


def run_episode(client, episode_id, prompt, max_steps=400, replan_every=8):
    """One episode. ``replan_every`` controls how many actions of the
    8-step chunk we execute before re-inferring. 8 = consume the full
    chunk (sim default); 4 stays more reactive on a real robot.
    """
    action_plan = []
    for t in range(max_steps):
        # Hardware glue: read all fields the orchestrator needs.
        # ``depth_m`` and ``ee_quat`` can be ``None`` if grasp recovery
        # is disabled.
        ext, wrist, joints, grip, depth_m, ee_quat = read_obs_from_robot()

        if not action_plan:
            wire = make_wire_obs(
                prompt, ext, wrist, joints, grip, t, episode_id,
                depth_m=depth_m, ee_quat_wxyz=ee_quat,
            )
            resp = client.infer(wire)

            # Flush handling: the orchestrator sets this when it has
            # rewritten the instruction (subgoal advance, grasp
            # handoff). Drop any cached chunk and use the fresh
            # actions in *this* response.
            if resp.get("orchestrator_flush_actions"):
                action_plan = []

            action_plan = list(resp["actions"][:replan_every])

        cmd = action_plan.pop(0)
        joints_target, gripper_target = cmd[:7], cmd[7]
        apply_action_on_robot(joints_target, gripper_target)

        if episode_done():
            break

    # Optional: flush metadata for this episode.
    client.infer({"__finalize_only": True, "__episode_id": episode_id})


def main():
    client = websocket_client_policy.WebsocketClientPolicy(
        host=ORCH_HOST, port=ORCH_PORT,
    )
    for ep_idx, prompt in enumerate(["pick up the red block", ...]):
        run_episode(client, episode_id=f"ep_{ep_idx}", prompt=prompt)
```

The `WebsocketClientPolicy` wrapper from `openpi_client` handles the
metadata read on connect, so the driver doesn't need to read it
explicitly. If you don't want to depend on `openpi-client`, the same
shape works on raw `websockets` + `vlm_orchestrator.codec.Packer` /
`codec.unpackb` — about 30 extra lines.

### When does a flush actually fire?

`orchestrator_flush_actions=True` is attached to the response of an
`infer()` call when the orchestrator has just decided that the
previous chunk is stale — typically because:
- subgoal advanced and the policy needs the new instruction;
- the failure monitor escalated to grasp recovery and the next
  chunk will be generated by the grasp tool, not the VLA;
- the grasp tool finished and control hands back to the VLA under a
  rewritten instruction.

If you replay the full 8-action chunk before re-inferring, flush will
typically arrive on the *next* `infer()` call, and clearing the
already-empty `action_plan` is a no-op — but the new chunk that
replaces it carries the new instruction's actions, which is the whole
point. If you replay shorter (e.g. 4 of 8), flush is what guarantees
you don't keep executing stale actions across the boundary.

## 9. Validating your driver — `examples/schema_smoke_test.py`

A tiny stand-in for a robot driver lives at
`examples/schema_smoke_test.py`. It connects to a running
orchestrator, sends synthetic observations matching every required
key, and asserts the response shape and `orchestrator_*` metadata.
Useful for:

- Verifying the schema layer end-to-end without robolab or hardware.
- Smoke-testing the public-IP path from another machine on the
  NVIDIA network.
- Catching schema regressions when adapting a new robot driver
  (e.g. wrong key name, wrong dtype, missing grasp tensor).

Same machine, exercising the full required schema + grasp keys::

    python examples/schema_smoke_test.py \\
        --orch-host 127.0.0.1 --orch-port 8001 --steps 3 --grasp

From a different machine on the NVIDIA network (orchestrator host at
`198.51.100.10`)::

    python examples/schema_smoke_test.py \\
        --orch-host 198.51.100.10 --orch-port 8001 --steps 3 --grasp

Exit code is `0` on PASS, `1` on FAIL. The script writes nothing to
disk; orchestrator-side artifacts (per-episode logs, `proxy.log`)
land under the orchestrator's `--log-dir` as usual.

It assumes the orchestrator's upstream VLA is already up. For a
fully offline test without a real policy server, run the existing
`pytest tests/test_proxy.py -v` — it spawns a mock VLA in-process.
