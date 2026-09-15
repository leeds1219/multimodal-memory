<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Robolab Robot Support Analysis

> **Date:** 2026-03-27 · **Scope:** `~/robolab` (IsaacLab sim-eval) + `~/openpi` (policy server)

---

## 1. What Robolab Currently Supports

### Robot Configurations (`robolab/robots/`)

Robolab ships **only Franka Panda** variants — there are no other robot families:

| File | Robot | Gripper | Notes |
|------|-------|---------|-------|
| `droid.py` | Franka Panda | Robotiq 2F-85 | Primary config. Custom USD (`franka_robotiq_2f_85_flattened.usd`). Joint-position control, wrist cam, binary gripper (0=open, 1=closed). **This is what all experiments use.** |
| `franka.py` | Franka Panda | Default Panda fingers | IsaacLab nucleus USD. Low-PD gains (stiffness 80). FrameTransformer-based EE tracking. |
| `franka_high_pd.py` | Franka Panda | Default Panda fingers | Same as `franka.py` but high-PD gains (stiffness 400, damping 80). |
| `franka_definitions.py` | — | — | Shared action configs (IK absolute, IK relative, joint-position) and helper observation functions for Franka. |
| `delta_actions.py` | — | — | Utility: converts target EE pose → delta-pose action for any single-arm with a 6-DoF delta + 1-DoF gripper interface. |

### Robot Assets (`robolab/assets/robots/`)

A single file: **`franka_robotiq_2f_85_flattened.usd`** — the Franka Panda arm with a Robotiq 2F-85 gripper.

### Action Space

All action configs assume a **single 7-DoF arm + 1 gripper** structure:
- `DroidJointPositionActionCfg`: 7 joint-position dims (`panda_joint.*`) + binary `finger_joint`.
- `FrankaJointPositionActionCfg` / `FrankaIKActionCfg` / `FrankaRelIKActionCfg`: 7 arm joints + binary finger action.

### Observation Space

`ProprioceptionObservationCfg` provides:
- `arm_joint_pos` — 7 joint angles
- `gripper_pos` — 1 scalar (0–1)
- `ee_pos` — 3D position
- `ee_quat` — 4D quaternion

Image observations (`ImageObsCfg`): one `external_cam` + one `wrist_cam`.

### Inference Clients (`policies/droid_jointpos/inference/`)

All six inference clients share the identical observation extraction code and expect `arm_joint_pos[7]`, `gripper_pos[1]`, and two RGB images:

| Client | Model | Notes |
|--------|-------|-------|
| `pi0_family.py` | π₀ / π₀-FAST / π₀.5 | WebSocket via openpi. Action chunks (horizon 8–16). |
| `openvla.py` | OpenVLA | REST `/act`. Single-step actions. |
| `openvla_oft.py` | OpenVLA-OFT | REST variant. |
| `gr00t.py` | GR00T (NVIDIA) | ZMQ protocol. EE pose → euler conversion. |
| `dreamzero.py` | DreamZero | WebSocket, multi-camera world model. |
| `droid_molmo.py` | Molmo-Act | REST, Molmo VLM. |

### Environment Factory

`auto_register_droid_envs()` hardcodes the robot:
```python
robot_cfg=DroidCfg,
actions_cfg=DroidJointPositionActionCfg(),
contact_gripper=contact_gripper,  # Robotiq finger regex
```
The factory (`EnvFactory`) passes `robot_cfg` to `generate_scene_env_cfg()` which uses it as a mixin base class for the scene. **The robot_cfg is pluggable** — it's passed as a parameter, not imported inside the factory itself.

---

## 2. What OpenPI Supports (Policy Side)

### Policy Transform Configs (`openpi/src/openpi/policies/`)

| File | Robot | State dim | Action dim | Cameras |
|------|-------|-----------|------------|---------|
| `droid_policy.py` | Franka (DROID) | 8 (7 joints + 1 gripper) | 8 | base + wrist (2) |
| `aloha_policy.py` | **ALOHA (dual-arm)** | **14** (2 × 6 joints + 2 grippers) | **14** | cam_high + cam_low + cam_left_wrist + cam_right_wrist (4) |
| `ur5e_policy.py` | **UR5e** | 7 (6 DoF + 1 gripper), padded to model dim | **7** | base + wrist (2) |
| `libero_policy.py` | Franka (LIBERO sim) | 8 | 7 | base + wrist (2) |

### Registered Training/Inference Configs (`openpi/src/openpi/training/config.py`)

| Config name(s) | Robot | Type |
|---------------|-------|------|
| `pi0_droid`, `pi0_fast_droid`, `pi05_droid`, `*_jointpos` variants | Franka/DROID | Single-arm |
| `pi0_aloha`, `pi05_aloha`, `pi0_aloha_towel`, `pi0_aloha_tupperware`, `pi0_aloha_pen_uncap`, `pi05_aloha_pen_uncap`, `pi0_aloha_sim` | **ALOHA (dual-arm Trossen ViperX)** | **Bimanual** |
| `pi0_ur5` | **UR5e** | Single-arm |
| `pi0_libero`, `pi0_fast_libero`, `pi05_libero`, LoRA variants | Franka (LIBERO) | Single-arm |

**Key finding:** OpenPI already has production-ready policy transforms and trained checkpoints for ALOHA (bimanual, 14-DoF) and UR5e (single-arm, 7-DoF). No humanoid policies exist in the codebase.

---

## 3. Gap Analysis: What Would Need to Change

### 3A. Dual-Arm (Bimanual) — e.g., ALOHA in IsaacLab

**Policy side (openpi): ✅ Ready.** `aloha_policy.py` handles 14-dim state/actions, 4 cameras, and joint-angle flipping between ALOHA and π₀ conventions.

**Sim side (robolab): ❌ Not supported.** Every layer assumes a single `{ENV_REGEX_NS}/robot`:

| Component | What needs to change | Effort |
|-----------|---------------------|--------|
| **Robot USD asset** | Need an ALOHA USD (two ViperX 300s on a shared base). IsaacLab community has ALOHA URDFs but no production USD in this repo. | Medium — import from URDF + tune collision/visual meshes |
| **Robot config** (`robots/aloha.py`) | New `AlohaCfg` class with two arm articulations (`left_arm`, `right_arm`), each with 6 joints + 1 gripper. Joint name regex patterns must differ per arm. | Medium |
| **Action config** | New `AlohaJointPositionActionCfg` with 14-dim actions split across two arm articulations + two grippers. IsaacLab's `JointPositionActionCfg` supports a single `asset_name` so you either: (a) model both arms as one articulation, or (b) create two action terms. | Medium |
| **Observation config** | State must concatenate both arms' joints/grippers (14-dim). Need 4 cameras (overhead, side, left-wrist, right-wrist). | Low |
| **Inference client** | New `AlohaJointposClient` that maps the 14-dim action chunk from π₀-ALOHA back to the two arms. Image extraction must feed 4 camera views. | Medium |
| **Environment registration** | New `auto_register_aloha_envs()` with the dual-arm configs above. | Low |
| **Tasks** | Existing pick-and-place tasks assume a single EE. Bimanual tasks (fold towel, open tupperware) need new task files with bimanual success criteria. | High |
| **Scene layout** | Table/fixture geometry changes — ALOHA has a wider workspace with the robot on a ~30 cm elevated base. | Low–Medium |

**Estimated total effort: 3–5 weeks** for a senior engineer (1 week sim asset, 1 week configs/actions/observations, 1 week inference client, 1–2 weeks bimanual tasks + debugging).

### 3B. Humanoid Robots

**Policy side (openpi): ❌ Not supported.** No humanoid policy transforms, no training configs, no norm stats.

**Sim side (robolab): ❌ Not supported.** The entire framework is built around table-top manipulation:

| Component | What needs to change | Effort |
|-----------|---------------------|--------|
| **Robot USD** | Need full-body humanoid USD (e.g., Unitree H1, Figure 01). IsaacLab has some humanoid assets in its Nucleus server, but locomotion + manipulation in a single articulation is complex. | High |
| **Robot config** | Must handle 20–50+ DoF: legs, torso, two arms, hands. The current `robot_cfg` mixin pattern can't easily mix locomotion + manipulation action spaces. | High |
| **Action space** | Fundamentally different — needs whole-body control (locomotion controller for legs, IK/joint-pos for arms). Would likely need a hierarchical controller or a single large joint-position vector. | High |
| **Observation space** | Head cameras, wrist cameras, IMU, foot contact sensors, full-body proprioception (50+ dims). | Medium |
| **Policy** | No existing π₀ humanoid checkpoint. Would need to either: (a) train a new model from scratch on humanoid data, or (b) use a separate humanoid policy (e.g., from NVIDIA Isaac GR00T Humanoid). | Very High |
| **Tasks** | All existing tasks are tabletop. Humanoid tasks involve navigation + manipulation (e.g., walk to shelf, pick object). Complete task redesign needed. | Very High |
| **Physics/sim** | Need ground plane, proper foot contact modeling, balance controllers. Current scenes are table-centric. | High |

**Estimated total effort: 3–6 months.** This is effectively a new project — the only reusable parts are the environment factory pattern and the task file format.

### 3C. Other Single-Arm Robots (UR5e, Kuka, xArm)

**Policy side (openpi):** ✅ UR5e ready (`ur5e_policy.py` + `pi0_ur5` config). Others would need custom policy transforms (straightforward — copy `ur5e_policy.py`, adjust DoF count).

**Sim side (robolab):** Low–medium effort — single-arm robots are architecturally identical to Franka:

| Component | What needs to change | Effort |
|-----------|---------------------|--------|
| **Robot USD** | Obtain/convert robot USD. IsaacLab Nucleus has UR10, Sawyer, Kuka KR210. For UR5e or xArm, import from URDF. | Low–Medium |
| **Robot config** | New `UR5eCfg` / `KukaCfg`. Same structure as `DroidCfg` — change joint names, DoF count, actuator limits, EE body name. | Low |
| **Action config** | Adjust `joint_names_expr` for the new robot. UR5e = 6 DoF + gripper; xArm = 6 or 7 DoF + gripper. | Low |
| **Gripper** | Depends on end-effector choice (Robotiq, OnRobot, etc.). Need USD and binary/continuous action mapping. | Low–Medium |
| **Inference client** | Minimal changes if action dim stays 7–8. For UR5e (6 joints + 1 gripper = 7 dims), adjust the slicing in `_extract_observation` and action binarization. | Low |
| **Tasks** | Existing tasks should mostly work — they define object placements relative to the robot origin, so just verify reachability with the new kinematic chain. | Low |

**Estimated total effort: 1–2 weeks** per robot.

---

## 4. Summary Table

| Robot Type | Robolab Sim | OpenPI Policy | Feasibility | Effort |
|-----------|:-----------:|:-------------:|:-----------:|:------:|
| **Franka Panda (DROID)** | ✅ Full | ✅ Full (pi0/pi05/FAST + OpenVLA + GR00T + DreamZero) | Production | — |
| **UR5e** | ❌ None | ✅ Policy transforms + train config | High | 1–2 weeks |
| **Other single-arm (Kuka, xArm, Sawyer)** | ❌ None | ⚠️ No policy transforms (easy to add) | High | 1–2 weeks each |
| **ALOHA (dual-arm bimanual)** | ❌ None | ✅ Full (pi0/pi05 + ALOHA data) | Medium | 3–5 weeks |
| **Humanoid** | ❌ None | ❌ None | Low | 3–6 months |

---

## 5. Architecture Notes for Extension

### The `robot_cfg` plug-in pattern is sound
The `EnvFactory.create_env_cfg()` takes `robot_cfg` as a parameter and uses it as a mixin base for the scene. Adding a new robot means:
1. Create `robots/new_robot.py` with a `@configclass` that defines `robot` (ArticulationCfg), optionally `wrist_cam`, `frames`, etc.
2. Create a matching `ActionCfg` class.
3. Create a matching `ProprioceptionObservationCfg`.
4. Write a new `auto_register_*_envs()` function (or pass the new cfg to the existing factory).

### What's **not** pluggable (hardcoded Franka assumptions)
- **Inference clients** (`policies/droid_jointpos/inference/`): All 6 clients hardcode `arm_joint_pos[7]` + `gripper_pos[1]` extraction and `action[-1]` binarization. A new robot with different DoF needs a parallel client or a refactored base class.
- **`delta_actions.py`**: Assumes a single EE frame and 7-dim (6 pose + 1 gripper) output. Would need extension for dual-arm.
- **Task success criteria**: Tasks reference `"robot"` as the single articulation name and use single-EE contact sensors. Bimanual tasks need dual-EE support in the termination/reward logic.

### Recommended first extension: UR5e
Lowest effort, highest policy readiness. OpenPI already has `pi0_ur5` with trained weights and norm stats. The sim-side work is purely mechanical — a new robot cfg file, a UR5e USD from IsaacLab Nucleus, and minor inference client tweaks.
