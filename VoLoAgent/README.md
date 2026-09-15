<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VoLoAgent

**VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation**

[Project page](https://chicychen.github.io/VoLo/) · [arXiv](https://arxiv.org/abs/2606.07723) · [Benchmark tasks (RoboVoLo)](https://github.com/NVlabs/RoboVoLo)

A proxy between a robot eval client (`robolab`) and a VLA policy server (`openpi`) that uses a VLM to improve task execution through subgoal decomposition, tool-based grasp/place execution, and ground-truth failure detection with grasp-based recovery.

This repository holds the **orchestrator code**. The benchmark task pack — 126 RoboLab tasks, USD scenes, and assets used to evaluate VoLo — lives in the companion repository [**RoboVoLo**](https://github.com/NVlabs/RoboVoLo).

> **RoboLab requirement:** use [RoboLab v0.3.0](https://github.com/NVlabs/RoboLab/tree/v0.3.0), which includes native VoLo and `robovolo` content-pack support.

## Quick Start: Running Experiments

All experiments require three services running in separate terminals. Each service uses a different conda environment.

### Prerequisites

| Service | Conda env | Default port | Purpose |
|---------|-----------|:------------:|---------|
| VLA policy server (openpi) | `openpi` | 8000 | Runs the robot policy (Pi0.5, etc.) |
| VLM orchestrator proxy | `vlm-orch` | 8001 | Sits between robolab and VLA, applies strategies |
| Grasp server (optional) | `graspgen` | 8003 | GPU grasp generation for recovery actions |
| Robolab eval client | `robolab` | — | Runs Isaac Sim, connects to orchestrator |

### Step 1: Start the VLA policy server

```bash
conda activate openpi
# (see openpi docs for the specific launch command)
# Server must be listening on port 8000 before continuing
```

### Step 2: Start the orchestrator proxy

```bash
conda activate vlm-orch
cd ~/vlm-orchestrator

# Failure detection + recovery (subgoal decomposition + VLM monitor)
vlm-orchestrator --vla-port 8000 --port 8001 --mode subgoal \
    --failure-monitor vlm --recovery-mode replan_grasp \
    --log-dir ./results/my_experiment --verbose

# Passthrough mode (baseline, no orchestration)
vlm-orchestrator --vla-port 8000 --port 8001 --mode passthrough \
    --log-dir ./results/my_passthrough_experiment --verbose

# Tool-chain mode (VLM drives grasp/place tools, bypassing the VLA)
vlm-orchestrator --vla-port 8000 --port 8001 --mode tool_chain \
    --log-dir ./results/my_experiment --verbose
```

### Step 3: (Optional) Start the grasp server

Required when using `--recovery-mode grasp_first` or `--recovery-mode vlm_grasp`:

```bash
conda activate graspgen
cd ~/vlm-orchestrator
python vlm_orchestrator/grasp/server.py \
    --gripper-config ~/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml \
    --port 8003 --verbose --enable-sam2 --enable-gdino
```

### Step 4: Run the eval client

```bash
conda activate robolab
cd ~/RoboLab

# Single task, 10 episodes, with GT state export
python policies/volo/run.py \
    --policy pi05 \
    --remote-port 8001 \
    --task BlockStackingSpecifiedOrderTask \
    --num-runs 10 \
    --enable-subtask \
    --enable-gt-state \
    --video-mode all \
    --headless

# Multiple tasks, 3 episodes per task
python policies/volo/run.py \
    --policy pi05 \
    --remote-port 8001 \
    --task BlockStackingSpecifiedOrderTask CondimentsInBinTask \
    --num-runs 3 \
    --enable-subtask \
    --enable-gt-state
```

Key `policies/volo/run.py` flags:
- `--policy pi05`: select the policy backend behind the VoLo proxy
- `--remote-port 8001`: connect to the orchestrator (not the VLA directly)
- `--enable-gt-state`: export Isaac Sim ground-truth state for GT failure detection
- `--enable-subtask`: enable subtask progress tracking and scoring
- `--task <TaskName>`: one or more task class names (see the RoboLab task list)
- `--num-runs N`: sequential episodes per task when `--num-envs` is 1
- `--video-mode all`: save sensor and viewport videos
- `--headless`: no GUI display

> **Depth data requirement:** When using `--recovery-mode grasp_first` or
> `--recovery-mode vlm_grasp`, the eval client **must** forward depth images.
> Robolab forwards depth automatically (camera config includes depth).
> LIBERO eval clients require `--enable-depth`. VLABench and RoboCasa eval
> clients also accept `--enable-depth`. Without depth data, the grasp tool
> will fail with "No depth data in observation" and fall back to instruction
> rewriting, which produces degraded results.

### Verifying Services

Before launching an eval, check that all services are up:

```bash
# Check VLA server
curl -s http://localhost:8000/health 2>/dev/null || echo "VLA not running on 8000"

# Check orchestrator
curl -s http://localhost:8001/health 2>/dev/null || echo "Orchestrator not running on 8001"

# Check grasp server (if needed)
curl -s http://localhost:8003/health 2>/dev/null || echo "Grasp server not running on 8003"
```

## Strategies

The proxy supports multiple orchestration strategies, selected via `--mode`:

| Strategy | Description |
|----------|-------------|
| `subgoal` | VLM decomposes task into subgoals, feeds one at a time, VLM checks progress (**default**) |
| `tool_chain` | VLM picks `grasp(target)` / `place(destination)` tool calls per subgoal; tool actions bypass the VLA |
| `next_goal` | VLM predicts next step on-the-fly every N action chunks; no upfront decomposition |
| `passthrough` | No-op baseline — forwards requests unmodified |

Archived exploration modes (retained but no longer recommended):
- `rewrite` — VLM rewrites the instruction once at episode start
- `adaptive` — compare-and-pick: probes policy with original vs rewritten instruction, picks lower TD
- `scene_edit` — GDino detects target object, edits exterior image (highlight/dim)
- `subgoal_scene_edit` — subgoal decomposition + GDino scene editing

## Next-Goal Mode

`--mode next_goal` is an alternative to subgoal decomposition. Instead of planning the full task up front, the VLM is called periodically to predict only the *next* step, given the overall goal, a rolling history of past checkpoint images, and the current scene.

**How it works:**
1. At episode start, the VLM immediately predicts the first step.
2. Every `--check-interval` action chunks, the VLM is called again with updated scene image.
3. The predicted instruction replaces the current VLA prompt and the action chunk is flushed so the policy re-infers with the new instruction.
4. The current scene image is added to a rolling history (capped at `--next-goal-max-history` images).

**VLM backend support:** Works with any OpenAI-compatible API. Set the endpoint and
model via `--vlm-base-url` / `--vlm-model` (defaults are the placeholders
`https://YOUR_VLM_ENDPOINT/v1` and `YOUR_VLM_MODEL`). Any vision-capable,
OpenAI-compatible model can be used.

> **API URL note:** The base URL must point at an OpenAI-compatible `/v1`
> endpoint. Pointing at the wrong path typically causes a 404 error.

**Incompatibilities:** `--mode next_goal` does not support `--failure-monitor` or `--recovery-mode`.

```bash
# Any OpenAI-compatible vision model
vlm-orchestrator --vla-port 8000 --port 8001 --mode next_goal \
    --vlm-model YOUR_VLM_MODEL \
    --vlm-base-url https://YOUR_VLM_ENDPOINT/v1 \
    --use-front-camera --verbose
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--check-interval N` | 10 | Action chunks between VLM predictions |
| `--next-goal-template` | `v1_freeform` | Prompt template: `v1_freeform` (CoT free-text), `v1` (simpler), `v2`/`mcq` (MCQ, requires `--next-goal-task-type`) |
| `--next-goal-task-type` | — | Closed-vocab task type for MCQ templates (`i3_closed_vocab`, `i4_closed_vocab`, `h8_closed_vocab`) |
| `--next-goal-max-history` | 5 | Max checkpoint images kept in rolling history (0 = unlimited) |

**Output:**

Each episode directory gains a `vlm_calls.jsonl` alongside `rewrites.jsonl`:
- **`rewrites.jsonl`**: one `next_goal_episode_start` entry (first VLM call) + `next_goal_check` entries (subsequent calls). Each entry contains `instruction`, `vlm_raw`, `vlm_latency_s`, `history_len`, `used_fallback`, `step_count`.
- **`vlm_calls.jsonl`**: full per-call log with prompt text, image count, complete VLM response, input/output token counts, and latency. Written by the strategy.

## Tool-Chain Mode

`--mode tool_chain` skips the VLA policy for manipulation actions. Instead the VLM decomposes the task into subgoals and, for each subgoal, picks a sequence of `grasp(target)` / `place(destination)` tool calls. Each tool is executed by the orchestrator itself: perception runs on the grasp server (SAM3 / SAM2 / GDinoV2 / Molmo2), grasp poses come from GraspGen (Contact-GraspNet head), placement uses a free-form 2D point (Claude or Molmo2) projected to 3D via depth, and motion is generated by the selected motion planner (cuRobo by default; see below) + Cartesian trajectory.

> **These grasp/place features are shared.** The motion planner, stack mode, front-camera routing, and the internal fixes (front-camera depth-resolution projection, Robotiq place gripper-depth) live inside the grasp/place tool executors, so they apply to **any mode that runs the orchestrator's own grasp/place pipeline** — both `--mode tool_chain` and `--mode subgoal` with a tool-using recovery mode (`--recovery-mode replan_tools` / `replan_grasp` / `grasp` / `grasp_first` / `vlm_grasp` / `tools_first`). The flags below are documented here but are not tool_chain-only.

**Motion planner (`--motion-planner`, default `curobo`):** the grasp/place approach is planned by collision-aware cuRobo motion generation. This requires the grasp server launched with `--enable-curobo`; otherwise `/plan_motion` returns 503 and grasps **loud-fail** (no silent fallback). Use `--motion-planner linear` for the historical straight-line joint-space interpolation (no collision checking).

**Stack mode (`--enable-stack-mode`, default ON; `--no-enable-stack-mode` to disable):** the VLM sets an optional `stack` arg on `grasp()`/`place()`. `stack: true` preserves the held object's grasp orientation on release (stacking blocks, nesting lids) and keeps the top-down grasp filter; `stack: false` (default when omitted) uses a top-down release for simple pick-and-place / drop-into-bin and disables the top-down grasp filter (keeps all GraspGen candidates). Set `stack` consistently on the matching `grasp()`/`place()` pair. `--no-enable-stack-mode` restores historical grasp-consistent placement and ignores the `stack` arg. A/B override: `STACK_FORCE={true,false}` env-var. Local A/B harness: `scripts/run_stack_ab_local.sh`.

**Front camera (`--use-front-camera`):** routes VLM grasp/place perception through the top-down `egocentric_mirrored_camera`; the VLA still sees the exterior camera. Point-cloud projection uses the front camera's depth resolution — the grasp tool raises (no silent exterior-camera fallback) if front depth is present but front RGB is missing.

**How it works:**
1. At episode start the VLM is asked for the full subgoal list (same prompt as `subgoal` mode).
2. For each subgoal the VLM is asked which tool to call next (`grasp(target_phrase)` / `place(destination_phrase)` / `next_subgoal`).
3. The orchestrator runs the tool against the live observation and forwards the resulting motion to the policy server as a chunked action stream — the policy server still runs but its outputs are overridden when a tool is active.
4. Subgoal advances when the VLM emits `next_subgoal` or its hard cap is hit.

**Required services:** grasp server with whichever perception backends the chosen seg modes need:
- `--grasp-seg-mode sam3` (default) → `--enable-sam3`
- `--grasp-seg-mode gdino_sam2` → `--enable-gdino --enable-sam2`
- `--grasp-seg-mode molmo_sam2` → `--enable-sam2 --enable-molmo`
- `--place-seg-mode molmo_point` (default) → `--enable-molmo`
- `--place-seg-mode vlm_point` → no extra (uses the orchestrator's main VLM)

The grasp server's `--enable-molmo` subprocess-launches `vlm_orchestrator/utils/molmo2_hf_server.py` in the `molmo-env` conda env (Molmo2-8B needs `transformers ≥ 4.55` which the GraspGen env can't host). Override via `--molmo-env`, `--molmo-model`, `--molmo-port`, `--molmo-quantize {bf16,int8,int4}`.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--grasp-seg-mode` | `sam3` | Segmentation backend for grasp: `sam3` / `gdino_sam2` / `molmo_sam2` / `gt_sim` |
| `--place-seg-mode` | `molmo_point` | Destination grounding: `molmo_point` / `vlm_point` / `sam3` / `gdino_sam2` / `gt_sim` |
| `--grasp-topdown-threshold` | 0.85 | Strict-top-down filter on Contact-GraspNet candidates (dot-product cutoff vs. gravity; 0.0 = keep all, 1.0 = perfectly vertical). Bypassed for a `stack: false` grasp under stack mode |
| `--motion-planner` | `curobo` | Approach motion planner: `curobo` (collision-aware, needs server `--enable-curobo`) / `linear` (straight-line, no collision) |
| `--enable-stack-mode` / `--no-enable-stack-mode` | ON | Honour the VLM's per-call `stack` arg (true=preserve grasp orientation + top-down filter; false=top-down release + no filter). `--no-` restores historical grasp-consistent placement |
| `--use-front-camera` | off | Route grasp/place VLM perception through the top-down front camera (VLA still sees exterior) |
| `--tool-chain-max-tools-per-subgoal` | 5 | Hard cap before forcing replan (avoids plan-thrashing) |
| `--tool-chain-max-tools-per-episode` | 30 | Hard cap before aborting (bounds runtime / VLM cost) |

**Local launch (all-in-one):** `scripts/run_tool_chain_molmo_local.sh` spins up VLA stub, grasp server (SAM2 + Molmo2 sidecar), orchestrator, and robolab eval in a single box. Per-service logs land in `$ORCH_LOG_DIR/_services/`.

**SAM3 + Molmo2 grasp server:** the SAM3-grasp + Molmo2-place tool_chain run uses the `Dockerfile.grasp-server-sam3-molmo2` image (prebuilt at `ghcr.io/chicychen/grasp-server-sam3-molmo2:latest`), which extends the base grasp-server image with SAM3 + accelerate + tensorflow-cpu + `transformers==4.57.1` + `huggingface-hub==0.34.4` (the last two pinned because Molmo2's `processing_molmo2.py` passes kwargs that transformers 5.x rejects).

```bash
# Manual launch (3 terminals)
# Terminal 1: VLA stub (tool_chain doesn't need a real policy)
python vlm_orchestrator/utils/vla_stub_server.py --port 8000

# Terminal 2: grasp server with SAM3 + Molmo2 sidecar
conda activate graspgen
python -m vlm_orchestrator.grasp.server \
    --gripper-config $HOME/graspgen/models/checkpoints/graspgen_franka_panda.yml \
    --port 8003 \
    --enable-sam3 --sam3-model facebook/sam3 \
    --enable-molmo --molmo-model allenai/Molmo2-8B \
    --molmo-port 8122 --molmo-quantize bf16 \
    --enable-curobo   # required for --motion-planner curobo (the default)

# Terminal 3: orchestrator
export GRASP_SERVER_HOST=127.0.0.1 GRASP_SERVER_PORT=8003
export MOLMO_BASE_URL=http://127.0.0.1:8122/v1 MOLMO_MODEL=allenai/Molmo2-8B
vlm-orchestrator --vla-host 127.0.0.1 --vla-port 8000 --port 8001 \
    --mode tool_chain \
    --vlm-model YOUR_VLM_MODEL \
    --grasp-seg-mode sam3 --place-seg-mode molmo_point \
    --grasp-topdown-threshold 0.85 --verbose
```

**Output:** standard `<task>/episode_<N>/{rewrites.jsonl, metadata.json, task_failures.jsonl}`. Each episode also gets per-tool diagnostic dirs `debug_grasp/` (detection / mask / depth / grasp-pose / IK images, plus `grasp_log.json`) and `debug_place/` (destination point / projected world target / post-place EE delta, plus `place_log.json`).

## Failure Detection and Recovery

`--failure-monitor` (requires `--mode subgoal`) watches execution and triggers recovery when a subgoal stalls or fails. The VLM model used for detection is set globally by `--vlm-model`.

### Failure Monitor

**`--failure-monitor vlm`** is the recommended monitor: periodic VLM checks handle both failure detection and subgoal completion. It works on any benchmark (no ground-truth physics state required).

| `--failure-monitor` | Description | Compatible `--recovery-mode` (default first) |
|---------------------|-------------|----------------------------------------------|
| `vlm` | Periodic VLM checks for failure detection and subgoal completion | `replan`, `replan_grasp`, `replan_tools`, `grasp` |

### Recovery Modes

For the `vlm` monitor, `--recovery-mode` controls what actions the VLM may take on a detected failure:

| `--recovery-mode` | Description |
|-------------------|-------------|
| `replan` | VLM context-aware instruction rewrite / replan (default, ~5s) |
| `replan_grasp` | VLM decides: replan OR activate the grasp tool (~5s) |
| `replan_tools` | VLM replans and can call the orchestrator's own grasp/place tool pipeline (see Tool-Chain Mode) |
| `grasp` | Activate the grasp tool on VLM decision |

### Legacy / experimental monitors

<details>
<summary>Signal-based and ground-truth (GT) monitors — retained but not recommended</summary>

These were earlier explorations. They remain functional but are not recommended for general use; **prefer `--failure-monitor vlm`**.

**Signal-based** — action/EE heuristics, no VLM cost (or VLM only to confirm). Recovery modes: `template` (default), `retry`, `vlm`, `vlm_grasp`.

| `--failure-monitor` | Description |
|---------------------|-------------|
| `signal_primary` | Action/EE signal heuristics only (no VLM cost) |
| `union_failure` | Maximizes recall (signal ∪ VLM) |
| `intersect_video` | Signal + video VLM confirmation |

**Ground-truth (GT)** — requires `--enable-gt-state` from the eval client (Isaac Sim only). Uses known object poses / CSM physics conditions. Recovery modes: `grasp_first` (default), `place_first`, `tools_first`.

| `--failure-monitor` | Recovery decision | Pauses robot? |
|---------------------|-------------------|:-------------:|
| `gt` | Auto rule-based (grasp → retry → skip) | No |
| `gt_hitl` | Human decides in browser UI | Yes |
| `gt_vlm` | VLM with GT context + image | No |

> ⚠ **GT-monitor caveat:** the GT detector's rules and thresholds were calibrated on the **block-stack** suite. They have not been validated on Memory (LH) / Common-Sense (LH-CS) suites — object-contact heuristics may misfire on non-block geometries and the `SUBGOAL_COMPLETE` rule depends on container/positional predicates not authored for every task.

**GT failure types:**

| Type | Detection source | Auto recovery |
|------|-----------------|---------------|
| `WRONG_OBJECT_PICKED` | robolab `grasped_object` contact sensor | Grasp correct target |
| `OBJECT_DROPPED` | CSM condition: grabbed True→False, not in container | Re-grasp dropped object |
| `NO_PROGRESS` | `subtask.score` stalled for 30 steps | Grasp first remaining target |
| `SUBTASK_REGRESSION` | `all_subtask_conditions` True→False (confirmed over 5 steps) | VLM replan with GT context |
| `SUBGOAL_COMPLETE` | All target objects completed in CSM | Advance to next subgoal |

**Subtask regression detection:** when a previously completed physics condition becomes unsatisfied (e.g., robot bumps a stacked block off), the detector (1) confirms the regression persists for 5 consecutive steps to filter transient flickers, (2) triggers a VLM replan with GT context injected into the prompt (which objects regressed, what the robot is holding, current subgoal), and (3) produces a new subgoal plan from the actual current state.

</details>

### Front Camera Mode (`--use-front-camera`)

The simulator has a top-down front camera (`egocentric_mirrored_camera`) that provides a much better view of the workspace than the oblique exterior camera. When `--use-front-camera` is enabled:

- **VLM calls** (planning, recycle/replan, subgoal checks) use the front camera image — clearer spatial relationships, easier to distinguish stacked vs. adjacent objects
- **Grasp tool** (GDino detection, SAM2 segmentation, 3D grasp pose) uses the front camera RGB + depth — less perspective distortion, less arm occlusion
- **Annotated video** (`*_annotated.mp4`) shows the front camera with text overlays — matches what the VLM sees
- **Policy** (pi0/pi0.5) still receives the exterior camera it was trained on — no impact on policy behavior

```bash
vlm-orchestrator --mode subgoal --failure-monitor vlm \
    --recovery-mode replan_grasp --use-front-camera --verbose
```

### HITL (Human-in-the-Loop)

Add `--hitl --hitl-port 8002` to any subgoal mode. Opens a browser UI at `http://localhost:8002` where a human can monitor, pause, rewrite instructions, and provide recovery decisions.

## Architecture

```
                          ┌─────────────────────────────────────────────────────┐
                          │              VLM Orchestrator (port 8001)           │
                          │                                                     │
                          │  proxy.py ─── strategy (passthrough/subgoal/etc.)  │
                          │                  │                                  │
                          │    detection/gt_detector.py  ← obs["gt_state"]     │
                          │    grasp/tool.py → grasp/server.py (port 8003)     │
                          │    hitl.py → browser UI (port 8002)                │
                          │                                                     │
robolab (Isaac Sim) ──ws──▶  recv obs ──▶ strategy.process() ──ws──▶  VLA (8000)
                          │                                                     │
                          └─────────────────────────────────────────────────────┘
```

### Source Layout

```
vlm_orchestrator/
├── cli.py                    # CLI entry point (all flags documented here)
├── proxy.py                  # WebSocket proxy between robolab and VLA
├── vlm.py                    # VLM backends (OpenAI-compatible)
├── codec.py                  # msgpack-numpy protocol (openpi compatible)
├── hitl.py                   # Human-in-the-loop browser UI
├── image_edit.py             # Image editing (highlight, dim, compose)
├── policy_prober.py          # Policy probing for TD computation
├── bench.py                  # Benchmark task definitions
├── video_annotator.py        # Annotated video writer
├── lerobot_converter.py      # Convert trajectories to LeRobot v3 format
├── trajectory_collector.py   # DAgger-style data collection
│
├── grasp/                    # Grasp-based recovery pipeline
│   ├── tool.py               #   GraspToolExecutor (planned grasping)
│   ├── client.py             #   HTTP client for grasp server
│   ├── server.py             #   HTTP server for GPU grasp generation (separate env)
│   ├── debug.py              #   Debug visualization
│   ├── camera.py             #   Camera intrinsics/extrinsics utilities
│   ├── ik.py                 #   Franka FK/IK for grasp execution
│   ├── sam3.py               #   SAM3 detection + segmentation
│   └── gdino.py              #   GroundingDINO object detection
│
├── detection/                # Failure detection and GT segmentation
│   ├── signals.py            #   Action-based signal detector (EE heuristics)
│   ├── recovery.py           #   Recovery action generation
│   ├── gt_detector.py        #   GT object-level failure detection from sim state
│   └── gt_segmentation.py    #   GT segmentation providers per simulator
│
├── failure_handlers/         # Unified failure handler abstraction
│   ├── base.py               #   FailureHandler ABC + HandlerResult
│   ├── vlm.py                #   VLM-based failure detection + action selection
│   ├── gt.py                 #   GT handler (thin wrapper)
│   └── signal.py             #   Signal handler (thin wrapper)
│
├── benchmarks/               # External benchmark support
│   ├── libero_gt.py          #   LIBERO ground-truth state exporter
│   ├── robocasa_gt.py        #   RoboCasa ground-truth state exporter
│   ├── vlabench_gt.py        #   VLABench ground-truth state exporter
│   └── libero_specs.py       #   LIBERO-Plus / LIBERO-PRO benchmark specs
│
├── strategies/               # Orchestration strategies
│   ├── base.py               #   Base strategy + StrategyContext
│   ├── passthrough.py        #   No-op passthrough
│   ├── subgoal_base.py       #   Shared base for subgoal strategies
│   ├── subgoal.py            #   Subgoal decomposition (default)
│   ├── next_goal.py          #   On-the-fly next-step prediction
│   ├── tool_chain.py         #   VLM-driven grasp/place tool calls
│   └── archive/              #   Archived exploration modes
│       ├── rewrite.py        #     Single-shot VLM rewrite
│       ├── adaptive.py       #     Adaptive compare-and-pick (TD-based)
│       ├── image_edit.py     #     Image-edit helpers
│       ├── scene_edit.py     #     Scene-edit-only
│       └── subgoal_scene_edit.py  # Subgoal + scene edit
│
├── metrics/                  # Pluggable metric framework (td, action_variance, etc.)
└── diagnostics/              # Internal policy probing
```

### Robolab-Side Components

The GT state export lives in robolab (not this repo):

| File | Description |
|------|-------------|
| `robolab/core/events/gt_state_exporter.py` | Packs sim state (object poses, CSM conditions, contact sensor) into `obs["gt_state"]` |
| `robolab/core/events/subtask_recorder.py` | Subtask progress tracking via `SubtaskStateMachine` |
| `robolab/core/task/subtask_state_machine.py` | Sequential subtask orchestration and scoring |
| `robolab/core/task/conditionals_state_machine.py` | Per-object condition tracking (grab → lift → drop → in_container) |
| `policies/volo/run.py` | VoLo eval entry point that connects to the orchestrator |
| `robolab/eval/episode.py` | Single episode runner (policy inference loop) |

## Setup

Install the pinned RoboLab release and follow its environment setup instructions. Install [RoboVoLo](https://github.com/NVlabs/RoboVoLo) into that checkout when running the companion benchmark tasks.

```bash
git clone --branch v0.3.0 --depth 1 https://github.com/NVlabs/RoboLab.git
# Follow RoboLab's installation instructions, then install RoboVoLo if needed.
```

Install the VoLoAgent environment separately:

```bash
conda create -n vlm-orch python=3.11 -y
conda activate vlm-orch
pip install -e .
```

Set API key (required for VLM calls in non-passthrough modes):
```bash
export VLM_API_KEY="..."   # API key for your OpenAI-compatible VLM endpoint
```

The orchestrator talks to an OpenAI-compatible endpoint. Configure it with
`--vlm-base-url` (default placeholder `https://YOUR_VLM_ENDPOINT/v1`) and
`--vlm-model` (default placeholder `YOUR_VLM_MODEL`), and provide the key via
`--vlm-api-key` or the `VLM_API_KEY` env var. The `OPENAI_API_KEY` env var is
checked as a fallback if `VLM_API_KEY` is not set.

## Output Structure

### Orchestrator output (`--log-dir`)

```
results/<experiment_name>/
├── <instruction_slug>/
│   └── episode_<N>/
│       ├── metadata.json     # Episode config, subgoals, timestamps
│       ├── rewrites.jsonl    # All VLM calls, GT failures, checks, advances
│       ├── vlm_calls.jsonl   # Per-call VLM log (next_goal mode; all backends)
│       └── *.mp4             # Symlinked from robolab output
└── proxy.log
```

### Robolab output (`~/robolab/output/<experiment>/`)

```
<TaskName>/
├── episode_results.json      # Per-episode success/score/reason/metrics
├── data.hdf5                 # Recorded trajectory data
├── env_cfg.json              # Environment config
├── log_<N>.json              # Per-step subtask status (score, conditions, events)
└── <instruction>_<N>.mp4     # Episode videos (obs, viewport, annotated)
```

### Key output files

- **`rewrites.jsonl`**: Every VLM call, GT failure detection event, grasp escalation, and subgoal advance. Primary file for analyzing orchestrator behavior. In `next_goal` mode, entries are typed `next_goal_episode_start` and `next_goal_check`.
- **`vlm_calls.jsonl`**: Full per-call VLM log written by `next_goal` mode. Contains prompt text, image count, complete response, input/output token counts, and latency.
- **`log_<N>.json`**: Array of per-step dicts with `{status, completed, total, info, score, all_status_codes}`. Note: the `score` field is currently always 0.0 due to a known issue where IsaacLab's `info["log"]` is only populated during `env.reset()`, not `env.step()`. Use `episode_results.json` for final scores.
- **`episode_results.json`**: Final per-episode results with `success`, `score`, `reason`, `duration`, trajectory metrics, and event counts.

## Example Experiment Recipes

### GT failure detection eval (block stacking, 10 episodes)

```bash
# Terminal 1: VLA server (openpi env, port 8000)
# Terminal 2: Grasp server
conda activate graspgen
python vlm_orchestrator/grasp/server.py \
    --gripper-config ~/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml \
    --port 8003 --verbose --enable-sam2 --enable-gdino

# Terminal 3: Orchestrator (with front camera for better VLM + grasp)
conda activate vlm-orch
vlm-orchestrator --vla-port 8000 --port 8001 --mode subgoal \
    --failure-monitor gt --recovery-mode grasp_first \
    --use-front-camera \
    --log-dir ./results/gt_eval_blockstacking --verbose

# Terminal 4: Eval client
conda activate robolab
cd ~/RoboLab
python policies/volo/run.py --policy pi05 --remote-port 8001 \
    --task BlockStackingSpecifiedOrderTask \
    --num-runs 10 --enable-subtask --enable-gt-state \
    --video-mode all --headless
```

### Passthrough baseline (same task, 10 episodes)

```bash
# Terminal 1: VLA server (same as above)
# Terminal 2: Orchestrator (no grasp server needed)
conda activate vlm-orch
vlm-orchestrator --vla-port 8000 --port 8001 --mode passthrough \
    --log-dir ./results/passthrough_blockstacking --verbose

# Terminal 3: Eval client
conda activate robolab
cd ~/RoboLab
python policies/volo/run.py --policy pi05 --remote-port 8001 \
    --task BlockStackingSpecifiedOrderTask \
    --num-runs 10 --enable-subtask \
    --video-mode all --headless
```

### Multi-task eval with subgoal decomposition

```bash
conda activate vlm-orch
vlm-orchestrator --vla-port 8000 --port 8001 --mode subgoal \
    --failure-monitor vlm --recovery-mode replan_grasp \
    --log-dir ./results/subgoal_eval --verbose

conda activate robolab
cd ~/RoboLab
python policies/volo/run.py --policy pi05 --remote-port 8001 \
    --task ToolOrganizationBothTask SpoonsInPotTask FruitsOnPlateTask \
    --num-runs 3 --enable-subtask --enable-gt-state \
    --video-mode all --headless
```

### Next-goal mode with Claude (API)

```bash
# Terminal 1: VLA server
# Terminal 2: Orchestrator
conda activate vlm-orch
vlm-orchestrator --vla-port 8000 --port 8001 --mode next_goal \
    --vlm-model YOUR_VLM_MODEL \
    --vlm-base-url https://YOUR_VLM_ENDPOINT/v1 \
    --use-front-camera \
    --log-dir ./results/nextgoal_claude --verbose

# Terminal 3: Eval client (no --enable-gt-state required)
conda activate robolab
cd ~/RoboLab
python policies/volo/run.py --policy pi05 --remote-port 8001 \
    --task BlockStack3BlueRedGreenTask \
    --num-runs 3 --enable-subtask --video-mode all --headless
```

### HITL (human-in-the-loop) session

```bash
conda activate vlm-orch
vlm-orchestrator --vla-port 8000 --port 8001 --mode subgoal \
    --hitl --hitl-port 8002 --failure-monitor gt_hitl --verbose
# Open http://localhost:8002 in browser to monitor and intervene
```

## Alternative VLA backends

Beyond π0/π0.5 over the openpi WebSocket protocol, the orchestrator supports
third-party VLAs:
- **Cosmos3** (`nvidia/Cosmos3-Nano-Policy-DROID`) via `--client-protocol cosmos3`,
  with the option of using the Cosmos3-Nano reasoner as the VLM brain.
- **DreamZero** (`GEAR-Dreams/DreamZero-DROID`) via `--client-protocol dreamzero`.

## External Benchmark Evaluations

Beyond the primary RoboLab / Isaac Sim pipeline, the orchestrator can also be
evaluated on external manipulation benchmarks — **LIBERO** (and the LIBERO-PRO,
LIBERO-Plus, LIBERO-Mem variants), **RoboCasa**, and **VLABench**. Each has its
own eval client in [`examples/`](examples/), conda environment, and checkpoint,
all routing through the same orchestrator proxy.

Setup and run instructions for these environments live in
[`examples/README.md`](examples/README.md).

## Protocol

Fully compatible with the openpi WebSocket + msgpack-numpy protocol. No changes needed in robolab or the policy server — just redirect the port.

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026volo,
  title   = {VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation},
  author  = {Chen, Siyi and Hadfield, Hugo and Zook, Alex and Uy, Mikaela Angelina and Song, Chan Hee and Coumans, Erwin and Yang, Xuning and Ladhak, Faisal and Qu, Qing and Birchfield, Stan and Tremblay, Jonathan and Blukis, Valts},
  journal = {arXiv preprint arXiv:2606.07723},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.07723}
}
```

Project page: https://chicychen.github.io/VoLo/

## Related Projects

- [**RoboVoLo**](https://github.com/NVlabs/RoboVoLo) — the benchmark task pack
  (126 RoboLab tasks, USD scenes, and assets) used to evaluate VoLo.
- [**SpaceTools**](https://spacetools.github.io/) (CVPR 2026) — Tool-Augmented
  Spatial Reasoning via Double Interactive RL, which equips VLMs with vision and
  robotic tools for spatial reasoning and real-world manipulation.

## License

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

VoLo is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE)
for copyright and third-party attribution notices.

## Contributing

External contributions are welcome. All contributions must be made under the
Apache License 2.0 and signed off in accordance with the Developer Certificate
of Origin (DCO). See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution
process and DCO instructions.
