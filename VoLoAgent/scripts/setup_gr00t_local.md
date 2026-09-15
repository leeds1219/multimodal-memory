<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local GR00T setup for orchestrator testing

End-to-end test goal:

```
robolab eval client (--policy gr00t)
  ──ZMQ──▶ orchestrator (--frontend gr00t-zmq --backend gr00t-zmq)
            ──ZMQ──▶ GR00T server
```

## 1. Clone the GR00T fork robolab is wired against

```bash
cd ~
git clone --recurse-submodules \
    https://github.com/nadunRanawaka1/Isaac-GR00T-n16-droid.git
cd Isaac-GR00T-n16-droid
git checkout fa1fd91f4798e333b7cd1e9d5a32fe55f105a16b
```

## 2. Set up the uv venv (separate from your other envs)

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
uv venv .venv-gr00t --python 3.10
uv sync
uv pip install -e .
```

Confirm CUDA toolkit 12.4 is installed; `nvcc --version` should report 12.4.

## 3. Download the `oss-droid-v0` checkpoint

The checkpoint lives on NVIDIA OneDrive (URL in `robolab/README.md` §gr00t).
Unzip into `Isaac-GR00T-n16-droid/models/oss-droid-v0/`.

```bash
unzip oss-droid-v0.zip -d models/
ls models/oss-droid-v0/checkpoint-25000  # should contain pytorch_model.bin etc.
```

About 3 GB of weights.

## 4. Smoke-test the GR00T server alone

Terminal 1:

```bash
cd ~/Isaac-GR00T-n16-droid
source .venv-gr00t/bin/activate
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path models/oss-droid-v0/checkpoint-25000 \
    --embodiment-tag OXE_DROID_JOINT_POSITION_RELATIVE \
    --use-sim-policy-wrapper \
    --host 0.0.0.0 --port 5555
```

Wait ~30 s for model load.  Then in Terminal 2:

```bash
cd ~/vlm-orchestrator
conda activate vlm-orch
python -c "
import zmq
from vlm_orchestrator.protocols.gr00t_zmq import _gr00t_pack, _gr00t_unpack
ctx = zmq.Context()
s = ctx.socket(zmq.REQ); s.connect('tcp://127.0.0.1:5555')
s.send(_gr00t_pack({'endpoint': 'ping'}))
print('ping →', _gr00t_unpack(s.recv()))
"
```

Expected: `ping → {'ok': True}`.  If yes, the GR00T server is healthy.

## 5. Direct robolab → GR00T (bypass orchestrator) — establish baseline

Terminal 1: GR00T server still running.

Terminal 3:

```bash
conda activate robolab
cd ~/robolab
python policies/gr00t/run.py \
    --headless \
    --remote-host 127.0.0.1 --remote-port 5555 \
    --num-runs 1 \
    --task BananaInBowlTask \
    --output-folder-name smoke_gr00t_direct
```

Watch the run.  Note success/fail and per-step latency.  This is your
**baseline** for what GR00T-on-robolab can do without orchestration.

## 6. With orchestrator (passthrough mode) — sanity check the protocol

Terminal 1: GR00T server still running on :5555.

Terminal 2: orchestrator (note `--vla-port 5555`, `--port 8001`, both
ends in gr00t-zmq mode, **passthrough** strategy so we just proxy):

```bash
conda activate vlm-orch
cd ~/vlm-orchestrator
vlm-orchestrator \
    --frontend gr00t-zmq --backend gr00t-zmq \
    --vla-host 127.0.0.1 --vla-port 5555 \
    --port 8001 \
    --mode passthrough \
    --log-dir results/smoke_gr00t_orchestrated_passthrough \
    --verbose
```

Terminal 3:

```bash
conda activate robolab
cd ~/robolab
python policies/gr00t/run.py \
    --headless \
    --remote-host 127.0.0.1 --remote-port 8001 \
    --num-runs 1 \
    --task BananaInBowlTask \
    --output-folder-name smoke_gr00t_orchestrated_pass
```

Same task as step 5.  Success rate should match the baseline (the
orchestrator is just proxying).  If it differs significantly, the
schema translation has a bug — compare obs/action shapes in
`results/smoke_gr00t_orchestrated_passthrough/.../rewrites.jsonl`.

## 7. With strategies (subgoal + replan_grasp + front-camera, no timeout)

Same as step 6 but with the orchestration features enabled.  This is
the configuration `bench-vague-vlm-rg-fc-notimeout` uses.

Terminal 2 (replace orchestrator command):

```bash
vlm-orchestrator \
    --frontend gr00t-zmq --backend gr00t-zmq \
    --vla-host 127.0.0.1 --vla-port 5555 \
    --port 8001 \
    --mode subgoal \
    --failure-monitor vlm \
    --recovery-mode replan_grasp \
    --use-front-camera \
    --grasp-seg-mode gdino_sam2 \
    --subgoal-timeout 9999 \
    --log-dir results/smoke_gr00t_subgoal_replan_grasp \
    --verbose
```

Terminal 4 (grasp server, in graspgen env):

```bash
conda activate graspgen
cd ~/vlm-orchestrator
python vlm_orchestrator/grasp/server.py \
    --gripper-config ~/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml \
    --port 8003 --verbose --enable-sam2 --sam2-model facebook/sam2.1-hiera-small --enable-gdino
```

Terminal 3: same eval client command as step 6 (gr00t still uses port
8001 to talk to the orchestrator).

Watch `rewrites.jsonl` for `vlm_detect`, `subgoal_complete`, and
`grasp_tool` events.  This validates the strategy layer works through
the gr00t-zmq frontend.

## Known limitations of step 7 right now

- **`__step` not sent by robolab gr00t client yet.**  The orchestrator
  falls back to ``infer_count * 8`` for cadence, which over-estimates
  by 25 % for gr00t (chunks are 10).  Fix: add ``request_data["__step"]
  = self._episode_step`` and the matching counter increment in
  ``robolab/policies/.../gr00t.py`` (mirroring the change we made in
  ``pi0_family.py``).  ~5 lines.

- **Depth / camera pose / gt_state not forwarded by gr00t client.**
  Replan-grasp recovery needs depth.  Without it, the grasp tool falls
  back to instruction rewriting — a real test of grasp recovery needs
  the gr00t client extended to forward these obs, same way
  ``pi0_family.py`` does at lines 78-103.

- **GPU contention.**  Running GR00T (~14 GB) + IsaacSim
  (~6 GB) + GroundingDINO/SAM2 (~3 GB) on one A6000 is workable but
  tight.  Watch ``nvidia-smi``.
