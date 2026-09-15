<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local OpenVLA setup for orchestrator testing

End-to-end test goal:

```
robolab eval client (--policy openvla)
  ──HTTP/REST──▶ orchestrator (--frontend openvla-rest --backend openvla-rest)
                  ──HTTP/REST──▶ OpenVLA deploy.py server
```

OpenVLA is fully open-source on HuggingFace — no SharePoint, no NDA, no
custom CUDA toolkit version pinning.  Setup is materially simpler than
GR00T.

## 1. Clone the OpenVLA repo

```bash
cd ~
git clone https://github.com/openvla/openvla.git
cd openvla
```

## 2. Set up an env

OpenVLA's pyproject pins to PyTorch 2.2 and Python 3.10.  Create a
fresh conda env so it doesn't fight your other envs.

```bash
conda create -n openvla python=3.10 -y
conda activate openvla
pip install -e .
pip install packaging ninja
# Flash-Attn build is slow (~5-10 min) but recommended for inference speed:
pip install "flash-attn==2.5.5" --no-build-isolation
```

If `flash-attn` build fails (it's notoriously brittle), it's optional —
remove the dep from `pyproject.toml` and skip.  Inference will be
~2× slower but functionally correct.

## 3. Install the deploy server's extra deps

The deploy script needs `json_numpy` and `flask`/`uvicorn`:

```bash
pip install json_numpy flask draccus
```

## 4. Smoke-test the OpenVLA server alone

The repo ships `vla-scripts/deploy.py`.  First run downloads
`openvla/openvla-7b` weights from HuggingFace (~14 GB) to
`~/.cache/huggingface/hub/`, then loads to GPU.  ~10 min first time;
subsequent runs ~30 s.

Terminal 1:

```bash
cd ~/openvla
conda activate openvla
python vla-scripts/deploy.py \
    --openvla_path openvla/openvla-7b \
    --host 0.0.0.0 \
    --port 8000
```

Wait for `* Running on http://0.0.0.0:8000` (or similar, depending on
deploy.py's framework).  Then in Terminal 2:

```bash
cd ~/vlm-orchestrator
conda activate vlm-orch
python -c "
import json, json_numpy, requests
import numpy as np
json_numpy.patch()
rgb = np.zeros((256, 256, 3), dtype=np.uint8)
body = {'image': rgb, 'instruction': 'pick up the block', 'unnorm_key': 'bridge_orig'}
r = requests.post('http://127.0.0.1:8000/act', data=json.dumps(body), timeout=30)
print('status:', r.status_code)
print('action:', np.asarray(r.json()))
"
```

Expected: an action array of length 7 (rough garbage values — the model
hasn't seen the actual scene, just blank input — but the request shape
is verified).  If you see an error about `unnorm_key`, try
`droid` or one of the keys printed in deploy.py logs at startup.

## 5. Validate the OpenVLA server directly

RoboLab v0.3.0 does not ship an OpenVLA policy runner. Validate this backend with an OpenVLA-native client or the repository's protocol smoke test instead:

```bash
python scripts/smoke_openvla_local.py
```

The smoke test validates request and action translation without Isaac Sim. OpenVLA was trained on real DROID + Bridge data, so even a compatible simulator client would have a large visual domain gap; protocol validation does not imply task success.

## 6. With orchestrator (passthrough mode) — verify protocol path

Terminal 1: OpenVLA server still on :8000.

Terminal 2: orchestrator with both ends in openvla-rest mode,
**passthrough** strategy:

```bash
conda activate vlm-orch
cd ~/vlm-orchestrator
vlm-orchestrator \
    --frontend openvla-rest --backend openvla-rest \
    --vla-host 127.0.0.1 --vla-port 8000 \
    --port 8001 \
    --openvla-unnorm-key bridge_orig \
    --mode passthrough \
    --log-dir results/smoke_openvla_orchestrated_passthrough \
    --verbose
```

Use an OpenVLA-native eval client to send REST requests to port 8001. RoboLab v0.3.0 cannot be used for this step because it has no OpenVLA policy runner. In passthrough mode, per-step actions should match direct server requests apart from the orchestrator hop.

## 7. With strategies (subgoal + replan-style only — no replan_grasp)

Terminal 2 (replace orchestrator command):

```bash
vlm-orchestrator \
    --frontend openvla-rest --backend openvla-rest \
    --vla-host 127.0.0.1 --vla-port 8000 \
    --port 8001 \
    --openvla-unnorm-key bridge_orig \
    --mode subgoal \
    --failure-monitor vlm \
    --recovery-mode replan \
    --subgoal-timeout 9999 \
    --log-dir results/smoke_openvla_subgoal_replan \
    --verbose
```

Note: `--recovery-mode replan_grasp` won't work cleanly with OpenVLA
because:
  - OpenVLA doesn't send depth or camera pose in its payload, so the
    grasp tool can't build a point cloud.
  - OpenVLA returns 1 action/step; the grasp tool's 8-step chunks get
    truncated to 1 by the frontend.

Stick to `--recovery-mode replan` (VLM-driven instruction rewrite, no
grasp tool) for OpenVLA experiments.

## Known limitations

1. **No proprioception in OpenVLA payload.**  OpenVLA only sends image
   + instruction.  The frontend synthesizes zero state arrays for
   ``observation/joint_position``, ``observation/gripper_position``,
   ``observation/ee_pos``, and identity quaternion for
   ``observation/ee_quat``.  Failure handlers/strategies that look at
   these will see static zeros — typically harmless for the VLM
   failure handler we use, but worth being aware of.

2. **Single image, no wrist.**  OpenVLA sends only the exterior
   camera.  The frontend stubs the wrist field with the same image so
   strategies that index ``observation/wrist_image_left`` don't error,
   but it's not a real wrist view.

3. **Domain gap.**  OpenVLA was trained on real-world Bridge + DROID
   data.  Performance on IsaacSim's rendered tabletop scenes will be
   limited.  This setup is for *protocol validation*, not for getting
   competitive success rates.

4. **`unnorm_key` matters.**  Each dataset OpenVLA was trained on has
   its own action normalization.  Common values: ``bridge_orig``,
   ``droid``, ``droid_wipe``,
   ``nyu_franka_play_dataset_converted_externally_to_rlds``.  Wrong
   key → actions decoded with the wrong scale → robot moves
   nonsensically.  Check deploy.py logs at startup to see which keys
   the loaded model supports.
