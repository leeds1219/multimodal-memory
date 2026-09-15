<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# External Benchmark Evaluation Clients

The primary VoLo pipeline runs on RoboLab / Isaac Sim (see the
[main README](../README.md)). In addition, the orchestrator can be evaluated on
several **external manipulation benchmarks**. Each benchmark has its own eval
client in this directory, its own conda environment, and its own model
checkpoint, but they all route through the **same orchestrator proxy**.

This document collects the setup and run instructions for those external
environments. It is supplementary to the main README — refer there first for the
core orchestrator usage.

All benchmarks route through the **same orchestrator proxy** on port 8019.
The general pattern is always:

```
┌────────────┐    ┌──────────────┐    ┌────────────────────┐
│  Eval       │ws  │  Orchestrator │ws  │  VLA Policy Server │
│  Client     │───▶│  Proxy (8019) │───▶│  (8002)            │
│  (benchmark)│    │  --env libero │    │  (openpi)          │
└────────────┘    └──────────────┘    └────────────────────┘
```

For passthrough baselines, point the eval client directly at port 8002 (VLA).

### Supported Benchmarks

| Benchmark | Model | Conda env | Eval client | Checkpoint |
|-----------|-------|-----------|-------------|------------|
| [LIBERO](#libero) | pi0.5 | `.libero-venv` | `examples/libero/run_eval.py` | `pi05_libero` (from openpi) |
| [LIBERO-PRO](#libero-pro) | pi0.5 | `.libero-venv` | `examples/libero/run_eval.py` | same as LIBERO |
| [LIBERO-Plus](#libero-plus) | pi0.5 | `.libero-venv` | `examples/libero/run_eval.py` | same as LIBERO |
| [LIBERO-Mem](#libero-mem) | pi0.5 | `.libero-venv` | `examples/libero/run_mem_eval.py` | same as LIBERO |
| [RoboCasa](#robocasa) | pi0 | `robocasa` | `examples/robocasa/robocasa_eval_client.py` | `pi0_robocasa_pretrain_human300` |
| [VLABench](#vlabench) | pi0.5 | `vlabench` | `examples/vlabench/vlabench_eval_client.py` | `pi05_ft_vlabench_primitive` |

---

### LIBERO

Standard LIBERO benchmark (4 suites × 10 tasks). Uses the base openpi pi0.5 checkpoint.

**Setup** (one-time):
```bash
# Clone LIBERO and create a Python 3.10 venv
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
cd ~/LIBERO && python3.10 -m venv .libero-venv
source .libero-venv/bin/activate
pip install -e libero/  # also installs robosuite + mujoco-py
pip install openpi-client imageio scipy
```

**Step 1 — Serve the model:**
```bash
cd ~/openpi
uv run scripts/serve_policy.py --port 8002 \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=<checkpoint_path>
```

**Step 2 — Start the orchestrator proxy:**
```bash
conda activate vlm-orch
vlm-orchestrator --env libero --mode subgoal \
    --vla-port 8002 --port 8019
```

**Step 3 — Run evaluation:**
```bash
source ~/LIBERO/.libero-venv/bin/activate
cd ~/vlm-orchestrator

# Passthrough baseline (direct to VLA)
python examples/libero/run_eval.py \
    --port 8002 \
    --task-suite-name libero_10 \
    --num-trials-per-task 50 \
    --log-dir results/libero/passthrough

# Orchestrated (through proxy)
python examples/libero/run_eval.py \
    --port 8019 \
    --task-suite-name libero_10 \
    --num-trials-per-task 50 \
    --enable-gt-state --enable-depth \
    --log-dir results/libero/orchestrated
```

Available suite names: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (long-horizon).

Key flags:
- `--task-suite-name <suite>`: which LIBERO suite to evaluate
- `--num-trials-per-task N`: episodes per task (default: 50)
- `--task-filter <prefix>`: only run tasks matching this prefix
- `--max-tasks N`: subsample N random tasks (0=all)
- `--enable-gt-state`: export ground-truth state for GT failure detector
- `--enable-depth`: render and forward depth images for grasp tool recovery (**required** when using `grasp_first` or `vlm_grasp` recovery modes)

---

### LIBERO-PRO

Object-level perturbation benchmark (swap/add/change objects). Uses the same
pi0.5 checkpoint as base LIBERO — tests robustness to visual changes.

**Setup**: Clone `~/LIBERO-PRO`, then update `~/.libero/config.yaml` to point
to the PRO data directory. See `docs/libero-extended-benchmarks.md` §3.

**Run evaluation** (same script, different suite names):
```bash
source ~/LIBERO/.libero-venv/bin/activate
cd ~/vlm-orchestrator

# Example: with_mug perturbation suites
python examples/libero/run_eval.py \
    --port 8002 \
    --task-suite-name libero_10_with_mug \
    --num-trials-per-task 3 \
    --log-dir results/libero_pro/passthrough
```

PRO suite names: `libero_10_with_mug`, `libero_spatial_with_mug`,
`libero_object_with_mug`, `libero_goal_with_mug`, and similarly for
`with_milk`, `with_yellow_book`, etc.

---

### LIBERO-Plus

Multi-dimensional perturbation benchmark (language, sensor noise, textures,
lighting, layout, camera, robot). Each perturbed task is unique — use 1 trial per task.

**Setup**: Clone `~/LIBERO-plus`, download `assets.zip` (~6.3GB) from HuggingFace.
See `docs/libero-extended-benchmarks.md` §2.

**Run evaluation:**
```bash
source ~/LIBERO/.libero-venv/bin/activate
cd ~/vlm-orchestrator

# Language dimension (30 random tasks from each base suite)
python examples/libero/run_eval.py \
    --port 8002 \
    --task-suite-name libero_10 \
    --task-filter language \
    --max-tasks 30 \
    --num-trials-per-task 1 \
    --log-dir results/libero_plus/language_passthrough
```

Dimension filters: `language`, `noise`, `texture`, `light`, `layout`, `camera`, `robot`.

---

### LIBERO-Mem

Memory-augmented benchmark with multi-step subgoal tasks. Uses subgoal
completion rate (SCR) instead of binary success. Requires dedicated eval script.

**Setup**: Clone `~/LIBERO-Mem`, copy init states. See `docs/libero-extended-benchmarks.md` §5.

**Run evaluation:**
```bash
source ~/LIBERO/.libero-venv/bin/activate
cd ~/vlm-orchestrator

python examples/libero/run_mem_eval.py \
    --port 8002 \
    --num-trials-per-task 1 \
    --log-dir results/libero_mem/passthrough

# Specific tasks only
python examples/libero/run_mem_eval.py \
    --port 8002 \
    --task-ids "1,2,3" \
    --num-trials-per-task 5
```

Key flags:
- `--task-ids "1,2,3"`: comma-separated task IDs (1-10)
- Output includes per-task SCR (subgoal completion rate) and tiered breakdown

---

### RoboCasa

365-task kitchen manipulation benchmark with PandaMobile robot. Uses pi0
fine-tuned on 300-task human demos.

**Setup** (one-time):
```bash
# Create conda env
conda create -n robocasa python=3.11 -y
conda activate robocasa
cd ~/robocasa && pip install -e .
pip install openpi-client imageio numpy==2.2.5

# Download kitchen assets (~23GB)
python -m robocasa.scripts.download_kitchen_assets

# Download pi0 checkpoint from HuggingFace (~12GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('robocasa/robocasa365_checkpoints',
    allow_patterns='pi0/pi0_robocasa_pretrain_human300/**',
    local_dir='$HOME/robocasa-checkpoints')
"
```

**Step 1 — Serve the model** (requires patched robocasa-openpi fork):
```bash
# The fork at ~/robocasa-openpi has robocasa imports patched for serve-only use
export PYTHONPATH=~/robocasa-openpi/src
cd ~/robocasa-openpi
~/openpi/.venv/bin/python scripts/serve_policy.py \
    --port 8002 \
    policy:checkpoint \
    --policy.config=pi0_robocasa_pretrain_human300 \
    --policy.dir=$HOME/robocasa-checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000
```

**Step 2 — Start the orchestrator proxy:**
```bash
conda activate vlm-orch
vlm-orchestrator --env libero --mode subgoal \
    --vla-port 8002 --port 8019
```

**Step 3 — Run evaluation:**
```bash
conda activate robocasa
cd ~/vlm-orchestrator

# Passthrough: all 18 atomic_seen tasks, 1 trial each (quick survey)
python examples/robocasa/robocasa_eval_client.py \
    --port 8002 \
    --task-set atomic_seen \
    --num-trials 1 \
    --log-dir results/robocasa/passthrough

# Orchestrated
python examples/robocasa/robocasa_eval_client.py \
    --port 8019 \
    --task-set atomic_seen \
    --num-trials 1 \
    --enable-depth \
    --log-dir results/robocasa/orchestrated

# Full eval (50 trials × all tasks)
python examples/robocasa/robocasa_eval_client.py \
    --port 8002 \
    --task-set atomic_seen \
    --num-trials 50 \
    --log-dir results/robocasa/full_passthrough
```

Key flags:
- `--task-set <name>`: `atomic_seen`, `atomic_unseen`, `composite_seen`, `composite_unseen`
- `--num-trials N`: episodes per task
- `--max-tasks N`: subsample N tasks (0=all 18)
- `--split <name>`: `pretrain` or `target`
- `--enable-depth`: forward depth images for grasp tool recovery

Or use the comparison shell script:
```bash
bash examples/robocasa/run_comparison.sh
```

---

### VLABench

10-task primitive manipulation benchmark (select, insert, add). Uses pi0.5
fine-tuned on VLABench data.

**Setup** (one-time):
```bash
# Create conda env
conda create -n vlabench python=3.10 -y
conda activate vlabench
cd ~/VLABench && pip install -e .
pip install mujoco==3.2.2 dm_control==1.0.22 open3d colorlog openai \
    mediapy openpi-client scipy

# Extract scene assets (already downloaded with VLABench)
cd ~/VLABench/VLABench/assets && unzip -o scene.zip -d scenes/

# Download pi0.5 checkpoint from HuggingFace (~12GB params)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('VLABench/pi05-primitive-10task',
    local_dir='$HOME/vlabench-checkpoints/pi05-primitive-10task')
"
# Optionally delete optimizer state to save 27GB:
rm -rf ~/vlabench-checkpoints/pi05-primitive-10task/train_state
```

**Step 1 — Serve the model** (uses openpi fork on `pi05` branch):
```bash
export PYTHONPATH=~/VLABench/third_party/openpi/src
cd ~/VLABench/third_party/openpi
~/openpi/.venv/bin/python scripts/serve_policy.py \
    --port 8002 \
    policy:checkpoint \
    --policy.config=pi05_ft_vlabench_primitive \
    --policy.dir=$HOME/vlabench-checkpoints/pi05-primitive-10task
```

**Step 2 — Start the orchestrator proxy:**
```bash
conda activate vlm-orch
vlm-orchestrator --env libero --mode subgoal \
    --vla-port 8002 --port 8019
```

**Step 3 — Run evaluation:**
```bash
conda activate vlabench
cd ~/vlm-orchestrator
export VLABENCH_ROOT=~/VLABench/VLABench
export MUJOCO_GL=egl

# Passthrough: Track 1 (in-distribution), 3 episodes per task
python examples/vlabench/vlabench_eval_client.py \
    --port 8002 \
    --eval-track track_1_in_distribution \
    --n-episode 3 \
    --log-dir results/vlabench/passthrough

# Orchestrated
python examples/vlabench/vlabench_eval_client.py \
    --port 8019 \
    --eval-track track_1_in_distribution \
    --n-episode 3 \
    --enable-depth \
    --log-dir results/vlabench/orchestrated

# Specific tasks only
python examples/vlabench/vlabench_eval_client.py \
    --port 8002 \
    --tasks select_fruit add_condiment \
    --n-episode 10
```

Key flags:
- `--eval-track <name>`: `track_1_in_distribution`, `track_2_cross_category`,
  `track_3_common_sense`, `track_4_semantic_instruction`, `track_6_unseen_texture`
- `--tasks <name> [<name>...]`: specific tasks (overrides track)
- `--n-episode N`: episodes per task (default: 50)
- `--max-tasks N`: subsample N tasks from track (0=all)
- `--enable-depth`: forward depth images for grasp tool recovery

Or use the comparison shell script:
```bash
VLABENCH_ROOT=~/VLABench/VLABench N_EPISODE=3 \
    bash examples/vlabench/run_comparison.sh
```

---

### GPU Memory Requirements

Only one benchmark model can be served at a time on a single GPU:

| Model | GPU memory | Notes |
|-------|-----------|-------|
| pi0.5 LIBERO | ~15 GB | From openpi base checkpoint |
| pi0 RoboCasa | ~15 GB | Full 12GB checkpoint |
| pi0.5 VLABench | ~15 GB | Full 12GB checkpoint |
| Orchestrator VLM | ~2.5 GB | API-based (no GPU), or local model |
| Grasp server | ~2.5 GB | Optional, for recovery mode |

To switch between benchmarks, kill the current model server and start the new
one on port 8002. The orchestrator proxy does not need to be restarted.

