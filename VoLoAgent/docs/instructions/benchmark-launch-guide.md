<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Launch Guide

How to launch evaluation benchmarks with the VLM-orchestrator pipeline.

Each benchmark requires **4 terminals** (T1–T4). The policy server (T1) and
orchestrator env flag (T3) must match the benchmark being run.

## Port Layout

| Terminal | Service          | Port |
|----------|------------------|------|
| T1       | Policy server    | 8002 |
| T2       | Grasp server     | 8003 |
| T3       | Orchestrator     | 8001 |
| T4       | Eval client      | —    |

> Optional: HITL browser UI on port 8004 — see [HITL Web UI](#hitl-web-ui)
> for setup. Off by default in all examples below.

---

## RoboLab (Isaac Sim)

### T1 — Policy server (openpi env)

```bash
cd ~/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py --port 8002 \
    policy:checkpoint \
    --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

> **Note:** `--port` must come **before** the `policy:checkpoint` subcommand.

### T2 — Grasp server (graspgen env)

```bash
conda activate graspgen
cd ~/vlm-orchestrator
python -m vlm_orchestrator.grasp.server \
    --gripper-config ~/code/toolshed/graspgen/models/checkpoints/graspgen_franka_panda.yml \
    --port 8003 --verbose \
    --enable-sam2 --sam2-model facebook/sam2.1-hiera-small \
    --enable-gdino
```

For pure `--recovery-mode replan` runs (language rewrite only) you can skip
T2 entirely; it's only needed for grasp-tool recovery
(`grasp_first` / `replan_grasp` / `vlm_grasp` / `grasp`). Keeping it
always-on is convenient since toggling recovery mode shouldn't
require a service restart.

### T3 — Orchestrator proxy (vlm-orch env)

```bash
conda activate vlm-orch
cd ~/vlm-orchestrator
vlm-orchestrator \
    --vla-host 127.0.0.1 --vla-port 8002 \
    --port 8001 \
    --mode subgoal \
    --failure-monitor vlm \
    --recovery-mode replan \
    --vlm-model YOUR_VLM_MODEL \
    --use-front-camera \
    --subgoal-timeout 9999 \
    --log-dir results/robolab_<run_label> \
    --verbose
```

Key flag rationale:

- `--mode subgoal` + `--failure-monitor vlm` — periodic VLM checks
  on BEFORE/NOW images, paired with subgoal decomposition. Used when no
  RobolabGTStateExporter is wired (GT-based monitor would be `gt` /
  `gt_vlm` / `gt_hitl` instead).
- `--recovery-mode replan` — language-only rewrite on detected
  failure. **For the grasp-tool variant, swap to** `--recovery-mode replan_grasp`
  (and keep T2 up + add `--grasp-seg-mode gdino_sam2`).
- `--vlm-model YOUR_VLM_MODEL` — the VLM used for reasoning, served
  from an OpenAI-compatible endpoint. Set the API key in your shell
  (`VLM_API_KEY` or `OPENAI_API_KEY`). Any vision-capable model can be
  swapped in via its own `--vlm-base-url` / `--vlm-model` / `--vlm-api-key`.
- `--use-front-camera` — VLM calls + grasp tool consume the
  egocentric mirrored camera (`observation/front_image_left`); the
  policy still receives the exterior camera it was trained on. Requires
  the eval client to forward the front camera (RoboLab forwards it by
  default when GT state is enabled).
- `--subgoal-timeout 9999` — disables time-based subgoal
  advancement; advances only on VLM/GT completion signals. Set a finite
  value to re-enable Mem-style fall-through.
- `--log-dir results/robolab_<run_label>` — per-episode `metadata.json`,
  `rewrites.jsonl`, and rollout videos land under this root, organized
  as `<task_slug>/episode_<N>/`.

For interactive runs, see [HITL Web UI](#hitl-web-ui) below.

### T4 — RoboLab eval (robolab env)

```bash
conda activate robolab
cd ~/robolab
python policies/volo/run.py \
    --headless \
    --remote-host 127.0.0.1 --remote-port 8001 \
    --policy pi05 \
    --num-runs 3 \
    --enable-subtask \
    --enable-gt-state \
    --video-mode all \
    --task-dirs benchmark \
    --instruction-type vague \
    --output-folder-name robolab_<run_label> \
    --task CleanUpToysTask FruitsOnPlateTask FruitsOnPlate3Task
```

Key flags:

- `--remote-host` / `--remote-port 8001` → connect to the orchestrator
  (not the policy directly).
- `--policy pi05` — wire-format selector for the openpi pi0.5 policy
  family (use `gr00t` / `pi0` / `pi0fast` for those policies).
- `--enable-subtask` → request per-task subgoal predicates from
  Robolab's task definition; required for GT-monitor mode and useful
  for VLM-monitor instruction grounding.
- `--enable-gt-state` → forward `obs["gt_state"]` for any GT-based
  failure detection. Harmless when the orchestrator runs `--failure-monitor vlm`.
- `--video-mode all` → save per-episode rollout videos under the output
  tree (independent of the orchestrator's annotated mp4).
- `--task-dirs benchmark` / `--instruction-type vague` → sub-select the
  vague-instruction benchmark task subset. Use `--task-dirs all` and
  drop `--instruction-type` to run the full registered task set.
- `--output-folder-name` → label the per-run output dir under
  `~/robolab/output/`. Match this to the orchestrator's `--log-dir`
  basename for easy correlation.
- `--task` → specific task list. Omit to run every task in the chosen
  `--task-dirs`. `--tag spatial` runs all tasks with a tag instead.

---

## LIBERO

All LIBERO benchmarks (standard, Plus, PRO, Mem) share the same T1–T3
setup. Only the `--task-suite-name` in T4 changes.

### T1 — Policy server (openpi env)

```bash
cd ~/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py --port 8002 \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

### T2 — Grasp server (graspgen env)

Same as RoboLab (see above).

### T3 — Orchestrator proxy (vlm-orch env)

```bash
conda activate vlm-orch
cd ~/vlm-orchestrator
vlm-orchestrator \
    --vla-port 8002 \
    --port 8001 \
    --env libero \
    --mode subgoal \
    --failure-monitor gt \
    --recovery-mode grasp_first \
    --grasp-seg-mode gt_sim \
    --verbose
```

For interactive runs, see [HITL Web UI](#hitl-web-ui) below.

### T4 — LIBERO eval client (.libero-venv)

```bash
cd ~/vlm-orchestrator
source .libero-venv/bin/activate
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name <SUITE_NAME> \
    --num-trials-per-task 3 \
    --enable-depth \
    --enable-gt-state
```

Key flags:
- `--task-suite-name` → see suite tables below
- `--task-filter "moka_pots"` → substring match to select specific task(s)
- `--max-tasks 1` → limit number of tasks (0 = all)
- `--enable-depth` → render depth for grasp recovery point cloud
- `--enable-gt-state` → forward GT state for failure detection
- `--video-out-path` / `--log-dir` → override output dirs (auto-timestamped by default)

---

### LIBERO Standard (4 suites, 10 tasks each)

The core benchmarks. 40 unique tasks total (no overlap with LIBERO-90).
LIBERO-Plus and LIBERO-PRO are perturbation/distractor variants of these
4 suites. Start here.

| Suite name | Description | Max steps |
|------------|-------------|-----------|
| `libero_spatial` | Spatial reasoning (left/right/front/back) | 220 |
| `libero_object` | Object recognition & manipulation | 280 |
| `libero_goal` | Goal-conditioned tasks | 300 |
| `libero_10` | Long-horizon multi-step tasks | 520 |

Example:
```bash
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 3 \
    --enable-depth --enable-gt-state
```

---

### LIBERO-Plus (perturbation robustness)

Tests generalization by perturbing the standard tasks across **7 perturbation
dimensions × 5 difficulty levels**.  See [arXiv:2510.13626](https://arxiv.org/abs/2510.13626)
and [github.com/sylvestf/LIBERO-Plus](https://github.com/sylvestf/LIBERO-Plus).

LIBERO-Plus is a **drop-in replacement** for the standard `libero` package.
It does **not** introduce new suite names — instead the existing four base
suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) are
each expanded from 10 tasks to thousands of perturbed variants.  Perturbation
**category** and **difficulty level** are recorded in
`libero/libero/benchmark/task_classification.json` shipped with the install.

#### Perturbation categories (in `task_classification.json`)

| Category | Description |
|---|---|
| `Camera Viewpoints` | position / orientation / FoV changes |
| `Robot Initial States` | manipulator initial pose variations |
| `Language Instructions` | LLM-rewritten instruction text |
| `Light Conditions` | intensity / direction / colour / shadows |
| `Background Textures` | scene / surface appearance |
| `Sensor Noise` | photometric distortions, image degradation |
| `Objects Layout` | confounding objects + target displacement |

#### Install (separate venv, do not clobber LIBERO-Mem)

```bash
# 1. Fresh venv — keeps the LIBERO-Mem install at .libero-venv intact
python -m venv ~/vlm-orchestrator/.libero-plus-venv
source ~/vlm-orchestrator/.libero-plus-venv/bin/activate

# 2. Install LIBERO-plus as the libero package
cd ~/LIBERO-plus
pip install -e .
pip install -r requirements.txt -r extra_requirements.txt

# 3. Install eval-client transports
pip install msgpack-numpy websockets imageio[ffmpeg] opencv-python-headless

# 4. Sanity check (suite list still uses standard names)
python -c "from libero.libero import benchmark; print(sorted(benchmark.get_benchmark_dict()))"
# → ['libero_10', 'libero_100', 'libero_90', 'libero_goal', 'libero_mix', 'libero_object', 'libero_spatial']

# 5. Download the new objects/textures asset bundle
huggingface-cli download Sylvest/LIBERO-plus assets.zip \
    --repo-type dataset \
    --local-dir ~/LIBERO-plus/libero/libero/
unzip ~/LIBERO-plus/libero/libero/assets.zip \
    -d ~/LIBERO-plus/libero/libero/
```

#### Launch

The eval client accepts `--task-category` and `--task-difficulty` flags
that read `task_classification.json` and sub-select tasks.  Use
`--num-trials-per-task 1` because each task IS already a perturbation —
running 20 episodes of the *same* perturbation is wasted compute.

The eval client itself is mode-agnostic.  Pick one of the two T2 launches
below; T3 is identical for both.

##### Passthrough baseline (no orchestration)

```bash
# T2 — orchestrator proxy
conda activate vlm-orch
vlm-orchestrator \
    --vla-port 8002 --port 8001 \
    --mode passthrough --env libero \
    --log-dir results/libero_plus_lan_passthrough_$(date +%Y%m%d_%H%M%S)

# T3 — eval client (.libero-plus-venv)
source ~/vlm-orchestrator/.libero-plus-venv/bin/activate
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name libero_goal \
    --task-category "Language Instructions" \
    --num-trials-per-task 1 \
    --enable-depth --enable-gt-state
```

##### Subgoal + GT + grasp_first

```bash
# T2 — orchestrator proxy
conda activate vlm-orch
vlm-orchestrator \
    --vla-port 8002 --port 8001 \
    --mode subgoal --env libero \
    --failure-monitor gt --recovery-mode grasp_first \
    --grasp-seg-mode gdino_sam2 \
    --log-dir results/libero_plus_lan_subgoal_$(date +%Y%m%d_%H%M%S)

# T3 — eval client (same as passthrough)
source ~/vlm-orchestrator/.libero-plus-venv/bin/activate
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name libero_goal \
    --task-category "Language Instructions" \
    --num-trials-per-task 1 \
    --enable-depth --enable-gt-state
```

##### Other category / difficulty filters

```bash
# Camera-viewpoint robustness across all four base suites' Spatial set,
# difficulty 3 only:
--task-category "Camera Viewpoints" --task-difficulty 3

# Sample 50 random tasks of any kind (good for a smoke test):
--max-tasks 50
```

##### Notes

- `libero_goal` filtered to `"Language Instructions"` ≈ 380 tasks (out
  of 1,537 total Language tasks across the four base suites).  At 1
  trial per task that's ~3-4 hours of wall-clock at passthrough's ~26
  s/episode rate.  Smoke-test with `--max-tasks 5` first, then drop
  the cap once you've confirmed an episode round-trips end-to-end
  with `eval_metadata.json` written.
- `--enable-depth` / `--enable-gt-state` aren't strictly required in
  passthrough (neither is consumed by the proxy in that mode), but
  keep them on for parity with the eventual subgoal run so the env
  itself runs identically and the only variable is orchestration.
- When `task_classification.json` isn't present (vanilla LIBERO /
  LIBERO-Mem install), `--task-category` / `--task-difficulty` log
  a warning and become no-ops — safe to leave them off in those runs.

---

### LIBERO-PRO (distractor object robustness)

Tests robustness to novel distractor objects placed in the scene.
Suite name pattern: `<base>_with_<distractor>`.

| Distractor | Spatial | Object | Goal | Long-Horizon |
|---|---|---|---|---|
| Red stick | `libero_spatial_with_red_stick` | `libero_object_with_red_stick` | `libero_goal_with_red_stick` | `libero_10_with_red_stick` |
| Blue stick | `libero_spatial_with_blue_stick` | `libero_object_with_blue_stick` | `libero_goal_with_blue_stick` | `libero_10_with_blue_stick` |
| Diffpos stick | `libero_spatial_with_diffpos_stick` | `libero_object_with_diffpos_stick` | `libero_goal_with_diffpos_stick` | `libero_10_with_diffpos_stick` |
| Red box | `libero_spatial_with_red_box` | `libero_object_with_red_box` | `libero_goal_with_red_box` | `libero_10_with_red_box` |
| Mug | `libero_spatial_with_mug` | `libero_object_with_mug` | `libero_goal_with_mug` | `libero_10_with_mug` |
| Green mug | `libero_spatial_with_green_mug` | — | `libero_goal_with_green_mug` | — |
| Yellow book | `libero_spatial_with_yellow_book` | `libero_object_with_yellow_book` | `libero_goal_with_yellow_book` | — |
| Alphabet soup | `libero_spatial_with_alphabet_soup` | `libero_object_with_alphabet_soup` | `libero_goal_with_alphabet_soup` | `libero_10_with_alphabet_soup` |
| Milk | `libero_spatial_with_milk` | — | `libero_goal_with_milk` | `libero_10_with_milk` |
| Rotated stick | — | — | `libero_goal_with_rotated_stick` | — |

Example:
```bash
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name libero_spatial_with_red_stick \
    --num-trials-per-task 3 \
    --enable-depth --enable-gt-state
```

Or run all PRO variants at once:
```bash
python examples/libero/run_extended_benchmarks.py \
    --benchmark pro \
    --port 8001 \
    --num-trials 3
```

---

### LIBERO-90 (large-scale evaluation)

**Separate from LIBERO Standard.** A distinct set of 90 tasks with zero
overlap with the 4 standard suites (spatial/object/goal/10). Covers 10
different scenes across kitchen, living room, and study environments.
Used on the VLA evaluation harness leaderboard as its own benchmark.

| Suite name | Tasks | Description |
|------------|-------|-------------|
| `libero_90` | 90 | 90 unique tasks across 10 scenes |

```bash
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name libero_90 \
    --num-trials-per-task 3 \
    --enable-depth --enable-gt-state
```

Filter by scene or task:
```bash
--task-filter "KITCHEN_SCENE2"     # all 7 tasks in kitchen scene 2
--task-filter "moka_pot"           # moka pot tasks
--task-filter "stack_"             # stacking tasks
```

---

### LIBERO-Mem (memory & multi-step reasoning)

**Separate benchmark** from LIBERO-90.  10 memory-focused tasks requiring
sequential subgoal completion, repetition counting, and spatial reasoning
(e.g. "place bowl on plate 5 times", "swap 2 bowls via empty plate").

- **Metric**: Subgoal Completion Rate (SCR) in addition to binary success
- **Suite name**: `libero_mem`
- **Tasks**: 10 (all in KITCHEN_SCENE1)

| # | Task | Subgoals |
|---|------|----------|
| T1 | Pick up bowl, place on plate | 1 |
| T2 | Lift bottle, put on plate | 1 |
| T3 | Bowl → plate × 3 | 3 |
| T4 | Bottle → plate × 3 | 3 |
| T5 | Bowl → plate × 5 | 5 |
| T6 | Bowl → plate × 7 | 7 |
| T7 | Swap 2 bowls via empty plate | 3 |
| T8 | Rotate 3 bowls left → right | 4 |
| T9 | Cream cheese → nearest basket → center | 2 |
| T10 | Cream cheese → nearest basket, empty basket → center | 2 |

Quick standalone run via the generic client (no per-episode video / metadata):
```bash
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name libero_mem \
    --num-trials-per-task 3 \
    --enable-depth --enable-gt-state
```

Filter single task:
```bash
--task-filter "swap_the_2_bowls"    # T7 only
--task-filter "5_times"             # T5 only
--task-filter "cream_cheese"        # T9 & T10
```

#### LIBERO-Mem dedicated runner — `run_mem_eval.py`

`examples/libero/run_mem_eval.py` is the recommended client for
LIBERO-Mem.  It emits per-step sim counters, reads the orchestrator's
`orchestrator_episode_log_dir`, writes per-episode
`rollout_raw.mp4` / `rollout_annotated.mp4` / `eval_metadata.json`
alongside `rewrites.jsonl`, and patches the Mem-specific
`_check_success(inc=True)` early-termination signal.

The eval client itself is **mode-agnostic** — what changes between
passthrough and subgoal+GT+grasp_first is the proxy launch (T2).
Always launch the proxy with `--log-dir` so per-episode outputs colocate.

##### Passthrough baseline (no orchestration)

```bash
# T2 — orchestrator proxy
conda activate vlm-orch
vlm-orchestrator \
    --vla-port 8002 --port 8001 \
    --mode passthrough --env libero \
    --log-dir results/libero_mem_passthrough_$(date +%Y%m%d_%H%M%S)

# T3 — eval client (.libero-venv)
source ~/vlm-orchestrator/.libero-venv/bin/activate
python examples/libero/run_mem_eval.py \
    --port 8001 \
    --num-trials-per-task 5 \
    --enable-depth --enable-gt-state
```

##### Subgoal + GT + grasp_first

```bash
# T2 — orchestrator proxy
conda activate vlm-orch
vlm-orchestrator \
    --vla-port 8002 --port 8001 \
    --mode subgoal --env libero \
    --failure-monitor gt --recovery-mode grasp_first \
    --grasp-seg-mode gdino_sam2 \
    --log-dir results/libero_mem_subgoal_$(date +%Y%m%d_%H%M%S)

# T3 — eval client (same as passthrough)
source ~/vlm-orchestrator/.libero-venv/bin/activate
python examples/libero/run_mem_eval.py \
    --port 8001 \
    --num-trials-per-task 5 \
    --enable-depth --enable-gt-state
```

The current default `subgoal_timeout = 9999` effectively disables
time-based subgoal advancement — subgoals advance on VLM/GT
completion signals only.  Pass `--subgoal-timeout 200` to the proxy
to re-enable a Mem-style timeout.

##### Subset / single-task runs

```bash
--task-ids 2          # only T3 (bowl-on-plate ×3, the prime debugging task)
--task-ids 0,2,5      # T1, T3, T6 (a quick mix of singleton, repetition-3, repetition-7)
```

##### Output layout

Each episode under the proxy's `--log-dir`:
```
results/libero_mem_<mode>_<ts>/
└── <task_slug>/
    └── episode_<N>/
        ├── rewrites.jsonl       # orchestrator events (decompose, vlm_detect, ...)
        ├── metadata.json        # proxy-side episode metadata
        ├── rollout_raw.mp4      # raw 224×224 frames
        ├── rollout_annotated.mp4 # PIP overlay with subgoal / failure / grasp state
        └── eval_metadata.json   # eval-client side: success, SCR, num_steps, ...
```

`--num-trials-per-task` multiplies through all tasks (3 × 10 = 30
episodes for the default suite, 5 × 10 = 50 if matching the
established baseline).  Omitting `--task-ids` runs every task.
Pass `--video-out-path PATH` only if you intentionally want a
fallback tree under that path when the proxy doesn't advertise an
episode log dir; without it, video saving is skipped with a single
warning so `results/libero_mem` never gets created by accident.

---

### RoboCerebra (long-horizon Franka-sim, NeurIPS 2025)

[RoboCerebra](https://github.com/qiuboxiang/RoboCerebra) is a long-horizon
manipulation benchmark built on the LIBERO simulation platform — same
Franka Panda, same robosuite `OffScreenRenderEnv`, same agentview camera,
same OSC_POSE actions.  Average trajectory length **~2,972 sim steps**
(6× standard LIBERO).  60 unique tasks across 6 perturbation types, each
with a multi-object compound goal tracked via `goal.json`.

- **Repo**: `~/RoboCerebra` (cloned from `qiuboxiang/RoboCerebra`)
- **Bench data**: `~/RoboCerebra_Bench` (selectively downloaded from HF
  dataset `qiukingballball/RoboCerebraBench` — see install step 4 below)
- **Venv**: `~/vlm-orchestrator/.robocerebra-venv`
- **Suite registration**: `vlm_orchestrator/benchmarks/robocerebra.py` —
  registers a `robocerebra` benchmark with `libero.libero.benchmark` so
  the standard eval client works with `--task-suite-name robocerebra`.

| Task type | Cases | Description |
|---|---|---|
| `Ideal` | 10 | Static baseline tasks |
| `Memory_Exploration` | 10 | Active object search |
| `Memory_Execution` | 10 | Memory of prior placements |
| `Observation_Mismatching` | 10 | Visual discrepancies (reuses `Ideal/` BDDLs at runtime) |
| `Random_Disturbance` | 10 | Environmental noise (reuses `Ideal/` BDDLs) |
| `Mix` | 10 | Combined types (only 5 have init_files; the rest skip with `use_init_files=True`) |

#### Install (separate venv, do not clobber other LIBERO installs)

```bash
# 1. Repo
git clone https://github.com/qiuboxiang/RoboCerebra.git ~/RoboCerebra

# 2. Selective HF download — eval-only, ~150 MB
#    (Skipping the full 137 GB train tree: no mp4/png previews, no RLDS.)
pip install --user huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='qiukingballball/RoboCerebraBench',
    repo_type='dataset',
    local_dir='/home/$USER/RoboCerebra_Bench',
    allow_patterns=['*.bddl', '*.json', '*.txt', '*.hdf5', '*.init'],
)
"

# 3. Venv (mirror .libero-plus-venv — Python 3.12 + bootstrap pip)
python -m venv --without-pip ~/vlm-orchestrator/.robocerebra-venv
~/vlm-orchestrator/.robocerebra-venv/bin/python -m ensurepip || \
    (curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
     ~/vlm-orchestrator/.robocerebra-venv/bin/python /tmp/get-pip.py)

PIP=~/vlm-orchestrator/.robocerebra-venv/bin/pip
# 4a. Bundled LIBERO from RoboCerebra repo (-e, namespace .pth)
$PIP install -e ~/RoboCerebra/LIBERO
echo "/home/$USER/RoboCerebra/LIBERO" \
    > ~/vlm-orchestrator/.robocerebra-venv/lib/python3.12/site-packages/libero.pth

# 4b. Robosuite from LIBERO-Mem's vendored copy (avoids evdev compile chain)
$PIP install --no-deps pynput
$PIP install --no-deps -e ~/LIBERO-Mem/thirdparty/robosuite

# 4c. Runtime + eval-client deps
$PIP install \
    'numpy<2' 'numba>=0.49.1' scipy 'mujoco>=2.3.0' Pillow opencv-python termcolor \
    hydra-core 'transformers>=4.30' robomimic einops bddl 'gym<0.27' \
    matplotlib cloudpickle easydict future Wand scikit-image h5py \
    msgpack-numpy websockets 'imageio[ffmpeg]' opencv-python-headless tyro torch \
    -e ~/openpi/packages/openpi-client \
    -e ~/vlm-orchestrator
$PIP install 'numpy<2'   # transformers/scikit-image pull 2.x; pin back
```

#### Per-venv config (essential — do once)

```bash
mkdir -p ~/vlm-orchestrator/.robocerebra-venv/.libero
cat > ~/vlm-orchestrator/.robocerebra-venv/.libero/config.yaml <<EOF
benchmark_root: $HOME/RoboCerebra_Bench
bddl_files:     $HOME/RoboCerebra_Bench
init_states:    $HOME/RoboCerebra_Bench/init_files
datasets:       $HOME/RoboCerebra_Bench
assets:         $HOME/RoboCerebra/LIBERO/libero/libero/assets
EOF

# Bake env vars into activate
cat >> ~/vlm-orchestrator/.robocerebra-venv/bin/activate <<'EOF'

# Per-venv LIBERO config + RoboCerebra bench root
export LIBERO_CONFIG_PATH="$VIRTUAL_ENV/.libero"
export ROBOCEREBRA_BENCH_ROOT="$HOME/RoboCerebra_Bench"
EOF
```

The two-tier config (`assets` from the bundled LIBERO; everything else
from `RoboCerebra_Bench`) matters: the BDDL files reference shared LIBERO
mesh / texture assets, and `bddl_files` must point at the bench root so
the eval client's `pathlib.Path(bddl_files) / problem_folder / bddl_file`
join lands on real files.

#### Sanity verify (optional)

```bash
source ~/vlm-orchestrator/.robocerebra-venv/bin/activate
python -c "
from vlm_orchestrator.benchmarks.robocerebra import register, RoboCerebra
register()
s = RoboCerebra()
print(f'tasks: {s.n_tasks}')
print(f'first: {s.get_task(0).language[:60]}...')
"
# → tasks: 60
```

#### Launch

The eval client itself is mode-agnostic — only T2 changes between modes.

##### Passthrough baseline

```bash
# T2 — orchestrator proxy (vlm-orch env)
TS=$(date +%Y%m%d_%H%M%S)
vlm-orchestrator \
    --vla-port 8002 --port 8001 \
    --mode passthrough --env libero \
    --log-dir "results/robocerebra_passthrough_$TS"

# T3 — eval client (.robocerebra-venv)
source ~/vlm-orchestrator/.robocerebra-venv/bin/activate
python examples/libero/libero_eval_client.py \
    --port 8001 \
    --task-suite-name robocerebra \
    --num-trials-per-task 1 \
    --enable-depth --enable-gt-state
```

##### Subgoal + VLM monitor + replan_grasp

```bash
# T2
TS=$(date +%Y%m%d_%H%M%S)
vlm-orchestrator \
    --vla-port 8002 --port 8001 \
    --mode subgoal --env libero \
    --failure-monitor vlm --recovery-mode replan_grasp \
    --grasp-seg-mode gdino_sam2 \
    --log-dir "results/robocerebra_vlm_replan_grasp_$TS"

# T3 — same as passthrough
```

##### Other useful T2 variants

```bash
# Subgoal + VLM monitor + replan only (text rewrite, no grasp tool)
--mode subgoal --env libero --failure-monitor vlm --recovery-mode replan

# Subgoal + VLM monitor + grasp only
--mode subgoal --env libero --failure-monitor vlm --recovery-mode grasp \
    --grasp-seg-mode gdino_sam2
```

`--failure-monitor gt` / `gt_hitl` / `gt_vlm` will not fire — there's no
`RoboCerebraGTStateExporter` yet.  `--enable-gt-state` on the eval client
is harmless (forwards an empty `gt_state`); the VLM monitor doesn't read
it.  Wiring a GT exporter for RoboCerebra is a future ~½-day item:
translate `goal.json` + sim state into the shape `LiberoGTStateExporter`
emits.

#### Subset / single-task runs

```bash
--task-filter "Ideal_case1!case10"      # exactly one Ideal task (excludes case10)
--task-filter "Memory_Exploration"      # all 10 cases of one task type
--task-filter "case5"                   # case5 across all task types
--max-tasks 5                           # random sample of 5 tasks
```

#### Per-task dynamics (RoboCerebra-specific quirks)

- **Per-task max_steps** — RoboCerebra tasks have 4-15 atomic
  segments; the per-task budget is `len(task_description.json) × 150`
  sim steps (mirrors upstream `cfg.switch_steps × segment_count`).  A
  4-segment task gets 600 steps, a 15-segment task 2,250.  Implemented
  via `RoboCerebra.get_task_max_steps(i)`; the eval client calls this
  per task instead of the suite-wide ceiling.
- **Authoritative success metric** — `env.step()`'s `done` flag fires
  on partial-goal predicate matches for these multi-object compound
  tasks (e.g. it returns `True` once one of five required object
  placements is satisfied).  Use only `RoboCerebra.compute_episode_success(env, i)`,
  which loads `goal.json`, calls `env._check_success(monitor_dict)`,
  and returns the third element (`all_done`).  The eval client uses
  this when `compute_episode_success` is present on the suite — and
  also keeps running to `max_steps` instead of breaking on `done`.
- **Init-state convention** — `.init` files are plain Python pickles
  (a single robosuite sim state per case), unlike LIBERO's
  `*.pruned_init` which is a list.  `RoboCerebra.get_task_init_states(i)`
  wraps the loaded state in a 1-element list so the eval client's
  `initial_states[ep_idx]` flow works.  Re-trials of the same case use
  the same deterministic init.
- **Mix init coverage** — only 5 of the 10 Mix cases ship init_files
  upstream.  With `use_init_files=True` (the default), the other 5 are
  silently skipped.

#### Output layout

```
results/robocerebra_<mode>_<ts>/
└── Task_<task_slug>/
    └── episode_<N>/
        ├── rewrites.jsonl          # orchestrator events (decompose, vlm_detect, ...)
        ├── metadata.json           # proxy episode metadata
        ├── rollout_raw.mp4
        ├── rollout_annotated.mp4   # PIP overlay with subgoal / failure / grasp PIP
        └── eval_metadata.json      # success, num_steps, duration_s, orchestrator_*
```

`success` in `eval_metadata.json` is the `compute_episode_success`
return — `True` only when the full multi-object goal in `goal.json` is
satisfied at episode end.

#### Known limitations / future work

- No `RoboCerebraGTStateExporter` → can't use GT-based failure
  monitoring or grasp_first recovery (use VLM-based instead).
- Cognitive metrics (planning accuracy / reflection / efficiency) from
  the upstream paper are NOT computed — we measure success only.
  Wiring them would require porting the anchor-aligned plan-vs-GT
  matching from `~/RoboCerebra/evaluation/`.
- pi0.5-libero is OOD on RoboCerebra's coffee_table / kitchen_table /
  study_table scenes; expect floor-level passthrough on the harder
  task types (Memory_Exploration, Observation_Mismatching).  This is
  a known "doing-no-harm" baseline finding, not an integration bug.

---

## RoboCasa (MuJoCo kitchen tasks)

> **Excluded from CoRL eval (2026-05-03).** The originally-targeted
> `pi05_robocasa` doesn't exist (was a planning placeholder; no such
> checkpoint in any openpi training config or HF org). The
> locally-available `pi0_robocasa_pretrain_human300` covers only 2 of
> 16 `composite_seen` tasks within its training distribution — same
> null-result wall as VLABench composites. The right checkpoint
> (`pi0_robocasa_target_composite_seen`) has a training config in
> `robocasa-openpi/src/openpi/training/config.py:929` but no shipped
> weights anywhere we can find; obtaining requires either non-public
> access or self-training (~50-100 H100-hours). See `corl-plan.md` §5
> RoboCasa footnote for full rationale. Infrastructure below remains
> in the repo and is ready to run if/when target weights become
> available — integration cost at that point ~½ day.

[RoboCasa](https://robocasa.ai/) is a large-scale simulation framework with
365 kitchen tasks across atomic skills (pick-place, doors, drawers, stove,
microwave, etc.) and composite tasks (cooking, cleaning, loading dishwasher,
etc.) using 2,500+ kitchen scenes and 3,200+ 3D objects.

- **Repo**: `~/robocasa` (installed), `~/robocasa-openpi` (eval scripts)
- **Sim**: MuJoCo/robosuite (same runtime as LIBERO)
- **Robot**: Franka Panda with OSC_POSE controller
- **Status (paper)**: Not run for the CoRL submission per the callout above.
- **Status (infrastructure)**: Eval client at
  `examples/robocasa/robocasa_eval_client.py` is a working openpi
  websocket client (depth + GT-state forwarding, composite-config seed
  handling); GT exporter at `vlm_orchestrator/benchmarks/robocasa_gt.py`
  is present. Routes through the same orchestrator proxy as LIBERO.

### Standalone eval (no orchestrator)

```bash
# T1 — Policy server (robocasa-openpi env)
cd ~/robocasa-openpi
uv run scripts/serve_policy.py --port 8002 \
    policy:checkpoint \
    --policy.config=pi05_robocasa \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_robocasa

# T2 — Eval client
cd ~/robocasa-openpi
uv run examples/robocasa/main.py --port 8002 \
    --task_set EvalTaskSet1 \
    --num_rollouts 50
```

### Task sets

| Task set | Description |
|----------|-------------|
| `EvalTaskSet1` | Standard evaluation (atomic tasks) |
| `EvalComposite` | Composite multi-step tasks |

Individual tasks can be selected via `--task_name`:
```bash
uv run examples/robocasa/main.py --port 8002 \
    --task_name PnPCounterToCab --num_rollouts 10
```

### Orchestrator integration (TODO)

To integrate with the orchestrator proxy, an eval client similar to
`examples/libero/libero_eval_client.py` is needed that:
1. Converts RoboCasa observations to the wire format
2. Connects to the orchestrator websocket (not policy directly)
3. Forwards GT state for failure detection

---

## VLABench (language-conditioned long-horizon reasoning, ICCV 2025)

[VLABench](https://vlabench.github.io/) ([arXiv:2412.18194](https://arxiv.org/abs/2412.18194))
is a MuJoCo (dm_control) benchmark for language-conditioned robotic manipulation
with long-horizon reasoning tasks.

- **Primitive tasks** (10 trained): `add_condiment`, `insert_flower`,
  `select_{book, chemistry_tube, drink, fruit, mahjong, painting, poker, toy}`.
  Evaluated via 5 robustness perturbation tracks (`track_1_in_distribution`,
  `track_2_cross_category`, `track_3_common_sense`, `track_4_semantic_instruction`,
  `track_6_unseen_texture`) with deterministic per-episode configs.
- **Composite tasks** (~50 registered, 36 with stable random_init): cluster
  series, rearrangement, cooking, set-dining, physical-QA. **No upstream
  deterministic configs** — random spawn poses with a known
  `# BUG some episodes are unstable` upstream comment in
  `evaluator/base.py:84`. We probe and ship per-task validated seeds.

### One-time setup

#### 1. Eval-client conda env

VLABench ships its own dependency stack incompatible with our other venvs:

```bash
conda create -n vlabench python=3.10 -y
conda activate vlabench
cd ~/VLABench && pip install -e .
pip install msgpack-numpy websockets 'imageio[ffmpeg]' opencv-python-headless
pip install ~/openpi/packages/openpi-client
```

Note: `vlm-orchestrator` is **not** installed in this env. The eval client
prepends the repo root to `sys.path` at import time so `EpisodeVideoWriter`
and other orchestrator-side classes load correctly. If you see
`ModuleNotFoundError: No module named 'vlm_orchestrator'` at video-write
time, check that `examples/vlabench/vlabench_eval_client.py:46-55` (the
`sys.path.insert(0, _REPO_ROOT)` block) hasn't been edited out.

#### 2. VLA server fork (separate venv)

VLABench's `pi05-primitive-10task` checkpoint requires the
[`Shiduo-zh/openpi`](https://github.com/Shiduo-zh/openpi) fork on the `pi05`
branch (different from upstream `openpi`). The lockfile is identical to
upstream so we use a separate venv for clean isolation:

```bash
git clone https://github.com/Shiduo-zh/openpi.git ~/openpi-vlabench
cd ~/openpi-vlabench && git checkout pi05
uv sync   # creates .venv; identical lockfile to upstream → near-zero disk

# Download the checkpoint (~12 GB, one-time)
~/openpi/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'VLABench/pi05-primitive-10task',
    local_dir='/home/$USER/vlabench-checkpoints/pi05-primitive-10task',
)"
```

#### 3. Composite seed JSON (one-time, ~3.5 hours)

Probe valid `random_init` seeds for composite tasks:

```bash
~/miniconda3/envs/vlabench/bin/python scripts/vlabench_probe_composite_seeds.py \
    --output data/vlabench_composite_seeds.json \
    --phase1-max 50 --phase2-max 500 --n-dummy-steps 5
```

Validation = `env.reset()` + 5 hold-pose steps without `PhysicsError`. Phase
1 (seeds 0-49) covers most tasks; phase 2 (50-499) is for stragglers but
empirically unproductive — most phase-1 failures are structurally broken
(over-packed workspace). Output JSON has one validated seed per task plus
`null` for skipped tasks. Resumable via `--resume`.

Current snapshot: **36/50 valid, 14 skipped**. Skipped tasks (excluded
from eval): cluster_drink, cluster_ingredients, cluster_toy,
find_fruit_to_make_juice, hang_picture_on_specific_nail,
insert_power_cord_to_make_juice, play_snooker, plug_cord_and_heat_food,
replace_wilted_flower, set_dining_chopstick (×2 variants),
set_dining_left_hand, set_dining_table, simple_cuestick_use.

### T1 — Policy server (`Shiduo-zh/openpi` fork)

```bash
cd ~/openpi-vlabench
.venv/bin/python scripts/serve_policy.py \
    --port 8005 \
    policy:checkpoint \
    --policy.config=pi05_ft_vlabench_primitive \
    --policy.dir=/home/$USER/vlabench-checkpoints/pi05-primitive-10task
```

Note the port: **8005** for VLABench (avoids collision with the LIBERO
`pi05_libero` server on 8002 if both are running).

### T2 — Grasp server (graspgen env)

Optional. Same as RoboLab; only needed when running `--recovery-mode replan_grasp`
or `grasp_first`. Skip for `replan`-only orchestrated runs and all passthrough
runs.

### T3 — Orchestrator proxy (vlm-orch env)

```bash
VLM_API_KEY=<key> ~/miniconda3/envs/vlm-orch/bin/vlm-orchestrator \
    --env libero \
    --mode subgoal \
    --failure-monitor vlm --recovery-mode replan \
    --prompt-style vlabench \
    --vla-host 127.0.0.1 --vla-port 8005 \
    --port 8019 \
    --log-dir results/vlabench_<run_label>
```

Key VLABench-specific flags:
- `--env libero` — VLABench's policy input format (front/right/wrist 3-cam,
  8-D state) matches LIBERO's wire convention closely enough that the libero
  preset works.
- `--vla-port 8005` — points at the VLABench-fork policy server, not LIBERO's.
- `--port 8019` — VLABench client convention; avoids collision with LIBERO's 8001.
- `--prompt-style vlabench` — **required** for VLABench. Activates a
  prompt variant in `subgoal_base.py` that tells the VLM to preserve the
  original instruction's wording on single-action tasks and gate object-name
  expansion on visible scene ambiguity. Without this, the VLM paraphrases
  in-distribution prompts (e.g. `"Take out the cola from the fridge_open"`
  → `"Pick the red cola can from the fridge shelf and place it on the table
  outside the fridge"`), which is OOD for the pi05-primitive-10task
  fine-tune and degrades SR by ~10pp.
- `--recovery-mode replan` (no grasp tool) is the cleanest first
  comparison — `replan_grasp` adds the grasp tool but requires T2 up.

### T4 — Eval client (vlabench env)

#### Primitive evaluation (5 official tracks)

```bash
~/miniconda3/envs/vlabench/bin/python examples/vlabench/vlabench_eval_client.py \
    --port 8019 \
    --eval-track track_1_in_distribution \
    --n-episode 3 --max-steps 200 \
    --log-dir results/vlabench_<run_label>
```

`--eval-track` choices: `track_1_in_distribution`, `track_2_cross_category`,
`track_3_common_sense`, `track_4_semantic_instruction`, `track_6_unseen_texture`.
Each uses VLABench's deterministic per-episode configs (no seed probe needed).

For passthrough comparison, change `--port 8019` → `--port 8005` (skip the
orchestrator entirely).

#### Composite evaluation (36 validated tasks)

```bash
~/miniconda3/envs/vlabench/bin/python examples/vlabench/vlabench_eval_client.py \
    --port 8019 \
    --tasks $(python -c "import json; d=json.load(open('data/vlabench_composite_seeds.json')); print(' '.join(t for t,v in d['tasks'].items() if v['seed'] is not None))") \
    --composite-config data/vlabench_composite_seeds.json \
    --n-episode 3 --max-steps 600 \
    --log-dir results/vlabench_<run_label>
```

Key flags:
- `--composite-config` — loads the probe JSON. Each task uses its
  pre-validated seed for **all** episodes (deterministic init across
  passthrough vs. orchestrated runs, so VLM stochasticity is the only
  source of variation in the orchestrated arm). Skips the `MAX_RESET_RETRIES`
  retry loop entirely. Tasks absent from the JSON or with `seed=null` are
  hard-skipped with a warning.
- `--tasks <list>` — overrides `--eval-track`; required for composites
  because they have no track JSON. The shell substitution above pulls the
  36 valid task names directly from the seed JSON.
- `--max-steps 600` — composites have no per-task `max_episode_length` in
  `task_config.json` and inherit upstream's hardcoded 200 fallback, which
  floors most composite trajectories. Bump on the CLI.

#### Output layout

```
results/vlabench_<run_label>/
└── <task_name>/                     # e.g. select_fruit, cluster_book
    └── episode_<N>/
        ├── rewrites.jsonl           # orchestrator events (only orchestrated runs)
        ├── metadata.json            # proxy episode metadata (only orchestrated)
        ├── rollout_raw.mp4          # 2x2 multi-view tile (right/left/front/wrist)
        ├── rollout_annotated.mp4    # subgoal/failure overlay (only orchestrated)
        └── eval_metadata.json       # success, num_steps, instruction, task_slug, ...
```

The 2x2 tile matches VLABench upstream's `evaluator/base.py:207` save_video
convention. `eval_metadata.json` includes both `task_slug` (stable family
name like `select_painting` — used for aggregation) and `instruction` (the
per-episode prompt — varies across primitive episodes since each ep targets
a different object/style).

### Grasp tool on VLABench: NOT SUPPORTED for CoRL deadline

> Running `--recovery-mode replan_grasp` / `grasp` / `grasp_first` /
> `vlm_grasp` against VLABench produces visibly broken arm motion:
> the grasp tool's planned EE-delta trajectory is designed for
> robosuite OSC_POSE controller compliance, but VLABench's joint-pos
> env applies each delta-derived IK setpoint discontinuously. Even
> with bumped physics substeps and cumulative-from-snapshot
> integration the arm wandered without reaching the target.
>
> Use `--recovery-mode replan` (language-only rewrite) on VLABench
> for the CoRL submission. The grasp tool path is left in the repo
> for post-deadline rework — the proper fix is `GraspEnvMode.VLABENCH`
> emitting absolute EE-targets that the eval client IKs once, in
> the same path policy actions go through.

### Known VLABench-specific pitfalls

- **Multi-phrasing instructions.** Some tasks (notably `density_qa`) define
  a list of alternative phrasings in `get_instruction()`, and `dm_task.py:295`
  randomly samples one per call. Calling `get_instruction()` every step
  flips the prompt mid-episode and fools the orchestrator's
  prompt-changed boundary detector into rotating episode logs every step.
  The eval client locks the instruction at `step==0` to avoid this.

- **Per-episode prompts vary even within a single task.** `select_painting`
  ep 0 vs ep 1 hit different style targets ("ukiyo-e" vs "surrealism"),
  so the prompt is NOT a stable per-task identifier. The eval client
  forwards `wire_obs["__task_slug"] = task_name` so the orchestrator's
  episode-dir naming uses the stable family name instead of the prompt.
  Without this, three episodes of `select_drink` would land in three
  separate `Take_out_the_*` dirs.

- **Physics divergence at reset on broken composites.** The 14 skipped
  tasks in our seed JSON exhausted 500 attempts without a stable seed.
  These are upstream-broken (e.g., `cluster_drink` packs 5 objects into
  a 0.012 m³ workspace). Hard-skip via the composite-config's `null` seed
  rather than running the legacy retry loop.

- **VLA distribution shift via prompt rewriting.** Discussed above — always
  pass `--prompt-style vlabench` to the orchestrator on VLABench runs.

- **CLI quirk: `--tasks` must be set explicitly to override `--eval-track`'s
  default.** Without `--tasks`, the client always loads `track_1_in_distribution`.

---

## Switching Between Benchmarks

When switching from one benchmark to another:

1. **Kill T1** (policy server) — each benchmark needs a different checkpoint
2. **Restart T3** with the correct `--env` flag (`droid` vs `libero`)
3. **T2** (grasp server) can stay running — it's benchmark-agnostic
4. **T4** uses a different script/env per benchmark

| Benchmark | T1 policy config          | T3 `--env` | T4 conda/venv          | Orchestrator |
|-----------|---------------------------|------------|------------------------|--------------|
| RoboLab   | `pi05_droid_jointpos`     | `droid`    | `conda: robolab`       | ✓ Integrated |
| LIBERO (all) | `pi05_libero`          | `libero`   | `.libero-venv`         | ✓ Integrated |
| LIBERO-Mem | `pi05_libero`            | `libero`   | `.libero-venv`         | ✓ Integrated |
| RoboCerebra | `pi05_libero`           | `libero`   | `.robocerebra-venv`    | ✓ Integrated |
| RoboCasa  | `pi0_robocasa_target_composite_seen` (target; weights not shipped) — fallback `pi0_robocasa_pretrain_human300` (local, mostly OOD on composite_seen) | `libero` | `~/robocasa-openpi` | ⚠ Wired, **excluded from CoRL eval** — see §RoboCasa for the checkpoint situation |
| VLABench  | `pi05_ft_vlabench_primitive` (Shiduo-zh fork, port 8005) | `libero` | `conda: vlabench` | ✓ Integrated (use `--prompt-style vlabench`) |

> **Tip:** When switching between LIBERO suites (e.g. standard → Plus → PRO),
> only T4 needs to change. T1, T2, and T3 stay the same.

---

## Orchestrator Modes Quick Reference

### Failure monitor (`--failure-monitor`)

| Mode      | Behavior | When to use |
|-----------|----------|-------------|
| `gt`      | Auto-detect from `obs["gt_state"]` (CSM, contact, subtask score) | LIBERO/robolab when `--enable-gt-state` is on |
| `gt_hitl` | GT detection + pause for human approval | data collection / debug |
| `gt_vlm`  | GT detection + VLM decides recovery | most expressive when GT is wired |
| `vlm`     | Periodic VLM-only checks on BEFORE/NOW images | RoboCerebra, VLABench (no GT exporter) |
| `signal_primary` | Action/EE-pose heuristics, no VLM cost | cheap baseline |
| `union_failure` | Signal ∪ VLM (max recall) | when missing failures is worse than spurious recovery |
| `intersect_video` | Signal + video VLM confirmation | high-precision detection |
| (none)    | No failure detection | passthrough only |

### Recovery mode (`--recovery-mode`)

| Mode          | Behavior | Compatible monitors |
|---------------|----------|---------------------|
| `grasp_first` | Immediately trigger grasp tool on any failure | `gt` only |
| `replan`      | VLM rewrites instruction (no grasp tool) | `vlm` |
| `replan_grasp` | VLM rewrites + falls back to grasp tool when target identified | `vlm` |
| `grasp`       | Grasp tool only, no instruction rewrite | `vlm`, signal-based |
| `vlm_grasp`   | VLM decides whether to retry or use grasp tool | `vlm`, signal-based |
| `vlm`         | VLM-only language rewrite | signal-based monitors |
| `retry`       | Re-issue current chunk | signal-based |
| `template`    | Rewrite instruction from template (no grasp tool) | signal-based |

`grasp_first` and `replan_grasp`/`grasp` require the grasp server running on
T2 with `--grasp-seg-mode {gt_sim, gdino_sam2, sam3}` matching what the eval
client forwards (`gt_sim` needs depth + GT seg masks; `gdino_sam2` works
with depth alone).

### Orchestration mode (`--mode`)

| Mode          | Behavior |
|---------------|----------|
| `passthrough` | Forward observations/actions without modification |
| `subgoal`     | VLM decomposes task into subgoals, tracks progress |
| `rewrite`     | Single VLM rewrite at episode start, no decomposition / monitoring |
| `next_goal`   | Self-contained: per-`--check-interval` VLM call predicts the next step on-the-fly |

### Other useful flags

| Flag | Purpose |
|---|---|
| `--prompt-style {default,vlabench}` | System-prompt variant. `vlabench` preserves original instruction wording on single-action tasks; gates name-expansion on visible scene ambiguity. Required for VLABench; do not use on robolab/LIBERO unless you re-validate. |
| `--subgoal-timeout 9999` | Disable time-based subgoal advancement (advance only on VLM/GT completion). Default for Mem-style runs. |
| `--use-front-camera` | VLM + grasp tool use the front/egocentric camera; policy still receives the trained exterior camera. |
| `--check-interval N` | VLM detection/check fires every N action chunks. Lower = more responsive, higher = cheaper. |

---

## HITL Web UI

Optional interactive browser UI for human-in-the-loop runs.
**All default examples in this guide omit HITL** — turn it on only when
you need pause/approve/override behavior (data collection, debugging
failure detection, or `gt_hitl` mode where human approval is required
on every detected failure).

### Enable

Add to **any** orchestrator launch (T3) in this guide:

```bash
    --hitl --hitl-port 8004
```

Then open `http://localhost:8004` in a browser. Works identically
across RoboLab, LIBERO, RoboCerebra, and VLABench — same flag, same
port, same UI.

### What the UI shows

- Live camera feed from the robot (primary scene image)
- Current subgoal and progress (subgoal index / total, decomposed list)
- Failure detection alerts (with reason text from VLM or GT detector)
- Controls: submit corrected subgoal, mark current step done, flag
  failure, skip step, abort episode

### When to use

| Use HITL | Use case |
|---|---|
| ✅ | Recording labeled human-correction data for downstream training |
| ✅ | Debugging why a subgoal/failure detector fires (or doesn't) |
| ✅ | `--failure-monitor gt_hitl` — pauses on every detected failure for human approval |
| ✅ | Demos / qualitative review with a human in the loop |
| ❌ | Headless eval sweeps (the UI websocket adds latency and stalls if no client connects) |
| ❌ | Headless / cluster runs where there's no browser to attach |

### Notes

- HITL is the only thing on port 8004; if 8004 is busy, pass
  `--hitl-port <other>` and update the browser URL.
- The eval client requires no special flag — HITL only affects the
  orchestrator side.
- `--failure-monitor gt_hitl` is a **detection mode**, not the
  same as `--hitl` (the UI flag). `gt_hitl` requires `--hitl` to be
  set so the UI is reachable; without `--hitl`, `gt_hitl` falls
  back to auto-recovery and the human-approval gate is never
  reached.

---

## Output Locations

| Benchmark | Default output |
|-----------|----------------|
| RoboLab   | `~/robolab/output/<timestamp>_<policy>/` |
| LIBERO    | `~/vlm-orchestrator/results/libero/eval_<timestamp>/` |
| Orchestrator logs | `~/vlm-orchestrator/results/<mode>_<timestamp>/` |

LIBERO saves:
- `videos/` — raw + annotated rollout videos
- `logs/results.json` — per-episode success/failure data

Orchestrator saves (all benchmarks):
- `rewrites.jsonl` — per-step orchestrator decisions (completions, failures, recoveries)
- `debug_images/` — grasp tool debug visualizations (if grasp recovery active)
