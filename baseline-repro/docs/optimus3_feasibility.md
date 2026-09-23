# Optimus-3 "as released" on the 70 tech-tree tasks: feasibility

Date: 2026-09-23. This is a read-only code analysis. No weights were downloaded, no packages were installed, and no GPU process or Minecraft instance was started.

Sources:

- **R** = `/home/rag/data/official/Optimus-3`. This is a shallow clone of JiuTian-VL/Optimus-3 at commit `a73c013` (2026-04-14). `MS/` means `R/MineStudio/minestudio/`, and `src/` means `R/src/minecraftoptimus/`.
- **GUI** = the reference client, lizaijing/OptimusGUI at commit `7775758`. It was cloned to the scratchpad for reading only.
- **HF** facts come from the Hugging Face model API (file lists and sizes) and from each repo's `config.json` and `added_tokens.json`. Only those small files were fetched.
- **Wheel** facts come from wheels saved with `pip download --no-deps` into the scratchpad. They were not installed.

---

## TL;DR

- **Verdict: GO, with caveats.** The code has no automated rollout or eval for long-horizon tasks. The only live-episode driver is an interactive FastAPI GUI server. That server's loop is small and easy to reproduce headless: about 60 lines of `gui_server.py` logic plus our monitor.
- **Effort:** about 12 h of engineering (range 10–16 h), plus smoke-test wall time.
- **Compute (estimate, not measured):** worst case about 1.28 M env steps for one pass over the 70 tasks. At roughly 10 steps/s that is about 36 process-hours. With 4 parallel workers it is about 9–10 h.
- **Things the table must disclose.** These are what "as released" means here:
  - (a) Crafting and smelting are scripted GUI macros, not learned.
  - (b) "dig down" subgoals force attack and a scripted look-down.
  - (c) The simulator injects ore under the agent through `random_ore`, at **80 % per new y-level**. Optimus-1's rate is 10 %.
  - (d) The MLLM is text-only in the action loop. It plans once, and per subgoal it emits one "task token" whose hidden state conditions a STEVE-1-style policy.
  - (e) There is no replanning, no reflection, and no use of the task router in the released loop.

---

## 1. How the released code drives a live episode

### Entry point and API

The entry point is `R/gui_server.py`, a FastAPI app. It runs with `uvicorn.run("gui_server:app", host="ip", port=9500)`, and the IP is a placeholder (`gui_server.py:576-579`). The endpoints are:

| Endpoint | Code | Behaviour |
|---|---|---|
| `startup` | `gui_server.py:232-258` | Calls `/reset` on the first GPU with at least 40 GB free, otherwise on the CPU. |
| `POST /reset {device}` | `gui_server.py:261-311` | Builds a **new** `MinecraftSim(obs_size=(128,128), preferred_spawn_biome="forest", callbacks=[CommandsCallback([...8 gamerules...])], seed=random.randint(1,1e8))`. Then it calls `env.reset()` and creates `helper = {craft: CraftWorker, smelt: SmeltWorker, equip: EquipWorker}`. It loads `Optimus3Agent(policy_dir, mllm_dir, router_dir)` once, and the checkpoint paths are placeholders (`gui_server.py:298-303`). |
| `POST /send_text {text, task}` | `gui_server.py:426-526` | Dispatches on `task`: `planning`, `captioning`/`embodied_qa`, `action`, or `grounding`. |
| `POST /pause`, `/resume`, `GET /get_obs`, `/receive_text`, `/status`, `/gpu`, `WS /ws/obs` | `gui_server.py:111-230, 389-573` | UI plumbing: pause, frame streaming, and status. |

The reset commands are the same 8 used in Env O and Env M: sendCommandFeedback off, commandBlockOutput off, keepInventory, night vision, doDaylightCycle off, time 0, doImmediateRespawn, and /spawnpoint (`gui_server.py:278-289`).

### How a task is given (two steps, as the README describes)

1. **`task="planning"`, `text="get a xxx"`** (the README's "Action rule: Planning must precede action") calls `model.plan(text)` (`gui_server.py:461-466`).
   - `Optimus3Agent.plan` sends one text-only chat message, `"How to " + task + " from scratch?"`. It generates with the **plan expert** (`tasks=0`) and `max_new_tokens=512` (`src/model/agent/optimus3.py:184-196`).
   - It regex-parses `<answer>` into `step N: ...` strings (`optimus3.py:42-45`) and into goals `{count, item}` (`optimus3.py:48-56`). Any item containing "log" becomes `"logs"`.
   - In the planner's own benchmark data the prompt is `"How to obtain 1 <item> from scratch?"`. Step verbs in that data: `craft N x` 638, `dig down and mine N x` 160, `chop trees to get N logs` 105, `smelt N x` 71 (`R/datas/Optimus3/benchmark/plan/plan_103.json`).
2. **`task="action"`** with no text. **Each HTTP call advances the current subgoal by one "tick"** (`gui_server.py:471-503`). The client simply loops `send_text("action")` until the user pauses. It has no success check and no step limit (GUI `gui/main_window.py:217-220`).

### The per-tick loop (`_step`, `gui_server.py:314-386`)

The subgoal string is routed by substring:

- **`"craft" in subgoal`** runs `CraftWorker.crafting(item, count)` (`gui_server.py:319-334`).
  - This is a scripted JARVIS-style GUI macro. It places or opens the table, moves the cursor, clicks, and mines the table back (`MS/models/shell/craft_agent.py:189-290, 463-578`). Every action is a real `env.step`, taken hundreds at a time inside the call.
  - Afterwards it equips the best pickaxe (`gui_server.py:331-333`) and takes one noop step.
- **`"smelt" in subgoal`** runs `SmeltWorker.smelting(item, count)`, which is scripted in the same way (`gui_server.py:336-346`, `MS/models/shell/smelt_agent.py:122-`).
- **Anything else** (chop trees, dig down and mine) runs one action-head step (`gui_server.py:347-371`):
  - `env._only_once = True`. This arms ore injection for this step (see §2.4).
  - `action = agent.get_action(obs, subgoal)`, which calls `Optimus3ActionAgent.optimus3_action(cached_mllm_embed, obs["image"], task)` (`optimus3.py:126-148`).
  - drop, inventory, use and hotbar are all forced to 0.
  - If the subgoal contains **"dig down"**: jump, left, right, sneak and sprint are set to 0, **attack is forced to 1**, and on the first tick the code equips the best pickaxe and runs the scripted `_look_down()` (`gui_server.py:357-365`).
  - Whenever `attack > 0`, the movement keys are zeroed (`gui_server.py:367-369`).
- **Subgoal completion** uses `check_inventory(info["inventory"], goal.item, goal.count)`. It is a substring match, and one slot must hold at least `count` (`optimus3.py:17-29`). On success `sub_task_index += 1` and `model.task = None` (`gui_server.py:492-494`). After the last subgoal the response is `"success"`.
- **There is no timeout, no retry limit, no replanning and no reflection** in this loop. A failing subgoal repeats forever.
  - `model.reflection` exists (`optimus3.py:198-212`) but the server never calls it.
  - `model.router` (the Task Router) is loaded (`optimus3.py:78`) but never called. Expert selection is the hard-coded `task_type` label (`optimus3.py:171`; MoE forward uses `task_type[0]`, `modeling_optimus3.py:330-351`).

### How often the MLLM is called

| When | Call | What it is |
|---|---|---|
| Once per episode | `plan()`, one `generate` | Text only, up to 512 new tokens (`optimus3.py:193`). |
| Once per subgoal, including craft and smelt subgoals where it is unused | `model.reset(subgoal)` (`gui_server.py:479-480`) | One short `generate` with the **action** expert. The output is a single task token from `<dirt> <tree> <diamond> <gold> <iron> <seed> <cobblestone> <house> <craft> <redstone>` (ids 151665–151674, HF `added_tokens.json`), and `[:-10]` strips `<|im_end|>`. Then one forward pass, `get_action_embedding`, takes the last-layer hidden state (3584-d) at that token (`optimus3.py:93-124`, `modeling_optimus3.py:1054-1061`). It is cached for the rest of the subgoal. **No image is passed.** |
| Every env step | None | The action head alone: `mllm_embed_linear` (3584→7168→512), the STEVE-1 prior VAE (sampled, `deterministic=False`), then the VPT/STEVE-1 policy with classifier-free guidance. `cond_scale=6.0` gives batch 2 and the policy acts stochastically (`src/model/steve1/agent.py:179-270`, `MineRLConditionalAgent.py:54-63, 88-107`). |

Two further details of the action head:

- Each step also runs a MineCLIP text encode of the subgoal. It is used only to get the shape (`agent.py:252-254`).
- The recurrent state is **never reset between subgoals**. `Optimus3Agent.reset` does not touch `mine_policy.agent`, which is reset only in the constructor (`agent.py:210`).

### Headless operation

It can run headless, and no GUI client is needed.

- The Minecraft client is launched by `MS/simulator/minerl/env/launchClient.sh:36-41`. It uses `xvfb-run -a java ...` when the render device is `cpu`, which is the default. It uses `vglrun` only when `MINESTUDIO_GPU_RENDER=1` (`gpu_utils.py:41`).
- `xvfb-run`, `Xvfb` and Java 1.8.0_504 are already on the host.
- The GUI server is not required. The harness imports `Optimus3Agent`, `MinecraftSim` and the workers directly, and re-implements `_step` without the websocket frame pump.

---

## 2. Automated rollout or eval code: none for long-horizon tasks

I searched the whole repo: `scripts/`, `configs/`, `src/evaluation/`, and MineStudio's `benchmark/`, `inference/`, `online/` and `tutorials/`.

- `scripts/optimus3/eval/benchmark_generate.sh` and `benchmark_eval.sh` run `minecraftoptimus.evaluation.evaluator` over **static JSON benchmarks** (plan_103, grounding_500, reflection_64, caption_134, vqa_400). The judge is an LLM. No environment is involved (`configs/optimus3/eval/9B/coldstart/*.yaml`, `src/evaluation/evaluator.py`).
- `MS/benchmark/test_pipeline.py` and `task_configs/{simple,hard}` are MineStudio's own short VPT tasks (build, carve, etc., 200 steps). They have nothing to do with Optimus-3 or tech-tree items.
- A grep for `horizon|max_steps|success_rate|long-horizon` finds nothing Optimus-3-specific. The paper's Table 2 (long-horizon) has no released harness.

### 2.1 Minimal harness to write (`envs/optimus3_episode.py`, one task per process)

```text
setup:
  seed python random, numpy and torch with the episode seed
  (random_ore and the VAE prior are stochastic)
  env = MinecraftSim(obs_size=(128,128), preferred_spawn_biome="forest",   # as released
                     seed=SEED, inventory={},                             # empty
                     callbacks=[CommandsCallback(<the 8 gui_server commands>),
                                StepMonitor(target, horizon_steps)])
  obs, info = env.reset()
  helper = {craft: CraftWorker(env), smelt: SmeltWorker(env), equip: EquipWorker(env)}
  agent = Optimus3Agent(...)        # load once per process
  text, subgoals, goals = agent.plan(<instruction>)    # store the raw plan text
  i = 0; agent.task = None; look_down_once = False
loop:
  loop until the monitor raises Horizon / Success, or subgoals are exhausted:
    if agent.task is None: agent.reset(subgoals[i])     # as released, also for craft/smelt
    obs, info, done = _step(env, agent, obs, subgoals[i], goals[i], helper)   # copied verbatim
    if done: i += 1; agent.task = None; look_down_once = False
    if i == len(subgoals): keep stepping? → no: the released loop stops (response "success").
       Record native_success=True. If the target is not held, the episode ends as a
       failure at that step.
```

**StepMonitor** is a `MinecraftCallback` whose `after_step` sees **every** `env.step`, including the hundreds of steps taken inside `crafting()` and `smelting()`. It:

- sums the inventory by item over slots (`info["inventory"][slot] = {"type","quantity"}`) and applies `configs/task_targets.yaml`;
- logs the inventory each step, or only when it changes;
- raises an exception at the horizon or on success.

That exception is the only way to stop a macro mid-way. `CraftWorker` catches only `AssertionError` (`craft_agent.py:574`), so the exception must be a different class.

**What else the harness needs:**

- A subprocess and chain runner that reuses `envs/common.py` (`EpisodeMonitor`, `SEEDS`, `TASKS`, `write_result`).
- Per-episode `env.close()`, and a new `MinecraftSim` per episode.
- `end_reason` values: `success`, `horizon`, `plan_exhausted`, `method_exception`.

### 2.2 Seed

- `MinecraftSim(seed=...)` calls `self.env.seed(seed)` once in the constructor (`MS/simulator/entry.py:126,155`).
- The MineRL layer adds the seed to the mission token (`MS/simulator/minerl/env/_multiagent.py:662-663`) and **clears it after each reset** (`_multiagent.py:526-530`). This is the same code path we verified for Env O (DECISIONS D15).
- So either build a new `MinecraftSim` per episode, which is what `gui_server` does, or call `env.env.seed(seed)` before every `env.reset()`.

### 2.3 Biome and inventory

- **Biome:** `preferred_spawn_biome`, which becomes the `PreferredSpawnBiome` handler (`entry.py:128,152`; `human_survival_specs.py:71-74`). The released value is always `"forest"`. Our per-group biomes can be passed instead if we choose to deviate.
- **Initial inventory:** `inventory={}` is the default and becomes `InventoryAgentStart({})` (`entry.py:127,151`; `human_survival_specs.py:69-70`), which gives an empty inventory. `InitInventoryCallback` exists (`MS/simulator/callbacks/init_inventory.py`) but is not needed.

### 2.4 Environment behaviour to record

- **Ore injection.**
  - On every action-head step, `MinecraftSim.step` calls `random_ore(..., thresold=0.2)` (`entry.py:195-197`, armed by `gui_server.py:348`).
  - With probability 0.8 it `/setblock`s an ore 3–5 blocks below the feet. The bands are: coal 45–50; iron 26–43; gold 15–26 when the target y ≥ 17; redstone when the target y ≤ 16; diamond when y ≤ 14 (`entry.py:54-90`).
  - The rule is at most once per y-level (`ORE_MAP`), and there are no per-ore caps.
  - This is far more generous than Env O's 10 % per step. It uses unseeded Python `random`, so the harness must seed it.
- **No death-ends-episode.** `HumanSurvival` has no `DoneOnDeath` (`human_survival_specs.py:77-87`). With keepInventory and immediate respawn, death does not end the episode. Env O's environment does end it.
- **Crafting and smelting take env steps** inside the macros, as in Env O's JARVIS helper. So the horizon counts them.

---

## 3. Environment requirements

| Item | Finding |
|---|---|
| Python | `>=3.11`, pinned to 3.11 (`R/.python-version`, `pyproject.toml:6`). The shipped `.pyc` files are cpython-311. |
| Lock file | `uv.lock` is empty (7 lines, the project only). `requirements.txt` is mostly unpinned. The only pins are `sentence-transformers==4.1.0`, `qwen-vl-utils==0.0.11`, `minecraft_data==3.20.0`, `x_transformers==0.27.13.11`, `av==11.0.0`, `pyrender==0.1.45` and `pyglet==1.5.27`. `fastapi` and `uvicorn` are not listed; they are only needed for the GUI server. |
| transformers | The code imports `Qwen2_5_VLFlashAttention2`, `Qwen2_5_VLSdpaAttention`, `Qwen2_5_VLVisionFlashAttention2`, `Qwen2_5_VLVisionSdpaAttention` and `Qwen2MLP` from `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl` (`src/model/optimus3/modeling_optimus3.py:15-29`). These classes exist in 4.51.3 and 4.52.4 (checked in the wheels) and not before 4.49. The HF config says `"transformers_version": "4.51.3"`. **Pin transformers==4.51.3.** Newer versions refactor these classes away. This env cannot share mcagent's 4.44.2. |
| LLaMA-Factory | Imported at module load for `_register_composite_model` (`modeling_optimus3.py:7,1423-1429`). **llamafactory==0.9.3** has it and allows transformers 4.45–4.52.4 (checked in the wheel METADATA). 0.9.2 caps transformers at 4.49.0. Install it with `--no-deps` plus its needed deps, or pin accelerate ≤1.7, peft ≤0.15.2 and trl ≤0.9.6. |
| torch | No pin (`torch>=2.3.1`). The host GPUs are **RTX PRO 6000 Blackwell (sm_120)**, so torch must be ≥2.7 with cu128. mcagent has 2.9.1+cu128 with sm_120, which shows this works on the host. Use torch 2.7.1 or 2.8 with cu128 in a new env. |
| flash-attn | Hard-coded: `attn_implementation="flash_attention_2"` (`src/model/agent/optimus3.py:75`), and the README says `pip install flash-attn`. The code imports flash_attn only `if is_flash_attn_2_available()` (`modeling_optimus3.py:40-52`), and the attention class tables include `eager` and `sdpa` (`modeling_optimus3.py:59-63, 410-414`). The decoder has an explicit sdpa mask path (`:681, :722`). **So sdpa works with a one-token patch** (`"flash_attention_2"` → `"sdpa"`). Numerics differ slightly and the outputs are the same model. Building flash-attn for sm_120 is possible but risky; I recommend sdpa and recording it as a deviation. |
| Java / simulator | Java 8 is needed, and the host has it. MineStudio's engine is `mcprec-6.13.jar`. `check_engine()` looks for `$MINESTUDIO_DIR/engine/build/libs/mcprec-6.13.jar`. If it is missing, it **prompts on stdin** (`entry.py:106-117`), which blocks when run headless. Pre-download it with `python -m minestudio.simulator.entry -y`, which fetches HF `CraftJarvis/SimulatorEngine/engine.zip` (**458 MB**) (`entry.py:93-103, 325-332`). The fork's Python code (`MS/`, 436 MB with the repo) must be installed with `pip install -e MineStudio`, because it contains the Optimus changes (`random_ore`, `find_best_pickaxe`, workers). It does not need a Gradle build. |
| Weights | MLLM `iLearn-Lab/Optimus-3` (the preview, which the GUI README points to): 19.45 GB, 5 safetensors shards. Action head `MinecraftOptimus/Optimus-3-ActionHead`: 2.70 GB. It bundles MineCLIP, the prior, the VPT policy and the linear layer, so it needs no separate STEVE-1 files (`agent.py:302-316`). Task router: 0.45 GB. `efederici/sentence-bert-base`: 0.44 GB. The MineCLIP tokenizer is `openai/clip-vit-base-patch16`, tokenizer only. **Total about 23.1 GB + 0.46 GB engine.** |
| Which MLLM | Use `iLearn-Lab/Optimus-3` (preview). The ActionHead and Router were published the same day as it (2025-06-16). `Optimus-3-v2` (2026-03) is a later "coldstart_action" checkpoint for the MineSys2 static benchmark; it has `model_type "optimus3"` and bf16. It is not documented as paired with the ActionHead. |
| GPU memory | The MLLM is about 9.7 B params (28 layers, 14 of them MoE with shared + 5 experts; HF config), about 19.5 GB in bf16. The action head is about 2.7 GB in fp32. The README asks for 28–32 GB, and the server looks for a GPU with ≥40 GB free (`gui_server.py:249`). **Budget about 26–30 GB per worker.** Three free 96 GB GPUs can hold about 3 workers each; CPU and RAM for the MC clients is the real limit. |
| Coexistence | Fine in a **separate conda env** (py3.11, torch 2.7/2.8 cu128, transformers 4.51.3, llamafactory 0.9.3). It has no shared site-packages with mcagent. It needs its own `MINESTUDIO_DIR`, which otherwise defaults to `/tmp/MineStudio` (`MS/utils/temp.py:6-20`). Disk: env about 10 GB, weights about 24 GB. |

### Hard-coded paths that must be patched (the released code crashes otherwise)

- `MS/models/shell/craft_agent.py:469, 544, 643`: `root_path = "/data7/Users/xyq/developer/MinecraftOptimus/MineStudio/minestudio"`. The first one is outside the `AssertionError` try, so every craft raises `FileNotFoundError`.
- `MS/models/shell/smelt_agent.py:130`: `root_path = "MineStudio/minestudio"` is relative, so the cwd must be the repo root.
- Recipes ship only as `MS/assets/mc_recipes.zip` (859 files). The workers read `assets/recipes/<item>.json`, so the zip must be unzipped in place.
- `src/model/optimus3/modeling_task_router.py:11`: `SentenceTransformer("/data1/Models/sentence-bert-base")`.
- `gui_server.py:299-301, 579`: placeholder checkpoint paths and host. The harness does not use them.

---

## 4. Latency and call frequency

- Nothing in the code or README states per-step latency. The paper's "Layer Router" for skipping layers during action inference is **not in the released code**: a grep for layer-router or skip finds nothing, and `Optimus3DecoderLayer` always runs (`modeling_optimus3.py:418-505`). It would not matter anyway, because the MLLM is not called per step.
- Estimate from our Env O measurements (PROGRESS.md:10-11: MineRL headless about 16 env steps/s; STEVE-1 plus env about 12 steps/s):
  - The per-step cost here is the same kind of VPT-2x policy with CFG, plus a MineCLIP text encode. So expect **about 10–12 steps/s** for action-head steps, and about 16 steps/s inside the craft and smelt macros.
  - The MLLM costs about 10–30 s for the plan (≤512 tokens, HF `generate`, sdpa) plus under 1 s per subgoal. That is negligible.
  - Reset is about 40 s.
- Worst case per pass, one seed per task, every episode running to the horizon: horizons (tasks.yaml) × task counts = 1067 min ≈ **1.28 M steps ≈ 30–36 process-hours**. Successes shorten this.

---

## 5. Blockers, risks and recommendation

### Blockers, all fixable

| # | Blocker | Fix |
|---|---|---|
| B1 | No rollout harness | Write about 250 lines (§2.1). |
| B2 | Hard-coded `/data7` and `/data1` paths, relative smelt path, zipped recipes | 4 one-line patches, or symlinks, plus an unzip. |
| B3 | flash-attn on sm_120 | Use sdpa (a one-token patch). |
| B4 | transformers/LLaMA-Factory version window | Separate env with 4.51.3 + 0.9.3. |
| B5 | `check_engine` stdin prompt | Pre-download the engine. |

### Risks

- **Planner prompt format.** The code uses `"How to " + task + " from scratch?"`. The README says `get a xxx`, and the planner data uses `obtain 1 <item>`. Our instructions ("Craft a wooden pickaxe") are a different surface form, and parse failures give an empty plan, which means immediate failure.
  - Mitigation: a cheap **planner-only dry run** over the 70 instructions (about 15–30 min on 1 GPU, no Minecraft).
  - Then decide between the verbatim instruction (the default, the same as other methods) and README-style `get a <item>`. Record the choice in DEVIATIONS.
- **`max_new_tokens=512`** may truncate long plans such as diamond or armor. This is released behaviour; keep it.
- **No subgoal timeout** in the released loop. One unobtainable subgoal, such as a mis-planned `craft 1 ghast_tear`, eats the horizon. This is released behaviour.
- **Unfair-advantage caveats** for the separate table: scripted craft and smelt, forced-attack dig-down, 80 % ore injection, forest spawn, and no death termination. The table should state these. Optionally run a variant with Env O's biomes; the ore rule is part of their simulator, so keep it.
- **Compute:** GPUs and CPUs are busy now, so start after Stage A frees resources.

### Effort estimate (engineering)

| Step | Hours |
|---|---|
| New conda env, deps, engine download, weights download (about 24 GB, mostly network time), import smoke test | 3–4 |
| Path, sdpa and recipes patches (kept as a patch file under `baselines/optimus3/`) | 0.5–1 |
| Harness: episode runner with StepMonitor, horizon exception, per-step inventory log, seed handling, result JSON through `common.py`, chain runner or parallel launcher | 4–6 |
| Planner dry run over the 70 tasks, then 3 smoke episodes (wooden, stone, iron) and a fix loop | 2–4 (plus wall time) |
| **Total** | **about 10–15 h**, then about 9–12 h wall for a full pass with 3–4 workers |

**Recommendation: GO**, as a separate "Optimus-3 (as released, own simulator)" table, once Stage A frees GPU and CPU. Do the planner dry run first; it is cheap and is the main risk to results. Record transformers 4.51.3, sdpa, the path patches and the instruction format in DEVIATIONS.md, and list the environment advantages from §2.4 in the table caption.
