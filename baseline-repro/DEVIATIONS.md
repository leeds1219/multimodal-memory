# Deviations from the official code

Allowed change kinds: **(a)** LLM swap to gemini-3-flash-preview through the
shared layer (same prompts and message format, images included); **(b)**
version / compatibility fixes needed to run at all; **(c)** interface glue to
run a method in an evaluation environment. Upstream files are left untouched
wherever possible: glue lives in `baseline-repro/envs/` and patches the methods
at run time; the few in-place edits are listed with their patch files.

## Shared LLM layer (all methods)
| Kind | Where | Change | Why |
|------|-------|--------|-----|
| a | `llm/gemini_client.py` | Drop-in for `openai.OpenAI()`: model forced from `configs/llm.yaml`, every call logged (JSONL + images), bounded exponential backoff (6 retries), global $3,000 cap, per-episode anomaly guard | One planner backbone; everything logged; no runaway spend |
| a | same | `max_tokens` sent = method value + 16,384 thinking headroom | Gemini counts thinking tokens against max_tokens (DECISIONS D3) |

## Shared environment glue (all methods, both envs)
| Kind | Where | Change | Why |
|------|-------|--------|-----|
| c | `envs/common.py` `EpisodeMonitor` | Wraps the raw MineRL `step`: counts steps, writes a compact trajectory, checks the paper success criterion every step, saves the first frame | Uniform success criterion and logs (DECISIONS D12) |
| c | launchers | `env.seed(seed)` before reset, seeds from `configs/seeds.yaml`; python/numpy/torch seeded | Identical (task, seed) worlds across methods (D15) |
| c | launchers | One task per process; task text injected; group settings from the env's own benchmark yaml | Per-task resumable runs, custom task orders |

## MineEvolve (`MC-MineEvolve/`, upstream a0a5f9b)
| Kind | File | Change | Why |
|------|------|--------|-----|
| b | `MC-MineEvolve/docs/fixes.md` §1–4 | Inherited from branch `fix/mineevolve-env-and-runtime` (chat no-op fix, MineRL chat patch, STEVE-1 runner, POV transport) | Upstream does not run on MineRL 1.0.2 / current MineStudio |
| a | `planner/backends/openai_compat.py` | `self._client = GeminiClient()` | LLM swap; upstream tenacity retry (3 attempts) kept |
| a | `conf/llm/gemini_shared.yaml` | New LLM config, temperature 0.2 / max_tokens 1536 from upstream `gemini_flash.yaml` | Model id from shared config |
| b | env | STEVE-1 loaded through the official `steve1` package (installed from Shalev-Lifshitz/STEVE-1) with the shared weight files; MineCLIP package installed with its unused `dense_reward` import guarded (`baselines/mineclip_compat.patch`) | MineStudio not installed (disk); `mineclip/__init__` imports MineDojo |
| c | `envs/mineevolve_episode.py` | `time` → `GameTime` inside `mineevolve.main` (1 s = 20 ticks) | D23 |
| c | same | `_run_subgoal`, `_run_helper_subgoal`, `client.repair` raise `EpisodeEnd` once MineRL reported done | D24 |
| c | same | `_episode_succeeded` wrapped only to record the native verdict | D12 |
| c | Env M | `CraftHelper` → `envs/functional_craft.py` | Upstream helper is a stub (D11) |
| c | Env M | Raw env's `execute_cmd` hidden so chat commands take MineEvolve's own "chat action = env step" path | Optimus's MineRL build (D10) adds `execute_cmd`, which stock MineRL lacks |
| c | Env M | `PlainInventoryObservation` added to `MineEvolveBaseSpec` | Makes the released auto-pickaxe work (D19) |
| c | Env O (`envs/cross_glue.py:mineevolve_in_O`) | Optimus-1 env spec + `CustomEnvWrapper` under MineEvolve's wrapper (MineEvolve's own ore / pickaxe mixins off, reset commands issued once); craft/smelt → Optimus-1 GUI helpers; MineEvolve's view refreshed after helper actions | Run MineEvolve in Optimus-1's environment |

## Optimus-1 (`/home/rag/data/official/NeurIPS24-Optimus-1`; in-place edits in `baselines/optimus1/optimus1.patch`)
| Kind | File | Change | Why |
|------|------|--------|-----|
| a | `src/optimus1/models/gpt4_planning.py` | `client = GeminiClient()`; `model="gpt-4o"` → `"FROM_CONFIG"` (4 calls). Prompts, images, max_tokens=2000 unchanged | LLM swap |
| b | `app.py` | `retrieval` / `plan` routes: `while True` → `while retry < 10` (same bound as the reflection / replan routes) | Never retry forever |
| b | `src/optimus1/memories/memory.py` | `retrieve_plan`: an empty plan directory returns `(None, False)` like a missing one | Crashed (`extractOne` → None) on the empty-memory variant |
| b | env | torch 2.0.1 → 2.9.1+cu128; transformers pinned 4.44.2 | Blackwell sm_120; 5.x breaks the bundled DeepSeek-VL import |
| c | `envs/optimus_episode.py` | Config composed from the absolute conf dir, `main.__wrapped__(cfg)`, HydraConfig set manually; Optimus repo root on `sys.path` | Hydra cannot import `optimus1.conf` from the editable install; env ids are `src.optimus1...` |
| c | same | Per-chain working directory (`envs/optimus_workdir.py`): symlinks to app.py / checkpoints / helper, own memory dir, own step-image dir | Optimus-1 resolves memory, recipes and images relative to cwd; parallel chains must not share them |
| c | same | Janitor deletes step images Optimus-1 can never read again (keeps the lowest-step image per subtask name, which `_filter_task_obs` reads) | Disk (≈1 GB per long episode) |
| c | same | video / action pickles off | Optional recordings |
| c | same | `RuntimeError("Timeout!")` at the horizon treated as the episode end | D22 |
| c | Env M (`envs/cross_glue.py:optimus_in_M`) | MineEvolve env spec (+ `PlainInventoryObservation`, `IsGuiOpen` observations) with Env M group biome/horizon, wrapped in Optimus-1's `BasaltTimeoutWrapper`; wrapper step without `/kill` heuristics and Optimus ore rule, with Env M ore bands every step; chat commands as env steps; craft/smelt/equip via the functional primitive, failures reported in Optimus-1's `missing material: {...}` format | Run Optimus-1 in MineEvolve's environment |

## Optimus-1 full pre-built memory (HF `MinecraftOptimus/Optimus1_Memory`, used by the `prebuilt` variant)
| Kind | File | Change | Why |
|------|------|--------|-----|
| b | 4 of 386+ memory JSON files | Repaired with `baselines/optimus1/repair_memory_json.py`; originals kept in `official/optimus1_full_memory/corrupt_originals/`; report `baselines/optimus1/memory_repair_report.txt`. `plan/success/craft_crafting_table.json`: truncated last record dropped (71 of 662,842 chars). `plan/success/dig_down_and_break_down_cobblestone.json`: a duplicated tail after the first JSON document dropped. `reflection/chop_trees.json`: truncated last record dropped (8 chars). `reflection/dig_down_and_mine_diamond.json`: missing commas / a lost bracket from interleaved writes; rebuilt exactly from its 4,184 image-name pairs (names encode env and category). | The released files are invalid JSON; as released, Optimus-1 crashes when reflection retrieval opens one, or silently falls back to its example plan when a corrupt plan file is matched. |

## Optimus-3 (Stage C, own models; `/home/rag/data/official/Optimus-3` a73c013; in-place edits in `baselines/optimus3/optimus3.patch`)
Env "C3" = Optimus-3's own MineStudio fork + its own MLLM (`iLearn-Lab/Optimus-3` preview, rev 6168839) + action head + task router. No Gemini (kind a does not apply).
| Kind | File | Change | Why |
|------|------|--------|-----|
| b | env (`baselines/optimus3/setup_env.sh`, `constraints.txt`) | Separate conda env `optimus3` (py3.11): torch 2.7.1+cu128, transformers 4.51.3, llamafactory 0.9.3 (accelerate 1.7.0, peft 0.15.2, trl 0.9.6), numpy 1.26.4 | Blackwell sm_120; modeling code imports Qwen2.5-VL classes that exist only in the 4.49–4.52 window (feasibility §3) |
| b | env | `av==11.0.0` → 12.3.0 | 11.0.0 has no py3.11 wheel (sdist needs pkg-config/ffmpeg); only used by video-recording code paths |
| b | env | `cuda-python<13` (12.9.x) | MineStudio `gpu_utils.py` does `from cuda import cuda, cudart`, removed in 13.x → every env reset raised "An error occured in gpu_utils.py" |
| b | env | flash-attn not installed | No sm_120 build; see sdpa row |
| b | `src/minecraftoptimus/model/agent/optimus3.py:75` | `attn_implementation="flash_attention_2"` → `os.environ.get("OPTIMUS3_ATTN","sdpa")` | flash-attn unavailable for sm_120; sdpa path exists in the released modeling code; same weights, small numeric differences |
| b | `src/minecraftoptimus/model/optimus3/modeling_task_router.py:11` | `/data1/Models/sentence-bert-base` → `$OPTIMUS3_SBERT_DIR` (default = old path) | Hard-coded author path (router is loaded but never called in the released loop) |
| b | `src/minecraftoptimus/model/steve1/mineclip/mineclip/tokenization.py:31` | `/data7/.../mineclip` tokenizer → `$OPTIMUS3_MINECLIP_TOKENIZER` (= `openai/clip-vit-base-patch16` tokenizer files) | Hard-coded author path |
| b | `MineStudio/minestudio/models/shell/craft_agent.py:469,544,643`, `smelt_agent.py:130` | `root_path` `/data7/...` and cwd-relative `MineStudio/minestudio` → the package's own `minestudio/` dir | Hard-coded author path (every craft raised FileNotFoundError); recipes are already unzipped in the repo (`assets/recipes`, 858 files, git-tracked), so no unzip was needed |
| b | weights | `MINESTUDIO_DIR=/home/rag/data/official/optimus3_weights/minestudio_dir`, engine pre-fetched (`CraftJarvis/SimulatorEngine` engine.zip) | `check_engine()` otherwise prompts on stdin |
| c | `baselines/optimus3/run_episode.py` | Headless re-implementation of the GUI server loop (`gui_server.py` `/reset`, `/send_text planning`, repeated `/send_text action`); `_step` copied verbatim minus pause flag / frame pump / `step_hook` | Released code has no automated rollout; only an interactive GUI server |
| c | same | `MinecraftSim(obs_size=(128,128), seed=<seeds.yaml>, inventory={}, preferred_spawn_biome=<MineEvolve group biome>, callbacks=[CommandsCallback(8 reset commands), StepMonitor])`; new sim per episode; python/numpy/torch seeded | Same (task, seed) worlds as other methods; biome see DECISIONS C2 |
| c | same | `StepMonitor` callback: every `env.step` (incl. inside craft/smelt/equip macros) goes through the shared `EpisodeMonitor` (paper criterion, trajectory, keyframes 320×180/100 steps); raises a `BaseException` on success or at the horizon (MineEvolve horizons ×1200) | Workers catch only `AssertionError`; BaseException ends a macro mid-way |
| c | same | Planner input `"obtain <count> <target item id>"` (→ `"How to obtain 1 stone_pickaxe from scratch?"`) instead of our instruction text | DECISIONS C1 |
| c | same | Released loop has no stop: episode ends when the plan is exhausted (`plan_exhausted`; GUI answers "success"), when a subgoal has no parsed goal (`method_exception`; GUI would raise IndexError every tick), at success, or at the horizon | Harness termination |
| c | same | `Optimus3Agent._generate` / `model.generate` wrapped to log every MLLM call (prompt, response, tokens, latency, finish reason) to `llm_calls.jsonl` | Logging only; calls unchanged |
