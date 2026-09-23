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
