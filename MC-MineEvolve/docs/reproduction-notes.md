# Reproduction notes — MineEvolve (arXiv 2603.13131 v3) vs. this repo

Status as of 2026-09-18. Written so we do not forget which differences between the
paper and this codebase are ours, which are upstream's, and which are still unknown.

## What is the same as the paper

- Code is the upstream repo `xzw-ustc/MC-MineEvolve` at commit `a0a5f9b3` (2026-07-05),
  which is also upstream HEAD (5 commits total; nothing newer to pull). Prompts
  (`planner/prompts.py`, `inducer/prompts.py`, `adaptor/prompts.py`) and the
  Algorithm 1–3 loop are unmodified.
- Task suite: 70 MCU tasks in 7 groups, wooden = 11 tasks / 2-minute horizon
  (`conf/benchmark/*.yaml`, paper Table 3).
- Knowledge-retrieval prompt budget B = 512 tokens (`runtime.budget_tokens`, paper Eq. 7).
- The planner is text-only in the paper too: it receives "position, health, hunger,
  GUI status, inventory, task progress"; RGB frames go only to STEVE-1. Adding
  vision to the planner is therefore an *extension*, not part of the reproduction.
- Planner backbone `gemini-3-flash-preview` is the only "Gemini 3 Flash" the API
  exposes (plain `gemini-3-flash` does not exist).

## Differences we introduced on purpose (fixes needed to run at all)

| Change | Why | Where |
|---|---|---|
| `max_tokens` no longer hard-coded to 1024 (inducer/adaptor/planner) or 1536 (server default); `MINEEVOLVE_LLM_MAX_TOKENS`, 8192 for Gemini | Gemini 3.x are thinking models; hidden reasoning counts against `max_tokens`, so upstream's 1024 truncated the plan/repair JSON (`finish=length`, 38–297 visible tokens). Upstream HEAD still has the 1024. The paper states no output-token limit. | `planner/__init__.py`, `server/api.py`, `scripts/server_gemini.sh` |
| Stop the episode when `env.step` returns `done=True` | Upstream keeps repairing after the horizon; every subgoal then ends on step 1 and the repair loop never terminates (283 LLM calls before we killed it). | `main.py` |
| Knowledge store per model (`memories/<model>`) | Upstream shares `memories/run` across all runs; remedies induced by one model were retrieved into another model's initial plan. | `scripts/server_gemini.sh` (`MINEEVOLVE_MEMORY_PATH`) |
| `gemini_flash.yaml` → `gemini-3-flash-preview` | Upstream config says `gemini-2.5-flash`, but the paper says Gemini-3-Flash, and 2.5-flash returns 404 "no longer available to new users" for our key. | `conf/llm/gemini_flash.yaml` |
| Per-call LLM logging (`logs/llm_calls.jsonl` + full prompt/response dumps), numbered evidence dirs | Needed to compare call composition / cost across models and to judge what the planner could know from text. Does not change behaviour. | `planner/backends/openai_compat.py`, `util/evidence.py` |
| **`move` executor primitive** (`executor_hint: move`, `params: {yaw_deg, pitch_deg, steps, jump}`) + `moved` check | Upstream's only non-STEVE primitives are stationary (`mc_craft`/`mc_smelt`/`place`/`use`); every movement goes through STEVE-1, which deadlocks on some spawns (below). The paper says "action primitives" were controlled but lists none. `move` is a deterministic turn-then-walk the Adaptor can pick, matching the paper's mechanism (failure → remedy → executable repair, Curator `V_exec`) rather than a hidden auto-unstick. It is described to the LLM in the vocabulary the prompts already render. | `main.py` (`_move_script`), `util/vocab.py`, `planner/base.py` |
| Only `inv_ge` checks were evaluated | `path_clear` / `ypos_*` / `gui_closed` are advertised to the LLM but never checked, so a repair subgoal without an item target always timed out. `moved` is now evaluated; the others still are not. | `main.py` |

## Known mismatches we have NOT resolved

1. **World seeds** — *resolved 2026-09-18 with JARVIS-1's spawn table.* Upstream never
   sets a seed (the `env.seed` option exists but no config uses it), so every episode
   was a random world: face-to-trunk oak forest one run, dark forest / birch grove /
   hillside the next. The paper evaluates on a fixed "task–seed split" it does not
   publish — but JARVIS-1 (CraftJarvis), which the paper compares against "under the
   same split", ships `jarvis/assets/spawn.json`: 556 close-ended spawns
   `(seed, biome, player_pos)` incl. 93 `oak_forest`. We now run every task once per
   entry in `seeds:` (`conf/evaluate.yaml`; default = JARVIS-1's first three
   `oak_forest` spawns; `main.py` seeds the world, resets, then `/tp`s to `pos`), and
   each run is appended to `<hydra output dir>/runs.jsonl`. `scripts/fetch_spawns.py`
   lists more. Whether MineEvolve used exactly these entries is still unknown.
2. **Knowledge-base state of the main results is unspecified.** Table 4's numbers
   (e.g. Gemini-3-Flash + MineEvolve: Wooden 98.6 %, Overall 52.0 %) come with a
   controlled "evaluation-time LLM-call budget" but the paper never says whether the
   KB is empty, warmed up, or written online during that evaluation. Only the
   accumulation study (Sec. 4.4, Tables 6–7) is explicit: KB frozen after
   M ∈ {0, 50, 100, 200, 400} episodes; and App. D.5 shows "Cold Start (Empty KB)"
   on Diamond stays < 3 %. Our runs are cold start with online writing.
3. **STEVE-1 gets the full subgoal sentence.** Repairs produce 20–40-word conditions;
   the repo passes them verbatim to STEVE-1's CLIP text encoder (77-token limit,
   keyword-driven). The paper only says the "observation/action interface" and
   "action primitives" were controlled, not how long conditions are handled.
4. **Config names vs. paper models.** `gpt_5_5.yaml` → `gpt-4o`, `glm_4_7.yaml` →
   `glm-4-plus`, `gemini_flash.yaml` → `gemini-2.5-flash` (upstream). The public repo
   does not encode the paper's exact backbones.
5. **Environment stack.** Paper: 8× A40, unspecified MineRL/STEVE-1 build. Ours:
   MineRL 1.0.2 with our chat/allowCommands jar patch (`patches/`), MineStudio STEVE-1
   official weights, one RTX 2080 Ti per server. Not verifiable against the paper.
6. **Duplicate remedy induction.** Each failed subgoal triggers `induce_remedy`
   twice with the identical prompt (`/induce`, then `/repair` re-induces a temporary
   remedy). 2 of every 3 calls per failure are this pair. Upstream behaviour; left as is.

## STEVE-1 deadlock (why task 8 fails on some spawns)

Open-loop test of the loaded policy on real frames (2026-09-18, GPU 2, no LLM):

| Frame | condition | STEVE-1 output (60 steps) |
|---|---|---|
| hillside in view (clean Gemini run, step 1) | "chop a tree" | attack 56, camera 0 |
| same | "chop a tree", cond_scale 0 | attack 60, camera 0 |
| same | "explore, walk forward" / "turn around and run" | attack 60 / 59, camera 0 |
| open grass, trees ahead (smoke run) | "chop a tree" | forward 40, sprint 6 → walks to trees |
| same | "explore, walk forward" | camera moves 38/60 |

So the model and the wiring are fine; on a wall-filled view the visual prior wins over
any text condition, the targeted block is out of reach, the frame never changes, and
the loop never breaks. Text repairs cannot help; only a real movement can — hence `move`.

STEVE-1-only "chop a tree" (600 steps): on our hand-picked seeds 103/108/109 → 0 / 2 / 0
oak logs; on JARVIS-1's first three `oak_forest` spawns → 5 / 0 / 0 (spawn 3 never
moved). 1/3 either way ≈ the paper's 25.6 % STEVE-1-only Wooden baseline.

## Seed survey (prefer_biome: forest, task 8 needs *oak*)

| seed | spawn view | oak? |
|---|---|---|
| 101, 102, 105, 106, 111 | birch grove (birch_log ≠ oak_log) | no |
| **103** | oak tree 5 blocks ahead | yes |
| 104 | dark forest (dark_oak) | no |
| 107 | snow | no |
| **108** | oak trunk at left | yes |
| **109** | oak canopy, trunk visible | yes |
| 110 | sandstone pit | no |

Minecraft's "forest" biome mixes oak and birch, so even the right biome is often
unwinnable for an oak-specific task. Superseded by the JARVIS-1 `oak_forest` spawns
(above); plain integer seeds still work for ad-hoc runs.

## MCU vs. this repo's task protocol

The paper's 70 tasks come from MCU (Zheng et al., 2025 — shipped inside MineStudio,
`minestudio/benchmark/task_configs/`). MCU tasks are *scenario-initialised*: each yaml
runs `custom_init_commands` such as `/give @s minecraft:oak_log 4`,
`/summon minecraft:sheep ~2 ~ ~`, `/setblock ~2 ~ ~ minecraft:crafting_table`. The
MineEvolve paper (B.2) and this repo (`evaluate.yaml`: "NO /give") instead start from an
empty inventory with nothing placed — a strictly harder variant. Which protocol produced
Table 4 is not verifiable from the paper. Worth asking the authors.

## Reference numbers from the paper (for sanity checks)

- Eval-time LLM calls per episode ≈ 7.8–8.3 (Table 7, hard tasks); our clean Gemini 3
  wooden run: 8.
- STEVE-1 low-level-only baseline on Wooden: 25.6 % (Table 4) — i.e. even without a
  planner, "chop a log" succeeds 1 in 4 episodes. Weakest planner (DEPS) with
  Gemini-3-Flash: 84 %. If our per-task success stays near 0 over several seeds, the
  environment/STEVE-1 stack is the first suspect, not the LLM.
- Retrieved tokens 628–1216 per call at 50–400 episodes of accumulated knowledge.

## Our runs so far (task 8, "Chop an oak log from a tree", wooden, 2-min horizon)

| Date | Planner | Outcome | LLM calls | Notes |
|---|---|---|---|---|
| 09-17 | Qwen3.5-2B local | 0/1 | 17 (283 before the horizon fix) | dark-forest spawn; agent attack-only |
| 09-18 | gemini-3.6-flash | 0/1 | 4 | invalid: JSON truncated at max_tokens 1536 |
| 09-18 | gemini-3-flash-preview | 0/1 | 8 | invalid: plan contaminated by Qwen's remedies from shared store |
| 09-18 | gemini-3-flash-preview (clean, random seed) | 0/1 | 17 (plan 1, remedy 11, repair 5), 182k prompt tok, $0.16 | spawned facing a hillside; 6 subgoals, all timed out; **0.0 blocks moved in 2,392 steps**, 100 % attack regardless of repair text — the deadlock above |

## Next steps we agreed on

- Run task 8 across the 3 default seeds with the `move` primitive available (not yet done with an API model).
- Ask the authors: seed list, KB state for Table 4, action-primitive set, MCU init commands or empty start.
- Compare `gemini-3-flash-preview` vs `gemini-3.8-flash` on the same seeds:
  repair count and call composition (from `logs/llm_calls.jsonl`).
- Only then decide the local-model size for repeated experiments.
