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
| **`move` executor primitive** (`executor_hint: move`, `params: {yaw_deg, pitch_deg, steps, jump}`) + `moved` check | Upstream's only non-STEVE primitives are stationary (`mc_craft`/`mc_smelt`/`place`/`use`); every movement goes through STEVE-1, which deadlocks on some spawns (below). The paper says "action primitives" were controlled but lists none. Precedent: DEPS (`CraftJarvis/MC-Planner/controller.py`) runs scripted `look_to(deg)`, `jump`, `pillar_jump`, `go_surface`, `place_down`, `equip` next to its policy; JARVIS-1 has craft/smelt/equip scripts only. `move` is a deterministic turn-then-walk the Adaptor can pick, matching the paper's mechanism (failure → remedy → executable repair, Curator `V_exec`) rather than a hidden auto-unstick. | `main.py` (`_move_script`), `util/vocab.py`, `planner/base.py` |
| **Adaptor prompt rule 8** | Without it the Adaptor kept writing 30-word `stevei` conditions ("move backward 2 blocks and jump…") that STEVE-1 cannot follow and used `move` in only 2 of 5 repairs. Rule 8 says: movement → `move` primitive, then a ≤8-word `stevei` step. This is our only change to an upstream prompt. | `adaptor/prompts.py` |
| **`reasoning_effort=low`, `max_tokens` 16384 for Gemini** | Gemini 3's hidden thinking is unbounded: repair calls of 30–34 s hit `finish=length` even at 8192 and the JSON was cut, ending 2 of 3 episodes after one subgoal. `MINEEVOLVE_LLM_REASONING_EFFORT` is passed through as the OpenAI-compatible `reasoning_effort`. | `planner/backends/openai_compat.py`, `scripts/server_gemini.sh` |
| **GUI crafting controller** (`executor/gui_craft.py`, used by `mc_craft`) + per-slot inventory (`env/slot_inventory.py`, jar patch adds `index`) + `IsGuiOpen` observation + vanilla 1.16.5 recipe/tag JSON in `assets/` | Upstream's `CraftHelper` is a stub: `/playsound` + 10 s of inventory polling, no GUI, no craft action — every `mc_craft` subgoal failed in 0 steps, so 8 of 11 wooden tasks (and most of the 70) were unwinnable. MineRL 1.0.2 has no craft command on the Java side, so crafting must drive the inventory / crafting-table GUI with the mouse, exactly as JARVIS-1's and DEPS' scripts do (their slot-pixel tables and camera-to-cursor scale are the reference; the code is ours, JARVIS-1 has no license). Verified locally with `scripts/test_craft.py`: log → 12 planks → sticks → table → wooden pickaxe (3×3, table placed/opened/recovered), ~470 steps. This is infrastructure the paper says it "controlled" for all STEVE-1 baselines, not the paper's contribution. | `executor/gui_craft.py`, `executor/craft_helper.py`, `env/slot_inventory.py`, `env/custom_env.py`, `patches/`, `assets/` |
| **`mc_smelt` furnace controller + `equip` primitive** | Upstream `mc_smelt` was the same polling stub; ingots (iron ×14, gold ×7 tasks) were unreachable. `smelt()` places the agent's furnace, loads raw item + fuel (vanilla burn values), waits 200 ticks/item, takes the output and recovers the furnace with a pickaxe. `equip` moves a tool into the hotbar and selects it (a crafted pickaxe in the main inventory could never be held). The wrapper's auto-pickaxe mixin also expected an `obs["plain_inventory"]` MineRL never produced; it is now fed from the slot observation. Tested in `scripts/test_craft.py` (2 iron ingots, 597 steps). | `executor/gui_craft.py`, `env/mods/status.py` |
| Only `inv_ge` checks were evaluated | `path_clear` / `ypos_*` / `gui_closed` are advertised to the LLM but never checked, so a repair subgoal without an item target always timed out. `moved` and `ypos_le` / `ypos_ge` are now evaluated; `path_clear` / `gui_closed` still are not. | `main.py` |
| **Task success scoring** (`_episode_succeeded`) | Upstream matched the item id ("oak planks") as a substring of the goal ("craft an oak plank") → plural/singular mismatches scored finished tasks as failures (task 7 runs 1-2 ended with 4 oak_planks and were counted 0/3). Now the goal's object phrase is extracted ("smelt iron ore into iron ingot" → "iron ingot", "kill a cow to obtain leather" → "leather"), matched head-word + qualifiers, with the named quantity ("mine eight cobblestone"). Unit-tested on all 70 phrasings' shapes. | `main.py` |
| **`inv_ge` is an absolute post-state check** | Upstream compared against a baseline snapshot at subgoal start (a delta), while the vocabulary tells the planner "inventory has >= n" and the paper's `CheckSuccess(z_i, s_post)` is a post-state check. With the delta, a re-issued "chop an oak log" subgoal could never pass once the log was already in hand (task 7 run 3 looped on this). | `env/mods/task_checker.py` |
| **Extensions beyond the paper — OFF by default** | `MINEEVOLVE_EXECUTOR_ERRORS=1` adds the GUI helpers' failure reason ("missing oak_planks x1 (have 2, need 3)") to the planner state; the paper's Monitor only emits a failure *type*. A plank-arithmetic hint briefly added to the `mc_craft` description on 2026-09-19 was removed (recipe knowledge does not belong in the prompt). Runs that used either are labelled below. | `main.py`, `util/vocab.py` |
| Not scripted (3 of 70 tasks) | "Trade with a villager" (villager + trade GUI), "Wash a leather chestplate in a cauldron" (cauldron + water), "Repair an iron helmet at a smithing table" (another GUI) have no primitive; they will fail unless STEVE-1 does them on its own. | — |

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
2. **Main-table numbers are the *accumulated-KB* regime, not cold start.** The paper
   never states it, but the numbers do: Table 6 (KB frozen after M episodes)
   gives MineEvolve Iron/Redstone/Diamond/Armor = 39.8/24.2/9.0/16.2 at 0 eps and
   53.4/31.0/15.8/24.6 at 400 eps, and Table 4's Qwen3.5-Flash row is
   52.43/30.77/14.58/23.47 — i.e. ≈ the 400-episode checkpoint, 10 pp above cold
   start. At 0 eps MineEvolve equals Static Store and Text Reflection (no knowledge →
   plain planner + repair). Our runs are cold start (a few skills/remedies at most).
   A warm-up of hundreds of episodes would be needed for a like-for-like comparison
   (~$0.02–0.1 per wooden episode on Gemini 3 Flash); the warm-up protocol is only
   described for the hard groups. Wooden is where the KB matters least (DEPS, no
   memory, reaches 84 % with Gemini-3-Flash), so our wooden gap is mostly executor/spawn.
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
| 09-18 | gemini-3-flash-preview, JARVIS-1 spawn 1, `move` available, no prompt rule | 0/1 | 15 | STEVE-1 chopped a log but the drop was out of reach; Adaptor used `move` in 2 of 5 repairs (both succeeded their `moved` check) but paired them with long `stevei` text again |
| 09-18 | gemini-3-flash-preview, task 7 "Craft an oak plank", cold start (frozen KB), 3 spawns | **0/3**, 127 calls, $1.0 | spawn 1 had the log at step 123; `mc_craft` failed in 0 steps (stub), Adaptor cycled for the horizon. Spawns 2/3: STEVE-1 got no log (variance). Led to the GUI crafting controller above. |
| 09-18 | gemini-3-flash-preview, task 8, 3 JARVIS-1 spawns, `move` + rule 8 + thinking cap + per-subgoal goal check | **3/3**, 10 calls, $0.05 | 321 / 783 / 662 steps |
| 09-18 | gemini-3-flash-preview, task 7 "Craft an oak plank", cold start, real `mc_craft` | scored 0/3, **re-scored 2/3** | runs 1-2 crafted 4 oak_planks (815 / 646 steps) but the old goal matcher missed the plural; run 3 held a log yet looped on re-issued chop subgoals (delta check). Both scoring bugs fixed after this run. |
| 09-18 | gemini-3-flash-preview, 3 JARVIS-1 spawns, `move` + rule 8, max_tokens 8192 | **1/3** | 15 total, 90k prompt tok, $0.08 | spawn 2 **success** (chop → `move` 20 steps onto the drop → "chop oak_log", 949 steps). Spawns 1 and 3 lost to truncated repair calls (30–34 s of thinking) → `llm_failed` → episode ended after one subgoal |

## Wooden tier, cold start — full result (2026-09-18/19)

Gemini 3 Flash (`reasoning_effort=low`), frozen empty KB, JARVIS-1 `oak_forest`
spawns (3 per task), paper horizon 2 min, all primitives above. Paper (Table 4,
accumulated KB): Wooden 98.6 %; STEVE-1-only 25.6 %.

| task | result | notes |
|---|---|---|
| 8 Chop an oak log | 3/3 | |
| 10 Punch a tree to gather wood | 3/3 | |
| 7 Craft an oak plank | 3/3 | |
| 6 Craft a crafting table | 3/3 | |
| 5 Craft a stick | 2/3 | run 2: log obtained at step ~2300, horizon hit mid-craft |
| 9 Collect a sapling | 0/3 | STEVE-1 hits trunks; sapling is a ~5 % leaf drop |
| 0 Wooden pickaxe | 0/3 (two batches: 0/6) | see below |
| 1 Wooden axe | 0/3 (0/6) | |
| 2 Wooden shovel | 0/3 (0/6) | |
| 3 Wooden hoe | 0/3 (0/6) | |
| 4 Wooden sword | 0/3 (0/6) | |
| **total** | **14/33 = 42 %** | |

Why the five tool tasks fail: a wooden tool from an empty inventory needs 3 logs
(9 planks: 4 table + 2 sticks + 2-3 head) and 4-5 GUI crafts (~65 steps each, the
table one ~250). STEVE-1 here collects roughly one log per 500 steps, so gathering
alone eats 1,500 of the 2,400 steps; add repairs and the horizon runs out — many
runs ended one craft short with table + planks + sticks in hand. The planner also
under-counts planks (plans 8, needs 9) on the first attempt; with the new
`last_executor_error` feedback it repairs correctly but has no time left. The
first tool batch had reported 15/15 — all false positives from the "wood"
substring rule (fixed; every success is now re-verified against the final inventory).

Provenance of the batches: tool batch 1 (false positives) and batch 2 (0/15) ran under paper
conditions except that batch 2 had the executor-error extension ON (now default off);
the 6-minute diagnostics (non-paper horizon) had it ON as well, and the second one also
carried the plank-arithmetic hint. None of the 14 wooden successes depended on either.

Interpretation: under the paper's *stated* conditions (empty inventory, 2 min,
cold start) this stack solves the single-step wooden tasks but not the 4-craft
chains. The paper's 98.6 % therefore implies a faster gatherer (their STEVE-1
interface / spawns) and/or accumulated skills that front-load "chop 3 logs"; the
6-minute diagnostic below separates horizon from pipeline.

## Next steps we agreed on

- Run task 8 across the 3 default seeds with the `move` primitive available (not yet done with an API model).
- Ask the authors: seed list, KB state for Table 4, action-primitive set, MCU init commands or empty start.
- Compare `gemini-3-flash-preview` vs `gemini-3.8-flash` on the same seeds:
  repair count and call composition (from `logs/llm_calls.jsonl`).
- Only then decide the local-model size for repeated experiments.
