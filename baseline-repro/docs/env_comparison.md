# Env M (MineEvolve) vs Env O (Optimus-1): environment comparison

Paths used in citations:

- **M** = `/home/rag/data/multimodal-memory/MC-MineEvolve` (branch `experiment/baseline-repro`). `src/` means `src/mineevolve/`.
- **O** = `/home/rag/data/official/NeurIPS24-Optimus-1`. `src/` means `src/optimus1/`. The MineRL fork lives in `minerl/`, and its MCP-Reborn tree in `minerl/minerl/MCP-Reborn/`.
- **UP** = upstream `minerllabs/minerl` at tag `v1.0.2`. I shallow-cloned it for the diff and deleted the clone afterwards.

This is a read-only code analysis. Nothing was built or run. When a behaviour is inferred from reading code rather than observed, the text says so.

---

## 0. Summary table

| # | Item | Env M (MineEvolve) | Env O (Optimus-1) | Same? |
|---|------|--------------------|-------------------|-------|
| 1 | MineRL / MCP build | Stock MineRL 1.0.2 from GitHub, plus `patches/minerl-1.0.2-chat-commands.patch`, which adds a `chat` verb and sets `allowCommands=true`. The jar is rebuilt by `scripts/patch_minerl.sh`. | A MineRL fork (1.0.2 plus later upstream fixes) with a prebuilt jar. Its `EnvServer.java` adds a `chat` verb, sets `allowCommands=true`, and makes the inventory JSON report all 36 slots with `slot_id` and `"none"` for empty slots. It also runs `xvfb-run` inside `launchClient.sh`. | Chat and commands are functionally the same. The inventory JSON is different. O's Python code **needs** O's jar. M's code works on O's jar. |
| 2 | Reset commands | 8 commands: sendCommandFeedback off, commandBlockOutput off, keepInventory, night vision, doDaylightCycle off, time 0, doImmediateRespawn, /spawnpoint | The same 8 commands, in the same order | Yes |
| 3 | Initial inventory | `[]`, and the code enforces it with a ValueError | `[]` in every YAML. A non-empty list is allowed (`CustomInventoryAgentStart`). | Yes, as configured |
| 4 | Biome | wooden = forest. All other 6 groups = plains. | wooden, gold, redstone, diamond, armor = forest. stone and iron = plains. | **No**, for gold, redstone, diamond and armor |
| 4b | World seed | There is an `env.seed` key, but it goes into `generatorOptions`, which the Java side **ignores**. Every world is random. | Not configurable. Every world is random. | Both random. M's seed knob is dead. |
| 5 | Ore spawning | 10% chance per step. Places the ore 3–5 blocks below the feet with `/setblock`. Bands: coal 45–50, iron 26–43, gold 17–26, **redstone 1–16 then diamond 1–14** (first match wins). Each ore has a per-episode cap. | 10% chance per STEVE-1 step. Same offset. Coal 45–50, iron 26–43, gold (14,26] with target ≥17, **redstone only when y is 15–21**, **diamond for every y ≤14**. No caps. | **No**. In M, diamond ore almost never spawns (see §5). |
| 6 | Auto-equip / automatic actions | Auto-pickaxe below y=70 is **dead code**, because `plain_inventory` is not observed. No stuck-kill. | Auto-pickaxe below y=70 works. `/kill` after 8000 steps at the same y. `/kill` if the goal is iron_ore and y<25. After a kill, the agent explores for 100 steps. | **No** |
| 7 | Craft / smelt / equip | `CraftHelper` is a **stub**. It sends a `/playsound`, sleeps, and polls the inventory. **Nothing is ever crafted or smelted.** `place` and `use` are only inventory checks. | JARVIS-style scripted GUI crafting, smelting and equipping. It places the crafting table or furnace, moves the cursor, clicks, and mines the block back. All of this takes env steps. | **No**. This blocks everything in M. |
| 8 | Horizon (minutes; 1 min = 1200 steps) | W 2, S 3, I 20, G 10, R 15, D 30, A 25 | W 2, S 5, I 20, G 30, R 30, D 30, A 30 | **No** for S, G, R, A |
| 8b | Task lists | 70 tasks: 11 / 10 / 16 / 7 / 6 / 7 / 13 | 73 YAML entries: 12 / 10 / 17 / 7 / 7 / 7 / 13. That is 67 craft tasks plus 6 "mine" entries. | Only diamond is identical. See §8. |
| 9 | Success | Episode success is a substring heuristic over the final inventory. It gives many false positives and some false negatives. | Episode success means every subgoal in the LLM plan completed and no timeout occurred. The target item itself is not checked. | **No** |
| 10 | Obs / action | 640×360, FOV 70, gamma 2, GUI scale 1. The POV is sent as lossless PNG and resized to 128×128. MineStudio STEVE-1 with cond_scale 4.0. Recurrent state resets when the subgoal text changes. | The same camera settings. The POV is sent as **JPEG** and resized to 128×128. Original `steve1` package with cond_scale **6.0**. State resets only once per episode. | Camera the same. The STEVE-1 wrapper differs. |
| 11 | "Overall" | Task-count-weighted mean over all 70 tasks (Easy = W+S+G, Hard = I+R+D+A). Stated in `docs/tasks.md`; no code computes it. | The README says "average on the five groups Iron, Gold, Diamond, Redstone, Armor". No code computes it. | **No** |
| 12 | Death / time / mobs | DoneOnDeath (death ends the episode). Time frozen. Mobs on. Difficulty HARD. | The same. But O's own `/kill` calls also end the episode because of DoneOnDeath. | Mostly the same |

**Blocking or silent bugs:**

- **M-B1:** Crafting and smelting are no-ops.
- **M-B2:** Auto-pickaxe is dead.
- **M-B3:** The episode success heuristic is wrong for many tasks.
- **M-B4:** The seed setting is ignored.
- **M-B5:** The repair loop is unbounded after a death or timeout.
- **M-B6:** Diamond ore is practically never spawned.
- **O-B1:** `env.close()` sits inside the per-task loop. Multi-task runs probably break.
- **O-B2:** The `/kill` heuristics end the episode under DoneOnDeath.
- **O-B3:** `planning = example` can hit an `UnboundLocalError`.
- **O-B4:** Relative paths and an entry point that force the working directory to be the repo root.
- **O-B5:** `pre_open_tabel` can hit an `UnboundLocalError`.

Details are in §13.

---

## 1. MineRL / MCP-Reborn build

### 1.1 What upstream MineRL 1.0.2 builds

- `minerl/scripts/setup_mcp.sh:6-8` clones **`https://github.com/Hexeption/MCP-Reborn.git`** and checks out **`1.16.5-20210115`**. It then runs `./gradlew setup` (line 10), which decompiles Minecraft 1.16.5. It is not `minerllabs/MCP-Reborn`. MineRL's own Java changes, including all of `com/minerl/multiagent/env/EnvServer.java`, come from `scripts/mcp_patch.diff`, which `patch_mcp.sh` applies.
- O's `setup_mcp.sh` and `mcp_patch.diff` are **byte-identical** to upstream v1.0.2 (checked with `diff`).
- In upstream `EnvServer.java` (extracted from `mcp_patch.diff`):
  - The world is created with `allowCommands=false`: `new WorldSettings(..., Difficulty.HARD, false, ...)`.
  - Action lines are parsed only as `<key> <0|1>`, `camera dx dy` or `dwheel`. There is no `chat` verb, so a line such as `chat /gamerule` hits `Integer.parseInt("/gamerule")` and crashes (as M's `docs/fixes.md` §2 describes).

### 1.2 What Env M expects

- Stock MineRL 1.0.x, pip-installed from GitHub (`docs/fixes.md`, the "Environment-level workarounds" section). On top of that, `patches/minerl-1.0.2-chat-commands.patch` is applied to the installed `MCP-Reborn`, and `scripts/patch_minerl.sh` rebuilds `mcprec-6.13.jar` with `./gradlew shadowJar`.
- Patch contents (`patches/minerl-1.0.2-chat-commands.patch`):
  - `allowCommands=true` (lines 7-9).
  - `execChatActions()` sends each `chat <text>` line with `mc.player.sendChatMessage` **on the client thread**, via `mc.execute` (lines 17-39, 42).
  - The keyboard and mouse parsers skip `chat` lines (lines 51, 60).
- The Python side uses stock MineRL. `env.action_space.noop()` plus `action["chat"]=cmd` is stepped through the gym wrappers (`src/env/wrapper.py:324-331`). Stock `spaces.Text.no_op()` does not accept `batch_shape`, which is why M adds `_ChatText` (`src/env/chat_action.py:16-28`).
- Observations: M's spec (`src/env/custom_env.py`) does not override `create_observables`. It therefore gets only the HumanSurvival observables (UP `minerl/herobraine/env_specs/human_survival_specs.py:21-43` and `human_controls.py:56-60`): `pov`, flat `inventory`, `equipped_items`, `life_stats`, `location_stats` and the stat counters. There is **no `plain_inventory`**. `isGuiOpen` is a *monitor*, so it arrives in `info`, not in `obs` (UP `_multiagent.py:233-236`).

### 1.3 What Env O expects (its MineRL fork)

Python differences from UP v1.0.2, found with `diff -r` and ignoring whitespace:

| File | Change |
|---|---|
| `minerl/minerl/env/_multiagent.py:265-266` | An empty `chat` value is skipped, so no `chat ` line is sent. |
| `minerl/minerl/env/_multiagent.py:~771-786` | `find_ip` uses `instance.port` directly instead of the `<Find>` handshake. |
| `minerl/minerl/env/_singleagent.py:35-41` | New `execute_cmd(cmd)`. It builds a noop with `chat=cmd` and calls `self.step`. |
| `minerl/minerl/herobraine/hero/spaces.py:454, 471` | `Text.no_op(batch_shape=None)` and `Text.sample(batch_size=None)`. This is the same fix as M's `_ChatText`. |
| `minerl/minerl/env/malmo.py` | `MINERL_TMP_INSTANCES` temp-dir instances. This change also exists in later upstream code. |
| `minerl/setup.py` | Honors `GRADLE_USER_HOME` and sets version `1.0.2`. The rest is cosmetic (black formatting, wiki URLs). |

**Custom MCP-Reborn source tree compared with UP-patched MCP.** Method: I copied the 161 files that `mcp_patch.diff` touches out of O's tree and ran `patch -R --dry-run`. Every hunk reverses cleanly except two files:

1. `launchClient.sh:52` wraps the launch in `xvfb-run -a java -Xmx$maxMem -jar $fatjar --envPort=$port`.
2. `src/main/java/com/minerl/multiagent/env/EnvServer.java`:
   - `:423`: `allowCommands = true` in `WorldSettings`.
   - `:564-583`, `getInventoryJson()`: emits **all 36 main-inventory slots**, each with `"slot_id"`, and uses `type:"none", quantity:0` for empty slots. Upstream emits only non-empty stacks and has no slot ids. O's `PlainInventoryObservation` depends on this (`src/env/plain_inventory.py:26-35` reads `item["slot_id"]`).
   - `:608, 616-625`: `doChatActions()` sends every `chat <msg>` line with `Minecraft.getInstance().player.sendChatMessage(msg)`. It is called directly from the env-server thread, not through `mc.execute` as in M's patch.
   - `:630`: the keyboard parser skips `chat` **and `equip`** lines. `:661`: the mouse parser skips `chat` lines.
   - Everything else is identical: `done` logic (`:516`), seed handling (`:249, 444`), biome predicate (`:415`).
   - No other patched file differs. The 177 `com/microsoft/Malmo/Schemas/*.java` files carry a 2024-10-18 mtime only because they are generated by xjc at build time. Vanilla Minecraft sources that `mcp_patch.diff` does not touch could not be diffed without a vanilla decompile, but their mtimes all match the bulk extraction date (2024-09-13).
- The prebuilt jar `build/libs/mcprec-6.13.jar` (2024-10-18) contains these changes. `strings` on `EnvServer.class` shows `doChatActions`, `sendChatMessage` and `slot_id`.

**Can O's build run M's chat commands natively?** Yes. `/gamerule`, `/effect`, `/time`, `/spawnpoint` and `/setblock` all go through `sendChatMessage` with cheats enabled. In the integrated server the single player is an operator when `allowCommands=true`. M's Python code also works on O's MineRL:

- `getattr(self.env, "execute_cmd")` resolves to O's `_SingleAgentEnv.execute_cmd` through `gym.Wrapper.__getattr__`.
- `FlatInventoryObservation` ignores the `"none"` stacks through its `KeyError` path (UP `inventory.py:57-64`).

**Do not apply M's patch on top of O's MCP tree.** The hunks will not apply cleanly, and if forced, every chat line would be sent twice.

**The reverse does not work.** O's Python code on M's patched stock MineRL crashes, because `PlainInventoryObservation.from_hero` needs `slot_id` (`src/env/plain_inventory.py:26-35`, `KeyError`).

Minor difference: M's `execute_cmd` steps through the gym `TimeLimit` wrapper, so command steps count toward the horizon. O's `self.env.execute_cmd` resolves to the innermost `_SingleAgentEnv.step` and bypasses both `TimeLimit` and `BasaltTimeoutWrapper`, so command steps are free.

---

## 2. Reset commands

| | Env M `src/conf/evaluate.yaml:52-60` | Env O `src/conf/evaluate.yaml:47-55` |
|---|---|---|
| 1 | `/gamerule sendCommandFeedback false` | same |
| 2 | `/gamerule commandBlockOutput false` | same |
| 3 | `/gamerule keepInventory true` | same |
| 4 | `/effect give @a night_vision 99999 250 true` | same |
| 5 | `/gamerule doDaylightCycle false` | same |
| 6 | `/time set 0` | same |
| 7 | `/gamerule doImmediateRespawn true` | same |
| 8 | `/spawnpoint` | same |

- M issues them in `src/env/wrapper.py:229-234`, after `env.reset()` at line 223. Each one is an env step through TimeLimit.
- O issues them in `src/env/wrapper.py:119-124` with `self.env.execute_cmd`, so the steps are not counted.
- Both envs also set `TimeInitialCondition(allow_passage_of_time=False)` and `SpawningInitialCondition(allow_spawning=True)`: M in `src/env/custom_env.py:73-74`, O in `src/env/custom_env.py:118-119`.

## 3. Initial inventory

- **M:** every benchmark YAML has `initial_inventory: []` (line 9). The spec always passes `StrictEmptyInventoryStart([])` (`src/env/custom_env.py:85`), which raises on a non-empty list (`src/env/inventory_agent_start.py:26-31`). The YAML value is never read.
- **O:** every YAML has `initial_inventory: []` (line 9), with commented-out examples. It is passed to `CustomInventoryAgentStart` (`src/env/__init__.py:69`, `src/env/custom_env.py:88`), which requires a `slot` key per item.
- Result: both start empty.

## 4. Biome and world seed

**Preferred biome.** Both use `handlers.PreferredSpawnBiome` (M `src/env/custom_env.py:86`, O `src/env/custom_env.py:89`). On the Java side this becomes `setSpawnBiomePredicate(category == biome)` (`EnvServer.java:415`).

| Group | M `prefer_biome` (YAML line 10) | O `prefer_biome` (YAML line 12; wooden line 14) |
|---|---|---|
| wooden | forest | forest |
| stone | plains | plains |
| iron | plains | plains |
| gold | plains | **forest** (`golden.yaml:12`) |
| redstone | plains | **forest** |
| diamond | plains | **forest** |
| armor | plains | **forest** |

O asserts that the biome is in `ALL_BIOMES` (`src/env/__init__.py:64`). M relies on the Java check (`checkValidBiome`).

**World seed.**

- The Java side takes a seed from only two places:
  1. The 6th `:`-field of the mission token, which `env.seed(int)` sets. It must be called **before every** `reset()`, because reset clears it (UP `_multiagent.py:136-152, 461-465, 599-600`).
  2. `<WorldSeed>` inside `AgentStart` (`EnvServer.java:444`). No Python handler for this exists in MineRL 1.0.2.
- Otherwise the seed is `new Random().nextLong()`.
- **M:** `benchmark.env.seed` (`src/env/__init__.py:45`) is serialized into `DefaultWorldGenerator(generator_options=json.dumps({"seed": ...}))` (`src/env/custom_env.py:61-67`). `generatorOptions` is **never read** by `EnvServer` (grep shows it only in the generated `Schemas/DefaultWorldGenerator.java`), so the seed is silently ignored. None of the YAMLs set it anyway. Worlds are random.
- **O:** there is no seed option. `DefaultWorldGenerator(force_reset=True)` (`src/env/custom_env.py:105`). Worlds are random.
- **To make seeds work in either env:** call `env.seed(s)` on the underlying MineRL env before each `reset()`, or add a custom AgentStart handler that emits `<WorldSeed>s</WorldSeed>`. Ore spawning uses Python's unseeded `random` in both envs, so it also needs `random.seed`.

## 5. Ore spawning

Both envs place ore with `/setblock ~ ~{dy} ~ minecraft:<ore>`, where `dy = randint(-5,-3)`. That is directly below the agent, 3–5 blocks under its feet.

**Env M** (`src/env/wrapper.py:66-120`, called on **every** wrapper `step` at `:292`):

- Chance per step: `random.random() > 0.10` returns without spawning (line 98), so 10% of steps spawn.
- The step is skipped if the current y is already a key of `spawned_at_y`. Only target y values are recorded (lines 100, 116).
- Bands are checked in order. **Both** the agent's y and the target y must lie inside the band, and a per-episode count cap applies (lines 108-118):

  | Ore | y_min–y_max | Cap |
  |---|---|---|
  | coal_ore | 45–50 | 6 |
  | iron_ore | 26–43 | 17 |
  | gold_ore | 17–26 | 10 |
  | redstone_ore | 1–16 | 12 |
  | diamond_ore | 1–14 | 14 |

- **Consequence:** for y ≤ 14 the redstone band always matches first. **Diamond ore can only spawn after 12 redstone ores have been spawned.** Each spawn needs a distinct target y, and there are only 16 possible values, so diamond placement is practically unreachable. The effective y ranges are coal 48–50, iron 29–43, gold 20–26 (y=26 gives gold), and redstone 4–16.
- There is no ore spawning while a helper runs, because M's helpers do not step the env.

**Env O** (`src/env/wrapper.py:14-51`). It only runs when `_only_once` is set, which `main.py:176` does before every STEVE-1 step (`wrapper.py:188-190`). It never runs during scripted helpers.

- Chance: `prob <= 0.9` returns without spawning (lines 15-17), so 10% of steps spawn.
- The step is skipped if the current y **or** the target y is already used (the current y is recorded too).
- Branches (an if/elif chain on the current y):

  | Ore | Condition | Source lines |
  |---|---|---|
  | coal | 45 ≤ y ≤ 50 and target ≥ 45 | 20-26 |
  | iron | 26 ≤ y ≤ 43 and target ≥ 26 (y=26 gives nothing) | 27-32 |
  | gold | 14 < y ≤ 26 and target ≥ 17 | 34-39 |
  | redstone | 14 < y ≤ 26, gold branch failed, target ≤ 16 (effectively y 15–21) | 40-44 |
  | diamond | y ≤ 14 and target ≥ 1: **every** spawn at y ≤ 14 is diamond | 45-51 |

- The "max: N" comments are **not enforced**, so there are no caps.

**Net difference.** Gold and iron behave the same. Redstone and diamond are swapped in priority. O guarantees diamond ore at y ≤ 14 about every 10 steps. In M, digging to y ≤ 14 mostly yields redstone and almost never diamond. Expect diamond-group success in Env M to be much lower, and redstone mining much easier.

## 6. Auto-equip and other automatic actions

**Auto-pickaxe.** Both envs force `hotbar.*`, `use` and `inventory` to 0 during STEVE-1 steps and then press the hotbar key of the best pickaxe (diamond > iron > stone > wooden) in slots 0–8 when y < 70.

- **M:** `src/env/wrapper.py:246-249, 132-151, 266-272`. It reads `status_mod.plain_inventory()`, which is filled from `obs["plain_inventory"]` (`src/env/mods/status.py:72-74`). **That key does not exist in M's observation space** (§1.2), so the function always returns `None`. **Auto-equip never happens in Env M.**
- **O:** `src/env/wrapper.py:153-161, 261-294`. It uses `plain_inventory` from `PlainInventoryObservation` (`src/env/custom_env.py:71`), so it works.

**Other automatic actions.**

| Action | Env M | Env O |
|---|---|---|
| Drop key | Forced 0 (`wrapper.py:252`) | Forced 0 (`wrapper.py:165`) |
| While attacking | Zeroes jump, left, right, sneak, sprint (`:253-256`) | Only in `raw_step` (`:134-136`). **Not** in the `step` path STEVE-1 uses. |
| Stuck recovery | None; deliberately removed (`wrapper.py` docstring, lines 17-26) | `/kill` when a single y level has more than 8000 cumulative steps, then 100 steps of the prompt `explore to find {goal}` (`wrapper.py:178-186`, `main.py:171-175`) |
| Iron-ore guard | None | `/kill` whenever the goal contains `iron_ore` and y < 25 (`wrapper.py:200-209`) |
| Kill interaction with DoneOnDeath | n/a | Both `/kill` paths end the episode (§12, O-B2) |
| Pickaxe inside craft helper | n/a | `pre_open_tabel` also equips the best pickaxe to hotbar slot 1 when y < 50 and the agent cannot jump, then mines around (`helper/jarvis_craft_helper.py:120-194`) |

## 7. Craft / smelt / equip helpers: what is scripted rather than STEVE-1

**Env M.** The planner emits an `executor_hint` for each subgoal. `stevei` goes to STEVE-1. `mc_craft`, `mc_smelt`, `place` and `use` go to `_run_helper_subgoal` (`src/main.py:400-409`), which calls `CraftHelper` (`src/executor/craft_helper.py`):

- `mc_craft` / `mc_smelt`: `_open_gui` only runs `/playsound ...` (lines 83-92). The helper then busy-waits `timeout_s` (default 10 s) and polls `env.info["inventory"]` (lines 59-67). **The env is never stepped, no GUI is opened, and nothing is crafted.** It returns True only if the item count had already increased. In practice it always fails and wastes about 10 s of wall-clock time.
- `place`: returns `baseline_count > 0`. Nothing is placed (lines 94-99).
- `use`: an inventory check only.
- STEVE-1 cannot craft on its own either, because `use` and `inventory` are forced to 0 on every STEVE-1 step (`wrapper.py:246-250`; `allow_change_hotbar` is never set to True).
- **Net: in Env M no craft or smelt step can ever succeed.** This is bug M-B1. Only raw collection (logs, cobblestone, ores, leather) can happen.

**Env O.** Any subgoal whose first word is `craft`, `smelt` or `equip` (or contains `create` or `smelt`) is handed to the scripted helpers (`src/main.py:105-168`, `src/helper/helper.py:38-51`). While they run, hotbar and inventory keys are unlocked (`main.py:113-116`).

- `CraftHelper.crafting` (`src/helper/jarvis_craft_helper.py:462-622`):
  1. Loads `recipes/<item>.json` (859 recipes), with fuzzy matching.
  2. For 3×3 recipes, places a crafting table (`open_crating_table_wo_recipe`, `:197-238`, and `_place_down`, `:267`).
  3. Moves the fake cursor with camera deltas to fixed slot pixel positions (`helper/slot.py`, 640×360, GUI scale 1), then clicks items into the grid and takes the result.
  4. Checks that the result item is in the inventory (`crafting_once`, `:687-745`).
  5. Mines the crafting table back (`return_crafting_table`, `:604-631`).
  - A missing material returns `False, 'missing material: {...}'`, which triggers an LLM replan (`main.py:130-168`).
- `SmeltHelper.smelting` (`jarvis_smelt_helper.py:79-145`): places a furnace and picks the fuel automatically (coals > planks→charcoal > planks > logs). It **requires a wooden_pickaxe** to retrieve the furnace (`:127`) and asserts that the furnace was recovered (`:175`).
- `EquipHelper.equip_item` (`jarvis_equip_helper.py:32-65`): moves the item to the hotbar through the inventory GUI and selects it.
- Every helper action is a real env step. It counts toward the horizon, and a timeout inside a helper sets `game_over` (`jarvis_craft_helper.py:100-101`, `main.py:135-137`).
- STEVE-1 is used only for subgoals such as mine, chop, dig, explore and kill.

## 8. Horizons and task lists

**Horizons.** `max_minutes × 1200` steps. In M: `src/env/custom_env.py:52`, enforced by the gym `TimeLimit`. In O: `src/env/custom_env.py:187`, enforced by `BasaltTimeoutWrapper` (`wrapper.py:54-72`) and `TimeLimit`.

| Group | M (`conf/benchmark/*.yaml:8`) | O (`conf/benchmark/*.yaml:8`) | M times / evaluate (lines 11/13) | O times / evaluate (lines 11/15) |
|---|---|---|---|---|
| wooden | 2 | 2 | 1 / all | 1 / `[2]` (`wooden.yaml:13,17`) |
| stone | **3** | **5** | 1 / all | 20 / `[0]` |
| iron | 20 | 20 | 1 / all | 20 / all |
| gold | **10** | **30** (`golden.yaml`) | 1 / all | 20 / `[5]` |
| redstone | **15** | **30** | 1 / all | 20 / `[5]` |
| diamond | 30 | 30 | 1 / all | 20 / `[5]` (`scripts/diamond.sh` overrides this to `[0..6]`, times=30) |
| armor | **25** | **30** | 1 / all | 20 / `[5]` |

M has extra limits on top of the horizon (`conf/evaluate.yaml:14-16`, `main.py:134-145, 398, 496`):

- `max_subgoals: 12`
- `max_steps_per_subgoal: 1200`
- a **wall-clock** `subgoal_timeout_s: 60`, which at about 7 steps/s caps a subgoal at roughly 420 steps
- abort after 3 consecutive unrepaired failures

O has no per-subgoal cap. Its reflection runs every 1200 steps (`main.py:240-253`).

**Task lists.** M has 70 tasks. O's YAMLs have 73 entries: 67 craft-type entries plus 6 `mine` entries (wooden 2, stone, iron, gold, redstone). "Same" below means the same target item, ignoring wording.

| Group | Shared | Only in M | Only in O |
|---|---|---|---|
| Wooden (M 11 / O 12) | wooden pickaxe, axe, shovel, hoe, sword; stick; crafting table; log (M "Chop an oak log" ≈ O "chop a tree") | oak plank; sapling; "punch a tree … without tools" | chest; bowl; ladder; "dig down to mine dirt" |
| Stone (10 / 10) | stone pickaxe, axe, shovel, hoe, sword; furnace; cobblestone (M "Mine cobblestone with a wooden pickaxe" ≈ O "dig down to mine cobblestone") | "upgrade wooden→stone pickaxe"; "stone slab material"; "eight cobblestone" | charcoal (smelt); smoker; torch |
| Iron (16 / 17) | iron pickaxe, axe, shovel, hoe, sword; bucket; hopper; rail; shears; smithing table; iron nugget; iron bars; iron ore ("dig down to mine iron ore") | smelt iron ingot; "mine three iron ore"; **villager trade for iron ingot** | tripwire hook; chain; blast furnace; stonecutter |
| Gold (7 / 7) | golden pickaxe, axe, sword, hoe; gold ingot (M "Smelt gold ore into gold ingot" ≈ O "Smelt and craft a gold ingot") | golden apple; golden carrot | golden shovel; "dig down to mine gold ore" |
| Redstone (6 / 7) | piston; redstone torch; redstone mining | redstone repeater; **comparator** (needs nether quartz, which cannot be obtained in an overworld-only env); clock | activator rail; compass; dropper; note block |
| Diamond (7 / 7) | **identical**: mine diamond; diamond pickaxe, axe, shovel, hoe, sword; jukebox | none | none |
| Armor (13 / 13) | shield; iron helmet, chestplate, leggings, boots | leather helmet, chestplate, leggings, boots; **turtle shell helmet** (needs scutes); **"wash leather chestplate in cauldron"** (only works on dyed leather); **"repair iron helmet at a smithing table"** (a 1.16 smithing table cannot repair); kill cow for leather | diamond helmet, chestplate, leggings, boots; golden helmet, chestplate, leggings, boots |

In M, the entries in bold are practically infeasible in MC 1.16.5 as worded. Only the diamond group is directly comparable between the two envs.

## 9. Success checker

**Env M.**

- Subgoal level: `TaskCheckerMod` (`src/env/mods/task_checker.py:40-55`) measures the inventory **delta** since the subgoal started. The item must match exactly; a target of `log`, `oak_log` or `wood` matches any `*_log` (`src/util/items.py:11-36`).
- **Episode level:** `_episode_succeeded` (`src/main.py:534-546`) checks the **final inventory** only:
  1. If the goal text contains `log`, `wood` or `tree` and any log is held, the episode succeeds.
  2. Otherwise it succeeds if any held item id, with `_` replaced by a space, is a substring of the goal text.
- **False positives:**
  - All 5 "Craft a *wood*en X" tasks, plus "Mine cobblestone with a *wood*en pickaxe" and "Upgrade a *wood*en pickaxe…", succeed with just one log. "wood" is a substring of "wooden".
  - "Collect a sapling from a *tree*" succeeds with a log.
  - All 5 "Craft a diamond X" tasks succeed with a raw `diamond`.
  - "Craft a redstone torch/repeater/comparator" succeeds with `redstone`.
  - "Craft a golden apple/carrot" succeeds with `apple` / `carrot`.
  - "Smelt iron/gold ore into … ingot" succeeds with the ore itself.
  - "Mine three iron ore" and "eight cobblestone" succeed with one block.
  - "Craft leather X" succeeds with `leather`.
  - "Repair an iron helmet at a smithing table" succeeds with a `smithing_table`.
- **False negatives:**
  - "Craft an oak plank": the held item is `oak_planks`, and "oak planks" is not a substring of the goal.
  - "Craft a turtle shell helmet": the item id is `turtle_helmet`.
- Combined with M-B1, most successes that Env M reports come from these false positives.

**Env O.**

- Episode `status = "success"` iff `plan_manager.remain_plans` is empty and not `game_over` (`src/main.py:258-265`). In other words, every subgoal of the LLM plan finished. The final task item is **not** checked separately; whether the last plan step matches the task depends on the planner.
- STEVE-1 subgoals: `TaskCheckerMod` (`src/env/mods/task_checker.py:21-40`) counts the inventory increase since the subgoal started (`s >= p + number`, line 38), with the `_expand_item` aliases (`:42-70`):
  - anything containing `log` → any log
  - `plank` → any planks
  - `redstone` → `redstone`
  - anything else containing **`stone`** → **`cobblestone`**
  - `coal` → `coal`
- Helper subgoals succeed when the helper returns True, which requires the result item in the inventory (`jarvis_craft_helper.py:698-740`).
- If the final subgoal completes on the same step that `done` fires, the episode is counted as failed, because the `game_over` check comes first (`main.py:225-238`).

## 10. Observation and action interface

| | Env M | Env O |
|---|---|---|
| POV | 640×360, FOV 70, gamma 2, GUI scale 1, cursor 16. These are the HumanControlEnvSpec defaults (UP `human_controls.py:42-46`). | Same, set explicitly (`src/env/custom_env.py:51-54`) |
| Extra obs | Flat inventory, `location_stats`, `life_stats`, `equipped_items`, stats. `isGuiOpen` is only in `info`, and **M overwrites it with False** (`wrapper.py:287`). | Flat inventory, `plain_inventory` (36 slots), `equipped_items`, `life_stats`, `location_stats`, `isGuiOpen` in obs (`src/env/custom_env.py:67-84`) |
| Action keys | MineRL keys + camera + `chat`. Hotbar, `use`, `inventory` and `drop` are masked during STEVE-1 steps. | Same keys + `chat`, same masking. The STEVE action also carries `ESC: 0` (`models/steve_action_model.py:69`). |
| Transport to STEVE-1 | Base64 **PNG**, lossless (`src/main.py:324-334`, `docs/fixes.md` §4) | Base64 **JPEG**, lossy (`src/util/image.py:44`, `util/server_api.py:112-118`) |
| STEVE-1 implementation | MineStudio `SteveOnePolicy` (`CraftJarvis/MineStudio_STEVE-1.official`). Frame resized to 128×128 with INTER_LINEAR. VPT action mapped back with `CameraHierarchicalMapping` + `ActionTransformer` (`executor/steve_runner.py:28-81, 158-176`). | Original `steve1` package: VPT `2x.model` + `steve1.weights` + prior VAE. `resize_image` to 128×128 with INTER_LINEAR (`models/steve1/VPT/agent.py:104-109`) |
| Classifier-free guidance scale | 4.0 (`MINEEVOLVE_STEVE_COND_SCALE`, `server/api.py:152`) | **6.0** (`models/steve_action_model.py:26, 51`) |
| Recurrent state reset | When the subgoal text or scale changes, and at episode reset (`steve_runner.py:104-131`, `server/agent.py:86`) | Only at episode reset (`/reset` recreates the agent, `app.py:45-50`). The state carries over across subgoals. The prompt embedding is recomputed every step. |
| Prompt | Subgoal `condition` string from the MineEvolve planner | Subgoal `task` string (e.g. "chop a tree"). "punch/collect/gather" are rewritten to "chop" (`util/prompt.py:32-33`). After a `/kill`, the prompt is "explore to find X". |

## 11. How "Overall" is aggregated

- **MineEvolve:** `docs/tasks.md:18-26` defines Easy = Wooden + Stone + Gold (28 tasks) and Hard = Iron + Redstone + Diamond + Armor (42 tasks). "Easy Avg., Hard Avg., Overall are computed as task-count-weighted means", so Overall is the mean over all 70 tasks. No code computes the cross-group aggregate; `print_results` only reports the active group.
- **Optimus-1:** `README.md:112` says "Overall represents the average result on the five groups of Iron, Gold, Diamond, Redstone, and Armor". Wooden and Stone are excluded. The README reads as a mean of group averages, not task-weighted, though that is not stated explicitly. **No code** computes it: `main.py:357-370` only logs per-task metrics and total steps.
- For a fair comparison, compute both aggregates from the per-task results. Also note that O's default YAMLs only evaluate one task per group, except iron (§8).

## 12. Other factors that could make results differ

- **Death:** both envs add `DoneOnDeath` (M `custom_env.py:87`, O `custom_env.py:90`). The Java side sets `done = player==null || !isAlive() || (doneOnDeath && hasPlayerRespawned)` (`EnvServer.java:516`). So `keepInventory` and `doImmediateRespawn` make no practical difference: any death ends the episode.
  - In M, a death shows up as `died=True` (`main.py:188-190`), and later `env.step` calls raise errors that are caught (`main.py:162-169`).
  - In O, `game_over=True`.
  - In M, **reaching the horizon is also recorded as `died`**, because `done` comes from `TimeLimit`.
- **Time and light:** both freeze time at 0 (morning) and give night vision at level 250.
- **Mobs:** both allow spawning at difficulty HARD (`EnvServer.java:423`), so hostile mobs spawn in dark caves.
- **Horizon accounting:**
  - O's scripted helpers consume steps; a table craft costs about 100–300 steps. M's helpers consume none because they do nothing.
  - M's reset commands and `/setblock` steps count toward the horizon; O's do not (§1.3).
  - M also has a 60 s wall-clock cap per subgoal.
- **Multiple ore spawns per y:** M records only the target y, so it can spawn again from later y values more freely. O records both.
- **Task execution loop:** M uses LLM-generated subgoals with a check type (`inv_ge`). O uses LLM plans with `(item, n)` goals and alias expansion (§9).
- **Frame compression:** JPEG vs PNG (§10) can slightly shift STEVE-1 behaviour.
- **Repetitions:** O's default `times: 20`; M's default `times: 1`.

## 13. Bugs that crash or silently break a run

**Env M**

- **M-B1 (blocking):** crafting and smelting helpers are stubs (`src/executor/craft_helper.py:42-99`). No crafted item can ever be produced, because STEVE-1 is also barred from `use` and `inventory` (`wrapper.py:246-250`). Every craft-type task is unreachable except through the success-heuristic false positives. To evaluate crafting baselines in Env M, port O's JARVIS helpers, which need `plain_inventory` and therefore O's `slot_id` jar or an equivalent.
- **M-B2 (silent):** auto-pickaxe never fires, because `plain_inventory` is not an M observation (`src/env/mods/status.py:72-74`; no `create_observables` override in `custom_env.py`). STEVE-1 mines stone and ores with whatever it is holding.
- **M-B3 (silent):** `_episode_succeeded` false positives and negatives (§9, `main.py:534-546`).
- **M-B4 (silent):** `env.seed` is ignored (`custom_env.py:61-67`).
- **M-B5 (hang risk):** in `run_episode` (`main.py:474-491`), a successful repair does `continue` without advancing `i` or counting failures. The server's `repair` has no attempt cap (`server/agent.py:254-330`). After a death or the horizon, every subgoal fails within one step, so the loop can call the LLM without limit.
- **M-B6 (design):** redstone is checked before diamond, and the redstone cap is 12 (`wrapper.py:66-70, 108-120`), so diamond ore is almost never placed.
- **Minor:**
  - `info["isGuiOpen"]` is forced to False (`wrapper.py:287`).
  - Health and hunger are read from `location_stats`, not `life_stats` (`status.py:58-65`), so they always report 20.
  - `_run_subgoal` labels a timeout as `died` (`main.py:188-190`).

**Env O**

- **O-B1 (likely crash, not run-verified):** `env.close()` is inside the `for task in evaluate_tasks` loop (`src/main.py:365`). The next task calls `env.reset()` on a closed env. MineRL's `close()` kills the instance but leaves it in `self.instances`, and `_setup_instances` starts no new one (UP `_multiagent.py:529-540, 650-666`). The second and later tasks (e.g. `scripts/diamond.sh` with 7 tasks) should fail or hang. Run one task per process, or move `close()` after the loop.
- **O-B2 (silent):** the `/kill` stuck-recovery and the "Return to ground" rule (`wrapper.py:178-186, 200-209`) end the episode under `DoneOnDeath` (`EnvServer.java:516`), so they turn into failures instead of recoveries. The iron-ore rule fires whenever the current goal contains `iron_ore` and y < 25, and iron ore is spawned at y ≥ 26, so digging slightly too deep during an iron-ore subgoal kills the episode.
- **O-B3 (crash):** in `main.py:300-328`, if `ServerAPI.get_retrieval` raises before `example` is assigned, the `except` branch does `planning = example`, which raises `UnboundLocalError`.
- **O-B4 (environmental):**
  - `CUSTOM_GYM_ENTRY_POINT = "src.optimus1.env.custom_env:..."` (`custom_env.py:29`) and `CraftHelper.root_path = "src/optimus1/helper"` (`jarvis_craft_helper.py:20`) require the working directory to be the O repo root.
  - The gym entry point also imports a second copy of the env module.
  - The MineRL launch runs under `xvfb-run` (`launchClient.sh:52`). Adding an outer xvfb is fine, but `xvfb-run` must be installed.
- **O-B5 (edge-case crash):** in `pre_open_tabel` (`jarvis_craft_helper.py:136-161`), when y < 50 and the agent has **no** pickaxe, `inventory_id` is never assigned, so `if inventory_id != "inventory_0"` raises `UnboundLocalError`. Only `AssertionError` and `RuntimeError` are caught, so this crashes the run.
- **Minor:** in the task checker, anything containing "stone" maps to `cobblestone` (`task_checker.py:65-66`). A STEVE subgoal for `stone` or a `stone_*` item is judged by cobblestone count.

**Cross-env**

- O's Python env code **requires** O's jar, because of `slot_id` in the inventory JSON.
- M's Python code runs on O's MineRL and jar. Do **not** also apply M's chat patch there.
- To evaluate "Env M" semantics with working crafting, the practical route is M's wrapper (ore bands, horizons, biomes, task list) on O's MineRL build, with O's JARVIS helpers.
