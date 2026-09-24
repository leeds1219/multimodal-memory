# Progress

## 2026-09-23 — setup

**Machine.** 8× RTX PRO 6000 Blackwell (97 GB). GPUs 3/4/5 hold other users' memory → never used. Using **GPU 6** (idle, matmul 416 TFLOPS). Disk: `/home/rag/data` and `/` are the same local NVMe (not NAS); ~12 GB free at time of writing.

**Environment.** conda env `mcagent` (py3.10, torch 2.9.1+cu128, gym 0.23.1, numpy 1.23.5), Java 8, Xvfb. Optimus-1 MineRL fork + authors' prebuilt MCP-Reborn (`mcprec-6.13.jar`).

**Verified so far**
- MineRL headless (Optimus fork): reset 37 s, ~16 env steps/s.
- STEVE-1 alone ("chop a tree", 1200 steps, GPU 6): first log at step 263, 5 oak logs, ~12 steps/s.
- Gemini 3 flash vision via OpenAI-compatible endpoint (2 images, Optimus-style call).
- Debug check 2 (LLM layer, no game): model forced from config; JSONL logs with image hashes; thinking tokens counted (e.g. 2 visible / 1,918 billed out tokens); backoff 4 attempts then `LLMFailed`; global cap → `BudgetExceeded`; episode guard → `EpisodeAnomaly` + `ANOMALY` marker; API key absent from logs.

**Baseline code availability**
| Method | Code | Notes |
|--------|------|-------|
| MineEvolve | official (xzw-ustc/MC-MineEvolve a0a5f9b) | no baseline reimplementations included |
| Optimus-1 | official (iLearn-Lab/NeurIPS24-Optimus-1) | memory v1 ships 5 success plans + reflection images |
| JARVIS-1 | partial (offline eval, fixed memory only) | Stage B port |
| DEPS | MineDojo + own controller, Codex-era | Stage B port |

**Next:** Env M vs Env O comparison table; offline (mock LLM) end-to-end check per method per env.

## 2026-09-23 — environments, glue, offline checks

**GPUs used:** GPU 6 (idle: 0 MiB, matmul 416 TFLOPS). GPUs 3/4/5 hold other users' memory — never used.

### Env M vs Env O (full analysis with file:line citations: `docs/env_comparison.md`)
| Item | Env M (MineEvolve) | Env O (Optimus-1) | Used in reproduction |
|------|--------------------|-------------------|----------------------|
| MineRL / MCP build | MineRL 1.0.2 + chat patch | Optimus fork (chat verb, allowCommands, per-slot inventory JSON), prebuilt jar | Optimus jar for both (functionally identical for M; D10) |
| Reset commands | 8 commands (gamerules, night vision, time 0, spawnpoint) | identical | identical |
| Initial inventory | empty | empty | empty |
| Biome | wooden forest, all others plains | wooden/gold/redstone/diamond/armor forest, stone/iron plains | per env |
| World seed | config ignored by Java → random | not configurable → random | `env.seed(s)` before reset, same seeds everywhere (D15) |
| Ore spawning | 10%/step, 3–5 below feet, bands coal/iron/gold/redstone/diamond with caps (diamond only after 12 redstone) | 10% per STEVE-1 step, diamond for every y ≤ 14, no caps | per env |
| Auto pickaxe | dead code as released (no per-slot obs) | best pickaxe when y < 70 | M: enabled (D19); O: as released |
| Kill heuristics | none | `/kill` after 8000 steps at one y, or iron-ore goal below y 25 | per env |
| Craft / smelt / equip | stub (never crafts) | scripted GUI (JARVIS helpers) | M: functional primitive (D11); O: GUI helpers |
| Horizon (min) W/S/I/G/R/D/A | 2/3/20/10/15/30/25 | 2/5/20/30/30/30/30 | per env |
| Task list | 70 tasks | 73 yaml entries | MineEvolve's 70 in both (D14) |
| Success checker | substring heuristic over final inventory | all plan steps done | paper criterion in both (target item, any step); native recorded (D12) |
| STEVE-1 wrapper | cond_scale 4.0, PNG frames, state reset per subgoal | cond_scale 6.0, JPEG, reset per episode | per env (D16); same weights (D2) |
| "Overall" in the paper | task-weighted mean over 70 tasks | mean of Iron/Gold/Diamond/Redstone/Armor groups | both reported, per env, never mixed |

### Stage C (Optimus-2 / Optimus-3)
| Method | Repo / license | Released | Depends on | Verdict |
|--------|----------------|----------|-----------|---------|
| Optimus-2 | github.com/iLearn-Lab/CVPR25-Optimus-2, MIT | README + figures only; MGOA training data (138 GB) | own GOAP policy + MLLM | **Skipped: no code, no weights** |
| Optimus-3 | github.com/JiuTian-VL/Optimus-3, MIT | code, weights (Qwen2.5-VL-7B MoE ≈19.5 GB + action head 2.7 GB + router), static benchmarks, interactive GUI server | own MLLM + own action head; own MineStudio/Malmo simulator | **Blocked:** ≈23 GB of weights exceed the disk budget; no automated tech-tree rollout harness released (only the interactive GUI). Needs disk + a rollout harness. |

### Offline (mock LLM) check — all four (method, env) pairs pass
"Craft a crafting table" end to end with canned plans: env reset, STEVE-1 acting, craft primitive, success check, trajectory / LLM log / result / memory snapshot written. Resume: chain killed mid-episode → rerun skipped the finished episode, restored memory from the last snapshot, moved the crashed episode aside and redid it.
Fairness: same seed → identical spawn position (570.5, 63.0, -659.5) under both methods in both envs; identical first actions within an env. First frames are saved per episode for image-level comparison.

### Key findings so far
- **MineEvolve as released cannot craft** (helper stub) and **cannot hold a pickaxe** (auto-pickaxe dead) in its own env; its success check passes "Craft a wooden pickaxe" with one oak log.
- **Optimus-1 with Gemini often never plans:** goal inference returns e.g. "wood logs"; `retrieve_graph` raises KeyError; `main.py` silently executes the built-in example plan (D20). Tracked per episode as `plan_source`.
- MineEvolve's LLM cost is dominated by the Inducer (≈10k input tokens per call, one call per subgoal).

### Disk freed on the host (≈217 GB free) — deferred items revisited
- Optimus-1 **full pre-built memory** (HF, 9.9 GB tar.gz): downloading; the `prebuilt` variant will use it (D21b). The repo-shipped 5-plan memory is dropped.
- **Optimus-3**: weights now fit; a feasibility study of its released code (live rollout / missing harness) is running → `docs/optimus3_feasibility.md`.
- Keyframes (320×180 every 100 steps) now saved per episode for visual failure analysis (D26).
- Kept on purpose (not disk-driven): one MineRL build, STEVE-1 via the `steve1` package with the shared weight files (D27).

### Real smoke tests (Gemini, 3 easy tasks × 1 seed per method/env) and projection
| env | method | wooden_00 (pickaxe) | wooden_06 (table) | stone_00 (stone pickaxe) | steps/s | $ / 1k steps | calls / 1k steps |
|-----|--------|------|------|------|------|------|------|
| M | MineEvolve | ✗ horizon, $0.20 | ✓ 175 steps, $0.07 | ✗ horizon, $0.31 | 3.9 | 0.094 | 7.9 |
| O | MineEvolve | ✗ horizon, $0.27 | ✓ 288 steps, $0.07 | ✗ plan ended, $0.17 | 4.0 | 0.081 | 6.2 |
| M | Optimus-1 (empty) | ✓ (fallback plan) | ✓ (fallback plan) | ✓ (fallback plan) | 10.2 | 0.003 | 0.95 |
| O | Optimus-1 (empty) | ✓ | ✓ | ✓ | 13.9 | 0.003 | 0.74 |
Stage B (DEPS, JARVIS-1) smoke: DEPS 3.0 / 6.4 steps/s, JARVIS-1 2.4 / 4.9 steps/s (M / O); still running at time of writing.

Checks: plans parse (MineEvolve JSON, Optimus retrieval/reflection formats); Optimus-1 image calls work (retrieval + reflection with 2 images). Two MineEvolve calls were truncated by thinking → headroom raised to 32k (D3b).
**Optimus-1 with Gemini never reached its planner in any smoke episode:** goal inference returns a list ("stone pickaxe, cobblestone, sticks, …"), `retrieve_graph` raises KeyError, and the built-in example plan is executed (D20). Kept as released; flagged per episode (`plan_source`).

**Projection** (every episode to its horizon = upper bound; mid = 60 %): LLM cost for all 22 planned chains ≈ **$1,224 upper / $735 mid** (cap $3,000). Longest chain ≈ 149 h (JARVIS-1, Env M), MineEvolve 92–114 h, Optimus-1 33–44 h — all chains run in parallel, so the run fits before 10/06 if started by ~09/27.

**Launch (09-23):** Stage A part 1 (12 chains: MineEvolve + Optimus-1 empty × 3 orders × 2 envs) via `scripts/supervise.py --plan configs/run_plan_stageA.yaml`. Optimus-1 full-memory chains (Stage A2) start once the authors' memory download finishes; Stage B chains after their smoke tests.

### 09-23 08:30 — Stage B smoke done, all stages launched
Stage B smoke (DEPS / JARVIS-1, 3 tasks × 2 envs): DEPS 4/6 successes, JARVIS-1 5/6; no truncations; JARVIS-1 stone_00 in Env M hit the 12-replan cap (145 calls, $1.19). One infra crash (Minecraft socket timeout) retried successfully.
Running: Stage A (12 chains, launched 07:54), Stage A2 (Optimus-1 with the authors' full memory, 6 chains, 08:24), Stage B (4 chains, 08:30). GPUs 0, 1, 2, 7 (idle at launch: 0 MiB + matmul test); GPU 6 used for tests and Optimus-3 setup. Stage C (Optimus-3) harness in progress.

### 09-23 08:45 — Stage A2 restarted
The authors' full Optimus-1 memory contains 4 corrupt JSON files (DEVIATIONS: Optimus-1 full memory). One A2 episode crashed on it three times; others could have silently used the example plan. A2 was stopped after ~5 episodes, the files repaired (all records kept except two truncated last records and one duplicated tail), A2 results wiped, and A2 relaunched from scratch. Rerun reason recorded here. Also added a reaper that kills Minecraft instances no episode is using (killed episodes can leave them behind).

### 09-23 09:05 — Optimus-1 degenerates with Gemini → added a labelled `goalfix` variant (Stage A3)
Faithful Optimus-1 (empty memory, Env O, order1): after its first success every episode ends at step 0 (14/15 `method_exception`): the first saved plan stores Gemini's list-style goal, fuzzy retrieval then returns that file for every task, `retrieve_graph` raises KeyError and the released `UnboundLocalError` bug ends the episode. Faithful runs continue unchanged; the `goalfix` variant (only change: keep the first item of the goal inference) runs alongside (12 chains, D29). **Needs the user's call: which Optimus-1 row goes in the paper.**

### 09-23 09:40 — Stage C: Optimus-3 launched (own models, own simulator, env "C3")
Setup (by a sub-agent, stopped by an API usage limit after finishing setup; resumed here): separate conda env `optimus3` (py3.11, torch 2.7.1+cu128, transformers 4.51.3, sdpa instead of flash-attn), weights ≈23 GB (`iLearn-Lab/Optimus-3` preview + action head + router), headless harness `baselines/optimus3/run_episode.py` re-implementing the released GUI server loop, compat patches in `baselines/optimus3/optimus3.patch`, decisions C1–C5, deviations section "Optimus-3".
Planner dry run over the 70 tasks: the released planner only works with its trained input form "obtain N <item>" (verbatim instructions: 38/70 truncated, 29/70 empty plans) → C1.
Smoke (3 tasks): wooden_00 ✓ 831 steps, stone_00 ✓ 1,826 steps, wooden_06 ✗ horizon; ≈10 steps/s; 5–8 MLLM calls/episode; ≈28 GB GPU per worker.
Full run: 3 shards on GPU 6 (`run_all.py --shard i/3`), results in `runs/C3/optimus3/order0/`; reported in a separate table (different setting: own policy + MLLM, not Gemini + STEVE-1).

## 2026-09-24 — daily summary (auto)

```
env chain                  order      done  succ%    cost$  calls anom crash fallbk  wall_h  running
C3  optimus3               order0   70/70    28.6     0.00    521    0     0      0    14.4  
M   deps                   order0   22/70    22.7     6.15    360    0     0      0    14.8  redstone_04 @10100
M   jarvis1                order0   32/70    21.9    22.61   3837    0     0      0    14.5  wooden_09 @500
M   mineevolve             order0   35/70    11.4    18.12    858    0     0      0    14.7  armor_07 @9700
M   mineevolve             order1   30/70    13.3    21.04   1075    0     0      0    15.7  iron_07 @2600
M   mineevolve             order2   31/70    16.1    19.62    963    0     0      0    14.8  armor_04 @6700
M   optimus1-empty         order0   70/70     7.1     1.39    380    0     0     64    12.2  
M   optimus1-empty         order1   70/70     8.6     0.51    144    0     0     63     3.7  
M   optimus1-empty         order2   70/70     2.9     0.49    149    0     0     65     3.6  
M   optimus1-empty-goalfix order0   70/70    17.1     0.67    193    0     0     53     5.5  
M   optimus1-empty-goalfix order1   70/70    11.4     1.11    273    0     0     55     9.2  
M   optimus1-empty-goalfix order2   70/70    14.3     1.59    404    0     0     43    14.0  
M   optimus1-prebuilt      order0   49/70    36.7     1.63    469    0     1     39    14.4  
M   optimus1-prebuilt      order1   38/70    26.3     1.61    446    0     0     35    14.3  stone_07 @2100
M   optimus1-prebuilt      order2   33/70    24.2     1.75    433    0     1     26    14.5  stone_09 @1300
M   optimus1-prebuilt-goalfix order0   35/70    28.6     1.48    396    0     1     20    13.7  armor_07 @27800
M   optimus1-prebuilt-goalfix order1   35/70    31.4     1.49    406    0     0     15    13.7  armor_08 @18800
M   optimus1-prebuilt-goalfix order2   27/70    14.8     1.76    430    0     1     12    14.2  diamond_06 @5900
O   deps                   order0   22/70    27.3    10.64    554    0     0      0    14.2  redstone_04 @6100
O   jarvis1                order0   34/70    14.7    24.69   4237    0     0      0    13.6  gold_03 @3200
O   mineevolve             order0   22/70    13.6    25.59   1258    0     0      0    15.6  redstone_04 @4800
O   mineevolve             order1   21/70     9.5    24.45   1422    0     0      0    15.8  stone_00 @1900
O   mineevolve             order2   26/70     7.7    19.61   1056    0     0      0    12.9  diamond_01 @20700
O   optimus1-empty         order0   70/70    18.6     0.53    187    0     0     58     5.0  
O   optimus1-empty         order1   70/70     1.4     0.24     91    0     0     70     1.5  
O   optimus1-empty         order2   70/70     5.7     0.38    133    0     0     66     3.5  
O   optimus1-empty-goalfix order0   70/70    20.0     1.46    387    0     0     39    11.4  
O   optimus1-empty-goalfix order1   70/70     1.4     0.24     96    0     0     70     1.7  
O   optimus1-empty-goalfix order2   70/70    15.7     1.28    356    0     0     48    11.5  
O   optimus1-prebuilt      order0   60/70    46.7     1.48    423    0     0     47    12.9  diamond_01 @7200
O   optimus1-prebuilt      order1   70/70    48.6     1.62    437    0     0     54    12.1  
O   optimus1-prebuilt      order2   43/70    41.9     1.46    402    0     0     32    11.7  armor_09 @28300
O   optimus1-prebuilt-goalfix order0   45/70    57.8     1.62    417    0     0     19    12.7  iron_00 @8800
O   optimus1-prebuilt-goalfix order1   37/70    43.2     1.60    417    0     0     17    12.4  diamond_06 @6200
O   optimus1-prebuilt-goalfix order2   50/70    44.0     1.86    514    0     0     23    13.7  iron_06 @2400

LLM spend (ledger, all runs incl. smoke/debug): $244.63 / cap $3000   per env: {'C3': 0.0, 'O': 118.74, 'M': 103.02}   disk free: 636.3 GB
```

## 2026-09-24 — daily summary (auto)

```
env chain                  order      done  succ%    cost$  calls anom crash fallbk  wall_h  running
C3  optimus3               order0   70/70    28.6     0.00    521    0     0      0    14.4  
M   deps                   order0   22/70    22.7     6.15    360    0     0      0    14.8  redstone_04 @10100
M   jarvis1                order0   32/70    21.9    22.61   3837    0     0      0    14.5  wooden_09 @500
M   mineevolve             order0   35/70    11.4    18.12    858    0     0      0    14.7  armor_07 @9800
M   mineevolve             order1   30/70    13.3    21.04   1075    0     0      0    15.7  iron_07 @2600
M   mineevolve             order2   31/70    16.1    19.62    963    0     0      0    14.8  armor_04 @6700
M   optimus1-empty         order0   70/70     7.1     1.39    380    0     0     64    12.2  
M   optimus1-empty         order1   70/70     8.6     0.51    144    0     0     63     3.7  
M   optimus1-empty         order2   70/70     2.9     0.49    149    0     0     65     3.6  
M   optimus1-empty-goalfix order0   70/70    17.1     0.67    193    0     0     53     5.5  
M   optimus1-empty-goalfix order1   70/70    11.4     1.11    273    0     0     55     9.2  
M   optimus1-empty-goalfix order2   70/70    14.3     1.59    404    0     0     43    14.0  
M   optimus1-prebuilt      order0   49/70    36.7     1.63    469    0     1     39    14.4  
M   optimus1-prebuilt      order1   38/70    26.3     1.61    446    0     0     35    14.3  stone_07 @2200
M   optimus1-prebuilt      order2   33/70    24.2     1.75    433    0     1     26    14.5  stone_09 @1300
M   optimus1-prebuilt-goalfix order0   35/70    28.6     1.48    396    0     1     20    13.7  armor_07 @27900
M   optimus1-prebuilt-goalfix order1   35/70    31.4     1.49    406    0     0     15    13.7  armor_08 @18900
M   optimus1-prebuilt-goalfix order2   27/70    14.8     1.76    430    0     1     12    14.2  diamond_06 @6000
O   deps                   order0   22/70    27.3    10.64    554    0     0      0    14.2  redstone_04 @6100
O   jarvis1                order0   34/70    14.7    24.69   4237    0     0      0    13.6  gold_03 @3200
O   mineevolve             order0   22/70    13.6    25.59   1258    0     0      0    15.6  redstone_04 @4900
O   mineevolve             order1   21/70     9.5    24.45   1422    0     0      0    15.8  stone_00 @2000
O   mineevolve             order2   26/70     7.7    19.61   1056    0     0      0    12.9  diamond_01 @20700
O   optimus1-empty         order0   70/70    18.6     0.53    187    0     0     58     5.0  
O   optimus1-empty         order1   70/70     1.4     0.24     91    0     0     70     1.5  
O   optimus1-empty         order2   70/70     5.7     0.38    133    0     0     66     3.5  
O   optimus1-empty-goalfix order0   70/70    20.0     1.46    387    0     0     39    11.4  
O   optimus1-empty-goalfix order1   70/70     1.4     0.24     96    0     0     70     1.7  
O   optimus1-empty-goalfix order2   70/70    15.7     1.28    356    0     0     48    11.5  
O   optimus1-prebuilt      order0   60/70    46.7     1.48    423    0     0     47    12.9  diamond_01 @7300
O   optimus1-prebuilt      order1   70/70    48.6     1.62    437    0     0     54    12.1  
O   optimus1-prebuilt      order2   43/70    41.9     1.46    402    0     0     32    11.7  armor_09 @28300
O   optimus1-prebuilt-goalfix order0   45/70    57.8     1.62    417    0     0     19    12.7  iron_00 @8900
O   optimus1-prebuilt-goalfix order1   37/70    43.2     1.60    417    0     0     17    12.4  diamond_06 @6300
O   optimus1-prebuilt-goalfix order2   50/70    44.0     1.86    514    0     0     23    13.7  iron_06 @2500

LLM spend (ledger, all runs incl. smoke/debug): $244.67 / cap $3000   per env: {'C3': 0.0, 'O': 118.74, 'M': 103.02}   disk free: 636.3 GB
```

## 2026-09-24 — daily summary (auto)

```
env chain                  order      done  succ%    cost$  calls anom crash fallbk  wall_h  running
C3  optimus3               order0   70/70    28.6     0.00    521    0     0      0    14.4  
M   deps                   order0   22/70    22.7     6.15    360    0     0      0    14.8  redstone_04 @10300
M   jarvis1                order0   32/70    21.9    22.61   3837    0     0      0    14.5  wooden_09 @600
M   mineevolve             order0   35/70    11.4    18.12    858    0     0      0    14.7  armor_07 @9900
M   mineevolve             order1   30/70    13.3    21.04   1075    0     0      0    15.7  iron_07 @2600
M   mineevolve             order2   31/70    16.1    19.62    963    0     0      0    14.8  armor_04 @6700
M   optimus1-empty         order0   70/70     7.1     1.39    380    0     0     64    12.2  
M   optimus1-empty         order1   70/70     8.6     0.51    144    0     0     63     3.7  
M   optimus1-empty         order2   70/70     2.9     0.49    149    0     0     65     3.6  
M   optimus1-empty-goalfix order0   70/70    17.1     0.67    193    0     0     53     5.5  
M   optimus1-empty-goalfix order1   70/70    11.4     1.11    273    0     0     55     9.2  
M   optimus1-empty-goalfix order2   70/70    14.3     1.59    404    0     0     43    14.0  
M   optimus1-prebuilt      order0   49/70    36.7     1.63    469    0     1     39    14.4  
M   optimus1-prebuilt      order1   38/70    26.3     1.61    446    0     0     35    14.3  stone_07 @2400
M   optimus1-prebuilt      order2   33/70    24.2     1.75    433    0     1     26    14.5  stone_09 @1500
M   optimus1-prebuilt-goalfix order0   35/70    28.6     1.48    396    0     1     20    13.7  armor_07 @28200
M   optimus1-prebuilt-goalfix order1   35/70    31.4     1.49    406    0     0     15    13.7  armor_08 @19100
M   optimus1-prebuilt-goalfix order2   27/70    14.8     1.76    430    0     1     12    14.2  diamond_06 @6200
O   deps                   order0   22/70    27.3    10.64    554    0     0      0    14.2  redstone_04 @6100
O   jarvis1                order0   34/70    14.7    24.69   4237    0     0      0    13.6  gold_03 @3200
O   mineevolve             order0   22/70    13.6    25.59   1258    0     0      0    15.6  redstone_04 @5000
O   mineevolve             order1   21/70     9.5    24.45   1422    0     0      0    15.8  stone_00 @2100
O   mineevolve             order2   26/70     7.7    19.61   1056    0     0      0    12.9  diamond_01 @20800
O   optimus1-empty         order0   70/70    18.6     0.53    187    0     0     58     5.0  
O   optimus1-empty         order1   70/70     1.4     0.24     91    0     0     70     1.5  
O   optimus1-empty         order2   70/70     5.7     0.38    133    0     0     66     3.5  
O   optimus1-empty-goalfix order0   70/70    20.0     1.46    387    0     0     39    11.4  
O   optimus1-empty-goalfix order1   70/70     1.4     0.24     96    0     0     70     1.7  
O   optimus1-empty-goalfix order2   70/70    15.7     1.28    356    0     0     48    11.5  
O   optimus1-prebuilt      order0   60/70    46.7     1.48    423    0     0     47    12.9  diamond_01 @7400
O   optimus1-prebuilt      order1   70/70    48.6     1.62    437    0     0     54    12.1  
O   optimus1-prebuilt      order2   43/70    41.9     1.46    402    0     0     32    11.7  armor_09 @28500
O   optimus1-prebuilt-goalfix order0   45/70    57.8     1.62    417    0     0     19    12.7  iron_00 @9100
O   optimus1-prebuilt-goalfix order1   37/70    43.2     1.60    417    0     0     17    12.4  diamond_06 @6600
O   optimus1-prebuilt-goalfix order2   50/70    44.0     1.86    514    0     0     23    13.7  iron_06 @2800

LLM spend (ledger, all runs incl. smoke/debug): $244.79 / cap $3000   per env: {'C3': 0.0, 'O': 118.74, 'M': 103.02}   disk free: 636.3 GB
```

## 2026-09-24 — daily summary (auto)

```
env chain                  order      done  succ%    cost$  calls anom crash fallbk  wall_h  running
C3  optimus3               order0   70/70    28.6     0.00    521    0     0      0    14.4  
M   deps                   order0   22/70    22.7     6.15    360    0     0      0    14.8  redstone_04 @10400
M   jarvis1                order0   32/70    21.9    22.61   3837    0     0      0    14.5  wooden_09 @600
M   mineevolve             order0   35/70    11.4    18.12    858    0     0      0    14.7  armor_07 @9900
M   mineevolve             order1   30/70    13.3    21.04   1075    0     0      0    15.7  iron_07 @2600
M   mineevolve             order2   31/70    16.1    19.62    963    0     0      0    14.8  armor_04 @6700
M   optimus1-empty         order0   70/70     7.1     1.39    380    0     0     64    12.2  
M   optimus1-empty         order1   70/70     8.6     0.51    144    0     0     63     3.7  
M   optimus1-empty         order2   70/70     2.9     0.49    149    0     0     65     3.6  
M   optimus1-empty-goalfix order0   70/70    17.1     0.67    193    0     0     53     5.5  
M   optimus1-empty-goalfix order1   70/70    11.4     1.11    273    0     0     55     9.2  
M   optimus1-empty-goalfix order2   70/70    14.3     1.59    404    0     0     43    14.0  
M   optimus1-prebuilt      order0   49/70    36.7     1.63    469    0     1     39    14.4  
M   optimus1-prebuilt      order1   38/70    26.3     1.61    446    0     0     35    14.3  stone_07 @2500
M   optimus1-prebuilt      order2   33/70    24.2     1.75    433    0     1     26    14.5  stone_09 @1500
M   optimus1-prebuilt-goalfix order0   35/70    28.6     1.48    396    0     1     20    13.7  armor_07 @28200
M   optimus1-prebuilt-goalfix order1   35/70    31.4     1.49    406    0     0     15    13.7  armor_08 @19200
M   optimus1-prebuilt-goalfix order2   27/70    14.8     1.76    430    0     1     12    14.2  diamond_06 @6200
O   deps                   order0   22/70    27.3    10.64    554    0     0      0    14.2  redstone_04 @6100
O   jarvis1                order0   34/70    14.7    24.69   4237    0     0      0    13.6  gold_03 @3200
O   mineevolve             order0   22/70    13.6    25.59   1258    0     0      0    15.6  redstone_04 @5000
O   mineevolve             order1   21/70     9.5    24.45   1422    0     0      0    15.8  stone_00 @2100
O   mineevolve             order2   26/70     7.7    19.61   1056    0     0      0    12.9  diamond_01 @20800
O   optimus1-empty         order0   70/70    18.6     0.53    187    0     0     58     5.0  
O   optimus1-empty         order1   70/70     1.4     0.24     91    0     0     70     1.5  
O   optimus1-empty         order2   70/70     5.7     0.38    133    0     0     66     3.5  
O   optimus1-empty-goalfix order0   70/70    20.0     1.46    387    0     0     39    11.4  
O   optimus1-empty-goalfix order1   70/70     1.4     0.24     96    0     0     70     1.7  
O   optimus1-empty-goalfix order2   70/70    15.7     1.28    356    0     0     48    11.5  
O   optimus1-prebuilt      order0   60/70    46.7     1.48    423    0     0     47    12.9  diamond_01 @7500
O   optimus1-prebuilt      order1   70/70    48.6     1.62    437    0     0     54    12.1  
O   optimus1-prebuilt      order2   43/70    41.9     1.46    402    0     0     32    11.7  armor_09 @28500
O   optimus1-prebuilt-goalfix order0   45/70    57.8     1.62    417    0     0     19    12.7  iron_00 @9100
O   optimus1-prebuilt-goalfix order1   37/70    43.2     1.60    417    0     0     17    12.4  diamond_06 @6700
O   optimus1-prebuilt-goalfix order2   50/70    44.0     1.86    514    0     0     23    13.7  iron_06 @2800

LLM spend (ledger, all runs incl. smoke/debug): $244.81 / cap $3000   per env: {'C3': 0.0, 'O': 118.74, 'M': 103.02}   disk free: 636.3 GB
```
