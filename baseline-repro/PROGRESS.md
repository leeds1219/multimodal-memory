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

## 2026-09-24 — MineEvolve reproduction gap: investigation and a fairness fix
Question: MineEvolve reaches ≈10–13 % (paper 52 %), Stone ≈10 % (paper 93 %). Findings so far:
- The functional craft primitive is not the cause (crafts fail only when materials are really missing).
- Most failed Stone episodes hit the 3-minute horizon while still collecting wood/stone: STEVE-1 is slow.
- Same method + same prompts (JARVIS-1): first log at median step 450 in Env M vs 165 in Env O. Controlled STEVE-1 test: **cond_scale 4.0 (MineEvolve's runner default) gives about half the log yield of 6.0** (3.4 vs 6.6 logs / 1200 steps; 3/5 vs 5/5 episodes get any log).
- This exposed a **fairness bug**: Stage A cross pairs used their own STEVE-1 wrapper instead of the env's (D32). Fixed; 15 chains rerun from scratch (old results archived as invalid).
- Dead-Minecraft episodes were recorded as method failures by MineEvolve's loop (6 of ~1,700) → now infra crashes, retried (D31).
- Remaining explanations for the MineEvolve gap (open): (1) released code ≠ the code behind the paper's numbers (stub crafting, broken success check, dead auto-pickaxe in the release); (2) STEVE-1 settings (the paper does not state cond_scale); (3) our strict target check ("oak log" task fails with a dark-oak log; 3 of 9 wooden failures).

## 2026-09-25 — daily summary (auto)

```
env chain                  order      done  succ%    cost$  calls anom crash fallbk  wall_h  running
C3  optimus3               order0   70/70    28.6     0.00    521    0     0      0    14.4  
M   deps                   order0   54/70    25.9    20.94   1141    0     0      0    38.7  armor_05 @700
M   jarvis1                order0   70/70    14.3    52.17   9009    0     1      0    32.7  
M   mineevolve             order0   70/70    10.0    46.09   2367    0     0      0    35.0  
M   mineevolve             order1   64/70    12.5    53.01   2661    0     0      0    36.9  iron_06 @21400
M   mineevolve             order2   70/70    14.3    47.70   2393    0     0      0    35.6  
M   optimus1-empty         order0   70/70     4.3     0.49    148    0     0     70     4.1  
M   optimus1-empty         order1   70/70     8.6     1.59    374    0     0     63    14.3  
M   optimus1-empty         order2   70/70     2.9     0.35    126    0     0     65     4.1  
M   optimus1-empty-goalfix order0   70/70     5.7     0.32    126    0     0     65     3.6  
M   optimus1-empty-goalfix order1   70/70    11.4     0.97    257    0     0     56    10.4  
M   optimus1-empty-goalfix order2   61/70    14.8     1.98    521    0     0     35    21.2  diamond_03 @28000
M   optimus1-prebuilt      order0   55/70    38.2     1.99    524    0     0     49    21.4  armor_03 @5300
M   optimus1-prebuilt      order1   53/70    28.3     2.05    549    0     0     41    21.6  diamond_01 @6200
M   optimus1-prebuilt      order2   52/70    30.8     2.01    533    0     1     47    22.0  
M   optimus1-prebuilt-goalfix order0   49/70    26.5     2.20    524    0     1     18    21.7  gold_05 @3400
M   optimus1-prebuilt-goalfix order1   42/70    31.0     2.07    525    0     0     19    21.8  redstone_01 @8600
M   optimus1-prebuilt-goalfix order2   35/70    20.0     2.00    501    0     1     11    20.9  armor_00 @26400
O   deps                   order0   50/70    20.0    45.27   2221    0     0      0    38.7  armor_08 @400
O   jarvis1                order0   70/70    10.0    52.60   9434    0     1      0    28.5  
O   mineevolve             order0   43/70    14.0    32.30   1578    0     0      0    22.3  armor_04 @500
O   mineevolve             order1   44/70     4.5    34.52   1733    0     0      0    21.9  redstone_00 @5200
O   mineevolve             order2   35/70     8.6    34.73   1972    0     0      0    20.6  armor_00 @30700
O   optimus1-empty         order0   70/70    18.6     0.53    187    0     0     58     5.0  
O   optimus1-empty         order1   70/70     1.4     0.24     91    0     0     70     1.5  
O   optimus1-empty         order2   70/70     5.7     0.38    133    0     0     66     3.5  
O   optimus1-empty-goalfix order0   70/70    20.0     1.46    387    0     0     39    11.4  
O   optimus1-empty-goalfix order1   70/70     1.4     0.24     96    0     0     70     1.7  
O   optimus1-empty-goalfix order2   70/70    15.7     1.28    356    0     0     48    11.5  
O   optimus1-prebuilt      order0   70/70    44.3     2.01    568    0     0     56    17.0  
O   optimus1-prebuilt      order1   70/70    48.6     1.62    437    0     0     54    12.1  
O   optimus1-prebuilt      order2   70/70    45.7     2.46    687    0     0     53    19.0  
O   optimus1-prebuilt-goalfix order0   70/70    51.4     2.92    687    0     0     29    20.4  
O   optimus1-prebuilt-goalfix order1   70/70    38.6     2.70    723    0     0     34    20.5  
O   optimus1-prebuilt-goalfix order2   70/70    41.4     2.86    756    0     0     30    20.0  

LLM spend (ledger, all runs incl. smoke/debug): $586.33 / cap $3000   per env: {'C3': 0.0, 'O': 218.12, 'M': 237.92}   disk free: 571.2 GB
```

## 2026-09-25 — daily summary (auto)

```
env chain                  order      done  succ%    cost$  calls anom crash fallbk  wall_h  running
C3  optimus3               order0   70/70    28.6     0.00    521    0     0      0    14.4  
M   deps                   order0   70/70    25.7    29.09   1559    0     0      0    52.3  
M   jarvis1                order0   70/70    14.3    52.17   9009    0     1      0    32.7  
M   mineevolve             order0   70/70    10.0    46.09   2367    0     0      0    35.0  
M   mineevolve             order1   70/70    11.4    65.11   3399    0     0      0    43.0  
M   mineevolve             order2   70/70    14.3    47.70   2393    0     0      0    35.6  
M   optimus1-empty         order0   70/70     4.3     0.49    148    0     0     70     4.1  
M   optimus1-empty         order1   70/70     8.6     1.59    374    0     0     63    14.3  
M   optimus1-empty         order2   70/70     2.9     0.35    126    0     0     65     4.1  
M   optimus1-empty-goalfix order0   70/70     5.7     0.32    126    0     0     65     3.6  
M   optimus1-empty-goalfix order1   70/70    11.4     0.97    257    0     0     56    10.4  
M   optimus1-empty-goalfix order2   70/70    12.9     2.39    623    0     0     40    26.1  
M   optimus1-prebuilt      order0   70/70    34.3     2.82    747    0     0     62    30.1  
M   optimus1-prebuilt      order1   70/70    25.7     2.76    739    0     1     55    28.5  
M   optimus1-prebuilt      order2   70/70    31.4     2.80    742    0     1     61    30.3  
M   optimus1-prebuilt-goalfix order0   70/70    25.7     3.73    862    0     1     25    35.4  
M   optimus1-prebuilt-goalfix order1   70/70    27.1     3.52    918    0     1     33    37.9  
M   optimus1-prebuilt-goalfix order2   70/70    31.4     3.50    897    0     1     24    37.0  
O   deps                   order0   70/70    18.6    63.85   3201    0     0      0    56.8  
O   jarvis1                order0   70/70    10.0    52.60   9434    0     1      0    28.5  
O   mineevolve             order0   70/70    10.0    57.18   2969    0     0      0    37.2  
O   mineevolve             order1   70/70     8.6    53.37   2755    0     0      0    33.6  
O   mineevolve             order2   64/70    12.5    63.47   3603    0     0      0    38.1  gold_00 @700
O   optimus1-empty         order0   70/70    18.6     0.53    187    0     0     58     5.0  
O   optimus1-empty         order1   70/70     1.4     0.24     91    0     0     70     1.5  
O   optimus1-empty         order2   70/70     5.7     0.38    133    0     0     66     3.5  
O   optimus1-empty-goalfix order0   70/70    20.0     1.46    387    0     0     39    11.4  
O   optimus1-empty-goalfix order1   70/70     1.4     0.24     96    0     0     70     1.7  
O   optimus1-empty-goalfix order2   70/70    15.7     1.28    356    0     0     48    11.5  
O   optimus1-prebuilt      order0   70/70    44.3     2.01    568    0     0     56    17.0  
O   optimus1-prebuilt      order1   70/70    48.6     1.62    437    0     0     54    12.1  
O   optimus1-prebuilt      order2   70/70    45.7     2.46    687    0     0     53    19.0  
O   optimus1-prebuilt-goalfix order0   70/70    51.4     2.92    687    0     0     29    20.4  
O   optimus1-prebuilt-goalfix order1   70/70    38.6     2.70    723    0     0     34    20.5  
O   optimus1-prebuilt-goalfix order2   70/70    41.4     2.86    756    0     0     30    20.0  

LLM spend (ledger, all runs incl. smoke/debug): $698.21 / cap $3000   per env: {'C3': 0.0, 'O': 309.18, 'M': 265.4}   disk free: 538.3 GB
```

### 09-26 06:05 — API-free diagnostic D1 prepared, launched, and stopped (GPUs taken by another user)
Prepared an API-free diagnostic (`configs/run_plan_diagD1.yaml`, LLM layer in mock mode): STEVE-1 only (no planner), Wood+Stone × 3 seeds, in Env M (cond 4.0), Env M (cond 6.0) and Env O — to compare with the MineEvolve paper's STEVE-1-only row and separate env/controller effects from method effects. GPUs 0/1/6 were idle at 05:47, but another user's job (~22 GB on each of GPUs 0,1,2,3,6,7) started before our launch at 05:57, so the run was stopped at once and its partial results deleted. To rerun when GPUs are free: re-check `nvidia-smi` (memory.used 0 on the chosen GPUs), set `gpus:` in the plan, then `python scripts/supervise.py --plan configs/run_plan_diagD1.yaml`.

## 2026-10-02 — Optimus-1 only; env restored; Wood fixed; Stone diagnosis

**Scope.** Decided to keep only Optimus-1 (its own 67-task suite, D34) as the baseline; MineEvolve, DEPS,
JARVIS-1, Optimus-3, Env M and VoLoAgent removed from the tree (still in git history).

**Isoworld results (10-01, 6 seeds, logfix+memfix+isoworld).** Wood 58/60 = 96.7% (paper 98.6%),
Stone 42/54 = 77.8% (paper 92.4%); 114/114 episodes ran in the requested world (`world_seed_ok`).
Released code on the same tasks: Wood 27/30 = 90.0% (all 3 failures = chest typo, fixed by memfix).

**Remaining Wood failures.** wooden_00 seed1: table placed over water, GUI never opened, released code
clicks on regardless until timeout → `craftfix` (D37); rerun of that world succeeded (fix path not hit).
wooden_03 seed4: plain timeout (chopping took ~3500/3600 steps).

**Stone failures.** 8 of 12 never got a log; the released stone.yaml uses `prefer_biome: plains`
(all other groups forest). Running plains vs forest (D38), 6 seeds each, GPUs 2-3 (LAUNCHES.md).
`stone_01` is a memory hit (stored plan used without a planner call), so `plan_source=example_fallback`
is a mislabel there, not a planner failure.

**Env (D39).** Container reset wiped /opt/conda/envs/mcagent and system Java. The original env was copied
from the old container to /home/rag/data/conda_envs/mcagent_orig (symlinked). `setup_env.sh` rebuilt to
pin envs/mcagent.lock.txt (= original freeze): a test build installed all 142 pinned packages identical
to the original. The authors' MCP-Reborn jar was deleted by minerl's setup.py during a rebuild and
restored from their tarball (same world verified on a same-seed rerun).

**Known infra crash.** `minerl spaces.sample: numpy.float64 cannot be interpreted as an integer` at
~step 6 (66 crashed attempts across all suite_optimus1 variants); the chain retries the same seed.

### 10-02 (cont.) — Stone: cause found (REPLAN never acted on); final evaluation launched

| Stone, 6 seeds (54 ep.) | success |
|---|---|
| logfix+memfix+isoworld, released biome (plains) — 10-01 | 42/54 = 77.8% |
| + craftfix, plains | 42/54 = 77.8% |
| + craftfix, forest (D38 diagnostic) | 49/54 = 90.7% |
| paper Table 9 | 92.4% |

* Plains failures: 11 of 12 had ≥1 reflector REPLAN that the released code ignores; without a REPLAN 39/40 succeed (D41).
  Forest only looks better because the agent rarely falls into a pit / digs into a hillside when trees are close.
* Biome: not stated in the paper ("random start point"); the authors' memory labels Stone episodes ~half forest / half
  plains (Wood ~95% forest, calibrated on our own runs), so the released `plains` is kept for the final run.
* replanfix (planner re-query, as the authors suggest in issue #11): 0/7 on previously failed worlds.
  escapefix (scripted build_tower / go_to_land, D42): 2/7, logs obtained after escaping in 3 of 4 stone_01 worlds.
* tagfix (D40): smoker with mixed log types; A/B in a fixed world: released fails the craft, tagfix crafts at once.

**Final evaluation** (10-02 09:45, GPUs 2-3, 18 chains): variant
`prebuilt-logfix-memfix-craftfix-tagfix-replanfix-escapefix-isoworld-g38`, all 67 tasks × 6 seeds, orders
`final<k>_<a|b|c>`, released biomes and paper horizons. Expected ~6 h, ~$30.

### 10-02 (cont.) — Stone in plains reaches the paper; final evaluation relaunched

| Stone, released biome (plains), 6 seeds | success |
|---|---|
| logfix+memfix+isoworld (+craftfix) | 42/54 = 77.8% |
| + tagfix, promptfix, replanfix, escapefix v2 → v5 (D41-D44) | 43 → 46 → 47 → 46 → **50/54 = 92.6%** |
| paper Table 9 | 92.4% |

Remaining v5 failures: ocean spawn (stone_04 seed0, never reaches trees), slow cobblestone mining (7/8 at timeout),
furnace/charcoal timeouts after escapes. The final run includes Stone again = an independent second sample.
Final evaluation relaunched 10-02 (variant `prebuilt-logfix-memfix-craftfix-tagfix-promptfix-replanfix-escapefix5-isoworld-g38`,
67 tasks × 6 seeds, orders `final<k>_<a|b|c>`, --retries 4). The aborted earlier final (escapefix v1) is superseded.

## 2026-10-03 — final evaluation (67 tasks × 6 seeds) and failure analysis

Variant `prebuilt-logfix-memfix-craftfix-tagfix-promptfix-replanfix-escapefix5-isoworld-g38`, released biomes and
horizons, Gemini 3.8 Flash, 402 episodes, $25.5. One crashed_final (socket timeout) and one anomaly (Gemini proxy 500)
were moved aside and rerun singly (LAUNCHES.md); 0 infra failures remain.

| Group | ours | paper (Table 1) | near-released run (`logfix` only, 10-01) |
|---|---|---|---|
| Wood | 56/60 = 93.3% | 98.6 | 26/31 = 83.9 |
| Stone | 48/54 = 88.9% (two runs: 98/108 = 90.7%) | 92.35 | 19/28 = 67.9 |
| Iron | 69/96 = 71.9% | 46.69 | 31/48 = 64.6 |
| Gold | 18/36 = 50.0% | 8.51 | 5/18 = 27.8 |
| Diamond | 20/42 = 47.6% | 11.61 | 7/21 = 33.3 |
| Redstone | 32/36 = 88.9% | 25.02 | 14/18 = 77.8 |
| Armor | 41/78 = 52.6% | 19.47 | 20/39 = 51.3 |
| Overall (mean of I/G/D/R/A) | 62.2 | 22.26 | 51.0 |

* Wood/Stone are within ~1.5 SE of the paper (Wood 10-01 + 10-03 = 114/120 = 95.0%). The 4 Wood worlds that failed
  all succeeded on 10-01: 2 crafting tables placed in/next to water (GUI never opened; one agent drowned), 1 stuck in
  water, 1 slow chopping — run-to-run variance, no new code path involved before the failure.
* The other groups are 2–6× the paper, and were already far above with near-released code, so this is not from our fixes.
  Likely: the released Env O helpers (ore spawning 10% per STEVE-1 step, diamond at any y ≤ 14; auto best pickaxe below
  y 70; see the 09-23 env table) and a stronger planner than GPT-4V. Whether the paper's runs used those helpers is not
  stated. **Wood/Stone reproduce; the harder groups do not match the paper (they are higher).**

### Failure analysis (`analysis/final_failures.py` → analysis/out/final_failures.{csv,txt}, not committed)

118 failures, 116 at the horizon. One primary cause each (first rule that matches):

| cause | all | Wood+Stone | Iron | G/D/R/A |
|---|---|---|---|---|
| ore not found within the horizon | 40 | 0 | 2 | 38 |
| terrain trap (pit / ravine / water), escape failed | 30 | 3 | 18 | 9 |
| ore not found, time lost in a trap earlier | 14 | 1 | 1 | 12 |
| slow gathering (wood / cobblestone) | 12 | 2 | 2 | 8 |
| ours: replan sub-goal with a dirt goal ("pillar up using dirt", "select block") | 7 | 0 | 1 | 6 |
| craft / smelt failed | 7 | 2 | 3 | 2 |
| ours: promptfix v1 rewrote a non-mined goal (D45) | 5 | 0 | 0 | 5 |
| crafting-table GUI never opened | 3 | 2 | 0 | 1 |

* A reflector REPLAN appears in 101/118 failures vs 24/284 successes: getting stuck is the main failure signal.
* Iron's 18 traps: the agent falls into a stone ravine/cave while looking for trees (plains spawn) with an empty
  inventory; build_tower has no block and cannot dig stone by hand (exited=False every time, up to 13 escapes per
  episode); the reflector still says drop_down each time.
* Our fixes' own issues: promptfix v1 over-match (fixed as promptfix2, D45, check running); replanfix accepts replan
  sub-goals whose goal is dirt, which pillaring consumes (8/8 such episodes failed) — not fixed yet.
* Planner goal `stone` for a mining step (mining stone yields cobblestone): 1/4 succeeded; matches the earlier
  "smelt stone" 0/90 finding.
