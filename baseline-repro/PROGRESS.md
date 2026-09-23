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
