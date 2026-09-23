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
