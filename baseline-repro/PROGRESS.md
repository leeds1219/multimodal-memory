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
