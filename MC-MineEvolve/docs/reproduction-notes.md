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

## Known mismatches we have NOT resolved

1. **World seeds** — *partly resolved 2026-09-18.* Upstream never sets a seed (the
   `env.seed` option exists but no config uses it), so every episode was a random
   world: face-to-trunk oak forest one run, dark forest / birch grove / hillside the
   next. The paper evaluates on a fixed "task–seed split" but does not publish the
   seeds. We now run every task once per seed in `seeds: [101, 102, 103]`
   (`conf/evaluate.yaml`; `main.py` calls `env.seed()` before each reset, and each
   run is appended to `<hydra output dir>/runs.jsonl` with its seed). This is *our*
   split, not the paper's — results are reproducible, not comparable seed-for-seed.
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
| 09-18 | gemini-3-flash-preview (clean) | see `logs/llm_calls.jsonl` | 8+ | spawned facing a hill, birch trees 15 blocks away; attack-only, moved 0 blocks |

## Next steps we agreed on

- Run task 8 across the 3 default seeds before drawing conclusions (`seeds` is now fixed).
- Compare `gemini-3-flash-preview` vs `gemini-3.8-flash` on the same seeds:
  repair count and call composition (from `logs/llm_calls.jsonl`).
- Only then decide the local-model size for repeated experiments.
