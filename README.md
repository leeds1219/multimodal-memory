# multimodal-memory

Reproduction of **Optimus-1** (NeurIPS 2024, [iLearn-Lab/NeurIPS24-Optimus-1](https://github.com/iLearn-Lab/NeurIPS24-Optimus-1))
with a Gemini planner, as the baseline for our multimodal-memory work. Other baselines
(MineEvolve, DEPS, JARVIS-1, Optimus-3) and VoLoAgent were dropped on 2026-10-02; they
are still in git history before that date.

Everything lives in `baseline-repro/`:

| Path | What |
|------|------|
| `scripts/setup_env.sh` | builds the whole setup: env `mcagent` pinned to the original env's freeze (`envs/mcagent.lock.txt`), authors' repos at pinned commits + our patches, prebuilt simulator jar (sha256-checked), checkpoints |
| `scripts/chain.py` | runs one chain of episodes (one seed offset, a task list) on one GPU |
| `envs/optimus_episode.py` | one Optimus-1 episode; labelled fix variants are switched on here |
| `configs/suites/optimus1/` | the 67 craft tasks of the paper (Table 5), seeds, horizons |
| `baselines/optimus1/` | our patches to the authors' code and MineCLIP, memory repair |
| `DECISIONS.md` | every non-obvious choice and every fix variant, with evidence (D1–D39) |
| `PROGRESS.md`, `RESUME.md` | what was run and how to resume / check runs |

## Setup (Linux, NVIDIA GPU)

```bash
cd baseline-repro
bash scripts/setup_env.sh          # idempotent; env goes to /home/rag/data/conda_envs/mcagent
```

The planner key goes in `/home/rag/data/env.yaml` (`GOOGLE_API_KEY=...`); never commit it.

## Run

```bash
cd baseline-repro
REPRO_SUITE=optimus1 /opt/conda/envs/mcagent/bin/python scripts/chain.py --env O --method optimus1 \
  --variant prebuilt-logfix-memfix-craftfix-isoworld-g38 --order seed0 --seed-offset 0 \
  --tasks o1_stone_00,o1_stone_01 --gpu 2 --port 10200
```

Variant tokens (combine with `-`): `prebuilt` (authors' full memory), `logfix` (D33),
`memfix` (D35), `isoworld` (D36), `craftfix` (D37), `forest` (D38, diagnostic), `g38`
(gemini-3.8-flash planner). Results: `/home/rag/data/repro_runs/suite_optimus1/runs/O/<variant>/<order>/<task>/<seed>/result.json`.

See `CLAUDE.md` for the repo rules and `CONTRIBUTING.md` for the PR workflow.
