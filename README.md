# multimodal-memory

Can a VLM agent in Minecraft learn a hidden rule of its world from its own failures? The agent (System 1) plays a
task. When it fails, System 2 reads the episode log, finds the decision that caused the failure, and proposes a
hypothesis. The agent then tests the hypothesis by replaying the world from a save, and verified statements become
World Knowledge for later episodes. The hidden rule is how ore appears (ENVIRONMENT.md).

- `EXPERIMENTS.md`: what is compared, and how to run each condition.
- `ENVIRONMENT.md`: versions, setup and the Minecraft world.

## Layout

```
system1/        the acting agent: VLM turn loop (vlm_agent.py), prompts, controller
  tools/        what a command runs: raw keys, STEVE-1, Optimus-1's equip/craft/smelt helpers
system2/        reasoning over episode logs: failure_point.py, hypothesis.py, judge.py, episode_log.py
world/          Minecraft: one episode (episode.py), world saves (save.py), replay world (replay.py),
                tasks / seeds / success check (benchmark.py), private world folder per episode
memory/         World Knowledge store, MineEvolve memory
experiments/    conditions.py (method x world), run.py (start a condition), chain.py (episodes on one GPU),
                ours_loop.py (the learning loop), optimus1/ (the Optimus-1 reproduction)
analysis/       results: summary tables, episode browser pages
llm/            the one LLM client (cost ledger, caps, retries, logs) and a mock for $0 tests
configs/        LLM configs, the 67 benchmark tasks, seeds, recipe graphs
setup/          environment setup, patches to the authors' code, pinned packages
scripts/        server start, status, cleanup, the replay-world test
```

## Quick start

```bash
bash setup/setup_env.sh && python setup/build_saveall_jar.py        # once (ENVIRONMENT.md)
python experiments/run.py --method baseline --prefix b1 --gpus 0 --port 29000 --dry-run   # print the 15 chains
python experiments/run.py --method baseline --prefix b1 --gpus 0 --port 29000             # run them
python experiments/chain.py --method baseline --order t1 --tasks o1_gold_00 --gpu 0 --port 29000 --mock  # $0 test
python analysis/browse.py baseline/standard && python -m http.server 8800   # episode pages: /run_data/site/
```

A chain skips episodes it already finished, so a new run of a condition needs a new `--prefix`.
Results: `results/<method>/<world>/<order>/<task>/<seed>/`; memory stores, chain state, logs and the LLM ledger:
`run_data/` (both git-ignored).
