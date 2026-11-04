# multimodal-memory

Can a VLM agent in Minecraft learn a hidden rule of its world from its own failures? The agent (the Executor) plays a
task. When it fails, the Designer reads the episode log, finds the decision that caused the failure, and proposes a
hypothesis. The agent then tests the hypothesis by replaying the world from a save, and verified statements become
World Knowledge for later episodes. The hidden rule is how ore appears (ENVIRONMENT.md).

- `EXPERIMENTS.md`: what is compared, and how to run each condition.
- `ENVIRONMENT.md`: versions, setup and the Minecraft world.

## Layout

```
executor/       what the method adds to any agent: World Knowledge and the experiment's change (extras.py)
designer/       reasoning over episode logs: failure_point.py, hypothesis.py, judge.py (domain-free prompts)
memory/         World Knowledge store, MineEvolve memory
experiments/    conditions.py (method x world), run.py (start a condition), chain.py (episodes on one GPU),
                ours_loop.py (the learning loop), optimus1/ (the Optimus-1 reproduction)
domains/        everything specific to one environment; base.py = what a domain gives the method
  minecraft/    domain.py (that interface), episode_log.py (the log the Designer reads), loop_ui.py (results page)
    agent/      the Executor: VLM turn loop (vlm_agent.py), prompts, controller, tools/ (keys, STEVE-1, helpers)
    world/      one episode (episode.py), world saves and replay, tasks / seeds / success check (benchmark.py),
                world rules (nolava.py), private world folder per episode
analysis/       results: summary tables, episode browser pages, loop results page (loop_ui/)
llm/            the one LLM client (cost ledger, caps, retries, logs); backends in providers.py (Gemini, OpenAI-compatible/local, mock)
configs/        LLM configs, the 67 benchmark tasks, seeds, recipe graphs
setup/          environment setup, patches to the authors' code, pinned packages
scripts/        server start, status, cleanup, the replay-world test
```

### Another domain or model

- **Another model:** a config in `configs/` with `provider:` (`llm/providers.py`); a local model runs behind vLLM or
  Ollama (`configs/llm_local_example.yaml`).
- **Another environment:** a folder `domains/<name>/` with `domain.py` implementing `domains/base.py` (read
  episodes, saves and states; run an episode), its own Executor agent that appends `executor/extras.py`'s text to its
  prompt, and a `Vocabulary` (the words the Designer's prompts use). Select it with `MM_DOMAIN=<name>`. The Designer, the
  loop and the LLM layer stay as they are.

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
