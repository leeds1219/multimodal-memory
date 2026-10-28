# Experiments

A condition is a method in a world. Conditions are compared on the 32 hard tasks (gold, redstone, diamond, armor)
in worlds A, B and C: 96 episodes each.

| # | Method | What the agent has | World | Command (`python experiments/run.py ...`) |
|---|---|---|---|---|
| 1 | Optimus-1 reproduction | released Optimus-1 planner, memory and reflection | standard | `--method optimus1` |
| 2 | Baseline | VLM that plays the whole task; no memory, no recipes | standard | `--method baseline` |
| 3 | Baseline + MineEvolve | MineEvolve memory, learned online | standard | `--method mineevolve` |
| 4 | Baseline + GT Recipe | the recipe graph in the prompt | standard | `--method gt_recipe` |
| 5 | Baseline + GT World Rule | the hidden ore rule in the prompt | standard | `--method gt_world_rule` |
| 6 | Ours | World Knowledge learned from its own failures | nolava | see below |

Every run also takes `--prefix <name> --gpus <list> --port <first port>`. Results are in
`results/<method>/<world>/`.

## 1. Optimus-1 reproduction

The released Optimus-1, run with gemini-3.8-flash as its planner. Corrections to bugs in the release are switches,
each described in `experiments/optimus1/optimus1_agent.py`.

## 2. Baseline

At each turn the game pauses. The VLM (gemini-3.8-flash) sees the screen and its height, position relative to the
start, facing, health and inventory. It also sees its last 10 commands and what they did. It sends one command,
which runs for the number of steps it chose: raw keys, a STEVE-1 instruction, or an equip / craft / smelt helper.
The helpers are Optimus-1's own and are kept as released, quirks included: coping with an imperfect tool is part of
what a method should learn. Code: `system1/`.

## 3. Baseline + MineEvolve

MineEvolve (arXiv 2603.13131v3), implemented from the paper on the Baseline agent (Eq. 1-8: monitor, inducer,
curator, adaptor). The paper is followed where its released code differs from it. The memory is shared by all
episodes of the run, in `run_data/memory/me_mineevolve.json`; it starts empty when that file does not exist (move
the file away to start over). Code: `memory/mineevolve_memory.py`.

## 4. Baseline + GT Recipe

The recipe graph Optimus-1's planner receives is added to the prompt. Two errors of the released graph are fixed:
gold ingots come from smelting gold ore, and each ore count matches the ingots or gems needed. Code:
`setup/build_craft_graphs.py`, which writes `configs/suites/optimus1/craft_graphs.json`.

## 5. Baseline + GT World Rule

The hidden ore rule (ENVIRONMENT.md) is stated in the prompt. Code: `ORE_HINT` in `system1/prompts.py`.

## 6. Ours

In the nolava world, lava near the agent becomes stone, so that deaths in lava do not hide what knowledge changes.

| Step | Command | Results |
|---|---|---|
| Upper bound: rule told | `run.py --method gt_world_rule --world nolava` | `results/gt_world_rule/nolava/` |
| Baseline with world saves | `run.py --method baseline --world nolava --save 600` | `results/baseline/nolava_save600/` |
| Learning loop on its failures | `ours_loop.py --name <name> --episodes <list> --store <name>` | `results/ours_loop/<name>/` |
| Evaluation with the knowledge | `run.py --method ours --knowledge <name> --world nolava` | `results/ours_<name>/nolava/` |

The learning loop runs on each failed episode:
1. System 2 finds the failure point, the turn whose decision caused the failure (`system2/failure_point.py`).
2. It proposes a hypothesis: a change, the result it predicts, how to try it, and a statement about the world
   (`system2/hypothesis.py`). It sees the agent's own turns around the failure point (8 before, 3 after).
3. The game restarts from the last save before that turn (saves are 600 steps apart), and the agent tries the
   change. It is told the change, the expected result and how to try it (`world/replay.py`).
4. System 2 judges whether the prediction was observed after the change, regardless of task success
   (`system2/judge.py`): Supported, Refuted, or Not tried (the change was never made). It reads only what the game
   recorded, not the agent's thoughts. A Supported statement goes to World Knowledge.
5. If the hypothesis was refuted, the loop proposes another at the same failure point. If it was supported, the
   loop looks for the next failure point. The loop stops on task success or after 5 experiments.

## Episode folder

| File | Content |
|---|---|
| `result.json` | success, steps, cost |
| `vlm/turns.jsonl` | one line per turn: command, thought, outcome, effect |
| `trajectory.jsonl.gz` | every step: position, health, inventory |
| `ckpt/` | world saves (only with `--save`) |
| `client.log` | Minecraft log. It lists the hidden ore placements, so no agent reads it |
