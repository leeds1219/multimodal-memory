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
| 7 | Baseline + Optimus-1 memory | Optimus-1's knowledge graph and experience pool, learned online on train tasks | standard | `--method optimus1_memory --o1m <name>` (see below) |

Every run also takes `--prefix <name> --gpus <list> --port <first port>`. Results are in
`results/<method>/<world>/`.

Two tasks need more gold than the ore rule can ever place (golden leggings, golden chestplate), so they need natural
ore, which is luck within the time limit. They stay in the totals; `analysis/summary.py` also reports success on
the tasks the rule can supply and on these two separately (`analysis/ore_supply.py`).

## 1. Optimus-1 reproduction

The released Optimus-1, run with gemini-3.8-flash as its planner. Corrections to bugs in the release are switches,
each described in `experiments/optimus1/optimus1_agent.py`.

## 2. Baseline

At each turn the game pauses. The VLM (gemini-3.8-flash) sees the screen and its height, position relative to the
start, facing, health and inventory. It also sees its last 10 commands and what they did. It sends one command,
which runs for the number of steps it chose: raw keys, a STEVE-1 instruction, or an equip / craft / smelt helper.
The helpers are Optimus-1's own and are kept as released, quirks included: coping with an imperfect tool is part of
what a method should learn. Code: `domains/minecraft/agent/` (with `executor/extras.py`).

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

The hidden ore rule (ENVIRONMENT.md) is stated in the prompt, in one of two versions (`domains/minecraft/agent/prompts.py`):

| Version | Method | Prompt text |
|---|---|---|
| v1 | `gt_world_rule` | the rule in short (`ORE_HINT`) |
| v2 | `gt_world_rule_v2` | what the placing code does, step by step (`ORE_RULE_CODE`) |

Episodes that fail with v1 on tasks the rule can supply are run again with v2. If v2 solves them, the no-lava
world is solvable once the rule is fully known.

## 6. Ours

In the nolava world, lava near the agent becomes stone, so that deaths in lava do not hide what knowledge changes.

| Step | Command | Results |
|---|---|---|
| Upper bound: rule told | `run.py --method gt_world_rule --world nolava` | `results/gt_world_rule/nolava/` |
| Baseline with world saves | `run.py --method baseline --world nolava --save 600` | `results/baseline/nolava_save600/` |
| Learning loop on its failures | `ours_loop.py --name <name> --episodes <list> --store <name>` | `results/ours_loop/<name>/` |
| Evaluation with the knowledge | `run.py --method ours --knowledge <name> --world nolava` | `results/ours_<name>/nolava/` |

The learning loop runs on each failed episode:
1. The Designer lists the failure points (`designer/failure_point.py`): turns whose decision was a mistake from what the
   agent could know then (or its response to an effect it did not decide, such as an item lost) and that likely
   caused the failure. The loop takes them earliest first.
2. It proposes a hypothesis (`designer/hypothesis.py`), seeing the other failure points of the list as well as the
   agent's own turns around this one (8 before, 3 after): the problem the decision caused later, the belief about
   how to avoid it, the change that tests the belief there, the moment the belief is checked at (the next time the
   agent needs what the problem was about) and what it predicts then.
3. The game restarts from the last save before that turn (saves are 600 steps apart), and the agent tries the
   change. It is told the change and how to make it, not the prediction (`domains/minecraft/world/replay.py`).
4. The Designer judges the experiment (`designer/judge.py`) from what the game recorded, not the agent's thoughts:
   Not reached (the check moment never came), Supported or Refuted (what the log shows at the first check
   moment). Task success does not decide it. A Supported problem and belief go to World Knowledge.
5. Otherwise the loop proposes again at the same failure point, with the hypotheses tested there and what their
   experiments showed; after 3 experiments there without support, it moves to the next failure point of the list.
   If it was supported, the loop lists the failure points again, in the experiment's log. When the failure points
   of a list are used up, it lists them again in the latest experiment's log. The loop stops on task success or
   after 10 experiments (or when the latest experiment's log has no failure point).

## 7. Baseline + Optimus-1 memory

Optimus-1's Hybrid Multimodal Memory (Li et al., NeurIPS 2024) on the Baseline agent, implemented from the paper:
the Hierarchical Directed Knowledge Graph (HDKG) and the Abstracted Multimodal Experience Pool (AMEP). Memory is the
only difference from the Baseline. The agent also names the sub-goal of each command and judges its previous one
(COMPLETE, CONTINUE or REPLAN, the verdicts of Optimus-1's reflector), so experiences can be stored per sub-goal.
- HDKG: only what the agent's own episodes showed (crafts and smelts that worked, materials a failed one named,
  items gathered); never the recipe files. The target's subgraph, with amounts, is in the system prompt.
- AMEP: per sub-goal window, up to 16 frames, the start and end state and the verdict. Every 1200 steps on one
  sub-goal, the closest past case for each verdict is shown.

| Phase | Command | Store |
|---|---|---|
| 1. Learn on the train tasks | `run.py --method optimus1_memory --o1m <name> --tasks <train tasks>` | starts empty, written after each episode |
| 2. Test on unseen tasks | `run.py --method optimus1_memory --o1m <name> --frozen` | read only |

The store is `run_data/memory/o1m_<name>/`, shared by the chains of a run. Results are in
`results/optimus1_memory_<name>[_frozen]/<world>/`. Code: `memory/optimus1_memory.py`; offline test:
`scripts/test_optimus1_memory.py`.

## Episode folder

| File | Content |
|---|---|
| `result.json` | success, steps, cost |
| `vlm/turns.jsonl` | one line per turn: command, thought, outcome, effect |
| `trajectory.jsonl.gz` | every step: position, health, inventory |
| `ckpt/` | world saves (only with `--save`) |
| `client.log` | Minecraft log. It lists the hidden ore placements, so no agent reads it |
