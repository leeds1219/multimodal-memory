# Environment

The machine, the software versions and the Minecraft world the experiments run in.

## Versions used

Rebuild exactly these; `setup/setup_env.sh` installs them.

| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU / driver | NVIDIA RTX PRO 6000 Blackwell, driver 580.105.08 |
| Java | OpenJDK 1.8.0_504 (`openjdk-8-jdk-headless` 8u504-ga-1ubuntu1~22.04.3); Minecraft needs Java 8 |
| Virtual display | Xvfb 21.1.4 (`xvfb-run`) |
| Python | 3.10.21 (conda env `mcagent`) |
| PyTorch | 2.9.1+cu128 (CUDA 12.8, cuDNN 9.10.02) |
| Python packages | pinned in `setup/mcagent.lock.txt` (key ones: minerl 1.0.2, gym 0.23.1, numpy 1.23.5, transformers 4.44.2, hydra-core 1.3.7, openai 3.19.0) |
| Minecraft | 1.16.5, the authors' prebuilt MCP-Reborn `mcprec-6.13.jar` (sha256 `ebd503dd…5357fdd095`) |
| Optimus-1 | github.com/iLearn-Lab/NeurIPS24-Optimus-1 @ `409356cc` + `setup/optimus1.patch` |
| STEVE-1 | github.com/Shalev-Lifshitz/STEVE-1 @ `903b2447`, weights from Optimus-1's `optimus1_steve1_ckpt.zip` |
| MineCLIP | github.com/MineDojo/MineCLIP @ `e6c06a02` + `setup/mineclip_compat.patch` |
| LLM | `gemini-3.8-flash` through Google's OpenAI-compatible endpoint (`configs/llm_g38mon.yaml`) |

Known traps:
- **Do not run `minerl/setup.py` as is.** Unless `READTHEDOCS=1` is set, it deletes the prebuilt MCP-Reborn and
  rebuilds it. The setup script sets the variable.
- **transformers 5.x breaks the import of Optimus-1.** Keep 4.44.2.
- **The patched Optimus-1 imports `gemini_client` by name.** The setup script writes `llm/` into a `.pth` file of
  the env.
- **World saves need the patched jar.** `setup/build_saveall_jar.py` builds it, and only runs with world saves use
  it. In the released client the player cannot run `/save-all`.

## Setup

```bash
bash setup/setup_env.sh           # apt packages, conda env, authors' repos at the commits above, patches, jar, weights
python setup/build_saveall_jar.py # jar that allows /save-all (world saves, experiment 6)
```
Outside the repository, the authors' repos, the conda env and caches live on a data volume given by `MM_DATA`
(`paths.py`, `setup/setup_env.sh`).
API key: set the environment variable `GOOGLE_API_KEY`, or write `GOOGLE_API_KEY="..."` into `env.yaml` at the
repository root (git-ignored).

## LLM calls

Every LLM call goes through `llm/gemini_client.py`:
- **Ledger.** Each call's cost is logged to `run_data/llm_ledger.jsonl`, and the call itself to
  `<episode>/llm/calls.jsonl`.
- **Caps.** There is a global cap, an hourly cap and a per-episode runaway guard (`configs/llm_g38mon.yaml`). When
  one trips, the client writes `STOP_BUDGET`, every later call fails at $0, and chains stop before their next
  episode.
- **Retries.** Rate limits, timeouts and server errors are retried with backoff. A 400 INVALID_ARGUMENT, which
  Gemini sometimes returns for an ordinary request, is retried up to 3 times. The first rejected request is saved
  as `bad_request_<time>.json`.

## Crashes

When Minecraft stops answering, the episode is marked crashed and `experiments/chain.py` retries it up to 3 times.
A crash never counts as a task failure.

## The Minecraft world

The world is Optimus-1's released benchmark env (Li et al., NeurIPS 2024), taken as it is, so results compare
directly with Optimus-1. Source: **Optimus-1 code** = the released code (`src/optimus1/env/`, `conf/`), **Optimus-1
paper** = a value from the paper where the code differs, **ours** = set by this repository.

| Setting | Value | Source |
|---|---|---|
| Observation | 640×360 screen, FOV 70, gamma 2, GUI scale 1 | Optimus-1 code (`env/custom_env.py`) |
| Time | always day: `/time set 0`, `doDaylightCycle false` | Optimus-1 code (`conf/evaluate.yaml` reset commands) |
| Night vision | permanent (`/effect give @a night_vision 99999 250`) | Optimus-1 code (reset commands) |
| Death | items kept (`keepInventory true`), immediate respawn | Optimus-1 code (reset commands) |
| Monsters | spawn as normal (`allow_spawning=True`); difficulty not set (MineRL default) | Optimus-1 code (`env/custom_env.py`) |
| Command output | hidden from the screen (`sendCommandFeedback false`) | Optimus-1 code (reset commands) |
| Spawn biome | forest for wooden, gold, redstone, diamond, armor; plains for stone, iron | Optimus-1 code (`conf/benchmark/<group>.yaml`) |
| Start inventory | empty | Optimus-1 code (`conf/benchmark/<group>.yaml`) |
| Hidden ore rule | when the agent first reaches a height, 10% chance per step that an ore is placed 3-5 blocks below it, once per height; coal 45-50, iron 26-43, gold 15-26, redstone ≤ 16, diamond ≤ 14 | Optimus-1 code (`env/wrapper.py` `random_ore`) |
| Action filters | dropping items is disabled; while attacking, jump/strafe/sneak/sprint are released; below y 70 the best pickaxe in the hotbar is selected automatically | Optimus-1 code (`env/wrapper.py`) |
| Tasks | the 67 craft tasks of the benchmark | Optimus-1 paper (Table 5) and code (`conf/benchmark/`) |
| Horizon | 3,600 / 7,200 / 12,000 / 36,000 steps (wooden / stone / iron / gold and harder); 20 steps = 1 s | Optimus-1 paper (Table 5); the code uses 2 / 5 / 20 / 30 minutes |
| Success | the target item in the inventory, checked every step | Optimus-1 paper (the code only checks that the plan finished) |
| World and seed | a new world per episode, seeded with `env.seed(seed)`; seeds in `configs/suites/optimus1/seeds.yaml`; worlds A, B, C = seed offsets 0, 1, 4 | ours |

What our code changes in every run (ours):

| Change | Why | Where |
|---|---|---|
| A private Minecraft folder per episode | In the released setup, parallel instances share `saves/`, so the same seed could give a different world | `world/private_world.py` |
| No "stuck" `/kill` | The released wrapper kills an agent that stays 8,000 steps at one height; the VLM decides instead | `system1/controller.py` |
| No code stop on health or falls | The VLM decides | `system1/controller.py` |

World options of an experiment (`--world`, `--save`):

| Option | Effect |
|---|---|
| `--world nolava` | Every 20 steps, lava within 12 blocks of the agent becomes stone. Nothing appears on screen, and no game time is added |
| `--save N` | The world is saved every N steps, as starting points for the replay world. No game time is added |
