# MC-MineEvolve

Open-source implementation of **MineEvolve** — a knowledge-driven self-evolution framework for long-horizon embodied Minecraft agents.

This repository contains a minimal end-to-end implementation of the four-stage MineEvolve pipeline (Monitor → Inducer → Curator → Adaptor) on top of MineRL + STEVE-1

---

## What is here

```
MC-MineEvolve/
├── app.py                       # FastAPI server entry (loads STEVE-1 + LLM planner)
├── scripts/
│   ├── server.{sh,bat}          # start the server on port 9000
│   └── run_eval.{sh,bat}        # start the env-side client and run a benchmark
├── checkpoints/                 # placeholder; download STEVE-1/VPT/MineCLIP weights here
├── docs/
│   ├── architecture.md          # 4-stage data-flow diagram and module map
│   └── tasks.md                 # the 70-task split (paper Table 3)
└── src/mineevolve/
    ├── conf/                    # Hydra configs (benchmark/, llm/, evaluate.yaml)
    ├── env/                     # MineRL env spec + wrapper (dynamic ore + auto pickaxe)
    ├── monitor/                 # paper §3.2: TypedFeedback + ProgressScore + stagnation
    ├── inducer/                 # paper §3.2: BuildSkill / BuildRemedy from feedback
    ├── curator/                 # paper §3.3: validate / merge / retrieve external KB
    ├── adaptor/                 # paper §3.3: knowledge-conditioned local plan repair
    ├── planner/                 # high-level LLM planner + 5 OpenAI-compatible backends
    ├── executor/                # STEVE-1 loader + runner + craft helper
    ├── server/                  # FastAPI app (plan/action/monitor/induce/repair/...)
    ├── client/                  # HTTP client used by the env-side process
    ├── monitors/                # SuccessMonitor / StepMonitor for evaluation
    └── main.py                  # Hydra @main: Algorithm 1 main loop
```

This implementation is **completely independent**: no source files are copied from any other Minecraft agent repository. All third-party dependencies (MineRL, STEVE-1 / MineStudio, OpenAI SDK) are installed via `pip`.

---

## Installation

### Quick path (Linux + NVIDIA, conda, no root) — recommended

One script builds the whole `mineevolve` conda env, including Java 8, Xvfb,
MineRL 1.0.2 and MineStudio, without `sudo`:

```bash
bash scripts/setup_env.sh        # ~15-25 min, mostly the MineRL/Gradle build
conda activate mineevolve
```

Then verify everything except the LLM works (launches Minecraft headless,
resets, steps 200 random actions, records the run under `logs/smoke/run-<timestamp>/`):

```bash
xvfb-run -a python scripts/smoke_test.py
python scripts/plot_smoke.py            # -> logs/smoke/run-<latest>/summary.png
```

Each run dir holds `summary.json` (timings, inventory delta, pass/fail),
`trajectory.jsonl` (per-step coords / health / hunger / inventory / keys held) and
`frames/step_*.png` (POV every 10 steps; `--frame-every N` to change). `plot_smoke.py`
renders those into one PNG: frame strip, keys-held raster, distance / health / items.

To also verify STEVE-1 inference (GPU), start the server in another shell and
point the smoke test at it — no API key is needed for this path, the planner
only warns:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/server.sh &          # loads STEVE-1 (downloads HF weights on first run)
xvfb-run -a python scripts/smoke_test.py --server http://127.0.0.1:9000 --condition "chop a tree"
```

Notes:
- Always run the env-side process (`smoke_test.py`, `run_eval.sh`) under `xvfb-run -a`
  on a headless machine; the server does not need it.
- `scripts/setup_env.sh` is idempotent — re-run it to pick up new requirements.
- If you add a dependency, add it to `requirements.txt` **and** check the script still passes.
- Pitfalls the script works around (so you don't have to): PyPI's `minerl` is 0.4.x
  (too old, we need 1.0.x from GitHub); `gym==0.23.1` needs `setuptools<66` to build;
  MineStudio pins `opencv-python==4.8.0.74`, whose wheel needs `libGL.so.1` / `libgthread`
  (supplied from conda-forge, no `apt`); headless GL needs Mesa (`mesalib`) on
  `LD_LIBRARY_PATH`, which the `xvfb-run` shim sets.
- **MineRL is patched** (`patches/minerl-1.0.2-chat-commands.patch`, applied by
  `scripts/patch_minerl.sh`, which rebuilds the Minecraft jar). The patch adds the `chat`
  action, enables commands, and adds the slot `index` to the inventory observation that
  the GUI crafting controller needs (`executor/gui_craft.py`; test it with
  `xvfb-run -a python scripts/test_craft.py`). Stock MineRL 1.0 has no
  `chat` action and creates the world with commands disabled, so every `/gamerule`,
  `/effect` and `/setblock` this repo issues (`conf/evaluate.yaml::commands`, ore
  spawning in `env/wrapper.py`) was silently rejected. If you reinstall `minerl`,
  re-run `bash scripts/patch_minerl.sh`.

### Manual path

#### 1. System prerequisites

- Linux or Windows (WSL2 / native both work).
- Python **3.10+**.
- Java **OpenJDK 8** (required by MineRL's MCP-Reborn build).
- An NVIDIA GPU is recommended for STEVE-1 inference.
- `git`, `clang` (Linux) or MSVC build tools (Windows).

#### 2. Create environment

```bash
conda create -n mineevolve python=3.10 -y
conda activate mineevolve
pip install -r requirements.txt
pip install -e .
```

#### 3. Install MineRL

We do not vendor MineRL. Install the official package from the MineRL Labs (the version that supports STEVE-1 weights, e.g. MineRL 1.0.x):

```bash
# PyPI only hosts MineRL 0.4.x, which lacks the HumanSurvival spec we subclass;
# install 1.0.x from GitHub (needs Java 8 on PATH; compiles Minecraft, 10-20 min)
pip install "git+https://github.com/minerllabs/minerl@v1.0.2"
```

If you need to (re)build the Java backend yourself, follow MineRL's official docs:
<https://minerl.readthedocs.io/>.

#### 4. Install STEVE-1

Choose **one** of the two paths:

**(a) Recommended: MineStudio**

```bash
pip install MineStudio
```

The first call to `mineevolve.executor.steve_loader.load_steve_policy()` will pull `CraftJarvis/MineStudio_STEVE-1.official` from HuggingFace automatically; nothing needs to live in `checkpoints/`.

**(b) Original STEVE-1 package**

```bash
pip install "git+https://github.com/Shalev-Lifshitz/STEVE-1.git"
```

Then download:

| Weight                | Place at                             |
| --------------------- | ------------------------------------ |
| VPT 2x model          | `checkpoints/vpt/2x.model`           |
| STEVE-1 weights       | `checkpoints/steve1/steve1.weights`  |
| STEVE-1 prior         | `checkpoints/steve1/steve1_prior.pt` |
| MineCLIP attn weights | `checkpoints/mineclip/attn.pth`      |

See `checkpoints/README.md` for source URLs.

#### 5. LLM API keys

The default backend is **Qwen Plus** via DashScope's OpenAI-compatible endpoint
(no source change needed — both `conf/evaluate.yaml` and the FastAPI server
boot in Qwen mode):

```bash
# DEFAULT: Qwen via DashScope
export DASHSCOPE_API_KEY=sk-...
```

Other supported backends (set the matching key + pass `llm=<name>` on the
command line):

```bash
export ZHIPUAI_API_KEY=...    # then: llm=glm_4_7
export GOOGLE_API_KEY=...     # then: llm=gemini_flash
export OPENAI_API_KEY=...     # then: llm=gpt_5_5
```

---

## Debugging without an API key (local open-weights model)

Everything after the environment — planner, inducer, curator, adaptor — normally
needs a paid API. For debugging the code path you can run a small open model
instead; MineEvolve is untouched, it just talks to a local OpenAI-compatible
server (`scripts/local_llm_server.py`) through the same `openai_compat` backend.

```bash
# shell 1: local LLM (≈4 GB VRAM for Qwen3.5-2B in fp16, ~25 tok/s on a 2080 Ti)
CUDA_VISIBLE_DEVICES=3 python scripts/local_llm_server.py --model Qwen/Qwen3.5-2B --truncate-prompt

# shell 2: MineEvolve server pointed at it (STEVE-1 on its own GPU)
CUDA_VISIBLE_DEVICES=2 bash scripts/server_local.sh

# shell 3: one wooden task through the full Algorithm 1 loop
xvfb-run -a python -m mineevolve.main benchmark=wooden llm=local evaluate='[8]'
```

Artifacts land in `plans/` (initial + repaired plans), `evidence/` (per-subgoal
frames and step logs) and `memories/` (skill/remedy store) — all git-ignored.

To see what the same traffic would cost on the API backends, point
`scripts/llm_usage.py` at the local LLM server's log (token totals + a per-model
cost table; `--tasks 11` scales one task to the wooden tier):

```bash
python scripts/llm_usage.py llm.log --tasks 11
```

Expectations: a 2B model will *not* solve tasks; it produces plausible-looking
plans and repairs that let you watch every branch (parse failures, repeated
failures → Adaptor repair, "abort after 3 consecutive failures", …) for free.
Prompts grow with the feedback history (7.5k tokens by the 4th call); on an 11 GB
GPU ~10k tokens is the ceiling, hence `--truncate-prompt` (or `--max-prompt-tokens`
to just reject long prompts with a 413). `--model Qwen/Qwen3.5-0.8B` is faster and
smaller if you only care about plumbing; `--think` enables Qwen's thinking mode.

## Running

### Start the FastAPI server (one-time, GPU)

```bash
bash scripts/server.sh            # Qwen via DashScope (default), needs DASHSCOPE_API_KEY
bash scripts/server_gemini.sh     # Gemini 3 Flash (preview), needs GOOGLE_API_KEY (sets provider+model+base_url together)
# Windows:
scripts\server.bat
```

Keys are read from the git-ignored `.env` at the monorepo root (`cp .env.example .env`
and fill it in) or from your shell environment; never put one in a Hydra config.

The server:
- loads STEVE-1 once,
- holds the external knowledge store K and feedback buffer B,
- executes Monitor / Inducer / Curator / Adaptor on each `/chat` request.

### Run a benchmark group (env-side process)

```bash
# DEFAULT: wooden tier with Qwen Plus (just needs DASHSCOPE_API_KEY)
bash scripts/run_eval.sh

# iron tier with Qwen Flash (cheaper)
bash scripts/run_eval.sh iron qwen_flash

# diamond tier with Qwen Plus
bash scripts/run_eval.sh diamond qwen_plus

# any tier with another vendor
bash scripts/run_eval.sh iron glm_4_7        # ZHIPUAI_API_KEY required
bash scripts/run_eval.sh iron gemini_flash   # GOOGLE_API_KEY required
bash scripts/run_eval.sh iron gpt_5_5        # OPENAI_API_KEY required
```

Every task runs once per spawn in `seeds` (`conf/evaluate.yaml`; default = the first
three `oak_forest` entries of JARVIS-1's close-ended spawn table, seed + teleport
position; `python scripts/fetch_spawns.py oak_forest 10` lists more). A single quick
run is `seeds='[{seed: 19961103, pos: [-79, 64, -512]}]'` (or a plain integer seed
without teleport), and `seeds='[]'` restores
random worlds (`env.times` runs). Every evaluation is self-contained under
`logs/eval/<date>/<time>/`: `runs.jsonl` (task / seed / success / steps per run),
`plans/`, `evidence/` (per-subgoal trajectories + keyframes) and `llm_calls.jsonl` +
`llm_calls/` (this run's LLM calls with stage, tokens, full prompt + response; the
server appends the same to `logs/llm_calls.jsonl` across runs). Nothing needs to be
re-run to look at an old result. See
[docs/reproduction-notes.md](docs/reproduction-notes.md) before comparing numbers
with the paper.

Per-task and aggregate results are printed via `rich.Table`. Per-episode logs and (optionally) videos go to `logs/eval/<date>/<time>/` and `videos/<date>/`.

### Knowledge accumulation with checkpoints (paper Table 7 protocol)

The paper accumulates knowledge over M episodes of the fixed task-seed split and
reports checkpoints at 50 / 100 / 200 / 400. One command does that in a single run
dir, so an interrupted run never pays for an episode twice:

```bash
# server with KB writes ON and a fresh store (cold start):
MINEEVOLVE_MEMORY_PATH=memories/acc-$(date +%Y%m%d) bash scripts/server_gemini.sh &
# 50 episodes = the 33 wooden episodes (pass 1) + the first 17 again (pass 2)
xvfb-run -a python -m mineevolve.main benchmark=wooden llm=gemini_flash \
    accumulate.episodes=50 accumulate.kb_store_dir=memories/acc-$(date +%Y%m%d)

# it died at episode 37?  continue in place (same server, same store):
xvfb-run -a python -m mineevolve.main benchmark=wooden llm=gemini_flash \
    accumulate.episodes=50 accumulate.kb_store_dir=memories/acc-... \
    resume_dir=logs/eval/<date>/<time>

# frozen evaluation of a checkpoint: copy it into a store, start the server frozen
mkdir -p memories/M50 && cp logs/eval/<date>/<time>/kb_checkpoints/M50/*.json memories/M50/
MINEEVOLVE_KB_FROZEN=1 MINEEVOLVE_MEMORY_PATH=memories/M50 bash scripts/server_gemini.sh &
xvfb-run -a python -m mineevolve.main benchmark=wooden llm=gemini_flash
```

Layout of an accumulation run: `runs.jsonl` (with `episode` and `pass` fields),
`pass<k>/` (the usual `evidence/`, `plans/`, `runs.jsonl` of that pass —
`scripts/analyze_failures.py logs/eval/<date>/<time>/pass1` works unchanged),
`kb_checkpoints/M<n>/{skills,remedies}.json` every `accumulate.kb_checkpoint_every`
(50) episodes and at the end, `llm_calls.jsonl` + `llm_calls/` (appended on resume).
Everything needed for offline analysis (`scripts/analyze_failures.py`,
`scripts/llm_usage.py`, `scripts/spend.py`) is inside the run dir; no API call is
needed to look at a result again.

### Evaluate a custom task subset

`conf/benchmark/<group>.yaml::evaluate` selects task ids; leave it `[]` for all 70 tasks. To run iron tasks #2 and #5 only:

```bash
python -m mineevolve.main benchmark=iron evaluate='[2, 5]'
```

(Default `llm=qwen_plus` is taken from `conf/evaluate.yaml`; override with
`llm=qwen_flash` etc. if needed.)

---

## Method overview

MineEvolve converts each subgoal execution into typed feedback, induces skills (from successful segments) and remedies (from failed/stagnant segments), validates and retrieves them under a prompt budget, and repairs the unfinished plan suffix when failures repeat.

```
                  Algorithm 1 main loop
+-----------------------------------------------------+
| reset task -> initial plan                          |
| for each subgoal i:                                  |
|   (a) STEVE-1 executes z_i                           |
|   (b) Monitor builds typed feedback e_i (Eq. 1-3)    |
|   (c) Inducer -> skill | remedy candidates           |
|   (d) Curator validates (Eq. 6) and stores K         |
|   (e) if repeated failure or stagnation:             |
|         Adaptor freezes prefix, repairs suffix (Eq. 8)|
+-----------------------------------------------------+
```

See [docs/architecture.md](docs/architecture.md) for a per-module breakdown, and
[docs/fixes.md](docs/fixes.md) for what this vendored copy changes versus upstream and why.
[docs/reproduction-notes.md](docs/reproduction-notes.md) tracks where our setup matches or
deviates from the paper (seeds, knowledge-base state, token limits, model names) and the runs done so far.

---

## License

MIT.
