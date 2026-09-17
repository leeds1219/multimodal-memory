# Fixes applied on top of upstream MC-MineEvolve

This folder is a vendored copy of
[xzw-ustc/MC-MineEvolve](https://github.com/xzw-ustc/MC-MineEvolve) at commit
`a0a5f9b` (see the top-level README). As vendored, the code did **not** run on a
headless Linux GPU box with the dependency versions available today. Every
deviation from upstream is listed here with the symptom, the root cause, the fix
and how to check it. Keep this file updated when you change behaviour that
upstream does not have.

Verification for all of it: `xvfb-run -a python scripts/smoke_test.py --server http://127.0.0.1:9000`
(server started with `scripts/server.sh`). No LLM key needed.

---

## 1. Reset-time chat commands were silently dropped (`env/chat_action.py`)

**Symptom.** Every entry of `conf/evaluate.yaml::commands` (`/gamerule keepInventory`,
night vision, day lock, `/spawnpoint`, …) logged
`execute_cmd failed: ... Text.no_op() got an unexpected keyword argument 'batch_shape'`
and was skipped. Same for the `/setblock` ore spawning in `env/wrapper.py`.

**Cause.** `env.action_space.noop()` (MineRL `Dict.no_op(batch_shape=())`) forwards
`batch_shape` to every sub-space; MineRL 1.0's `spaces.Text.no_op()` does not accept
it (nor does `Text.sample(bs)`), so building the no-op action raised, and the
wrapper's `except` turned it into a warning.

**Fix.** `ChatAction` now uses a `_ChatText(spaces.Text)` subclass whose `no_op()` /
`sample()` accept and ignore the batch arguments.

**Check.** Minecraft log (`logs/mc_<port>.log`) shows
`Gamerule keepInventory is now set to: true`, `Applied effect Night Vision`, etc.
right after `MineRLAgent0 joined the game`.

## 2. MineRL itself cannot execute chat commands (`patches/`, `scripts/patch_minerl.sh`)

**Symptom.** After fix 1, the first chat command crashed the Java side with
`NumberFormatException: For input string: "/gamerule"` and MineRL terminated the
episode (`Attempted to step an environment server with done=True`).

**Cause.** MineRL 1.0.x's `EnvServer.java` (MCP-Reborn) parses every action line as
`<key> <0|1>` or `camera <dx> <dy>`; there is no `chat` verb at all. Additionally the
world is created with `allowCommands=false`, so even a delivered `/gamerule` would be
refused. Other agent repos built on MineRL 1.0 (Optimus-1, MCU, …) ship a private
MineRL fork for this; upstream MineEvolve's README does not mention it.

**Fix.** `patches/minerl-1.0.2-chat-commands.patch` (applied to the installed MineRL by
`scripts/patch_minerl.sh`, which then rebuilds `mcprec-6.13.jar` with Gradle):
- `chat <text>` action lines are sent via `player.sendChatMessage` on the client
  thread and skipped by the keyboard/mouse parsers;
- the world is created with `allowCommands=true`.

**Caveat.** The patch lives in the conda env's `site-packages`, not in git. A
`pip install --force-reinstall minerl` drops it; rerun `bash scripts/patch_minerl.sh`
(`setup_env.sh` does this automatically). Marker file:
`<site-packages>/minerl/MCP-Reborn/build/libs/.patched`.

## 3. STEVE-1 runner did not match the installed MineStudio (`executor/steve_runner.py`, `executor/steve_loader.py`)

**Symptom.** `/chat type=action` returned `{"action": null, "error": "'condition'"}`
after ~7 s.

**Cause.** Three things:
- `SteveRunner` looked for `policy.get_steve_action(...)` (an older MineStudio API)
  and fell back to `get_action(obs)`; MineStudio 1.1.x's `get_action` expects
  `input={'image': ..., 'condition': ...}`.
- The loader never moved the policy to the GPU, so inference ran on CPU (7 s/step).
- The server calls `set_condition()` on **every** step, and the runner re-embedded the
  prompt and re-created the recurrent state each time, so STEVE-1 had no temporal
  context.

**Fix.**
- `_MineStudioAdapter` resizes the POV to the checkpoint's `img_shape` (128×128),
  calls `get_action` with a pre-batched `[1,1,H,W,3]` frame (`input_shape="BT*"`, so
  the prepared condition is not batchified again), and maps the VPT factored action
  back to MineRL keys with the same `CameraHierarchicalMapping` + `ActionTransformer`
  MineStudio's own simulator uses.
- `set_condition()` only re-embeds and resets state when the text/scale changes.
- `_try_minestudio` does `policy.to(device).eval()`.

**Check.** ~16 ms/step on an RTX 2080 Ti; `nvidia-smi` shows the server process on
the chosen GPU; the agent moves and breaks blocks in the smoke test.

## 4. Per-step latency dominated by the POV transport (`main.py`, `util/image.py`)

**Symptom.** ~1 env step per second end-to-end even with STEVE-1 at 16 ms.

**Cause.** `_safe_pov` sent `pov.tolist()` — a 640×360×3 nested list, ~6.6 MB of JSON
per request, twice (`pov` and `image`). Serialise + transfer + parse ≈ 1 s. With
`runtime.subgoal_timeout_s: 60` this capped each subgoal at ~60 env steps instead of
the intended `max_steps_per_subgoal: 1200`.

**Fix.** `_safe_pov` now returns a base64 PNG (`util/image.py::encode_pov_to_base64`,
which upstream defined but never used); the server decodes it with the new
`decode_pov`, which still accepts the old nested-list format. ~275 KB per step,
lossless.

**Check.** Smoke test reports ~7 steps/s (was 0.9).

---

## Environment-level workarounds (not code changes)

All in `scripts/setup_env.sh`; see the README "Quick path" for the list:
MineRL 1.0.x from GitHub (PyPI stops at 0.4.x), `gym==0.23.1` needs `setuptools<66`
to build, MineStudio pins `opencv-python==4.8.0.74` whose wheel needs `libGL`/`glib`
(conda-forge), headless GL needs Mesa on `LD_LIBRARY_PATH` (done by `scripts/xvfb-run`;
without it LWJGL hangs silently at `Backend library: LWJGL`).

## 5. Runtime artifact directories were not git-ignored (`.gitignore`)

`main.py` and the server write `plans/`, `evidence/` (PNG frames + `steps.jsonl`
per subgoal), `memories/` and `resolved_config.yaml` into the repo directory. Only
`memories/run_*/` was ignored, so the first real run would have staged megabytes of
frames. All four are now ignored.

## Debugging without an API (branch `debug/local-llm`)

`scripts/local_llm_server.py` serves a small HF model (default `Qwen/Qwen3.5-2B`,
fp16, ~4 GB) behind an OpenAI-compatible endpoint; `conf/llm/local.yaml` and
`scripts/server_local.sh` point MineEvolve at it. See the README section
"Debugging without an API key". Verified: one wooden task runs through the whole
Algorithm 1 loop (initial plan → STEVE-1 execution → Monitor feedback → Inducer →
Adaptor repair → abort after 3 consecutive failures) with zero API calls.

**Observation (upstream behaviour, not changed):** the inducer/repair prompts grow
with the feedback history — 2.3k tokens for the initial plan, 7.5k by the 4th LLM
call, ~14k by the 8th, within a single 2-minute wooden task. `runtime.budget_tokens`
(paper Eq. 7) only bounds the *retrieved knowledge*, not the feedback buffer. This
is a cost problem with paid APIs (input tokens dominate) and a memory problem for
local models (~10k tokens is the ceiling on an 11 GB GPU, hence `--truncate-prompt`).
Worth capping in `server/agent.py` before running the full 70-task benchmark.

## Not yet verified

- The LLM path with a real API model (only exercised with a local 2B model; see above).
- The original `steve1` package fallback in `steve_loader.py`.
- Full benchmark runs (`scripts/run_eval.sh`); only a 600-step "chop a tree" episode was
  exercised.
