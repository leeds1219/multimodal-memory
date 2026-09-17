# multimodal-memory — working rules

Monorepo holding two vendored sub-projects that are developed by several people.
These rules apply to every contributor, human or AI. `CONTRIBUTING.md` has the
longer explanation; this file is the short checklist.

## Repo layout

| Path | What it is | Python | Env name |
|------|------------|--------|----------|
| `MC-MineEvolve/` | LLM planner + STEVE-1 executor for Minecraft tech-tree tasks | 3.10 | `mineevolve` |
| `VoLoAgent/` | VLM orchestrator agent | 3.11 | `volo` |

Each sub-project has its own README, its own `requirements.txt` / `pyproject.toml`,
and its own setup script under `<sub-project>/scripts/`. Top-level files are for
things that concern the whole repo only.

## Git rules

1. **Never commit or push directly to `main`.** Always work on a branch and open a PR.
   Branch names: `feature/…`, `fix/…`, `docs/…`, `refactor/…`, `experiment/…`.
2. **Never force-push** to a shared branch, and never rewrite history on `main`.
3. `git pull` on `main` before branching so you start from the latest code.
4. Small, focused commits; one-line imperative subject (`Fix progress score for stagnant subgoals`).
5. Do not commit: secrets/API keys, model weights, videos, logs, `.venv/`, conda envs,
   `__pycache__/`, editor files. Check `.gitignore` before adding a new artefact type.
6. A PR that touches both `MC-MineEvolve/` and `VoLoAgent/` must say so in its description.

## Environment rules

1. **One isolated environment per sub-project.** Never install a sub-project's
   dependencies into the base/system Python or into the other sub-project's env.
   - conda (Linux/GPU boxes): `conda activate mineevolve` / `conda activate volo`
   - uv (laptops): `<sub-project>/.venv/`
2. Create/refresh an env only through the sub-project's setup script
   (`MC-MineEvolve/scripts/setup_env.sh`, `VoLoAgent` → `pip install -e ".[dev]"`),
   so everyone ends up with the same thing. If you need a new dependency, add it to
   that sub-project's `requirements.txt` / `pyproject.toml` **and** the setup script.
3. Prefer no-root installs (conda-forge, pip) so setup works inside containers
   without `sudo`. If something truly needs a system package, document it in the
   sub-project README under "System prerequisites".
4. Keep caches (pip, HuggingFace, torch, conda pkgs) out of `$HOME` on shared
   machines — point them at the big data volume (see the machine's own README).
5. Shared GPU box: check `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` before launching.

## Code rules

1. Follow the style of the file you are editing; don't reformat unrelated code.
2. Don't vendor code from other repos into `src/`; add a pip dependency instead.
3. Config lives in the sub-project's `conf/` (Hydra YAML); don't hard-code paths,
   ports, or model names in Python.
4. Anything that must be run to reproduce a result (env setup, download, eval)
   goes in a script under `scripts/` **and** gets a line in the README. If you had
   to run a manual command to make something work, turn it into a script.
5. `VoLoAgent/`: run `ruff check . && ruff format .` and `pytest` before a PR.

## For AI assistants (Claude Code etc.)

- Read the sub-project README and `scripts/` before proposing a setup.
- Activate the right conda env / venv before running anything for that sub-project.
- Never run `git push` to `main`, `git push --force`, or `git reset --hard` on shared
  branches; never commit without being asked.
- Do not run large jobs on GPUs 0–1 without checking they are free.
