# multimodal-memory — working rules

Reproduction of Optimus-1 (Gemini planner) in `baseline-repro/`; see `README.md`.
These rules apply to every contributor, human or AI. `CONTRIBUTING.md` has the
longer explanation; this file is the short checklist.

## Git rules

1. **Never push to `main`.** Work on a branch, push the branch freely, and merge via PR.
   Branch names: `feature/…`, `fix/…`, `docs/…`, `refactor/…`, `experiment/…`.
2. **Never force-push** to a shared branch, and never rewrite history on `main`.
3. `git pull` on `main` before branching so you start from the latest code.
4. Small, focused commits; one-line imperative subject (`Fix progress score for stagnant subgoals`).
5. Do not commit: secrets/API keys, model weights, videos, logs, run outputs, large analysis
   outputs (`analysis/out/`, `analysis/explorer/`), `.venv/`, conda envs, `__pycache__/`,
   editor files. Run `git status` before every commit; if a new artefact type shows up,
   add it to `.gitignore` first. Never commit a file over 5 MB.

## Environment rules

1. One env, `mcagent`, built only by `baseline-repro/scripts/setup_env.sh`. Its Python
   packages are pinned to `baseline-repro/envs/mcagent.lock.txt` (the original env's
   freeze). A new dependency goes into the lock file **and** is checked by rebuilding.
2. The env lives on the data volume (`/home/rag/data/conda_envs/`); `/opt/conda/envs/mcagent`
   is only a symlink, so a container reset loses nothing but the symlink and apt packages:
   rerun the setup script.
3. **Never run a third-party `setup.py` / install script before reading all of it.**
   `minerl/setup.py` deletes and rebuilds `MCP-Reborn` unless `READTHEDOCS=1` (D39).
4. Keep caches (pip, HuggingFace, torch, conda pkgs) out of `$HOME`; point them at
   `/home/rag/data/.cache`.
5. Shared GPU box: check `nvidia-smi` and pass a free `--gpu`; GPUs 0, 1 and 6 are used
   by other people.

## Code rules

1. Follow the style of the file you are editing; don't reformat unrelated code.
2. Don't vendor code from other repos into `src/`; add a pip dependency instead.
3. Config lives in `baseline-repro/configs/` (YAML); don't hard-code ports or model names
   in Python.
4. Anything that must be run to reproduce a result (env setup, download, eval)
   goes in a script under `scripts/` **and** gets a line in the README. If you had
   to run a manual command to make something work, turn it into a script.
5. Changes to the released method go in as a labelled variant (env var + variant token +
   a DECISIONS entry), never as a silent edit, so released and fixed runs stay comparable.

## For AI assistants (Claude Code etc.)

- Read `README.md`, `baseline-repro/RESUME.md` and `scripts/` before proposing a setup.
- Use `/opt/conda/envs/mcagent/bin/python` for everything in `baseline-repro/`.
- Never push to `main`, never `git push --force`, never `git reset --hard` on shared
  branches. Committing and pushing to a feature branch is fine when asked.
- Do not run jobs on GPUs 0, 1 or 6 without checking they are free.
- Record every launch in `/home/rag/data/repro_runs/suite_optimus1/LAUNCHES.md`.
