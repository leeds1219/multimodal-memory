# Contributing

## Getting started

```bash
git clone https://github.com/leeds1219/multimodal-memory.git
cd multimodal-memory
```

Then build the virtual env for the project you're working on (see `README.md`).
Each sub-project has its own `.venv/`; they are git-ignored.

## Workflow

`main` is protected: nobody pushes to it directly. All changes go through a branch + Pull Request.

```bash
git checkout main && git pull                 # 1. start from the latest main
git checkout -b <type>/<short-description>    # 2. e.g. feature/curator-retrieval, fix/steve-loader
# ... make changes ...
git add -A && git commit -m "Short imperative summary"   # 3. small, focused commits
git push -u origin <branch>                   # 4. push your branch
gh pr create                                  # 5. open a PR (or use the GitHub UI)
```

After the PR is merged, update your local `main` (`git checkout main && git pull`) and delete the old branch.

Branch prefixes: `feature/`, `fix/`, `docs/`, `refactor/`, `experiment/`.

## Commit messages

- One line, imperative mood, under ~70 chars: `Fix progress score for stagnant subgoals`.
- Add a blank line and a short body if the *why* isn't obvious.

## Rules of thumb

- **Pull before you start** so you're not working on stale code.
- **Announce what you're working on** (issue, PR draft, or a message) to avoid two people editing the same module.
- **Never commit secrets.** API keys go in `.env` (git-ignored) or `export VAR=...` in your shell.
- **Never commit large binaries** — model weights, videos, logs. They are git-ignored; keep it that way.
- Keep PRs focused. One PR = one logical change; big refactors get their own PR.
- If a PR touches `MC-MineEvolve/` and `VoLoAgent/` at the same time, say so in the description.

## Resolving conflicts

If GitHub says the PR has conflicts:

```bash
git checkout <branch>
git pull origin main          # merge main into your branch
# fix files marked <<<<<<< / >>>>>>>, then:
git add -A && git commit
git push
```

## Code style

- Python ≥ 3.10, follow the style already in the file you're editing.
- `VoLoAgent/` uses `ruff` (`ruff check . && ruff format .` from its venv).
- Run `pytest` in `VoLoAgent/` before opening a PR if you touched it.
