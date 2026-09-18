# multimodal-memory

Workspace grouping two codebases:

| Folder | Upstream | Vendored commit |
|--------|----------|-----------------|
| `MC-MineEvolve/` | https://github.com/xzw-ustc/MC-MineEvolve | `a0a5f9b36626fd544dfd3c05e5ec27a1c1b48081` |
| `VoLoAgent/` | https://github.com/NVlabs/VoLoAgent (Apache-2.0) | `b4e623079ca8498a16bcd5016920d71f76c44d30` |

Each folder is a plain copy (upstream `.git` removed); see each folder's own README for setup.

## API keys

```bash
cp .env.example .env      # .env is git-ignored; paste your own keys into it
```

`MC-MineEvolve/scripts/server*.sh` load `.env` on start (`GOOGLE_API_KEY` for Gemini,
`DASHSCOPE_API_KEY` for Qwen, …). Never commit `.env` or put a key in a config file.

## Local environments (macOS, via [uv](https://docs.astral.sh/uv/))

```bash
# MC-MineEvolve (Python 3.10; MineStudio/MineRL skipped — Linux+NVIDIA only)
cd MC-MineEvolve && uv venv .venv --python 3.10 \
  && uv pip install -r <(grep -vi '^MineStudio' requirements.txt) && uv pip install --no-deps -e .

# VoLoAgent (Python 3.11)
cd VoLoAgent && uv venv .venv --python 3.11 && uv pip install -e ".[dev]"
```

## Linux + GPU servers (conda)

One conda env per sub-project; never install into `base`. Env names are fixed so
everyone's shell looks the same:

| Sub-project | Env | Setup |
|-------------|-----|-------|
| `MC-MineEvolve/` | `mineevolve` (py3.10) | `bash MC-MineEvolve/scripts/setup_env.sh` — installs Java 8, Xvfb, MineRL 1.0.2, MineStudio, torch; no root needed |
| `VoLoAgent/` | `volo` (py3.11) | `conda create -n volo python=3.11 && conda activate volo && pip install -e "VoLoAgent[dev]"` |

Smoke-test MC-MineEvolve without any LLM key: `conda activate mineevolve && cd MC-MineEvolve && xvfb-run -a python scripts/smoke_test.py`.

On shared boxes, point pip / HF / torch / conda caches at the big data volume instead
of `$HOME` (e.g. `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `CONDA_ENVS_DIRS`, `CONDA_PKGS_DIRS`).

See `CLAUDE.md` for the repo rules (branching, envs, what not to commit) and
`CONTRIBUTING.md` for the PR workflow.
