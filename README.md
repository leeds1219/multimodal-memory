# multimodal-memory

Workspace grouping two codebases:

| Folder | Upstream | Vendored commit |
|--------|----------|-----------------|
| `MC-MineEvolve/` | https://github.com/xzw-ustc/MC-MineEvolve | `a0a5f9b36626fd544dfd3c05e5ec27a1c1b48081` |
| `VoLoAgent/` | https://github.com/NVlabs/VoLoAgent (Apache-2.0) | `b4e623079ca8498a16bcd5016920d71f76c44d30` |

Each folder is a plain copy (upstream `.git` removed); see each folder's own README for setup.

## Local environments (macOS, via [uv](https://docs.astral.sh/uv/))

```bash
# MC-MineEvolve (Python 3.10; MineStudio/MineRL skipped — Linux+NVIDIA only)
cd MC-MineEvolve && uv venv .venv --python 3.10 \
  && uv pip install -r <(grep -vi '^MineStudio' requirements.txt) && uv pip install --no-deps -e .

# VoLoAgent (Python 3.11)
cd VoLoAgent && uv venv .venv --python 3.11 && uv pip install -e ".[dev]"
```
