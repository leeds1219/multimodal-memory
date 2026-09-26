# Baselines

Reference implementations of the Minecraft agents MineEvolve compares against,
cloned here so their interfaces (action primitives, observations, memory) can be
read next to our reproduction. Nothing here is a dependency of `MC-MineEvolve`.

| repo | what is released | state here |
|---|---|---|
| `NeurIPS24-Optimus-1` (iLearn-Lab, ex JiuTian-VL) | full code, MCP-Reborn build, STEVE-1 checkpoints, the Hybrid Multimodal Memory dataset | **installed and starting** — conda env `optimus1`, server on port 9020 |
| `Optimus-3` (JiuTian-VL) | full code + weights on HF, bundles MineStudio | cloned only; needs 28–32 GB VRAM, our GPUs have 11 GB |
| `CVPR25-Optimus-2` | README and assets only (authors say they are refactoring it into the Optimus-3 framework) | cloned for reference |
| `JARVIS-1` (CraftJarvis) | offline evaluation only; online (growing memory) "coming soon"; some GROOT controller weights unreleased | cloned; we already use its `spawn.json` for our fixed task–seed split |

## Optimus-1 setup (what it took)

The README's steps are not sufficient on a headless box without root:

1. `conda create -n optimus1 python=3.10`, then `pip install -r requirements.txt`
2. **Pin `transformers==4.44.2`** — the repo asks for `>=4.38.2`, pip installs 5.x,
   which refuses torch 2.0.1 and disables every model ("PyTorch was not found").
3. **Pin `numpy==1.24.4`** — numpy 2.x breaks torch 2.0.1 and the opencv wheel.
4. Use `opencv-python-headless==4.8.0.74`; the normal wheel needs `libGL.so.1`.
5. `conda install -c conda-forge openjdk=8` and export `JAVA_HOME=$CONDA_PREFIX`
   before `pip install -e minerl` (its build runs `./gradlew downloadAssets`).
6. Download from the README's Google Drive links: `MCP-Reborn.tar.gz` (1.3 GB,
   extract into `minerl/minerl/`, the jar is prebuilt) and
   `optimus1_steve1_ckpt.zip` (1.4 GB → `checkpoints/{vpt,steve1,mineclip}`).
7. Xvfb: reuse the `mineevolve` env's `Xvfb` and the `MC-MineEvolve/scripts/xvfb-run`
   shim (`LD_LIBRARY_PATH=$CONDA_PREFIX/lib:.../mineevolve/lib`).
8. The planner was `openai.OpenAI(api_key="<your api key>")` with `model="gpt-4o"`
   hard-coded. Patched to read `OPTIMUS1_LLM_{API_KEY,BASE_URL,MODEL}` so it can run
   against the same Gemini endpoint as the MineEvolve reproduction.

Run:

```bash
conda activate optimus1
CUDA_VISIBLE_DEVICES=5 uvicorn app:app --port 9020 &          # STEVE-1 + MineCLIP
export OPTIMUS1_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export OPTIMUS1_LLM_API_KEY=$GOOGLE_API_KEY OPTIMUS1_LLM_MODEL=gemini-3-flash-preview
xvfb-run -a python -m optimus1.test_optimus1 server.port=9020 evaluate="[0]"
```
