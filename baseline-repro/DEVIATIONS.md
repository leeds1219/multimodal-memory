# Deviations from the official code

Only three kinds of change are allowed: (a) LLM swap to gemini-3-flash-preview
through the shared layer, same prompts and message format; (b) version /
compatibility fixes needed to run at all; (c) interface glue for an evaluation
environment. Every change is listed here with file, change and reason.

## Shared LLM layer (all methods)
| Kind | Where | Change | Why |
|------|-------|--------|-----|
| a | `baseline-repro/llm/gemini_client.py` | Drop-in for `openai.OpenAI()`: forces the model from `configs/llm.yaml`, logs every call, bounded exponential backoff, spend guards | Rule: one planner backbone, everything logged |
| a | same | `max_tokens` sent = method value + 16384 thinking headroom | Gemini counts thinking tokens against max_tokens (DECISIONS D3) |

## MineEvolve (`MC-MineEvolve/`, upstream a0a5f9b)
| Kind | File | Change | Why |
|------|------|--------|-----|
| b | see `MC-MineEvolve/docs/fixes.md` §1–4 | Inherited from branch `fix/mineevolve-env-and-runtime`: chat no-op fix, MineRL chat-command patch, STEVE-1 runner for installed MineStudio, POV transport as base64 PNG | Upstream does not run on MineRL 1.0.2 / current MineStudio as released |
| a | `src/mineevolve/planner/backends/openai_compat.py` | `self._client = GeminiClient()` instead of `OpenAI(...)` | LLM swap. Upstream's own tenacity retry (3 attempts) is kept. |
| a | `src/mineevolve/conf/llm/gemini_shared.yaml` | New LLM config: provider gemini, temperature 0.2 / max_tokens 1536 copied from upstream `gemini_flash.yaml` | Model id comes from the shared config |

## Optimus-1 (`/home/rag/data/official/NeurIPS24-Optimus-1`, patches exported to `baseline-repro/baselines/optimus1/`)
| Kind | File | Change | Why |
|------|------|--------|-----|
| a | `src/optimus1/models/gpt4_planning.py` | `client = GeminiClient()`; `model="gpt-4o"` → `"FROM_CONFIG"` (4 calls). Prompts, images, max_tokens=2000 unchanged | LLM swap |
| b | environment | torch 2.0.1 → 2.9.1+cu128 | torch 2.0.1 has no sm_120 kernels (RTX PRO 6000 Blackwell) |
| b | environment | transformers pinned 4.44.2 | 5.x fails importing the bundled DeepSeek-VL module (DECISIONS D6) |
