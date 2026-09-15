# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal OpenAI-compatible HTTP server for Molmo2-8B via HF transformers.

We need this because vLLM 0.9.1 in the `tool_vlm` env can't load
Molmo2-8B (Molmo2 modeling needs transformers>=4.55; vLLM 0.9.1
crashes on aimv2 duplicate registration with that transformers
version).  Running Molmo2 directly via HF transformers in the
`molmo-env` env works, but the rest of the orchestrator expects
an OpenAI-compatible `/v1/chat/completions` endpoint — this server
provides exactly that, just enough fields for our `vlm_orchestrator/
vision/molmo.py` client to consume.

Run inside `molmo-env`:

    conda run -n molmo-env \\
        python vlm_orchestrator/utils/molmo2_hf_server.py --port 8122

Endpoints:
    POST /v1/chat/completions   OpenAI-compatible (image + text)
    GET  /health                liveness probe
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import time
import uuid
from typing import Any

import torch
import PIL.Image
from fastapi import FastAPI, HTTPException, Request
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

logger = logging.getLogger("molmo2_hf_server")


# ─── Model setup ─────────────────────────────────────────────────────
_model = None
_processor = None
_model_id = "allenai/Molmo2-8B"


def _load_model(model_id: str, quantize: str = "bf16"):
    """Load Molmo2 with optional 4-bit / 8-bit quantization via bitsandbytes.

    ``quantize`` options:
      - ``bf16`` (default): full precision bf16, ~17 GB VRAM
      - ``int8``: bnb 8-bit, ~10 GB
      - ``int4``: bnb 4-bit (NF4), ~5 GB — preferred when sharing GPU
        with Isaac Sim + grasp server on a single 48 GB card.
    """
    global _model, _processor, _model_id
    _model_id = model_id
    logger.info(f"[molmo2_hf] loading {model_id} on CUDA (quantize={quantize}) ...")
    t0 = time.time()

    if quantize == "bf16":
        _processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda",
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda",
        )
    elif quantize in ("int8", "int4"):
        from transformers import BitsAndBytesConfig
        if quantize == "int4":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        # Processor doesn't quantize; load it normally.
        _processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )
        _model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_cfg,
            device_map="cuda",
        )
    else:
        raise ValueError(f"unknown quantize value {quantize!r} (use bf16/int8/int4)")

    logger.info(f"[molmo2_hf] loaded in {time.time() - t0:.1f}s")


# ─── Request parsing helpers ────────────────────────────────────────
_DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,(.+)$", re.IGNORECASE)


def _decode_image_data_url(url: str) -> PIL.Image.Image:
    m = _DATA_URL_RE.match(url)
    if not m:
        raise ValueError(f"image_url not a data URL: {url[:60]!r}")
    img_bytes = base64.b64decode(m.group(1))
    return PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _flatten_user_content(content: Any) -> tuple[str, PIL.Image.Image | None]:
    """Convert OpenAI-style content (str or list of content-parts) to
    (text, image).  Only the first image is used (Molmo2 takes one
    image per turn).  Multiple text parts are concatenated."""
    if isinstance(content, str):
        return content, None
    texts: list[str] = []
    image: PIL.Image.Image | None = None
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url" and image is None:
            url = part.get("image_url", {}).get("url", "")
            image = _decode_image_data_url(url)
    return "\n".join(t for t in texts if t), image


# ─── Generation ──────────────────────────────────────────────────────
def _generate(text_prompt: str, image: PIL.Image.Image, *,
              max_new_tokens: int, temperature: float) -> str:
    assert _model is not None and _processor is not None
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text_prompt},
        ],
    }]
    inputs = _processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(_model.device, dtype=torch.bfloat16)
    do_sample = temperature > 0
    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
        )
    gen = out[0, inputs["input_ids"].size(1):]
    return _processor.tokenizer.decode(gen, skip_special_tokens=True)


# ─── FastAPI ─────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True, "model": _model_id, "loaded": _model is not None}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")
    body = await request.json()

    messages = body.get("messages") or []
    # We only honour the first non-system user turn (Molmo2 is a
    # single-turn pointing model in our use).
    text_prompt = ""
    image: PIL.Image.Image | None = None
    for m in messages:
        if m.get("role") != "user":
            continue
        text, img = _flatten_user_content(m.get("content"))
        if text:
            text_prompt = text
        if img is not None:
            image = img
        if image is not None and text_prompt:
            break

    if image is None:
        raise HTTPException(400, "Request must include at least one image part")
    if not text_prompt:
        raise HTTPException(400, "Request must include at least one text part")

    max_tokens = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.0))
    t0 = time.time()
    completion = _generate(
        text_prompt, image,
        max_new_tokens=max_tokens, temperature=temperature,
    )
    dt = time.time() - t0
    logger.info(
        f"[molmo2_hf] {dt:.2f}s prompt={text_prompt[:80]!r} "
        f"-> {completion[:100]!r}"
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8122)
    p.add_argument("--model", default="allenai/Molmo2-8B")
    p.add_argument("--quantize", default="bf16",
                   choices=["bf16", "int8", "int4"],
                   help="bnb quantization: bf16 (default ~17 GB), "
                        "int8 (~10 GB), int4 (~5 GB).  Use int4 when "
                        "sharing GPU with Isaac Sim + grasp server.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _load_model(args.model, quantize=args.quantize)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
