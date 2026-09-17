"""Minimal OpenAI-compatible chat server backed by a local HuggingFace model.

Lets you run the whole MineEvolve loop (planner / inducer / curator / adaptor)
with a small open-weights model instead of a paid API, so the code path can be
debugged for free. MineEvolve is not modified: it talks to this server through
the same ``openai_compat`` backend it uses for Qwen / GPT / Gemini / GLM, only
``base_url`` differs (see ``conf/llm/local.yaml``).

    CUDA_VISIBLE_DEVICES=3 python scripts/local_llm_server.py --model Qwen/Qwen3.5-2B
    curl -s localhost:8001/v1/chat/completions -H 'content-type: application/json' \\
         -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}'

Implements only what the ``openai`` SDK needs for non-streaming chat:
``POST /v1/chat/completions`` and ``GET /v1/models``. One request at a time.

Small models are much worse than the API models at following the JSON plan
format; that is the point - it exercises every parsing / repair / fallback
branch. Use ``--think`` to allow Qwen3's thinking mode (slower, sometimes
better), default is off. fp16 is used on GPUs without bf16 (e.g. Turing).

Memory: MineEvolve prompts grow with the feedback history (7.5k tokens after
four calls in a wooden task). On an 11 GB GPU Qwen3.5-2B/0.8B with eager
attention handle ~10k tokens; beyond that prefill OOMs. Prompts longer than
``--max-prompt-tokens`` get HTTP 413 (MineEvolve logs "LLM call failed" and
carries on) unless ``--truncate-prompt`` is given, which drops the middle of
the user message instead. A CUDA OOM is caught, the cache is freed and HTTP
503 is returned, so one bad request no longer poisons the server.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch import

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("local_llm_server")

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str = "local"
    messages: List[Message]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[Any] = None


class LocalLLM:
    def __init__(
        self,
        name: str,
        dtype: str,
        think: bool,
        max_new_tokens: int,
        attn: str = "eager",
        max_prompt_tokens: int = 9000,
        truncate_prompt: bool = False,
    ) -> None:
        self.name = name
        self.think = think
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.truncate_prompt = truncate_prompt
        self._lock = threading.Lock()

        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(name)
        # eager peaks lower than sdpa for Qwen3.5's hybrid layers on Turing (7.8 vs 10.4 GB at 7.5k tokens)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype, attn_implementation=attn).to(device).eval()
        self.device = device
        logger.info(
            "loaded %s (%s, %s) in %.0fs; GPU mem %.1f GB",
            name, type(self.model).__name__, dtype, time.time() - t0,
            torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0,
        )

    def _render(self, msgs: List[Dict[str, str]]) -> str:
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=self.think
            )
        except TypeError:  # template without a thinking switch
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _prompt(self, messages: List[Message]) -> str:
        msgs = [{"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)} for m in messages]
        text = self._render(msgs)
        n = len(self.tok(text)["input_ids"])
        if n <= self.max_prompt_tokens:
            return text
        if not self.truncate_prompt:
            raise HTTPException(
                status_code=413,
                detail=f"prompt is {n} tokens > --max-prompt-tokens {self.max_prompt_tokens} "
                       f"(would OOM); pass --truncate-prompt to drop the middle of the user message",
            )
        # Drop the middle of the longest message so head (instructions) and tail (current state) survive.
        idx = max(range(len(msgs)), key=lambda i: len(msgs[i]["content"]))
        body = msgs[idx]["content"]
        excess = n - self.max_prompt_tokens
        cut_chars = int(excess * 4.5) + 200  # ~4 chars/token, with slack
        head = (len(body) - cut_chars) // 2
        msgs[idx]["content"] = body[:head] + "\n[... truncated by local_llm_server ...]\n" + body[-head:]
        logger.warning("prompt %d tokens > %d: truncated middle of message %d", n, self.max_prompt_tokens, idx)
        return self._render(msgs)

    def chat(self, req: ChatRequest) -> Dict[str, Any]:
        max_new = int(req.max_tokens or req.max_completion_tokens or self.max_new_tokens)
        temperature = float(req.temperature if req.temperature is not None else 0.0)
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new}
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=float(req.top_p or 0.95))
        else:
            gen_kwargs.update(do_sample=False)

        prompt = self._prompt(req.messages)
        with self._lock:
            ids = self.tok(prompt, return_tensors="pt").to(self.device)
            n_prompt = int(ids["input_ids"].shape[1])
            t0 = time.time()
            try:
                with torch.no_grad():
                    out = self.model.generate(**ids, **gen_kwargs)
            except torch.OutOfMemoryError as exc:
                del ids
                torch.cuda.empty_cache()
                logger.error("CUDA OOM on prompt of %d tokens: %s", n_prompt, str(exc).split(".")[0])
                raise HTTPException(status_code=503, detail=f"CUDA OOM on a {n_prompt}-token prompt; lower --max-prompt-tokens")
            gen_ids = out[0][n_prompt:]
            dt = time.time() - t0

        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        if not self.think:
            text = THINK_RE.sub("", text)
        n_gen = int(gen_ids.shape[0])
        finish = "length" if n_gen >= max_new else "stop"
        logger.info("prompt=%d gen=%d tok in %.1fs (%.1f tok/s) finish=%s", n_prompt, n_gen, dt, n_gen / max(dt, 1e-6), finish)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text.strip()},
                "finish_reason": finish,
            }],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen},
        }


def create_app(llm: LocalLLM) -> FastAPI:
    app = FastAPI(title="local-llm (OpenAI-compatible)")

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": llm.name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict:
        return llm.chat(req)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": llm.name}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B", help="HF model id or local path")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--think", action="store_true", help="allow the model's thinking mode")
    ap.add_argument("--max-new-tokens", type=int, default=1536, help="default when the request has no max_tokens")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"], help="attention implementation")
    ap.add_argument("--max-prompt-tokens", type=int, default=9000, help="reject (413) longer prompts; ~10k fits an 11 GB GPU")
    ap.add_argument("--truncate-prompt", action="store_true", help="truncate over-long prompts instead of rejecting them")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    llm = LocalLLM(args.model, args.dtype, args.think, args.max_new_tokens,
                   attn=args.attn, max_prompt_tokens=args.max_prompt_tokens, truncate_prompt=args.truncate_prompt)
    uvicorn.run(create_app(llm), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
