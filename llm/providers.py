"""LLM providers behind the shared client (llm/client.py). The config's ``provider:`` picks one.

Every caller speaks the OpenAI chat-completions format (messages with text and base64 ``image_url`` parts; the
answer read as ``resp.choices[0].message.content``). A provider sends such a request to its backend and returns a
response in that shape, plus the token counts the client bills. The client does everything else (ledger, caps,
retries, logs, cache), so a new backend only adds a class here and an entry in PROVIDERS.

    openai   any OpenAI-compatible endpoint: OpenAI, vLLM, SGLang, Ollama, LM Studio, OpenRouter (base_url)
    gemini   Google's OpenAI-compatible endpoint, with Gemini's thinking-token handling and its random 400s
    mock     canned answers for $0 tests (llm/mock.py); also forced by LLM_MOCK=1

A local model is served by vLLM or Ollama and used through ``openai`` with a localhost base_url and prices 0.
A backend without an OpenAI-compatible API (e.g. a model loaded in-process) gets its own class that converts the
messages in ``create`` and returns the same shape.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace


def response(content: str, finish_reason: str = "stop", n_in: int = 0, n_out: int = 0) -> SimpleNamespace:
    """An OpenAI-shaped response, for providers whose backend answers in another format"""
    usage = SimpleNamespace(prompt_tokens=n_in, completion_tokens=n_out, total_tokens=n_in + n_out,
                            completion_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                    finish_reason=finish_reason)], usage=usage)


class Provider:
    name = "base"
    retryable: tuple = ()  # exceptions retried with backoff

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def prepare(self, kw: dict) -> dict:
        """Backend-specific changes to the request (after the client set model and defaults)"""
        return kw

    def create(self, kw: dict):
        raise NotImplementedError

    def usage(self, resp) -> tuple[int, int, int]:
        """(input tokens, visible output tokens, billed output tokens)"""
        u = getattr(resp, "usage", None)
        n_in = (getattr(u, "prompt_tokens", 0) or 0) if u else 0
        n_out = (getattr(u, "completion_tokens", 0) or 0) if u else 0
        return n_in, n_out, n_out

    def flaky(self, err: Exception) -> bool:
        """An error the backend returns at random for valid requests: retried a few times and logged"""
        return False


class OpenAICompat(Provider):
    name = "openai"
    default_key_env = "OPENAI_API_KEY"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import openai
        self.retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                          openai.InternalServerError)
        self.raw = openai.OpenAI(api_key=self._api_key(), base_url=cfg.get("base_url"),
                                 timeout=cfg.get("request_timeout_s", 300), max_retries=0)

    def _api_key(self) -> str:
        """$<api_key_env>, else that name in api_key_file; a server without keys (local) gets a placeholder"""
        var = self.cfg.get("api_key_env", self.default_key_env)
        if not var:
            return "EMPTY"
        if os.environ.get(var):
            return os.environ[var]
        f = self.cfg.get("api_key_file")
        if f and Path(f).exists():
            m = re.search(rf"{var}\s*[:=]\s*[\"']?([^\"'\s]+)", Path(f).read_text())
            if m:
                return m.group(1)
        raise RuntimeError(f"{var} not set and not found in {f}")

    def create(self, kw: dict):
        return self.raw.chat.completions.create(**kw)

    def usage(self, resp):
        n_in, n_vis, _ = super().usage(resp)
        u = getattr(resp, "usage", None)
        det = getattr(u, "completion_tokens_details", None) if u else None
        n_think = (getattr(det, "reasoning_tokens", 0) or 0) if det is not None else 0
        return n_in, n_vis, n_vis + n_think


class Gemini(OpenAICompat):
    name = "gemini"
    default_key_env = "GOOGLE_API_KEY"

    def prepare(self, kw):
        # Gemini counts thinking tokens against max_tokens; the methods' max_tokens bound the visible answer.
        if kw.get("max_tokens") is not None and self.cfg.get("thinking_headroom_tokens"):
            kw["max_tokens"] = int(kw["max_tokens"]) + int(self.cfg["thinking_headroom_tokens"])
        return kw

    def usage(self, resp):
        n_in, n_vis, n_out = super().usage(resp)
        u = getattr(resp, "usage", None)
        n_tot = (getattr(u, "total_tokens", 0) or 0) if u else 0
        return n_in, n_vis, max(n_out, n_tot - n_in)  # the endpoint reports thinking tokens only in total_tokens

    def flaky(self, err):
        import openai
        return isinstance(err, openai.BadRequestError)  # 400 INVALID_ARGUMENT to a few valid requests at random


class Mock(Provider):
    name = "mock"

    def create(self, kw):
        from mock import mock_response  # llm/mock.py
        from client import context
        return mock_response(kw, context())

    def usage(self, resp):
        return 0, 0, 0


PROVIDERS = {p.name: p for p in (OpenAICompat, Gemini, Mock)}


def make(cfg: dict) -> Provider:
    name = "mock" if os.environ.get("LLM_MOCK") == "1" else cfg.get("provider", "gemini")
    if name not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}; known: {', '.join(PROVIDERS)}")
    return PROVIDERS[name](cfg)
