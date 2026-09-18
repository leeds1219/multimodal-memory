"""Generic OpenAI-compatible chat backend.

Used directly for the OpenAI GPT family and as the base class for Qwen
(via DashScope's OpenAI-compatible endpoint), GLM (via Zhipu BigModel's
OpenAI-compatible endpoint), and Gemini (via Google AI Studio's
OpenAI-compatible endpoint).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..base import PlannerBackend


logger = logging.getLogger("mineevolve.planner.openai_compat")

# Per-call usage log (one JSON line per LLM request) so a run can be broken
# down by stage / tokens afterwards: scripts/llm_usage.py reads it.
LLM_LOG_PATH = os.environ.get("MINEEVOLVE_LLM_LOG", "logs/llm_calls.jsonl")
# Full prompt + response of each call, one file per call, next to the jsonl
# (logs/llm_calls/0001_repair.json). Needed to judge what the LLM could and
# could not know from text alone (e.g. vs. the POV keyframes in evidence/).
LLM_DUMP_DIR = os.environ.get("MINEEVOLVE_LLM_DUMP_DIR", os.path.splitext(LLM_LOG_PATH)[0])
_call_counter = 0


def _stage_for(system: str) -> str:
    """Name the pipeline stage from the system prompt it uses."""
    try:
        from ...adaptor.prompts import ADAPTOR_SYSTEM_PROMPT
        from ...inducer.prompts import REMEDY_SYSTEM_PROMPT, SKILL_SYSTEM_PROMPT
        from ...planner.prompts import PLANNER_SYSTEM_PROMPT
    except Exception:  # pragma: no cover
        return "unknown"
    for name, prompt in (
        ("plan", PLANNER_SYSTEM_PROMPT),
        ("induce_skill", SKILL_SYSTEM_PROMPT),
        ("induce_remedy", REMEDY_SYSTEM_PROMPT),
        ("repair", ADAPTOR_SYSTEM_PROMPT),
    ):
        if system == prompt:
            return name
    return "other"


def _log_call(record: dict, system: str = "", user: str = "", response: str = "") -> None:
    global _call_counter
    _call_counter += 1
    record["n"] = _call_counter
    logger.info(
        "llm call stage=%s model=%s prompt=%s gen=%s finish=%s %.1fs",
        record["stage"], record["model"], record["prompt_tokens"], record["completion_tokens"],
        record["finish"], record["s"],
    )
    try:
        os.makedirs(os.path.dirname(LLM_LOG_PATH) or ".", exist_ok=True)
        with open(LLM_LOG_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if LLM_DUMP_DIR:
            os.makedirs(LLM_DUMP_DIR, exist_ok=True)
            dump = os.path.join(LLM_DUMP_DIR, f"{_call_counter:04d}_{record['stage']}.json")
            with open(dump, "w") as fh:
                json.dump({**record, "system": system, "user": user, "response": response}, fh, indent=1)
            record["dump"] = dump
    except OSError as exc:  # pragma: no cover
        logger.warning("could not write %s: %s", LLM_LOG_PATH, exc)


class OpenAICompatibleBackend(PlannerBackend):
    """Thin wrapper around the official ``openai`` python SDK.

    The vendor is selected solely by ``base_url`` and ``api_key``.
    """

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        default_temperature: float = 0.2,
        default_max_tokens: int = 1024,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The `openai` package is required. Install it with `pip install openai`."
            ) from exc

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            logger.warning(
                "API key not set (env=%s). Backend %s will fail at call time.",
                api_key_env,
                self.name,
            )
        self._client = OpenAI(
            api_key=resolved_key or "missing",
            base_url=base_url,
            timeout=timeout,
        )
        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
        retry=retry_if_exception_type(Exception),
    )
    def _do_chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        t0 = time.monotonic()
        extra = {}
        # Thinking models (Gemini 3.x, o-series): cap hidden reasoning so the
        # visible JSON is not truncated by max_tokens. "low" | "medium" | "high".
        effort = os.environ.get("MINEEVOLVE_LLM_REASONING_EFFORT")
        if effort:
            extra["reasoning_effort"] = effort
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        choice = response.choices[0]
        content = choice.message.content if choice and choice.message else ""
        usage = getattr(response, "usage", None)
        _log_call({
            "t": time.time(),
            "stage": _stage_for(system),
            "backend": self.name,
            "model": self._model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "finish": getattr(choice, "finish_reason", None),
            "reasoning_effort": effort,
            "s": round(time.monotonic() - t0, 2),
            "content_chars": len(content or ""),
        }, system=system, user=user, response=content or "")
        return content or ""

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 0,
        temperature: float = -1.0,
    ) -> str:
        if max_tokens <= 0:
            max_tokens = self._default_max_tokens
        if temperature < 0:
            temperature = self._default_temperature
        try:
            return self._do_chat(system, user, max_tokens, temperature)
        except Exception as exc:
            logger.error("LLM call failed (%s/%s): %s", self.name, self._model, exc)
            return ""
