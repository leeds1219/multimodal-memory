"""One LLM call that must answer a JSON object (System 2 roles)."""
from __future__ import annotations

import json
import re


def ask_json(system: str, user: str, log: list, kind: str, max_tokens: int = 4000) -> dict | None:
    """The JSON object in the answer, or None; every call is appended to `log`."""
    from gemini_client import GeminiClient
    try:
        resp = GeminiClient().chat.completions.create(
            model="FROM_CONFIG", max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        txt = resp.choices[0].message.content or ""
    except Exception as e:
        log.append({"ev": f"{kind}_llm_error", "error": repr(e)[:300]})
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        out = json.loads(m.group(0)) if m else None
    except Exception:
        out = None
    log.append({"ev": kind, "answer": out, "text": None if isinstance(out, dict) else txt[:1000]})
    return out if isinstance(out, dict) else None
