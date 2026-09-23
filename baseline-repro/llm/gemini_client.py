"""Shared LLM layer for every baseline (drop-in for ``openai.OpenAI()``).

``GeminiClient().chat.completions.create(**kw)`` keeps the OpenAI signature and
message format (text + base64 ``image_url`` parts), and adds:

* model forced from ``configs/llm.yaml`` (baselines' hardcoded names are ignored);
* one JSONL record per call: run/episode context, messages (images replaced by
  a sha1 and saved once as files), response, finish reason, tokens incl.
  thinking tokens, latency, attempts, cost;
* exponential backoff retries up to ``max_retries``, then ``LLMFailed``;
* a global spend cap shared across processes (append-only ledger + flock);
* a per-episode runaway guard (cost / call count vs. smoke-test means) that
  raises ``EpisodeAnomaly`` and drops an ``ANOMALY`` marker in the episode dir;
* an optional response cache (off for evaluation) and a mock mode
  (``LLM_MOCK=1``) for offline checks.

Episode context comes from the JSON file named by ``$LLM_CTX_FILE`` (written
by the episode runner before each episode), because the baselines call the
LLM from a server process that the runner does not control directly.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import openai
import yaml

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(os.environ.get("LLM_CONFIG", REPO / "configs" / "llm.yaml")))
_LOCK = threading.Lock()
_EPISODE = {"key": None, "cost": 0.0, "calls": 0}


class BudgetExceeded(RuntimeError):
    pass


class EpisodeAnomaly(RuntimeError):
    pass


class LLMFailed(RuntimeError):
    pass


def _api_key() -> str:
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    txt = Path(CFG["api_key_file"]).read_text()
    m = re.search(r"GOOGLE_API_KEY\s*[:=]\s*[\"']?([^\"'\s]+)", txt)
    if not m:
        raise RuntimeError(f"GOOGLE_API_KEY not found in {CFG['api_key_file']}")
    return m.group(1)


def context() -> dict:
    """Current episode context (env, method, order, task, seed, episode_dir)."""
    f = os.environ.get("LLM_CTX_FILE")
    ctx = {}
    if f and os.path.exists(f):
        try:
            ctx = json.loads(Path(f).read_text())
        except json.JSONDecodeError:
            ctx = {}
    ctx.setdefault("method", os.environ.get("METHOD", "unknown"))
    ctx.setdefault("run_id", os.environ.get("RUN_ID", "adhoc"))
    return ctx


def _log_dir(ctx: dict) -> Path:
    if ctx.get("episode_dir"):
        return Path(ctx["episode_dir"]) / "llm"
    return Path(CFG["log_root"]) / "adhoc_llm" / ctx["method"] / ctx["run_id"]


# ----------------------------------------------------------------------------
# Global ledger (all processes)
# ----------------------------------------------------------------------------

def ledger_total() -> float:
    p = Path(CFG["ledger_path"])
    if not p.exists():
        return 0.0
    total = 0.0
    with open(p) as f:
        for line in f:
            try:
                total += json.loads(line)["cost"]
            except (ValueError, KeyError):
                pass
    return total


def _ledger_append(rec: dict) -> float:
    p = Path(CFG["ledger_path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return rec["cost"]


_TOTAL_CACHE = {"t": 0.0, "value": 0.0}


def _global_spent() -> float:
    # Re-read the ledger at most every 5 s; add our own spend in between.
    now = time.time()
    if now - _TOTAL_CACHE["t"] > 5:
        _TOTAL_CACHE.update(t=now, value=ledger_total())
    return _TOTAL_CACHE["value"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _strip_images(messages, img_dir: Path):
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    head, _, b64 = url.partition(",")
                    h = hashlib.sha1(b64.encode()).hexdigest()[:16]
                    ext = "png" if "png" in head else "jpg"
                    f = img_dir / f"{h}.{ext}"
                    if not f.exists():
                        img_dir.mkdir(parents=True, exist_ok=True)
                        f.write_bytes(base64.b64decode(b64))
                    parts.append({"type": "image", "sha1": h, "file": f.name})
                else:
                    parts.append(p)
            out.append({**msg, "content": parts})
        else:
            out.append(msg)
    return out


def _cache_key(kw: dict) -> str:
    return hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()


def _usage(resp):
    u = getattr(resp, "usage", None)
    n_in = (getattr(u, "prompt_tokens", 0) or 0) if u else 0
    n_vis = (getattr(u, "completion_tokens", 0) or 0) if u else 0
    n_tot = (getattr(u, "total_tokens", 0) or 0) if u else 0
    n_think = 0
    det = getattr(u, "completion_tokens_details", None) if u else None
    if det is not None:
        n_think = getattr(det, "reasoning_tokens", 0) or 0
    # Gemini's OpenAI endpoint reports thinking tokens only inside total_tokens.
    n_billed_out = max(n_vis + n_think, n_tot - n_in)
    return n_in, n_vis, n_billed_out


def _check_episode_guard(ctx: dict, method: str):
    key = ctx.get("episode_dir")
    if key != _EPISODE["key"]:
        _EPISODE.update(key=key, cost=0.0, calls=0)
    if key and (Path(key) / "ANOMALY").exists():
        raise EpisodeAnomaly(f"episode already marked anomalous: {key}")
    # Thresholds = multiplier x (smoke-test rate per env step) x this episode's
    # horizon, with small floors (DECISIONS D28); legacy per-episode means kept.
    rate = (CFG.get("episode_rates_per_step") or {}).get(method) or {}
    horizon = ctx.get("horizon_steps")
    mean_cost = (CFG.get("episode_mean_cost_usd") or {}).get(method)
    mean_calls = (CFG.get("episode_mean_calls") or {}).get(method)
    if rate and horizon:
        mean_cost = max(rate["cost_per_step"] * horizon, 0.05)
        mean_calls = max(rate["calls_per_step"] * horizon, 3)
    reason = None
    if mean_cost and _EPISODE["cost"] > CFG["episode_cost_multiplier"] * mean_cost:
        reason = f"episode cost ${_EPISODE['cost']:.3f} > {CFG['episode_cost_multiplier']}x expected ${mean_cost:.3f}"
    if mean_calls and _EPISODE["calls"] > CFG["episode_call_multiplier"] * mean_calls:
        reason = f"episode calls {_EPISODE['calls']} > {CFG['episode_call_multiplier']}x expected {mean_calls:.1f}"
    if reason:
        if key:
            Path(key).mkdir(parents=True, exist_ok=True)
            (Path(key) / "ANOMALY").write_text(reason + "\n")
        raise EpisodeAnomaly(reason)


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------

class _Completions:
    def __init__(self, raw):
        self._raw = raw

    def create(self, **kw):
        ctx = context()
        method = ctx["method"]
        requested_model = kw.get("model")
        kw["model"] = CFG["model"]
        if kw.get("temperature") is None and CFG.get("default_temperature") is not None:
            kw["temperature"] = CFG["default_temperature"]
        if kw.get("max_tokens") is None and CFG.get("default_max_tokens") is not None:
            kw["max_tokens"] = CFG["default_max_tokens"]
        kw = {k: v for k, v in kw.items() if v is not None}
        method_max_tokens = kw.get("max_tokens")
        if method_max_tokens is not None and CFG.get("thinking_headroom_tokens"):
            kw["max_tokens"] = int(method_max_tokens) + int(CFG["thinking_headroom_tokens"])

        with _LOCK:
            spent = _global_spent()
            if spent >= CFG["global_cap_usd"]:
                raise BudgetExceeded(f"global cap ${CFG['global_cap_usd']} reached (${spent:.2f})")
            _check_episode_guard(ctx, method)

        log_dir = _log_dir(ctx)
        log_dir.mkdir(parents=True, exist_ok=True)
        mock = os.environ.get("LLM_MOCK") == "1"
        use_cache = CFG.get("cache") and os.environ.get("LLM_CACHE", "1") == "1" and not mock
        ckey = _cache_key(kw) if use_cache else None
        cfile = Path(CFG["cache_dir"]) / f"{ckey}.json" if ckey else None

        attempts, err, cached = 0, None, False
        t0 = time.time()
        if mock:
            from mock import mock_response  # llm/mock.py
            resp = mock_response(kw, ctx)
            attempts = 1
        elif cfile is not None and cfile.exists():
            d = json.loads(cfile.read_text())
            resp = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=d["content"]), finish_reason=d["finish_reason"])],
                usage=None,
            )
            cached, attempts = True, 0
        else:
            for attempts in range(1, int(CFG["max_retries"]) + 2):
                t0 = time.time()
                try:
                    resp = self._raw.chat.completions.create(**kw)
                    err = None
                    break
                except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                        openai.InternalServerError) as e:
                    err = e
                    if attempts <= int(CFG["max_retries"]):
                        time.sleep(min(60, 2 ** (attempts - 1)))
            if err is not None:
                self._write(log_dir, ctx, kw, requested_model, None, None, 0, 0, 0, 0.0, attempts, time.time() - t0,
                            error=repr(err), method_max_tokens=method_max_tokens)
                raise LLMFailed(f"LLM failed after {attempts} attempts: {err!r}")
            if cfile is not None:
                cfile.parent.mkdir(parents=True, exist_ok=True)
                cfile.write_text(json.dumps({"content": resp.choices[0].message.content,
                                             "finish_reason": resp.choices[0].finish_reason}))
        dt = time.time() - t0

        n_in, n_vis, n_out = (0, 0, 0) if cached or mock else _usage(resp)
        cost = n_in / 1e6 * CFG["price_in_per_m"] + n_out / 1e6 * CFG["price_out_per_m"]
        with _LOCK:
            _EPISODE["cost"] += cost
            _EPISODE["calls"] += 1
            _TOTAL_CACHE["value"] += cost
            if cost:
                _ledger_append({"ts": time.time(), "cost": cost, "method": method, "env": ctx.get("env"),
                                "episode_dir": ctx.get("episode_dir")})
        content = resp.choices[0].message.content
        self._write(log_dir, ctx, kw, requested_model, content, resp.choices[0].finish_reason,
                    n_in, n_vis, n_out, cost, attempts, dt, cached=cached, mock=mock,
                    method_max_tokens=method_max_tokens)
        return resp

    @staticmethod
    def _write(log_dir, ctx, kw, requested_model, content, finish, n_in, n_vis, n_out, cost, attempts, dt, **extra):
        rec = {
            "id": uuid.uuid4().hex, "ts": time.time(),
            **{k: ctx.get(k) for k in ("run_id", "env", "method", "order_id", "task", "seed", "step")},
            "caller": _caller(),
            "model": kw["model"], "requested_model": requested_model,
            "method_max_tokens": extra.pop("method_max_tokens", None),
            "params": {k: v for k, v in kw.items() if k not in ("messages", "model")},
            "messages": _strip_images(kw["messages"], log_dir / "images"),
            "response": content, "finish_reason": finish,
            "tokens_in": n_in, "tokens_out_visible": n_vis, "tokens_out_billed": n_out,
            "cost_usd": round(cost, 6), "latency_s": round(dt, 3), "attempts": attempts,
            "episode_cost_usd": round(_EPISODE["cost"], 5), "episode_calls": _EPISODE["calls"],
            **extra,
        }
        with _LOCK, open(log_dir / "calls.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _caller() -> str:
    import traceback
    for fr in reversed(traceback.extract_stack()[:-3]):
        if "gemini_client" not in fr.filename:
            return f"{Path(fr.filename).name}:{fr.lineno}:{fr.name}"
    return "?"


class _Chat:
    def __init__(self, raw):
        self.completions = _Completions(raw)


class GeminiClient:
    """Accepts and ignores the baselines' own OpenAI() kwargs (key, base_url, retries)."""

    def __init__(self, **_ignored):
        raw = None
        if os.environ.get("LLM_MOCK") != "1":
            raw = openai.OpenAI(api_key=_api_key(), base_url=CFG["base_url"],
                                timeout=CFG["request_timeout_s"], max_retries=0)
        self.chat = _Chat(raw)


def episode_stats() -> dict:
    return dict(_EPISODE)
