"""LLM spending guard: pause for human review when calls look abnormal.

Every paid LLM request in the server goes through ``GUARD.before_call`` /
``GUARD.after_call``. Four rolling limits are checked (env-configurable):

    MINEEVOLVE_GUARD_MAX_CALLS_PER_EPISODE   default 80   (healthy episodes: <= 48)
    MINEEVOLVE_GUARD_MAX_USD_PER_EPISODE     default 0.60 (healthy: <= 0.30)
    MINEEVOLVE_GUARD_MAX_USD_PER_HOUR        default 3.00 (healthy: ~1.1/h; runaway: 6.4/h)
    MINEEVOLVE_GUARD_MAX_USD_TOTAL           default 20   (per server session)
    MINEEVOLVE_GUARD_DIR                     default logs/llm_guard

When a limit trips the guard does NOT kill anything: it writes
``<dir>/PAUSED`` (reason + counters), raises ``GuardPaused`` so the request
returns HTTP 503, and the env-side client waits. Nothing else is spent until a
human looks and runs

    python scripts/llm_guard.py status | resume | abort

``resume`` removes PAUSED and drops a ``RESUME`` marker: the episode / hourly
counters restart and the session total gets a fresh allowance. ``abort`` drops
``ABORT``: the next call raises ``GuardAborted`` and the evaluation exits.
A PAUSED file left over from a previous session keeps a freshly started server
paused, on purpose.

Cost is estimated at MINEEVOLVE_LLM_PRICE_IN / _OUT USD per 1M tokens
(default Gemini 3 Flash list price 0.50 / 3.00).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("mineevolve.llm_guard")


class GuardPaused(RuntimeError):
    """Spending guard tripped or a PAUSED file exists: no LLM call until resumed."""


class GuardAborted(RuntimeError):
    """A human dropped an ABORT marker: stop the evaluation."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class LLMGuard:
    def __init__(self, guard_dir: str | None = None) -> None:
        self.dir = Path(guard_dir or os.environ.get("MINEEVOLVE_GUARD_DIR", "logs/llm_guard"))
        self.max_calls_episode = int(_env_float("MINEEVOLVE_GUARD_MAX_CALLS_PER_EPISODE", 80))
        self.max_usd_episode = _env_float("MINEEVOLVE_GUARD_MAX_USD_PER_EPISODE", 0.60)
        self.max_usd_hour = _env_float("MINEEVOLVE_GUARD_MAX_USD_PER_HOUR", 3.00)
        self.max_usd_total = _env_float("MINEEVOLVE_GUARD_MAX_USD_TOTAL", 20.0)
        self.price_in = _env_float("MINEEVOLVE_LLM_PRICE_IN", 0.50)
        self.price_out = _env_float("MINEEVOLVE_LLM_PRICE_OUT", 3.00)
        self._lock = threading.RLock()  # status() is called while paused inside before_call
        self._episode_calls = 0
        self._episode_usd = 0.0
        self._episode_goal = ""
        self._episode_no = 0
        self._total_calls = 0
        self._total_usd = 0.0
        self._hour: deque[tuple[float, float]] = deque()  # (t, usd)
        self._session_start = time.time()

    # -- files -----------------------------------------------------------
    @property
    def paused_file(self) -> Path:
        return self.dir / "PAUSED"

    @property
    def resume_file(self) -> Path:
        return self.dir / "RESUME"

    @property
    def abort_file(self) -> Path:
        return self.dir / "ABORT"

    def _hour_usd(self) -> float:
        cutoff = time.time() - 3600
        while self._hour and self._hour[0][0] < cutoff:
            self._hour.popleft()
        return sum(u for _, u in self._hour)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "paused": self.paused_file.exists(),
                "paused_reason": self._read_reason(),
                "abort_requested": self.abort_file.exists(),
                "episode": {"n": self._episode_no, "goal": self._episode_goal,
                            "calls": self._episode_calls, "usd": round(self._episode_usd, 3),
                            "max_calls": self.max_calls_episode, "max_usd": self.max_usd_episode},
                "hour": {"usd": round(self._hour_usd(), 3), "max_usd": self.max_usd_hour},
                "session": {"calls": self._total_calls, "usd": round(self._total_usd, 3),
                            "max_usd": round(self.max_usd_total, 2),
                            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._session_start))},
                "guard_dir": str(self.dir),
            }

    def _read_reason(self) -> str:
        try:
            return json.loads(self.paused_file.read_text()).get("reason", "")
        except Exception:
            return ""

    def _pause(self, reason: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {"reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **self.status()}
        payload["paused"] = True
        self.paused_file.write_text(json.dumps(payload, indent=1))
        logger.error("LLM GUARD PAUSED: %s  -> review, then `python scripts/llm_guard.py resume` (or abort)", reason)

    def _apply_resume(self) -> None:
        """A human wrote RESUME: restart the windows and give the session a fresh allowance."""
        self._episode_calls = 0
        self._episode_usd = 0.0
        self._hour.clear()
        self.max_usd_total = self._total_usd + _env_float("MINEEVOLVE_GUARD_MAX_USD_TOTAL", 20.0)
        try:
            self.resume_file.unlink()
        except OSError:
            pass
        logger.warning("LLM guard resumed by operator; counters reset, session cap now $%.2f", self.max_usd_total)

    # -- hooks -----------------------------------------------------------
    def episode_start(self, goal: str = "") -> None:
        with self._lock:
            self._episode_no += 1
            self._episode_goal = goal
            self._episode_calls = 0
            self._episode_usd = 0.0

    def before_call(self, stage: str = "") -> None:
        with self._lock:
            if self.abort_file.exists():
                raise GuardAborted("ABORT marker present")
            if self.resume_file.exists():
                self._apply_resume()
            if self.paused_file.exists():
                raise GuardPaused(self._read_reason() or "PAUSED file present")
            checks = (
                (self._episode_calls >= self.max_calls_episode,
                 f"{self._episode_calls} LLM calls in one episode (limit {self.max_calls_episode}; healthy <= 48)"),
                (self._episode_usd >= self.max_usd_episode,
                 f"${self._episode_usd:.2f} in one episode (limit ${self.max_usd_episode:.2f})"),
                (self._hour_usd() >= self.max_usd_hour,
                 f"${self._hour_usd():.2f} in the last hour (limit ${self.max_usd_hour:.2f})"),
                (self._total_usd >= self.max_usd_total,
                 f"${self._total_usd:.3f} this server session (limit ${self.max_usd_total:.3f})"),
            )
            for tripped, reason in checks:
                if tripped:
                    self._pause(f"{reason}; next call would be stage={stage}, episode {self._episode_no} '{self._episode_goal}'")
                    raise GuardPaused(reason)

    def after_call(self, prompt_tokens: int | None, completion_tokens: int | None) -> float:
        usd = ((prompt_tokens or 0) * self.price_in + (completion_tokens or 0) * self.price_out) / 1e6
        with self._lock:
            self._episode_calls += 1
            self._episode_usd += usd
            self._total_calls += 1
            self._total_usd += usd
            self._hour.append((time.time(), usd))
        return usd


GUARD = LLMGuard()
