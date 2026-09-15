# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM grasp-target alias resolver — string match + optional VLM grounding.

This module exists to address a well-defined failure mode of the live
``VLMGraspTargetQADetector`` in ``detectors.py``: when a VLM names a grasp
target in natural language ("white bottle with green cap") and the GT
``object_id`` is rigid ("ranch_dressing"), a literal string/token match
fires a false-positive ``vlm_grasp_target_mismatch / target_mismatch``.
Hand-curated aliases in ``_SCENE_OBJECT_PHRASE_ALIASES`` plug the most
common cases, but new tasks / new objects break this coverage silently.

The ``TargetResolver`` class below provides a SINGLE entry point that
combines two strategies, with the choice made explicit at construction
time:

  1. **String matching** — fast, deterministic, free, no extra deps.
     Identical logic to the live detector (delegates to the same
     helpers in detectors.py so the two never drift apart). Always
     tried first.

  2. **VLM grounding** — slow, robust to new objects, costs API calls.
     Sends the VLM-spoken phrase, the candidate ``expected_targets``,
     all ``scene_objects``, and the original ``subgoal_text`` to a
     small VLM and asks: "which scene_object is the user describing?"
     Caches the answer on disk (per ``observed/expected/subgoal``
     triplet) so re-runs are free.

Recommended usage by context:

  ┌──────────────────────────────┬───────────────────────┬─────────────────────────────────┐
  │ Context                      │ Constructor           │ Why                             │
  ├──────────────────────────────┼───────────────────────┼─────────────────────────────────┤
  │ Live orchestrator eval       │ ``vlm_client=None``   │ Detector path must be cheap +   │
  │ (runs on every step)         │                       │ deterministic. Failures of      │
  │                              │                       │ string matching are still       │
  │                              │                       │ recorded for offline review.    │
  │                              │                       │                                 │
  │ Offline reprocessing         │ ``vlm_client=<X>``    │ Resolves alias gaps in already- │
  │ (``scripts/resolve_target_*``│ ``cache_path=<.json>``│ collected ``metrics.jsonl``     │
  │                              │                       │ files. ~$0.001 per event,       │
  │                              │                       │ cached forever.                 │
  │                              │                       │                                 │
  │ Dashboard build              │ ``vlm_client=None``   │ Should not call VLM at build    │
  │                              │                       │ time. Reads pre-resolved data   │
  │                              │                       │ from reprocessor if available. │
  └──────────────────────────────┴───────────────────────┴─────────────────────────────────┘

The ``_SCENE_OBJECT_PHRASE_ALIASES`` dict in ``detectors.py`` is the
single source of truth for known aliases. ``TargetResolver`` reads it
directly — there is no second copy here. To permanently teach the
system about a new alias, edit that dict.

Failure-aware behavior:
  - String match win → ``ResolveResult(aligned=True, method="string")``
  - Alias dict win  → ``ResolveResult(aligned=True, method="alias")``
  - VLM win         → ``ResolveResult(aligned=True, method="vlm", confidence=…)``
  - VLM rejection   → ``ResolveResult(aligned=False, method="vlm_reject")``
  - VLM disabled
    + no string hit → ``ResolveResult(aligned=False, method="unmatched")``

The ``method`` field is propagated into ``metrics.jsonl`` annotations
(``alias_resolution_method``) when the reprocessor runs, so downstream
analyses can split "true VLM error" from "auto-resolved alias".
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm_orchestrator.gt_metrics.detectors import (
    _SCENE_OBJECT_PHRASE_ALIASES,
    _target_matches_observed,
    _resolve_observed_scene_target,
)
from vlm_orchestrator.vlm.api import chat_create, parse_json

logger = logging.getLogger(__name__)


@dataclass
class ResolveResult:
    """Outcome of a single grasp-target resolution."""

    aligned: bool
    """True if observed_target maps to one of expected_targets."""

    matched_object: str | None
    """The scene object_id the observed phrase resolved to (if aligned)."""

    method: str
    """How the verdict was reached. One of:
       'string', 'alias', 'vlm', 'vlm_reject', 'unmatched', 'cached_vlm'."""

    confidence: float = 1.0
    """1.0 for deterministic methods; ≤1.0 from the VLM's stated confidence."""

    vlm_reason: str | None = None
    """Free-text rationale, only populated when method ∈ {vlm, vlm_reject}."""

    cache_key: str | None = field(default=None, repr=False)
    """Internal — used by the resolver to write back to its cache."""


class TargetResolver:
    """Stateful resolver — combines string match + optional VLM grounding.

    Construct ONCE per run / process / script invocation. Pass
    ``vlm_client=None`` to disable VLM grounding (default; equivalent to
    the live detector logic). To enable VLM grounding, supply any
    OpenAI-compatible ``client`` and a model id.

    Cache file format::

        {
          "<sha1 of (observed, expected_sorted, subgoal)>": {
            "aligned": true,
            "matched_object": "ranch_dressing",
            "confidence": 0.95,
            "vlm_reason": "Both phrases ...",
            "model": "YOUR_VLM_MODEL",
            "ts": "2026-05-20T00:00:00Z"
          },
          ...
        }
    """

    def __init__(
        self,
        *,
        vlm_client: Any = None,
        vlm_model: str | None = None,
        cache_path: str | os.PathLike | None = None,
    ) -> None:
        self.vlm_client = vlm_client
        self.vlm_model = vlm_model
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = self._load_cache()
        self._cache_dirty = False

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def resolve(
        self,
        observed_target: str,
        expected_targets: list[str],
        scene_objects: list[str],
        subgoal_text: str = "",
    ) -> ResolveResult:
        """Try to resolve ``observed_target`` to one of ``expected_targets``.

        Strategy order (first hit wins, returns immediately):
          1. Identity / token-overlap match  (free, deterministic)
          2. Alias dict                      (free, deterministic, uses
                                              live detector's dict)
          3. Subgoal-text-grounded rescue    (free, deterministic, only
                                              for events captured BEFORE
                                              the ``_scene_objects``
                                              union fix landed —
                                              matches the GT-extractor
                                              mis-attribution pattern)
          4. VLM grounding                   (optional, costs API calls,
                                              cached on disk)

        Returns a ``ResolveResult`` regardless of outcome. Inspect
        ``aligned`` and ``method`` to decide downstream behavior.
        """
        if not observed_target or not expected_targets:
            return ResolveResult(False, None, "unmatched")

        # 1) Direct token / 2) alias dict — both handled by the live
        # detector's _target_matches_observed (which checks both).
        for exp in expected_targets:
            if _target_matches_observed(observed_target, exp):
                method = (
                    "alias"
                    if any(
                        a in observed_target.lower()
                        for a in _SCENE_OBJECT_PHRASE_ALIASES.get(exp, set())
                    )
                    else "string"
                )
                return ResolveResult(True, exp, method)

        # 3) Subgoal-text-grounded rescue. If the subgoal text and the
        # observed_target share a non-stopword noun that is NOT in any
        # of the expected_targets, it's likely a GT-extractor
        # mis-attribution (e.g. subgoal "pick the silver spoon" +
        # observed "silver spoon" + expected ["bowl"] — the VLM
        # correctly followed the subgoal; the GT extractor returned the
        # container). The ``_scene_objects`` union fix in
        # ``detectors.py`` prevents NEW events of this shape; this
        # check rescues legacy data captured before the fix.
        if _subgoal_grounded_rescue(observed_target, expected_targets, subgoal_text):
            return ResolveResult(True, None, "subgoal_aligned")

        # 4) VLM grounding (only if a client is provided)
        if self.vlm_client is not None and self.vlm_model:
            return self._resolve_via_vlm(
                observed_target, expected_targets, scene_objects, subgoal_text,
            )

        return ResolveResult(False, None, "unmatched")

    def save_cache(self) -> None:
        """Persist VLM resolutions to ``cache_path``. Safe to call
        multiple times — no-op if cache hasn't changed."""
        if not self.cache_path or not self._cache_dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, indent=2))
        tmp.replace(self.cache_path)
        self._cache_dirty = False

    # ──────────────────────────────────────────────────────────────────
    # VLM grounding
    # ──────────────────────────────────────────────────────────────────

    def _resolve_via_vlm(
        self,
        observed_target: str,
        expected_targets: list[str],
        scene_objects: list[str],
        subgoal_text: str,
    ) -> ResolveResult:
        key = self._cache_key(observed_target, expected_targets, subgoal_text)

        # Cached?
        if key in self._cache:
            entry = self._cache[key]
            aligned = bool(entry.get("aligned"))
            return ResolveResult(
                aligned=aligned,
                matched_object=entry.get("matched_object") if aligned else None,
                method="cached_vlm" if aligned else "vlm_reject",
                confidence=float(entry.get("confidence", 1.0)),
                vlm_reason=entry.get("vlm_reason"),
                cache_key=key,
            )

        # Fresh VLM call
        prompt = _build_resolver_prompt(
            observed_target, expected_targets, scene_objects, subgoal_text,
        )
        try:
            resp = chat_create(
                self.vlm_client,
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = parse_json(raw) or {}
        except Exception as e:
            logger.warning(
                "TargetResolver VLM call failed: %s — falling back to "
                "string-only verdict (unmatched).", e,
            )
            return ResolveResult(False, None, "unmatched")

        matched = parsed.get("matched_object")
        aligned = bool(matched) and matched in expected_targets
        confidence = float(parsed.get("confidence", 0.8))
        vlm_reason = parsed.get("reason") or parsed.get("rationale") or ""

        # Write cache
        from datetime import datetime, timezone
        self._cache[key] = {
            "observed_target": observed_target,
            "expected_targets": sorted(expected_targets),
            "subgoal_text": subgoal_text,
            "aligned": aligned,
            "matched_object": matched if aligned else None,
            "confidence": confidence,
            "vlm_reason": vlm_reason[:500],
            "model": self.vlm_model,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._cache_dirty = True

        return ResolveResult(
            aligned=aligned,
            matched_object=matched if aligned else None,
            method="vlm" if aligned else "vlm_reject",
            confidence=confidence,
            vlm_reason=vlm_reason,
            cache_key=key,
        )

    # ──────────────────────────────────────────────────────────────────
    # Cache I/O
    # ──────────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path or not self.cache_path.is_file():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except Exception as e:
            logger.warning("TargetResolver cache unreadable (%s); starting empty.", e)
            return {}

    @staticmethod
    def _cache_key(
        observed_target: str,
        expected_targets: list[str],
        subgoal_text: str,
    ) -> str:
        import hashlib
        payload = "|".join([
            observed_target.strip().lower(),
            ",".join(sorted(s.strip().lower() for s in expected_targets)),
            subgoal_text.strip().lower(),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────
# Subgoal-text-grounded rescue (deterministic, free)
# ──────────────────────────────────────────────────────────────────────

_SUBGOAL_RESCUE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "from",
    "with", "into", "this", "that", "is", "are", "be",
    "table", "scene", "place", "pick", "put", "small", "large",
    "object", "objects", "container", "item", "items",
    "fallen", "lying", "loose",
})


def _subgoal_rescue_tokens(text: str) -> set[str]:
    """Tokenize for the subgoal-grounded rescue check — alpha-only,
    lowercased, len>2, stopwords removed."""
    import re as _re
    return {
        t for t in _re.findall(r"[a-z]+", (text or "").lower())
        if t not in _SUBGOAL_RESCUE_STOPWORDS and len(t) > 2
    }


def _subgoal_grounded_rescue(
    observed_target: str,
    expected_targets: list[str],
    subgoal_text: str,
) -> bool:
    """Return True iff observed and subgoal share a noun token absent
    from every expected_target — signature of a GT-extractor bug.

    This is a legacy-data rescue. New runs with the patched
    ``_scene_objects`` shouldn't produce events that match this pattern.
    """
    if not subgoal_text or not observed_target or not expected_targets:
        return False
    obs_t = _subgoal_rescue_tokens(observed_target)
    sg_t = _subgoal_rescue_tokens(subgoal_text)
    exp_t: set[str] = set()
    for exp in expected_targets:
        exp_t |= _subgoal_rescue_tokens(exp.replace("_", " "))
    shared = obs_t & sg_t
    return bool(shared) and not (shared & exp_t)


# ──────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a precise robot-perception adjudicator. Your job is to decide "
    "whether a natural-language object description and a canonical object "
    "ID refer to the SAME physical scene object. You receive only text — "
    "no image — but the description usually contains color, shape, and "
    "container cues that disambiguate. Answer in strict JSON only."
)


def _build_resolver_prompt(
    observed_target: str,
    expected_targets: list[str],
    scene_objects: list[str],
    subgoal_text: str,
) -> str:
    parts: list[str] = []
    if subgoal_text:
        parts.append(f"SUBGOAL: {subgoal_text}")
    parts.append(f"VLM-NAMED GRASP TARGET: \"{observed_target}\"")
    parts.append(
        "GT-EXPECTED CANDIDATE OBJECT IDs (the VLM is correct if it "
        "refers to ANY of these): " + ", ".join(expected_targets)
    )
    if scene_objects:
        # Cap the scene list to keep prompts small.
        scene_str = ", ".join(scene_objects[:30])
        parts.append(f"ALL SCENE OBJECTS (for context): {scene_str}")
    parts.append(
        "Decide: does the VLM-named grasp target plausibly refer to "
        "any of the GT-expected candidates?"
    )
    parts.append(
        'Respond with strict JSON: '
        '{"matched_object": "<exact_candidate_id_or_null>", '
        '"confidence": <0..1>, '
        '"reason": "<one-sentence justification>"}'
    )
    parts.append(
        "Rules:\n"
        "- If the description clearly matches one candidate (e.g., "
        "'white bottle with green cap' ↔ 'ranch_dressing'), set "
        "matched_object to that candidate.\n"
        "- If the description matches NONE of the candidates, set "
        "matched_object to null.\n"
        "- Do not invent objects. matched_object must be exactly one "
        "of the listed candidate IDs (case-sensitive), or null.\n"
        "- Visual cues only: ignore subgoal-text noise like \"on the "
        "table\" or \"lying down\"."
    )
    return "\n\n".join(parts)
