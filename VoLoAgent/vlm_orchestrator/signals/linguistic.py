# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linguistic Vagueness metric.

Pure text analysis — no VLA or VLM needed.  Counts grounding words
(colors, spatial refs, quantities, sequencing), penalises vague pronouns
and implicit actions, and produces a composite vagueness score.

Higher score → more vague.
"""

from __future__ import annotations

import re

from vlm_orchestrator.signals.base import Metric, MetricResult

# ── Word lists ──────────────────────────────────────────────────────────────

DEST_WORDS = {
    "bowl", "bin", "crate", "plate", "tray", "shelf", "box",
    "table", "pot", "mug", "pail", "inside", "into", "onto",
}
VAGUE_OBJECTS = {
    "it", "them", "things", "items", "stuff", "objects",
    "food", "fruit", "animals", "toys", "tools", "dishes",
}
COLORS = {
    "red", "blue", "green", "yellow", "orange", "white", "black",
    "grey", "gray", "brown", "pink", "purple", "silver", "gold",
    "dark", "light", "multicolored",
}
SIZE_WORDS = {
    "big", "large", "small", "little", "tiny", "bigger", "larger", "smaller",
}
SPATIAL_WORDS = {
    "left", "right", "top", "bottom", "front", "back", "behind", "above",
    "below", "next", "beside", "between",
}
QUANTITY_WORDS = {
    "one", "two", "three", "four", "both", "all", "each", "every", "exactly",
}
SEQUENCING_WORDS = {"first", "then", "after", "before", "next", "finally"}
VAGUE_ACTIONS = {
    "put away", "clean up", "toss", "throw away", "tidy",
    "sort", "organize", "make sure", "identify",
}
REASONING_WORDS = {
    "bigger", "larger", "smaller", "taller", "shorter",
    "closest", "nearest", "most", "least", "compare",
    "identify", "correct", "right", "wrong",
}
MULTI_STEP = {
    "then", "after", "before", "first", "next",
    "both", "all", "each", "every",
}

ALL_GROUNDING = COLORS | SIZE_WORDS | SPATIAL_WORDS | QUANTITY_WORDS | SEQUENCING_WORDS


def analyze_instruction(text: str) -> dict:
    """Compute linguistic vagueness features for a single instruction.

    Returns a flat dict of boolean/int features plus a composite
    ``vagueness_score`` (0 = very specific, higher = more vague).
    """
    text_lower = text.lower().strip()
    words = set(text_lower.split())
    n_words = len(text_lower.split())

    has_destination = bool(DEST_WORDS & words)
    has_vague_objects = bool(VAGUE_OBJECTS & words)
    has_color = bool(COLORS & words)
    has_spatial = bool(SPATIAL_WORDS & words)
    has_quantity = bool(QUANTITY_WORDS & words)
    has_sequencing = bool(SEQUENCING_WORDS & words)
    has_vague_action = any(va in text_lower for va in VAGUE_ACTIONS)
    needs_reasoning = bool(REASONING_WORDS & words)
    is_multi_step = bool(MULTI_STEP & words)
    has_count = bool(re.search(r"\b\d+\b|\bone\b|\btwo\b|\bthree\b|\bfour\b", text_lower))

    total_grounding = len(ALL_GROUNDING & words)

    # Composite score (higher = more vague)
    score = 0
    if not has_destination:
        score += 2
    if has_vague_objects:
        score += 2
    if not has_color:
        score += 1
    if has_vague_action:
        score += 1
    if needs_reasoning:
        score += 1
    if n_words <= 5:
        score += 1

    return {
        "n_words": n_words,
        "has_destination": has_destination,
        "has_vague_objects": has_vague_objects,
        "has_color": has_color,
        "has_spatial": has_spatial,
        "has_quantity": has_quantity,
        "has_sequencing": has_sequencing,
        "has_vague_action": has_vague_action,
        "needs_reasoning": needs_reasoning,
        "is_multi_step": is_multi_step,
        "has_count": has_count,
        "total_grounding": total_grounding,
        "vagueness_score": score,
    }


class Linguistic(Metric):
    """Pure-text linguistic vagueness metric."""

    name = "linguistic"
    needs_policy = False
    needs_vlm = False
    needs_obs = False

    def compute(self, instruction, **kwargs) -> MetricResult:
        features = analyze_instruction(instruction)
        return MetricResult(
            name=self.name,
            score=features["vagueness_score"],
            detail=features,
        )
