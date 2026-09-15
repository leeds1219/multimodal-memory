# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth evaluation metric event records."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_event_id() -> str:
    return f"gtm_{uuid.uuid4().hex[:12]}"


def stable_id(prefix: str, *parts: Any) -> str:
    """Build a short stable id from JSON-ish values."""
    payload = "|".join(str(_json_safe(part)) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _json_safe(value: Any) -> Any:
    """Convert common simulator/numpy values into JSON-serializable data."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


@dataclass
class GTMetricEvent:
    """Passive metric event derived from ground truth and control logs.

    These events are evaluation-only. They are written to ``metrics.jsonl``
    and are not surfaced to the VLM or policy server.
    """

    metric: str
    category: str
    step_count: int = 0
    episode_step: int | None = None
    subgoal_idx: int = 0
    decision_id: str | None = None
    decision_kind: str | None = None
    subject_object: str | None = None
    expected: str | None = None
    observed: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    parent_event_id: str | None = None
    causal_role: str = "observation"
    confidence: float = 1.0
    event_id: str = field(default_factory=_new_event_id)

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "type": "gt_metric",
            "event_id": self.event_id,
            "metric": self.metric,
            "category": self.category,
            "step_count": int(self.step_count),
            "subgoal_idx": int(self.subgoal_idx),
            "confidence": float(self.confidence),
            "causal_role": self.causal_role,
        }
        if self.episode_step is not None:
            row["episode_step"] = int(self.episode_step)
        if self.decision_id is not None:
            row["decision_id"] = self.decision_id
        if self.decision_kind is not None:
            row["decision_kind"] = self.decision_kind
        if self.subject_object is not None:
            row["subject_object"] = self.subject_object
        if self.expected is not None:
            row["expected"] = self.expected
        if self.observed is not None:
            row["observed"] = self.observed
        if self.parent_event_id is not None:
            row["parent_event_id"] = self.parent_event_id
        if self.evidence:
            row["evidence"] = _json_safe(self.evidence)
        return row
