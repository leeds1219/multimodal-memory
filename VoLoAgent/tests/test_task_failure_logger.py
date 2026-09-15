# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the passive TaskFailureLogger (Aspect 1).

These tests drive the logger through synthetic obs sequences and check
that the resulting ``task_failures.jsonl`` contains the expected event
types in the expected order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_orchestrator.diagnostics.metrics1_failure_logger import TaskFailureLogger
from vlm_orchestrator.diagnostics.metrics1_gt_rules import (
    WRONG_TARGET_PLACE_CONFIRM_CHUNKS,
)


def _ladder(obj, *, grabbed=False, above=False, dropped=False, in_cont=False):
    return [
        {"object": obj, "condition_idx": 0, "satisfied": grabbed, "info": ""},
        {"object": obj, "condition_idx": 1, "satisfied": above, "info": ""},
        {"object": obj, "condition_idx": 2, "satisfied": dropped, "info": ""},
        {"object": obj, "condition_idx": 3, "satisfied": in_cont, "info": ""},
    ]


def _gt(*, score=0.0, conditions=None, object_completed=None,
        all_subtask_conditions=None, grasped=None, objects_in_contact=None):
    return {
        "subtask": {
            "score": score,
            "conditions": conditions or [],
            "object_completed": object_completed or {},
            "all_subtask_conditions": all_subtask_conditions or {},
        },
        "robot": {
            "grasped_object": grasped,
            "objects_in_contact": objects_in_contact or [],
        },
    }


def _read_events(log_dir: Path) -> list[dict]:
    """Read task_failures.jsonl as a list of event dicts."""
    path = log_dir / "task_failures.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestLifecycle:

    def test_no_log_when_gt_state_missing(self, tmp_path):
        L = TaskFailureLogger()
        L.on_episode_start(episode_log_dir=tmp_path, episode_id=1, instruction="t")
        # observe with no gt_state
        L.observe({"some": "obs"}, step=1)
        L.on_episode_end()
        # No file created
        assert not (tmp_path / "task_failures.jsonl").exists()

    def test_episode_start_and_end_events(self, tmp_path):
        L = TaskFailureLogger()
        L.on_episode_start(episode_log_dir=tmp_path, episode_id=42, instruction="put X in Y")
        L.observe({"gt_state": _gt(object_completed={"red": False})}, step=1)
        L.on_episode_end()
        events = _read_events(tmp_path)
        assert events[0]["type"] == "episode_start"
        assert events[0]["episode_id"] == 42
        assert events[0]["instruction"] == "put X in Y"
        assert events[1]["type"] == "task_targets_resolved"
        assert events[1]["targets"] == ["red"]
        assert events[-1]["type"] == "episode_end"

    def test_session_log_dir_fallback(self, tmp_path):
        """If on_episode_start has no episode_log_dir, falls back to session dir."""
        L = TaskFailureLogger(session_log_dir=str(tmp_path))
        L.on_episode_start()  # no per-episode dir
        L.observe({"gt_state": _gt(object_completed={"red": False})}, step=1)
        L.on_episode_end()
        events = _read_events(tmp_path)
        assert any(e["type"] == "task_targets_resolved" for e in events)

    def test_no_log_dir_no_crash(self, tmp_path):
        """If no log dir is configured at all, observe() is a no-op."""
        L = TaskFailureLogger()
        L.on_episode_start()  # no log dir anywhere
        L.observe({"gt_state": _gt(object_completed={"red": False})}, step=1)
        # Doesn't crash; no files written
        L.on_episode_end()


class TestEventEmission:

    def test_no_progress_is_not_emitted(self, tmp_path):
        """``no_progress`` was deliberately removed from the failure
        type list — score stagnation alone is low-information.  The
        logger must NEVER emit ``no_progress`` events even when score
        stagnates for a long stretch."""
        L = TaskFailureLogger(session_log_dir=str(tmp_path))
        L.on_episode_start(episode_log_dir=tmp_path)
        gt = _gt(score=0.0, object_completed={"red": False})
        # Long stagnation period
        for i in range(100):
            L.observe({"gt_state": gt}, step=i)
        L.on_episode_end()
        events = _read_events(tmp_path)
        no_progress = [e for e in events if e["type"] == "no_progress"]
        assert no_progress == [], f"unexpected no_progress events: {events}"

    def test_wrong_target_place_event(self, tmp_path):
        L = TaskFailureLogger(session_log_dir=str(tmp_path))
        L.on_episode_start(episode_log_dir=tmp_path)
        # Object released, not grabbed, not in container — for streak chunks
        conds = _ladder("red", above=True, dropped=True)
        gt = _gt(
            conditions=conds,
            object_completed={"red": False},
        )
        for i in range(WRONG_TARGET_PLACE_CONFIRM_CHUNKS + 2):
            L.observe({"gt_state": gt}, step=i)
        L.on_episode_end()
        events = _read_events(tmp_path)
        wtp = [e for e in events if e["type"] == "wrong_target_place"]
        assert len(wtp) >= 1, f"events: {events}"

    def test_object_complete_event(self, tmp_path):
        L = TaskFailureLogger(session_log_dir=str(tmp_path))
        L.on_episode_start(episode_log_dir=tmp_path)
        # First observation: target visible, not yet completed
        gt_pending = _gt(score=0.5, object_completed={"red": False})
        L.observe({"gt_state": gt_pending}, step=1)
        # Then completion holds for streak
        gt_done = _gt(score=1.0, object_completed={"red": True})
        for i in range(10):
            L.observe({"gt_state": gt_done}, step=i + 2)
        L.on_episode_end()
        events = _read_events(tmp_path)
        complete = [e for e in events if e["type"] == "object_complete"]
        assert len(complete) >= 1, f"events: {events}"

    def test_wrong_object_picked_event(self, tmp_path):
        L = TaskFailureLogger(session_log_dir=str(tmp_path))
        L.on_episode_start(episode_log_dir=tmp_path)
        gt = _gt(
            object_completed={"banana": False},
            grasped="apple",
            objects_in_contact=["apple"],  # banana not in contact → fires
        )
        L.observe({"gt_state": gt}, step=1)
        L.on_episode_end()
        events = _read_events(tmp_path)
        wop = [e for e in events if e["type"] == "wrong_object_picked"]
        assert len(wop) >= 1, f"events: {events}"
        assert wop[0]["grasped_object"] == "apple"

    def test_event_includes_score_and_grasped(self, tmp_path):
        L = TaskFailureLogger(session_log_dir=str(tmp_path))
        L.on_episode_start(episode_log_dir=tmp_path)
        gt = _gt(
            score=0.42,
            object_completed={"banana": False},
            grasped="apple",
            objects_in_contact=["apple"],
        )
        L.observe({"gt_state": gt}, step=1)
        L.on_episode_end()
        events = _read_events(tmp_path)
        wop = next(e for e in events if e["type"] == "wrong_object_picked")
        assert wop["score"] == pytest.approx(0.42)
        assert wop["grasped_object"] == "apple"
        assert wop["step"] == 1


class TestRobustness:

    def test_observe_handles_malformed_gt_state(self, tmp_path):
        """Logger shouldn't crash on gt_state with missing fields."""
        L = TaskFailureLogger(session_log_dir=str(tmp_path))
        L.on_episode_start(episode_log_dir=tmp_path)
        # Various weird shapes
        L.observe({"gt_state": {}}, step=1)
        L.observe({"gt_state": {"subtask": {}}}, step=2)
        L.observe({"gt_state": {"subtask": {"conditions": "not_a_list"}}}, step=3)
        L.on_episode_end()
        # Just confirm no crash; events file may or may not exist.

    def test_episode_rotation(self, tmp_path):
        """Calling on_episode_start a second time closes the first file."""
        L = TaskFailureLogger()
        ep1 = tmp_path / "ep1"
        ep2 = tmp_path / "ep2"
        L.on_episode_start(episode_log_dir=ep1, episode_id=1)
        L.observe({"gt_state": _gt(object_completed={"red": False})}, step=1)
        L.on_episode_start(episode_log_dir=ep2, episode_id=2)
        L.observe({"gt_state": _gt(object_completed={"blue": False})}, step=1)
        L.on_episode_end()
        # ep1 has its own log; ep2 has its own
        ev1 = _read_events(ep1)
        ev2 = _read_events(ep2)
        assert ev1[0]["episode_id"] == 1
        assert ev2[0]["episode_id"] == 2
        # Targets are episode-local
        assert ev1[1]["targets"] == ["red"]
        assert ev2[1]["targets"] == ["blue"]

    def test_observe_before_episode_start(self, tmp_path):
        """observe before on_episode_start is a no-op (no log dir)."""
        L = TaskFailureLogger()
        L.observe({"gt_state": _gt(object_completed={"red": False})}, step=1)
        # Doesn't crash; no file created anywhere.
