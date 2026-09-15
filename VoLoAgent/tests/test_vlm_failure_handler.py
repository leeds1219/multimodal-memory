# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VLMFailureHandler."""

import time

import numpy as np
import pytest

from vlm_orchestrator.failure_handlers.base import (
    ACTION_CONTINUE,
    ACTION_GRASP,
    ACTION_NEXT,
    ACTION_REPLAN,
    HandlerResult,
    STATUS_COMPLETE,
    STATUS_FAILURE,
    STATUS_IN_PROGRESS,
)
from vlm_orchestrator.failure_handlers.vlm import (
    VLMFailureHandler,
    _ACTION_SETS,
    _build_system_prompt,
    _parse_vlm_detection,
)


# ======================================================================
# Prompt generation
# ======================================================================


class TestBuildSystemPrompt:
    def test_replan_mode(self):
        prompt = _build_system_prompt(_ACTION_SETS["replan"])
        assert '"next"' in prompt
        assert '"continue"' in prompt
        assert '"replan"' in prompt
        assert '"grasp_tool"' not in prompt

    def test_replan_grasp_mode(self):
        prompt = _build_system_prompt(_ACTION_SETS["replan_grasp"])
        assert '"next"' in prompt
        assert '"continue"' in prompt
        assert '"replan"' in prompt
        assert '"grasp_tool"' in prompt

    def test_grasp_mode(self):
        prompt = _build_system_prompt(_ACTION_SETS["grasp"])
        assert '"next"' in prompt
        assert '"continue"' in prompt
        assert '"replan"' not in prompt
        assert '"grasp_tool"' in prompt

    def test_stack_note_gated_on_stack_mode(self):
        """The place_stack guidance is advertised only when stack mode is
        enabled AND place_tool is an available action — never a no-op arg."""
        # replan_tools exposes place_tool.
        on = _build_system_prompt(
            _ACTION_SETS["replan_tools"], stack_mode_enabled=True,
        )
        off = _build_system_prompt(
            _ACTION_SETS["replan_tools"], stack_mode_enabled=False,
        )
        assert '"place_tool"' in on and '"place_tool"' in off
        assert "place_stack" in on
        assert "place_stack" not in off
        # No place_tool action → no stack note even with the switch on.
        no_place = _build_system_prompt(
            _ACTION_SETS["replan"], stack_mode_enabled=True,
        )
        assert "place_stack" not in no_place


# ======================================================================
# Parsing
# ======================================================================


class TestParseVLMDetection:
    def test_valid_complete(self):
        raw = '{"status": "complete", "action": "next", "reason": "done"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_COMPLETE
        assert r.action == ACTION_NEXT
        assert r.reason == "done"

    def test_valid_failure_replan(self):
        raw = '{"status": "failure", "action": "replan", "reason": "dropped"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_FAILURE
        assert r.action == ACTION_REPLAN

    def test_valid_in_progress(self):
        raw = '{"status": "in_progress", "action": "continue", "reason": "ok"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_IN_PROGRESS
        assert r.action == ACTION_CONTINUE

    def test_valid_grasp_tool(self):
        raw = '{"status": "failure", "action": "grasp_tool", "grasp_target": "red block", "reason": "wrong object"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_grasp"])
        assert r.action == ACTION_GRASP
        assert r.grasp_target == "red block"

    def test_place_stack_parsed(self):
        raw = ('{"status": "failure", "action": "place_tool", '
               '"place_destination": "on the red block", '
               '"place_stack": true, "reason": "stack on top"}')
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_tools"])
        assert r.place_destination == "on the red block"
        assert r.place_stack is True

    def test_place_stack_defaults_false(self):
        raw = ('{"status": "failure", "action": "place_tool", '
               '"place_destination": "in the white bowl"}')
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan_tools"])
        assert r.place_stack is False

    def test_invalid_json_defaults_to_continue(self):
        raw = "not json at all"
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_IN_PROGRESS
        assert r.action == ACTION_CONTINUE
        assert r.reason == "parse_failure"

    def test_invalid_status_defaults(self):
        raw = '{"status": "exploded", "action": "next"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_IN_PROGRESS

    def test_unavailable_action_defaults_to_continue(self):
        # grasp_tool not available in "replan" mode
        raw = '{"status": "failure", "action": "grasp_tool"}'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.action == ACTION_CONTINUE

    def test_markdown_code_fence(self):
        raw = '```json\n{"status": "complete", "action": "next", "reason": "moved"}\n```'
        r = _parse_vlm_detection(raw, _ACTION_SETS["replan"])
        assert r.status == STATUS_COMPLETE
        assert r.action == ACTION_NEXT


# ======================================================================
# HandlerResult validation
# ======================================================================


class TestHandlerResult:
    def test_valid_result(self):
        r = HandlerResult(status="complete", action="next")
        assert r.status == STATUS_COMPLETE

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            HandlerResult(status="exploded", action="next")

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action"):
            HandlerResult(status="complete", action="dance")


# ======================================================================
# VLMFailureHandler lifecycle
# ======================================================================


class _FakeState:
    """Minimal state stub for handler tests."""
    def __init__(self):
        self.subgoals = ["pick up red block", "place in bin"]
        self.current_subgoal_idx = 0
        self.original_instruction = "put the red block in the bin"
        self.initial_image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.initial_extra_images = None
        self.infer_count = 10
        self.episode_id = 1
        self.episode_step = 0  # bumped manually in tests to drive cadence
        self.vlm_check_result = None
        self.vlm_check_infer_step = None
        self.log_entries = []
        self.metric_entries = []

    def tick(self):
        """Advance both counters by one chunk worth of step (=1 in tests)."""
        self.infer_count += 1
        self.episode_step += 1

    def log(self, entry):
        """Match SessionState.log signature so the handler can record events."""
        entry.setdefault("timestamp", time.time())
        entry.setdefault("episode_id", self.episode_id)
        self.log_entries.append(entry)


class TestVLMFailureHandler:
    def _make_handler(self, vlm_response, check_interval=5):
        """Create handler with a mock VLM that returns fixed response."""
        def mock_vlm_call(system_prompt, user_content):
            return vlm_response

        def mock_build_check(text, init_img, cur_img, extra, **kw):
            return [{"type": "text", "text": text}]

        def mock_get_image(obs):
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def mock_get_extra(obs):
            return None

        return VLMFailureHandler(
            vlm_call_fn=mock_vlm_call,
            image_builder_fn=mock_build_check,
            get_vlm_image_fn=mock_get_image,
            get_extra_images_fn=mock_get_extra,
            check_interval=check_interval,
            recovery_mode="replan",
        )

    def test_returns_none_before_interval(self):
        handler = self._make_handler('{"status": "complete", "action": "next"}')
        state = _FakeState()
        handler.on_episode_start({}, state)

        # Steps 1-4: should return None (not time yet)
        for _ in range(4):
            state.tick()
            assert handler.step({}, state) is None

    def test_returns_result_at_interval(self):
        handler = self._make_handler(
            '{"status": "complete", "action": "next", "reason": "block moved"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        # Steps 1-4: not time yet
        for _ in range(4):
            state.tick()
            handler.step({}, state)

        # Step 5: should check and return result
        state.tick()
        result = handler.step({}, state)
        assert result is not None
        assert result.status == STATUS_COMPLETE
        assert result.action == ACTION_NEXT

    def test_in_progress_continue_returns_none(self):
        """in_progress + continue = nothing to report."""
        handler = self._make_handler(
            '{"status": "in_progress", "action": "continue", "reason": "ok"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        for _ in range(5):
            state.tick()
            result = handler.step({}, state)
        # in_progress + continue is filtered to None
        assert result is None
        # But check result is stored on state
        assert state.vlm_check_result is not None
        assert state.vlm_check_result["status"] == STATUS_IN_PROGRESS

    def test_failure_replan_returns_result(self):
        handler = self._make_handler(
            '{"status": "failure", "action": "replan", "reason": "stuck"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        for _ in range(5):
            state.tick()
            result = handler.step({}, state)
        assert result is not None
        assert result.status == STATUS_FAILURE
        assert result.action == ACTION_REPLAN

    def test_on_subgoal_advanced_resets_counter(self):
        handler = self._make_handler(
            '{"status": "complete", "action": "next"}'
        )
        state = _FakeState()
        handler.on_episode_start({}, state)

        # 3 steps in
        for _ in range(3):
            state.tick()
            handler.step({}, state)

        # Advance resets counter
        handler.on_subgoal_advanced({}, state, 1)

        # Now need 5 more steps to trigger
        for i in range(4):
            state.tick()
            assert handler.step({}, state) is None
        state.tick()
        result = handler.step({}, state)
        assert result is not None

    def test_invalid_recovery_mode(self):
        with pytest.raises(ValueError, match="Invalid recovery_mode"):
            VLMFailureHandler(
                vlm_call_fn=lambda s, u: "",
                image_builder_fn=lambda *a, **k: [],
                get_vlm_image_fn=lambda o: None,
                get_extra_images_fn=lambda o: None,
                recovery_mode="nonexistent",
            )

    def test_no_subgoals_returns_none(self):
        handler = self._make_handler('{"status": "complete", "action": "next"}')
        state = _FakeState()
        state.subgoals = []
        handler.on_episode_start({}, state)

        for _ in range(10):
            assert handler.step({}, state) is None
