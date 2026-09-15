# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the subgoal strategy."""

import json

import numpy as np
import pytest

from vlm_orchestrator.strategies.base import SessionState, StrategyContext
from vlm_orchestrator.failure_handlers.base import (
    ACTION_REPLAN,
    HandlerResult,
    STATUS_FAILURE,
)
from vlm_orchestrator.strategies.subgoal import (
    SubgoalConfig,
    SubgoalStrategy,
)
from vlm_orchestrator.strategies.subgoal_base import MAX_RECYCLES
from vlm_orchestrator.vlm import parse_json as _parse_json
from vlm_orchestrator.vlm import VLMBackend


# ---- Mock VLM that returns canned decompositions / checks ----


class MockSubgoalVLM(VLMBackend):
    """VLM backend whose behaviour is controlled by test fixtures."""

    def __init__(self):
        # Map instruction -> (list of subgoals, ordered) for decompose
        self.decompose_responses: dict[str, list[str]] = {}
        # Map instruction -> ordered flag (default False)
        self.decompose_ordered: dict[str, bool] = {}
        # Sequence of check responses (popped in order)
        self.check_responses: list[dict] = []

    def rewrite_instruction(
        self,
        instruction: str,
        image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        return instruction  # unused by subgoal strategy


# ---- Patched SubgoalStrategy that uses MockSubgoalVLM ----


class TestableSubgoalStrategy(SubgoalStrategy):
    """SubgoalStrategy with VLM calls replaced by mock responses."""

    def __init__(self, ctx: StrategyContext, config: SubgoalConfig, mock_vlm: MockSubgoalVLM):
        super().__init__(ctx, config)
        self._mock = mock_vlm

    def _vlm_call(self, system_prompt: str, user_content: list[dict]) -> str:
        """Return mock responses instead of calling OpenAI."""
        # Determine if this is a decompose or check call by inspecting
        # the system prompt.
        if "subgoals" in system_prompt.lower():
            # Decompose call — find instruction from user_content text
            text = ""
            for item in user_content:
                if item.get("type") == "text":
                    text = item["text"]
                    break
            # Find the instruction in quotes
            import re
            m = re.search(r'"([^"]+)"', text)
            instruction = m.group(1) if m else ""
            subgoals = self._mock.decompose_responses.get(
                instruction, [instruction]
            )
            ordered = self._mock.decompose_ordered.get(instruction, False)
            return json.dumps({"subgoals": subgoals, "ordered": ordered})
        elif '{"done": true}' in system_prompt:
            # Binary check call (hybrid mode) — identified by the
            # CHECK_DONE_SYSTEM_PROMPT which contains '{"done": true}'
            if self._mock.check_responses:
                resp = self._mock.check_responses.pop(0)
                # Translate action-based mock to done-based
                done = resp.get("action") == "next" or resp.get("done", False)
                return json.dumps({"done": done})
            return json.dumps({"done": False})
        else:
            # Full check call (vlm mode)
            if self._mock.check_responses:
                resp = self._mock.check_responses.pop(0)
            else:
                resp = {"action": "continue"}
            return json.dumps(resp)


# ---- Helpers ----


def _make_obs(prompt: str) -> dict:
    return {
        "prompt": prompt,
        "observation/exterior_image_1_left": np.zeros(
            (224, 224, 3), dtype=np.uint8
        ),
    }


def _make_strategy(
    mock_vlm: MockSubgoalVLM,
    check_interval: int = 5,
    subgoal_timeout: int = 20,
    check_mode: str = "vlm",
) -> TestableSubgoalStrategy:
    ctx = StrategyContext(vlm=mock_vlm)
    config = SubgoalConfig(
        check_mode=check_mode,
        check_interval=check_interval,
        subgoal_timeout=subgoal_timeout,
    )
    return TestableSubgoalStrategy(ctx, config, mock_vlm)


# ---- Tests ----


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        text = '```json\n{"subgoals": ["x"]}\n```'
        assert _parse_json(text) == {"subgoals": ["x"]}

    def test_json_in_prose(self):
        text = 'Here is the plan:\n{"action": "next"}\nDone.'
        assert _parse_json(text) == {"action": "next"}

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_json("no json here")


class TestDecomposition:
    def test_single_subgoal(self):
        """Simple task → 1 subgoal (effective rewrite)."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["pick up the banana"] = [
            "Pick up the yellow banana and place it in the red bowl"
        ]
        strategy = _make_strategy(mock)

        obs = _make_obs("pick up the banana")
        state = SessionState()

        obs, state = strategy.process(obs, state)

        assert state.subgoals == [
            "Pick up the yellow banana and place it in the red bowl"
        ]
        assert state.current_subgoal_idx == 0
        assert obs["prompt"] == (
            "Pick up the yellow banana and place it in the red bowl"
        )

    def test_multi_subgoal(self):
        """Complex task → multiple subgoals; only first is active."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["put the apple and yogurt in the bowl"] = [
            "Pick up the red apple and place it in the black bowl",
            "Pick up the white yogurt cup and place it in the black bowl",
        ]
        strategy = _make_strategy(mock)

        obs = _make_obs("put the apple and yogurt in the bowl")
        state = SessionState()

        obs, state = strategy.process(obs, state)

        assert len(state.subgoals) == 2
        assert state.current_subgoal_idx == 0
        assert obs["prompt"] == (
            "Pick up the red apple and place it in the black bowl"
        )


class TestNormalizeSubgoals:
    """Verify that dict-format subgoals are normalised to plain strings."""

    def test_dict_subgoals_extract_instruction(self):
        """VLM returns structured dicts → only instruction text is kept."""
        from vlm_orchestrator.strategies.subgoal import _normalize_subgoals

        raw = [
            {"instruction": "Pick up the green cube and place it in the bin",
             "target_object": "green cube"},
            {"instruction": "Pick up the red cube and place it in the bin",
             "target_object": "red cube"},
        ]
        result = _normalize_subgoals(raw)
        assert result == [
            "Pick up the green cube and place it in the bin",
            "Pick up the red cube and place it in the bin",
        ]

    def test_string_subgoals_pass_through(self):
        """Plain string subgoals are returned unchanged."""
        from vlm_orchestrator.strategies.subgoal import _normalize_subgoals

        raw = ["Pick up the banana", "Place it in the bowl"]
        assert _normalize_subgoals(raw) == raw

    def test_mixed_string_and_dict(self):
        """Mix of string and dict entries handled correctly."""
        from vlm_orchestrator.strategies.subgoal import _normalize_subgoals

        raw = [
            "Open the drawer",
            {"instruction": "Pick up the apple", "target_object": "apple"},
        ]
        assert _normalize_subgoals(raw) == [
            "Open the drawer",
            "Pick up the apple",
        ]

    def test_dict_without_instruction_key_falls_back(self):
        """Dict missing 'instruction' key falls back to str(dict)."""
        from vlm_orchestrator.strategies.subgoal import _normalize_subgoals

        raw = [{"target_object": "banana"}]
        result = _normalize_subgoals(raw)
        assert len(result) == 1
        assert "banana" in result[0]

    def test_decompose_with_dict_subgoals(self):
        """End-to-end: VLM returns dict subgoals → policy gets clean text."""
        mock = MockSubgoalVLM()
        # Simulate VLM returning structured dicts (like the recycle prompt)
        mock.decompose_responses["sort the blocks"] = [
            {"instruction": "Pick up the red block and place it in the bin",
             "target_object": "red block"},
            {"instruction": "Pick up the blue block and place it in the bin",
             "target_object": "blue block"},
        ]
        strategy = _make_strategy(mock)
        obs = _make_obs("sort the blocks")
        state = SessionState()

        obs, state = strategy.process(obs, state)

        # state.subgoals must be plain strings, not dict reprs
        assert state.subgoals == [
            "Pick up the red block and place it in the bin",
            "Pick up the blue block and place it in the bin",
        ]
        assert obs["prompt"] == "Pick up the red block and place it in the bin"
        # Verify no dict-like characters leaked into the instruction
        assert "{" not in obs["prompt"]
        assert "target_object" not in obs["prompt"]


class TestPeriodicCheck:
    def test_continue_keeps_instruction(self):
        """VLM says 'continue' → instruction unchanged."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        mock.check_responses = [{"action": "continue"}]
        strategy = _make_strategy(mock, check_interval=2)

        state = SessionState()
        obs = _make_obs("task")

        # First call: decompose
        obs, state = strategy.process(obs, state)
        assert obs["prompt"] == "step 1"
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # 2 more chunks (trigger check at chunk 2)
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # After check, still on step 1
        assert state.current_subgoal_idx == 0
        assert obs["prompt"] == "step 1"

    def test_next_advances_subgoal(self):
        """VLM says 'next' → advance to second subgoal."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        mock.check_responses = [{"action": "next"}]
        strategy = _make_strategy(mock, check_interval=2)

        state = SessionState()
        obs = _make_obs("task")

        # Decompose
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # 2 more chunks → triggers check
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.current_subgoal_idx == 1
        assert obs["prompt"] == "step 2"

    def test_refine_updates_instruction(self):
        """VLM says 'refine' → instruction updated in-place."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        mock.check_responses = [
            {"action": "refine", "instruction": "step 1 improved"},
        ]
        strategy = _make_strategy(mock, check_interval=2)

        state = SessionState()
        obs = _make_obs("task")

        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.current_subgoal_idx == 0
        assert obs["prompt"] == "step 1 improved"
        assert state.subgoals[0] == "step 1 improved"


class TestSubgoalTimeout:
    def test_timeout_advances(self):
        """After subgoal_timeout chunks, auto-advance to next subgoal."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        # All checks return continue (never completes)
        mock.check_responses = [{"action": "continue"}] * 20
        strategy = _make_strategy(mock, check_interval=3, subgoal_timeout=6)

        state = SessionState()
        obs = _make_obs("task")

        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # Run 6 more chunks → timeout
        for _ in range(6):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.current_subgoal_idx == 1
        assert obs["prompt"] == "step 2"

    def test_timeout_on_last_subgoal_stays(self):
        """Timeout on the last subgoal does not crash."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["only step"]
        mock.check_responses = [{"action": "continue"}] * 20
        strategy = _make_strategy(mock, check_interval=3, subgoal_timeout=4)

        state = SessionState()
        obs = _make_obs("task")

        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        for _ in range(5):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # Still on index 0 (only subgoal), no crash
        assert state.current_subgoal_idx == 0
        assert obs["prompt"] == "only step"


class TestNewEpisodeReset:
    def test_new_episode_resets_state(self):
        """When the prompt changes, subgoal state is fully reset."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task A"] = ["A1", "A2"]
        mock.decompose_responses["task B"] = ["B1"]
        strategy = _make_strategy(mock, check_interval=100)

        state = SessionState()

        # Episode 1
        obs = _make_obs("task A")
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1
        assert len(state.subgoals) == 2
        assert state.episode_id == 1

        # Episode 2 (prompt changes)
        obs = _make_obs("task B")
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.episode_id == 2
        assert state.subgoals == ["B1"]
        assert state.current_subgoal_idx == 0
        # subgoal just (re)started; allow off-by-one because the
        # test driver bumps episode_step *after* process() runs
        assert state.episode_step - state.step_at_subgoal_start <= 1
        assert obs["prompt"] == "B1"


class TestLogEntries:
    def test_decompose_and_check_produce_logs(self):
        """Strategy produces structured log entries."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        mock.check_responses = [{"action": "next"}]
        strategy = _make_strategy(mock, check_interval=2)

        state = SessionState()
        obs = _make_obs("task")

        # Decompose — no explicit log entry (logged at higher level)
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # Trigger check
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        check_logs = [e for e in state.log_entries if e["type"] == "check"]
        assert len(check_logs) == 1
        assert check_logs[0]["action"] == "next"


class TestTimerCheckMode:
    def test_timer_mode_skips_vlm_check(self):
        """In timer mode, no VLM check calls are made — only timeout advances."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        # If VLM check were called, it would return "next" immediately —
        # but in timer mode it should never be called.
        mock.check_responses = [{"action": "next"}] * 20
        strategy = _make_strategy(
            mock, check_interval=2, subgoal_timeout=4, check_mode="timer",
        )

        state = SessionState()
        obs = _make_obs("task")

        # Decompose
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1
        assert obs["prompt"] == "step 1"

        # Run 2 chunks — in vlm mode this would trigger a check,
        # but in timer mode nothing happens.
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # Still on step 1 (no VLM check fired "next")
        assert state.current_subgoal_idx == 0
        assert obs["prompt"] == "step 1"

        # 2 more chunks → timeout at 4 → advance
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.current_subgoal_idx == 1
        assert obs["prompt"] == "step 2"
        # check_responses should be untouched (never popped)
        assert len(mock.check_responses) == 20

    def test_timer_mode_produces_timeout_logs(self):
        """Timer mode still produces subgoal_timeout log entries."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        strategy = _make_strategy(
            mock, check_interval=100, subgoal_timeout=3, check_mode="timer",
        )

        state = SessionState()
        obs = _make_obs("task")
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        for _ in range(3):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        timeout_logs = [
            e for e in state.log_entries if e["type"] == "subgoal_timeout"
        ]
        assert len(timeout_logs) == 1
        assert timeout_logs[0]["instruction"] == "step 2"


class TestHandlerReplan:
    def test_replan_action_does_not_recycle_after_max_attempts(self):
        mock = MockSubgoalVLM()
        strategy = _make_strategy(mock)
        state = SessionState()
        state.original_instruction = "recover the scene"
        state.subgoals = ["Pick the tuna can from the lower shelf"]
        state.current_subgoal_idx = 0
        state.rewritten_instruction = state.subgoals[0]
        state.initial_image = np.zeros((224, 224, 3), dtype=np.uint8)
        state.episode_id = 1
        state.episode_step = 10
        strategy._recycle_count = MAX_RECYCLES

        obs = _make_obs("recover the scene")
        result = HandlerResult(
            status=STATUS_FAILURE,
            action=ACTION_REPLAN,
            reason="target is no longer reachable",
        )

        strategy._execute_handler_result(obs, state, result)

        assert strategy._recycle_count == MAX_RECYCLES
        assert any(
            entry["type"] == "replan_failed"
            and entry["reason"] == "max recycles reached"
            for entry in state.log_entries
        )


class TestHybridCheckMode:
    def test_ordered_task_uses_vlm_check(self):
        """Hybrid mode: ordered task uses VLM check, advances on 'done'."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        mock.decompose_ordered["task"] = True
        # VLM says done on first check
        mock.check_responses = [{"action": "next"}]
        strategy = _make_strategy(
            mock, check_interval=2, subgoal_timeout=30,
            check_mode="hybrid",
        )

        state = SessionState()
        obs = _make_obs("task")

        # Decompose
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1
        assert state.subgoals_ordered is True
        assert obs["prompt"] == "step 1"

        # 2 chunks → triggers check → VLM says done → advance
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.current_subgoal_idx == 1
        assert obs["prompt"] == "step 2"

    def test_unordered_task_uses_timer(self):
        """Hybrid mode: unordered task ignores VLM check, uses timer."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        mock.decompose_ordered["task"] = False
        # VLM would say done, but should never be called
        mock.check_responses = [{"action": "next"}] * 10
        strategy = _make_strategy(
            mock, check_interval=2, subgoal_timeout=4,
            check_mode="hybrid",
        )

        state = SessionState()
        obs = _make_obs("task")

        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1
        assert state.subgoals_ordered is False

        # 2 chunks → would trigger check in vlm mode, but not in hybrid/unordered
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # Still on step 1 (no VLM check fired)
        assert state.current_subgoal_idx == 0

        # 2 more chunks → timeout at 4 → advance
        for _ in range(2):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.current_subgoal_idx == 1
        assert obs["prompt"] == "step 2"
        # Check responses untouched
        assert len(mock.check_responses) == 10

    def test_timeout_is_same_for_ordered_and_unordered(self):
        """Both ordered and unordered tasks use the same subgoal_timeout."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["task"] = ["step 1", "step 2"]
        mock.decompose_ordered["task"] = True
        # VLM always says not done → should hit timeout
        mock.check_responses = [{"action": "continue"}] * 50
        strategy = _make_strategy(
            mock, check_interval=3, subgoal_timeout=8,
            check_mode="hybrid",
        )

        state = SessionState()
        obs = _make_obs("task")

        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        # After 7 chunks, still on subgoal 0
        for _ in range(7):
            obs = _make_obs("task")
            obs, state = strategy.process(obs, state)
            state.infer_count += 1
            state.episode_step += 1  # mimic proxy: 1 sim step per chunk in tests
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1
        assert state.current_subgoal_idx == 0

        # 1 more = 8 total → timeout → advance
        obs = _make_obs("task")
        obs, state = strategy.process(obs, state)
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1

        assert state.current_subgoal_idx == 1
        assert obs["prompt"] == "step 2"


class TestInitialStrategyComposition:
    def test_rewrite_then_decompose(self):
        """Initial RewriteStrategy runs before decomposition."""
        from vlm_orchestrator.strategies.archive.rewrite import RewriteStrategy

        class UppercaseVLM(VLMBackend):
            def rewrite_instruction(self, instruction, image, extra_images=None):
                return instruction.upper()

        vlm = UppercaseVLM()
        ctx = StrategyContext(vlm=vlm)
        initial = RewriteStrategy(ctx, mode="first_per_episode")

        mock = MockSubgoalVLM()
        # The decomposition will receive the UPPERCASED instruction
        mock.decompose_responses["PICK UP THE BANANA"] = [
            "Grasp the yellow banana",
            "Place it in the red bowl",
        ]

        config = SubgoalConfig(check_interval=100, subgoal_timeout=100)
        strategy = TestableSubgoalStrategy(ctx, config, mock)
        strategy.initial_strategy = initial

        state = SessionState()
        obs = _make_obs("pick up the banana")

        obs, state = strategy.process(obs, state)

        assert state.subgoals == [
            "Grasp the yellow banana",
            "Place it in the red bowl",
        ]
        assert obs["prompt"] == "Grasp the yellow banana"
        # original_instruction is the raw input
        assert state.original_instruction == "pick up the banana"

    def test_no_initial_strategy_decomposes_directly(self):
        """Without initial strategy, decomposition gets the raw instruction."""
        mock = MockSubgoalVLM()
        mock.decompose_responses["pick up the banana"] = ["grab banana"]
        strategy = _make_strategy(mock)

        state = SessionState()
        obs = _make_obs("pick up the banana")
        obs, state = strategy.process(obs, state)

        assert state.subgoals == ["grab banana"]
        assert obs["prompt"] == "grab banana"


# ======================================================================
# HITL integration tests
# ======================================================================


def _make_hitl_strategy(
    mock_vlm: MockSubgoalVLM,
    hitl_state=None,
    check_interval: int = 5,
    subgoal_timeout: int = 20,
    check_mode: str = "vlm",
) -> TestableSubgoalStrategy:
    """Build a testable strategy with a HITLState attached."""
    ctx = StrategyContext(vlm=mock_vlm)
    config = SubgoalConfig(
        check_mode=check_mode,
        check_interval=check_interval,
        subgoal_timeout=subgoal_timeout,
    )
    strategy = TestableSubgoalStrategy(ctx, config, mock_vlm)
    strategy._hitl = hitl_state
    return strategy


class TestHITLEpisodeStart:
    """HITL mode: episode start blocks until human provides subgoals."""

    def test_wait_for_subgoals_blocks_then_applies(self):
        """Submit Subgoals (START_SUBGOAL) unblocks the episode."""
        import threading
        from vlm_orchestrator.hitl import HITLAction, HITLState

        hitl = HITLState()
        mock = MockSubgoalVLM()
        # The decompose response should NEVER be used in HITL mode.
        mock.decompose_responses["sort blocks"] = ["should not appear"]
        strategy = _make_hitl_strategy(mock, hitl_state=hitl)

        state = SessionState()
        obs = _make_obs("sort blocks")

        # Fire the human action after a short delay.
        def submit():
            import time
            time.sleep(0.1)
            hitl.set_action(
                HITLAction.START_SUBGOAL,
                {"instruction": "pick red block\nplace in bin"},
            )
        threading.Thread(target=submit, daemon=True).start()

        obs, state = strategy.process(obs, state)

        assert state.subgoals == ["pick red block", "place in bin"]
        assert state.rewritten_instruction == "pick red block"
        assert obs["prompt"] == "pick red block"
        # The VLM decompose response must NOT have been used.
        assert "should not appear" not in state.subgoals

    def test_wait_pushes_first_image(self):
        """The first camera frame is pushed to HITLState before blocking."""
        import threading
        from vlm_orchestrator.hitl import HITLAction, HITLState

        hitl = HITLState()
        mock = MockSubgoalVLM()
        strategy = _make_hitl_strategy(mock, hitl_state=hitl)

        state = SessionState()
        obs = _make_obs("sort blocks")

        # Submit immediately so we don't block forever.
        def submit():
            import time
            time.sleep(0.05)
            hitl.set_action(
                HITLAction.START_SUBGOAL,
                {"instruction": "pick block"},
            )
        threading.Thread(target=submit, daemon=True).start()

        strategy.process(obs, state)

        img, _ = hitl.get_image()
        assert img is not None, "first frame should be pushed at episode start"

    def test_abort_during_wait(self):
        """ABORT during subgoal wait sets instruction to 'stop'."""
        import threading
        from vlm_orchestrator.hitl import HITLAction, HITLState

        hitl = HITLState()
        mock = MockSubgoalVLM()
        strategy = _make_hitl_strategy(mock, hitl_state=hitl)

        state = SessionState()
        obs = _make_obs("sort blocks")

        def abort():
            import time
            time.sleep(0.1)
            hitl.set_action(HITLAction.ABORT)
        threading.Thread(target=abort, daemon=True).start()

        obs, state = strategy.process(obs, state)
        assert state.rewritten_instruction == "stop"


class TestHITLActions:
    """HITL mode: human actions are consumed and applied every step."""

    def _setup(self):
        from vlm_orchestrator.hitl import HITLState

        hitl = HITLState()
        mock = MockSubgoalVLM()
        strategy = _make_hitl_strategy(
            mock, hitl_state=hitl, check_interval=5, subgoal_timeout=999,
        )
        state = SessionState()
        obs = _make_obs("sort blocks")

        # Bootstrap: set up as if episode already started with subgoals.
        state.original_instruction = "sort blocks"
        state.subgoals = ["pick red", "pick blue", "pick green"]
        state.rewritten_instruction = "pick red"
        state.current_subgoal_idx = 0
        state.infer_count = 1  # past episode start
        return hitl, strategy, state, obs

    def test_rewrite_changes_instruction(self):
        from vlm_orchestrator.hitl import HITLAction

        hitl, strategy, state, obs = self._setup()
        hitl.set_action(
            HITLAction.REWRITE_INSTRUCTION,
            {"instruction": "carefully pick the red block"},
        )

        obs, state = strategy.process(obs, state)

        assert state.rewritten_instruction == "carefully pick the red block"
        assert obs["prompt"] == "carefully pick the red block"

    def test_done_advances_subgoal(self):
        from vlm_orchestrator.hitl import HITLAction

        hitl, strategy, state, obs = self._setup()
        hitl.set_action(HITLAction.SUBGOAL_DONE)

        obs, state = strategy.process(obs, state)

        assert state.current_subgoal_idx == 1
        assert state.rewritten_instruction == "pick blue"
        assert obs["prompt"] == "pick blue"

    def test_skip_advances_subgoal(self):
        from vlm_orchestrator.hitl import HITLAction

        hitl, strategy, state, obs = self._setup()
        hitl.set_action(HITLAction.SKIP)

        obs, state = strategy.process(obs, state)

        assert state.current_subgoal_idx == 1
        assert state.rewritten_instruction == "pick blue"

    def test_recovery_replaces_instruction(self):
        from vlm_orchestrator.hitl import HITLAction

        hitl, strategy, state, obs = self._setup()
        hitl.set_action(
            HITLAction.RECOVERY,
            {"instruction": "try grasping from the side"},
        )

        obs, state = strategy.process(obs, state)

        assert state.rewritten_instruction == "try grasping from the side"
        assert obs["prompt"] == "try grasping from the side"
        # subgoal just (re)started; allow off-by-one because the
        # test driver bumps episode_step *after* process() runs
        assert state.episode_step - state.step_at_subgoal_start <= 1  # counters reset

    def test_abort_sets_stop(self):
        from vlm_orchestrator.hitl import HITLAction

        hitl, strategy, state, obs = self._setup()
        hitl.set_action(HITLAction.ABORT)

        obs, state = strategy.process(obs, state)

        assert state.rewritten_instruction == "stop"
        assert obs["prompt"] == "stop"

    def test_no_vlm_check_in_hitl_mode(self):
        """VLM checks should NOT fire in HITL mode, even past check_interval."""
        hitl, strategy, state, obs = self._setup()

        # Run many steps — far past check_interval (5) and timeout (999).
        for i in range(30):
            state.infer_count = i + 1
            obs_step = _make_obs("sort blocks")
            obs_step, state = strategy.process(obs_step, state)

        # Instruction should NOT have changed (no VLM check advanced it).
        assert state.rewritten_instruction == "pick red"
        assert state.current_subgoal_idx == 0

    def test_replan_replaces_subgoals_midexecution(self):
        """START_SUBGOAL mid-execution replaces the full subgoal list."""
        from vlm_orchestrator.hitl import HITLAction

        hitl, strategy, state, obs = self._setup()

        # Advance to 2nd subgoal first.
        hitl.set_action(HITLAction.SUBGOAL_DONE)
        obs, state = strategy.process(obs, state)
        assert state.current_subgoal_idx == 1
        assert state.rewritten_instruction == "pick blue"

        # Now re-plan with a completely new list.
        hitl.set_action(
            HITLAction.START_SUBGOAL,
            {"instruction": "move cup to shelf\nwipe table"},
        )
        state.infer_count += 1
        # In unit tests the strategy is exercised without a real proxy.
        # Bump episode_step by 1 per chunk so that test ``check_interval``
        # values stay legible (a value of 2 means "fire after 2 chunks"
        # in the test, even though under the proxy they'd be sim steps).
        state.episode_step += 1
        obs = _make_obs("sort blocks")
        obs, state = strategy.process(obs, state)

        assert state.subgoals == ["move cup to shelf", "wipe table"]
        assert state.current_subgoal_idx == 0
        assert state.rewritten_instruction == "move cup to shelf"
        assert obs["prompt"] == "move cup to shelf"
        # subgoal just (re)started; allow off-by-one because the
        # test driver bumps episode_step *after* process() runs
        assert state.episode_step - state.step_at_subgoal_start <= 1


class TestHITLPause:
    """HITL mode: pause blocks the proxy loop, resume unblocks it."""

    def test_pause_then_resume(self):
        import threading
        from vlm_orchestrator.hitl import HITLAction, HITLState

        hitl = HITLState()
        mock = MockSubgoalVLM()
        strategy = _make_hitl_strategy(mock, hitl_state=hitl)

        state = SessionState()
        state.original_instruction = "sort blocks"
        state.subgoals = ["pick red"]
        state.rewritten_instruction = "pick red"
        state.infer_count = 1

        obs = _make_obs("sort blocks")

        # Click Pause now.
        hitl.set_action(HITLAction.PAUSE)

        # Resume after a short delay.
        def resume():
            import time
            time.sleep(0.2)
            hitl.set_action(HITLAction.RESUME)
        threading.Thread(target=resume, daemon=True).start()

        # process() should BLOCK here until resume.
        import time
        t0 = time.time()
        obs, state = strategy.process(obs, state)
        elapsed = time.time() - t0

        assert elapsed >= 0.15, "process() should have blocked while paused"
        assert not hitl.is_paused
        assert obs["prompt"] == "pick red"

    def test_failure_pauses_then_recovery_resumes(self):
        """Clicking Failure pauses; sending Recovery resumes with new instruction."""
        import threading
        from vlm_orchestrator.hitl import HITLAction, HITLState

        hitl = HITLState()
        mock = MockSubgoalVLM()
        strategy = _make_hitl_strategy(mock, hitl_state=hitl)

        state = SessionState()
        state.original_instruction = "sort blocks"
        state.subgoals = ["pick red"]
        state.rewritten_instruction = "pick red"
        state.infer_count = 1

        obs = _make_obs("sort blocks")

        # Click Failure now.
        hitl.set_action(HITLAction.FAILURE)

        # After a short delay, send recovery instruction.
        def recover():
            import time
            time.sleep(0.2)
            hitl.set_action(
                HITLAction.RECOVERY,
                {"instruction": "approach from the left side"},
            )
        threading.Thread(target=recover, daemon=True).start()

        obs, state = strategy.process(obs, state)

        assert state.rewritten_instruction == "approach from the left side"
        assert obs["prompt"] == "approach from the left side"
        assert not hitl.is_paused

    def test_image_pushed_every_chunk(self):
        """Images are pushed every process() call (every action chunk)."""
        from vlm_orchestrator.hitl import HITLState

        hitl = HITLState()
        mock = MockSubgoalVLM()
        strategy = _make_hitl_strategy(mock, hitl_state=hitl)

        state = SessionState()
        state.original_instruction = "sort blocks"
        state.subgoals = ["pick red"]
        state.rewritten_instruction = "pick red"
        state.infer_count = 1

        pushes = 0
        for step in range(20):
            state.infer_count = step + 1
            obs = _make_obs("sort blocks")
            obs, state = strategy.process(obs, state)
            _, updated = hitl.get_image()
            if updated:
                pushes += 1

        # HITL_IMAGE_INTERVAL=1 → every process() call pushes an image.
        assert pushes == 20, f"expected 20 image pushes in 20 steps, got {pushes}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
