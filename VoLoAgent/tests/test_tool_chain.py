# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``--mode tool_chain``.

Three layers:

1. **Prompt parser** — exhaustive coverage of valid and invalid VLM
   responses, including tool-arg validation.
2. **Strategy ``_on_step`` transitions** — feed canned VLM responses
   through a strategy with a mocked ``_vlm_call`` and assert the
   bookkeeping (counters, advance, replan, abort, hold).
3. **CLI dispatch** — that ``--mode tool_chain`` constructs the right
   strategy and rejects incompatible flags.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vlm_orchestrator.strategies.base import (
    SessionState, StrategyContext,
)
from vlm_orchestrator.strategies.tool_chain import (
    ToolChainConfig, ToolChainStrategy,
)
from vlm_orchestrator.strategies.tool_chain import (
    ToolChainDecision,
    build_tool_chain_user_content,
    parse_tool_chain_response,
)
from vlm_orchestrator.vlm import PassthroughVLM


# ======================================================================
# Layer 1 — prompt parser
# ======================================================================


class TestParseValid:
    def test_noop_advance(self):
        d = parse_tool_chain_response(json.dumps({
            "subgoal_action": "advance",
            "tool": "noop",
            "args": {},
            "reason": "subgoal 0 is done",
        }))
        assert d.subgoal_action == "advance"
        assert d.tool == "noop"
        assert d.args == {}
        assert d.reason == "subgoal 0 is done"

    def test_place_continue(self):
        d = parse_tool_chain_response(json.dumps({
            "subgoal_action": "continue",
            "tool": "place",
            "args": {
                "destination": "in the blue bowl",
                "held_object_hint": "red block",
            },
            "reason": "drop the block",
        }))
        assert d.subgoal_action == "continue"
        assert d.tool == "place"
        assert d.args["destination"] == "in the blue bowl"

    def test_noop_replan(self):
        d = parse_tool_chain_response(json.dumps({
            "subgoal_action": "replan",
            "tool": "noop",
            "args": {},
            "reason": "scene shifted",
        }))
        assert d.subgoal_action == "replan"
        assert d.tool == "noop"
        assert d.args == {}

    def test_abort_noop(self):
        d = parse_tool_chain_response(json.dumps({
            "subgoal_action": "abort",
            "tool": "noop",
            "args": {},
            "reason": "unrecoverable",
        }))
        assert d.subgoal_action == "abort"
        assert d.tool == "noop"

    def test_place_freeform_spatial_destination(self):
        # Spatial language flows through destination; no `relation` field.
        d = parse_tool_chain_response(json.dumps({
            "subgoal_action": "continue",
            "tool": "place",
            "args": {
                "destination": "on top of the table",
                "held_object_hint": "block",
            },
            "reason": "put on top",
        }))
        assert d.args["destination"] == "on top of the table"
        assert "relation" not in d.args or d.args.get("relation") is None


class TestParseInvalid:
    def test_not_json(self):
        with pytest.raises(ValueError, match="parse"):
            parse_tool_chain_response("hello world")

    def test_missing_subgoal_action(self):
        with pytest.raises(ValueError, match="missing"):
            parse_tool_chain_response(json.dumps({
                "tool": "grasp",
                "args": {"target": "x"},
            }))

    def test_missing_tool(self):
        with pytest.raises(ValueError, match="missing"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "continue",
                "args": {},
            }))

    def test_invalid_subgoal_action(self):
        with pytest.raises(ValueError, match="invalid subgoal_action"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "stop",
                "tool": "noop",
                "args": {},
            }))

    def test_invalid_tool(self):
        with pytest.raises(ValueError, match="invalid tool"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "continue",
                "tool": "wave_hello",
                "args": {},
            }))

    def test_grasp_without_target(self):
        with pytest.raises(ValueError, match="target"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "continue",
                "tool": "grasp",
                "args": {},
            }))

    def test_grasp_empty_target(self):
        with pytest.raises(ValueError, match="target"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "continue",
                "tool": "grasp",
                "args": {"target": "   "},
            }))

    def test_place_without_destination(self):
        with pytest.raises(ValueError, match="destination"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "continue",
                "tool": "place",
                "args": {"held_object_hint": "block"},
            }))

    def test_advance_with_tool_rejected(self):
        with pytest.raises(ValueError, match="must be paired"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "advance",
                "tool": "grasp",
                "args": {"target": "red block"},
                "reason": "subgoal 0 wants the red block",
            }))

    def test_place_extra_relation_is_ignored(self):
        # If the model still emits `relation`, parse succeeds and the
        # field is left in args untouched — downstream activation
        # ignores it.  Spatial meaning is carried by `destination`.
        d = parse_tool_chain_response(json.dumps({
            "subgoal_action": "continue",
            "tool": "place",
            "args": {
                "destination": "next to the green lemon",
                "relation": "next_to",
                "held_object_hint": "orange",
            },
            "reason": "consolidate near the lemon",
        }))
        assert d.tool == "place"
        assert d.args["destination"] == "next to the green lemon"

    def test_args_not_object(self):
        with pytest.raises(ValueError, match="JSON object"):
            parse_tool_chain_response(json.dumps({
                "subgoal_action": "continue",
                "tool": "noop",
                "args": [1, 2, 3],
            }))


# ======================================================================
# Layer 2 — user-content builder
# ======================================================================


def _img(h=32, w=32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestBuildUserContent:
    def test_first_cycle_no_last_tool(self):
        content = build_tool_chain_user_content(
            task="put cube in bowl",
            subgoals=["pick cube", "place cube in bowl"],
            current_idx=0,
            before_image=_img(),
            now_image=_img(),
            last_tool=None,
            last_args=None,
            last_status=None,
            last_reason=None,
        )
        text = content[0]["text"]
        assert "Task: put cube in bowl" in text
        assert "-> [0] pick cube" in text
        assert "  [1] place cube in bowl" in text
        assert "first cycle" in text

    def test_after_grasp(self):
        content = build_tool_chain_user_content(
            task="put cube",
            subgoals=["pick cube", "place"],
            current_idx=1,
            before_image=_img(),
            now_image=_img(),
            last_tool="grasp",
            last_args={"target": "cube"},
            last_status="DONE",
            last_reason="grasped",
        )
        text = content[0]["text"]
        assert "Last tool:" in text
        assert "grasp(target='cube')" in text
        assert "DONE" in text
        assert "-> [1] place" in text

    def test_no_before_image(self):
        # Should still produce content with the now image and no crash.
        content = build_tool_chain_user_content(
            task="x", subgoals=["y"], current_idx=0,
            before_image=None, now_image=_img(),
            last_tool=None, last_args=None,
            last_status=None, last_reason=None,
        )
        # Text + 1 image (no before)
        assert sum(c["type"] == "image_url" for c in content) == 1


# ======================================================================
# Layer 3 — strategy transitions
# ======================================================================


def _ctx() -> StrategyContext:
    return StrategyContext(vlm=PassthroughVLM())


def _state() -> SessionState:
    s = SessionState()
    s.episode_id = 1
    s.original_instruction = "put cube in bowl"
    s.subgoals = ["pick cube", "place cube in bowl"]
    s.current_subgoal_idx = 0
    s.subgoals_ordered = True
    s.initial_image = _img()
    s.tool_chain_active = True
    return s


def _config(**overrides) -> ToolChainConfig:
    base = dict(
        max_tools_per_subgoal=5,
        max_tools_per_episode=30,
    )
    base.update(overrides)
    return ToolChainConfig(**base)


def _obs() -> dict:
    return {
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.array([0.0]),
        "observation/ee_pos": np.array([0.4, 0.0, 0.5]),
        "observation/exterior_image_1_left": _img(480, 640),
    }


def _make_strategy(config: ToolChainConfig | None = None) -> ToolChainStrategy:
    cfg = config or _config()
    strat = ToolChainStrategy(_ctx(), cfg)
    # Force the parent's get_vlm_image to return our test image.
    strat.ctx.get_vlm_image = lambda obs: _img()
    return strat


def _patch_vlm(strategy: ToolChainStrategy, response_dict: dict) -> MagicMock:
    """Patch ``_vlm_call`` to return a JSON-encoded canned response."""
    raw = json.dumps(response_dict)
    mock = MagicMock(return_value=raw)
    strategy._vlm_call = mock
    return mock


class TestStepTransitions:
    def test_first_cycle_picks_grasp(self):
        s = _make_strategy()
        state = _state()
        _patch_vlm(s, {
            "subgoal_action": "continue",
            "tool": "grasp",
            "args": {"target": "cube"},
            "reason": "first pick",
        })
        with patch(
            "vlm_orchestrator.strategies.tool_chain"
            ".ToolChainStrategy._activate_grasp_for_tool_chain"
        ) as activate:
            obs, state = s._on_step(_obs(), state)
        assert activate.called
        assert state.tool_chain_subgoal_calls == 1
        assert state.tool_chain_tool_calls == 1
        assert state.tool_chain_pending_tool == "grasp"

    def test_advance_to_next_subgoal(self):
        s = _make_strategy()
        state = _state()
        _patch_vlm(s, {
            "subgoal_action": "advance",
            "tool": "noop",
            "args": {},
            "reason": "ok",
        })
        with patch.object(s, "_activate_place_for_tool_chain") as activate:
            s._on_step(_obs(), state)
        assert state.current_subgoal_idx == 1
        assert not activate.called
        assert state.tool_chain_subgoal_calls == 0
        assert state.tool_chain_tool_calls == 0

    def test_advance_past_last_subgoal_marks_done(self):
        s = _make_strategy()
        state = _state()
        state.current_subgoal_idx = 1   # already on last subgoal
        _patch_vlm(s, {
            "subgoal_action": "advance",
            "tool": "noop",
            "args": {},
            "reason": "all done",
        })
        s._on_step(_obs(), state)
        assert state.tool_chain_task_done is True
        assert state.tool_chain_tool_calls == 0  # noop not counted

    def test_abort_marks_aborted(self):
        s = _make_strategy()
        state = _state()
        _patch_vlm(s, {
            "subgoal_action": "abort",
            "tool": "noop",
            "args": {},
            "reason": "unrecoverable",
        })
        s._on_step(_obs(), state)
        assert state.tool_chain_aborted is True

    def test_noop_does_not_increment_counters(self):
        s = _make_strategy()
        state = _state()
        _patch_vlm(s, {
            "subgoal_action": "continue",
            "tool": "noop",
            "args": {},
            "reason": "wait",
        })
        s._on_step(_obs(), state)
        assert state.tool_chain_subgoal_calls == 0
        assert state.tool_chain_tool_calls == 0

    def test_replan_calls_recycle(self):
        s = _make_strategy()
        state = _state()
        _patch_vlm(s, {
            "subgoal_action": "replan",
            "tool": "noop",
            "args": {},
            "reason": "scene changed",
        })
        with patch.object(s, "_recycle", return_value=True) as recycle:
            s._on_step(_obs(), state)
        assert recycle.called

    def test_per_subgoal_cap_forces_replan(self):
        s = _make_strategy(_config(max_tools_per_subgoal=2))
        state = _state()
        state.tool_chain_subgoal_calls = 2     # at the cap
        # Patch _vlm_call to a Mock so we can assert it wasn't called.
        s._vlm_call = MagicMock(return_value="should not be called")
        with patch.object(s, "_recycle", return_value=True) as recycle:
            s._on_step(_obs(), state)
        assert recycle.called
        assert not s._vlm_call.called

    def test_per_episode_cap_marks_aborted(self):
        s = _make_strategy(_config(max_tools_per_episode=3))
        state = _state()
        state.tool_chain_tool_calls = 3        # at the cap
        s._on_step(_obs(), state)
        assert state.tool_chain_aborted is True

    def test_malformed_vlm_response_does_not_crash(self):
        s = _make_strategy()
        state = _state()
        s._vlm_call = MagicMock(return_value="not valid json {{{ ")
        # Should log a parse error and return without activating anything.
        s._on_step(_obs(), state)
        assert state.tool_chain_subgoal_calls == 0
        assert state.tool_chain_tool_calls == 0
        # Loud parse-error event in the log.
        types = [e.get("type") for e in state.log_entries]
        assert "tool_chain_parse_error" in types

    def test_vlm_exception_does_not_crash(self):
        s = _make_strategy()
        state = _state()
        s._vlm_call = MagicMock(side_effect=RuntimeError("network down"))
        s._on_step(_obs(), state)
        types = [e.get("type") for e in state.log_entries]
        assert "tool_chain_vlm_error" in types

    def test_done_state_no_more_vlm_calls(self):
        s = _make_strategy()
        state = _state()
        state.tool_chain_task_done = True
        s._vlm_call = MagicMock()
        s._on_step(_obs(), state)
        assert not s._vlm_call.called

    def test_aborted_state_no_more_vlm_calls(self):
        s = _make_strategy()
        state = _state()
        state.tool_chain_aborted = True
        s._vlm_call = MagicMock()
        s._on_step(_obs(), state)
        assert not s._vlm_call.called


# ======================================================================
# Layer 3b — config + decision dataclass
# ======================================================================


class TestDecision:
    def test_decision_validation(self):
        with pytest.raises(ValueError, match="invalid subgoal_action"):
            ToolChainDecision(
                subgoal_action="weird",  # type: ignore[arg-type]
                tool="grasp",
                args={"target": "x"},
            )
        with pytest.raises(ValueError, match="invalid tool"):
            ToolChainDecision(
                subgoal_action="continue",
                tool="dance",  # type: ignore[arg-type]
            )

    def test_is_no_action(self):
        d = ToolChainDecision(subgoal_action="continue", tool="noop")
        assert d.is_no_action()
        d = ToolChainDecision(
            subgoal_action="continue", tool="grasp",
            args={"target": "x"},
        )
        assert not d.is_no_action()


class TestConfig:
    def test_defaults(self):
        c = ToolChainConfig()
        assert c.max_tools_per_subgoal == 5
        assert c.max_tools_per_episode == 30
        assert c.vlm_model == "YOUR_VLM_MODEL"


# ======================================================================
# Layer 4 — CLI dispatch (smoke)
# ======================================================================


class TestCLI:
    def test_mode_in_choices(self):
        # The cli.main wraps argparse setup — verify --help mentions tool_chain.
        import subprocess
        out = subprocess.run(
            ["python", "-m", "vlm_orchestrator.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        assert "tool_chain" in out
        assert "--tool-chain-max-tools-per-subgoal" in out
        assert "--tool-chain-max-tools-per-episode" in out

    def test_failure_monitor_rejected(self):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "vlm_orchestrator.cli",
             "--mode", "tool_chain",
             "--failure-monitor", "vlm"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "tool_chain" in result.stderr or "subgoal" in result.stderr

    def test_recovery_mode_rejected(self):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "vlm_orchestrator.cli",
             "--mode", "tool_chain",
             "--recovery-mode", "replan"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert (
            "tool_chain does not use --recovery-mode" in result.stderr
            or "tool_chain" in result.stderr
        )
