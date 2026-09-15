# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subgoal strategy — VLM decomposes tasks and monitors progress.

Three ``check_mode`` options control subgoal transitions:

* ``"vlm"``    — VLM progress check every *check_interval* chunks.
* ``"timer"``  — pure timer cycling, no VLM calls after decomposition.
* ``"hybrid"`` — VLM-gated for ordered tasks, timer for unordered.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .base import OrchestrationStrategy, SessionState, StrategyContext
from .subgoal_base import (
    CHECK_DONE_SHARED_PROMPT,
    CHECK_DONE_SHARED_PROMPT_VLABENCH,
    DECOMPOSE_SHARED_PROMPT,
    DECOMPOSE_SHARED_PROMPT_VLABENCH,
    MAX_RECYCLES,
    SubgoalBaseStrategy,
    _with_vla_front_cam_note,
    parse_json,
)


def _normalize_subgoals(raw_subgoals: list) -> list[str]:
    """Extract plain instruction strings from a list of subgoal entries.

    The VLM may return subgoals as plain strings::

        ["Pick up the red block and place it in the bin", ...]

    or as structured dicts (especially from the recycle prompt)::

        [{"instruction": "Pick up ...", "target_object": "red block"}, ...]

    This helper normalises both to a flat list of instruction strings
    so that ``state.subgoals`` always contains policy-ready text.
    """
    out: list[str] = []
    for sg in raw_subgoals:
        if isinstance(sg, dict):
            # Prefer 'instruction' key; fall back to full dict repr
            out.append(sg.get("instruction", str(sg)))
        else:
            out.append(str(sg))
    return out

logger = logging.getLogger(__name__)

# ======================================================================
# Prompts
# ======================================================================

_DECOMPOSE_FORMAT_SUFFIX = """

Output ONLY valid JSON (no markdown, no explanation):
{"subgoals": ["first subgoal", "second subgoal", ...], "ordered": true}\
"""

DECOMPOSE_SYSTEM_PROMPT = DECOMPOSE_SHARED_PROMPT + _DECOMPOSE_FORMAT_SUFFIX
DECOMPOSE_SYSTEM_PROMPT_VLABENCH = (
    DECOMPOSE_SHARED_PROMPT_VLABENCH + _DECOMPOSE_FORMAT_SUFFIX
)

CHECK_SYSTEM_PROMPT = """\
You are monitoring a robot arm's progress on a manipulation task.

You receive two sets of images:
- **BEFORE** images: the scene at the start of this episode, before the \
  robot acted.
- **NOW** images: the current scene.

Compare them and decide what the robot should do next:

1. "continue" — The current step is NOT yet complete.
2. "refine"   — The current step is NOT complete, but the instruction \
   should be updated to better guide the robot. Provide the updated text.
3. "next"     — The current step IS complete. The target object has \
   moved from its BEFORE position to its destination.

IMPORTANT: Only answer "next" if you see clear visual evidence that the \
step is done. When in doubt, say "continue".

Output ONLY valid JSON:
{"action": "continue"}
or {"action": "refine", "instruction": "updated instruction"}
or {"action": "next"}\
"""

CHECK_DONE_SYSTEM_PROMPT = CHECK_DONE_SHARED_PROMPT
CHECK_DONE_SYSTEM_PROMPT_VLABENCH = CHECK_DONE_SHARED_PROMPT_VLABENCH


# ======================================================================
# Config
# ======================================================================


@dataclass
class SubgoalConfig:
    """Configuration for the subgoal strategy."""

    check_mode: str = "vlm"
    """``"vlm"`` | ``"timer"`` | ``"hybrid"``."""

    # Sim-step units (was chunks); ratios approximate the previous
    # behavior on pi05 (chunk=8) and remain meaningful for any VLA
    # behind a shim that uses pi0_family with horizon=8.
    check_interval: int = 40
    # Effectively-disabled timeout — advancement is gated on VLM/GT
    # completion checks instead.  Time-based advancement was hurting
    # repetition tasks (forcing the orchestrator to race through
    # subgoals before the VLA could finish each rep) and short tasks
    # alike (preempting the VLA on tasks it would have completed
    # autonomously).  Set explicitly per-run to re-enable.
    subgoal_timeout: int = 9999

    vlm_model: str = "YOUR_VLM_MODEL"
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 512
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None


# ======================================================================
# Strategy
# ======================================================================


class SubgoalStrategy(SubgoalBaseStrategy):
    """Decompose instructions into subgoals with periodic VLM monitoring.

    Optionally wraps an *initial strategy* (rewrite / adaptive) that runs
    before decomposition.
    """

    def __init__(
        self,
        ctx: StrategyContext,
        config: SubgoalConfig,
        initial_strategy: OrchestrationStrategy | None = None,
        failure_monitor: str | None = None,
        recovery_mode: str = "template",
        hitl_state=None,
        grasp_seg_mode: str = "gdino_sam2",
        place_seg_mode: str | None = None,
        env_mode: str = "robolab",
        collect_trajectories: str | None = None,
        use_front_camera: bool = False,
        gt_failure_types: set[str] | None = None,
        grasp_topdown_threshold: float | None = None,
        max_grasp_attempts: int | None = None,
        motion_planner: str = "linear",
        stack_mode_enabled: bool = False,
    ):
        super().__init__(
            ctx,
            initial_strategy=initial_strategy,
            failure_monitor=failure_monitor,
            recovery_mode=recovery_mode,
            hitl_state=hitl_state,
            grasp_seg_mode=grasp_seg_mode,
            place_seg_mode=place_seg_mode,
            env_mode=env_mode,
            collect_trajectories=collect_trajectories,
            use_front_camera=use_front_camera,
            gt_failure_types=gt_failure_types,
            grasp_topdown_threshold=grasp_topdown_threshold,
            max_grasp_attempts=max_grasp_attempts,
            motion_planner=motion_planner,
            stack_mode_enabled=stack_mode_enabled,
        )
        self.config = config

    # ------------------------------------------------------------------
    # Hooks: check-mode helpers
    # ------------------------------------------------------------------

    def _should_check(self, state: SessionState) -> bool:
        if self.config.check_mode == "timer":
            return False
        if self.config.check_mode == "hybrid" and not state.subgoals_ordered:
            return False
        return True

    def _get_timeout(self, state: SessionState) -> int:
        return self.config.subgoal_timeout

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def _decompose_and_setup(self, obs, state, instruction, image):
        extra = self.ctx.get_vlm_extra_images(obs)
        user_content = self._build_image_message(
            f'Scene instruction: "{instruction}"', image, extra,
        )

        decompose_prompt = (
            DECOMPOSE_SYSTEM_PROMPT_VLABENCH
            if self.ctx.prompt_style == "vlabench"
            else DECOMPOSE_SYSTEM_PROMPT
        )
        decompose_prompt = _with_vla_front_cam_note(
            decompose_prompt,
            use_front_camera=bool(
                getattr(self.ctx, "front_image_key", None)
            ),
        )
        t0 = time.time()
        try:
            raw = self._vlm_call(decompose_prompt, user_content)
            elapsed = time.time() - t0
        except Exception as e:
            logger.warning(f"  Decomposition VLM call failed: {e}")
            state.subgoals = [instruction]
            state.subgoals_ordered = False
            state.rewritten_instruction = instruction
            state.log({
                "type": "decompose",
                "instruction": instruction,
                "subgoals": [instruction],
                "ordered": False,
                "fallback": "vlm_call_failed",
                "error": str(e),
            })
            return

        logger.info(f"  Decompose raw ({elapsed:.1f}s): {raw[:500]}")

        parse_error: str | None = None
        try:
            data = parse_json(raw)
            subgoals = data.get("subgoals", [])
            if not isinstance(subgoals, list) or not subgoals:
                raise ValueError("empty or invalid subgoals list")
            subgoals = _normalize_subgoals(subgoals)
            ordered = bool(data.get("ordered", False))
        except (ValueError, KeyError) as e:
            logger.warning(f"  Cannot parse decomposition: {e}")
            subgoals = [instruction]
            ordered = False
            parse_error = str(e)

        state.subgoals = subgoals
        state.subgoals_ordered = ordered

        log_entry = {
            "type": "decompose",
            "instruction": instruction,
            "subgoals": list(subgoals),
            "ordered": ordered,
            "vlm_latency_s": round(elapsed, 2),
        }
        if parse_error is not None:
            log_entry["fallback"] = "parse_failed"
            log_entry["error"] = parse_error
        state.log(log_entry)

        if subgoals:
            state.rewritten_instruction = subgoals[0]
            obs = self.ctx.set_prompt(obs, subgoals[0])
            tag = "ORDERED" if ordered else "UNORDERED"
            logger.info(
                f"Episode {state.episode_id}: "
                f"{len(subgoals)} subgoal(s) [{tag}] "
                f"for: \"{instruction}\""
            )
            for i, sg in enumerate(subgoals):
                logger.info(f"  [{i + 1}/{len(subgoals)}] {sg}")
        else:
            state.rewritten_instruction = instruction

    # ------------------------------------------------------------------
    # VLM check: routes to binary or full
    # ------------------------------------------------------------------

    def _check_subgoal(self, obs, state, image):
        if self.config.check_mode == "hybrid":
            return self._check_binary(obs, state, image)
        return self._check_full(obs, state, image)

    def _check_binary(self, obs, state, image):
        """Simplified binary check: done or not done."""
        current_sg = state.subgoals[state.current_subgoal_idx]
        total = len(state.subgoals)

        text = f'Step: "{current_sg}"\nIs this step done?'
        extra = self.ctx.get_vlm_extra_images(obs)
        primary_label, extra_labels = self.ctx.vlm_camera_labels
        user_content = self._build_check_message(
            text, state.initial_image, image, extra,
            initial_extra_images=state.initial_extra_images,
            primary_label=primary_label,
            extra_labels=extra_labels,
        )

        check_prompt = (
            CHECK_DONE_SYSTEM_PROMPT_VLABENCH
            if self.ctx.prompt_style == "vlabench"
            else CHECK_DONE_SYSTEM_PROMPT
        )
        t0 = time.time()
        try:
            raw = self._vlm_call(check_prompt, user_content)
            elapsed = time.time() - t0
        except Exception as e:
            logger.warning(f"  Check VLM call failed: {e}; continuing")
            return obs, state

        logger.info(f"  Check-done raw ({elapsed:.1f}s): {raw[:500]}")

        done = False
        try:
            data = parse_json(raw)
            done = bool(data.get("done", False))
        except ValueError:
            pass

        if done and state.current_subgoal_idx < total - 1:
            self._advance_subgoal(obs, state)
            state.log({
                "type": "check",
                "action": "next",
                "subgoal_idx": state.current_subgoal_idx,
                "instruction": state.rewritten_instruction,
                "vlm_latency_s": round(elapsed, 2),
            })
        elif done:
            if self._recycle_count < MAX_RECYCLES:
                if self._recycle(obs, state, image):
                    obs = self.ctx.set_prompt(
                        obs, state.rewritten_instruction
                    )
                    state.log({
                        "type": "check",
                        "action": "next",
                        "subgoal_idx": state.current_subgoal_idx,
                        "instruction": state.rewritten_instruction,
                        "vlm_latency_s": round(elapsed, 2),
                    })
                    self._store_vlm_check(state, done=True, action="next")
                    return obs, state
            logger.info(
                "  Check-done: DONE on last subgoal, continuing execution"
            )
            state.log({
                "type": "check",
                "action": "next",
                "subgoal_idx": state.current_subgoal_idx,
                "instruction": state.rewritten_instruction,
                "vlm_latency_s": round(elapsed, 2),
            })
        else:
            logger.debug("  Check-done: NOT DONE")
            state.log({
                "type": "check",
                "action": "continue",
                "subgoal_idx": state.current_subgoal_idx,
                "vlm_latency_s": round(elapsed, 2),
            })

        self._store_vlm_check(
            state,
            done=done,
            action="next" if done else "continue",
        )
        return obs, state

    def _check_full(self, obs, state, image):
        """Full 3-way check: continue / refine / next."""
        current_sg = state.subgoals[state.current_subgoal_idx]
        total = len(state.subgoals)
        idx = state.current_subgoal_idx + 1

        remaining = state.subgoals[state.current_subgoal_idx + 1:]
        if remaining:
            remaining_str = "Remaining steps after this: " + "; ".join(
                f"[{i}] {s}"
                for i, s in enumerate(remaining, start=idx + 1)
            )
        else:
            remaining_str = "This is the last step."

        text = (
            f'Overall task: "{state.original_instruction}"\n'
            f'Current step ({idx}/{total}): "{current_sg}"\n'
            f"{remaining_str}\n\n"
            f"Look at the scene and decide: continue, refine, or next?"
        )

        extra = self.ctx.get_vlm_extra_images(obs)
        primary_label, extra_labels = self.ctx.vlm_camera_labels
        user_content = self._build_check_message(
            text, state.initial_image, image, extra,
            initial_extra_images=state.initial_extra_images,
            primary_label=primary_label,
            extra_labels=extra_labels,
        )

        check_prompt = _with_vla_front_cam_note(
            CHECK_SYSTEM_PROMPT,
            use_front_camera=bool(
                getattr(self.ctx, "front_image_key", None)
            ),
        )
        t0 = time.time()
        try:
            raw = self._vlm_call(check_prompt, user_content)
            elapsed = time.time() - t0
        except Exception as e:
            logger.warning(f"  Check VLM call failed: {e}; continuing")
            return obs, state

        logger.info(f"  Check raw ({elapsed:.1f}s): {raw[:500]}")

        try:
            data = parse_json(raw)
            action = data.get("action", "continue")
        except ValueError:
            action = "continue"

        if action == "next":
            if state.current_subgoal_idx < total - 1:
                self._advance_subgoal(obs, state)
            else:
                if self._recycle_count < MAX_RECYCLES:
                    if self._recycle(obs, state, image):
                        obs = self.ctx.set_prompt(
                            obs, state.rewritten_instruction
                        )
                        self._store_vlm_check(
                            state, done=True, action="next",
                        )
                        state.log({
                            "type": "check",
                            "action": "next",
                            "subgoal_idx": state.current_subgoal_idx,
                            "instruction": state.rewritten_instruction,
                            "vlm_latency_s": round(elapsed, 2),
                        })
                        return obs, state
                logger.info(
                    "  Check: NEXT on last subgoal, continuing execution"
                )

            state.log({
                "type": "check",
                "action": "next",
                "subgoal_idx": state.current_subgoal_idx,
                "instruction": state.rewritten_instruction,
                "vlm_latency_s": round(elapsed, 2),
            })

        elif action == "refine":
            new_instruction = data.get("instruction", current_sg)
            state.rewritten_instruction = new_instruction
            obs = self.ctx.set_prompt(obs, new_instruction)
            state.subgoals[state.current_subgoal_idx] = new_instruction
            logger.info(f'  Check: REFINE -> "{new_instruction}"')
            state.log({
                "type": "check",
                "action": "refine",
                "subgoal_idx": state.current_subgoal_idx,
                "instruction": new_instruction,
                "vlm_latency_s": round(elapsed, 2),
            })

        else:
            logger.debug("  Check: CONTINUE")
            state.log({
                "type": "check",
                "action": "continue",
                "subgoal_idx": state.current_subgoal_idx,
                "vlm_latency_s": round(elapsed, 2),
            })

        self._store_vlm_check(
            state,
            done=action == "next",
            action=action,
        )
        return obs, state

    # ------------------------------------------------------------------
    # Recycling
    # ------------------------------------------------------------------

    def _apply_recycle_subgoals(self, obs, state, data):
        subgoals = _normalize_subgoals(data["subgoals"])

        state.subgoals = subgoals
        state.current_subgoal_idx = 0
        state.rewritten_instruction = subgoals[0]
        # Detection reset, GT config, flush, and subclass hooks are
        # handled by _recycle() after this method returns True.

        logger.info(f"  Recycle: {len(subgoals)} new subgoals")
        for i, sg in enumerate(subgoals):
            logger.info(f"    [{i + 1}] {sg}")

        state.log({
            "type": "recycle",
            "recycle_count": self._recycle_count,
            "new_subgoals": subgoals,
        })
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _store_vlm_check(
        state: SessionState, *, done: bool, action: str,
    ) -> None:
        """Store VLM check result for annotated video."""
        state.vlm_check_result = {"done": done, "action": action}
        state.vlm_check_infer_step = state.infer_count
