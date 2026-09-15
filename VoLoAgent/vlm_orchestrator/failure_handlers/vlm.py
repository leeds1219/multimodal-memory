# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM-based failure detection and recovery handler.

Periodically sends BEFORE + NOW images to a VLM and asks it to assess
the current subgoal status and decide the next action.  Replaces both
GT-based failure detection and the old VLM subgoal completion check
with a single unified call.

The VLM outputs two independent fields:

* **status** — diagnosis (``complete`` / ``failure`` / ``in_progress``)
* **action** — decision (``next`` / ``replan`` / ``continue`` /
  ``grasp_tool``)

The available actions are configured by ``--recovery-mode``:

* ``replan``       → next, continue, replan
* ``replan_grasp`` → next, continue, replan, grasp_tool
* ``grasp``        → next, continue, grasp_tool
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from vlm_orchestrator.vlm import parse_json

from .base import (
    ACTION_CONTINUE,
    ACTION_GRASP,
    ACTION_NEXT,
    ACTION_PLACE,
    ACTION_REPLAN,
    FailureHandler,
    HandlerResult,
    STATUS_COMPLETE,
    STATUS_FAILURE,
    STATUS_IN_PROGRESS,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Available action sets per recovery mode
# ======================================================================

_ACTION_SETS: dict[str, set[str]] = {
    "replan": {ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN},
    "replan_grasp": {
        ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN, ACTION_GRASP,
    },
    "grasp": {ACTION_NEXT, ACTION_CONTINUE, ACTION_GRASP},
    # Place-only analogs of the grasp tokens.
    "place": {ACTION_NEXT, ACTION_CONTINUE, ACTION_PLACE},
    "replan_place": {
        ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN, ACTION_PLACE,
    },
    # Combined: VLM may pick either tool based on failure context.
    "tools": {ACTION_NEXT, ACTION_CONTINUE, ACTION_GRASP, ACTION_PLACE},
    "replan_tools": {
        ACTION_NEXT, ACTION_CONTINUE, ACTION_REPLAN,
        ACTION_GRASP, ACTION_PLACE,
    },
}


# ======================================================================
# Prompt
# ======================================================================

def _build_system_prompt(
    available_actions: set[str],
    *,
    use_front_camera: bool = False,
    stack_mode_enabled: bool = False,
) -> str:
    """Build the system prompt with the available action vocabulary.

    When ``use_front_camera`` is True, the prompt warns the VLM that
    the front (egocentric_mirrored) camera view is left-right flipped
    relative to the robot's actual perspective, so the VLM should use
    the robot's frame when emitting any directional references.
    """

    action_descriptions = []
    if ACTION_NEXT in available_actions:
        action_descriptions.append(
            '- "next" — this subgoal is COMPLETE. The target object has '
            "moved to its destination. Advance to the next subgoal."
        )
    if ACTION_CONTINUE in available_actions:
        action_descriptions.append(
            '- "continue" — the robot is making progress or the subgoal '
            "is not yet complete. Keep working with the current instruction."
        )
    if ACTION_REPLAN in available_actions:
        action_descriptions.append(
            '- "replan" — something went wrong (object dropped, stuck, '
            "wrong object picked, no visible progress) OR the current "
            "plan is not working. Replan the remaining subtasks from "
            "the current scene state."
        )
    if ACTION_GRASP in available_actions:
        action_descriptions.append(
            '- "grasp_tool" — fire only for WRONG-OBJECT recovery: the '
            "gripper holds (or is closing on) an object whose identity "
            "differs from the subgoal's target. Do NOT fire while the "
            "robot is still attempting the correct object — that is "
            'in-progress execution; emit "continue". When firing, set '
            '"grasp_target" to the noun phrase of the correct object.'
        )
    if ACTION_PLACE in available_actions:
        action_descriptions.append(
            '- "place_tool" — fire only for WRONG-DESTINATION recovery: '
            "the held object is heading to or already released at a "
            "destination different from the subgoal's destination. Do "
            "NOT fire while the robot is en route to the correct "
            'destination — that is in-progress execution; emit '
            '"continue". When firing, set "place_destination" to a '
            "FREEFORM destination phrase. PHRASING RULES:\n"
            "  1. Inside a container → \"in <container>\" "
            '(e.g. "in the white bowl").\n'
            "  2. Beside/near an anchor → prepend \"empty space\" "
            '(e.g. "empty space next to the green lemon").\n'
            "  3. Open table → \"empty space on the table\".\n"
            "  4. On a surface object → \"on <surface>\" "
            '(e.g. "on the wire rack shelf").\n'
            "Optionally also set \"place_held_object\" with the noun "
            "phrase of the object in the gripper.\n"
            "LIMITATION: place_tool is unreliable for high containers "
            '(e.g. a shelf, a high vase). For those destinations prefer '
            '"continue" (let the VLA finish) or "replan" instead of '
            "firing place_tool."
            + (
                "\n"
                "STACK CONTROL: place_tool accepts an optional boolean "
                '"place_stack" arg.\n'
                '  - "place_stack": false (DEFAULT) — plain top-down '
                "release. Use for ordinary pick-and-place: dropping an "
                "object into a bowl/bin, onto an open surface, or beside "
                "another object.\n"
                '  - "place_stack": true — keep the held object\'s current '
                "orientation at release instead of forcing top-down. Use "
                "ONLY when setting an object ON TOP of another in a way "
                "that preserves how it is held (stacking a block on a "
                "block, nesting a lid on a container). When in doubt, "
                'prefer false.'
                if stack_mode_enabled
                else ""
            )
        )

    actions_block = "\n".join(action_descriptions)

    front_cam_block = ""
    if use_front_camera:
        front_cam_block = """\

NOTE on camera orientation: the "Front camera" view is BOTH L/R \
and front/behind FLIPPED from the robot's perspective. \
image-LEFT↔robot-RIGHT, image-TOP↔robot-BEHIND.

To avoid frame confusion, describe targets by VISUAL FEATURES \
(color, type, proximity to landmarks) in every output field — \
NOT by left/right/front/behind. Example: "the grey container next \
to the red block" instead of "the right bin". If the subgoal uses a \
direction, first apply the flip to find which object it means in the \
image, then re-describe by visual features."""

    return f"""\
You are monitoring a robot arm executing a tabletop manipulation task.

You receive two sets of images:
- **BEFORE** images: the scene at the START of the episode (for reference).
- **NOW** images: the current scene.
{front_cam_block}

You also receive the overall task instruction, the current subgoal, and \
any remaining subgoals after the current one.

HOW TO REASON:
- Use the BEFORE images and overall task to understand the goal.
- Use the NOW images, current subgoal, and remaining subgoals to \
  understand the current state and what still needs to happen.
- Decide whether the robot can keep executing the current subgoal \
  from the NOW scene. If yes → "in_progress" / "continue". \
  If the current subgoal is already done → "complete" / "next". \
  If something actively prevents progress (wrong object in gripper, \
  object dropped, robot stuck) → "failure".
- The subgoal list may have been replanned mid-episode. The current \
  subgoal and remaining subgoals already reflect what needs to be \
  done from the current state — do not treat them as stale.

Your job:
1. **Assess status** — look at the NOW images and determine the status \
   of the CURRENT SUBGOAL.
2. **Decide action** — choose the best action for the robot.

STATUS:
- "complete" — the current subgoal is done (the target object is at \
  its destination in the NOW images).
- "failure" — the robot is clearly stuck, holding the wrong object, \
  or the target object is unreachable.
- "in_progress" — the subgoal is not yet complete but is still \
  achievable. The robot may be approaching, reaching, or grasping.

ACTION (what to do?):
{actions_block}

IMPORTANT RULES:
- Only say "complete" if you see CLEAR visual evidence the subgoal is \
  done in the NOW images (target object at its destination).
- Say "in_progress" if the robot has not yet started or is still \
  working on the current subgoal — this is NORMAL, not a failure.
- Only say "failure" if the robot is clearly stuck, has picked up the \
  WRONG object, or has been making no progress for a long time.
- When in doubt, say "in_progress" with action "continue".
- Do NOT replan just because the scene looks different from BEFORE. \
  Only replan if the current subgoal is impossible or the plan no \
  longer makes sense given the NOW scene.

Output ONLY valid JSON (no markdown, no explanation):
{{"status": "...", "action": "...", "reason": "brief explanation"}}"""


# ======================================================================
# Parser
# ======================================================================

def _parse_vlm_detection(
    raw: str,
    available_actions: set[str],
) -> HandlerResult:
    """Parse VLM JSON response into a HandlerResult.

    Falls back to ``(in_progress, continue)`` on parse failure.
    """
    try:
        data = parse_json(raw)
    except (ValueError, KeyError):
        logger.warning(f"  VLM detection: failed to parse JSON: {raw[:200]}")
        return HandlerResult(
            status=STATUS_IN_PROGRESS,
            action=ACTION_CONTINUE,
            reason="parse_failure",
        )

    status = data.get("status", STATUS_IN_PROGRESS)
    action = data.get("action", ACTION_CONTINUE)
    reason = data.get("reason", "")
    grasp_target = data.get("grasp_target")
    place_destination = data.get("place_destination")
    place_target = data.get("place_target")
    place_relation = data.get("place_relation", "in")
    place_held_object = data.get("place_held_object")
    place_stack = bool(data.get("place_stack", False))

    # Validate status
    if status not in (STATUS_COMPLETE, STATUS_FAILURE, STATUS_IN_PROGRESS):
        logger.warning(
            f"  VLM detection: invalid status {status!r}, "
            f"defaulting to in_progress"
        )
        status = STATUS_IN_PROGRESS

    # Validate action against available set
    if action not in available_actions:
        logger.warning(
            f"  VLM detection: action {action!r} not in "
            f"{available_actions}, defaulting to continue"
        )
        action = ACTION_CONTINUE

    if place_relation not in ("in", "on", "on_top_of"):
        logger.warning(
            f"  VLM detection: invalid place_relation {place_relation!r}, "
            f"defaulting to 'in'"
        )
        place_relation = "in"

    return HandlerResult(
        status=status,
        action=action,
        reason=reason,
        grasp_target=grasp_target,
        place_destination=place_destination,
        place_target=place_target,
        place_relation=place_relation,
        place_held_object=place_held_object,
        place_stack=place_stack,
    )


# ======================================================================
# Handler
# ======================================================================

class VLMFailureHandler(FailureHandler):
    """Periodic VLM-based failure detection and action selection.

    Parameters
    ----------
    vlm_call_fn:
        Callable ``(system_prompt, user_content) -> str`` that makes
        a VLM API call.  Provided by the strategy layer.
    image_builder_fn:
        Callable ``(text, initial_image, current_image, extra_images,
        initial_extra_images) -> list[dict]`` that builds the
        multi-image user content.  Provided by the strategy layer.
    get_vlm_image_fn:
        Callable ``(obs) -> np.ndarray | None`` that extracts the VLM
        image from an observation dict.
    get_extra_images_fn:
        Callable ``(obs) -> list[np.ndarray] | None`` that extracts
        extra camera images from an observation dict.
    check_interval:
        Number of sim steps between VLM checks.  Gated by step delta
        against ``state.episode_step`` so cadence is invariant across
        VLAs whose action chunks differ in length.
    recovery_mode:
        Controls available actions: ``"replan"``, ``"replan_grasp"``,
        or ``"grasp"``.
    """

    def __init__(
        self,
        vlm_call_fn: Callable,
        image_builder_fn: Callable,
        get_vlm_image_fn: Callable,
        get_extra_images_fn: Callable,
        check_interval: int = 80,
        recovery_mode: str = "replan",
        primary_label: str = "External camera",
        extra_labels: list[str] | None = None,
        use_front_camera: bool = False,
        stack_mode_enabled: bool = False,
    ):
        self._vlm_call = vlm_call_fn
        self._build_check_message = image_builder_fn
        self._get_vlm_image = get_vlm_image_fn
        self._get_extra_images = get_extra_images_fn
        self._check_interval = check_interval
        self._primary_label = primary_label
        self._extra_labels = extra_labels

        if recovery_mode not in _ACTION_SETS:
            raise ValueError(
                f"Invalid recovery_mode {recovery_mode!r} for VLM handler, "
                f"expected one of {list(_ACTION_SETS)}"
            )
        self._available_actions = _ACTION_SETS[recovery_mode]
        self._system_prompt = _build_system_prompt(
            self._available_actions,
            use_front_camera=use_front_camera,
            stack_mode_enabled=stack_mode_enabled,
        )

        # Per-episode state — sim step at which the last VLM check
        # fired.  Cadence is gated by step delta against
        # ``state.episode_step`` so that VLA chunks of different sizes
        # produce the same wall-clock check frequency.
        self._step_at_last_check: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(self, obs: dict, state: Any) -> None:
        self._step_at_last_check = state.episode_step

    def on_subgoal_advanced(self, obs: dict, state: Any, idx: int) -> None:
        self._step_at_last_check = state.episode_step

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def step(self, obs: dict, state: Any) -> HandlerResult | None:
        """Run periodic VLM detection.

        Returns ``None`` if it's not time to check yet.
        """
        # Not time to check yet
        if (state.episode_step - self._step_at_last_check
                < self._check_interval):
            return None

        # No subgoals to check against
        if not state.subgoals:
            logger.warning("  VLM handler: no subgoals, skipping")
            return None

        self._step_at_last_check = state.episode_step
        logger.info(
            f"  VLM handler: checking subgoal {state.current_subgoal_idx+1}/"
            f"{len(state.subgoals)} at step {state.episode_step}"
        )

        # Get current image
        current_image = self._get_vlm_image(obs)
        if current_image is None:
            logger.warning("  VLM handler: current_image is None, skipping")
            return None

        # Build context
        sg_idx = state.current_subgoal_idx
        current_sg = state.subgoals[sg_idx]
        total = len(state.subgoals)

        remaining = state.subgoals[sg_idx + 1:]
        if remaining:
            remaining_str = "Remaining subgoals after this:\n" + "\n".join(
                f"  {i}. {s}"
                for i, s in enumerate(remaining, start=sg_idx + 2)
            )
        else:
            remaining_str = "This is the LAST subgoal."

        text = (
            f'Overall task: "{state.original_instruction}"\n'
            f'Current subgoal ({sg_idx + 1}/{total}): "{current_sg}"\n'
            f"{remaining_str}\n\n"
            f"Assess the current subgoal status and decide the action."
        )

        # Build message with BEFORE + NOW images
        extra_images = self._get_extra_images(obs)
        user_content = self._build_check_message(
            text,
            state.initial_image,
            current_image,
            extra_images,
            initial_extra_images=state.initial_extra_images,
            primary_label=self._primary_label,
            extra_labels=self._extra_labels,
        )

        # VLM call
        t0 = time.time()
        try:
            raw = self._vlm_call(self._system_prompt, user_content)
            elapsed = time.time() - t0
        except Exception as e:
            logger.warning(f"  VLM detection call failed: {e}; continuing")
            return None

        logger.info(f"  VLM detect raw ({elapsed:.1f}s): {raw[:500]}")

        # Parse result
        result = _parse_vlm_detection(raw, self._available_actions)
        result.extra["vlm_latency_s"] = round(elapsed, 2)
        result.extra["vlm_raw"] = raw

        # Log
        status_icon = {
            STATUS_COMPLETE: "✓",
            STATUS_FAILURE: "⚠",
            STATUS_IN_PROGRESS: "…",
        }.get(result.status, "?")

        action_icon = {
            ACTION_NEXT: "→",
            ACTION_REPLAN: "↻",
            ACTION_CONTINUE: "…",
            ACTION_GRASP: "🤏",
        }.get(result.action, "?")

        logger.info(
            f"  {status_icon} VLM detect [{sg_idx + 1}/{total}]: "
            f"status={result.status}, action={result.action} "
            f"{action_icon} — {result.reason} ({elapsed:.1f}s)"
        )

        # Always log the VLM check to state (even for in_progress/continue)
        self._store_check_result(state, result)
        state.log({
            "type": "vlm_detect",
            "status": result.status,
            "action": result.action,
            "reason": result.reason,
            "subgoal_idx": sg_idx,
            "subgoal": current_sg,
            "grasp_target": result.grasp_target,
            "vlm_latency_s": round(elapsed, 2),
            "vlm_raw": raw[:500],
            "step_count": state.infer_count,
        })

        # Return None for in_progress + continue (no action needed)
        if (result.status == STATUS_IN_PROGRESS
                and result.action == ACTION_CONTINUE):
            return None

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _store_check_result(state: Any, result: HandlerResult) -> None:
        """Store check result on state for video annotation."""
        state.vlm_check_result = {
            "done": result.status == STATUS_COMPLETE,
            "status": result.status,
            "action": result.action,
            "reason": result.reason,
        }
        state.vlm_check_infer_step = state.infer_count
