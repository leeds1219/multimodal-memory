# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure-aware recovery: generate corrective instructions from failure diagnosis.

Two approaches:
1. **Template-based** — zero-cost, zero-latency. Maps failure type + subgoal
   context → parameterised corrective instruction.
2. **VLM-generated** — one VLM call (~5-10s). Sends failure context + current
   image to VLM and asks for a specific corrective instruction.

Both return a *recovery instruction* — a plain string that replaces the
current subgoal instruction sent to the VLA.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vlm_orchestrator.vlm import encode_image_b64

if TYPE_CHECKING:
    from vlm_orchestrator.failure_handlers.signal_detector import ClassificationResult

logger = logging.getLogger(__name__)


# ======================================================================
# Recovery result
# ======================================================================

@dataclass
class RecoveryAction:
    """What the recovery system decided to do."""
    instruction: str           # New instruction for VLA
    method: str                # "template" | "vlm" | "human"
    failure_type: str          # Original failure type (fumble, open_high, ...)
    original_instruction: str  # What we were trying before
    reasoning: str = ""        # VLM/human reasoning (empty for template)


# ======================================================================
# Template-based recovery
# ======================================================================

# Failure type → recovery instruction template.
# {object} is replaced with the target object from the current subgoal.
# {instruction} is replaced with the original subgoal instruction.
RECOVERY_TEMPLATES: dict[str, list[str]] = {
    # Object was dropped mid-transport (gripper opened at height)
    "open_high": [
        "The object was dropped. Move back to the table, pick up the {object}, and carefully place it in the target location",
        "Re-grasp the {object} that was just dropped on the table and place it where it needs to go",
    ],
    # Repeated open/close without successful grasp
    "fumble": [
        "Move directly above the {object}, descend slowly, close the gripper firmly, then lift and place it in the target",
        "Approach the {object} carefully from above, grasp it firmly, and complete the placement",
    ],
    # Robot not moving / EE stuck in one position
    "stall": [
        "Pull back from the current position, then approach the {object} and pick it up",
        "Move the arm away, reposition above the {object}, and attempt to grasp it",
    ],
    # Gripper never closed on anything
    "never_gripped": [
        "Move to the {object}, close the gripper to grasp it firmly, then lift and place it in the target",
        "The gripper never grasped anything. Move directly to the {object} and pick it up",
    ],
    # Generic / unknown failure type
    "unknown": [
        "Retry: {instruction}",
    ],
}


def _extract_target_object(subgoal: str) -> str:
    """Best-effort extraction of target object from a subgoal string.

    Handles both plain instruction strings and JSON-style subgoal dicts.
    """
    import json
    import re

    # Try parsing as dict (subgoals sometimes stored as stringified dicts)
    if subgoal.strip().startswith("{"):
        try:
            # Try Python literal eval first (handles single quotes, apostrophes)
            import ast
            data = ast.literal_eval(subgoal)
            if isinstance(data, dict):
                if "target_object" in data:
                    return data["target_object"]
                if "instruction" in data:
                    subgoal = data["instruction"]
        except (ValueError, SyntaxError):
            pass
        # Fallback to JSON
        try:
            data = json.loads(subgoal.replace("'", '"'))
            if "target_object" in data:
                return data["target_object"]
            if "instruction" in data:
                subgoal = data["instruction"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Regex: "Pick up the X and place it..."
    m = re.search(
        r"(?:pick up|grasp|grab|lift|move)\s+(?:the\s+)?(.+?)"
        r"\s+(?:and|then|to|into|in|from)",
        subgoal, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(",.")

    # Fallback: "... the X in the bin"
    m = re.search(r"(?:the|a)\s+(.+?)\s+(?:in|into|on|onto)\s+", subgoal, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(",.")

    return "object"


def template_recovery(
    failure_type: str,
    subgoal_instruction: str,
    retry_count: int = 0,
) -> RecoveryAction:
    """Generate a recovery instruction from templates.

    Uses ``retry_count`` to pick different templates on successive retries
    (so the VLA sees a different instruction each time).
    """
    obj = _extract_target_object(subgoal_instruction)
    templates = RECOVERY_TEMPLATES.get(
        failure_type, RECOVERY_TEMPLATES["unknown"]
    )
    # Cycle through templates on successive retries
    template = templates[retry_count % len(templates)]
    instruction = template.format(object=obj, instruction=subgoal_instruction)

    logger.info(f"  Template recovery [{failure_type}]: \"{instruction}\"")

    return RecoveryAction(
        instruction=instruction,
        method="template",
        failure_type=failure_type,
        original_instruction=subgoal_instruction,
    )


# ======================================================================
# VLM-generated recovery
# ======================================================================

RECOVERY_SYSTEM_PROMPT = """\
You are a robot manipulation recovery planner. A robot arm was attempting \
a pick-and-place task but failed. You will see the current camera image \
showing the scene after the failure.

Your job: write ONE specific, actionable instruction that tells the robot \
what to do RIGHT NOW to recover from this failure and complete the task.

Rules:
- Be specific about movements: "move above", "descend to", "grasp firmly", \
  "pull back", "approach from the side"
- Reference the actual object by name
- The instruction must be a single step the robot can execute directly
- Do NOT repeat the original instruction verbatim — adapt it to the failure
- Keep it concise: one sentence, max two

Output ONLY the recovery instruction text, nothing else.\
"""

# Human-readable failure descriptions for VLM context.
FAILURE_DESCRIPTIONS: dict[str, str] = {
    "open_high": (
        "The robot dropped the object during transport — the gripper "
        "opened while the arm was still high above the target."
    ),
    "fumble": (
        "The robot repeatedly opened and closed its gripper without "
        "successfully grasping the object (fumbled grasp)."
    ),
    "stall": (
        "The robot stopped moving — the arm is stuck in one position "
        "and has not made progress for many steps."
    ),
    "never_gripped": (
        "The robot moved around but never closed its gripper on the "
        "object. It failed to even attempt a grasp."
    ),
}


def vlm_recovery(
    failure_type: str,
    subgoal_instruction: str,
    task_instruction: str,
    current_image: np.ndarray,
    vlm_call_fn,
    retry_count: int = 0,
) -> RecoveryAction:
    """Generate a recovery instruction by calling the VLM.

    Parameters
    ----------
    failure_type : str
        Detected failure type (fumble, open_high, stall, ...).
    subgoal_instruction : str
        The subgoal instruction that failed.
    task_instruction : str
        The overall task instruction.
    current_image : np.ndarray
        Current camera image (scene after failure).
    vlm_call_fn : callable
        ``(system_prompt: str, user_content: list[dict]) -> str``
    retry_count : int
        How many times we've already retried this subgoal.
    """
    failure_desc = FAILURE_DESCRIPTIONS.get(
        failure_type,
        f"The robot failed (type: {failure_type}) during the task.",
    )

    user_text = (
        f"Overall task: \"{task_instruction}\"\n"
        f"Current subgoal: \"{subgoal_instruction}\"\n"
        f"Retry attempt: {retry_count + 1}\n\n"
        f"FAILURE: {failure_desc}\n\n"
        f"The image shows the current scene after the failure. "
        f"What should the robot do right now to recover?"
    )

    user_content = [
        {"type": "text", "text": user_text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_b64(current_image)}"
            },
        },
    ]

    t0 = time.time()
    try:
        raw = vlm_call_fn(RECOVERY_SYSTEM_PROMPT, user_content)
        elapsed = time.time() - t0
        logger.info(
            f"  VLM recovery ({elapsed:.1f}s) [{failure_type}]: \"{raw[:150]}\""
        )
        # Clean up: remove quotes, "Recovery: " prefix, etc.
        instruction = raw.strip().strip('"').strip("'")
        if instruction.lower().startswith("recovery:"):
            instruction = instruction[len("recovery:"):].strip()

        return RecoveryAction(
            instruction=instruction,
            method="vlm",
            failure_type=failure_type,
            original_instruction=subgoal_instruction,
            reasoning=raw,
        )
    except Exception as e:
        logger.warning(f"  VLM recovery call failed ({e}), falling back to template")
        return template_recovery(failure_type, subgoal_instruction, retry_count)


# ======================================================================
# Recovery strategy dispatcher
# ======================================================================

def retry_recovery(
    failure_type: str,
    subgoal_instruction: str,
) -> RecoveryAction:
    """Blind retry: re-send the exact same instruction (no rewriting).

    This is the simplest possible recovery — reset counters and try again.
    Useful as a baseline to measure whether instruction changes actually help.
    """
    logger.info(f"  Retry recovery [{failure_type}]: re-sending original instruction")

    return RecoveryAction(
        instruction=subgoal_instruction,
        method="retry",
        failure_type=failure_type,
        original_instruction=subgoal_instruction,
    )


# ======================================================================
# VLM recovery with grasp-tool option
# ======================================================================

RECOVERY_GRASP_SYSTEM_PROMPT = """\
You are a robot manipulation recovery planner. A robot arm was attempting \
a pick-and-place task but failed. You will see the current camera image \
showing the scene after the failure.

You have TWO recovery options:

1. **retry** — Give the robot a new, refined instruction and let it try \
   again with its learned policy.  Best when the robot was close to \
   succeeding, the object moved slightly, or the approach angle was wrong.

2. **grasp_tool** — Activate the planned-grasp pipeline.  This uses a \
   grasp-pose prediction model to compute an exact grasp pose, then \
   executes a precise IK-planned trajectory to pick up the object.  \
   After grasping, the robot's learned policy takes over for placement.  \
   Best when the robot fundamentally cannot grasp the object (fumbling, \
   never gripping, stuck in a loop).

Decision guidelines:
- If the object is still on the table and the robot failed to grasp it \
  → prefer **grasp_tool** (the learned policy already failed at grasping)
- If the robot grasped but dropped the object → **retry** (grasping works, \
  just needs a better trajectory)
- If the robot is stuck / frozen → **grasp_tool** (policy is confused)
- If the scene changed (object moved to a reachable spot) → **retry**
- When in doubt → **grasp_tool** (it's more reliable for grasping)

Output ONLY valid JSON:
{"action": "retry", "instruction": "new instruction for the robot"}
or
{"action": "grasp_tool", "target_object": "the red block"}\
"""


def vlm_grasp_recovery(
    failure_type: str,
    subgoal_instruction: str,
    task_instruction: str,
    current_image: np.ndarray,
    vlm_call_fn,
    retry_count: int = 0,
) -> RecoveryAction:
    """VLM decides: retry with new instruction OR use grasp tool.

    Returns a RecoveryAction. When the VLM chooses ``grasp_tool``, the
    ``method`` field is ``"grasp_tool"`` and ``instruction`` contains the
    target object name (for the caller to activate the grasp executor).
    """
    import json as _json

    failure_desc = FAILURE_DESCRIPTIONS.get(
        failure_type,
        f"The robot failed (type: {failure_type}) during the task.",
    )

    user_text = (
        f"Overall task: \"{task_instruction}\"\n"
        f"Current subgoal: \"{subgoal_instruction}\"\n"
        f"Retry attempt: {retry_count + 1}\n\n"
        f"FAILURE: {failure_desc}\n\n"
        f"Look at the image. Decide: should the robot **retry** with "
        f"a new instruction, or use **grasp_tool** for a precise "
        f"planned grasp?"
    )

    user_content = [
        {"type": "text", "text": user_text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_b64(current_image)}"
            },
        },
    ]

    t0 = time.time()
    try:
        raw = vlm_call_fn(RECOVERY_GRASP_SYSTEM_PROMPT, user_content)
        elapsed = time.time() - t0
        logger.info(
            f"  VLM grasp-recovery ({elapsed:.1f}s): {raw[:200]}"
        )

        # Parse JSON response
        from vlm_orchestrator.vlm import parse_json
        data = parse_json(raw)
        action = data.get("action", "retry")

        if action == "grasp_tool":
            target = data.get("target_object", _extract_target_object(subgoal_instruction))
            logger.info(f"  VLM chose grasp_tool for '{target}'")
            return RecoveryAction(
                instruction=target,  # target object name
                method="grasp_tool",
                failure_type=failure_type,
                original_instruction=subgoal_instruction,
                reasoning=raw,
            )
        else:
            instruction = data.get("instruction", subgoal_instruction)
            logger.info(f"  VLM chose retry: \"{instruction[:100]}\"")
            return RecoveryAction(
                instruction=instruction,
                method="vlm",
                failure_type=failure_type,
                original_instruction=subgoal_instruction,
                reasoning=raw,
            )
    except Exception as e:
        logger.warning(f"  VLM grasp-recovery failed ({e}), using template")
        return template_recovery(failure_type, subgoal_instruction, retry_count)


def generate_recovery(
    failure_result: "ClassificationResult",
    subgoal_instruction: str,
    task_instruction: str,
    current_image: np.ndarray | None,
    vlm_call_fn=None,
    retry_count: int = 0,
    mode: str = "template",
) -> RecoveryAction:
    """Main entry point: pick the right recovery method and generate.

    Parameters
    ----------
    mode : str
        "retry"    — blind retry, re-send same instruction
        "template" — fast, free, deterministic instruction rewriting
        "vlm"      — calls VLM for context-aware recovery instruction
        "vlm_grasp" — VLM decides: retry with new instruction OR use grasp tool
        "human"    — placeholder; returns None (handled by HITL controller)
    """
    # Parse failure type from result.reason
    # reason is like "fumble: 12 grip transitions..." — take first word
    failure_type = failure_result.reason.split(":")[0].strip() if failure_result.reason else "unknown"

    if mode == "retry":
        return retry_recovery(
            failure_type=failure_type,
            subgoal_instruction=subgoal_instruction,
        )

    if mode == "vlm_grasp" and current_image is not None and vlm_call_fn is not None:
        return vlm_grasp_recovery(
            failure_type=failure_type,
            subgoal_instruction=subgoal_instruction,
            task_instruction=task_instruction,
            current_image=current_image,
            vlm_call_fn=vlm_call_fn,
            retry_count=retry_count,
        )

    if mode == "vlm" and current_image is not None and vlm_call_fn is not None:
        return vlm_recovery(
            failure_type=failure_type,
            subgoal_instruction=subgoal_instruction,
            task_instruction=task_instruction,
            current_image=current_image,
            vlm_call_fn=vlm_call_fn,
            retry_count=retry_count,
        )
    else:
        if mode in ("vlm", "vlm_grasp"):
            logger.warning("  VLM recovery requested but no image/client — using template")
        return template_recovery(
            failure_type=failure_type,
            subgoal_instruction=subgoal_instruction,
            retry_count=retry_count,
        )
