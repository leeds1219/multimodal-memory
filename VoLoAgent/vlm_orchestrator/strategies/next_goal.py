# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Next-goal strategy — the VLM predicts the next step on-the-fly.

Runs in parallel with (not under) the subgoal strategy.  Instead of
decomposing a task upfront, this strategy fires every *check_interval*
action chunks and asks the VLM: "given the overall goal, the history of
past checkpoints, and the current scene — what should the robot do next?"

The predicted instruction replaces the current VLA prompt immediately.
Each checkpoint image is appended to a rolling history that is forwarded
to every subsequent VLM call.

Usage::

    vlm-orchestrator --mode next_goal
    vlm-orchestrator --mode next_goal --check-interval 15
    vlm-orchestrator --mode next_goal \\
        --next-goal-template v2 --next-goal-task-type i3_closed_vocab
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np


# === merged from prompts.py =======================================
import json
import re

# =============================================================================
# Task-specific configurations
# =============================================================================

# Subtask candidate sets for different tasks
SUBTASK_CANDIDATE_SETS: dict[str, list[str]] = {
    "i3_closed_vocab": [
        "pick up the mail with the left arm while using the right arm to keep the letter steady",
        "pick up the mail with the right arm while using the left arm to keep the letter steady",
        "put the mail in the red mailbox",
        "put the mail in the blue mailbox",
        "retract the arms to the original position",
        "hand the mail held in one arm to the other arm (after it's picked up)",
        "other",
    ],
    "i4_closed_vocab": [
        "put the candy in the white box",
        "close the white box",
        "put both arms in a neutral position",
        "lower the arms to the original position",
        "place the box in the red tray",
        "place the box in the green tray",
        "place the box in the yellow tray",
        "place the box in the orange tray",
        "other",
    ],
    "h8_closed_vocab": [
        "lower the arms to the original position",
        "pick up and place a yellow m&m bag in the pumpkin bowl",
        "pick up and place a gold twix bar in the pumpkin bowl",
        "pick up and place an orange reeses bar in the pumpkin bowl",
        "pick up and place a red kitkat bar in the pumpkin bowl",
        "pick up and place a green skittles bag in the pumpkin bowl",
        "move both arms to a neutral raised position",
        "other",
    ],
}

# Additional context/information to include in prompts for each task type
# Key: task_type name, Value: context string to prepend to prompts
TASK_CONTEXT: dict[str, str] = {
    "i3_closed_vocab": (
        "For this task, the red mailbox is for international mails, "
        "and the blue mailbox is for domestic mails. "
        "Make sure to pick up the mail with the arm that is closest to the correct mailbox. "
        "For example, if the correct mailbox is on the left, pick up the mail with the left arm.\n"
    ),
    "i4_closed_vocab": "",
    "h8_closed_vocab": "",
}


def get_subtask_candidates(
    task_type: str | None, fallback_candidates: list[str] | None = None
) -> list[str]:
    """Get subtask candidates for a given task type.

    Args:
        task_type: Name of the task type (e.g., "i3", "i4"). If None, uses fallback.
        fallback_candidates: Candidates to use if task_type is None or not found.

    Returns:
        List of subtask candidate strings.
    """
    if task_type is None:
        return fallback_candidates or []

    if task_type not in SUBTASK_CANDIDATE_SETS:
        available = ", ".join(SUBTASK_CANDIDATE_SETS.keys())
        print(  # noqa: T201
            f"Warning: Unknown task_type '{task_type}'. Available: {available}. Using fallback candidates."
        )
        return fallback_candidates or []

    return SUBTASK_CANDIDATE_SETS[task_type]


def get_task_context(task_type: str | None) -> str:
    """Get additional context string for a given task type.

    Args:
        task_type: Name of the task type (e.g., "i3", "i4"). If None, returns empty string.

    Returns:
        Context string to include in prompts, or empty string if no context.
    """
    if task_type is None:
        return ""
    return TASK_CONTEXT.get(task_type, "")


# =============================================================================
# MCQ label utilities
# =============================================================================


def make_option_labels(n: int) -> list[str]:
    """Generate option labels for n choices: A, B, ..., Z, AA, AB, ..., AZ, BA, ...

    Supports up to 702 options (26 single-letter + 676 double-letter).
    """
    labels = []
    for i in range(n):
        if i < 26:
            labels.append(chr(ord("A") + i))
        else:
            j = i - 26
            labels.append(chr(ord("A") + j // 26) + chr(ord("A") + j % 26))
    return labels


def label_to_index(label: str, valid_labels: list[str]) -> int | None:
    """Convert an option label back to a 0-based index, or None if invalid."""
    label = label.strip().upper().rstrip(".),:;")
    try:
        return valid_labels.index(label)
    except ValueError:
        return None


# =============================================================================
# Prompt building utilities
# =============================================================================


def _build_context_description(
    history_text: str,
    num_history_images: int,
    num_current_images: int,
    task_type: str | None = None,
) -> str:
    """Build a description of the context being provided to the model.

    This creates a unified description that works for all memory types.

    Args:
        history_text: Text summary of previous actions.
        num_history_images: Number of history images provided.
        num_current_images: Number of current/recent images provided.
        task_type: Optional task type for task-specific context.

    Returns:
        Description string to include in the prompt.
    """
    parts = []

    # Add task-specific context if available
    task_context = get_task_context(task_type)
    if task_context:
        parts.append(task_context)

    # Describe images
    if num_history_images > 0 and num_current_images > 0:
        parts.append(
            f"You are provided with {num_history_images + num_current_images} images in total: "
            f"the first {num_history_images} images show the history of past states, "
            f"and the last {num_current_images} images show the current/recent state of the robot."
        )
    elif num_current_images > 0:
        parts.append(
            f"You are provided with {num_current_images} images showing the current/recent state of the robot."
        )

    # Describe text history
    if history_text:
        parts.append(
            f"Additionally, here is a summary of previous predicted actions. "
            f"Note that an action may need multiple steps to complete,"
            f"and the predicted action may not be accurate. "
            f"So, it is possible that the actions in the history text are not complete, or even incorrect. "
            f"Use the images to guide your reasoning, only consider past action predictions as loose guidance.\n"
            f"{history_text}\n"
        )

    if parts:
        return "\n".join(parts) + "\n\n"
    return ""


# =============================================================================
# V1 Templates (freeform instruction output)
# =============================================================================


def next_action_template_v1_freeform(
    goal: str,
    subtask_candidates: list | None = None,
    history_text: str = "",
    num_history_images: int = 0,
    num_current_images: int = 0,
    task_type: str | None = None,
) -> str:
    """Freeform template with structured CoT output (observation, progress, reasoning, instruction).

    Like v2 but produces a free-text instruction instead of selecting from MCQ options.
    Designed to be embodiment-neutral and produce concise action phrases (3-15 words)
    matching the style of ground-truth annotations across diverse data sources.

    Args:
        goal: The overall task goal.
        subtask_candidates: Ignored for freeform mode (kept for API compatibility).
        history_text: Text summary of previous actions (for text-based memory).
        num_history_images: Number of history images provided (for image-based memory).
        num_current_images: Number of current/recent images provided.
        task_type: Optional task type for task-specific context.

    Returns:
        The formatted prompt string.
    """
    _ = subtask_candidates  # Not used in freeform mode

    context_description = _build_context_description(
        history_text, num_history_images, num_current_images, task_type
    )

    format_example = {
        "observation": "<what you see in the current image(s)>",
        "progress": "<current progress toward the goal>",
        "reasoning": "<what should happen next and why>",
        "instruction": "<concise action phrase, e.g. 'pick up the red cup'>",
    }

    out = (
        f"You are a robot task planner. The robot's goal is:\n"
        f"{goal}\n\n"
        f"{context_description}"
        f"Based on the images, predict the next sub-task the robot should perform.\n\n"
        f"Instructions:\n"
        f"1. Describe what you see in the current images (one sentence)\n"
        f"2. Assess the current progress towards the goal (one sentence)\n"
        f"3. Explain what action should be taken next and why (one sentence)\n"
        f"4. Provide a concise action phrase for the next sub-task (roughly 3-15 words).\n"
        f"   Describe WHAT to do, not HOW. Be specific about which object(s) and where.\n\n"
        f"You MUST respond with a JSON object containing exactly these four fields. "
        f"Do not include any text outside the JSON block.\n\n"
        f"Output format:\n"
        f"```json\n{json.dumps(format_example, indent=2)}\n```\n"
    )
    return out


def next_action_template_v1(
    goal: str,
    subtask_candidates: list | None = None,
    history_text: str = "",
    num_history_images: int = 0,
    num_current_images: int = 0,
    task_type: str | None = None,
) -> str:
    """V1 template with memory support.

    Args:
        goal: The overall task goal.
        subtask_candidates: Optional list of candidate subtasks to choose from.
        history_text: Text summary of previous actions (for text-based memory).
        num_history_images: Number of history images provided (for image-based memory).
        num_current_images: Number of current/recent images provided.
        task_type: Optional task type for task-specific context.

    Returns:
        The formatted prompt string.
    """
    format_example = {"instruction": "<INSTRUCTION>"}
    if subtask_candidates is not None:
        subtask_candidates_str = (
            "You MUST select ONE instruction from the following list: \n"
            + "\n".join([f"{i+1}: {subtask}" for i, subtask in enumerate(subtask_candidates)])
        )
    else:
        subtask_candidates_str = ""

    # Build context description based on what memory is provided
    context_description = _build_context_description(
        history_text, num_history_images, num_current_images, task_type
    )

    out = (
        f"# You are an expert robot control engineer. The robot is given this goal:\n"
        f"{goal}\n\n"
        f"{context_description}"
        f"Please observe the images from the camera and provide a simple instruction to the robot. "
        f"The instruction should be a single sentence in plain english, returned in the JSON format below. "
        f"{subtask_candidates_str}\n"
        f"You should follow the steps below:\n"
        f"1. Provide the description of the current images in one sentence\n"
        f"2. Considering any history provided, explain the current progress towards the goal in one sentence\n"
        f"3. Motivate what action should be taken next in one sentence\n"
        f"4. Provide a simple instruction to the robot and return the instruction in the JSON format "
        f"as shown in the example below: "
        f"```json\n{json.dumps(format_example)}\n```\n"
    )
    return out


# =============================================================================
# V2 Templates (MCQ format with structured output)
# =============================================================================


def next_action_template_v2(
    goal: str,
    subtask_candidates: list | None = None,
    history_text: str = "",
    num_history_images: int = 0,
    num_current_images: int = 0,
    task_type: str | None = None,
) -> str:
    """MCQ format template with memory support and structured output.

    Args:
        goal: The overall task goal.
        subtask_candidates: List of candidate subtasks (required for MCQ format).
        history_text: Text summary of previous actions (for text-based memory).
        num_history_images: Number of history images provided (for image-based memory).
        num_current_images: Number of current/recent images provided.
        task_type: Optional task type for task-specific context.

    Returns:
        The formatted prompt string.

    Raises:
        ValueError: If ``subtask_candidates`` is ``None`` or empty.
    """
    if subtask_candidates is None or len(subtask_candidates) == 0:
        raise ValueError("subtask_candidates is required for MCQ format")

    option_labels = make_option_labels(len(subtask_candidates))
    options_str = "\n".join(
        [
            f"{label}. {subtask}"
            for label, subtask in zip(option_labels, subtask_candidates, strict=False)
        ]
    )
    if len(option_labels) <= 10:
        valid_options = ", ".join(option_labels)
    else:
        valid_options = f"{option_labels[0]}-{option_labels[-1]}"

    # Build context description based on what memory is provided
    context_description = _build_context_description(
        history_text, num_history_images, num_current_images, task_type
    )

    # Example output format with structured fields
    format_example = {
        "observation": "<one sentence describing what you see in the current images>",
        "progress": "<one sentence explaining the current progress towards the goal>",
        "reasoning": "<one sentence explaining what action should be taken next and why>",
        "option": "<OPTION_LETTER>",
    }

    out = (
        f"# You are an expert robot control engineer. The robot is given this goal:\n"
        f"{goal}\nYour task is to predict the next sub-task instruction for the robot to complete the goal.\n"
        f"{context_description}"
        f"Please observe the images from the camera and select the best instruction for the robot.\n\n"
        f"Options:\n{options_str}\n\n"
        f"Instructions:\n"
        f"1. Describe what you see in the current images (one sentence)\n"
        f"2. Assess the current progress towards the goal (one sentence)\n"
        f"3. Explain what action should be taken next and why (one sentence)\n"
        f"4. Select the best option from the list above ({valid_options})\n\n"
        f"You MUST respond with a JSON object containing exactly these four fields. "
        f"Do not include any text outside the JSON block.\n\n"
        f"Output format:\n"
        f"```json\n{json.dumps(format_example, indent=2)}\n```\n"
    )
    return out


NEXT_ACTION_TEMPLATE = {
    # Canonical names
    "v1": next_action_template_v1,
    "v2": next_action_template_v2,
    "v1_freeform": next_action_template_v1_freeform,
    # Aliases
    "mcq": next_action_template_v2,
}


# =============================================================================
# Parsers
# =============================================================================


def next_action_parser_v1(
    response: str,
    subtask_candidates: list = None,  # Not used, kept for API compatibility  # noqa: RUF013
    *,
    fallback: str,
) -> tuple[str, bool, dict | None]:
    """Parse the model response to extract the instruction.

    Args:
        response: Raw response from the model
        subtask_candidates: Not used in v1 mode, kept for compatibility
        fallback: Fallback instruction to return if parsing fails

    Returns:
        Tuple of (instruction, used_fallback, metadata) where:
        - instruction: The parsed instruction string, or fallback if parsing fails
        - used_fallback: True if fallback was used, False if parsing succeeded
        - metadata: None for v1 (no structured output)
    """
    _ = subtask_candidates  # Explicitly mark as unused
    try:
        if not response:
            print("Warning: Empty/None response from model. Using fallback.")  # noqa: T201
            return fallback, True, None
        # Check if response contains JSON code block
        if "```json" in response and "```" in response:
            match = re.search(r"```json(.*)```", response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                print(  # noqa: T201
                    "Warning: Found ```json marker but couldn't extract JSON content. Using fallback."
                )
                return fallback, True, None

        # Try to parse JSON
        parsed = json.loads(response)

        # Extract instruction
        if "instruction" in parsed:
            return parsed["instruction"], False, None
        else:
            print(  # noqa: T201
                f"Warning: JSON response missing 'instruction' key. "
                f"Available keys: {list(parsed.keys())}. Using fallback."
            )
            return fallback, True, None

    except json.JSONDecodeError as e:
        print(  # noqa: T201
            f"Warning: Failed to parse JSON: {e}. Response text: {response[:200]}. Using fallback."
        )
        return fallback, True, None
    except Exception as e:  # noqa: BLE001
        print(  # noqa: T201
            f"Warning: Unexpected error parsing response: {e}. Response text: {response[:200]}. Using fallback."
        )
        return fallback, True, None


def next_action_parser_v2(
    response: str, subtask_candidates: list, *, fallback: str
) -> tuple[str, bool, dict | None]:
    """Parse the MCQ JSON response to extract the instruction and reasoning.

    Args:
        response: Raw response from the model (should contain JSON with required fields)
        subtask_candidates: List of candidate instructions corresponding to options
        fallback: Fallback instruction to return if parsing fails

    Returns:
        Tuple of (instruction, used_fallback, metadata) where:
        - instruction: The parsed instruction string, or fallback if parsing fails
        - used_fallback: True if fallback was used, False if parsing succeeded
        - metadata: Dict containing observation, progress, reasoning (if parsed successfully)
    """
    if not subtask_candidates:
        print(  # noqa: T201
            "Warning: No subtask candidates provided for MCQ parser. Using fallback."
        )
        return fallback, True, None

    try:
        if not response:
            print("Warning: Empty/None response from model. Using fallback.")  # noqa: T201
            return fallback, True, None
        json_content = response

        # Check if response contains JSON code block
        if "```json" in response and "```" in response:
            match = re.search(r"```json(.*)```", response, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                print(  # noqa: T201
                    f"Warning: Found ```json marker but couldn't extract JSON content. "
                    f"Raw response: {response}. Using fallback."
                )
                return fallback, True, None

        # Try to parse JSON
        parsed = json.loads(json_content)

        # Extract option letter
        if "option" not in parsed:
            print(  # noqa: T201
                f"Warning: JSON response missing 'option' key. "
                f"Available keys: {list(parsed.keys())}. Raw response: {response}. Using fallback."
            )
            return fallback, True, None

        valid_labels = make_option_labels(len(subtask_candidates))
        index = label_to_index(str(parsed["option"]), valid_labels)

        if index is not None:
            metadata = {
                "observation": parsed.get("observation"),
                "progress": parsed.get("progress"),
                "reasoning": parsed.get("reasoning"),
                "selected_option": valid_labels[index],
            }
            return subtask_candidates[index], False, metadata
        else:
            print(  # noqa: T201
                f"Warning: Invalid option value '{parsed['option']}'. "
                f"Expected one of {valid_labels[0]}-{valid_labels[-1]}. "
                f"Raw response: {response}. Using fallback."
            )
            return fallback, True, None

    except json.JSONDecodeError as e:
        print(  # noqa: T201
            f"Warning: Failed to parse JSON: {e}. Response text: {response}. Using fallback."
        )
        return fallback, True, None
    except Exception as e:  # noqa: BLE001
        print(  # noqa: T201
            f"Warning: Unexpected error parsing MCQ response: {e}. "
            f"Response text: {response}. Using fallback."
        )
        return fallback, True, None


def next_action_parser_v1_freeform(
    response: str,
    subtask_candidates: list = None,  # Not used, kept for API compatibility  # noqa: RUF013
    *,
    fallback: str,
) -> tuple[str, bool, dict | None]:
    """Parse freeform structured CoT response (observation, progress, reasoning, instruction).

    Args:
        response: Raw response from the model (should contain JSON with required fields)
        subtask_candidates: Not used in freeform mode, kept for compatibility
        fallback: Fallback instruction to return if parsing fails

    Returns:
        Tuple of (instruction, used_fallback, metadata) where:
        - instruction: The parsed instruction string, or fallback if parsing fails
        - used_fallback: True if fallback was used, False if parsing succeeded
        - metadata: Dict containing observation, progress, reasoning (if parsed successfully)
    """
    _ = subtask_candidates  # Explicitly mark as unused
    try:
        if not response:
            print("Warning: Empty/None response from model. Using fallback.")  # noqa: T201
            return fallback, True, None

        json_content = response

        # Check if response contains JSON code block
        if "```json" in response and "```" in response:
            match = re.search(r"```json(.*)```", response, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                print(  # noqa: T201
                    "Warning: Found ```json marker but couldn't extract JSON content. Using fallback."
                )
                return fallback, True, None

        # Try to parse JSON
        parsed = json.loads(json_content)

        # Extract instruction
        if "instruction" not in parsed:
            print(  # noqa: T201
                f"Warning: JSON response missing 'instruction' key. "
                f"Available keys: {list(parsed.keys())}. Using fallback."
            )
            return fallback, True, None

        instruction = parsed["instruction"]
        if not instruction or not isinstance(instruction, str):
            print(  # noqa: T201
                f"Warning: Invalid instruction value: {instruction!r}. Using fallback."
            )
            return fallback, True, None

        # Extract metadata from the structured response
        metadata = {
            "observation": parsed.get("observation"),
            "progress": parsed.get("progress"),
            "reasoning": parsed.get("reasoning"),
        }

        return instruction, False, metadata

    except json.JSONDecodeError as e:
        print(  # noqa: T201
            f"Warning: Failed to parse JSON: {e}. Response text: {response[:200]}. Using fallback."
        )
        return fallback, True, None
    except Exception as e:  # noqa: BLE001
        print(  # noqa: T201
            f"Warning: Unexpected error parsing freeform response: {e}. "
            f"Response text: {response[:200]}. Using fallback."
        )
        return fallback, True, None


NEXT_ACTION_PARSER = {
    "v1": next_action_parser_v1,
    "v2": next_action_parser_v2,
    "v1_freeform": next_action_parser_v1_freeform,
    # Aliases
    "mcq": next_action_parser_v2,
}

# === end of merged prompts ========================================
from vlm_orchestrator.vlm import encode_image_b64

from .base import OrchestrationStrategy, SessionState, StrategyContext

logger = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================


@dataclass
class NextGoalConfig:
    """Configuration for the next-goal strategy."""

    # VLM connection (mirrors SubgoalConfig fields)
    vlm_model: str = "YOUR_VLM_MODEL"
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 1024
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None

    # Prediction cadence
    check_interval: int = 80
    """Sim steps between VLM predictions.  Gated by step delta against
    ``state.episode_step`` so cadence is invariant across VLAs whose
    chunk sizes differ (pi05=8, gr00t=10, openvla=1)."""

    # Prompt selection
    template_name: str = "v1_freeform"
    """Key into ``prompts.NEXT_ACTION_TEMPLATE``.
    ``v1_freeform`` (default) — free-text output with CoT.
    ``v1`` — free-text, simpler.
    ``v2`` / ``mcq`` — MCQ selection; requires ``task_type``.
    """

    task_type: str | None = None
    """Task type key for ``prompts.SUBTASK_CANDIDATE_SETS`` and
    ``prompts.TASK_CONTEXT``.  Required for MCQ templates; optional
    for freeform templates (adds task-specific context if set).
    """

    max_history_images: int = 0
    """Maximum checkpoint images kept in the rolling history.
    Oldest image is dropped when the limit is exceeded.
    ``0`` (default) means unlimited — all checkpoints are forwarded to the VLM.
    With check_interval=10 over a ~170-step episode this reaches ~17 images.
    For API models (Claude, Llama-vision) each extra image adds ~0.6 s and
    ~1200 input tokens, so 17 images ≈ 10 s/call — acceptable for offline eval.
    """

    vla_capabilities: str | None = None
    """Optional description of VLA capabilities to include in the system
    prompt. When set, the orchestrator will tailor its subgoal instructions
    to match what the VLA can actually execute. Built-in presets:
    ``molmobot`` — MolmoBot pick/pick-and-place skill set.
    Any other string is used verbatim as the capabilities description.
    """


# ======================================================================
# Strategy
# ======================================================================


class NextGoalStrategy(OrchestrationStrategy):
    """Periodic next-step prediction via the VLM.

    Every *check_interval* action chunks the strategy:

    1. Collects the current scene image.
    2. Builds a prompt from the overall goal, the rolling history of
       past checkpoint images, and the current image.
    3. Calls the VLM and parses the next-step instruction.
    4. Replaces the current VLA prompt with the predicted instruction.
    5. Appends the current image to the history.

    The history resets at every episode boundary.
    """

    def __init__(self, ctx: StrategyContext, config: NextGoalConfig) -> None:
        super().__init__(ctx)
        self.config = config

        # Lazy OpenAI-compatible client
        self._client = None
        self._call_count = 0
        self._last_api_usage: dict | None = None  # set as side-effect in _vlm_call

        # Validate template name early — better error than a KeyError later
        if config.template_name not in NEXT_ACTION_TEMPLATE:
            raise ValueError(
                f"Unknown --next-goal-template {config.template_name!r}. "
                f"Valid options: {list(NEXT_ACTION_TEMPLATE)}"
            )
        self._template_fn = NEXT_ACTION_TEMPLATE[config.template_name]
        self._parser_fn = NEXT_ACTION_PARSER[config.template_name]
        self._subtask_candidates: list[str] = get_subtask_candidates(config.task_type)

        # Per-episode state — reset in _on_new_episode.  Cadence is
        # gated by step delta against ``state.episode_step`` rather than
        # a chunk counter, so chunk size differences across VLAs don't
        # change the wall-clock prediction frequency.
        self._history_images: list[np.ndarray] = []
        self._step_at_last_check: int = 0

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    _VLA_CAPABILITIES_PRESETS: dict[str, str] = {
        "molmobot": (
            "The robot policy (VLA) you are controlling can ONLY execute "
            "the following primitive skills:\n"
            "- PICK: Grasp a specified object and lift it. "
            "Instruction format: 'pick up the <object>'\n"
            "- PICK AND PLACE: Grasp a specified object and place it into/onto "
            "a specified receptacle or location. "
            "Instruction format: 'pick up the <object> and place it on/in the <target>'\n"
            "- PICK AND PLACE NEXT TO: Grasp a specified object and place it "
            "adjacent to a reference object on the same surface. "
            "Instruction format: 'pick up the <object> and place it next to the <reference>'\n"
            "\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- There is NO 'place' or 'put down' command alone — every action "
            "that moves an object MUST start with picking it up.\n"
            "- The robot cannot push, slide, rotate in-hand, pour, or do any "
            "action other than pick-and-place.\n"
            "- Each instruction must refer to exactly ONE object to pick.\n"
            "- Be explicit about colors, sizes, or positions when multiple "
            "similar objects are present (e.g. 'the red block', "
            "'the left can', 'the top block on the stack').\n"
            "- Decompose complex goals into a sequence of individual "
            "pick-and-place steps."
        ),
    }

    def _build_system_prompt(self) -> str:
        """Construct the system prompt, optionally with VLA capabilities."""
        base = (
            "You are a robot task planner. Given the overall goal, "
            "the history of past checkpoints, and the current scene, "
            "predict the next step the robot should execute."
        )
        caps = self.config.vla_capabilities
        if not caps:
            return base

        # Look up preset or use verbatim
        caps_text = self._VLA_CAPABILITIES_PRESETS.get(caps, caps)
        return f"{base}\n\n{caps_text}"

    # ------------------------------------------------------------------
    # VLM infrastructure (mirrors SubgoalBaseStrategy verbatim so that
    # NextGoalStrategy is fully self-contained)
    # ------------------------------------------------------------------

    @property
    def _use_ember(self) -> bool:
        """Local Ember VLM backend is not bundled in the public release."""
        return False

    @property
    def client(self):
        """Lazy-initialised OpenAI-compatible client."""
        if self._client is None:
            import openai
            kwargs: dict = {}
            if self.config.vlm_base_url:
                kwargs["base_url"] = self.config.vlm_base_url
            if self.config.vlm_api_key:
                kwargs["api_key"] = self.config.vlm_api_key
            elif os.environ.get("VLM_API_KEY"):
                kwargs["api_key"] = os.environ["VLM_API_KEY"]
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _vlm_call(
        self,
        system_prompt: str,
        user_content: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        """Single VLM call via the OpenAI-compatible backend."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        from vlm_orchestrator.vlm import chat_create
        response = chat_create(
            self.client,
            model=self.config.vlm_model,
            temperature=self.config.vlm_temperature,
            max_tokens=max_tokens or self.config.vlm_max_tokens,
            messages=messages,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "NextGoal: VLM response truncated "
                f"(max_tokens={max_tokens or self.config.vlm_max_tokens})"
            )
        usage = response.usage
        self._last_api_usage = {
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
        }
        content = choice.message.content
        return content.strip() if content is not None else ""

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _is_new_episode(self, obs: dict, state: SessionState) -> bool:
        """Episode detection that trusts the proxy's episode_id.

        The base-class implementation uses prompt comparison, which fires
        false positives on orchestrator flush re-infers: after a next-goal
        prediction the proxy sends ``orchestrator_instruction`` (= VLM
        instruction) in its response, robolab re-infers with that as the
        prompt, and the base class sees ``current != state.original_instruction``
        and wrongly starts a new episode — resetting the chunk counter and
        overwriting ``state.original_instruction`` on every flush.

        Fix: when running under the proxy (``state.episode_id > 0``), use
        only the proxy-managed ``episode_id`` to detect boundaries.  The
        proxy never increments ``episode_id`` on flush re-infers, only on
        genuine new-task boundaries.  Falls back to the base-class logic
        when running without the proxy (tests / standalone).
        """
        ep_id = state.episode_id
        last_ep = getattr(self, "_last_seen_episode_id", 0)

        if ep_id > 0:
            # Proxy-driven: trust episode_id exclusively.
            is_new = state.infer_count == 0 or ep_id != last_ep
        else:
            # No proxy — fall back to prompt comparison.
            current = self.ctx.get_prompt(obs)
            is_new = (
                state.infer_count == 0
                or current != state.original_instruction
            )
            if is_new:
                state.episode_id = 1

        if is_new:
            self._last_seen_episode_id = state.episode_id
        return is_new

    def process(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        prompt = self.ctx.get_prompt(obs)
        if prompt is None:
            return obs, state
        if self._is_new_episode(obs, state):
            return self._on_new_episode(obs, state, prompt)
        return self._on_step(obs, state)

    def _on_new_episode(
        self, obs: dict, state: SessionState, prompt: str
    ) -> tuple[dict, SessionState]:
        """Reset per-episode state; fire an initial VLM prediction immediately."""
        state.original_instruction = prompt
        state.rewritten_instruction = prompt

        self._history_images = []
        self._all_history_images: list[np.ndarray] = []  # unbounded, for disk saving
        self._step_at_last_check = state.episode_step

        image = self.ctx.get_vlm_image(obs)
        if image is not None:
            state.initial_image = image.copy()

        logger.info(
            f"NextGoal episode {state.episode_id}: \"{prompt[:80]}\""
        )

        # Fire an initial prediction immediately — the first VLA action chunk
        # should already use a VLM-derived instruction, not the raw task string.
        # The episode-start log entry is written inside _run_prediction so it
        # carries the full VLM output (raw response, latency, etc.).
        if image is not None:
            obs, state = self._run_prediction(obs, state, image, is_episode_start=True)
        else:
            # No image available yet — log a minimal start entry.
            state.log({
                "type": "next_goal_episode_start",
                "instruction": prompt,
                "template": self.config.template_name,
                "check_interval": self.config.check_interval,
            })

        return obs, state

    def _on_step(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        """Apply current instruction; fire VLM prediction every check_interval sim steps."""
        if (state.episode_step - self._step_at_last_check
                < self.config.check_interval):
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(obs, state.rewritten_instruction)
            return obs, state

        # — Check interval reached —
        self._step_at_last_check = state.episode_step

        image = self.ctx.get_vlm_image(obs)
        if image is None:
            logger.warning("NextGoal: no image available, skipping prediction")
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(obs, state.rewritten_instruction)
            return obs, state

        return self._run_prediction(obs, state, image)

    def _run_prediction(
        self, obs: dict, state: SessionState, image: np.ndarray,
        is_episode_start: bool = False,
    ) -> tuple[dict, SessionState]:
        """Fire one VLM next-goal prediction and update obs/state.

        Builds the prompt from current history + image, calls the VLM,
        parses the result, appends image to history, and sets the new
        instruction on obs and state.  On VLM failure, keeps the current
        instruction unchanged.
        """
        # Build prompt text.  The template describes image layout based on
        # counts; text history is left empty because checkpoint images are
        # the primary history signal.
        num_history = len(self._history_images)
        prompt_text = self._template_fn(
            goal=state.original_instruction,
            subtask_candidates=self._subtask_candidates or None,
            history_text="",
            num_history_images=num_history,
            num_current_images=1,
            task_type=self.config.task_type,
        )

        # user_content: history images (oldest→newest) + current image + text
        user_content: list[dict] = []
        for hist_img in self._history_images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_b64(hist_img)}"
                },
            })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_b64(image)}"
            },
        })
        user_content.append({"type": "text", "text": prompt_text})

        # Save images to disk for inspection (only when episode_log_dir is set).
        self._save_vlm_images(state, self._history_images, self._all_history_images, image)

        # VLM call
        system_prompt = self._build_system_prompt()
        t0 = time.time()
        try:
            raw = self._vlm_call(
                system_prompt,
                user_content,
            )
            elapsed = time.time() - t0
        except Exception as e:
            import traceback
            err_str = traceback.format_exc()
            logger.warning(
                f"NextGoal: VLM call failed: {e}; keeping current instruction\n{err_str}"
            )
            state.log({
                "type": "next_goal_vlm_error",
                "is_episode_start": is_episode_start,
                "error": str(e),
                "traceback": err_str,
                "step_count": state.infer_count,
                "vlm_model": self.config.vlm_model,
                "vlm_base_url": self.config.vlm_base_url,
                **({"template": self.config.template_name,
                    "check_interval": self.config.check_interval,
                    "original_instruction": state.original_instruction}
                   if is_episode_start else {}),
            })
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(obs, state.rewritten_instruction)
            return obs, state

        # Parse response
        fallback = state.rewritten_instruction or state.original_instruction or ""
        instruction, used_fallback, metadata = self._parser_fn(
            raw,
            subtask_candidates=self._subtask_candidates or None,
            fallback=fallback,
        )

        # Append current image to history AFTER the call so the next
        # prediction sees this checkpoint as part of the history.
        img_copy = image.copy()
        self._history_images.append(img_copy)
        self._all_history_images.append(img_copy)  # unbounded, for disk saving
        if (self.config.max_history_images > 0
                and len(self._history_images) > self.config.max_history_images):
            self._history_images.pop(0)

        # Update instruction
        state.rewritten_instruction = instruction
        state.flush_actions = True
        obs = self.ctx.set_prompt(obs, instruction)

        # Expose to video annotator (rendered as bottom-banner overlay)
        state.vlm_check_result = {"action": "next_goal", "instruction": instruction}
        state.vlm_check_infer_step = state.infer_count

        logger.info(
            f"  NextGoal ({elapsed:.1f}s): \"{instruction[:80]}\" "
            f"[history={len(self._history_images)}, fallback={used_fallback}]"
        )
        log_entry: dict = {
            "type": "next_goal_episode_start" if is_episode_start else "next_goal_check",
            "instruction": instruction,
            "used_fallback": used_fallback,
            "vlm_latency_s": round(elapsed, 2),
            "history_len": len(self._history_images),
            "step_count": state.infer_count,
            "vlm_raw": raw,
            **(metadata or {}),
        }
        if is_episode_start:
            log_entry["original_instruction"] = state.original_instruction
            log_entry["template"] = self.config.template_name
            log_entry["check_interval"] = self.config.check_interval
        state.log(log_entry)

        # Write per-call log for the API backend.
        if not self._use_ember:
            self._write_vlm_log(state, user_content, raw, elapsed)

        return obs, state

    def _save_vlm_images(
        self,
        state: SessionState,
        vlm_history: list[np.ndarray],
        all_history: list[np.ndarray],
        current_image: np.ndarray,
    ) -> None:
        """Save the images for this prediction step.

        Layout::

            episode_log_dir/
              vlm_images/
                step_0000/
                  current.jpg          ← current scene (sent to VLM)
                  vlm_history_00.jpg   ← images actually sent to VLM (rolling window)
                  vlm_history_01.jpg
                  all_history_00.jpg   ← full unbounded history (all past checkpoints)
                  all_history_01.jpg
                  ...
        """
        if not state.episode_log_dir:
            return
        from PIL import Image as PILImage

        step_dir = os.path.join(
            state.episode_log_dir,
            "vlm_images",
            f"step_{state.infer_count:04d}",
        )
        os.makedirs(step_dir, exist_ok=True)
        try:
            PILImage.fromarray(current_image).save(
                os.path.join(step_dir, "current.jpg")
            )
            for i, img in enumerate(vlm_history):
                PILImage.fromarray(img).save(
                    os.path.join(step_dir, f"vlm_history_{i:02d}.jpg")
                )
            for i, img in enumerate(all_history):
                PILImage.fromarray(img).save(
                    os.path.join(step_dir, f"all_history_{i:02d}.jpg")
                )
        except Exception as e:
            logger.warning(f"NextGoal: failed to save VLM images: {e}")

    def _write_vlm_log(
        self,
        state: SessionState,
        user_content: list[dict],
        response: str,
        elapsed: float,
    ) -> None:
        """Append one VLM call record to vlm_calls.jsonl in the episode dir."""
        import json
        from datetime import datetime, timezone

        if not state.episode_log_dir:
            return

        self._call_count += 1

        user_text_parts = []
        n_images = 0
        for item in user_content:
            if item.get("type") == "text":
                user_text_parts.append(item["text"])
            elif item.get("type") == "image_url":
                n_images += 1

        usage = self._last_api_usage or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        record = {
            "call_id": self._call_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.vlm_model,
            "user_text": "\n".join(user_text_parts),
            "n_images": n_images,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed, 2),
        }
        if input_tokens and output_tokens:
            record["tok_per_s"] = round(output_tokens / max(elapsed, 0.01), 1)

        log_path = os.path.join(state.episode_log_dir, "vlm_calls.jsonl")
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"NextGoal: failed to write vlm_calls.jsonl: {e}")
