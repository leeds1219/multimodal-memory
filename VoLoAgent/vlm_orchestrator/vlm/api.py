# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM backends for instruction rewriting.

Also hosts shared VLM utility functions (:func:`encode_image_b64`,
:func:`parse_json`) imported throughout the codebase.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Shared VLM utilities
# --------------------------------------------------------------------------- #


# Models that reject `temperature` (deprecated by the provider). Includes
# OpenAI reasoning models AND newer Claude (≥ 4.7).
_NO_TEMPERATURE_RE = re.compile(
    r"/(gpt-5(?:\.\d+)?|o[134]|(?:bedrock-)?claude-opus-4-(?:[7-9]|\d\d))(?:[-/]|$)"
)
# Subset that accepts the OpenAI `reasoning_effort` parameter — gpt-5+, o-series.
# Claude 4.7+ rejects this param (uses its own `thinking` config instead).
_OPENAI_REASONING_RE = re.compile(r"/(gpt-5(?:\.\d+)?|o[134])(?:[-/]|$)")


def is_reasoning_model(model: str) -> bool:
    """Whether the model rejects ``temperature`` (covers gpt-5+, o-series,
    Claude Opus 4.7+)."""
    return bool(_NO_TEMPERATURE_RE.search(model))


def chat_create(client, *, model: str, messages, temperature: float = 0.0,
                max_tokens: int | None = None, **extra):
    """Wrapper around `client.chat.completions.create` that picks the right
    parameter set for reasoning vs standard chat models. All call sites should
    use this so a single check covers every VLM call path.

    Categories:
      - **standard chat** (Claude ≤ 4.6, GPT-4*): ``temperature`` + ``max_tokens``.
      - **OpenAI reasoning** (gpt-5+, o1/o3/o4): no ``temperature``;
        ``max_completion_tokens`` (×4 of caller's max_tokens to leave room
        for invisible reasoning tokens) + ``reasoning_effort=low`` (or
        ``medium`` for ``-pro``).
      - **Claude reasoning** (Opus 4.7+): no ``temperature``; ``max_tokens``
        as given. No reasoning_effort (rejected; use Anthropic's
        ``thinking`` config separately if desired).

    Caller can override any of these by passing the param in **extra.
    """
    kwargs = dict(model=model, messages=messages, **extra)
    if is_reasoning_model(model):
        is_openai_reasoning = bool(_OPENAI_REASONING_RE.search(model))
        if is_openai_reasoning:
            if max_tokens is not None and "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = max(max_tokens * 4, 2048)
            if "reasoning_effort" not in kwargs:
                kwargs["reasoning_effort"] = "medium" if "-pro" in model else "low"
        else:
            # Claude 4.7+: just skip temperature; max_tokens passes through.
            if max_tokens is not None and "max_tokens" not in kwargs:
                kwargs["max_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
    return client.chat.completions.create(**kwargs)


def encode_image_b64(image: np.ndarray) -> str:
    """Encode an RGB uint8 numpy array as a base64 JPEG string."""
    pil = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json(text: str) -> dict:
    """Extract and parse JSON from a VLM response.

    Handles markdown code fences, surrounding prose, and nested braces.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from: {text[:200]}")


# --------------------------------------------------------------------------- #
#  Prompt: direct rewrite (no reasoning trace)
# --------------------------------------------------------------------------- #
REWRITE_SYSTEM_PROMPT = """\
You are a vision-language assistant helping rewrite vague robot manipulation instructions \
into specific ones that a Vision-Language-Action (VLA) policy can execute.

You will see the robot's current camera view of a tabletop scene and a task instruction. \
Rewrite the instruction following these rules:

FORMAT RULES:
- Start with an action verb: "Pick up", "Grasp", "Place", "Put", "Take", or "Remove".
- Keep it to 1-2 concise sentences, roughly 10-25 words.
- Use simple, direct language. No reasoning, no "identify", no "compare".
- For multi-step tasks, connect with "then" or "and": "Pick up X and place it in Y, then pick up Z and place it in Y."

OBJECT GROUNDING (use the scene image):
- Name each object by its color and type: "the yellow banana", "the red bowl", "the orange pumpkin".
- If there are multiple similar objects, distinguish by size or position: "the larger pumpkin", "the bowl on the left side".
- Count objects when relevant: "the two sauce bottles", "the three bananas".

PLACEMENT TARGET (critical — always specify WHERE):
- Look at the scene image for containers: bowls, bins, crates, plates, boxes, shelves.
- "Put away", "clean up", "toss", "throw away" → place into the nearest visible container (bin, bowl, crate).
- "Stack X on Y" → place X on top of Y.
- "Put X on the plate/shelf" → place onto that surface.
- Never say "place it to the side" or "place it away" — always name the specific target container or surface you see in the image.

Output ONLY the rewritten instruction. No quotes, no explanation.
"""

# --------------------------------------------------------------------------- #
#  Prompt: chain-of-thought — VLM reasons about the scene, then writes the
#  instruction on a clearly-delimited final line.
# --------------------------------------------------------------------------- #
COT_REWRITE_SYSTEM_PROMPT = """\
You are a vision-language assistant helping a robot arm. You will see the robot's camera \
view of a tabletop scene and a vague task instruction. Your job is to reason about the \
scene and then produce a single, specific instruction the robot can execute.

Think step by step:

STEP 1 — SCENE INVENTORY:
List every object you see on the table. For each object give its color, approximate size, \
and common name. Be precise — a Rubik's cube is NOT a "red cube"; it is a "multicolored \
Rubik's cube". A rectangular open-top container is a "bin", NOT a "bowl".

STEP 2 — IDENTIFY CONTAINERS / RECEPTACLES:
Which objects could serve as placement targets? Bins and crates are large, open-top, \
rectangular containers. Bowls are round and concave. Do NOT confuse them.

STEP 3 — INTERPRET THE INSTRUCTION:
Given the vague instruction, decide:
  • Which object(s) should be picked up?
  • Where should they be placed? (Always a specific container or surface from Step 2.)

STEP 4 — WRITE THE INSTRUCTION:
Write a concise robot instruction (10-25 words). Rules:
  • Start with "Pick up" or "Grasp".
  • Name objects by common name + color (e.g., "the yellow banana", "the grey bin").
  • For multi-step: "Pick up X and place it in Y, then pick up Z and place it in Y."
  • NEVER say "place it to the side" or "place it away".

Output your reasoning for Steps 1-3, then on the very last line write ONLY:
INSTRUCTION: <your final instruction>
"""

# --------------------------------------------------------------------------- #
#  Prompt: self-verification — VLM writes, critiques, and optionally revises.
# --------------------------------------------------------------------------- #
VERIFY_REWRITE_SYSTEM_PROMPT = """\
You are a vision-language assistant helping a robot arm execute tabletop manipulation tasks.

You will see the robot's camera view and a vague task instruction. Perform the following:

PHASE 1 — DRAFT:
Write a specific robot instruction based on the scene and vague instruction.
Rules: start with "Pick up" or "Grasp"; name objects by color and common name; \
always specify the target container or surface visible in the image; 10-25 words.

PHASE 2 — SELF-CHECK:
Critique your draft. Ask yourself:
  • Did I name each object correctly? (A Rubik's cube is NOT a "red cube".)
  • Is the placement target actually a container in the scene? (A bin is rectangular and open-top; a bowl is round.)
  • Did I mention the right number of objects?
  • Is the instruction concise (10-25 words)?

PHASE 3 — FINAL:
If your self-check found any problems, write a corrected instruction.

On the very last line, write ONLY:
INSTRUCTION: <your final instruction>
"""

# --------------------------------------------------------------------------- #
#  Prompt V2: CoT + scene-aware strategy (clutter, ordering, approach)
# --------------------------------------------------------------------------- #
COT_STRATEGY_PROMPT = """\
You are a vision-language assistant helping a robot arm. The robot is a Franka Panda \
with a parallel jaw gripper on a tabletop. You will see its camera view and a vague \
task instruction. Your job is to reason about the scene and produce a specific, \
executable instruction.

Think step by step:

STEP 1 — SCENE INVENTORY:
List every object on the table with color, size, and common name.
Be precise:
  • A Rubik's cube is a "multicolored Rubik's cube", NOT a "red cube".
  • Name containers by their specific type based on shape: a "bowl" is round and concave; \
    a "bin" is rectangular and open-top; a "plate" is flat and round; a "tray" is flat \
    and rectangular; a "crate" is a deep rectangular box. Always use the specific type — \
    never use the generic word "container".

STEP 2 — IDENTIFY TARGET OBJECTS AND DESTINATION:
Which objects need to be picked up? Where should they go? \
(Always a specific container or surface visible in the scene.)

STEP 3 — ASSESS DIFFICULTY:
Look at the scene carefully:
  • Is the target object surrounded by clutter? Are other objects touching it or very close?
  • Is there more than one target object? Which is easiest to reach (most isolated)?
  • Is the target object small, round, or slippery — hard for a parallel gripper?
  • Does the task require implicit reasoning (e.g., size comparison between objects)?

STEP 4 — PLAN STRATEGY:
Based on Step 3, decide the best approach:
  • If clutter blocks the target: instruct robot to first push or move the blocking \
    object out of the way, THEN pick up the target.
  • If multiple objects: start with the most accessible one first.
  • If the target is near the edge or in a tight spot: describe its location precisely \
    (e.g., "the red block between the two cubes, closer to the bin").
  • If size comparison needed: resolve it — directly name the correct object by its \
    visual properties (larger, taller, etc.) instead of asking the robot to "identify" it.

STEP 5 — WRITE THE INSTRUCTION:
Write a concise robot instruction (10-30 words). Rules:
  • Start with "Pick up", "Grasp", "Push", or "Move".
  • Name objects by common name + color.
  • For multi-step: "First pick up X and place it in Y, then pick up Z and place it in Y."
  • If clutter is an issue, prepend: "First push [obstacle] aside, then ..."
  • NEVER say "place it to the side" or "place it away" or "identify".

Output your reasoning for Steps 1-4, then on the very last line write ONLY:
INSTRUCTION: <your final instruction>
"""

# --------------------------------------------------------------------------- #
#  Prompt V3: Minimalist — short, DROID-style instruction; trust the policy's
#  visual grounding and keep text as simple as possible.
# --------------------------------------------------------------------------- #
MINIMALIST_PROMPT = """\
You help rewrite vague robot instructions into short, clear ones.

Look at the scene image. Rewrite the instruction in at most 12 words. Rules:
- Name the target object(s) by color and common name.
- Name the target container (bowl, bin, crate) by color.
- Use simple verbs: "put", "place", "pick up".
- For multi-object tasks, just say "put [objects] in [container]".
- Do NOT use the word "identify", "carefully", or "approach".
- Do NOT list objects one-by-one. Keep it natural and short.
- A Rubik's cube is a "Rubik's cube", not a "red cube".
- A rectangular open-top container is a "bin", not a "bowl".

Output ONLY the instruction. No quotes, no explanation.
"""

# --------------------------------------------------------------------------- #
#  Prompt V4: Scene-aware planner — analyze clutter, then give step-by-step
#  instruction that handles obstacles.
# --------------------------------------------------------------------------- #
PLANNER_PROMPT = """\
You are a robot task planner. You see a tabletop from the robot's camera and a vague \
task instruction. The robot has a parallel-jaw gripper.

Analyze the scene and output a step-by-step instruction the robot can follow.

ANALYSIS (think about these but only output the final instruction):
1. What objects need to be manipulated?
2. What is the destination (bin, bowl, plate)?
3. Are the target objects blocked by other objects? If yes, the robot should \
   push the blocking object away first.
4. For multiple targets: which object is most isolated / easiest to grasp? Do that first.
5. Is an object too large to grasp from the side? Then describe grasping from top.

OUTPUT RULES:
- Write one instruction with sequential steps connected by "then".
- Start each step with an action verb: "push", "pick up", "place", "grasp".
- Name every object by color + type + location if needed.
- Name the container by color + type (e.g., "the grey bin", "the blue crate").
- Keep total length under 35 words.
- A Rubik's cube = "Rubik's cube", NOT "red cube".
- Rectangular open-top container = "bin" or "crate", NOT "bowl".

Output ONLY the instruction. No reasoning, no quotes.
"""

# --------------------------------------------------------------------------- #
#  Prompt V5: "hint" — trust the vague instruction for object names,
#  use the VLM ONLY to identify the placement container from the image.
#  Designed for low-resolution (224×224) images where object details are hard
#  to see but container shapes (bin=rectangular, bowl=round) are still visible.
# --------------------------------------------------------------------------- #
HINT_PROMPT = """\
You help rewrite vague robot instructions. The camera image is low resolution, so \
do NOT try to identify objects in detail — trust the object names from the original \
instruction instead.

Your job is simple:
1. Read the original instruction to understand WHAT objects to manipulate.
2. Look at the image ONLY to identify the placement CONTAINER — is it a bowl \
   (round, concave), a bin (rectangular, open-top), a crate, or a plate?
3. Note the container's COLOR.

Then rewrite the instruction using:
- The object names from the original instruction (keep them as-is, do NOT rename objects)
- The container you identified from the image (by color + type)

Keep it under 20 words. Use simple verbs: "pick up", "put", "place".
For "put away" / "clean up": the destination is the container you see in the image.
For "stack X on Y": just clarify which goes on top.
For multi-step: use "then" to connect steps.

Output ONLY the instruction. No quotes, no explanation.
"""


class VLMBackend(ABC):
    """Base class for VLM backends."""

    @abstractmethod
    def rewrite_instruction(
        self,
        instruction: str,
        image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        """Examine the scene image and possibly rewrite the instruction.

        Args:
            instruction: The original task instruction.
            image: RGB uint8 image array (H, W, 3) from the robot's primary camera.
            extra_images: Optional additional camera views (e.g. wrist camera).
                Each is an RGB uint8 array (H, W, 3).

        Returns:
            The (possibly rewritten) instruction string.
        """


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic markers (``**``, ``*``, ``__``)."""
    import re
    return re.sub(r"\*{1,2}|_{1,2}", "", text)


def _extract_instruction_tag(text: str) -> str:
    """Extract the instruction after the last ``INSTRUCTION:`` tag.

    Falls back to the last non-empty line if the tag is missing.
    Handles markdown formatting (e.g. ``**INSTRUCTION:**``).
    """
    for line in reversed(text.strip().splitlines()):
        cleaned = _strip_markdown(line.strip())
        if cleaned.upper().startswith("INSTRUCTION:"):
            result = cleaned.split(":", 1)[1].strip()
            # Strip quotes
            if (result.startswith('"') and result.endswith('"')) or (
                result.startswith("'") and result.endswith("'")
            ):
                result = result[1:-1]
            return result
    # Fallback: last non-empty line
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            if (line.startswith('"') and line.endswith('"')) or (
                line.startswith("'") and line.endswith("'")
            ):
                line = line[1:-1]
            return line
    return text.strip()


class OpenAIVLM(VLMBackend):
    """VLM backend using OpenAI's vision models (GPT-4o, etc.).

    ``strategy`` controls which prompt template is used:

    * ``"direct"`` — single-shot rewrite (default, fastest).
    * ``"cot"``    — chain-of-thought: VLM inventories the scene, reasons,
                     then writes the instruction on a tagged final line.
    * ``"verify"`` — draft → self-critique → revise in a single call.
    """

    STRATEGY_PROMPTS = {
        "direct": REWRITE_SYSTEM_PROMPT,
        "cot": COT_REWRITE_SYSTEM_PROMPT,
        "verify": VERIFY_REWRITE_SYSTEM_PROMPT,
        "cot_strategy": COT_STRATEGY_PROMPT,
        "minimalist": MINIMALIST_PROMPT,
        "planner": PLANNER_PROMPT,
        "hint": HINT_PROMPT,
    }

    # Strategies that use reasoning traces need more tokens
    _HIGH_TOKEN_STRATEGIES = {"cot", "verify", "cot_strategy"}
    _DEFAULT_MAX_TOKENS = {
        "direct": 300,
        "minimalist": 200,
        "planner": 400,
        "cot": 1200,
        "verify": 800,
        "cot_strategy": 1500,
        "hint": 200,
    }

    def __init__(
        self,
        model: str = "YOUR_VLM_MODEL",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        strategy: str = "direct",
    ):
        import openai
        import os

        if strategy not in self.STRATEGY_PROMPTS:
            raise ValueError(
                f"Unknown strategy {strategy!r}; choose from {list(self.STRATEGY_PROMPTS)}"
            )
        self.strategy = strategy

        kwargs: dict = {}
        if base_url:
            kwargs["base_url"] = base_url
        # Resolve API key: explicit arg > VLM_API_KEY > OPENAI_API_KEY
        if api_key:
            kwargs["api_key"] = api_key
        elif os.environ.get("VLM_API_KEY"):
            kwargs["api_key"] = os.environ["VLM_API_KEY"]
        # else: fall through to openai lib's own OPENAI_API_KEY lookup
        self.client = openai.OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else self._DEFAULT_MAX_TOKENS.get(strategy, 300)
        self.system_prompt = system_prompt or self.STRATEGY_PROMPTS[strategy]

    # ------------------------------------------------------------------ #

    def _call_vlm(
        self,
        instruction: str,
        image_b64: str,
        extra_images_b64: list[str] | None = None,
    ) -> str:
        """Single VLM API call; returns raw response text."""
        # Build user content: text preamble + image(s)
        has_extra = extra_images_b64 and len(extra_images_b64) > 0
        if has_extra:
            n_total = 1 + len(extra_images_b64)
            text_preamble = (
                f"You have {n_total} camera views:\n"
                f"1. First image: External camera — shows the FULL scene "
                f"(all objects and containers on the table)\n"
            )
            for i in range(len(extra_images_b64)):
                text_preamble += (
                    f"{i + 2}. Image {i + 2}: Wrist camera — shows a CLOSE-UP "
                    f"of some objects near the gripper\n"
                )
            text_preamble += (
                "\nIMPORTANT:\n"
                "- The external camera is the PRIMARY view — it shows ALL objects "
                "and containers.\n"
                "- The wrist camera shows only a PARTIAL close-up — use it only to "
                "refine colors or shapes of objects you already identified in the "
                "external view.\n"
                "- The wrist camera can be misleading about object type due to "
                "extreme close-up angle. If the instruction names a specific object "
                "type (e.g., \"blocks\", \"banana\"), keep that type — do NOT "
                "rename it based on the wrist view. Always add colors and details "
                "from the images.\n\n"
                f'Scene instruction: "{instruction}"'
            )
        else:
            text_preamble = f'Scene instruction: "{instruction}"'

        user_content: list[dict] = [{"type": "text", "text": text_preamble}]

        # Primary image
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })

        # Extra images (wrist camera, etc.)
        if extra_images_b64:
            for extra_b64 in extra_images_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{extra_b64}"},
                })

        response = chat_create(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #

    def rewrite_instruction(
        self,
        instruction: str,
        image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        image_b64 = encode_image_b64(image)
        extra_b64 = (
            [encode_image_b64(img) for img in extra_images]
            if extra_images
            else None
        )
        raw = self._call_vlm(instruction, image_b64, extra_b64)

        if self.strategy == "direct":
            rewritten = raw
            # Strip surrounding quotes
            if (rewritten.startswith('"') and rewritten.endswith('"')) or (
                rewritten.startswith("'") and rewritten.endswith("'")
            ):
                rewritten = rewritten[1:-1]
        else:
            # cot / verify — extract from INSTRUCTION: tag
            rewritten = _extract_instruction_tag(raw)
            logger.debug("VLM reasoning trace:\n%s", raw)

        return rewritten


class PassthroughVLM(VLMBackend):
    """No-op backend that returns the instruction unchanged."""

    def rewrite_instruction(
        self,
        instruction: str,
        image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        return instruction


class ReplayVLM(VLMBackend):
    """Replay pre-computed rewrites from a JSON mapping file.

    The JSON should map original instructions to rewritten ones, either as::

        {"original instruction text": "rewritten text", ...}

    or as a task-keyed dict (auto-detected)::

        {"TaskName": {"rewritten": "text", "original_instruction": "orig", ...}, ...}

    Any instruction not found in the map is returned unchanged.
    """

    def __init__(self, rewrite_map_path: str):
        import json

        with open(rewrite_map_path) as f:
            raw = json.load(f)

        # Build instruction→rewrite lookup
        self._map: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(value, str):
                # Direct mapping: original → rewritten
                self._map[key] = value
            elif isinstance(value, dict) and "rewritten" in value:
                # Task-keyed: index by every original instruction variant
                rewritten = value["rewritten"]
                for orig_key in ("original_vague", "original_instruction", "original"):
                    if orig_key in value:
                        self._map[value[orig_key]] = rewritten
                # Also index by task name for fallback
                self._map[f"__task__{key}"] = rewritten

        logger.info(f"ReplayVLM loaded {len(self._map)} rewrite mappings from {rewrite_map_path}")

    def rewrite_instruction(
        self,
        instruction: str,
        image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        rewritten = self._map.get(instruction)
        if rewritten is not None:
            return rewritten
        # Fuzzy match: strip whitespace/case
        stripped = instruction.strip().lower()
        for orig, rw in self._map.items():
            if orig.strip().lower() == stripped:
                return rw
        logger.warning(f"ReplayVLM: no rewrite found for: \"{instruction[:80]}\"")
        return instruction
