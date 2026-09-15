# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class and utilities for subgoal-based strategies.

Provides shared VLM plumbing (client, message building, JSON parsing),
episode lifecycle (initial strategy, decompose, step, timeout, advance),
and recycling infrastructure.  Subclasses implement the actual prompts,
decomposition format, check logic, and any per-frame processing.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from vlm_orchestrator.failure_handlers.base import (
    HandlerResult,
    ACTION_CONTINUE,
    ACTION_GRASP,
    ACTION_NEXT,
    ACTION_PLACE,
    ACTION_REPLAN,
    STATUS_FAILURE,
)
from vlm_orchestrator.failure_handlers.instruction import generate_recovery
from vlm_orchestrator.failure_handlers.signal_detector import (
    CombinedDetector,
    ManipulationStatus,
)
from vlm_orchestrator.failure_handlers.gt_detector import (
    GTFailureDetector,
    GTFailureResult,
    GTFailureType,
)
from vlm_orchestrator.vlm import encode_image_b64, parse_json

from .base import OrchestrationStrategy, SessionState, StrategyContext

if TYPE_CHECKING:
    from vlm_orchestrator.failure_handlers.base import FailureHandler
    from vlm_orchestrator.hitl import HITLState

logger = logging.getLogger(__name__)

# ======================================================================
# Shared constants
# ======================================================================

MAX_RECYCLES = 3

# Failure types that strongly suggest a grasp problem.
_GRASP_FAILURE_TYPES = {
    "stall_never_gripped", "fumble", "stall_frozen",
    "no_progress",  # usually means the robot failed to grasp / lost the object
    "object_dropped",
}

# Verbs that indicate a subgoal involves grasping an object.
# Includes "put" — tasks like "put X on Y" require grasping X first.
_GRASP_VERBS = {"pick up", "grasp", "grab", "lift", "pick", "put", "place"}

# Prepositions that separate object from destination.
_DEST_PREPS = {
    " and place ",
    " and put ",
    " and drop ",
    " into ",
    " onto ",
    " in ",
    " on ",
    " to ",
}


def _is_grasp_subgoal(instruction: str, failure_reason: str = "") -> bool:
    """Return True if *instruction* involves grasping an object.

    Uses a lightweight heuristic — checks for grasp verbs in the
    instruction text and/or grasp-related failure signals.
    """
    low = instruction.lower()
    # Check instruction text
    for verb in _GRASP_VERBS:
        if verb in low:
            return True
    # Check failure signal
    ftype = failure_reason.split(":")[0].strip().lower()
    if ftype in _GRASP_FAILURE_TYPES:
        return True
    return False


def _extract_target_object(instruction: str) -> str:
    """Extract the target object phrase from a pick-and-place instruction.

    Examples::

        "Pick up the red block and place it in the bin"  →  "red block"
        "Grab the mug"                                   →  "mug"
        "Lift the yellow banana and put it on the plate"  →  "yellow banana"

    Returns the full instruction as fallback (GDino can still try).
    """
    low = instruction.lower()

    # Find the verb start
    verb_end = -1
    for verb in sorted(_GRASP_VERBS, key=len, reverse=True):
        idx = low.find(verb)
        if idx >= 0:
            verb_end = idx + len(verb)
            break
    if verb_end < 0:
        return instruction  # fallback

    # Trim "the " after verb
    rest = instruction[verb_end:].strip()
    if rest.lower().startswith("the "):
        rest = rest[4:]
    elif rest.lower().startswith("a "):
        rest = rest[2:]
    elif rest.lower().startswith("an "):
        rest = rest[3:]

    # Cut at destination preposition
    rest_low = rest.lower()
    cut = len(rest)
    for prep in _DEST_PREPS:
        idx = rest_low.find(prep)
        if 0 < idx < cut:
            cut = idx

    result = rest[:cut].strip().rstrip(".,;")
    return result if result else instruction


def _make_dummy_classification(result: "HandlerResult"):
    """Build a minimal ClassificationResult for grasp-tool compatibility."""
    from vlm_orchestrator.failure_handlers.signal_detector import (
        ClassificationResult, SignalFeatures,
    )
    return ClassificationResult(
        status=ManipulationStatus.FAILURE,
        reason=result.reason,
        confidence=result.confidence,
        features=SignalFeatures(),
    )


# HITL: push a new camera frame to the browser every N process() calls.
# Each process() call corresponds to one action chunk (~8 env steps).
# A value of 1 means every chunk → ~1 image/sec, which gives the human
# near-real-time feedback without meaningful bandwidth cost.
HITL_IMAGE_INTERVAL = 1

DECOMPOSE_SHARED_PROMPT = """\
You are a vision-language assistant helping a robot arm plan tabletop \
manipulation tasks. You will see the robot's camera view and a task \
instruction. Produce an ordered list of subgoals the robot should \
execute one at a time.

Planning principles:
- **Atomic actions.** Each subgoal is one pick-and-place / push / \
  reorient / open / close action.
- **Implicit prerequisites.** Add steps the instruction omits when \
  physics requires them (clear an obstacle, reorient an object too large \
  for a side grasp, etc.).
- **Avoid unnecessary re-handling.** Where physics permit, place each \
  object directly in its final pose on the first lift, instead of staging \
  it at a temporary location.
- **Naming.** Refer to objects by colour + type ("the yellow banana", \
  "the grey bin"). For visually-identical duplicates use a generic phrase \
  ("a Rubik's cube"). Keep each instruction short (≤15 words) and start \
  with an action verb.
- **Ordering.** Mark `ordered: true` when a later step depends on an \
  earlier one; mark `false` when steps are independent (e.g. sorting many \
  items into one container).
- **Trivial tasks.** A single pick-and-place with no obstacles should \
  be one subgoal with `ordered: false`.\
"""

# VLABench variant: tells the VLM to preserve the original instruction's
# wording when the task is already a single clear action and to gate
# object-name expansion on visible scene ambiguity.  Used when the
# underlying VLA was fine-tuned on specific prompt phrasings (pi05-
# primitive-10task), where the orchestrator's paraphrasing creates
# distribution shift on otherwise in-distribution episodes.
DECOMPOSE_SHARED_PROMPT_VLABENCH = """\
You are a vision-language assistant helping a robot arm plan tabletop \
manipulation tasks. You will see the robot's camera view and a task \
instruction. Produce an ordered list of subgoals the robot should \
execute one at a time.

Planning principles:
- **Atomic actions.** Each subgoal is one pick-and-place / push / \
  reorient / open / close action.
- **Implicit prerequisites.** Add steps the instruction omits when \
  physics requires them (clear an obstacle, reorient an object too large \
  for a side grasp, etc.).
- **Avoid unnecessary re-handling.** Where physics permit, place each \
  object directly in its final pose on the first lift, instead of staging \
  it at a temporary location.
- **Naming.** Use the original instruction's object names by default. \
  Only expand to a more specific descriptor (e.g. "the red cola") when \
  the scene contains multiple visually-similar candidates that the \
  original name cannot distinguish.
- **Ordering.** Mark `ordered: true` when a later step depends on an \
  earlier one; mark `false` when steps are independent (e.g. sorting many \
  items into one container).
- **Trivial tasks.** When the original instruction is already a single \
  clear action (e.g. pick-and-place, insert, pour, open, activate), \
  emit it verbatim as one subgoal with `ordered: false`.\
"""

CHECK_DONE_SHARED_PROMPT = """\
Robot progress checker. Decide if the current pick-and-place step is \
done by comparing BEFORE and NOW images. NOW images may include visual \
hints (green border, dimming) marking the targeted object — use them \
when present.

Rules:
- "Object in container" = the object's region overlaps with or is inside \
  the container's region. Overlap is enough.
- "Object moved" = the object is in a clearly different position than in \
  BEFORE.
- If the object is no longer visible at its BEFORE position and the \
  target area looks different, the object likely moved there.

Answer {"done": true} if the object has reached the target.
Answer {"done": false} if the object is still at its original location \
or in the gripper.

Do NOT explain. Output ONLY the JSON.\
"""

# VLABench variant: same logic but generalised away from pick-and-place
# wording, so non-pick-place tasks (insert, pour, open, activate) aren't
# implicitly framed as if they were pick-and-place.
CHECK_DONE_SHARED_PROMPT_VLABENCH = """\
Robot progress checker. Decide if the current step is done by comparing \
BEFORE and NOW images. NOW images may include visual hints (green \
border, dimming) marking the targeted object — use them when present.

Rules:
- "Object in container" = the object's region overlaps with or is inside \
  the container's region. Overlap is enough.
- "Object moved" = the object is in a clearly different position than in \
  BEFORE.
- If the object is no longer visible at its BEFORE position and the \
  target area looks different, the object likely moved there.

Answer {"done": true} if the step's goal has been reached.
Answer {"done": false} if the step is still in progress (object still at \
original location, in the gripper, or target state not yet achieved).

Do NOT explain. Output ONLY the JSON.\
"""

# When --use-front-camera is active, the "Front camera" view is both
# left/right AND front/behind FLIPPED relative to the robot's actual
# perspective.  Rather than ask the VLM to maintain two reference
# frames (image frame for tool args + robot frame for subgoal text),
# we instruct it to describe targets by visual features (color, type,
# proximity to landmarks) and to AVOID directional words.  When the
# user's instruction does use a direction, the VLM resolves it to a
# feature-based phrase using the flip rules.
_FRONT_CAM_NOTE = """

NOTE on camera orientation: the "Front camera" view is BOTH L/R \
and front/behind FLIPPED from the robot's perspective. \
image-LEFT↔robot-RIGHT, image-TOP↔robot-BEHIND.

To avoid frame confusion, describe targets by VISUAL FEATURES \
(color, type, proximity to landmarks) in every output field — \
NOT by left/right/front/behind. Example: "the grey container next \
to the red block" instead of "the right bin". If the user's \
instruction uses a direction, first apply the flip to find which \
object they mean in the image, then re-describe by visual features."""


def _with_vla_front_cam_note(
    base_prompt: str, use_front_camera: bool,
) -> str:
    """Append ``_FRONT_CAM_NOTE`` to a VLA-bound system prompt
    when the front camera is active.  No-op otherwise.

    Function kept under its historical name for back-compat with
    call sites in subgoal.py; the note itself now instructs the VLM
    to use feature-based descriptions in ALL output fields (rather
    than maintaining separate robot-frame vs image-frame outputs).
    """
    if not use_front_camera:
        return base_prompt
    return base_prompt + _FRONT_CAM_NOTE


RECYCLE_SYSTEM_PROMPT = """\
You check whether a robot finished a pick-and-place task by comparing \
BEFORE and NOW images of the scene.

Rules:
- Only consider objects matching the task instruction. Ignore all others \
  (figurines, decorations, containers, robot arm, etc.).
- An object is DONE if it moved from its BEFORE table position into the \
  target container / stacking position.
- An object is REMAINING if it is still clearly visible on the table in \
  the NOW image, or if the physics engine reports it as regressed.
- Do NOT invent objects you cannot clearly see on the table in NOW.
- Rubik's cubes are standard multi-colored cubes — never name them by a \
  single face color. Just call them "Rubik's cube".
- If you are unsure whether an object is still on the table or already \
  in the correct position, assume it is DONE (be conservative — fewer \
  subgoals is better than hallucinated ones).
- **IMPORTANT**: When an ORIGINAL SUBGOAL LIST is provided, you MUST \
  select subgoals ONLY from that list. Do NOT rephrase, rewrite, or \
  invent new subgoal instructions. Return the exact text of the \
  original subgoals that still need to be completed.

You MUST respond with ONLY a single JSON object — no analysis, no \
bullet points, no markdown, no explanation before or after. Any \
non-JSON output will cause a system error.

JSON format (task not done):
{"remaining": ["green cube"], "subgoals": [{"instruction": "Pick up \
the green cube and place it in the bin", "target_object": "green \
cube"}], "done": false}

JSON format (task complete):
{"remaining": [], "subgoals": [], "done": true}\
"""

# VLABench variant: same recycle logic, generalised away from
# pick-and-place / "moved to container" framing so non-pick-place tasks
# (insert, pour, open, activate) aren't implicitly framed wrong.
RECYCLE_SYSTEM_PROMPT_VLABENCH = """\
You check whether a robot finished the current task by comparing BEFORE \
and NOW images of the scene.

Rules:
- Only consider objects / state changes matching the task instruction. \
  Ignore unrelated items (figurines, decorations, robot arm, etc.).
- A subgoal is DONE if its goal state is visibly achieved in NOW \
  (object placed, container opened, fluid poured, device activated, etc.).
- A subgoal is REMAINING if its goal state has not yet been reached, or \
  if the physics engine reports it as regressed.
- Do NOT invent objects or state changes you cannot clearly observe.
- Rubik's cubes are standard multi-colored cubes — never name them by a \
  single face color. Just call them "Rubik's cube".
- If you are unsure whether a subgoal is achieved, assume it is DONE \
  (be conservative — fewer subgoals is better than hallucinated ones).
- **IMPORTANT**: When an ORIGINAL SUBGOAL LIST is provided, you MUST \
  select subgoals ONLY from that list. Do NOT rephrase, rewrite, or \
  invent new subgoal instructions. Return the exact text of the \
  original subgoals that still need to be completed.

You MUST respond with ONLY a single JSON object — no analysis, no \
bullet points, no markdown, no explanation before or after. Any \
non-JSON output will cause a system error.

JSON format (task not done):
{"remaining": ["green cube"], "subgoals": [{"instruction": "Pick up \
the green cube and place it in the bin", "target_object": "green \
cube"}], "done": false}

JSON format (task complete):
{"remaining": [], "subgoals": [], "done": true}\
"""


# ======================================================================
# Base strategy
# ======================================================================


class SubgoalBaseStrategy(OrchestrationStrategy):
    """Base class for strategies that decompose tasks into subgoals.

    Subclasses **must** implement:

    * ``_decompose_and_setup(obs, state, instruction, image)``
      — Decompose the instruction, populate ``state.subgoals``, and
      perform any strategy-specific setup (e.g. GDino detection).
    * ``_check_subgoal(obs, state, image) -> (obs, state)``
      — VLM progress check for the current subgoal.  Handles
      advancement and recycling internally.
    * ``_apply_recycle_subgoals(obs, state, data) -> bool``
      — Apply parsed recycle response to state.  Return ``True``
      if new subgoals were set up.

    Optional hooks:

    * ``_on_frame(obs, state)`` — per-frame processing (default: no-op).
    * ``_on_subgoal_advanced(obs, state, idx)`` — after subgoal advance.
    * ``_post_process(obs, state) -> (obs, state)`` — after each step.
    * ``_should_check(state) -> bool`` — whether VLM checks are active.
    * ``_get_timeout(state) -> int`` — subgoal timeout in chunks.
    """

    # Minimum chunks on a subgoal before failure detection kicks in.
    # Needs enough data for meaningful signal features (speed, grip
    # transitions) but not so long that failures go unnoticed.
    # 10 chunks × 8 steps = 80 sim steps.
    MIN_STEPS_BEFORE_FAILURE_CHECK = 10

    # Maximum retries per subgoal before escalating to replan.
    MAX_FAILURE_RETRIES = 2

    # Default cap on grasp tool attempts per subgoal in VLM-handler /
    # template / retry modes.  ``grasp_first`` and ``vlm_grasp`` modes
    # are always exempt (they have their own natural backoff).
    # Overridable via constructor + CLI ``--max-grasp-attempts``.
    DEFAULT_MAX_GRASP_ATTEMPTS = 2

    def __init__(
        self,
        ctx: StrategyContext,
        initial_strategy: OrchestrationStrategy | None = None,
        failure_monitor: str | None = None,
        recovery_mode: str = "template",
        hitl_state: "HITLState | None" = None,
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
        super().__init__(ctx)
        self.initial_strategy = initial_strategy
        self._client = None
        self._recycle_count = 0
        self._original_subgoals: list[str] = []
        self._grasp_seg_mode = grasp_seg_mode
        self._grasp_topdown_threshold = grasp_topdown_threshold
        # Motion-planner kind for grasp/place trajectory segments; built
        # lazily on first use so the linear default carries no import cost.
        self._motion_planner_kind = motion_planner
        self._motion_planner_obj = None
        # Master switch for stack/no-stack placement semantics (CLI
        # --enable-stack-mode). When False the place tool's per-call
        # ``stack`` arg is ignored (historical grasp-consistent placement).
        self._stack_mode_enabled = bool(stack_mode_enabled)
        self._max_grasp_attempts: int = (
            max_grasp_attempts
            if max_grasp_attempts is not None
            else self.DEFAULT_MAX_GRASP_ATTEMPTS
        )
        # ``place_seg_mode is not None`` doubles as the on/off switch for
        # the place tool; ``None`` keeps the tool disabled.
        self._place_seg_mode: str | None = place_seg_mode
        self._env_mode = env_mode
        self._use_front_camera = use_front_camera

        # Trajectory collection for fine-tuning data generation
        self._trajectory_collector = None
        if collect_trajectories is not None:
            from vlm_orchestrator.utils.trajectory_collector import (
                CollectorConfig,
                TrajectoryCollector,
            )
            self._trajectory_collector = TrajectoryCollector(
                CollectorConfig(output_dir=collect_trajectories)
            )
            logger.info(
                f"Trajectory collection enabled: {collect_trajectories}"
            )

        # Recovery mode: "template" (free), "vlm" (VLM call), "human" (HITL)
        self._recovery_mode: str = recovery_mode

        # Human-in-the-loop state (None = disabled)
        self._hitl: "HITLState | None" = hitl_state

        # Failure monitoring (None = disabled)
        self._failure_monitor_mode = failure_monitor
        self._failure_handler: "FailureHandler | None" = None
        self._failure_detector: CombinedDetector | None = None
        self._gt_detector: GTFailureDetector | None = None
        self._gt_all_done: bool = False  # set when all subgoals complete
        self._failure_retries: int = 0  # retries on current subgoal
        self._grasp_attempts: int = 0   # grasp tool attempts on current subgoal

        if failure_monitor == "vlm":
            # VLM handler is created lazily in _create_vlm_handler()
            # because it needs self._vlm_call which requires self.client
            # (lazy-init OpenAI client).  The handler is created on the
            # first episode start.
            logger.info(
                f"VLM failure handler enabled: "
                f"recovery={recovery_mode}"
            )
        elif failure_monitor is not None:
            if failure_monitor.startswith("gt"):
                # GT-based failure detection: gt, gt_hitl, gt_vlm
                self._gt_detector = GTFailureDetector(
                    enabled_failure_types=gt_failure_types,
                )
                logger.info(
                    f"GT failure monitor enabled: mode={failure_monitor}, "
                    f"recovery={recovery_mode}, "
                    f"gt_failure_types={gt_failure_types or 'all'}"
                )
            else:
                # Signal-based failure detection
                from vlm_orchestrator.failure_handlers.signal_detector import get_signal_config
                signal_config = get_signal_config(env_mode)
                self._failure_detector = CombinedDetector(
                    mode=failure_monitor, window_size=80,
                    signal_config=signal_config,
                )
                logger.info(
                    f"Signal failure monitor enabled: mode={failure_monitor}, "
                    f"recovery={recovery_mode}"
                )

        if ctx.front_image_key:
            logger.info(
                f"Front camera enabled for VLM: {ctx.front_image_key}"
            )

    # ------------------------------------------------------------------
    # Motion planner (lazy creation, shared across grasp + place tools)
    # ------------------------------------------------------------------

    def _get_motion_planner(self):
        """Build the motion planner once and cache it.

        The ``linear`` default builds a trivial planner; ``curobo`` builds
        the remote client (which resolves the grasp-server URL lazily).
        """
        if self._motion_planner_obj is None:
            from vlm_orchestrator.motion import build_motion_planner
            self._motion_planner_obj = build_motion_planner(
                self._motion_planner_kind
            )
        return self._motion_planner_obj

    # ------------------------------------------------------------------
    # VLM failure handler (lazy creation)
    # ------------------------------------------------------------------

    def _create_vlm_handler(self) -> None:
        """Create the VLM failure handler (deferred until first use).

        Deferred because it depends on ``self._vlm_call`` and
        ``self.config`` which require the lazy-init OpenAI client.
        """
        from vlm_orchestrator.failure_handlers.vlm import VLMFailureHandler

        primary_label, extra_labels = self.ctx.vlm_camera_labels

        self._failure_handler = VLMFailureHandler(
            vlm_call_fn=self._vlm_call,
            image_builder_fn=self._build_check_message,
            get_vlm_image_fn=self.ctx.get_vlm_image,
            get_extra_images_fn=self.ctx.get_vlm_extra_images,
            check_interval=self.config.check_interval,
            recovery_mode=self._recovery_mode,
            primary_label=primary_label,
            extra_labels=extra_labels,
            use_front_camera=bool(
                getattr(self.ctx, "front_image_key", None)
            ),
            stack_mode_enabled=self._stack_mode_enabled,
        )
        logger.info(
            f"VLM failure handler created: "
            f"check_interval={self.config.check_interval}, "
            f"recovery_mode={self._recovery_mode}, "
            f"use_front_camera="
            f"{bool(getattr(self.ctx, 'front_image_key', None))}"
        )

    # ------------------------------------------------------------------
    # VLM client
    # ------------------------------------------------------------------

    @property
    def _use_ember(self) -> bool:
        """Local Ember VLM backend is not bundled in the public release —
        always use the OpenAI-compatible API backend."""
        return False

    @property
    def client(self):
        """Lazy-initialised OpenAI-compatible client."""
        if self._client is None:
            import openai
            import os

            kwargs: dict = {}
            if self.config.vlm_base_url:
                kwargs["base_url"] = self.config.vlm_base_url
            # Resolve API key: explicit config > VLM_API_KEY > OPENAI_API_KEY
            if self.config.vlm_api_key:
                kwargs["api_key"] = self.config.vlm_api_key
            elif os.environ.get("VLM_API_KEY"):
                kwargs["api_key"] = os.environ["VLM_API_KEY"]
            # else: fall through to openai lib's own OPENAI_API_KEY lookup
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _vlm_call(
        self,
        system_prompt: str,
        user_content: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        """Single VLM call via the OpenAI-compatible backend.

        *max_tokens* overrides the default ``config.vlm_max_tokens``
        for calls that need more room (e.g. recycle with many subgoals).
        """
        from vlm_orchestrator.vlm import chat_create
        response = chat_create(
            self.client,
            model=self.config.vlm_model,
            temperature=self.config.vlm_temperature,
            max_tokens=max_tokens or self.config.vlm_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning(
                "VLM response truncated by max_tokens "
                f"({max_tokens or self.config.vlm_max_tokens})"
            )
        return choice.message.content.strip()

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_image_message(
        text: str,
        image: np.ndarray,
        extra_images: list[np.ndarray] | None,
    ) -> list[dict]:
        """Build user-content list with text and image(s)."""
        has_extra = extra_images and len(extra_images) > 0
        if has_extra:
            preamble = (
                f"You have {1 + len(extra_images)} camera views.\n"
                f"Image 1: External camera (full scene).\n"
            )
            for i in range(len(extra_images)):
                preamble += (
                    f"Image {i + 2}: Wrist camera "
                    f"(close-up near gripper).\n"
                )
            preamble += (
                "\nUse the external camera as the primary view. "
                "The wrist camera shows a partial close-up.\n\n"
            )
            text = preamble + text

        content: list[dict] = [{"type": "text", "text": text}]

        def _add(img: np.ndarray) -> None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,"
                    f"{encode_image_b64(img)}"
                },
            })

        _add(image)
        if extra_images:
            for img in extra_images:
                _add(img)

        return content

    @staticmethod
    def _build_check_message(
        text: str,
        initial_image: np.ndarray | None,
        current_image: np.ndarray,
        extra_images: list[np.ndarray] | None,
        initial_extra_images: list[np.ndarray] | None = None,
        *,
        primary_label: str = "External camera",
        extra_labels: list[str] | None = None,
    ) -> list[dict]:
        """Build user-content with BEFORE and NOW images.

        Images are **interleaved** with their text labels so that
        VLMs using chat templates that reorder content (e.g. Qwen3-VL)
        still see each label adjacent to its image:

          [BEFORE label] [BEFORE img] [BEFORE extra label] [BEFORE extra img] ...
          [NOW label] [NOW img] [NOW extra label] [NOW extra img] ...
          [task text]

        *primary_label* and *extra_labels* control camera name strings
        (e.g. "Front camera", "External camera", "Wrist camera").

        Falls back to current-only if *initial_image* is ``None``.
        """
        if initial_image is None:
            return SubgoalBaseStrategy._build_image_message(
                text, current_image, extra_images,
            )

        has_now_extra = extra_images and len(extra_images) > 0
        has_before_extra = (
            initial_extra_images and len(initial_extra_images) > 0
        )

        content: list[dict] = []

        def _add_text(t: str) -> None:
            content.append({"type": "text", "text": t})

        def _add_img(img: np.ndarray) -> None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,"
                    f"{encode_image_b64(img)}"
                },
            })

        # BEFORE images — interleaved label + image
        _add_text(f"BEFORE — {primary_label} (scene at episode start):")
        _add_img(initial_image)
        if has_before_extra:
            for i, img in enumerate(initial_extra_images):
                label = (
                    extra_labels[i]
                    if extra_labels and i < len(extra_labels)
                    else "Wrist camera"
                )
                _add_text(f"BEFORE — {label}:")
                _add_img(img)

        # NOW images — interleaved label + image
        _add_text(f"NOW — {primary_label} (current scene):")
        _add_img(current_image)
        if has_now_extra:
            for i, img in enumerate(extra_images):
                label = (
                    extra_labels[i]
                    if extra_labels and i < len(extra_labels)
                    else "Wrist camera"
                )
                _add_text(f"NOW — {label}:")
                _add_img(img)

        # Task context at the end
        _add_text(text)

        return content

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def process(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        prompt = self.ctx.get_prompt(obs)
        if prompt is None:
            return obs, state

        if self._is_new_episode(obs, state):
            obs, state = self._on_new_episode(obs, state, prompt)
        else:
            obs, state = self._on_step(obs, state)

        return self._post_process(obs, state)

    def _on_new_episode(
        self, obs: dict, state: SessionState, prompt: str,
    ) -> tuple[dict, SessionState]:
        state.original_instruction = prompt
        state.subgoals = []
        state.current_subgoal_idx = 0
        state.step_at_last_check = state.episode_step
        state.step_at_subgoal_start = state.episode_step
        self._recycle_count = 0
        self._failure_retries = 0
        self._grasp_attempts = 0
        if self._failure_detector is not None:
            self._failure_detector.reset()
        if self._gt_detector is not None:
            self._gt_detector.reset()
        self._gt_all_done = False

        # Drop any grasp-tool state carried over from the previous
        # episode.  Without this, episodes 3+ can immediately resume
        # an interrupted grasp from episode 1's final state — the
        # robot tries to grasp before the VLA gets a single chunk.
        state.grasp_tool_active = False
        state.grasp_tool_phase = "idle"
        state.grasp_tool_target = ""
        if state.grasp_tool_executor is not None:
            try:
                state.grasp_tool_executor.reset()
            except Exception:
                pass

        # Mirror the same reset for the place tool (same reasoning).
        state.place_tool_active = False
        state.place_tool_phase = "idle"
        state.place_tool_target = ""
        state.place_tool_held_object = ""
        state.place_debug_image = None
        state.place_debug_label = ""
        state.place_debug_queue.clear()
        if state.place_tool_executor is not None:
            try:
                state.place_tool_executor.reset()
            except Exception:
                pass
        state.grasp_debug_queue.clear()
        state.grasp_debug_image = None
        state.grasp_debug_label = ""

        # VLM failure handler: create on first episode, reset on subsequent
        if (self._failure_monitor_mode == "vlm"
                and self._failure_handler is None):
            self._create_vlm_handler()
        if self._failure_handler is not None:
            self._failure_handler.on_episode_start(obs, state)

        # Attach trajectory collector to session state (proxy reads it)
        if self._trajectory_collector is not None:
            state.trajectory_collector = self._trajectory_collector

        # Grab the first frame as early as possible so HITL can show it.
        image = self.ctx.get_vlm_image(obs)

        # Notify HITL UI of new episode
        if self._hitl is not None:
            self._hitl.update_status(
                step_count=0,
                instruction=prompt,
                subgoals=[],
                subgoal_idx=0,
                failure_type="",
                gt_failure=None,
                status_message=f"New episode: {prompt[:60]}",
                is_active=True,
                is_paused=False,
            )
            self._hitl.episode_id = state.episode_id
            self._hitl.task_instruction = prompt

            # Push the very first frame so the human sees the scene
            # *before* any decomposition / VLM work happens.
            if image is not None:
                self._hitl.update_image(image)

        # Save initial images for before/after comparison
        if image is not None:
            state.initial_image = image.copy()
        extra = self.ctx.get_vlm_extra_images(obs)
        state.initial_extra_images = (
            [e.copy() for e in extra] if extra else None
        )

        # ---- HITL-driven episode start ----
        # In HITL mode the human writes the subgoal decomposition — no
        # VLM calls, no initial strategy.  Block until the human submits
        # subgoals through the UI (START_SUBGOAL action with instruction
        # text, one subgoal per line).
        if self._hitl is not None:
            obs, state = self._hitl_wait_for_subgoals(obs, state, prompt)
            return obs, state

        # ---- Autonomous (non-HITL) episode start ----
        # Run initial strategy (rewrite / adaptive) if provided
        if self.initial_strategy is not None:
            obs, state = self.initial_strategy.process(obs, state)
            prompt = self.ctx.get_prompt(obs) or prompt
            logger.info(
                f"  Initial strategy "
                f"({type(self.initial_strategy).__name__}) "
                f"-> \"{prompt}\""
            )
            # Reset bookkeeping the initial strategy may have touched
            state.subgoals = []
            state.current_subgoal_idx = 0
            state.step_at_last_check = state.episode_step
            state.step_at_subgoal_start = state.episode_step

        if image is None:
            logger.warning(
                f"No image at '{self.ctx.image_key}', "
                f"skipping decomposition"
            )
            state.rewritten_instruction = prompt
            return obs, state

        # Subclass decomposes and sets up state.subgoals
        self._decompose_and_setup(obs, state, prompt, image)

        # Save the original subgoal list so recycle/replan can
        # reference it (ensures VLM picks from GT-trackable subgoals).
        if state.subgoals:
            self._original_subgoals = list(state.subgoals)

        # Configure GT detector for the first subgoal
        if state.subgoals:
            self._configure_gt_detector(obs, state, state.subgoals[0])

        # Apply the decomposed instruction to obs.  _decompose_and_setup
        # sets state.rewritten_instruction but ctx.set_prompt() returns a
        # *new* dict, so the caller's obs is not mutated.  We must apply
        # the instruction here so the first frame uses the correct prompt.
        if state.rewritten_instruction:
            obs = self.ctx.set_prompt(obs, state.rewritten_instruction)

        return obs, state

    def _on_step(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState]:
        # Step-delta gating: cadence is determined by
        # ``state.episode_step`` advancing past the markers
        # ``step_at_last_check`` / ``step_at_subgoal_start``.  No
        # explicit per-chunk counters needed.
        # Per-frame hook (e.g. GDino re-detection)
        self._on_frame(obs, state)

        # ==============================================================
        # HITL mode — human monitors the scene via browser UI.
        #
        # When a failure handler is active (GT or VLM), it runs
        # automatically and the HITL UI shows what's happening.
        # The human can watch, and optionally override via the UI.
        # ==============================================================
        if self._hitl is not None:
            # --- Grasp-tool lifecycle ---
            obs, state, still_active = self._tick_grasp_lifecycle(
                obs, state,
            )
            if still_active:
                # Keep the HITL UI live so the human can watch the
                # grasp execution in real-time.
                self._hitl_push_ui(obs, state)
                return obs, state

            # --- Place-tool lifecycle (parallel to grasp) ---
            obs, state, still_active = self._tick_place_lifecycle(
                obs, state,
            )
            if still_active:
                self._hitl_push_ui(obs, state)
                return obs, state

            # --- Consume pending HITL action (non-blocking) ---
            # Must run BEFORE detection so that a human replan /
            # recovery submitted while the grasp tool was running is
            # picked up before the detector re-fires.
            obs, state = self._hitl_consume_action(obs, state)

            # --- VLM failure handler ---
            if self._failure_handler is not None:
                handler_result = self._failure_handler.step(obs, state)
                if handler_result is not None:
                    # Show the VLM decision to the human via HITL UI
                    self._hitl.update_status(
                        status_message=(
                            f"VLM: {handler_result.status} → "
                            f"{handler_result.action} — "
                            f"{handler_result.reason[:60]}"
                        ),
                    )
                    obs, state = self._execute_handler_result(
                        obs, state, handler_result,
                    )
                    if state.rewritten_instruction:
                        obs = self.ctx.set_prompt(
                            obs, state.rewritten_instruction,
                        )

            # --- GT failure detection ---
            obs, state, outcome = self._tick_gt_detection(obs, state)
            if outcome == self._GT_DETECTION_FAILURE:
                return obs, state
            # On ADVANCE or None, fall through to HITL step

            return self._hitl_on_step(obs, state)

        # ==============================================================
        # Autonomous mode (no HITL) — VLM checks, failure detection,
        # timeouts, grasp-tool lifecycle, etc.
        # ==============================================================

        # --- Grasp-tool lifecycle ---
        obs, state, still_active = self._tick_grasp_lifecycle(obs, state)
        if still_active:
            return obs, state

        # --- Place-tool lifecycle ---
        obs, state, still_active = self._tick_place_lifecycle(obs, state)
        if still_active:
            return obs, state

        # ==============================================================
        # VLM failure handler — unified detection + action selection.
        # Replaces both GT detection and the old VLM subgoal check.
        # ==============================================================
        if self._failure_handler is not None:
            handler_result = self._failure_handler.step(obs, state)
            if handler_result is not None:
                obs, state = self._execute_handler_result(
                    obs, state, handler_result,
                )
                if state.rewritten_instruction:
                    obs = self.ctx.set_prompt(
                        obs, state.rewritten_instruction,
                    )
                # After next/replan, skip the rest of this step
                if handler_result.action != ACTION_CONTINUE:
                    return obs, state

        # ==============================================================
        # Signal-based failure detection (legacy path)
        # ==============================================================
        if self._failure_detector is not None:
            # Accumulate signals every step (~0.1ms)
            self._failure_detector.update_from_obs(obs)

            # Check for failure after minimum steps
            steps_on_subgoal = state.episode_step - state.step_at_subgoal_start
            if (
                state.subgoals
                and steps_on_subgoal
                >= self.MIN_STEPS_BEFORE_FAILURE_CHECK
            ):
                result = self._failure_detector.classify(
                    chunks_elapsed=steps_on_subgoal,
                )
                if result.status == ManipulationStatus.FAILURE:
                    obs, state = self._handle_failure(obs, state, result)
                    if state.rewritten_instruction:
                        obs = self.ctx.set_prompt(
                            obs, state.rewritten_instruction,
                        )
                    return obs, state

        # ==============================================================
        # GT failure detection (legacy path)
        # ==============================================================
        obs, state, outcome = self._tick_gt_detection(obs, state)
        if outcome is not None:
            return obs, state

        # ==============================================================
        # Old periodic VLM check (only when no handler and no GT)
        # ==============================================================
        steps_since_check = state.episode_step - state.step_at_last_check
        steps_on_subgoal = state.episode_step - state.step_at_subgoal_start
        if (
            self._failure_handler is None
            and self._gt_detector is None
            and state.subgoals
            and self._should_check(state)
            and steps_since_check >= self.config.check_interval
        ):
            state.step_at_last_check = state.episode_step
            image = self.ctx.get_vlm_image(obs)
            if image is not None:
                obs, state = self._check_subgoal(obs, state, image)

        # Subgoal timeout
        timeout = self._get_timeout(state)
        if state.subgoals and steps_on_subgoal >= timeout:
            if state.current_subgoal_idx < len(state.subgoals) - 1:
                self._advance_subgoal(obs, state)
                state.log({
                    "type": "subgoal_timeout",
                    "subgoal_idx": state.current_subgoal_idx,
                    "instruction": state.rewritten_instruction,
                })
            elif (
                # Last subgoal: throttle recycle to once per timeout
                # window so we don't burn ``MAX_RECYCLES`` VLM calls in
                # back-to-back chunks if the first attempt fails to
                # produce new subgoals.  Replaces the original
                # ``chunks_on_subgoal == timeout`` one-shot, which broke
                # under variable chunk sizes.
                state.episode_step - getattr(
                    self, "_step_at_last_recycle_attempt", -timeout,
                ) >= timeout
            ):
                self._step_at_last_recycle_attempt = state.episode_step
                # Last subgoal timed out — try to recycle (identify
                # remaining work) before giving up.
                if self._recycle_count < MAX_RECYCLES:
                    image = self.ctx.get_vlm_image(obs)
                    if image is not None and self._recycle(obs, state, image):
                        obs = self.ctx.set_prompt(
                            obs, state.rewritten_instruction
                        )
                        state.log({
                            "type": "subgoal_timeout_recycle",
                            "subgoal_idx": state.current_subgoal_idx,
                            "instruction": state.rewritten_instruction,
                            "new_subgoals": state.subgoals,
                        })
                    else:
                        logger.debug(
                            "  Last subgoal timed out, recycle failed — "
                            "running until episode ends"
                        )
                else:
                    logger.debug(
                        "  Last subgoal timed out, max recycles reached — "
                        "running until episode ends"
                    )

        # Apply current instruction
        if state.rewritten_instruction:
            obs = self.ctx.set_prompt(obs, state.rewritten_instruction)

        return obs, state

    # ------------------------------------------------------------------
    # Handler result execution
    # ------------------------------------------------------------------

    def _execute_handler_result(
        self,
        obs: dict,
        state: SessionState,
        result: HandlerResult,
    ) -> tuple[dict, SessionState]:
        """Execute the action from a :class:`HandlerResult`.

        The handler decided *what* to do; this method *does* it.
        """
        sg_idx = state.current_subgoal_idx
        sg = state.subgoals[sg_idx] if state.subgoals else "?"

        # VLMFailureHandler.step() already emitted a "vlm_detect" event
        # carrying status / action / reason / vlm_latency_s / vlm_raw —
        # no need to duplicate it here.  (Was previously logged as
        # "vlm_detection" from the strategy layer; consolidated to a
        # single event type to make post-hoc analysis simpler.)

        # Surface VLM-detected failures in the annotated video via the
        # same ``orchestrator_failure`` path that signal/GT handlers use.
        if (result.status == STATUS_FAILURE
                or result.action in (ACTION_REPLAN, ACTION_GRASP, ACTION_PLACE)):
            state._last_failure_event = {
                "reason": result.reason,
                "confidence": round(result.confidence, 2),
                "recovery_mode": result.action,
            }

        if result.action == ACTION_NEXT:
            # Advance to next subgoal
            if sg_idx < len(state.subgoals) - 1:
                self._advance_subgoal(obs, state)
            else:
                # Last subgoal done — try to recycle
                if self._recycle_count < MAX_RECYCLES:
                    image = self.ctx.get_vlm_image(obs)
                    if image is not None and self._recycle(
                        obs, state, image,
                    ):
                        logger.info(
                            "  VLM handler: last subgoal done, "
                            f"recycled → {len(state.subgoals)} "
                            "new subgoal(s)"
                        )
                    else:
                        logger.info(
                            "  VLM handler: last subgoal done, "
                            "recycle failed — running until episode ends"
                        )
                else:
                    logger.info(
                        "  VLM handler: last subgoal done, "
                        "all subgoals complete"
                    )

        elif result.action == ACTION_REPLAN:
            logger.info(
                f"  ↻ VLM handler: replan (reason: {result.reason})"
            )
            if self._recycle_count >= MAX_RECYCLES:
                logger.warning(
                    "  ↻ Replan skipped — max recycle attempts reached"
                )
                state.log({
                    "type": "replan_failed",
                    "reason": "max recycles reached",
                    "subgoal_idx": sg_idx,
                    "recycle_count": self._recycle_count,
                })
            else:
                image = self.ctx.get_vlm_image(obs)
                if image is None:
                    logger.warning(
                        "  ↻ Replan failed — no image available"
                    )
                elif self._recycle(obs, state, image):
                    logger.info(
                        f"  ↻ Replanned → {len(state.subgoals)} "
                        f"subgoal(s), starting with "
                        f"\"{state.subgoals[0][:60]}\""
                    )
                else:
                    logger.warning(
                        "  ↻ Replan failed — VLM recycle returned no new "
                        "subgoals (done=true, parse error, or empty list). "
                        "Continuing with current subgoal."
                    )
                    state.log({
                        "type": "replan_failed",
                        "reason": "recycle returned False",
                        "subgoal_idx": sg_idx,
                    })

        elif result.action == ACTION_GRASP:
            target = result.grasp_target or _extract_target_object(sg)
            logger.info(
                f"  🤏 VLM handler: grasp_tool for '{target}'"
            )
            activated = self._activate_grasp_tool(
                obs, state, sg,
                # Build a minimal ClassificationResult for compatibility
                _make_dummy_classification(result),
                target_override=target,
            )
            if not activated:
                logger.warning(
                    "  🤏 Grasp tool failed to start — "
                    "continuing with current subgoal"
                )

        elif result.action == ACTION_PLACE:
            # VLM picked place_tool — activate the place executor.
            # Prefer the new ``place_destination`` freeform phrase
            # (full natural language like "in the white bowl" or
            # "empty space near the orange").  Fall back to combining
            # the legacy ``place_target`` + ``place_relation`` pair.
            destination = (result.place_destination or "").strip()
            held_hint = (result.place_held_object or "").strip()
            if not destination:
                target = (result.place_target or "").strip()
                if not target:
                    logger.warning(
                        "  📍 VLM handler: place_tool requested but "
                        "no place_destination / place_target — ignoring."
                    )
                    destination = ""
                else:
                    relation = (result.place_relation or "in").strip()
                    destination = (
                        f"{relation} the {target}".strip()
                        if relation else target
                    )
                    logger.info(
                        f"  📍 VLM handler: place_destination not set; "
                        f"combined legacy fields → '{destination}'"
                    )
            if destination:
                place_stack = bool(getattr(result, "place_stack", False))
                logger.info(
                    f"  📍 VLM handler: place_tool → '{destination}'"
                    + (f" (held={held_hint})" if held_hint else "")
                    + (
                        f" [stack={place_stack}]"
                        if self._stack_mode_enabled else ""
                    )
                )
                # Pass the freeform phrase via the legacy payload
                # encoding so the existing parser fills target_object
                # with the full phrase (relation is forced to "in" —
                # raycast just needs to know which side of the
                # surface to drop on, and the phrase carries the
                # spatial intent itself).
                payload = f"in|{destination}|{held_hint}"
                activated = self._activate_place_tool_from_hitl(
                    obs, state, payload, source="vlm_handler",
                    stack=place_stack,
                )
                if not activated:
                    logger.warning(
                        "  📍 Place tool failed to start — "
                        "continuing with current subgoal"
                    )

        elif result.action == ACTION_CONTINUE:
            # Nothing to do — keep working
            if result.instruction:
                # Handler provided a refined instruction
                state.rewritten_instruction = result.instruction
                state.flush_actions = True

        # Reset chunk counters after any action (except continue)
        if result.action != ACTION_CONTINUE:
            state.step_at_subgoal_start = state.episode_step
            state.step_at_last_check = state.episode_step
            if self._failure_handler is not None:
                self._failure_handler.on_subgoal_advanced(
                    obs, state, state.current_subgoal_idx,
                )

        return obs, state

    # ------------------------------------------------------------------
    # HITL integration
    # ------------------------------------------------------------------

    def _hitl_push_ui(
        self, obs: dict, state: SessionState, *, force_image: bool = False,
    ) -> None:
        """Push state and (periodically) image to the HITL browser UI."""
        hitl = self._hitl

        hitl.update_status(
            step_count=state.infer_count,
            instruction=state.rewritten_instruction or "",
            subgoals=list(state.subgoals),
            subgoal_idx=state.current_subgoal_idx,
            is_active=True,
        )

        if force_image or state.infer_count % HITL_IMAGE_INTERVAL == 0:
            image = self.ctx.get_vlm_image(obs)
            if image is not None:
                hitl.update_image(image)

    @staticmethod
    def _hitl_log_context(state: SessionState) -> dict:
        """Build common context fields for every HITL log entry."""
        ctx: dict = {
            "step_count": state.infer_count,
            "subgoal_idx": state.current_subgoal_idx,
        }
        if state.subgoals and state.current_subgoal_idx < len(state.subgoals):
            ctx["subgoal"] = state.subgoals[state.current_subgoal_idx]
        if state.rewritten_instruction:
            ctx["current_instruction"] = state.rewritten_instruction
        return ctx

    def _hitl_consume_action(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState]:
        """Consume ONE pending human action and apply it to state.

        Returns the (possibly modified) ``(obs, state)`` pair.
        Does nothing if no action is pending.
        """
        hitl = self._hitl
        if not hitl.has_pending_action():
            return obs, state

        from vlm_orchestrator.hitl import HITLAction

        action, data = hitl.consume_action()

        if action == HITLAction.SUBGOAL_DONE:
            logger.info("  HITL: human marked subgoal DONE")
            ctx = self._hitl_log_context(state)
            hitl.update_status(
                failure_type="",
                status_message="Human: subgoal done → advancing",
            )
            if (state.subgoals
                    and state.current_subgoal_idx < len(state.subgoals) - 1):
                self._advance_subgoal(obs, state)
            state.log({"type": "hitl_done", **ctx})

        elif action == HITLAction.FAILURE:
            logger.info("  HITL: human flagged FAILURE → pausing for recovery")
            ctx = self._hitl_log_context(state)
            hitl.update_status(
                failure_type="human",
                status_message=(
                    "⚠ Human flagged failure — send a Recovery "
                    "instruction, Skip, or Resume to continue"
                ),
                is_paused=True,
            )
            state.log({"type": "hitl_failure", **ctx})
            # Block the proxy loop until human provides recovery / resume
            obs, state = self._hitl_pause_loop(obs, state)

        elif action == HITLAction.RECOVERY:
            instruction = data.get("instruction", "")
            if instruction:
                logger.info(f"  HITL: human recovery → \"{instruction}\"")
                ctx = self._hitl_log_context(state)
                previous = state.rewritten_instruction
                self._apply_new_instruction(obs, state, instruction)
                hitl.update_status(
                    failure_type="",
                    status_message=f"Recovery: {instruction[:60]}",
                    is_paused=False,
                )
                state.log({
                    "type": "hitl_recovery",
                    "instruction": instruction,
                    "previous_instruction": previous,
                    **ctx,
                })

        elif action == HITLAction.REWRITE_INSTRUCTION:
            instruction = data.get("instruction", "")
            if instruction:
                logger.info(f"  HITL: rewrite → \"{instruction}\"")
                ctx = self._hitl_log_context(state)
                previous = state.rewritten_instruction
                state.rewritten_instruction = instruction
                state.flush_actions = True
                hitl.update_status(
                    status_message=f"Rewritten: {instruction[:60]}",
                )
                state.log({
                    "type": "hitl_rewrite",
                    "instruction": instruction,
                    "previous_instruction": previous,
                    **ctx,
                })

        elif action == HITLAction.START_SUBGOAL:
            # Human submitted subgoals (one per line).
            instruction = data.get("instruction", "")
            if instruction:
                lines = [
                    ln.strip() for ln in instruction.splitlines()
                    if ln.strip()
                ]
                if lines:
                    self._apply_new_subgoals(obs, state, lines)
                    hitl.update_status(
                        subgoals=lines,
                        subgoal_idx=0,
                        failure_type="",
                        gt_failure=None,
                        is_paused=False,
                        status_message=(
                            f"Human: {len(lines)} subgoal(s) → "
                            f"executing \"{lines[0][:50]}\""
                        ),
                    )
                    logger.info(
                        f"  HITL: human submitted {len(lines)} subgoal(s)"
                    )
                    for i, sg in enumerate(lines):
                        logger.info(f"    [{i + 1}] {sg}")
                    state.log({
                        "type": "hitl_start_subgoals",
                        "subgoals": lines,
                        "step_count": state.infer_count,
                    })

        elif action == HITLAction.SKIP:
            logger.info("  HITL: human SKIP subgoal")
            ctx = self._hitl_log_context(state)
            if (state.subgoals
                    and state.current_subgoal_idx < len(state.subgoals) - 1):
                self._advance_subgoal(obs, state)
                hitl.update_status(
                    failure_type="",
                    status_message="Skipped subgoal",
                )
            state.log({"type": "hitl_skip", **ctx})

        elif action == HITLAction.PAUSE:
            logger.info("  HITL: PAUSE → blocking proxy loop")
            ctx = self._hitl_log_context(state)
            hitl.update_status(
                is_paused=True,
                status_message="⏸ Paused — click Resume to continue",
            )
            state.log({"type": "hitl_pause", **ctx})
            # Block the proxy loop so the robot stops receiving actions.
            obs, state = self._hitl_pause_loop(obs, state)

        elif action == HITLAction.RESUME:
            logger.info("  HITL: RESUMED")
            ctx = self._hitl_log_context(state)
            hitl.update_status(
                is_paused=False,
                status_message="▶ Resumed",
            )
            state.log({"type": "hitl_resume", **ctx})

        elif action == HITLAction.ABORT:
            logger.warning("  HITL: ABORT requested")
            ctx = self._hitl_log_context(state)
            hitl.update_status(
                is_active=False,
                status_message="Aborted by human",
            )
            state.log({"type": "hitl_abort", **ctx})
            state.rewritten_instruction = "stop"

        elif action == HITLAction.GRASP_WITH_TOOL:
            target_object = data.get("instruction", "")
            if target_object:
                logger.info(
                    f"  HITL: grasp_with_tool for '{target_object}'"
                )
                # Lazy-create executor if needed
                if state.grasp_tool_executor is None:
                    from vlm_orchestrator.grasp.tool import GraspToolExecutor
                    state.grasp_tool_executor = GraspToolExecutor(seg_mode=self._grasp_seg_mode, env_mode=self._env_mode, use_front_camera=self._use_front_camera, topdown_threshold=self._grasp_topdown_threshold, motion_planner=self._get_motion_planner(), stack_mode_enabled=self._stack_mode_enabled)
                # Expose HITL handle so executor can push debug images
                state._hitl = self._hitl
                state.grasp_tool_executor.start(
                    target_object=target_object, obs=obs, state=state,
                )
                state.grasp_tool_active = (
                    state.grasp_tool_executor.is_active
                )
                state.grasp_tool_phase = (
                    state.grasp_tool_executor.phase.value
                )
                state.grasp_tool_target = target_object
                hitl.update_status(
                    status_message=(
                        state.grasp_tool_executor.status_message
                    ),
                )
                state.log({
                    "type": "hitl_grasp_tool",
                    "target_object": target_object,
                    "phase": state.grasp_tool_phase,
                    **self._hitl_log_context(state),
                })

        elif action == HITLAction.PLACE_WITH_TOOL:
            payload = data.get("instruction", "")
            if payload:
                state._hitl = self._hitl
                if not self._recovery_admits_place():
                    hitl.update_status(
                        status_message=(
                            f"❌ place_tool not admitted by "
                            f"--recovery-mode {self._recovery_mode!r}; "
                            f"set 'place' / 'tools' / 'place_first' / "
                            f"'tools_first'."
                        ),
                    )
                else:
                    self._activate_place_tool_from_hitl(obs, state, payload)
                    if state.place_tool_active:
                        hitl.update_status(
                            status_message=(
                                state.place_tool_executor.status_message
                            ),
                        )

        return obs, state

    def _hitl_on_step(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState]:
        """Per-step logic when HITL is active.

        Replaces the autonomous VLM/failure/timeout pipeline with a
        purely human-driven loop:

        1. If paused → block until resumed.
        2. Push state + image to browser.
        3. Consume any pending human action.
        4. Apply current instruction and return.

        No VLM calls, no automatic failure detection, no timeouts.
        """
        hitl = self._hitl

        # 1. If already paused (e.g. from a previous step), block first.
        if hitl.is_paused:
            obs, state = self._hitl_pause_loop(obs, state)

        # 2. Push state and camera image to the browser.
        self._hitl_push_ui(obs, state)

        # 3. Consume a pending human action (non-blocking check).
        obs, state = self._hitl_consume_action(obs, state)

        # NOTE: grasp-tool lifecycle and GT detection are handled by
        # _tick_grasp_lifecycle / _tick_gt_detection in the caller
        # (_on_step) BEFORE _hitl_on_step is called.

        # 4. Apply current instruction.
        if state.rewritten_instruction:
            obs = self.ctx.set_prompt(obs, state.rewritten_instruction)

        return obs, state

    def _hitl_pause_loop(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState]:
        """Block the proxy loop until the human resumes or provides input.

        While blocked the robot receives no new actions, so it stops
        moving.  We keep pushing camera images so the human can see the
        (static) scene.

        Exits when the human sends: RESUME, RECOVERY, SKIP, ABORT, or
        SUBGOAL_DONE.
        """
        from vlm_orchestrator.hitl import HITLAction

        hitl = self._hitl
        logger.info("  HITL: entering pause loop (robot stopped)")

        while True:
            # Push a fresh image each iteration so the UI stays live.
            self._hitl_push_ui(obs, state, force_image=True)

            # Wait up to 0.25 s for the human to act.  Short timeout
            # keeps the image feed responsive; long enough to avoid
            # busy-spinning.
            got_action = hitl.wait_for_decision(timeout=0.25)
            if not got_action:
                continue

            action, data = hitl.consume_action()

            if action == HITLAction.RESUME:
                logger.info("  HITL: resumed from pause")
                hitl.update_status(
                    is_paused=False,
                    status_message="▶ Resumed",
                )
                state.log({
                    "type": "hitl_resume",
                    **self._hitl_log_context(state),
                })
                break

            if action == HITLAction.RECOVERY:
                instruction = data.get("instruction", "")
                if instruction:
                    logger.info(
                        f"  HITL: recovery during pause → \"{instruction}\""
                    )
                    ctx = self._hitl_log_context(state)
                    previous = state.rewritten_instruction
                    self._apply_new_instruction(obs, state, instruction)
                    hitl.update_status(
                        failure_type="",
                        gt_failure=None,
                        is_paused=False,
                        status_message=f"Recovery: {instruction[:60]}",
                    )
                    state.log({
                        "type": "hitl_recovery",
                        "instruction": instruction,
                        "previous_instruction": previous,
                        **ctx,
                    })
                    break

            if action == HITLAction.REWRITE_INSTRUCTION:
                instruction = data.get("instruction", "")
                if instruction:
                    logger.info(
                        f"  HITL: rewrite during pause → \"{instruction}\""
                    )
                    ctx = self._hitl_log_context(state)
                    previous = state.rewritten_instruction
                    state.rewritten_instruction = instruction
                    hitl.update_status(
                        is_paused=False,
                        status_message=f"Rewritten: {instruction[:60]}",
                    )
                    state.log({
                        "type": "hitl_rewrite",
                        "instruction": instruction,
                        "previous_instruction": previous,
                        **ctx,
                    })
                    break

            if action == HITLAction.SKIP:
                logger.info("  HITL: skip during pause")
                ctx = self._hitl_log_context(state)
                if (state.subgoals
                        and state.current_subgoal_idx
                        < len(state.subgoals) - 1):
                    self._advance_subgoal(obs, state)
                hitl.update_status(
                    failure_type="",
                    is_paused=False,
                    status_message="Skipped subgoal",
                )
                state.log({"type": "hitl_skip", **ctx})
                break

            if action == HITLAction.SUBGOAL_DONE:
                logger.info("  HITL: done during pause")
                ctx = self._hitl_log_context(state)
                if (state.subgoals
                        and state.current_subgoal_idx
                        < len(state.subgoals) - 1):
                    self._advance_subgoal(obs, state)
                hitl.update_status(
                    failure_type="",
                    is_paused=False,
                    status_message="Human: subgoal done → advancing",
                )
                state.log({"type": "hitl_done", **ctx})
                break

            if action == HITLAction.ABORT:
                logger.warning("  HITL: abort during pause")
                ctx = self._hitl_log_context(state)
                hitl.update_status(
                    is_active=False,
                    is_paused=False,
                    status_message="Aborted by human",
                )
                state.rewritten_instruction = "stop"
                state.log({"type": "hitl_abort", **ctx})
                break

            if action == HITLAction.START_SUBGOAL:
                instruction = data.get("instruction", "")
                if instruction:
                    lines = [
                        ln.strip() for ln in instruction.splitlines()
                        if ln.strip()
                    ]
                    if lines:
                        logger.info(
                            f"  HITL: replan during pause → "
                            f"{len(lines)} subgoal(s)"
                        )
                        ctx = self._hitl_log_context(state)
                        self._apply_new_subgoals(obs, state, lines)
                        hitl.update_status(
                            subgoals=lines,
                            subgoal_idx=0,
                            failure_type="",
                            gt_failure=None,
                            is_paused=False,
                            status_message=(
                                f"Replanned: {len(lines)} subgoal(s) → "
                                f"\"{lines[0][:50]}\""
                            ),
                        )
                        state.log({
                            "type": "hitl_replan_during_pause",
                            "subgoals": lines,
                            **ctx,
                        })
                        break

            if action == HITLAction.GRASP_WITH_TOOL:
                target_object = data.get("instruction", "")
                if target_object:
                    logger.info(
                        f"  HITL: grasp_with_tool during pause "
                        f"for '{target_object}'"
                    )
                    if state.grasp_tool_executor is None:
                        from vlm_orchestrator.grasp.tool import (
                            GraspToolExecutor,
                        )
                        state.grasp_tool_executor = GraspToolExecutor(seg_mode=self._grasp_seg_mode, env_mode=self._env_mode, use_front_camera=self._use_front_camera, topdown_threshold=self._grasp_topdown_threshold, motion_planner=self._get_motion_planner(), stack_mode_enabled=self._stack_mode_enabled)
                    # Expose HITL handle for debug visualization
                    state._hitl = hitl
                    state.grasp_tool_executor.start(
                        target_object=target_object,
                        obs=obs, state=state,
                    )
                    state.grasp_tool_active = (
                        state.grasp_tool_executor.is_active
                    )
                    state.grasp_tool_phase = (
                        state.grasp_tool_executor.phase.value
                    )
                    state.grasp_tool_target = target_object
                    hitl.update_status(
                        is_paused=False,
                        status_message=(
                            state.grasp_tool_executor.status_message
                        ),
                    )
                    state.log({
                        "type": "hitl_grasp_tool",
                        "target_object": target_object,
                        "phase": state.grasp_tool_phase,
                        **self._hitl_log_context(state),
                    })
                    break

            if action == HITLAction.PLACE_WITH_TOOL:
                payload = data.get("instruction", "")
                if payload:
                    state._hitl = hitl
                    if not self._recovery_admits_place():
                        hitl.update_status(
                            status_message=(
                                f"❌ place_tool not admitted by "
                                f"--recovery-mode {self._recovery_mode!r}; "
                                f"set 'place' / 'tools' / 'place_first' / "
                                f"'tools_first'."
                            ),
                        )
                    else:
                        self._activate_place_tool_from_hitl(
                            obs, state, payload,
                        )
                        if state.place_tool_active:
                            hitl.update_status(
                                is_paused=False,
                                status_message=(
                                    state.place_tool_executor.status_message
                                ),
                            )
                            break

            # Any other action (e.g. PAUSE again, NONE) — keep waiting.

        return obs, state

    def _hitl_wait_for_subgoals(
        self, obs: dict, state: SessionState, prompt: str,
    ) -> tuple[dict, SessionState]:
        """Block at episode start until the human provides subgoals.

        The human sees the first camera frame and the task instruction,
        then types subgoals (one per line) and clicks "Submit Subgoals".
        Until that happens the proxy loop is blocked and the robot stays
        at its start pose.
        """
        from vlm_orchestrator.hitl import HITLAction

        hitl = self._hitl
        hitl.update_status(
            status_message=(
                "⏳ Waiting for human — type subgoals and click "
                "'Submit Subgoals'"
            ),
            is_paused=True,
        )
        logger.info("  HITL: waiting for human to submit subgoals …")

        while True:
            # Keep the image feed alive while waiting.
            self._hitl_push_ui(obs, state, force_image=True)

            got = hitl.wait_for_decision(timeout=0.25)
            if not got:
                continue

            action, data = hitl.consume_action()

            if action in (HITLAction.START_SUBGOAL,
                          HITLAction.REWRITE_INSTRUCTION):
                instruction = data.get("instruction", "")
                if not instruction:
                    continue
                lines = [
                    ln.strip() for ln in instruction.splitlines()
                    if ln.strip()
                ]
                if not lines:
                    continue

                self._apply_new_subgoals(obs, state, lines)
                hitl.update_status(
                    subgoals=lines,
                    subgoal_idx=0,
                    is_paused=False,
                    status_message=(
                        f"▶ {len(lines)} subgoal(s) — executing "
                        f"\"{lines[0][:50]}\""
                    ),
                )
                logger.info(
                    f"  HITL: human submitted {len(lines)} subgoal(s)"
                )
                for i, sg in enumerate(lines):
                    logger.info(f"    [{i + 1}] {sg}")

                state.log({
                    "type": "hitl_start_subgoals",
                    "subgoals": lines,
                    "step_count": state.infer_count,
                })
                break

            if action == HITLAction.ABORT:
                logger.warning("  HITL: abort before subgoals submitted")
                hitl.update_status(
                    is_active=False,
                    is_paused=False,
                    status_message="Aborted by human",
                )
                state.rewritten_instruction = "stop"
                state.log({
                    "type": "hitl_abort",
                    "step_count": state.infer_count,
                })
                break

            # Ignore other actions while waiting for subgoals.

        if state.rewritten_instruction:
            obs = self.ctx.set_prompt(obs, state.rewritten_instruction)
        return obs, state

    # ------------------------------------------------------------------
    # Failure recovery
    # ------------------------------------------------------------------

    def _handle_failure(
        self,
        obs: dict,
        state: SessionState,
        result: "ClassificationResult",  # noqa: F821
    ) -> tuple[dict, SessionState]:
        """React to a detected failure with corrective recovery.

        Recovery depends on ``self._recovery_mode``:

        ``"grasp_first"``
            On ANY failure involving a grasp subgoal, immediately lift
            the arm and activate grasp_with_tool.  No VLM call, fastest
            possible recovery.  Falls through to replan/skip only if
            the grasp tool itself fails or the subgoal isn't a grasp.

        ``"vlm_grasp"``
            One VLM call that sees the scene and decides:
            *retry* (new instruction for VLA) or *grasp_tool* (planned
            grasp).  Falls through to replan/skip on exhausted retries.

        ``"template"`` / ``"vlm"`` / ``"retry"``
            Original ladder: recover × 2 → replan → skip.
            Grasp tool activates only after retries are exhausted AND
            the subgoal involves grasping.
        """
        sg_idx = state.current_subgoal_idx
        sg = state.subgoals[sg_idx] if state.subgoals else "?"

        logger.warning(
            f"  ⚠ Failure detected on subgoal [{sg_idx + 1}/"
            f"{len(state.subgoals)}] \"{sg}\" — "
            f"{result.reason} (conf={result.confidence:.2f})"
        )

        # Store event for video annotation overlay
        state._last_failure_event = {
            "reason": result.reason,
            "confidence": round(result.confidence, 2),
            "recovery_mode": self._recovery_mode,
        }

        # ==============================================================
        # Mode: grasp_first — skip retries, go straight to grasp tool
        # ==============================================================
        if self._recovery_mode == "grasp_first":
            if _is_grasp_subgoal(sg, result.reason):
                activated = self._activate_grasp_tool(
                    obs, state, sg, result,
                )
                if activated:
                    return obs, state
                # Grasp tool failed to start → fall through to replan
            # Not a grasp subgoal → use template recovery
            return self._recover_with_instruction(
                obs, state, sg, result,
            )

        # ==============================================================
        # Mode: vlm_grasp — VLM decides retry vs grasp_tool on FIRST failure
        # ==============================================================
        if self._recovery_mode == "vlm_grasp":
            self._failure_retries += 1
            state.step_at_subgoal_start = state.episode_step
            state.step_at_last_check = state.episode_step
            if self._failure_detector is not None:
                self._failure_detector.reset()

            image = self.ctx.get_vlm_image(obs)
            recovery = generate_recovery(
                failure_result=result,
                subgoal_instruction=sg,
                task_instruction=state.original_instruction or sg,
                current_image=image,
                vlm_call_fn=self._vlm_call,
                retry_count=self._failure_retries - 1,
                mode="vlm_grasp",
            )

            if recovery.method == "grasp_tool":
                # VLM chose grasp tool
                target = recovery.instruction  # target object name
                logger.info(
                    f"  🤏 VLM chose grasp_tool for '{target}'"
                )
                activated = self._activate_grasp_tool(
                    obs, state, sg, result, target_override=target,
                )
                if activated:
                    state.log({
                        "type": "failure_recovery",
                        "action": "grasp_tool_vlm",
                        "target_object": target,
                        "failure_reason": result.reason,
                        "confidence": round(result.confidence, 3),
                        "subgoal_idx": sg_idx,
                        "retry_count": self._failure_retries,
                    })
                    return obs, state
                # Grasp tool failed → treat as a retry
                logger.warning(
                    "  Grasp tool failed to start, using VLM instruction"
                )

            # VLM chose retry (or grasp_tool failed to start)
            state.rewritten_instruction = recovery.instruction
            logger.info(
                f"  ↻ VLM RECOVER [{recovery.method}]: "
                f"\"{recovery.instruction[:80]}\""
            )
            state.log({
                "type": "failure_recovery",
                "action": "recover",
                "method": recovery.method,
                "retry_count": self._failure_retries,
                "failure_type": recovery.failure_type,
                "failure_reason": result.reason,
                "confidence": round(result.confidence, 3),
                "recovery_instruction": recovery.instruction,
                "subgoal_idx": sg_idx,
            })

            # If too many retries, fall through to replan/skip
            if self._failure_retries >= self.MAX_FAILURE_RETRIES:
                return self._replan_or_skip(obs, state, result)
            return obs, state

        # ==============================================================
        # Original modes: template / vlm / retry
        # Ladder: recover × N → grasp (if applicable) → replan → skip
        # ==============================================================
        if self._failure_retries < self.MAX_FAILURE_RETRIES:
            return self._recover_with_instruction(
                obs, state, sg, result,
            )

        # Retries exhausted → try grasp tool if applicable
        if _is_grasp_subgoal(sg, result.reason):
            activated = self._activate_grasp_tool(
                obs, state, sg, result,
            )
            if activated:
                return obs, state

        # Grasp not applicable or failed → replan/skip
        return self._replan_or_skip(obs, state, result)

    # ------------------------------------------------------------------
    # Shared grasp-tool & GT detection helpers (single implementation)
    # ------------------------------------------------------------------

    def _tick_grasp_lifecycle(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState, bool]:
        """Drive the grasp-tool executor through DONE / FAILED states.

        Returns ``(obs, state, still_active)`` where *still_active* is
        True when the executor is still running (caller should return
        early and skip other logic).
        """
        if not (state.grasp_tool_active
                and state.grasp_tool_executor is not None):
            return obs, state, False

        from vlm_orchestrator.grasp.tool import GraspPhase

        executor = state.grasp_tool_executor
        state.grasp_tool_phase = executor.phase.value

        # Optionally push status to HITL UI
        hitl = self._hitl
        if hitl is not None:
            hitl.update_status(status_message=executor.status_message)

        if executor.phase == GraspPhase.DONE:
            logger.info("  grasp_with_tool: DONE → returning to VLA")
            state.grasp_tool_active = False
            state.grasp_tool_phase = "idle"
            state.grasp_debug_image = None
            state.grasp_debug_label = ""
            state.grasp_debug_queue.clear()
            if hitl is not None:
                hitl.update_status(
                    status_message="✅ Grasp complete — VLA control resumed",
                )
            state.log({
                "type": "grasp_tool_done",
                "target_object": state.grasp_tool_target,
                "confidence": executor._grasp_confidence,
            })
            executor.reset()

        elif executor.phase == GraspPhase.FAILED:
            logger.warning(
                f"  grasp_with_tool: FAILED ({executor.status_message})"
            )
            state.grasp_tool_active = False
            state.grasp_tool_phase = "idle"
            state.grasp_debug_image = None
            state.grasp_debug_label = ""
            state.grasp_debug_queue.clear()
            state.log({
                "type": "grasp_tool_failed",
                "target_object": state.grasp_tool_target,
                "reason": executor.status_message,
            })
            executor.reset()

            # In gt_hitl mode, pause so human can decide next step.
            # In gt (auto) / other modes, resume VLA immediately.
            if (self._failure_monitor_mode == "gt_hitl"
                    and hitl is not None):
                hitl.update_status(
                    status_message=f"❌ {executor.status_message}",
                    is_paused=True,
                )
                obs, state = self._hitl_pause_loop(obs, state)
            elif hitl is not None:
                hitl.update_status(
                    status_message=(
                        f"❌ {executor.status_message} — resuming VLA"
                    ),
                )

        if state.grasp_tool_active:
            # Still running — caller should return early
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(
                    obs, state.rewritten_instruction,
                )
            return obs, state, True

        # Grasp just finished (DONE or FAILED) — reset for fresh detection
        state.step_at_subgoal_start = state.episode_step
        state.step_at_last_check = state.episode_step
        if self._failure_detector is not None:
            self._failure_detector.reset()
        logger.info("  Failure detector reset — VLA resumes from clean state")

        return obs, state, False

    # ------------------------------------------------------------------
    # Place-tool lifecycle (mirror of _tick_grasp_lifecycle)
    # ------------------------------------------------------------------

    # Recovery-mode tokens that admit the place_tool action.
    _PLACE_RECOVERY_MODES = frozenset({
        "place", "replan_place",          # VLM-mode, place-only
        "tools", "replan_tools",          # VLM-mode, combined grasp+place
        "place_first", "tools_first",     # GT-mode, admits place
    })

    def _recovery_admits_place(self) -> bool:
        """Whether the configured --recovery-mode admits place_tool."""
        return self._recovery_mode in self._PLACE_RECOVERY_MODES

    def _activate_place_tool_from_hitl(
        self,
        obs: dict,
        state: SessionState,
        payload: str,
        *,
        source: str = "hitl",
        stack: bool = False,
    ) -> bool:
        """Parse a place_tool payload and start the executor.

        Payload shape: ``"<relation>|<target>|<held_hint>"``.  Older
        clients may pass just ``"<target>"`` — treated as
        ``relation='in'`` and no held-hint.

        ``stack`` is the advisory release-orientation control forwarded to
        the executor; only honoured when ``--enable-stack-mode`` is on
        (otherwise ignored, historical grasp-consistent placement).
        ``False`` = top-down release; ``True`` = preserve grasp orientation
        (stacking on top of another object).

        ``source`` identifies which caller invoked the activation (used
        for the JSONL log event):
          * ``"hitl"`` — a human pressed the place button in the browser UI.
          * ``"vlm_handler"`` — the VLM failure handler picked
            ``action=place_tool``.
          * (Any other string is preserved verbatim in the log.)

        Despite the name, this function is shared by both paths — the
        ``_from_hitl`` suffix is legacy.  The emitted log event is
        ``place_tool_invoked`` (not ``hitl_place_tool``) so both call
        sites produce consistent records, distinguishable by ``source``.

        Returns True if the executor started; False if the parse / start
        failed *or* the active --recovery-mode does not admit place_tool.
        Per the placement plan, --recovery-mode is the single gate;
        neither HITL nor the VLM handler can route around it.
        """
        if not self._recovery_admits_place():
            logger.warning(
                f"  HITL place_tool requested but --recovery-mode "
                f"{self._recovery_mode!r} does not admit place_tool. "
                f"Use one of {sorted(self._PLACE_RECOVERY_MODES)}."
            )
            return False

        from vlm_orchestrator.place import (
            DestinationSpec, PlaceToolExecutor,
        )

        parts = payload.split("|")
        if len(parts) >= 2:
            relation = parts[0].strip() or "in"
            target = parts[1].strip()
            held_hint = parts[2].strip() if len(parts) >= 3 else ""
        else:
            relation = "in"
            target = payload.strip()
            held_hint = ""

        if not target:
            logger.warning("  HITL place_tool: empty target — ignoring")
            return False

        if relation not in ("in", "on", "on_top_of"):
            logger.warning(
                f"  HITL place_tool: unknown relation {relation!r}, "
                f"defaulting to 'in'"
            )
            relation = "in"

        if state.place_tool_executor is None:
            state.place_tool_executor = PlaceToolExecutor(
                seg_mode=self._place_seg_mode,
                use_front_camera=self._use_front_camera,
                vlm=self.ctx.vlm,
                motion_planner=self._get_motion_planner(),
                stack_mode_enabled=self._stack_mode_enabled,
            )

        spec = DestinationSpec(
            target_object=target,
            relation=relation,
        )
        try:
            state.place_tool_executor.start(
                spec, obs, state,
                held_object_hint=held_hint or None,
                instruction=(
                    state.rewritten_instruction
                    or state.original_instruction
                    or ""
                ),
                stack=stack,
            )
        except Exception as e:
            logger.error(f"  HITL place_tool start failed: {e}", exc_info=True)
            return False

        state.place_tool_active = state.place_tool_executor.is_active
        state.place_tool_phase = state.place_tool_executor.phase.value
        state.place_tool_target = spec.describe()
        state.place_tool_held_object = held_hint
        state.flush_actions = True
        state.log({
            "type": "place_tool_invoked",
            "source": source,
            "destination": spec.describe(),
            "relation": relation,
            "held_object_hint": held_hint,
            "seg_mode": self._place_seg_mode,
            "stack": bool(stack) if self._stack_mode_enabled else None,
            "stack_mode_enabled": self._stack_mode_enabled,
            **self._hitl_log_context(state),
        })
        return state.place_tool_active

    def _tick_place_lifecycle(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState, bool]:
        """Drive the place-tool executor through DONE / FAILED states.

        Mirrors :meth:`_tick_grasp_lifecycle`.  Returns
        ``(obs, state, still_active)``.
        """
        if not (
            state.place_tool_active
            and state.place_tool_executor is not None
        ):
            return obs, state, False

        from vlm_orchestrator.place import PlacePhase

        executor = state.place_tool_executor
        state.place_tool_phase = executor.phase.value

        hitl = self._hitl
        if hitl is not None:
            hitl.update_status(status_message=executor.status_message)

        if executor.phase == PlacePhase.DONE:
            logger.info("  place_with_tool: DONE → returning to VLA")
            state.place_tool_active = False
            state.place_tool_phase = "idle"
            state.place_debug_image = None
            state.place_debug_label = ""
            state.place_debug_queue.clear()
            if hitl is not None:
                hitl.update_status(
                    status_message="✅ Place complete — VLA control resumed",
                )
            state.log({
                "type": "place_tool_done",
                "destination": state.place_tool_target,
                "held_object_hint": state.place_tool_held_object,
                "err_m": executor._place_log.get("post_move_err_m"),
                "source": executor._place_log.get("point_2d_source"),
            })
            executor.reset()

        elif executor.phase == PlacePhase.FAILED:
            logger.warning(
                f"  place_with_tool: FAILED ({executor.failure_reason}: "
                f"{executor.status_message})"
            )
            state.place_tool_active = False
            state.place_tool_phase = "idle"
            state.place_debug_image = None
            state.place_debug_label = ""
            state.place_debug_queue.clear()
            state.log({
                "type": "place_tool_failed",
                "destination": state.place_tool_target,
                "held_object_hint": state.place_tool_held_object,
                "failure_reason": executor.failure_reason,
                "message": executor.status_message,
            })
            executor.reset()

            if (
                self._failure_monitor_mode == "gt_hitl"
                and hitl is not None
            ):
                hitl.update_status(
                    status_message=f"❌ {executor.status_message}",
                    is_paused=True,
                )
                obs, state = self._hitl_pause_loop(obs, state)
            elif hitl is not None:
                hitl.update_status(
                    status_message=(
                        f"❌ {executor.status_message} — resuming VLA"
                    ),
                )

        if state.place_tool_active:
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(
                    obs, state.rewritten_instruction,
                )
            return obs, state, True

        # Place just finished (DONE or FAILED) — reset detection state
        state.step_at_subgoal_start = state.episode_step
        state.step_at_last_check = state.episode_step
        if self._failure_detector is not None:
            self._failure_detector.reset()
        logger.info("  Failure detector reset after place — VLA resumes")
        return obs, state, False

    _GT_DETECTION_ADVANCE = "advance"
    _GT_DETECTION_FAILURE = "failure"

    def _tick_gt_detection(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState, str | None]:
        """Run one GT failure-detection cycle.

        Returns ``(obs, state, outcome)`` where *outcome* is:
        - ``None``            — nothing happened (in-progress)
        - ``"advance"``       — subgoal complete, state already updated
        - ``"failure"``       — failure detected, recovery already applied
        """
        if (self._gt_detector is None
                or not state.subgoals):
            return obs, state, None

        gt_state = obs.get("gt_state")

        if gt_state is None:
            if state.infer_count == 1:
                logger.warning(
                    "[GT] obs has no 'gt_state' key — "
                    "did you pass --enable-gt-state to the RoboLab VoLo runner?"
                )
            return obs, state, None

        # When all subgoals are complete, only check for regression
        # (previously-completed work coming undone).  Skip normal
        # failure detection so the VLA can keep running toward episode
        # end without spurious WRONG_OBJECT / NO_PROGRESS fires.
        if self._gt_all_done:
            subtask = gt_state.get("subtask", {})
            reg = self._gt_detector._check_regression(subtask)
            if reg is not None and reg.is_failure:
                logger.warning(
                    f"  ⚠ GT regression after all-done: "
                    f"{reg.reason}"
                )
                self._gt_all_done = False
                obs, state = self._handle_gt_failure(obs, state, reg)
                if state.rewritten_instruction:
                    obs = self.ctx.set_prompt(
                        obs, state.rewritten_instruction,
                    )
                return obs, state, self._GT_DETECTION_FAILURE
            return obs, state, None

        gt_result = self._gt_detector.update(gt_state)

        if gt_result.failure_type == GTFailureType.SUBGOAL_COMPLETE:
            logger.info(
                f"  ✓ GT: subgoal complete — {gt_result.reason}"
            )
            state.log({
                "type": "gt_subgoal_complete",
                "subgoal_idx": state.current_subgoal_idx,
                "reason": gt_result.reason,
            })
            if state.current_subgoal_idx < len(state.subgoals) - 1:
                self._advance_subgoal(obs, state)
            else:
                logger.info(
                    "  ✓ GT: all subgoals complete — "
                    "disabling GT detection"
                )
                self._gt_all_done = True
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(
                    obs, state.rewritten_instruction,
                )
            return obs, state, self._GT_DETECTION_ADVANCE

        elif gt_result.is_failure:
            obs, state = self._handle_gt_failure(
                obs, state, gt_result,
            )
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(
                    obs, state.rewritten_instruction,
                )
            return obs, state, self._GT_DETECTION_FAILURE

        return obs, state, None

    # ------------------------------------------------------------------
    # GT failure handling
    # ------------------------------------------------------------------

    def _handle_gt_failure(
        self,
        obs: dict,
        state: SessionState,
        gt_result: GTFailureResult,
    ) -> tuple[dict, SessionState]:
        """React to a GT-detected object-level failure.

        All modes log the failure and push context to the HITL UI
        (when ``--hitl`` is active) for visualization.  The recovery
        decision depends on the mode:

        ``gt``      — automatic rule-based recovery (grasp correct
                      object).  HITL UI shows what happened but does
                      NOT pause.
        ``gt_hitl`` — pauses and waits for the human to pick an action
                      from the HITL UI.
        ``gt_vlm``  — asks VLM to decide recovery.
        """
        sg_idx = state.current_subgoal_idx
        sg = state.subgoals[sg_idx] if state.subgoals else "?"
        ft = gt_result.failure_type

        logger.warning(
            f"  ⚠ GT failure [{ft.value}] on subgoal [{sg_idx + 1}/"
            f"{len(state.subgoals)}] \"{sg}\" — {gt_result.reason}"
        )

        # Store event for video annotation
        state._last_failure_event = {
            "reason": f"GT:{ft.value} — {gt_result.reason}",
            "confidence": 1.0,
            "recovery_mode": self._recovery_mode,
            "gt_failure_type": ft.value,
            "grasped_object": gt_result.grasped_object,
            "target_objects": gt_result.target_objects,
        }

        # Log the GT failure
        state.log({
            "type": "gt_failure_detected",
            "failure_type": ft.value,
            "reason": gt_result.reason,
            "subgoal_idx": sg_idx,
            "grasped_object": gt_result.grasped_object,
            "target_objects": gt_result.target_objects,
            "objects_completed": gt_result.objects_completed,
            "objects_remaining": gt_result.objects_remaining,
            "suggested_actions": gt_result.suggested_actions,
        })

        # Build GT failure context dict for HITL UI
        gt_failure_ctx = {
            "type": ft.value,
            "reason": gt_result.reason,
            "grasped": gt_result.grasped_object,
            "targets": gt_result.target_objects,
            "container": gt_result.target_container,
            "remaining": gt_result.objects_remaining,
            "completed": gt_result.objects_completed,
            "near_miss_distance": gt_result.nearest_miss_distance,
            "near_miss_object": gt_result.nearest_miss_object,
            "suggested_actions": gt_result.suggested_actions,
        }

        # Always push GT failure context to HITL UI (if --hitl active)
        # so the human can see what's happening regardless of mode.
        if self._hitl is not None:
            self._hitl.update_status(
                failure_type=f"GT:{ft.value}",
                gt_failure=gt_failure_ctx,
                status_message=(
                    f"⚠ GT: {ft.value} — {gt_result.reason[:80]}"
                ),
            )

        # Determine target for grasp tool.
        #
        # Two names matter:
        #   _gt_name     — simulator object name (e.g. "husky_hammer",
        #                  "red_block").  Needed by GT_SIM segmentation
        #                  to resolve body/instance IDs.
        #   grasp_target — instruction-level name (e.g. "black hammer",
        #                  "red block").  Needed by SAM3/GDino to find
        #                  the object visually.
        #
        # When using gt_sim segmentation we pass the sim name so the
        # grasp tool can look up the correct body ID.  For visual
        # segmentation modes (gdino_sam2, sam3) we use the instruction
        # name which describes the object's appearance.
        grasp_target = None
        _gt_name = None
        if gt_result.objects_remaining:
            _gt_name = gt_result.objects_remaining[0]
        elif gt_result.target_objects:
            _gt_name = gt_result.target_objects[0]

        if _gt_name and self._gt_detector is not None:
            grasp_target = self._gt_detector.instruction_name(_gt_name)
        elif _gt_name:
            grasp_target = _gt_name.replace("_", " ")

        # For GT_SIM segmentation, prefer the sim name so that
        # _resolve_gt_body_id can find the object by its exact name.
        grasp_target_for_tool = grasp_target
        if self._grasp_seg_mode == "gt_sim" and _gt_name:
            grasp_target_for_tool = _gt_name

        logger.debug(
            f"  [GT-HANDLE] mode={self._failure_monitor_mode}, "
            f"target={grasp_target_for_tool!r}"
        )

        # --- SUBTASK_REGRESSION: always replan, never grasp ---
        # Regression means previously-completed work is undone (e.g. a
        # block knocked off a stack).  Grasping a single object can't
        # fix that — the task needs to be replanned.  Escalation order:
        #   1. HITL (if available): pause and let human replan
        #   2. VLM replan (if available): ask VLM for new subgoals
        #   3. Acknowledge and resume (GT detector will re-fire)
        if gt_result.failure_type == GTFailureType.SUBTASK_REGRESSION:
            return self._handle_regression(
                obs, state, gt_result, grasp_target_for_tool,
            )

        # --- Mode: gt_hitl — human decides (PAUSE) ---
        if self._failure_monitor_mode == "gt_hitl" and self._hitl is not None:
            return self._gt_hitl_recovery(
                obs, state, gt_result, grasp_target_for_tool,
            )

        # --- Mode: gt_vlm — VLM decides ---
        if self._failure_monitor_mode == "gt_vlm":
            return self._gt_vlm_recovery(
                obs, state, gt_result, grasp_target_for_tool,
            )

        # --- Mode: gt (auto) — rule-based recovery (no pause) ---
        return self._gt_auto_recovery(obs, state, gt_result, grasp_target_for_tool)

    def _gt_auto_recovery(
        self,
        obs: dict,
        state: SessionState,
        gt_result: GTFailureResult,
        grasp_target: str | None,
    ) -> tuple[dict, SessionState]:
        """Automatic GT recovery: grasp correct object, skip, or retry.

        Does NOT pause.  The HITL UI (if active) shows a brief flash
        of the GT failure context for visibility, then auto-recovery
        proceeds.
        """
        ft = gt_result.failure_type

        logger.info(
            f"  [GT-AUTO-RECOVERY] ft={ft.value}, "
            f"grasp_target={grasp_target!r}, "
            f"remaining={gt_result.objects_remaining}, "
            f"targets={gt_result.target_objects}, "
            f"grasped={gt_result.grasped_object}"
        )

        # SUBTASK_REGRESSION is intercepted in _handle_gt_failure
        # before mode dispatch.  This guard is a safety net.
        if ft == GTFailureType.SUBTASK_REGRESSION:
            return self._handle_regression(
                obs, state, gt_result, grasp_target,
            )

        # For failures where grasp tool makes sense
        _grasp_types = (
            GTFailureType.WRONG_OBJECT_PICKED,
            GTFailureType.OBJECT_DROPPED,
            GTFailureType.NO_PROGRESS,
        )
        logger.info(
            f"  [GT-AUTO] grasp check: ft={ft.value} in types={ft in _grasp_types}, "
            f"grasp_target={grasp_target!r} (bool={bool(grasp_target)})"
        )
        if ft in _grasp_types and grasp_target:
            activated = self._activate_grasp_tool_for_gt(
                obs, state, gt_result, grasp_target,
            )
            logger.info(f"  [GT-AUTO] _activate_grasp_tool_for_gt returned {activated}")
            if activated:
                return obs, state
            # Grasp tool failed to start (e.g. IK failure after pose
            # prediction).  In grasp_first mode, just acknowledge and
            # let the GT detector re-fire on the next cycle rather than
            # falling through to VLM replan / retry text instructions.
            if self._recovery_mode == "grasp_first":
                logger.warning(
                    "  grasp_first: grasp tool failed — "
                    "acknowledging, will retry on next GT detection"
                )
                if self._gt_detector is not None:
                    self._gt_detector.acknowledge()
                return obs, state

        # Cooldown GT detector to prevent immediate re-fire
        if self._gt_detector is not None:
            self._gt_detector.acknowledge()

        # Fallback: retry with corrective instruction
        self._failure_retries += 1
        state.step_at_subgoal_start = state.episode_step
        state.step_at_last_check = state.episode_step

        if grasp_target:
            state.rewritten_instruction = (
                f"Pick up the {grasp_target.replace('_', ' ')} and place it "
                f"in the {(gt_result.target_container or 'target').replace('_', ' ')}"
            )
        else:
            state.rewritten_instruction = gt_result.current_subgoal
        state.flush_actions = True

        logger.info(
            f"  ↻ GT auto-recovery: \"{state.rewritten_instruction[:80]}\""
        )

        if self._failure_retries >= self.MAX_FAILURE_RETRIES:
            # Dummy signal-based result for replan_or_skip compatibility
            from vlm_orchestrator.failure_handlers.signal_detector import (
                ClassificationResult, SignalFeatures,
            )
            dummy = ClassificationResult(
                status=ManipulationStatus.FAILURE,
                reason=gt_result.reason,
                confidence=1.0,
                features=SignalFeatures(),
            )
            return self._replan_or_skip(obs, state, dummy)

        return obs, state

    # ------------------------------------------------------------------
    # Regression recovery (shared across all GT failure-monitor modes)
    # ------------------------------------------------------------------

    def _handle_regression(
        self,
        obs: dict,
        state: SessionState,
        gt_result: GTFailureResult,
        grasp_target: str | None,
    ) -> tuple[dict, SessionState]:
        """Handle SUBTASK_REGRESSION regardless of failure-monitor mode.

        Regression means previously-completed work is undone (e.g. a
        block knocked off a stack).  Grasping a single object cannot
        fix that — the task needs a full replan.  Escalation order:

        1. **HITL** (if ``--hitl`` active): pause and let the human
           replan via the HITL UI.
        2. **VLM replan** (if VLM available): ask the VLM for new
           subgoals from the current physical state.
        3. **Acknowledge** and resume — the GT detector will re-detect
           the regression on subsequent cycles.
        """
        print(f"\n{'='*60}")
        print(f"[REGRESSION] detected: {gt_result.reason[:80]}")
        print(f"{'='*60}")
        state.log({
            "type": "gt_regression_replan",
            "regressed_objects": gt_result.objects_remaining,
            "reason": gt_result.reason,
            "subgoal_idx": state.current_subgoal_idx,
        })

        # ── Escalation 1: VLM auto-replan ──
        image = self.ctx.get_vlm_image(obs)
        print(f"[REGRESSION] image={'YES' if image is not None else 'NO'}, "
              f"initial_image={'YES' if state.initial_image is not None else 'NO'}")
        regression_ctx = self._build_regression_context(
            obs, state, gt_result,
        )
        recycle_ok = False
        if image is not None:
            recycle_ok = self._recycle(
                obs, state, image,
                regression_context=regression_ctx,
            )
        print(f"[REGRESSION] _recycle returned: {recycle_ok}")
        if recycle_ok:
            # _recycle handles detection reset, GT config, flush,
            # and subclass hooks.
            logger.info(
                f"  ✅ Regression → VLM replanned: "
                f"{len(state.subgoals)} subgoal(s), "
                f"starting with \"{state.subgoals[0][:60]}\""
            )
            if state.rewritten_instruction:
                obs = self.ctx.set_prompt(
                    obs, state.rewritten_instruction,
                )
            return obs, state

        # ── Escalation 2: HITL — let the human decide ──
        if self._hitl is not None:
            logger.info(
                "  👤 Regression → VLM replan failed, "
                "falling back to HITL pause"
            )
            self._hitl.update_status(
                is_paused=True,
                failure_type="GT:subtask_regression",
                status_message=(
                    f"⚠ REGRESSION: {gt_result.reason[:80]} — "
                    f"VLM replan failed, please replan manually"
                ),
            )
            return self._gt_hitl_recovery(
                obs, state, gt_result, grasp_target,
            )

        # ── Escalation 3: no HITL, VLM failed — retry current subgoal ──
        logger.warning(
            "  Replan failed after regression (no HITL, VLM failed) — "
            "retrying current subgoal from scratch"
        )
        state.flush_actions = True
        state.step_at_subgoal_start = state.episode_step
        state.step_at_last_check = state.episode_step
        self._failure_retries = 0
        self._grasp_attempts = 0
        if self._failure_detector is not None:
            self._failure_detector.reset()
        if self._gt_detector is not None:
            self._gt_detector.acknowledge()
        return obs, state

    def _build_regression_context(
        self,
        obs: dict,
        state: SessionState,
        gt_result: GTFailureResult,
    ) -> str:
        """Build a context string describing the regression for VLM."""
        gt_state = obs.get("gt_state", {})
        grasped = gt_state.get("robot", {}).get("grasped_object")
        current_subgoal = (
            state.subgoals[state.current_subgoal_idx]
            if state.subgoals
            else "unknown"
        )
        ctx_lines = [
            f"Previously completed conditions that are now "
            f"UNDONE: {gt_result.reason}",
            f"The robot was working on: \"{current_subgoal}\"",
        ]
        if grasped:
            ctx_lines.append(
                f"The robot is currently holding: {grasped}. "
                f"You may need to put it down first or use it."
            )
        else:
            ctx_lines.append(
                "The robot is not holding anything."
            )
        ctx_lines.append(
            "Do NOT include steps for objects that are already "
            "in their correct position (e.g. do not say 'place "
            "red block on table' if red is already the base)."
        )
        return "\n".join(ctx_lines)

    def _gt_vlm_recovery(
        self,
        obs: dict,
        state: SessionState,
        gt_result: GTFailureResult,
        grasp_target: str | None,
    ) -> tuple[dict, SessionState]:
        """VLM-assisted GT recovery: VLM sees GT context + image."""
        image = self.ctx.get_vlm_image(obs)
        ft = gt_result.failure_type

        # Build rich prompt with GT context
        gt_context = (
            f"GROUND TRUTH FAILURE DETECTED:\n"
            f"  Type: {ft.value}\n"
            f"  Reason: {gt_result.reason}\n"
            f"  Subgoal: \"{gt_result.current_subgoal}\"\n"
            f"  Target objects: {gt_result.target_objects}\n"
            f"  Container: {gt_result.target_container}\n"
            f"  Currently grasped: {gt_result.grasped_object}\n"
            f"  Completed: {gt_result.objects_completed}\n"
            f"  Remaining: {gt_result.objects_remaining}\n"
        )
        if gt_result.nearest_miss_distance is not None:
            gt_context += (
                f"  Nearest miss: {gt_result.nearest_miss_object} "
                f"at {gt_result.nearest_miss_distance*100:.1f}cm\n"
            )
        gt_context += (
            f"\nSuggested actions: {gt_result.suggested_actions}\n\n"
            f"Respond with JSON: "
            f'{{ "action": "grasp_tool" | "retry" | "skip" | "replan", '
            f'"target": "<object_name>", '
            f'"instruction": "<new instruction for VLA>" }}'
        )

        user_content = []
        if image is not None:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_b64(image)}",
                },
            })
        user_content.append({"type": "text", "text": gt_context})

        try:
            raw = self._vlm_call(
                "You are a robot recovery planner. A ground-truth failure "
                "detector reports exactly what went wrong. Decide the best "
                "recovery action.",
                user_content,
            )
            data = parse_json(raw)
        except Exception as e:
            logger.warning(f"  GT VLM recovery call failed: {e}")
            return self._gt_auto_recovery(obs, state, gt_result, grasp_target)

        action = data.get("action", "retry")
        target = data.get("target", grasp_target)
        instruction = data.get("instruction", gt_result.current_subgoal)

        logger.info(
            f"  🧠 GT VLM decision: action={action}, target={target}"
        )

        state.log({
            "type": "gt_vlm_recovery",
            "action": action,
            "target": target,
            "instruction": instruction,
            "gt_failure_type": ft.value,
        })

        if action == "grasp_tool" and target:
            activated = self._activate_grasp_tool_for_gt(
                obs, state, gt_result, target,
            )
            if activated:
                return obs, state
            # Grasp tool failed to start — fall through to retry

        elif action == "replan":
            logger.info("  🔄 GT VLM chose replan")
            replan_image = self.ctx.get_vlm_image(obs)
            if replan_image is not None and self._recycle(
                obs, state, replan_image,
            ):
                # _recycle handles detection reset, GT config, flush,
                # and subclass hooks.  Just acknowledge the GT failure.
                if self._gt_detector is not None:
                    self._gt_detector.acknowledge()
                if state.rewritten_instruction:
                    obs = self.ctx.set_prompt(
                        obs, state.rewritten_instruction,
                    )
                return obs, state
            logger.warning("  Replan failed, falling through to retry")

        elif action == "skip":
            if self._gt_detector is not None:
                self._gt_detector.acknowledge()
            if state.current_subgoal_idx < len(state.subgoals) - 1:
                self._advance_subgoal(obs, state)
                return obs, state
            # Last subgoal — can't skip, fall through to retry

        # Default / fallback: retry with new instruction
        if self._gt_detector is not None:
            self._gt_detector.acknowledge()
        self._failure_retries += 1
        state.step_at_subgoal_start = state.episode_step
        state.step_at_last_check = state.episode_step
        state.rewritten_instruction = instruction
        return obs, state

    def _gt_hitl_recovery(
        self,
        obs: dict,
        state: SessionState,
        gt_result: GTFailureResult,
        grasp_target: str | None,
    ) -> tuple[dict, SessionState]:
        """Human-in-the-loop GT recovery: pause and wait for human.

        The HITL UI shows the GT failure panel with context (what was
        grasped, what should have been grasped, distances, etc.) and
        suggested action buttons.  The robot stops and waits for the
        human to click an action: Grasp <target>, Resume, Skip, etc.

        When ``--recovery-mode grasp_first`` is set, automatically
        activates the grasp tool for grasp-related failures instead
        of pausing.  The HITL UI still shows what happened (the human
        can monitor the grasp execution), but the robot doesn't stop.
        """
        if self._hitl is None:
            # Fallback if HITL not available
            return self._gt_auto_recovery(obs, state, gt_result, grasp_target)

        ft = gt_result.failure_type

        # ── grasp_first: auto-activate grasp tool, don't pause ──
        # For grasp-related GT failures, skip the human decision loop
        # and go straight to the grasp tool.  The HITL UI still shows
        # the failure context so the human can watch and intervene if
        # needed (e.g. click Abort or Skip).
        _grasp_gt_types = (
            GTFailureType.WRONG_OBJECT_PICKED,
            GTFailureType.OBJECT_DROPPED,
            GTFailureType.NO_PROGRESS,
        )
        if (
            self._recovery_mode == "grasp_first"
            and ft in _grasp_gt_types
            and grasp_target
        ):
            logger.info(
                f"  🤏 GT HITL + grasp_first: auto-activating grasp "
                f"tool for '{grasp_target}' (ft={ft.value})"
            )
            self._hitl.update_status(
                status_message=(
                    f"⚠ GT: {ft.value} — grasp_first: auto-grasping "
                    f"'{grasp_target}'"
                ),
            )
            activated = self._activate_grasp_tool_for_gt(
                obs, state, gt_result, grasp_target,
            )
            if activated:
                state.log({
                    "type": "gt_hitl_grasp_first",
                    "failure_type": ft.value,
                    "target": grasp_target,
                    "subgoal_idx": state.current_subgoal_idx,
                })
                return obs, state
            # Grasp tool failed to start — fall through to manual pause
            logger.warning(
                "  grasp_first: grasp tool failed to start, "
                "falling back to manual HITL pause"
            )

        logger.info(
            f"  👤 GT HITL: pausing for human decision "
            f"(failure={ft.value}, target={grasp_target})"
        )

        # Update HITL UI: mark paused so human sees the failure
        self._hitl.update_status(
            is_paused=True,
            status_message=(
                f"⚠ GT FAILURE: {ft.value} — choose an action below"
            ),
        )

        state.log({
            "type": "gt_hitl_pause",
            "failure_type": ft.value,
            "reason": gt_result.reason,
            "suggested_target": grasp_target,
            "subgoal_idx": state.current_subgoal_idx,
        })

        # Block the proxy loop — robot stops moving.
        # Human sees the GT failure panel and clicks an action button.
        # The existing _hitl_pause_loop handles: Resume, Recovery,
        # Skip, Abort, Grasp, Subgoal Done, etc.
        obs, state = self._hitl_pause_loop(obs, state)

        # After human acts, clear GT failure from UI
        self._hitl.update_status(
            gt_failure=None,
            failure_type="",
        )

        # Reset the GT detector's per-subgoal counters so that
        # no_progress doesn't immediately re-fire.
        if self._gt_detector is not None:
            self._gt_detector.acknowledge()

        state.log({
            "type": "gt_hitl_resumed",
            "subgoal_idx": state.current_subgoal_idx,
        })

        return obs, state

    def _activate_grasp_tool_for_gt(
        self,
        obs: dict,
        state: SessionState,
        gt_result: GTFailureResult,
        target: str,
    ) -> bool:
        """Activate grasp tool for a GT-identified target object."""
        sg_idx = state.current_subgoal_idx
        self._grasp_attempts += 1

        logger.info(
            f"  🤏 GT GRASP [attempt {self._grasp_attempts}]: "
            f"'{target}' (gt_type={gt_result.failure_type.value})"
        )

        if state.grasp_tool_executor is None:
            from vlm_orchestrator.grasp.tool import GraspToolExecutor
            state.grasp_tool_executor = GraspToolExecutor(
                seg_mode=self._grasp_seg_mode,
                env_mode=self._env_mode,
                use_front_camera=self._use_front_camera,
                topdown_threshold=self._grasp_topdown_threshold,
                motion_planner=self._get_motion_planner(),
                stack_mode_enabled=self._stack_mode_enabled,
            )

        if self._hitl is not None:
            state._hitl = self._hitl

        state.grasp_tool_executor.start(
            target_object=target, obs=obs, state=state,
        )
        state.grasp_tool_active = state.grasp_tool_executor.is_active
        state.grasp_tool_phase = state.grasp_tool_executor.phase.value
        state.grasp_tool_target = target

        if state.grasp_tool_active:
            # Flush the eval client's cached action chunk so it
            # re-infers immediately — otherwise the client keeps
            # replaying stale VLA actions and never receives the
            # grasp-tool metadata / planned actions.
            state.flush_actions = True
            if self._gt_detector is not None:
                self._gt_detector.acknowledge()
            state.log({
                "type": "gt_grasp_escalation",
                "target_object": target,
                "gt_failure_type": gt_result.failure_type.value,
                "gt_reason": gt_result.reason,
                "subgoal_idx": sg_idx,
                "attempt": self._grasp_attempts,
            })
            return True

        fail_msg = getattr(state.grasp_tool_executor, '_status_message', '')
        logger.error(
            f"\n{'='*60}\n"
            f"  ❌ GT GRASP TOOL FAILED TO START\n"
            f"  target  = '{target}'\n"
            f"  seg_mode= {self._grasp_seg_mode}\n"
            f"  reason  = {fail_msg}\n"
            f"  phase   = {state.grasp_tool_executor.phase.value}\n"
            f"{'='*60}"
        )
        state.log({
            "type": "gt_grasp_failed_to_start",
            "target_object": target,
            "seg_mode": self._grasp_seg_mode,
            "reason": fail_msg,
            "gt_failure_type": gt_result.failure_type.value,
        })
        if self._gt_detector is not None:
            self._gt_detector.acknowledge()
        return False

    # ------------------------------------------------------------------
    # Failure recovery helpers
    # ------------------------------------------------------------------

    def _recover_with_instruction(
        self, obs, state, sg, result,
    ) -> tuple[dict, SessionState]:
        """Generate a corrective instruction and feed it to the VLA."""
        sg_idx = state.current_subgoal_idx
        self._failure_retries += 1
        state.step_at_subgoal_start = state.episode_step
        state.step_at_last_check = state.episode_step
        if self._failure_detector is not None:
            self._failure_detector.reset()

        image = self.ctx.get_vlm_image(obs)
        mode = self._recovery_mode
        if mode in ("grasp_first", "vlm_grasp"):
            mode = "template"  # these modes handle their own logic above
        recovery = generate_recovery(
            failure_result=result,
            subgoal_instruction=sg,
            task_instruction=state.original_instruction or sg,
            current_image=image,
            vlm_call_fn=(
                self._vlm_call if mode == "vlm" else None
            ),
            retry_count=self._failure_retries - 1,
            mode=mode,
        )

        state.rewritten_instruction = recovery.instruction
        state.flush_actions = True
        logger.info(
            f"  ↻ RECOVER {self._failure_retries}/"
            f"{self.MAX_FAILURE_RETRIES} [{recovery.method}]: "
            f"\"{recovery.instruction[:80]}\""
        )
        state.log({
            "type": "failure_recovery",
            "action": "recover",
            "method": recovery.method,
            "retry_count": self._failure_retries,
            "failure_type": recovery.failure_type,
            "failure_reason": result.reason,
            "confidence": round(result.confidence, 3),
            "original_instruction": recovery.original_instruction,
            "recovery_instruction": recovery.instruction,
            "subgoal_idx": sg_idx,
        })
        return obs, state

    def _activate_grasp_tool(
        self, obs, state, sg, result, *, target_override=None,
    ) -> bool:
        """Activate the grasp tool executor.  Returns True if activated.

        Per-subgoal attempt cap (``self._max_grasp_attempts``, default 2,
        configurable via ``--max-grasp-attempts``) applies in VLM-handler
        / template / retry modes.  ``grasp_first`` and ``vlm_grasp``
        modes are exempt: ``grasp_first``'s GT detector resets after
        each grasp, and ``vlm_grasp`` lets the VLM pick on each failure
        so it self-limits via ``MAX_FAILURE_RETRIES``.
        """
        if (self._recovery_mode not in ("grasp_first", "vlm_grasp")
                and self._grasp_attempts >= self._max_grasp_attempts):
            logger.info(
                f"  🤏 Grasp attempts exhausted "
                f"({self._grasp_attempts}/{self._max_grasp_attempts}), "
                f"skipping grasp tool"
            )
            return False

        sg_idx = state.current_subgoal_idx
        target = target_override or _extract_target_object(sg)
        self._grasp_attempts += 1

        logger.info(
            f"  🤏 GRASP ESCALATION [attempt {self._grasp_attempts}]: "
            f"lifting arm + planned grasp for '{target}'"
        )
        if state.grasp_tool_executor is None:
            from vlm_orchestrator.grasp.tool import GraspToolExecutor
            state.grasp_tool_executor = GraspToolExecutor(seg_mode=self._grasp_seg_mode, env_mode=self._env_mode, use_front_camera=self._use_front_camera, topdown_threshold=self._grasp_topdown_threshold, motion_planner=self._get_motion_planner(), stack_mode_enabled=self._stack_mode_enabled)

        if self._hitl is not None:
            state._hitl = self._hitl

        state.grasp_tool_executor.start(
            target_object=target, obs=obs, state=state,
        )
        state.grasp_tool_active = state.grasp_tool_executor.is_active
        state.grasp_tool_phase = state.grasp_tool_executor.phase.value
        state.grasp_tool_target = target

        if state.grasp_tool_active:
            # Flush the eval client's cached action chunk so it
            # re-infers immediately — otherwise the client keeps
            # replaying stale VLA actions and never receives the
            # grasp-tool metadata / planned actions.
            state.flush_actions = True
            state.log({
                "type": "failure_recovery",
                "action": "grasp_escalation",
                "target_object": target,
                "failure_reason": result.reason,
                "confidence": round(result.confidence, 3),
                "subgoal_idx": sg_idx,
            })
            return True

        logger.warning("  Grasp tool failed to start")
        return False

    def _replan_or_skip(
        self, obs, state, result,
    ) -> tuple[dict, SessionState]:
        """Replan remaining subgoals via VLM, or skip to next."""
        sg_idx = state.current_subgoal_idx

        if self._recycle_count < MAX_RECYCLES:
            logger.info("  ↻ REPLAN: re-decomposing task")
            image = self.ctx.get_vlm_image(obs)
            if image is not None and self._recycle(obs, state, image):
                # _recycle handles detection reset, GT config, flush,
                # and subclass hooks.
                state.log({
                    "type": "failure_recovery",
                    "action": "replan",
                    "failure_reason": result.reason,
                    "confidence": round(result.confidence, 3),
                    "new_subgoals": state.subgoals,
                })
                return obs, state
            else:
                logger.warning("  Replan failed, falling through to skip")

        if sg_idx < len(state.subgoals) - 1:
            logger.info("  ⏭ SKIP: giving up on subgoal, advancing")
            self._advance_subgoal(obs, state)
            self._failure_retries = 0
            state.log({
                "type": "failure_recovery",
                "action": "skip",
                "failure_reason": result.reason,
                "confidence": round(result.confidence, 3),
                "subgoal_idx": state.current_subgoal_idx,
            })
        else:
            logger.info("  ⚠ Last subgoal, cannot skip — continuing")
            self._failure_retries = 0
            if self._failure_detector is not None:
                self._failure_detector.reset()
            state.log({
                "type": "failure_recovery",
                "action": "continue_last",
                "failure_reason": result.reason,
                "confidence": round(result.confidence, 3),
                "subgoal_idx": sg_idx,
            })

        return obs, state

    # ------------------------------------------------------------------
    # Shared state-transition helpers
    # ------------------------------------------------------------------

    def _reset_detection_state(
        self, obs: dict, state: SessionState,
    ) -> None:
        """Reset all failure detection counters and detectors.

        Called after any subgoal change (advance, replan, recovery).
        """
        state.step_at_subgoal_start = state.episode_step
        state.step_at_last_check = state.episode_step
        self._failure_retries = 0
        self._grasp_attempts = 0
        if self._failure_detector is not None:
            self._failure_detector.reset()
        if self._failure_handler is not None:
            self._failure_handler.on_subgoal_advanced(
                obs, state, state.current_subgoal_idx,
            )

    def _configure_gt_detector(
        self, obs: dict, state: SessionState, instruction: str,
    ) -> None:
        """Configure the GT detector for a new subgoal instruction."""
        if self._gt_detector is not None:
            gt_state = obs.get("gt_state")
            scene_objects = (
                gt_state.get("scene_objects", []) if gt_state else []
            )
            gt_conditions = (
                gt_state.get("subtask", {}).get("conditions", [])
                if gt_state else []
            )
            self._gt_detector.set_subgoal(
                instruction=instruction,
                scene_objects=scene_objects,
                gt_conditions=gt_conditions,
            )

    def _apply_new_instruction(
        self, obs: dict, state: SessionState, instruction: str,
    ) -> None:
        """Set a new instruction with full detection reset.

        Used for recovery and rewrite actions.
        """
        state.rewritten_instruction = instruction
        state.flush_actions = True
        self._reset_detection_state(obs, state)
        self._configure_gt_detector(obs, state, instruction)

    def _apply_new_subgoals(
        self, obs: dict, state: SessionState, subgoals: list[str],
    ) -> None:
        """Replace the subgoal list and configure detectors.

        Used for replan, HITL START_SUBGOAL, etc.
        """
        state.subgoals = subgoals
        state.subgoals_ordered = True
        state.current_subgoal_idx = 0
        state.rewritten_instruction = subgoals[0]
        state.flush_actions = True
        # Save original subgoals so _recycle can constrain the VLM
        # to pick from the original list.  Only set on the first call
        # (episode start); subsequent replans should keep the original.
        if not self._original_subgoals:
            self._original_subgoals = list(subgoals)
        self._reset_detection_state(obs, state)
        self._configure_gt_detector(obs, state, subgoals[0])

    # ------------------------------------------------------------------
    # Subgoal advancement
    # ------------------------------------------------------------------

    def _advance_subgoal(self, obs: dict, state: SessionState) -> None:
        """Move to the next subgoal.  Does NOT log — caller logs."""
        state.current_subgoal_idx += 1
        new_sg = state.subgoals[state.current_subgoal_idx]
        state.rewritten_instruction = new_sg
        self._reset_detection_state(obs, state)
        self._configure_gt_detector(obs, state, new_sg)
        logger.info(
            f"  → Subgoal [{state.current_subgoal_idx + 1}/"
            f"{len(state.subgoals)}]: \"{new_sg}\""
        )
        self._on_subgoal_advanced(obs, state, state.current_subgoal_idx)

    # ------------------------------------------------------------------
    # Recycling
    # ------------------------------------------------------------------

    def _recycle(
        self,
        obs: dict,
        state: SessionState,
        current_image: np.ndarray,
        regression_context: str | None = None,
    ) -> bool:
        """Send before/after to VLM, delegate result to subclass.

        Parameters
        ----------
        regression_context:
            When set, the recycle was triggered by a SUBTASK_REGRESSION.
            Contains a human-readable description of which physics
            conditions regressed (e.g. "blue block is no longer stacked
            on red block").  Injected into the VLM prompt so it can
            correctly replan even when the visual difference is subtle.

        Returns ``True`` if new subgoals were set up.
        """
        self._recycle_count += 1
        logger.info(
            f"  Recycling attempt {self._recycle_count}/{MAX_RECYCLES}"
        )

        text = (
            f'Task: "{state.original_instruction}"\n\n'
            f"Image 1 is BEFORE (episode start). "
            f"Image 2 is NOW (current).\n"
        )
        if self._original_subgoals:
            numbered = "\n".join(
                f"  {i+1}. {sg}"
                for i, sg in enumerate(self._original_subgoals)
            )
            text += (
                f"\nORIGINAL SUBGOAL LIST (select ONLY from these, "
                f"using the exact wording):\n{numbered}\n"
            )
        if regression_context:
            text += (
                f"\n⚠️ SUBTASK REGRESSION DETECTED by the physics "
                f"engine:\n{regression_context}\n"
                f"Trust the physics report over visual appearance "
                f"(objects may look close together but are not actually "
                f"in the correct position). "
            )
        text += (
            "\nList ALL remaining subgoals to complete the task "
            "from the current state."
        )

        # Only send primary camera — wrist close-ups can confuse
        # the VLM into hallucinating extra objects.
        primary_label, _ = self.ctx.vlm_camera_labels
        user_content = self._build_check_message(
            text, state.initial_image, current_image,
            extra_images=None, initial_extra_images=None,
            primary_label=primary_label,
        )

        recycle_prompt = (
            RECYCLE_SYSTEM_PROMPT_VLABENCH
            if self.ctx.prompt_style == "vlabench"
            else RECYCLE_SYSTEM_PROMPT
        )
        recycle_prompt = _with_vla_front_cam_note(
            recycle_prompt,
            use_front_camera=bool(
                getattr(self.ctx, "front_image_key", None)
            ),
        )
        t0 = time.time()
        try:
            raw = self._vlm_call(
                recycle_prompt, user_content, max_tokens=1024,
            )
            elapsed = time.time() - t0
        except Exception as e:
            print(f"[RECYCLE] ❌ VLM call FAILED: {e}")
            state.log({
                "type": "recycle_vlm_error",
                "error": str(e),
                "recycle_count": self._recycle_count,
            })
            return False

        print(f"[RECYCLE] VLM responded in {elapsed:.1f}s: {raw[:200]}")

        try:
            data = parse_json(raw)
            if data.get("done", False) or not data.get("subgoals"):
                print(f"[RECYCLE] ❌ VLM says done/no subgoals: {data}")
                state.log({
                    "type": "recycle_rejected",
                    "reason": "done=true or empty subgoals",
                    "vlm_raw": raw[:500],
                    "recycle_count": self._recycle_count,
                })
                return False
            result = self._apply_recycle_subgoals(obs, state, data)
            if result:
                # Centralized post-recycle cleanup: reset detectors,
                # configure GT, flush stale action chunks, and notify
                # subclass hooks (e.g. scene-edit GDino re-detection).
                state.flush_actions = True
                self._reset_detection_state(obs, state)
                self._configure_gt_detector(
                    obs, state,
                    state.subgoals[state.current_subgoal_idx],
                )
                self._on_subgoal_advanced(
                    obs, state, state.current_subgoal_idx,
                )
            print(f"[RECYCLE] ✅ new subgoals: {data.get('subgoals')}")
            return result
        except (ValueError, KeyError) as e:
            print(f"[RECYCLE] ❌ parse failed: {e}")
            state.log({
                "type": "recycle_parse_error",
                "error": str(e),
                "vlm_raw": raw[:500],
                "recycle_count": self._recycle_count,
            })
            return False

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _decompose_and_setup(
        self, obs: dict, state: SessionState,
        instruction: str, image: np.ndarray,
    ) -> None:
        """Decompose instruction and set up ``state.subgoals``."""
        raise NotImplementedError

    def _check_subgoal(
        self, obs: dict, state: SessionState, image: np.ndarray,
    ) -> tuple[dict, SessionState]:
        """VLM progress check.  Handles advance + recycle internally."""
        raise NotImplementedError

    def _apply_recycle_subgoals(
        self, obs: dict, state: SessionState, data: dict,
    ) -> bool:
        """Apply parsed recycle response.  Return True if new subgoals."""
        raise NotImplementedError

    def _on_frame(self, obs: dict, state: SessionState) -> None:
        """Per-frame processing (e.g. GDino re-detection)."""

    def _on_subgoal_advanced(
        self, obs: dict, state: SessionState, idx: int,
    ) -> None:
        """Called after advancing to subgoal *idx*."""

    def _post_process(
        self, obs: dict, state: SessionState,
    ) -> tuple[dict, SessionState]:
        """Post-step processing (e.g. scene edits)."""
        return obs, state

    def _should_check(self, state: SessionState) -> bool:
        """Whether VLM checks are active for this state."""
        return True

    def _get_timeout(self, state: SessionState) -> int:
        """Subgoal timeout in chunks."""
        return self.config.subgoal_timeout
