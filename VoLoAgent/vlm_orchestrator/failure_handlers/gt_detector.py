# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth object-level failure detector.

Consumes ``gt_state`` dicts (produced by the robolab-side
``GTStateExporter``) and classifies object-level failures with rich
contextual information for recovery decision-making.

The detector operates per-subgoal:
  1. ``set_subgoal()`` — configure target objects and container
  2. ``update(gt_state)`` — feed one step, returns current status
  3. On failure, the ``GTFailureResult`` includes enough context for
     a human (HITL) or VLM to pick a sensible recovery action.

⚠ **Scope warning.**  The rules in this module (WRONG_OBJECT_PICKED,
OBJECT_DROPPED, NO_PROGRESS, SUBTASK_REGRESSION, SUBGOAL_COMPLETE) and
their thresholds (e.g. the 5-step regression confirmation) were
developed and tuned against the robolab **block-stack** task suite.
They have NOT been validated on the Memory (LH) or Common Sense
(LH-CS) suites — CSM-condition coverage is sparser there, object-
contact / gripper heuristics may misfire on non-block geometries, and
SUBGOAL_COMPLETE depends on container / positional predicates that
aren't authored for every task.  Treat any GT-monitor result outside
block-stack as exploratory; prefer the ``vlm`` failure monitor for
general benchmarks (see ``failure_handlers/vlm.py``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Failure types
# ──────────────────────────────────────────────────────────────────────

class GTFailureType(str, Enum):
    """Object-level failure categories detectable from GT state."""
    SUBGOAL_COMPLETE = "subgoal_complete"
    IN_PROGRESS = "in_progress"
    WRONG_OBJECT_PICKED = "wrong_object_picked"
    OBJECT_DROPPED = "object_dropped"
    # Stable-state placement failure: object released and remained released
    # for a confirmation streak, but not in the target container.  Replaces
    # OBJECT_DROPPED for new code (the passive task-failure logger).  See
    # docs/instructions/lh-task-encoding-rules.md § "Failure-type design" for rationale.
    # The active GT recovery handler in subgoal_base.py still uses
    # OBJECT_DROPPED until the planned refactor.
    WRONG_TARGET_PLACE = "wrong_target_place"
    NO_PROGRESS = "no_progress"
    SUBTASK_REGRESSION = "subtask_regression"
    # Aspect-1 recovery: a previously-emitted failure event's
    # underlying condition has resolved.  ``GTFailureResult.reason``
    # describes which failure recovered (wrong-object grasp resolved;
    # misplaced object reached its target; regressed subtask satisfied
    # again; score climbed back to peak).
    RECOVERY = "recovery"


@dataclass
class GTFailureResult:
    """Rich failure context for decision-making."""
    failure_type: GTFailureType
    confidence: float = 1.0  # always 1.0 for ground-truth
    reason: str = ""

    # Subgoal context
    current_subgoal: str = ""
    target_objects: list[str] = field(default_factory=list)
    target_container: str | None = None

    # What's happening
    grasped_object: str | None = None
    grasped_is_correct: bool = False

    # Progress tracking
    objects_completed: list[str] = field(default_factory=list)
    objects_remaining: list[str] = field(default_factory=list)
    objects_displaced: dict[str, float] = field(default_factory=dict)

    # Placement precision
    nearest_miss_distance: float | None = None
    nearest_miss_object: str | None = None

    # Suggested recovery actions (ranked)
    suggested_actions: list[str] = field(default_factory=list)

    @property
    def is_failure(self) -> bool:
        return self.failure_type not in (
            GTFailureType.IN_PROGRESS,
            GTFailureType.SUBGOAL_COMPLETE,
        )


# ──────────────────────────────────────────────────────────────────────
# Subgoal → GT object mapping
# ──────────────────────────────────────────────────────────────────────

# Common container keywords
_CONTAINER_PATTERNS = {
    "bin": "bin", "tray": "tray", "bowl": "bowl", "pot": "pot",
    "plate": "plate", "mug": "mug", "cup": "cup", "box": "box",
    "shelf": "shelf", "container": "container", "basket": "basket",
}

# Known distinguishing-attribute words for conflict detection.
# If the instruction says "black" and an object name contains "red",
# that object should be excluded even if its base word ("hammer") matches.
# Despite the legacy name, this set isn't restricted to colors — it
# also includes material/size attributes that disambiguate sibling
# objects in the LH suite (e.g. wooden_bowl vs bowl, fork_big vs
# fork_small, blackandbrassbowl_small).  See docs/instructions/lh-task-encoding-rules.md.
_COLOR_WORDS = {
    # colors
    "red", "blue", "green", "yellow", "black", "white", "orange",
    "purple", "grey", "gray", "brown", "pink", "silver", "gold",
    # material / size attributes used in LH task object names
    "wooden", "brass", "ceramic", "big", "small",
}


def _core_noun(words: list[str]) -> str | None:
    """Return the core noun from a list of object-name words.

    The core noun is the longest non-color, non-trivial word (≥ 3 chars).
    E.g. ``["white", "yellow", "mug"]`` → ``"mug"``,
         ``["porcelain", "mug"]`` → ``"porcelain"`` (longer).
    """
    candidates = [
        w for w in words
        if len(w) >= 3 and w not in _COLOR_WORDS
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _extract_phrase_for_gt_name(instruction: str, gt_name: str) -> str | None:
    """Extract the instruction phrase that corresponds to a GT object name.

    E.g. gt_name="husky_hammer", instruction="pick up the black hammer"
    → finds "hammer" is the base word, looks at surrounding words in the
    instruction, returns "black hammer".
    """
    inst_lower = instruction.lower()
    gt_words = gt_name.lower().replace("_", " ").split()

    # Find which gt_words appear in the instruction (the "base" words)
    base_words = [w for w in gt_words if w in inst_lower and len(w) > 2]
    if not base_words:
        return None

    # Use the last base word as anchor (usually the noun: "hammer")
    anchor = base_words[-1]
    # Other gt_words serve as context to disambiguate duplicate anchors
    context_words = set(gt_words) - {anchor}

    # Tokenize instruction
    tokens = inst_lower.split()

    # Find ALL occurrences of the anchor in the token list
    anchor_indices = [
        i for i, tok in enumerate(tokens) if tok == anchor
    ]
    if not anchor_indices:
        # Anchor might be a substring; find tokens containing it
        anchor_indices = [
            i for i, tok in enumerate(tokens) if anchor in tok
        ]
    if not anchor_indices:
        return None

    # Collect adjective-like words before the anchor (up to 2 words back)
    # Stop at articles, prepositions, verbs.  Grasp verbs MUST stay
    # synced with strategies/subgoal_base._GRASP_VERBS — otherwise a
    # missing verb (e.g. "lift") leaks into the phrase as a fake
    # adjective ("lift bowl"), which GDino then grounds onto whatever
    # is actually being lifted in the scene (typically the gripper).
    _STOP_WORDS = {
        "the", "a", "an", "and", "or", "in", "on", "to", "from",
        "into", "with", "up",
        # grasp / placement verbs
        "pick", "put", "place", "grab", "move", "lift", "grasp", "drop",
        "it", "them", "is", "are", "then", "next", "first",
    }

    def _phrase_at(idx: int) -> str:
        """Build the phrase for the anchor at token index *idx*."""
        parts = []
        for i in range(max(0, idx - 2), idx):
            tok = tokens[i]
            if tok not in _STOP_WORDS:
                parts.append(tok)
        parts.append(tokens[idx])
        return " ".join(parts)

    # If only one occurrence, use it directly
    if len(anchor_indices) == 1:
        best_idx = anchor_indices[0]
    else:
        # Multiple occurrences — pick the one whose surrounding words
        # best match the other gt_words (e.g. "red" near "block" for
        # red_block).
        best_idx = anchor_indices[0]
        best_overlap = -1
        for idx in anchor_indices:
            # Check 2 words before the anchor for context overlap
            nearby = set()
            for i in range(max(0, idx - 2), idx):
                nearby.add(tokens[i])
            overlap = len(nearby & context_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

    phrase = _phrase_at(best_idx)

    # Reject if the extracted phrase has a color that conflicts with the
    # GT name's color.  E.g. gt_name="red_hammer", phrase="black hammer"
    # → the GT object isn't the one mentioned in this instruction.
    gt_colors = set(gt_words) & _COLOR_WORDS
    phrase_colors = set(phrase.split()) & _COLOR_WORDS
    if gt_colors and phrase_colors and gt_colors.isdisjoint(phrase_colors):
        return None

    return phrase


def _build_gt_to_instruction_map(
    instruction: str,
    gt_names: list[str],
) -> dict[str, str]:
    """Build a mapping from GT object names to instruction-level phrases.

    E.g. {"husky_hammer": "black hammer", "red_hammer": "red hammer"}
    """
    mapping = {}
    for gt_name in gt_names:
        phrase = _extract_phrase_for_gt_name(instruction, gt_name)
        if phrase:
            mapping[gt_name] = phrase
        else:
            # Fallback: replace underscores with spaces
            mapping[gt_name] = gt_name.replace("_", " ")
    return mapping


def map_subgoal_to_objects(
    instruction: str,
    scene_objects: list[str],
    gt_conditions: list[dict] | None = None,
) -> tuple[list[str], str | None]:
    """Map a subgoal instruction to GT scene object names.

    Returns (target_objects, target_container).

    Strategy:
    1. If GT conditions are available, use them directly.
    2. If the subgoal instruction is more specific than the GT conditions
       (e.g. "pick up the *black* hammer" vs GT listing both hammers),
       intersect GT targets with instruction-matched targets to narrow
       down to the ones the subgoal actually refers to.
    3. Otherwise, fuzzy-match instruction words against scene object names.
    """
    gt_targets: list[str] = []
    gt_container: str | None = None

    if gt_conditions:
        gt_targets, gt_container = _from_gt_conditions(gt_conditions, scene_objects)

    instr_targets, instr_container = _from_instruction(instruction, scene_objects)

    # If GT conditions gave us targets, try to narrow them using
    # the instruction.  This handles the case where the task has
    # multiple objects (e.g. both hammers) but the current subgoal
    # only refers to one (e.g. "the black hammer").
    if gt_targets:
        if instr_targets:
            # Intersect: keep only GT targets that instruction also matched
            narrowed = [t for t in gt_targets if t in instr_targets]
            if narrowed:
                # Instruction successfully narrowed the GT targets
                container = gt_container or instr_container
                return narrowed, container
            # No intersection — the orchestrator's subgoal refers to
            # different objects than robolab's current SSM subtask
            # (e.g. VLM says "pick up the cube" but SSM is tracking
            # banana).  Trust the instruction over GT conditions.
            return instr_targets, instr_container or gt_container
        # No instruction targets — use all GT targets
        return gt_targets, gt_container or instr_container

    # No GT conditions — rely on instruction matching alone
    if instr_targets:
        return instr_targets, instr_container

    return [], instr_container


def _from_gt_conditions(
    conditions: list[dict],
    scene_objects: list[str] | None = None,
) -> tuple[list[str], str | None]:
    """Extract target objects and container from GT subtask conditions.

    Filters out generic CSM group names (e.g. ``"conditions"``) that
    don't correspond to actual scene objects.

    Objects that appear **only** in articulated-action predicates
    (``Close``, ``Open``, ``Turnon``, ``Turnoff``) are NOT grasp
    targets — you don't pick up a microwave to close it.
    """
    # Predicates where the object is NOT a grasp target.
    _ARTICULATED_PREDICATES = {"Close", "Open", "Turnon", "Turnoff"}

    targets = []
    container = None
    scene_set = set(scene_objects) if scene_objects else None
    for cond in conditions:
        obj = cond.get("object", "")
        if not obj:
            continue
        # Skip generic group names that aren't real scene objects
        if scene_set is not None and obj not in scene_set:
            continue
        # Skip articulated-action predicates (Close, Open, Turnon, etc.)
        predicate = cond.get("predicate", "")
        if predicate in _ARTICULATED_PREDICATES:
            continue
        if not cond.get("satisfied", False):
            targets.append(obj)
        # Try to extract container from info string
        info = cond.get("info", "")
        for kw in _CONTAINER_PATTERNS:
            if kw in info.lower():
                # Parse "object_in_container(red_block, grey_bin)"
                m = re.search(r"(?:container|bin|pot|plate|bowl|mug|shelf)\w*", info, re.I)
                if m:
                    container = m.group(0)
    return targets, container


def _has_color_conflict(obj_name: str, instruction: str) -> bool:
    """Check if an object's color attribute conflicts with the instruction.

    E.g. obj_name="red_hammer", instruction="pick up the black hammer"
    → obj has color "red", instruction has color "black" → conflict.
    """
    obj_words = set(obj_name.lower().replace("_", " ").split())
    inst_words = set(instruction.lower().split())

    obj_colors = obj_words & _COLOR_WORDS
    inst_colors = inst_words & _COLOR_WORDS

    if obj_colors and inst_colors:
        # Both have color words — conflict if they don't overlap
        return obj_colors.isdisjoint(inst_colors)

    return False


def _split_pick_and_place(instruction: str) -> tuple[str, str]:
    """Split an instruction into a *pick* part and a *place* part.

    Returns ``(pick_part, place_part)`` where either may be empty.

    Examples
    --------
    >>> _split_pick_and_place("Pick up the yellow block and place it on top of the green block")
    ('pick up the yellow block', 'place it on top of the green block')
    >>> _split_pick_and_place("Put the red block in the bin")
    ('put the red block', 'in the bin')
    """
    inst = instruction.lower().strip()

    # Pattern 1: explicit "pick/grab/get X and/then place/put Y"
    m = re.match(
        r"(.*?\b(?:pick|grab|grasp|get)\b.+?)"
        r"\s+(?:and\s+|then\s+)"
        r"((?:place|put|drop|set|stack)\b.*)",
        inst,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Pattern 2: "put/place/stack X in/on/into Y"
    m = re.match(
        r"(.*?\b(?:put|place|move|drop|stack)\b.+?)"
        r"\s+((?:in|on|into|onto|to|inside)\b.*)",
        inst,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # No split possible
    return inst, ""


def _match_objects_in_text(
    text: str,
    scene_objects: list[str],
) -> tuple[list[str], str | None]:
    """Match scene objects mentioned in a text fragment.

    Returns ``(matched_objects, container_or_None)``.

    Matching tiers (highest priority first):
    1. **Full match** — every word in the scene object name appears in
       the text (ignoring numeric suffixes like ``_1``).
    2. **Partial match** — at least ``len(words) - 1`` non-trivial words
       match (also ignoring numeric suffixes).
    3. **Core-word match** — the object's core noun (longest non-numeric,
       non-color word ≥ 3 chars, e.g. ``"mug"``, ``"hammer"``) appears
       in the text.  This handles short human instructions like
       ``"place the mug"`` matching ``"white_yellow_mug_1"``.
    """
    if not text:
        return [], None

    text_lower = text.lower()
    full_matches: list[str] = []
    partial_matches: list[str] = []
    core_matches: list[str] = []
    full_container: str | None = None
    partial_container: str | None = None
    core_container: str | None = None

    for obj_name in scene_objects:
        obj_words = obj_name.lower().replace("_", " ").split()
        # Filter out numeric suffixes for matching purposes
        meaningful_words = [w for w in obj_words if not w.isdigit()]
        is_container = any(
            kw in obj_name.lower() for kw in _CONTAINER_PATTERNS
        )

        if all(w in text_lower for w in meaningful_words):
            if is_container:
                if full_container is None:
                    full_container = obj_name
            else:
                full_matches.append(obj_name)
        elif len(meaningful_words) > 1:
            matching = sum(
                1 for w in meaningful_words
                if w in text_lower and len(w) > 2
            )
            if matching >= max(1, len(meaningful_words) - 1):
                if is_container:
                    if partial_container is None:
                        partial_container = obj_name
                else:
                    partial_matches.append(obj_name)
            else:
                # Core-word fallback: match if the object's core noun
                # (longest meaningful word ≥ 3 chars) is in the text.
                core = _core_noun(meaningful_words)
                if core and core in text_lower:
                    if is_container:
                        if core_container is None:
                            core_container = obj_name
                    else:
                        core_matches.append(obj_name)

    matched = full_matches or partial_matches or core_matches
    container = full_container or partial_container or core_container
    return matched, container


def _from_instruction(
    instruction: str,
    scene_objects: list[str],
) -> tuple[list[str], str | None]:
    """Parse instruction verb structure to separate pick targets from
    placement destinations, then fuzzy-match against scene objects.

    Objects mentioned in the *place* clause (e.g. "on top of the green
    block") are treated as the destination / container — **not** as
    grasp targets.
    """
    inst_lower = instruction.lower()

    # --- Step 1: split into pick / place parts ---
    pick_part, place_part = _split_pick_and_place(instruction)

    # --- Step 2: match objects in each part ---
    pick_targets, pick_container = _match_objects_in_text(
        pick_part, scene_objects,
    )
    place_targets, place_container = _match_objects_in_text(
        place_part, scene_objects,
    )

    # Objects in the place clause are destinations, not pick targets.
    # If an object appears in both clauses (e.g. shared base word like
    # "block"), only keep it in pick_targets if it's NOT also a place
    # destination.
    destination_objects = set(place_targets)
    targets = [t for t in pick_targets if t not in destination_objects]

    # If the split didn't help (no pick_part / no split possible), fall
    # back to the old full-instruction matching but still try to exclude
    # objects that appear only after placement prepositions.
    if not targets and not pick_container:
        # Full-instruction fallback
        all_matched, fallback_container = _match_objects_in_text(
            inst_lower, scene_objects,
        )
        # Remove objects only found in the place part
        targets = [t for t in all_matched if t not in destination_objects]
        if not targets:
            targets = all_matched  # last resort: keep everything
        pick_container = fallback_container

    # Container priority: explicit place destination > place_container > pick_container
    container: str | None = None
    if place_targets:
        # The object in the place clause is the destination (even if
        # it's not a traditional "container" like a bin).
        container = place_targets[0]
    container = container or place_container or pick_container

    # If no container from objects, check instruction for container references
    if container is None:
        for kw, _ in _CONTAINER_PATTERNS.items():
            if kw in inst_lower:
                for obj_name in scene_objects:
                    if kw in obj_name.lower() and obj_name not in targets:
                        container = obj_name
                        break
                if container:
                    break

    # Filter color conflicts
    if len(targets) > 1:
        filtered = [
            t for t in targets if not _has_color_conflict(t, instruction)
        ]
        if filtered:
            targets = filtered

    return targets, container


# ──────────────────────────────────────────────────────────────────────
# Main detector
# ──────────────────────────────────────────────────────────────────────

class GTFailureDetector:
    """Ground-truth failure detector using sim object states.

    Operates per-subgoal: call ``set_subgoal()`` when the orchestrator
    advances to a new subgoal, then ``update()`` each step.
    """

    # NOTE on units: the constants below are in *chunks* (one per
    # ``update()`` call), not sim steps.  ``update()`` is called once
    # per VLA chunk request from the proxy.  At pi05's chunk rate
    # (~15 Hz / 8 sim steps per chunk), 30 chunks ≈ 2 s.  Cadence
    # drifts ~38% across VLAs with different chunk sizes (libero=5,
    # gr00t=10).  Migration to step-based gating is deferred — see
    # ``docs`` and the step-based monitor cadence migration.

    # Chunks without any target object displacement before declaring NO_PROGRESS.
    NO_PROGRESS_PATIENCE_CHUNKS = 30  # ~2 seconds at pi05's 15 Hz chunk rate

    # Minimum chunks before detecting failures (let policy settle).
    MIN_CHUNKS_BEFORE_FAILURE = 8

    # After a failure is acknowledged (recovery started), suppress all
    # failure detection for this many chunks.
    COOLDOWN_CHUNKS = 15

    # All failure types that can be filtered via enabled_failure_types.
    ALL_FAILURE_TYPES = {
        GTFailureType.WRONG_OBJECT_PICKED,
        GTFailureType.OBJECT_DROPPED,
        GTFailureType.NO_PROGRESS,
        GTFailureType.SUBTASK_REGRESSION,
    }

    def __init__(
        self,
        enabled_failure_types: set[str] | None = None,
    ):
        # If caller provides a set of failure-type *value* strings
        # (e.g. {"wrong_object_picked", "object_dropped"}), only
        # those failure types will be reported.  ``None`` means all
        # failure types are enabled (default / backward-compatible).
        if enabled_failure_types is not None:
            self._enabled: set[GTFailureType] = {
                GTFailureType(v) for v in enabled_failure_types
            }
        else:
            self._enabled: set[GTFailureType] = set(self.ALL_FAILURE_TYPES)

        self._subgoal_instruction: str = ""
        self._target_objects: list[str] = []
        self._target_container: str | None = None
        self._scene_objects: list[str] = []
        # Mapping from GT name → instruction-level phrase
        # e.g. {"husky_hammer": "black hammer", "right_bin": "grey bin"}
        self._gt_to_instruction: dict[str, str] = {}

        # Per-subgoal tracking
        # Chunk counters (incremented once per ``update()`` call).
        # Naming retained as ``_chunks_*`` for clarity; pre-migration
        # these were called ``_steps_*`` but always ticked per chunk.
        self._chunks_on_subgoal: int = 0
        self._chunks_since_progress: int = 0
        self._last_progress_score: float = 0.0

        # Object tracking (minimal — rely on robolab's CSM conditions)
        self._prev_grasped: str | None = None
        self._prev_grabbed: dict[str, bool] = {}  # per-object grab history for drop detection
        self._completed_objects: set[str] = set()

        # Completion persistence: require per-object completion to
        # hold for COMPLETION_CONFIRM_CHUNKS consecutive frames (one
        # per ``update()`` call) before firing SUBGOAL_COMPLETE.
        # Guards against transient physics states (e.g. block balanced
        # momentarily then falling).
        self.COMPLETION_CONFIRM_CHUNKS: int = 5
        self._obj_completion_streak: int = 0

        # Global completion tracking (persists across subgoal advances).
        # Used to detect SUBTASK_REGRESSION: a condition that was satisfied
        # becomes unsatisfied (e.g. block falls off stack after being
        # bumped).  Only reset on full reset() (new episode), NOT on
        # set_subgoal() or acknowledge().
        self._globally_completed: dict[str, bool] = {}
        self._peak_score: float = 0.0

        # Persistence tracking: a condition must be True for
        # REGRESSION_CONFIRM_CHUNKS consecutive chunks before it's
        # recorded in _globally_completed.  This prevents transient
        # flickers (block momentarily balanced) from triggering false
        # regression later.
        self.REGRESSION_CONFIRM_CHUNKS: int = 5
        self._condition_true_streak: dict[str, int] = {}

        # Cooldown: after a failure is handled, suppress all failure
        # detection for COOLDOWN_CHUNKS to let the recovery action take
        # effect before re-checking.
        self._cooldown_remaining: int = 0

    def instruction_name(self, gt_name: str) -> str:
        """Convert a GT object name to its instruction-level phrase.

        E.g. ``"husky_hammer"`` → ``"black hammer"``
        """
        return self._gt_to_instruction.get(
            gt_name, gt_name.replace("_", " "),
        )

    def acknowledge(self) -> None:
        """Call after a failure has been handled (recovery started).

        Starts a cooldown period during which ``update()`` returns
        IN_PROGRESS, giving the recovery action time to take effect.
        Also resets tracking state so the same failure doesn't
        immediately re-fire:
        - ``_chunks_since_progress`` reset → no-progress restarts count
        - ``_prev_grabbed`` cleared → drop detection restarts
        """
        self._cooldown_remaining = self.COOLDOWN_CHUNKS
        self._chunks_since_progress = 0
        self._prev_grabbed = {}
        logger.info(
            f"[GT] Failure acknowledged — cooldown for "
            f"{self.COOLDOWN_CHUNKS} chunks, tracking reset"
        )

    def reset(self) -> None:
        """Full reset (new episode)."""
        self._subgoal_instruction = ""
        self._target_objects = []
        self._target_container = None
        self._scene_objects = []
        self._gt_to_instruction = {}
        self._chunks_on_subgoal = 0
        self._chunks_since_progress = 0
        self._last_progress_score = 0.0
        self._prev_grasped = None
        self._prev_grabbed = {}
        self._completed_objects = set()
        self._obj_completion_streak = 0
        self._globally_completed = {}
        self._peak_score = 0.0
        self._condition_true_streak = {}
        self._cooldown_remaining = 0

    def set_subgoal(
        self,
        instruction: str,
        scene_objects: list[str] | None = None,
        gt_conditions: list[dict] | None = None,
        target_objects: list[str] | None = None,
        target_container: str | None = None,
    ) -> None:
        """Configure detector for a new subgoal.

        Either provide explicit target_objects/target_container, or let
        the mapper infer them from the instruction + scene.
        """
        self._subgoal_instruction = instruction
        self._scene_objects = scene_objects or self._scene_objects

        if target_objects is not None:
            self._target_objects = list(target_objects)
            self._target_container = target_container
        else:
            self._target_objects, self._target_container = map_subgoal_to_objects(
                instruction, self._scene_objects, gt_conditions,
            )

        # Build GT-name → instruction-phrase mapping
        self._gt_to_instruction = _build_gt_to_instruction_map(
            instruction, self._target_objects,
        )
        if self._target_container:
            container_phrase = _extract_phrase_for_gt_name(
                instruction, self._target_container,
            )
            if container_phrase:
                self._gt_to_instruction[self._target_container] = container_phrase

        # Reset per-subgoal state
        self._chunks_on_subgoal = 0
        self._chunks_since_progress = 0
        self._last_progress_score = 0.0
        self._prev_grasped = None
        self._prev_grabbed = {}
        self._cooldown_remaining = 0
        self._completed_objects = set()
        self._obj_completion_streak = 0

        logger.info(
            f"[GT] Subgoal set: targets={self._target_objects}, "
            f"container={self._target_container}, "
            f"name_map={self._gt_to_instruction}, "
            f"instruction={instruction!r}"
        )

    def update(self, gt_state: dict) -> GTFailureResult:
        """Process one step of GT state. Returns current status/failure."""
        self._chunks_on_subgoal += 1

        subtask = gt_state.get("subtask", {})

        # --- Regression check (always, even during cooldown) ---
        # A previously completed object becoming uncompleted is critical
        # and should not be suppressed by per-subgoal cooldown.
        result = self._check_regression(subtask)
        if result is not None:
            return self._filter_disabled(result)

        # Cooldown: after a failure was acknowledged, suppress detection
        # for a few steps so the recovery action can take effect.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return self._make_result(
                GTFailureType.IN_PROGRESS,
                f"Cooldown ({self._cooldown_remaining} chunks remaining)",
            )

        objects = gt_state.get("objects", {})
        robot = gt_state.get("robot", {})

        # Lazy re-init: if set_subgoal was called before gt_state was
        # available (empty scene_objects), retry mapping now.
        if (
            not self._target_objects
            and self._subgoal_instruction
            and objects
        ):
            scene_objects = list(
                gt_state.get("scene_objects", objects.keys())
            )
            gt_conditions = subtask.get("conditions", [])
            self._scene_objects = scene_objects
            self._target_objects, self._target_container = (
                map_subgoal_to_objects(
                    self._subgoal_instruction, scene_objects,
                    gt_conditions,
                )
            )
            if self._target_objects:
                logger.info(
                    f"[GT] Lazy target init: targets="
                    f"{self._target_objects}, "
                    f"container={self._target_container}"
                )

        # Set scene objects if not already set
        if not self._scene_objects:
            self._scene_objects = list(gt_state.get("scene_objects", objects.keys()))

        grasped = robot.get("grasped_object")
        objects_in_contact = robot.get("objects_in_contact") or []
        conditions = subtask.get("conditions", [])

        # Build per-object condition map from CSM:
        # obj_conds[obj_name] = {0: bool, 1: bool, 2: bool, 3: bool}
        # For pick_and_place: 0=grabbed, 1=above_bottom, 2=dropped, 3=in_container
        obj_conds: dict[str, dict[int, bool]] = {}
        for cond in conditions:
            obj = cond.get("object", "")
            idx = cond.get("condition_idx", -1)
            sat = cond.get("satisfied", False)
            obj_conds.setdefault(obj, {})[idx] = sat

        # Track completed objects from robolab's object_completed.
        # For composite subtasks (pick_and_place), keys are object names
        # like "red_block".  For atomic subtasks (stacked), the key is a
        # generic group name like "conditions".  In the latter case, when
        # the generic group is done we mark ALL target objects as done.
        obj_completed = subtask.get("object_completed", {})
        target_set = set(self._target_objects)
        if self._chunks_on_subgoal % 100 == 0 or any(obj_completed.values()):
            logger.info(
                f"[GT-DETECT] chunk={self._chunks_on_subgoal} "
                f"obj_completed={obj_completed} "
                f"target_set={target_set} "
                f"_completed_objects={self._completed_objects}"
            )
        for obj_name, done in obj_completed.items():
            if done and obj_name in target_set:
                self._completed_objects.add(obj_name)

        # --- Check subgoal completion (always, even during warmup) ---
        result = self._check_subgoal_complete(subtask)
        if result is not None:
            self._prev_grasped = grasped
            return result

        # --- Too early to detect failures ---
        if self._chunks_on_subgoal < self.MIN_CHUNKS_BEFORE_FAILURE:
            self._prev_grasped = grasped
            return self._make_result(GTFailureType.IN_PROGRESS, "Warming up")

        # --- Failure detection using robolab's CSM conditions ---
        # All checks use the per-object condition map + grasped_object
        # from robolab, not custom heuristics.

        # 1. Wrong object: gripper holding a non-target
        result = self._check_wrong_object(grasped, objects_in_contact, obj_conds)
        if result is not None:
            self._prev_grasped = grasped
            return self._filter_disabled(result)

        # 2. Object dropped: target was grabbed (cond 0 was True),
        #    now not grabbed, and not yet in container (cond 3 False)
        result = self._check_object_dropped(grasped, obj_conds)
        if result is not None:
            self._prev_grasped = grasped
            return self._filter_disabled(result)

        # 3. No progress: score hasn't improved for many steps
        result = self._check_no_progress(subtask)
        if result is not None:
            self._prev_grasped = grasped
            return self._filter_disabled(result)

        self._prev_grasped = grasped
        return self._make_result(GTFailureType.IN_PROGRESS, "Operating normally")

    # ------------------------------------------------------------------
    # Failure checks
    # ------------------------------------------------------------------

    def _check_regression(
        self, subtask: dict,
    ) -> GTFailureResult | None:
        """Check if any previously completed object has become uncompleted.

        This detects **cross-subgoal regression**: e.g. the robot bumps
        a block stack while working on a later subgoal, undoing earlier
        work.  The check runs every step, even during cooldown, because
        regression is a critical event that changes what work remains.

        Uses ``object_completed`` from robolab's CSM, which already
        tracks regression internally (``object_completed_table`` flips
        ``True → False`` when conditions are no longer met).
        """
        # all_subtask_conditions: maps "subtask_0" → bool, etc.
        # Re-evaluated every step for ALL robolab subtasks (not just current).
        all_conds = subtask.get("all_subtask_conditions", {})

        # Track peak score for logging
        score = float(subtask.get("score", 0.0))
        if score > self._peak_score:
            self._peak_score = score

        # Periodic debug logging
        if self._chunks_on_subgoal % 200 == 1:
            logger.debug(
                f"[GT-REGCHECK] conds={dict(all_conds)} "
                f"confirmed={dict(self._globally_completed)}"
            )

        if not all_conds:
            return None

        # --- Persistence: only promote to "completed" after N
        # consecutive True steps to avoid transient flickers ---
        for key, is_done in all_conds.items():
            if is_done:
                self._condition_true_streak[key] = (
                    self._condition_true_streak.get(key, 0) + 1
                )
            else:
                self._condition_true_streak[key] = 0

            # Promote to globally_completed only after confirmed streak
            if (
                self._condition_true_streak[key]
                >= self.REGRESSION_CONFIRM_CHUNKS
                and not self._globally_completed.get(key, False)
            ):
                logger.info(
                    f"[GT-REGTRACK] '{key}' confirmed satisfied "
                    f"({self.REGRESSION_CONFIRM_CHUNKS} consecutive "
                    f"chunks)"
                )
                self._globally_completed[key] = True

        # Compare current conditions against confirmed snapshot
        newly_regressed: list[str] = []
        for key, is_done in all_conds.items():
            was_done = self._globally_completed.get(key, False)
            if was_done and not is_done:
                newly_regressed.append(key)
                # Mark as regressed in snapshot so it doesn't re-fire
                self._globally_completed[key] = False

        if not newly_regressed:
            return None

        # Also remove regressed objects from per-subgoal completed set
        for obj in newly_regressed:
            self._completed_objects.discard(obj)

        logger.warning(
            f"[GT] SUBTASK REGRESSION: {newly_regressed} were completed "
            f"but are now undone (score {self._peak_score:.2f} → "
            f"{score:.2f})"
        )

        return self._make_result(
            GTFailureType.SUBTASK_REGRESSION,
            f"Previously completed subtasks regressed: "
            f"{newly_regressed} (score dropped "
            f"{self._peak_score:.2f} → {score:.2f})",
            suggested_actions=[
                "replan",
                "resume",
            ],
        )

    def _check_subgoal_complete(self, subtask: dict) -> GTFailureResult | None:
        """Check if this subgoal's target objects are completed.

        Uses per-object completion from robolab's ``object_completed``
        dict (which now covers ALL subtasks, not just the current SSM
        subtask).  When *all* target objects for the current subgoal
        appear in ``_completed_objects``, confirm for
        ``COMPLETION_CONFIRM_CHUNKS`` consecutive chunks to guard against
        transient physics states.
        """
        score = float(subtask.get("score", 0.0))

        # Periodic diagnostic logging
        if self._chunks_on_subgoal % 200 == 0:
            logger.info(
                f"[GT-COMPLETION] chunk={self._chunks_on_subgoal} "
                f"score={score:.3f} "
                f"targets={self._target_objects} "
                f"completed={self._completed_objects} "
                f"streak={self._obj_completion_streak}"
            )

        # ``_completed_objects`` is populated from robolab's
        # ``object_completed`` dict every step (see update()).
        # Check whether ALL target objects for the current subgoal
        # are individually completed.
        unique_targets = set(self._target_objects)
        if unique_targets and unique_targets.issubset(self._completed_objects):
            self._obj_completion_streak += 1
            if self._obj_completion_streak == 1:
                logger.info(
                    f"[GT-COMPLETION] pending: "
                    f"targets={unique_targets} all in "
                    f"completed={self._completed_objects}, "
                    f"score={score:.3f}, confirming..."
                )
            if self._obj_completion_streak >= self.COMPLETION_CONFIRM_CHUNKS:
                logger.info(
                    f"[GT-COMPLETION] ★ CONFIRMED after "
                    f"{self._obj_completion_streak} chunks: "
                    f"targets={unique_targets} completed, "
                    f"score={score:.3f}"
                )
                self._obj_completion_streak = 0
                return self._make_result(
                    GTFailureType.SUBGOAL_COMPLETE,
                    f"per-object completion: "
                    f"{unique_targets} all done (score={score:.2f})",
                )
        else:
            # Not all targets completed — reset streak
            if self._obj_completion_streak > 0:
                logger.info(
                    f"[GT-COMPLETION] reset: "
                    f"targets={unique_targets}, "
                    f"completed={self._completed_objects}, "
                    f"streak was {self._obj_completion_streak}"
                )
            self._obj_completion_streak = 0

        return None

    def _check_wrong_object(
        self,
        grasped: str | None,
        objects_in_contact: list[str] | None = None,
        obj_conds: dict[str, dict[int, bool]] | None = None,
    ) -> GTFailureResult | None:
        """Check if robot is holding a non-target object.

        Uses robolab's ``grasped_object`` from the contact sensor
        (already exported in ``gt_state.robot``).

        To avoid false positives when the gripper simultaneously
        contacts both a target and a non-target (e.g. blocks touching
        each other), we apply two suppression checks:

        1. **Contact list**: robolab exports ``objects_in_contact``
           (all objects the gripper is touching).  If any target
           appears in this list, the gripper IS touching the correct
           object — ``grasped_object`` just happened to pick a
           non-target as the "primary" contact.

        2. **CSM grabbed condition**: if any target has CSM condition 0
           (``grabbed``) = True, the physics engine considers the
           target held.  This catches cases where the contact list is
           unavailable or incomplete.
        """
        if grasped is None:
            return None
        if not self._target_objects:
            return None
        if grasped in self._target_objects:
            return None  # Correct object — primary contact is target

        # Suppression 1: target appears in the full contact list.
        # grasped_object is just objects_in_contact[0]; if the target
        # is anywhere in the list, the gripper IS touching it.
        if objects_in_contact:
            for t in self._target_objects:
                if t in self._completed_objects:
                    continue
                if t in objects_in_contact:
                    logger.info(
                        f"[GT] Primary contact is '{grasped}' but "
                        f"target '{t}' also in contact list "
                        f"{objects_in_contact} — "
                        f"suppressing wrong_object_picked"
                    )
                    return None

        # Suppression 2: CSM says target is grabbed (physics-level).
        if obj_conds:
            for t in self._target_objects:
                if t in self._completed_objects:
                    continue
                conds = obj_conds.get(t, {})
                if conds.get(0, False):  # condition 0 = grabbed
                    logger.info(
                        f"[GT] Primary contact is '{grasped}' but "
                        f"target '{t}' has CSM grabbed=True — "
                        f"suppressing wrong_object_picked"
                    )
                    return None

        first_target = self._first_remaining_target()
        return self._make_result(
            GTFailureType.WRONG_OBJECT_PICKED,
            f"Grasped '{grasped}' instead of target '{first_target}'",
            grasped_object=grasped,
            grasped_is_correct=False,
            suggested_actions=[
                f"grasp_tool({first_target})",
                "retry",
                "resume",
            ],
        )

    def _check_object_dropped(
        self, grasped: str | None,
        obj_conds: dict[str, dict[int, bool]],
    ) -> GTFailureResult | None:
        """Check if a target was grabbed then released before reaching
        the container.

        Uses the CSM condition table directly:
        - condition 0 (grabbed) was True at some point → tracked via
          ``_prev_grabbed``
        - condition 0 now False AND condition 3 (in_container) False
          → object was dropped
        """
        for t in self._target_objects:
            if t in self._completed_objects:
                continue
            conds = obj_conds.get(t, {})
            was_grabbed = self._prev_grabbed.get(t, False)
            is_grabbed = conds.get(0, False)
            in_container = conds.get(3, False)

            # Track grab history
            if is_grabbed:
                self._prev_grabbed[t] = True

            # Drop: previously grabbed, now not grabbed, not in container
            if was_grabbed and not is_grabbed and not in_container:
                # Only fire once per drop — clear history
                self._prev_grabbed[t] = False
                return self._make_result(
                    GTFailureType.OBJECT_DROPPED,
                    f"'{t}' was grabbed then released "
                    f"(not in container)",
                    grasped_object=None,
                    suggested_actions=[
                        f"grasp_tool({t})",
                        "retry",
                    ],
                )

        return None

    def _check_no_progress(
        self, subtask: dict,
    ) -> GTFailureResult | None:
        """Check if the task score hasn't improved for many steps."""
        score = float(subtask.get("score", 0.0))
        if score > self._last_progress_score:
            self._last_progress_score = score
            self._chunks_since_progress = 0
            return None

        self._chunks_since_progress += 1

        if self._chunks_since_progress >= self.NO_PROGRESS_PATIENCE_CHUNKS:
            first_target = self._first_remaining_target()
            remaining = [
                t for t in self._target_objects
                if t not in self._completed_objects
            ]
            return self._make_result(
                GTFailureType.NO_PROGRESS,
                f"No score improvement for "
                f"{self._chunks_since_progress} chunks. "
                f"Remaining: {remaining}",
                suggested_actions=[
                    f"grasp_tool({first_target})",
                    "retry",
                    "skip",
                ],
            )

        return None

    def _first_remaining_target(self) -> str:
        """Return the first uncompleted target object name."""
        remaining = [
            t for t in self._target_objects
            if t not in self._completed_objects
        ]
        if remaining:
            return remaining[0]
        if self._target_objects:
            return self._target_objects[0]

        # Last resort: extract from instruction
        instr = self._subgoal_instruction or ""
        for pattern in [
            r"pick\s+up\s+(?:the\s+)?(.+?)(?:\s+and\b|\s+from\b|$)",
            r"(?:grab|grasp|get)\s+(?:the\s+)?(.+?)(?:\s+and\b|\s+from\b|$)",
            r"(?:place|put|move)\s+(?:the\s+)?(.+?)(?:\s+in\b|\s+on\b|\s+to\b|$)",
        ]:
            m = re.search(pattern, instr, re.I)
            if m:
                return m.group(1).strip()
        return instr[:40] if instr else "unknown"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_disabled(self, result: GTFailureResult) -> GTFailureResult:
        """Downgrade a failure to IN_PROGRESS if its type is disabled.

        Non-failure types (SUBGOAL_COMPLETE, IN_PROGRESS) are never
        filtered — they pass through unconditionally.
        """
        if not result.is_failure:
            return result
        if result.failure_type in self._enabled:
            return result
        # Failure type is disabled — suppress by returning IN_PROGRESS.
        logger.debug(
            f"[GT] Suppressed disabled failure type "
            f"{result.failure_type.value}: {result.reason}"
        )
        return self._make_result(
            GTFailureType.IN_PROGRESS,
            f"Operating normally (suppressed {result.failure_type.value})",
        )

    def _make_result(
        self,
        failure_type: GTFailureType,
        reason: str,
        grasped_object: str | None = None,
        grasped_is_correct: bool = False,
        nearest_miss_distance: float | None = None,
        nearest_miss_object: str | None = None,
        suggested_actions: list[str] | None = None,
    ) -> GTFailureResult:
        """Build a GTFailureResult with current tracking state."""
        remaining = [
            t for t in self._target_objects
            if t not in self._completed_objects
        ]
        completed = list(self._completed_objects)

        # Object displacements
        displaced = {}
        # (filled from the last gt_state — callers should have just called update)

        # Translate GT names to instruction-level phrases in
        # suggested_actions so the HITL UI shows human-readable names
        # and the grasp tool gets a visual description for SAM3.
        translated_actions = []
        for act in (suggested_actions or []):
            m = re.match(r"grasp_tool\((.+)\)", act)
            if m:
                gt_name = m.group(1)
                instr_name = self.instruction_name(gt_name)
                translated_actions.append(f"grasp_tool({instr_name})")
            else:
                translated_actions.append(act)

        return GTFailureResult(
            failure_type=failure_type,
            confidence=1.0,
            reason=reason,
            current_subgoal=self._subgoal_instruction,
            target_objects=list(self._target_objects),
            target_container=self._target_container,
            grasped_object=grasped_object,
            grasped_is_correct=grasped_is_correct,
            objects_completed=completed,
            objects_remaining=remaining,
            objects_displaced=displaced,
            nearest_miss_distance=nearest_miss_distance,
            nearest_miss_object=nearest_miss_object,
            suggested_actions=translated_actions,
        )

    def get_status_summary(self) -> str:
        """Human-readable summary of current detector state."""
        remaining = [
            t for t in self._target_objects
            if t not in self._completed_objects
        ]
        return (
            f"Subgoal: {self._subgoal_instruction!r}\n"
            f"  Targets: {self._target_objects}\n"
            f"  Container: {self._target_container}\n"
            f"  Completed: {list(self._completed_objects)}\n"
            f"  Remaining: {remaining}\n"
            f"  Chunks: {self._chunks_on_subgoal}, "
            f"since progress: {self._chunks_since_progress}"
        )
