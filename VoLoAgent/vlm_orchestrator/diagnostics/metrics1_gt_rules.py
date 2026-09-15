# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aspect-1 stateless rule functions.

These functions consume ``gt_state`` snapshots (from the eval client's
GTStateExporter) and emit classified ``Aspect1Event`` events for the
passive task-failure logger.

Used **only** by:
- :mod:`vlm_orchestrator.diagnostics.metrics1_failure_logger` — passive
  per-step observer attached at the proxy level.  Runs in every eval
  mode whenever ``gt_state`` is in obs.

Independent from the aspect-2 active detector
(:mod:`vlm_orchestrator.failure_handlers.gt_detector`).  Aspect 1 is purely
diagnostic logging; aspect 2 drives recovery routing.  The two have
separate event taxonomies (``Aspect1EventType`` here vs
``GTFailureType`` there) so they can evolve independently.

Rules are stateless functions: each takes a ``RuleState`` (caller-owned
mutable scratchpad) plus inputs derived from ``gt_state``, and returns
``Aspect1Event | None``.  The caller decides what to do with the
result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Aspect-1 event types (independent from the aspect-2 ``GTFailureType``
# in :mod:`vlm_orchestrator.failure_handlers.gt_detector`).
#
# Aspect 1 is the passive task-failure logger.  It runs in every eval
# mode and emits per-object diagnostic events without driving any
# recovery action.  Its event taxonomy is intentionally kept separate
# from the active-detector enum so the two can evolve independently.
# ──────────────────────────────────────────────────────────────────────

class Aspect1EventType(str, Enum):
    """Diagnostic events emitted by the aspect-1 rules."""
    # A target object reached its destination (full pick-and-place
    # ladder satisfied for ``COMPLETION_CONFIRM_CHUNKS`` consecutive
    # chunks).  Multi-object subtasks emit one event per object as
    # each one completes.
    OBJECT_COMPLETE = "object_complete"
    # The gripper holds an object that isn't a current-subtask target.
    WRONG_OBJECT_PICKED = "wrong_object_picked"
    # A target object is in a stable released-and-not-in-target state.
    WRONG_TARGET_PLACE = "wrong_target_place"
    # An object that previously emitted OBJECT_COMPLETE has stopped
    # being at its target (e.g. a placed cube was picked up again
    # and moved to the wrong container).  For single-object subtasks
    # this also covers whole-subtask regression.
    OBJECT_REGRESSION = "object_regression"
    # No progress has been made for ``STUCK_FIRE_CHUNKS`` consecutive
    # chunks (no score increase, no new object completion).  Re-fires
    # every ``STUCK_FIRE_CHUNKS`` chunks while still stuck.  Resets
    # whenever another failure event fires (the policy is doing
    # *something* even if wrong) or progress is observed.
    STUCK = "stuck"
    # A previously-emitted failure event's underlying condition has
    # resolved.  The ``reason`` field describes which one recovered.
    RECOVERY = "recovery"


@dataclass
class Aspect1Event:
    """Aspect-1 diagnostic event.

    Carries enough context for the logger to write a structured JSONL
    entry — but no recovery actions, no confidence scores, no
    detector-specific scaffolding.  Independent from the active
    detector's ``GTFailureResult``.
    """
    event_type: Aspect1EventType
    reason: str = ""
    target_objects: list[str] = field(default_factory=list)
    objects_completed: list[str] = field(default_factory=list)
    objects_remaining: list[str] = field(default_factory=list)
    grasped_object: Optional[str] = None
    suggested_actions: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Tunable constants
# ──────────────────────────────────────────────────────────────────────

# All thresholds are in *chunks* (one update() call per VLA action chunk).
# At pi05's ~15 Hz chunk rate, 30 chunks ≈ 2 s.  Cadence drifts ~38%
# across VLAs with different chunk sizes (libero=5, gr00t=10).

# Per-object full-ladder completion must hold for this many consecutive
# chunks before OBJECT_COMPLETE fires.  Set to 1 — the CSM's own
# completion check already gates on stable physics state (contact
# force cones, ``require_gripper_detached`` etc.), and any transient
# completion that flips back to False quickly will trigger
# OBJECT_REGRESSION (and a subsequent RECOVERY when it stabilizes).
# Higher values delay emission past the natural completion moment
# and are particularly problematic when the policy briefly re-grasps
# a placed object during a later subtask, which resets the streak.
COMPLETION_CONFIRM_CHUNKS = 1

# An object that previously emitted OBJECT_COMPLETE must have
# ``object_completed[obj] == False`` for this many consecutive chunks
# before OBJECT_REGRESSION fires.  Filters single-tick flickers
# (gripper momentarily near a placed cube, contact-force jitter)
# without delaying genuine regression detection by much.
REGRESSION_CONFIRM_CHUNKS = 2

# A target object must have cond_idx 2 (released) without cond_idx 3
# (in target) and without being currently held for this many
# consecutive chunks before WRONG_TARGET_PLACE fires.  Filters out
# transient mid-maneuver drops the policy recovers from.
WRONG_TARGET_PLACE_CONFIRM_CHUNKS = 2

# Fire STUCK every this many chunks of no-progress.  "Progress" =
# either ``subtask.score`` increases or a new object's
# ``object_completed`` flips True.  ~30 chunks ≈ 10 s at pi05's chunk
# rate.  Re-fires every interval while still stuck (so the log
# surfaces a steady "still stuck" beacon rather than going silent),
# but does NOT spam every chunk.  Reset whenever another failure
# event fires (the policy is doing something rather than nothing) or
# when progress is observed.
STUCK_FIRE_CHUNKS = 30



# ──────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RuleState:
    """Mutable scratchpad shared across rule invocations within one scope.

    The caller (active detector, passive logger) owns one of these and
    passes it to each :func:`check_*` call.

    Reset on scope changes:
    - ``reset_for_scope()`` — call when the active detector advances
      to a new VLM subgoal (clears per-scope completion + streak state
      but preserves the outstanding-failure trackers so cross-scope
      regression + recovery still work).
    - construct fresh — call on a new episode.
    """
    # Per-tick counter (incremented by the caller).
    chunks: int = 0
    # Per-scope per-object completion (filled from ``object_completed``).
    completed_objects: set[str] = field(default_factory=set)
    # Per-object confirmation streak for OBJECT_COMPLETE.  Object is
    # admitted to ``completed_objects`` (and an OBJECT_COMPLETE event
    # emitted) once its raw ``object_completed`` flag has been True for
    # ``COMPLETION_CONFIRM_CHUNKS`` consecutive chunks.  Filters out
    # transient physics states.
    completion_streak: dict[str, int] = field(default_factory=dict)
    # Set of objects we've already emitted an OBJECT_COMPLETE event for.
    # Prevents re-emission on the same completion (the rule fires once
    # per object per scope, not on every tick the object stays
    # complete).
    emitted_completions: set[str] = field(default_factory=set)
    # WRONG_TARGET_PLACE confirmation streak per object.
    released_streak: dict[str, int] = field(default_factory=dict)
    # Per-object regression confirmation streak (for OBJECT_REGRESSION).
    # Counts consecutive chunks where a previously-emitted-complete
    # object's ``object_completed`` flag has been False.  Streak resets
    # if the flag flips back to True.
    object_regression_streak: dict[str, int] = field(default_factory=dict)

    # ── Outstanding failure tracking (for RECOVERY emission) ─────────
    # Each entry below records a failure event whose underlying
    # condition hasn't resolved yet.  ``check_recoveries`` scans these
    # every tick and emits a RECOVERY event (clearing the entry) when
    # the recovery condition holds.

    # The wrong-grasped object name from the last WRONG_OBJECT_PICKED
    # fire.  None when no outstanding wrong-grasp.  Also used for
    # dedup: if the same wrong object is still grasped on a subsequent
    # tick, ``check_wrong_object_picked`` doesn't re-fire.
    outstanding_wrong_object: Optional[str] = None
    # Set of object names with outstanding WRONG_TARGET_PLACE.
    # Recovery fires when the object reaches its target (cond_idx 3
    # True).  Also used for dedup.
    outstanding_wrong_target_place: set[str] = field(default_factory=set)
    # Set of object names with outstanding OBJECT_REGRESSION.  Recovery
    # fires when ``object_completed[obj]`` flips True again.
    outstanding_object_regression: set[str] = field(default_factory=set)

    # ── STUCK tracking ───────────────────────────────────────────────
    # Streak of chunks since the last observed progress (score
    # increase or new object completion).  Reset to 0 on progress or
    # whenever another failure event fires this tick.  STUCK fires
    # every ``STUCK_FIRE_CHUNKS`` chunks the streak crosses that
    # threshold.
    stuck_streak: int = 0
    # Highest score seen so far.  Progress = score > this value.
    last_progress_score: float = 0.0
    # Number of objects that had ``object_completed=True`` at the
    # last progress observation.  Progress also = this count
    # increasing.
    last_progress_completions: int = 0
    # True iff a STUCK event has been emitted and no recovery yet.
    # Drives the RECOVERY emission when progress resumes.
    outstanding_stuck: bool = False

    def reset_for_scope(self) -> None:
        """Reset per-scope state.  Preserves the outstanding-failure
        trackers so cross-scope recovery still works."""
        self.chunks = 0
        self.completed_objects = set()
        self.completion_streak = {}
        self.emitted_completions = set()
        self.released_streak = {}
        self.object_regression_streak = {}
        self.stuck_streak = 0


# ──────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────

def build_obj_conds(gt_state: dict) -> dict[str, dict[int, bool]]:
    """Build per-object {condition_idx → satisfied} map from
    ``gt_state.subtask.conditions``.

    For composite ``pick_and_place`` subtasks, each target object has
    cond_idx 0 (grabbed), 1 (above_bottom), 2 (dropped), 3 (in_container).
    Atomic / list-of-partial subtasks export differently — see the
    encoding-rules doc.
    """
    conds_list = gt_state.get("subtask", {}).get("conditions", [])
    out: dict[str, dict[int, bool]] = {}
    for c in conds_list:
        obj = c.get("object", "")
        idx = c.get("condition_idx", -1)
        sat = bool(c.get("satisfied", False))
        out.setdefault(obj, {})[idx] = sat
    return out


def current_subtask_targets(gt_state: dict) -> list[str]:
    """Targets of the **currently active** subtask only.

    Derived from ``gt_state.subtask.conditions`` — the exporter
    populates this list from the current CSM's
    ``object_completed_table`` keys (one entry per cond_idx per
    object).  We collect the unique object names.

    Returns ``[]`` when there is no subtask info or the current
    subtask uses generic group keys (e.g. atomic compound predicates
    where the CSM isn't tracking per-object ladders).  Callers should
    fall back to the task-level union when this returns empty.

    Used by ``check_wrong_object_picked`` and
    ``check_wrong_target_place`` to scope failure detection to the
    objects the policy is *supposed* to be working on right now —
    avoids false-positives where the policy correctly works on a
    later subtask's target while a future-subtask object happens to
    be in scope of the task-level target list.
    """
    conds_list = gt_state.get("subtask", {}).get("conditions", [])
    out: list[str] = []
    seen: set[str] = set()
    for c in conds_list:
        obj = c.get("object")
        if isinstance(obj, str) and obj and obj not in seen:
            seen.add(obj)
            out.append(obj)
    return out


def update_completed_set(
    state: RuleState, gt_state: dict, target_set: set[str],
) -> None:
    """Add any newly-completed target objects to ``state.completed_objects``.

    Reads ``gt_state.subtask.object_completed`` (a dict of
    ``{obj_name: bool}``) and adds the True entries that intersect
    ``target_set``.
    """
    obj_completed = gt_state.get("subtask", {}).get("object_completed", {})
    for obj_name, done in obj_completed.items():
        if done and obj_name in target_set:
            state.completed_objects.add(obj_name)


# ──────────────────────────────────────────────────────────────────────
# Rule functions
# ──────────────────────────────────────────────────────────────────────

def check_object_complete(
    state: RuleState, gt_state: dict, target_objects: list[str],
) -> Optional[Aspect1Event]:
    """OBJECT_COMPLETE: a target object's CSM ladder is done.

    Per-object emission: each target fires its own event once
    ``gt_state.subtask.object_completed[obj]`` has been True for
    ``COMPLETION_CONFIRM_CHUNKS`` consecutive chunks.  With the
    threshold set to 1, the event fires on the first True
    observation — the CSM's own completion check already gates on
    stable physics state, so an additional logger-side stability
    streak would only delay reporting.

    Multi-phase tasks like ``LH_A01_UnstackCubesBottomToBin`` produce
    one event per cube as each one completes its phase.  Multi-object
    subtasks (e.g. ``pick_and_place_grouped`` / atomic compound
    predicates) produce one event per object as each one is admitted.

    ``state.emitted_completions`` records emitted objects to prevent
    re-firing on the same completion.  If the object regresses (via
    ``check_object_regression``), the entry is removed from both
    ``completed_objects`` and ``emitted_completions`` so a future
    re-completion can re-emit OBJECT_COMPLETE.

    Multiple objects may complete on the same tick.  This function
    fires for at most one per call — the next tick picks up the next.
    """
    target_set = set(target_objects)
    obj_completed = gt_state.get("subtask", {}).get("object_completed", {})

    # Per-target streak update.
    for t in target_set:
        if obj_completed.get(t, False):
            state.completion_streak[t] = state.completion_streak.get(t, 0) + 1
        else:
            state.completion_streak[t] = 0

    # Find the first target whose streak just crossed the threshold and
    # hasn't been emitted yet.  Sorted iteration for determinism.
    for t in sorted(target_set):
        if t in state.emitted_completions:
            continue
        if state.completion_streak.get(t, 0) >= COMPLETION_CONFIRM_CHUNKS:
            state.emitted_completions.add(t)
            state.completed_objects.add(t)
            score = float(gt_state.get("subtask", {}).get("score", 0.0))
            return Aspect1Event(
                event_type=Aspect1EventType.OBJECT_COMPLETE,
                reason=(
                    f"object completed: {t} (score={score:.2f})"
                ),
                target_objects=[t],
                objects_completed=[t],
            )

    return None


def check_wrong_object_picked(
    state: RuleState,
    gt_state: dict,
    target_objects: list[str],
    completed_objects: set[str],
    obj_conds: dict[str, dict[int, bool]],
) -> Optional[Aspect1Event]:
    """WRONG_OBJECT_PICKED: gripper holds an object that isn't a
    (remaining) target.

    Suppression: if any uncompleted target appears in
    ``robot.objects_in_contact`` — the gripper IS touching a target,
    ``grasped_object`` just happened to pick a non-target as the
    "primary" contact.

    The CSM ``cond_idx 0`` (grabbed) bit is intentionally NOT used
    here because it's *latched* — once the agent has grasped the
    target at any point in the episode, cond_idx 0 stays True even
    after release.  Using it for current-grasp suppression would
    cause the rule to silently ignore subsequent wrong-object grasps
    after the policy has touched the target once.  ``robot.objects_in_contact``
    is current-tick state and gives the right semantic.

    Dedup: when the same wrong object is still grasped on a subsequent
    tick, no re-fire — the failure stays "outstanding" in
    ``state.outstanding_wrong_object`` until ``check_recoveries``
    detects recovery and clears it.
    """
    robot = gt_state.get("robot", {})
    grasped = robot.get("grasped_object")
    objects_in_contact = robot.get("objects_in_contact") or []

    if grasped is None:
        return None
    if not target_objects:
        return None
    if grasped in target_objects:
        return None

    # Suppression: any uncompleted target is currently in contact.
    for t in target_objects:
        if t in completed_objects:
            continue
        if t in objects_in_contact:
            return None

    # Dedup: same wrong object as last fire — don't re-fire.
    if state.outstanding_wrong_object == grasped:
        return None

    state.outstanding_wrong_object = grasped
    first_target = next(
        (t for t in target_objects if t not in completed_objects),
        target_objects[0],
    )
    return Aspect1Event(
        event_type=Aspect1EventType.WRONG_OBJECT_PICKED,
        reason=f"Grasped '{grasped}' instead of target '{first_target}'",
        grasped_object=grasped,
        target_objects=list(target_objects),
        suggested_actions=[
            f"grasp_tool({first_target})", "retry", "resume",
        ],
    )


def check_wrong_target_place(
    state: RuleState,
    gt_state: dict,
    target_objects: list[str],
    completed_objects: set[str],
    obj_conds: dict[str, dict[int, bool]],
) -> Optional[Aspect1Event]:
    """WRONG_TARGET_PLACE: a target object is in a stable
    released-and-not-in-target state for ``WRONG_TARGET_PLACE_CONFIRM_CHUNKS``
    consecutive chunks.

    This replaces the old OBJECT_DROPPED check.  Where OBJECT_DROPPED
    fired on the cond_idx 0 transition (every release, including
    deliberate buffering moves), this fires only on a stable misplaced
    state — the policy thinks it's done with the object, but the
    object isn't in its target container.

    Required CSM signals (from ``pick_and_place(_grouped/_on_surface)``
    composites): per-object cond_idx 2 (released / dropped), 3
    (in_container / on_top).  Tasks encoded with inline list-of-partials
    don't expose cond_idx 2 per-object — this rule is silent on those.

    The "currently held" check uses ``gt_state.robot.grasped_object``
    rather than CSM cond_idx 0 because the CSM table is *latched* —
    once cond_idx 0 advances it stays True even after the agent
    releases — which would prevent the rule from ever firing for an
    object the agent grasped, then released into the wrong target.
    """
    grasped_now = gt_state.get("robot", {}).get("grasped_object")
    fired_for: list[str] = []
    for t in target_objects:
        if t in completed_objects:
            state.released_streak[t] = 0
            continue
        conds = obj_conds.get(t, {})
        is_released = conds.get(2, False)
        in_container = conds.get(3, False)
        is_held_now = grasped_now == t
        if is_released and not is_held_now and not in_container:
            state.released_streak[t] = state.released_streak.get(t, 0) + 1
            # Dedup: if this object is already in outstanding, don't
            # re-fire — failure stays open until recovery resolves it.
            if (
                state.released_streak[t] >= WRONG_TARGET_PLACE_CONFIRM_CHUNKS
                and t not in state.outstanding_wrong_target_place
            ):
                state.released_streak[t] = 0
                state.outstanding_wrong_target_place.add(t)
                fired_for.append(t)
        else:
            state.released_streak[t] = 0

    if not fired_for:
        return None

    obj = fired_for[0]
    return Aspect1Event(
        event_type=Aspect1EventType.WRONG_TARGET_PLACE,
        reason=(
            f"'{obj}' released but not in target "
            f"(stable for {WRONG_TARGET_PLACE_CONFIRM_CHUNKS} chunks)"
        ),
        target_objects=list(target_objects),
        objects_remaining=fired_for,
        suggested_actions=[f"grasp_tool({obj})", "retry"],
    )


def check_object_regression(
    state: RuleState,
    gt_state: dict,
    current_targets: Optional[list[str]] = None,
) -> Optional[Aspect1Event]:
    """OBJECT_REGRESSION: a previously-completed object is no longer
    at its target.

    Fires when an object that previously emitted OBJECT_COMPLETE
    (i.e. is in ``state.emitted_completions``) has had
    ``object_completed[obj] == False`` for ``REGRESSION_CONFIRM_CHUNKS``
    consecutive ticks **and** is not currently a target of the active
    subtask.

    The current-subtask guard handles the legitimate cross-subtask
    re-handling case: if an earlier subtask placed object X at place P1
    and a later subtask's target is X→P2, the policy must lift X off P1
    to satisfy the new subtask.  That lift makes the prior subtask's
    predicate False, but the move is intentional, not a regression.
    While X is in the active subtask's target list, suppress regression
    detection for it.

    For single-object subtasks (most LH tasks), this also covers
    whole-subtask regression — when the placed object moves, the
    subtask's terminal predicate also flips False, but rather than
    detect that at the subtask level we report it per-object via
    this rule.

    Mutates ``state.object_regression_streak``,
    ``state.completed_objects`` (the regressed object is removed so
    ``check_object_complete`` will fire again if it recovers), and
    ``state.emitted_completions`` (so a recovery-then-re-completion
    can re-emit OBJECT_COMPLETE).
    """
    obj_completed = gt_state.get("subtask", {}).get("object_completed", {})
    cur_set = set(current_targets or ())
    fired_for: list[str] = []
    for obj in sorted(state.emitted_completions):
        if obj in state.outstanding_object_regression:
            # Already firing — wait for recovery before re-arming.
            continue
        if obj in cur_set:
            # The active subtask is supposed to be re-handling this
            # object (it was placed by a prior subtask and now needs
            # to move to a new destination).  Whatever the predicate
            # says about the prior subtask's target, this isn't a
            # regression — it's progress on a later phase.  Reset the
            # streak so we don't accumulate spurious ticks.
            state.object_regression_streak[obj] = 0
            continue
        if obj_completed.get(obj, False):
            state.object_regression_streak[obj] = 0
            continue
        state.object_regression_streak[obj] = (
            state.object_regression_streak.get(obj, 0) + 1
        )
        if state.object_regression_streak[obj] >= REGRESSION_CONFIRM_CHUNKS:
            state.object_regression_streak[obj] = 0
            state.outstanding_object_regression.add(obj)
            fired_for.append(obj)

    if not fired_for:
        return None

    obj = fired_for[0]
    # Allow this object to re-emit OBJECT_COMPLETE if it recovers.
    state.emitted_completions.discard(obj)
    state.completed_objects.discard(obj)
    return Aspect1Event(
        event_type=Aspect1EventType.OBJECT_REGRESSION,
        reason=(
            f"'{obj}' was previously complete but is no longer at "
            f"its target (held below for "
            f"{REGRESSION_CONFIRM_CHUNKS} chunks)"
        ),
        target_objects=[obj],
        objects_remaining=[obj],
        suggested_actions=[f"return {obj} to its target"],
    )


def check_stuck(
    state: RuleState,
    gt_state: dict,
    other_event_fired: bool = False,
    target_objects: Optional[list[str]] = None,
) -> Optional[Aspect1Event]:
    """STUCK: no progress for ``STUCK_FIRE_CHUNKS`` consecutive chunks.

    Progress = either ``subtask.score`` increases past its previous
    high-water mark, or a new object's ``object_completed`` flips True,
    or the gripper is currently holding a current-subtask target (the
    pick-half of pick-and-place doesn't always show up in score before
    placement, but the policy is clearly making meaningful progress).

    Behavior:
    - Re-fires every ``STUCK_FIRE_CHUNKS`` chunks while still stuck
      (sparse heartbeat — does NOT spam every chunk).
    - The streak resets when *any* other failure event fires this tick
      (caller passes ``other_event_fired=True``) — the policy is doing
      something even if wrong, so it's not "stuck".
    - The streak also resets on observed progress, including when the
      gripper is holding a current-subtask target.
    - When progress resumes after a STUCK fire, a single RECOVERY
      event fires.

    Caller is expected to invoke this rule **last** in its tick, after
    all other failure rules, with ``other_event_fired`` reflecting
    whether any other rule emitted in this tick.  ``target_objects``
    should be the current-subtask target list (same one passed to
    ``check_wrong_object_picked``); when provided, holding any of
    those objects counts as progress.
    """
    subtask = gt_state.get("subtask", {})
    score = float(subtask.get("score", 0.0))
    obj_completed = subtask.get("object_completed", {})
    completions = sum(1 for v in obj_completed.values() if v)
    grasped = gt_state.get("robot", {}).get("grasped_object")
    holding_target = bool(
        target_objects and grasped and grasped in target_objects
    )

    score_or_completion_progress = (
        score > state.last_progress_score
        or completions > state.last_progress_completions
    )
    progress = score_or_completion_progress or holding_target

    if progress:
        state.last_progress_score = max(state.last_progress_score, score)
        state.last_progress_completions = max(
            state.last_progress_completions, completions,
        )
        state.stuck_streak = 0
        if state.outstanding_stuck:
            state.outstanding_stuck = False
            if score_or_completion_progress:
                reason = (
                    f"STUCK recovered: progress resumed "
                    f"(score={score:.2f}, completions={completions})"
                )
            else:
                reason = (
                    f"STUCK recovered: gripper holds target {grasped!r} "
                    f"(score={score:.2f}, completions={completions})"
                )
            return Aspect1Event(
                event_type=Aspect1EventType.RECOVERY,
                reason=reason,
            )
        return None

    if other_event_fired:
        # Another failure event already described this tick — the
        # policy is doing something, just not productive yet.  Reset
        # the stuck timer; we'll start counting fresh from here.
        state.stuck_streak = 0
        return None

    state.stuck_streak += 1
    if state.stuck_streak >= STUCK_FIRE_CHUNKS:
        state.stuck_streak = 0
        state.outstanding_stuck = True
        return Aspect1Event(
            event_type=Aspect1EventType.STUCK,
            reason=(
                f"No progress for {STUCK_FIRE_CHUNKS} chunks "
                f"(score={score:.2f})"
            ),
        )
    return None


# NOTE: ``check_no_progress`` and ``check_subtask_regression`` were
# removed from this stateless rules module.  Score stagnation /
# score-level drops alone are low-information signals; aspect-1 keys
# regression detection off the per-object signal
# (``check_object_regression``) and uses ``check_stuck`` for
# absence-of-progress signaling.  Single-object subtasks naturally
# get whole-subtask regression coverage that way; multi-object
# subtasks get per-object granularity instead of an aggregated
# subtask flip.  The active GT recovery handler in
# ``gt_detector.py`` retains its own ``_check_subtask_regression``
# and ``_check_no_progress`` for Aspect-2 grasp-recovery routing —
# separate code path with a separate refactor schedule.


def check_recoveries(
    state: RuleState,
    gt_state: dict,
    obj_conds: dict[str, dict[int, bool]],
    target_objects: Optional[list[str]] = None,
) -> Optional[Aspect1Event]:
    """RECOVERY: a previously-emitted failure event's underlying
    condition has resolved.

    Scans the three ``RuleState.outstanding_*`` trackers and emits one
    RECOVERY event per call (sorted, deterministic).  Recovery
    conditions per failure type:

    - WRONG_OBJECT_PICKED → gripper now holds a current target object.
      Just dropping the wrong object (or picking another wrong one)
      is NOT a recovery — the agent has to actually grab a correct
      target for the failure to be considered resolved.  Requires
      ``target_objects`` to be passed in; if omitted, falls back to
      the looser "no longer holds the wrong object" semantic.
    - WRONG_TARGET_PLACE → the object reaches its target container
      (cond_idx 3 True).  Per-object — multiple outstanding
      WRONG_TARGET_PLACE entries each get their own recovery event.
    - OBJECT_REGRESSION → ``object_completed[obj]`` is True again.

    Multiple recoveries may resolve on the same tick.  This function
    emits at most one per call (logger picks up the rest on subsequent
    ticks).  Order: wrong-object → wrong-target-place → object
    regression.
    """
    # 1. Wrong-object recovery
    if state.outstanding_wrong_object is not None:
        grasped = gt_state.get("robot", {}).get("grasped_object")
        recovered_now = False
        if target_objects is not None:
            # Strict: recovery requires agent to be holding a current
            # target.  Releasing the wrong object alone — or grabbing
            # another wrong one — does NOT count.
            recovered_now = grasped is not None and grasped in target_objects
        else:
            # Loose fallback: any change of grasp counts.  Used when
            # the caller can't supply target_objects (e.g. legacy
            # callers).
            recovered_now = grasped != state.outstanding_wrong_object
        if recovered_now:
            recovered = state.outstanding_wrong_object
            state.outstanding_wrong_object = None
            return Aspect1Event(
                event_type=Aspect1EventType.RECOVERY,
                reason=(
                    f"WRONG_OBJECT_PICKED recovered: gripper now holds "
                    f"target '{grasped}' (was '{recovered}')"
                ),
                target_objects=[recovered],
                grasped_object=grasped,
            )

    # 2. Wrong-target-place recovery (one per call)
    for obj in sorted(state.outstanding_wrong_target_place):
        conds = obj_conds.get(obj, {})
        if conds.get(3, False):  # in_container
            state.outstanding_wrong_target_place.discard(obj)
            return Aspect1Event(
                event_type=Aspect1EventType.RECOVERY,
                reason=(
                    f"WRONG_TARGET_PLACE recovered: '{obj}' now in "
                    f"target container"
                ),
                target_objects=[obj],
                objects_completed=[obj],
            )

    # 3. Object-regression recovery (one per call)
    obj_completed = gt_state.get("subtask", {}).get("object_completed", {})
    for obj in sorted(state.outstanding_object_regression):
        if obj_completed.get(obj, False):
            state.outstanding_object_regression.discard(obj)
            return Aspect1Event(
                event_type=Aspect1EventType.RECOVERY,
                reason=(
                    f"OBJECT_REGRESSION recovered: '{obj}' is back "
                    f"at its target"
                ),
                target_objects=[obj],
                objects_completed=[obj],
            )

    return None
