# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive task-level failure logger (Aspect 1).

Watches ``gt_state`` from each step's obs and emits structured events
to ``<episode_log_dir>/task_failures.jsonl``.  Runs in every eval mode
(passthrough / signal / vlm / gt — anything) whenever the eval client
was launched with ``--enable-gt-state``.

Does not influence orchestration: pure observer.  Failures here are
informational only — recovery routing in the active GT mode lives in
:mod:`vlm_orchestrator.failure_handlers.gt` /
``strategies/subgoal_base.py:_tick_gt_detection``.

Scope = whole task, not per-VLM-subgoal: the logger's ``target_objects``
is the union of all CSM subtasks' targets (read from
``gt_state.subtask.object_completed`` keys, which the exporter
populates for every CSM-target object across the task).  This avoids
the granularity mismatch the active per-VLM-subgoal detector hits.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, TextIO

from vlm_orchestrator.diagnostics import metrics1_gt_rules as gt_rules
from vlm_orchestrator.diagnostics.metrics1_gt_rules import Aspect1Event, Aspect1EventType

logger = logging.getLogger(__name__)


# Aspect-1 event types the logger emits.  Per-target-object
# OBJECT_COMPLETE fires once per object as its CSM ladder satisfies
# (not aggregated to whole-task).  OBJECT_REGRESSION covers regression
# at the object level — single-object subtasks naturally get whole-
# subtask coverage; multi-object subtasks get per-object granularity.
# Aspect-2's NO_PROGRESS / SUBTASK_REGRESSION / SUBGOAL_COMPLETE are
# intentionally not used here — the active GT detector keeps those
# for recovery routing on a separate code path.  RECOVERY pairs with
# each failure event (one fire / one recover).
_LOG_TYPES = (
    Aspect1EventType.WRONG_OBJECT_PICKED,
    Aspect1EventType.WRONG_TARGET_PLACE,
    Aspect1EventType.OBJECT_REGRESSION,
    Aspect1EventType.OBJECT_COMPLETE,
    Aspect1EventType.STUCK,
    Aspect1EventType.RECOVERY,
)


class TaskFailureLogger:
    """Per-session observer.  One instance per orchestrator session;
    each episode opens its own ``task_failures.jsonl`` file inside the
    episode's log dir.

    Lifecycle:
      1. ``on_episode_start(episode_log_dir, episode_id, instruction)``
         — called when the proxy detects a new episode.  Resets state
         and opens a fresh ``task_failures.jsonl``.
      2. ``observe(obs, step)`` — called every step.  Reads
         ``obs["gt_state"]``, runs the rule functions, writes any
         emitted events.
      3. ``on_episode_end()`` — called when the episode ends or the
         session terminates.  Closes the file.

    All methods are no-ops if ``gt_state`` is missing from obs (e.g.
    eval client launched without ``--enable-gt-state``).
    """

    def __init__(self, session_log_dir: Optional[str] = None) -> None:
        # Optional session-level log dir — only used as a default
        # when ``on_episode_start`` is called without an explicit
        # episode_log_dir.  Most callers pass episode_log_dir per-episode.
        self._session_log_dir = session_log_dir
        # Per-episode state.  None when no episode is active.
        self._fp: Optional[TextIO] = None
        self._episode_log_dir: Optional[Path] = None
        self._episode_id: Optional[int] = None
        self._instruction: str = ""
        self._state: gt_rules.RuleState = gt_rules.RuleState()
        self._target_objects: list[str] = []
        self._step: int = 0
        # Track whether we've ever seen gt_state for this episode.
        # If we go through a whole episode without seeing it,
        # on_episode_end shouldn't open empty files.
        self._observed_any: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(
        self,
        episode_log_dir: Optional[str | Path] = None,
        episode_id: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> None:
        """Called by the proxy when a new episode begins."""
        # Close any leftover handle from a previous episode.
        self._close_file()

        self._state = gt_rules.RuleState()
        self._target_objects = []
        self._step = 0
        self._observed_any = False
        self._episode_id = episode_id
        self._instruction = instruction or ""
        self._episode_log_dir = (
            Path(episode_log_dir) if episode_log_dir
            else (Path(self._session_log_dir) if self._session_log_dir else None)
        )

    def observe(self, obs: dict, step: Optional[int] = None) -> None:
        """Called every step.  Runs the rules and writes any events."""
        gt_state = obs.get("gt_state")
        if gt_state is None:
            return
        if step is not None:
            self._step = int(step)
        else:
            self._step += 1

        # Lazy file open on first observation — defers creating the
        # file until we know there's gt_state to log.
        if self._fp is None:
            if self._episode_log_dir is None:
                # No log dir configured; can't write anything.  Skip.
                return
            try:
                self._episode_log_dir.mkdir(parents=True, exist_ok=True)
                path = self._episode_log_dir / "task_failures.jsonl"
                self._fp = path.open("w")
                self._write_header()
            except OSError as e:
                logger.warning(f"[task_failure_logger] failed to open log: {e}")
                return

        # Grow ``target_objects`` cumulatively across the episode.  The
        # exporter (with the future-subtask skip) only exposes
        # ``object_completed`` keys for the current + past subtasks at
        # any given step, so the union grows as the SSM advances.  We
        # take the union over time so SUBGOAL_COMPLETE / OBJECT_REGRESSION
        # can fire for objects across all subtasks the policy has
        # touched, while ``check_wrong_object_picked`` /
        # ``check_wrong_target_place`` get a tighter current-subtask
        # scope via ``current_subtask_targets`` below.
        obj_completed = gt_state.get("subtask", {}).get("object_completed", {})
        new_targets = [k for k in obj_completed.keys() if k not in self._target_objects]
        if new_targets:
            self._target_objects.extend(new_targets)
            self._write_event({
                "type": "task_targets_resolved",
                "step": self._step,
                "targets": list(self._target_objects),
                "newly_added": new_targets,
            })

        self._observed_any = True
        try:
            self._run_rules(gt_state)
        except Exception:
            # Never let logger errors break the proxy.
            logger.exception("[task_failure_logger] rule evaluation failed")

    def on_episode_end(self) -> None:
        """Called when the episode ends or the session terminates."""
        if self._fp is not None and self._observed_any:
            self._write_event({
                "type": "episode_end",
                "step": self._step,
                "instruction": self._instruction,
                "episode_id": self._episode_id,
            })
        self._close_file()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_rules(self, gt_state: dict) -> None:
        # Task-level union — used for OBJECT_COMPLETE and
        # OBJECT_REGRESSION which want cross-subtask scope.
        task_targets = list(self._target_objects)
        # Current-active subtask targets — used for WRONG_OBJECT_PICKED
        # and WRONG_TARGET_PLACE so the policy isn't penalized for a
        # future-subtask object that happens to be off-target.  Falls
        # back to task-level union if the current subtask has no
        # per-object CSM groups (e.g. atomic compound predicates).
        cur_targets = gt_rules.current_subtask_targets(gt_state) or task_targets
        completed = self._state.completed_objects
        obj_conds = gt_rules.build_obj_conds(gt_state)
        # Track whether any non-STUCK event fires this tick.  STUCK is
        # the *fallback* event — it should reset its streak whenever
        # another rule has something to say.
        other_event_fired = False

        # Recoveries first — surfaces resolution of any outstanding
        # failures from prior ticks before potentially re-firing them
        # this tick.  ``cur_targets`` is passed so WRONG_OBJECT_PICKED
        # recovery requires the agent to actually grab a target, not
        # just drop the wrong one.
        res = gt_rules.check_recoveries(
            self._state, gt_state, obj_conds, target_objects=cur_targets,
        )
        if res is not None:
            self._emit_event(res, gt_state)
            other_event_fired = True

        # Object-complete (task-level — fire as objects across all
        # subtasks reach their final position).
        res = gt_rules.check_object_complete(self._state, gt_state, task_targets)
        if res is not None:
            self._emit_event(res, gt_state)
            other_event_fired = True

        # Object-level regression — a previously-emitted-complete
        # object stopped being at its target.  Single-object subtasks
        # get whole-subtask regression coverage via this rule too;
        # multi-object subtasks get per-object granularity.
        # ``cur_targets`` suppresses spurious regression when a later
        # subtask's job is to re-handle an object placed by an earlier
        # subtask (the lift is intentional, not regression).
        res = gt_rules.check_object_regression(
            self._state, gt_state, current_targets=cur_targets,
        )
        if res is not None:
            self._emit_event(res, gt_state)
            other_event_fired = True

        # Wrong-object: scope to the current subtask only — picking a
        # future-subtask target while a current-subtask target is still
        # pending is itself a wrong-order failure.
        res = gt_rules.check_wrong_object_picked(
            self._state, gt_state, cur_targets, completed, obj_conds,
        )
        if res is not None:
            self._emit_event(res, gt_state)
            other_event_fired = True

        # Wrong-target-place: also current-subtask scope.
        res = gt_rules.check_wrong_target_place(
            self._state, gt_state, cur_targets, completed, obj_conds,
        )
        if res is not None:
            self._emit_event(res, gt_state)
            other_event_fired = True

        # Stuck check runs last — sparse fallback heartbeat for
        # episodes where nothing else describes what's happening.
        # The streak resets if any other event fired this tick, OR if
        # the gripper is currently holding a current-subtask target
        # (the pick half of pick-and-place is real progress even when
        # the score formula hasn't updated yet).
        res = gt_rules.check_stuck(
            self._state, gt_state,
            other_event_fired=other_event_fired,
            target_objects=cur_targets,
        )
        if res is not None:
            self._emit_event(res, gt_state)

    def _emit_event(self, res: Aspect1Event, gt_state: dict) -> None:
        if res.event_type not in _LOG_TYPES:
            return
        score = float(gt_state.get("subtask", {}).get("score", 0.0))
        grasped = gt_state.get("robot", {}).get("grasped_object")
        event = {
            "type": res.event_type.value,
            "step": self._step,
            "score": score,
            "grasped_object": grasped,
            "reason": res.reason,
            "target_objects": list(res.target_objects),
            "objects_completed": list(res.objects_completed),
            "objects_remaining": list(res.objects_remaining),
            "suggested_actions": list(res.suggested_actions),
        }
        self._write_event(event)

    def _write_event(self, event: dict) -> None:
        if self._fp is None:
            return
        event.setdefault("ts", time.time())
        try:
            self._fp.write(json.dumps(event) + "\n")
            self._fp.flush()
        except OSError as e:
            logger.warning(f"[task_failure_logger] write failed: {e}")
            self._close_file()

    def _write_header(self) -> None:
        """One-line header naming the episode + instruction."""
        if self._fp is None:
            return
        self._write_event({
            "type": "episode_start",
            "step": 0,
            "episode_id": self._episode_id,
            "instruction": self._instruction,
            "wrong_target_place_confirm_chunks":
                gt_rules.WRONG_TARGET_PLACE_CONFIRM_CHUNKS,
            "completion_confirm_chunks":
                gt_rules.COMPLETION_CONFIRM_CHUNKS,
            "regression_confirm_chunks":
                gt_rules.REGRESSION_CONFIRM_CHUNKS,
        })

    def _close_file(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except OSError:
                pass
            self._fp = None
