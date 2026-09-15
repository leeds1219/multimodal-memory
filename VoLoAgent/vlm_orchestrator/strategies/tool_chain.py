# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-chain strategy: VLA-bypassed, VLM-driven tool calls only.

Per-cycle flow:
  1. Tick any active tool (grasp / place) lifecycle.
  2. If a tool just transitioned DONE/FAILED, snapshot its result so
     the next VLM call can reason about it.
  3. If episode-level cap not yet reached, call ``vlm_step`` with the
     unified prompt that returns ``{subgoal_action, tool, args, reason}``.
  4. Apply ``subgoal_action`` (advance / continue / replan / abort).
  5. Activate the chosen tool (grasp / place / noop / done).
  6. ``state.tool_chain_active`` stays True for the whole episode so
     the proxy serves hold-position chunks whenever no tool is active.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .base import OrchestrationStrategy, SessionState, StrategyContext
from .subgoal import SubgoalConfig, SubgoalStrategy

# === merged from tool_chain_prompts.py ============================

import numpy as np

from vlm_orchestrator.vlm import encode_image_b64, parse_json

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Decision types (parsed VLM output)
# ----------------------------------------------------------------------

SubgoalAction = Literal["advance", "continue", "replan", "abort"]
ToolName = Literal["grasp", "place", "noop"]


@dataclass
class ToolChainDecision:
    """One cycle's decision: what to do at the subgoal level + which tool to run."""
    subgoal_action: SubgoalAction
    tool: ToolName
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    raw_response: str = ""              # for logs / debug

    def is_no_action(self) -> bool:
        return self.tool == "noop"

    def __post_init__(self):
        # Validation runs on construction so the parser surfaces malformed
        # output as ValueError, never as a silent fallback.
        if self.subgoal_action not in ("advance", "continue", "replan", "abort"):
            raise ValueError(
                f"invalid subgoal_action {self.subgoal_action!r}; "
                f"expected one of advance/continue/replan/abort"
            )
        if self.tool not in ("grasp", "place", "noop"):
            raise ValueError(
                f"invalid tool {self.tool!r}; expected one of grasp/place/noop"
            )


# ----------------------------------------------------------------------
# System prompt
# ----------------------------------------------------------------------

TOOL_CHAIN_SYSTEM_PROMPT = """\
You are driving a robot arm.  At every step you (a) judge how the
last tool call went and where the plan stands, then (b) pick the
next tool to run.  Output BOTH decisions in a single JSON object.

Inputs you receive:
  - the overall task instruction
  - the planned subgoal list with the current subgoal index marked
  - a BEFORE image (start of the episode)
  - a NOW image (current scene)
  - the last tool that ran, its arguments, and its outcome
    (DONE / FAILED + failure_reason).  These are null on the first
    cycle (no tool has run yet).

SUBGOAL ACTION — judges what has ALREADY happened to the current
subgoal as of the NOW image.  Do NOT use it predictively to describe
what the tool you're about to call WILL do.  Pick exactly one:

  "continue" -- the current subgoal is not yet finished in the NOW
                image.  Use this whenever you still need to run a
                tool (grasp or place) to advance it.  THIS IS THE
                DEFAULT.  In particular:
                  * just grasped an object → still need to place →
                    "continue" + tool=place
                  * grasp / place failed → "continue" + retry
                  * scene partially progressed → "continue"

  "advance"  -- the current subgoal is ALREADY satisfied in the NOW
                image (e.g. the target object is already inside the
                container, the gripper is empty, no more action
                needed for this subgoal).  Pair with tool="noop" —
                "advance" never runs a tool itself, it only moves
                the subgoal pointer.

                ⚠ If you would describe what you're "about to do"
                ("now placing it…", "next I will grasp…") that is
                NOT an "advance" — that is "continue" with the
                corresponding tool call.  Tools execute the action;
                "advance" only acknowledges that the action is
                already complete.

  "replan"   -- something went wrong or the plan no longer fits the
                NOW scene; re-decompose the task.  Pair with
                tool="noop".

  "abort"    -- task is unrecoverable.  Pair with tool="noop".

NEXT TOOL (the action to run THIS cycle, after applying subgoal_action):

  grasp(target: str)
    Pick up `target` from the scene.  Use a noun phrase the detector
    can localize ("red block", "blue bowl") — no pronouns, no
    relations, no "the X that …" descriptions.

  place(destination: str, held_object_hint: str)
    Place the currently held object at `destination`.  The
    destination is a free-form natural-language phrase that names
    *where the held object should end up*.  The VLM-pointing
    perception layer reads the whole phrase and picks the target
    pixel — there is NO separate relation field.

    PHRASING RULES — follow these literally, they materially affect
    where the perception layer points:

    1. **Containment** — when the destination is *inside* a
       container (bowl, bin, tray, cup, basket, box), use
       "in <container>":
         - "in the white bowl"
         - "in the grey bin"

    2. **Spatial relation to an anchor object** — when you want to
       place NEAR / LEFT OF / RIGHT OF / NEXT TO / BEHIND / IN
       FRONT OF an anchor (i.e. not on top of it), prepend
       "empty space":
         - "empty space next to the green lemon"
         - "empty space left of the rubiks cube"
         - "empty space near the orange"
       (Without "empty space", VLM pointers tend to land *on* the
       anchor instead of beside it.)

    3. **Open table surface** — when there's no specific anchor,
       use "empty space on the table" rather than just "on the
       table" (same reason as rule 2).

    4. **On top of a surface object** — when the destination IS the
       surface itself (a shelf, a tray, a rack), use "on <surface>":
         - "on the wire rack shelf"
         - "on the rim of the grey bin"

    `held_object_hint` is the noun phrase of the object in the
    gripper (typically the same `target` you grasped).

  noop
    Skip a cycle without running a tool.  REQUIRED pairing for
    subgoal_action ∈ {"advance", "replan", "abort"}.  Also fine
    as a passive re-check when you want to look again next cycle.

PAIRING RULES (enforced — break them and your action is dropped):
  - "continue" → tool MUST be "grasp" or "place" (the work isn't
    done; you need to do something).  Using "noop" here wastes a
    cycle.
  - "advance" / "replan" / "abort" → tool MUST be "noop".  These
    are subgoal-pointer moves, not actions.  If you want to do
    something AND the subgoal happens to finish from doing it,
    that's still "continue" + the tool — the next cycle will see
    the result and you can emit "advance" + "noop" then.

OUTPUT: a single JSON object, exactly these four top-level keys, no
markdown, no commentary outside the JSON:
  {
    "subgoal_action": "advance" | "continue" | "replan" | "abort",
    "tool":           "grasp"   | "place"    | "noop",
    "args":           { <tool-specific> },
    "reason":         "<one short sentence describing what HAS
                       happened or why you picked this tool;
                       do not promise future actions here>"
  }

Examples (all good — follow these patterns):

  First cycle, nothing has happened yet — grasp first:
  {"subgoal_action": "continue",
   "tool": "grasp", "args": {"target": "red block"},
   "reason": "Subgoal 0 needs the red block; grasping it."}

  Grasp succeeded, the held object now needs to go into a container.
  Use "continue" — the subgoal is NOT done until the place finishes.
  {"subgoal_action": "continue",
   "tool": "place", "args": {"destination": "in the blue bowl",
                             "held_object_hint": "red block"},
   "reason": "Block is in gripper; placing into the bowl now."}

  Place finished and the NOW image shows the block IS in the bowl.
  ONLY now use "advance" — and pair with noop, never a tool.
  {"subgoal_action": "advance",
   "tool": "noop", "args": {},
   "reason": "Red block visible in the blue bowl; subgoal 0 done."}

  Place near an anchor object (no container) — note "empty space":
  {"subgoal_action": "continue",
   "tool": "place", "args": {"destination":
                                "empty space next to the green lemon",
                             "held_object_hint": "orange fruit"},
   "reason": "Subgoal asks to consolidate fruit by the lemon."}

  Place on the open table — note "empty space":
  {"subgoal_action": "continue",
   "tool": "place", "args": {"destination": "empty space on the table",
                             "held_object_hint": "yellow lemon"},
   "reason": "Lemon is grasped, setting it down on the table."}

  Previous grasp picked the wrong object; retry the same subgoal:
  {"subgoal_action": "continue",
   "tool": "grasp", "args": {"target": "the leftmost red cube"},
   "reason": "Last grasp picked a different red block; disambiguate."}

  Scene shifted (e.g. an object knocked over), replan:
  {"subgoal_action": "replan",
   "tool": "noop", "args": {},
   "reason": "Blue bowl tipped over; re-decompose from current scene."}

ANTI-EXAMPLES (do NOT do these):

  ✗ Combining "advance" with a tool — the tool is silently dropped
    when subgoal_action ≠ "continue", and on the FINAL subgoal this
    immediately ends the task without running it:
    {"subgoal_action": "advance",
     "tool": "place", "args": {…},
     "reason": "now placing it on the table to finish the task"}
    ↑ wrong — write this as "continue" + tool="place".  Emit
    "advance" + tool="noop" only AFTER the place is done and the
    NOW image confirms the object is in its destination.

  ✗ Using "reason" to narrate a future action without issuing the
    tool:
    {"subgoal_action": "advance",
     "tool": "noop", "args": {},
     "reason": "Cube grasped; now I will place it on the table."}
    ↑ wrong — if the cube is in the gripper, the place hasn't
    happened yet.  Emit "continue" + tool="place".

  ✗ Extra fields / wrapping the JSON in markdown / preface text.
    Output the raw JSON object only.
"""


_TOOL_CHAIN_FRONT_CAM_NOTE = """

NOTE on camera orientation: the "Front camera" view is BOTH L/R \
and front/behind FLIPPED from the robot's perspective. \
image-LEFT↔robot-RIGHT, image-TOP↔robot-BEHIND.

To avoid frame confusion, describe targets by VISUAL FEATURES \
(color, type, proximity to landmarks) in every `target` / \
`destination` — NOT by left/right/front/behind. Example: "the grey \
container next to the red block" instead of "the right bin". If the \
subgoal uses a direction, first apply the flip to find which object \
it means in the image, then re-describe by visual features."""


_TOOL_CHAIN_STACK_NOTE = """

STACK CONTROL (available on place): place() accepts an optional \
boolean `stack` arg.
  - `stack: false` (DEFAULT) — the robot puts the object down with a \
    plain top-down release. Use this for ordinary pick-and-place: \
    dropping an object into a bowl/bin, setting it on an open table \
    surface, or placing it next to another object.
  - `stack: true` — the robot keeps the held object's current \
    orientation at release instead of forcing top-down. Use this ONLY \
    when the object must be set down ON TOP OF another object in a way \
    that preserves how it is currently held (e.g. stacking a block on \
    a block, nesting a lid on a container). When in doubt, prefer \
    `stack: false`.
Example: {"tool": "place", "args": {"destination": "on the red \
block", "held_object_hint": "blue block", "stack": true}}

IMPORTANT — set `stack` CONSISTENTLY on grasp() AND place() for the same \
object. grasp() also accepts the same optional `stack` arg:
  - `stack: false` (DEFAULT) — grasp the object from the best available \
    angle (side or top). Use for objects you will drop from above.
  - `stack: true` — grasp with a controlled top-down hold, so the object \
    can later be set down precisely on top of another object.
Whatever `stack` you intend for the place(), pass the SAME value on the \
grasp() that picks that object up.
Example: {"tool": "grasp", "args": {"target": "blue block", "stack": true}}"""


# ----------------------------------------------------------------------
# User content builder
# ----------------------------------------------------------------------

def build_tool_chain_user_content(
    *,
    task: str,
    subgoals: list[str],
    current_idx: int,
    before_image: np.ndarray | None,
    now_image: np.ndarray,
    last_tool: str | None,
    last_args: dict[str, Any] | None,
    last_status: str | None,
    last_reason: str | None,
) -> list[dict]:
    """Build the user-content list for one tool_chain VLM call.

    Returns a list of message-content dicts compatible with
    OpenAI / NVIDIA-inference chat.completions: a leading text block
    describing the plan + history, then BEFORE and NOW images.
    """
    lines = [f"Task: {task}", "", "Subgoals:"]
    if not subgoals:
        lines.append("  (no subgoals yet)")
    else:
        for i, sg in enumerate(subgoals):
            marker = "->" if i == current_idx else "  "
            lines.append(f"  {marker} [{i}] {sg}")

    if last_tool is None:
        lines.append("")
        lines.append("No tool has run yet (this is the first cycle).")
    else:
        lines.append("")
        lines.append("Last tool:")
        args_str = ", ".join(f"{k}={v!r}" for k, v in (last_args or {}).items())
        lines.append(f"  {last_tool}({args_str})")
        lines.append(f"  status: {last_status}")
        if last_reason:
            lines.append(f"  details: {last_reason}")

    lines.append("")
    lines.append("Return JSON only. No markdown. No commentary.")

    user_content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]

    if before_image is not None:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_b64(before_image)}",
            },
        })
    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image_b64(now_image)}",
        },
    })
    return user_content


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------

def parse_tool_chain_response(raw: str) -> ToolChainDecision:
    """Parse a VLM response into a :class:`ToolChainDecision`.

    Raises:
        ValueError when the JSON is malformed or required fields are
        missing / invalid.  Caller decides what to do (the strategy
        treats parse failures as a forced ``continue + noop`` to
        avoid wedging the loop, with a loud warning log).
    """
    try:
        data = parse_json(raw)
    except Exception as exc:
        raise ValueError(
            f"could not parse VLM response as JSON: {exc!s}; "
            f"raw[:200]={raw[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"VLM response is not a JSON object; got {type(data).__name__}"
        )

    sg = data.get("subgoal_action")
    tool = data.get("tool")
    if sg is None or tool is None:
        raise ValueError(
            f"missing required field(s) in VLM response: "
            f"subgoal_action={sg!r}, tool={tool!r}"
        )

    args = data.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError(
            f"`args` must be a JSON object; got {type(args).__name__}"
        )

    reason = str(data.get("reason", "")).strip()

    decision = ToolChainDecision(
        subgoal_action=sg,
        tool=tool,
        args=args,
        reason=reason,
        raw_response=raw,
    )

    # Tool-arg validation: required fields per tool.
    if decision.tool == "grasp":
        target = decision.args.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(
                f"grasp tool requires `args.target` (non-empty string); "
                f"got {target!r}"
            )
    elif decision.tool == "place":
        destination = decision.args.get("destination")
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError(
                f"place tool requires `args.destination` (non-empty string); "
                f"got {destination!r}"
            )
        # No relation field — spatial language lives in the destination
        # phrase itself and is resolved by the VLM-pointing perception
        # layer.  Any `relation` the model emits is ignored downstream.
    # noop: no required args.

    # Pairing rule: advance / replan / abort never run a tool.  The
    # strategy will silently drop the tool field anyway (it returns
    # before tool activation when subgoal_action != "continue"), so
    # accepting it from the VLM hides bugs — the most visible one
    # being premature tool_chain_done when advance fires on the
    # final subgoal AND the VLM thought a place would still run.
    # Surface the mismatch with a clear ValueError so the caller can
    # log it and force a retry instead of silently ending the episode.
    if decision.subgoal_action != "continue" and decision.tool != "noop":
        raise ValueError(
            f"subgoal_action={decision.subgoal_action!r} must be paired "
            f"with tool='noop' (got tool={decision.tool!r}).  "
            f"'advance' / 'replan' / 'abort' do not execute tools; "
            f"if you wanted to run {decision.tool!r}, use "
            f"subgoal_action='continue' instead."
        )

    return decision

# === end of merged prompts ========================================
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

@dataclass
class ToolChainConfig:
    """Configuration for the ``tool_chain`` strategy.

    Inherits VLM client config (`vlm_model`, `vlm_temperature`,
    `vlm_max_tokens`, `vlm_base_url`, `vlm_api_key`) from
    :class:`SubgoalConfig` so the same flags configure both modes.
    """

    # VLM client settings — passed through to SubgoalConfig.
    vlm_model: str = "YOUR_VLM_MODEL"
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 512
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None

    # Tool-chain-specific.
    max_tools_per_subgoal: int = 5
    """Hard cap on tool calls before forcing a replan (avoid thrash)."""

    max_tools_per_episode: int = 30
    """Hard cap on tool calls before forcing abort (bound runtime cost)."""


# ----------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------

class ToolChainStrategy(SubgoalStrategy):
    """VLA-bypassed strategy: subgoal decomposition + VLM-driven tool calls.

    Inherits from :class:`SubgoalStrategy` to reuse decomposition and
    replan logic; overrides ``_on_step`` to drive the tool-chain loop.
    """

    def __init__(
        self,
        ctx: StrategyContext,
        config: ToolChainConfig,
        initial_strategy: OrchestrationStrategy | None = None,
        hitl_state=None,
        grasp_seg_mode: str = "gdino_sam2",
        place_seg_mode: str | None = None,
        env_mode: str = "robolab",
        collect_trajectories: str | None = None,
        use_front_camera: bool = False,
        grasp_topdown_threshold: float | None = None,
        motion_planner: str = "linear",
        stack_mode_enabled: bool = False,
    ):
        # ``failure_monitor`` and ``recovery_mode`` are forced None /
        # "template" — tool_chain mode does not use the failure-detector
        # / recovery-mode pipeline, the post-tool VLM check supplants
        # them.  CLI validation rejects the user passing those flags.
        sg_config = SubgoalConfig(
            check_mode="timer",                  # no periodic VLM checks
            check_interval=999_999,              # never trigger
            subgoal_timeout=999_999,
            vlm_model=config.vlm_model,
            vlm_temperature=config.vlm_temperature,
            vlm_max_tokens=config.vlm_max_tokens,
            vlm_base_url=config.vlm_base_url,
            vlm_api_key=config.vlm_api_key,
        )
        super().__init__(
            ctx,
            sg_config,
            initial_strategy=initial_strategy,
            failure_monitor=None,
            recovery_mode="template",
            hitl_state=hitl_state,
            grasp_seg_mode=grasp_seg_mode,
            place_seg_mode=place_seg_mode,
            env_mode=env_mode,
            collect_trajectories=collect_trajectories,
            use_front_camera=use_front_camera,
            gt_failure_types=None,
            grasp_topdown_threshold=grasp_topdown_threshold,
            motion_planner=motion_planner,
            stack_mode_enabled=stack_mode_enabled,
        )
        self._tc_config = config

        # Optional forced stack override for A/B testing: when
        # STACK_FORCE={true,false} the VLM's per-tool ``stack`` arg is
        # IGNORED and every grasp+place uses the forced value.  Unset (the
        # normal path) → the VLM decides.  Only meaningful when stack-mode is
        # enabled; logged loudly so the override is observable in rewrites.jsonl.
        import os as _os
        _sf = _os.environ.get("STACK_FORCE", "").strip().lower()
        if _sf in ("true", "1", "yes"):
            self._stack_force: bool | None = True
        elif _sf in ("false", "0", "no"):
            self._stack_force = False
        else:
            self._stack_force = None
        if self._stack_force is not None:
            logger.info(
                f"tool_chain: STACK_FORCE={self._stack_force} — overriding all "
                f"VLM stack decisions (A/B test mode)"
            )

    # ------------------------------------------------------------------
    # Episode start
    # ------------------------------------------------------------------

    def _on_new_episode(self, obs, state, prompt):
        # Reuse the parent's full decomposition flow + state reset.
        obs, state = super()._on_new_episode(obs, state, prompt)

        # Tool-chain bookkeeping.
        state.tool_chain_active = True
        state.tool_chain_subgoal_calls = 0
        state.tool_chain_tool_calls = 0
        state.tool_chain_task_done = False
        state.tool_chain_aborted = False
        state.tool_chain_last_tool = None
        state.tool_chain_last_args = None
        state.tool_chain_last_status = None
        state.tool_chain_last_reason = None
        state.tool_chain_pending_tool = None
        state.tool_chain_pending_args = None
        return obs, state

    # ------------------------------------------------------------------
    # Per-step loop (overrides SubgoalBaseStrategy._on_step)
    # ------------------------------------------------------------------

    def _on_step(self, obs, state):
        state.tool_chain_active = True

        # 1. Snapshot last-tool result before lifecycle ticks reset the
        # executors, then tick.  The ticks log their own DONE/FAILED
        # events; we just need the result for the next VLM call.
        self._snapshot_active_tool_result(state)

        obs, state, still_active = self._tick_grasp_lifecycle(obs, state)
        if still_active:
            return obs, state
        obs, state, still_active = self._tick_place_lifecycle(obs, state)
        if still_active:
            return obs, state

        # 2. If we're already done or aborted, just hold.
        if state.tool_chain_task_done or state.tool_chain_aborted:
            return obs, state

        # 3. Per-episode cap → abort and hold.
        if state.tool_chain_tool_calls >= self._tc_config.max_tools_per_episode:
            logger.warning(
                f"tool_chain: per-episode cap reached "
                f"({state.tool_chain_tool_calls}/"
                f"{self._tc_config.max_tools_per_episode}); aborting"
            )
            state.tool_chain_aborted = True
            state.log({
                "type": "tool_chain_episode_cap",
                "tool_calls": state.tool_chain_tool_calls,
                "max": self._tc_config.max_tools_per_episode,
            })
            return obs, state

        # 4. Per-subgoal cap → force replan instead of asking the VLM.
        if state.tool_chain_subgoal_calls >= self._tc_config.max_tools_per_subgoal:
            logger.warning(
                f"tool_chain: per-subgoal cap reached "
                f"({state.tool_chain_subgoal_calls}/"
                f"{self._tc_config.max_tools_per_subgoal}); forcing replan"
            )
            self._do_replan(
                obs, state,
                reason=(
                    f"per-subgoal cap "
                    f"({self._tc_config.max_tools_per_subgoal}) reached"
                ),
            )
            return obs, state

        # 5. VLM call.
        image = self.ctx.get_vlm_image(obs)
        if image is None:
            logger.warning(
                "tool_chain: no VLM image available; serving hold chunk"
            )
            return obs, state

        decision = self._vlm_step(obs, state, image)
        if decision is None:
            # Parse / VLM error already logged; serve a hold chunk
            # and try again next cycle.
            return obs, state

        # Log every cycle so post-hoc analysis can reconstruct the loop.
        state.log({
            "type": "tool_chain_step",
            "subgoal_action": decision.subgoal_action,
            "tool": decision.tool,
            "args": decision.args,
            "reason": decision.reason,
            "subgoal_idx": state.current_subgoal_idx,
            "subgoals": list(state.subgoals),
            "last_tool": state.tool_chain_last_tool,
            "last_status": state.tool_chain_last_status,
            "tool_calls_in_subgoal": state.tool_chain_subgoal_calls,
            "tool_calls_in_episode": state.tool_chain_tool_calls,
        })

        # 6. Apply subgoal action.
        if decision.subgoal_action == "advance":
            if state.current_subgoal_idx < len(state.subgoals) - 1:
                old_idx = state.current_subgoal_idx
                self._advance_subgoal(obs, state)
                state.tool_chain_subgoal_calls = 0
                state.log({
                    "type": "tool_chain_advance",
                    "from": old_idx,
                    "to": state.current_subgoal_idx,
                    "subgoal": state.subgoals[state.current_subgoal_idx],
                })
            else:
                logger.info(
                    "tool_chain: VLM advanced past last subgoal; task done"
                )
                state.tool_chain_task_done = True
                state.log({
                    "type": "tool_chain_done",
                    "reason": decision.reason,
                })
                return obs, state
        elif decision.subgoal_action == "replan":
            self._do_replan(obs, state, reason=decision.reason)
        elif decision.subgoal_action == "abort":
            logger.warning(
                f"tool_chain: VLM aborted (reason: {decision.reason})"
            )
            state.tool_chain_aborted = True
            state.log({
                "type": "tool_chain_abort",
                "reason": decision.reason,
            })
            return obs, state
        # else "continue": stay on current subgoal

        # 7. Activate the chosen tool.
        if decision.tool == "noop":
            # Reset last-tool tracking — next cycle has nothing fresh
            # to assess.  Don't count noop against the budget.
            state.tool_chain_last_tool = None
            state.tool_chain_last_args = None
            state.tool_chain_last_status = None
            state.tool_chain_last_reason = None
            return obs, state

        if decision.tool == "grasp":
            # Optional per-grasp stack intent (only honoured under stack-mode).
            # Default True preserves top-down filtering; the VLM sets False when
            # the object will be dropped from above (grasp angle irrelevant).
            g_stack_arg = decision.args.get("stack")
            g_stack = bool(g_stack_arg) if isinstance(g_stack_arg, bool) else True
            if self._stack_force is not None:
                g_stack = self._stack_force  # A/B override
            self._activate_grasp_for_tool_chain(
                obs, state, decision.args.get("target", ""), stack=g_stack,
            )
        elif decision.tool == "place":
            self._activate_place_for_tool_chain(obs, state, decision.args)
        # (validation in parse_tool_chain_response prevents other values)

        # Increment counters whether or not the tool actually started —
        # a failed start is still a budget hit so we don't loop forever.
        state.tool_chain_subgoal_calls += 1
        state.tool_chain_tool_calls += 1
        state.tool_chain_pending_tool = decision.tool
        state.tool_chain_pending_args = decision.args
        return obs, state

    # ------------------------------------------------------------------
    # Disable the parent's periodic VLM check and timeout machinery.
    # ------------------------------------------------------------------

    def _should_check(self, state):
        return False

    def _get_timeout(self, state):
        return 999_999_999

    # ------------------------------------------------------------------
    # VLM step call
    # ------------------------------------------------------------------

    def _vlm_step(self, obs, state, now_image):
        """Single VLM call returning a :class:`ToolChainDecision`.

        Returns None on parse / VLM error (caller serves a hold chunk
        and retries next cycle — the alternative would be a forced
        decision which silently re-introduces the kind of fallback
        the design rules warn against).
        """
        user_content = build_tool_chain_user_content(
            task=state.original_instruction or "",
            subgoals=list(state.subgoals),
            current_idx=state.current_subgoal_idx,
            before_image=state.initial_image,
            now_image=now_image,
            last_tool=state.tool_chain_last_tool,
            last_args=state.tool_chain_last_args,
            last_status=state.tool_chain_last_status,
            last_reason=state.tool_chain_last_reason,
        )
        t0 = time.time()
        system_prompt = TOOL_CHAIN_SYSTEM_PROMPT
        if self._use_front_camera:
            system_prompt = system_prompt + _TOOL_CHAIN_FRONT_CAM_NOTE
        # Only expose the `stack` arg to the planner when the master switch
        # is on — otherwise the arg would be ignored and the prompt would
        # advertise a no-op control.
        if self._stack_mode_enabled:
            system_prompt = system_prompt + _TOOL_CHAIN_STACK_NOTE
        try:
            raw = self._vlm_call(system_prompt, user_content)
        except Exception as e:
            logger.error(
                f"tool_chain: VLM call failed: {e}; "
                f"will retry next cycle"
            )
            state.log({
                "type": "tool_chain_vlm_error",
                "error": repr(e),
            })
            return None
        elapsed = time.time() - t0
        logger.info(
            f"tool_chain: VLM step ({elapsed:.1f}s) raw={raw[:300]!r}"
        )

        try:
            decision = parse_tool_chain_response(raw)
        except ValueError as e:
            logger.warning(
                f"tool_chain: malformed VLM response ({e}); "
                f"will retry next cycle"
            )
            state.log({
                "type": "tool_chain_parse_error",
                "error": str(e),
                "raw": raw[:500],
                "vlm_latency_s": round(elapsed, 2),
            })
            return None

        # Stash the raw + timing on the decision for downstream logging.
        decision.raw_response = raw
        return decision

    # ------------------------------------------------------------------
    # Replan helper
    # ------------------------------------------------------------------

    def _do_replan(self, obs, state, *, reason: str) -> None:
        old_subgoals = list(state.subgoals)
        image = self.ctx.get_vlm_image(obs)
        if image is None:
            logger.warning(
                "tool_chain: replan requested but no image available; "
                "staying on current plan"
            )
            return
        if not self._recycle(obs, state, image):
            logger.warning(
                "tool_chain: replan failed (VLM returned no new subgoals); "
                "staying on current plan"
            )
            state.log({
                "type": "tool_chain_replan_failed",
                "reason": reason,
                "old_subgoals": old_subgoals,
            })
            return
        state.tool_chain_subgoal_calls = 0
        state.log({
            "type": "tool_chain_replan",
            "reason": reason,
            "old_subgoals": old_subgoals,
            "new_subgoals": list(state.subgoals),
        })

    # ------------------------------------------------------------------
    # Tool result snapshot
    # ------------------------------------------------------------------

    def _snapshot_active_tool_result(self, state: SessionState) -> None:
        """Capture an active tool's result before its lifecycle ticker
        resets the executor.  Stores onto ``state.tool_chain_last_*``.
        """
        from vlm_orchestrator.grasp.tool import GraspPhase
        from vlm_orchestrator.place import PlacePhase

        # Grasp.
        if state.grasp_tool_active and state.grasp_tool_executor is not None:
            ex = state.grasp_tool_executor
            if ex.phase in (GraspPhase.DONE, GraspPhase.FAILED):
                state.tool_chain_last_tool = "grasp"
                state.tool_chain_last_args = {
                    "target": getattr(state, "grasp_tool_target", ""),
                }
                state.tool_chain_last_status = (
                    "DONE" if ex.phase == GraspPhase.DONE else "FAILED"
                )
                state.tool_chain_last_reason = (
                    getattr(ex, "status_message", "") or ""
                )
                return

        # Place.
        if state.place_tool_active and state.place_tool_executor is not None:
            ex = state.place_tool_executor
            if ex.phase in (PlacePhase.DONE, PlacePhase.FAILED):
                state.tool_chain_last_tool = "place"
                # Reconstruct args from session state.
                state.tool_chain_last_args = {
                    "destination": getattr(state, "place_tool_target", ""),
                    "held_object_hint": getattr(
                        state, "place_tool_held_object", "",
                    ),
                }
                state.tool_chain_last_status = (
                    "DONE" if ex.phase == PlacePhase.DONE else "FAILED"
                )
                # Place tool exposes failure_reason; combine with status_message.
                fr = getattr(ex, "failure_reason", "") or ""
                sm = getattr(ex, "status_message", "") or ""
                state.tool_chain_last_reason = (
                    f"[{fr}] {sm}" if fr else sm
                )

    # ------------------------------------------------------------------
    # Tool activation
    # ------------------------------------------------------------------

    def _activate_grasp_for_tool_chain(
        self, obs, state, target: str, stack: bool = True,
    ) -> None:
        """Direct grasp activation for tool_chain mode.

        Bypasses ``_activate_grasp_tool``'s grasp-attempt counters and
        recovery-mode plumbing — those are part of the failure-detector
        path which tool_chain does not use.

        ``stack`` couples the grasp to the intended placement: when stack-mode
        is enabled and ``stack=False`` (drop-from-above), the grasp's top-down
        filter is disabled so all GraspGen candidates are kept.
        """
        target = (target or "").strip()
        if not target:
            logger.warning(
                "tool_chain: grasp request with empty target; skipping"
            )
            return
        from vlm_orchestrator.grasp.tool import GraspToolExecutor
        if state.grasp_tool_executor is None:
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
        try:
            state.grasp_tool_executor.start(
                target_object=target, obs=obs, state=state, stack=stack,
            )
        except Exception as e:
            logger.error(
                f"tool_chain: grasp_tool start failed for {target!r}: {e}",
                exc_info=True,
            )
            return
        state.grasp_tool_active = state.grasp_tool_executor.is_active
        state.grasp_tool_phase = state.grasp_tool_executor.phase.value
        state.grasp_tool_target = target
        state.flush_actions = True
        state.log({
            "type": "tool_chain_grasp_activated",
            "target": target,
        })

    def _activate_place_for_tool_chain(
        self, obs, state, args: dict,
    ) -> None:
        """Direct place activation for tool_chain mode.

        Bypasses ``_activate_place_tool_from_hitl``'s recovery-mode
        gate (tool_chain has no recovery mode).
        """
        from vlm_orchestrator.place import (
            DestinationSpec, PlaceToolExecutor,
        )

        destination = (args.get("destination") or "").strip()
        held_hint = (args.get("held_object_hint") or "").strip()
        if not destination:
            logger.warning(
                "tool_chain: place request with empty destination; skipping"
            )
            return

        if state.place_tool_executor is None:
            state.place_tool_executor = PlaceToolExecutor(
                seg_mode=self._place_seg_mode or "gdino_sam2",
                use_front_camera=self._use_front_camera,
                vlm=self.ctx.vlm,
                motion_planner=self._get_motion_planner(),
                stack_mode_enabled=self._stack_mode_enabled,
            )

        # Optional per-call stack semantics from the VLM tool args. Only
        # honoured when the master switch is on (enforced in the executor).
        # When the switch is ON and the VLM omits ``stack``, default to
        # False (top-down simple pick-and-place) per the stack-mode design.
        # When the switch is OFF the executor ignores this entirely and
        # uses the historical grasp-consistent placement.
        stack_arg = args.get("stack")
        stack = bool(stack_arg) if isinstance(stack_arg, bool) else False
        if self._stack_force is not None:
            stack = self._stack_force  # A/B override (matches grasp side)
        if self._hitl is not None:
            state._hitl = self._hitl
        # The destination phrase carries all spatial meaning ("in the
        # blue bowl", "next to the green lemon") — perception resolves
        # it directly via vlm_point.  DestinationSpec.relation defaults
        # to "in" but is not consulted in any Z computation under the
        # drop-from-safe-height model.
        spec = DestinationSpec(target_object=destination)
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
            logger.error(
                f"tool_chain: place_tool start failed for "
                f"{destination!r}: {e}",
                exc_info=True,
            )
            return
        state.place_tool_active = state.place_tool_executor.is_active
        state.place_tool_phase = state.place_tool_executor.phase.value
        state.place_tool_target = spec.describe()
        state.place_tool_held_object = held_hint
        state.flush_actions = True
        state.log({
            "type": "tool_chain_place_activated",
            "destination": destination,
            "held_object_hint": held_hint,
            "stack": stack,
            "stack_mode_enabled": self._stack_mode_enabled,
        })
