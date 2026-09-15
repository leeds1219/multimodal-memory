# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes for orchestration strategies."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from vlm_orchestrator.vlm import VLMBackend

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Per-connection state, shared across all strategies.

    Common fields are always present. Strategy-specific fields (adaptive_*,
    subgoal_*) are only meaningful when the corresponding strategy is active.
    """

    infer_count: int = 0
    # Sim step from the eval client (sent as `__step` in obs).  Falls
    # back to ``infer_count * 8`` if the client doesn't send it.  Used by
    # strategies/handlers to gate VLM checks in step-units rather than
    # chunk-units, which keeps cadence consistent across VLAs whose
    # action chunks differ in length (pi05=8, gr00t=10, openvla=1).
    episode_step: int = 0
    # Step indices at which the last VLM check fired and the current
    # subgoal began.  Replace the chunk-counters
    # ``chunks_since_check`` / ``chunks_on_subgoal`` with step deltas:
    #   ``episode_step - step_at_last_check >= check_interval``
    #   ``episode_step - step_at_subgoal_start >= subgoal_timeout``
    step_at_last_check: int = 0
    step_at_subgoal_start: int = 0
    original_instruction: str | None = None
    rewritten_instruction: str | None = None
    episode_id: int = 0
    robolab_episode_idx: int = 0  # 0-based, resets per task

    # --- Adaptive strategy fields ---
    adaptive_decision: str | None = None
    probe_variance: float | None = None
    probe_result: object | None = None

    # --- Subgoal strategy fields ---
    subgoals: list[str] = field(default_factory=list)
    subgoals_ordered: bool = False
    current_subgoal_idx: int = 0
    initial_image: np.ndarray | None = field(default=None, repr=False)
    initial_extra_images: list[np.ndarray] | None = field(
        default=None, repr=False
    )

    # --- Action flush signal (forces eval client to discard cached chunk) ---
    flush_actions: bool = False

    # --- Frame-level orchestrator metadata (for annotated video) ---
    frame_edited: bool = False
    edit_bbox_dict: dict | None = None
    edit_mode: str = "none"
    vlm_check_result: dict | None = None
    vlm_check_infer_step: int = -1

    # --- Grasp-with-tool fields ---
    grasp_tool_active: bool = False
    grasp_tool_phase: str = "idle"
    grasp_tool_target: str = ""
    grasp_tool_executor: object | None = field(default=None, repr=False)

    # Queue of grasp pipeline debug images [(rgb_array, label), ...].
    # Populated by grasp_debug helpers during perception (synchronous).
    # The proxy pops one per chunk and forwards to annotated video as PIP.
    grasp_debug_queue: list = field(default_factory=list)
    # Currently displayed debug image (popped from queue).
    grasp_debug_image: np.ndarray | None = field(default=None, repr=False)
    grasp_debug_label: str = ""

    # --- Place-with-tool fields (mirror of grasp_tool_* above) ---
    place_tool_active: bool = False
    place_tool_phase: str = "idle"
    place_tool_target: str = ""           # description string for UI / logs
    place_tool_held_object: str = ""      # advisory hint passed to executor
    place_tool_executor: object | None = field(default=None, repr=False)

    place_debug_queue: list = field(default_factory=list)
    place_debug_image: np.ndarray | None = field(default=None, repr=False)
    place_debug_label: str = ""

    # Held-object geometry handed grasp → place for collision-aware place
    # planning.  Captured by the grasp executor at grasp time: the masked
    # TARGET-object point cloud in world/base frame, plus the FK end-effector
    # transform at the grasp config.  The place executor rigidly re-maps the
    # cloud from the grasp EE pose to the current EE pose (the object is held
    # rigidly) to build the cuRobo attached-object spheres.  ``None`` when the
    # last manipulation wasn't a grasp or the planner is collision-unaware.
    last_grasped_object_pc_world: np.ndarray | None = field(
        default=None, repr=False,
    )
    last_grasped_ee_pose: np.ndarray | None = field(
        default=None, repr=False,
    )  # (4, 4) FK EE transform at the grasp config

    # --- Tool-chain (--mode tool_chain) fields ---
    # When True the proxy serves a hold-position action chunk whenever
    # no grasp / place tool is active, instead of forwarding to the VLA.
    tool_chain_active: bool = False
    tool_chain_subgoal_calls: int = 0     # tool calls on the current subgoal
    tool_chain_tool_calls: int = 0        # tool calls in the episode
    tool_chain_task_done: bool = False    # last subgoal advanced
    tool_chain_aborted: bool = False      # VLM said abort or cap hit
    # Last tool's outcome (None on the first cycle).  Read by the
    # next vlm_step call so the VLM can reason about what just
    # happened.  Snapshotted before lifecycle ticks reset the executor.
    tool_chain_last_tool: str | None = None
    tool_chain_last_args: dict | None = field(default=None, repr=False)
    tool_chain_last_status: str | None = None
    tool_chain_last_reason: str | None = None
    # Currently-running (or just-activated) tool — for log breadcrumbs.
    tool_chain_pending_tool: str | None = None
    tool_chain_pending_args: dict | None = field(default=None, repr=False)

    # --- Trajectory collector for fine-tuning data generation ---
    trajectory_collector: object | None = field(default=None, repr=False)

    # --- Structured log entries (drained by proxy after each process()) ---
    log_entries: list[dict] = field(default_factory=list)

    # --- Passive GT metric entries (drained to metrics.jsonl by proxy) ---
    metric_entries: list[dict] = field(default_factory=list)

    # --- Current episode log directory (set by proxy on each episode rotate) ---
    episode_log_dir: str | None = None

    # --- Stable per-task slug for episode-dir naming. Set from
    # ``wire_obs["__task_slug"]`` when the eval client provides one
    # (used by VLABench, where the prompt varies per episode and so
    # cannot be used as a stable task identifier).  When None, the proxy
    # falls back to slugifying ``original_instruction``. ---
    task_slug: str | None = None

    def log(self, entry: dict) -> None:
        """Append a structured log entry (written to JSONL by the proxy)."""
        entry.setdefault("timestamp", time.time())
        entry.setdefault("episode_id", self.episode_id)
        self.log_entries.append(entry)

    def log_metric(self, event) -> None:
        """Append a passive GT metric event for ``metrics.jsonl``."""
        if hasattr(event, "to_dict"):
            entry = event.to_dict()
        else:
            entry = dict(event)
        entry.setdefault("timestamp", time.time())
        entry.setdefault("episode_id", self.episode_id)
        self.metric_entries.append(entry)


@dataclass
class StrategyContext:
    """Shared configuration and helpers available to every strategy."""

    vlm: VLMBackend
    image_key: str = "observation/exterior_image_1_left"
    prompt_key: str = "prompt"
    extra_image_keys: list[str] = field(
        default_factory=lambda: ["observation/wrist_image_left"]
    )
    vla_host: str = "127.0.0.1"
    vla_port: int = 8000
    # When set, VLM calls (planning, recycle, subgoal checks) prefer
    # this camera over the default *image_key*.  The policy still
    # receives the original camera (pi0 was trained on external_cam).
    front_image_key: str | None = None
    # System-prompt variant for subgoal decomposition / monitor / recycle.
    # "default" preserves the original robolab/LIBERO behaviour (stack-
    # and-place leaning, name-expansion encouraged).  "vlabench" tells
    # the VLM to keep the original instruction's wording when the task
    # is already a single clear action and to only expand object names
    # when the scene visibly demands disambiguation — chosen to avoid
    # injecting prompt distribution shift into VLABench's pi05 fine-tune.
    prompt_style: str = "default"

    def get_image(self, obs: dict) -> np.ndarray | None:
        """Return the primary camera image from an observation, or None."""
        img = obs.get(self.image_key)
        if isinstance(img, np.ndarray):
            return img
        # Fallback for env mismatch (e.g. --env not set for LIBERO)
        for key in self._FALLBACK_IMAGE_KEYS:
            val = obs.get(key)
            if isinstance(val, np.ndarray):
                return val
        return None

    def get_extra_images(self, obs: dict) -> list[np.ndarray] | None:
        """Return additional camera images (e.g. wrist), or None."""
        extras = []
        for key in self.extra_image_keys:
            img = obs.get(key)
            if img is not None and isinstance(img, np.ndarray):
                extras.append(img)
        return extras if extras else None

    # Common image keys across environments (robolab, LIBERO, etc.).
    # Used as last-resort fallback when the configured image_key is
    # not present in the observation dict.
    _FALLBACK_IMAGE_KEYS = (
        "observation/image_raw",
        "observation/image",
        "observation/exterior_image_1_left_raw",
        "observation/exterior_image_1_left",
    )

    def get_vlm_image(self, obs: dict) -> np.ndarray | None:
        """Return the highest-resolution primary image available.

        When *front_image_key* is configured, prefers the front camera
        for VLM calls (better top-down view for scene understanding).
        Falls back to the default exterior camera if the front camera
        is not present in the observation.  As a last resort, tries
        well-known image keys from other environments so that a
        mismatched ``--env`` flag doesn't silently break VLM / HITL
        image display.
        """
        # Try front camera first (if configured)
        if self.front_image_key:
            raw = obs.get(self.front_image_key + "_raw")
            if isinstance(raw, np.ndarray):
                return raw
            img = obs.get(self.front_image_key)
            if isinstance(img, np.ndarray):
                return img
        # Fall back to default exterior camera
        raw = obs.get(self.image_key + "_raw")
        if isinstance(raw, np.ndarray):
            return raw
        img = self.get_image(obs)
        if img is not None:
            return img
        # Last resort: try well-known keys from other environments
        for key in self._FALLBACK_IMAGE_KEYS:
            val = obs.get(key)
            if isinstance(val, np.ndarray):
                return val
        return None

    def get_vlm_extra_images(self, obs: dict) -> list[np.ndarray] | None:
        """Return high-res extra images, falling back to policy-resolution.

        When *front_image_key* is active (front camera is the primary VLM
        image), the default exterior camera is prepended to the extras so
        that the VLM receives all three viewpoints: front + external + wrist.
        """
        extras = []
        # When front camera is primary, include the external camera
        # as the first extra image so the VLM still sees it.
        if self.front_image_key:
            raw = obs.get(self.image_key + "_raw")
            if isinstance(raw, np.ndarray):
                extras.append(raw)
            else:
                img = obs.get(self.image_key)
                if isinstance(img, np.ndarray):
                    extras.append(img)
        for key in self.extra_image_keys:
            raw = obs.get(key + "_raw")
            if isinstance(raw, np.ndarray):
                extras.append(raw)
            else:
                img = obs.get(key)
                if isinstance(img, np.ndarray):
                    extras.append(img)
        return extras if extras else None

    @property
    def vlm_camera_labels(self) -> tuple[str, list[str]]:
        """Return (primary_label, extra_labels) for VLM image messages.

        When front camera is active:
          primary = "Front camera", extras = ["External camera", "Wrist camera"]
        Otherwise:
          primary = "External camera", extras = ["Wrist camera"]
        """
        if self.front_image_key:
            extra_labels = ["External camera"] + [
                "Wrist camera" for _ in self.extra_image_keys
            ]
            return "Front camera", extra_labels
        return "External camera", [
            "Wrist camera" for _ in self.extra_image_keys
        ]

    def get_prompt(self, obs: dict) -> str | None:
        return obs.get(self.prompt_key)

    def set_prompt(self, obs: dict, prompt: str) -> dict:
        """Return a shallow copy of *obs* with the prompt replaced."""
        obs = dict(obs)
        obs[self.prompt_key] = prompt
        return obs


class OrchestrationStrategy(ABC):
    """Base class for all orchestration strategies.

    A strategy controls *how* and *when* the instruction sent to the VLA
    policy is modified during an episode.  The proxy calls :meth:`process`
    once per action-chunk request from the eval client.

    To add a new strategy:

    1. Subclass ``OrchestrationStrategy``.
    2. Implement :meth:`process`.
    3. Register it in ``cli.py`` under a new ``--mode`` value.
    """

    def __init__(self, ctx: StrategyContext):
        self.ctx = ctx

    @abstractmethod
    def process(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        """Process an observation, possibly modifying the instruction.

        Called once per action-chunk request (every ~8 env steps).
        Must handle episode-boundary detection internally.

        Returns:
            ``(obs, state)`` — *obs* may have a modified prompt key;
            *state* is updated in-place and also returned.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to all strategies
    # ------------------------------------------------------------------

    def _is_new_episode(self, obs: dict, state: SessionState) -> bool:
        """Detect an episode boundary (prompt changed or first observation).

        When the proxy is in front of the strategy it sets
        ``state.episode_id`` *before* calling ``process()``.  When
        strategies run without the proxy (e.g. in tests), we fall back
        to a simple counter so ``state.episode_id`` is always ≥ 1 by the
        time ``process()`` uses it.

        The proxy also bumps ``state.episode_id`` when it detects a
        client-sent ``__episode_id`` marker, so we track the last seen
        value to detect proxy-driven episode boundaries even when the
        prompt is unchanged.
        """
        current = self.ctx.get_prompt(obs)
        ep_id = state.episode_id
        last_ep = getattr(self, "_last_seen_episode_id", 0)
        is_new = (
            state.infer_count == 0
            or current != state.original_instruction
            or (ep_id > 0 and ep_id != last_ep)
        )
        if is_new:
            if ep_id == 0:
                # First episode (no proxy)
                state.episode_id = 1
            elif state.infer_count > 0 and ep_id == last_ep:
                # Mid-connection prompt change without proxy — increment
                state.episode_id += 1
            self._last_seen_episode_id = state.episode_id
        return is_new

    def _rewrite_with_retry(
        self,
        instruction: str,
        image: np.ndarray,
        obs: dict,
    ) -> tuple[str, float]:
        """Call VLM rewrite_instruction, retrying single-cam if dual-cam
        returns the original unchanged.  Returns ``(rewritten, elapsed_s)``."""
        extra = self.ctx.get_vlm_extra_images(obs)
        t0 = time.time()
        rewritten = self.ctx.vlm.rewrite_instruction(
            instruction, image, extra_images=extra,
        )
        # If dual-cam returned identical, retry without extra images
        if rewritten == instruction and extra:
            logger.info("  VLM returned identical with dual cam, retrying single-cam")
            try:
                rewritten = self.ctx.vlm.rewrite_instruction(
                    instruction, image, extra_images=None,
                )
            except Exception:
                pass  # keep rewritten as-is
        elapsed = time.time() - t0
        return rewritten, elapsed
