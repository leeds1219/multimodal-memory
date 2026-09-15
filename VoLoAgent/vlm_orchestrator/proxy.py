# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proxy that sits between eval client and VLA policy server.

Pluggable transport: a :class:`Frontend` handles the eval-client side,
a :class:`Backend` handles the VLA-server side, and strategies in
between always see canonical openpi-schema observations.  Adding a new
VLA = add a new protocol module in ``vlm_orchestrator/protocols/``;
no changes to strategies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import time

import numpy as np

from vlm_orchestrator.diagnostics.metrics1_failure_logger import TaskFailureLogger
from vlm_orchestrator.protocols.base import (
    Backend,
    BackendConnection,
    Frontend,
    FrontendSession,
)
from vlm_orchestrator.strategies.base import (
    OrchestrationStrategy,
    SessionState,
    StrategyContext,
)
from vlm_orchestrator.strategies.passthrough import PassthroughStrategy

logger = logging.getLogger(__name__)


def _enumerate_listen_addrs(host: str) -> list[tuple[str, str]]:
    """Return ``(ifname, ipv4)`` pairs the orchestrator is reachable on.

    If ``host`` is a specific IP, returns just that. If ``host`` is
    ``0.0.0.0`` (or empty), enumerates every IPv4 address on the box via
    ``ip -4 -o addr show``, falling back to a UDP-connect trick + loopback
    if iproute2 is unavailable. The list is intended to be printed at
    startup so the operator can pick the right IP for a remote eval
    client's ``--remote-host`` flag.
    """
    if host and host not in ("0.0.0.0", "::", "*", ""):
        return [("bind", host)]

    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode()
        results: list[tuple[str, str]] = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[2] == "inet":
                results.append((parts[1], parts[3].split("/")[0]))
        if results:
            return results
    except Exception:
        pass

    results = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 1))
        results.append(("primary", s.getsockname()[0]))
        s.close()
    except Exception:
        pass
    results.append(("lo", "127.0.0.1"))
    return results


def _sanitize_filename(text: str, max_len: int = 80) -> str:
    """Convert an instruction string into a safe directory name."""
    slug = re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:max_len] or "unknown"


def _make_hold_chunk_for_obs(obs: dict) -> np.ndarray:
    """Synthesize a hold-position robolab action chunk from current obs.

    8-DOF action: 7 joints + 1 gripper (current values), tiled across an
    8-step horizon to match Pi05's expected chunk length.  Used by the
    proxy's ``tool_chain`` bypass branch when no tool is active and
    we don't want to forward to the VLA.
    """
    joints = np.asarray(
        obs.get("observation/joint_position", np.zeros(7)),
        dtype=np.float64,
    ).reshape(-1)[:7]
    gripper = float(np.asarray(
        obs.get("observation/gripper_position", [0.0]),
    ).reshape(-1)[0])
    action = np.concatenate([joints, [gripper]])
    return np.tile(action, (8, 1))


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

class ProxyConfig:
    """Configuration for the orchestrator proxy.

    The ``strategy`` field controls all instruction-management behaviour.
    Build it in the CLI or test harness and pass it here.
    """

    def __init__(
        self,
        *,
        vla_host: str = "127.0.0.1",
        vla_port: int = 8000,
        host: str = "0.0.0.0",
        port: int = 8001,
        strategy: OrchestrationStrategy | None = None,
        image_key: str = "observation/exterior_image_1_left",
        prompt_key: str = "prompt",
        extra_image_keys: list[str] | None = None,
        log_dir: str | None = None,
        robolab_output_dir: str | None = None,
        enable_gt_metrics: bool = False,
        gt_metric_types: set[str] | list[str] | tuple[str, ...] | None = None,
        placement_confirm_steps: int = 5,
        gt_metric_attribution_window_steps: int = 10,
        frontend: Frontend | None = None,
        backend: Backend | None = None,
        vla_obs_key_remap: str = "none",
        client_protocol: str = "openpi",
        # ----- backward-compat shim (used by tests / old call-sites) -----
        vlm=None,
        rewrite_strategy: str = "first_per_episode",
        adaptive_probe_n: int = 30,
    ):
        self.vla_host = vla_host
        self.vla_port = vla_port
        self.host = host
        self.port = port
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.extra_image_keys = (
            extra_image_keys
            if extra_image_keys is not None
            else ["observation/wrist_image_left"]
        )
        self.log_dir = log_dir
        self.robolab_output_dir = robolab_output_dir
        self.enable_gt_metrics = enable_gt_metrics
        self.gt_metric_types = gt_metric_types
        self.placement_confirm_steps = placement_confirm_steps
        self.gt_metric_attribution_window_steps = (
            gt_metric_attribution_window_steps
        )
        self.vla_obs_key_remap = vla_obs_key_remap
        self.client_protocol = client_protocol

        # Frontend / backend default to openpi WebSocket on both sides
        # (the only supported configuration before Option D).  Other
        # combinations are constructed by ``cli.py`` based on
        # ``--frontend`` / ``--backend`` flags.
        if frontend is None or backend is None:
            from vlm_orchestrator.protocols.openpi_ws import (
                OpenpiWsBackend, OpenpiWsFrontend,
            )
            if frontend is None:
                frontend = OpenpiWsFrontend()
            if backend is None:
                backend = OpenpiWsBackend(vla_host, vla_port)
        self.frontend = frontend
        self.backend = backend

        # If a strategy is provided, use it directly.
        # Otherwise fall back to building one from the legacy vlm / mode args
        # so that existing tests and callers keep working.
        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = _build_legacy_strategy(
                vlm=vlm,
                rewrite_strategy=rewrite_strategy,
                adaptive_probe_n=adaptive_probe_n,
                image_key=image_key,
                prompt_key=prompt_key,
                extra_image_keys=self.extra_image_keys,
                vla_host=vla_host,
                vla_port=vla_port,
            )


def _build_legacy_strategy(
    *,
    vlm,
    rewrite_strategy: str,
    adaptive_probe_n: int,
    image_key: str,
    prompt_key: str,
    extra_image_keys: list[str],
    vla_host: str,
    vla_port: int,
) -> OrchestrationStrategy:
    """Create a strategy from legacy ProxyConfig fields (backward compat)."""
    from vlm_orchestrator.vlm import PassthroughVLM

    if vlm is None:
        vlm = PassthroughVLM()

    ctx = StrategyContext(
        vlm=vlm,
        image_key=image_key,
        prompt_key=prompt_key,
        extra_image_keys=extra_image_keys,
        vla_host=vla_host,
        vla_port=vla_port,
    )

    if isinstance(vlm, PassthroughVLM):
        return PassthroughStrategy(ctx)

    if rewrite_strategy == "adaptive":
        from vlm_orchestrator.strategies.archive.adaptive import AdaptiveStrategy

        return AdaptiveStrategy(ctx, probe_n=adaptive_probe_n)

    from vlm_orchestrator.strategies.archive.rewrite import RewriteStrategy

    return RewriteStrategy(ctx, mode=rewrite_strategy)


# ------------------------------------------------------------------
# Proxy
# ------------------------------------------------------------------


class OrchestratorProxy:
    """Async websocket proxy with pluggable orchestration strategies."""

    def __init__(self, config: ProxyConfig):
        self.config = config

        # Per-episode logging state
        self._current_episode_id = 0
        self._last_instruction: str | None = None
        self._last_episode_marker: str | None = None
        self._log_file = None
        self._metric_file = None
        self._episode_log_dir = None
        self._gt_metrics = None
        if config.enable_gt_metrics:
            from vlm_orchestrator.gt_metrics import GTMetricsManager

            self._gt_metrics = GTMetricsManager(
                metric_types=config.gt_metric_types,
                placement_confirm_steps=config.placement_confirm_steps,
                attribution_window_steps=(
                    config.gt_metric_attribution_window_steps
                ),
            )

        # Per-instruction (task) episode counter for robolab symlinks.
        # Robolab numbers episodes 0-based per task, but the proxy's
        # _current_episode_id is a global counter.  This dict maps
        # cleaned instruction → number of episodes seen so far.
        self._per_task_episode_count: dict[str, int] = {}

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        log_dir_msg = (
            f", log_dir={self.config.log_dir}"
            if self.config.log_dir
            else ", log_dir=NONE (results will NOT be saved)"
        )
        logger.info(
            f"VLM Orchestrator: frontend={type(self.config.frontend).__name__} "
            f"on {self.config.host}:{self.config.port}, "
            f"backend={type(self.config.backend).__name__} "
            f"→ {self.config.vla_host}:{self.config.vla_port}, "
            f"strategy={type(self.config.strategy).__name__}"
            f"{log_dir_msg}"
        )

        addrs = _enumerate_listen_addrs(self.config.host)
        bar = "=" * 64
        lines = [bar, f" vlm-orchestrator listening on port {self.config.port}"]
        for ifname, ip in addrs:
            lines.append(f"   ws://{ip}:{self.config.port}    ({ifname})")
        lines.append(" Eval-client flag: --remote-host <one of the above>")
        lines.append(bar)
        for line in lines:
            logger.info(line)

        await self.config.frontend.serve(
            self.config.host, self.config.port, self._handle_session,
        )

    # ------------------------------------------------------------------

    async def _handle_session(self, session: FrontendSession):
        """Handle one eval-client session.

        Lifecycle:
          1. Open a backend connection to the VLA server.
          2. Forward VLA metadata to the eval client (with an
             ``orchestrator`` marker).
          3. Loop: receive canonical obs from frontend, run strategy,
             forward to backend (or grasp tool), attach orchestrator
             metadata to response, send back to frontend.
        """
        try:
            backend_conn: BackendConnection = (
                await self.config.backend.connect()
            )
        except Exception as e:
            logger.error(f"Cannot open backend connection: {e}")
            return
        remote = "client"  # protocol-agnostic placeholder for log strings

        # Receive metadata from VLA server, forward to client.  Some
        # backends (e.g. gr00t ZMQ) synthesize a minimal metadata dict
        # since the wire protocol has no metadata frame.
        try:
            metadata = await backend_conn.recv_metadata()
        except Exception as e:
            logger.error(f"Failed to read backend metadata: {e}")
            await backend_conn.close()
            return
        metadata["orchestrator"] = True
        await session.send_metadata(metadata)

        state = SessionState()

        # Aspect-1 passive task-failure logger.  Runs in every eval mode
        # whenever ``gt_state`` is in obs (i.e. the eval client was
        # launched with ``--enable-gt-state``).  Side-effecting observer
        # only — does not influence orchestration routing.
        task_failure_logger = TaskFailureLogger(
            session_log_dir=self.config.log_dir,
        )

        try:
            while True:
                # Receive observation from eval client (already in
                # canonical openpi schema — frontend has translated
                # if necessary).
                obs = await session.recv_obs()
                if obs is None:
                    logger.info(
                        f"Session ended for {remote} at infer #{state.infer_count}"
                    )
                    break

                # One-shot obs dump for offline grasp-tool validation. Gated by
                # GRASP_OBS_DUMP=<path>; writes the first obs that carries depth
                # then unsets so it fires once. Debug aid only.
                _dump_path = os.environ.get("GRASP_OBS_DUMP")
                if _dump_path and isinstance(obs, dict) and obs.get("observation/depth_external") is not None:
                    try:
                        import pickle as _pkl
                        with open(_dump_path, "wb") as _df:
                            _pkl.dump(obs, _df)
                        logger.info(f"[GRASP_OBS_DUMP] wrote obs with depth to {_dump_path}")
                    except Exception as _de:
                        logger.error(f"[GRASP_OBS_DUMP] failed: {_de}")
                    os.environ.pop("GRASP_OBS_DUMP", None)

                # DreamZero-native clients (robolab `--policy dreamzero`)
                # send ``{"endpoint": "reset"}`` at episode boundaries so
                # the DZ server can clear its AR (autoregressive) frame
                # buffer. The payload carries no observation — feeding
                # it to strategy.process() would crash on missing image
                # keys. Forward the raw bytes to the backend (the DZ
                # server's reset path returns a small ack), send a
                # no-op action back to the client (whose .reset() just
                # awaits any response and discards it), and rotate
                # per-episode bookkeeping ourselves so the NEXT real
                # infer lands in a fresh log dir.
                if (self.config.client_protocol == "dreamzero"
                        and obs.get("endpoint") == "reset"):
                    import numpy as _np
                    new_ep_marker = obs.get("__episode_id")
                    logger.info(
                        f"[DZ-RESET] forwarding reset to backend; "
                        f"client __episode_id={new_ep_marker}"
                    )
                    try:
                        await backend_conn.infer(obs)
                    except Exception as exc:
                        logger.warning(
                            f"[DZ-RESET] backend reset failed: {exc}"
                        )
                    if new_ep_marker is not None:
                        self._last_episode_marker = new_ep_marker
                    # Finalize the prior episode (if any) before the
                    # next real infer rotates state. Mirrors what the
                    # ``is_new_episode`` branch below would do.
                    if state.infer_count > 0:
                        self._drain_log_entries(state)
                        self._finalize_trajectory_episode(state)
                        self._finalize_episode_metadata(state)
                        try:
                            task_failure_logger.on_episode_end()
                        except Exception:
                            logger.exception(
                                "[task_failure_logger] end failed"
                            )
                    await session.send_action(
                        {"actions": _np.zeros((1, 8), dtype=_np.float32)}
                    )
                    continue

                # Finalize-only sentinel from the frontend (e.g. gr00t-zmq
                # ``end_episode`` endpoint).  Flush per-episode metadata
                # + create video symlinks for the just-ended episode
                # without invoking the strategy or sending an action
                # back — the frontend already ACKed the eval client.
                if obs.get("__finalize_only"):
                    if state.infer_count > 0:
                        self._drain_log_entries(state)
                        self._finalize_trajectory_episode(state)
                        self._finalize_episode_metadata(state)
                        try:
                            task_failure_logger.on_episode_end()
                        except Exception:
                            logger.exception(
                                "[task_failure_logger] end failed"
                            )
                        logger.info(
                            f"Finalized episode {state.episode_id} on "
                            f"end_episode signal "
                            f"(client ep={obs.get('__episode_id')})"
                        )
                    continue

                # Debug: log incoming observation
                client_prompt = obs.get(self.config.prompt_key)
                client_img = obs.get(self.config.image_key)
                logger.debug(
                    f"[RECV #{state.infer_count}] "
                    f"prompt={client_prompt!r}, "
                    f"img={getattr(client_img, 'shape', type(client_img))}"
                )

                # Detect episode boundary BEFORE process() so we can
                # set up the per-episode log directory in time for
                # debug images saved during strategy.process().
                # We use a proxy-level _last_instruction (not
                # state.original_instruction) so we don't interfere
                # with strategy-level episode detection.
                #
                # Clients may send ``__episode_id`` to signal a new
                # episode even when the prompt is unchanged (e.g. same
                # task, different trial).
                episode_marker = obs.pop("__episode_id", None)

                # Stable per-task slug from the eval client.  Used in
                # place of slugifying the prompt for episode-dir naming
                # when the prompt varies per episode (VLABench).  Falls
                # back to prompt-slug for clients that don't send it
                # (LIBERO, RoboCerebra, RoboCasa).
                client_task_slug = obs.pop("__task_slug", None)

                # Sim-step counter from the eval client.  When present
                # this is the ground truth.  Without it (older clients,
                # other VLAs that haven't been updated), fall back to
                # estimating from infer_count assuming pi05's 8-step
                # chunks.  Strategies should prefer state.episode_step
                # over state.infer_count for cadence gating.
                client_step = obs.pop("__step", None)
                if client_step is not None:
                    state.episode_step = int(client_step)
                else:
                    # Fall back to estimating sim steps from chunk
                    # consumption. The multiplier matches the client's
                    # open_loop_horizon: pi05=8, DreamZeroClient=24,
                    # Cosmos3 release client=32.
                    if self.config.client_protocol == "dreamzero":
                        _chunk_size = 24
                    elif self.config.client_protocol == "cosmos3":
                        _chunk_size = 32
                    else:
                        _chunk_size = 8
                    state.episode_step = state.infer_count * _chunk_size

                # Detect genuine new episode vs orchestrator flush re-infer.
                # When ``orchestrator_flush_actions`` triggers the eval
                # client to re-infer, it sends ``orchestrator_instruction``
                # (a subgoal) as the prompt.  This differs from the
                # original task instruction but is NOT a new episode.
                # Accept the prompt if it matches either the original task
                # instruction OR the current rewritten instruction.
                _prompt_changed = (
                    client_prompt != self._last_instruction
                    and client_prompt != getattr(
                        state, "rewritten_instruction", None
                    )
                    and client_prompt != getattr(
                        state, "original_instruction", None
                    )
                )
                is_new_episode = (
                    state.infer_count == 0
                    or _prompt_changed
                    or (
                        episode_marker is not None
                        and episode_marker != self._last_episode_marker
                    )
                )
                if episode_marker is not None:
                    self._last_episode_marker = episode_marker
                if is_new_episode:
                    # Finalize the PREVIOUS episode before rotating to
                    # the next one.  The session-end ``finally`` block
                    # also calls finalize, but that's unreliable for
                    # ZMQ REP frontends (recv blocks forever on peer
                    # disconnect — no exception fires) and for any
                    # frontend if the orchestrator process is killed
                    # without a clean shutdown.  Per-episode finalize
                    # ensures each completed episode gets its
                    # ``end_timestamp``, robolab video symlinks, and
                    # data.hdf5 symlink as soon as the next one starts.
                    if state.infer_count > 0:
                        self._finalize_trajectory_episode(state)
                        self._finalize_episode_metadata(state)
                        try:
                            task_failure_logger.on_episode_end()
                        except Exception:
                            logger.exception(
                                "[task_failure_logger] end failed"
                            )

                    self._last_instruction = client_prompt
                    next_ep_id = self._current_episode_id + 1
                    self._current_episode_id = next_ep_id
                    # Sync episode_id to state so strategies and log
                    # entries use the correct (proxy-authoritative) value.
                    state.episode_id = next_ep_id

                    # Track per-task (per-instruction) episode index
                    # so robolab symlinks use the correct 0-based
                    # episode number.  Robolab restarts episode
                    # numbering from 0 for each task.
                    task_key = (client_prompt or "").strip()
                    prev_count = self._per_task_episode_count.get(
                        task_key, 0
                    )
                    state.robolab_episode_idx = prev_count
                    self._per_task_episode_count[task_key] = (
                        prev_count + 1
                    )

                    # Seed original_instruction so strategies which never
                    # set it (e.g. Passthrough) still get correct symlinks
                    # and metadata.  We update on every new-episode boundary
                    # — not just infer_count==0 — because a single
                    # orchestrator session (notably gr00t-zmq REP) can serve
                    # multiple tasks back-to-back and the previous task's
                    # instruction would otherwise leak into the new
                    # episode's annotated video / metadata.  Strategies
                    # that manage their own original_instruction will
                    # overwrite this seed inside their process() call.
                    state.original_instruction = client_prompt
                    if client_task_slug:
                        state.task_slug = str(client_task_slug)
                    if self.config.log_dir:
                        self._rotate_episode_log(
                            next_ep_id, client_prompt or "unknown",
                            state=state,
                            task_slug_override=(
                                str(client_task_slug)
                                if client_task_slug else None
                            ),
                        )

                    # Aspect-1: rotate the passive logger to the new episode.
                    # Hooks after _rotate_episode_log so state.episode_log_dir
                    # is set.
                    try:
                        task_failure_logger.on_episode_start(
                            episode_log_dir=getattr(state, "episode_log_dir", None),
                            episode_id=next_ep_id,
                            instruction=client_prompt,
                        )
                    except Exception:
                        logger.exception("[task_failure_logger] start failed")

                # ---- Orchestration strategy ----
                # Run in a thread so that long-running VLM API calls
                # inside process() don't block the asyncio event loop.
                # This keeps websocket ping/pong alive and prevents the
                # eval client from disconnecting due to ping timeouts.
                obs, state = await asyncio.to_thread(
                    self.config.strategy.process, obs, state,
                )

                # Aspect-1: passive failure log.  Reads ``obs["gt_state"]``
                # (if present) and emits structured events.  Side-effect
                # only; never influences routing.
                try:
                    task_failure_logger.observe(
                        obs, step=state.episode_step,
                    )
                except Exception:
                    logger.exception("[task_failure_logger] observe failed")

                # Debug: log outgoing observation
                vla_prompt = obs.get(self.config.prompt_key)
                logger.debug(
                    f"[SEND #{state.infer_count}] prompt={vla_prompt!r}"
                )

                # Verify image data integrity (debug only).
                # Scene-edit strategies intentionally modify images,
                # so only warn for strategies that shouldn't.
                vla_img = obs.get(self.config.image_key)
                if (
                    hasattr(client_img, "tobytes")
                    and hasattr(vla_img, "tobytes")
                    and client_img.tobytes() != vla_img.tobytes()
                ):
                    strategy_name = type(self.config.strategy).__name__
                    if "SceneEdit" in strategy_name:
                        logger.debug("[OK] Image edited by SceneEditStrategy")
                    else:
                        logger.error("[BUG] Image data modified during rewrite!")

                # Strip keys the policy server doesn't need:
                #   - *_raw: full-resolution images (for VLM only)
                #   - gt_state: robolab ground-truth state machine
                #   - ground_truth_done: LIBERO success flag (logging only)
                #   - observation/ee_pos, observation/gripper_position:
                #     failure detector inputs (proxy-only)
                #   - observation/depth_*, observation/camera_*:
                #     grasp tool inputs (proxy-only)
                _STRIP_PREFIXES = (
                    "observation/depth_",
                    "observation/camera_",
                    "gt_seg/",
                )
                _STRIP_EXACT = {"gt_state", "ground_truth_done"}
                vla_obs = {
                    k: v for k, v in obs.items()
                    if (
                        not k.endswith("_raw")
                        and k not in _STRIP_EXACT
                        and not any(k.startswith(p) for p in _STRIP_PREFIXES)
                    )
                }

                # ---- grasp_with_tool: bypass VLA ----
                # When a planned grasp is active, generate actions
                # locally instead of forwarding to the VLA server.
                if (
                    getattr(state, "grasp_tool_active", False)
                    and state.grasp_tool_executor is not None
                ):
                    response = state.grasp_tool_executor.step(obs, state)
                    response["orchestrator_grasp_tool"] = {
                        "active": True,
                        "phase": state.grasp_tool_phase,
                        "target": getattr(state, "grasp_tool_target", ""),
                    }
                    logger.debug(
                        f"[GRASP] phase={state.grasp_tool_phase} "
                        f"actions_shape="
                        f"{response.get('actions', 'N/A')}"
                    )
                elif (
                    getattr(state, "place_tool_active", False)
                    and state.place_tool_executor is not None
                ):
                    # ---- place_with_tool: bypass VLA ----
                    response = state.place_tool_executor.step(obs, state)
                    response["orchestrator_place_tool"] = {
                        "active": True,
                        "phase": state.place_tool_phase,
                        "target": getattr(state, "place_tool_target", ""),
                        "held_object": getattr(
                            state, "place_tool_held_object", "",
                        ),
                    }
                    logger.debug(
                        f"[PLACE] phase={state.place_tool_phase} "
                        f"actions_shape="
                        f"{response.get('actions', 'N/A')}"
                    )
                elif getattr(state, "tool_chain_active", False):
                    # ---- tool_chain: serve a hold-position chunk ----
                    # The strategy synchronously called the VLM (or
                    # finished a tool) inside process(); now that
                    # process() returned, no tool is active so we keep
                    # the robot still rather than handing control to
                    # the VLA (which is the entire point of this mode).
                    response = {
                        "actions": _make_hold_chunk_for_obs(obs),
                    }
                    response["orchestrator_tool_chain"] = {
                        "active": True,
                        "task_done": getattr(
                            state, "tool_chain_task_done", False,
                        ),
                        "aborted": getattr(
                            state, "tool_chain_aborted", False,
                        ),
                        "subgoal_calls": getattr(
                            state, "tool_chain_subgoal_calls", 0,
                        ),
                        "tool_calls": getattr(
                            state, "tool_chain_tool_calls", 0,
                        ),
                    }
                    logger.debug(
                        "[TOOL_CHAIN] hold chunk served"
                    )
                else:
                    # ---- Normal path: forward to VLA server via backend ----
                    # Backend may run sync I/O internally (it uses
                    # asyncio.to_thread for ZMQ / sync WS) so the event
                    # loop stays unblocked even when the VLA blocks on
                    # JAX JIT compilation (30-60 s on first inference).
                    if (self.config.vla_obs_key_remap == "dreamzero"
                            and self.config.client_protocol == "openpi"):
                        # DreamZero uses the roboarena WebSocket protocol on
                        # top of openpi-ws. Five adjustments beyond plain
                        # openpi:
                        #   (1) Cameras: 0-indexed (_0, _1) instead of
                        #       DROID-convention 1-indexed (_1, _2). Robolab/
                        #       openpi sends _1/_2; we shift down by 1.
                        #   (2) Routing: every request must carry an
                        #       ``endpoint`` field ("infer" or "reset").
                        #       Without it the server raises KeyError.
                        #   (3) State reset: DreamZero's ARDroidRoboarenaPolicy
                        #       clears its frame buffers when ``session_id``
                        #       changes. Send the orchestrator episode id so
                        #       each new episode gets a clean buffer.
                        #   (4) Image resolution: DreamZero-DROID's policy
                        #       requires (H=180, W=320) input images
                        #       (DROID training res). Robolab/pi05 sends
                        #       224×224. Resize the three cameras here.
                        #   (5) Second exterior camera: DreamZero's internal
                        #       model requires both video.exterior_image_1_left
                        #       AND video.exterior_image_2_left to be present.
                        #       Robolab pi05 only has a single exterior camera.
                        #       Duplicate _0_left into _1_left when missing so
                        #       the model gets both populated (better than
                        #       crashing; still a single-view input).
                        import cv2
                        import numpy as np
                        _DZ_RES = (320, 180)  # cv2.resize takes (W, H)
                        def _resize_img(arr):
                            arr = np.asarray(arr)
                            if arr.ndim != 3 or arr.shape[2] != 3:
                                return arr
                            return cv2.resize(arr, _DZ_RES,
                                              interpolation=cv2.INTER_AREA)
                        renamed = {}
                        for k, v in vla_obs.items():
                            if k == "observation/exterior_image_2_left":
                                renamed["observation/exterior_image_1_left"] = _resize_img(v)
                            elif k == "observation/exterior_image_1_left":
                                renamed["observation/exterior_image_0_left"] = _resize_img(v)
                            elif k == "observation/wrist_image_left":
                                renamed[k] = _resize_img(v)
                            else:
                                renamed[k] = v
                        # Duplicate single exterior camera if second is missing
                        if ("observation/exterior_image_0_left" in renamed
                                and "observation/exterior_image_1_left" not in renamed):
                            renamed["observation/exterior_image_1_left"] = (
                                renamed["observation/exterior_image_0_left"]
                            )
                        renamed["endpoint"] = "infer"
                        renamed["session_id"] = f"ep{state.episode_id}"
                        vla_obs = renamed
                    logger.debug(
                        f"[VLA SEND #{state.infer_count}] "
                        f"keys={list(vla_obs)[:5]}…"
                    )
                    try:
                        response = await backend_conn.infer(vla_obs)
                    except Exception as vla_err:
                        logger.error(
                            f"[VLA ERROR #{state.infer_count}] "
                            f"{type(vla_err).__name__}: {vla_err}"
                        )
                        raise

                    # DreamZero/roboarena VLA backends return a bare
                    # ndarray (N, 8) instead of openpi's
                    # ``{"actions": ndarray}``. Wrap unconditionally so
                    # downstream orchestrator_* metadata attachment
                    # always works — this is also required when the
                    # client is robolab's native DreamZeroClient
                    # (``--policy dreamzero``) talking to us in
                    # passthrough mode without the pi05→DZ remap.
                    if not isinstance(response, dict):
                        import numpy as _np
                        response = {"actions": _np.asarray(response)}

                # Attach orchestrator metadata to response
                if state.rewritten_instruction is not None:
                    response["orchestrator_instruction"] = (
                        state.rewritten_instruction
                    )
                if state.original_instruction is not None:
                    response["orchestrator_original_instruction"] = (
                        state.original_instruction
                    )
                if state.subgoals:
                    response["orchestrator_subgoals"] = state.subgoals
                    response["orchestrator_subgoal_idx"] = (
                        state.current_subgoal_idx
                    )

                # Per-episode output directory (orchestrator-owned).  Eval
                # clients use this to colocate their videos / metadata
                # with rewrites.jsonl in the orchestrator's per-episode dir.
                if getattr(state, "episode_log_dir", None):
                    response["orchestrator_episode_log_dir"] = (
                        state.episode_log_dir
                    )

                # Log ground-truth success signal (from LIBERO or similar).
                # Informational only — not used for orchestration decisions.
                if obs.get("ground_truth_done"):
                    response["orchestrator_gt_done"] = True

                # Signal eval client to flush cached action chunk
                if getattr(state, "flush_actions", False):
                    response["orchestrator_flush_actions"] = True
                    state.flush_actions = False

                # Frame-level metadata for annotated videos
                response["orchestrator_use_front_camera"] = bool(
                    getattr(
                        getattr(self.config.strategy, "ctx", None),
                        "front_image_key", None,
                    )
                )
                response["orchestrator_frame_edited"] = state.frame_edited
                if state.edit_bbox_dict is not None:
                    response["orchestrator_edit_bbox"] = (
                        state.edit_bbox_dict
                    )
                    response["orchestrator_edit_mode"] = state.edit_mode
                if state.vlm_check_result is not None:
                    response["orchestrator_vlm_check"] = (
                        state.vlm_check_result
                    )
                    response["orchestrator_vlm_check_step"] = (
                        state.vlm_check_infer_step
                    )

                # Failure detection + grasp tool metadata
                if state.grasp_tool_active:
                    response["orchestrator_grasp_tool"] = {
                        "active": True,
                        "phase": state.grasp_tool_phase,
                        "target": state.grasp_tool_target,
                    }
                # Place tool metadata (mirrors grasp tool block)
                if getattr(state, "place_tool_active", False):
                    response["orchestrator_place_tool"] = {
                        "active": True,
                        "phase": state.place_tool_phase,
                        "target": state.place_tool_target,
                        "held_object": getattr(
                            state, "place_tool_held_object", "",
                        ),
                    }
                # Forward grasp pipeline debug images (one per chunk)
                # Pop from queue when current image has been shown enough.
                _dbg_q = getattr(state, "grasp_debug_queue", [])
                if _dbg_q and state.grasp_debug_image is None:
                    # Load next image from queue
                    img, lbl = _dbg_q.pop(0)
                    state.grasp_debug_image = img
                    state.grasp_debug_label = lbl
                    state._grasp_dbg_chunks = 0
                if getattr(state, "grasp_debug_image", None) is not None:
                    import cv2 as _cv2
                    _bgr = _cv2.cvtColor(
                        state.grasp_debug_image, _cv2.COLOR_RGB2BGR,
                    )
                    _, _buf = _cv2.imencode(
                        ".jpg", _bgr,
                        [_cv2.IMWRITE_JPEG_QUALITY, 80],
                    )
                    response["orchestrator_grasp_debug_jpg"] = (
                        _buf.tobytes()
                    )
                    response["orchestrator_grasp_debug_label"] = (
                        state.grasp_debug_label
                    )
                    # Show each debug image for ~2 chunks (16 sim steps)
                    # then advance to next in queue
                    state._grasp_dbg_chunks = getattr(
                        state, "_grasp_dbg_chunks", 0,
                    ) + 1
                    if state._grasp_dbg_chunks >= 2:
                        if _dbg_q:
                            img, lbl = _dbg_q.pop(0)
                            state.grasp_debug_image = img
                            state.grasp_debug_label = lbl
                            state._grasp_dbg_chunks = 0
                        # else: keep showing last image until cleared

                # Forward place pipeline debug images (mirrors grasp loop)
                _place_dbg_q = getattr(state, "place_debug_queue", [])
                if _place_dbg_q and state.place_debug_image is None:
                    img, lbl = _place_dbg_q.pop(0)
                    state.place_debug_image = img
                    state.place_debug_label = lbl
                    state._place_dbg_chunks = 0
                if getattr(state, "place_debug_image", None) is not None:
                    import cv2 as _cv2
                    _bgr = _cv2.cvtColor(
                        state.place_debug_image, _cv2.COLOR_RGB2BGR,
                    )
                    _, _buf = _cv2.imencode(
                        ".jpg", _bgr,
                        [_cv2.IMWRITE_JPEG_QUALITY, 80],
                    )
                    response["orchestrator_place_debug_jpg"] = (
                        _buf.tobytes()
                    )
                    response["orchestrator_place_debug_label"] = (
                        state.place_debug_label
                    )
                    state._place_dbg_chunks = getattr(
                        state, "_place_dbg_chunks", 0,
                    ) + 1
                    if state._place_dbg_chunks >= 2:
                        if _place_dbg_q:
                            img, lbl = _place_dbg_q.pop(0)
                            state.place_debug_image = img
                            state.place_debug_label = lbl
                            state._place_dbg_chunks = 0
                if hasattr(state, "_last_failure_event"):
                    response["orchestrator_failure"] = (
                        state._last_failure_event
                    )
                    response["orchestrator_failure_step"] = (
                        state.infer_count
                    )
                    del state._last_failure_event

                # ---- Trajectory collection (for fine-tuning data) ----
                self._record_trajectory_step(
                    obs, response, state, is_new_episode,
                    client_prompt,
                )

                # ---- Passive GT metrics (evaluation-only) ----
                self._record_gt_metrics(obs, state)

                await session.send_action(response)

                # Drain structured log entries from the strategy
                self._drain_log_entries(state)
                self._drain_metric_entries(state)

                state.infer_count += 1

        except Exception as e:
            if (
                "ConnectionClosed" in type(e).__name__
                or "close" in str(e).lower()
            ):
                logger.warning(
                    f"Connection closed for {remote} at infer #{state.infer_count}: "
                    f"{type(e).__name__}: {e}"
                )
            else:
                logger.error(
                    f"Error handling client {remote} at infer #{state.infer_count}: "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
        finally:
            # Finalize trajectory collection for the last episode
            self._finalize_trajectory_episode(state)

            # Drain any remaining log entries before closing
            self._drain_log_entries(state)
            self._drain_metric_entries(state)
            self._finalize_episode_metadata(state)
            try:
                task_failure_logger.on_episode_end()
            except Exception:
                logger.exception("[task_failure_logger] end failed")
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            if self._metric_file:
                self._metric_file.close()
                self._metric_file = None
            try:
                await backend_conn.close()
            except Exception:
                pass
            logger.info(
                f"Session ended for {remote}: "
                f"{state.infer_count} inferences"
            )

    # ------------------------------------------------------------------
    # Trajectory collection (for fine-tuning data)
    # ------------------------------------------------------------------

    def _record_trajectory_step(
        self,
        obs: dict,
        response: dict,
        state: "SessionState",
        is_new_episode: bool,
        client_prompt: str | None,
    ) -> None:
        """Feed the trajectory collector if one is attached to state.

        Called once per proxy loop iteration, after the response is
        obtained (from VLA or grasp tool) but before it's sent to the
        eval client.
        """
        collector = getattr(state, "trajectory_collector", None)
        if collector is None:
            return

        # Episode boundary: end previous episode, start new one
        if is_new_episode and state.infer_count > 0:
            # End the previous episode.  Use GT score from obs if available.
            gt_state = obs.get("gt_state")
            prev_score = 0.0
            prev_success = False
            if gt_state is not None:
                prev_score = float(
                    gt_state.get("subtask", {}).get("score", 0.0)
                )
            prev_success = bool(obs.get("ground_truth_done", False))
            collector.end_episode(
                success=prev_success, final_score=prev_score,
            )

        if is_new_episode:
            collector.begin_episode(
                episode_id=state.episode_id,
                task_instruction=client_prompt or "",
            )

        # Record this step
        actions = response.get("actions")
        if actions is None:
            return

        action_source = "policy"
        is_recovery = False
        failure_type = ""

        if getattr(state, "grasp_tool_active", False):
            action_source = "grasp_tool"
            is_recovery = True
        # Check if the last log entry was a failure recovery
        if state.log_entries:
            for entry in reversed(state.log_entries):
                if entry.get("type") in (
                    "failure_recovery", "gt_failure_detected",
                    "gt_grasp_escalation",
                ):
                    is_recovery = True
                    failure_type = entry.get(
                        "failure_type",
                        entry.get("gt_failure_type", ""),
                    )
                    break

        collector.record_step(
            obs=obs,
            actions=actions,
            prompt=state.rewritten_instruction or client_prompt or "",
            action_source=action_source,
            gt_state=obs.get("gt_state"),
            metadata={
                "subgoal_idx": state.current_subgoal_idx,
                "is_recovery": is_recovery,
                "failure_type": failure_type,
            },
        )

    def _finalize_trajectory_episode(
        self, state: "SessionState",
    ) -> None:
        """End the trajectory episode on connection close."""
        collector = getattr(state, "trajectory_collector", None)
        if collector is None:
            return
        if not getattr(collector, "_episode_active", False):
            return
        # Best-effort: end with unknown success
        collector.end_episode(success=False, final_score=0.0)
        collector.write_manifest()

    # ------------------------------------------------------------------
    # Per-episode logging
    # ------------------------------------------------------------------

    def _rotate_episode_log(
        self, episode_id: int, instruction: str, state=None,
        task_slug_override: str | None = None,
    ):
        """Create a new per-episode log directory and rotate the log file.

        Called BEFORE ``strategy.process()`` so that debug images saved
        during processing land in the correct episode directory.

        ``task_slug_override`` lets the eval client supply a stable
        per-task slug (via ``wire_obs["__task_slug"]``) for benchmarks
        whose prompt varies per episode (VLABench).  When None, the slug
        is derived from the instruction string.
        """
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        if self._metric_file:
            self._metric_file.close()
            self._metric_file = None
        if self._gt_metrics is not None:
            self._gt_metrics.reset()

        self._current_episode_id = episode_id
        if task_slug_override:
            task_slug = _sanitize_filename(task_slug_override)
        else:
            task_slug = _sanitize_filename(instruction)
        ep_dir = os.path.join(
            self.config.log_dir, task_slug, f"episode_{episode_id}"
        )
        os.makedirs(ep_dir, exist_ok=True)
        self._episode_log_dir = ep_dir
        if state is not None:
            state.episode_log_dir = ep_dir

        self._log_file = open(
            os.path.join(ep_dir, "rewrites.jsonl"), "a"
        )
        self._metric_file = open(
            os.path.join(ep_dir, "metrics.jsonl"), "a"
        )

        # Write episode metadata
        ep_start = time.time()
        metadata = {
            "episode_id": episode_id,
            "original_instruction": instruction,
            "start_timestamp": ep_start,
        }
        with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Store start time on state for symlink mtime filtering
        if state is not None:
            state.start_timestamp = ep_start

        # Update strategy debug image dir if supported
        strategy = self.config.strategy
        if hasattr(strategy, "config") and hasattr(
            strategy.config, "debug_image_dir"
        ):
            strategy.config.debug_image_dir = os.path.join(
                ep_dir, "debug_images"
            )

        logger.info(
            f"Episode log dir: {ep_dir}"
        )

    def _drain_log_entries(self, state: SessionState):
        """Write any structured log entries the strategy produced."""
        if not self._log_file or not state.log_entries:
            return
        for entry in state.log_entries:
            self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()
        state.log_entries.clear()

    def _record_gt_metrics(self, obs: dict, state: SessionState) -> None:
        """Run passive GT metric detectors over this proxy step."""
        if self._gt_metrics is None:
            return
        try:
            events = self._gt_metrics.step(
                obs,
                state,
                control_entries=list(state.log_entries),
            )
            for event in events:
                state.log_metric(event)
        except Exception as e:
            logger.warning(f"GT metrics detector failed: {e}", exc_info=True)

    def _drain_metric_entries(self, state: SessionState):
        """Write passive GT metric entries to metrics.jsonl."""
        if not self._metric_file or not state.metric_entries:
            return
        for entry in state.metric_entries:
            self._metric_file.write(json.dumps(entry) + "\n")
        self._metric_file.flush()
        state.metric_entries.clear()

    def _resolve_episode_log_dir(self, state: SessionState) -> str | None:
        """Reconstruct the per-connection episode log directory.

        Uses ``state.episode_id`` and ``state.original_instruction`` which
        are set per-connection, making this safe when multiple robolab
        connections overlap (robolab keeps earlier connections alive while
        starting new episodes).
        """
        if not self.config.log_dir:
            return None
        if not state.original_instruction or state.episode_id == 0:
            return None
        if getattr(state, "task_slug", None):
            task_slug = _sanitize_filename(state.task_slug)
        else:
            task_slug = _sanitize_filename(state.original_instruction)
        return os.path.join(
            self.config.log_dir, task_slug,
            f"episode_{state.episode_id}",
        )

    def _finalize_episode_metadata(self, state: SessionState):
        """Update the episode metadata.json with final info (subgoals, etc)."""
        episode_log_dir = self._resolve_episode_log_dir(state)
        if not episode_log_dir:
            return
        meta_path = os.path.join(episode_log_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            metadata["end_timestamp"] = time.time()
            metadata["infer_count"] = state.infer_count
            if state.subgoals:
                metadata["subgoals"] = state.subgoals
                metadata["subgoals_ordered"] = state.subgoals_ordered

            # Aggregate HITL action stats from the JSONL log
            hitl_stats = self._aggregate_hitl_stats(episode_log_dir)
            if hitl_stats:
                metadata["hitl"] = hitl_stats

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to finalize episode metadata: {e}")

        # Create symlinks to robolab output files (videos, logs, hdf5)
        self._symlink_robolab_outputs(state, episode_log_dir)

    @staticmethod
    def _aggregate_hitl_stats(episode_log_dir: str) -> dict | None:
        """Read the JSONL log and aggregate HITL action counts."""
        jsonl_path = os.path.join(episode_log_dir, "rewrites.jsonl")
        if not os.path.exists(jsonl_path):
            return None
        counts: dict[str, int] = {}
        recoveries: list[str] = []
        failures: list[dict] = []
        try:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    etype = entry.get("type", "")
                    if etype.startswith("hitl_"):
                        counts[etype] = counts.get(etype, 0) + 1
                    if etype == "hitl_recovery":
                        recoveries.append(entry.get("instruction", ""))
                    if etype == "hitl_failure":
                        failures.append({
                            "subgoal_idx": entry.get("subgoal_idx"),
                            "subgoal": entry.get("subgoal"),
                            "step": entry.get("step_count"),
                        })
        except Exception as e:
            logger.warning(f"Failed to aggregate HITL stats: {e}")
            return None
        if not counts:
            return None
        return {
            "mode": True,
            "action_counts": counts,
            "total_actions": sum(counts.values()),
            "recoveries": recoveries,
            "failures": failures,
        }

    def _symlink_robolab_outputs(
        self, state: SessionState, episode_log_dir: str
    ):
        """Create symlinks from the episode log dir to robolab output files.

        Searches subdirectories of ``robolab_output_dir`` (up to 2 levels
        deep) for video files whose names match the robolab naming
        convention::

          <cleaned_instruction>_<episode>.mp4

        This allows the user to pass either the specific experiment dir
        (``~/robolab/output/<experiment>/``) or the general output root
        (``~/robolab/output/``).

        The proxy episode_id is 1-based; robolab episodes are 0-based.
        Uses ``state.episode_id`` (per-connection) instead of the shared
        ``self._current_episode_id`` so that concurrent robolab
        connections each symlink to the correct episode.
        """
        if not self.config.robolab_output_dir or not episode_log_dir:
            return
        if not state.original_instruction:
            return

        instruction = state.original_instruction
        cleaned = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")
        # Use the per-task episode index (0-based, resets per task)
        # which matches robolab's per-task video numbering.
        robolab_ep = getattr(state, "robolab_episode_idx", None)
        if robolab_ep is None:
            # Fallback for sessions started before this field existed
            robolab_ep = state.episode_id - 1

        # Video files that robolab writes per episode
        video_files = [
            f"{cleaned}_{robolab_ep}.mp4",
            f"{cleaned}_{robolab_ep}_viewport.mp4",
            f"{cleaned}_{robolab_ep}_annotated.mp4",
            f"{cleaned}_{robolab_ep}_policy.mp4",
        ]
        log_file = f"log_{robolab_ep}.json"
        primary = video_files[0]

        # Find the task subdirectory by checking which one contains the
        # primary video file.  Search up to 2 levels deep so the user can
        # pass either ~/robolab/output/<experiment>/ (1 level to TaskName)
        # or ~/robolab/output/ (2 levels: experiment/TaskName).
        #
        # Use the episode start time as a lower bound to avoid matching
        # stale video files from previous experiment runs that happen to
        # share the same filename (same task, same episode index).
        robolab_dir = self.config.robolab_output_dir
        mtime_after = getattr(state, "start_timestamp", None)
        task_dir = self._find_task_dir(
            robolab_dir, primary, max_depth=2, mtime_after=mtime_after,
        )

        if task_dir is None:
            logger.warning(
                f"Could not find robolab task dir for instruction "
                f"{instruction!r} (looked for {primary} under "
                f"{robolab_dir})"
            )
            return

        # Create symlinks for videos and per-episode log
        for fname in video_files + [log_file]:
            src = os.path.join(task_dir, fname)
            if not os.path.exists(src):
                continue
            dst = os.path.join(episode_log_dir, fname)
            try:
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
                logger.info(f"Symlinked {dst} -> {src}")
            except OSError as e:
                logger.warning(f"Failed to create symlink {dst}: {e}")

        # Symlink data.hdf5 (shared across episodes in robolab)
        hdf5_src = os.path.join(task_dir, "data.hdf5")
        if os.path.exists(hdf5_src):
            hdf5_dst = os.path.join(episode_log_dir, "data.hdf5")
            try:
                if os.path.lexists(hdf5_dst):
                    os.remove(hdf5_dst)
                os.symlink(os.path.abspath(hdf5_src), hdf5_dst)
                logger.info(f"Symlinked {hdf5_dst} -> {hdf5_src}")
            except OSError as e:
                logger.warning(f"Failed to create symlink {hdf5_dst}: {e}")

    @staticmethod
    def _find_task_dir(
        root: str,
        filename: str,
        max_depth: int = 2,
        mtime_after: float | None = None,
    ) -> str | None:
        """Search *root* for a directory containing *filename*, up to
        *max_depth* levels deep.  When multiple matches exist the most
        recently modified one wins (so the current experiment is picked
        over stale ones).

        If *mtime_after* is given (epoch seconds), only files modified
        after that timestamp are considered.  This prevents matching
        stale video files from previous experiment runs.
        """
        best: str | None = None
        best_mtime: float = -1

        def _search(directory: str, depth: int) -> None:
            nonlocal best, best_mtime
            try:
                entries = os.listdir(directory)
            except OSError:
                return
            for entry in entries:
                candidate = os.path.join(directory, entry)
                if not os.path.isdir(candidate):
                    continue
                target = os.path.join(candidate, filename)
                if os.path.exists(target):
                    mtime = os.path.getmtime(target)
                    if mtime_after is not None and mtime < mtime_after:
                        continue  # stale file from a previous run
                    if mtime > best_mtime:
                        best = candidate
                        best_mtime = mtime
                elif depth > 1:
                    _search(candidate, depth - 1)

        _search(root, max_depth)
        return best
