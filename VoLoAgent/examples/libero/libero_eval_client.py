# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO evaluation client with VLM-orchestrator proxy support.

Adapts openpi's ``examples/libero/main.py`` to route through the
orchestrator proxy instead of connecting directly to the policy server.
Openpi is Copyright The openpi authors and licensed under Apache-2.0;
see NOTICE for attribution.

Key differences from the vanilla openpi LIBERO client:
  - Connects to the orchestrator proxy (default port 8001) not directly
    to the policy server (port 8000).
  - Sends ``_raw`` full-resolution images alongside 224×224 policy images
    so the proxy's VLM has better visual input for subgoal decomposition
    and progress checking.
  - Sends ``observation/ee_pos`` and ``observation/gripper_position`` for
    the proxy's failure signal detector.
  - Reads orchestrator metadata from the response (subgoals, failure
    events, flush signals) and handles ``orchestrator_flush_actions``.
  - Logs per-episode results (JSON) and saves annotated replay videos.

Usage::

    # Terminal 1: openpi policy server
    cd ~/openpi && uv run scripts/serve_policy.py --env LIBERO

    # Terminal 2: VLM orchestrator proxy
    cd ~/vlm-orchestrator
    vlm-orchestrator --env libero --mode subgoal_scene_edit \\
        --vla-port 8000 --port 8001 \\
        --log-dir results/libero/experiment_$(date +%Y%m%d_%H%M%S)

    # Terminal 3: LIBERO eval client (this script)
    cd ~/vlm-orchestrator
    python examples/libero/libero_eval_client.py \\
        --proxy-port 8001 \\
        --task-suite-name libero_10 \\
        --num-trials-per-task 50
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
import torch

# LIBERO ships init-state pickles containing numpy.core.multiarray._reconstruct.
# torch.load default flipped to weights_only=True in PyTorch 2.6+, which
# rejects those pickles.  Patch back to weights_only=False so init states
# load on every torch version.  (Same as run_mem_eval.py.)
_orig_torch_load = torch.load
def _patched_torch_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _patched_torch_load

logger = logging.getLogger(__name__)


# ── robosuite depth conversion ────────────────────────────────────────

def _robosuite_depth_to_metres(
    depth_norm: np.ndarray,
    near: float = 0.01,
    far: float = 50.0,
) -> np.ndarray:
    """Convert MuJoCo normalised [0,1] depth buffer to metres.

    Uses the standard MuJoCo zbuffer formula:
        depth_m = near / (1 - d * (1 - near / far))

    See ``robosuite.utils.camera_utils.get_real_depth_map``.

    ``near`` and ``far`` should be extracted from the sim at runtime via
    ``sim.model.vis.map.znear * sim.model.stat.extent`` (and zfar).
    The defaults (0.01, 50.0) match MuJoCo defaults with extent=1.
    """
    # Clamp to avoid division by zero at d=1
    d = np.clip(depth_norm, 0.0, 0.9999)
    return (near / (1.0 - d * (1.0 - near / far))).astype(np.float32)

# ── LIBERO dummy action: 6D zeros (no EE delta) + open gripper ──
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Resolution that LIBERO training data was rendered at.
LIBERO_ENV_RESOLUTION = 256


# ──────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    # ── Connection ──
    host: str = "0.0.0.0"
    port: int = 8001  # orchestrator proxy (not the raw policy server)

    # ── Policy settings ──
    resize_size: int = 224
    replan_steps: int = 5

    # ── LIBERO env ──
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # ── Grasp tool support ──
    enable_depth: bool = False
    """Enable depth rendering and camera matrix forwarding for the
    orchestrator's grasp tool pipeline. Adds ~10% overhead to sim step."""

    # ── GT failure detection ──
    enable_gt_state: bool = False
    """Enable ground-truth state export for the orchestrator's GT failure
    detector. Extracts per-object positions, BDDL goal predicate values,
    and grasp detection from the MuJoCo sim each step. Requires the
    orchestrator to be running with ``--failure-monitor gt`` or similar."""

    # ── Output ──
    video_out_path: str = ""
    """Fallback rollout-video dir, used ONLY when the orchestrator
    proxy doesn't advertise an episode log dir (i.e. the proxy was
    launched without ``--log-dir``).  Empty default → no fallback;
    video saving is skipped with a warning so we never accidentally
    create ``results/libero/...`` trees in parallel with the proxy's
    ``--log-dir``."""
    log_dir: str = ""
    """Unused — kept for argparse compatibility.  Per-episode
    metadata lands under the proxy's ``orchestrator_episode_log_dir``.
    Set the proxy's ``--log-dir`` to control where outputs go."""
    seed: int = 7

    prompt_override: str = "__keep__"
    """Override every task's prompt with this string before sending
    to the policy.  Use ``""`` to send an empty prompt (tests how
    much the VLA depends on language vs. visual context).  The
    sentinel ``__keep__`` (default) means use the per-task prompt
    derived from the filename."""

    # ── Plus / large-benchmark task filtering ──
    task_filter: str = ""
    """Only evaluate tasks whose name contains this substring.
    E.g. '_language_' for LIBERO-Plus language perturbation.
    Empty string means evaluate all tasks."""

    max_tasks: int = 0
    """Maximum number of tasks to evaluate (0 = no limit).
    Combined with task_filter, allows sampling e.g. 30 language tasks."""

    # ── LIBERO-plus category / difficulty filtering ──
    # LIBERO-plus replaces each base suite (libero_spatial / libero_object /
    # libero_goal / libero_10) with thousands of perturbed variants.  Each
    # task's perturbation category and difficulty are recorded in
    # libero/libero/benchmark/task_classification.json shipped with the
    # LIBERO-plus install.  These flags let you sub-select by that metadata
    # without editing suite names.  Both are no-ops when the JSON is absent
    # (vanilla LIBERO / LIBERO-Mem installs).
    task_category: str = ""
    """LIBERO-plus perturbation category to keep.  Examples:
    'Language Instructions', 'Camera Viewpoints', 'Robot Initial States',
    'Light Conditions', 'Background Textures', 'Sensor Noise',
    'Objects Layout'.  Empty string = no category filter."""

    task_difficulty: int = 0
    """LIBERO-plus difficulty level to keep (1-5).  0 = no level filter."""


# ──────────────────────────────────────────────────────────────────
# Observation packaging
# ──────────────────────────────────────────────────────────────────

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) → axis-angle (3,).

    Adapted from robosuite ``transform_utils.quat2axisangle``. Robosuite is
    Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot
    Perception and Learning Lab, licensed under the MIT License. See NOTICE.
    """
    # Clip w component
    w = float(quat[3])
    w = max(-1.0, min(1.0, w))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def build_wire_obs(
    obs: dict,
    task_description: str,
    resize_size: int,
    *,
    ground_truth_done: bool = False,
    camera_K: np.ndarray | None = None,
    camera_extrinsic: np.ndarray | None = None,
    depth_near: float = 0.01,
    depth_far: float = 50.0,
) -> dict:
    """Convert LIBERO raw env observation to orchestrator wire format.

    The dict returned here is sent over WebSocket (msgpack-numpy) to
    the orchestrator proxy, which forwards it (minus ``_raw`` keys) to
    the policy server.

    Keys produced:
      - ``observation/image``           – 224×224 policy input (180° rotated)
      - ``observation/wrist_image``     – 224×224 policy input (180° rotated)
      - ``observation/image_raw``       – 256×256 full-res for VLM
      - ``observation/wrist_image_raw`` – 256×256 full-res for VLM
      - ``observation/state``           – 8D [eef_pos(3), axisangle(3), gripper_qpos(2)]
      - ``observation/ee_pos``          – 3D EE world position (failure detector)
      - ``observation/gripper_position``– 1D mean gripper width (failure detector)
      - ``prompt``                      – task language description
      - ``ground_truth_done``           – bool (optional, for logging only)
    """
    from openpi_client import image_tools

    # 180° rotation — matches LIBERO's training data preprocessing.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    # Resize + pad to policy resolution (224×224 with letterboxing).
    img_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_resized = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )

    # State vector: [eef_pos(3), axisangle(3), gripper_qpos(2)]
    state = np.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ])

    # Gripper width: mean of two finger joint positions.
    # gripper_qpos values are ~0.04 when open, ~0.0 when closed.
    # The failure detector expects a scalar where higher = more closed,
    # so we invert: 1 - (mean_qpos / 0.04) gives ~0 open, ~1 closed.
    raw_grip = float(np.mean(obs["robot0_gripper_qpos"]))
    # Normalise: robosuite Panda gripper range is [0, 0.04]
    grip_normalised = np.array([1.0 - min(raw_grip / 0.04, 1.0)],
                               dtype=np.float64)

    wire = {
        # ── Policy inputs (forwarded to policy server) ──
        "observation/image": img_resized,
        "observation/wrist_image": wrist_resized,
        "observation/state": state.astype(np.float64),
        "prompt": str(task_description),

        # ── Full-resolution for VLM (stripped by proxy before forwarding) ──
        "observation/image_raw": img,
        "observation/wrist_image_raw": wrist,

        # ── Failure detector inputs ──
        "observation/ee_pos": obs["robot0_eef_pos"].astype(np.float64),
        "observation/gripper_position": grip_normalised,
    }

    # EE quaternion for grasp tool orientation planning.
    # robosuite stores as (x,y,z,w); the grasp tool expects (w,x,y,z).
    eef_quat_xyzw = obs.get("robot0_eef_quat")
    if eef_quat_xyzw is not None:
        q = np.asarray(eef_quat_xyzw, dtype=np.float64).flatten()
        wire["observation/ee_quat"] = np.array([q[3], q[0], q[1], q[2]])

    if ground_truth_done:
        wire["ground_truth_done"] = True

    # ── Grasp tool support: depth + camera matrices ──
    depth = obs.get("agentview_depth")
    if depth is not None:
        # robosuite depth is (H, W, 1), squeeze to (H, W).
        # MuJoCo returns depth normalised to [0, 1]; convert to metres.
        depth_squeezed = np.squeeze(depth)[::-1, ::-1].astype(np.float32)
        depth_m = _robosuite_depth_to_metres(depth_squeezed, near=depth_near, far=depth_far)
        wire["observation/depth_agentview"] = np.ascontiguousarray(depth_m)

    if camera_K is not None:
        wire["observation/camera_K"] = camera_K.flatten().astype(np.float64)
    if camera_extrinsic is not None:
        wire["observation/camera_extrinsic"] = camera_extrinsic.flatten().astype(np.float64)

    # ── GT segmentation support (for grasp_tool GT_SIM mode) ──
    # When gt_seg_provider is set (via enable_gt_seg), render the raw
    # segmentation image so the grasp tool can extract a perfect mask
    # for whatever target object it chooses — no GDino/SAM needed.
    gt_seg_provider = getattr(build_wire_obs, "_gt_seg_provider", None)
    if gt_seg_provider is not None:
        try:
            gt_seg_provider.render_mask_for_obs(
                # We don't know the target yet; render a full seg image
                # and send it as gt_seg/seg_body_ids for the grasp tool
                # to extract per-object masks from.
                # _render_full_seg returns the raw body-ID image.
                None, wire,
            )
        except Exception:
            pass
        try:
            # Render full segmentation buffer
            sim = gt_seg_provider._env.sim
            seg = sim.render(
                camera_name="agentview",
                height=img.shape[0],
                width=img.shape[1],
                segmentation=True,
            )
            # seg: (H, W, 2) — channel 0 = objtype, channel 1 = objid.
            # For standard geoms objtype=5, objid=geom_id.
            # Convert geom_ids → body_ids via model.geom_bodyid.
            geom_ids = seg[::-1, :, 1].astype(np.int32)
            geom_bodyid = sim.model.geom_bodyid
            body_ids = np.where(
                geom_ids >= 0,
                geom_bodyid[np.clip(geom_ids, 0, len(geom_bodyid) - 1)],
                -1,
            ).astype(np.int32)
            wire["gt_seg/body_ids"] = body_ids
            wire["gt_seg/obj_body_id"] = gt_seg_provider._obj_body_id
        except Exception as e:
            logger.debug(f"GT seg render failed: {e}")

    return wire


# ──────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────

def get_libero_env(task, resolution: int, seed: int, enable_depth: bool = False):
    """Initialise and return a LIBERO environment + task description.

    When ``enable_depth=True``, the environment renders depth images
    alongside RGB, and camera intrinsic/extrinsic matrices are extracted
    for the grasp tool pipeline.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if enable_depth:
        env_args["camera_depths"] = True

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    # Extract camera matrices (one-time, after env creation)
    camera_K = None
    camera_extrinsic = None
    depth_near = 0.01
    depth_far = 50.0
    if enable_depth:
        try:
            from robosuite.utils.camera_utils import (
                get_camera_intrinsic_matrix,
                get_camera_extrinsic_matrix,
            )
            camera_K = get_camera_intrinsic_matrix(
                env.env.sim, "agentview", resolution, resolution,
            )
            camera_extrinsic = get_camera_extrinsic_matrix(
                env.env.sim, "agentview",
            )
            # Extract near/far clip planes for depth conversion
            sim = env.env.sim
            extent = sim.model.stat.extent
            depth_near = float(sim.model.vis.map.znear * extent)
            depth_far = float(sim.model.vis.map.zfar * extent)
            logger.info(
                f"  Camera K:\n{camera_K}\n"
                f"  Camera extrinsic (cam→world):\n{camera_extrinsic}\n"
                f"  Depth clip: near={depth_near}, far={depth_far}"
            )
        except Exception as e:
            logger.warning(f"  Failed to extract camera matrices: {e}")

    return env, task_description, camera_K, camera_extrinsic, depth_near, depth_far


def get_max_steps(task_suite_name: str) -> int:
    """Per-suite episode length limits (from openpi's defaults).

    Extended suites (LIBERO-Plus, LIBERO-PRO) inherit limits from their
    base suite.  For example ``libero_goal_swap`` inherits from ``libero_goal``.

    For suites with **per-task** budgets (RoboCerebra: a task's segment
    count drives its step budget), the eval loop calls
    ``task_suite.get_task_max_steps(i)`` directly and the value
    returned here is only a coarse fallback when that method is absent.
    """
    limits = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
        "libero_composite": 700,  # 3-step novel chains; libero_10 baseline 520 is too tight
        "libero_composite_vague": 700,  # vague-prompt mirror of libero_composite
        "libero_composite_creative": 700,  # intent-style mirror of libero_composite
        "libero_composite_creative_v2": 700,  # scenario-style mirror of libero_composite
        "libero_mem": 1400,  # T6 has 7 subgoals × 200 steps
        # RoboCerebra: 4-15 segments × 150 sim steps per segment (per
        # their cfg.switch_steps default).  This per-suite ceiling is
        # only used when get_task_max_steps(i) isn't called; the eval
        # loop should prefer the per-task value.
        "robocerebra": 2250,  # 15 × 150 — covers the deepest task
    }
    # Direct match
    if task_suite_name in limits:
        return limits[task_suite_name]
    # Match by base suite prefix (for LIBERO-Plus and LIBERO-PRO suites
    # like libero_goal_swap, libero_spatial_lan, etc.)
    for base_name, limit in sorted(limits.items(), key=lambda x: -len(x[0])):
        if task_suite_name.startswith(base_name):
            return limit
    # Default for unknown suites
    logger.warning(
        f"Unknown task suite '{task_suite_name}', using default max_steps=400"
    )
    return 400


# ──────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────

def eval_libero(args: Args) -> None:
    """Run LIBERO evaluation through the VLM-orchestrator proxy."""
    from libero.libero import benchmark as libero_benchmark
    from openpi_client import websocket_client_policy

    # RoboCerebra registers a custom suite that scans the
    # ROBOCEREBRA_BENCH_ROOT directory tree.  Idempotent — safe to call
    # always; no-op for non-RoboCerebra runs.
    from vlm_orchestrator.aux_benchmarks.robocerebra import (
        register as _register_robocerebra,
    )
    _register_robocerebra()

    np.random.seed(args.seed)

    # Initialise LIBERO task suite.
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    # Coarse per-suite ceiling (used as fallback for suites without
    # per-task budgets).  Per-task max_steps is recomputed inside the
    # per-task loop when ``task_suite.get_task_max_steps(i)`` exists.
    suite_max_steps = get_max_steps(args.task_suite_name)
    max_steps = suite_max_steps

    is_mem_suite = "libero_mem" in args.task_suite_name
    # Suites with multi-predicate goals where partial completion is
    # informative (e.g. composite chains).  We re-evaluate each goal
    # predicate at episode end and report the fraction satisfied.
    is_composite_suite = "libero_composite" in args.task_suite_name

    logger.info(f"Task suite: {args.task_suite_name} ({num_tasks} tasks)")
    logger.info(f"Max steps per episode: {max_steps}")
    if is_mem_suite:
        logger.info("LIBERO-Mem mode: tracking subgoal completion rate (SCR)")
    if is_composite_suite:
        logger.info("Composite mode: tracking per-predicate partial score (PSR)")
    logger.info(f"Connecting to orchestrator proxy at {args.host}:{args.port}")

    # Outputs land in the orchestrator's per-episode log dir
    # (advertised via response["orchestrator_episode_log_dir"]).
    # No upfront mkdir — directories are created lazily on the first
    # response that exposes the log dir, so we never create stray
    # ``results/libero/eval_<ts>/`` trees alongside the proxy's
    # ``--log-dir`` tree.
    video_out_fallback = (
        pathlib.Path(args.video_out_path) if args.video_out_path else None
    )

    # Connect to orchestrator proxy (same WebSocket protocol as openpi).
    client = websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port
    )

    # ── Evaluation loop ──
    total_episodes = 0
    total_successes = 0
    results: list[dict] = []

    # Build task list (with optional filtering for LIBERO-Plus)
    task_ids = list(range(num_tasks))
    if args.task_filter:
        # Support "INCLUDE!EXCLUDE" syntax: e.g. "_table_!from_table_center"
        parts = args.task_filter.split("!")
        include_pat = parts[0]
        exclude_pat = parts[1] if len(parts) > 1 else ""
        task_ids = [i for i in task_ids
                    if include_pat in task_suite.get_task(i).name
                    and (not exclude_pat or exclude_pat not in task_suite.get_task(i).name)]
        logger.info(f"Filter '{args.task_filter}': {len(task_ids)}/{num_tasks} tasks match")

    # LIBERO-plus per-task classification filter.  task_classification.json
    # ships with LIBERO-plus and maps task names → {category, difficulty_level}.
    # Absent JSON (vanilla LIBERO / LIBERO-Mem) → both filters silently no-op.
    if args.task_category or args.task_difficulty:
        try:
            import libero as _libero
            # libero ships as a namespace package (no top-level
            # __init__.py), so _libero.__file__ is None.  Walk via
            # __path__ instead.
            libero_root = pathlib.Path(next(iter(_libero.__path__)))
            classif_path = (
                libero_root
                / "libero" / "benchmark" / "task_classification.json"
            )
            if classif_path.exists():
                raw = json.loads(classif_path.read_text())
                # raw is {suite_name: [{id, name, category, difficulty_level}, ...]}
                name_to_meta = {
                    entry["name"]: entry
                    for suite_entries in raw.values()
                    for entry in suite_entries
                }
                kept = []
                for tid in task_ids:
                    meta = name_to_meta.get(task_suite.get_task(tid).name)
                    if meta is None:
                        continue
                    if (args.task_category
                            and meta.get("category") != args.task_category):
                        continue
                    if (args.task_difficulty
                            and meta.get("difficulty_level") != args.task_difficulty):
                        continue
                    kept.append(tid)
                logger.info(
                    f"LIBERO-plus filter (category={args.task_category!r}, "
                    f"difficulty={args.task_difficulty}): "
                    f"{len(kept)}/{len(task_ids)} tasks match"
                )
                task_ids = kept
            else:
                logger.warning(
                    "task_classification.json not found at "
                    f"{classif_path} — --task-category / --task-difficulty "
                    "ignored.  Install LIBERO-plus to use these filters."
                )
        except Exception as e:
            logger.warning(f"LIBERO-plus filter failed ({e}) — ignoring")

    if args.max_tasks > 0 and len(task_ids) > args.max_tasks:
        import random
        random.seed(args.seed)
        task_ids = sorted(random.sample(task_ids, args.max_tasks))
        logger.info(f"Sampled {args.max_tasks} tasks")
    num_eval_tasks = len(task_ids)
    # Position-in-filtered-list lookup so per-task logging can show
    # `i-of-N matched` rather than the raw suite-wide task_id (which
    # may be far larger than the matched count after filtering).
    task_position = {tid: i for i, tid in enumerate(task_ids)}

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # Per-task step budget: prefer the suite's per-task value
        # (RoboCerebra computes this from the task's segment count),
        # fall back to the suite-wide ceiling otherwise.
        if hasattr(task_suite, "get_task_max_steps"):
            try:
                max_steps = int(task_suite.get_task_max_steps(task_id))
            except Exception as e:
                logger.warning(
                    f"  get_task_max_steps({task_id}) failed ({e}); "
                    f"falling back to suite-wide max_steps={suite_max_steps}"
                )
                max_steps = suite_max_steps
        else:
            max_steps = suite_max_steps
        env, task_description, camera_K, camera_extrinsic, depth_near, depth_far = get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed,
            enable_depth=args.enable_depth,
        )
        if args.prompt_override != "__keep__":
            task_description = args.prompt_override
            logger.info(f"  Prompt override active: '{task_description}'")

        # Optional: GT state exporter for GT failure detection
        gt_exporter = None
        if args.enable_gt_state:
            try:
                from vlm_orchestrator.aux_benchmarks.libero_gt import (
                    LiberoGTStateExporter,
                )
                gt_exporter = LiberoGTStateExporter(env)
            except Exception as e:
                logger.warning(
                    f"  Failed to init GT state exporter: {e}"
                )

            # GT segmentation provider (for grasp tool GT_SIM mode)
            # Attached to build_wire_obs as a closure variable so it
            # can render segmentation without changing the function sig.
            try:
                from vlm_orchestrator.perception.gt_segmentation import (
                    RobosuiteSegProvider,
                )
                build_wire_obs._gt_seg_provider = RobosuiteSegProvider(
                    env,
                    camera_name="agentview",
                    height=LIBERO_ENV_RESOLUTION,
                    width=LIBERO_ENV_RESOLUTION,
                )
                logger.info(
                    "  GT segmentation provider initialized "
                    f"({len(build_wire_obs._gt_seg_provider._obj_body_id)} bodies)"
                )
            except Exception as e:
                logger.debug(f"  GT seg provider init failed: {e}")
                build_wire_obs._gt_seg_provider = None

        task_episodes = 0
        task_successes = 0

        for episode_idx in range(args.num_trials_per_task):
            logger.info(
                f"\n{'='*60}\n"
                f"Task {task_position[task_id]+1}/{num_eval_tasks} "
                f"(suite_id={task_id}): {task_description}\n"
                f"Episode {episode_idx+1}/{args.num_trials_per_task}\n"
                f"{'='*60}"
            )

            # Reset environment.
            env.reset()
            # Cycle through available init states so suites that ship
            # fewer initial states than num_trials_per_task (e.g.
            # RoboCerebra ships 1 deterministic init per case) re-use
            # the available state(s) rather than IndexError'ing.
            init_idx = episode_idx % max(len(initial_states), 1)
            obs = env.set_init_state(initial_states[init_idx])

            action_plan: collections.deque = collections.deque()
            replay_images: list[np.ndarray] = []
            last_response: dict = {}  # most recent proxy response

            # Video writers are lazily constructed on the first proxy
            # response that exposes orchestrator_episode_log_dir.
            # Frames captured before that get buffered.
            raw_writer = None
            annotated_writer = None
            video_initialised = False
            pending_raw: list = []
            pending_annot: list = []  # (raw_full, step, response)

            episode_metadata: dict = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "start_time": time.time(),
                "orchestrator_subgoals": None,
                "orchestrator_failures": [],
                "episode_log_dir": None,
            }

            t = 0
            done = False

            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to settle in sim.
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Build wire-format observation.
                    wire_obs = build_wire_obs(
                        obs, task_description, args.resize_size,
                        ground_truth_done=done,
                        camera_K=camera_K,
                        camera_extrinsic=camera_extrinsic,
                        depth_near=depth_near,
                        depth_far=depth_far,
                    )

                    # Episode marker so the proxy can detect episode
                    # boundaries even when the prompt is unchanged
                    # (same task, different trial).
                    wire_obs["__episode_id"] = (
                        f"{task_id}_{episode_idx}"
                    )
                    # Sim step counter — lets the orchestrator gate VLM
                    # cadence in step units rather than chunk units.
                    # Without this, the proxy falls back to a pi05-
                    # calibrated estimate (8 steps/chunk) which is
                    # wrong for libero (5 steps/chunk → ~38% off).
                    wire_obs["__step"] = t

                    # Pack GT state for proxy's GT failure detector
                    if gt_exporter is not None:
                        try:
                            wire_obs["gt_state"] = gt_exporter.export(obs)
                        except Exception as e:
                            if t == args.num_steps_wait:
                                logger.warning(
                                    f"  GT state export failed: {e}"
                                )

                    # Capture per-step frames.  Annotated writer needs
                    # the response, so we buffer until first proxy
                    # response gives us one.
                    raw_frame = wire_obs["observation/image"]
                    raw_full = wire_obs.get(
                        "observation/image_raw", raw_frame
                    )
                    replay_images.append(raw_frame)

                    if not action_plan:
                        # ── Query orchestrator proxy for actions ──
                        response = client.infer(wire_obs)
                        last_response = response

                        # Read orchestrator metadata.
                        if "orchestrator_subgoals" in response:
                            episode_metadata["orchestrator_subgoals"] = (
                                response["orchestrator_subgoals"]
                            )
                        if "orchestrator_instruction" in response:
                            episode_metadata.setdefault(
                                "orchestrator_instructions", []
                            ).append(response["orchestrator_instruction"])
                        if "orchestrator_failure" in response:
                            episode_metadata["orchestrator_failures"].append({
                                "step": t,
                                "failure": response["orchestrator_failure"],
                            })
                        if (episode_metadata["episode_log_dir"] is None
                                and "orchestrator_episode_log_dir" in response):
                            episode_metadata["episode_log_dir"] = (
                                response["orchestrator_episode_log_dir"]
                            )

                        # Lazy video-writer init once the proxy
                        # advertises an episode log dir.  Falls back
                        # to --video-out-path when explicitly set;
                        # otherwise skips video saving with a warning.
                        if not video_initialised:
                            ep_dir = None
                            if episode_metadata["episode_log_dir"]:
                                ep_dir = pathlib.Path(
                                    episode_metadata["episode_log_dir"]
                                )
                            elif video_out_fallback:
                                task_slug = task_description.replace(" ", "_")[:80]
                                ep_dir = (
                                    video_out_fallback
                                    / f"{task_slug}_ep{episode_idx}"
                                )
                            if ep_dir is not None:
                                ep_dir.mkdir(parents=True, exist_ok=True)
                                episode_metadata["episode_log_dir"] = str(ep_dir)
                                from vlm_orchestrator.utils.video_annotator import (
                                    EpisodeVideoWriter, RawVideoWriter,
                                )
                                raw_writer = RawVideoWriter(
                                    str(ep_dir / "rollout_raw.mp4"),
                                    fps=10,
                                )
                                annotated_writer = EpisodeVideoWriter(
                                    str(ep_dir / "rollout_annotated.mp4"),
                                    fps=10,
                                )
                                video_initialised = True
                                # Flush buffered pre-init frames.
                                for f in pending_raw:
                                    raw_writer.add_frame(f)
                                for raw_full_b, step_b, resp_b in pending_annot:
                                    annotated_writer.add_frame(
                                        raw_full_b, step=step_b, response=resp_b,
                                    )
                                pending_raw.clear()
                                pending_annot.clear()
                                logger.info(f"  Videos → {ep_dir}")
                            else:
                                logger.warning(
                                    "  Skipping video save: orchestrator did "
                                    "not advertise an episode_log_dir and "
                                    "--video-out-path was not set.  Launch "
                                    "the proxy with --log-dir or pass "
                                    "--video-out-path explicitly."
                                )
                                video_initialised = True   # don't retry
                                pending_raw.clear()
                                pending_annot.clear()

                        # Handle flush signal: orchestrator changed the
                        # instruction mid-chunk, discard cached actions.
                        if response.get("orchestrator_flush_actions"):
                            action_plan.clear()
                            logger.info(
                                "[FLUSH] Proxy signalled action flush — "
                                "discarding cached chunk"
                            )

                        # Extract action chunk.
                        action_chunk = response["actions"]
                        if len(action_chunk) < args.replan_steps:
                            logger.warning(
                                f"Action chunk too short: "
                                f"{len(action_chunk)} < {args.replan_steps}"
                            )
                        action_plan.extend(
                            action_chunk[: args.replan_steps]
                        )

                    # Record the frame.
                    if raw_writer is not None:
                        raw_writer.add_frame(raw_frame)
                        annotated_writer.add_frame(
                            raw_full, step=t, response=last_response,
                        )
                    elif not video_initialised:
                        pending_raw.append(raw_frame)
                        pending_annot.append((raw_full, t, last_response))

                    # Pop next action and execute in environment.
                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    # LIBERO-Mem: advance subgoal counters each step.
                    # Capture the return so the sequence-tracker `done`
                    # propagates — base env.step calls _check_success
                    # without inc, so its `done` never fires for Mem
                    # (see run_mem_eval.py:559-573 for context).
                    if is_mem_suite:
                        mem_done = bool(env._check_success(inc=True))
                        if mem_done:
                            done = True

                    # Suites with a custom ``compute_episode_success``
                    # (e.g. RoboCerebra) override env.step's `done` —
                    # their BDDL predicate fires on partial-goal matches
                    # and would break the loop early with a false
                    # success, so we ignore `done` and run to max_steps.
                    if done and not hasattr(
                        task_suite, "compute_episode_success",
                    ):
                        break

                    t += 1

                except Exception as e:
                    logger.error(f"Exception during episode: {e}")
                    break

            # ── Episode finished ──
            task_episodes += 1
            total_episodes += 1

            # Authoritative success: prefer a suite-defined check
            # (RoboCerebra parses goal.json and inspects per-object
            # progress).  Fall back to env.step's `done` for suites
            # that don't override.
            if hasattr(task_suite, "compute_episode_success"):
                try:
                    episode_success = bool(
                        task_suite.compute_episode_success(env, task_id)
                    )
                except Exception as e:
                    logger.warning(
                        f"  compute_episode_success failed ({e}); "
                        f"falling back to env.step `done`"
                    )
                    episode_success = bool(done)
            else:
                episode_success = bool(done)

            if episode_success:
                task_successes += 1
                total_successes += 1

            episode_metadata["end_time"] = time.time()
            episode_metadata["num_steps"] = t
            episode_metadata["success"] = episode_success
            episode_metadata["duration_s"] = (
                episode_metadata["end_time"] - episode_metadata["start_time"]
            )

            # LIBERO-Mem: record subgoal completion rate
            if is_mem_suite:
                total_subgoals = env.get_goal_sequence_len()
                completed_subgoals = len(env._satisfied_subgoals)
                # If done (full success), all subgoals are completed
                if done:
                    completed_subgoals = total_subgoals
                scr = completed_subgoals / max(total_subgoals, 1)
                episode_metadata["subgoals_completed"] = completed_subgoals
                episode_metadata["subgoals_total"] = total_subgoals
                episode_metadata["scr"] = scr

            # Composite suite: re-evaluate each goal predicate at the
            # final state and report the fraction satisfied as PSR
            # (Predicate Satisfaction Rate).  Lets us distinguish a
            # 0/N policy from a "got 2 of 3" policy when binary
            # success is too coarse on novel multi-step chains.
            if is_composite_suite:
                try:
                    # The libero env wraps a robosuite problem;
                    # parsed_problem['goal_state'] is the list of
                    # primitive predicates (the suite parser flattens
                    # `(And ...)` into individual entries).
                    raw_env = getattr(env, "env", env)
                    goal_state = raw_env.parsed_problem["goal_state"]
                    sat = [
                        bool(raw_env._eval_predicate(p)) for p in goal_state
                    ]
                    n_total = len(sat)
                    n_done = int(sum(sat))
                    psr = n_done / max(n_total, 1)
                    episode_metadata["predicates_total"] = n_total
                    episode_metadata["predicates_satisfied"] = n_done
                    episode_metadata["predicates_per"] = sat
                    episode_metadata["psr"] = psr
                except Exception as e:
                    logger.debug(f"  PSR compute failed: {e}")
                    psr = None
                    n_done = 0
                    n_total = 0

            results.append(episode_metadata)

            # Close video writers (no-op if never initialised).
            if raw_writer is not None:
                try:
                    raw_writer.close()
                except Exception as e:
                    logger.debug(f"  Raw video close failed: {e}")
            if annotated_writer is not None:
                try:
                    annotated_writer.close(success=done)
                except Exception as e:
                    logger.debug(f"  Annotated video close failed: {e}")

            # Per-episode eval-side metadata next to videos under the
            # proxy's episode log dir.
            if episode_metadata["episode_log_dir"]:
                try:
                    ep_dir = pathlib.Path(episode_metadata["episode_log_dir"])
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    with open(ep_dir / "eval_metadata.json", "w") as f:
                        json.dump(episode_metadata, f, indent=2, default=str)
                except Exception as e:
                    logger.debug(f"  eval_metadata save failed: {e}")

            # Log progress.
            success_rate = total_successes / total_episodes * 100
            extra = ""
            if is_mem_suite:
                extra = f" | SCR: {scr:.0%} ({completed_subgoals}/{total_subgoals})"
            elif is_composite_suite and episode_metadata.get("psr") is not None:
                extra = (
                    f" | PSR: {episode_metadata['psr']:.0%} "
                    f"({episode_metadata['predicates_satisfied']}"
                    f"/{episode_metadata['predicates_total']})"
                )
            logger.info(
                f"  Result: {'SUCCESS' if episode_success else 'FAILURE'} | "
                f"Steps: {t}{extra} | "
                f"Running: {total_successes}/{total_episodes} "
                f"({success_rate:.1f}%)"
            )

        # Per-task summary.
        task_rate = task_successes / max(task_episodes, 1) * 100
        logger.info(
            f"\nTask '{task_description}': "
            f"{task_successes}/{task_episodes} ({task_rate:.1f}%)"
        )

    # ── Final summary ──
    final_rate = total_successes / max(total_episodes, 1) * 100
    extra_summary = ""
    if is_mem_suite:
        avg_scr = (
            sum(r.get("scr", 0) for r in results)
            / max(len(results), 1)
        )
        extra_summary = f"  Average SCR: {avg_scr:.1%}\n"
    elif is_composite_suite:
        psrs = [r["psr"] for r in results if r.get("psr") is not None]
        if psrs:
            avg_psr = sum(psrs) / len(psrs)
            tot_done = sum(
                r.get("predicates_satisfied", 0) for r in results
            )
            tot_pred = sum(
                r.get("predicates_total", 0) for r in results
            )
            extra_summary = (
                f"  Average PSR: {avg_psr:.1%}  "
                f"(predicates {tot_done}/{tot_pred} across episodes)\n"
            )
    logger.info(
        f"\n{'='*60}\n"
        f"FINAL RESULTS: {args.task_suite_name}\n"
        f"  Total: {total_successes}/{total_episodes} ({final_rate:.1f}%)\n"
        f"{extra_summary}"
        f"{'='*60}"
    )

    # No aggregate results JSON written here — per-episode results live
    # in eval_metadata.json next to each rollout video under the proxy's
    # per-episode log dir.  Roll up across episodes / runs post-hoc when
    # needed.


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tyro

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    eval_libero(tyro.cli(Args))
