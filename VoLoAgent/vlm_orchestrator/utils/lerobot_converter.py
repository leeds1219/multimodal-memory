# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert collected trajectories to LeRobot v3 format for openpi training.

Bridges the gap between our TrajectoryCollector output and openpi's
LeRobot-based training pipeline.  Produces a dataset structurally identical
to ``physical-intelligence/libero`` (or DROID, VLABench, RoboCasa — see
``EnvConfig``) so the existing openpi training configs work with minimal
modification.

The module provides two pieces:

1. **StepLevelRecorder** — a lightweight, env-agnostic recorder that
   plugs into *any* eval client's step loop.  It consumes the **wire-
   format** observation dict that every eval client already builds, plus
   the single-step action, so no env-specific code is needed.

2. **convert_to_lerobot()** — offline conversion from the recorder's
   per-episode ``data.npz`` + ``metadata.json`` output into a LeRobot v3
   dataset that openpi can consume directly.

Supported environments
~~~~~~~~~~~~~~~~~~~~~~

Each environment uses different image keys, state/action dimensions, and
camera names in the LeRobot dataset.  ``EnvConfig`` presets capture these
differences:

.. list-table::
   :header-rows: 1

   * - Env
     - Image keys (wire format)
     - State dim
     - Action dim
     - LeRobot camera names
   * - LIBERO
     - ``observation/image``, ``observation/wrist_image``
     - 8
     - 7
     - ``image``, ``wrist_image``
   * - RoboCasa
     - ``observation/image``, ``observation/wrist_image``
     - 16
     - 7
     - ``image``, ``wrist_image``
   * - VLABench
     - ``observation/image``, ``observation/second_image``,
       ``observation/wrist_image``
     - 8
     - 7
     - ``image``, ``second_image``, ``wrist_image``
   * - DROID Sim
     - ``observation/exterior_image_1_left``,
       ``observation/wrist_image_left``
     - varies
     - 8
     - ``exterior_image_1_left``, ``wrist_image_left``

Usage (recording)::

    from vlm_orchestrator.utils.lerobot_converter import StepLevelRecorder, ENV_CONFIGS

    recorder = StepLevelRecorder(
        output_dir="results/training_data",
        env_config=ENV_CONFIGS["libero"],      # or "robocasa", "vlabench", "droid_sim"
    )
    recorder.begin_episode(episode_id=0, task="pick up the red block")

    for step in range(max_steps):
        wire_obs = build_wire_obs(obs, ...)     # env-specific, already exists
        response = client.infer(wire_obs)

        action = action_plan.popleft()          # single-step action
        obs, reward, done, info = env.step(action)

        recorder.record_step(
            wire_obs=wire_obs,                  # env-agnostic wire dict
            action=action,                      # single-step action
            response=response,                  # proxy response (has source metadata)
        )

    recorder.end_episode(success=done)

Usage (conversion)::

    python -m vlm_orchestrator.utils.lerobot_converter \\
        --input results/training_data \\
        --repo-id my_user/libero_corrections \\
        --env libero \\
        --flavor corrective
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Environment configurations
# ======================================================================


@dataclass(frozen=True)
class EnvConfig:
    """Describes how to extract data from wire-format observations for a
    specific environment.

    Each eval client produces a wire-format dict before calling
    ``client.infer()``.  This config tells the recorder which keys to
    read for images, state, and actions, and what shapes/dims to expect
    in the LeRobot dataset.
    """

    name: str

    # Wire-format keys → LeRobot feature names.
    # Maps {lerobot_name: wire_key}.  Order matters for stacking.
    image_keys: dict[str, str] = field(default_factory=dict)

    # Wire-format key for proprioceptive state.
    state_key: str = "observation/state"

    # Expected dimensions (used for padding/truncation during conversion).
    state_dim: int = 8
    action_dim: int = 7

    # Image resolution stored in LeRobot dataset.
    image_size: int = 256

    # FPS of the evaluation environment.
    fps: int = 10

    # Robot type string for LeRobot metadata.
    robot_type: str = "panda"


# Pre-built configs for each supported environment.
LIBERO_CONFIG = EnvConfig(
    name="libero",
    image_keys={
        "image": "observation/image",
        "wrist_image": "observation/wrist_image",
    },
    state_dim=8,
    action_dim=7,
    fps=10,
    robot_type="panda",
)

ROBOCASA_CONFIG = EnvConfig(
    name="robocasa",
    image_keys={
        "image": "observation/image",
        "wrist_image": "observation/wrist_image",
    },
    state_dim=16,
    action_dim=7,
    fps=20,
    robot_type="panda",
)

VLABENCH_CONFIG = EnvConfig(
    name="vlabench",
    image_keys={
        "image": "observation/image",
        "second_image": "observation/second_image",
        "wrist_image": "observation/wrist_image",
    },
    state_dim=8,
    action_dim=7,
    fps=10,
    robot_type="panda",
)

DROID_SIM_CONFIG = EnvConfig(
    name="droid_sim",
    image_keys={
        "exterior_image_1_left": "observation/exterior_image_1_left",
        "wrist_image_left": "observation/wrist_image_left",
    },
    state_key="observation/joint_position",
    state_dim=8,    # joint_pos + gripper (padded to 8)
    action_dim=8,
    fps=10,
    robot_type="droid",
)

ENV_CONFIGS: dict[str, EnvConfig] = {
    "libero": LIBERO_CONFIG,
    "robocasa": ROBOCASA_CONFIG,
    "vlabench": VLABENCH_CONFIG,
    "droid_sim": DROID_SIM_CONFIG,
}


# ======================================================================
# Per-step recorder (env-agnostic, runs inside any eval client)
# ======================================================================


class StepLevelRecorder:
    """Records per-sim-step data during evaluation.

    Env-agnostic: reads from the **wire-format** observation dict that
    every eval client already constructs.  The ``EnvConfig`` describes
    which keys to look for.

    Typical integration (any eval client)::

        recorder = StepLevelRecorder("results/data", ENV_CONFIGS["libero"])
        recorder.begin_episode(0, "pick up the red block")

        for step in range(max_steps):
            wire_obs = build_wire_obs(obs, ...)   # already exists
            if not action_plan:
                response = client.infer(wire_obs)
                action_plan.extend(response["actions"][:replan_steps])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action)

            recorder.record_step(wire_obs=wire_obs, action=action,
                                 response=response)

        recorder.end_episode(success=done)

    The per-step output is independent of the env: images as uint8 numpy
    arrays, state/action as float32 arrays, metadata as JSON.  The
    downstream ``convert_to_lerobot()`` uses the same ``EnvConfig`` to
    write the correctly-shaped LeRobot dataset.
    """

    def __init__(
        self,
        output_dir: str,
        env_config: EnvConfig | None = None,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cfg = env_config or LIBERO_CONFIG
        self._episode_id = -1
        self._task = ""
        self._frames: list[dict] = []
        self._active = False

        # Global stats
        self._total_episodes = 0
        self._total_frames = 0

    # ── Episode lifecycle ──────────────────────────────────────────

    def begin_episode(self, episode_id: int, task: str) -> None:
        """Start recording a new episode."""
        if self._active:
            self.end_episode(success=False)
        self._episode_id = episode_id
        self._task = task
        self._frames = []
        self._active = True

    def record_step(
        self,
        wire_obs: dict,
        action: np.ndarray | list,
        response: dict | None = None,
        *,
        action_source: str | None = None,
        gt_score: float | None = None,
        is_recovery: bool | None = None,
    ) -> None:
        """Record one sim step.

        Parameters
        ----------
        wire_obs : dict
            The wire-format observation dict that was (or would be) sent to
            ``client.infer()``.  Must contain the image and state keys
            defined in ``self._cfg``.
        action : array-like
            The single-step action executed this step.
        response : dict, optional
            The proxy response dict.  Used to auto-detect ``action_source``,
            ``gt_score``, and ``is_recovery`` from orchestrator metadata.
        action_source : str, optional
            Override: ``"policy"`` or ``"grasp_tool"``.  If None, inferred
            from ``response``.
        gt_score : float, optional
            Override: GT subtask score.  If None, read from
            ``wire_obs["gt_state"]``.
        is_recovery : bool, optional
            Override: whether this step is part of a failure recovery.
        """
        if not self._active:
            return

        response = response or {}

        # ── Extract images ──
        images = {}
        for lerobot_name, wire_key in self._cfg.image_keys.items():
            img = wire_obs.get(wire_key)
            if img is not None and hasattr(img, "shape"):
                images[lerobot_name] = np.asarray(img, dtype=np.uint8)
            else:
                # Placeholder zeros — will be filled during conversion
                images[lerobot_name] = None

        # ── Extract state ──
        state_raw = wire_obs.get(self._cfg.state_key)
        if state_raw is not None:
            state = np.asarray(state_raw, dtype=np.float32).flatten()
        else:
            state = np.zeros(self._cfg.state_dim, dtype=np.float32)

        # ── Extract action ──
        action_arr = np.asarray(action, dtype=np.float32).flatten()

        # ── Auto-detect metadata from proxy response ──
        if action_source is None:
            grasp_info = response.get("orchestrator_grasp_tool", {})
            action_source = "grasp_tool" if grasp_info.get("active") else "policy"

        if gt_score is None:
            gt_state = wire_obs.get("gt_state")
            if gt_state is not None:
                gt_score = float(
                    gt_state.get("subtask", {}).get("score", 0.0)
                )
            else:
                gt_score = 0.0

        if is_recovery is None:
            is_recovery = bool(response.get("orchestrator_failure"))

        self._frames.append({
            "images": images,
            "state": state,
            "action": action_arr,
            "task": wire_obs.get("prompt", self._task),
            "action_source": action_source,
            "gt_score": float(gt_score),
            "is_recovery": bool(is_recovery),
        })

    def end_episode(self, success: bool) -> None:
        """Finalize episode and write to disk."""
        if not self._active:
            return
        self._active = False

        n = len(self._frames)
        if n < 5:
            logger.debug(
                f"Episode {self._episode_id} too short ({n}), skipping"
            )
            return

        ep_dir = self._output_dir / f"episode_{self._episode_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # ── Stack arrays ──
        states = np.stack([f["state"] for f in self._frames])
        actions = np.stack([f["action"] for f in self._frames])
        gt_scores = np.array(
            [f["gt_score"] for f in self._frames], dtype=np.float32
        )

        arrays_to_save = {
            "states": states,
            "actions": actions,
            "gt_scores": gt_scores,
        }

        # Stack images per camera (handle missing frames with zeros)
        for cam_name in self._cfg.image_keys:
            cam_images = []
            ref_shape = None
            for f in self._frames:
                img = f["images"].get(cam_name)
                if img is not None and ref_shape is None:
                    ref_shape = img.shape
                cam_images.append(img)

            if ref_shape is not None:
                filled = []
                for img in cam_images:
                    if img is not None and img.shape == ref_shape:
                        filled.append(img)
                    else:
                        filled.append(np.zeros(ref_shape, dtype=np.uint8))
                arrays_to_save[f"images_{cam_name}"] = np.stack(filled)

        np.savez_compressed(str(ep_dir / "data.npz"), **arrays_to_save)

        # ── Metadata ──
        meta = {
            "episode_id": self._episode_id,
            "task": self._task,
            "env": self._cfg.name,
            "num_frames": n,
            "success": success,
            "final_score": float(gt_scores[-1]) if len(gt_scores) > 0 else 0.0,
            "action_sources": [f["action_source"] for f in self._frames],
            "has_recovery": any(f["is_recovery"] for f in self._frames),
            "num_recovery_frames": sum(
                1 for f in self._frames if f["is_recovery"]
            ),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "image_keys": list(self._cfg.image_keys.keys()),
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._total_episodes += 1
        self._total_frames += n
        logger.info(
            f"[Recorder] Episode {self._episode_id}: {n} frames, "
            f"success={success}, score={meta['final_score']:.3f}, "
            f"env={self._cfg.name}"
        )
        self._frames = []


# ======================================================================
# LeRobot v3 converter (env-aware via EnvConfig)
# ======================================================================


def convert_to_lerobot(
    input_dir: str,
    repo_id: str = "local/libero_corrections",
    env_config: EnvConfig | None = None,
    flavor: str = "all",
    min_score: float = 0.0,
) -> Path:
    """Convert per-step recordings to a LeRobot v3 dataset.

    Parameters
    ----------
    input_dir : str
        Directory containing ``episode_NNNNNN/`` subdirs with ``data.npz``
        and ``metadata.json`` (written by ``StepLevelRecorder``).
    repo_id : str
        LeRobot repo ID for the output dataset.
    env_config : EnvConfig, optional
        Environment config.  If None, auto-detected from episode metadata.
    flavor : str
        ``"all"`` — include all episodes.
        ``"successful"`` — only episodes where success=True.
        ``"corrective"`` — only episodes containing recovery segments.
    min_score : float
        Minimum final score to include an episode.

    Returns
    -------
    Path to the created LeRobot dataset.
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import (
            HF_LEROBOT_HOME,
            LeRobotDataset,
        )
    except ImportError:
        logger.error(
            "lerobot package not found. Install with: pip install lerobot"
        )
        raise

    import shutil

    input_path = Path(input_dir)
    output_path = HF_LEROBOT_HOME / repo_id

    # Auto-detect env from first episode if not specified
    if env_config is None:
        env_config = _detect_env_config(input_path)

    cfg = env_config

    # Clean existing output
    if output_path.exists():
        shutil.rmtree(output_path)

    # Build feature dict from EnvConfig
    features: dict = {}
    for cam_name in cfg.image_keys:
        features[cam_name] = {
            "dtype": "image",
            "shape": (cfg.image_size, cfg.image_size, 3),
            "names": ["height", "width", "channel"],
        }
    features["state"] = {
        "dtype": "float32",
        "shape": (cfg.state_dim,),
        "names": ["state"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (cfg.action_dim,),
        "names": ["actions"],
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=cfg.robot_type,
        fps=cfg.fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=2,
    )

    # Find and filter episodes
    episode_dirs = sorted(input_path.glob("episode_*"))
    included = 0
    skipped = 0

    for ep_dir in episode_dirs:
        meta_path = ep_dir / "metadata.json"
        data_path = ep_dir / "data.npz"
        if not meta_path.exists() or not data_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Apply flavor filter
        if flavor == "successful" and not meta.get("success", False):
            skipped += 1
            continue
        if flavor == "corrective" and not meta.get("has_recovery", False):
            skipped += 1
            continue
        if meta.get("final_score", 0) < min_score:
            skipped += 1
            continue

        # Load data
        data = np.load(str(data_path))
        states = data["states"]
        actions = data["actions"]
        n_frames = len(actions)
        task = meta["task"]

        for i in range(n_frames):
            frame: dict = {}

            # Images
            for cam_name in cfg.image_keys:
                arr_key = f"images_{cam_name}"
                if arr_key in data:
                    img = data[arr_key][i]
                    if img.shape[:2] != (cfg.image_size, cfg.image_size):
                        import cv2
                        img = cv2.resize(
                            img, (cfg.image_size, cfg.image_size)
                        )
                    frame[cam_name] = img
                else:
                    frame[cam_name] = np.zeros(
                        (cfg.image_size, cfg.image_size, 3), dtype=np.uint8
                    )

            # State — pad/truncate to expected dim
            state = states[i]
            if len(state) < cfg.state_dim:
                state = np.pad(state, (0, cfg.state_dim - len(state)))
            state = state[: cfg.state_dim].astype(np.float32)
            frame["state"] = state

            # Action — pad/truncate to expected dim
            action = actions[i]
            if len(action) < cfg.action_dim:
                action = np.pad(action, (0, cfg.action_dim - len(action)))
            action = action[: cfg.action_dim].astype(np.float32)
            frame["actions"] = action

            # Task instruction
            frame["task"] = task

            dataset.add_frame(frame)

        dataset.save_episode()
        included += 1

    logger.info(
        f"LeRobot dataset created: {output_path}\n"
        f"  Episodes: {included} included, {skipped} skipped\n"
        f"  Env: {cfg.name} (state_dim={cfg.state_dim}, "
        f"action_dim={cfg.action_dim})\n"
        f"  Cameras: {list(cfg.image_keys.keys())}\n"
        f"  Flavor: {flavor}\n"
        f"  Repo ID: {repo_id}"
    )
    return output_path


def _detect_env_config(input_path: Path) -> EnvConfig:
    """Auto-detect EnvConfig from episode metadata."""
    for ep_dir in sorted(input_path.glob("episode_*")):
        meta_path = ep_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            env_name = meta.get("env", "libero")
            if env_name in ENV_CONFIGS:
                logger.info(f"Auto-detected env: {env_name}")
                return ENV_CONFIGS[env_name]
    logger.warning("Could not auto-detect env, defaulting to LIBERO")
    return LIBERO_CONFIG


# ======================================================================
# CLI
# ======================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert collected trajectories to LeRobot format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory with per-step episode recordings",
    )
    parser.add_argument(
        "--repo-id", default="local/libero_corrections",
        help="LeRobot repo ID (default: local/libero_corrections)",
    )
    parser.add_argument(
        "--env", choices=list(ENV_CONFIGS.keys()), default=None,
        help="Environment (auto-detected from metadata if not set)",
    )
    parser.add_argument(
        "--flavor", choices=["all", "successful", "corrective"],
        default="all",
        help="Which episodes to include (default: all)",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum final GT score to include (default: 0.0)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    env_config = ENV_CONFIGS[args.env] if args.env else None

    output = convert_to_lerobot(
        input_dir=args.input,
        repo_id=args.repo_id,
        env_config=env_config,
        flavor=args.flavor,
        min_score=args.min_score,
    )
    print(f"\nDataset created at: {output}")
    print(f"\nTo train with openpi:")
    print(f"  uv run scripts/train.py \\")
    print(f"    --config <your_config> \\")
    print(f"    --config.data.repo_id {args.repo_id}")


if __name__ == "__main__":
    main()
