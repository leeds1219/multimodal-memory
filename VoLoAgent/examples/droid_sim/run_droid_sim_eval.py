#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DROID Sim evaluation through VLM-orchestrator proxy.

Runs the sim-evals Isaac Sim DROID environment through the orchestrator
proxy for passthrough vs orchestrated comparison.

Requirements:
  - Isaac Sim / IsaacLab (installed in sim-evals venv)
  - sim-evals assets downloaded to ~/sim-evals/assets/
  - DROID pi0.5 server on port 8000
  - ~12GB free GPU memory for Isaac Sim

Usage::

    # From the sim-evals venv (which has Isaac Sim):
    cd ~/sim-evals
    source .venv/bin/activate

    # Passthrough (direct to VLA):
    python ~/vlm-orchestrator/examples/droid_sim/run_droid_sim_eval.py \
        --proxy-port 8000 --episodes 10 --headless

    # With orchestrator proxy:
    python ~/vlm-orchestrator/examples/droid_sim/run_droid_sim_eval.py \
        --proxy-port 8019 --episodes 10 --headless
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("droid_sim_eval")

SCENES = {
    1: "put the cube in the bowl",
    2: "put the can in the mug",
    3: "put banana in the bin",
}


def parse_args():
    p = argparse.ArgumentParser(description="DROID sim evaluation")
    p.add_argument("--proxy-host", default="127.0.0.1")
    p.add_argument("--proxy-port", type=int, default=8000,
                    help="Port of proxy or VLA server")
    p.add_argument("--episodes", type=int, default=10,
                    help="Episodes per scene")
    p.add_argument("--scenes", type=str, default="1,2,3",
                    help="Comma-separated scene IDs")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--open-loop-horizon", type=int, default=8)
    # These are passed through to Isaac Sim but we parse them out
    return p.parse_known_args()


def main():
    args, unknown = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    scenes = [int(s) for s in args.scenes.split(",")]
    from datetime import datetime
    log_dir = Path(args.log_dir or
                   f"results/droid_sim/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Launch Isaac Sim ──
    logger.info("Launching Isaac Sim (headless)...")
    from isaaclab.app import AppLauncher
    import argparse as _ap
    sim_parser = _ap.ArgumentParser()
    AppLauncher.add_app_launcher_args(sim_parser)
    args_cli, _ = sim_parser.parse_known_args([])
    args_cli.enable_cameras = True
    args_cli.headless = args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import after app launch (Isaac Sim requirement)
    import gymnasium as gym
    import torch

    # Add sim-evals to path
    sim_evals_path = str(Path(__file__).resolve().parent / "../../../sim-evals/src")
    if sim_evals_path not in sys.path:
        sys.path.insert(0, sim_evals_path)
    # Also try the home directory version
    home_sim = str(Path.home() / "sim-evals/src")
    if home_sim not in sys.path:
        sys.path.insert(0, home_sim)

    import sim_evals.environments  # noqa: F401 — registers DROID env
    from isaaclab_tasks.utils import parse_env_cfg
    from openpi_client import websocket_client_policy, image_tools

    logger.info(f"Connecting to proxy/VLA at {args.proxy_host}:{args.proxy_port}")

    all_results = []
    overall_successes = 0
    overall_episodes = 0

    for scene_id in scenes:
        instruction = SCENES[scene_id]
        logger.info(f"\n{'='*60}")
        logger.info(f"Scene {scene_id}: {instruction}")
        logger.info(f"{'='*60}")

        # Create environment
        env_cfg = parse_env_cfg(
            "DROID", device="cuda:0", num_envs=1, use_fabric=True,
        )
        env_cfg.set_scene(scene_id)
        env = gym.make("DROID", cfg=env_cfg)
        obs, _ = env.reset()
        obs, _ = env.reset()  # second reset for correct materials

        # Policy client
        client = websocket_client_policy.WebsocketClientPolicy(
            args.proxy_host, args.proxy_port
        )

        max_steps = env.env.max_episode_length
        scene_results = []

        for ep in range(args.episodes):
            logger.info(f"  Episode {ep+1}/{args.episodes}")
            obs, _ = env.reset()
            obs, _ = env.reset()

            actions_completed = 0
            pred_chunk = None
            ep_start = time.time()

            with torch.no_grad():
                for step in range(max_steps):
                    # Extract observations
                    right_img = obs["policy"]["external_cam"][0].cpu().numpy()
                    wrist_img = obs["policy"]["wrist_cam"][0].cpu().numpy()
                    joint_pos = obs["policy"]["arm_joint_pos"].cpu().numpy().flatten()
                    gripper_pos = obs["policy"]["gripper_pos"].cpu().numpy().flatten()

                    # Query policy when chunk exhausted
                    if actions_completed == 0 or actions_completed >= args.open_loop_horizon:
                        actions_completed = 0
                        request = {
                            "observation/exterior_image_1_left":
                                image_tools.resize_with_pad(right_img, 224, 224),
                            "observation/wrist_image_left":
                                image_tools.resize_with_pad(wrist_img, 224, 224),
                            "observation/joint_position": joint_pos,
                            "observation/gripper_position": gripper_pos,
                            "prompt": instruction,
                            # Sim step counter for orchestrator
                            # cadence gating (see proxy.py).
                            "__step": step,
                        }
                        # TODO: Add depth and gt_state for grasp tool / GT failure monitor.
                        # Requires adding data_types=["rgb", "depth"] to camera configs
                        # in sim-evals/src/sim_evals/environments/droid_environment.py
                        # and forwarding observation/depth_external, observation/camera_pos,
                        # observation/camera_quat keys (similar to robolab pi0_family.py).
                        response = client.infer(request)
                        pred_chunk = response["actions"]

                        if response.get("orchestrator_flush_actions"):
                            actions_completed = 0

                    action = pred_chunk[actions_completed]
                    actions_completed += 1

                    # Binarize gripper
                    if action[-1] > 0.5:
                        action = np.concatenate([action[:-1], np.ones(1)])
                    else:
                        action = np.concatenate([action[:-1], np.zeros(1)])

                    # Step environment
                    action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                    obs, _, term, trunc, _ = env.step(action_t)

                    if term or trunc:
                        break

            ep_time = time.time() - ep_start
            success = bool(term)
            scene_results.append({
                "episode": ep,
                "success": success,
                "steps": step + 1,
                "duration_s": ep_time,
            })

            if success:
                overall_successes += 1
            overall_episodes += 1

            logger.info(f"    {'SUCCESS' if success else 'FAILURE'} | "
                        f"steps={step+1} | time={ep_time:.1f}s")

        # Scene summary
        n_succ = sum(1 for r in scene_results if r["success"])
        logger.info(f"  Scene {scene_id}: {n_succ}/{len(scene_results)} "
                     f"({100*n_succ/len(scene_results):.1f}%)")

        all_results.append({
            "scene_id": scene_id,
            "instruction": instruction,
            "episodes": scene_results,
            "successes": n_succ,
            "total": len(scene_results),
        })

        env.close()

    # Final summary
    rate = 100 * overall_successes / max(overall_episodes, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS: DROID Sim")
    logger.info(f"  Total: {overall_successes}/{overall_episodes} ({rate:.1f}%)")
    for r in all_results:
        sr = 100 * r["successes"] / r["total"]
        logger.info(f"  Scene {r['scene_id']}: {r['successes']}/{r['total']} "
                     f"({sr:.1f}%) — {r['instruction']}")
    logger.info(f"{'='*60}")

    # Save results
    results_path = log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "benchmark": "droid_sim",
            "total_episodes": overall_episodes,
            "total_successes": overall_successes,
            "success_rate": rate,
            "scenes": all_results,
        }, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
