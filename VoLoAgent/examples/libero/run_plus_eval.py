#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate LIBERO-Plus with sampled tasks per perturbation dimension."""
import argparse, json, os, sys, time, random
import numpy as np, requests

import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

DIM_PATTERNS = {
    "language":    "_language_",
    "noise":       "_level",
    "texture":     "_table_",
    "texture_tb":  "_tb_",
    "layout":      "_add_",
    "light":       "_light_",
}
STEP_LIMITS = {
    "libero_spatial": 220, "libero_object": 280,
    "libero_goal": 300, "libero_10": 520,
}


def get_action(host, port, obs, language, gt_state=None):
    payload = {
        "observation": {
            "image": {"agent_image": obs["agentview_image"].tolist()},
            "state": obs["robot0_eye_in_hand_image"].tolist(),
        },
        "prompt": language,
    }
    if gt_state is not None:
        payload["gt_state"] = gt_state
    resp = requests.post(f"http://{host}:{port}/act", json=payload, timeout=30)
    resp.raise_for_status()
    return np.array(resp.json()["actions"][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--dimension", required=True, choices=list(DIM_PATTERNS.keys()))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--log-dir", default="results/plus")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--enable-gt-state", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    max_steps = STEP_LIMITS.get(args.suite, 300)
    pattern = DIM_PATTERNS[args.dimension]

    bench = get_benchmark(args.suite)(task_order_index=0)
    dim_indices = [i for i in range(bench.n_tasks) if pattern in bench.get_task(i).name]
    print(f"Suite {args.suite}, dimension {args.dimension}: {len(dim_indices)} tasks")

    if len(dim_indices) > args.sample_size:
        sampled = sorted(random.sample(dim_indices, args.sample_size))
    else:
        sampled = dim_indices
    print(f"Sampled {len(sampled)} tasks")

    os.makedirs(args.log_dir, exist_ok=True)
    successes = 0
    total = 0
    results = []

    for idx, task_i in enumerate(sampled):
        task = bench.get_task(task_i)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        try:
            init_states = bench.get_task_init_states(task_i)
        except Exception as e:
            print(f"  [{idx+1}/{len(sampled)}] SKIP init error: {e}")
            continue

        for trial in range(args.num_trials):
            total += 1
            init_idx = trial % len(init_states)
            env = None
            try:
                env = OffScreenRenderEnv(bddl_file_name=bddl,
                                         camera_heights=128, camera_widths=128)
                env.seed(args.seed + task_i + trial)
                obs = env.reset()
                env.set_init_state(init_states[init_idx])
                # Re-get obs after setting init state
                obs, _, _, _ = env.step(np.zeros(7))

                success = False
                for step in range(max_steps):
                    gt_state = None
                    if args.enable_gt_state:
                        gt_state = {
                            "ee_pos": env.sim.data.site_xpos[env.robots[0].eef_site_id].tolist(),
                            "gripper_open": float(obs.get("robot0_gripper_qpos", [0.04])[0] > 0.03),
                        }
                    action = get_action(args.host, args.port, obs, task.language, gt_state)
                    obs, reward, done, info = env.step(action)
                    if done:
                        success = True
                        break

                if success:
                    successes += 1
                results.append({
                    "task": task.name[:80], "language": task.language[:80],
                    "trial": trial, "success": success, "steps": step + 1,
                })
                status = "✅" if success else "❌"
                print(f"  [{idx+1}/{len(sampled)}] t{trial} {status} steps={step+1:3d} | {task.language[:60]}")
                env.close()
            except Exception as e:
                print(f"  [{idx+1}/{len(sampled)}] t{trial} ERROR: {str(e)[:80]}")
                results.append({
                    "task": task.name[:80], "trial": trial,
                    "success": False, "error": str(e)[:200],
                })
                if env:
                    try: env.close()
                    except: pass

    rate = successes / total * 100 if total > 0 else 0
    print(f"\nFINAL RESULTS: {args.suite} / {args.dimension}")
    print(f"  Total: {successes}/{total} ({rate:.1f}%)")

    with open(os.path.join(args.log_dir, "results.json"), "w") as f:
        json.dump({
            "suite": args.suite, "dimension": args.dimension,
            "sample_size": len(sampled), "total_episodes": total,
            "successes": successes, "success_rate": rate, "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
