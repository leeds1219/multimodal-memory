#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe valid random_init seeds for VLABench composite tasks.

For each registered composite task, find the first seed in
[0, phase1_max) that yields a stable initial state.  Validation =
``env.reset()`` succeeds AND ``--n-dummy-steps`` "hold-current-pose"
actions don't trigger ``PhysicsError``.  Tasks that fail phase 1 are
deferred to phase 2 (seeds [phase1_max, phase2_max)).  Tasks that
fail phase 2 are skipped with ``"seed": null``.

Output: ``data/vlabench_composite_seeds.json`` — per-task seed for
deterministic loading, designed so the same init can be reused across
passthrough vs. orchestrated runs of the same composite suite.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

# VLABench env vars must be set before importing VLABench
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("VLABENCH_ROOT", os.path.expanduser("~/VLABench/VLABench"))

# noqa imports — modules execute side-effecting registers
import VLABench.tasks  # noqa: F401
import VLABench.robots  # noqa: F401
from VLABench.envs import load_env  # noqa: E402
from dm_control.rl.control import PhysicsError  # noqa: E402


COMPOSITE_TASKS = [
    "assemble_hammer",
    "book_rearrange",
    "cluster_billiards",
    "cluster_book",
    "cluster_dessert",
    "cluster_drink",
    "cluster_ingredients",
    "cluster_toy",
    "complex_seesaw_use",
    "cook_dishes",
    "cool_drink",
    "density_qa",
    "find_fruit_to_make_juice",
    "find_unseen_object",
    "friction_qa",
    "get_coffee",
    "get_coffee_with_milk",
    "get_coffee_with_sugar",
    "hammer_loose_nail",
    "hammer_nail_and_hang_picture",
    "hang_picture_on_specific_nail",
    "heat_food",
    "insert_bloom_flower",
    "insert_power_cord_to_make_juice",
    "magnetism_qa",
    "make_juice",
    "play_mahjong",
    "play_math_game",
    "play_snooker",
    "plug_cord_and_heat_food",
    "put_box_on_painting",
    "rearrange_tube",
    "reflection_qa",
    "replace_wilted_flower",
    "select_specific_type_book",
    "set_dining_chopstick",
    "set_dining_chopstick_left_hand",
    "set_dining_left_hand",
    "set_dining_table",
    "set_study_table",
    "simple_cuestick_use",
    "simple_seesaw_use",
    "speed_of_sound_qa",
    "store_food",
    "take_chemistry_experiment",
    "take_out_cool_drink",
    "texas_holdem",
    "texas_holdem_explore",
    "thermal_expansion_qa",
    "weight_qa",
]


def _hold_pose_action(env) -> np.ndarray:
    """9-D action that holds the robot at its current configuration.

    Matches the action format vlabench_eval_client expects:
    [7 arm joint targets, 2 gripper finger positions].
    """
    qpos = np.asarray(env.task.robot.get_qpos(env.physics)).reshape(-1)
    obs = env.get_observation(require_pcd=False)
    gripper_state = float(np.asarray(obs["ee_state"]).reshape(-1)[-1])
    gripper_action = (
        np.ones(2) * 0.04 if gripper_state >= 0.1 else np.zeros(2)
    )
    return np.concatenate([qpos[:7], gripper_action])


def validate_seed(task_name: str, seed: int, n_dummy_steps: int) -> bool:
    """Returns True iff (load_env, reset, n dummy hold-pose steps) all
    succeed without PhysicsError or other exception."""
    env = None
    try:
        np.random.seed(seed)
        random.seed(seed)
        env = load_env(task_name, random_init=True, run_mode="eval")
        env.reset()
        for _ in range(n_dummy_steps):
            action = _hold_pose_action(env)
            env.step(action)
        return True
    except PhysicsError:
        return False
    except Exception:
        # Don't print full traceback for every fail — too noisy. The
        # caller logs aggregate failures.
        return False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def probe_phase(
    task_name: str,
    seed_start: int,
    seed_end: int,
    n_dummy_steps: int,
) -> tuple[int | None, int]:
    """Iterate seeds in [seed_start, seed_end). Return (first_valid_seed,
    attempts_used). first_valid_seed is None if no seed worked."""
    for s in range(seed_start, seed_end):
        if validate_seed(task_name, s, n_dummy_steps):
            return s, s - seed_start + 1
    return None, seed_end - seed_start


def get_vlabench_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.expanduser("~/VLABench"),
             "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/vlabench_composite_seeds.json",
        help="Output JSON path",
    )
    parser.add_argument("--phase1-max", type=int, default=50,
                        help="Phase 1: seeds [0, phase1_max)")
    parser.add_argument("--phase2-max", type=int, default=500,
                        help="Phase 2: seeds [phase1_max, phase2_max)")
    parser.add_argument("--n-dummy-steps", type=int, default=5,
                        help="Hold-pose steps after reset to validate")
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help=f"Specific tasks to probe (default: all {len(COMPOSITE_TASKS)})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip tasks already populated with a non-null seed",
    )
    args = parser.parse_args()

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        if "tasks" not in data:
            data["tasks"] = {}
    else:
        data = {"_meta": {}, "tasks": {}}

    data["_meta"] = {
        "vlabench_rev": get_vlabench_rev(),
        "mujoco_version": "3.2.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase1_max": args.phase1_max,
        "phase2_max": args.phase2_max,
        "n_dummy_steps": args.n_dummy_steps,
        "validation": f"reset_plus_{args.n_dummy_steps}_dummy_steps",
    }

    tasks = args.tasks if args.tasks else COMPOSITE_TASKS
    print(f"Probing {len(tasks)} tasks. Output → {out_path}")
    print()

    # Phase 1
    print(f"=== Phase 1: seeds 0..{args.phase1_max - 1} ===")
    deferred = []
    for i, task in enumerate(tasks):
        existing = data["tasks"].get(task, {})
        if args.resume and existing.get("seed") is not None:
            print(f"  [{i+1}/{len(tasks)}] {task}: SKIP (already "
                  f"seed={existing['seed']})")
            continue
        t0 = time.time()
        seed, attempts = probe_phase(
            task, 0, args.phase1_max, args.n_dummy_steps,
        )
        elapsed = time.time() - t0
        if seed is not None:
            data["tasks"][task] = {
                "seed": seed,
                "phase": 1,
                "attempts": attempts,
                "probe_s": round(elapsed, 1),
            }
            print(f"  [{i+1}/{len(tasks)}] {task}: seed={seed} "
                  f"(attempts={attempts}, {elapsed:.1f}s)")
        else:
            data["tasks"][task] = {
                "seed": None,
                "phase": "phase1_failed",
                "attempts": attempts,
                "probe_s": round(elapsed, 1),
            }
            deferred.append(task)
            print(f"  [{i+1}/{len(tasks)}] {task}: PHASE1_FAILED "
                  f"(no seed in {attempts}, {elapsed:.1f}s)")
        # Persist after each task (resumable on crash / interrupt)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        sys.stdout.flush()

    # Phase 2 — deferred tasks only
    if deferred:
        print()
        print(f"=== Phase 2: seeds {args.phase1_max}..{args.phase2_max - 1} "
              f"for {len(deferred)} deferred tasks ===")
        for i, task in enumerate(deferred):
            t0 = time.time()
            seed, attempts = probe_phase(
                task, args.phase1_max, args.phase2_max, args.n_dummy_steps,
            )
            elapsed = time.time() - t0
            prev = data["tasks"][task]
            total_attempts = prev["attempts"] + attempts
            total_s = round(prev["probe_s"] + elapsed, 1)
            if seed is not None:
                data["tasks"][task] = {
                    "seed": seed,
                    "phase": 2,
                    "attempts": total_attempts,
                    "probe_s": total_s,
                }
                print(f"  [{i+1}/{len(deferred)}] {task}: seed={seed} "
                      f"(phase2 attempts={attempts}, total={total_attempts}, "
                      f"{elapsed:.1f}s)")
            else:
                data["tasks"][task] = {
                    "seed": None,
                    "phase": "skipped",
                    "attempts": total_attempts,
                    "probe_s": total_s,
                    "reason": (
                        f"no valid seed in {total_attempts} attempts "
                        f"(phase1={prev['attempts']}, phase2={attempts})"
                    ),
                }
                print(f"  [{i+1}/{len(deferred)}] {task}: SKIPPED "
                      f"(no seed in {total_attempts} attempts)")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            sys.stdout.flush()

    # Final summary
    n_valid = sum(
        1 for v in data["tasks"].values() if v.get("seed") is not None
    )
    n_skipped = sum(
        1 for v in data["tasks"].values() if v.get("seed") is None
    )
    print()
    print("=== SUMMARY ===")
    print(f"  Valid seeds: {n_valid}/{len(data['tasks'])}")
    print(f"  Skipped: {n_skipped}")
    if n_skipped:
        skipped_names = [
            t for t, v in data["tasks"].items() if v.get("seed") is None
        ]
        print(f"  Skipped tasks: {sorted(skipped_names)}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
