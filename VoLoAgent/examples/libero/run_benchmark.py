#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automated LIBERO benchmark runner for the VLM-orchestrator.

Evaluates one or more orchestrator modes across LIBERO task suites and
generates a summary table.

Usage::

    # Prerequisite: start policy server + orchestrator proxy separately.
    # Terminal 1:
    cd ~/openpi && uv run scripts/serve_policy.py --env LIBERO

    # Terminal 2:
    cd ~/vlm-orchestrator
    vlm-orchestrator --env libero --mode subgoal_scene_edit \
        --vla-port 8000 --port 8001

    # Terminal 3: run this benchmark
    python examples/libero/run_benchmark.py \
        --suites libero_10 \
        --num-trials 50 \
        --proxy-port 8001

    # Or run all suites:
    python examples/libero/run_benchmark.py \
        --suites libero_spatial libero_object libero_goal libero_10
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

ALL_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]

# Extended benchmark suites (LIBERO-Plus and LIBERO-PRO).
# These use the same eval client and proxy; only the installed libero
# package differs.  See run_extended_benchmarks.py for full support.
LIBERO_MEM_SUITE = "libero_mem"  # 10 memory tasks, subgoal completion metric
LIBERO_PLUS_SUITES = ALL_SUITES  # same names, but ~2400 tasks/suite
LIBERO_PRO_SUITES = [
    # Per-dimension suites (4 bases × 5 perturbation dims = 20 suites)
    f"{base}_{dim}"
    for base in ALL_SUITES
    for dim in ["lan", "object", "swap", "task", "env"]
]


def run_suite(
    suite: str,
    *,
    proxy_host: str,
    proxy_port: int,
    num_trials: int,
    log_dir: str,
    video_dir: str,
    enable_depth: bool,
    seed: int,
) -> dict | None:
    """Run the LIBERO eval client for one task suite."""
    suite_log_dir = str(pathlib.Path(log_dir) / suite)
    suite_video_dir = str(pathlib.Path(video_dir) / suite)

    cmd = [
        sys.executable,
        "examples/libero/libero_eval_client.py",
        "--host", proxy_host,
        "--port", str(proxy_port),
        "--task-suite-name", suite,
        "--num-trials-per-task", str(num_trials),
        "--log-dir", suite_log_dir,
        "--video-out-path", suite_video_dir,
        "--seed", str(seed),
    ]
    if enable_depth:
        cmd.append("--enable-depth")

    logger.info(f"\n{'='*60}")
    logger.info(f"Running suite: {suite}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Suite {suite} failed with exit code {e.returncode}")
        return None
    except FileNotFoundError:
        logger.error(
            "Could not find libero_eval_client.py. "
            "Run from the vlm-orchestrator root directory."
        )
        return None

    elapsed = time.time() - t0

    # Read results
    results_path = pathlib.Path(suite_log_dir) / "results.json"
    if not results_path.exists():
        logger.warning(f"No results.json found for {suite}")
        return None

    with open(results_path) as f:
        data = json.load(f)

    data["elapsed_s"] = elapsed
    return data


def print_summary(results: dict[str, dict]) -> str:
    """Format a results table similar to openpi's published benchmarks."""
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("LIBERO BENCHMARK RESULTS")
    lines.append("=" * 72)
    lines.append(
        f"{'Suite':<20} {'Episodes':>10} {'Successes':>10} "
        f"{'Rate':>8} {'Time':>10}"
    )
    lines.append("-" * 72)

    total_ep = 0
    total_succ = 0
    for suite, data in results.items():
        if data is None:
            lines.append(f"{suite:<20} {'FAILED':>10}")
            continue
        ep = data["total_episodes"]
        succ = data["total_successes"]
        rate = data["success_rate"]
        elapsed = data.get("elapsed_s", 0)
        total_ep += ep
        total_succ += succ
        lines.append(
            f"{suite:<20} {ep:>10} {succ:>10} "
            f"{rate:>7.1f}% {elapsed:>9.0f}s"
        )

    lines.append("-" * 72)
    if total_ep > 0:
        avg_rate = total_succ / total_ep * 100
        lines.append(
            f"{'AVERAGE':<20} {total_ep:>10} {total_succ:>10} "
            f"{avg_rate:>7.1f}%"
        )
    lines.append("=" * 72)

    summary = "\n".join(lines)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Automated LIBERO benchmark runner"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_10"],
        choices=ALL_SUITES + ["all"],
        help="Task suites to evaluate (default: libero_10)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Trials per task (default: 50)",
    )
    parser.add_argument(
        "--proxy-host",
        default="0.0.0.0",
        help="Orchestrator proxy host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=8001,
        help="Orchestrator proxy port (default: 8001)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Output directory (default: results/libero/<timestamp>)",
    )
    parser.add_argument(
        "--enable-depth",
        action="store_true",
        help="Enable depth rendering for grasp tool support",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Resolve suites
    suites = args.suites
    if "all" in suites:
        suites = ALL_SUITES

    # Resolve log dir
    if args.log_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"results/libero/benchmark_{ts}"
    log_dir = args.log_dir
    video_dir = str(pathlib.Path(log_dir) / "videos")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Benchmark log dir: {log_dir}")
    logger.info(f"Suites: {suites}")
    logger.info(f"Trials per task: {args.num_trials}")
    logger.info(f"Proxy: {args.proxy_host}:{args.proxy_port}")

    # Run each suite
    results: dict[str, dict | None] = {}
    for suite in suites:
        data = run_suite(
            suite,
            proxy_host=args.proxy_host,
            proxy_port=args.proxy_port,
            num_trials=args.num_trials,
            log_dir=log_dir,
            video_dir=video_dir,
            enable_depth=args.enable_depth,
            seed=args.seed,
        )
        results[suite] = data

    # Print and save summary
    summary = print_summary(results)
    print(summary)

    summary_path = pathlib.Path(log_dir) / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info(f"Summary saved to {summary_path}")

    # Save combined results JSON
    combined_path = pathlib.Path(log_dir) / "benchmark_results.json"
    with open(combined_path, "w") as f:
        json.dump(
            {
                "suites": suites,
                "num_trials": args.num_trials,
                "results": {
                    k: v for k, v in results.items() if v is not None
                },
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
