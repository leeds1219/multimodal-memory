#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified benchmark runner for LIBERO, LIBERO-Plus, and LIBERO-PRO.

Supports all three LIBERO benchmark families through a single CLI, with
per-perturbation-dimension result breakdowns.

Usage examples::

    # Original LIBERO (same as run_benchmark.py)
    python examples/libero/run_extended_benchmarks.py \\
        --benchmark libero --suites libero_10 \\
        --num-trials 50 --proxy-port 8001

    # LIBERO-Plus: all suites, 1 trial per perturbed task
    python examples/libero/run_extended_benchmarks.py \\
        --benchmark libero_plus --suites all \\
        --proxy-port 8001

    # LIBERO-Plus: single category (e.g. language perturbations only)
    python examples/libero/run_extended_benchmarks.py \\
        --benchmark libero_plus --plus-category "Language Instructions" \\
        --proxy-port 8001

    # LIBERO-PRO: all dimensions on libero_goal
    python examples/libero/run_extended_benchmarks.py \\
        --benchmark libero_pro --suites libero_goal \\
        --proxy-port 8001

    # LIBERO-PRO: position perturbation only (the hardest dimension)
    python examples/libero/run_extended_benchmarks.py \\
        --benchmark libero_pro --pro-dimension position \\
        --proxy-port 8001

Prerequisites:
    # Terminal 1: Policy server (same for all benchmarks)
    cd ~/openpi && uv run scripts/serve_policy.py --env LIBERO

    # Terminal 2: Orchestrator proxy
    cd ~/vlm-orchestrator
    vlm-orchestrator --env libero --mode subgoal_scene_edit \\
        --vla-port 8000 --port 8001

    # For LIBERO-Plus: install LIBERO-plus repo as the libero package
    # For LIBERO-PRO: install LIBERO-PRO repo as the libero package
    # Both are drop-in replacements for the original LIBERO package.

Notes on LIBERO-Plus vs LIBERO-PRO installation:
    Both benchmarks extend the base LIBERO package. Only ONE can be installed
    at a time (they both provide the ``libero`` Python package). Switch between
    them by doing:
        cd ~/LIBERO-plus && pip install -e .   # for LIBERO-Plus
        cd ~/LIBERO-PRO && pip install -e .    # for LIBERO-PRO
    The eval client code is the same — only the benchmark registration and
    task maps differ.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import subprocess
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

# Import benchmark metadata module
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from vlm_orchestrator.aux_benchmarks.libero_specs import (
    PLUS_CATEGORIES,
    PLUS_SUITES,
    PRO_BASE_SUITES,
    PlusResults,
    ProResults,
    aggregate_plus_results,
    aggregate_pro_results,
    format_plus_report,
    format_pro_report,
    get_all_pro_suites,
    get_pro_suite_names,
    infer_pro_dimension_from_suite,
    load_plus_classification,
    make_benchmark_spec,
)


# ──────────────────────────────────────────────────────────────────
# Max steps per suite
# ──────────────────────────────────────────────────────────────────

def get_max_steps(suite: str) -> int:
    """Episode step limits. Extended suites inherit from their base suite."""
    base_limits = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    # Direct match
    if suite in base_limits:
        return base_limits[suite]
    # Try to match base suite prefix
    for base, limit in base_limits.items():
        if suite.startswith(base):
            return limit
    # Default for unknown suites
    return 400


# ──────────────────────────────────────────────────────────────────
# Suite runner
# ──────────────────────────────────────────────────────────────────

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
        str(pathlib.Path(__file__).parent / "libero_eval_client.py"),
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
    logger.info(f"Running suite: {suite} ({num_trials} trials/task)")
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


# ──────────────────────────────────────────────────────────────────
# Summary formatters
# ──────────────────────────────────────────────────────────────────

def format_libero_summary(results: dict[str, dict | None]) -> str:
    """Format standard LIBERO results table."""
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
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for LIBERO, LIBERO-Plus, LIBERO-PRO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--benchmark",
        choices=["libero", "libero_plus", "libero_pro"],
        default="libero",
        help="Which benchmark family to run (default: libero)",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=None,
        help="Explicit suite names. Use 'all' for all suites in the benchmark.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Trials per task. Defaults: libero=50, libero_plus=1, libero_pro=50",
    )

    # LIBERO-Plus specific
    parser.add_argument(
        "--plus-category",
        default=None,
        choices=PLUS_CATEGORIES,
        help="LIBERO-Plus: filter to a single perturbation category",
    )
    parser.add_argument(
        "--plus-difficulty",
        type=int,
        nargs="+",
        default=None,
        help="LIBERO-Plus: filter to specific difficulty levels (1-5)",
    )

    # LIBERO-PRO specific
    parser.add_argument(
        "--pro-dimension",
        default=None,
        choices=["object", "position", "semantic", "task", "environment"],
        help="LIBERO-PRO: run only a single perturbation dimension",
    )
    parser.add_argument(
        "--pro-base-suite",
        default=None,
        choices=PRO_BASE_SUITES,
        help="LIBERO-PRO: run only a single base suite (default: all 4)",
    )

    # Common options
    parser.add_argument(
        "--proxy-host", default="0.0.0.0",
        help="Orchestrator proxy host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--proxy-port", type=int, default=8001,
        help="Orchestrator proxy port (default: 8001)",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Output directory (default: results/libero/<benchmark>_<timestamp>)",
    )
    parser.add_argument(
        "--enable-depth", action="store_true",
        help="Enable depth rendering for grasp tool support",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Random seed (default: 7)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Resolve suites ──
    benchmark = args.benchmark
    suites: list[str]

    if args.suites and "all" in args.suites:
        args.suites = None  # let make_benchmark_spec pick defaults

    if benchmark == "libero":
        spec = make_benchmark_spec(
            "libero",
            suites=args.suites,
            num_trials=args.num_trials or 50,
        )
        suites = spec.suites
        num_trials = spec.num_trials_per_task

    elif benchmark == "libero_plus":
        spec = make_benchmark_spec(
            "libero_plus",
            suites=args.suites,
            num_trials=args.num_trials or 1,
            plus_categories=[args.plus_category] if args.plus_category else None,
            plus_difficulty=args.plus_difficulty,
        )
        suites = spec.suites
        num_trials = spec.num_trials_per_task

    elif benchmark == "libero_pro":
        # Build suite list from dimensions and base suites
        if args.suites:
            suites = args.suites
        else:
            bases = [args.pro_base_suite] if args.pro_base_suite else PRO_BASE_SUITES
            if args.pro_dimension:
                suites = []
                for base in bases:
                    suites.extend(get_pro_suite_names(base, args.pro_dimension))
            else:
                suites = []
                for base in bases:
                    suites.extend(get_pro_suite_names(base))
            # Optionally add base suites for reference
            if not args.pro_dimension:
                suites = list(bases) + suites

        num_trials = args.num_trials or 50

    else:
        raise SystemExit(f"Unknown benchmark: {benchmark}")

    # ── Resolve log dir ──
    if args.log_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"results/libero/{benchmark}_{ts}"
    log_dir = args.log_dir
    video_dir = str(pathlib.Path(log_dir) / "videos")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Suites ({len(suites)}): {suites}")
    logger.info(f"Trials per task: {num_trials}")
    logger.info(f"Proxy: {args.proxy_host}:{args.proxy_port}")

    # ── Run each suite ──
    all_results: dict[str, dict | None] = {}
    for suite in suites:
        data = run_suite(
            suite,
            proxy_host=args.proxy_host,
            proxy_port=args.proxy_port,
            num_trials=num_trials,
            log_dir=log_dir,
            video_dir=video_dir,
            enable_depth=args.enable_depth,
            seed=args.seed,
        )
        all_results[suite] = data

    # ── Generate reports ──
    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if benchmark == "libero":
        summary = format_libero_summary(all_results)
        print(summary)
        report_text = summary

    elif benchmark == "libero_plus":
        # Generate per-dimension breakdown
        # Collect all episode results from the per-suite results.json files
        all_episodes: list[dict] = []
        for suite_name, suite_data in valid_results.items():
            if suite_data and "episodes" in suite_data:
                for ep in suite_data["episodes"]:
                    # Extract task_name from the episode data
                    desc = ep.get("task_description", "")
                    ep_enriched = dict(ep)
                    ep_enriched["suite"] = suite_name
                    # For LIBERO-Plus, the task_name in episode data should
                    # match names in task_classification.json
                    if "task_name" not in ep_enriched:
                        ep_enriched["task_name"] = desc.replace(" ", "_")
                    all_episodes.append(ep_enriched)

        try:
            classification = load_plus_classification()
            plus_results = aggregate_plus_results(all_episodes, classification)
            report_text = format_plus_report(plus_results)
            print(report_text)
        except FileNotFoundError as e:
            logger.warning(f"Cannot generate per-dimension report: {e}")
            summary = format_libero_summary(all_results)
            print(summary)
            report_text = summary

    elif benchmark == "libero_pro":
        pro_results = aggregate_pro_results(valid_results)
        report_text = format_pro_report(pro_results)
        print(report_text)

    # ── Save outputs ──
    report_path = pathlib.Path(log_dir) / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_path}")

    combined_path = pathlib.Path(log_dir) / "benchmark_results.json"
    with open(combined_path, "w") as f:
        json.dump(
            {
                "benchmark": benchmark,
                "suites": suites,
                "num_trials": num_trials,
                "results": valid_results,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
