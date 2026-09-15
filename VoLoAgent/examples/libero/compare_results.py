#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare results from two LIBERO evaluation runs.

Reads results.json files from two mode directories and generates a
side-by-side comparison report.

Usage::

    python examples/libero/compare_results.py \
        --mode-a-dir results/libero/comparison_*/passthrough \
        --mode-a-name passthrough \
        --mode-b-dir results/libero/comparison_*/subgoal_gt_grasp \
        --mode-b-name "subgoal+gt+grasp" \
        --output results/libero/comparison_*/COMPARISON.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load_suite_results(mode_dir: pathlib.Path) -> dict[str, dict]:
    """Load per-suite results.json files from a mode directory."""
    results = {}
    for suite in SUITES:
        results_path = mode_dir / suite / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results[suite] = json.load(f)
        else:
            # Try direct path
            for p in mode_dir.glob(f"**/results.json"):
                if suite in str(p):
                    with open(p) as f:
                        results[suite] = json.load(f)
                    break
    return results


def per_task_breakdown(suite_data: dict) -> dict[str, dict]:
    """Extract per-task success rates from episodes list."""
    tasks: dict[str, list[bool]] = {}
    for ep in suite_data.get("episodes", []):
        desc = ep.get("task_description", "unknown")
        success = bool(ep.get("success", False))
        tasks.setdefault(desc, []).append(success)

    return {
        desc: {
            "episodes": len(outcomes),
            "successes": sum(outcomes),
            "rate": sum(outcomes) / len(outcomes) * 100 if outcomes else 0.0,
        }
        for desc, outcomes in tasks.items()
    }


def generate_report(
    name_a: str,
    results_a: dict[str, dict],
    name_b: str,
    results_b: dict[str, dict],
) -> str:
    """Generate a markdown comparison report."""
    lines = []
    lines.append(f"# LIBERO Evaluation Comparison")
    lines.append("")
    lines.append(f"**Mode A**: `{name_a}`  ")
    lines.append(f"**Mode B**: `{name_b}`  ")
    lines.append("")

    # ── Suite-level summary ──
    lines.append("## Suite-Level Results")
    lines.append("")
    lines.append(
        f"| {'Suite':<20} | {name_a + ' Rate':>18} | {name_b + ' Rate':>18} | "
        f"{'Delta':>8} | {'Winner':>10} |"
    )
    lines.append(
        f"|{'-'*22}|{'-'*20}|{'-'*20}|{'-'*10}|{'-'*12}|"
    )

    total_a_ep, total_a_succ = 0, 0
    total_b_ep, total_b_succ = 0, 0

    for suite in SUITES:
        data_a = results_a.get(suite, {})
        data_b = results_b.get(suite, {})
        rate_a = data_a.get("success_rate", 0.0)
        rate_b = data_b.get("success_rate", 0.0)
        ep_a = data_a.get("total_episodes", 0)
        ep_b = data_b.get("total_episodes", 0)
        succ_a = data_a.get("total_successes", 0)
        succ_b = data_b.get("total_successes", 0)

        total_a_ep += ep_a
        total_a_succ += succ_a
        total_b_ep += ep_b
        total_b_succ += succ_b

        delta = rate_b - rate_a
        delta_str = f"{delta:+.1f}pp"
        winner = name_b if delta > 1 else (name_a if delta < -1 else "tie")

        a_str = f"{succ_a}/{ep_a} ({rate_a:.1f}%)" if ep_a else "—"
        b_str = f"{succ_b}/{ep_b} ({rate_b:.1f}%)" if ep_b else "—"

        lines.append(
            f"| {suite:<20} | {a_str:>18} | {b_str:>18} | "
            f"{delta_str:>8} | {winner:>10} |"
        )

    # Totals
    avg_a = total_a_succ / total_a_ep * 100 if total_a_ep else 0
    avg_b = total_b_succ / total_b_ep * 100 if total_b_ep else 0
    delta_avg = avg_b - avg_a
    winner_avg = name_b if delta_avg > 1 else (name_a if delta_avg < -1 else "tie")

    lines.append(
        f"|{'-'*22}|{'-'*20}|{'-'*20}|{'-'*10}|{'-'*12}|"
    )
    a_tot = f"{total_a_succ}/{total_a_ep} ({avg_a:.1f}%)"
    b_tot = f"{total_b_succ}/{total_b_ep} ({avg_b:.1f}%)"
    lines.append(
        f"| {'**AVERAGE**':<20} | {a_tot:>18} | {b_tot:>18} | "
        f"{delta_avg:+.1f}pp{'':>2} | {winner_avg:>10} |"
    )
    lines.append("")

    # ── Per-task breakdown ──
    lines.append("## Per-Task Breakdown")
    lines.append("")
    for suite in SUITES:
        data_a = results_a.get(suite, {})
        data_b = results_b.get(suite, {})
        tasks_a = per_task_breakdown(data_a)
        tasks_b = per_task_breakdown(data_b)

        all_tasks = sorted(set(list(tasks_a.keys()) + list(tasks_b.keys())))
        if not all_tasks:
            continue

        lines.append(f"### {suite}")
        lines.append("")
        lines.append(
            f"| {'Task':<60} | {name_a:>10} | {name_b:>10} | {'Δ':>6} |"
        )
        lines.append(f"|{'-'*62}|{'-'*12}|{'-'*12}|{'-'*8}|")

        for task in all_tasks:
            ta = tasks_a.get(task, {"rate": 0.0, "episodes": 0, "successes": 0})
            tb = tasks_b.get(task, {"rate": 0.0, "episodes": 0, "successes": 0})
            ra = ta["rate"]
            rb = tb["rate"]
            delta = rb - ra

            task_short = task[:58] + ".." if len(task) > 60 else task
            a_cell = f"{ta['successes']}/{ta['episodes']}" if ta["episodes"] else "—"
            b_cell = f"{tb['successes']}/{tb['episodes']}" if tb["episodes"] else "—"
            delta_cell = f"{delta:+.0f}" if (ta["episodes"] and tb["episodes"]) else "—"

            lines.append(
                f"| {task_short:<60} | {a_cell:>10} | {b_cell:>10} | {delta_cell:>6} |"
            )
        lines.append("")

    # ── Analysis ──
    lines.append("## Analysis")
    lines.append("")

    # Find tasks where B improved most
    improvements = []
    regressions = []
    for suite in SUITES:
        tasks_a = per_task_breakdown(results_a.get(suite, {}))
        tasks_b = per_task_breakdown(results_b.get(suite, {}))
        for task in set(list(tasks_a.keys()) + list(tasks_b.keys())):
            ra = tasks_a.get(task, {}).get("rate", 0.0)
            rb = tasks_b.get(task, {}).get("rate", 0.0)
            if rb > ra + 1:
                improvements.append((suite, task, ra, rb, rb - ra))
            elif ra > rb + 1:
                regressions.append((suite, task, ra, rb, ra - rb))

    if improvements:
        improvements.sort(key=lambda x: -x[4])
        lines.append(f"### Tasks where `{name_b}` improved over `{name_a}`")
        lines.append("")
        for suite, task, ra, rb, delta in improvements[:10]:
            lines.append(f"- **{task}** ({suite}): {ra:.0f}% → {rb:.0f}% (+{delta:.0f}pp)")
        lines.append("")

    if regressions:
        regressions.sort(key=lambda x: -x[4])
        lines.append(f"### Tasks where `{name_b}` regressed vs `{name_a}`")
        lines.append("")
        for suite, task, ra, rb, delta in regressions[:10]:
            lines.append(f"- **{task}** ({suite}): {ra:.0f}% → {rb:.0f}% (-{delta:.0f}pp)")
        lines.append("")

    if not improvements and not regressions:
        lines.append("No significant per-task differences detected.")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two LIBERO eval runs")
    parser.add_argument("--mode-a-dir", required=True, help="Path to mode A results")
    parser.add_argument("--mode-a-name", default="mode_a", help="Display name for mode A")
    parser.add_argument("--mode-b-dir", required=True, help="Path to mode B results")
    parser.add_argument("--mode-b-name", default="mode_b", help="Display name for mode B")
    parser.add_argument("--output", default=None, help="Output markdown file")
    args = parser.parse_args()

    dir_a = pathlib.Path(args.mode_a_dir)
    dir_b = pathlib.Path(args.mode_b_dir)

    results_a = load_suite_results(dir_a)
    results_b = load_suite_results(dir_b)

    if not results_a:
        print(f"WARNING: No results found in {dir_a}", file=sys.stderr)
    if not results_b:
        print(f"WARNING: No results found in {dir_b}", file=sys.stderr)

    report = generate_report(args.mode_a_name, results_a, args.mode_b_name, results_b)
    print(report)

    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\nReport saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
