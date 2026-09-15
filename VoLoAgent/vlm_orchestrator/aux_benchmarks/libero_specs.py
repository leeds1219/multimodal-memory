# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support for extended LIBERO benchmarks: LIBERO-Plus and LIBERO-PRO.

This module provides unified task enumeration, perturbation classification,
and result aggregation for three LIBERO benchmark families:

  - **LIBERO** (original): 4 standard suites (spatial, object, goal, 10)
  - **LIBERO-Plus**: 10,030 tasks across 7 perturbation dimensions (drop-in
    replacement, same API as LIBERO)
  - **LIBERO-PRO**: 5 generalization dimensions with YAML-driven perturbation
    config, combinable flags, and dynamic temp-suite generation

Architecture note:
  Both extended benchmarks use the same MuJoCo/robosuite backend as original
  LIBERO.  The existing ``--env libero`` CLI preset, eval client, and all 6
  orchestrator strategies work with all three benchmarks.  This module handles
  only the *benchmark metadata layer* — task enumeration, perturbation
  categorisation, and per-dimension result aggregation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

# Default paths (can be overridden via constructor args)
LIBERO_PLUS_ROOT = Path(os.path.expanduser("~/LIBERO-plus"))
LIBERO_PRO_ROOT = Path(os.path.expanduser("~/LIBERO-PRO"))

# LIBERO-Plus perturbation dimensions
PLUS_CATEGORIES = [
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
]

# LIBERO-Plus base suites (each contains ~2400+ perturbed tasks)
PLUS_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]

# LIBERO-PRO perturbation dimensions and their flag names
PRO_DIMENSIONS = {
    "object": "use_object",
    "position": "use_swap",     # "swap" = spatial position swap
    "semantic": "use_language",  # language paraphrase
    "task": "use_task",
    "environment": "use_environment",
}

# LIBERO-PRO pre-generated suite naming convention:
#   libero_{base}_{perturbation}
# where base ∈ {spatial, object, goal, 10}
# and perturbation ∈ {lan, object, swap, task, env, temp, ...}
PRO_BASE_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# Map from PRO dimension name → suite suffix in libero_suite_task_map
PRO_DIM_TO_SUFFIX = {
    "object": "object",     # libero_{base}_object
    "position": "swap",     # libero_{base}_swap
    "semantic": "lan",      # libero_{base}_lan
    "task": "task",         # libero_{base}_task
    "environment": "env",   # libero_{base}_env
}

# Also available: _object_ood, _relation_ood, _semantic_ood, _temp variants


# ═══════════════════════════════════════════════════════════════════
#  LIBERO-Plus Support
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PlusTask:
    """A single LIBERO-Plus task with perturbation metadata."""
    id: int
    name: str
    category: str        # perturbation dimension
    difficulty_level: int
    suite: str           # e.g. "libero_spatial"
    language: str = ""   # task language instruction


@dataclass
class PlusClassification:
    """Parsed task_classification.json for LIBERO-Plus."""
    tasks_by_suite: dict[str, list[PlusTask]] = field(default_factory=dict)
    _loaded: bool = False

    @property
    def all_tasks(self) -> list[PlusTask]:
        return [t for tasks in self.tasks_by_suite.values() for t in tasks]

    @property
    def total_count(self) -> int:
        return sum(len(t) for t in self.tasks_by_suite.values())

    def tasks_by_category(self, category: str) -> list[PlusTask]:
        """Get all tasks for a specific perturbation category."""
        return [t for t in self.all_tasks if t.category == category]

    def tasks_by_suite_and_category(
        self, suite: str, category: str
    ) -> list[PlusTask]:
        """Get tasks for a specific suite × category combination."""
        return [
            t for t in self.tasks_by_suite.get(suite, [])
            if t.category == category
        ]

    def category_counts(self, suite: str | None = None) -> dict[str, int]:
        """Count tasks per perturbation category, optionally filtered by suite."""
        tasks = self.tasks_by_suite.get(suite, []) if suite else self.all_tasks
        counts: dict[str, int] = {}
        for t in tasks:
            counts[t.category] = counts.get(t.category, 0) + 1
        return counts

    def difficulty_distribution(
        self, suite: str | None = None, category: str | None = None
    ) -> dict[int, int]:
        """Count tasks per difficulty level."""
        tasks = self.all_tasks
        if suite:
            tasks = [t for t in tasks if t.suite == suite]
        if category:
            tasks = [t for t in tasks if t.category == category]
        dist: dict[int, int] = {}
        for t in tasks:
            dist[t.difficulty_level] = dist.get(t.difficulty_level, 0) + 1
        return dict(sorted(dist.items()))


def load_plus_classification(
    root: Path | str = LIBERO_PLUS_ROOT,
) -> PlusClassification:
    """Load and parse LIBERO-Plus task_classification.json.

    Returns a ``PlusClassification`` with tasks indexed by suite and category.
    """
    root = Path(root)
    json_path = root / "libero" / "libero" / "benchmark" / "task_classification.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"LIBERO-Plus task_classification.json not found at {json_path}. "
            f"Clone https://github.com/sylvestf/LIBERO-plus to {root}"
        )

    with open(json_path) as f:
        data = json.load(f)

    classification = PlusClassification()
    for suite_name, tasks_list in data.items():
        suite_tasks = []
        for entry in tasks_list:
            task = PlusTask(
                id=entry["id"],
                name=entry["name"],
                category=entry["category"],
                difficulty_level=entry["difficulty_level"],
                suite=suite_name,
            )
            suite_tasks.append(task)
        classification.tasks_by_suite[suite_name] = suite_tasks

    classification._loaded = True
    logger.info(
        f"Loaded LIBERO-Plus classification: {classification.total_count} tasks "
        f"across {len(classification.tasks_by_suite)} suites"
    )
    return classification


@dataclass
class PlusResults:
    """Aggregated LIBERO-Plus results with per-dimension breakdowns."""
    # Overall
    total_episodes: int = 0
    total_successes: int = 0

    # Per suite
    suite_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per perturbation category (across all suites)
    category_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per suite × category
    suite_category_results: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )

    # Per difficulty level
    difficulty_results: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def overall_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_successes / self.total_episodes * 100


def aggregate_plus_results(
    episode_results: list[dict],
    classification: PlusClassification,
) -> PlusResults:
    """Aggregate per-episode results by LIBERO-Plus perturbation dimensions.

    Parameters
    ----------
    episode_results : list[dict]
        Each dict must have at minimum:
          - ``task_name`` (str): task name matching PlusClassification
          - ``success`` (bool): whether episode succeeded
        Optionally:
          - ``suite`` (str): suite name (if not provided, looked up from classification)
    classification : PlusClassification
        Loaded LIBERO-Plus classification data.

    Returns
    -------
    PlusResults
        Aggregated results broken down by category, suite, difficulty.
    """
    # Build lookup: task_name → PlusTask
    task_lookup: dict[str, PlusTask] = {}
    for task in classification.all_tasks:
        task_lookup[task.name] = task

    results = PlusResults()

    # Accumulator dicts
    suite_acc: dict[str, list[bool]] = {}
    cat_acc: dict[str, list[bool]] = {}
    suite_cat_acc: dict[str, dict[str, list[bool]]] = {}
    diff_acc: dict[int, list[bool]] = {}

    for ep in episode_results:
        task_name = ep.get("task_name", "")
        success = bool(ep.get("success", False))
        results.total_episodes += 1
        if success:
            results.total_successes += 1

        plus_task = task_lookup.get(task_name)
        if plus_task is None:
            continue

        suite = plus_task.suite
        category = plus_task.category
        difficulty = plus_task.difficulty_level

        suite_acc.setdefault(suite, []).append(success)
        cat_acc.setdefault(category, []).append(success)
        suite_cat_acc.setdefault(suite, {}).setdefault(category, []).append(success)
        diff_acc.setdefault(difficulty, []).append(success)

    def _summarize(outcomes: list[bool]) -> dict[str, Any]:
        n = len(outcomes)
        s = sum(outcomes)
        return {"episodes": n, "successes": s, "rate": s / n * 100 if n else 0.0}

    results.suite_results = {k: _summarize(v) for k, v in suite_acc.items()}
    results.category_results = {k: _summarize(v) for k, v in cat_acc.items()}
    results.difficulty_results = {
        k: _summarize(v) for k, v in sorted(diff_acc.items())
    }
    for suite, cats in suite_cat_acc.items():
        results.suite_category_results[suite] = {
            cat: _summarize(outcomes) for cat, outcomes in cats.items()
        }

    return results


def format_plus_report(results: PlusResults) -> str:
    """Format LIBERO-Plus results into a readable markdown report."""
    lines: list[str] = []
    lines.append("# LIBERO-Plus Benchmark Results")
    lines.append("")
    lines.append(
        f"**Overall**: {results.total_successes}/{results.total_episodes} "
        f"({results.overall_rate:.1f}%)"
    )
    lines.append("")

    # Per-category summary table
    lines.append("## Results by Perturbation Dimension")
    lines.append("")
    lines.append(f"| {'Dimension':<25} | {'Episodes':>10} | {'Successes':>10} | {'Rate':>8} |")
    lines.append(f"|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*10}|")
    for cat in PLUS_CATEGORIES:
        data = results.category_results.get(cat, {"episodes": 0, "successes": 0, "rate": 0.0})
        lines.append(
            f"| {cat:<25} | {data['episodes']:>10} | {data['successes']:>10} | "
            f"{data['rate']:>7.1f}% |"
        )
    lines.append("")

    # Per-suite summary
    lines.append("## Results by Suite")
    lines.append("")
    lines.append(f"| {'Suite':<20} | {'Episodes':>10} | {'Successes':>10} | {'Rate':>8} |")
    lines.append(f"|{'-'*22}|{'-'*12}|{'-'*12}|{'-'*10}|")
    for suite in PLUS_SUITES:
        data = results.suite_results.get(suite, {"episodes": 0, "successes": 0, "rate": 0.0})
        lines.append(
            f"| {suite:<20} | {data['episodes']:>10} | {data['successes']:>10} | "
            f"{data['rate']:>7.1f}% |"
        )
    lines.append("")

    # Per difficulty
    if results.difficulty_results:
        lines.append("## Results by Difficulty Level")
        lines.append("")
        lines.append(f"| {'Level':>6} | {'Episodes':>10} | {'Successes':>10} | {'Rate':>8} |")
        lines.append(f"|{'-'*8}|{'-'*12}|{'-'*12}|{'-'*10}|")
        for level, data in sorted(results.difficulty_results.items()):
            lines.append(
                f"| {level:>6} | {data['episodes']:>10} | {data['successes']:>10} | "
                f"{data['rate']:>7.1f}% |"
            )
        lines.append("")

    # Suite × category breakdown
    lines.append("## Suite × Dimension Breakdown")
    lines.append("")
    header = f"| {'Suite':<20} |"
    for cat in PLUS_CATEGORIES:
        short = cat.split()[0][:6]
        header += f" {short:>8} |"
    lines.append(header)
    lines.append("|" + "-" * 22 + "|" + (("-" * 10 + "|") * len(PLUS_CATEGORIES)))
    for suite in PLUS_SUITES:
        row = f"| {suite:<20} |"
        for cat in PLUS_CATEGORIES:
            data = results.suite_category_results.get(suite, {}).get(
                cat, {"rate": 0.0}
            )
            row += f" {data['rate']:>7.1f}% |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  LIBERO-PRO Support
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProPerturbConfig:
    """LIBERO-PRO perturbation configuration (from evaluation_config.yaml)."""
    bddl_files_path: str = ""
    script_path: str = ""
    init_file_dir: str = ""

    use_environment: bool = False
    use_swap: bool = False
    use_object: bool = False
    use_language: bool = False
    use_task: bool = False

    ood_task_configs: dict[str, str] = field(default_factory=dict)

    @property
    def active_dimensions(self) -> list[str]:
        """Return list of active perturbation dimension names."""
        dims = []
        if self.use_object:
            dims.append("object")
        if self.use_swap:
            dims.append("position")
        if self.use_language:
            dims.append("semantic")
        if self.use_task:
            dims.append("task")
        if self.use_environment:
            dims.append("environment")
        return dims

    @property
    def config_tag(self) -> str:
        """Short tag describing the active perturbation combo."""
        if not self.active_dimensions:
            return "original"
        return "+".join(self.active_dimensions)


def load_pro_config(
    config_path: str | Path | None = None,
    root: Path | str = LIBERO_PRO_ROOT,
) -> ProPerturbConfig:
    """Load LIBERO-PRO evaluation_config.yaml.

    Parameters
    ----------
    config_path : str or Path, optional
        Direct path to the YAML config. If None, uses
        ``{root}/evaluation_config.yaml``.
    root : Path or str
        LIBERO-PRO repository root.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for LIBERO-PRO config parsing")

    root = Path(root)
    if config_path is None:
        config_path = root / "evaluation_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"LIBERO-PRO evaluation_config.yaml not found at {config_path}. "
            f"Clone https://github.com/Zxy-MLlab/LIBERO-PRO to {root}"
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Resolve relative paths to absolute based on root
    def _resolve(p: str) -> str:
        if not p:
            return ""
        pp = Path(p)
        if not pp.is_absolute():
            pp = root / pp
        return str(pp)

    config = ProPerturbConfig(
        bddl_files_path=_resolve(raw.get("bddl_files_path", "")),
        script_path=_resolve(raw.get("script_path", "")),
        init_file_dir=_resolve(raw.get("init_file_dir", "")),
        use_environment=bool(raw.get("use_environment", False)),
        use_swap=bool(raw.get("use_swap", False)),
        use_object=bool(raw.get("use_object", False)),
        use_language=bool(raw.get("use_language", False)),
        use_task=bool(raw.get("use_task", False)),
        ood_task_configs={
            k: _resolve(v) for k, v in raw.get("ood_task_configs", {}).items()
        },
    )

    logger.info(
        f"Loaded LIBERO-PRO config: active dimensions = {config.active_dimensions}"
    )
    return config


def get_pro_suite_names(
    base_suite: str = "libero_goal",
    dimension: str | None = None,
) -> list[str]:
    """Get LIBERO-PRO suite name(s) for a base suite and dimension.

    Parameters
    ----------
    base_suite : str
        One of: libero_spatial, libero_object, libero_goal, libero_10
    dimension : str, optional
        One of: object, position, semantic, task, environment.
        If None, returns names for all 5 dimensions.

    Returns
    -------
    list[str]
        Suite names like ``["libero_goal_swap"]``.
    """
    if dimension:
        suffix = PRO_DIM_TO_SUFFIX.get(dimension)
        if suffix is None:
            raise ValueError(
                f"Unknown PRO dimension: {dimension}. "
                f"Options: {list(PRO_DIM_TO_SUFFIX.keys())}"
            )
        return [f"{base_suite}_{suffix}"]

    return [f"{base_suite}_{s}" for s in PRO_DIM_TO_SUFFIX.values()]


def get_all_pro_suites() -> list[str]:
    """Get all 20 standard LIBERO-PRO perturbation suites (4 bases × 5 dims)."""
    suites = []
    for base in PRO_BASE_SUITES:
        for suffix in PRO_DIM_TO_SUFFIX.values():
            suites.append(f"{base}_{suffix}")
    return suites


def infer_pro_dimension_from_suite(suite_name: str) -> str | None:
    """Infer which PRO dimension a suite tests based on its name suffix.

    Returns dimension name (object, position, semantic, task, environment)
    or None if unrecognised.
    """
    suffix_to_dim = {v: k for k, v in PRO_DIM_TO_SUFFIX.items()}
    # Extract suffix after the base suite name
    for base in PRO_BASE_SUITES:
        if suite_name.startswith(base + "_"):
            suffix = suite_name[len(base) + 1:]
            if suffix in suffix_to_dim:
                return suffix_to_dim[suffix]
            # Also handle _object_ood, _relation_ood, _semantic_ood variants
            if suffix.endswith("_ood"):
                prefix = suffix[:-4]
                return suffix_to_dim.get(prefix)
    return None


@dataclass
class ProResults:
    """Aggregated LIBERO-PRO results with per-dimension breakdowns."""
    total_episodes: int = 0
    total_successes: int = 0

    # Per base suite (original LIBERO, for reference)
    base_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per dimension (across all base suites)
    dimension_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per base_suite × dimension
    suite_dimension_results: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )

    @property
    def overall_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_successes / self.total_episodes * 100


def aggregate_pro_results(
    suite_results: dict[str, dict],
) -> ProResults:
    """Aggregate per-suite results by LIBERO-PRO generalization dimensions.

    Parameters
    ----------
    suite_results : dict[str, dict]
        Maps suite name → result dict with at least:
          - ``total_episodes`` (int)
          - ``total_successes`` (int)
          - ``success_rate`` (float)

    Returns
    -------
    ProResults
        Aggregated results broken down by dimension and base suite.
    """
    results = ProResults()
    dim_acc: dict[str, list[tuple[int, int]]] = {}  # dim → [(ep, succ), ...]
    suite_dim_acc: dict[str, dict[str, tuple[int, int]]] = {}

    for suite_name, data in suite_results.items():
        ep = data.get("total_episodes", 0)
        succ = data.get("total_successes", 0)
        results.total_episodes += ep
        results.total_successes += succ

        # Check if it's a base suite (original LIBERO)
        if suite_name in PRO_BASE_SUITES:
            results.base_results[suite_name] = {
                "episodes": ep, "successes": succ,
                "rate": succ / ep * 100 if ep else 0.0,
            }
            continue

        # Infer which dimension this suite tests
        dim = infer_pro_dimension_from_suite(suite_name)
        if dim is None:
            continue

        dim_acc.setdefault(dim, []).append((ep, succ))

        # Figure out base suite
        for base in PRO_BASE_SUITES:
            if suite_name.startswith(base + "_"):
                suite_dim_acc.setdefault(base, {})[dim] = (ep, succ)
                break

    # Aggregate per-dimension
    for dim, entries in dim_acc.items():
        total_ep = sum(e for e, _ in entries)
        total_succ = sum(s for _, s in entries)
        results.dimension_results[dim] = {
            "episodes": total_ep,
            "successes": total_succ,
            "rate": total_succ / total_ep * 100 if total_ep else 0.0,
        }

    # Aggregate per base × dimension
    for base, dims in suite_dim_acc.items():
        results.suite_dimension_results[base] = {}
        for dim, (ep, succ) in dims.items():
            results.suite_dimension_results[base][dim] = {
                "episodes": ep,
                "successes": succ,
                "rate": succ / ep * 100 if ep else 0.0,
            }

    return results


def format_pro_report(results: ProResults) -> str:
    """Format LIBERO-PRO results into a readable markdown report."""
    lines: list[str] = []
    lines.append("# LIBERO-PRO Benchmark Results")
    lines.append("")
    lines.append(
        f"**Overall**: {results.total_successes}/{results.total_episodes} "
        f"({results.overall_rate:.1f}%)"
    )
    lines.append("")

    # Original LIBERO baselines
    if results.base_results:
        lines.append("## Original LIBERO Baseline")
        lines.append("")
        lines.append(f"| {'Suite':<20} | {'Rate':>8} |")
        lines.append(f"|{'-'*22}|{'-'*10}|")
        for base in PRO_BASE_SUITES:
            data = results.base_results.get(base, {"rate": 0.0})
            lines.append(f"| {base:<20} | {data['rate']:>7.1f}% |")
        lines.append("")

    # Per-dimension summary
    dim_order = ["object", "position", "semantic", "task", "environment"]
    lines.append("## Results by Generalization Dimension")
    lines.append("")
    lines.append(f"| {'Dimension':<15} | {'Episodes':>10} | {'Successes':>10} | {'Rate':>8} |")
    lines.append(f"|{'-'*17}|{'-'*12}|{'-'*12}|{'-'*10}|")
    for dim in dim_order:
        data = results.dimension_results.get(dim, {"episodes": 0, "successes": 0, "rate": 0.0})
        lines.append(
            f"| {dim:<15} | {data['episodes']:>10} | {data['successes']:>10} | "
            f"{data['rate']:>7.1f}% |"
        )
    lines.append("")

    # Suite × dimension breakdown
    lines.append("## Suite × Dimension Breakdown")
    lines.append("")
    header = f"| {'Suite':<20} |"
    for dim in dim_order:
        header += f" {dim:>11} |"
    lines.append(header)
    lines.append("|" + "-" * 22 + "|" + (("-" * 13 + "|") * len(dim_order)))
    for base in PRO_BASE_SUITES:
        row = f"| {base:<20} |"
        for dim in dim_order:
            data = results.suite_dimension_results.get(base, {}).get(
                dim, {"rate": 0.0}
            )
            row += f" {data['rate']:>10.1f}% |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  LIBERO-Mem Result Aggregation
# ═══════════════════════════════════════════════════════════════════

# Task difficulty tiers
MEM_TIERS = {
    "easy":   [0, 1],         # T1-T2: 1 subgoal (pick and place)
    "medium": [2, 3, 8, 9],   # T3-T4, T9-T10: 2-3 subgoals
    "hard":   [4, 5, 6, 7],   # T5-T8: 4-7 subgoals (repetition, swap, rotate)
}

MEM_TASK_NAMES = [
    "pick up bowl → plate",                    # T1
    "lift bottle → plate",                      # T2
    "bowl → plate × 3",                         # T3
    "bottle → plate × 3",                       # T4
    "bowl → plate × 5",                         # T5
    "bowl → plate × 7",                         # T6
    "swap 2 bowls (3 steps)",                   # T7
    "rotate 3 bowls (4 steps)",                 # T8
    "cheese → nearest basket → center",         # T9
    "cheese → nearest basket, empty → center",  # T10
]


@dataclass
class MemResults:
    """Aggregated LIBERO-Mem evaluation results."""
    total_episodes: int = 0
    total_successes: int = 0
    avg_subgoal_completion: float = 0.0
    total_subgoals_completed: int = 0
    total_subgoals_possible: int = 0

    # Per-task results
    per_task: list[dict] = field(default_factory=list)

    # Per-tier results
    tier_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def binary_success_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_successes / self.total_episodes * 100

    @property
    def subgoal_rate(self) -> float:
        if self.total_subgoals_possible == 0:
            return 0.0
        return self.total_subgoals_completed / self.total_subgoals_possible * 100


def aggregate_mem_results(results_json: dict) -> MemResults:
    """Aggregate LIBERO-Mem results from a results.json file.

    Parameters
    ----------
    results_json : dict
        The JSON output from ``run_mem_eval.py``.
    """
    mem = MemResults(
        total_episodes=results_json.get("total_episodes", 0),
        total_successes=results_json.get("total_successes", 0),
        avg_subgoal_completion=results_json.get("subgoal_completion_rate", 0.0),
        total_subgoals_completed=results_json.get("total_subgoals_completed", 0),
        total_subgoals_possible=results_json.get("total_subgoals_possible", 0),
    )

    # Process per-task data
    for task_data in results_json.get("per_task", []):
        tid = task_data["task_id"]
        eps = task_data["episodes"]
        n_succ = sum(1 for e in eps if e["success"])
        tiered_rates = [e["tiered_success"] for e in eps]
        avg_scr = sum(tiered_rates) / len(tiered_rates) if tiered_rates else 0.0

        mem.per_task.append({
            "task_id": tid,
            "short_name": MEM_TASK_NAMES[tid] if tid < len(MEM_TASK_NAMES) else f"T{tid+1}",
            "language": task_data.get("language", ""),
            "n_subgoals": task_data.get("n_subgoals", 1),
            "n_episodes": len(eps),
            "n_successes": n_succ,
            "binary_rate": 100 * n_succ / len(eps) if eps else 0,
            "subgoal_completion_rate": avg_scr,
        })

    # Compute tier results
    for tier_name, task_ids in MEM_TIERS.items():
        tier_eps = 0
        tier_succ = 0
        tier_scr_sum = 0.0
        for pt in mem.per_task:
            if pt["task_id"] in task_ids:
                tier_eps += pt["n_episodes"]
                tier_succ += pt["n_successes"]
                tier_scr_sum += pt["subgoal_completion_rate"] * pt["n_episodes"]
        mem.tier_results[tier_name] = {
            "episodes": tier_eps,
            "successes": tier_succ,
            "binary_rate": 100 * tier_succ / tier_eps if tier_eps else 0,
            "avg_scr": tier_scr_sum / tier_eps if tier_eps else 0,
        }

    return mem


def format_mem_results(results: MemResults) -> str:
    """Format LIBERO-Mem results as a markdown report."""
    lines = [
        "# LIBERO-Mem Results\n",
        f"**Episodes**: {results.total_episodes}",
        f"**Binary Success Rate**: {results.binary_success_rate:.1f}%",
        f"**Subgoal Completion Rate**: {results.avg_subgoal_completion:.1%}",
        f"**Total Subgoals**: {results.total_subgoals_completed}"
        f"/{results.total_subgoals_possible}",
        "",
        "## Per-Task Breakdown\n",
        "| Task | SGs | Binary Success | Subgoal Rate |",
        "|------|-----|----------------|--------------|",
    ]
    for pt in results.per_task:
        lines.append(
            f"| T{pt['task_id']+1}: {pt['short_name']:<35s} "
            f"| {pt['n_subgoals']:>3d} "
            f"| {pt['n_successes']:>2d}/{pt['n_episodes']:<2d} "
            f"({pt['binary_rate']:5.1f}%) "
            f"| {pt['subgoal_completion_rate']:>6.1%} |"
        )
    lines.append("")
    lines.append("## Tier Summary\n")
    lines.append("| Tier   | Binary Success | Subgoal Rate |")
    lines.append("|--------|----------------|--------------|")
    for tier_name in ["easy", "medium", "hard"]:
        td = results.tier_results.get(tier_name, {})
        lines.append(
            f"| {tier_name:<6s} "
            f"| {td.get('successes',0):>2d}/{td.get('episodes',0):<3d} "
            f"({td.get('binary_rate',0):5.1f}%) "
            f"| {td.get('avg_scr',0):>6.1%} |"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Unified Benchmark Abstraction
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkSpec:
    """Specification for running a benchmark evaluation."""
    name: str                  # "libero", "libero_plus", "libero_pro"
    suites: list[str]          # suite names to evaluate
    num_trials_per_task: int = 50
    max_steps_override: dict[str, int] = field(default_factory=dict)

    # LIBERO-Plus specific
    plus_categories: list[str] | None = None  # filter by category
    plus_difficulty: list[int] | None = None   # filter by difficulty

    # LIBERO-PRO specific
    pro_config_path: str | None = None
    pro_dimensions: list[str] | None = None    # filter dimensions
    pro_seed: int = 42


def make_benchmark_spec(
    benchmark: str,
    *,
    suites: list[str] | None = None,
    num_trials: int = 50,
    pro_dimensions: list[str] | None = None,
    plus_categories: list[str] | None = None,
    plus_difficulty: list[int] | None = None,
) -> BenchmarkSpec:
    """Create a BenchmarkSpec from high-level parameters.

    Parameters
    ----------
    benchmark : str
        One of "libero", "libero_plus", "libero_pro".
    suites : list[str], optional
        Explicit suite list. If None, uses defaults for the benchmark.
    """
    benchmark = benchmark.lower().replace("-", "_")

    if benchmark == "libero":
        default_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        return BenchmarkSpec(
            name="libero",
            suites=suites or default_suites,
            num_trials_per_task=num_trials,
        )

    elif benchmark == "libero_plus":
        # For LIBERO-Plus, each suite has ~2400 tasks, 1 trial per task
        default_suites = PLUS_SUITES
        return BenchmarkSpec(
            name="libero_plus",
            suites=suites or default_suites,
            num_trials_per_task=1,  # Each perturbed task is unique
            plus_categories=plus_categories,
            plus_difficulty=plus_difficulty,
        )

    elif benchmark == "libero_pro":
        default_suites = get_all_pro_suites()
        if pro_dimensions:
            # Filter to only requested dimensions
            filtered: list[str] = []
            for base in PRO_BASE_SUITES:
                for dim in pro_dimensions:
                    filtered.extend(get_pro_suite_names(base, dim))
            default_suites = filtered

        return BenchmarkSpec(
            name="libero_pro",
            suites=suites or default_suites,
            num_trials_per_task=num_trials,
            pro_dimensions=pro_dimensions,
        )

    elif benchmark == "libero_mem":
        return BenchmarkSpec(
            name="libero_mem",
            suites=suites or ["libero_mem"],
            num_trials_per_task=num_trials if num_trials != 50 else 20,
            max_steps_override={
                # 200 steps per subgoal, with per-task granularity
                # managed by the dedicated run_mem_eval.py script
                "libero_mem": 1400,
            },
        )

    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. "
            f"Options: libero, libero_plus, libero_pro, libero_mem"
        )


# ═══════════════════════════════════════════════════════════════════
#  Orchestrator Strategy × Perturbation Analysis
# ═══════════════════════════════════════════════════════════════════

# Hypothesis matrix: which strategies should help with which perturbations
STRATEGY_PERTURBATION_HYPOTHESES = {
    # LIBERO-Plus dimensions
    "Language Instructions": {
        "relevant_strategies": ["subgoal", "rewrite", "adaptive"],
        "hypothesis": (
            "VLM-based subgoal decomposition may be robust to paraphrased "
            "instructions since the VLM understands semantic equivalence. "
            "Instruction rewriting could normalise varied phrasings back to "
            "the policy's training distribution."
        ),
        "expected_impact": "positive",
    },
    "Camera Viewpoints": {
        "relevant_strategies": ["scene_edit", "subgoal_scene_edit"],
        "hypothesis": (
            "VLM-based progress checking may degrade with shifted viewpoints "
            "since visual grounding is viewpoint-dependent. Scene editing "
            "(GDino highlighting) should still work if the object is visible."
        ),
        "expected_impact": "neutral_to_negative",
    },
    "Objects Layout": {
        "relevant_strategies": ["scene_edit", "subgoal_scene_edit"],
        "hypothesis": (
            "When objects are rearranged, GDino-based visual highlighting can "
            "help the policy attend to the correct object regardless of position. "
            "Subgoal decomposition is layout-agnostic."
        ),
        "expected_impact": "positive",
    },
    "Robot Initial States": {
        "relevant_strategies": ["subgoal", "subgoal_scene_edit"],
        "hypothesis": (
            "Different initial arm poses may cause the policy to fail early in "
            "the trajectory. Failure recovery (grasp tool) could help if the "
            "initial reach fails. Subgoal decomposition is not affected."
        ),
        "expected_impact": "neutral",
    },
    "Background Textures": {
        "relevant_strategies": ["scene_edit", "subgoal_scene_edit"],
        "hypothesis": (
            "Background texture changes can confuse object detection. GDino-based "
            "highlighting can counteract this by making the target object visually "
            "salient regardless of background."
        ),
        "expected_impact": "positive",
    },
    "Light Conditions": {
        "relevant_strategies": ["scene_edit"],
        "hypothesis": (
            "Lighting changes affect both the policy and VLM/GDino detection. "
            "Scene editing may help if GDino can still detect objects under "
            "changed lighting, but extreme darkness will break everything."
        ),
        "expected_impact": "neutral",
    },
    "Sensor Noise": {
        "relevant_strategies": [],
        "hypothesis": (
            "Photometric noise (blur, JPEG artifacts, Gaussian noise) degrades "
            "all vision-based components equally. No orchestration strategy "
            "specifically addresses pixel-level corruption."
        ),
        "expected_impact": "neutral_to_negative",
    },

    # LIBERO-PRO dimensions
    "object (PRO)": {
        "relevant_strategies": ["scene_edit", "subgoal_scene_edit"],
        "hypothesis": (
            "When objects have different visual appearance (color, shape), "
            "GDino can still detect them by text description. Scene editing "
            "highlights the correct object for the policy."
        ),
        "expected_impact": "positive",
    },
    "position (PRO)": {
        "relevant_strategies": ["subgoal_scene_edit", "subgoal"],
        "hypothesis": (
            "Position perturbation causes catastrophic failure in all models "
            "(0% success). Failure recovery with grasp tool is the most promising "
            "approach — when the policy fails to reach the displaced object, "
            "the grasp tool can use GDino+depth to plan a direct approach. "
            "This is the HIGHEST-IMPACT opportunity for the orchestrator."
        ),
        "expected_impact": "high_positive",
    },
    "semantic (PRO)": {
        "relevant_strategies": ["subgoal", "rewrite", "adaptive"],
        "hypothesis": (
            "Paraphrased instructions should be handled well by subgoal "
            "decomposition since the VLM understands paraphrases. Rewriting "
            "could map non-standard phrasings to standard ones."
        ),
        "expected_impact": "positive",
    },
    "task (PRO)": {
        "relevant_strategies": ["subgoal", "subgoal_scene_edit"],
        "hypothesis": (
            "New task logic (different goals, different object interactions) "
            "is the hardest perturbation for any approach. Subgoal decomposition "
            "might help if the VLM can reason about the new task structure."
        ),
        "expected_impact": "uncertain",
    },
    "environment (PRO)": {
        "relevant_strategies": ["scene_edit", "subgoal_scene_edit"],
        "hypothesis": (
            "New environments (different tables, backgrounds, room layouts) are "
            "similar to LIBERO-Plus's Background Textures. Scene editing can "
            "highlight targets regardless of environment."
        ),
        "expected_impact": "positive",
    },
}


def get_experiment_plan(benchmark: str) -> list[dict]:
    """Generate an experiment plan for testing strategies under perturbations.

    Returns a list of experiment configurations to run.
    """
    experiments = []

    strategies = [
        "passthrough",
        "subgoal",
        "scene_edit",
        "subgoal_scene_edit",
        "rewrite",
    ]

    if benchmark == "libero_plus":
        for strategy in strategies:
            for suite in PLUS_SUITES:
                for category in PLUS_CATEGORIES:
                    hyp = STRATEGY_PERTURBATION_HYPOTHESES.get(category, {})
                    relevant = hyp.get("relevant_strategies", [])
                    # Run all strategies but mark relevance
                    experiments.append({
                        "benchmark": "libero_plus",
                        "strategy": strategy,
                        "suite": suite,
                        "category": category,
                        "is_relevant": strategy in relevant,
                        "expected_impact": hyp.get("expected_impact", "unknown"),
                        "hypothesis": hyp.get("hypothesis", ""),
                    })

    elif benchmark == "libero_pro":
        dim_order = ["object", "position", "semantic", "task", "environment"]
        for strategy in strategies:
            for base in PRO_BASE_SUITES:
                for dim in dim_order:
                    pro_key = f"{dim} (PRO)"
                    hyp = STRATEGY_PERTURBATION_HYPOTHESES.get(pro_key, {})
                    relevant = hyp.get("relevant_strategies", [])
                    suite_name = f"{base}_{PRO_DIM_TO_SUFFIX[dim]}"
                    experiments.append({
                        "benchmark": "libero_pro",
                        "strategy": strategy,
                        "suite": suite_name,
                        "dimension": dim,
                        "base_suite": base,
                        "is_relevant": strategy in relevant,
                        "expected_impact": hyp.get("expected_impact", "unknown"),
                        "hypothesis": hyp.get("hypothesis", ""),
                    })

    return experiments
