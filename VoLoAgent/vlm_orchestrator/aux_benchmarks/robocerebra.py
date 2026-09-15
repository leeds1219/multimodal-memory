# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboCerebra benchmark loader for our LIBERO eval client.

RoboCerebra (https://github.com/qiuboxiang/RoboCerebra, NeurIPS 2025)
is built on the LIBERO simulation platform but does NOT register a
``robocerebra`` suite via libero's ``@register_benchmark`` machinery —
their upstream eval (``evaluation/utils.py:get_task_directories``)
discovers tasks by scanning a directory tree at ``ROBOCEREBRA_BENCH_ROOT``:

    {root}/
        Ideal/
            case1/{*.bddl, demo.hdf5, goal.json, task_description.{json,txt}, ...}
            case2/...
        Memory_Exploration/...
        Memory_Execution/...
        Mix/...
        Observation_Mismatching/...
        init_files/
            Ideal/
                case1.init  case2.init  ...
            ...

Per their ``utils.py``, three task types share the ``Ideal/`` BDDL
directory (since the BDDL itself is identical, only the runtime
disturbance differs):

    use_ideal_files = {"Ideal", "Observation_Mismatching", "Random_Disturbance"}

This module wraps that scan in a libero-style ``Benchmark`` so our
existing ``examples/libero/libero_eval_client.py`` works unchanged with
``--task-suite-name robocerebra``.

Output convention: each discovered ``(task_type, case_dir)`` pair becomes
one task with ``name = f"{task_type}_{case_dir.name}"``, e.g.
``Memory_Exploration_case3``.  This makes ``--task-filter
"Memory_Exploration"`` a free task-type filter.
"""

from __future__ import annotations

import logging
import os
from collections import namedtuple
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# Six task types that together comprise the RoboCerebra eval set.
DEFAULT_TASK_TYPES = (
    "Ideal",
    "Memory_Exploration",
    "Memory_Execution",
    "Observation_Mismatching",
    "Random_Disturbance",
    "Mix",
)

# Task types that reuse Ideal/'s BDDL directory at runtime
# (mirrors RoboCerebra/evaluation/utils.py).
_USE_IDEAL_BDDL = {"Ideal", "Observation_Mismatching", "Random_Disturbance"}

# RoboCerebra's eval allocates this many sim steps per subgoal "segment"
# (matches ``cfg.switch_steps`` default in their config.py).  A task with
# N segments runs for ``_SWITCH_STEPS_DEFAULT × N`` steps.
_SWITCH_STEPS_DEFAULT = 150


# Mimic libero's Task NamedTuple shape so the eval client's
# ``task.name`` / ``task.language`` / ``task.bddl_file`` accesses work.
Task = namedtuple(
    "Task",
    ["name", "language", "problem", "problem_folder",
     "bddl_file", "init_states_file"],
)


def _parse_goal_json(raw: dict) -> dict:
    """Convert ``goal.json`` to the ``monitor_dict`` _check_success expects.

    Both legacy (list of [verb, subj] / [verb, subj, region]) and the
    newer step-annotated format ({"state_pair": [...], "task_step": N})
    are supported, mirroring
    ``RoboCerebra/evaluation/utils.py:load_actions``.
    """
    out: dict[str, list] = {}
    for obj_id, entries in raw.items():
        states: list = []
        for item in entries:
            if isinstance(item, dict) and "state_pair" in item:
                triple = item["state_pair"]
            elif isinstance(item, list):
                triple = item
            else:
                continue
            if len(triple) == 2:
                verb, subj = triple
                states.append([verb.lower(), subj])
            elif len(triple) == 3:
                verb, subj, region = triple
                states.append([verb.lower(), subj, region])
        out[obj_id] = states
    return out


def _read_language(case_dir: Path) -> str:
    """Read the human-readable task language from task_description.txt.

    Falls back to the case dir name if the txt is missing or empty.
    """
    txt = case_dir / "task_description.txt"
    if txt.is_file():
        try:
            line = txt.read_text(encoding="utf-8").splitlines()[0].strip()
            if line:
                return line
        except Exception:
            pass
    return case_dir.name


def _enumerate_tasks(root: Path) -> List[Task]:
    """Scan {root}/{task_type}/case*/{*.bddl} into Task records."""
    tasks: List[Task] = []
    for task_type in DEFAULT_TASK_TYPES:
        # Three task types share Ideal's BDDLs — same source dir but
        # distinct task_type prefix in the task name so init_files
        # resolve to the right disturbance-overlay file.
        bddl_root = root / ("Ideal" if task_type in _USE_IDEAL_BDDL else task_type)
        if not bddl_root.is_dir():
            logger.warning(
                f"[robocerebra] missing {task_type} → {bddl_root}, skipping"
            )
            continue
        cases = sorted(p for p in bddl_root.iterdir() if p.is_dir())
        for case_dir in cases:
            bddls = list(case_dir.glob("*.bddl"))
            if not bddls:
                continue
            bddl = bddls[0]
            # ``problem_folder`` is the path *relative to the bench root*
            # of the directory containing the BDDL file, so
            # ``{root}/{problem_folder}/{bddl_file}`` resolves correctly.
            tasks.append(Task(
                name=f"{task_type}_{case_dir.name}",
                language=_read_language(case_dir),
                problem="RoboCerebra",
                problem_folder=str(case_dir.relative_to(root)),
                bddl_file=bddl.name,
                init_states_file=f"{case_dir.name}.init",
            ))
    return tasks


class RoboCerebra:
    """LIBERO-style ``Benchmark`` for RoboCerebra task tree.

    Implements the same ``get_task`` / ``get_task_init_states`` /
    ``get_task_bddl_file_path`` / ``n_tasks`` interface the LIBERO eval
    client expects.  Resolves ``ROBOCEREBRA_BENCH_ROOT`` lazily so this
    module imports cleanly even before the env var is set.
    """

    name = "robocerebra"

    def __init__(self, task_order_index: int = 0) -> None:
        self.task_order_index = task_order_index
        self.task_embs = None
        self._tasks: List[Task] | None = None

    # ── Lazy init ──────────────────────────────────────────────────

    def _root(self) -> Path:
        env_root = os.environ.get("ROBOCEREBRA_BENCH_ROOT")
        if not env_root:
            raise RuntimeError(
                "ROBOCEREBRA_BENCH_ROOT not set — point it at your "
                "downloaded RoboCerebraBench root.  Activating "
                "~/vlm-orchestrator/.robocerebra-venv exports it."
            )
        root = Path(env_root)
        if not root.is_dir():
            raise FileNotFoundError(f"ROBOCEREBRA_BENCH_ROOT={root} not a dir")
        return root

    @property
    def tasks(self) -> List[Task]:
        if self._tasks is None:
            self._tasks = _enumerate_tasks(self._root())
            logger.info(
                f"[robocerebra] enumerated {len(self._tasks)} tasks "
                f"across {len(DEFAULT_TASK_TYPES)} task types"
            )
        return self._tasks

    # ── libero Benchmark API ──────────────────────────────────────

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task(self, i: int) -> Task:
        return self.tasks[i]

    def get_task_names(self) -> List[str]:
        return [t.name for t in self.tasks]

    def get_task_problems(self) -> List[str]:
        return [t.problem for t in self.tasks]

    def get_task_bddl_files(self) -> List[str]:
        return [t.bddl_file for t in self.tasks]

    def get_task_bddl_file_path(self, i: int) -> str:
        t = self.tasks[i]
        return str(self._root() / t.problem_folder / t.bddl_file)

    def get_task_init_states(self, i: int):
        """Load the per-case init-state from {root}/init_files/{type}/{case}.init.

        RoboCerebra's ``.init`` files are plain Python pickles holding a
        SINGLE robosuite sim state (np.ndarray), unlike LIBERO which
        ships a list of pruned init states.  The libero eval client's
        ``initial_states[episode_idx]`` expects a list, so we wrap the
        loaded state in a 1-element list — every trial against the same
        case re-uses the same deterministic init.  This matches
        RoboCerebra's upstream eval (``utils.py:load_init_state``).

        Lookup order:
          1. ``init_files/{task_type}/{case}.init`` — task-type-specific
          2. ``init_files/Ideal/{case}.init`` — fallback for the three
             types that reuse Ideal's overlays at runtime (see
             ``_USE_IDEAL_BDDL``).
        """
        import pickle
        t = self.tasks[i]
        # Recover task_type from the name we constructed.
        task_type = t.name.split("_case", 1)[0]
        candidates = [
            self._root() / "init_files" / task_type / t.init_states_file,
            self._root() / "init_files" / "Ideal" / t.init_states_file,
        ]
        for path in candidates:
            if path.is_file():
                with open(path, "rb") as f:
                    state = pickle.load(f)
                # libero eval client: ``initial_states[ep_idx]``
                return [state]
        raise FileNotFoundError(
            f"No init_states file for task '{t.name}'.  Tried: "
            + ", ".join(str(p) for p in candidates)
        )

    def get_task_demonstration(self, i: int) -> str:
        """Path to the per-case demonstration HDF5 (state replay)."""
        t = self.tasks[i]
        return f"{t.problem_folder}/{t.name.split('_case', 1)[0]}/{t.init_states_file}"

    def get_task_emb(self, i: int):
        return self.task_embs[i] if self.task_embs is not None else None

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs

    # ── Per-task success metric (RoboCerebra-specific) ───────────

    def compute_episode_success(self, env, task_id: int) -> bool:
        """Return ``True`` iff the RoboCerebra goal is fully satisfied.

        ``env.step()`` returns ``done=True`` whenever the BDDL goal
        predicate fires, but RoboCerebra's multi-step goals are
        sequence-tracked separately in ``goal.json`` and the BDDL
        predicate fires on PARTIAL progress (e.g. one of three sub-goals
        completed) — leading to false-positive successes if the eval
        relies on ``done`` alone.

        This calls ``env._check_success(monitor_dict)`` exactly as
        RoboCerebra's upstream eval does (``evaluation/episode.py:484``)
        where ``monitor_dict`` is the parsed ``goal.json``.  The
        third return value (``all_done``) is the only true success bit.
        """
        import json
        t = self.tasks[task_id]
        goal_path = self._root() / t.problem_folder / "goal.json"
        if not goal_path.is_file():
            logger.warning(
                f"[robocerebra] no goal.json at {goal_path} — falling "
                "back to env.step's `done` (may report false positives)"
            )
            return False
        try:
            raw = json.loads(goal_path.read_text(encoding="utf-8"))
            monitor_dict = _parse_goal_json(raw)
            inner = env.env if hasattr(env, "env") else env
            result = inner._check_success(monitor_dict)
        except Exception as e:
            logger.warning(
                f"[robocerebra] _check_success failed for {t.name}: {e}"
            )
            return False
        # _check_success returns (completion_dict, total_completed, all_done).
        if isinstance(result, tuple) and len(result) >= 3:
            return bool(result[2])
        # Some wrappers strip down to a single bool; respect that.
        return bool(result)

    # ── Per-task max-step budget (RoboCerebra-specific) ──────────

    def get_task_max_steps(self, i: int) -> int:
        """Compute the per-task sim-step budget the upstream eval uses.

        RoboCerebra's ``eval_openvla.py`` sets
        ``max_steps = cfg.switch_steps * segment_count``
        where ``switch_steps = 150`` (default) and ``segment_count``
        = number of steps in ``task_description.json``.  A 6-step
        task gets 900 sim steps; a 15-step task gets 2,250.

        Reading the JSON every time is cheap (the file is < 1 KB)
        and avoids caching state on the suite, which keeps episodes
        deterministic across re-imports.
        """
        import json
        t = self.tasks[i]
        case_dir = self._root() / t.problem_folder
        json_path = case_dir / "task_description.json"
        switch_steps = _SWITCH_STEPS_DEFAULT
        try:
            steps = json.loads(json_path.read_text(encoding="utf-8"))
            segment_count = max(1, len(steps))
        except Exception:
            segment_count = 1
        return switch_steps * segment_count


def register() -> None:
    """Register ``RoboCerebra`` with libero's benchmark dict.

    Idempotent: re-registration is a no-op.  Call once at import time
    of any client that wants ``--task-suite-name robocerebra`` to work.
    """
    try:
        from libero.libero.benchmark import benchmark_map
    except ImportError:
        # Older libero exposes the dict differently; try the public
        # accessor.  If neither is available, the caller will surface
        # the missing libero install.
        from libero.libero import benchmark as _bm
        if hasattr(_bm, "register_benchmark"):
            _bm.register_benchmark(RoboCerebra)
        return
    if "robocerebra" in benchmark_map:
        return
    benchmark_map["robocerebra"] = RoboCerebra
