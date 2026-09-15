#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin launcher for libero_eval_client that doesn't require tyro.

Parses CLI args manually and calls eval_libero(Args(...)) directly.
Also patches torch.load for PyTorch ≥2.6 compatibility with LIBERO's
init state files (which use numpy arrays serialized via pickle).
"""

from __future__ import annotations

import argparse
import logging
import sys

# ── PyTorch ≥2.6 compatibility ──
# LIBERO's init state files contain numpy arrays serialized with pickle.
# PyTorch 2.6+ changed torch.load to weights_only=True by default,
# which blocks numpy globals. We need to allow them.
import torch
import numpy as np

# Monkey-patch torch.load to always use weights_only=False for LIBERO
# init state files which contain plain numpy arrays.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Ensure the eval client module is importable
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())

from libero_eval_client import Args, eval_libero


def main():
    parser = argparse.ArgumentParser(description="LIBERO eval client (no-tyro wrapper)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--enable-depth", action="store_true", default=False)
    parser.add_argument("--enable-gt-state", action="store_true", default=False)
    parser.add_argument("--video-out-path", default="results/libero/videos")
    parser.add_argument("--log-dir", default="results/libero/logs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-filter", default="",
                       help="Only eval tasks containing this substring (e.g. '_language_')")
    parser.add_argument("--max-tasks", type=int, default=0,
                       help="Max tasks to evaluate (0=all). Sampled randomly with --seed.")

    cli = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = Args(
        host=cli.host,
        port=cli.port,
        resize_size=cli.resize_size,
        replan_steps=cli.replan_steps,
        task_suite_name=cli.task_suite_name,
        num_steps_wait=cli.num_steps_wait,
        num_trials_per_task=cli.num_trials_per_task,
        enable_depth=cli.enable_depth,
        enable_gt_state=cli.enable_gt_state,
        video_out_path=cli.video_out_path,
        log_dir=cli.log_dir,
        seed=cli.seed,
        task_filter=cli.task_filter,
        max_tasks=cli.max_tasks,
    )
    eval_libero(args)


if __name__ == "__main__":
    main()
