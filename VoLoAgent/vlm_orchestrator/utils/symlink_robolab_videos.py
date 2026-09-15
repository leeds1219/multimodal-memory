# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retroactively symlink robolab videos into orch episode dirs.

When the orchestrator and eval client run on separate machines, the proxy's
auto-symlink (proxy.py:_symlink_robolab_outputs) runs where
``robolab_output_dir`` is unknown, so no symlinks land in the orch results
dir.  Once both trees are available locally, this wires them up.

Slug + per-task index logic mirrors proxy.py exactly:

    cleaned = re.sub(r"[^\\w\\s]", "", instruction).replace(" ", "_")
    # Within the orch's <slug>/episode_<id>/ dirs sorted by episode_id,
    # the N-th maps to robolab's <cleaned>_<N>.mp4 (0-based).

Usage:

    python vlm_orchestrator/utils/symlink_robolab_videos.py \\
        --orch-dir ~/vlm-orchestrator/results/lh_cs_tool_chain_sam3_molmo2 \\
        --robolab-dir ~/robolab/output/lh_cs_tool_chain_sam3_molmo2
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

VIDEO_SUFFIXES = ["", "_viewport", "_annotated", "_policy"]


def _slug(instruction: str) -> str:
    """Match proxy.py's robolab slug."""
    return re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")


def _find_task_dir(robolab_root: Path, slug: str) -> Path | None:
    """Find the task subdirectory containing videos with this slug."""
    for cand in robolab_root.iterdir():
        if not cand.is_dir():
            continue
        # robolab writes videos directly into <TaskName>/<slug>_<idx>.mp4
        if any(p.name.startswith(slug + "_") for p in cand.glob(f"{slug}_*.mp4")):
            return cand
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orch-dir", required=True, type=Path,
                   help="Orchestrator results dir (e.g. results/lh_cs_tool_chain_sam3_molmo2)")
    p.add_argument("--robolab-dir", required=True, type=Path,
                   help="Robolab output dir (e.g. robolab/output/lh_cs_tool_chain_sam3_molmo2)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without creating symlinks")
    args = p.parse_args()

    orch = args.orch_dir.expanduser().resolve()
    robolab = args.robolab_dir.expanduser().resolve()
    if not orch.is_dir():
        raise SystemExit(f"Orch dir not found: {orch}")
    if not robolab.is_dir():
        raise SystemExit(f"Robolab dir not found: {robolab}")

    n_linked = 0
    n_missing = 0
    for slug_dir in sorted(orch.iterdir()):
        if not slug_dir.is_dir():
            continue
        # Collect (episode_id, episode_dir) pairs.
        episodes = []
        for ep_dir in slug_dir.iterdir():
            if not (ep_dir.is_dir() and ep_dir.name.startswith("episode_")):
                continue
            md_path = ep_dir / "metadata.json"
            if not md_path.exists():
                continue
            md = json.loads(md_path.read_text())
            ep_id = md.get("episode_id")
            instruction = md.get("original_instruction")
            if ep_id is None or instruction is None:
                continue
            episodes.append((ep_id, ep_dir, instruction))

        if not episodes:
            continue
        # Sort by orch episode_id ascending; N-th item gets robolab idx N (0-based).
        episodes.sort(key=lambda t: t[0])
        instruction = episodes[0][2]
        slug = _slug(instruction)

        task_dir = _find_task_dir(robolab, slug)
        if task_dir is None:
            print(f"  [skip] {slug_dir.name}: no robolab task dir matches {slug!r}")
            continue

        for robolab_idx, (ep_id, ep_dir, _) in enumerate(episodes):
            for suffix in VIDEO_SUFFIXES:
                fname = f"{slug}_{robolab_idx}{suffix}.mp4"
                src = task_dir / fname
                if not src.exists():
                    n_missing += 1
                    continue
                dst = ep_dir / fname
                if args.dry_run:
                    print(f"  [dry] {dst} -> {src}")
                else:
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    dst.symlink_to(src)
                    n_linked += 1

            # Also link the per-episode log if present
            log_src = task_dir / f"log_{robolab_idx}.json"
            if log_src.exists():
                log_dst = ep_dir / f"log_{robolab_idx}.json"
                if args.dry_run:
                    print(f"  [dry] {log_dst} -> {log_src}")
                else:
                    if log_dst.is_symlink() or log_dst.exists():
                        log_dst.unlink()
                    log_dst.symlink_to(log_src)
                    n_linked += 1

    verb = "would link" if args.dry_run else "linked"
    print(f"\n{verb} {n_linked} files ({n_missing} candidate files not found in robolab tree)")


if __name__ == "__main__":
    main()
