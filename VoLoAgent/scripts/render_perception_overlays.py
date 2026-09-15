# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render debug_perception overlays from results.json.

Reads `debug_perception/results.json` (override the directory with the
``DEBUG_PERCEPTION_DIR`` env var) and produces:

- One stacked overlay per backend with N panels (one per prompt group).
- A backends × groups grid for at-a-glance comparison.

Numbered markers + on-image legend per panel so overlapping points are
still readable.  Idempotent — overwrites `*_compare.png` and
`comparison_grid.png`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DEBUG_DIR = Path(os.environ.get("DEBUG_PERCEPTION_DIR", "debug_perception"))
RESULTS_JSON = DEBUG_DIR / "results.json"

# Vivid, distinguishable BGR colors (10 unique).
COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (255, 0, 255), (255, 100, 0), (200, 50, 200), (50, 200, 200), (180, 180, 255),
]


def _draw_panel(img_bgr, title, prompts, results_per_prompt):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    cv2.putText(out, title, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    legend_w = 500
    legend_h = 22 * len(prompts) + 14
    legend_x = w - legend_w - 8
    legend_y = 8
    overlay = out.copy()
    cv2.rectangle(overlay, (legend_x, legend_y),
                  (legend_x + legend_w, legend_y + legend_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    for i, prompt in enumerate(prompts):
        c = COLORS[i % len(COLORS)]
        val = results_per_prompt.get(prompt)
        ly = legend_y + 22 * i + 24
        cv2.circle(out, (legend_x + 18, ly - 6), 8, c, -1)
        cv2.circle(out, (legend_x + 18, ly - 6), 8, (255, 255, 255), 1)
        label = f"{i + 1}. {prompt}"
        if val is None:
            label += "  [NONE]"
        cv2.putText(out, label, (legend_x + 36, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if val is None:
            continue
        xn, yn = val
        px = int(xn * w); py = int(yn * h)
        cv2.circle(out, (px, py), 18, c, 3)
        cv2.circle(out, (px, py), 4, c, -1)
        cv2.putText(out, str(i + 1), (px + 20, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2, cv2.LINE_AA)

    return out


def main():
    if not RESULTS_JSON.exists():
        sys.exit(f"missing {RESULTS_JSON} — run molmo_compare_standalone.py first")

    data = json.loads(RESULTS_JSON.read_text())
    input_path = Path(data["input"])
    img_bgr = cv2.imread(str(input_path))
    print(f"input: {input_path}  shape={img_bgr.shape}")
    groups: dict[str, list[str]] = data["groups"]
    results: dict[str, dict] = data["results"]

    backends = list(results.keys())
    group_names = list(groups.keys())

    per_backend_panels: dict[str, list[np.ndarray]] = {}
    for b in backends:
        panels = []
        for g in group_names:
            panels.append(_draw_panel(
                img_bgr, f"{b} | {g}", groups[g], results[b],
            ))
        stacked = np.vstack(panels)
        out_path = DEBUG_DIR / f"{b}_compare.png"
        cv2.imwrite(str(out_path), stacked)
        print(f"  wrote {out_path.name}  ({stacked.shape[1]}x{stacked.shape[0]})")
        per_backend_panels[b] = panels

    # Big grid: rows = backends, columns = groups
    rows = []
    for b in backends:
        rows.append(np.hstack(per_backend_panels[b]))
    grid = np.vstack(rows)
    grid_path = DEBUG_DIR / "comparison_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"  wrote {grid_path.name}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
