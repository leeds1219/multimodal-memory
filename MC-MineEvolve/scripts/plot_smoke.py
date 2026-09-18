"""Render a smoke-test run (see ``smoke_test.py``) to a single ``summary.png``.

    python scripts/plot_smoke.py                     # latest run under logs/smoke/
    python scripts/plot_smoke.py logs/smoke/run-...  # a specific run

The figure has a strip of POV frames on top, a raster of which action keys
were held on each step, and three small panels: distance travelled from the
spawn point, health / hunger, and total items in the inventory. Also prints
the ``summary.json`` as a one-line table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

# Categorical slots 1/2 + neutral inks (light surface).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MAX_FRAMES = 8
# Row order for the key raster; anything else STEVE-1 presses is appended.
KEY_ORDER = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "drop", "inventory"]


def _latest_run(root: Path) -> Path:
    runs = sorted(p for p in root.glob("run-*") if (p / "summary.json").exists())
    if not runs:
        sys.exit(f"no runs with summary.json under {root}")
    return runs[-1]


def _load(run: Path):
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(l) for l in (run / "trajectory.jsonl").read_text().splitlines() if l.strip()]
    return summary, rows


def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def render(run: Path, out: Path | None = None) -> Path:
    summary, rows = _load(run)
    out = out or run / "summary.png"

    steps = np.array([r["step"] for r in rows])
    coords = np.array([r["coords"] if len(r["coords"]) == 3 else [np.nan] * 3 for r in rows], dtype=float)
    origin = coords[~np.isnan(coords[:, 0])][0] if (~np.isnan(coords[:, 0])).any() else np.zeros(3)
    dist_xz = np.hypot(coords[:, 0] - origin[0], coords[:, 2] - origin[2])
    health = np.array([r["health"] if r["health"] is not None else np.nan for r in rows], dtype=float)
    hunger = np.array([r["hunger"] if r["hunger"] is not None else np.nan for r in rows], dtype=float)
    items = np.array([sum(r["inventory"].values()) for r in rows])

    frame_paths = sorted((run / "frames").glob("step_*.png"))
    if len(frame_paths) > MAX_FRAMES:
        idx = np.linspace(0, len(frame_paths) - 1, MAX_FRAMES).round().astype(int)
        frame_paths = [frame_paths[i] for i in idx]

    keys_seen = sorted({k for r in rows for k in r["keys"]})
    key_rows = [k for k in KEY_ORDER if k in keys_seen] + [k for k in keys_seen if k not in KEY_ORDER]
    key_rows = key_rows or ["(none)"]
    held = np.zeros((len(key_rows), len(rows)), dtype=float)
    for j, r in enumerate(rows):
        for k in r["keys"]:
            if k in key_rows:
                held[key_rows.index(k), j] = 1.0
    cam_moved = np.array([any(abs(c) > 0 for c in r["camera"]) for r in rows])

    n_frames = max(len(frame_paths), 1)
    fig = plt.figure(figsize=(14, 8.4), facecolor="#fcfcfb")
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.9 * 13.2 / n_frames * 0.5625 / 1.2, 0.28 * (len(key_rows) + 1), 1.6],
        hspace=0.5, wspace=0.3, left=0.08, right=0.98, top=0.87, bottom=0.07,
    )

    status = "PASSED" if summary.get("passed") else f"FAILED ({summary.get('error')})"
    cond = f'  ·  "{summary["condition"]}"' if summary.get("condition") else ""
    fig.suptitle(f"Smoke test {status}  ·  {summary['mode']}{cond}", x=0.08, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(
        0.08, 0.905,
        f"{run.name}  ·  env {summary.get('env')}  ·  reset {summary.get('reset_s', '?')}s  ·  "
        f"{summary.get('steps', 0)} steps @ {summary.get('steps_per_s', '?')} steps/s  ·  "
        f"inventory {summary.get('inventory_start')} → {summary.get('inventory_end')}",
        fontsize=9, color=INK_2,
    )

    # --- frame strip -----------------------------------------------------------
    strip = fig.add_subplot(gs[0, :])
    strip.axis("off")
    if frame_paths:
        n = len(frame_paths)
        sub = strip.get_subplotspec().subgridspec(1, n, wspace=0.04)
        for k, p in enumerate(frame_paths):
            ax = fig.add_subplot(sub[0, k])
            ax.imshow(Image.open(p))
            ax.set_title(f"step {int(p.stem.split('_')[1])}", fontsize=8, color=INK_2, pad=3)
            ax.axis("off")

    # --- key raster: which binary keys were held on each step ------------------
    ax = fig.add_subplot(gs[1, :])
    raster = np.vstack([held, cam_moved[None, :].astype(float)])
    ax.imshow(raster, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#fcfcfb", BLUE]), vmin=0, vmax=1,
              extent=(steps[0] - 0.5, steps[-1] + 0.5, raster.shape[0] - 0.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(raster.shape[0]))
    ax.set_yticklabels(key_rows + ["camera moved"])
    _style(ax, "Keys held per step")
    ax.grid(False)
    ax.set_xlabel("step", fontsize=8, color=INK_2)
    for y in range(raster.shape[0] - 1):
        ax.axhline(y + 0.5, color=GRID, linewidth=0.8)

    # --- panels ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(steps, dist_xz, color=BLUE, linewidth=2)
    _style(ax, "Distance from spawn (xz, blocks)")
    ax.set_xlabel("step", fontsize=8, color=INK_2)
    ax.set_ylim(0, max(1.0, float(np.nanmax(dist_xz)) * 1.1) if np.isfinite(dist_xz).any() else 1.0)
    if not (np.nan_to_num(dist_xz) > 0).any():
        ax.text(0.5, 0.5, "did not move", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=INK_2)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(steps, health, color=BLUE, linewidth=2.5, label="health")
    ax.plot(steps, hunger, color=ORANGE, linewidth=2, linestyle=(0, (4, 3)), label="hunger")
    _style(ax, "Health / hunger (0–20)")
    ax.set_xlabel("step", fontsize=8, color=INK_2)
    ax.set_ylim(0, 21)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")

    ax = fig.add_subplot(gs[2, 2])
    ax.step(steps, items, where="post", color=BLUE, linewidth=2)
    _style(ax, "Items in inventory (total)")
    ax.set_xlabel("step", fontsize=8, color=INK_2)
    ax.set_ylim(0, max(1, int(items.max())) + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if items.max() == 0:
        ax.text(0.5, 0.5, "nothing collected", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=INK_2)

    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default=None, help="run dir (default: latest under logs/smoke)")
    ap.add_argument("--out", default=None, help="output PNG (default <run>/summary.png)")
    args = ap.parse_args()

    run = Path(args.run) if args.run else _latest_run(Path("logs/smoke"))
    summary, rows = _load(run)
    print(
        f"{run.name}: {'PASSED' if summary.get('passed') else 'FAILED'} | {summary['mode']}"
        f"{' | ' + repr(summary['condition']) if summary.get('condition') else ''}"
        f" | reset {summary.get('reset_s')}s | {summary.get('steps')} steps @ {summary.get('steps_per_s')}/s"
        f" | inventory {summary.get('inventory_start')} -> {summary.get('inventory_end')}"
    )
    out = render(run, Path(args.out) if args.out else None)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
