"""Slide figure: Env O places diamond ore under the agent.

    conda run -n optimus3 python analysis/fig_diamond_harness.py   (any env with matplotlib + PIL)
    -> analysis/out/fig_diamond_harness.png

Same method (Optimus-1, full memory), same task (craft a diamond axe), same seed,
order0, in both environments. Top: Env O, depth over time, the steps at which the
environment's `random_ore` placed diamond ore under the agent (from the
"diamond ore at" lines in client.log, mapped from log time to env step through
the keyframe file times), and the step the first diamond entered the inventory.
Bottom: the same run in Env M, where no diamond is ever obtained.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

RUNS = Path("/home/rag/data/repro_runs/runs")
OUT = Path(__file__).resolve().parent / "out" / "fig_diamond_harness.png"
TASK, CHAIN, ORDER = "diamond_02", "optimus1-prebuilt", "order0"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
BLUE, ORANGE, GREEN, INK, MUTED, BAND = "#2a78d6", "#eb6834", "#0ca30c", "#141917", "#6c7670", "#eef1ed"


def episode(env: str) -> Path:
    return next((RUNS / env / CHAIN / ORDER / TASK).iterdir())


def clock(ep: Path):
    pts = sorted((p.stat().st_mtime, int(p.stem)) for p in (ep / "keyframes").glob("*.jpg"))
    def step(ts):
        for (t0, s0), (t1, s1) in zip(pts, pts[1:]):
            if t0 <= ts <= t1:
                return s0 + (s1 - s0) * (ts - t0) / max(t1 - t0, 1e-6)
        return pts[0][1] if ts < pts[0][0] else pts[-1][1]
    return step


def traj(ep: Path):
    ys, first = [], {}
    for line in gzip.open(ep / "trajectory.jsonl.gz", "rt"):
        r = json.loads(line)
        if "pos" in r:
            ys.append((r["t"], r["pos"][1]))
        for k in (r.get("inv") or {}):
            first.setdefault(k, r["t"])
    return ys, first


def placements(ep: Path):
    step, ts, out = clock(ep), None, []
    for line in ANSI.sub("", (ep / "client.log").read_text(errors="ignore")).splitlines():
        m = re.match(r"\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d)\]", line)
        if m:
            ts = dt.datetime.strptime(m[1], "%m/%d/%y %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
        m = re.match(r"diamond ore at (-?\d+)", line.strip())
        if m and ts is not None:
            out.append((step(ts), int(m[1])))
    return out


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 1, figsize=(13.33, 7.5), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28})
    epO, epM = episode("O"), episode("M")
    xmax = 12000
    for ax, ep, title in ((axes[0], epO, "Env O (Optimus-1's own environment)"),
                          (axes[1], epM, "Env M (MineEvolve's environment): same method, task and seed")):
        ys, first = traj(ep)
        ax.axhspan(0, 14, color=BAND, zorder=0)
        ax.plot([t for t, _ in ys], [y for _, y in ys], color=BLUE, lw=1.8, zorder=2)
        ax.set_xlim(0, xmax); ax.set_ylim(0, 80)
        ax.set_ylabel("depth (y)")
        ax.set_title(title, loc="left", fontsize=14, color=INK, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#dbe0dc", lw=0.8, ls=":")
        for k, lab in (("iron_pickaxe", "iron pickaxe"),):
            if k in first:
                ax.axvline(first[k], color=MUTED, lw=1, ls="--", zorder=1)
                ax.text(first[k] + 80, 74, lab, color=MUTED, fontsize=11, va="top")
        if ep is epO:
            ax.text(150, 7, "y ≤ 14: ore placed 3–5 blocks below the agent, 10%/step",
                    color=MUTED, fontsize=11, va="center")
            pl = placements(ep)
            ax.scatter([s for s, _ in pl], [y for _, y in pl], marker="D", s=46, color=ORANGE, edgecolor="white",
                       lw=0.8, zorder=4, label=f"diamond ore placed by the environment ({len(pl)}×)")
            if "diamond" in first:
                t = first["diamond"]
                ax.axvline(t, color=GREEN, lw=2, zorder=3)
                reach = next(tt for tt, y in ys if y <= 14)
                ax.annotate(f"first diamond at step {t:,}\n({t - reach:,} steps after reaching y ≤ 14)",
                            xy=(t, 40), xytext=(t + 450, 52), color=INK, fontsize=12,
                            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.5})
            ax.legend(loc="upper right", frameon=False, fontsize=11)
            kf = ep / "keyframes" / f"{(first.get('diamond', 6700) // 100) * 100:06d}.jpg"
            if kf.exists():
                inset = ax.inset_axes([0.73, 0.08, 0.25, 0.45])
                inset.imshow(Image.open(kf)); inset.set_xticks([]); inset.set_yticks([])
                for s in inset.spines.values():
                    s.set_edgecolor(GREEN); s.set_linewidth(2)
                inset.set_title("agent view near that step", fontsize=10, color=MUTED)
        else:
            ax.text(150, 7, "no diamond in 36,000 steps (Env M's depth bands almost never place diamond ore)", color=MUTED, fontsize=11, va="center")
    axes[1].set_xlabel("environment step (20 steps = 1 s of game time)")
    fig.suptitle("Optimus-1 · \"Craft a diamond axe\" · seed and task order identical in both environments",
                 x=0.06, ha="left", fontsize=12, color=MUTED, y=0.995)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
