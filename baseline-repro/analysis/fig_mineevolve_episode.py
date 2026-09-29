"""Slide figure: what MineEvolve does in the world where Optimus-1 succeeded.

    conda run -n optimus3 python analysis/fig_mineevolve_episode.py   (any env with matplotlib + PIL)
    -> analysis/out/fig_mineevolve_episode.png

Env O, task "Craft a diamond axe", order0, same seed (same world), same STEVE-1
action model for both methods. Top: Optimus-1's log count over time. Bottom:
MineEvolve's subgoals in execution order (from artifacts/evidence/*/summary.json,
ordered by write time, placed back to back by their step counts), its log count,
and where the episode was aborted. Footer: how MineEvolve's failed episodes end
across all of Env O.
"""
from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

RUNS = Path("/home/rag/data/repro_runs/runs")
OUT = Path(__file__).resolve().parent / "out" / "fig_mineevolve_episode.png"
TASK, ORDER = "diamond_02", "order0"
BLUE, ORANGE, RED, GREEN, INK, MUTED, LINE = "#2a78d6", "#eb6834", "#d03b3b", "#0ca30c", "#141917", "#6c7670", "#dbe0dc"


def ep(env, chain):
    return next(p for p in (RUNS / env / chain / ORDER / TASK).iterdir() if "." not in p.name)


def log_curve(e: Path):
    """Logs collected so far (running maximum, so crafting them into planks does not look like a loss)."""
    pts = [(0, 0)]
    for line in gzip.open(e / "trajectory.jsonl.gz", "rt"):
        r = json.loads(line)
        inv = r.get("inv")
        if inv is not None:
            n = sum(v for k, v in inv.items() if k.endswith("_log"))
            if n > pts[-1][1]:
                pts.append((r["t"], n))
    return pts


def mineevolve_subgoals(e: Path):
    rows = []
    for s in glob.glob(str(e / "artifacts" / "evidence" / "*" / "*" / "*" / "summary.json")):
        j = json.loads(Path(s).read_text())
        rows.append((Path(s).stat().st_mtime, j["condition"], int(j.get("steps") or 0), bool(j.get("success"))))
    rows.sort()
    out, t = [], 0
    for _, cond, n, ok in rows:
        out.append((t, n, cond, ok)); t += n
    return out


def abort_stats(env="O"):
    frac, early = [], 0
    for f in glob.glob(str(RUNS / env / "mineevolve" / "*" / "*" / "*" / "result.json")):
        if ".crash" in f:
            continue
        r = json.loads(Path(f).read_text())
        if r.get("success"):
            continue
        frac.append(r["steps"] / r["horizon_steps"])
        early += r.get("end_reason") == "plan_finished"
    frac.sort()
    return len(frac), early, frac[len(frac) // 2], sum(f < 0.5 for f in frac) / len(frac)


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    eo, em = ep("O", "optimus1-prebuilt"), ep("O", "mineevolve")
    ro, rm = (json.loads((x / "result.json").read_text()) for x in (eo, em))
    fig, axes = plt.subplots(2, 1, figsize=(13.33, 7.5), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [2, 3], "hspace": 0.32})
    xmax = 6600

    # --- Optimus-1
    ax = axes[0]
    lc = log_curve(eo)
    ax.step([t for t, _ in lc] + [xmax], [n for _, n in lc] + [lc[-1][1]], where="post", color=BLUE, lw=2)
    first = next(t for t, n in lc if n >= 1); four = next(t for t, n in lc if n >= 4)
    ax.annotate(f'STEVE-1 prompt: "chop trees"\nfirst log at step {first}, 4 logs by step {four}',
                xy=(four, 4), xytext=(2400, 5), fontsize=12, color=INK,
                arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.4})
    ax.set_ylim(0, 12); ax.set_ylabel("logs collected")
    ax.set_title(f"Optimus-1: gets its wood, goes on to succeed (diamond axe at step {ro['success_step']:,})",
                 loc="left", fontsize=14, fontweight="bold", color=INK)

    # --- MineEvolve
    ax = axes[1]
    sg = mineevolve_subgoals(em)
    steve = [s for s in sg if s[1] > 0]
    crafts = [s for s in sg if s[1] == 0]
    for k, (t0, n, cond, ok) in enumerate(steve):
        y = 7.6 - k * 1.7
        ax.barh(y, n, left=t0, height=0.9, color=ORANGE if not ok else GREEN, alpha=0.28, edgecolor=ORANGE, lw=1)
        ax.text(t0 + 25, y, f'"{cond}"', va="center", fontsize=10.2, color=INK)
    if crafts:
        t = crafts[0][0]
        ax.scatter([t] * len(crafts), [1.2 + 0.3 * i for i in range(len(crafts))], marker="x", color=RED, s=40, zorder=4)
        ax.text(t + 60, 1.8, f"{len(crafts)} crafting attempts, reworded each time; no logs to craft from", fontsize=10.5,
                color=RED, va="center")
    lm = log_curve(em)
    ax.step([t for t, _ in lm] + [rm["steps"]], [n for _, n in lm] + [lm[-1][1]], where="post", color=BLUE, lw=2)
    ax.axvline(rm["steps"], color=INK, lw=1.5)
    ax.text(rm["steps"] + 60, 9.2, f"episode aborted after repeated\nsubgoal failures: step {rm['steps']:,}\n"
            f"of {rm['horizon_steps']:,} available", fontsize=11, color=INK, va="top")
    ax.set_ylim(0, 10); ax.set_yticks([])
    ax.text(40, 0.35, "logs collected: 0 for the whole episode", color=BLUE, fontsize=10.5)
    ax.set_title("MineEvolve: same world and STEVE-1, never gets a log; the fixes only reword the prompt",
                 loc="left", fontsize=14, fontweight="bold", color=INK)
    kf = sorted((em / "keyframes").glob("*.jpg"))
    if kf:
        pick = min(kf, key=lambda p: abs(int(p.stem) - 3000))
        ins = ax.inset_axes([0.005, 0.12, 0.17, 0.3])
        ins.imshow(Image.open(pick)); ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_edgecolor(ORANGE); s.set_linewidth(2)
        ins.set_title(f"agent view, step {int(pick.stem):,}", fontsize=9.5, color=MUTED)

    for a in axes:
        a.set_xlim(0, xmax)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", color=LINE, lw=0.8, ls=":")
    axes[1].set_xlabel("environment step")
    n, early, med, half = abort_stats("O")
    fig.suptitle('Env O · "Craft a diamond axe" · same seed (same world) and the same STEVE-1 action model for both methods',
                 x=0.06, ha="left", fontsize=12, color=MUTED, y=0.995)
    fig.text(0.01, -0.02, f"Across all {n} failed MineEvolve episodes in Env O: {early} ({100 * early / n:.0f}%) end because its loop gives up "
             f"before the time limit; median episode length {100 * med:.0f}% of the limit, {100 * half:.0f}% end before half.",
             fontsize=11, color=INK)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
