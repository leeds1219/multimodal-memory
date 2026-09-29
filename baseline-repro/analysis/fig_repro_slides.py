"""Slide figures for the reproduction section.

    conda run -n optimus3 python analysis/fig_repro_slides.py   (any env with matplotlib)
    -> analysis/out/fig_repro_optimus1.png   Optimus-1 reproduces in its own environment
    -> analysis/out/fig_repro_optimus1_envM.png   the same Optimus-1 in MineEvolve's environment
    -> analysis/out/fig_repro_mineevolve_env{M,O}.png   MineEvolve, same format
    -> analysis/out/fig_repro_release.png    the others vs how complete their release is

Only measured numbers (analysis/our_numbers.json, Wilson 95% CIs) and
paper-reported numbers (analysis/paper_numbers.json). Paper "overall" for a
method whose paper reports groups only = task-count-weighted over our 70 tasks.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OURS = json.loads((HERE / "our_numbers.json").read_text())
PAPERS = json.loads((HERE / "paper_numbers.json").read_text())
GROUPS = ["wooden", "stone", "iron", "gold", "redstone", "diamond", "armor"]
GL = {"wooden": "Wood", "stone": "Stone", "iron": "Iron", "gold": "Gold", "redstone": "Redstone", "diamond": "Diamond", "armor": "Armor"}
NT = {"wooden": 11, "stone": 10, "iron": 16, "gold": 7, "redstone": 6, "diamond": 7, "armor": 13}
INK, MUTED, LINE, GREEN, BLUE, ORANGE, RED = "#141917", "#6c7670", "#dbe0dc", "#2f6b4f", "#2a78d6", "#eb6834", "#d03b3b"


def weighted(g: dict) -> float:
    return sum(g[k] * NT[k] for k in GROUPS) / sum(NT.values())


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})


def fig_groups(env: str, chain: str, paper: dict, paper_label: str, ours_label: str, title: str, note: str, out: str):
    """Per-group success, ours (dot + 95% range) vs the paper (black line), plus overall."""
    o = OURS[f"{env}/{chain}"]["groups"]
    cats = GROUPS + ["overall"]
    pv = [paper[g] for g in GROUPS] + [weighted(paper)]
    fig, ax = plt.subplots(figsize=(13.33, 6.2), dpi=150)
    for i, g in enumerate(cats):
        r = o[g]; lo, hi = r["ci95"]
        inside = lo <= pv[i] <= hi
        x = i + (0.6 if g == "overall" else 0)
        ax.plot([x, x], [lo, hi], color=BLUE, lw=9, alpha=0.25, solid_capstyle="round", zorder=1)
        ax.scatter([x], [r["sr"]], s=90, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
        ax.plot([x - 0.28, x + 0.28], [pv[i]] * 2, color=INK, lw=2.5, zorder=2)
        # no mark when there were 0 successes: the range then reaches the paper value only because it is wide
        mark, mcol = ("", MUTED) if inside and r["sr"] == 0 else ("✓", GREEN) if inside else ("↓" if r["sr"] < pv[i] else "↑", RED)
        if mark:
            ax.text(x, max(hi, pv[i]) + 4, mark, ha="center", fontsize=15, color=mcol, fontweight="bold")
        ax.text(x, -9, f"{r['sr']:.0f} / {pv[i]:.0f}", ha="center", fontsize=11, color=MUTED)
    ax.set_xticks([i + (0.6 if g == "overall" else 0) for i, g in enumerate(cats)])
    ax.set_xticklabels([f"{GL[g]}\n({NT[g]} tasks)" for g in GROUPS] + ["Overall\n(70 tasks)"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(-14, 108); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("success rate (%)")
    ax.axvline(len(GROUPS) - 0.2, color=LINE, lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LINE, lw=0.8, ls=":")
    ax.text(-0.5, -12.5, "ours / paper", fontsize=10, color=MUTED)
    h1 = ax.scatter([], [], s=90, color=BLUE, label=ours_label)
    h2, = ax.plot([], [], color=INK, lw=2.5, label=paper_label)
    ax.legend(handles=[h1, h2], loc="upper right", frameon=False, fontsize=11)
    ov, pov = o["overall"]["sr"], weighted(paper)
    ax.set_title(title.format(ov=ov, pov=pov), loc="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.01, -0.05, "✓ paper value inside our 95% range; ↓/↑ outside it (no mark: 0 successes). Bars are 95% ranges and lean toward 50% "
             "near 0 or 100, so the dot is not centred. Paper overall = its group values weighted by our task counts.\n" + note,
             fontsize=10, color=MUTED)
    fig.savefig(OUT / out, bbox_inches="tight", facecolor="white")


def fig_optimus1(env: str = "O"):
    p = PAPERS["optimus1"]["groups"]["Optimus-1|-"]
    lab = f"ours: Gemini-3-Flash, Env {env}, authors' full memory, 3 task orders (bar = 95% range)"
    if env == "O":
        fig_groups(env, "optimus1-prebuilt", p, "Optimus-1 paper, Table 1 (GPT-4V)", lab,
                   "Optimus-1 reproduces in its own environment: overall {ov:.1f}% vs {pov:.1f}% in the paper",
                   "Wood/Stone below the paper, Diamond above it (Env O places diamond ore under the agent).",
                   "fig_repro_optimus1.png")
    else:
        fig_groups(env, "optimus1-prebuilt", p, "Optimus-1 paper, Table 1 (GPT-4V)", lab,
                   "Same Optimus-1, MineEvolve's environment: overall {ov:.1f}% vs {pov:.1f}% in the paper",
                   "Same code, memory, planner, tasks, seeds and orders as in Env O; the environment differs "
                   "(ore placement almost never gives diamond, no auto-pickaxe, no /kill, functional crafting; STEVE-1 guidance 4.0 vs 6.0).",
                   f"fig_repro_optimus1_env{env}.png")


def fig_mineevolve(env: str):
    p = PAPERS["mineevolve"]["groups"]["MineEvolve|Gemini-3-Flash"]
    lab = f"ours: released MineEvolve, Gemini-3-Flash, Env {env}, 3 task orders (bar = 95% range)"
    where = "its own environment (as released + a minimal crafting primitive)" if env == "M" else "Optimus-1's environment"
    fig_groups(env, "mineevolve", p, "MineEvolve paper, Table 4 (Gemini-3-Flash)", lab,
               "MineEvolve in " + where.split(" (")[0] + ": overall {ov:.1f}% vs {pov:.1f}% in the paper",
               f"Env {env} = {where}. Same planner LLM as the paper's row; knowledge accumulates across the 70 tasks of each order.",
               f"fig_repro_mineevolve_env{env}.png")


def fig_release():
    me = PAPERS["mineevolve"]["groups"]
    rows = [  # label, ours M, ours O / own, paper, paper source, release note, complete?
        ("Optimus-1", "M/optimus1-prebuilt", "O/optimus1-prebuilt", weighted(PAPERS["optimus1"]["groups"]["Optimus-1|-"]),
         "Complete: code, prompts and the authors' memory", 2),
        ("DEPS", "M/deps", "O/deps", me["DEPS|Gemini-3-Flash"]["overall"],
         "Planner released; learned Selector missing (= paper's own DEP ablation)", 1),
        ("Optimus-3", None, "C3/optimus3", weighted(PAPERS["optimus3"]["groups"]["Optimus-3|-"]),
         "Weights released; evaluation loop not (GUI demo only, no reflection)", 1),
        ("JARVIS-1", "M/jarvis1", "O/jarvis1", me["JARVIS-1|Gemini-3-Flash"]["overall"],
         "Stored-plan replay only: no planner, no image memory, no self-instruct", 0),
        ("MineEvolve", "M/mineevolve", "O/mineevolve", me["MineEvolve|Gemini-3-Flash"]["overall"],
         "Crafting helper is a stub; success check and auto-pickaxe broken", 0),
    ]
    fig = plt.figure(figsize=(13.33, 6.4), dpi=150)
    ax = fig.add_axes([0.13, 0.22, 0.40, 0.64])
    tx = fig.add_axes([0.55, 0.22, 0.44, 0.64]); tx.axis("off")
    n = len(rows)
    for i, (lab, m, o, pv, note, comp) in enumerate(rows):
        y = n - 1 - i
        for key, col, dy, mk in ((m, ORANGE, 0.14, "o"), (o, BLUE, -0.14, "o")):
            if key is None:
                continue
            r = OURS[key]["groups"]["overall"]; lo, hi = r["ci95"]
            ax.plot([lo, hi], [y + dy] * 2, color=col, lw=7, alpha=0.25, solid_capstyle="round")
            ax.scatter([r["sr"]], [y + dy], s=70, color=col, edgecolor="white", lw=1.2, zorder=3, marker=mk)
        ax.scatter([pv], [y], marker="D", s=70, color=INK, zorder=4)
        ax.text(-3, y, lab, ha="right", va="center", fontsize=14, fontweight="bold", color=INK)
        mark, col = {2: ("●", GREEN), 1: ("◐", "#b77e00"), 0: ("○", RED)}[comp]
        tx.text(0.0, (y + 0.5) / n, mark, fontsize=17, color=col, va="center", transform=tx.transAxes)
        tx.text(0.06, (y + 0.5) / n, note, fontsize=12.5, color=INK, va="center", transform=tx.transAxes, wrap=True)
    ax.set_ylim(-0.6, n - 0.4); ax.set_yticks([]); ax.set_xlim(0, 60)
    ax.set_xlabel("overall success, 70 tasks (%)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LINE, lw=0.8, ls=":")
    h = [ax.scatter([], [], s=70, color=ORANGE, label="ours, Env M"),
         ax.scatter([], [], s=70, color=BLUE, label="ours, Env O (Optimus-3: its own sim)"),
         ax.scatter([], [], marker="D", s=70, color=INK, label="paper")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=10.5)
    tx.text(0.0, 1.03, "What the public release contains (code inspection)", fontsize=12, color=MUTED,
            transform=tx.transAxes, fontweight="bold")
    fig.text(0.02, 0.93, "The more complete the release, the closer the reproduction", fontsize=16,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.0, "Same planner LLM (Gemini-3-Flash), tasks, seeds and task orders for every method. Paper values: "
             "MineEvolve Table 4 Gemini-3-Flash rows; Optimus-1 and Optimus-3 from their own tables. Bars = 95% range.",
             fontsize=10, color=MUTED)
    fig.savefig(OUT / "fig_repro_release.png", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    style(); OUT.mkdir(exist_ok=True)
    fig_optimus1("O"); fig_optimus1("M"); fig_mineevolve("M"); fig_mineevolve("O"); fig_release()
    print(OUT / "fig_repro_optimus1.png", OUT / "fig_repro_release.png")
