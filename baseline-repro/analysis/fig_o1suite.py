"""Figures for the Optimus-1 native suite (its own 67 tasks, paper horizons, gemini-3.8-flash).

    conda run -n optimus3 python analysis/fig_o1suite.py      (any env with matplotlib + PIL + yaml)
    -> analysis/out/o1suite_groups.png       per group: ours (logfix; released on Wood/Stone) vs paper
    -> analysis/out/o1suite_woodstone.png    every Wood/Stone episode, coloured by outcome / failure cause
    -> analysis/out/o1suite_notree.png       episodes that never got wood: spawn view, middle, end
    -> analysis/out/o1suite_episodes.json    one row per episode with its failure category

Failure categories (first matching rule):
  crash          released-code exception (end_reason method_exception)
  plan ended     plan finished without the target item (Planner)
  no wood        time limit reached and no log was ever collected (Policy: tree not found/chopped)
  stuck digging  time limit reached with a wooden pickaxe but no cobblestone (Policy: dig down)
  out of time    time limit reached after partial progress
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
OUT = HERE / "out"
RUNS = Path("/home/rag/data/repro_runs/suite_optimus1/runs/O")
SUITE = REPRO / "configs" / "suites" / "optimus1"
TASKS = {t["uid"]: t for t in yaml.safe_load(open(SUITE / "tasks.yaml"))["tasks"]}
PAPER = json.loads((HERE / "paper_numbers.json").read_text())["optimus1"]
PG = PAPER["groups"]["Optimus-1|-"]
G = ["wooden", "stone", "iron", "gold", "redstone", "diamond", "armor"]
GL = {"wooden": "Wood", "stone": "Stone", "iron": "Iron", "gold": "Gold", "redstone": "Redstone", "diamond": "Diamond", "armor": "Armor"}
NT = {g: sum(t["group"] == g for t in TASKS.values()) for g in G}
LF, RL = "optimus1-prebuilt-logfix-g38", "optimus1-prebuilt-g38"
INK, MUTED, LINE, BLUE, ORANGE, GREEN, RED = "#141917", "#6c7670", "#dbe0dc", "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b"
CAT_COL = {"success": GREEN, "no wood": "#8c5a2b", "stuck digging": "#b77e00", "plan ended": "#7b5bd6",
           "crash": RED, "out of time": "#9aa3a0"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def items(ep: Path) -> dict:
    first = {}
    f = ep / "trajectory.jsonl.gz"
    if f.exists():
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            for k in (r.get("inv") or {}):
                first.setdefault(k, r["t"])
    return first


def category(r: dict, first: dict) -> str:
    if r.get("success"):
        return "success"
    end = r.get("end_reason")
    if end == "method_exception":
        return "crash"
    if end in ("plan_finished", "plan_exhausted", "plan_failed_or_timeout"):
        return "plan ended"
    if not any(k.endswith("_log") for k in first):
        return "no wood"
    if "wooden_pickaxe" in first and "cobblestone" not in first:
        return "stuck digging"
    return "out of time"


def load() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(RUNS / "*" / "seed*" / "*" / "*" / "result.json"))):
        if ".crash" in f:
            continue
        p = Path(f).parent
        r = json.loads(Path(f).read_text())
        first = items(p)
        task = p.parts[-2]
        log = re.sub(r"\x1b\[[0-9;]*m", "", (p / "client.log").read_text(errors="ignore")) if (p / "client.log").exists() else ""
        rows.append({"chain": p.parts[-4], "world": p.parts[-3].split("_")[0], "task": task, "group": TASKS[task]["group"],
                     "text": TASKS[task]["instruction"], "ok": bool(r.get("success")), "end": r.get("end_reason"),
                     "steps": r.get("steps"), "src": r.get("plan_source"), "cat": category(r, first),
                     "subgoals": re.findall(r"Current Task: ([^,]+),", log), "dir": str(p)})
    return rows


def paper_task_sr() -> dict:
    out = {}
    for v in PAPER["tasks"]["Optimus-1|GPT-4V"].values():
        if v.get("sr") is not None and v.get("match") != "none":
            out[" ".join(re.sub(r"\s*\(Table.*", "", v["paper_task"]).lower().split())] = v["sr"]
    return out


def fig_groups(rows):
    fig, ax = plt.subplots(figsize=(13.33, 6.2), dpi=150)
    cats = G + ["o5", "all"]
    pv = [PG[g] for g in G] + [sum(PG[g] for g in G[2:]) / 5, sum(PG[g] * NT[g] for g in G) / 67]
    ours = {}
    for i, g in enumerate(cats):
        x = i + (0.6 if i == len(G) else 1.0 if i > len(G) else 0)
        for chain, col, dx in ((LF, BLUE, -0.12), (RL, ORANGE, 0.12)):
            if g == "o5":
                rs = [r for r in rows if r["chain"] == chain and r["group"] in G[2:]]
                if chain != LF or not rs:
                    continue
                v = sum(100 * sum(r["ok"] for r in rs if r["group"] == gg) / max(1, sum(r["group"] == gg for r in rs)) for gg in G[2:]) / 5
                ax.scatter([x + dx], [v], s=90, color=col, edgecolor="white", lw=1.5, zorder=3); ours[(chain, g)] = v
                continue
            rs = [r for r in rows if r["chain"] == chain and (g == "all" or r["group"] == g)]
            if not rs or (chain == RL and g == "all"):
                continue
            k, n = sum(r["ok"] for r in rs), len(rs); lo, hi = wilson(k, n)
            ax.plot([x + dx] * 2, [lo, hi], color=col, lw=8, alpha=0.25, solid_capstyle="round")
            ax.scatter([x + dx], [100 * k / n], s=90, color=col, edgecolor="white", lw=1.5, zorder=3)
            ours[(chain, g)] = 100 * k / n
        ax.plot([x - 0.3, x + 0.3], [pv[i]] * 2, color=INK, lw=2.5, zorder=2)
        lab = f"{ours.get((LF, g), float('nan')):.0f} / {pv[i]:.0f}"
        ax.text(x, -9, lab, ha="center", fontsize=11, color=MUTED)
    ax.set_xticks([i + (0.6 if i == len(G) else 1.0 if i > len(G) else 0) for i in range(len(cats))])
    ax.set_xticklabels([f"{GL[g]}\n({NT[g]} tasks)" for g in G] + ["Overall\n5 hard groups\n(paper's def.)", "Overall\nall 67 tasks\n(weighted)"], fontsize=11)
    ax.axvline(len(G) - 0.2, color=LINE, lw=1)
    ax.set_ylim(-14, 108); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("success rate (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LINE, lw=0.8, ls=":")
    ax.text(-0.5, -12.5, "ours / paper", fontsize=10, color=MUTED)
    h = [ax.scatter([], [], s=90, color=BLUE, label="ours, logfix (released code + one-line graph fix)"),
         ax.scatter([], [], s=90, color=ORANGE, label="ours, released code (Wood/Stone only)"),
         ax.plot([], [], color=INK, lw=2.5, label="Optimus-1 paper (GPT-4V)")[0]]
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=11)
    ax.set_title("Optimus-1 on its own 67 tasks (paper time limits, gemini-3.8-flash, 3 worlds per task)",
                 loc="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, -0.05, "Bars: 95% range. Iron and harder groups are above the paper; Wood and Stone remain below it.\n"
             "Env O as released (ore placed under the agent, GUI macros, always day); time limits from paper Table 5; the paper ran >=30 worlds per task.",
             fontsize=10, color=MUTED)
    fig.savefig(OUT / "o1suite_groups.png", bbox_inches="tight", facecolor="white")


def fig_woodstone(rows):
    ps = paper_task_sr()
    tasks = [u for u in TASKS if TASKS[u]["group"] in ("wooden", "stone")]
    fig, ax = plt.subplots(figsize=(13.33, 7.4), dpi=150)
    for i, u in enumerate(tasks):
        y = len(tasks) - 1 - i
        for j, (chain, w) in enumerate([(c, w) for c in (RL, LF) for w in ("seed0", "seed1", "seed2")]):
            r = next((r for r in rows if r["chain"] == chain and r["task"] == u and r["world"] == w), None)
            x = j + (0.4 if j >= 3 else 0)
            if r is None:
                continue
            ax.add_patch(plt.Rectangle((x, y - 0.4), 0.9, 0.8, color=CAT_COL[r["cat"]], ec="white", lw=1))
        sr = ps.get(" ".join(TASKS[u]["instruction"].lower().split()))
        ax.text(-0.2, y, TASKS[u]["instruction"], ha="right", va="center", fontsize=11, color=INK)
        ax.text(6.7, y, f"{sr:.0f}%" if sr is not None else "–", ha="left", va="center", fontsize=11, color=MUTED)
    ax.set_xlim(-0.1, 7.4); ax.set_ylim(-0.7, len(tasks) - 0.3)
    ax.set_xticks([0.45, 1.45, 2.45, 3.85, 4.85, 5.85]); ax.set_xticklabels(["world 1", "world 2", "world 3"] * 2, fontsize=10)
    ax.text(1.45, len(tasks) - 0.1, "released code", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(4.85, len(tasks) - 0.1, "logfix", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(6.7, len(tasks) - 0.1, "paper", ha="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COL.values()]
    labels = ["success", "no wood (tree never found/chopped)", "stuck digging down", "plan ended without the item (Planner)",
              "released-code crash", "out of time after progress"]
    ax.legend(hs, labels, loc="upper center", bbox_to_anchor=(0.45, -0.04), ncol=3, frameon=False, fontsize=10.5)
    ax.set_title("Wood and Stone: every episode (same three worlds for both versions)", loc="left", fontsize=15,
                 fontweight="bold", color=INK, pad=22)
    fig.savefig(OUT / "o1suite_woodstone.png", bbox_inches="tight", facecolor="white")


def fig_notree(rows):
    sel = [r for r in rows if r["cat"] == "no wood" and r["group"] in ("wooden", "stone")]
    sel.sort(key=lambda r: (r["chain"], r["task"], r["world"]))
    sel = sel[:8]
    if not sel:
        return
    fig, axes = plt.subplots(len(sel), 3, figsize=(10, 2.0 * len(sel)), dpi=130)
    for i, r in enumerate(sel):
        ep = Path(r["dir"])
        kf = sorted((ep / "keyframes").glob("*.jpg"))
        picks = [ep / "first_frame.png", kf[len(kf) // 2] if kf else None, kf[-1] if kf else None]
        names = ["spawn", f"step {int(picks[1].stem):,}" if picks[1] else "", f"step {int(picks[2].stem):,}" if picks[2] else ""]
        for j in range(3):
            a = axes[i][j]; a.set_xticks([]); a.set_yticks([])
            if picks[j] is not None and Path(picks[j]).exists():
                a.imshow(Image.open(picks[j]))
            a.set_title(names[j], fontsize=9, color=MUTED)
            if j == 0:
                tag = "released" if r["chain"] == RL else "logfix"
                a.set_ylabel(f"{r['text']}\n{tag}, {r['world']}", fontsize=9, rotation=0, ha="right", va="center", color=INK)
    fig.suptitle("Episodes that never collected wood: where the agent started and where it went", fontsize=13,
                 fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "o1suite_notree.png", bbox_inches="tight", facecolor="white")


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})
    OUT.mkdir(exist_ok=True)
    rows = load()
    (OUT / "o1suite_episodes.json").write_text(json.dumps(rows, indent=1))
    fig_groups(rows); fig_woodstone(rows); fig_notree(rows)
    from collections import Counter
    for chain in (LF, RL):
        print(chain, Counter(r["cat"] for r in rows if r["chain"] == chain and r["group"] in ("wooden", "stone")))
        print("  all groups:", Counter(r["cat"] for r in rows if r["chain"] == chain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
