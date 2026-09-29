"""Pick representative episodes and export small keyframe filmstrips.

    python analysis/build_filmstrips.py   -> analysis/explorer/film.js

Per configuration (env/chain) up to four episodes, chosen deterministically:
  success      the first successful Stone/Iron episode (order0 preferred)
  stuck-early  a failed Wood/Stone episode that never got past logs
  mid-way      a failed episode that reached cobblestone or further
  planner      a failed episode whose plan came from the example fallback or
               ended in a method exception (if any)
Each gets 6 keyframes spread over the steps actually played, 192x108 JPEG,
captioned with the step and the three largest inventory stacks at that step.
"""
from __future__ import annotations

import base64
import glob
import gzip
import io
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "envs"))
from common import RUNS_ROOT, TASKS  # noqa: E402

SKIP = {"smoke", "mocktest", "mockfix"}
N_FRAMES = 6
EARLY = ("log", "planks")


def inv_at(ep: Path, steps: list[int]) -> dict[int, dict]:
    """Inventory snapshot at (or just before) each requested step."""
    out, want, last = {}, sorted(steps), {}
    f = ep / "trajectory.jsonl.gz"
    if not f.exists():
        return out
    try:
        i = 0
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            while i < len(want) and r["t"] > want[i]:
                out[want[i]] = dict(last); i += 1
            if r.get("inv") is not None:
                last = r["inv"]
        while i < len(want):
            out[want[i]] = dict(last); i += 1
    except (OSError, EOFError, ValueError):
        pass
    return out


def frames(ep: Path) -> list[dict]:
    kf = sorted((ep / "keyframes").glob("*.jpg")) if (ep / "keyframes").exists() else []
    if not kf:
        return []
    idx = sorted({round(i * (len(kf) - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)})
    pick = [kf[i] for i in idx]
    steps = [int(p.stem) for p in pick]
    invs = inv_at(ep, steps)
    from PIL import Image
    out = []
    for p, s in zip(pick, steps):
        img = Image.open(p).convert("RGB").resize((192, 108))
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=60)
        inv = sorted((invs.get(s) or {}).items(), key=lambda kv: -kv[1])[:3]
        out.append({"t": s, "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                    "inv": ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in inv) or "empty"})
    return out


def stage(ms: dict) -> int:
    order = ["log", "planks", "crafting_table", "wooden_pickaxe", "cobblestone", "stone_pickaxe",
             "furnace", "iron_ore", "iron_ingot", "iron_pickaxe"]
    return max([order.index(k) + 1 for k in ms if k in order] or [0])


def main() -> int:
    from build_explorer import milestones  # reuse the same milestone definition
    by_cfg: dict[str, list] = {}
    for rf in sorted(glob.glob(str(RUNS_ROOT / "runs" / "*/*/*/*/*/result.json"))):
        p = Path(rf)
        env, chain, order, task, seed = p.parts[-6:-1]
        if order in SKIP or "." in seed or chain.startswith("steve1"):
            continue
        by_cfg.setdefault(f"{env}/{chain}", []).append((order, task, p.parent, json.loads(p.read_text())))
    film = {}
    for cfg, eps in sorted(by_cfg.items()):
        eps.sort(key=lambda x: (x[0] != "order0", x[0], x[1]))
        picks = {}
        for order, task, ep, r in eps:
            g = TASKS[task]["group"]
            ok = bool(r.get("success"))
            if ok and "success" not in picks and g in ("stone", "iron"):
                picks["success"] = (order, task, ep, r)
            if ok or r.get("infra_env_died"):
                continue
            ms = milestones(ep)
            st = stage(ms)
            if "stuck-early" not in picks and g in ("wooden", "stone") and st <= 2 and not set(ms) - set(EARLY):
                picks["stuck-early"] = (order, task, ep, r)
            elif "mid-way" not in picks and st >= 5:
                picks["mid-way"] = (order, task, ep, r)
            if "planner" not in picks and (r.get("plan_source") == "example_fallback" or r.get("end_reason") == "method_exception"):
                picks["planner"] = (order, task, ep, r)
        film[cfg] = []
        for kind in ("success", "stuck-early", "mid-way", "planner"):
            if kind not in picks:
                continue
            order, task, ep, r = picks[kind]
            fr = frames(ep)
            if not fr:
                continue
            film[cfg].append({"kind": kind, "order": order, "task": task, "text": TASKS[task]["instruction"],
                              "ok": bool(r.get("success")), "st": r.get("success_step"), "steps": r.get("steps"),
                              "hz": r.get("horizon_steps"), "end": r.get("end_reason"), "src": r.get("plan_source"),
                              "ms": milestones(ep), "frames": fr})
        print(cfg, [f["kind"] for f in film[cfg]])
    out = REPRO / "analysis" / "explorer" / "film.js"
    out.write_text("window.FILM = " + json.dumps(film, separators=(",", ":")) + ";\n")
    print(f"{out.stat().st_size / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
