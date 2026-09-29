"""Build the per-episode dataset for the results explorer page.

    python analysis/build_explorer.py   -> analysis/explorer/data.js

Per episode: result fields, tech-tree milestones reached (first step each
item entered the inventory), the plan / executed subgoals as far as each
method's logs record them, and for failed episodes a small last-keyframe
thumbnail (where the agent ended up). Also embeds our group numbers with CIs
and the papers' group numbers.
"""
from __future__ import annotations

import base64
import glob
import gzip
import io
import json
import math
import re
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "envs"))
from common import ORDERS, RUNS_ROOT, TASKS  # noqa: E402

SKIP = {"smoke", "mocktest", "mockfix"}
MILESTONES = [  # (label, predicate on item id)
    ("log", lambda i: i.endswith("_log") or i.endswith("_wood")),
    ("planks", lambda i: i.endswith("_planks")),
    ("crafting_table", lambda i: i == "crafting_table"),
    ("wooden_pickaxe", lambda i: i == "wooden_pickaxe"),
    ("cobblestone", lambda i: i == "cobblestone"),
    ("stone_pickaxe", lambda i: i == "stone_pickaxe"),
    ("furnace", lambda i: i == "furnace"),
    ("iron_ore", lambda i: i == "iron_ore"),
    ("iron_ingot", lambda i: i == "iron_ingot"),
    ("iron_pickaxe", lambda i: i == "iron_pickaxe"),
    ("gold_ingot", lambda i: i in ("gold_ore", "gold_ingot")),
    ("redstone", lambda i: i == "redstone"),
    ("diamond", lambda i: i == "diamond"),
]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def milestones(ep: Path) -> dict:
    out = {}
    f = ep / "trajectory.jsonl.gz"
    if not f.exists():
        return out
    try:
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            for item in (r.get("inv") or {}):
                for name, pred in MILESTONES:
                    if name not in out and pred(item):
                        out[name] = r["t"]
    except (OSError, EOFError, ValueError):
        pass
    return out


def thumb(ep: Path) -> str | None:
    kf = sorted((ep / "keyframes").glob("*.jpg")) if (ep / "keyframes").exists() else []
    if not kf:
        return None
    try:
        from PIL import Image
        img = Image.open(kf[-1]).convert("RGB").resize((160, 90))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def clip(s: str, n: int = 90) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def plan_info(method: str, ep: Path) -> dict:
    """Initial plan, executed subgoals (with outcome when known), replans."""
    info: dict = {"plan": [], "exec": [], "replans": 0}
    art = ep / "artifacts"
    try:
        if method == "mineevolve":
            plans = sorted((art / "plans").glob("*.json")) if (art / "plans").exists() else []
            ini = [p for p in plans if "_initial_" in p.name]
            if ini:
                info["plan"] = [clip(s.get("condition", ""), 70) for s in json.loads(ini[0].read_text())["plan"].get("subgoals", [])]
            info["replans"] = sum(1 for p in plans if "_repair_" in p.name)
            log = (ep / "client.log").read_text(errors="ignore") if (ep / "client.log").exists() else ""
            for m in re.finditer(r"craft (mc_\w+) (\S+) x(\d+) -> (True|False)\s*([^|]*)", log):
                info["exec"].append({"s": f"{m[1][3:]} {m[2]} ×{m[3]}", "ok": m[4] == "True", "why": clip(m[5], 60)})
        elif method == "optimus1":
            log = ANSI.sub("", (ep / "client.log").read_text(errors="ignore")) if (ep / "client.log").exists() else ""
            for line in log.splitlines():
                m = re.search(r"Current Task: (.+?), Goal", line)
                if m:
                    info["exec"].append({"s": clip(m[1], 60), "ok": None})
                elif re.search(r"(\S.*?) Success", line) and info["exec"]:
                    info["exec"][-1]["ok"] = True
                elif "failed..." in line and info["exec"]:
                    info["exec"][-1]["ok"] = False
                    info["exec"][-1]["why"] = clip(line.split("Beacuse")[-1], 60)
                elif "Replanning" in line:
                    info["replans"] += 1
            info["plan"] = [e["s"] for e in info["exec"][:12]]
        elif method in ("deps", "jarvis1"):
            ev = json.loads((art / "events.json").read_text()) if (art / "events.json").exists() else []
            for e in ev:
                if e.get("event") == "plan" and not info["plan"]:
                    info["plan"] = [clip(g.get("line") or g.get("name") or json.dumps(g.get("object")), 70) for g in (e.get("plan") or e.get("goals") or [])]
                elif e.get("event") == "replan":
                    info["replans"] += 1
                elif e.get("event") == "goal":
                    info["exec"].append({"s": clip(e.get("line", ""), 60), "ok": e.get("ok"), "why": clip(e.get("info") or "", 60)})
                elif e.get("event") in ("craft", "smelt"):
                    info["exec"].append({"s": f"{e['event']} {e.get('item')} ×{e.get('n')}", "ok": e.get("ok"), "why": clip(e.get("info") or "", 60)})
        elif method == "optimus3":
            if (ep / "plan.json").exists():
                info["plan"] = [clip(x, 70) for x in json.loads((ep / "plan.json").read_text()).get("subgoals", [])][:14]
            if (ep / "events.jsonl").exists():
                started = {}
                for line in open(ep / "events.jsonl"):
                    e = json.loads(line)
                    if e.get("kind") == "subgoal_start":
                        started[e["i"]] = len(info["exec"])
                        info["exec"].append({"s": clip(e.get("subgoal", ""), 60), "ok": False})
                    elif e.get("kind") == "subgoal_done" and e.get("i") in started:
                        info["exec"][started[e["i"]]]["ok"] = True
    except Exception as exc:  # never let one malformed log break the page
        info["err"] = repr(exc)[:120]
    info["exec"] = info["exec"][:40]
    return info


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def our_numbers(eps: list) -> dict:
    """Per env/chain: SR with Wilson 95 % CI per group and overall (episodes
    pooled over task orders), per-task successes. Same numbers as
    analysis/our_numbers.json, which this rewrites so compare_papers.py stays current."""
    out = {}
    for e in eps:
        o = out.setdefault(f"{e['env']}/{e['chain']}", {"orders": set(), "tasks": {}, "cnt": {}, "infra_or_crash_episodes": 0})
        o["orders"].add(e["order"])
        t = o["tasks"].setdefault(e["task"], {"succ": 0, "runs": 0})
        t["succ"] += e["ok"]; t["runs"] += 1
        for g in (e["group"], "overall"):
            c = o["cnt"].setdefault(g, [0, 0]); c[0] += e["ok"]; c[1] += 1
        o["infra_or_crash_episodes"] += bool(e["dead"] or e["status"] == "crashed_final")
    groups = [t["group"] for t in TASKS.values()]
    for o in out.values():
        o["orders"] = len(o["orders"])
        cnt = o.pop("cnt")
        o["groups"] = {g: {"sr": round(100 * k / n, 1) if n else 0.0, "k": k, "n": n, "ci95": wilson(k, n)}
                       for g in list(dict.fromkeys(groups)) + ["overall"] for k, n in [cnt.get(g, [0, 0])]}
    return out


def main() -> int:
    eps = []
    for rf in sorted(glob.glob(str(RUNS_ROOT / "runs" / "*/*/*/*/*/result.json"))):
        p = Path(rf)
        env, chain, order, task, seed = p.parts[-6:-1]
        if order in SKIP or "." in seed or chain.startswith("steve1"):
            continue
        r = json.loads(p.read_text())
        method = chain.split("-")[0]
        ep = p.parent
        inv = r.get("final_inventory") or {}
        rec = {
            "env": env, "chain": chain, "method": method, "order": order, "task": task,
            "group": TASKS[task]["group"], "ok": bool(r.get("success")), "st": r.get("success_step"),
            "steps": r.get("steps"), "hz": r.get("horizon_steps"), "end": r.get("end_reason"),
            "status": r.get("status"), "calls": r.get("llm_calls"), "usd": round(r.get("cost_usd") or 0, 4),
            "inv": dict(sorted(inv.items(), key=lambda kv: -kv[1])[:10]),
            "ms": milestones(ep), "src": r.get("plan_source"), "dead": bool(r.get("infra_env_died")),
            "idx": (ORDERS.get(order, []).index(task) + 1) if task in ORDERS.get(order, []) else None,
            **plan_info(method, ep),
        }
        if not rec["ok"]:
            rec["img"] = thumb(ep)
        eps.append(rec)
    ours = our_numbers(eps)
    (REPRO / "analysis" / "our_numbers.json").write_text(json.dumps(ours, indent=1) + "\n")
    papers = json.loads((REPRO / "analysis" / "paper_numbers.json").read_text())
    paper_groups = {pk: {row: {k: v for k, v in g.items() if not isinstance(v, (dict, str))}
                         for row, g in (pv.get("groups") or {}).items()} for pk, pv in papers.items()}
    tasks = [{"uid": u, "group": t["group"], "text": t["instruction"]} for u, t in TASKS.items()]
    import datetime
    data = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "runs_root": str(RUNS_ROOT),
            "episodes": eps, "ours": {k: v["groups"] for k, v in ours.items()}, "papers": paper_groups,
            "tasks": tasks, "milestones": [m[0] for m in MILESTONES]}
    out = REPRO / "analysis" / "explorer" / "data.js"
    out.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"{len(eps)} episodes, {out.stat().st_size / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
