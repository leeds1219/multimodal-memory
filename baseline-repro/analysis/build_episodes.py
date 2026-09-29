"""Per-episode detail files for the explorer's episode viewer.

    python analysis/build_episodes.py                 # every episode -> analysis/explorer/episodes/
    python analysis/build_episodes.py --bundle DIR    # example subset, frames embedded -> DIR

Each episode becomes `<id>.js` calling `window.__ep(id, {...})` (a script, so the
page also works from file://), plus `index.js`: one summary row per episode
(task, result, steps, furthest tech-tree stage) for the viewer's lists.
Bundle subset: the film.js episodes plus, per configuration, the first success
and the first failure of each task group (order0 first), at most BUNDLE_PER_KIND
of each. Contents:
  frames  every keyframe (step, src). Local mode: a relative URL under runs/
          (scripts/explorer.sh symlinks analysis/explorer/runs -> $RUNS_ROOT/runs);
          bundle mode: downscaled data URIs, at most MAX_FRAMES.
  inv     inventory whenever it changed; pos (sampled); hp changes
  events  plan / subgoal / craft / replan events with the env step. Steps come
          from the method's own logs where it records them; otherwise the wall
          time of the log line is mapped to a step through the keyframe file
          times (marked approx).
  llm     every LLM call: caller, prompt, response, image, tokens, cost, step
          (mapped the same way).
"""
from __future__ import annotations

import argparse
import base64
import bisect
import datetime as dt
import glob
import gzip
import io
import json
import re
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "envs"))
sys.path.insert(0, str(REPRO / "analysis"))
from common import RUNS_ROOT, TASKS  # noqa: E402

SKIP = {"smoke", "mocktest", "mockfix"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MAX_FRAMES = 48          # bundle only: frames kept per episode
FRAME_W = 208            # bundle only: embedded frame width (px)
BUNDLE_PER_KIND = 5
STAGES = [("none", []), ("wood", ["log"]), ("wooden tools", ["planks", "crafting_table", "wooden_pickaxe"]),
          ("stone", ["cobblestone", "stone_pickaxe", "furnace"]), ("iron", ["iron_ore", "iron_ingot", "iron_pickaxe"]),
          ("gold / redstone / diamond", ["gold_ingot", "redstone", "diamond"])]
CLIP_PROMPT, CLIP_RESP = 4000, 3000


def ep_id(env, chain, order, task) -> str:
    return f"{env}~{chain}~{order}~{task}"


def clip(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


class Clock:
    """Wall time -> env step, from keyframe file modification times."""

    def __init__(self, ep: Path):
        pts = []
        for p in (ep / "keyframes").glob("*.jpg") if (ep / "keyframes").exists() else []:
            try:
                pts.append((p.stat().st_mtime, int(p.stem)))
            except ValueError:
                pass
        pts.sort()
        self.ts = [p[0] for p in pts]
        self.st = [p[1] for p in pts]

    def step(self, ts):
        if ts is None or len(self.ts) < 2:
            return None
        i = bisect.bisect_left(self.ts, ts)
        if i <= 0:
            return self.st[0]
        if i >= len(self.ts):
            return self.st[-1]
        t0, t1, s0, s1 = self.ts[i - 1], self.ts[i], self.st[i - 1], self.st[i]
        return int(s0 + (s1 - s0) * (ts - t0) / max(t1 - t0, 1e-6))


def trajectory(ep: Path):
    inv, pos, hp, last_hp = [], [], [], None
    f = ep / "trajectory.jsonl.gz"
    if not f.exists():
        return inv, pos, hp
    try:
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            t = r["t"]
            if "inv" in r and r["inv"] is not None:
                inv.append([t, r["inv"]])
            if "pos" in r and (not pos or t - pos[-1][0] >= 20):
                pos.append([t] + [round(v, 1) for v in r["pos"]])
            if "hp" in r and r["hp"] != last_hp:
                hp.append([t, r["hp"]]); last_hp = r["hp"]
    except (OSError, EOFError, ValueError):
        pass
    return inv, pos, hp


def msg_text(messages) -> tuple[str, list]:
    out, imgs = [], []
    for m in messages or []:
        c = m.get("content")
        parts = c if isinstance(c, list) else [{"type": "text", "text": c}]
        txt = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                txt.append(p.get("text") or "")
            elif isinstance(p, dict) and p.get("type") == "image":
                txt.append(f"[image {p.get('file', '')}]"); imgs.append(p.get("file"))
            elif isinstance(p, str):
                txt.append(p)
        out.append(f"── {m.get('role', '?')}\n" + "\n".join(txt).strip())
    return "\n\n".join(out), imgs


def llm_calls(ep: Path, clock: Clock, rel: str):
    calls = []
    f = ep / "llm" / "calls.jsonl"
    if f.exists():
        for line in open(f):
            try:
                c = json.loads(line)
            except ValueError:
                continue
            prompt, imgs = msg_text(c.get("messages"))
            st = c.get("step")
            calls.append({"t": st if st is not None else clock.step(c.get("ts")), "approx": st is None,
                          "caller": c.get("caller", ""), "prompt": clip(prompt, CLIP_PROMPT), "resp": clip(c.get("response"), CLIP_RESP),
                          "img": [f"{rel}/llm/images/{i}" for i in imgs if i and (ep / "llm" / "images" / i).exists()],
                          "tin": c.get("tokens_in"), "tout": c.get("tokens_out_billed"), "usd": round(c.get("cost_usd") or 0, 4),
                          "lat": round(c.get("latency_s") or 0, 1), "fin": c.get("finish_reason")})
    f = ep / "llm_calls.jsonl"  # Optimus-3's own model
    if f.exists():
        for line in open(f):
            try:
                c = json.loads(line)
            except ValueError:
                continue
            calls.append({"t": c.get("env_step"), "approx": False, "caller": c.get("caller") or c.get("task_type", ""),
                          "prompt": clip(c.get("prompt"), CLIP_PROMPT), "resp": clip(c.get("response"), CLIP_RESP), "img": [],
                          "tin": c.get("tokens_in"), "tout": c.get("tokens_out"), "usd": 0, "lat": round(c.get("latency_s") or 0, 1),
                          "fin": c.get("finish_reason")})
    return calls


def parse_time(s: str):
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%m/%d/%y %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def events(method: str, ep: Path, clock: Clock):
    ev = []
    art = ep / "artifacts"

    def add(t, kind, text, ok=None, approx=False):
        ev.append({"t": t, "approx": approx, "kind": kind, "text": clip(text, 160), "ok": ok})

    if method == "optimus1" and (ep / "client.log").exists():
        ts = None
        for line in ANSI.sub("", (ep / "client.log").read_text(errors="ignore")).splitlines():
            m = re.match(r"\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d)\]", line)
            if m:
                ts = parse_time(m[1])
            if (m := re.search(r"Current Task: ([^,]+),", line)):
                add(clock.step(ts), "subgoal", m[1], None, True)
            elif re.search(r"\S Success", line):
                add(clock.step(ts), "done", line.split("INFO")[-1].split("main.py")[0].strip(), True, True)
            elif "failed..." in line:
                add(clock.step(ts), "fail", line.split("Beacuse")[-1].strip() or "failed", False, True)
            elif "Replanning" in line:
                add(clock.step(ts), "replan", "replanning", None, True)
    elif method == "mineevolve":
        for p in sorted((art / "plans").glob("*.json")) if (art / "plans").exists() else []:
            try:
                j = json.loads(p.read_text())
                sub = [s.get("condition", "") for s in j["plan"].get("subgoals", [])]
            except Exception:
                sub = []
            kind = "plan" if "_initial_" in p.name else "replan"
            add(clock.step(p.stat().st_mtime), kind, f"{kind}: " + " → ".join(sub), None, True)
        if (ep / "client.log").exists():
            for line in (ep / "client.log").read_text(errors="ignore").splitlines():
                m = re.search(r"craft (mc_\w+) (\S+) x(\d+) -> (True|False)\s*([^|]*)", line)
                if m:
                    add(clock.step(parse_time(line[:23])), "craft", f"{m[1][3:]} {m[2]} ×{m[3]}" + ("" if m[4] == "True" else f" — {m[5].strip()}"),
                        m[4] == "True", True)
    elif method in ("deps", "jarvis1") and (art / "events.json").exists():
        for e in json.loads((art / "events.json").read_text()):
            k, t = e.get("event"), e.get("t")
            if k == "plan":
                add(t, "plan", "plan: " + " → ".join(str(g.get("name") or g.get("line") or "") for g in (e.get("plan") or e.get("goals") or [])))
            elif k == "replan":
                add(t, "replan", "replan")
            elif k == "goal":
                add(t, "subgoal", (e.get("line") or "") + (f" — {e.get('info')}" if e.get("info") and not e.get("ok") else ""), e.get("ok"))
            elif k in ("craft", "smelt"):
                add(t, "craft", f"{k} {e.get('item')} ×{e.get('n')}" + (f" — {e.get('info')}" if e.get("info") and not e.get("ok") else ""), e.get("ok"))
    elif method == "optimus3" and (ep / "events.jsonl").exists():
        for line in open(ep / "events.jsonl"):
            e = json.loads(line)
            if e.get("kind") == "subgoal_start":
                add(e.get("step"), "subgoal", e.get("subgoal", ""))
            elif e.get("kind") == "subgoal_done":
                add(e.get("step"), "done", e.get("subgoal", "") or "subgoal done", True)
    return ev


def embed(path: Path, w: int, q: int = 55) -> str | None:
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = img.resize((w, round(w * img.height / img.width)))
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def detail(env, chain, order, task, ep: Path, r: dict, bundle: bool) -> dict:
    method = chain.split("-")[0]
    rel = "runs/" + "/".join(ep.parts[-5:])
    clock = Clock(ep)
    kf = sorted((ep / "keyframes").glob("*.jpg")) if (ep / "keyframes").exists() else []
    if bundle and len(kf) > MAX_FRAMES:
        kf = [kf[round(i * (len(kf) - 1) / (MAX_FRAMES - 1))] for i in range(MAX_FRAMES)]
    frames = []
    for p in kf:
        try:
            t = int(p.stem)
        except ValueError:
            continue
        frames.append([t, embed(p, FRAME_W, 50) if bundle else f"{rel}/keyframes/{p.name}"])
    inv, pos, hp = trajectory(ep)
    calls = llm_calls(ep, clock, rel)
    if bundle:
        for c in calls:
            c["img"] = [x for x in (embed(ep / "llm" / "images" / Path(i).name, 320, 60) for i in c["img"][:2]) if x]
    return {"id": ep_id(env, chain, order, task), "env": env, "chain": chain, "order": order, "task": task,
            "seed": ep.name, "text": TASKS[task]["instruction"], "group": TASKS[task]["group"],
            "ok": bool(r.get("success")), "st": r.get("success_step"), "steps": r.get("steps"), "hz": r.get("horizon_steps"),
            "end": r.get("end_reason"), "src": r.get("plan_source"), "usd": round(r.get("cost_usd") or 0, 4),
            "frames": frames, "inv": inv, "pos": pos, "hp": hp, "events": events(method, ep, clock), "llm": calls}


def summary(env, chain, order, task, r: dict, ms: dict) -> dict:
    stage = max([i for i, (_, items) in enumerate(STAGES) if any(m in ms for m in items)] or [0])
    return {"id": ep_id(env, chain, order, task), "env": env, "chain": chain, "order": order, "task": task,
            "group": TASKS[task]["group"], "text": TASKS[task]["instruction"], "ok": bool(r.get("success")),
            "st": r.get("success_step"), "steps": r.get("steps"), "hz": r.get("horizon_steps"), "end": r.get("end_reason"),
            "src": r.get("plan_source"), "dead": bool(r.get("infra_env_died")), "stage": STAGES[stage][0],
            "ms": ms}


def bundle_ids(rows: list) -> set:
    film = (REPRO / "analysis" / "explorer" / "film.js").read_text()
    film = json.loads(film[film.index("=") + 1: film.rindex(";")])
    ids = {ep_id(*k.split("/"), f["order"], f["task"]) for k, v in film.items() for f in v}
    rows = sorted(rows, key=lambda x: (x["order"] != "order0", x["order"], x["task"]))
    for cfg in {(x["env"], x["chain"]) for x in rows}:
        for ok in (True, False):
            n, seen = 0, set()
            for x in rows:
                if (x["env"], x["chain"]) != cfg or x["ok"] != ok or x["group"] in seen or x["dead"]:
                    continue
                seen.add(x["group"]); ids.add(x["id"]); n += 1
                if n >= BUNDLE_PER_KIND:
                    break
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="write only the film.js episodes, frames embedded, into this directory")
    a = ap.parse_args()
    from build_explorer import milestones
    out = Path(a.bundle) if a.bundle else REPRO / "analysis" / "explorer" / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    eps = []
    for rf in sorted(glob.glob(str(RUNS_ROOT / "runs" / "*/*/*/*/*/result.json"))):
        p = Path(rf)
        env, chain, order, task, seed = p.parts[-6:-1]
        if order in SKIP or "." in seed or chain.startswith("steve1"):
            continue
        r = json.loads(p.read_text())
        eps.append((env, chain, order, task, p.parent, r, summary(env, chain, order, task, r, milestones(p.parent))))
    wanted = bundle_ids([e[6] for e in eps]) if a.bundle else None
    rows = []
    for env, chain, order, task, ep, r, row in eps:
        if wanted is not None and row["id"] not in wanted:
            continue
        d = detail(env, chain, order, task, ep, r, bool(a.bundle))
        (out / f"{row['id']}.js").write_text(f"window.__ep({json.dumps(row['id'])},{json.dumps(d, separators=(',', ':'))});\n")
        rows.append(row)
    ids = rows
    (out / "index.js").write_text("window.EP_INDEX = " + json.dumps({"bundle": bool(a.bundle), "total": len(eps), "episodes": rows},
                                                                      separators=(",", ":")) + ";\n")
    size = sum(f.stat().st_size for f in out.glob("*.js"))
    print(f"{len(ids)} episodes, {size / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
