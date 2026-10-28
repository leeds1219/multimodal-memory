"""Episode browser: static web pages to look through episodes by hand.

    python analysis/browse.py baseline/standard gt_world_rule/nolava      # build run_data/site/
    python -m http.server 8800                                            # from the repository root     # open http://<host>:8800/run_data/site/

index.html lists the episodes of each condition (filter by task group, world, success). An episode page shows the
height over time with deaths, the keyframes (every 100 steps), and every turn: command, the VLM's thought, outcome
and what changed. Images are linked from results/, not copied.
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

from benchmark import RESULTS, RUN_DATA, SEEDS  # noqa: E402

WORLDS = {0: "A", 1: "B", 4: "C"}
SITE = RUN_DATA / "site"
CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--ok:#1a7f37;--bad:#c62828;--acc:#0a66c2}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f7;--mut:#a1a1a6;--line:#2c2c2e;--ok:#4ac26b;--bad:#ff6b6b;--acc:#5aa9ff}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}
a{color:var(--acc)} table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid var(--line);padding:4px 6px;
text-align:left;vertical-align:top} th{color:var(--mut);font-weight:600} .ok{color:var(--ok)} .bad{color:var(--bad)}
.mut{color:var(--mut)} .strip{display:flex;gap:4px;overflow-x:auto;padding:6px 0} .strip img{height:90px}
.strip figure{margin:0;text-align:center;font-size:11px;color:var(--mut)} code{font-size:12px} select{margin-right:8px}
td.th{max-width:420px} img.turn{height:72px}
"""


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>")


def rel(target: Path, from_dir: Path) -> str:
    import os
    return os.path.relpath(target, from_dir)


def episodes(cond: str):
    for f in sorted((RESULTS / cond).glob("*/*/*/result.json")):
        order, task, seed = f.parts[-4:-1]
        if not seed.isdigit():
            continue
        r = json.loads(f.read_text())
        if r.get("status") not in ("finished", "anomaly", "crashed_final"):
            continue
        yield f.parent, order, task, int(seed), r


def trajectory(ep: Path):
    ts, ys, deaths, prev = [], [], [], None
    try:
        with gzip.open(ep / "trajectory.jsonl.gz", "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if "pos" in d:
                    p = d["pos"]
                    if prev and sum((a - b) ** 2 for a, b in zip(p, prev)) > 100:
                        deaths.append(d["t"])
                    prev = p
                    ts.append(d["t"])
                    ys.append(p[1])
    except (OSError, EOFError):
        pass
    return ts, ys, deaths


def height_svg(ts, ys, deaths, horizon) -> str:
    if not ts:
        return ""
    W, H = 1100, 160
    lo, hi = min(ys + [0]), max(ys + [70])
    x = lambda t: 40 + (W - 50) * t / max(1, horizon)
    y = lambda v: 10 + (H - 30) * (hi - v) / max(1, hi - lo)
    step = max(1, len(ts) // 1500)
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in list(zip(ts, ys))[::step])
    grid = "".join(f"<line x1='40' x2='{W - 10}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='var(--line)'/>"
                   f"<text x='4' y='{y(v) + 4:.1f}' font-size='10' fill='var(--mut)'>y{v}</text>"
                   for v in (14, 26, 45, 63) if lo <= v <= hi)
    dead = "".join(f"<line x1='{x(t):.1f}' x2='{x(t):.1f}' y1='10' y2='{H - 20}' stroke='var(--bad)' stroke-dasharray='3 3'/>"
                   for t in deaths)
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='height over steps'>{grid}{dead}"
            f"<polyline points='{pts}' fill='none' stroke='var(--acc)' stroke-width='1.2'/>"
            f"<text x='{W - 10}' y='{H - 4}' font-size='10' text-anchor='end' fill='var(--mut)'>step {horizon:,}</text></svg>")


def episode_page(ep: Path, cond: str, order: str, task: str, world: str, r: dict, out: Path) -> None:
    ts, ys, deaths = trajectory(ep)
    kf = sorted((ep / "keyframes").glob("*.jpg"))
    strip = "".join(f"<figure><a href='{rel(k, out.parent)}'><img loading='lazy' src='{rel(k, out.parent)}'></a>"
                    f"<figcaption>{int(k.stem):,}</figcaption></figure>" for k in kf[::max(1, len(kf) // 60)])
    rows = []
    turns = [json.loads(l) for l in open(ep / "vlm" / "turns.jsonl")] if (ep / "vlm" / "turns.jsonl").exists() else []
    kf_steps = [int(k.stem) for k in kf]
    for t in turns:
        near = max((s for s in kf_steps if s <= t["t1"]), default=None)
        img = (f"<img class='turn' loading='lazy' src='{rel(ep / 'keyframes' / f'{near:06d}.jpg', out.parent)}'>"
               if near is not None else "")
        rows.append(f"<tr><td>{t['turn']}</td><td class='mut'>{t['t0']:,}-{t['t1']:,}</td>"
                    f"<td><code>{html.escape(json.dumps(t['cmd']))}</code></td><td class='th'>{html.escape(t.get('thought') or '')}</td>"
                    f"<td>{html.escape(str(t['outcome']))}<div class='mut'>{html.escape(t.get('effect') or '')}</div></td><td>{img}</td></tr>")
    res = "<span class='ok'>success</span>" if r.get("success") else "<span class='bad'>failed</span>"
    head = (f"<p><a href='index.html'>&larr; all episodes</a></p><h1>{task} &middot; world {world}</h1>"
            f"<p>{cond} &middot; {order} &middot; {res} &middot; {r.get('steps', 0):,} / {r.get('horizon_steps', 0):,} steps "
            f"&middot; {len(turns)} turns &middot; {len(deaths)} deaths &middot; ${r.get('cost_usd') or 0:.2f}</p>"
            f"<p class='mut'>Final inventory: {html.escape(json.dumps(r.get('final_inventory') or {}))}</p>")
    body = (head + "<h2>Height</h2>" + height_svg(ts, ys, deaths, r.get("horizon_steps") or (ts[-1] if ts else 1))
            + "<h2>Keyframes</h2><div class='strip'>" + strip + "</div><h2>Turns</h2><table><tr><th>#</th><th>steps</th>"
            "<th>command</th><th>thought</th><th>outcome / effect</th><th>view</th></tr>" + "".join(rows) + "</table>")
    out.write_text(page(f"{task} {world}", body))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="+", help="results/<method>/<world>, e.g. baseline/standard")
    ap.add_argument("--orders", default="", help="comma list of chain-name prefixes to keep")
    a = ap.parse_args()
    pre = [p for p in a.orders.split(",") if p]
    SITE.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in a.conditions:
        for ep, order, task, seed, r in episodes(cond):
            if pre and not any(order.startswith(p) for p in pre):
                continue
            world = WORLDS.get(seed - SEEDS[task], "?")
            name = f"{cond.replace('/', '__')}__{order}__{task}.html"
            episode_page(ep, cond, order, task, world, r, SITE / name)
            rows.append((cond, task.split("_")[1], world, bool(r.get("success")), order, task, r, name))
    trs = "".join(
        f"<tr data-c='{c}' data-g='{g}' data-w='{w}' data-s='{int(s)}'><td>{c}</td><td><a href='{n}'>{t}</a></td><td>{w}</td>"
        f"<td class='{'ok' if s else 'bad'}'>{'yes' if s else 'no'}</td><td>{r.get('steps', 0):,}</td>"
        f"<td>${r.get('cost_usd') or 0:.2f}</td><td class='mut'>{o}</td></tr>" for c, g, w, s, o, t, r, n in rows)
    opts = lambda vals: "".join(f"<option>{v}</option>" for v in sorted(set(vals)))
    filt = (f"<p><select id='c'><option value=''>all conditions</option>{opts(r[0] for r in rows)}</select>"
            f"<select id='g'><option value=''>all groups</option>{opts(r[1] for r in rows)}</select>"
            f"<select id='w'><option value=''>all worlds</option>{opts(r[2] for r in rows)}</select>"
            "<select id='s'><option value=''>success and failure</option><option value='1'>success</option>"
            "<option value='0'>failure</option></select> <span id='n' class='mut'></span></p>")
    js = ("<script>const f=['c','g','w','s'];function u(){let n=0;document.querySelectorAll('tbody tr').forEach(r=>{"
          "const ok=f.every(k=>{const v=document.getElementById(k).value;return !v||r.dataset[k]===v});r.hidden=!ok;n+=ok});"
          "document.getElementById('n').textContent=n+' episodes'}f.forEach(k=>document.getElementById(k).onchange=u);u()</script>")
    body = ("<h1>Episodes</h1>" + filt + "<table><thead><tr><th>condition</th><th>task</th><th>world</th><th>success</th>"
            "<th>steps</th><th>cost</th><th>chain</th></tr></thead><tbody>" + trs + "</tbody></table>" + js)
    (SITE / "index.html").write_text(page("Episodes", body))
    print(f"{len(rows)} episodes -> {SITE / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
