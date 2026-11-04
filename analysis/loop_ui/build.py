"""Build the loop results page (pipeline, algorithm, per-run retry chain, each stage's real input/output) from the logs.

    python analysis/loop_ui/build.py                       # every experiment in results/ours_loop -> run_data/site/loop/
    python analysis/loop_ui/build.py --exp pt2-pt3         # one experiment (a name in groups.json, or a folder name)

An experiment = the loop folders results/ours_loop/<name>/<task>_<seed>/ of one or more names; groups.json joins
names that are one experiment run in parts. Each experiment goes to run_data/site/loop/<experiment>/ (index.html,
data.json, frames/), with run_data/site/loop/index.html listing them; loops still running are included as they
are. Look at it next to the episode browser (analysis/browse.py links loop/ from its index):
    python -m http.server 8800    # from the repository root; open http://<host>:8800/run_data/site/loop/
or publish one experiment folder as an artifact: page = index.html, files = data.json, meta.json, frames/*.

What is read: loop.json (the loop's own record: failure points, hypotheses, verdicts, restart steps, settings) and,
through the domain adapter (domains/<domain>/loop_ui.py, chosen with --domain), each episode's result, turns, progress and
keyframes. Loops run before the settings were recorded (2026-11-03) are marked: they stopped at "no failure point
left" instead of listing the failure points again in the latest retry's log.
"""
import argparse
import glob
import html
import importlib
import json
import re
import shutil
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPRO))  # the domain packages, domains/<name>/
HERE = Path(__file__).parent
LOOPS = REPRO / "results" / "ours_loop"
SITE = REPRO / "run_data" / "site" / "loop"
DEFAULT = {"max_iter": 10, "tries": 3}  # the loop's defaults before loop.json recorded its settings


def experiments():
    """{experiment: [loop folder names]}: groups.json first, then every other folder with loops on its own"""
    groups = json.loads((HERE / "groups.json").read_text())
    taken = {n for ns in groups.values() for n in ns}
    rest = sorted(p.parent.parent.name for p in LOOPS.glob("*/*/loop.json"))
    for n in rest:
        if n not in taken:
            groups.setdefault(n, [n])
    return {e: ns for e, ns in groups.items() if any((LOOPS / n).glob("*/loop.json"))}


def build(exp, names, dom, out):
    shutil.rmtree(out / "frames", ignore_errors=True)
    (out / "frames").mkdir(parents=True)

    def frame(ep, step, name):
        f = dom.frame(ep, step)
        if not f:
            return None
        shutil.copy(f, out / "frames" / name)
        return f"frames/{name}"

    runs, settings, horizon = [], [], None
    for f in sorted(p for n in names for p in (LOOPS / n).glob("*/loop.json")):
        L = json.load(open(f))
        if not L["iterations"]:
            continue
        settings.append(L.get("settings"))
        src = Path(L["source"])
        res0 = json.load(open(src / "result.json"))
        H = res0.get("horizon_steps") or 36000
        horizon = horizon or H
        g = dom.goal(res0)
        rid = f"{L['task']}_{L['seed']}"
        o = dom.episode(src, g, H)
        orig = {"start": 0, "end": res0["steps"], "success": bool(res0.get("success")), **o}
        fps = {(p["turn"], p["checkpoint_step"]): p for p in L["failure_points"]}
        src_turns = dom.turns(src)
        its = []
        for it in L["iterations"]:
            ep = Path(it["episode"])
            res = json.load(open(ep / "result.json"))
            fp = fps.get((it["failure_point_turn"], it["checkpoint_step"]), {})
            fp_ep = Path(fp.get("episode", src))
            ft = src_turns if fp_ep == src else dom.turns(fp_ep)
            tt = dom.turns(ep)
            k, fpk, ck = it["k"], it["failure_point_turn"], it.get("check_turn")
            fpt = ft.get(fpk)
            ct = tt.get(ck) if ck is not None else None
            end = res.get("success_step") or res.get("steps") or H
            h = it["hypothesis"]
            its.append({
                "k": k, "parent": "orig" if fp_ep == src else int(fp_ep.name[2:]), "start": it["checkpoint_step"],
                "end": end, "success": it["task_success"], "verdict": it["verdict"],
                "fp": {"turn": fpk, "step": fpt["t0"] if fpt else None, "decision": fp.get("decision", ""),
                       "better": fp.get("better", ""), "reason": fp.get("reason", ""), "rec": fpt,
                       "frame": frame(fp_ep, fpt["t1"], f"{rid}_it{k}_fp.jpg") if fpt else None},
                "check": {"turn": ck, "step": ct["t0"] if ct else None, "rec": ct,
                          "frame": frame(ep, ct["t1"], f"{rid}_it{k}_check.jpg") if ct else None},
                "hyp": {x: h.get(x, "") for x in ("problem", "hypothesis", "change", "check_when", "prediction", "plan")},
                "fp_ctx": [ft[j] for j in range(fpk - 3, fpk + 2) if j in ft],
                "check_ctx": [tt[j] for j in range(ck - 2, ck + 1) if j in tt] if ck is not None else [],
                "fails": [x for x in tt.values() if str(x.get("outcome", "")).startswith("failed")][:8],
                "n_fails": sum(str(x.get("outcome", "")).startswith("failed") for x in tt.values()),
                "observed": it.get("observed", ""), "reason": it.get("reason", ""), **dom.episode(ep, g, H),
                "end_frame": frame(ep, end - 1, f"{rid}_it{k}_end.jpg")})
        runs.append({"id": rid, "task": L["task"], "seed": L["seed"], "world": dom.world(src),
                     "instruction": res0.get("instruction", ""), "item": g["label"], "unit": g["unit"],
                     "unit_short": g["unit_short"], "need": g["need"], "needs": g["needs"], "orig": orig, "its": its,
                     "done": L.get("done") or "running", "knowledge": [x["statement"] for x in L["knowledge"]],
                     "orig_end_frame": frame(src, res0["steps"] - 1, f"{rid}_orig_end.jpg")})
        print(f"  {rid} {g['label']}: {len(its)} retries, {runs[-1]['done']}")
    st = next((s for s in settings if s), DEFAULT)
    horizon = horizon or 36000
    meta = {"experiment": exp, "old_rule": any(s is None for s in settings), "horizon": horizon, **st}
    (out / "data.json").write_text(json.dumps(runs, ensure_ascii=False))
    (out / "meta.json").write_text(json.dumps(meta))
    page = (HERE / "index.html").read_text()
    for a, b in (("{{H}}", str(horizon)), ("{{H_TXT}}", f"{horizon:,}"), ("{{N}}", str(st["max_iter"])),
                 ("{{M}}", str(st["tries"]))):
        page = page.replace(a, b)
    (out / "index.html").write_text(page)
    return runs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", help="one experiment (default: all)")
    ap.add_argument("--domain", default="minecraft", help="adapter domains/<domain>/loop_ui.py")
    ap.add_argument("--site", default=str(SITE))
    a = ap.parse_args()
    dom = importlib.import_module(f"domains.{a.domain}.loop_ui")
    site = Path(a.site)
    exps = experiments()
    if a.exp:
        exps = {a.exp: exps.get(a.exp, [a.exp])}
    rows = []
    for e, names in exps.items():
        print(f"{e} ({', '.join(names)})")
        runs, meta = build(e, names, dom, site / e)
        ok = sum(any(i["success"] for i in r["its"]) for r in runs)
        rows.append(f"<tr><td><a href='{html.escape(e)}/index.html'>{html.escape(e)}</a></td><td>{html.escape(', '.join(names))}</td>"
                    f"<td>{len(runs)}</td><td>{ok}</td><td>{sum(len(r['its']) for r in runs)}</td>"
                    f"<td>{sum(r['done'] == 'running' for r in runs)}</td></tr>")
    if not a.exp:
        (site / "index.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Loop experiments</title><style>body{font-family:Arial,sans-serif;"
            "margin:24px;color:#262626}td,th{padding:4px 12px;border-bottom:1px solid #bfbfbf;text-align:left}"
            "a{color:#7e0015;font-weight:700}</style><h1>Loop experiments</h1><table><tr><th>experiment</th>"
            "<th>loop folders</th><th>failed baseline runs</th><th>Task success</th><th>retries</th><th>running</th></tr>"
            + "".join(rows) + "</table><p><a href='../index.html'>&larr; all episodes</a></p>")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
