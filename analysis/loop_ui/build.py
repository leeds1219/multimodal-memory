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

Two loop.json formats. The pt1-pt3 loops ("iterations": one episode per hypothesis) go to the interactive page
index.html reads from data.json. The current loop ("experiments": milestones, a timeline of parts, failure points
with their kind, S/A/R hypotheses tested by rollouts, a verdict with its frequency k/n) goes to a static page
written here (render_attempts): per attempt the milestones, the timeline parts with their replay markers, and per
experiment its failure point, hypothesis, change, prediction, each rollout's occurrence, the verdict and the
rollout the timeline continued from. An experiment whose loops are all in the current format gets that page.
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


def attempt(L: dict) -> dict:
    """One attempt in the current format, as render_attempts shows it."""
    src = Path(L["source"])
    res0 = json.load(open(src / "result.json")) if (src / "result.json").exists() else {}
    fps = {(f["episode"], f["turn"]): f for f in L.get("failure_points", [])}
    exps = []
    for e in L.get("experiments", []):
        fp = fps.get((e["failure_point"]["episode"], e["failure_point"]["turn"]), {})
        cont_from = None
        if e.get("continuation"):  # the loop continues from the first rollout where R followed
            cont_from = next((i for i, r in enumerate(e["rollouts"], 1)
                              if r.get("occurrence") and r["occurrence"].get("result_followed")), None)
        exps.append({**e, "fp": {**e["failure_point"], "decision": fp.get("decision", ""), "shown": fp.get("shown", ""),
                                 "knowledge": fp.get("knowledge", ""), "checkpoint_step": fp.get("checkpoint_step")},
                     "continued_from": cont_from})
    done = L.get("done") or "running"
    return {"id": f"{L['task']}_{L['seed']}", "task": L["task"], "seed": L["seed"],
            "instruction": res0.get("instruction", ""), "source_steps": res0.get("steps"), "done": done,
            "success": done.startswith("task succeeded"), "settings": L.get("settings") or {},
            "milestones": L.get("milestones") or [], "timeline": L.get("timeline") or [], "experiments": exps}


def render_attempts(exp: str, runs: list) -> str:
    """The static page of the current-format loops of one experiment."""
    e_ = html.escape

    def name(ep):
        return e_(Path(ep).name) if ep else ""

    def rows(head, body):
        return ("<table><tr>" + "".join(f"<th>{e_(h)}</th>" for h in head) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body) + "</table>")

    st = next((r["settings"] for r in runs if r["settings"]), {})
    out = [f"<h1>Loop: {e_(exp)}</h1>",
           "<p class='note'>" + e_(", ".join(f"{k} {v}" for k, v in st.items())) + "</p>" if st else ""]
    for r in runs:
        out.append(f"<section><h2>{e_(r['task'])} <span class='quiet'>seed {r['seed']}</span></h2>"
                   f"<p>{e_(r['instruction'])}. Result: <b>{e_(r['done'])}</b>"
                   + ("" if r["source_steps"] is None else f"; first episode {r['source_steps']:,} steps") + ".</p>")
        out.append("<h3>Milestones</h3>" + (rows(["#", "item", "what for"], [[str(i), e_(m["item"]), e_(m.get("why", ""))]
                                                                         for i, m in enumerate(r["milestones"], 1)])
                                           if r["milestones"] else "<p class='quiet'>not decided yet</p>"))
        out.append("<h3>Timeline</h3>" + rows(
            ["part", "episode", "from step", "until step", "marker"],
            [[str(i), name(p["episode"]), f"{p.get('start_step', 0):,}",
              "end" if p.get("until") is None else f"{p['until']:,}", e_(p.get("marker") or "")]
             for i, p in enumerate(r["timeline"], 1)]))
        for x in r["experiments"]:
            fp, h = x["fp"], x.get("hypothesis", {})
            ch, pr = x.get("change", {}), x.get("prediction", {})
            k, n = (x.get("frequency") or [0, 0])
            ro = [[str(i), name(o["episode"]), "" if o.get("steps") is None else f"{o['steps']:,}",
                   f"T{oc['turn']}" if (oc := o.get("occurrence")) else "none in the window",
                   ("yes" if oc["result_followed"] else "no") if oc else "",
                   e_(oc.get("evidence", "")) if oc else "",
                   ("continued the timeline" if x["continued_from"] == i else "") + (" task succeeded" if o.get("success") else "")]
                  for i, o in enumerate(x["rollouts"], 1)]
            out.append(
                f"<div class='exp'><h4>Experiment {x['k']}: <span class='v {e_(x['verdict'].split()[0].lower())}'>"
                f"{e_(x['verdict'])} {k}/{n}</span></h4>"
                f"<p><b>Failure point</b> T{fp['turn']} in {name(fp['episode'])} ({e_(fp.get('kind', ''))}), restart step "
                f"{'' if fp.get('checkpoint_step') is None else format(fp['checkpoint_step'], ',')}: {e_(fp['decision'])}"
                f"<br><span class='quiet'>Shown: {e_(fp['shown'])}</span></p>"
                f"<p><b>Hypothesis</b> {e_(x.get('statement', ''))}<br>"
                f"<span class='quiet'>S: {e_(h.get('situation', ''))} &middot; A: {e_(h.get('action', ''))} &middot; "
                f"R: {e_(h.get('result', ''))}</span></p>"
                f"<p><b>Change</b> when {e_(ch.get('start', ''))}: {e_(ch.get('do', ''))}; stop when {e_(ch.get('stop', ''))}"
                + (f"<br><span class='quiet'>Plan: {e_(x.get('plan', ''))}</span>" if x.get("plan") else "") + "</p>"
                f"<p><b>Prediction</b> when {e_(pr.get('when', ''))}: {e_(pr.get('what', ''))}</p>"
                + rows(["rollout", "episode", "steps", "occurrence", "R followed", "evidence", ""], ro)
                + (f"<p class='quiet'>Continuation: {name(x['continuation'])}</p>" if x.get("continuation") else "")
                + "</div>")
        out.append("</section>")
    css = (":root{--bg:#fff;--fg:#262626;--quiet:#6b6b6b;--rule:#d9d9d9;--good:#1a7f37;--bad:#b42318;--mid:#9a6700}"
           "@media (prefers-color-scheme:dark){:root{--bg:#1c1c1c;--fg:#e6e6e6;--quiet:#a0a0a0;--rule:#3a3a3a;"
           "--good:#4ac26b;--bad:#f47067;--mid:#d4a72c}}"
           "body{font-family:Arial,sans-serif;margin:24px;max-width:1100px;background:var(--bg);color:var(--fg)}"
           "table{border-collapse:collapse;margin:6px 0 12px}td,th{padding:4px 10px;border-bottom:1px solid var(--rule);"
           "text-align:left;vertical-align:top;font-size:14px}.quiet,.note{color:var(--quiet)}"
           ".exp{border-left:3px solid var(--rule);padding-left:12px;margin:16px 0}.v.supported{color:var(--good)}"
           ".v.refuted{color:var(--bad)}.v.not{color:var(--mid)}section{margin-bottom:40px}")
    return (f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Loop {e_(exp)}</title><style>{css}</style>" + "".join(out)
            + "<p><a href='../index.html'>&larr; all experiments</a></p>")


def build(exp, names, dom, out):
    shutil.rmtree(out / "frames", ignore_errors=True)
    (out / "frames").mkdir(parents=True)

    def frame(ep, step, name):
        f = dom.frame(ep, step)
        if not f:
            return None
        shutil.copy(f, out / "frames" / name)
        return f"frames/{name}"

    files = sorted(p for n in names for p in (LOOPS / n).glob("*/loop.json"))
    current = [L for L in map(lambda f: json.load(open(f)), files) if "experiments" in L]
    if current and len(current) == len(files):  # the current loop format: the static page
        runs = [attempt(L) for L in current]
        for r in runs:
            print(f"  {r['id']}: {len(r['experiments'])} experiments, {r['done']}")
        (out / "data.json").write_text(json.dumps(runs, ensure_ascii=False))
        (out / "index.html").write_text(render_attempts(exp, runs))
        return [{"its": [{"success": r["success"]}] * len(r["experiments"]) or [], "done": r["done"]} for r in runs], {}
    runs, settings, horizon = [], [], None
    for f in files:
        L = json.load(open(f))
        if not L.get("iterations"):
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
    global LOOPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", help="one experiment (default: all)")
    ap.add_argument("--domain", default="minecraft", help="adapter domains/<domain>/loop_ui.py")
    ap.add_argument("--site", default=str(SITE))
    ap.add_argument("--loops", default=str(LOOPS), help="the loop folders (default results/ours_loop)")
    a = ap.parse_args()
    LOOPS = Path(a.loops)
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
