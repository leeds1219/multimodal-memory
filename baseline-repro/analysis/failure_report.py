"""Failure viewer for an Optimus-1 run (suite_optimus1): one self-contained HTML page listing every
failed episode with where and why it failed, keyframes, depth over time, the sub-goal timeline,
reflector verdicts and escapes, plus Claude's analysis and a notes box for the reader.

    python analysis/failure_report.py                       # final eval -> analysis/out/failures.html
    python analysis/failure_report.py --variant <chain> --prefix pf2 --out analysis/out/pf2.html
    python analysis/failure_report.py --all                 # successes too (for comparison)

Inputs per episode (runs/O/<chain>/<order>/<task>/<seed>/): result.json, client.log (monitors dump,
reflector verdicts, escapes, ore spawns), trajectory.jsonl.gz (pos, hp), keyframes/.
Hand-written notes: analysis/failure_notes.yaml ({episode_id: text}, optional) are shown as
"Claude's notes" beside the generated analysis. Reader notes live in the published artifact's db
(collection `notes`, doc id = episode id); Claude reads them back with the Artifact tool's read_db.
"""
from __future__ import annotations

import argparse
import ast
import base64
import collections
import gzip
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from final_failures import OVERMATCH, PAPER, ROOT, cause, episode, messages  # noqa: E402
from subgoal_steps import kind  # noqa: E402

KIND_STAGE = {"mine diamond": "diamond", "mine gold": "gold_ore", "mine iron": "iron_ore", "mine redstone": "redstone",
              "mine coal": "coal", "mine cobblestone": "cobblestone", "chop wood": "wood", "dig sand": "sand",
              "craft": "craft", "smelt": "smelt"}

DEFAULT_VARIANT = "optimus1-prebuilt-logfix-memfix-craftfix-tagfix-promptfix-replanfix-escapefix5-isoworld-g38"
N_FRAMES = 8
GROUP_ORDER = ["wooden", "stone", "iron", "gold", "diamond", "redstone", "armor"]


def frames(d: Path, steps: int) -> list[dict]:
    from PIL import Image
    kf = sorted((d / "keyframes").glob("*.jpg"))
    if not kf:
        return []
    want = [round(i * (len(kf) - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)]
    out = []
    for i in dict.fromkeys(want):
        im = Image.open(kf[i]).convert("RGB").resize((224, 126))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=62)
        out.append({"step": int(kf[i].stem), "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
    return out


def track(d: Path) -> tuple[list, list, int]:
    """(t, y) sampled to <= 240 points, death steps (hp jumps back up), steps played."""
    pts, deaths, hp_prev, last = [], [], None, 0
    for line in gzip.open(d / "trajectory.jsonl.gz", "rt"):
        x = json.loads(line)
        last = x["t"]
        if "pos" in x:
            pts.append((x["t"], round(x["pos"][1], 1)))
        if "hp" in x:
            if hp_prev is not None and x["hp"] >= 19.5 and x["hp"] > hp_prev + 8:  # respawn at full health
                deaths.append(x["t"])
            hp_prev = x["hp"]
    k = max(1, len(pts) // 240)
    return pts[::k] + pts[-1:], deaths, last


def subgoals(msgs: list[str]) -> list[dict]:
    dump = [m for m in msgs if m.startswith("{'") and "SuccessMonitor" in m]
    if not dump:
        return []
    try:
        mon = ast.literal_eval(re.sub(r"\s+", " ", dump[-1]))
    except (SyntaxError, ValueError):
        return []
    out = []
    for key, v in mon.items():
        sub, _, idx = key.rpartition("_")
        out.append({"name": sub, "done": bool(v.get("SuccessMonitor")), "steps": int(v.get("StepMonitor", 0))})
    return out


def events(msgs: list[str]) -> list[str]:
    ev = []
    for m in msgs:
        if m.startswith("Current Task:"):
            ev.append("▶ " + m[len("Current Task: "):])
        elif m.startswith("Enviroment:"):
            ev.append("◆ reflector — " + m.replace("Enviroment:", "env:"))
        elif m.startswith("escapefix:"):
            ev.append("⇡ " + m)
        elif "could not be opened" in m or "Stuck...." in m or "Return to ground" in m:
            ev.append("✖ " + m[:140])
    # collapse consecutive repeats
    out = []
    for e in ev:
        if out and out[-1][0] == e:
            out[-1][1] += 1
        else:
            out.append([e, 1])
    return [e if n == 1 else f"{e}  ×{n}" for e, n in out]


def analysis(e: dict, sg: list, ys: list, deaths: list, spawns: collections.Counter, msgs: list[str]) -> list[str]:
    """Claude's reading of the episode, generated from the evidence by fixed rules."""
    s = []
    horizon, played = e["horizon"], e["steps"] or 0
    done = [g for g in sg if g["done"]]
    stuck = [g for g in sg if not g["done"]]
    tot = sum(g["steps"] for g in sg) or 1
    if stuck:
        big = max(stuck, key=lambda g: g["steps"])
        s.append(f"막힌 곳: \"{big['name']}\" — 이 sub-goal에 {big['steps']:,} step "
                 f"(monitor가 기록한 step의 {100 * big['steps'] / tot:.0f}%). 완료한 sub-goal {len(done)}/{len(sg)}개.")
        if played and tot < 0.6 * played:
            s.append(f"monitor 기록은 {tot:,} / {played:,} step뿐 — 나머지는 제작 helper·탈출 스크립트 등 monitor 밖에서 쓰임.")
    elif e["last_subgoal"]:
        s.append(f"마지막 sub-goal: \"{e['last_subgoal']}\" (monitor 기록 없음).")
    if e["end_reason"] == "horizon":
        s.append(f"{horizon:,} step 제한에 걸려 종료.")
    elif e["end_reason"]:
        s.append(f"종료 사유: {e['end_reason']}.")
    if ys:
        y0, ymin = ys[0][1], min(y for _, y in ys)
        below14 = sum(1 for _, y in ys if y <= 14) / len(ys)
        line = f"시작 y={y0:.0f}, 최저 y={ymin:.0f}"
        if below14 > 0.05:
            line += f", 기록된 위치의 {100 * below14:.0f}%를 y≤14(다이아 층)에서 보냄"
        s.append(line + ".")
    rep = e["replans"]
    if rep:
        preds = collections.Counter(re.findall(r"Predicament: (\w+)", "\n".join(msgs)))
        s.append(f"reflector가 REPLAN {rep}회 판정 (" + ", ".join(f"{k} {v}" for k, v in preds.most_common()) + ").")
    if e["escapes"]:
        s.append(f"탈출 스크립트 {e['escapes']}회 실행, 빠져나옴 {e['escape_ok']}회.")
    if deaths:
        s.append(f"사망(리스폰) 추정 {len(deaths)}회.")
    if spawns:
        s.append("환경이 생성한 광석: " + ", ".join(f"{k} {v}" for k, v in spawns.most_common()) + ".")
    c = e["cause"]
    inv = e["inventory"]
    picks = [k for k in inv if k.endswith("pickaxe")]
    if c.startswith("ore not found"):
        ore = e["stage"]
        msg = f"판단: {ore} 단계에서 시간 초과."
        if ore == "diamond" and spawns.get("diamond") and "iron_pickaxe" in inv and not inv.get("diamond"):
            msg += (f" 철곡괭이를 갖고 다이아 광석이 {spawns['diamond']}개나 생성됐는데 다이아가 0개 → "
                    "생성 위치를 STEVE-1이 못 지나갔거나, 손에 든 도구가 철곡괭이가 아니었을 가능성 "
                    "(find_best_pickaxe가 0번 슬롯/핫바 밖 곡괭이를 못 고름, 미검증).")
        elif ore == "gold_ore" and ys and min(y for _, y in ys) <= 10:
            msg += (f" 금 광석이 {spawns.get('gold', 0)}개 생성됐지만 금을 얻지 못했고, 금이 생성되는 y15–26보다 "
                    "깊이 내려가 오래 머묾 (dig down이 금 층을 지나침).")
        if "trap" in c:
            msg += " 그 전에 구덩이/물에서 시간을 잃음."
        s.append(msg)
    elif c.startswith("terrain trap"):
        noblk = "build_tower (0 placed" in "\n".join(msgs)
        s.append("판단: 지형에 갇힘 (구덩이·협곡·물). " + ("쌓을 블록이 없어 탈출 스크립트가 블록을 못 놓음. " if noblk else "")
                 + "reflector는 매번 같은 predicament를 반복 판정.")
    elif c.startswith("slow gathering"):
        s.append(f"판단: 재료 채집이 느림 ({e['stage']}). 막힌 위치/수량을 보면 나무·돌이 드문 지형.")
    elif c.startswith("crafting-table GUI"):
        s.append("판단: 제작대를 놓았지만 GUI가 열리지 않음 (물가/좁은 곳). craftfix 재시도도 실패하며 시간을 소모.")
    elif c.startswith("craft/smelt"):
        s.append(f"판단: 제작/제련 단계 실패 (missing material {e['missing_material']}회, GUI 실패 {e['gui_fail']}회).")
    elif c.startswith("ours: promptfix"):
        s.append("판단: 우리 promptfix v1 버그 — 광물이 아닌 목표를 'dig down and mine …'으로 바꿔 STEVE-1이 엉뚱하게 팜 (D45에서 수정).")
    elif c.startswith("ours: replan"):
        s.append("판단: 우리 replanfix 한계 — replan이 만든 'pillar up/select block' sub-goal의 목표가 dirt 수량이라, "
                 "기둥을 쌓을수록 dirt가 줄어 절대 완료되지 않음.")
    else:
        s.append(f"판단: 기타 ({c}).")
    if picks:
        s.append("최종 곡괭이: " + ", ".join(picks) + ".")
    return s


def build(variant: str, prefix: str, include_success: bool) -> dict:
    notes = {}
    nf = HERE / "failure_notes.yaml"
    if nf.exists():
        import yaml
        notes = yaml.safe_load(nf.read_text()) or {}
    eps = []
    for res in sorted(ROOT.glob(f"{variant}/{prefix}*/*/*/result.json")):
        d = res.parent
        if "." in d.name:
            continue
        e = episode(d)
        if e["success"] and not include_success:
            continue
        msgs = messages(d / "client.log") if (d / "client.log").exists() else []
        sg = subgoals(msgs)
        stuck = [g for g in sg if not g["done"]]
        big = max(stuck, key=lambda g: g["steps"])["name"].lower() if stuck else ""
        if e["dirt_goal"] and big and not re.search(r"pillar|select|block|dirt", big):
            e["dirt_goal"] = 0
            e["stage"] = KIND_STAGE.get(kind(big), e["stage"])
        if e["pf_overmatch"] and big and not OVERMATCH.search("dig down and mine " + big.split("mine ", 1)[-1] + ","):
            e["pf_overmatch"] = 0
            e["stage"] = KIND_STAGE.get(kind(big), e["stage"])
        e["cause"] = cause(e) if not e["success"] else "success"
        ys, deaths, played = track(d) if (d / "trajectory.jsonl.gz").exists() else ([], [], 0)
        raw = (d / "client.log").read_text(errors="replace") if (d / "client.log").exists() else ""
        spawns = collections.Counter(re.findall(r"(\w+) ore at -?\d+", raw))  # short print lines, not in msgs
        eid = f"{e['order']}__{e['task']}__{e['seed']}"
        eps.append({
            "id": eid, **{k: e[k] for k in ("order", "task", "group", "seed", "instruction", "success", "steps",
                                            "horizon", "end_reason", "last_subgoal", "stage", "cause", "replans",
                                            "escapes", "escape_ok", "in_water", "gui_fail", "llm_calls", "cost")},
            "inventory": e["inventory"], "subgoals": sg, "y": ys, "deaths": deaths, "spawns": dict(spawns),
            "events": events(msgs)[:160], "frames": frames(d, played),
            "analysis": analysis(e, sg, ys, deaths, spawns, msgs), "note": notes.get(eid, ""),
        })
    eps.sort(key=lambda x: (GROUP_ORDER.index(x["group"]) if x["group"] in GROUP_ORDER else 9, x["task"], x["order"]))
    allres = [json.loads(p.read_text()) for p in ROOT.glob(f"{variant}/{prefix}*/*/*/result.json") if "." not in p.parent.name]
    groups = []
    for g in GROUP_ORDER:
        v = [r for r in allres if r["group"] == g]
        if v:
            groups.append({"group": g, "n": len(v), "succ": sum(bool(r["success"]) for r in v), "paper": PAPER[g]})
    return {"variant": variant, "prefix": prefix, "episodes": eps, "groups": groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--all", action="store_true", help="include successful episodes")
    ap.add_argument("--out", default=str(HERE / "out" / "failures.html"))
    a = ap.parse_args()
    data = build(a.variant, a.prefix, a.all)
    html = (HERE / "failure_report.html").read_text().replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{len(data['episodes'])} episodes -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
