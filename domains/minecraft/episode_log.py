"""Compressed view of one vlmtask episode for System 2.

A timeline first: the turn each item was first obtained, and commands that failed more than once. Then one line per
VLM turn: step range, height y before -> after (from the trajectory's positions), what the command was, how it
ended, the change in position / health / inventory, deaths, and the start of the agent's own thought (what it saw
and meant to do; the full thoughts go in the raw window around a failure point, raw_window()). Reads only
turns.jsonl, trajectory.jsonl.gz and result.json: client.log, which names the ore placements, is never read.

    python domains/minecraft/episode_log.py <episode dir> [--raw T0 T1]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def _positions(ep: Path) -> tuple[list[int], list[list[float]], set[int]]:
    ts, ps, died, prev = [], [], set(), None
    for line in gzip.open(ep / "trajectory.jsonl.gz", "rt"):
        d = json.loads(line)
        if "pos" in d:
            if prev is not None and sum((a - b) ** 2 for a, b in zip(d["pos"], prev)) > 100:
                died.add(d["t"])  # respawn: a jump of more than 10 blocks
            ts.append(d["t"])
            ps.append(d["pos"])
            prev = d["pos"]
    return ts, ps, died


def _pos_at(ts, ps, t):
    import bisect
    i = bisect.bisect_right(ts, t) - 1
    return ps[max(i, 0)] if ps else [0, 0, 0]


def _cmd_short(c) -> str:
    if not isinstance(c, dict):
        return str(c)[:60]
    typ = c.get("type")
    if typ == "keys":
        cam = c.get("camera") or [0, 0]
        cam = f" cam {cam}" if any(cam) else ""
        return f"keys {'+'.join(c.get('keys') or []) or '-'}{cam} x{c.get('steps')}"
    if typ == "steve":
        return f"steve {c.get('prompt')!r} x{c.get('steps')}"
    if typ in ("craft", "smelt"):
        return f"{typ} {c.get('item')} {c.get('n', 1)}"
    if typ == "equip":
        return f"equip {c.get('item')}"
    return json.dumps(c)[:80]


def _effect_short(eff: str) -> str:
    m_hp = re.search(r"health ([\d.]+)->([\d.]+)", eff or "")
    m_inv = re.search(r"inventory change (\{.*?\}|none)", eff or "")
    out = []
    if m_hp and m_hp.group(1) != m_hp.group(2):
        out.append(f"hp {float(m_hp.group(1)):g}->{float(m_hp.group(2)):g}")
    if m_inv and m_inv.group(1) != "none":
        d = json.loads(m_inv.group(1))
        out.append("inv " + ", ".join(f"{k} {v:+d}" for k, v in sorted(d.items())))
    return "; ".join(out)


THOUGHT_CHARS = 160


def _thought(t) -> str:
    th = " ".join(str(t.get("thought") or "").split())
    return f'thought: "{th[:THOUGHT_CHARS]}{"..." if len(th) > THOUGHT_CHARS else ""}"' if th else ""


def _turn_at(turns, t):
    return next((x["turn"] for x in turns if x["t0"] <= t < x["t1"]), turns[-1]["turn"] if turns else 0)


def timeline(parts: list, turns: list) -> list[str]:
    """Facts already in the log, gathered: first time each item was in the inventory; repeated failures.
    `parts`: [(episode, until_step or None)], the attempt from step 0."""
    first = {}
    for ep, until in parts:
        for line in gzip.open(ep / "trajectory.jsonl.gz", "rt"):
            d = json.loads(line)
            if until is not None and d["t"] >= until:
                break
            for k, v in (d.get("inv") or {}).items():
                if v > 0 and k not in first:
                    first[k] = d["t"]
    out = ["Timeline. First time each item was in the inventory (turn, step):",
           "  " + ", ".join(f"{k} T{_turn_at(turns, t)} ({t})" for k, t in sorted(first.items(), key=lambda x: x[1]))]
    fails = {}
    for t in turns:
        if str(t.get("outcome", "")).startswith("failed"):
            fails.setdefault(_cmd_short(t["cmd"]), []).append(t)
    rep = [(c, f) for c, f in fails.items() if len(f) >= 2]
    if rep:
        out.append("Commands that failed more than once:")
        for c, f in sorted(rep, key=lambda x: x[1][0]["t0"]):
            why = sorted({str(x.get("outcome"))[:60] for x in f})
            out.append(f"  {c}: {len(f)} times, T{f[0]['turn']} ({f[0]['t0']}) to T{f[-1]['turn']} ({f[-1]['t0']}); "
                       + " / ".join(why))
    return out


def checkpoints(ep: Path) -> list[int]:
    return sorted(int(p.stem[1:]) for p in (ep / "ckpt").glob("s*.zip")) if (ep / "ckpt").exists() else []


def _rows(ep: Path, turns: list) -> list:
    ts, ps, died = _positions(ep)
    ck = checkpoints(ep)
    rows = []
    for t in turns:
        p0, p1 = _pos_at(ts, ps, t["t0"]), _pos_at(ts, ps, t["t1"])
        d = [f"T{t['turn']}", f"{t['t0']}-{t['t1']}", f"y {p0[1]:.0f}->{p1[1]:.0f}", _cmd_short(t["cmd"]),
             str(t.get("outcome"))[:60]]
        ch = _effect_short(t.get("effect", ""))
        mv = ((p1[0] - p0[0]) ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5
        if mv >= 1:
            ch = f"moved {mv:.0f} blocks sideways" + (f"; {ch}" if ch else "")
        if any(t["t0"] < s <= t["t1"] for s in died):
            ch = "DIED (respawned)" + (f"; {ch}" if ch else "")
        d.append(ch)
        d.append(_thought(t))
        rows.append((t, d, t["t1"] in ck))
    return rows


def compress(ep: Path, parts: list | None = None) -> str:
    """`parts`: the whole attempt from step 0, [(episode, until_step, note)], the last being `ep` (until None): each
    part's turns before until_step, the parts after the first replayed from a save of the one before, `note` printed
    where the part starts. None: `ep` alone."""
    ep = Path(ep)
    parts = [(Path(e), u, n) for e, u, n in parts] if parts else [(ep, None, None)]
    res = json.loads((ep / "result.json").read_text())
    pturns = [[t for t in (json.loads(l) for l in open(e / "vlm" / "turns.jsonl")) if u is None or t["t0"] < u]
              for e, u, _ in parts]
    turns = [t for ts_ in pturns for t in ts_]
    ck = any(checkpoints(e) for e, _, _ in parts)
    summ = json.loads((ep / "vlm" / "summary.json").read_text()) if (ep / "vlm" / "summary.json").exists() else {}
    if len(parts) > 1:
        start = [f"This attempt is shown from step 0: the turns before step {summ.get('start_step', '?')} come from the "
                 f"earlier attempts it was replayed from (marked below)."]
    elif summ.get("start_step"):
        start = [f"This attempt started from a saved moment of an earlier attempt, at step {summ['start_step']} "
                 f"(turn {turns[0]['turn'] - 1 if turns else '?'} of that attempt)."]
    else:
        start = []
    head = [f"Task: {res.get('instruction')}",
            f"Result: {'succeeded' if res.get('success') else 'failed'} after {res.get('steps')} of {res.get('horizon_steps')} steps, "
            f"{len(turns)} turns. Final inventory: {json.dumps(res.get('final_inventory') or {})}",
            *start,
            *timeline([(e, u) for e, u, _ in parts], turns),
            "Turns. Columns: turn | steps | height y start->end | command | how it ended | changes | the agent's thought"
            + (" | [saved] = the world was saved after this turn" if ck else "")]
    body = []
    for (e, _, note), ts_ in zip(parts, pturns):
        if note:
            body.append(f"-- {note} --")
        body += _merge(_rows(e, ts_))
    return "\n".join(head + body)


def _merge(rows) -> list[str]:
    """Runs of turns that changed nothing (same height, no change listed, same command type, no save) -> one line."""
    out, run = [], []

    def flush():
        if len(run) < 3:
            out.extend(" | ".join(x for x in d if x) for _, d, _ in run)
        else:
            (t0, d0, _), (t1, d1, _) = run[0], run[-1]
            kinds = sorted({_cmd_short(t["cmd"]).split(" ")[0] for t, _, _ in run})
            out.append(" | ".join([f"T{t0['turn']}-T{t1['turn']}", f"{t0['t0']}-{t1['t1']}", d0[2],
                                   f"{len(run)} turns of {'/'.join(kinds)}, e.g. {d0[3]}", "nothing changed"]
                                  + ([f"first {d0[6]}"] if d0[6] else [])))
        run.clear()

    for t, d, saved in rows:
        quiet = not d[5] and d[2].split("->")[0][2:] == d[2].split("->")[1] and not saved
        if quiet and run and _cmd_short(run[-1][0]["cmd"]).split(" ")[0] == _cmd_short(t["cmd"]).split(" ")[0] \
                and run[-1][1][2] == d[2]:
            run.append((t, d, saved))
            continue
        flush()
        if quiet:
            run.append((t, d, saved))
        else:
            out.append(" | ".join([x for x in d[:6]] + ([d[6]] if d[6] else []) + (["[saved]"] if saved else [])))
    flush()
    return out


def raw_window(ep: Path, t_from: int, t_to: int) -> list[dict]:
    """The turns t_from..t_to as logged (command, thought, outcome, effect), for the failure point's context."""
    out = []
    for l in open(Path(ep) / "vlm" / "turns.jsonl"):
        t = json.loads(l)
        if t_from <= t["turn"] <= t_to:
            out.append({k: t.get(k) for k in ("turn", "t0", "t1", "cmd", "thought", "outcome", "effect")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ep")
    ap.add_argument("--raw", nargs=2, type=int)
    a = ap.parse_args()
    if a.raw:
        print(json.dumps(raw_window(Path(a.ep), *a.raw), indent=1))
    else:
        print(compress(Path(a.ep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
