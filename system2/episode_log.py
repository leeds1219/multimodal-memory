"""Compressed view of one vlmtask episode for System 2.

One line per VLM turn: step range, height y before -> after (from the trajectory's positions), what the command was,
how it ended, the change in position / health / inventory, deaths. No VLM thoughts (those go in the raw window
around a failure point, raw_window()). Reads only turns.jsonl, trajectory.jsonl.gz and result.json: client.log,
which names the ore placements, is never read.

    python system2/episode_log.py <episode dir> [--raw T0 T1]
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


def checkpoints(ep: Path) -> list[int]:
    return sorted(int(p.stem[1:]) for p in (ep / "ckpt").glob("s*.zip")) if (ep / "ckpt").exists() else []


def compress(ep: Path) -> str:
    ep = Path(ep)
    res = json.loads((ep / "result.json").read_text())
    turns = [json.loads(l) for l in open(ep / "vlm" / "turns.jsonl")]
    ts, ps, died = _positions(ep)
    ck = checkpoints(ep)
    summ = json.loads((ep / "vlm" / "summary.json").read_text()) if (ep / "vlm" / "summary.json").exists() else {}
    head = [f"Task: {res.get('instruction')}",
            f"Result: {'succeeded' if res.get('success') else 'failed'} after {res.get('steps')} of {res.get('horizon_steps')} steps, "
            f"{len(turns)} turns. Final inventory: {json.dumps(res.get('final_inventory') or {})}",
            *([f"This attempt started from a saved moment of an earlier attempt, at step {summ['start_step']} "
               f"(turn {turns[0]['turn'] - 1 if turns else '?'} of that attempt)."] if summ.get("start_step") else []),
            "Columns: turn | steps | height y start->end | command | how it ended | changes"
            + (" | [saved] = the world was saved after this turn" if ck else "")]
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
        rows.append((t, d, t["t1"] in ck))
    return "\n".join(head + _merge(rows))


def _merge(rows) -> list[str]:
    """Runs of turns that changed nothing (same height, no change listed, same command type, no save) -> one line."""
    out, run = [], []

    def flush():
        if len(run) < 3:
            out.extend(" | ".join(d) for _, d, _ in run)
        else:
            (t0, d0, _), (t1, d1, _) = run[0], run[-1]
            kinds = sorted({_cmd_short(t["cmd"]).split(" ")[0] for t, _, _ in run})
            out.append(" | ".join([f"T{t0['turn']}-T{t1['turn']}", f"{t0['t0']}-{t1['t1']}", d0[2],
                                   f"{len(run)} turns of {'/'.join(kinds)}, e.g. {d0[3]}", "nothing changed"]))
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
            out.append(" | ".join(d + (["[saved]"] if saved else [])))
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
