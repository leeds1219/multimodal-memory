"""The log of one vlmtask episode, or of a timeline of episodes, as the Designer reads it (designer/terms.py
LOG_LAYOUT): one turn row per command, with the step range, height y before -> after (from the trajectory's
positions), the command, how it ended, the changes (inventory, health, movement, deaths) and the agent's whole
thought; turns in a row that changed nothing are one line. With milestones, the rows are grouped by segment under a
heading line. Reads only turns.jsonl, trajectory.jsonl.gz and the saves: client.log, which names the ore
placements, is never read.

    python domains/minecraft/episode_log.py <episode dir> [--milestones log,planks:4,...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import zlib
from pathlib import Path


def _trajectory(ep: Path):
    """The trajectory records; of an episode still playing, the ones written so far (the gzip may end mid-block)."""
    try:
        for line in gzip.open(ep / "trajectory.jsonl.gz", "rt"):
            yield json.loads(line)
    except (OSError, EOFError, ValueError, zlib.error):
        return


def _positions(ep: Path) -> tuple[list[int], list[list[float]], set[int]]:
    ts, ps, died, prev = [], [], set(), None
    for d in _trajectory(ep):
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



FAMILIES = ("log", "planks", "wool")  # generic item names: any <kind>_log counts as log (agent/planner.py)


def _thought(t) -> str:
    th = " ".join(str(t.get("thought") or "").split())
    return f'thought: "{th}"' if th else ""


def checkpoints(ep: Path) -> list[int]:
    return sorted(int(p.stem[1:]) for p in (ep / "ckpt").glob("s*.zip")) if (ep / "ckpt").exists() else []


def turns_of(ep: Path, until: int | None = None) -> list:
    """The turns of episode ep that started before step `until` (None: all)."""
    out = [json.loads(l) for l in open(Path(ep) / "vlm" / "turns.jsonl")]
    return [t for t in out if until is None or t["t0"] < until]


def _rows(ep: Path, turns: list) -> list:
    """(turn, [turn, steps, height, command, how it ended, changes, thought]) per turn."""
    ts, ps, died = _positions(ep)
    rows = []
    for t in turns:
        p0, p1 = _pos_at(ts, ps, t["t0"]), _pos_at(ts, ps, t["t1"])
        d = [f"T{t['turn']}", f"{t['t0']}-{t['t1']}", f"y {p0[1]:.0f}->{p1[1]:.0f}", _cmd_short(t["cmd"]),
             str(t.get("outcome"))]
        ch = _effect_short(t.get("effect", ""))
        mv = ((p1[0] - p0[0]) ** 2 + (p1[2] - p0[2]) ** 2) ** 0.5
        if mv >= 1:
            ch = f"moved {mv:.0f} blocks sideways" + (f"; {ch}" if ch else "")
        if any(t["t0"] < s <= t["t1"] for s in died):
            ch = "DIED (respawned)" + (f"; {ch}" if ch else "")
        d += [ch, _thought(t)]
        rows.append((t, d))
    return rows


def _gains(t) -> dict:
    """Items the turn added to the inventory: {item: n}."""
    m = re.search(r"inventory change (\{.*?\})", t.get("effect") or "")
    return {k: v for k, v in json.loads(m.group(1)).items() if v > 0} if m else {}


def _is(item_id: str, milestone: str) -> bool:
    return item_id == milestone or (milestone in FAMILIES and item_id.endswith("_" + milestone))


def _held(ep: Path, turns: list) -> list:
    """The inventory {item: n} at the end of each turn, from the trajectory."""
    out, inv, recs = [], {}, iter(_trajectory(ep))
    nxt = next(recs, None)
    for t in turns:
        while nxt is not None and nxt["t"] <= t["t1"]:
            inv = nxt.get("inv", inv)
            nxt = next(recs, None)
        out.append(inv)
    return out


def _has(inv: dict, milestone: tuple) -> bool:
    item, n = milestone
    return sum(v for k, v in inv.items() if _is(k, item)) >= n


def _merge(rows, keep: set) -> list[str]:
    """Runs of 3+ turns that changed nothing (same height, no change listed, same command type) -> one line. A turn
    in `keep` (a marker goes before it) starts a new run."""
    out, run = [], []

    def flush():
        if len(run) < 3:
            out.extend(" | ".join(d) for _, d in run)
        else:
            (t0, d0), (t1, _) = run[0], run[-1]
            kinds = sorted({_cmd_short(t["cmd"]).split(" ")[0] for t, _ in run})
            out.append(" | ".join([f"T{t0['turn']}-T{t1['turn']}", f"{t0['t0']}-{t1['t1']}", d0[2],
                                   f"{len(run)} turns of {'/'.join(kinds)}, e.g. {d0[3]}", "nothing changed"]))
        run.clear()

    for t, d in rows:
        quiet = not d[5] and d[2].split("->")[0][2:] == d[2].split("->")[1] and t["turn"] not in keep
        if quiet and run and _cmd_short(run[-1][0]["cmd"]).split(" ")[0] == _cmd_short(t["cmd"]).split(" ")[0] \
                and run[-1][1][2] == d[2]:
            run.append((t, d))
            continue
        flush()
        if quiet:
            run.append((t, d))
        else:
            out.append(" | ".join(d))
    flush()
    return out


def segments(turns: list, held: list, milestones: list) -> list:
    """The turns cut where each milestone (item, count) is first held, `held` the inventory after each turn:
    [(turns, milestone or None at the timeline end)]."""
    left, segs, cur = list(milestones), [], []
    for t, inv in zip(turns, held):
        cur.append(t)
        got = [m for m in left if _has(inv, m)]
        if got:
            for m in got:
                left.remove(m)
            segs.append((cur, got[-1]))
            cur = []
    if cur:
        segs.append((cur, None))
    return segs


def milestones_reached(parts: list, milestones: list) -> int:
    """How far the turns of the parts [(episode dir, until_step, marker)] got: 1 + the order of the furthest milestone
    gained (a later milestone counts the ones before it as reached), 0 when none."""
    got = [i for ep, until, _ in parts for inv in _held(Path(ep), turns_of(Path(ep), until))
           for i, m in enumerate(milestones) if _has(inv, m)]
    return max(got) + 1 if got else 0


def render(parts: list, milestones: list | None = None, markers: dict | None = None) -> str:
    """The log of a timeline: `parts` [(episode, until_step or None, marker line or None)], each part's turns before
    until_step, the marker printed where the part starts. `milestones` ((item, count), in order): rows grouped by
    segment under heading lines; None: rows only. `markers` {turn: [lines]}: printed just before that turn."""
    markers = {k: list(v) for k, v in (markers or {}).items()}
    rows, turns, held = [], [], []
    for ep, until, mark in parts:
        ts_ = turns_of(Path(ep), until)
        if mark and ts_:
            markers.setdefault(ts_[0]["turn"], []).insert(0, mark)
        rows += _rows(Path(ep), ts_)
        turns += ts_
        held += _held(Path(ep), ts_) if milestones is not None else []
    by_turn = {t["turn"]: (t, d) for t, d in rows}

    def lines(ts_):
        out, run = [], []
        for t in ts_:
            if t["turn"] in markers:
                out += _merge(run, set())
                run = []
                out += markers[t["turn"]]
            run.append(by_turn[t["turn"]])
        return out + _merge(run, set())

    if milestones is None:
        return "\n".join(lines(turns))
    out = []
    for n, (ts_, m) in enumerate(segments(turns, held, milestones), 1):
        ended = f"ended with milestone {m[0]} {m[1]}" if m else "ended at the timeline end"
        out.append(f"== Segment {n}: T{ts_[0]['turn']}-T{ts_[-1]['turn']}, steps {ts_[0]['t0']}-{ts_[-1]['t1']} "
                   f"({ts_[-1]['t1'] - ts_[0]['t0']} steps), {ended} ==")
        out += lines(ts_)
    return "\n".join(out)


def state_line(ep: Path, step: int, window: int) -> str:
    """The restart point as the Designer is told it: step, the turn before it, the window, height, inventory."""
    ep = Path(ep)
    before = [t for t in turns_of(ep) if t["t1"] <= step]
    ts, ps, _ = _positions(ep)
    inv = {}
    for d in _trajectory(ep):
        if d["t"] > step:
            break
        inv = d.get("inv", inv)
    inv_s = ", ".join(f"{k} {v}" for k, v in sorted(inv.items()) if v) or "empty"
    after = f"after T{before[-1]['turn']}" if before else "before the first turn"
    return f"step {step} ({after}), window {window} steps, y {_pos_at(ts, ps, step)[1]:.0f}, inventory: {inv_s}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ep")
    ap.add_argument("--milestones", default=None, help="comma list of item[:count]: group the rows by segment")
    a = ap.parse_args()
    ms = [(m.split(":")[0], int(m.split(":")[1]) if ":" in m else 1) for m in a.milestones.split(",")] \
        if a.milestones else None
    print(render([(Path(a.ep), None, None)], ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
