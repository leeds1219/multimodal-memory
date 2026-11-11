"""Offline test of the Optimus-1 memory (memory/optimus1_memory.py): no game, no LLM call.

Replays finished Baseline episodes (vlm/turns.jsonl) through the memory's hooks in a fresh store: the HDKG is learned
from their craft/smelt/gather turns, the AMEP from reflection windows whose verdicts are made up from the log (the
Baseline gave none: sub-goal = the next item the episode gained, COMPLETE when it arrived). Then checks retrieval,
the system prompt, the frozen switch and the file lock with parallel writers.

    python scripts/test_optimus1_memory.py [--results results/baseline/nolava_save600] [--episodes 12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

import optimus1_memory as O  # noqa: E402
import prompts as P  # noqa: E402
from benchmark import load_yaml  # noqa: E402

TARGETS = load_yaml("task_targets.yaml")["targets"]
STATE = {"pos": [0.0, 64.0, 0.0], "hp": 20.0}


def turns_of(ep: Path) -> list:
    """(turn, inventory before, inventory after) with the inventory rebuilt from the logged changes."""
    inv, out = {}, []
    for line in (ep / "vlm" / "turns.jsonl").read_text().splitlines():
        d = json.loads(line)
        m = re.search(r"inventory change (\{.*?\})", d["effect"])
        dv = json.loads(m.group(1)) if m else {}
        inv1 = {k: v for k, v in ((k, inv.get(k, 0) + dv.get(k, 0)) for k in set(inv) | set(dv)) if v > 0}
        out.append((d, dict(inv), inv1, dv))
        inv = inv1
    return out


def replay(store: Path, ep: Path, frames: list) -> O.Optimus1Memory:
    task_id = ep.parent.name
    target = TARGETS[task_id]["any_of"][0]
    instr = json.loads((ep / "result.json").read_text()).get("instruction", task_id) if (ep / "result.json").exists() else task_id
    mem = O.Optimus1Memory(str(store), instr, task_id, target, int(ep.name), "test")
    rows = turns_of(ep)
    gains = [(i, k) for i, (_, _, _, dv) in enumerate(rows) for k, v in dv.items() if v > 0]
    prev_goal = None
    for i, (d, inv0, inv1, dv) in enumerate(rows):
        nxt = next((k for j, k in gains if j >= i), None)
        goal = f"get {nxt}" if nxt else "finish the task"
        verdict = "COMPLETE" if prev_goal and i and prev_goal[4:] in rows[i - 1][3] else "CONTINUE"
        raw = json.dumps({"thought": "", "reflection": verdict, "subgoal": goal, "command": d["cmd"]})
        mem.turn_content(d["t0"])
        mem.after_answer(raw, frames[i % len(frames)], STATE, inv0, d["t0"])
        mem.after_turn(d["cmd"], d["outcome"], inv0, inv1)
        prev_goal = goal
    success = json.loads((ep / "vlm" / "summary.json").read_text())["success"]
    mem.finish(success, STATE, rows[-1][2] if rows else {}, rows[-1][0]["t1"] if rows else 0)
    return mem


def _writer(store, frame, k):
    mem = O.Optimus1Memory(store, "parallel", f"par_{k}", "stick", k)
    mem.after_answer(json.dumps({"reflection": "", "subgoal": f"get stick {k}"}), frame, STATE, {}, 0)
    mem.after_turn({"type": "craft", "item": "stick", "n": 4}, "done", {"oak_planks": 2}, {"stick": 4})
    mem.finish(True, STATE, {"stick": 4}, 50)


def digest(store: Path) -> str:
    return hashlib.md5(b"".join((store / f).read_bytes() for f in ("hdkg.json", "amep.json"))).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(paths.RESULTS / "baseline" / "nolava_save600"))
    ap.add_argument("--episodes", type=int, default=12)
    a = ap.parse_args()
    eps = sorted(p.parent.parent for p in Path(a.results).rglob("vlm/turns.jsonl") if (p.parent / "summary.json").exists())
    eps = eps[:a.episodes]
    frames = [f.read_bytes() for f in sorted(Path(a.results).rglob("vlm/frames/*.jpg"))[:40]]
    assert eps and frames, "no episodes or frames found"
    store = Path(tempfile.mkdtemp(prefix="o1m_test_"))
    ok = True

    # 1. Phase 1: the store starts empty and fills from each episode in turn
    for i, ep in enumerate(eps):
        mem = replay(store, ep, frames)
        if i == 0:
            ok &= mem.events[0]["graph_items"] == 0 and mem.events[0]["experiences"] == 0
    g, x = json.loads((store / "hdkg.json").read_text()), json.loads((store / "amep.json").read_text())
    print(f"[1] {len(eps)} episodes -> HDKG {len(g)} items, AMEP {len(x)} experiences, "
          f"{len(list((store / 'frames').iterdir()))} frames; verdicts "
          f"{ {v: sum(e['verdict'] == v for e in x) for v in O.VERDICTS} }")
    ok &= len(g) > 0 and len(x) > 0

    # 2. HDKG retrieval for a target (planner input) and the system prompt
    for item in ("iron_pickaxe", "furnace", "diamond_pickaxe"):
        print(f"[2] compile {item}:\n{O.compile_graph(g, item) or O.NO_GRAPH.format(item=item)}")
    mem = O.Optimus1Memory(str(store), "craft an iron pickaxe", "o1_x", "iron_pickaxe", 0)
    system = P.SYSTEM.format(task="t", subgoal=P.NO_SUBGOAL, commands="").replace(O.ANSWER_LINE, O.REFLECT_FORMAT) + mem.prompt_text()
    ok &= O.REFLECT_FORMAT in system and "Knowledge graph" in system
    print(f"[2] system prompt tail:\n{system[-700:]}")

    # 3. failed craft -> the missing materials' graphs until the next craft/smelt
    mem.after_turn({"type": "craft", "item": "iron_pickaxe", "n": 1}, 'failed: missing material: {"stick": 2}', {}, {})
    ok &= "missing materials" in mem.prompt_text()
    print(f"[3] after a failed craft:{mem.prompt_text()[-300:]}")
    mem.after_turn({"type": "craft", "item": "stick", "n": 4}, "done", {"oak_planks": 2}, {"stick": 4})
    ok &= "missing materials" not in mem.prompt_text()

    # 4. AMEP retrieval at a periodic check: one case per verdict, two frames each
    goal = max(x, key=lambda e: e["verdict"] == "COMPLETE")["subgoal"]
    mem.after_answer(json.dumps({"reflection": "", "subgoal": goal}), frames[0], STATE, {}, 0)
    early = mem.turn_content(10)
    content = mem.turn_content(O.H["reflect_every_steps"])
    n_img = sum(c["type"] == "image_url" for c in content)
    print(f"[4] sub-goal {goal!r}: before the check {len(early)} parts; at the check {n_img} images, texts: "
          f"{[c['text'][:90] for c in content if c['type'] == 'text']}")
    ok &= not early and 0 < n_img <= O.H["frames_shown"] * 3

    # 5. frozen: an episode reads the store and writes nothing
    before = digest(store)
    os.environ["OPTIMUS_VLM_O1M_FROZEN"] = "1"
    mem = replay(store, eps[0], frames)
    del os.environ["OPTIMUS_VLM_O1M_FROZEN"]
    print(f"[5] frozen: unchanged={digest(store) == before}; {mem.events[-1]}")
    ok &= digest(store) == before and mem.frozen

    # 6. parallel chains: 8 writers at once, nothing lost
    n0 = len(json.loads((store / "amep.json").read_text()))
    with mp.Pool(8) as pool:
        pool.starmap(_writer, [(str(store), frames[0], k) for k in range(8)])
    n1 = len(json.loads((store / "amep.json").read_text()))
    st = json.loads((store / "hdkg.json").read_text())["stick"]["recipes"]
    print(f"[6] parallel: AMEP {n0} -> {n1}; stick recipes {st}")
    ok &= n1 == n0 + 8

    print("store:", store, "\nALL OK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
