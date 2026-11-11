"""Optimus-1's Hybrid Multimodal Memory (Li et al., NeurIPS 2024, Sec. 2.1-2.3) on the Baseline agent: the
Hierarchical Directed Knowledge Graph (HDKG) and the Abstracted Multimodal Experience Pool (AMEP).

Written from the paper; the release (official/NeurIPS24-Optimus-1/src/optimus1/memories/, main.py) was audited and
differs in ways we do not copy: its graph is loaded from Minecraft's recipe files (ground truth; here it holds only
what the agent's own episodes showed), its replan prompt states the pickaxe tiers (not shown here), it gives the
planner a stored successful plan as an example (the paper's planner gets the HDKG only), and it picks reflection
cases at random within the best fuzzy-matched task (here: the most similar sub-goal, as the paper's "most relevant").

Mapping onto an agent without a planner or reflector (the VLM sends one command per turn):
  sub-goal      the VLM names the sub-goal of each command (Optimus-1's planner writes them) and, every turn, judges the
                sub-goal of its previous command: COMPLETE, CONTINUE or REPLAN (the reflector's three verdicts, defined
                as in the release's reflection prompt). A reworded sub-goal under CONTINUE stays the same sub-goal.
  HDKG          nodes are items; an edge says what an item was made from and how many (paper App. E.2: "3 planks,
                2 sticks -> wooden pickaxe"). Written from the agent's own turns: a craft/smelt that succeeded (its
                inventory change), one that failed for a missing material (the helper names it), and items gathered
                by other commands. Retrieval (paper: the node and its subgraph, topologically sorted, with amounts):
                for the task's target item, in the system prompt every turn; for the missing materials of a failed
                craft/smelt (the release's replan retrieval), until the next craft/smelt.
  AMEP          one experience per reflection window: the frames between the start of a sub-goal (or the previous
                periodic check) and the verdict that ended it, abstracted by the paper's 16-frame buffer that drops the
                most similar frame, with the sub-goal, the agent's initial and final state, the plan (sub-goals of the
                episode so far), the verdict and the task's success. A window ends at COMPLETE / REPLAN, or with
                CONTINUE once it is reflect_every_steps old (the release reflects every 1200 steps). Retrieval at those
                periodic checks: for each verdict, the most similar sub-goal's case (first and last frame, as the
                release's before/now pair), in the turn's message before the current view.
Writing: after the episode (paper App. E: memory is updated after a task), under a file lock, so parallel chains of a
run share one store and a crashed attempt writes nothing. Frozen (OPTIMUS_VLM_O1M_FROZEN=1, Phase 2): read only.
Store: a directory <runs>/memory/o1m_<name>/ (hdkg.json, amep.json, frames/). Hyperparameters:
configs/hyperparams.yaml optimus1_memory.*.
"""
from __future__ import annotations

import difflib
import fcntl
import io
import json
import math
import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from hyperparams import HP

H = HP["optimus1_memory"]
VERDICTS = ("COMPLETE", "CONTINUE", "REPLAN")
TOOL_SUFFIXES = ("_pickaxe", "_axe", "_shovel", "_sword", "_hoe")

ANSWER_LINE = 'Answer with JSON only: {"thought": "<what you see and what you expect this command to do>", "command": {...}}'
REFLECT_FORMAT = ('Answer with JSON only: {"thought": "<what you see and what you expect this command to do>", '
                  '"reflection": "COMPLETE" | "CONTINUE" | "REPLAN", "subgoal": "<the sub-goal this command works on, '
                  'e.g. mine 3 iron ore>", "command": {...}}\n'
                  '"reflection" judges the sub-goal of your previous command from what you see now. COMPLETE: it is '
                  'done. CONTINUE: it is not done, but you are in a good state and likely to finish it. REPLAN: it is '
                  'not done and you are in a bad state or situation (low health, in danger, trapped, or this way '
                  'cannot work), so change your plan. "subgoal" is one step of your plan that gets an item or reaches '
                  'a place; keep it until you judge it COMPLETE or REPLAN.')
GRAPH_HEAD = ("\n\nKnowledge graph from earlier episodes: crafting and smelting that worked there (or failed for a "
              "missing item) and items gathered. It holds only what those episodes showed, so it may be incomplete.\n")
NO_GRAPH = "No way to make {item} was observed yet."
MISSING_HEAD = "\nYour last {kind} failed for missing materials. What the knowledge graph has on them:\n"
CASES_HEAD = ("Past cases of a similar sub-goal from earlier episodes, one per verdict. Each has two views: when the "
              "sub-goal (or its last check) started, and when it was judged.")


# --------------------------------------------------------------------------------------------- store
class Store:
    def __init__(self, path):
        self.dir = Path(path)
        self.frames = self.dir / "frames"
        self.frames.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with open(self.dir / "store.lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def read(self, name: str, default):
        f = self.dir / name
        return json.loads(f.read_text()) if f.exists() else default

    def write(self, name: str, data):  # call inside locked()
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self.dir / name)


# --------------------------------------------------------------------------------------------- HDKG
def _norm(qty: dict, out: int) -> tuple[dict, int]:
    """The smallest whole batch: 4 oak_log -> 16 oak_planks is stored as 1 -> 4."""
    g = math.gcd(out, *qty.values()) if qty else out
    return {k: qty[k] // g for k in sorted(qty)}, out // g


def graph_observe(g: dict, ev: dict) -> None:
    """Merge one observed event into the graph {item: {"recipes", "needs", "gathered"}}."""
    node = g.setdefault(ev["item"], {"recipes": [], "needs": {}, "gathered": None})
    if ev["ev"] == "made":
        inn, out = _norm(ev["in"], ev["out"])
        r = next((r for r in node["recipes"] if r["kind"] == ev["kind"] and r["in"] == inn and r["out"] == out), None)
        if r:
            r["n"] += 1
        else:
            node["recipes"].append({"kind": ev["kind"], "in": inn, "out": out, "n": 1})
    elif ev["ev"] == "missing":  # the helper's shortfall: at least this many were needed
        for m, k in ev["missing"].items():
            node["needs"][m] = max(node["needs"].get(m, 0), int(k))
    else:  # gathered: how often, and how often each tool was in the inventory then
        old = node["gathered"] or {"n": 0, "tools": {}}
        node["gathered"] = {"n": old["n"] + 1,
                            "tools": {t: old["tools"].get(t, 0) + (t in ev["tools"]) for t in set(old["tools"]) | set(ev["tools"])}}


def _recipe(node: dict | None):
    return max(node["recipes"], key=lambda r: r["n"]) if node and node["recipes"] else None


def _kept(node: dict, r: dict | None) -> list:
    """Items a failed attempt named that the recipe does not use up (e.g. a table). The helper names some materials
    generically ("planks" for oak_planks), so a name inside an ingredient's name counts as that ingredient."""
    return [m for m in node["needs"] if not (r and any(m == c or m in c or c in m for c in r["in"]))]


def compile_graph(g: dict, item: str, number: int = 1) -> str | None:
    """The item's subgraph, leaves first, with the amount of each node needed for `number` items (paper: retrieve the
    node and its subgraph, topological sort). None when the graph has no way to get the item."""
    node = g.get(item)
    if not node or not (node["recipes"] or node["needs"] or node["gathered"]):
        return None
    order, seen = [], set()

    def visit(x, stack):
        if x in seen or x in stack:
            return
        n = g.get(x)
        r = _recipe(n)
        for c in sorted(r["in"] if r else {}) + sorted(_kept(n, r) if n else []):
            visit(c, stack | {x})
        seen.add(x)
        order.append(x)
    visit(item, set())
    need = {x: 0 for x in order}
    need[item] = number
    for x in reversed(order):  # parents before children
        n, q = g.get(x), need[x]
        r = _recipe(n)
        if r and q > 0:
            runs = -(-q // r["out"])
            for c, k in r["in"].items():
                need[c] += runs * k
        for m in (_kept(n, r) if n else []):  # needed but not used up (e.g. a table), or no success seen yet
            need[m] = max(need[m], n["needs"][m] if not r else 1)
    lines = []
    for i, x in enumerate(order, 1):
        n, r = g.get(x), _recipe(g.get(x))
        if r:
            how = f"{r['kind']}: {' + '.join(f'{k} {c}' for c, k in r['in'].items())} -> {r['out']}"
            kept = _kept(n, r)
            how += f"; also needs {', '.join(kept)} (not used up)" if kept else ""
        elif n and n["needs"]:
            how = (f"needs at least {', '.join(f'{k} {m}' for m, k in n['needs'].items())} (named by a failed "
                   f"craft/smelt; no success seen)")
        elif n and n["gathered"]:
            ga = n["gathered"]
            tools = [t for t, k in sorted(ga["tools"].items()) if k >= H["tool_share"] * ga["n"]]
            how = f"gathered {ga['n']} times; tools usually in the inventory then: {', '.join(tools) or 'none'}"
        else:
            how = "no way to get it observed yet"
        lines.append(f"{i}. {x}: need {need[x]} ({how})")
    return f"get {number} {item}:\n" + "\n".join(lines)


# --------------------------------------------------------------------------------------------- AMEP
def frame_bytes(pov) -> bytes:
    """One frame as stored and shown."""
    from images import jpeg
    return jpeg(pov, tuple(H["frame_px"]), 70)


def _thumb(frame: bytes):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(frame)).convert("L").resize((32, 18)), dtype=np.float32).ravel()
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def buffer_add(buf: list, frame: bytes, size: int) -> None:
    """Paper App. E.1.1: a buffer of `size` frames; when full, the frame most similar to its predecessor goes (the
    first frame, the state before the sub-goal, always stays)."""
    buf.append((frame, _thumb(frame)))
    if len(buf) > size:
        sims = [float(buf[i][1] @ buf[i - 1][1]) for i in range(1, len(buf))]
        buf.pop(1 + max(range(len(sims)), key=sims.__getitem__))


def retrieve_cases(exps: list, subgoal: str, task_id: str) -> list:
    """Per verdict, the experiences of the most similar sub-goal (ties: same task, then newest)."""
    out = []
    for v in VERDICTS:
        scored = []
        for i, e in enumerate(exps):
            if e["verdict"] != v:
                continue
            s = difflib.SequenceMatcher(None, subgoal.lower(), e["subgoal"].lower()).ratio()
            if s >= H["min_subgoal_similarity"]:
                scored.append((s, e["task_id"] == task_id, i, e))
        scored.sort(key=lambda x: x[:3], reverse=True)
        out += [e for *_, e in scored[:H["cases_per_verdict"]]]
    return out


# --------------------------------------------------------------------------------------------- per episode
class Optimus1Memory:
    def __init__(self, store_path: str, task: str, task_id: str, target: str | None, seed, order: str = ""):
        self.store = Store(store_path)
        self.frozen = os.environ.get("OPTIMUS_VLM_O1M_FROZEN") == "1"
        self.task, self.task_id, self.target = task, task_id, target
        self.source = {"task": task_id, "seed": seed, "order": order}
        with self.store.locked():  # the memory as it is when the episode starts
            self.graph = self.store.read("hdkg.json", {})
            self.exps = self.store.read("amep.json", [])
        self.graph_text = (compile_graph(self.graph, target) or NO_GRAPH.format(item=target)) if target else ""
        self.missing_text = ""
        self.win, self.due, self.plan = None, False, []
        self.new_events, self.new_exps = [], []  # written at finish()
        self.events = [{"ev": "start", "graph_items": len(self.graph), "experiences": len(self.exps),
                        "target_known": target in self.graph if target else None}]

    # ------------------------------------------------------------ what the VLM gets
    def prompt_text(self) -> str:
        """Knowledge-Guided Planner input: the HDKG subgraph (system prompt)."""
        out = GRAPH_HEAD + self.graph_text if self.graph_text else ""
        return out + self.missing_text

    def turn_content(self, t: int) -> list:
        """Experience-Driven Reflector input at a periodic check: retrieved cases (turn message, before the current view)."""
        from images import img
        self.due = bool(self.win) and t - self.win["t0"] >= H["reflect_every_steps"]
        if not self.due:
            return []
        cases = retrieve_cases(self.exps, self.win["subgoal"], self.task_id)
        self.events.append({"ev": "reflect_retrieval", "t": t, "subgoal": self.win["subgoal"], "ids": [e["id"] for e in cases]})
        if not cases:
            return []
        out = [{"type": "text", "text": CASES_HEAD}]
        for e in cases:
            out.append({"type": "text", "text": f"<{e['verdict']}> task: {e['task']}; sub-goal: {e['subgoal']}; judged "
                                                f"after {e['steps']} steps; start y={e['start']['y']:.0f}, health {e['start']['hp']}"})
            fs = e["frames"]
            k = min(H["frames_shown"], len(fs))
            for j in sorted({round(i * (len(fs) - 1) / max(1, k - 1)) for i in range(k)}):
                f = self.store.frames / fs[j]
                if f.exists():
                    out.append(img(f.read_bytes()))
        return out

    # ------------------------------------------------------------ AMEP: reflection windows
    def after_answer(self, raw: str, frame: bytes, state: dict, inv: dict, t: int):
        """The VLM's verdict on its previous sub-goal and the sub-goal of this command (before the command runs)."""
        d = parse_answer(raw)
        verdict, subgoal = d["reflection"], d["subgoal"]
        w = self.win
        if w is not None:
            buffer_add(w["buf"], frame, H["buffer_frames"])
            if verdict in ("COMPLETE", "REPLAN") or (self.due and verdict == "CONTINUE"):
                self._close(verdict, state, inv, t)
                if verdict == "CONTINUE":  # the same sub-goal goes on in a new window from this check
                    self._open(w["subgoal"], frame, state, inv, t)
        if self.win is None and subgoal:
            self._open(subgoal, frame, state, inv, t)
        self.due = False

    def _open(self, subgoal, frame, state, inv, t):
        if not self.plan or self.plan[-1] != subgoal:
            self.plan.append(subgoal)
        self.win = {"subgoal": subgoal, "t0": t, "start": {"y": state["pos"][1], "hp": state["hp"], "inv": dict(inv)},
                    "buf": []}
        buffer_add(self.win["buf"], frame, H["buffer_frames"])

    def _close(self, verdict, state, inv, t):
        w, self.win = self.win, None
        e = {"id": f"x_{uuid.uuid4().hex[:8]}", "task": self.task, "task_id": self.task_id, "subgoal": w["subgoal"],
             "verdict": verdict, "steps": t - w["t0"], "start": w["start"],
             "end": {"y": state["pos"][1], "hp": state["hp"], "inv": dict(inv)}, "plan": list(self.plan),
             "source": self.source, "frames": [f for f, _ in w["buf"]]}
        self.new_exps.append(e)
        self.events.append({"ev": "experience", **{k: e[k] for k in ("id", "subgoal", "verdict", "steps")},
                            "frames": len(e["frames"])})

    # ------------------------------------------------------------ HDKG: what the turn showed
    def after_turn(self, cmd, outcome: str, inv0: dict, inv1: dict):
        typ = cmd.get("type") if isinstance(cmd, dict) else None
        dv = {k: inv1.get(k, 0) - inv0.get(k, 0) for k in set(inv0) | set(inv1) if inv1.get(k, 0) != inv0.get(k, 0)}
        ev = None
        if typ in ("craft", "smelt"):
            item = str(cmd.get("item", "")).replace("minecraft:", "").strip()
            if outcome == "done":
                got = [k for k, v in dv.items() if v > 0 and (k == item or k.endswith(item) or item.endswith(k))]
                used = {k: -v for k, v in dv.items() if v < 0}
                if got and used:
                    ev = {"ev": "made", "kind": typ, "item": got[0], "in": used, "out": dv[got[0]]}
            else:
                m = re.search(r"missing material:\s*(\{.*\})", str(outcome))
                try:
                    miss = {str(k): int(v) for k, v in json.loads(m.group(1)).items()} if m else None
                except (ValueError, TypeError):
                    miss = None
                if miss and item:
                    ev = {"ev": "missing", "kind": typ, "item": item, "missing": miss}
            self.missing_text = ""
            if ev and ev["ev"] == "missing":  # the release's replan retrieval: the graph of each missing material
                parts = [compile_graph(self.graph, m, k) or NO_GRAPH.format(item=m) for m, k in ev["missing"].items()]
                self.missing_text = MISSING_HEAD.format(kind=typ) + "\n".join(parts)
        elif typ != "equip":
            tools = [k for k in inv0 if k.endswith(TOOL_SUFFIXES)]
            for k, v in dv.items():
                if v > 0:
                    self._learn({"ev": "gathered", "item": k, "tools": tools})
        if ev:
            self._learn(ev)

    def _learn(self, ev):
        self.new_events.append(ev)
        self.events.append({"ev": "graph_event", **ev})

    # ------------------------------------------------------------ after the episode
    def finish(self, success: bool, state: dict, inv: dict, t: int):
        if self.win is not None and success:  # the task item arrived: the sub-goal in progress was the last one
            self._close("COMPLETE", state, inv, t)
        self.win = None
        for e in self.new_exps:
            e["task_success"] = success
        if self.frozen:
            self.events.append({"ev": "frozen_no_write", "experiences": len(self.new_exps), "graph_events": len(self.new_events)})
            return
        rows = []
        for e in self.new_exps:
            names = []
            for j, f in enumerate(e["frames"]):
                names.append(f"{e['id']}_{j:02d}.jpg")
                (self.store.frames / names[-1]).write_bytes(f)
            rows.append({**e, "frames": names})
        with self.store.locked():
            g = self.store.read("hdkg.json", {})
            for ev in self.new_events:
                graph_observe(g, ev)
            self.store.write("hdkg.json", g)
            self.store.write("amep.json", self.store.read("amep.json", []) + rows)
        self.events.append({"ev": "written", "experiences": len(rows), "graph_events": len(self.new_events)})


def parse_answer(raw: str) -> dict:
    """The "reflection" verdict and "subgoal" of the VLM's JSON answer ("" when missing)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    v = str(d.get("reflection", "")).strip().upper()
    return {"reflection": v if v in VERDICTS else "", "subgoal": str(d.get("subgoal") or "").strip()[:120]}
