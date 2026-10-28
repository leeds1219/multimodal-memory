"""MineEvolve's method (paper arXiv 2603.13131v3, Sec. 3, Eq. 1-8 and Algorithms 1-3) on the VLM baseline.

Written from the paper, not from the release. The
release differs from the paper in ways we do not copy: its Inducer prompt says "use the POV keyframes" but sends
only file paths (here: the frames themselves); Vmatch always returns True (here: a real check); p_goal is always 0
(here: progress towards the subgoal's check); every successful segment of the buffer is re-induced after every
subgoal (here: each maximal successful run once); the remedy is induced twice per failure, once by the Inducer and
once inside the Adaptor (here: once, shared).

Mapping onto an agent without a plan (the VLM sends one command per turn):
  subgoal z_i   the VLM names its current subgoal and a check with every command
                ("subgoal": {"name": ..., "check": {"type": "inv_ge"|"ypos_le"|"ypos_ge", ...}}).
                One execution = the turns spent on one subgoal; it ends when the check holds (y=1), the VLM moves
                to another subgoal, the player dies, or after SEG_STEPS steps (as one subgoal budget of the
                release, 1200 steps = 1 game minute), in which case the same subgoal continues as a new execution.
  Monitor       Eq. 1-3 typed feedback per execution; failure types of Fig. 3 from env signals and helper results.
  Inducer       Eq. 4-5: skill from each maximal run of successful executions whose last check holds; remedy when
                the recent failure rate >= ETA_FAIL over H executions, the last one is stagnant, or a cross-subgoal
                deadlock repeats. Input: typed feedback + the frames the VLM saw (up to N_FRAMES).
  Curator       Eq. 6 validation (schema, match, exec, spec, conflict), merge with support counts, Eq. 7 retrieval
                under a token budget (BUDGET, TOP_K).
  Adaptor       Eq. 8: the VLM keeps what it has done (there is no plan to freeze); on a repair trigger the retrieved
                knowledge K^R and the active remedies A_i (retrieved + the temporary one from this failure) are put
                in its prompt until an execution succeeds. Before the first turn: retrieval for (goal, start state)
                (Algorithm 1, lines 5-6).
Store: one JSON file shared by all chains of a run (file lock), so knowledge accumulates online across episodes.
Frozen (OPTIMUS_VLM_ME_FROZEN=1, paper Sec. 4.4): no knowledge is generated or written, and entries that came
from the same (task, seed) are not retrieved (held-out evaluation).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SEG_STEPS = 1200
H, ETA_FAIL = 4, 0.5
DEADLOCK_WINDOW, DEADLOCK_SUBGOALS, DEADLOCK_REPEATS = 8, 2, 3
LAMBDA_X, LAMBDA_V, LAMBDA_G, EPS_X, EPS_V, EPS_P = 0.4, 0.4, 0.2, 1.0, 1.0, 0.10
LOW_SPAN, HIGH_SPAN = 1.0, 8.0
BUDGET, TOP_K = 512, 16
N_FRAMES = 4
CONFLICT_RHO, CONFLICT_OVERLAP, DUP_TRIGGER, DUP_CONTENT = 0.8, 0.5, 0.5, 0.6
CHECKS = ("inv_ge", "ypos_le", "ypos_ge")
FAILURES = ("NONE", "NAV_STUCK", "TARGET_UNREACHABLE", "MISSING_ITEM", "RECIPE_FAIL", "SMELT_FAIL", "DEADLOCK",
            "MOB_KILLED", "TIMEOUT", "UNKNOWN")
SCOPES = ("subgoal_local", "task_global")
CATEGORIES = ("generic", "missing_prerequisite", "deadlock_pattern", "recovery_strategy")
BLACKLIST = ("try again", "be careful", "be more careful", "do better", "use common sense", "think harder", "keep trying")
EXEC_WORDS = ("craft", "smelt", "equip", "steve", "keys", "mine", "dig", "attack", "place", "use", "walk", "move",
              "jump", "pillar", "climb", "descend", "ascend", "tunnel", "look", "turn", "sneak", "sprint", "chop",
              "collect", "pick", "go ", "break", "kill", "eat", "select", "hotbar", "forward", "back", "strafe", "swim")
MATCH_WORDS = ("inventory", "health", "height", "depth", "level", "position", "underground", "surface", "cave",
               "tunnel", "ore", "y=", "y <", "y >", "y<", "y>", "y ", "hotbar", "night", "lava", "water", "bedrock")

ANSWER_LINE = 'Answer with JSON only: {"thought": "<what you see and what you expect this command to do>", "command": {...}}'
SUBGOAL_FORMAT = ('Answer with JSON only: {"thought": "<what you see and what you expect this command to do>", '
                  '"subgoal": {"name": "<the sub-goal this command works on, e.g. mine 3 iron ore>", "check": '
                  '<when it is done: {"type": "inv_ge", "item": "<item id>", "n": <count in inventory>} or '
                  '{"type": "ypos_le", "n": <height>} or {"type": "ypos_ge", "n": <height>}>}, '
                  '"command": {...}}\nThe sub-goal is one step of your plan that gets an item or reaches a height '
                  '(e.g. "get 3 oak logs", "mine 3 iron ore", "reach y 12"). Keep the same name and check until the '
                  'check holds; moving, looking around, digging toward it or fetching a missing tool all stay under the '
                  'current sub-goal. Change it only when it is done or when you give it up.')
KNOW_HEAD = ("\n\nKnowledge from earlier executions (skills: steps that worked before; remedies: what to do when a "
             "failure like this happens). It was learned in other worlds; use it if it fits what you see.\n")
REPAIR_HEAD = "\n\nRepair guidance for your current failure ({why}):\n"

SKILL_SYSTEM = """You are the Inducer of a Minecraft agent's knowledge system. You get a run of consecutive successful
sub-goal executions (typed feedback: sub-goal, success, inventory change, state change, failure type, progress) and
frames the agent saw. Turn the run into ONE reusable skill that the agent can execute again and verify from its state.
The agent acts with commands: "keys" (hold keys / turn the camera for N steps), "steve" (a learned policy following a
short instruction), "equip", "craft", "smelt". Write steps as such commands or concrete actions, with the numbers
that worked (steps, counts, heights). Use only what the feedback and frames show.
Answer with JSON only:
{"trigger_context": ["<state phrase: task, inventory, height, place>", ...], "preconditions": ["<inventory or world
condition, with counts>", ...], "steps": ["<one executable step>", ...], "verification": {"type": "inv_ge" | "ypos_le" |
"ypos_ge", "item": "<item id, for inv_ge>", "n": <int>}, "observed_effects": ["<inventory or state change>", ...],
"confidence": <0-1, 0.9+ only if the inventory change verifies it>}"""

REMEDY_SYSTEM = """You are the Inducer of a Minecraft agent's knowledge system. Recent sub-goal executions failed or
stagnated. You get their typed feedback (sub-goal, success, inventory change, state change, failure type, progress,
stagnation flag), a deadlock signal if the same failure repeats across sub-goals, and frames the agent saw.
Diagnose the risk pattern and write ONE remedy: concrete repair actions the agent can execute (commands: "keys",
"steve", "equip", "craft", "smelt", or concrete actions with numbers). No generic advice ("try again", "be careful").
Failure types: NAV_STUCK (stays near one spot, no gain), TARGET_UNREACHABLE (moves a lot, never gets the target),
MISSING_ITEM (a tool, table, fuel or material is missing), RECIPE_FAIL, SMELT_FAIL, DEADLOCK (same failure across
sub-goals), MOB_KILLED (died), TIMEOUT (sub-goal budget used up), UNKNOWN.
Scope: "subgoal_local" (changes only the next sub-goal) or "task_global" (inserts prerequisite or recovery sub-goals).
Category: "generic", "missing_prerequisite", "deadlock_pattern" or "recovery_strategy".
Answer with JSON only:
{"trigger_context": ["<state phrase that can be matched: inventory, height, place, failure>", ...], "failure_type":
"<one type>", "risk_pattern": "<one sentence>", "repair_action": ["<one executable step>", ...], "scope": "...",
"category": "...", "applicability": ["<state condition that must hold>", ...], "confidence": <0-1>}"""


def _toks(text) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if len(t) > 2}


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _content(e: dict) -> list:
    return e["u"].get("steps", []) if e["type"] == "skill" else e["u"].get("repair_action", [])


def _count(inv: dict, item: str) -> int:
    item = str(item).lower().replace("minecraft:", "")
    if item in inv:
        return int(inv[item])
    return sum(int(v) for k, v in inv.items() if k.endswith(item) or item.endswith(k) or item in k)


# --------------------------------------------------------------------------------------------- store (Curator)
class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with open(str(self.path) + ".lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield json.loads(self.path.read_text()) if self.path.exists() else []
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def save(self, entries: list):  # call inside locked()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1))
        os.replace(tmp, self.path)


def validate(e: dict, store: list, state_words: set) -> str:
    """Eq. 6: returns "ok" or the rule that rejected it."""
    need = ("trigger_context", "steps", "verification") if e["type"] == "skill" else \
        ("trigger_context", "repair_action", "risk_pattern")
    raw = e["raw"]
    if any(not raw.get(k) for k in need) or not e["c"]:
        return "schema"
    if e["type"] == "skill" and raw["verification"].get("type") not in CHECKS:
        return "schema"
    trig = " ".join(map(str, e["c"])).lower()
    if not (any(w in trig for w in MATCH_WORDS) or _toks(trig) & state_words):
        return "match"  # Vmatch: the trigger must refer to state fields (inventory items, height, place, health)
    items = [str(x).lower() for x in _content(e)]
    if not items or any(not any(w in x for w in EXEC_WORDS) for x in items):
        return "exec"
    if any(len(x) < 12 or any(b in x for b in BLACKLIST) for x in items):
        return "specificity"
    for o in store:  # Cconflict: overlapping trigger, same type/failure, different content, existing rho >= 0.8
        if (o["type"] == e["type"] and o["failure_type"] == e["failure_type"] and o["rho"] >= CONFLICT_RHO
                and _jacc(_toks(" ".join(o["c"])), _toks(trig)) >= CONFLICT_OVERLAP
                and _jacc(_toks(" ".join(map(str, _content(o)))), _toks(" ".join(items))) < DUP_CONTENT):
            return "conflict"
    return "ok"


def ingest(store: Store, cands: list, state_words: set, source: dict, journal: list | None = None) -> list:
    """Validate, then merge into a duplicate (support count, confidence average, sources) or insert.
    journal: every write is recorded so rollback() can undo a crashed attempt."""
    report = []
    with store.locked() as entries:
        for e in cands:
            why = validate(e, entries, state_words)
            if why != "ok":
                report.append({"id": e["id"], "type": e["type"], "accepted": False, "reason": why})
                continue
            e = {k: v for k, v in e.items() if k != "raw"}
            dup = next((o for o in entries if o["type"] == e["type"] and o["failure_type"] == e["failure_type"]
                        and _jacc(_toks(" ".join(o["c"])), _toks(" ".join(e["c"]))) >= DUP_TRIGGER
                        and _jacc(_toks(" ".join(map(str, _content(o)))), _toks(" ".join(map(str, _content(e)))))
                        >= DUP_CONTENT), None)
            if dup:
                dup["rho"] = (dup["rho"] * dup["support"] + e["rho"]) / (dup["support"] + 1)
                dup["support"] += 1
                dup["E"] = dup["E"] + e["E"]
                added = source not in dup["sources"]
                if added:
                    dup["sources"].append(source)
                if journal is not None:
                    journal.append({"op": "merge", "id": dup["id"], "rho": e["rho"], "E": e["E"], "source": source if added else None})
                report.append({"id": e["id"], "type": e["type"], "accepted": True, "merged_into": dup["id"]})
            else:
                entries.append(e)
                if journal is not None:
                    journal.append({"op": "insert", "id": e["id"]})
                report.append({"id": e["id"], "type": e["type"], "accepted": True})
        store.save(entries)
    return report


def retrieve(store: Store, ctx_text: str, failures: list, exclude: dict | None = None, journal: list | None = None) -> list:
    """Eq. 7: greedy by relevance (token overlap x confidence, + bonus for a remedy of a recent failure type)
    under BUDGET tokens and TOP_K entries. exclude: held-out source {task, seed} (frozen evaluation)."""
    ctx = _toks(ctx_text)
    with store.locked() as entries:
        scored = []
        for e in entries:
            if exclude and any(s.get("task") == exclude["task"] and s.get("seed") == exclude["seed"] for s in e["sources"]):
                continue
            rel = _jacc(ctx, _toks(" ".join(e["c"]) + " " + " ".join(map(str, _content(e)))))
            rel = rel * (0.5 + 0.5 * e["rho"]) + (0.25 if e["type"] == "remedy" and e["failure_type"] in failures else 0)
            if rel > 0:
                scored.append((rel, len(render([e])) // 4, e))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out, used = [], 0
        for rel, n, e in scored:
            if used + n > BUDGET:
                continue
            out.append(e)
            used += n
            if len(out) >= TOP_K:
                break
        ids = {e["id"] for e in out}
        if ids and not os.environ.get("OPTIMUS_VLM_ME_FROZEN"):
            for e in entries:
                if e["id"] in ids:
                    e["uses"] = e.get("uses", 0) + 1
                    if journal is not None:
                        journal.append({"op": "use", "id": e["id"]})
            store.save(entries)
    return out


def rollback(store: Store, journal: list) -> int:
    """Undo a crashed attempt's writes (newest first) so a crash leaves no trace in the shared store, like chain.py's
    memory snapshots. A merge's confidence average is reverted arithmetically."""
    with store.locked() as entries:
        by_id = {e["id"]: e for e in entries}
        for j in reversed(journal):
            e = by_id.get(j["id"])
            if e is None:
                continue
            if j["op"] == "insert":
                entries.remove(e)
                by_id.pop(j["id"])
            elif j["op"] == "use":
                e["uses"] = max(0, e.get("uses", 0) - 1)
            else:
                n = e["support"] - 1
                e["rho"] = (e["rho"] * e["support"] - j["rho"]) / n if n > 0 else e["rho"]
                e["support"] = max(1, n)
                e["E"] = e["E"][:len(e["E"]) - len(j["E"])]
                if j["source"] in e["sources"]:
                    e["sources"].remove(j["source"])
        store.save(entries)
    return len(journal)


def render(entries: list) -> str:
    lines = []
    for e in entries:
        if e["type"] == "skill":
            lines.append(f"- skill (when: {'; '.join(map(str, e['c']))}; needs: {'; '.join(map(str, e['u'].get('preconditions', [])))}): "
                         f"{' -> '.join(map(str, e['u']['steps']))}")
        else:
            lines.append(f"- remedy for {e['failure_type']} (when: {'; '.join(map(str, e['c']))}): {e['u'].get('risk_pattern', '')} "
                         f"Do: {' -> '.join(map(str, e['u']['repair_action']))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- Monitor / Inducer / Adaptor
class MineEvolve:
    """Per-episode Monitor, Inducer and Adaptor; the Curator's store is shared on disk."""

    def __init__(self, store_path: str, task: str, task_id: str, seed, order: str = ""):
        self.store = Store(Path(store_path))
        self.frozen = os.environ.get("OPTIMUS_VLM_ME_FROZEN") == "1"
        self.task, self.task_id, self.seed = task, task_id, seed
        self.source = {"task": task_id, "seed": seed, "order": order}
        self.buffer, self.seg, self.run, self.n_seg = [], None, [], 0
        self.initial, self.active, self.active_why = "", "", ""
        self.events = []  # everything the module did, for the episode record
        self.journal = []  # every store write, for abort()

    # ------------------------------------------------------------ prompt text
    def start(self, state: dict, inv: dict):
        ents = retrieve(self.store, self._ctx("", state, inv), [], self._exclude(), self.journal)
        self.initial = render(ents)
        self.events.append({"ev": "initial_retrieval", "ids": [e["id"] for e in ents]})

    def prompt_text(self) -> str:
        out = KNOW_HEAD + self.initial if self.initial else ""
        if self.active:
            out += REPAIR_HEAD.format(why=self.active_why) + self.active
        return out

    def _exclude(self):
        return {"task": self.task_id, "seed": self.seed} if self.frozen else None

    def _ctx(self, subgoal: str, state: dict, inv: dict) -> str:
        y = state["pos"][1]
        band = "surface" if y >= 60 else "underground deep diamond level" if y <= 16 else "underground"
        return f"{self.task} {subgoal} {' '.join(inv)} height {band}"

    # ------------------------------------------------------------ executions
    def before_turn(self, decl, n, state, inv, t):
        """The VLM's declared subgoal for this turn; closes the open execution when it moved to another one."""
        name, check = self._parse(decl)
        # same subgoal if the name or the success check matches (the VLM rewords names between turns)
        if self.seg and name and name.lower() != self.seg["z"].lower() and not (check and check == self.seg["check"]):
            self._close(state, inv, t, success=self._holds(self.seg["check"], state, inv), reason="switched")
        if self.seg is None:
            name = name or (self.buffer[-1]["z"] if self.buffer else "unspecified")
            self.n_seg += 1
            self.seg = {"z": name, "check": check, "sid": f"s{self.n_seg}", "t0": t, "s0": state, "inv0": dict(inv),
                        "turns": [], "frames": [], "coords": []}
        elif check and not self.seg["check"]:
            self.seg["check"] = check

    def after_turn(self, turn: dict, frame: bytes, rows: list, state, inv, t, died: bool, helper_fail: str | None):
        if self.seg is None:
            return
        self.seg["turns"].append(turn)
        self.seg["frames"].append(frame)
        self.seg["coords"] += [r["pos"] for r in rows]
        self.seg["died"] = self.seg.get("died") or died
        if helper_fail:
            self.seg["helper_fail"] = helper_fail
        if self._holds(self.seg["check"], state, inv):
            self._close(state, inv, t, success=True, reason="check")
        elif died:
            self._close(state, inv, t, success=False, reason="died")
        elif t - self.seg["t0"] >= SEG_STEPS:
            z, check = self.seg["z"], self.seg["check"]
            self._close(state, inv, t, success=False, reason="timeout")
            self.before_turn({"name": z, "check": check}, None, state, inv, t)

    def abort(self) -> int:
        """The env crashed: chain.py retries the episode, so remove everything this attempt wrote."""
        n = rollback(self.store, self.journal)
        self.journal = []
        self.events.append({"ev": "abort_rollback", "writes": n})
        return n

    def finish(self, state, inv, t):
        if self.seg is not None and self.seg["turns"]:
            self._close(state, inv, t, success=self._holds(self.seg["check"], state, inv), reason="episode_end")
        self._flush_run()

    # ------------------------------------------------------------ Monitor (Eq. 1-3, Fig. 3 failure types)
    def _close(self, state, inv, t, success: bool, reason: str):
        g = self.seg
        self.seg = None
        if not g["turns"]:
            return
        dv = {k: inv.get(k, 0) - g["inv0"].get(k, 0) for k in set(inv) | set(g["inv0"]) if inv.get(k, 0) != g["inv0"].get(k, 0)}
        var = 0.0
        if len(g["coords"]) > 1:
            for ax in range(3):
                vals = [c[ax] for c in g["coords"]]
                m = sum(vals) / len(vals)
                var += sum((v - m) ** 2 for v in vals) / len(vals)
        l1 = sum(abs(v) for v in dv.values())
        p_goal = self._goal_progress(g["check"], g["s0"], state, g["inv0"], inv)
        p = LAMBDA_X * var / (EPS_X + var) + LAMBDA_V * l1 / (EPS_V + l1) + LAMBDA_G * p_goal
        stagnant = p < EPS_P and not success
        span = max((max(c[a] for c in g["coords"]) - min(c[a] for c in g["coords"]) for a in range(3)), default=0) if g["coords"] else 0
        f = "NONE"
        if not success:
            hf = g.get("helper_fail") or ""
            if g.get("died"):
                f = "MOB_KILLED"
            elif "missing material" in hf or "no wooden_pickaxe" in hf:
                f = "MISSING_ITEM"
            elif hf.startswith("craft"):
                f = "RECIPE_FAIL"
            elif hf.startswith("smelt"):
                f = "SMELT_FAIL"
            elif reason == "timeout":
                f = "TIMEOUT"
            elif not dv and span < LOW_SPAN:
                f = "NAV_STUCK"
            elif span > HIGH_SPAN and p_goal < 1:
                f = "TARGET_UNREACHABLE"
            else:
                f = "UNKNOWN"
        fb = {"id": f"e_{uuid.uuid4().hex[:8]}", "sid": g["sid"], "z": g["z"], "check": g["check"], "y": int(success),
              "dv": dv, "ds": {"start": {"y": g["s0"]["pos"][1], "hp": g["s0"]["hp"]}, "end": {"y": state["pos"][1], "hp": state["hp"]},
                               "steps": t - g["t0"], "span": round(span, 1), "end_reason": reason,
                               "commands": [f"{x['cmd']} -> {x['outcome']}; {x['effect']}" for x in g["turns"]][-12:]},
              "f": f, "p": round(p, 3), "l": stagnant, "t0": g["t0"], "t1": t}
        self.buffer.append(fb)
        self.events.append({"ev": "feedback", **{k: fb[k] for k in ("id", "sid", "z", "y", "f", "p", "l", "dv", "t0", "t1")}})
        frames = g["frames"]
        if success:
            self.run.append((fb, frames))
            self.active, self.active_why = "", ""  # the repair worked
            return
        self._flush_run()
        self._remedy_and_repair(frames, state, inv)

    def _holds(self, check, state, inv) -> bool:
        if not check:
            return False
        if check["type"] == "inv_ge":
            return _count(inv, check.get("item", "")) >= int(check.get("n", 1))
        if check["type"] == "ypos_le":
            return state["pos"][1] <= float(check["n"])
        return state["pos"][1] >= float(check["n"])

    def _goal_progress(self, check, s0, s1, inv0, inv1) -> float:
        if not check:
            return 0.0
        if check["type"] == "inv_ge":
            a, b, n = _count(inv0, check["item"]), _count(inv1, check["item"]), int(check.get("n", 1))
            return 1.0 if b >= n else max(0.0, (b - a) / max(1, n - a))
        y0, y1, n = s0["pos"][1], s1["pos"][1], float(check["n"])
        if (check["type"] == "ypos_le" and y1 <= n) or (check["type"] == "ypos_ge" and y1 >= n):
            return 1.0
        return max(0.0, min(1.0, (y1 - y0) / (n - y0))) if n != y0 else 0.0

    @staticmethod
    def _parse(decl):
        if not isinstance(decl, dict):
            return (str(decl)[:120] if decl else ""), None
        name = str(decl.get("name", ""))[:120]
        c = decl.get("check")
        if isinstance(c, dict) and c.get("type") in CHECKS and str(c.get("n", "")).lstrip("-").replace(".", "").isdigit():
            c = {"type": c["type"], "n": float(c["n"]) if c["type"] != "inv_ge" else int(float(c["n"])),
                 **({"item": str(c.get("item", ""))} if c["type"] == "inv_ge" else {})}
        else:
            c = None
        return name, c

    # ------------------------------------------------------------ Inducer (Eq. 4-5) + Adaptor (Eq. 8)
    def _flush_run(self):
        run, self.run = self.run, []
        if not run or self.frozen:
            return
        fbs = [fb for fb, _ in run]  # every execution of the run ended with its check holding (verified final state)
        frames = [f for _, fr in run for f in fr]
        raw = self._llm(SKILL_SYSTEM, {"task_goal": self.task, "successful_executions": [self._fb_text(x) for x in fbs]}, frames, "skill")
        if raw:
            self._ingest([self._entry("skill", raw, fbs, "NONE")])

    def _remedy_and_repair(self, frames, state, inv):
        recent = self.buffer[-H:]
        dead = self._deadlock()
        trigger = (sum(1 for e in recent if not e["y"]) / len(recent) >= ETA_FAIL) or recent[-1]["l"] or bool(dead)
        if not trigger:
            return
        why = (f"deadlock: {dead['failure_type']} x{dead['count']} across {dead['n']} sub-goals" if dead else
               f"{recent[-1]['f']} on '{recent[-1]['z']}'")
        failures = [e["f"] for e in recent if e["f"] != "NONE"] + ([dead["failure_type"]] if dead else [])
        tmp = None
        if not self.frozen:
            raw = self._llm(REMEDY_SYSTEM, {"current_subgoal": recent[-1]["z"], "recent_executions": [self._fb_text(x) for x in recent],
                                            "deadlock_signal": dead}, frames, "remedy")
            if raw:
                tmp = self._entry("remedy", raw, recent, str(raw.get("failure_type") or recent[-1]["f"]))
                if dead and tmp["u"]["category"] == "generic":
                    tmp["u"].update(category="deadlock_pattern", scope="task_global")
                self._ingest([tmp])
        ents = retrieve(self.store, self._ctx(recent[-1]["z"], state, inv), failures, self._exclude(), self.journal)
        if tmp is not None and tmp["u"].get("repair_action"):  # A_i includes the temporary remedy even if not stored
            ents = [e for e in ents if e["id"] != tmp["id"]] + [{k: v for k, v in tmp.items() if k != "raw"}]
        self.active, self.active_why = render(ents), why
        self.events.append({"ev": "repair", "why": why, "ids": [e["id"] for e in ents], "temporary": tmp["id"] if tmp else None})

    def _deadlock(self):
        per, cnt = {}, {}
        for e in self.buffer[-DEADLOCK_WINDOW:]:
            if e["y"] or e["f"] in ("NONE", ""):
                continue
            per.setdefault(e["f"], set()).add(e["z"])
            cnt[e["f"]] = cnt.get(e["f"], 0) + 1
        for f, zs in per.items():
            if cnt[f] >= DEADLOCK_REPEATS and len(zs) >= DEADLOCK_SUBGOALS:
                return {"failure_type": f, "count": cnt[f], "n": len(zs)}
        return None

    def _entry(self, typ, raw, fbs, failure_type) -> dict:
        try:
            rho = max(0.0, min(1.0, float(raw.get("confidence", 0.6))))
        except (TypeError, ValueError):
            rho = 0.6
        if typ == "skill":
            u = {"preconditions": list(raw.get("preconditions") or []), "steps": list(raw.get("steps") or []),
                 "observed_effects": list(raw.get("observed_effects") or []), "task_goal": self.task}
            phi = {"verification": raw.get("verification") or {}}
        else:
            u = {"risk_pattern": str(raw.get("risk_pattern") or ""), "repair_action": list(raw.get("repair_action") or []),
                 "scope": raw.get("scope") if raw.get("scope") in SCOPES else "subgoal_local",
                 "category": raw.get("category") if raw.get("category") in CATEGORIES else "generic"}
            phi = {"applicability": list(raw.get("applicability") or [])}
        if failure_type not in FAILURES:
            failure_type = "UNKNOWN"
        return {"id": f"k_{uuid.uuid4().hex[:8]}", "type": typ, "c": [str(x) for x in raw.get("trigger_context") or []],
                "u": u, "phi": phi, "rho": rho, "E": [x["id"] for x in fbs], "failure_type": failure_type, "support": 1,
                "uses": 0, "sources": [self.source], "created": time.time(), "raw": raw}

    def _ingest(self, cands):
        words = set()
        for fb in self.buffer[-H:]:
            words |= _toks(" ".join(fb["dv"]) + " " + fb["z"] + " " + json.dumps(fb.get("check")))
        words |= _toks(self.task)
        rep = ingest(self.store, cands, words, self.source, self.journal)
        self.events.append({"ev": "ingest", "report": rep,
                            "entries": [{k: v for k, v in c.items() if k != "raw"} for c in cands]})

    @staticmethod
    def _fb_text(fb) -> dict:
        return {k: fb[k] for k in ("sid", "z", "check", "y", "dv", "ds", "f", "p", "l")}

    def _llm(self, system, payload, frames, kind):
        from images import img
        from gemini_client import GeminiClient
        pick = frames if len(frames) <= N_FRAMES else [frames[round(i * (len(frames) - 1) / (N_FRAMES - 1))] for i in range(N_FRAMES)]
        content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=1)}]
        for i, f in enumerate(pick):
            content += [{"type": "text", "text": f"Frame {i + 1} of {len(pick)} (in time order):"}, img(f)]
        try:
            resp = GeminiClient().chat.completions.create(model="FROM_CONFIG", max_tokens=1200,
                                                          messages=[{"role": "system", "content": system},
                                                                    {"role": "user", "content": content}])
            txt = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            out = json.loads(m.group(0)) if m else None
        except Exception as e:  # a failed induction is recorded and skipped, the episode goes on
            self.events.append({"ev": f"{kind}_llm_error", "error": repr(e)[:200]})
            return None
        self.events.append({"ev": f"{kind}_llm", "ok": isinstance(out, dict)})
        return out if isinstance(out, dict) else None
