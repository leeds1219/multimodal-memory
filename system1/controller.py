"""Runs the agent's commands in the game, one MineRL step at a time, and keeps the state the agent is told about.

A command is run by a tool (system1/tools/): raw keys held for N steps, a STEVE-1 instruction run for N steps, or one
of Optimus-1's own helpers (equip / craft / smelt). Every step of every tool goes through Controller.step, which records position, facing, health and
deaths. No code stop on health or falls: the VLM decides (the released wrapper's "Stuck" /kill after 8000 steps at
one height is switched off as well).

Runaway guards (a trial ends with no further VLM call): the env reports done; VLM calls reach horizon / 5; 30 turns
in a row ran at most 1 step and changed nothing.
"""
from __future__ import annotations

import json
import math

import craft_helpers
import keys
import steve1_policy
from inventory import f, inv_counts

MIN_STEPS_PER_CALL = 5  # VLM calls <= horizon / this
STALL_TURNS = 30  # this many turns in a row that ran <= 1 game step and changed nothing -> "no_progress"


class Controller:
    def __init__(self, env, cfg, steps: int, log):
        self.env, self.cfg, self.steps, self.log = env, cfg, steps, log
        self.t, self.rows, self.deaths, self.prev = 0, [], 0, None
        self.result = None  # "success" / "dead" / None
        self.stop = None  # guard that ended the trial: "env_ended" / "call_cap" / "no_progress" / None
        self.calls, self.stall = 0, 0
        self.success_t = None
        self.obs, self.info = None, {}
        self._helper = None

    # ------------------------------------------------------------ state
    def state(self, obs):
        loc, life = obs["location_stats"], obs.get("life_stats") or {}
        return {"pos": [round(f(loc["xpos"]), 2), round(f(loc["ypos"]), 2), round(f(loc["zpos"]), 2)],
                "yaw": round(f(loc.get("yaw", 0)), 1), "pitch": round(f(loc.get("pitch", 0)), 1),
                "hp": round(f(life.get("life", 20)), 1)}

    def check(self, s):
        """Task success; overridden by the agent."""
        return None

    def step(self, action, tag):
        import numpy as np
        self.env.cache["ypos"] = {}  # no "Stuck" /kill of the released wrapper (8000 steps at one height)
        obs, _, done, info = self.env.step(action)
        if done and self.stop is None:  # the env's own episode end (wrapper timeout, server gone): nothing after it is real
            self.stop = "env_ended"
            self.log(f"  env returned done at step {self.t}: trial ends (env_ended)")
        s = self.state(obs)
        died = self.prev is not None and ((self.prev["hp"] > 0 and s["hp"] <= 0) or
                                          sum((a - b) ** 2 for a, b in zip(s["pos"], self.prev["pos"])) > 100)
        self.deaths += died
        a = {k: (v if isinstance(v, (int, float)) else np.asarray(v).tolist()) for k, v in action.items()
             if k in ("attack", "forward", "back", "left", "right", "jump", "sneak", "sprint", "camera", "use")}
        row = {"t": self.t, "c": tag, **s, "a": a}
        if died:
            row["died"] = True
        self.rows.append(row)
        self.prev = s
        self.t += 1
        if died and self.result is None:
            self.result = "dead"
        elif self.result is None:
            self.result = self.check(s)
        self.obs, self.info = obs, info
        return obs, s, done or self.t >= self.steps

    def over(self):
        return self.result is not None or self.stop is not None or self.t >= self.steps

    def guard_turn(self, t_start, s0, inv0=None, inv1=None):
        """After each VLM turn: count the call, stop on the call cap or on a run of turns that did nothing."""
        self.calls += 1
        cap = math.ceil(self.steps / MIN_STEPS_PER_CALL)
        same = (self.prev["pos"] == s0["pos"] and self.prev["hp"] == s0["hp"] and inv0 == inv1)
        self.stall = self.stall + 1 if (self.t - t_start <= 1 and same) else 0
        if self.stop is None and self.calls >= cap:
            self.stop = "call_cap"
        elif self.stop is None and self.stall >= STALL_TURNS:
            self.stop = "no_progress"
        if self.stop in ("call_cap", "no_progress"):
            self.log(f"  {self.stop} after {self.calls} VLM calls at step {self.t}: trial ends")

    # ------------------------------------------------------------ commands
    def noop(self):
        return self.env.noop_action()

    def execute(self, cmd, tag):
        """Run one command with its tool (system1/tools/)."""
        typ = cmd.get("type")
        if typ == "steve":
            return steve1_policy.run(self, cmd.get("prompt"), cmd.get("steps"), tag)
        if typ in ("equip", "craft", "smelt"):
            return craft_helpers.run(self, typ, str(cmd.get("item", "")).strip(), int(cmd.get("n", 1) or 1), tag)
        if typ == "keys":
            return keys.run(self, cmd, tag)
        return f"unknown command {typ!r}"

    # ------------------------------------------------------------ what the agent is told
    def describe(self):
        s, (x0, y0, z0) = self.prev, self.start
        inv = inv_counts(self.obs)
        return (f"Height y={s['pos'][1]:.0f} (sea level is y=63). Position relative to start: dx={s['pos'][0] - x0:+.1f} (east), "
                f"dy={s['pos'][1] - y0:+.1f} (up), dz={s['pos'][2] - z0:+.1f} (south). Facing yaw {s['yaw']:.0f} "
                f"(0 = south, 90 = west, -90 = east, 180 = north), pitch {s['pitch']:.0f}. Health {s['hp']}/20. "
                f"Inventory: {json.dumps(inv)}.")

    def effect(self, s0, s1, inv0, inv1, seg):
        def _on(v):
            while isinstance(v, list):
                v = v[0] if v else 0
            return bool(v)
        att = sum(_on(r["a"].get("attack")) for r in seg) / max(1, len(seg))
        turned = (s1["yaw"] - s0["yaw"] + 180) % 360 - 180
        dinv = {k: inv1.get(k, 0) - inv0.get(k, 0) for k in set(inv0) | set(inv1) if inv1.get(k, 0) != inv0.get(k, 0)}
        return (f"moved dx={s1['pos'][0] - s0['pos'][0]:+.1f} dy={s1['pos'][1] - s0['pos'][1]:+.1f} "
                f"dz={s1['pos'][2] - s0['pos'][2]:+.1f}, turned {turned:+.0f} deg, pitch {s0['pitch']:.0f}->{s1['pitch']:.0f}, "
                f"health {s0['hp']}->{s1['hp']}, inventory change {json.dumps(dinv) if dinv else 'none'}, "
                f"attack held {att:.0%} of the time")
