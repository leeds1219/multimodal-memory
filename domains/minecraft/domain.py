"""The Minecraft domain (Optimus-1 benchmark on MineRL) as the method sees it (domains/base.py).

Episodes are the folders experiments/chain.py writes: result.json, vlm/turns.jsonl, trajectory.jsonl.gz and the
world saves ckpt/s<step>.zip (world/save.py). The log the Designer reads is episode_log.py; nothing from client.log (the
ore placements) reaches it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import episode_log
import prompts as P
from domains.base import Domain, Vocabulary


class Minecraft(Domain):
    name = "minecraft"
    vocabulary = Vocabulary(name="Minecraft")
    env_label = "O"

    def result(self, ep):
        return json.loads((Path(ep) / "result.json").read_text())

    def turns(self, ep, until=None):
        return episode_log.turns_of(Path(ep), until)

    def commands(self):
        return "\n".join([P.CMD_KEYS, P.CMD_STEVE, P.CMD_HELPERS])

    def log(self, parts, milestones=None, markers=None):
        return episode_log.render(parts, milestones, markers)

    def milestones_reached(self, parts, milestones):
        return episode_log.milestones_reached(parts, milestones)

    def milestone_held(self, ep, from_turn, milestone):
        return episode_log.milestone_held(Path(ep), from_turn, milestone)

    def success_segment(self, ep, prev, target, max_rows):
        return episode_log.success_segment(Path(ep), self.result(ep)["task"], prev, target, max_rows)

    def checkpoints(self, ep):
        return episode_log.checkpoints(Path(ep))

    def save_before(self, ep, turn):
        """Saves are taken at turn ends, so the last one at or before the turn's first step."""
        ep = Path(ep)
        rows = {t["turn"]: t for t in self.turns(ep)}
        if turn not in rows:
            return None
        ok = [s for s in episode_log.checkpoints(ep) if s <= rows[turn]["t0"]]
        return (ep / "ckpt" / f"s{max(ok):06d}.zip", max(ok)) if ok else ("", 0)

    def state_line(self, ep, clock, window):
        return episode_log.state_line(Path(ep), clock, window)

    def plan(self, ep):
        f = Path(ep) / "vlm" / "plan.json"
        plan = json.loads(f.read_text()).get("plan") if f.exists() else None
        return "\n".join(f"{i}. have {s['count']} {s['item']}" for i, s in enumerate(plan, 1)) if plan else None

    def llm_config(self, opts):
        from conditions import Condition
        from paths import ROOT, RUN_DATA
        return Condition("baseline", opts["world"], opts["save_every"]).env_vars(RUN_DATA, ROOT)["LLM_CONFIG"]

    def run_episode(self, task, seed, ep, start_save, extras_file, opts):
        """The Baseline agent (agent/vlm_agent.py) in world opts["world"], saving every opts["save_every"] steps,
        ending opts["max_steps"] steps after its start when set, planning first when opts["plan"] (an episode from a
        save keeps its plan); opts also holds name (the chain's order is loop_<name>), gpu, port, mock."""
        from benchmark import SEEDS
        from chain import Chain
        ns = argparse.Namespace(method="baseline", world=opts["world"], save=opts["save_every"], knowledge=None,
                                order=f"loop_{opts['name']}", gpu=opts["gpu"], port=opts["port"], retries=2,
                                stall_min=60.0, mock=opts["mock"], seed_offset=seed - SEEDS[task],
                                plan=bool(opts.get("plan")))
        ch = Chain(ns)
        ch.env_vars.update(OPTIMUS_EXPERIMENT=str(extras_file))
        if start_save:
            ch.env_vars.update(OPTIMUS_LOAD_CKPT=str(Path(start_save).resolve()))
        if opts.get("max_steps"):  # a rollout: the episode ends this many steps after its start
            ch.env_vars.update(OPTIMUS_MAX_STEPS=str(opts["max_steps"]))
        if opts.get("stop_file"):  # a rollout: the episode ends once the loop writes this file
            ch.env_vars.update(OPTIMUS_STOP_FILE=str(opts["stop_file"]))
        return ch.run_episode(task, Path(ep))


DOMAIN = Minecraft()
