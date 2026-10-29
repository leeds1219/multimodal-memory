"""Experiment conditions: a method in a world. A condition decides what the episode code switches on (environment
variables) and where its results go: <runs>/results/<method>/<world>/<order>/<task>/<seed>/.

    Condition("baseline", "standard")                    results/baseline/standard/
    Condition("gt_world_rule", "nolava")                 results/gt_world_rule/nolava/
    Condition("baseline", "nolava", save_every=600)      results/baseline/nolava_save600/
    Condition("ours", "nolava", knowledge="pl1")         results/ours_pl1/nolava/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Agents (EXPERIMENTS.md). Each is a list of switches (environment variable, value); LLM = configs/llm_<name>.yaml.
VLM = [("OPTIMUS_VLMTASK", "1")]  # the Baseline agent, system1/vlm_agent.py
METHODS = {
    "optimus1": {"memory": "prebuilt", "llm": "g38", "switches": [  # released Optimus-1 + labelled fixes
        ("OPTIMUS_LOGFIX", "1"), ("OPTIMUS_CRAFTFIX", "1"), ("OPTIMUS_TAGFIX", "1"), ("OPTIMUS_PROMPTFIX", "1"),
        ("OPTIMUS_REPLANFIX", "1"), ("OPTIMUS_ESCAPEFIX", "1")], "memfix": True},
    "baseline": {"switches": VLM},
    "mineevolve": {"switches": VLM, "mineevolve": "mineevolve"},  # store run_data/memory/me_mineevolve.json
    "gt_recipe": {"switches": VLM + [("OPTIMUS_VLM_RECIPES", "fixed")]},
    "gt_world_rule": {"switches": VLM + [("OPTIMUS_VLM_OREHINT", "1")]},  # the rule in short
    "gt_world_rule_v2": {"switches": VLM + [("OPTIMUS_VLM_OREHINT", "2")]},  # what the placing code does, step by step
    "ours": {"switches": VLM},  # + the World Knowledge store, Condition(knowledge=...)
}
WORLDS = {
    "standard": [],  # Optimus-1's env as released (with its hidden ore rule)
    "nolava": [("OPTIMUS_NOLAVA", "1")],  # lava near the agent becomes stone
}


@dataclass
class Condition:
    method: str
    world: str = "standard"
    save_every: int = 0  # save the world every N steps (starting points of the replay world)
    knowledge: str | None = None  # method ours: World Knowledge store memory/wk_<name>.json

    def __post_init__(self):
        if self.method not in METHODS or self.world not in WORLDS:
            raise ValueError(f"unknown method {self.method!r} or world {self.world!r}")
        if (self.method == "ours") != bool(self.knowledge):
            raise ValueError("method 'ours' needs --knowledge, and only it takes one")

    @property
    def name(self) -> str:
        """results/<method>/<world>"""
        method = f"ours_{self.knowledge}" if self.method == "ours" else self.method
        world = self.world + (f"_save{self.save_every}" if self.save_every else "")
        return f"{method}/{world}"

    @property
    def memory(self) -> str:
        """Optimus-1's own memory at the start: 'empty' or 'prebuilt' (the authors' released memory)."""
        return METHODS[self.method].get("memory", "empty")

    @property
    def memfix(self) -> bool:
        return METHODS[self.method].get("memfix", False)

    def env_vars(self, runs_root: Path, repo: Path) -> dict:
        a = METHODS[self.method]
        out = dict(a["switches"]) | dict(WORLDS[self.world])
        out["OPTIMUS_ISOWORLD"] = "1"  # every episode in its own Minecraft world folder (world/private_world.py)
        out["LLM_CONFIG"] = str(repo / "configs" / f"llm_{a.get('llm', 'g38mon')}.yaml")
        if a.get("mineevolve"):
            out["OPTIMUS_VLM_ME"] = str(runs_root / "memory" / f"me_{a['mineevolve']}.json")
        if self.save_every:
            out["OPTIMUS_CKPT_EVERY"] = str(self.save_every)
            out["OPTIMUS_SAVEALL_JAR"] = str(runs_root / "infra" / "mcprec-6.13-saveall.jar")
        if self.knowledge:
            out["OPTIMUS_VLM_WK"] = str(runs_root / "memory" / f"wk_{self.knowledge}.json")
        return out


def add_args(ap) -> None:
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--world", default="standard", choices=list(WORLDS))
    ap.add_argument("--save", type=int, default=0, help="save the world every N steps (replay starting points)")
    ap.add_argument("--knowledge", default=None, help="method ours: World Knowledge store memory/wk_<name>.json")


def from_args(a) -> Condition:
    return Condition(a.method, a.world, a.save, a.knowledge)
