"""Experiment conditions: a method in a world. A condition decides what the episode code switches on (environment
variables) and where its results go: <runs>/results/<method>/<world>/<order>/<task>/<seed>/.

    Condition("baseline", "standard")                    results/baseline/standard/
    Condition("gt_world_rule", "nolava")                 results/gt_world_rule/nolava/
    Condition("baseline", "nolava", save_every=600)      results/baseline/nolava_save600/
    Condition("ours", "nolava", knowledge="pl1")         results/ours_pl1/nolava/
    Condition("baseline", "nolava", plan=True)           results/baseline_plan/nolava/
    Condition("mineevolve", "nolava", store="p1")        results/mineevolve_p1/nolava/ (store me_p1.json)
    Condition("optimus1_memory", o1m="tr1")              results/optimus1_memory_tr1/standard/  (Phase 1: learns)
    Condition("optimus1_memory", o1m="tr1", frozen=True) results/optimus1_memory_tr1_frozen/standard/  (Phase 2)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Agents (EXPERIMENTS.md). Each is a list of switches (environment variable, value); LLM = configs/llm_<name>.yaml.
VLM = [("OPTIMUS_VLMTASK", "1")]  # the Baseline agent, domains/minecraft/agent/vlm_agent.py
METHODS = {
    "optimus1": {"memory": "prebuilt", "llm": "g38", "switches": [  # released Optimus-1 + labelled fixes
        ("OPTIMUS_LOGFIX", "1"), ("OPTIMUS_CRAFTFIX", "1"), ("OPTIMUS_TAGFIX", "1"), ("OPTIMUS_PROMPTFIX", "1"),
        ("OPTIMUS_REPLANFIX", "1"), ("OPTIMUS_ESCAPEFIX", "1")], "memfix": True},
    "baseline": {"switches": VLM},
    "mineevolve": {"switches": VLM, "mineevolve": "mineevolve"},  # store run_data/memory/me_mineevolve.json
    "optimus1_memory": {"switches": VLM},  # + the store run_data/memory/o1m_<name>/, Condition(o1m=..., frozen=...)
    "gt_recipe": {"switches": VLM + [("OPTIMUS_VLM_RECIPES", "fixed")]},
    "gt_world_rule": {"switches": VLM + [("OPTIMUS_VLM_OREHINT", "1")]},  # the rule in short
    "gt_world_rule_v2": {"switches": VLM + [("OPTIMUS_VLM_OREHINT", "2")]},  # what the placing code does, step by step
    "ours": {"switches": VLM},  # + the World Knowledge store, Condition(knowledge=...)
}
WORLDS = {
    "standard": [],  # Optimus-1's env as released (with its hidden ore rule)
    "nolava": [("OPTIMUS_NOLAVA", "1")],  # lava near the agent becomes stone
    "nolava_equipfix": [("OPTIMUS_NOLAVA", "1"), ("OPTIMUS_EQUIPFIX", "1")],  # + equip drops nothing (domains/minecraft/agent/tools/craft_helpers.py)
}


@dataclass
class Condition:
    method: str
    world: str = "standard"
    save_every: int = 0  # save the world every N steps (starting points of the replay world)
    knowledge: str | None = None  # method ours: World Knowledge store memory/wk_<name>.json
    plan: bool = False  # initial planning (domains/minecraft/agent/planner.py), on top of any VLM method
    store: str | None = None  # method mineevolve: its memory store memory/me_<name>.json (default: the method's own)
    o1m: str | None = None  # method optimus1_memory: its store run_data/memory/o1m_<name>/, one per run
    frozen: bool = False  # mineevolve --store / optimus1_memory: the store is read only (Phase 2, unseen tasks)

    def __post_init__(self):
        if self.method not in METHODS or self.world not in WORLDS:
            raise ValueError(f"unknown method {self.method!r} or world {self.world!r}")
        if self.plan and ("OPTIMUS_VLMTASK", "1") not in METHODS[self.method]["switches"]:
            raise ValueError("--plan needs a method run by the Baseline agent")
        if (self.method == "ours") != bool(self.knowledge):
            raise ValueError("method 'ours' needs --knowledge, and only it takes one")
        if self.store and not METHODS[self.method].get("mineevolve"):
            raise ValueError("--store is for method 'mineevolve' only")
        if (self.method == "optimus1_memory") != bool(self.o1m) or (self.frozen and not (self.o1m or self.store)):
            raise ValueError("method 'optimus1_memory' needs --o1m, and only it takes --o1m; --frozen needs --o1m or --store")

    @property
    def name(self) -> str:
        """results/<method>/<world>"""
        method = (f"ours_{self.knowledge}" if self.method == "ours" else
                  f"optimus1_memory_{self.o1m}" + ("_frozen" if self.frozen else "") if self.o1m else
                  f"mineevolve_{self.store}" + ("_frozen" if self.frozen else "") if self.store else
                  self.method) + ("_plan" if self.plan else "")
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
        out["OPTIMUS_ISOWORLD"] = "1"  # every episode in its own Minecraft world folder (domains/minecraft/world/private_world.py)
        out["LLM_CONFIG"] = str(repo / "configs" / f"llm_{a.get('llm', 'g38mon')}.yaml")
        if a.get("mineevolve"):
            out["OPTIMUS_VLM_ME"] = str(runs_root / "memory" / f"me_{self.store or a['mineevolve']}.json")
            if self.frozen:
                out["OPTIMUS_VLM_ME_FROZEN"] = "1"
        if self.save_every:
            out["OPTIMUS_CKPT_EVERY"] = str(self.save_every)
            out["OPTIMUS_SAVEALL_JAR"] = str(runs_root / "infra" / "mcprec-6.13-saveall.jar")
        if self.knowledge:
            out["OPTIMUS_VLM_WK"] = str(runs_root / "memory" / f"wk_{self.knowledge}.json")
        if self.plan:
            out["OPTIMUS_VLM_PLAN"] = "1"
        if self.o1m:
            out["OPTIMUS_VLM_O1M"] = str(runs_root / "memory" / f"o1m_{self.o1m}")
            if self.frozen:
                out["OPTIMUS_VLM_O1M_FROZEN"] = "1"
        return out


def add_args(ap) -> None:
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--world", default="standard", choices=list(WORLDS))
    ap.add_argument("--save", type=int, default=0, help="save the world every N steps (replay starting points)")
    ap.add_argument("--knowledge", default=None, help="method ours: World Knowledge store memory/wk_<name>.json")
    ap.add_argument("--plan", action="store_true", help="initial planning: sub-goals written before the first turn")
    ap.add_argument("--store", default=None, help="method mineevolve: memory store memory/me_<name>.json")
    ap.add_argument("--o1m", default=None, help="method optimus1_memory: store run_data/memory/o1m_<name>/")
    ap.add_argument("--frozen", action="store_true", help="--store / --o1m: read the store, never write it (Phase 2)")


def from_args(a) -> Condition:
    return Condition(a.method, a.world, a.save, a.knowledge, getattr(a, "plan", False), getattr(a, "store", None),
                     getattr(a, "o1m", None), getattr(a, "frozen", False))
