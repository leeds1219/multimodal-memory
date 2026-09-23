"""Functional craft / smelt primitive for Env M (MineEvolve's environment).

Upstream MineEvolve's ``executor/craft_helper.py`` only plays a sound and polls
the inventory, so no craft can ever succeed. The MineEvolve paper lists
"crafting" as an action primitive of the shared interface (§4.1, App. B.2).
This module implements that primitive with the Minecraft 1.16 recipe files
shipped by Optimus-1 (``helper/recipes`` + ``helper/tag_items.json``):

* a craft succeeds only if every ingredient is in the inventory; recipes that
  do not fit the 2x2 grid also need a ``crafting_table`` in the inventory;
* a smelt needs a ``furnace`` in the inventory, the input, and fuel
  (vanilla burn values: coal/charcoal 8 items, logs/planks 1.5, stick 0.5);
* inputs are removed with ``/clear`` and the product is added with ``/give``,
  one chat command = one env step. Nothing is created without its inputs.

The placement semantics match upstream MineEvolve's helper (``_place`` treats
an item in the inventory as available), so a crafting table / furnace only has
to be held, not placed.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("baseline_repro.functional_craft")

RECIPE_DIR = Path("/home/rag/data/official/NeurIPS24-Optimus-1/src/optimus1/helper/recipes")
TAG_FILE = RECIPE_DIR.parent / "tag_items.json"

FUEL_VALUE = {  # items smelted per fuel item (vanilla 1.16)
    "coal": 8.0, "charcoal": 8.0, "coal_block": 80.0, "lava_bucket": 100.0,
    "stick": 0.5,
}
for _w in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak"):
    FUEL_VALUE[f"{_w}_planks"] = 1.5
    FUEL_VALUE[f"{_w}_log"] = 1.5

# Planner vocabulary → canonical item id (only unambiguous aliases).
ALIASES = {
    "planks": "#planks", "wooden_planks": "#planks", "wood_planks": "#planks",
    "sticks": "stick", "crafting_bench": "crafting_table", "workbench": "crafting_table",
    "table": "crafting_table", "iron": "iron_ingot", "gold": "gold_ingot",
    "golden_ingot": "gold_ingot", "logs": "#logs", "log": "#logs",
}


def _strip(name: str) -> str:
    return name.split(":", 1)[-1]


class RecipeBook:
    def __init__(self, recipe_dir: Path = RECIPE_DIR, tag_file: Path = TAG_FILE) -> None:
        self.tags: Dict[str, List[str]] = {
            _strip(k): [_strip(v) for v in vs] for k, vs in json.loads(tag_file.read_text()).items()
        }
        self.by_result: Dict[str, List[dict]] = {}
        for f in sorted(recipe_dir.glob("*.json")):
            r = json.loads(f.read_text())
            kind = _strip(r.get("type", ""))
            if kind not in ("crafting_shaped", "crafting_shapeless", "smelting"):
                continue  # blasting / smoking / campfire / stonecutting need other blocks
            res = r.get("result")
            if isinstance(res, dict):
                item, count = _strip(res["item"]), int(res.get("count", 1))
            else:
                item, count = _strip(res), 1
            r["_file"], r["_kind"], r["_count"] = f.stem, kind, count
            self.by_result.setdefault(item, []).append(r)

    @staticmethod
    def tags_only(tag_file: Path = TAG_FILE) -> Dict[str, List[str]]:
        raw = {_strip(k): [_strip(v) for v in vs] for k, vs in json.loads(tag_file.read_text()).items()}
        return {k: [i for m in vs for i in (raw.get(m, [m]) if m in raw else [m])] for k, vs in raw.items()}

    # Each ingredient slot is a list of acceptable item ids.
    def _choices(self, spec: Any) -> List[str]:
        if isinstance(spec, list):
            out: List[str] = []
            for s in spec:
                out += self._choices(s)
            return out
        if "item" in spec:
            return [_strip(spec["item"])]
        tag = _strip(spec["tag"])
        items = []
        for member in self.tags.get(tag, []):
            items += self.tags.get(member, [member]) if member in self.tags else [member]
        return items

    def slots(self, r: dict) -> List[List[str]]:
        if r["_kind"] == "crafting_shaped":
            cnt = Counter(ch for row in r["pattern"] for ch in row if ch != " ")
            return [self._choices(r["key"][ch]) for ch, n in cnt.items() for _ in range(n)]
        if r["_kind"] == "crafting_shapeless":
            return [self._choices(i) for i in r["ingredients"]]
        return [self._choices(r["ingredient"])]

    @staticmethod
    def needs_table(r: dict) -> bool:
        if r["_kind"] == "crafting_shaped":
            return len(r["pattern"]) > 2 or max(len(row) for row in r["pattern"]) > 2
        if r["_kind"] == "crafting_shapeless":
            return len(r["ingredients"]) > 4
        return False

    def resolve(self, target: str) -> List[str]:
        t = target.strip().lower().replace("minecraft:", "").replace(" ", "_")
        t = ALIASES.get(t, t)
        if t.startswith("#"):
            return [i for i in self.tags.get(t[1:], []) if i in self.by_result]
        return [t]


@dataclass
class CraftRequest:
    kind: str        # "mc_craft" | "mc_smelt" | "place" | "use"
    target: str
    quantity: int = 1
    timeout_s: float = 10.0


class FunctionalCraftHelper:
    """Drop-in for MineEvolve's ``CraftHelper`` (same ``execute`` API)."""

    book: Optional[RecipeBook] = None

    SYNC_STEPS = 5  # no-op steps allowed for the observation to catch up

    def __init__(self, env: Any, execute_cmd: Optional[Callable[[str], None]] = None,
                 inventory: Optional[Callable[[], Mapping[str, int]]] = None,
                 noop_step: Optional[Callable[[], Any]] = None) -> None:
        if FunctionalCraftHelper.book is None:
            FunctionalCraftHelper.book = RecipeBook()
        self.env = env
        self._exec = execute_cmd or getattr(env, "execute_cmd")
        self._inv_fn = inventory or (lambda: dict((getattr(env, "info", {}) or {}).get("inventory") or {}))
        self._noop = noop_step or (lambda: env.step(env.action_space.noop()))
        self.last_error: str = ""
        self.commands: List[str] = []
        self._local: Dict[str, int] = {}

    # --------------------------------------------------------------------- API
    def execute(self, req: CraftRequest) -> bool:
        ok = self._execute(req)
        logger.info("craft %s %s x%s -> %s %s | inv=%s | cmds=%s", req.kind, req.target, req.quantity, ok,
                    self.last_error, self._inv(), self.commands)
        return ok

    def _execute(self, req: CraftRequest) -> bool:
        if req.kind in ("place", "use"):
            ok = self._count(self.book.resolve(req.target)) > 0
            self.last_error = "" if ok else f"no {req.target} in inventory"
            return ok
        if req.kind not in ("mc_craft", "mc_smelt"):
            self.last_error = f"unknown helper kind {req.kind}"
            return False
        targets = self.book.resolve(req.target)
        if not targets:
            self.last_error = f"unknown item {req.target}"
            return False
        # The observation lags chat commands by a tick, so decisions use a
        # local copy of the inventory; afterwards wait for the env to agree.
        self._local = self._inv()
        start = self._local_count(targets)
        want = max(1, int(req.quantity))
        ok = True
        while self._local_count(targets) - start < want:
            if not self._one(targets, smelt=(req.kind == "mc_smelt")):
                ok = False
                break
        if self.commands:
            self._sync()
        if ok:
            self.last_error = ""
        return ok

    def _local_count(self, items: List[str]) -> int:
        return sum(self._local.get(i, 0) for i in items)

    def _sync(self) -> None:
        for _ in range(self.SYNC_STEPS):
            obs = self._inv()
            if all(obs.get(k, 0) == v for k, v in self._local.items()):
                return
            self._noop()
        logger.warning("inventory did not sync: local=%s observed=%s", self._local, self._inv())

    # ------------------------------------------------------------- internals
    def _inv(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self._inv_fn().items() if int(v) > 0}

    def _count(self, items: List[str]) -> int:
        inv = self._inv()
        return sum(inv.get(i, 0) for i in items)

    def _one(self, targets: List[str], smelt: bool) -> bool:
        inv = {k: v for k, v in self._local.items() if v > 0}
        reasons = []
        for target in targets:
            for r in self.book.by_result.get(target, []):
                if (r["_kind"] == "smelting") != smelt:
                    continue
                plan = self._allocate(r, inv)
                if plan is None:
                    reasons.append(f"{r['_file']}: missing ingredients")
                    continue
                if smelt:
                    fuel = self._fuel(inv, plan)
                    if "furnace" not in inv:
                        reasons.append("furnace not in inventory"); continue
                    if fuel is None:
                        reasons.append("no fuel"); continue
                    plan = plan + fuel
                elif self.book.needs_table(r) and "crafting_table" not in inv:
                    reasons.append("crafting_table not in inventory"); continue
                self._apply(plan, target, r["_count"])
                return True
        self.last_error = "; ".join(sorted(set(reasons))) or f"no recipe for {targets}"
        return False

    def _allocate(self, r: dict, inv: Dict[str, int]) -> Optional[List[Tuple[str, int]]]:
        left = dict(inv)
        used: Counter = Counter()
        for choices in self.book.slots(r):
            pick = next((c for c in choices if left.get(c, 0) > 0), None)
            if pick is None:
                return None
            left[pick] -= 1
            used[pick] += 1
        return list(used.items())

    def _fuel(self, inv: Dict[str, int], plan: List[Tuple[str, int]]) -> Optional[List[Tuple[str, int]]]:
        left = dict(inv)
        for item, n in plan:
            left[item] -= n
        for item, value in sorted(FUEL_VALUE.items(), key=lambda kv: -kv[1]):
            if left.get(item, 0) > 0:
                # one smelt consumes 1/value of a fuel item; charge a whole item
                # the first time and keep the remainder as furnace burn time
                if getattr(self.env, "_fc_burn", 0.0) >= 1.0:
                    self.env._fc_burn -= 1.0
                    return []
                self.env._fc_burn = value - 1.0
                return [(item, 1)]
        if getattr(self.env, "_fc_burn", 0.0) >= 1.0:
            self.env._fc_burn -= 1.0
            return []
        return None

    def _apply(self, consume: List[Tuple[str, int]], product: str, count: int) -> None:
        for item, n in consume:
            self._cmd(f"/clear @s minecraft:{item} {n}")
            self._local[item] = self._local.get(item, 0) - n
        self._cmd(f"/give @s minecraft:{product} {count}")
        self._local[product] = self._local.get(product, 0) + count

    def _cmd(self, cmd: str) -> None:
        self.commands.append(cmd)
        self._exec(cmd)
