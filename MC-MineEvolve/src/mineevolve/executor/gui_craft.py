"""GUI crafting controller for MineRL 1.0 (640x360), used by ``executor_hint: mc_craft``.

MineRL 1.0 has no craft action; crafting is done the way a player does it:
open the inventory (2x2 grid) or a placed crafting table (3x3), move the
mouse over slots, left-click to pick up / put down a stack, right-click to
place one item, take the result, close the GUI. This is the same mechanism
JARVIS-1 / DEPS use (their scripts are the reference for the slot pixel
positions and the camera-to-cursor scale); the code here is our own.

Requirements (both in this repo):
  * the MineEvolve jar patch, which adds the slot index to the inventory
    observation (``env/slot_inventory.py``), and
  * vanilla 1.16.5 recipe / tag JSON in ``assets/`` (extracted from the
    Minecraft client jar), so any shaped / shapeless recipe resolves.

The controller steps the *inner* MineRL env: the MineEvolve wrapper zeroes
``inventory`` / ``hotbar.*`` / ``use`` keys by default.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..env.slot_inventory import decode_slots

logger = logging.getLogger("mineevolve.executor.gui_craft")

WIDTH, HEIGHT = 640, 360
CAMERA_PER_PIXEL = 360.0 / 2400.0  # camera degrees per cursor pixel while a GUI is open
ASSETS = Path(__file__).resolve().parents[3] / "assets"

# Slot centres (pixels) for the two GUIs, recipe book closed. Layout is the
# vanilla 1.16 one; JARVIS-1's tables use the same numbers.
_INV_GRID = {"lt": (329, 114), "rb": (365, 150), "rows": 2, "cols": 2}
_INV_RESULT = (394, 133)
_TABLE_GRID = {"lt": (261, 113), "rb": (315, 167), "rows": 3, "cols": 3}
_TABLE_RESULT = (364, 140)
_HOTBAR = {"lt": (239, 238), "rb": (401, 256), "rows": 1, "cols": 9}
_MAIN = {"lt": (239, 180), "rb": (401, 234), "rows": 3, "cols": 9}


def _centres(spec: dict) -> List[Tuple[int, int]]:
    (x0, y0), (x1, y1) = spec["lt"], spec["rb"]
    w, h = (x1 - x0) // spec["cols"], (y1 - y0) // spec["rows"]
    return [(x0 + c * w + w // 2, y0 + r * h + h // 2) for r in range(spec["rows"]) for c in range(spec["cols"])]


INV_SLOT_PX = {i: p for i, p in enumerate(_centres(_HOTBAR))}          # slots 0-8
INV_SLOT_PX.update({9 + i: p for i, p in enumerate(_centres(_MAIN))})  # slots 9-35
GRID_PX = {2: _centres(_INV_GRID), 3: _centres(_TABLE_GRID)}
RESULT_PX = {2: _INV_RESULT, 3: _TABLE_RESULT}


# ----------------------------------------------------------------------
# Recipes
# ----------------------------------------------------------------------

class Recipes:
    def __init__(self) -> None:
        self.recipes: Dict[str, dict] = json.loads((ASSETS / "recipes_1.16.5.json").read_text())
        tags = json.loads((ASSETS / "item_tags_1.16.5.json").read_text())
        self.tags: Dict[str, List[str]] = {}
        for name, body in tags.items():
            self.tags[name] = [self._resolve_tag_value(v, tags) for v in body.get("values", [])]
            self.tags[name] = [x for sub in self.tags[name] for x in (sub if isinstance(sub, list) else [sub])]

    def _resolve_tag_value(self, v, tags, depth=0):
        v = str(v)
        if v.startswith("#"):
            inner = v[1:].replace("minecraft:", "")
            if depth > 4 or inner not in tags:
                return []
            return [self._resolve_tag_value(x, tags, depth + 1) for x in tags[inner].get("values", [])]
        return v.replace("minecraft:", "")

    def options(self, ingredient) -> List[str]:
        """All item ids that satisfy one recipe ingredient (item, tag, or list)."""
        if isinstance(ingredient, list):
            return [x for ing in ingredient for x in self.options(ing)]
        if "item" in ingredient:
            return [ingredient["item"].replace("minecraft:", "")]
        if "tag" in ingredient:
            return list(self.tags.get(ingredient["tag"].replace("minecraft:", ""), []))
        return []

    def find(self, target: str) -> List[dict]:
        """Crafting recipes producing ``target`` (vanilla names, e.g. wooden_pickaxe)."""
        target = target.replace("minecraft:", "")
        out = []
        for name, r in self.recipes.items():
            if r.get("type") not in ("minecraft:crafting_shaped", "minecraft:crafting_shapeless"):
                continue
            res = r.get("result", {})
            item = (res.get("item") if isinstance(res, dict) else res) or ""
            if item.replace("minecraft:", "") == target:
                out.append(r)
        # prefer 2x2-able recipes (no table needed), then fewer ingredients
        return sorted(out, key=lambda r: (self.grid_size(r), len(self.cells(r))))

    def cells(self, r: dict) -> List[Tuple[int, int, Any]]:
        """[(row, col, ingredient)] on a grid; shapeless recipes are laid out row-major."""
        if r["type"] == "minecraft:crafting_shaped":
            out = []
            for i, row in enumerate(r["pattern"]):
                for j, ch in enumerate(row):
                    if ch != " ":
                        out.append((i, j, r["key"][ch]))
            return out
        ings = r.get("ingredients", [])
        n = 2 if len(ings) <= 4 else 3
        return [(k // n, k % n, ing) for k, ing in enumerate(ings)]

    def grid_size(self, r: dict) -> int:
        cells = self.cells(r)
        rows = max((c[0] for c in cells), default=0) + 1
        cols = max((c[1] for c in cells), default=0) + 1
        return 2 if rows <= 2 and cols <= 2 else 3

    @staticmethod
    def result_count(r: dict) -> int:
        res = r.get("result", {})
        return int(res.get("count", 1)) if isinstance(res, dict) else 1


# ----------------------------------------------------------------------
# GUI controller
# ----------------------------------------------------------------------

class GuiCraftController:
    def __init__(self, env: Any, recipes: Optional[Recipes] = None) -> None:
        self.wrapper = env                       # MineEvolveEnvWrapper
        self.env = getattr(env, "env", env)      # inner MineRL env (no key sanitising)
        self.recipes = recipes or Recipes()
        self.obs: Dict[str, Any] = {}
        self.cursor = [WIDTH // 2, HEIGHT // 2]
        self.steps = 0

    # -- low level ------------------------------------------------------
    def _noop(self):
        return self.env.action_space.noop()

    def _step(self, action, n: int = 1):
        for _ in range(n):
            self.obs, _r, done, _i = self.env.step(action)
            self.steps += 1
            if done:
                raise RuntimeError("episode ended during crafting")
        return self.obs

    def _key(self, name: str, hold: int = 1, settle: int = 3):
        a = self._noop(); a[name] = np.array(1)
        self._step(a, hold)
        self._step(self._noop(), settle)

    def slots(self) -> Dict[int, Tuple[str, int]]:
        return decode_slots(self.obs.get("inventory_slots"))

    def gui_open(self) -> bool:
        return bool(self.obs.get("isGuiOpen", False))

    def move_to(self, x: int, y: int):
        dx, dy = x - self.cursor[0], y - self.cursor[1]
        n = max(1, int(math.ceil(max(abs(dx), abs(dy)) / 40.0)))
        for _ in range(n):
            a = self._noop()
            a["camera"] = np.array([dy / n * CAMERA_PER_PIXEL, dx / n * CAMERA_PER_PIXEL], dtype=np.float32)
            self._step(a)
        self.cursor = [x, y]
        self._step(self._noop())

    def left_click(self):
        self._key("attack", settle=2)

    def right_click(self, n: int = 1):
        for _ in range(n):
            self._key("use", settle=2)

    # -- GUI open / close ------------------------------------------------
    def open_inventory(self):
        self._step(self._noop(), 2)
        if not self.gui_open():
            self._key("inventory", settle=6)
        self.cursor = [WIDTH // 2, HEIGHT // 2]
        if not self.gui_open():
            raise RuntimeError("inventory GUI did not open")

    def close_gui(self):
        if self.gui_open():
            self._key("inventory", settle=4)
        self._step(self._noop(), 2)

    def open_table(self):
        """Place the crafting table under the agent and open it (returns after crafting via ``pickup_table``)."""
        slot = self._find("crafting_table")
        if slot is None:
            raise RuntimeError("no crafting_table in inventory")
        if slot > 8:  # move it to hotbar slot 0 first
            self.open_inventory()
            self._swap_to_hotbar(slot, 0)
            self.close_gui()
            slot = 0
        self._key(f"hotbar.{slot + 1}", settle=2)
        # look straight down, jump and place under the feet, then use it
        a = self._noop(); a["camera"] = np.array([88.0, 0.0], dtype=np.float32); self._step(a, 2)
        a = self._noop(); a["jump"] = np.array(1); self._step(a, 1)
        a = self._noop(); a["use"] = np.array(1); self._step(a, 1)
        self._step(self._noop(), 4)
        for _ in range(4):
            self._key("use", settle=5)
            if self.gui_open():
                break
        if not self.gui_open():
            raise RuntimeError("crafting table GUI did not open")
        self.cursor = [WIDTH // 2, HEIGHT // 2]

    def pickup_table(self):
        """Break the placed table (still looking down; ~75 ticks by hand) so it drops
        under the agent and is picked up, then look back up."""
        before = sum(q for n, q in self.slots().values() if n == "crafting_table")
        a = self._noop(); a["attack"] = np.array(1)
        for _ in range(120):
            self._step(a)
            if sum(q for n, q in self.slots().values() if n == "crafting_table") > before:
                break
        self._step(self._noop(), 15)  # let the drop get picked up
        a = self._noop(); a["camera"] = np.array([-88.0, 0.0], dtype=np.float32); self._step(a, 2)
        if sum(q for n, q in self.slots().values() if n == "crafting_table") <= before:
            logger.warning("crafting table was not recovered after crafting")

    # -- inventory helpers -------------------------------------------------
    def _find(self, item: str, min_qty: int = 1) -> Optional[int]:
        for s, (name, qty) in sorted(self.slots().items()):
            if name == item and qty >= min_qty:
                return s
        return None

    def _empty_slot(self) -> Optional[int]:
        used = set(self.slots())
        for s in list(range(9, 36)) + list(range(0, 9)):
            if s not in used:
                return s
        return None

    def _swap_to_hotbar(self, src: int, dst: int):
        """With a GUI open: pick up stack at src, drop into dst; if dst was occupied the
        cursor now holds that stack, so put it where src was (never close a GUI with an
        item on the cursor - vanilla drops it on the ground)."""
        dst_occupied = dst in self.slots()
        self.move_to(*INV_SLOT_PX[src]); self.left_click()
        self.move_to(*INV_SLOT_PX[dst]); self.left_click()
        if dst_occupied:
            self.move_to(*INV_SLOT_PX[src]); self.left_click()
        self._step(self._noop(), 2)

    # -- crafting ----------------------------------------------------------
    def craft(self, target: str, count: int = 1) -> bool:
        """Craft ``count`` of ``target`` (vanilla item id). Returns True on success."""
        target = target.replace("minecraft:", "")
        self._step(self._noop(), 4)  # let pickups / commands land in the inventory observation
        have0 = sum(q for n, q in self.slots().values() if n == target)
        if not self.recipes.find(target):
            logger.warning("no crafting recipe for %s", target)
            return False
        for r in self.recipes.find(target):
            plan = self._plan(r, count)
            if plan is None:
                continue
            grid, per_cell, crafts, result_n = plan
            logger.info("craft %s x%d via %s grid (%d crafts)", target, count, f"{grid}x{grid}", crafts)
            try:
                if grid == 3:
                    self.open_table()
                else:
                    self.open_inventory()
                self._fill_and_take(grid, per_cell, crafts, target)
                for _retry in range(2):  # a missed output click leaves ingredients in the grid
                    have = sum(q for n, q in self.slots().values() if n == target)
                    if have - have0 >= count:
                        break
                    missing = int(math.ceil((count - (have - have0)) / result_n))
                    logger.info("craft %s: %d short, clicking the output %d more time(s)", target, count - (have - have0), missing)
                    self.move_to(*RESULT_PX[grid])
                    for _ in range(missing):
                        self._key("attack", settle=5)
                    dst = self._empty_slot()
                    if dst is None:
                        break
                    self.move_to(*INV_SLOT_PX[dst]); self.left_click(); self._step(self._noop(), 3)
                self.close_gui()
                if grid == 3:
                    self.pickup_table()
            except RuntimeError as exc:
                logger.warning("craft %s failed: %s", target, exc)
                self.close_gui()
            have = sum(q for n, q in self.slots().values() if n == target)
            if have - have0 >= count:
                return True
        return sum(q for n, q in self.slots().values() if n == target) - have0 >= count

    def _plan(self, r: dict, count: int):
        """Choose concrete items from the inventory for each recipe cell."""
        grid = self.recipes.grid_size(r)
        crafts = int(math.ceil(count / self.recipes.result_count(r)))
        avail = {}
        for s, (name, qty) in self.slots().items():
            avail[name] = avail.get(name, 0) + qty
        per_cell: List[Tuple[int, str]] = []  # (grid index, item)
        need: Dict[str, int] = {}
        for row, col, ing in self.recipes.cells(r):
            choice = None
            for opt in self.recipes.options(ing):
                if avail.get(opt, 0) - need.get(opt, 0) >= crafts:
                    choice = opt
                    break
            if choice is None:
                return None
            need[choice] = need.get(choice, 0) + crafts
            per_cell.append((row * grid + col, choice))
        if grid == 3 and avail.get("crafting_table", 0) < 1:
            return None
        return grid, per_cell, crafts, self.recipes.result_count(r)

    def _fill_and_take(self, grid: int, per_cell: List[Tuple[int, str]], crafts: int, target: str):
        cells = GRID_PX[grid]
        # place `crafts` items of the chosen ingredient into each cell
        by_item: Dict[str, List[int]] = {}
        for idx, item in per_cell:
            by_item.setdefault(item, []).append(idx)
        for item, idxs in by_item.items():
            remaining = list(idxs)
            while remaining:
                src = self._find(item, 1)
                if src is None:
                    raise RuntimeError(f"ran out of {item}")
                qty = self.slots()[src][1]
                self.move_to(*INV_SLOT_PX[src]); self.left_click()   # pick up the whole stack
                placed_from_stack = 0
                while remaining and placed_from_stack + crafts <= qty:
                    idx = remaining.pop(0)
                    self.move_to(*cells[idx]); self.right_click(crafts)
                    placed_from_stack += crafts
                if placed_from_stack < qty:                            # put the remainder back
                    self.move_to(*INV_SLOT_PX[src]); self.left_click()
        # take the result: each left-click on the output crafts once onto the
        # cursor stack; clicks need a few ticks between them to register.
        rx, ry = RESULT_PX[grid]
        self.move_to(rx, ry)
        for _ in range(crafts):
            self._key("attack", settle=5)
        dst = self._empty_slot()
        if dst is None:
            raise RuntimeError("inventory full")
        self.move_to(*INV_SLOT_PX[dst]); self.left_click()
        self._step(self._noop(), 3)
        if self._find(target) is None:
            raise RuntimeError(f"{target} did not appear in the inventory")
