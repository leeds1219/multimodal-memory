"""Crafting helper for executor_hint in {mc_craft, mc_smelt, place, use}.

``mc_craft`` drives the real inventory / crafting-table GUI through
``gui_craft.GuiCraftController`` (upstream's version only played a sound and
polled the inventory, so no crafting task could ever succeed). ``mc_smelt``
places and drives the agent's furnace, ``equip`` moves an item to the hotbar
and selects it. No /give shortcuts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping


logger = logging.getLogger("mineevolve.executor.craft_helper")


@dataclass
class CraftRequest:
    """A single craft / smelt / place request issued by the planner."""

    kind: str        # "mc_craft" | "mc_smelt" | "place" | "use"
    target: str      # canonical item id, e.g. "iron_pickaxe"
    quantity: int = 1
    timeout_s: float = 10.0


class CraftHelper:
    """Run a craft / smelt request using inventory-delta polling."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.last_steps = 0
        self.episode_ended = False  # the env returned done while the GUI script was stepping it

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, req: CraftRequest) -> bool:
        """Execute a CraftRequest. Returns True iff target item count reached."""

        if req.kind not in ("mc_craft", "mc_smelt", "place", "use", "equip"):
            logger.warning("Unknown craft kind %s", req.kind)
            return False

        baseline = self._inventory_count(req.target)
        deadline = time.monotonic() + max(1.0, float(req.timeout_s))
        self.last_steps = 0

        if req.kind in ("mc_craft", "mc_smelt", "equip"):
            from .gui_craft import GuiCraftController

            ctl = GuiCraftController(self.env)
            try:
                if req.kind == "mc_craft":
                    ok = ctl.craft(req.target, int(req.quantity))
                elif req.kind == "mc_smelt":
                    ok = ctl.smelt(req.target, int(req.quantity))
                else:
                    ok = ctl.equip(req.target)
            except Exception as exc:
                logger.warning("GUI crafting of %s failed: %s", req.target, exc)
                ok = False
                if "episode ended" in str(exc):
                    self.episode_ended = True
            finally:
                self.last_steps = ctl.steps
                try:  # the controller stepped the inner env; refresh the wrapper's status/inventory view
                    self.env.step(self.env.action_space.noop())
                except Exception:
                    pass
            return bool(ok)

        if req.kind == "place":
            return self._place(req.target, baseline)
        if req.kind == "use":
            return self._use(req.target)

        # mc_craft / mc_smelt: open the relevant GUI then poll inventory
        # for the target item count to reach baseline + quantity.
        self._open_gui(req.kind)

        while time.monotonic() < deadline:
            current = self._inventory_count(req.target)
            if current - baseline >= int(req.quantity):
                return True
            time.sleep(0.2)

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inventory_count(self, item: str) -> int:
        info = getattr(self.env, "info", {}) or {}
        inv = info.get("inventory") if isinstance(info, Mapping) else None
        if not isinstance(inv, Mapping):
            return 0
        try:
            return int(inv.get(item, 0))
        except (TypeError, ValueError):
            return 0

    def _open_gui(self, kind: str) -> None:
        execute = getattr(self.env, "execute_cmd", None)
        if not callable(execute):
            return
        # We do not have a privileged crafting API; we use chat to ensure
        # the player has the GUI open by switching to inventory hotkey.
        if kind == "mc_smelt":
            execute("/playsound minecraft:block.furnace.fire_crackle master @s")
        else:
            execute("/playsound minecraft:block.crafting_table.fall master @s")

    def _place(self, item: str, baseline_count: int) -> bool:
        # We cannot programmatically place a block from this layer alone;
        # the planner is expected to issue a follow-up STEVE-1 subgoal that
        # actually drops + places the item. Returning True here just means
        # the item is available in inventory.
        return baseline_count > 0

    def _use(self, item: str) -> bool:
        return self._inventory_count(item) > 0
