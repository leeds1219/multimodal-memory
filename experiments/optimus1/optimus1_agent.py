"""Optimus-1 reproduction (EXPERIMENTS.md 1): labelled fixes and diagnostics applied to the released code.

Each block is switched on by one variant token (experiments/chain.py -> OPTIMUS_* environment variable) and patches
the released module in memory; the authors' files are not edited. Called by world/episode.py before the episode runs.
The Baseline agent and everything after it use none of these except `isoworld` (world/private_world.py).
"""
from __future__ import annotations

import os


def apply(om, mon) -> None:
    """Patch optimus1.main (`om`) and its modules for the variant tokens of this episode."""
    import numpy as np

    if os.environ.get("OPTIMUS_LOGFIX") == "1":
        # Token logfix: KnowledgeGraph._pretty_result indexes
        # summary[item] for every node of the sub-graph, but raw materials (e.g.
        # oak_log, reached through the hard-coded oak planks path) never enter
        # summary -> KeyError. Released code then either crashes at step 0
        # (retrieve_plan, UnboundLocalError on `example`) or during replanning.
        # Fix: an uncounted node prints as "need ??", the code's own placeholder.
        from optimus1.memories.graph import KnowledgeGraph
        _orig_pretty = KnowledgeGraph._pretty_result

        def _pretty_result(self, summary, base, sub_graph, in_degree=None):
            from collections import defaultdict
            return _orig_pretty(self, defaultdict(int, summary), base, sub_graph, in_degree)

        KnowledgeGraph._pretty_result = _pretty_result

    if os.environ.get("OPTIMUS_CRAFTFIX") == "1":
        # Token craftfix: open_crating_table_wo_recipe places the
        # table at the agent's feet and presses "use", but never checks that the
        # table GUI opened. Over water the placement silently fails and the helper
        # clicks slots of a GUI that is not there ("fail for unkown reason") until
        # the horizon. Fix: if the GUI is closed, step aside and retry the released
        # placement up to 3 times; if it is still closed, fail the craft honestly.
        from optimus1.helper.jarvis_craft_helper import CraftHelper as _CH
        _orig_open = _CH.open_crating_table_wo_recipe

        def open_crating_table_wo_recipe(self):
            _orig_open(self)
            for _ in range(3):
                if self.info["isGuiOpen"]:
                    return
                self.turn_left(); self.turn_left()  # 90 deg away from the failed spot
                for _ in range(10):
                    self._call_func("forward")
                # authors' later mitigation for confined spaces (Optimus-3 craft_agent: attack_num 40)
                self.pre_open_tabel(attack_num=40)
                self._place_down()
                for _ in range(5):
                    self._call_func("use")
                    if self.info["isGuiOpen"]:
                        break
            self._assert(self.info["isGuiOpen"], "crafting table could not be opened")

        _CH.open_crating_table_wo_recipe = open_crating_table_wo_recipe

    if os.environ.get("OPTIMUS_TAGFIX") == "1":
        # Token tagfix: crafting_shaped fills every cell of a tag
        # ingredient (minecraft:logs, :planks) from the FIRST matching inventory stack
        # and fails "missing material" if that one stack is short, even when stacks of
        # other wood types cover it (smoker with 2 birch + 2 oak logs; Minecraft accepts
        # mixed types). Fix: only when such a mix is needed, fill the cells stack by
        # stack; every other craft runs the released code unchanged.
        import json as _json
        from optimus1.helper import jarvis_craft_helper as _jch
        _CH2 = _jch.CraftHelper
        _orig_shaped = _CH2.crafting_shaped

        def _members(self, key):
            if key.get("item"):
                return {key["item"][10:]}
            tags = _json.load(open(os.path.join(self.root_path, "tag_items.json")))
            return {x[10:] for x in tags[key["tag"]]}

        def _stacks(labels, members):
            return sorted(((s, v["quantity"]) for s, v in labels.items()
                           if s.startswith("inventory_") and v.get("type") in members),
                          key=lambda t: -t[1])

        def _cells(pattern, sym, width):
            return [i * width + j for i in range(len(pattern)) for j in range(len(pattern[i])) if pattern[i][j] == sym]

        def crafting_shaped(self, target, iter_num, recipe_info):
            pattern, keys = recipe_info.get("pattern"), recipe_info.get("key")
            labels = self.get_labels()
            mix = False
            for sym, key in keys.items():
                if key.get("tag"):
                    need = len(_cells(pattern, sym, 1)) * iter_num
                    first = self.find_in_inventory(labels, key["tag"][10:], "tag")
                    have = labels[first]["quantity"] if first else 0
                    mix |= have < need <= sum(q for _, q in _stacks(labels, _members(self, key)))
            if not mix:
                return _orig_shaped(self, target, iter_num, recipe_info)
            width = 3 if "table" in self.current_gui_type else 2
            for sym, key in _jch.random_dic(keys).items():
                labels = self.get_labels()
                cells = _cells(pattern, sym, width)
                stacks = [s for s in _stacks(labels, _members(self, key)) if s[1] >= iter_num]
                name = (key.get("item") or key.get("tag"))[10:]
                self._assert(sum(q for _, q in stacks) >= len(cells) * iter_num,
                             _jch.MISSING_MATERIAL_FORMAT.format(name, len(cells) * iter_num))
                holding, held = None, 0
                for cell in cells:
                    if held < iter_num:
                        if holding:
                            self.pull_item_return(self.crafting_slotpos, holding)
                        holding, qty = stacks.pop(0)
                        self.pull_item(self.crafting_slotpos, holding, f"resource_{cell}", iter_num)
                        held = qty - iter_num
                    else:
                        self.pull_item_continue(self.crafting_slotpos, f"resource_{cell}", name, iter_num)
                        held -= iter_num
                if held > 0:
                    self.pull_item_return(self.crafting_slotpos, holding)

        _CH2.crafting_shaped = crafting_shaped

        # crafting(<tag>, n) in the release only succeeds if ONE member recipe can make all n
        # (e.g. 19 planks from oak logs alone) and otherwise fails "missing material" although
        # several log types together suffice (stone_01 seed3: 4 birch + 2 oak logs, "craft
        # planks 19" failed, then a chop/craft loop until timeout). On that failure, craft the
        # members one after another from what the inventory holds.
        _orig_crafting = _CH2.crafting

        def crafting(self, target, target_num=1):
            done, info = _orig_crafting(self, target, target_num)
            key = "minecraft:" + str(target)
            if done or "missing material" not in str(info) or key not in getattr(self, "tag_info", {}):
                return done, info
            remaining = int(target_num)
            for member in self.tag_info[key]:
                sub = member[10:]
                path = os.path.join(self.recipe_path, sub + ".json")
                if remaining <= 0 or not os.path.exists(path):
                    continue
                recipe = _json.load(open(path))
                ingr = recipe.get("ingredients") or []
                if len(ingr) != 1:
                    continue
                k = ingr[0]
                name, kind = ((k["item"][10:], "item") if k.get("item") else (k["tag"][10:], "tag"))
                labels = self.get_labels()
                have = sum(v["quantity"] for s_, v in labels.items() if s_.startswith("inventory_")
                           and (v.get("type") == name if kind == "item"
                                else v.get("type") in {x[10:] for x in self.tag_info.get("minecraft:" + name, [])}))
                per = int(recipe.get("result", {}).get("count", 1))
                n = min(-(-remaining // per), have)
                if n <= 0:
                    continue
                d, i = _orig_crafting(self, sub, n * per)
                if d:
                    remaining -= n * per
            return (True, None) if remaining <= 0 else (False, info)

        _CH2.crafting = crafting

        # The released task checker expands "logs" to the six *_log items only, but the
        # minecraft:logs tag the recipes use also holds stripped logs and *_wood: an agent
        # holding 3 stripped_oak_log could never finish "chop trees" (stone_00 seed0, twice).
        from optimus1.env.mods.task_checker import TaskCheckerMod as _TC
        _orig_expand = _TC._expand_item

        def _expand_item(self, item):
            out = _orig_expand(self, item)
            if "log" in item:
                woods = ["acacia", "birch", "dark_oak", "jungle", "oak", "spruce"]
                out = out + [f"stripped_{w}_log" for w in woods] + [f"{w}_wood" for w in woods] \
                    + [f"stripped_{w}_wood" for w in woods]
            return out

        _TC._expand_item = _expand_item

    if os.environ.get("OPTIMUS_PROMPTFIX") == "1":
        # Token promptfix: the sub-goal text is STEVE-1's prompt. The authors' memory always says "dig down and
        # mine/break down <ore>" (thousands of plans); Gemini sometimes writes "mine cobblestone" / "find and mine
        # coal", which STEVE-1 cannot do on the surface (all earlier Optimus-1 runs: cobblestone/stone with "dig down"
        # 377/390, without 19/122; "mine cobblestone" 0/10). Mining sub-goals for underground blocks get the memory's
        # wording; nothing else in the plan changes.
        UNDERGROUND = ("cobblestone", "stone", "coal", "iron", "gold", "diamond", "redstone", "lapis")
        _orig_render = om.render_gpt4_plan

        def render_gpt4_plan(plan, is_replan=False):
            plans = _orig_render(plan, is_replan)
            for p in plans or []:
                t, g = str(p.get("task", "")), p.get("goal") or [""]
                item = str(g[0]).lower()
                if (t.split(" ")[0] not in ("craft", "smelt", "equip") and "smelt" not in t
                        and "dig" not in t and any(u in item for u in UNDERGROUND)):
                    p["task"] = "dig down and mine " + item.replace("_", " ").replace("coals", "coal")
            return plans

        om.render_gpt4_plan = render_gpt4_plan

    if os.environ.get("OPTIMUS_REPLANFIX") == "1":
        # Token replanfix: the reflector's REPLAN verdict is parsed
        # (main.py: situation, replan_type) but never acted on - `match situation` only
        # has `case "done" | "continue": pass`. The authors (issue #11) say only a subset
        # of replanning is implemented and to add cases "after line 223" by querying the
        # planner. Done here exactly so: one `case "replan"` is compiled into agent_do,
        # which sends the predicament (the reflection prompt's own definitions) to the
        # released replan prompt, inserts the new sub-goals before the current one and
        # leaves the current sub-goal. A sub-goal whose goal is not an inventory item
        # (e.g. "climb out of the cave") would be marked done at once by the released
        # checker (KeyError -> finished), so it gets a fixed step budget instead.
        import inspect as _inspect
        import re as _re2
        import textwrap as _tw
        from optimus1.env.mods.task_checker import TaskCheckerMod

        ESCAPE_STEPS = int(os.environ.get("OPTIMUS_ESCAPE_STEPS", "600"))
        PREDICAMENT = {  # gpt4_planning.reflection_systerm
            "drop_down": '"drop_down" means that the agent has fallen into a cave or is trapped in a mountain or river',
            "in_water": '"in_water" means that the agent is in the ocean and needs to return to land immediately',
        }
        _n_escape = [0]

        # Token escapefix, on top of replanfix: re-querying the planner
        # gives "find trees"-like sub-goals that STEVE-1 cannot use to leave a pit / a dug-in
        # hillside (test on 4 failed worlds: 0/4). The release still calls
        # replan_helper.build_tower() for drop_down and go_to_land() for in_water
        # (helper.py), but never shipped that class; this is our reconstruction of it.
        PLACEABLE = ("dirt", "cobblestone", "stone", "andesite", "diorite", "granite", "netherrack",
                     "oak_planks", "birch_planks", "spruce_planks", "jungle_planks", "acacia_planks", "dark_oak_planks")

        def _act(env, n=1, **keys):
            out = None
            for _ in range(n):
                a = env.noop_action()
                for k, v in keys.items():
                    a[k] = np.array(v)
                out = env.step(a)
                if out[2]:  # horizon reached: stop, agent_do's own loop ends the episode
                    break
            return out

        OCEAN = {0, 10, 24, 44, 45, 46, 47, 48, 49, 50}  # 1.16 ocean biome ids

        def _loc(out):
            obs, _, _, info = out
            return (obs.get("location_stats") if isinstance(obs, dict) else None) or info.get("location_stats", {})

        def _xz(out):
            loc = _loc(out)
            return float(loc.get("xpos", 0)), float(loc.get("zpos", 0))

        def _go_to_land(env, logger):
            # Swim toward the spawn point (always land) until out of the ocean biome or back at spawn.
            import math
            sp, steps, out = mon.first_pos, 0, _act(env)
            while steps < 1200 and not out[2]:
                loc = _loc(out)
                x, z = _xz(out)
                if sp and math.hypot(sp[0] - x, sp[2] - z) < 4:
                    break
                if steps > 40 and int(np.asarray(loc.get("biome_id", 0))) not in OCEAN:
                    break
                cam = [-float(np.asarray(loc.get("pitch", 0))), 0.0]
                if sp and "yaw" in loc:  # Minecraft yaw: 0 = +z, 90 = -x
                    target = math.degrees(math.atan2(-(sp[0] - x), sp[2] - z))
                    cam[1] = max(-90.0, min(90.0, (target - float(np.asarray(loc["yaw"])) + 180) % 360 - 180))
                _act(env, 1, camera=cam)
                out = _act(env, 20, forward=1, jump=1, sprint=1)
                steps += 21
            logger.warning(f"escapefix: go_to_land ({steps} steps)")

        def _probe_exit(env, y_start):
            # Look level, walk+jump 8 steps. Escaped if the agent can roam (>= 3 blocks) or has
            # climbed out (>= 1.5 blocks moved while >= 1.5 above where the escape started).
            # can_see_sky is not used: it read False in every escapefix3 episode.
            out = _act(env)
            x0, z0 = _xz(out)
            _act(env, 1, camera=[-float(np.asarray(_loc(out).get("pitch", 0))), 0])
            _act(env, 8, forward=1, jump=1)
            out = _act(env, 6)
            x1, z1 = _xz(out)
            moved = ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5
            y1 = float(np.asarray(_loc(out).get("ypos", 0)))
            return moved >= 3 or (moved >= 1.5 and y1 >= y_start + 1.5), out

        def _build_tower(env, logger):
            out = _act(env)
            inv = out[3].get("plain_inventory", {})
            slot = next((s for s in range(9) if inv.get(s, {}).get("type") in PLACEABLE), None)
            if slot is None:  # no block to stand on yet: dig two from the wall in front (not the floor)
                _act(env, 1, camera=[-float(np.asarray(_loc(out).get("pitch", 0))), 0])
                for _ in range(2):
                    _act(env, 40, attack=1)
                    _act(env, 10, forward=1)
                out = _act(env, 5)
                inv = out[3].get("plain_inventory", {})
                slot = next((s for s in range(9) if inv.get(s, {}).get("type") in PLACEABLE), None)
            built = dug = stall = 0
            ok = False
            y_start = float(np.asarray(_loc(out).get("ypos", 0)))
            for i in range(20):
                ok, out = _probe_exit(env, y_start)
                if ok or out[2]:
                    break
                if slot is None:
                    break
                inv = out[3].get("plain_inventory", {})
                if inv.get(slot, {}).get("type") not in PLACEABLE:
                    slot = next((s for s in range(9) if inv.get(s, {}).get("type") in PLACEABLE), None)
                    if slot is None:
                        break
                _act(env, 1, **{f"hotbar.{slot + 1}": 1})
                _act(env, 2, camera=[88, 0])
                out = _act(env, 8)  # land first: the exit probe ends with jumps, so ypos would read high
                y0 = float(np.asarray(_loc(out).get("ypos", 0)))
                _act(env, 1, jump=1)
                _act(env, 2)
                _act(env, 1, use=1)
                out = _act(env, 3)
                if float(np.asarray(_loc(out).get("ypos", 0))) > y0 + 0.5:
                    built += 1
                    stall = 0
                    if built >= 12:  # never a taller pillar: fall damage
                        break
                else:  # ceiling: dig the block above, then try again (at most 3 times in a row)
                    stall += 1
                    if stall > 3:
                        break
                    _act(env, 4, camera=[-88, 0])
                    _act(env, 30, attack=1)
                    dug += 1
            if not ok:  # leave the spot anyway
                _act(env, 6, camera=[0, 30])
                out = _act(env, 40, forward=1, jump=1)
            logger.warning(f"escapefix: build_tower ({built} placed, {dug} dug, exited={ok}, slot {slot})")

        def _escape(env, predicament, logger):
            hot = env.can_change_hotbar
            env.can_change_hotbar = True
            try:
                # The reflector's label is not reliable (an ocean spawn was called drop_down):
                # also swim when the agent is in an ocean biome at sea level.
                loc = _loc(_act(env))
                at_sea = (int(np.asarray(loc.get("biome_id", -1))) in OCEAN
                          and float(np.asarray(loc.get("ypos", 0))) >= float(np.asarray(loc.get("sea_level", 62))) - 2)
                (_go_to_land if predicament == "in_water" or at_sea else _build_tower)(env, logger)
            finally:
                env.can_change_hotbar = hot

        def _reflect_replan(task, predicament, obs, current_plan, plan_manager, memory_bank, cfg, pbar, all_task, logger, env):
            if (os.environ.get("OPTIMUS_ESCAPEFIX") == "1" and predicament in ("drop_down", "in_water")
                    and "dig" not in str(task)):  # under ground on purpose while digging down
                _escape(env, predicament, logger)
                return False  # keep executing the current sub-goal from the new position
            info = f"predicament: {predicament}. " + PREDICAMENT.get(predicament or "", "the agent is in trouble")
            try:
                examples = memory_bank.retrieve_replan(task, info)
                new = om.render_gpt4_plan(om.ServerAPI.get_plan(cfg["server"], obs, task, info, examples, ""), is_replan=True)
            except Exception as e:  # keep the current sub-goal, as the released code does on parse errors
                logger.warning(f"replanfix: replan failed ({e!r})")
                return False
            if not new:
                return False
            for p in new:
                names = env.task_checker_mod._expand_item(str((p.get("goal") or [""])[0]))
                if not all(n in obs["inventory"] for n in names):
                    _n_escape[0] += 1
                    p["goal"] = [f"__escape__{_n_escape[0]}", ESCAPE_STEPS]
            if new[-1]["task"] != task:
                new.append(current_plan)
            plan_manager.insert_plan(new, is_replan=True)
            om.set_pbar_total(pbar, all_task, len(plan_manager.all))
            logger.warning(f"[yellow]Reflection replan ({predicament})...\n{new}[/yellow]")
            memory_bank.save_replan(task, info, new)
            return True

        om._reflect_replan = _reflect_replan
        src = _tw.dedent(_inspect.getsource(om.agent_do))
        m = list(_re2.finditer(r'\n( +)case "done" \| "continue":\n(?:.*\n)*?\1    pass\n', src))
        assert len(m) == 1, "replanfix: anchor not found exactly once in agent_do"
        ind = m[0].group(1)
        case = (f'{ind}case "replan" if _reflect_replan(task, replan_type, obs, current_plan, plan_manager, '
                f'memory_bank, cfg, pbar, all_task, logger, env):\n{ind}    break\n')
        src = "\n" * (om.agent_do.__code__.co_firstlineno - 1) + src[:m[0].end()] + case + src[m[0].end():]
        exec(compile(src, _inspect.getsourcefile(om.agent_do), "exec"), om.__dict__)

        _orig_tc_step, _orig_tc_reset = TaskCheckerMod.step, TaskCheckerMod.reset

        def tc_step(self, inventory, goal):
            if goal is not None and str(goal[0]).startswith("__escape__"):
                self._cache["escape_n"] = self._cache.get("escape_n", 0) + 1
                return self._cache["escape_n"] >= int(goal[1])
            return _orig_tc_step(self, inventory, goal)

        def tc_reset(self, inventory=None):
            self._cache["escape_n"] = 0
            return _orig_tc_reset(self, inventory)

        TaskCheckerMod.step, TaskCheckerMod.reset = tc_step, tc_reset
