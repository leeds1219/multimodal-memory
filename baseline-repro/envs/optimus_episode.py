"""Run ONE Optimus-1 episode (client side) in a given evaluation env.

The Optimus-1 server (app.py: STEVE-1 + Gemini planner/reflector) is started by
the chain runner with the same per-chain working directory, because Optimus-1
resolves memory, recipes, checkpoints and step images relative to the cwd.

Glue applied here (see DEVIATIONS.md):
  * one task per process, task text injected through Hydra (``all_task`` /
    ``evaluate``), the group's own benchmark yaml for horizon / biome;
  * config composed from the absolute conf dir and ``main.__wrapped__(cfg)``
    called directly (Hydra cannot import ``optimus1.conf`` from the editable
    install; no chdir, so the relative paths keep resolving);
  * world seeded with ``env.seed(seed)`` before reset;
  * per-step monitor on the raw MineRL step (paper success criterion);
    Optimus-1's own verdict (all plan steps done) kept as ``native_success``;
  * video / action pickles off (optional recordings);
  * a janitor keeps, per subtask name, only the image Optimus-1 can read back
    (the lowest step, see app.py ``_filter_task_obs``) plus recent files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm"))

from common import EpisodeMonitor, SEEDS, TASKS, disk_free_gb, is_method_exception, llm_summary, write_result  # noqa: E402

MINUTE = 1200
O_GROUP = {"wooden": "wooden", "stone": "stone", "iron": "iron", "gold": "golden",
           "redstone": "redstone", "diamond": "diamond", "armor": "armor"}
_IMG = re.compile(r"^(?P<uuid>[^_]+)_(?P<task>.+)_(?P<step>\d+)\.jpg$")


def janitor(img_dir: Path, stop: threading.Event, keep_recent_s: float = 60.0) -> None:
    """Delete step images Optimus-1 can never read again (infra only)."""
    while not stop.wait(30.0):
        try:
            best: dict[str, tuple[int, Path]] = {}
            files = []
            now = time.time()
            for f in img_dir.iterdir():
                m = _IMG.match(f.name)
                if not m:
                    continue
                task, step = m["task"], int(m["step"])
                files.append((f, task, step))
                if task not in best or step < best[task][0]:
                    best[task] = (step, f)
            keep = {v[1] for v in best.values()}
            for f, _, _ in files:
                if f not in keep and now - f.stat().st_mtime > keep_recent_s:
                    f.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["M", "O"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--order-id", default="order0")
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--steve-port", type=int, default=None, help="env-native STEVE-1 server (Env M)")
    ap.add_argument("--steve-workdir", default=None)
    args = ap.parse_args()

    task = TASKS[args.task]
    seed = int(args.seed if args.seed is not None else SEEDS[args.task])
    ep = Path(args.episode_dir)
    ep.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)

    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)

    # Optimus-1 registers its envs as ``src.optimus1.env...`` (it is run from
    # its repo root); make that importable while cwd is the chain workdir.
    sys.path.append("/home/rag/data/official/NeurIPS24-Optimus-1")
    import yaml
    import optimus1.main as om

    from cross_glue import env_m_group, env_o_group, optimus_in_M

    genv = env_o_group(task["group"]) if args.env == "O" else env_m_group(task["group"])
    horizon = int(genv["max_minutes"]) * MINUTE
    mon = EpisodeMonitor(ep, args.task, horizon)
    if args.env == "M":
        if args.steve_port is None:
            raise SystemExit("Env M needs --steve-port (MineEvolve STEVE-1 server)")
        optimus_in_M(om, mon, task["group"], seed, steve_port=args.steve_port)

    # ---- Env O: seed + per-step monitor on Optimus-1's own env ------------
    orig_make = om.env_make

    def env_make(env_id, cfg, logger):
        env = orig_make(env_id, cfg, logger)
        raw = env.env
        inner_step = raw.step

        def monitored_step(action):
            obs, reward, done, info = inner_step(action)
            mon.on_step(action, obs, done, info)
            return obs, reward, done, info

        raw.step = monitored_step
        env.seed(seed)
        return env

    if args.env == "O":
        om.env_make = env_make

    mcdir = None
    if args.env == "O" and os.environ.get("OPTIMUS_ISOWORLD") == "1":  # D36
        from isoworld import make_private_mcdir
        mcdir = make_private_mcdir(ep)

    if os.environ.get("OPTIMUS_GOALFIX") == "1":
        # Variant "goalfix" (DECISIONS D29): Gemini answers <goal inference> with a
        # list ("stone pickaxe, cobblestone, sticks, ..."), the prompt's own example
        # with a single item. Keep the first listed item; nothing else changes.
        import re as _re
        _orig_info = om.get_info_from_plan

        def get_info_from_plan(data):
            goal, visual, env_ = _orig_info(data)
            first = _re.split(r"[,;(/]|\band\b", goal)[0].strip().rstrip(".").strip()
            return (first or goal), visual, env_

        om.get_info_from_plan = get_info_from_plan

    if os.environ.get("OPTIMUS_LOGFIX") == "1":
        # Variant "logfix" (DECISIONS D33): KnowledgeGraph._pretty_result indexes
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
        # Variant "craftfix" (DECISIONS D37): open_crating_table_wo_recipe places the
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
                self._place_down()
                for _ in range(5):
                    self._call_func("use")
                    if self.info["isGuiOpen"]:
                        break
            self._assert(self.info["isGuiOpen"], "crafting table could not be opened")

        _CH.open_crating_table_wo_recipe = open_crating_table_wo_recipe

    if os.environ.get("OPTIMUS_TAGFIX") == "1":
        # Variant "tagfix" (DECISIONS D40): crafting_shaped fills every cell of a tag
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
        # Variant "promptfix" (DECISIONS D43): the sub-goal text is STEVE-1's prompt. The
        # authors' memory always says "dig down and mine/break down <ore>" (thousands of
        # plans); Gemini sometimes writes "mine cobblestone" / "find and mine coal", which
        # STEVE-1 cannot do on the surface (all suite_optimus1 runs: cobblestone/stone with
        # "dig down" 377/390, without 19/122; "mine cobblestone" 0/10). Mining sub-goals for
        # underground blocks get the memory's wording; nothing else in the plan changes.
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
        # Variant "replanfix" (DECISIONS D41): the reflector's REPLAN verdict is parsed
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

        # Variant "escapefix" (DECISIONS D42), on top of replanfix: re-querying the planner
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
            if os.environ.get("OPTIMUS_ESCAPEFIX") == "1" and predicament in ("drop_down", "in_water"):
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

    native = {}
    orig_do = om.agent_do

    def agent_do(*a, **k):
        status, steps, plan = orig_do(*a, **k)
        native.update(status=status, steps=steps)
        return status, steps, plan

    om.agent_do = agent_do

    stop = threading.Event()
    img_dir = Path("imgs"); img_dir.mkdir(exist_ok=True)
    threading.Thread(target=janitor, args=(img_dir, stop), daemon=True).start()

    instr = task["instruction"].replace('"', "'")
    overrides = [
        f"benchmark={O_GROUP[task['group']]}",
        f'all_task=[{{id:0,type:{task["type"]},instruction:"{instr}"}}]',
        "evaluate=[0]", "env.times=1", f"server.port={args.port}",
        "record.video.save=false", "record.action.save=false",
    ]
    if args.env == "O":
        overrides.append(f"env.max_minutes={int(genv['max_minutes'])}")  # suite may pin the paper's horizon
        if os.environ.get("OPTIMUS_BIOME"):  # variant "forest" (DECISIONS D38): released stone/iron yaml use plains
            overrides.append(f"env.prefer_biome={os.environ['OPTIMUS_BIOME']}")
        if os.environ.get("OPTIMUS_INIT_INV"):  # test-only: hydra list, e.g. [{type:oak_log,quantity:2,slot:0}]
            overrides.append(f"env.initial_inventory={os.environ['OPTIMUS_INIT_INV']}")
    log_file = open(ep / "client.log", "a")
    sys.stdout = sys.stderr = log_file
    from hydra import compose, initialize_config_dir

    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict

    with initialize_config_dir(config_dir=str(Path(om.__file__).parent / "conf"), version_base=None):
        cfg = compose(config_name="evaluate", overrides=overrides, return_hydra_config=True)
    (ep / "hydra").mkdir(exist_ok=True)
    with open_dict(cfg):
        cfg.hydra.runtime.output_dir = str(ep / "hydra")
    HydraConfig.instance().set_config(cfg)  # Optimus-1's logger reads output_dir from it
    status, err = "finished", None
    t0 = time.time()
    try:
        om.main.__wrapped__(cfg)
    except SystemExit:
        pass
    except BaseException as e:
        err = traceback.format_exc()
        print(err)
        # The JARVIS helpers raise RuntimeError("Timeout!") when the horizon
        # runs out mid-GUI-action (uncaught in optimus1.main as released):
        # that is the episode's normal end, not a crash.
        if mon.steps >= horizon or mon.over or "Timeout!" in str(e):
            mon.end_reason = mon.end_reason or "horizon"
            native.setdefault("status", "failed")
        elif is_method_exception(e):  # released code failed: episode over (D25)
            mon.end_reason = "method_exception"
            native.setdefault("status", "failed")
        else:
            status = "crashed"
    finally:
        mon.close()
        stop.set()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__

    if not native:
        status = "crashed" if status == "finished" else status
        err = err or "agent_do did not return (see client.log)"
    llm = llm_summary(ep)
    # Optimus-1 silently falls back to its built-in example plan when the
    # planning request is never made (e.g. retrieve_graph KeyError on the
    # inferred goal): record where the executed plan came from (DECISIONS D20).
    plan_source = "llm" if llm["llm_calls_by_caller"].get("planning") else "example_fallback"
    anomaly = (ep / "ANOMALY").exists()
    result = {
        "env": args.env, "method": os.environ.get("METHOD", "optimus1"), "task": args.task,
        "group": task["group"], "instruction": task["instruction"], "seed": seed, "order_id": args.order_id,
        "status": "budget_stopped" if (ep / "BUDGET_STOP").exists() else ("anomaly" if anomaly else status),
        "end_reason": mon.end_reason or ("plan_finished" if native.get("status") == "success" else "plan_failed_or_timeout"),
        "success": mon.success_step is not None, "success_step": mon.success_step,
        "native_success": (native.get("status") == "success") if native else None,
        "steps": mon.steps, "horizon_steps": horizon, "wall_time_s": round(time.time() - t0, 1),
        "final_inventory": mon.last_inventory, "plan_source": plan_source, **llm, "error": err,
        "disk_free_gb": round(disk_free_gb(), 1),
    }
    if mcdir is not None:  # D36: the seed the world was actually generated with
        from isoworld import cleanup, world_seeds
        ws = world_seeds(mcdir)
        result["world_seeds"] = ws
        result["world_seed_ok"] = bool(ws) and all(v == seed for v in ws.values())
        cleanup(mcdir)
    write_result(ep, result)
    print(json.dumps({k: result[k] for k in ("task", "status", "success", "steps", "llm_calls", "cost_usd")}))
    return 0 if status != "crashed" else 3


if __name__ == "__main__":
    sys.exit(main())
