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

        def _escape(env, predicament, logger):
            hot, use_ok = env.can_change_hotbar, True
            env.can_change_hotbar = True
            try:
                _, _, done, info = _act(env)
                if predicament == "in_water":  # go_to_land
                    _act(env, 2, camera=[-30, 0])
                    _act(env, 200, forward=1, jump=1, sprint=1)
                    logger.warning("escapefix: go_to_land (200 steps)")
                    return
                inv = info.get("plain_inventory", {})
                slot = next((s for s in range(9) if inv.get(s, {}).get("type") in PLACEABLE), None)
                built = 0
                if slot is not None:  # build_tower: pillar up while blocks last and height grows
                    _act(env, 1, **{f"hotbar.{slot + 1}": 1})
                    _act(env, 2, camera=[88, 0])
                    stalls = 0
                    for _ in range(15):
                        y0 = info["location_stats"]["ypos"]
                        _act(env, 1, jump=1)
                        _act(env, 2)
                        _, _, done, info = _act(env, 1, use=1)
                        _, _, done, info = _act(env, 3)
                        if done or inv.get(slot, {}).get("type") != info["plain_inventory"].get(slot, {}).get("type"):
                            break
                        stalls = stalls + 1 if info["location_stats"]["ypos"] <= y0 + 0.5 else 0
                        built += stalls == 0
                        if stalls >= 2:
                            break
                    _act(env, 2, camera=[-88, 0])
                _act(env, 6, camera=[0, 30])  # turn ~180 deg, then leave the spot
                _act(env, 60, forward=1, jump=1)
                logger.warning(f"escapefix: build_tower ({built} blocks, hotbar slot {slot}) + walk away")
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
