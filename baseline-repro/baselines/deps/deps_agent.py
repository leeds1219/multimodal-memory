"""DEPS ("Describe, Explain, Plan and Select", Wang et al. 2023) ported onto
the shared stack (MineRL 1.0 + STEVE-1 + Gemini) from the released code
github.com/CraftJarvis/MC-Planner (planner.py, main.py).

Kept verbatim from the release: task_prompt.txt, deps_prompt.txt,
parse_prompt.txt, the dialogue strings, the loop in ``Evaluator.eval_step``
(goal update, replan triggers: mine goal whose precondition/tool is missing;
craft not done after 150 steps; smelt after 200; stop after 12 replans),
LLM settings (planner T=0.7, max_tokens 1024, stop "Human:"; parser T=0,
max_tokens 256, prompt truncated to its last 4000 characters), and the
"default mine_log" goal for an empty plan.

Deviations (DEVIATIONS.md, Stage B / DEPS):
  * Codex / text-davinci-003 text completion → Gemini chat: the transcript is
    sent as one user message with a short system instruction to continue it;
    output cut at "Human:" (the code's stop sequence).
  * Selector: not in the released code (stub) → not used, as in the code
    ("DEP").
  * Controller: MineDojo policy / scripted dig-down → STEVE-1 text prompts:
    the release's own ``goal_mapping.json['mineclip']`` text where present,
    else JARVIS-1's released ``skill.json`` prompt for that item, else
    "get {item}" (JARVIS-1 convention).
  * Craft/smelt: MineDojo functional craft → the env's craft/smelt primitive
    (Env M functional, Env O Optimus GUI helpers); a failed craft keeps the
    goal active and the env idles until the release's 150/200-step replan
    trigger, as a failed MineDojo craft action does.
  * goal_lib.json extended (``goal_lib_ext.json``) with the 1.16 recipes of the
    targets it lacks; unknown goals are otherwise dropped by the code and DEPS
    would score 0 on those tasks by construction.
  * Item names: MineRL 1.16 inventory names are mapped to the release's
    MineDojo names (any *_log → log, *_planks → planks, ...) for the inventory
    description and goal checks.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
MCP = Path("/home/rag/data/official/MC-Planner/data")
JARVIS_SKILL = Path("/home/rag/data/official/JARVIS-1/jarvis/assets/skill.json")
log = logging.getLogger("baseline_repro.deps")

TOOLS = {"wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe",
         "wooden_axe", "stone_axe", "iron_axe", "diamond_axe"}
SYSTEM = ("You are a text-completion engine. Continue the transcript below exactly where it ends, "
          "in the same format. Output only the continuation.")  # ours: chat emulation of completion


def generic_name(item: str) -> str:
    """MineRL 1.16 item name → MineDojo name used by DEPS prompts / goal_lib."""
    if item.endswith("_log") or item.endswith("_wood"):
        return "log"
    if item.endswith("_planks"):
        return "planks"
    if item.endswith("_wool"):
        return "wool"
    if item.endswith("_sapling"):
        return "sapling"
    return item


def generic_inventory(inv: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in inv.items():
        g = generic_name(k)
        out[g] = out.get(g, 0) + int(v)
    return out


class DepsPlanner:
    """planner.py, with the OpenAI completion calls routed to Gemini."""

    def __init__(self, client) -> None:
        self.client = client
        self.dialogue = ""
        self.logging_dialogue = ""
        self.goal_lib = json.loads((MCP / "goal_lib.json").read_text())
        ext = HERE / "goal_lib_ext.json"
        if ext.exists():
            for k, v in json.loads(ext.read_text()).items():
                self.goal_lib.setdefault(k, v)
        self.supported_objects = {}
        for key, v in self.goal_lib.items():
            obj = list(v["output"].keys())[0]
            self.supported_objects[obj] = {**v, "name": key}
        self.task_prompt = (MCP / "task_prompt.txt").read_text()
        self.replan_prompt = (MCP / "deps_prompt.txt").read_text()
        self.parser_prompt = (MCP / "parse_prompt.txt").read_text()

    # --- LLM ---------------------------------------------------------------
    def query_codex(self, prompt_text: str) -> str:
        r = self.client.chat.completions.create(
            model="code-davinci-002",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_text}],
            temperature=0.7, max_tokens=1024, top_p=1, stop=["Human:"])
        text = r.choices[0].message.content or ""
        return text.split("Human:")[0]

    def query_gpt3(self, prompt_text: str) -> str:
        prompt_text = prompt_text[-4000:]
        r = self.client.chat.completions.create(
            model="text-davinci-003",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_text}],
            temperature=0, max_tokens=256, top_p=1)
        return r.choices[0].message.content or ""

    # --- planner.py --------------------------------------------------------
    def reset(self) -> None:
        self.dialogue = ""
        self.logging_dialogue = ""

    def online_parser(self, text: str):
        parsed = self.query_gpt3(self.parser_prompt + text)
        name = obj = rank = None
        for line in parsed.split("\n"):
            line = line.replace(" ", "")
            try:
                if "action:" in line:
                    pass
                elif "name:" in line:
                    name = line[5:]
                elif "object:" in line:
                    obj = ast.literal_eval(line[7:])  # release: eval()
                elif "rank:" in line:
                    rank = int(line[5:])
            except (ValueError, SyntaxError):
                pass
        return name, obj, rank

    def check_object(self, obj):
        try:
            name = list(obj.keys())[0]
            for goal in self.goal_lib:
                if name == list(self.goal_lib[goal]["output"].keys())[0]:
                    return goal
        except Exception:
            pass
        return False

    def generate_goal_list(self, plan: str) -> List[dict]:
        goal_list = []
        for line in plan.split("\n"):
            if "#" not in line:
                continue
            name, obj, rank = self.online_parser(f"input: {line}")
            if name in self.goal_lib:
                g = self.goal_lib[name]
                goal_list.append({"name": name, "type": g["type"], "object": obj,
                                  "precondition": {**g["precondition"], **g["tool"]}, "ranking": rank})
            elif self.check_object(obj):
                obj_name = list(obj.keys())[0]
                g = self.supported_objects[obj_name]
                goal_list.append({"name": g["name"], "type": g["type"], "object": obj,
                                  "precondition": {**g["precondition"], **g["tool"]}, "ranking": rank})
            else:
                log.info("parsed goal is not supported by current controller: %s %s", name, obj)
        log.info("Current Plan is %s", goal_list)
        return goal_list

    def initial_planning(self, task_question: str) -> str:
        question = f"Human: {task_question}\n"
        plan = self.query_codex(self.task_prompt + self.replan_prompt + question)
        self.dialogue = self.task_prompt + self.replan_prompt + question + plan
        self.logging_dialogue = question + plan
        return plan

    def generate_inventory_description(self, inventory: Dict[str, int]) -> str:
        text = "Human: My inventory now has "
        for name, q in inventory.items():
            if name in ("diamond_axe", "air"):
                continue
            text += f"{q} {name}, "
        text += "\n"
        self.dialogue += text
        self.logging_dialogue += text
        return text

    def generate_success_description(self, step) -> str:
        t = f"Human: I succeed on step {step}.\n"
        self.dialogue += t
        self.logging_dialogue += t
        return t

    def generate_failure_description(self, step) -> str:
        t = f"Human: I fail on step {step}"
        self.dialogue += t
        self.logging_dialogue += t
        detail = self.query_codex(self.dialogue)
        self.dialogue += detail
        self.logging_dialogue += detail
        return detail

    def generate_explanation(self) -> str:
        e = self.query_codex(self.dialogue)
        self.dialogue += e
        self.logging_dialogue += e
        return e

    def replan(self, task_question: str) -> str:
        t = f"Human: Please fix above errors and replan the task '{task_question}'.\n"
        self.dialogue += t
        self.logging_dialogue += t
        plan = self.query_codex(self.dialogue)
        self.dialogue += plan
        self.logging_dialogue += plan
        return plan


DEFAULT_GOAL = {"name": "mine_log", "type": "mine", "object": {"log": 1}, "precondition": {}, "ranking": 1}


class DepsAgent:
    """main.Evaluator.reset / eval_step on the shared env API."""

    def __init__(self, env, client, task_item: str, artifact_dir: Path) -> None:
        self.env = env
        self.planner = DepsPlanner(client)
        self.task_obj = task_item
        self.task_question = f"How to obtain {task_item}?"  # data/task_info.json template
        self.dir = Path(artifact_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        mapping = json.loads((MCP / "goal_mapping.json").read_text())
        self.mineclip_text = dict(mapping["mineclip"])
        self.jarvis_skill = json.loads(JARVIS_SKILL.read_text()) if JARVIS_SKILL.exists() else {}
        self.replan_rounds = 0
        self.events: List[dict] = []

    # --- helpers --------------------------------------------------------------
    def inv(self) -> Dict[str, int]:
        return generic_inventory(self.env.inventory())

    @staticmethod
    def check_inventory(inv: Dict[str, int], items: Optional[dict]) -> bool:
        if not isinstance(items, dict):
            return False
        return all(inv.get(k, 0) >= int(v) for k, v in items.items())

    def steve_prompt(self, item: str) -> str:
        """Release's goal_mapping['mineclip'] text, else JARVIS-1 skill.json: the
        mine-type skill that names the item (else the first mine-type one), else
        "get {item}" (JARVIS-1 convention)."""
        if item in self.mineclip_text:
            return self.mineclip_text[item]
        skills = self.jarvis_skill.get(item) or self.jarvis_skill.get(item + "s") or []
        mines = [s for s in skills if isinstance(s, dict) and s.get("type") == "mine"]
        word = item.replace("_", " ")
        named = [s for s in mines if word in s.get("text", "")]
        if named or mines:
            return (named or mines)[0]["text"]
        return f"get {item}"

    def _set_goals(self, plan: str) -> None:
        self.goal_list = self.planner.generate_goal_list(plan)
        self.curr_goal = self.goal_list[0] if self.goal_list else dict(DEFAULT_GOAL)
        self.goal_eps = 0

    def save(self) -> None:
        (self.dir / "dialogue.txt").write_text(self.planner.logging_dialogue)
        (self.dir / "events.json").write_text(json.dumps(self.events, indent=1))

    def replan_task(self) -> None:
        self.planner.generate_failure_description(self.curr_goal["ranking"])
        self.planner.generate_inventory_description(self.inv())
        self.planner.generate_explanation()
        plan = self.planner.replan(self.task_question)
        self._set_goals(plan)
        self.replan_rounds += 1
        self.events.append({"t": self.env.mon.steps, "event": "replan", "round": self.replan_rounds,
                            "plan": self.goal_list})
        self.save()

    # --- main loop ------------------------------------------------------------
    def run(self) -> str:
        """main.Evaluator.eval_step: one iteration = one env step of the release."""
        self.env.reset()
        self.planner.reset()
        plan = self.planner.initial_planning(self.task_question)
        self._set_goals(plan)
        self.events.append({"t": 0, "event": "plan", "plan": self.goal_list})
        self.save()
        limit = {"craft": 150, "smelt": 200}
        while not self.env.over:
            inv = self.inv()
            # update_goal
            if self.check_inventory(inv, self.curr_goal["object"]) and self.goal_eps > 1:
                self.planner.generate_success_description(self.curr_goal["ranking"])
                if self.goal_list:
                    self.goal_list.remove(self.goal_list[0])
                if not self.goal_list:  # release: IndexError on goal_list[0]; DEPS stops here
                    self.save()
                    return "plan_exhausted"
                self.curr_goal = self.goal_list[0]
                self.goal_eps = 0
            gtype = self.curr_goal["type"]
            obj = self.curr_goal["object"] if isinstance(self.curr_goal["object"], dict) else {"log": 1}
            item = list(obj.keys())[0]
            if gtype in limit:
                if self.goal_eps == 0:  # one primitive attempt per activation of the goal
                    fn = self.env.craft if gtype == "craft" else self.env.smelt
                    ok, info = fn(item, int(obj.get(item, 1)))
                    self.events.append({"t": self.env.mon.steps, "event": gtype, "item": item,
                                        "n": obj.get(item, 1), "ok": ok, "info": info})
                    self.goal_eps += 2  # the release's craft agent takes >= 2 steps
                    if ok:
                        continue
                # failed craft: the env passes steps with no effect until the release's
                # replan trigger (goal_eps > 150 / 200)
                idle = limit[gtype] + 1 - self.goal_eps
                self.env.noop(idle)
                self.goal_eps += idle
                self.replan_task()
            elif gtype == "mine":
                self.env.steve(self.steve_prompt(item), 1)
                self.goal_eps += 1
                if not self.check_inventory(self.inv(), self.curr_goal["precondition"]):
                    self.replan_task()
            else:
                self.env.noop(1)
                self.goal_eps += 1
            if self.replan_rounds > 12:
                self.save()
                return "replan_rounds_exceeded"
            if self.task_obj in self.env.inventory() or self.task_obj in self.inv():
                self.planner.generate_success_description(self.curr_goal["ranking"])
                self.save()
                return "task_done"
        self.save()
        return "horizon"

