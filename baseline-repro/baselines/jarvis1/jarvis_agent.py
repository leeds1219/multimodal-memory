"""JARVIS-1 (Wang et al. 2023) ported onto the shared stack, paper-faithful
where the release is silent (DECISIONS: Stage B / JARVIS-1).

The release (github.com/CraftJarvis/JARVIS-1) contains only offline replay of
stored plans plus an LLM skill selector; the interactive planner exists only
in the paper. This port follows the paper (Sec. 3, App. A.2) and reuses every
released piece:

  released, verbatim   memory.json (frozen, 188 successful plans), skill.json,
                       ``core.get_skill`` prompt (T=1, max_tokens=256), 600-step
                       STEVE-1 mine attempts, inventory-count goal check
  paper, verbatim      Prompt 1 planning, Prompt 2 goal parsing, Prompt 3
                       self-explain, Prompt 4 self-check (baselines/jarvis1/prompts)
  ours (not published) - query generation is not used: memory is retrieved by the
                         task item (exact key, else the closest key), top-1, and
                         rendered as an extra few-shot turn in Prompt 1's format;
                       - <visual observation>: the MineCLIP descriptor and its
                         sentence bank were never released → the symbolic
                         location sentence of Prompt 1's own example
                         ("I current locate in <biome>.");
                       - the turn that feeds a self-check / self-explain result back
                         into the planning dialogue (text below, marked OURS);
                       - bounds: 2 self-check rounds per plan, 12 replans.
  memory               frozen during evaluation (paper's default setting); no
                       cross-task accumulation → one task order.
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
J1 = Path("/home/rag/data/official/JARVIS-1/jarvis/assets")
log = logging.getLogger("baseline_repro.jarvis1")

MINE_ATTEMPT_STEPS = 600      # released offline_evaluation: STEVE-1 timeout per attempt
MAX_REPLANS = 12
MAX_SELF_CHECK = 2
SKILL_SYSTEM = ("You are a helpful assistant in Minecraft. I will give you a task in Minecraft and a set of skills to "
                "finish such task. And you need to choose a suitable skill for the agent to finish the task object.  "
                "Only choose one skill once. Output reasoning thought and the number of the skill as action. You can "
                "follow the history dialogues to make a decision.")
SKILL_SHOTS = [  # released core.get_skill, verbatim
    ("user", "Task: Obtain iron_ore.\nSkills: 1. dig down, 2. equip stone pickaxe, 3. break stone blocks, obtain iron ore,\nAgent State: Now I have 1 stone pickaxe, 1 crafting_table, 4 stick, 6 planks in inventory. Now I equip the crafting_table in hand. Now I locate in height of 50."),
    ("assistant", "Thought: Mine iron ore should use the tool stone pickaxe. I have stone pickaxe in inventory. But I do not equip it now. So I should equip the stone pickaxe. \nAction: 2"),
    ("user", "Task: Obtain logs.\nSkills: 1. chop down the tree, 2. equip iron axe to chop down the tree, \nAgent State: Now I have 1 iron_axe in inventory. Now I equip the air in hand. Now I locate in height of 60."),
    ("assistant", "Thought: Equip the iron axe will accelerate the speed to chop trees. I have an iron axe in the inventory. So I should equip the iron axe first.\nAction: 2"),
    ("user", "Task: Obtain diamond.\nSkills: 1. dig down, 2. equip iron pickaxe, 3. break stone blocks, obtain diamond\nAgent State: Now I have 1 iron pickaxe, 1 crafting_table, 4 stick, 6 planks in inventory. Now I equip the iron_pickaxe in hand. Now I locate in height of 30."),
    ("assistant", "Thought: Diamond in Minecraft only exists in layers under height 15 and above height 5. Now my height is 30, which does not exist diamonds. So I should dig down to lower layers.\nAction: 1"),
]


def _split_dialogue(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Paper prompt text → (system, [(role, content), ...])."""
    lines = text.strip().split("\n")
    system = lines[0].split("System:", 1)[1].strip()
    turns: List[Tuple[str, str]] = []
    for line in lines[1:]:
        if line.strip() in ("==========", "###"):
            continue
        m = re.match(r"^(User|Assistant):\s?(.*)$", line)
        if m:
            turns.append(("user" if m[1] == "User" else "assistant", m[2]))
        elif turns:
            turns[-1] = (turns[-1][0], turns[-1][1] + ("\n" if turns[-1][1] else "") + line)
    return system, turns


def _generic(item: str) -> str:
    if item.endswith("_log") or item.endswith("_wood"):
        return "log"
    if item.endswith("_planks"):
        return "planks"
    return item


def inventory_text(inv: Dict[str, int]) -> str:
    items = [f"{v} {k}" for k, v in inv.items() if v > 0]
    return ", ".join(items) if items else "nothing"


class JarvisAgent:
    def __init__(self, env, client, task_item: str, artifact_dir: Path) -> None:
        self.env, self.client, self.item = env, client, task_item
        self.dir = Path(artifact_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        p = HERE / "prompts"
        self.plan_sys, plan_turns = _split_dialogue((p / "prompt1_planning.txt").read_text())
        self.plan_shots = plan_turns[:-2]      # drop the paper's own stone_sword query + empty Assistant
        self.parse_sys, self.parse_shots = _split_dialogue((p / "prompt2_goal_parsing.txt").read_text())
        self.explain_sys, self.explain_shots = _split_dialogue((p / "prompt3_self_explain.txt").read_text())
        self.check_sys, self.check_shots = _split_dialogue((p / "prompt4_self_check.txt").read_text())
        self.memory = json.loads((J1 / "memory.json").read_text())
        self.skills = json.loads((J1 / "skill.json").read_text())
        self.replan_rounds = 0
        self.events: List[dict] = []
        self.dialogue: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------ LLM
    def chat(self, system: str, turns: List[Tuple[str, str]], **kw) -> str:
        msgs = [{"role": "system", "content": system}] + [{"role": r, "content": c} for r, c in turns]
        r = self.client.chat.completions.create(model="gpt-4", messages=msgs, **kw)
        return r.choices[0].message.content or ""

    # ------------------------------------------------------------ memory (frozen)
    def retrieve(self) -> Optional[Tuple[str, dict]]:
        keys = list(self.memory)
        if self.item in self.memory:
            return self.item, self.memory[self.item]
        from thefuzz import process
        best = process.extractOne(self.item, keys)
        if best and best[1] >= 80:
            return best[0], self.memory[best[0]]
        return None

    @staticmethod
    def render_plan(item: str, plan: List[dict]) -> str:
        """Memory JSON plan → Prompt 1 code style (OURS: renderer)."""
        lines = [f"def craft_{item}(initial_inventory={{}}):"]
        for k, g in enumerate(plan, 1):
            (obj, n), = g["goal"].items()
            if g["type"] == "mine":
                lines.append(f' mine(obj = {{"{obj}":{n}}}, tool = None) # step {k}: {g.get("text", "mine " + obj)}')
            else:
                lines.append(f' {g["type"]}(obj = {{"{obj}":{n}}}, materials = {{}}, tool = None) # step {k}: {g["type"]} {n} {obj}')
        lines.append(f' return "{item}"')
        return "\n".join(lines)

    def location(self) -> str:
        return f"I current locate in {self.env.biome}"

    def query(self) -> str:
        return (f"My current inventory has {inventory_text(self.env.inventory())}. {self.location()}. "
                f"How to obtain 1 {self.item} in Minecraft step-by-step?")

    # ------------------------------------------------------------- planning
    def initial_dialogue(self) -> List[Tuple[str, str]]:
        turns = list(self.plan_shots)
        ref = self.retrieve()
        if ref is not None:
            key, entry = ref
            n = len(entry["plan"])
            turns += [("user", f"My current inventory has nothing. {self.location()}. How to obtain 1 {key} in Minecraft step-by-step?"),
                      ("assistant", f"The code for obtaining 1 {key} is as follows:\n" + self.render_plan(key, entry["plan"])),
                      ("user", f"[Description] I succeed in step {', '.join(str(i) for i in range(1, n + 1))}.\n"
                               f"I finish all steps and I obtain 1 {key} successfully.")]
            self.events.append({"event": "retrieve", "key": key})
        return turns

    def plan(self) -> str:
        self.dialogue.append(("user", self.query()))
        text = self.chat(self.plan_sys, self.dialogue)
        self.dialogue.append(("assistant", text))
        for _ in range(MAX_SELF_CHECK):
            check = self.chat(self.check_sys, self.check_shots + [(
                "user", f"My current inventory has {inventory_text(self.env.inventory())}. {self.location()}. "
                        f"My task is to obtain 1 {self.item} in Minecraft step-by-step. This is my plan:\n{text}")])
            m = re.search(r"Return:\s*(.*will fail.*)", check)
            self.events.append({"event": "self_check", "result": check[-300:]})
            if not m:
                break
            # OURS: feed the self-check verdict back into the planning dialogue
            self.dialogue.append(("user", f"[Check] {m[1].strip()} Please fix the plan. {self.query()}"))
            text = self.chat(self.plan_sys, self.dialogue)
            self.dialogue.append(("assistant", text))
        return text

    def parse(self, plan_text: str) -> List[dict]:
        goals = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if not re.match(r"^(mine|craft|smelt|equip)\s*\(", line):
                continue
            out = self.chat(self.parse_sys, self.parse_shots + [("user", f"input: {line}")])
            g = {"line": line}
            for l in out.split("\n"):
                if ":" in l:
                    k, v = l.split(":", 1)
                    g[k.strip()] = v.strip()
            try:
                g["object_number"] = int(g.get("object_number", 1))
            except ValueError:
                g["object_number"] = 1
            if g.get("action") in ("mine", "craft", "smelt", "equip") and g.get("object_item"):
                goals.append(g)
        return goals

    # ------------------------------------------------------------ execution
    def have(self, item: str, n: int) -> bool:
        inv = self.env.inventory()
        if item in ("log", "logs", "planks"):
            item = "log" if item.startswith("log") else "planks"
            return max([v for k, v in inv.items() if _generic(k) == item] or [0]) >= n  # release: max over tag members
        return inv.get(item, 0) >= n

    def get_skill(self, item: str) -> dict:
        key = "logs" if item in ("log", "logs") else item
        if key not in self.skills:
            return {"text": f"get {item}", "type": "mine", "object_item": None}
        if len(self.skills[key]) == 1:
            return self.skills[key][0]
        content = "".join(f"{i + 1}. {s['text']}, " for i, s in enumerate(self.skills[key]))
        inv = self.env.inventory()
        state = (f"Now my inventory has {inventory_text(inv)}." if inv else "Now my inventory has nothing.")
        q = f"Task: Obtain {key}.\nSkills: {content}\nAgent State: {state}"
        out = self.chat(SKILL_SYSTEM, SKILL_SHOTS + [("user", q)], temperature=1, max_tokens=256, top_p=1)
        m = re.search(r"^Action:\s*(\d+)", out, re.M)
        if not m or not (1 <= int(m[1]) <= len(self.skills[key])):
            return random.choice(self.skills[key])
        return self.skills[key][int(m[1]) - 1]

    def execute(self, g: dict) -> Tuple[bool, str]:
        item, n, act = g["object_item"], g["object_number"], g["action"]
        if self.have(item, n):
            return True, ""
        if act == "mine":
            skill = self.get_skill(item)
            if skill.get("type") == "equip" and skill.get("object_item"):
                ok, info = self.env.equip(skill["object_item"])
                return self.have(item, n), f"equip {skill['object_item']}: {info}"
            self.env.steve(skill["text"], MINE_ATTEMPT_STEPS, stop=lambda _inv: self.have(item, n))
            return self.have(item, n), f"could not {skill['text']} within {MINE_ATTEMPT_STEPS} steps"
        if act == "equip":
            ok, info = self.env.equip(item)
            return ok, str(info)
        fn = self.env.craft if act == "craft" else self.env.smelt
        ok, info = fn(item, n)
        return ok and self.have(item, n), str(info)

    def explain(self, line: str) -> str:
        return self.chat(self.explain_sys, self.explain_shots + [
            ("user", f"Failed Action: {line}\nCurrent Inventory: {inventory_text(self.env.inventory())}")])

    def save(self) -> None:
        (self.dir / "dialogue.json").write_text(json.dumps(self.dialogue, indent=1))
        (self.dir / "events.json").write_text(json.dumps(self.events, indent=1))

    def run(self) -> str:
        self.env.reset()
        self.dialogue = self.initial_dialogue()
        goals = self.parse(self.plan())
        self.events.append({"event": "plan", "goals": goals})
        while not self.env.over:
            if self.have(self.item, 1) or self.item in self.env.inventory():
                return "task_done"
            if not goals:
                return "plan_exhausted"
            done_steps = []
            failed = None
            for k, g in enumerate(goals, 1):
                ok, info = self.execute(g)
                self.events.append({"t": self.env.mon.steps, "event": "goal", "line": g["line"], "ok": ok, "info": info})
                if self.have(self.item, 1):
                    self.save()
                    return "task_done"
                if not ok:
                    failed = (k, g, info)
                    break
                done_steps.append(k)
            if failed is None:
                goals = []
                continue
            if self.replan_rounds >= MAX_REPLANS:
                self.save()
                return "replan_rounds_exceeded"
            k, g, info = failed
            expl = self.explain(g["line"])
            self.replan_rounds += 1
            succeeded = ", ".join(str(i) for i in done_steps) or "none"
            # OURS: failure description turn, in Prompt 1's "[Description]" format
            self.dialogue.append(("user", f"[Description] I succeed in step {succeeded}. I fail in step {k}. {expl.strip()}"))
            goals = self.parse(self.plan())
            self.events.append({"event": "replan", "round": self.replan_rounds, "goals": goals})
            self.save()
        return "horizon"
