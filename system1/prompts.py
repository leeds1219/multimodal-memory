"""Every text the Baseline agent's VLM gets, in one place.

The system prompt is SYSTEM with the command list (CMD_KEYS, CMD_STEVE, CMD_HELPERS), plus the additions of the
experiment: GRAPH_NOTE (GT Recipe), ORE_HINT (GT World Rule), EXP_NOTE (an experiment of Ours), World Knowledge
(memory/world_knowledge.py) and MineEvolve's own text (memory/mineevolve_memory.py).
"""

SYSTEM = """You control a player in Minecraft. The game is paused while you decide; after each command
it runs (20 game steps per second) and pauses again.
Overall task: {task}
Current sub-goal: {subgoal}

Send one command per turn. Commands:
{commands}

Answer with JSON only: {{"thought": "<what you see and what you expect this command to do>", "command": {{...}}}}"""
NO_SUBGOAL = "none given; decide the steps yourself"

CMD_KEYS = """  {"type": "keys", "keys": [<any of: forward, back, left, right, jump, sneak, sprint, attack, use,
                                 hotbar.1 ... hotbar.9>], "camera": [<pitch deg>, <yaw deg>], "steps": <1-100>}
      hold these keys for that many steps; camera turns once at the first step (pitch > 0 = look down,
      yaw > 0 = turn right); "use" places the held block or uses the held item; hotbar.N selects slot N"""
CMD_STEVE = """  {"type": "steve", "prompt": "<short English instruction>", "steps": <1-200>}
      STEVE-1, a learned policy, turns the instruction into actions (e.g. "chop a tree", "dig down")"""
CMD_HELPERS = """  {"type": "equip", "item": "<item id>"}   puts the item in hand
  {"type": "craft", "item": "<item id>", "n": <count>}         crafts it if the materials are there
  {"type": "smelt", "item": "<item id>", "n": <count>}         smelts it in a furnace if the materials are there"""

N_HISTORY = 10  # earlier commands shown each turn
TURN = "Your previous commands and how they ended (most recent last):\n{history}\n\nTurn {n}. Next command?"

GRAPH_NOTE = ("\n<craft graph>: a top-down list of all the tools and materials needed to complete the task, "
              "counted from an empty inventory.\n{graph}")
ORE_HINT = ("\n\nHow ores appear in this world: at each step there is a 10% chance that one ore block is placed "
            "3 to 5 blocks directly below you (same x and z). Which ore depends on your height y: y 45-50 coal, "
            "y 26-43 iron, y 15-26 gold or redstone, y 14 and below diamond. Each height level is used at most once "
            "per episode (both your level and the level the ore is placed at), so standing on a level you already "
            "used, or walking sideways on it, gives no new ore; descending to new levels does. Ore generated with the world also exists, but you have to find it.")
EXP_NOTE = ("\n\nThis attempt starts from a saved moment of an earlier attempt at this task: the world, your position, "
            "inventory and your last commands are as they were then. In the earlier attempt the task failed. "
            "From here, try this change:\n{change}\nExpected result of the change: {prediction}\nHow to try it: {plan}\n"
            "Keep working on the overall task after trying it.")
