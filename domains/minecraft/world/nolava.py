"""World nolava: lava near the agent becomes stone every NOLAVA_EVERY steps (experiments/conditions.py WORLDS).
The commands ride on the chat field of the agent's own action (agent/vlm_agent.py), one per step."""
from __future__ import annotations

from hyperparams import HP

NOLAVA_EVERY, NOLAVA_R = HP["world"]["nolava_every"], HP["world"]["nolava_radius"]  # (2R+1)^3 = 15625 blocks, under /fill's 32768
# The fill is run by an invisible marker armor stand at the agent: a /fill that finds no lava fails, and a failure
# run by the player prints a red "No blocks were filled" on its screen; run by the stand it prints nothing.
# /fill refuses the whole box when a corner is below y 0, so the box is centred at max(y, R): near the bottom it is
# y 0 to 2R, which still holds the agent.


def nolava_cmds(y: float) -> tuple[str, str, str]:
    r = NOLAVA_R
    return ('/execute at @p run summon minecraft:armor_stand ~ ~ ~ {Invisible:1b,Marker:1b,NoGravity:1b,Silent:1b,Tags:["nolava"]}',
            f"/execute as @e[tag=nolava] at @s positioned ~ {max(int(y), r)} ~ run fill ~-{r} ~-{r} ~-{r} ~{r} ~{r} ~{r} "
            "minecraft:stone replace minecraft:lava",
            "/kill @e[tag=nolava]")
