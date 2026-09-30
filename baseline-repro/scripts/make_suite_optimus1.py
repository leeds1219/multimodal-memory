"""Build the Optimus-1 native task suite from the released benchmark yamls.

    python scripts/make_suite_optimus1.py   -> configs/suites/optimus1/{tasks,task_targets,seeds,task_orders}.yaml

Tasks: the `type: craft` entries of NeurIPS24-Optimus-1/src/optimus1/conf/benchmark/*.yaml
(67 tasks, the set the paper reports); the 6 `type: mine` entries are left out.
Targets: the item named in the instruction ("Craft a X" -> x, "Smelt a charcoal" ->
charcoal, "Smelt and craft a gold ingot" -> gold_ingot, "Dig down and mine a diamond"
-> diamond). Seeds: one base seed per task (random.Random(20260930)); extra worlds
come from the chain's --seed-offset. One task order: the paper's group order.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import yaml

REPRO = Path(__file__).resolve().parents[1]
SRC = Path("/home/rag/data/official/NeurIPS24-Optimus-1/src/optimus1/conf/benchmark")
OUT = REPRO / "configs" / "suites" / "optimus1"
GROUPS = [("wooden", "wooden"), ("stone", "stone"), ("iron", "iron"), ("gold", "golden"),
          ("redstone", "redstone"), ("diamond", "diamond"), ("armor", "armor")]
PARTS = {"a": ("wooden", "stone", "iron"), "b": ("gold", "redstone"), "c": ("diamond", "armor")}
SPECIAL = {"smelt a charcoal": "charcoal", "smelt and craft a gold ingot": "gold_ingot",
           "dig down and mine a diamond": "diamond"}


def target(instr: str) -> str:
    s = " ".join(instr.lower().split())
    if s in SPECIAL:
        return SPECIAL[s]
    m = re.match(r"craft (?:an? )?(.+)$", s)
    if not m:
        raise ValueError(f"cannot parse target from {instr!r}")
    return m[1].replace(" ", "_")


def main() -> int:
    tasks, targets, seeds, order, tg = [], {}, {}, [], {}
    rng = random.Random(20260930)
    for group, fname in GROUPS:
        y = yaml.safe_load(open(SRC / f"{fname}.yaml"))
        for t in y["all_task"]:
            if t["type"] != "craft":
                continue
            uid = f"o1_{group}_{int(t['id']):02d}"
            instr = " ".join(str(t["instruction"]).split())
            tasks.append({"uid": uid, "group": group, "local_id": int(t["id"]), "type": t["type"], "instruction": instr})
            targets[uid] = {"any_of": [target(instr)], "count": 1}
            seeds[uid] = rng.randrange(1, 2**31 - 1)
            order.append(uid); tg[uid] = group
    OUT.mkdir(parents=True, exist_ok=True)
    head = {"source": str(SRC), "note": "type: craft entries only (the 67 tasks the Optimus-1 paper reports)",
            # Paper Table 5 "Max. Steps" (3600/7200/12000/36000); the released benchmark
            # yamls use 2/5/20/30 min instead. 1 min = 1200 steps. See DECISIONS D34.
            "horizon_minutes": {"wooden": 3, "stone": 6, "iron": 10, "gold": 30, "redstone": 30,
                                "diamond": 30, "armor": 30}}
    yaml.safe_dump({**head, "tasks": tasks}, open(OUT / "tasks.yaml", "w"), sort_keys=False)
    yaml.safe_dump({"rule": "target item named in the instruction, in inventory at any step before the horizon",
                    "targets": targets}, open(OUT / "task_targets.yaml", "w"), sort_keys=False)
    yaml.safe_dump({"note": "base seed per task; seed k of a task = base + chain --seed-offset",
                    "seeds": seeds}, open(OUT / "seeds.yaml", "w"), sort_keys=False)
    yaml.safe_dump({"note": "the paper's group order; seed0/1/2 = the same order in three worlds (--seed-offset 0/1/2)",
                    "orders": {"paper": order, **{f"seed{k}": order for k in range(3)},
                               # the same three worlds split into shorter chains that can run in parallel
                               **{f"seed{k}_{part}": [u for u in order if tg[u] in groups]
                                  for k in range(3) for part, groups in PARTS.items()}}},
                   open(OUT / "task_orders.yaml", "w"), sort_keys=False)
    from collections import Counter
    print(len(tasks), "tasks", dict(Counter(t["group"] for t in tasks)), "->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
