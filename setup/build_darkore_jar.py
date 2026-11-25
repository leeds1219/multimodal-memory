"""a copy of the /save-all client jar (setup/build_saveall_jar.py) with one hidden world rule: an ore block mined in the
dark drops nothing.

The rule is in the ore loot tables (data/minecraft/loot_tables/blocks/<ore>.json, vanilla 1.16.5 format): every pool
gets a location_check condition that passes when the block or one of its 6 neighbours has light level >= MIN_LIGHT.
The neighbours are checked because the ore block itself is opaque (light 0); the air next to it is what a torch lights.
Daylight counts as light, so ore open to the sky drops as usual. Deterministic: no random condition is added.

    python setup/build_darkore_jar.py [--src run_data/infra/mcprec-6.13-saveall.jar] [--out run_data/infra/mcprec-6.13-darkore.jar]
"""
from __future__ import annotations

import sys
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ROOT  # noqa: E402

ORES = ("coal_ore", "iron_ore", "gold_ore", "diamond_ore", "redstone_ore")
MIN_LIGHT = 10  # above the 9 a redstone ore gives off while it is being hit
OFFSETS = ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def lit_condition() -> dict:
    return {"condition": "minecraft:alternative",
            "terms": [{"condition": "minecraft:location_check", "offsetX": x, "offsetY": y, "offsetZ": z,
                       "predicate": {"light": {"light": {"min": MIN_LIGHT}}}} for x, y, z in OFFSETS]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "run_data/infra/mcprec-6.13-saveall.jar"))
    ap.add_argument("--out", default=str(ROOT / "run_data/infra/mcprec-6.13-darkore.jar"))
    a = ap.parse_args()
    out = Path(a.out)
    tmp = out.with_suffix(".part")
    tables = {f"data/minecraft/loot_tables/blocks/{o}.json" for o in ORES}
    done = set()
    with zipfile.ZipFile(a.src) as zi, zipfile.ZipFile(tmp, "w") as zo:
        for item in zi.infolist():
            data = zi.read(item.filename)
            if item.filename in tables:
                t = json.loads(data)
                for pool in t["pools"]:
                    pool.setdefault("conditions", []).append(lit_condition())
                data = json.dumps(t, indent=2).encode()
                done.add(item.filename)
            zo.writestr(item, data, compress_type=item.compress_type)
    assert done == tables, f"loot tables not found: {tables - done}"
    shutil.move(tmp, out)
    print(out, hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
