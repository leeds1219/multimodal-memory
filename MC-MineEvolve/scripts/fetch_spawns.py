"""Print close-ended spawns (seed + player position) from JARVIS-1's spawn table.

JARVIS-1 (CraftJarvis/JARVIS-1, jarvis/assets/spawn.json) evaluates on fixed
(seed, biome, player_pos) triples; MineEvolve compares against it "under the
same task-seed split" but publishes none, so we borrow entries from there.
The file is downloaded on demand and not vendored (the repo has no license file).

    python scripts/fetch_spawns.py                 # biome counts
    python scripts/fetch_spawns.py oak_forest 10   # first 10 entries, YAML-ready
"""

from __future__ import annotations

import collections
import json
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/CraftJarvis/JARVIS-1/main/jarvis/assets/spawn.json"


def main() -> int:
    with urllib.request.urlopen(URL, timeout=30) as r:
        spawns = json.load(r)
    if len(sys.argv) < 2:
        for biome, n in collections.Counter(s["biome"] for s in spawns).most_common():
            print(f"{biome:14} {n}")
        return 0
    biome, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for s in [s for s in spawns if s["biome"] == biome][:n]:
        print(f"  - {{seed: {s['seed']}, pos: {list(s['player_pos'])}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
