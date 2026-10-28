"""Test of the replay world: start an episode from one save and compare the loaded state with the saved one.

    python scripts/test_world_save.py --ckpt <ep>/ckpt/s000600.zip --task o1_stone_00 --seed <seed> --out <dir> \
        --world nolava --save 60 --gpu 2 --port 28300
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402,F401

from benchmark import SEEDS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--ckpt", "--task", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--world", default="nolava")
    ap.add_argument("--save", type=int, default=60)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mock", action="store_true", help="LLM_MOCK: no API calls")
    a = ap.parse_args()
    from chain import Chain
    ch = Chain(argparse.Namespace(method="baseline", world=a.world, save=a.save, knowledge=None, order="test_world_save", gpu=a.gpu,
                                  port=a.port, retries=0, stall_min=30.0, mock=a.mock, seed_offset=a.seed - SEEDS[a.task]))
    ch.env_vars["OPTIMUS_LOAD_CKPT"] = str(Path(a.ckpt).resolve())
    ep = Path(a.out).resolve()  # the server script changes directory before it writes its log here
    ep.mkdir(parents=True, exist_ok=True)
    res = ch.run_episode(a.task, ep)
    side = json.loads(Path(a.ckpt).with_suffix(".json").read_text())
    first = None
    for line in gzip.open(ep / "trajectory.jsonl.gz", "rt"):
        d = json.loads(line)
        if "pos" in d and "inv" in d:
            first = d
            break
    print(json.dumps({"status": res.get("status"), "success": res.get("success"), "steps": res.get("steps"),
                      "saved": {"step": side["step"], "pos": side["pos"], "inv": side["inventory"]},
                      "loaded_first": first}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
