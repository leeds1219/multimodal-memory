"""The replay world: an episode that starts from a save of an earlier episode (world/save.py).

OPTIMUS_LOAD_CKPT=<zip>: the env spec gets MineRL's LoadWorldAgentStart, so the env reset opens the saved world
instead of generating one (the empty initial inventory and absent placement leave the saved player as is). Each
load gets a copy of the zip whose world folder has a new unique name: MCP-Reborn unzips into
java.io.tmpdir/<hex>/saves/<name>, and the unique name is how a later save in the replayed episode finds that folder.
world/episode.py then restores what the zip does not hold (spawn point, ore rule state, step counters) and
system1/vlm_agent.py the VLM's command history and start position.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from save import JAVA_TMP, PREFIX, entry


def _world_name(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as z:
        return z.namelist()[0].split("/")[2]


def prepare_load(zip_path: Path, dst_dir: Path) -> tuple[Path, dict]:
    """Copy of the save with a fresh world folder name (exported as OPTIMUS_CKPT_WORLD) and its side state."""
    zip_path = Path(zip_path).resolve()
    side = json.loads(zip_path.with_suffix(".json").read_text())
    old, new = _world_name(zip_path), f"ck_{uuid.uuid4().hex[:12]}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = (dst_dir / "loaded_world.zip").resolve()
    with zipfile.ZipFile(zip_path) as zi, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zi.infolist():
            zo.writestr(entry(new, item.filename[len(PREFIX + old + "/"):]), zi.read(item.filename))
    os.environ["OPTIMUS_CKPT_WORLD"] = new
    return dst, side


def cleanup_loaded() -> None:
    """Remove the unzipped copy of the loaded world (one per load, left in /tmp by MCP-Reborn)."""
    name = os.environ.get("OPTIMUS_CKPT_WORLD")
    for h in glob.glob(os.path.join(JAVA_TMP, "*", "saves", name)) if name else []:
        shutil.rmtree(Path(h).parent.parent, ignore_errors=True)


def patch_env_spec(zip_file: str) -> None:
    """Make every env spec built from here on start from the saved world."""
    from minerl.herobraine.hero import handlers
    from optimus1.env.custom_env import CustomBaseEnvSpec
    orig = CustomBaseEnvSpec.create_agent_start

    def create_agent_start(self):
        return orig(self) + [handlers.LoadWorldAgentStart(zip_file)]

    CustomBaseEnvSpec.create_agent_start = create_agent_start


def timeout_wrapper(env):
    """The wrapper that counts the horizon (BasaltTimeoutWrapper: timeout, num_steps)."""
    e = env
    while e is not None:
        if "num_steps" in vars(e) and "timeout" in vars(e):
            return e
        e = getattr(e, "env", None)
    raise RuntimeError("no timeout wrapper found")


def ore_map_from_json(rows: list) -> dict:
    return {(float(k) if isinstance(k, float) else int(k)): v for k, v in rows}
