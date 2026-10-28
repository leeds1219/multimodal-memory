"""World saves during an episode (token ckpt<N>), the starting points of the replay world (world/replay.py).

After the first VLM turn that ends at or past every N steps, "/save-all flush" rides on the chat field of the next
step's action (no extra tick). The integrated server writes the world and the player (position, facing, inventory,
health) to disk on the following tick; once level.dat is rewritten, the world folder is zipped to
<episode>/ckpt/s<step>.zip as ./saves/<name>/... (the layout MCP-Reborn's ReplaySender.loadWorldFromZip reads: it
takes the world name as entry.split("/")[2]). Next to it, s<step>.json holds what Minecraft does not store: the
episode step, the ore rule's used heights (ORE_MAP), the spawn point, the start position the VLM's dx/dz are counted
from, and the VLM's command history.

/save-all is not available to the player in the released client; ckpt<N> runs use the jar built by
setup/build_saveall_jar.py (world/private_world.py links it in).
"""
from __future__ import annotations

import glob
import json
import os
import time
import zipfile
from pathlib import Path

SAVE_CMD = "/save-all flush"
SAVE_WITHIN = 20  # agent steps to wait for the save; the server runs a chat command on the tick after it is sent
SKIP = ("session.lock",)
JAVA_TMP = "/tmp"  # java.io.tmpdir of the Minecraft JVM (MineRL sets none; Linux default)
PREFIX = "./saves/"


def entry(name: str, rel: str) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(PREFIX + name + "/" + rel, time.localtime()[:6])  # ZipInfo keeps the leading "./"
    zi.compress_type = zipfile.ZIP_DEFLATED
    return zi


def world_dir(mcdir: Path | None) -> Path:
    """The folder of the world this instance is running: the loaded copy if the episode started from a save
    (world/replay.py), else the one world of the episode's private MCP-Reborn/saves (world/private_world.py)."""
    name = os.environ.get("OPTIMUS_CKPT_WORLD")
    if name:
        hits = glob.glob(os.path.join(JAVA_TMP, "*", "saves", name))
        if len(hits) != 1:
            raise RuntimeError(f"loaded world {name}: {len(hits)} folders found")
        return Path(hits[0])
    worlds = sorted((Path(mcdir) / "saves").glob("*/level.dat")) if mcdir else []
    if len(worlds) != 1:
        raise RuntimeError(f"{len(worlds)} worlds in {mcdir}/saves (isoworld needed)")
    return worlds[0].parent


def level_mtime(mcdir: Path | None) -> float:
    lvl = world_dir(mcdir) / "level.dat"
    return lvl.stat().st_mtime if lvl.exists() else 0.0


def saved(mcdir: Path | None, before: float) -> bool:
    """True once level.dat was rewritten after `before` (the save ends with level.dat)."""
    return level_mtime(mcdir) > before


def write(mcdir: Path | None, out: Path, step: int, side: dict) -> Path:
    """Zip the saved world to out/s<step>.zip and the side state to out/s<step>.json."""
    wd = world_dir(mcdir)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    z = out / f"s{step:06d}.zip"
    tmp = z.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(wd.rglob("*")):
            if f.is_file() and f.name not in SKIP:
                zf.writestr(entry(wd.name, str(f.relative_to(wd))), f.read_bytes())
    tmp.rename(z)
    (out / f"s{step:06d}.json").write_text(json.dumps({**side, "step": step, "world": wd.name,
                                                       "save_s": round(time.time() - t0, 1)}))
    return z


def saves(ep: Path) -> list[int]:
    """Steps of the saves an episode has."""
    return sorted(int(p.stem[1:]) for p in (Path(ep) / "ckpt").glob("s*.zip"))


def ore_map_to_json(m: dict) -> list:
    return [[k, v] for k, v in m.items() if not isinstance(k, tuple)]
