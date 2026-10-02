"""Variant "isoworld" (DECISIONS D36): one private Minecraft run dir per episode.

Released MineRL (Optimus-1's fork) starts every Minecraft instance with
cwd = minerl/MCP-Reborn, so all instances share MCP-Reborn/saves. World folders
are named "mcpworld" + hex from java.util.Random(0) (RandomHelper.seedsRand), i.e.
the n-th world of EVERY instance gets the same name. An episode can therefore
open a folder that another (earlier or concurrent) instance created with a
different seed: its terrain and spawn are not determined by its own seed.

Fix (infra only): before the env is made, point InstanceManager.MINECRAFT_DIR at
a per-episode directory that symlinks every entry of MCP-Reborn except the ones
Minecraft writes (saves/, logs/, options.txt, usercache.json), which are private.
After the episode the actual world seed is read back from level.dat.
"""

from __future__ import annotations

import gzip
import os
import shutil
import struct
from pathlib import Path

PRIVATE_COPY = ("options.txt", "usercache.json")
PRIVATE_EMPTY = ("saves", "logs")


def make_private_mcdir(root: Path) -> Path:
    from minerl.env.malmo import InstanceManager
    src = Path(InstanceManager.MINECRAFT_DIR).resolve()
    dst = root / "MCP-Reborn"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for e in src.iterdir():
        if e.name in PRIVATE_EMPTY:
            (dst / e.name).mkdir()
        elif e.name in PRIVATE_COPY:
            shutil.copy2(e, dst / e.name)
        else:
            os.symlink(e, dst / e.name)
    InstanceManager.MINECRAFT_DIR = str(dst)
    return dst


def _nbt(b: bytes, i: int, t: int):
    if t == 1: return b[i], i + 1
    if t == 2: return struct.unpack(">h", b[i:i + 2])[0], i + 2
    if t == 3: return struct.unpack(">i", b[i:i + 4])[0], i + 4
    if t == 4: return struct.unpack(">q", b[i:i + 8])[0], i + 8
    if t == 5: return None, i + 4
    if t == 6: return None, i + 8
    if t == 7: return None, i + 4 + struct.unpack(">i", b[i:i + 4])[0]
    if t == 8:
        n = struct.unpack(">H", b[i:i + 2])[0]
        return b[i + 2:i + 2 + n].decode("utf8", "replace"), i + 2 + n
    if t == 9:
        et, n = b[i], struct.unpack(">i", b[i + 1:i + 5])[0]
        i += 5
        out = []
        for _ in range(n):
            v, i = _nbt(b, i, et)
            out.append(v)
        return out, i
    if t == 10:
        d = {}
        while True:
            tt = b[i]; i += 1
            if tt == 0:
                return d, i
            n = struct.unpack(">H", b[i:i + 2])[0]
            name = b[i + 2:i + 2 + n].decode("utf8", "replace"); i += 2 + n
            d[name], i = _nbt(b, i, tt)
    if t == 11: return None, i + 4 + 4 * struct.unpack(">i", b[i:i + 4])[0]
    if t == 12: return None, i + 4 + 8 * struct.unpack(">i", b[i:i + 4])[0]
    raise ValueError(f"nbt tag {t}")


def world_seeds(mcdir: Path) -> dict[str, int]:
    """{world folder: seed stored in its level.dat}."""
    out = {}
    for lvl in sorted((mcdir / "saves").glob("*/level.dat")):
        try:
            b = gzip.open(lvl).read()
            d, _ = _nbt(b, 3 + struct.unpack(">H", b[1:3])[0], 10)
            out[lvl.parent.name] = int(d["Data"]["WorldGenSettings"]["seed"])
        except Exception:
            out[lvl.parent.name] = None
    return out


def cleanup(mcdir: Path) -> None:
    shutil.rmtree(mcdir, ignore_errors=True)
