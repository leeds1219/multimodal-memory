"""a copy of MCP-Reborn's client jar with /save-all available to the MineRL player.

The MineRL player has level 2 (it can /setblock, /fill, /give, /tp) but not 4: /save-all answers "Unknown or
incomplete command", MineRL's own autosave is off (no world-zip prefix) and the game never pauses (the integrated
server is opened to LAN), so a running world is never written to disk. Vanilla registers /save-all only on a
dedicated server and the player has level 2, not the 4 it needs. Two changes: (1) Commands.<init> registers
SaveAllCommand where it registered PublishCommand (the integrated-server-only /publish, unused here): one
invokestatic constant-pool index; (2) SaveAllCommand's permission check iconst_4 -> iconst_2. The released jar is not touched; isoworld's private
MCP-Reborn dir uses the copy only when OPTIMUS_SAVEALL_JAR points at it (token ckpt<N>).

    python setup/build_saveall_jar.py [--out run_data/infra/mcprec-6.13-saveall.jar]
"""
from __future__ import annotations

import sys
import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import OPTIMUS1, ROOT  # noqa: E402

SRC = OPTIMUS1 / "minerl/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar"
CLASS = "net/minecraft/command/impl/SaveAllCommand.class"
OLD = bytes([0x2A, 0x07, 0xB6, 0x00, 0x1B])  # aload_0, iconst_4, invokevirtual #27 (CommandSource.hasPermissionLevel)
NEW = bytes([0x2A, 0x05, 0xB6, 0x00, 0x1B])  # aload_0, iconst_2, ...
CMDS = "net/minecraft/command/Commands.class"
PUBLISH = bytes([0xB8, 0x00, 0x4E])  # invokestatic #78 PublishCommand.register(CommandDispatcher)
SAVEALL = bytes([0xB8, 0x00, 0x47])  # invokestatic #71 SaveAllCommand.register(CommandDispatcher)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "run_data/infra/mcprec-6.13-saveall.jar"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(SRC) as zi, zipfile.ZipFile(tmp, "w") as zo:
        for item in zi.infolist():
            data = zi.read(item.filename)
            if item.filename == CLASS:
                assert data.count(OLD) == 1, "permission check not found exactly once"
                data = data.replace(OLD, NEW)
            if item.filename == CMDS:
                assert data.count(PUBLISH) == 1, "PublishCommand.register call not found exactly once"
                data = data.replace(PUBLISH, SAVEALL)
            zo.writestr(item, data, compress_type=item.compress_type)
    shutil.move(tmp, out)
    print(out, hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
