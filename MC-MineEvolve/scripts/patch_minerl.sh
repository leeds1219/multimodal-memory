#!/usr/bin/env bash
# Apply MineEvolve's MineRL patch and rebuild the Minecraft jar.
#
# MineRL 1.0.x's Java EnvServer has no "chat" action verb and creates the
# world with allowCommands=false, so every chat command this repo relies on
# (/gamerule at reset, /setblock ore spawning, ...) is rejected. The patch in
# patches/minerl-1.0.2-chat-commands.patch adds both. Run inside the
# `mineevolve` env after `pip install git+.../minerl@v1.0.2`:
#
#   bash scripts/patch_minerl.sh
#
# Idempotent: skips if the patch is already applied. Rebuild takes ~1-2 min
# (the Gradle decompile cache from the pip install is reused).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$HERE/patches/minerl-1.0.2-chat-commands.patch"

MCP="$(python -c 'import minerl, os; print(os.path.join(os.path.dirname(minerl.__file__), "MCP-Reborn"))')"
[ -d "$MCP" ] || { echo "MCP-Reborn not found at $MCP (is minerl installed in this env?)" >&2; exit 1; }

cd "$MCP"
if patch -p1 --dry-run -R -s -f < "$PATCH" >/dev/null 2>&1; then
  echo "minerl: chat-command patch already applied"
else
  patch -p1 < "$PATCH"
fi

if grep -q "MineEvolve patch" build/libs/.patched 2>/dev/null; then
  echo "minerl: jar already rebuilt with the patch"
else
  echo "minerl: rebuilding MCP-Reborn jar (Gradle) ..."
  ./gradlew shadowJar -x test --no-daemon -q
  echo "MineEvolve patch: $(sha256sum "$PATCH" | cut -c1-16)" > build/libs/.patched
fi
ls -la build/libs/*.jar
