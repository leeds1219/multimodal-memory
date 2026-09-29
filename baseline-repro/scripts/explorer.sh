#!/usr/bin/env bash
# Results explorer: rebuild its data from the runs and (optionally) serve it.
#
#   scripts/explorer.sh            rebuild data.js + film.js
#   scripts/explorer.sh serve      rebuild, then serve on http://localhost:8765
#   PORT=9000 scripts/explorer.sh serve
#
# Reads every result under $RUNS_ROOT (envs/common.py), so it also works while
# chains are still running: re-run it to refresh. No GPU, no API calls.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-conda run -n mcagent python}"
cd "$HERE"
$PY analysis/project.py >/dev/null 2>&1 || echo "note: analysis/project.py failed; group numbers may be stale"
$PY analysis/build_explorer.py
$PY analysis/build_filmstrips.py | tail -1
if [[ "${1:-}" == "serve" ]]; then
  echo "open http://localhost:${PORT:-8765}/"
  cd analysis/explorer && exec python3 -m http.server "${PORT:-8765}" --bind 127.0.0.1
fi
