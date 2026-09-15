#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install VLABench constructive composite tasks into the user's
# editable VLABench install (typically ~/VLABench/).
#
# 1. Copies data/vlabench_composite/constructive_tasks/*.py into
#    ~/VLABench/VLABench/tasks/hierarchical_tasks/composite/constructive/.
# 2. Patches VLABench/tasks/hierarchical_tasks/composite/__init__.py to
#    import the constructive subpackage.
# 3. Patches VLABench/configs/__init__.py's name2config so each new
#    composite is associated with a primitive series (so load_env can
#    find its task config).
#
# Idempotent: re-running is safe.
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_TASKS="${REPO_ROOT}/data/vlabench_composite/constructive_tasks"

VLABENCH_ROOT_DEFAULT="$HOME/VLABench/VLABench"
VLABENCH_ROOT="${VLABENCH_ROOT:-$VLABENCH_ROOT_DEFAULT}"

if [[ ! -d "$VLABENCH_ROOT" ]]; then
    echo "ERROR: VLABench root not found at $VLABENCH_ROOT"
    echo "       Set VLABENCH_ROOT to your editable VLABench checkout."
    exit 1
fi

DST_TASKS="$VLABENCH_ROOT/tasks/hierarchical_tasks/composite/constructive"
INIT_PY="$VLABENCH_ROOT/tasks/hierarchical_tasks/composite/__init__.py"
NAME2CONFIG_PY="$VLABENCH_ROOT/configs/__init__.py"

echo "[1/3] Copying constructive task files →"
echo "      $DST_TASKS"
mkdir -p "$DST_TASKS"
cp -v "$SRC_TASKS"/*.py "$DST_TASKS/"

echo "[2/3] Patching $INIT_PY"
if grep -q "constructive import" "$INIT_PY"; then
    echo "      already imports constructive subpackage — skipping"
else
    cat >> "$INIT_PY" <<'EOF'
from VLABench.tasks.hierarchical_tasks.composite.constructive import *  # noqa: F401, F403
EOF
    echo "      added import"
fi

echo "[3/3] Patching $NAME2CONFIG_PY (name2config)"
PYTHON="${VLABENCH_VENV_PYTHON:-$HOME/miniconda3/envs/vlabench/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python)"
$PYTHON - "$NAME2CONFIG_PY" <<'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()
# select_two_books_in_order belongs to the select_book_series config family
if "select_two_books_in_order" not in src:
    src = re.sub(
        r'(\"select_book_series\":\s*\[[^\]]*?)\]',
        r'\1, "select_two_books_in_order"]',
        src, count=1,
    )
    open(path, 'w').write(src)
    print(f"  patched {path}")
else:
    print(f"  {path} already lists select_two_books_in_order — skipping")
PY

echo
echo "=== install_vlabench_composite.sh DONE ==="
echo "Verify:"
echo "  MUJOCO_GL=egl VLABENCH_ROOT=$VLABENCH_ROOT $PYTHON -c \\"
echo "    \"import VLABench.tasks, VLABench.robots; from VLABench.envs import load_env; "
echo "     env=load_env('select_two_books_in_order', random_init=True, run_mode='eval'); "
echo "     env.reset(); print('OK,', env.task.get_instruction())\""
