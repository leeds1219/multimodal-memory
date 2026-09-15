#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install the LIBERO constructive composite suite into the active
# LIBERO benchmark install. Copies the 3 BDDLs + 3 init pickles from
# vlm-orchestrator/data/libero_composite/ to the directories the
# active libero config (~/.libero/config.yaml) points at, then patches
# the LIBERO benchmark python package (whichever one is installed
# editably in your venv) to register the libero_composite suite and
# pick up its tasks.
#
# Idempotent: re-running is safe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_BDDL="${REPO_ROOT}/data/libero_composite/bddl_files"
SRC_INIT="${REPO_ROOT}/data/libero_composite/init_files"

# ── 1. Resolve where active LIBERO expects bddl_files / init_files ──

LIBERO_CONFIG="${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml"
if [[ ! -f "$LIBERO_CONFIG" ]]; then
    echo "ERROR: LIBERO config not found at $LIBERO_CONFIG"
    echo "       Set LIBERO_CONFIG_PATH or create ~/.libero/config.yaml first."
    exit 1
fi

DST_BDDL="$(grep '^bddl_files:' "$LIBERO_CONFIG" | awk '{print $2}')"
DST_INIT="$(grep '^init_states:' "$LIBERO_CONFIG" | awk '{print $2}')"

if [[ -z "$DST_BDDL" || -z "$DST_INIT" ]]; then
    echo "ERROR: Couldn't parse bddl_files / init_states from $LIBERO_CONFIG"
    exit 1
fi

echo "[1/3] Active LIBERO paths:"
echo "      bddl_files → $DST_BDDL"
echo "      init_states → $DST_INIT"

mkdir -p "$DST_BDDL/libero_composite" "$DST_INIT/libero_composite"
cp -v "$SRC_BDDL"/*.bddl "$DST_BDDL/libero_composite/"
cp -v "$SRC_INIT"/*.pruned_init "$DST_INIT/libero_composite/"

# ── 2. Locate the editably-installed LIBERO python package ──

# Try .libero-venv first, fall back to PATH python.
PYTHON="${LIBERO_VENV_PYTHON:-$REPO_ROOT/.libero-venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python)"

LIBERO_PKG="$($PYTHON -c "import libero, os; print(os.path.dirname(libero.__file__))")"
echo "[2/3] LIBERO python package at: $LIBERO_PKG"

# ── 3. Patch benchmark/__init__.py and benchmark/libero_suite_task_map.py ──

INIT_PY="$LIBERO_PKG/libero/benchmark/__init__.py"
TASK_MAP="$LIBERO_PKG/libero/benchmark/libero_suite_task_map.py"

if grep -q '"libero_composite"' "$TASK_MAP"; then
    echo "[3/3] libero_composite already in $TASK_MAP — skipping task-map patch"
else
    # Insert libero_composite at the top of libero_task_map dict
    $PYTHON - "$TASK_MAP" <<'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()
insert = '''libero_task_map = {
    "libero_composite": [
        "KITCHEN_SCENE3_first_turn_on_the_stove_then_put_the_frypan_on_the_stove_then_put_the_moka_pot_on_the_stove",
        "LIVING_ROOM_SCENE5_first_put_the_white_mug_on_the_left_plate_then_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE1_first_put_the_alphabet_soup_in_the_basket_then_put_the_cream_cheese_in_the_basket",
    ],
'''
new = re.sub(r'^libero_task_map = \{\n', insert, src, count=1, flags=re.M)
if new == src:
    raise SystemExit("FAIL: couldn't find libero_task_map = { entry to patch")
open(path, 'w').write(new)
print(f"  patched {path}")
PY
fi

if grep -q 'libero_composite' "$INIT_PY"; then
    echo "      libero_composite already registered in $INIT_PY — skipping registration patch"
else
    $PYTHON - "$INIT_PY" <<'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()

# 1. Add libero_composite to libero_suites list
src = re.sub(
    r'libero_suites = \[\n',
    'libero_suites = [\n    "libero_composite",\n',
    src, count=1,
)

# 2. Allow libero_composite to use natural file order (n_tasks != 10)
src = re.sub(
    r'if self\.name == "libero_90":\n',
    'if self.name in ("libero_90", "libero_composite"):\n',
    src, count=1,
)

# 3. Patch torch.load to set weights_only=False (newer torch defaults
#    to True which rejects numpy reconstructors in init pickles).
src = re.sub(
    r'init_states = torch\.load\(init_states_path\)',
    'init_states = torch.load(init_states_path, weights_only=False)',
    src, count=1,
)

# 4. Add LIBERO_COMPOSITE benchmark class after LIBERO_MEM
add = '''

@register_benchmark
class LIBERO_COMPOSITE(Benchmark):
    """Constructive composite tasks: multi-step BDDLs chaining
    in-distribution LIBERO primitives via (:goal (Sequence ...))."""
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_composite"
        self._make_benchmark()
'''
src = re.sub(
    r'(@register_benchmark\nclass LIBERO_MEM\(Benchmark\):\n.*?self\._make_benchmark\(\)\n)',
    r'\1' + add,
    src, count=1, flags=re.S,
)

open(path, 'w').write(src)
print(f"  patched {path}")
PY
fi

echo
echo "=== install_libero_composite.sh DONE ==="
echo "Verify:"
echo "  $PYTHON -c \"from libero.libero import benchmark; s=benchmark.get_benchmark_dict()['libero_composite'](); print(s.n_tasks, 'tasks')\""
