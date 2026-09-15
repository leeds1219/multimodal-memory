#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full end-to-end test: robolab → proxy → policy server.

Runs BananaOnPlateTask (40s episodes) × 3 episodes through the real robolab
eval pipeline via the vlm-orchestrator proxy to verify the episode logging fix.

Results are left in place under ~/vlm-orchestrator/results/ for inspection.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

TASK = "BananaOnPlateTask"
NUM_EPISODES = 3
VLA_PORT = 8000
PROXY_PORT = 8001
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_NAME = f"e2e_episode_fix_{TIMESTAMP}"

LOG_DIR = os.path.expanduser(
    f"~/vlm-orchestrator/results/e2e_episode_fix_{TIMESTAMP}"
)
ROBOLAB_OUTPUT_DIR = os.path.expanduser(f"~/robolab/output/{OUTPUT_NAME}")


def port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def kill_port(port: int):
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    time.sleep(1)


def main():
    print("=" * 70)
    print("  E2E Robolab Episode Logging Test")
    print(f"  Task: {TASK} (40s episodes)")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Proxy log dir: {LOG_DIR}")
    print(f"  Robolab output: {ROBOLAB_OUTPUT_DIR}")
    print("=" * 70)
    print()

    # Pre-flight
    if not port_in_use(VLA_PORT):
        print(f"ABORT: Policy server not running on port {VLA_PORT}")
        sys.exit(1)
    print(f"✓ Policy server on port {VLA_PORT}")

    if port_in_use(PROXY_PORT):
        print(f"  Killing stale process on port {PROXY_PORT}...")
        kill_port(PROXY_PORT)
    print(f"✓ Port {PROXY_PORT} available")

    os.makedirs(LOG_DIR, exist_ok=True)

    # ===== Step 1: Start the proxy =====
    print("\n--- Step 1: Starting proxy ---")
    proxy_script = f"""\
import sys, os, logging
sys.path.insert(0, os.path.expanduser("~/vlm-orchestrator"))
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr)

from vlm_orchestrator.proxy import ProxyConfig, OrchestratorProxy

config = ProxyConfig(
    vla_host="127.0.0.1",
    vla_port={VLA_PORT},
    host="0.0.0.0",
    port={PROXY_PORT},
    log_dir="{LOG_DIR}",
    robolab_output_dir="{ROBOLAB_OUTPUT_DIR}",
)
proxy = OrchestratorProxy(config)
proxy.serve_forever()
"""
    proxy_log_path = os.path.join(LOG_DIR, "proxy.log")
    proxy_log_fh = open(proxy_log_path, "w")
    proxy_proc = subprocess.Popen(
        [sys.executable, "-c", proxy_script],
        stdout=proxy_log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Wait for proxy to be ready
    for i in range(20):
        time.sleep(1)
        if port_in_use(PROXY_PORT):
            break
    else:
        print("ABORT: Proxy failed to start")
        proxy_log_fh.close()
        with open(proxy_log_path) as f:
            print(f.read()[-1000:])
        sys.exit(1)
    print(f"  Proxy running (PID {proxy_proc.pid})")

    # ===== Step 2: Run robolab eval =====
    print(f"\n--- Step 2: Running robolab eval ({NUM_EPISODES} episodes) ---")
    eval_log_path = os.path.join(LOG_DIR, "eval.log")
    eval_cmd = [
        "conda", "run", "--no-capture-output", "-n", "robolab",
        "python", "policies/volo/run.py",
        "--headless",
        "--policy", "pi05",
        "--instruction-type", "default",
        "--num-runs", str(NUM_EPISODES),
        "--task", TASK,
        "--remote-port", str(PROXY_PORT),
        "--enable-subtask",
        "--output-folder-name", OUTPUT_NAME,
    ]
    eval_log_fh = open(eval_log_path, "w")
    try:
        result = subprocess.run(
            eval_cmd,
            cwd=os.path.expanduser("~/RoboLab"),
            stdout=eval_log_fh,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        eval_rc = result.returncode
    except subprocess.TimeoutExpired:
        print("  WARNING: Eval timed out")
        eval_rc = 124
    finally:
        eval_log_fh.close()

    if eval_rc == 0:
        print("  ✓ Eval completed successfully")
    else:
        print(f"  ✗ Eval exit code: {eval_rc}")

    # Give proxy a moment to finalize the last episode
    time.sleep(3)

    # ===== Step 3: Stop proxy =====
    print("\n--- Step 3: Stopping proxy ---")
    try:
        os.killpg(os.getpgid(proxy_proc.pid), signal.SIGTERM)
        proxy_proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proxy_proc.pid), signal.SIGKILL)
        except Exception:
            pass
    proxy_log_fh.close()
    kill_port(PROXY_PORT)
    print("  Proxy stopped")

    # ===== Step 4: Verify results =====
    print("\n" + "=" * 70)
    print("  VERIFICATION")
    print("=" * 70)

    all_ok = True

    # 4a: Check robolab output exists
    print(f"\n[1] Robolab output ({ROBOLAB_OUTPUT_DIR}):")
    if os.path.isdir(ROBOLAB_OUTPUT_DIR):
        robolab_task_dir = os.path.join(ROBOLAB_OUTPUT_DIR, TASK)
        if os.path.isdir(robolab_task_dir):
            robolab_files = sorted(os.listdir(robolab_task_dir))
            videos = [f for f in robolab_files if f.endswith(".mp4")]
            logs = [f for f in robolab_files if f.endswith(".json")]
            hdf5 = [f for f in robolab_files if f.endswith(".hdf5")]
            print(f"    Videos: {len(videos)}  Logs: {len(logs)}  HDF5: {len(hdf5)}")
            for v in videos:
                sz = os.path.getsize(os.path.join(robolab_task_dir, v))
                print(f"      {v}  ({sz:,} bytes)")
        else:
            print("    FAIL: Task dir not found")
            all_ok = False
    else:
        print("    FAIL: Output dir not found")
        all_ok = False

    # episode_results.json
    results_path = os.path.join(ROBOLAB_OUTPUT_DIR, "episode_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            episode_results = json.load(f)
        print(f"\n[2] Robolab episode results ({len(episode_results)} episodes):")
        for r in episode_results:
            print(f"    ep={r.get('episode')}, success={r.get('success')}, "
                  f"score={r.get('score', '?')}")

    # 4b: Check per-episode directories
    print(f"\n[3] Per-episode directories ({LOG_DIR}):")
    task_slugs = [d for d in os.listdir(LOG_DIR)
                  if os.path.isdir(os.path.join(LOG_DIR, d))]
    if not task_slugs:
        print("    FAIL: No task directories found")
        all_ok = False
    else:
        for task_slug in sorted(task_slugs):
            task_slug_dir = os.path.join(LOG_DIR, task_slug)
            episode_dirs = sorted([
                d for d in os.listdir(task_slug_dir)
                if d.startswith("episode_")
            ])
            print(f"    {task_slug}/: {episode_dirs}")

            for ep_id in range(1, NUM_EPISODES + 1):
                ep_name = f"episode_{ep_id}"
                ep_dir = os.path.join(task_slug_dir, ep_name)

                if not os.path.isdir(ep_dir):
                    print(f"      ✗ {ep_name}/ MISSING")
                    all_ok = False
                    continue

                meta_path = os.path.join(ep_dir, "metadata.json")
                if not os.path.exists(meta_path):
                    print(f"      ✗ {ep_name}/metadata.json MISSING")
                    all_ok = False
                    continue

                with open(meta_path) as f:
                    meta = json.load(f)

                checks = []
                if meta.get("episode_id") != ep_id:
                    checks.append(f"episode_id={meta.get('episode_id')} (expected {ep_id})")
                    all_ok = False
                if "end_timestamp" not in meta:
                    checks.append("not finalized")
                    all_ok = False
                if meta.get("infer_count", 0) == 0:
                    checks.append("infer_count=0")
                    all_ok = False

                if checks:
                    print(f"      ✗ {ep_name}: {', '.join(checks)}")
                else:
                    print(f"      ✓ {ep_name}: episode_id={meta['episode_id']}, "
                          f"infer_count={meta['infer_count']}")

    # 4c: Check symlinks
    print("\n[4] Symlinks:")
    symlink_count = 0
    broken_count = 0
    ep_targets = {}  # ep_name -> list of video basenames
    for task_slug in task_slugs:
        task_slug_dir = os.path.join(LOG_DIR, task_slug)
        for ep_name in sorted(os.listdir(task_slug_dir)):
            if not ep_name.startswith("episode_"):
                continue
            ep_dir = os.path.join(task_slug_dir, ep_name)
            for f in sorted(os.listdir(ep_dir)):
                fpath = os.path.join(ep_dir, f)
                if os.path.islink(fpath):
                    target = os.readlink(fpath)
                    exists = os.path.exists(fpath)
                    symlink_count += 1
                    if not exists:
                        broken_count += 1
                        all_ok = False
                    status = "✓" if exists else "✗ BROKEN"
                    print(f"    {status} {ep_name}/{f}")
                    print(f"        → {target}")
                    if f.endswith(".mp4"):
                        ep_targets.setdefault(ep_name, []).append(
                            os.path.basename(target))

    if symlink_count == 0:
        print("    WARNING: No symlinks found")
    else:
        print(f"    Total: {symlink_count} symlinks, {broken_count} broken")

    # 4d: Episode-specificity
    print("\n[5] Episode-specificity (each episode points to different files):")
    if len(ep_targets) >= 2:
        all_targets = list(ep_targets.values())
        all_unique = True
        for i in range(len(all_targets)):
            for j in range(i + 1, len(all_targets)):
                if set(all_targets[i]) == set(all_targets[j]):
                    names = list(ep_targets.keys())
                    print(f"    ✗ {names[i]} and {names[j]} point to SAME files!")
                    all_unique = False
                    all_ok = False
        if all_unique:
            for ep, targets in sorted(ep_targets.items()):
                print(f"    ✓ {ep} → robolab files: {targets}")
    else:
        print(f"    Only {len(ep_targets)} episode(s) have video symlinks")

    # ===== Verdict =====
    print()
    print("=" * 70)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print("  ❌ SOME CHECKS FAILED")
    print()
    print("  Results kept at:")
    print(f"    Proxy logs: {LOG_DIR}")
    print(f"    Robolab out: {ROBOLAB_OUTPUT_DIR}")
    print("=" * 70)

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
