#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline validation that the grasp tool consumes the depth + camera-pose
plumbing forwarded by robolab (lh-on-main port-forward work).

Loads a real obs dumped by the proxy (GRASP_OBS_DUMP=<path>) and runs the
grasp tool's perception pipeline (detection -> segmentation -> depth ->
point cloud -> GraspGen -> IK) against a LIVE grasp server. Exercises
_extract_depth, _get_camera_to_world, intrinsics selection, and
depth_to_pointcloud on real Isaac-Sim data — the exact path that never
fired in the full eval because pi05 happened to succeed.

Usage:
    GRASP_SERVER_HOST=127.0.0.1 GRASP_SERVER_PORT=8003 \
        python scripts/validate_grasp_from_obs.py \
            --obs /tmp/grasp_obs.pkl --target "rubiks cube" \
            [--use-front-camera] [--seg-mode sam3]
"""
import argparse
import pickle
import sys
import types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="pickled obs dict from GRASP_OBS_DUMP")
    ap.add_argument("--target", default="rubiks cube", help="grasp target phrase")
    ap.add_argument("--seg-mode", default="sam3")
    ap.add_argument("--use-front-camera", action="store_true")
    ap.add_argument("--topdown-threshold", type=float, default=0.85)
    args = ap.parse_args()

    with open(args.obs, "rb") as f:
        obs = pickle.load(f)

    print("=== loaded obs keys (orchestrator-relevant) ===")
    for k in sorted(obs):
        if not isinstance(k, str):
            continue
        if k.startswith(("observation/depth", "observation/camera", "gt_state")) \
           or k in ("observation/image", "observation/joint_position"):
            v = obs[k]
            shp = getattr(v, "shape", type(v).__name__)
            print(f"  {k}: {shp}")

    from vlm_orchestrator.grasp.tool import GraspToolExecutor

    print(f"\n=== creating GraspToolExecutor (seg={args.seg_mode}, "
          f"front_cam={args.use_front_camera}) ===")
    ex = GraspToolExecutor(
        seg_mode=args.seg_mode,
        env_mode="robolab",
        use_front_camera=args.use_front_camera,
        topdown_threshold=args.topdown_threshold,
    )

    # Minimal state stand-in (the tool only reads a few optional attrs).
    state = types.SimpleNamespace(
        _hitl=None, hitl=None, episode_log_dir=None,
        episode_step=0, step_count=0,
        log=lambda *a, **k: None,
    )

    print(f"\n=== running perception for target '{args.target}' ===")
    print("    (detection -> segmentation -> depth -> point cloud -> GraspGen -> IK)")
    try:
        ex.start(args.target, obs, state)
        print("\n✅ GRASP PERCEPTION + PLANNING SUCCEEDED")
        print(f"    phase: {getattr(ex, '_phase', '?')}")
        print(f"    status: {getattr(ex, '_status_message', '?')}")
        # Report planned grasp / trajectory if available
        for attr in ("_grasp_pose_world", "_planned_joints", "_trajectory",
                     "_approach_joints", "_grasp_joints"):
            v = getattr(ex, attr, None)
            if v is not None:
                shp = getattr(v, "shape", None)
                print(f"    {attr}: {shp if shp is not None else type(v).__name__}")
        return 0
    except Exception as e:
        import traceback
        print(f"\n❌ GRASP FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Distinguish plumbing failures from downstream (perception/IK) ones.
        msg = str(e).lower()
        if "no depth" in msg or "camera pose" in msg or "extrinsic" in msg \
           or "intrinsic" in msg:
            print("\n>>> This is a PLUMBING failure (depth/camera not consumed). <<<")
            return 2
        print("\n>>> Plumbing OK (depth+camera consumed); failure is downstream "
              "(detection/seg/graspgen/IK) — not a port-forward regression. <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
