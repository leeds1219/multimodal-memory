# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for the VLM Orchestrator."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="VLM Orchestrator: proxy between eval client and VLA policy server"
    )
    parser.add_argument(
        "--vla-host",
        default="127.0.0.1",
        help="VLA policy server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--vla-port",
        type=int,
        default=8000,
        help="VLA policy server port (default: 8000)",
    )
    parser.add_argument(
        "--frontend",
        choices=["openpi-ws", "gr00t-zmq", "openvla-rest"],
        default="openpi-ws",
        help="Eval-client-facing protocol.  'openpi-ws' (default) for "
             "pi0/pi05/paligemma eval clients (--policy pi05).  "
             "'gr00t-zmq' for GR00T eval clients (--policy gr00t).",
    )
    parser.add_argument(
        "--backend",
        choices=["openpi-ws", "gr00t-zmq", "openvla-rest", "file-ipc"],
        default="openpi-ws",
        help="VLA-server-facing protocol.  'openpi-ws' (default) for "
             "openpi policy server.  'gr00t-zmq' for NVIDIA Isaac-GR00T "
             "policy server.  'file-ipc' for serving over a shared "
             "filesystem when direct TCP is blocked — combine "
             "with --vla-file-ipc-dir.",
    )
    parser.add_argument(
        "--vla-file-ipc-dir",
        default=None,
        help="Shared directory for --backend file-ipc. Both the "
             "orchestrator and the policy server must see this path "
             "(via a shared filesystem). Created if missing.",
    )
    parser.add_argument(
        "--gr00t-api-token",
        default=None,
        help="Optional API token for the GR00T server (when --backend gr00t-zmq).",
    )
    parser.add_argument(
        "--openvla-unnorm-key",
        default="droid",
        help="OpenVLA action-denorm key (when --backend openvla-rest).  "
             "Identifies which dataset's normalization stats to use; must "
             "match a key the deployed OpenVLA model was trained with.  "
             "Default 'droid'.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for this orchestrator to listen on (default: 8001)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for this orchestrator to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--env",
        choices=["robolab", "libero"],
        default="robolab",
        help="Simulation environment preset. Sets default image keys and "
             "timing parameters. 'robolab' (default): IsaacLab/PhysX, "
             "exterior_image_1_left + wrist_image_left, 8-step chunks. "
             "'libero': MuJoCo/robosuite, observation/image + "
             "observation/wrist_image, 5-step replan chunks.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "rewrite", "passthrough", "adaptive",
            "subgoal", "scene_edit", "subgoal_scene_edit",
            "next_goal", "tool_chain",
        ],
        default="subgoal",
        help="Operation mode: "
             "subgoal (decompose + periodic check, recommended), "
             "tool_chain (no VLA at all; VLM picks grasp/place tools and "
             "checks status only after each tool finishes), "
             "next_goal (VLM predicts next step on-the-fly, no upfront decomposition), "
             "passthrough (never). "
             "Archived exploration modes: rewrite (single-shot rewrite), "
             "adaptive (compare-and-pick), scene_edit / subgoal_scene_edit "
             "(GDino-based visual highlighting) "
             "(default: subgoal)",
    )
    parser.add_argument(
        "--vlm-model",
        default="YOUR_VLM_MODEL",
        help="VLM model to use (default: YOUR_VLM_MODEL)",
    )
    parser.add_argument(
        "--vlm-temperature",
        type=float,
        default=0.0,
        help="VLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--vlm-base-url",
        default="https://YOUR_VLM_ENDPOINT/v1",
        help="Base URL for VLM API (default: https://YOUR_VLM_ENDPOINT/v1)",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=None,
        help="API key for VLM (default: VLM_API_KEY env var, "
             "falls back to OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--vlm-strategy",
        choices=[
            "direct", "cot", "verify", "cot_strategy",
            "minimalist", "planner", "hint",
        ],
        default="direct",
        help="VLM prompting strategy for rewrite/adaptive modes "
             "(default: direct)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt for the VLM (default: built-in)",
    )
    parser.add_argument(
        "--rewrite-strategy",
        choices=["first_per_episode", "first_per_connection"],
        default="first_per_episode",
        help="When to invoke the VLM in rewrite mode (default: first_per_episode)",
    )
    parser.add_argument(
        "--probe-n",
        type=int,
        default=30,
        help="Adaptive mode: number of policy probes per instruction "
             "(default: 30)",
    )

    # --- Subgoal mode options ---
    parser.add_argument(
        "--subgoal-check-mode",
        choices=["vlm", "timer", "hybrid"],
        default="vlm",
        help="Subgoal mode: how to decide subgoal transitions. "
             "'vlm' (default) always uses VLM progress checks. "
             "'timer' advances purely by timeout with no VLM checks. "
             "'hybrid' uses VLM-gated advancement for ordered "
             "tasks and timer cycling for unordered tasks.",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=80,
        help="Subgoal mode: sim steps between VLM progress checks "
             "(used in 'vlm' mode and for ordered tasks in 'hybrid'). "
             "Step-based since multi-VLA support; pi05 chunks=8 so "
             "80 steps ≈ 10 chunks of pi05 (default: 80)",
    )
    parser.add_argument(
        "--subgoal-timeout",
        type=int,
        default=9999,
        help="Subgoal mode: max sim steps per subgoal before "
             "auto-advancing.  Default 9999 effectively disables "
             "time-based advancement — subgoals advance on VLM / GT "
             "completion signals instead.  Set a smaller value "
             "explicitly to enable a timeout.",
    )
    parser.add_argument(
        "--subgoal-initial-mode",
        choices=["direct", "rewrite", "adaptive"],
        default="direct",
        help="Subgoal mode: how to process the instruction before "
             "decomposition. 'direct' decomposes as-is, 'rewrite' rewrites "
             "first, 'adaptive' does compare-and-pick first (default: direct)",
    )

    # --- Next-goal mode options ---
    parser.add_argument(
        "--next-goal-template",
        choices=["v1", "v2", "v1_freeform", "mcq"],
        default="v1_freeform",
        help="Next-goal mode: prompt template variant. "
             "'v1_freeform' (default) free-text output with CoT. "
             "'v1' simpler free-text. "
             "'v2'/'mcq' MCQ selection; requires --next-goal-task-type.",
    )
    parser.add_argument(
        "--next-goal-task-type",
        default=None,
        help="Next-goal mode: task type key for closed-vocab candidate sets "
             "and task-specific context (from vlm_orchestrator/prompts.py). "
             "Required for --next-goal-template v2/mcq. "
             "Available: i3_closed_vocab, i4_closed_vocab, h8_closed_vocab. "
             "(default: None = freeform, no candidates)",
    )
    parser.add_argument(
        "--next-goal-max-history",
        type=int,
        default=0,
        help="Next-goal mode: maximum checkpoint images kept in rolling "
             "history (0 = unlimited, default). Oldest image dropped when "
             "exceeded.",
    )
    parser.add_argument(
        "--vla-capabilities",
        default=None,
        help="Next-goal mode: description of VLA capabilities to include "
             "in the system prompt. The orchestrator will tailor subgoal "
             "instructions to match what the VLA can execute. "
             "Built-in presets: 'molmobot' (pick/pick-and-place only). "
             "Any other string is used verbatim. "
             "(default: None = no capability constraints)",
    )

    # --- Scene edit mode options ---
    parser.add_argument(
        "--scene-edit-mode",
        choices=["none", "highlight", "dim", "both"],
        default="highlight",
        help="Scene edit mode: image edit type "
             "(default: highlight)",
    )
    parser.add_argument(
        "--scene-edit-requery-interval",
        type=int,
        default=5,
        help="Scene edit: re-query for updated bbox every N chunks "
             "(default: 5)",
    )
    parser.add_argument(
        "--scene-edit-instruction",
        default=None,
        help="Scene edit: override the client instruction with this fixed "
             "string (default: use client instruction)",
    )
    parser.add_argument(
        "--scene-edit-detector",
        choices=["gdino", "vlm"],
        default="gdino",
        help="Scene edit: detection backend. 'gdino' uses GroundingDINO "
             "(fast, local GPU), 'vlm' uses VLM API (default: gdino)",
    )
    parser.add_argument(
        "--scene-edit-gdino-prompt",
        default="small green wooden block.",
        help="Scene edit: text prompt for GroundingDINO "
             "(default: 'small green wooden block.')",
    )
    parser.add_argument(
        "--scene-edit-no-wrist",
        action="store_true",
        default=False,
        help="Scene edit: disable wrist camera editing "
             "(default: edit both cameras)",
    )

    # --- Failure monitoring ---
    parser.add_argument(
        "--failure-monitor",
        choices=[
            "vlm",
            "signal_primary", "union_failure", "intersect_video",
            "gt", "gt_hitl", "gt_vlm",
        ],
        default=None,
        help="Enable failure detection and recovery. "
             "'vlm' uses periodic VLM checks for both failure detection "
             "and subgoal completion (recommended). "
             "'signal_primary' uses action/EE signals only (no VLM cost). "
             "'union_failure' maximizes failure recall (uses VLM). "
             "'intersect_video' signal + video VLM confirmation. "
             "'gt' uses ground-truth sim state (auto grasp recovery). "
             "'gt_vlm' GT detection + VLM recovery decision. "
             "'gt_hitl' GT detection + human decision. "
             "WARNING: gt / gt_hitl / gt_vlm were developed and tuned "
             "against the block-stack suite; their rules and thresholds "
             "have not been validated on Memory (LH) or Common Sense "
             "(LH-CS) tasks and will misfire on non-block geometries / "
             "tasks with sparse CSM coverage. Prefer --failure-monitor "
             "vlm for general benchmarks. "
             "(default: disabled)",
    )
    parser.add_argument(
        "--recovery-mode",
        choices=[
            # VLM handler modes (configures VLM action vocabulary)
            "replan", "replan_grasp", "grasp",
            "place", "replan_place",          # place-only analogs
            "tools", "replan_tools",           # combined grasp + place
            # Signal handler modes (hardcoded recovery policy)
            "retry", "template", "vlm", "vlm_grasp",
            # GT handler modes
            "grasp_first",
            "place_first", "tools_first",      # GT analogs admitting place
        ],
        default=None,
        help="Recovery strategy on failure detection. "
             "For --failure-monitor vlm: controls available VLM actions. "
             "  'replan' (default) — VLM can choose: next/replan/continue. "
             "  'replan_grasp' — adds grasp_tool to VLM options. "
             "  'grasp' — VLM can choose: next/continue/grasp_tool. "
             "  'place' / 'replan_place' — place_tool analogs of grasp. "
             "  'tools' / 'replan_tools' — combined grasp + place. "
             "For --failure-monitor signal*: hardcoded recovery policy. "
             "  'retry' / 'template' (default) / 'vlm' / 'vlm_grasp'. "
             "For --failure-monitor gt*: 'grasp_first' (default), "
             "'place_first', or 'tools_first' (the latter two admit "
             "place_tool — required for HITL place to fire). "
             "(default: auto-selected based on --failure-monitor)",
    )

    parser.add_argument(
        "--gt-failure-types",
        default=None,
        help="Comma-separated subset of GT failure types to monitor: "
             "wrong_object_picked, object_dropped, no_progress, "
             "subtask_regression. When specified, the GT detector "
             "skips failure types not in this set. "
             "subgoal_complete and in_progress are always active. "
             "(default: all failure types enabled)",
    )
    parser.add_argument(
        "--enable-gt-metrics",
        action="store_true",
        default=False,
        help="Enable passive ground-truth evaluation metrics. Writes "
             "metrics.jsonl next to rewrites.jsonl and does not expose "
             "metrics to the VLM or policy.",
    )
    parser.add_argument(
        "--gt-metric-types",
        default="perception,placement,tool_causality",
        help="Comma-separated passive GT metric detector groups to enable: "
             "perception, placement, tool_causality, gt_state, scene_qa, "
             "target_qa, plan_qa, format_qa, failure_qa, or all. "
             "(default: perception,placement,tool_causality)",
    )

    parser.add_argument(
        "--grasp-seg-mode",
        choices=["gdino_sam2", "sam3", "molmo_sam2", "vlm_sam2", "gt_sim"],
        default="sam3",
        help="Segmentation backend for grasp_with_tool pipeline. "
             "'sam3' (default) uses SAM3 for unified text-prompted "
             "detection + segmentation in one forward pass — best "
             "single-shot grasp perception (requires --enable-sam3 on "
             "the grasp server). "
             "'gdino_sam2' uses GroundingDINO detection → SAM2 "
             "segmentation (two-stage; lighter on memory but a bit "
             "less accurate on cluttered scenes). "
             "'molmo_sam2' uses Molmo2 VLM pointing (free-form prompts, "
             "MOLMO_BASE_URL/MOLMO_MODEL env vars override defaults) → "
             "SAM2 point-prompt segmentation. Requires --enable-sam2 on "
             "the grasp server and a running Molmo2 vLLM server. "
             "Recommended when GPU budget is tight and you want a "
             "single Molmo2 instance shared with --place-seg-mode "
             "molmo_point. "
             "'vlm_sam2' uses the orchestrator's main VLM "
             "(any OpenAI-compatible model — VLM_BASE_URL/VLM_MODEL env "
             "vars override the placeholder defaults) "
             "for pointing → SAM2 point-prompt segmentation. Requires "
             "--enable-sam2 on the grasp server and VLM_API_KEY (or "
             "OPENAI_API_KEY) in the environment. Recommended for "
             "single-VLM perception (same model picks grasp point AND "
             "place point) without standing up a separate Molmo2 server. "
             "'gt_sim' uses ground-truth segmentation masks from the "
             "simulator (requires --enable-gt-state on the eval client "
             "and a GTSegProvider in the eval client loop). Eliminates "
             "perception failures entirely. "
             "(default: sam3 — assumes the grasp server is launched "
             "with --enable-sam3)",
    )

    # ─── tool_chain mode (--mode tool_chain) ───
    parser.add_argument(
        "--tool-chain-max-tools-per-subgoal",
        type=int,
        default=5,
        help="tool_chain mode: hard cap on tool calls before forcing a "
             "replan (avoid plan-thrashing). (default: 5)",
    )
    parser.add_argument(
        "--tool-chain-max-tools-per-episode",
        type=int,
        default=30,
        help="tool_chain mode: hard cap on tool calls before forcing "
             "abort (bound runtime / VLM cost). (default: 30)",
    )

    parser.add_argument(
        "--max-grasp-attempts",
        type=int,
        default=2,
        help="Per-subgoal cap on grasp tool escalations in VLM-handler / "
             "template / retry recovery modes (replan_grasp, replan_tools, "
             "grasp). The grasp_first and vlm_grasp modes are exempt and "
             "self-limit. Counter resets when the subgoal advances. "
             "(default: 2)",
    )

    parser.add_argument(
        "--grasp-topdown-threshold",
        type=float,
        default=0.85,
        help="Strict-top-down filter for Contact-GraspNet output, "
             "passed through to the grasp server. Dot-product cutoff "
             "in [0, 1]: only candidates whose approach axis aligns "
             "with gravity (top-down) by at least this much survive. "
             "0.0 = keep everything (any direction). "
             "0.85 (default) = within ~32° of vertical — handles "
             "asymmetric / mug-by-handle scenes where Contact-GraspNet "
             "would otherwise pick a tilted grasp that the place tool "
             "then has to compensate for. "
             "1.0 = perfectly vertical only. Set to 0.0 if you want "
             "the old (permissive 0.3) behavior.",
    )

    parser.add_argument(
        "--motion-planner",
        choices=["linear", "curobo"],
        default="curobo",
        help="Joint-space trajectory planner for grasp/place tool segments. "
             "'curobo' (default) = collision-aware cuRobo motion generation "
             "served by the grasp server; REQUIRES the grasp server to be "
             "launched with --enable-curobo (else /plan_motion returns 503 "
             "and grasps loud-fail — no silent fallback). 'linear' = straight "
             "linear interpolation in joint space (historical behaviour, no "
             "server dependency). Pose selection / IK are unchanged either "
             "way.",
    )

    parser.add_argument(
        "--enable-stack-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Master switch for stack/no-stack placement semantics. ON "
             "(default) = the VLM's optional 'stack' arg on place()/grasp() is "
             "honoured: stack=True preserves grasp orientation (stacking), "
             "stack=False forces a top-down release (simple pick-and-place, "
             "descent axis matches release rotation). Pass --no-enable-stack-mode "
             "to restore the historical grasp-consistent placement (the place "
             "tool reuses the current EE rotation first, top-down fallback) "
             "and IGNORE the per-call 'stack' tool arg — zero behaviour change.",
    )

    parser.add_argument(
        "--place-seg-mode",
        choices=["gt_sim", "sam3", "gdino_sam2", "vlm_point", "molmo_point"],
        default="molmo_point",
        help="Destination-grounding backend for place_with_tool pipeline. "
             "'molmo_point' (default) asks a Molmo2 vLLM server "
             "(MOLMO_BASE_URL / MOLMO_MODEL env vars override defaults) "
             "for a normalized 2D pixel — handles free-form spatial "
             "destinations like 'empty space near the orange' that "
             "detection-only backends can't (validated empirically; "
             "see debug_perception/comparison_grid.png). "
             "'vlm_point' uses the orchestrator's main VLM (Claude) "
             "for the same role — slightly less precise than Molmo2 "
             "but no extra server required. "
             "'sam3' — single-call detect+segment via SAM3 (requires "
             "--enable-sam3 on the grasp server). Works well on bare "
             "noun phrases ('white bowl') but cannot understand "
             "spatial relations ('near X' returns X's centroid). "
             "'gdino_sam2' — GroundingDINO bbox → SAM2 mask, "
             "centroid is the chosen 2D pixel. Similar caveats to "
             "sam3 re. spatial prepositions. "
             "'gt_sim' reads gt_state.objects[<target>].pos directly "
             "(sim-only, requires --enable-gt-state). "
             "Whether the place tool *fires* is controlled by "
             "--recovery-mode (place / replan_place / tools / "
             "replan_tools / place_first / tools_first).",
    )

    parser.add_argument(
        "--hitl",
        action="store_true",
        default=False,
        help="Enable human-in-the-loop mode. Launches a web UI at "
             "--hitl-port where a human operator can monitor, pause, "
             "rewrite instructions, flag failures, and provide recovery "
             "instructions in real-time.",
    )
    parser.add_argument(
        "--hitl-port",
        type=int,
        default=8002,
        help="Port for HITL web UI (default: 8002)",
    )

    # --- Fine-tuning data collection ---
    parser.add_argument(
        "--collect-trajectories",
        default=None,
        metavar="DIR",
        help="Enable trajectory collection for fine-tuning data. "
             "Records observations, actions, and GT labels during "
             "evaluation. Data is saved to DIR in a format suitable "
             "for DAgger-style fine-tuning. Requires --failure-monitor "
             "gt* for labeled recovery segments. "
             "(default: disabled)",
    )

    # --- Common options ---
    parser.add_argument(
        "--image-key",
        default="observation/exterior_image_1_left",
        help="Key in observation dict for the scene image",
    )
    parser.add_argument(
        "--use-front-camera",
        action="store_true",
        help="Use the front/egocentric camera for VLM calls (planning, "
             "recycle, grasp tool) instead of the default exterior camera. "
             "The policy still receives the exterior camera. "
             "Requires robolab to forward the front camera image.",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Key in observation dict for the instruction",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["default", "vlabench"],
        default="default",
        help="System-prompt variant for subgoal decomposition / monitor / "
             "recycle. 'default' is tuned for robolab/LIBERO pick-and-place "
             "and is the historical behaviour. 'vlabench' tells the VLM to "
             "preserve the original instruction's wording on single-action "
             "tasks and only expand object names when the scene visibly "
             "demands disambiguation — recommended when the underlying VLA "
             "was fine-tuned on specific prompt phrasings (e.g. VLABench's "
             "pi05-primitive-10task).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to log rewrites as JSONL (default: no logging)",
    )
    parser.add_argument(
        "--robolab-output-dir",
        default=os.path.expanduser("~/robolab/output"),
        help="Path to robolab output dir. The proxy searches up to 2 "
             "levels deep for video files matching each episode's "
             "instruction and creates symlinks in the per-episode log "
             "dir. Can be the general root (~/robolab/output/) or a "
             "specific experiment dir (~/robolab/output/<experiment>/). "
             "Set to empty string to disable. "
             "(default: ~/robolab/output)",
    )
    parser.add_argument(
        "--vla-obs-key-remap",
        choices=["none", "dreamzero"],
        default="none",
        help="Rename exterior-image keys before forwarding to the VLA. "
             "'dreamzero' maps DROID-convention 1-indexed cameras "
             "(exterior_image_{1,2}_left) to roboarena-convention "
             "0-indexed (exterior_image_{0,1}_left) for DreamZero-DROID's "
             "WAM server. **Only valid with --client-protocol openpi** "
             "(legacy pi05→DZ bridge). For DZ-native clients use "
             "--client-protocol dreamzero instead. (default: none)",
    )
    parser.add_argument(
        "--client-protocol",
        choices=["openpi", "dreamzero", "cosmos3"],
        default="openpi",
        help="Wire protocol / obs schema of the robot-side client. "
             "'openpi' (default) expects pi0/pi05-family observations "
             "(DROID-convention exterior_image_{1,2}_left, 224×224 padded "
             "images, __episode_id/__step metadata). 'dreamzero' expects "
             "robolab's native DreamZeroClient — roboarena-format "
             "obs with 0-indexed cameras at the DZ-training resolution, "
             "session_id + endpoint fields. With 'dreamzero', the "
             "orchestrator does NO observation translation: it strips "
             "orchestrator-only keys (depth/camera/_raw/gt) before "
             "forwarding to the VLA, but the DZ-native fields pass "
             "through untouched. 'cosmos3' expects robolab's native "
             "Cosmos3Client (policies/cosmos3): all three camera views "
             "are pre-stitched into a SINGLE observation/image (wrist on "
             "top, over-shoulder left/right on the bottom) with no named "
             "cameras, plus joint/gripper/eef proprio. The VLA server is "
             "openpi-ws so no obs/action translation is needed; the proxy "
             "only points VLM image reads at the stitched observation/image "
             "and uses a 32-step open-loop chunk for episode_step "
             "estimation. Evaluate Cosmos3 on any --env (e.g. robolab).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Auto-generate --log-dir so results are always saved
    # ------------------------------------------------------------------
    if not args.log_dir:
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = f"hitl_{ts}" if args.hitl else f"{args.mode}_{ts}"
        args.log_dir = os.path.expanduser(
            f"~/vlm-orchestrator/results/{mode_tag}"
        )
        os.makedirs(args.log_dir, exist_ok=True)
        logging.getLogger(__name__).info(
            f"Auto-created log dir → {args.log_dir}"
        )

    # ------------------------------------------------------------------
    # Apply --env preset: override image keys and timing defaults
    # for LIBERO if the user hasn't explicitly set them.
    # ------------------------------------------------------------------
    _apply_env_preset(args)

    # ------------------------------------------------------------------
    # Validate --gt-failure-types
    # ------------------------------------------------------------------
    _VALID_GT_FAILURE_TYPES = {
        "wrong_object_picked", "object_dropped",
        "no_progress", "subtask_regression",
    }
    if args.gt_failure_types is not None:
        raw = {t.strip() for t in args.gt_failure_types.split(",") if t.strip()}
        invalid = raw - _VALID_GT_FAILURE_TYPES
        if invalid:
            parser.error(
                f"Invalid --gt-failure-types: {invalid}. "
                f"Valid choices: {sorted(_VALID_GT_FAILURE_TYPES)}"
            )
        args.gt_failure_types = raw  # set of strings
    # else: stays None → all types enabled (default)

    # ------------------------------------------------------------------
    # Validate --client-protocol + --vla-obs-key-remap combination
    # ------------------------------------------------------------------
    if (args.client_protocol == "dreamzero"
            and args.vla_obs_key_remap == "dreamzero"):
        parser.error(
            "--client-protocol dreamzero is incompatible with "
            "--vla-obs-key-remap dreamzero. The latter is a "
            "pi05→DZ obs translation that only applies when the "
            "client is openpi-format (pi0/pi05). With a DZ-native "
            "client, no translation is needed — drop "
            "--vla-obs-key-remap."
        )

    # ------------------------------------------------------------------
    # Validate --gt-metric-types
    # ------------------------------------------------------------------
    _VALID_GT_METRIC_TYPES = {
        "all", "perception", "placement", "tool_causality", "gt_state",
        "scene_qa", "target_qa", "plan_qa", "format_qa", "failure_qa",
    }
    raw_metric_types = {
        t.strip() for t in args.gt_metric_types.split(",") if t.strip()
    }
    invalid_metric_types = raw_metric_types - _VALID_GT_METRIC_TYPES
    if invalid_metric_types:
        parser.error(
            f"Invalid --gt-metric-types: {invalid_metric_types}. "
            f"Valid choices: {sorted(_VALID_GT_METRIC_TYPES)}"
        )
    args.gt_metric_types = (
        None if "all" in raw_metric_types else raw_metric_types
    )

    # ------------------------------------------------------------------
    # Validate next_goal mode
    # ------------------------------------------------------------------
    if args.mode == "next_goal":
        if args.failure_monitor is not None:
            parser.error("--mode next_goal does not use --failure-monitor.")
        if args.recovery_mode is not None:
            parser.error("--mode next_goal does not use --recovery-mode.")
        next_goal_template = getattr(args, "next_goal_template", "v1_freeform")
        if (next_goal_template in ("v2", "mcq")
                and getattr(args, "next_goal_task_type", None) is None):
            parser.error(
                f"--next-goal-template {next_goal_template} (MCQ) requires "
                "--next-goal-task-type to supply closed-vocab candidates."
            )

    # ------------------------------------------------------------------
    # Validate --failure-monitor / --recovery-mode compatibility
    # ------------------------------------------------------------------
    _SUBGOAL_MODES = ("subgoal", "subgoal_scene_edit")
    _SIGNAL_MONITORS = ("signal_primary", "union_failure", "intersect_video")
    _GT_MONITORS = ("gt", "gt_hitl", "gt_vlm")
    _VLM_RECOVERY_MODES = (
        "replan", "replan_grasp", "grasp",
        "place", "replan_place",
        "tools", "replan_tools",
    )
    _SIGNAL_RECOVERY_MODES = ("retry", "template", "vlm", "vlm_grasp")
    _GT_RECOVERY_MODES = ("grasp_first", "place_first", "tools_first")

    # Failure detection requires a subgoal-based mode
    if args.failure_monitor is not None and args.mode not in _SUBGOAL_MODES:
        parser.error(
            f"--failure-monitor requires --mode subgoal or "
            f"subgoal_scene_edit (got --mode {args.mode})"
        )

    # tool_chain mode owns its own per-tool VLM checks; the failure-
    # detector / recovery-mode pipeline is irrelevant.  Reject those
    # flags explicitly so the user is not surprised by them being silently
    # ignored.
    if args.mode == "tool_chain":
        if args.failure_monitor is not None:
            parser.error(
                "--mode tool_chain does not use --failure-monitor "
                "(the post-tool VLM check supplants it)."
            )
        if args.recovery_mode is not None:
            parser.error(
                "--mode tool_chain does not use --recovery-mode "
                "(the per-cycle VLM step decides advance/replan/abort)."
            )

    # Auto-default --recovery-mode based on --failure-monitor
    if args.recovery_mode is None:
        if args.failure_monitor == "vlm":
            args.recovery_mode = "replan"
        elif args.failure_monitor in _GT_MONITORS:
            args.recovery_mode = "grasp_first"
        elif args.failure_monitor in _SIGNAL_MONITORS:
            args.recovery_mode = "template"
        # else: None (no failure monitor → no recovery)

    # recovery-mode without failure-monitor makes no sense
    if args.recovery_mode is not None and args.failure_monitor is None:
        parser.error(
            "--recovery-mode requires --failure-monitor to be set"
        )

    # Validate recovery-mode / failure-monitor combinations
    if args.failure_monitor == "vlm":
        if args.recovery_mode not in _VLM_RECOVERY_MODES:
            parser.error(
                f"--failure-monitor vlm supports --recovery-mode "
                f"{list(_VLM_RECOVERY_MODES)} "
                f"(got '{args.recovery_mode}')"
            )

    if args.failure_monitor in _GT_MONITORS:
        if args.recovery_mode not in _GT_RECOVERY_MODES:
            parser.error(
                f"--failure-monitor {args.failure_monitor} supports "
                f"--recovery-mode {list(_GT_RECOVERY_MODES)} "
                f"(got '{args.recovery_mode}')"
            )

    if args.failure_monitor in _SIGNAL_MONITORS:
        if args.recovery_mode not in _SIGNAL_RECOVERY_MODES:
            parser.error(
                f"--failure-monitor {args.failure_monitor} supports "
                f"--recovery-mode {list(_SIGNAL_RECOVERY_MODES)} "
                f"(got '{args.recovery_mode}')"
            )

    # ------------------------------------------------------------------
    # Build the orchestration strategy
    # ------------------------------------------------------------------
    strategy = _build_strategy(args)

    _preflight_motion_planner(args)

    from vlm_orchestrator.proxy import ProxyConfig, OrchestratorProxy

    frontend, backend = _build_protocols(args)

    config = ProxyConfig(
        vla_host=args.vla_host,
        vla_port=args.vla_port,
        host=args.host,
        port=args.port,
        strategy=strategy,
        image_key=args.image_key,
        prompt_key=args.prompt_key,
        log_dir=args.log_dir,
        robolab_output_dir=args.robolab_output_dir or None,
        enable_gt_metrics=args.enable_gt_metrics,
        gt_metric_types=args.gt_metric_types,
        frontend=frontend,
        backend=backend,
        vla_obs_key_remap=args.vla_obs_key_remap,
        client_protocol=args.client_protocol,
    )

    proxy = OrchestratorProxy(config)
    proxy.serve_forever()


# Modes whose grasp/place tools route trajectories through the motion planner.
_GRASP_TOOL_MODES = {"tool_chain", "subgoal", "subgoal_scene_edit", "scene_edit"}
_GRASP_RECOVERY_MODES = {
    "grasp", "grasp_first", "vlm_grasp", "replan_grasp", "replan_tools",
}


def _preflight_motion_planner(args):
    """Fail loudly-early if --motion-planner curobo but the grasp server
    can't provide it.

    cuRobo is now the default motion planner, but it only works when the
    grasp server was launched with --enable-curobo (otherwise /plan_motion
    returns 503 and every grasp loud-fails mid-episode). Rather than let
    that surface deep inside the first episode, probe /health at startup.
    Per the design rules we do NOT silently downgrade to linear — we raise with an
    actionable message so the operator either starts the server with
    --enable-curobo or passes --motion-planner linear explicitly.
    """
    if getattr(args, "motion_planner", "curobo") != "curobo":
        return
    # Only relevant when a grasp-using mode is active.
    mode = getattr(args, "mode", None)
    recovery = getattr(args, "recovery_mode", None)
    uses_grasp = mode in _GRASP_TOOL_MODES and (
        mode == "tool_chain" or recovery in _GRASP_RECOVERY_MODES
    )
    if not uses_grasp:
        return

    from vlm_orchestrator.grasp.client import GraspClient

    client = GraspClient()
    try:
        health = client.health()
    except Exception as e:  # noqa: BLE001
        # Server may not be up yet (some launch orders start it in parallel).
        # Don't hard-fail on unreachable — just warn; the loud-fail on first
        # /plan_motion still protects correctness.
        logging.getLogger("vlm_orchestrator.cli").warning(
            "motion-planner=curobo: could not reach grasp server for a "
            "capability probe (%s: %s). Proceeding; if cuRobo is not loaded "
            "server-side, grasps will loud-fail. Start the grasp server with "
            "--enable-curobo, or pass --motion-planner linear.",
            type(e).__name__, e,
        )
        return

    if "curobo" not in health.get("models", []):
        raise SystemExit(
            "FATAL: --motion-planner curobo (the default) requires the grasp "
            "server to be launched with --enable-curobo, but the server at "
            f"{client._url} reports models={health.get('models')} (no "
            "'curobo'). Either restart the grasp server with --enable-curobo, "
            "or run the orchestrator with --motion-planner linear."
        )
    logging.getLogger("vlm_orchestrator.cli").info(
        "motion-planner=curobo: grasp server reports cuRobo loaded ✓"
    )


def _build_protocols(args):
    """Construct Frontend / Backend instances from CLI flags."""
    from vlm_orchestrator.protocols.openpi_ws import (
        OpenpiWsBackend, OpenpiWsFrontend,
    )

    if args.frontend == "openpi-ws":
        frontend = OpenpiWsFrontend()
    elif args.frontend == "gr00t-zmq":
        from vlm_orchestrator.protocols.gr00t_zmq import Gr00tZmqFrontend
        frontend = Gr00tZmqFrontend()
    elif args.frontend == "openvla-rest":
        from vlm_orchestrator.protocols.openvla_rest import OpenVlaRestFrontend
        frontend = OpenVlaRestFrontend()
    else:
        raise ValueError(f"Unknown --frontend: {args.frontend}")

    # tool_chain mode bypasses the VLA — strategies emit action chunks
    # themselves and ``backend_conn.infer()`` is never called.  Skip the
    # real VLA backend and use an in-process stub that synthesizes the
    # openpi metadata handshake.  Saves one GPU (no policy-server task)
    # and removes the need for vla_stub_server.py on local runners.
    if args.mode == "tool_chain":
        from vlm_orchestrator.protocols.stub import StubBackend
        backend = StubBackend()
    elif args.backend == "openpi-ws":
        backend = OpenpiWsBackend(args.vla_host, args.vla_port)
    elif args.backend == "gr00t-zmq":
        from vlm_orchestrator.protocols.gr00t_zmq import Gr00tZmqBackend
        backend = Gr00tZmqBackend(
            args.vla_host, args.vla_port,
            api_token=getattr(args, "gr00t_api_token", None),
        )
    elif args.backend == "openvla-rest":
        from vlm_orchestrator.protocols.openvla_rest import OpenVlaRestBackend
        backend = OpenVlaRestBackend(
            args.vla_host, args.vla_port,
            unnorm_key=getattr(args, "openvla_unnorm_key", "droid"),
        )
    elif args.backend == "file-ipc":
        from vlm_orchestrator.protocols.file_ipc import FileIPCBackend
        if not args.vla_file_ipc_dir:
            raise ValueError(
                "--backend file-ipc requires --vla-file-ipc-dir <path>"
            )
        backend = FileIPCBackend(args.vla_file_ipc_dir)
    else:
        raise ValueError(f"Unknown --backend: {args.backend}")

    return frontend, backend


def _build_strategy(args):
    """Construct the appropriate strategy from CLI arguments."""
    from vlm_orchestrator.strategies.base import StrategyContext

    # Extra image keys depend on the simulation environment.
    if getattr(args, "env", "robolab") == "libero":
        extra_image_keys = ["observation/wrist_image"]
    else:
        extra_image_keys = ["observation/wrist_image_left"]

    # The VLA *policy* (--client-protocol) can override the obs image
    # schema regardless of --env: the Cosmos3 client pre-stitches all
    # three camera views into a single observation/image (wrist on top,
    # over-shoulder L/R on the bottom) and sends no named cameras. Point
    # VLM/HITL image reads at that stitched frame EXPLICITLY (rather than
    # via the silent _FALLBACK_IMAGE_KEYS path) and drop the wrist extra,
    # since no standalone wrist key exists. Only override --image-key when
    # the user left it at the parser default.
    if getattr(args, "client_protocol", "openpi") == "cosmos3":
        if args.image_key == "observation/exterior_image_1_left":
            args.image_key = "observation/image"
        extra_image_keys = []

    front_image_key = (
        "observation/front_image_left"
        if getattr(args, "use_front_camera", False)
        else None
    )

    if args.mode == "passthrough":
        from vlm_orchestrator.vlm import PassthroughVLM
        from vlm_orchestrator.strategies.passthrough import (
            PassthroughStrategy,
        )

        ctx = StrategyContext(
            vlm=PassthroughVLM(),
            image_key=args.image_key,
            prompt_key=args.prompt_key,
            extra_image_keys=extra_image_keys,
            front_image_key=front_image_key,
            vla_host=args.vla_host,
            vla_port=args.vla_port,
        )
        return PassthroughStrategy(ctx)

    if args.mode == "next_goal":
        from vlm_orchestrator.vlm import PassthroughVLM
        from vlm_orchestrator.strategies.next_goal import (
            NextGoalConfig,
            NextGoalStrategy,
        )

        ctx = StrategyContext(
            vlm=PassthroughVLM(),  # ctx.vlm unused; NextGoalStrategy calls _vlm_call directly
            image_key=args.image_key,
            prompt_key=args.prompt_key,
            extra_image_keys=extra_image_keys,
            front_image_key=front_image_key,
            vla_host=args.vla_host,
            vla_port=args.vla_port,
        )
        config = NextGoalConfig(
            vlm_model=args.vlm_model,
            vlm_temperature=args.vlm_temperature,
            vlm_base_url=args.vlm_base_url,
            vlm_api_key=args.vlm_api_key,
            check_interval=args.check_interval,
            template_name=getattr(args, "next_goal_template", "v1_freeform"),
            task_type=getattr(args, "next_goal_task_type", None),
            max_history_images=getattr(args, "next_goal_max_history", 5),
            vla_capabilities=getattr(args, "vla_capabilities", None),
        )
        return NextGoalStrategy(ctx, config)

    # Modes that need a live VLM
    vlm = _build_vlm(args)
    ctx = StrategyContext(
        vlm=vlm,
        image_key=args.image_key,
        prompt_key=args.prompt_key,
        extra_image_keys=extra_image_keys,
        front_image_key=front_image_key,
        vla_host=args.vla_host,
        vla_port=args.vla_port,
        prompt_style=getattr(args, "prompt_style", "default"),
    )

    if args.mode == "rewrite":
        from vlm_orchestrator.strategies.archive.rewrite import RewriteStrategy

        return RewriteStrategy(ctx, mode=args.rewrite_strategy)

    if args.mode == "adaptive":
        from vlm_orchestrator.strategies.archive.adaptive import AdaptiveStrategy

        return AdaptiveStrategy(ctx, probe_n=args.probe_n)

    # --- HITL setup (shared across subgoal modes) ---
    hitl_state = None
    if getattr(args, "hitl", False):
        from vlm_orchestrator.hitl import HITLServer, HITLState
        hitl_state = HITLState()
        hitl_server = HITLServer(hitl_state, port=args.hitl_port)
        hitl_server.start()

    if args.mode == "tool_chain":
        from vlm_orchestrator.strategies.tool_chain import (
            ToolChainConfig,
            ToolChainStrategy,
        )

        tc_config = ToolChainConfig(
            vlm_model=args.vlm_model,
            vlm_temperature=args.vlm_temperature,
            vlm_base_url=args.vlm_base_url,
            vlm_api_key=args.vlm_api_key,
            max_tools_per_subgoal=getattr(
                args, "tool_chain_max_tools_per_subgoal", 5,
            ),
            max_tools_per_episode=getattr(
                args, "tool_chain_max_tools_per_episode", 30,
            ),
        )
        initial_strategy = _build_initial_strategy(args, ctx)
        return ToolChainStrategy(
            ctx, tc_config, initial_strategy=initial_strategy,
            hitl_state=hitl_state,
            grasp_seg_mode=args.grasp_seg_mode,
            place_seg_mode=getattr(args, "place_seg_mode", None),
            grasp_topdown_threshold=getattr(
                args, "grasp_topdown_threshold", None,
            ),
            env_mode=getattr(args, "env", "robolab"),
            collect_trajectories=getattr(args, "collect_trajectories", None),
            use_front_camera=getattr(args, "use_front_camera", False),
            motion_planner=getattr(args, "motion_planner", "curobo"),
            stack_mode_enabled=getattr(args, "enable_stack_mode", False),
        )

    if args.mode == "subgoal":
        from vlm_orchestrator.strategies.subgoal import (
            SubgoalConfig,
            SubgoalStrategy,
        )

        subgoal_config = SubgoalConfig(
            check_mode=args.subgoal_check_mode,
            check_interval=args.check_interval,
            subgoal_timeout=args.subgoal_timeout,
            vlm_model=args.vlm_model,
            vlm_temperature=args.vlm_temperature,
            vlm_base_url=args.vlm_base_url,
            vlm_api_key=args.vlm_api_key,
        )

        initial_strategy = _build_initial_strategy(args, ctx)

        return SubgoalStrategy(
            ctx, subgoal_config, initial_strategy=initial_strategy,
            failure_monitor=args.failure_monitor,
            recovery_mode=args.recovery_mode,
            hitl_state=hitl_state,
            grasp_seg_mode=args.grasp_seg_mode,
            place_seg_mode=getattr(args, "place_seg_mode", None),
            grasp_topdown_threshold=getattr(args, "grasp_topdown_threshold", None),
            env_mode=getattr(args, "env", "robolab"),
            collect_trajectories=getattr(args, "collect_trajectories", None),
            use_front_camera=getattr(args, "use_front_camera", False),
            gt_failure_types=getattr(args, "gt_failure_types", None),
            max_grasp_attempts=getattr(args, "max_grasp_attempts", None),
            motion_planner=getattr(args, "motion_planner", "curobo"),
            stack_mode_enabled=getattr(args, "enable_stack_mode", False),
        )

    if args.mode == "scene_edit":
        from vlm_orchestrator.strategies.archive.scene_edit import (
            SceneEditConfig,
            SceneEditStrategy,
        )

        scene_config = SceneEditConfig(
            edit_mode=args.scene_edit_mode,
            requery_interval=args.scene_edit_requery_interval,
            edit_wrist=not args.scene_edit_no_wrist,
            fixed_instruction=args.scene_edit_instruction,
            detector=args.scene_edit_detector,
            gdino_prompt=args.scene_edit_gdino_prompt,
            vlm_model=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            vlm_api_key=args.vlm_api_key,
            vlm_temperature=args.vlm_temperature,
            save_debug_images=True,
            debug_image_dir=os.path.join(args.log_dir, "debug_images")
            if args.log_dir
            else None,
        )

        return SceneEditStrategy(ctx, scene_config)

    if args.mode == "subgoal_scene_edit":
        from vlm_orchestrator.strategies.archive.subgoal_scene_edit import (
            SubgoalSceneEditConfig,
            SubgoalSceneEditStrategy,
        )

        config = SubgoalSceneEditConfig(
            edit_mode=args.scene_edit_mode,
            edit_wrist=not args.scene_edit_no_wrist,
            requery_interval=args.scene_edit_requery_interval,
            gdino_score_threshold=0.20,
            check_interval=args.check_interval,
            subgoal_timeout=args.subgoal_timeout,
            vlm_model=args.vlm_model,
            vlm_base_url=args.vlm_base_url,
            vlm_api_key=args.vlm_api_key,
            vlm_temperature=args.vlm_temperature,
            save_debug_images=True,
            debug_image_dir=os.path.join(args.log_dir, "debug_images")
            if args.log_dir
            else None,
        )

        # Build optional initial strategy
        initial_strategy = _build_initial_strategy(args, ctx)

        return SubgoalSceneEditStrategy(
            ctx, config, initial_strategy=initial_strategy,
            failure_monitor=args.failure_monitor,
            recovery_mode=args.recovery_mode,
            hitl_state=hitl_state,
            grasp_seg_mode=args.grasp_seg_mode,
            place_seg_mode=getattr(args, "place_seg_mode", None),
            grasp_topdown_threshold=getattr(args, "grasp_topdown_threshold", None),
            env_mode=getattr(args, "env", "robolab"),
            collect_trajectories=getattr(args, "collect_trajectories", None),
            gt_failure_types=getattr(args, "gt_failure_types", None),
            motion_planner=getattr(args, "motion_planner", "curobo"),
        )

    raise SystemExit(f"Unknown mode: {args.mode}")


def _apply_env_preset(args):
    """Apply simulation-environment presets for image keys and timing.

    When ``--env libero`` is selected, override default values that were
    designed for robolab (IsaacLab/PhysX) with LIBERO-appropriate ones.
    Only overrides values the user has NOT explicitly set (i.e. still at
    their parser defaults).

    LIBERO differences vs robolab:
      - Image key: ``observation/image`` (not ``exterior_image_1_left``)
      - Wrist key: ``observation/wrist_image`` (not ``wrist_image_left``)
      - Controller: OSC_POSE → 5-step replan (not 8-step joint-pos chunks)
      - Timing: faster control loop → slightly tighter check/timeout windows
      - robolab output symlinking: not applicable for LIBERO
    """
    if args.env != "libero":
        return

    log = logging.getLogger(__name__)

    # ── Image keys ──
    if args.image_key == "observation/exterior_image_1_left":
        args.image_key = "observation/image"
        log.info("--env libero: image_key → observation/image")

    # ── Timing ──
    # LIBERO replans every 5 sim steps (vs robolab's 8-step open-loop),
    # so VLM checks fire ~2× more often per sim time.  Halve
    # check_interval to keep the per-real-time check rate consistent.
    if args.check_interval == 80:  # parser default
        args.check_interval = 40
        log.info("--env libero: check_interval → 40 sim steps")
    # subgoal_timeout intentionally NOT overridden for libero — the
    # global default (9999) effectively disables time-based subgoal
    # advancement across all benchmarks.  Set explicitly per run if
    # you actually want a timeout.

    # ── Disable robolab output symlinking ──
    if args.robolab_output_dir == os.path.expanduser("~/robolab/output"):
        args.robolab_output_dir = ""
        log.info("--env libero: disabled robolab output symlinking")


def _build_initial_strategy(args, ctx):
    """Build the optional initial strategy for subgoal modes."""
    if args.subgoal_initial_mode == "rewrite":
        from vlm_orchestrator.strategies.archive.rewrite import RewriteStrategy
        return RewriteStrategy(ctx, mode="first_per_episode")
    if args.subgoal_initial_mode == "adaptive":
        from vlm_orchestrator.strategies.archive.adaptive import AdaptiveStrategy
        return AdaptiveStrategy(ctx, probe_n=args.probe_n)
    return None


def _build_vlm(args):
    """Construct the OpenAI-compatible VLM backend from CLI arguments."""
    from vlm_orchestrator.vlm import OpenAIVLM

    return OpenAIVLM(
        model=args.vlm_model,
        temperature=args.vlm_temperature,
        system_prompt=args.system_prompt,
        base_url=args.vlm_base_url,
        api_key=args.vlm_api_key,
        strategy=args.vlm_strategy,
    )


if __name__ == "__main__":
    main()
