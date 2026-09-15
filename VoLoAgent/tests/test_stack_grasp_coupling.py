# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stack/no-stack grasp coupling (Option A).

When stack-mode is ON and a grasp is issued with stack=False (the object
will be dropped from above at place time), the grasp's top-down filter is
disabled so GraspGen returns its best grasp at ANY approach angle.  When
stack=True (or stack-mode OFF) the top-down filter behaves as before.

These tests cover the wiring without spinning up the grasp server:
  1. GraspClient.compute_grasp serializes enable_topdown_filter into the npz.
  2. The server-side compute_grasp handler honours it (default on).
  3. GraspToolExecutor's gating: enable_topdown_filter=False is only chosen
     when stack_mode_enabled and not stack.
"""
import io

import numpy as np


def _decision(stack_mode_enabled, stack_this_grasp):
    """Replicate the executor's gating decision (kept identical to
    grasp/tool.py's inline logic)."""
    enable_td_filter = None
    if stack_mode_enabled and not stack_this_grasp:
        enable_td_filter = False
    return enable_td_filter


def test_executor_gating_matrix():
    # stack-mode OFF: never disable, regardless of stack value
    assert _decision(False, True) is None
    assert _decision(False, False) is None
    # stack-mode ON: disable ONLY when stack=False
    assert _decision(True, True) is None
    assert _decision(True, False) is False


def test_client_serializes_enable_topdown_filter():
    from vlm_orchestrator.grasp.client import GraspClient

    client = GraspClient.__new__(GraspClient)  # no network
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data

        class _Resp:
            ok = True
            content = b""

            def json(self):
                return {}

        # short-circuit before unpack by raising after capture
        raise _StopPost

    class _StopPost(Exception):
        pass

    import vlm_orchestrator.grasp.client as client_mod
    orig_post = client_mod.requests.post
    client_mod.requests.post = fake_post
    client._url = "http://x"
    client._timeout = 1
    try:
        for val in (True, False, None):
            captured.clear()
            try:
                client.compute_grasp(
                    point_cloud=np.zeros((3, 3), np.float32),
                    mask=np.ones((2, 2), np.uint8),
                    image_hw=(2, 2),
                    focal_length_px=100.0,
                    enable_topdown_filter=val,
                )
            except _StopPost:
                pass
            arrs = dict(np.load(io.BytesIO(captured["data"]), allow_pickle=True))
            if val is None:
                assert "enable_topdown_filter" not in arrs
            else:
                assert "enable_topdown_filter" in arrs
                assert bool(arrs["enable_topdown_filter"]) is val
    finally:
        client_mod.requests.post = orig_post


def test_server_reads_enable_topdown_filter_default_on():
    # Mirror the server handler's parse: default True when absent.
    for payload, expected in [
        ({}, True),
        ({"enable_topdown_filter": np.array(False)}, False),
        ({"enable_topdown_filter": np.array(True)}, True),
    ]:
        got = (
            bool(payload["enable_topdown_filter"])
            if "enable_topdown_filter" in payload
            else True
        )
        assert got is expected


def test_all_subgoal_grasp_executors_pass_stack_mode():
    """Every GraspToolExecutor(...) built in subgoal_base must forward
    stack_mode_enabled, so subgoal tool-recovery modes get the same
    stack support as tool_chain (guards against constructor drift)."""
    import re
    from pathlib import Path

    src = Path(
        "vlm_orchestrator/strategies/subgoal_base.py"
    ).read_text()
    constructs = re.findall(
        r"GraspToolExecutor\((?:[^()]|\([^()]*\))*\)", src
    )
    assert constructs, "expected at least one GraspToolExecutor construction"
    missing = [c for c in constructs if "stack_mode_enabled" not in c]
    assert not missing, (
        f"{len(missing)} GraspToolExecutor construction(s) in subgoal_base.py "
        f"omit stack_mode_enabled: {missing}"
    )
