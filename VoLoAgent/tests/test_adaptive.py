# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive compare-and-pick rewrite gating."""

import sys
import threading
import time

import numpy as np
import websockets.sync.server as ws_server_sync

from vlm_orchestrator.utils import codec
from vlm_orchestrator.signals.policy_prober import (
    CompareResult,
    PolicyProber,
    ProbeResult,
    compare_and_pick,
    compute_traj_disagreement,
    compute_variance_metrics,
    probe_once,
)
from vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig, SessionState
from vlm_orchestrator.vlm import VLMBackend


# ---------------------------------------------------------------------------
#  Pure-logic tests (no I/O)
# ---------------------------------------------------------------------------


class TestComputeVarianceMetrics:
    """Test variance computation from action arrays."""

    def test_identical_actions_zero_variance(self):
        """N identical action chunks → variance = 0."""
        action = np.ones((50, 8), dtype=np.float32)
        stacked = np.stack([action] * 10)  # (10, 50, 8)
        m = compute_variance_metrics(stacked)
        assert m["mean_action_var"] == 0.0
        assert m["first_step_var"] == 0.0
        assert m["action_spread"] == 0.0
        assert m["gripper_agreement"] == 1.0

    def test_high_variance_actions(self):
        """Random actions → positive variance."""
        rng = np.random.default_rng(42)
        stacked = rng.standard_normal((20, 50, 8)).astype(np.float32)
        m = compute_variance_metrics(stacked)
        assert m["mean_action_var"] > 0.5
        assert m["first_step_var"] > 0.0
        assert m["action_spread"] > 0.0

    def test_single_sample(self):
        """N=1 → degenerate case, variance = 0."""
        stacked = np.zeros((1, 50, 8), dtype=np.float32)
        m = compute_variance_metrics(stacked)
        assert m["mean_action_var"] == 0.0
        assert m["action_spread"] == 0.0
        assert m["gripper_agreement"] == 1.0

    def test_gripper_agreement_split(self):
        """Half open, half closed → agreement = 0.5."""
        stacked = np.zeros((20, 50, 8), dtype=np.float32)
        stacked[:10, :, -1] = 1.0   # first 10: gripper closed (>0.5)
        stacked[10:, :, -1] = 0.0   # last 10: gripper open (<0.5)
        m = compute_variance_metrics(stacked)
        assert m["gripper_agreement"] == 0.5

    def test_known_variance(self):
        """Verify mean_action_var against manual calculation."""
        # 5 samples, horizon=2, dim=2
        actions = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.2, 1.8], [3.2, 3.8]],
            [[0.8, 2.2], [2.8, 4.2]],
            [[1.1, 1.9], [3.1, 3.9]],
            [[0.9, 2.1], [2.9, 4.1]],
        ], dtype=np.float32)
        m = compute_variance_metrics(actions)
        # Manual: var over axis 0, then mean
        expected = float(np.mean(np.var(actions, axis=0)))
        assert abs(m["mean_action_var"] - expected) < 1e-6


# ---------------------------------------------------------------------------
#  Trajectory disagreement tests
# ---------------------------------------------------------------------------


class TestComputeTrajDisagreement:
    """Test the trajectory disagreement metric."""

    def test_identical_actions_zero_disagreement(self):
        """N identical action chunks → disagreement = 0."""
        action = np.ones((50, 8), dtype=np.float32)
        stacked = np.stack([action] * 10)  # (10, 50, 8)
        td = compute_traj_disagreement(stacked, horizon=8)
        assert td == 0.0

    def test_single_sample_zero_disagreement(self):
        """N=1 → degenerate, disagreement = 0."""
        stacked = np.ones((1, 50, 8), dtype=np.float32)
        td = compute_traj_disagreement(stacked, horizon=8)
        assert td == 0.0

    def test_random_actions_positive_disagreement(self):
        """Random actions → positive disagreement."""
        rng = np.random.default_rng(42)
        stacked = rng.standard_normal((20, 50, 8)).astype(np.float32)
        td = compute_traj_disagreement(stacked, horizon=8)
        assert 0.0 < td <= 1.5  # can exceed 1.0 with anti-correlated samples

    def test_two_clusters_high_disagreement(self):
        """Two opposing clusters → high disagreement."""
        # Half move steadily in +direction, half in -direction
        # Need actual trajectories with step-to-step changes (not constant)
        stacked = np.zeros((20, 50, 8), dtype=np.float32)
        for t in range(50):
            stacked[:10, t, :7] = 0.1 * t       # cluster 1: increasing joints
            stacked[10:, t, :7] = -0.1 * t       # cluster 2: decreasing joints
        td = compute_traj_disagreement(stacked, horizon=8)
        # Opposing directions → high disagreement (close to 1.0)
        assert td > 0.9

    def test_horizon_truncation(self):
        """Only first `horizon` steps should matter."""
        rng = np.random.default_rng(99)
        # First 8 steps: all samples move identically (+0.1 per step)
        stacked = np.zeros((10, 50, 8), dtype=np.float32)
        for t in range(8):
            stacked[:, t, :7] = 0.1 * t  # same trajectory for all
        # After step 8: divergent noise (should be ignored)
        stacked[:, 8:, :] = rng.standard_normal((10, 42, 8))

        td = compute_traj_disagreement(stacked, horizon=8)
        assert td < 1e-5  # only first 8 steps matter, and they're identical

    def test_structured_variation_lower_than_random(self):
        """Samples with a shared signal have lower disagreement than pure noise.

        After StandardScaler, structured variation (shared direction + small
        perturbation) preserves directional consensus, while IID noise does not.
        """
        rng = np.random.default_rng(42)
        n, horizon, dim = 20, 8, 8

        # Structured: shared trajectory direction + small perturbation
        shared_direction = rng.standard_normal((horizon, dim)).astype(np.float32)
        structured = np.array([
            shared_direction + rng.standard_normal((horizon, dim)).astype(np.float32) * 0.2
            for _ in range(n)
        ])
        # Pad to 50 timesteps
        padded_struct = np.zeros((n, 50, dim), dtype=np.float32)
        padded_struct[:, :horizon, :] = structured
        td_structured = compute_traj_disagreement(padded_struct, horizon=horizon)

        # Pure IID noise (no shared direction)
        pure_noise = rng.standard_normal((n, 50, dim)).astype(np.float32)
        td_noise = compute_traj_disagreement(pure_noise, horizon=horizon)

        # Structured variation should have LOWER disagreement
        assert td_structured < td_noise

    def test_disagreement_increases_with_cluster_split(self):
        """More balanced cluster split → higher disagreement.

        Simulates the real use case: clear instructions lead all samples to
        one trajectory; vague instructions split samples across trajectories.
        """
        n, horizon, dim = 20, 8, 8

        # Two distinct trajectories (deterministic)
        rng_base = np.random.default_rng(42)
        traj_a = rng_base.standard_normal((horizon, dim)).astype(np.float32) * 2
        traj_b = -traj_a  # opposite direction

        tds = []
        for n_in_b in [0, 5, 10]:
            # Use same noise seed each time so only the split varies
            rng_noise = np.random.default_rng(99)
            samples = np.zeros((n, 50, dim), dtype=np.float32)
            for i in range(n):
                base = traj_b if i < n_in_b else traj_a
                samples[i, :horizon, :] = base + rng_noise.standard_normal((horizon, dim)).astype(np.float32) * 0.3
            tds.append(compute_traj_disagreement(samples, horizon=horizon))

        # n_in_b=0: all same cluster → low disagreement
        # n_in_b=10: 50/50 split → highest disagreement
        assert tds[0] < tds[2], f"no split ({tds[0]:.4f}) should < 50/50 ({tds[2]:.4f})"


# ---------------------------------------------------------------------------
#  Compare-and-pick tests
# ---------------------------------------------------------------------------


class TestCompareAndPick:
    """Test the compare_and_pick decision function."""

    def _make_probe(self, instruction: str, traj_disagreement: float) -> ProbeResult:
        return ProbeResult(
            instruction=instruction,
            n_samples=20,
            elapsed_s=1.0,
            mean_action_var=0.005,
            first_step_var=0.001,
            action_spread=0.3,
            gripper_agreement=0.9,
            traj_disagreement=traj_disagreement,
        )

    def test_rewrite_wins_when_lower(self):
        """Rewrite has lower disagreement → rewrite wins."""
        orig = self._make_probe("do the thing", traj_disagreement=0.5)
        rewr = self._make_probe("pick up the banana", traj_disagreement=0.2)
        cmp = compare_and_pick(orig, rewr)
        assert cmp.winner == "rewritten"
        assert cmp.winning_instruction == "pick up the banana"
        assert cmp.improvement > 0

    def test_original_wins_when_lower(self):
        """Original has lower disagreement → original wins."""
        orig = self._make_probe("pick up the banana", traj_disagreement=0.1)
        rewr = self._make_probe("grab the yellow banana from the table", traj_disagreement=0.3)
        cmp = compare_and_pick(orig, rewr)
        assert cmp.winner == "original"
        assert cmp.winning_instruction == "pick up the banana"
        assert cmp.improvement < 0

    def test_tie_goes_to_original(self):
        """Equal disagreement → original wins (prefer user's wording)."""
        orig = self._make_probe("put it there", traj_disagreement=0.25)
        rewr = self._make_probe("place the cup on the shelf", traj_disagreement=0.25)
        cmp = compare_and_pick(orig, rewr)
        assert cmp.winner == "original"
        assert cmp.winning_instruction == "put it there"
        assert cmp.improvement == 0.0

    def test_to_dict(self):
        """CompareResult.to_dict() returns serialisable summary."""
        orig = self._make_probe("do the thing", traj_disagreement=0.5)
        rewr = self._make_probe("pick up the banana", traj_disagreement=0.2)
        cmp = compare_and_pick(orig, rewr)
        d = cmp.to_dict()
        assert d["winner"] == "rewritten"
        assert "original" in d
        assert "rewritten" in d
        assert d["improvement"] > 0

    def test_improvement_sign(self):
        """Improvement is positive when rewrite is better, negative otherwise."""
        orig = self._make_probe("a", traj_disagreement=0.6)
        rewr = self._make_probe("b", traj_disagreement=0.4)
        cmp = compare_and_pick(orig, rewr)
        assert abs(cmp.improvement - 0.2) < 1e-9

        # Reverse
        cmp2 = compare_and_pick(rewr, orig)
        assert abs(cmp2.improvement - (-0.2)) < 1e-9


class TestProbeResult:
    """Test ProbeResult dataclass."""

    def test_to_dict(self):
        pr = ProbeResult(
            instruction="test",
            n_samples=20,
            elapsed_s=1.234,
            mean_action_var=0.00523,
            first_step_var=0.00112,
            action_spread=0.305,
            gripper_agreement=0.95,
            traj_disagreement=0.152,
            all_actions=np.zeros((20, 50, 8)),
        )
        d = pr.to_dict()
        assert "all_actions" not in d
        assert d["instruction"] == "test"
        assert d["n_samples"] == 20
        assert d["elapsed_s"] == 1.23  # rounded
        assert d["mean_action_var"] == 0.00523
        assert d["traj_disagreement"] == 0.152

    def test_traj_disagreement_default(self):
        """traj_disagreement defaults to 0.0."""
        pr = ProbeResult(
            instruction="test",
            n_samples=1,
            elapsed_s=0.0,
            mean_action_var=0.0,
            first_step_var=0.0,
            action_spread=0.0,
            gripper_agreement=1.0,
        )
        assert pr.traj_disagreement == 0.0


# ---------------------------------------------------------------------------
#  Integration tests (mock policy server)
# ---------------------------------------------------------------------------

# A mock VLA server that adds controllable noise to actions
_MOCK_ACTION_NOISE = 0.0  # module-level, set per test


def _noisy_vla_handler(ws):
    """Mock VLA that returns actions with configurable noise."""
    packer = codec.Packer()
    ws.send(packer.pack({"model": "mock-vla-noisy", "version": "test"}))
    rng = np.random.default_rng(42)
    while True:
        try:
            raw = ws.recv()
            obs = codec.unpackb(raw)
            base_action = np.zeros((50, 8), dtype=np.float32)
            base_action[:, :7] = 0.1  # small constant arm action
            base_action[:, 7] = 1.0   # gripper closed
            noise = rng.standard_normal(base_action.shape).astype(np.float32) * _MOCK_ACTION_NOISE
            response = {
                "actions": base_action + noise,
                "received_prompt": obs.get("prompt", ""),
            }
            ws.send(packer.pack(response))
        except Exception:
            break


def _start_mock_vla(port: int) -> ws_server_sync.WebSocketServer:
    server = ws_server_sync.serve(
        _noisy_vla_handler, "127.0.0.1", port, compression=None, max_size=None
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    return server


class TestPolicyProber:
    """Integration tests for PolicyProber with mock server."""

    def test_probe_low_noise(self):
        """Low noise → low variance, traj_disagreement is computed."""
        global _MOCK_ACTION_NOISE
        _MOCK_ACTION_NOISE = 0.001
        port = 19990

        server = _start_mock_vla(port)
        try:
            obs = {
                "prompt": "pick up the banana",
                "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
            }
            with PolicyProber("127.0.0.1", port) as prober:
                result = prober.probe(obs, n_samples=10)

            assert result.n_samples == 10
            assert result.mean_action_var < 0.01
            # traj_disagreement is computed (value depends on mock noise structure)
            assert isinstance(result.traj_disagreement, float)
        finally:
            server.shutdown()

    def test_probe_high_noise(self):
        """High noise → high disagreement."""
        global _MOCK_ACTION_NOISE
        _MOCK_ACTION_NOISE = 0.5
        port = 19991

        server = _start_mock_vla(port)
        try:
            obs = {
                "prompt": "put the thing somewhere",
                "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
            }
            with PolicyProber("127.0.0.1", port) as prober:
                result = prober.probe(obs, n_samples=10)

            assert result.mean_action_var > 0.01
            assert result.traj_disagreement > 0  # noise causes some disagreement
        finally:
            server.shutdown()

    def test_probe_once_convenience(self):
        """Test the one-shot ``probe_once()`` helper."""
        global _MOCK_ACTION_NOISE
        _MOCK_ACTION_NOISE = 0.0
        port = 19992

        server = _start_mock_vla(port)
        try:
            obs = {
                "prompt": "stack the bowls",
                "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
            }
            result = probe_once(obs, n_samples=5, host="127.0.0.1", port=port)
            assert result.n_samples == 5
            assert result.mean_action_var < 1e-9   # zero noise → zero variance
            # Zero noise means identical actions → early-exit → disagreement = 0
            assert result.traj_disagreement == 0.0
        finally:
            server.shutdown()

    def test_probe_returns_traj_disagreement(self):
        """Probe result always includes traj_disagreement field."""
        global _MOCK_ACTION_NOISE
        _MOCK_ACTION_NOISE = 0.1
        port = 19997

        server = _start_mock_vla(port)
        try:
            obs = {
                "prompt": "move robot arm",
                "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
            }
            result = probe_once(obs, n_samples=10, host="127.0.0.1", port=port)
            assert hasattr(result, "traj_disagreement")
            assert isinstance(result.traj_disagreement, float)
            # Also present in to_dict
            assert "traj_disagreement" in result.to_dict()
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
#  Proxy adaptive mode integration test (compare-and-pick)
# ---------------------------------------------------------------------------


class UppercaseVLM(VLMBackend):
    """Test VLM that uppercases the instruction."""
    call_count = 0

    def rewrite_instruction(
        self, instruction: str, image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        UppercaseVLM.call_count += 1
        return instruction.upper()


class IdentityVLM(VLMBackend):
    """Test VLM that returns the instruction unchanged."""
    call_count = 0

    def rewrite_instruction(
        self, instruction: str, image: np.ndarray,
        extra_images: list[np.ndarray] | None = None,
    ) -> str:
        IdentityVLM.call_count += 1
        return instruction


class TestProxyAdaptiveMode:
    """Test the proxy with rewrite_strategy='adaptive' (compare-and-pick)."""

    def test_adaptive_vlm_always_called(self):
        """In compare-and-pick, VLM is always called (no threshold gating).
        The mock VLA has low noise, so both probes will have ~same disagreement,
        but the VLM is still invoked to generate the rewrite candidate."""
        global _MOCK_ACTION_NOISE
        _MOCK_ACTION_NOISE = 0.001
        vla_port = 19993
        proxy_port = 19994

        vla_server = _start_mock_vla(vla_port)
        UppercaseVLM.call_count = 0

        config = ProxyConfig(
            vla_host="127.0.0.1",
            vla_port=vla_port,
            host="127.0.0.1",
            port=proxy_port,
            vlm=UppercaseVLM(),
            rewrite_strategy="adaptive",
            adaptive_probe_n=5,
        )
        proxy = OrchestratorProxy(config)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        time.sleep(0.5)

        import websockets.sync.client as ws_client

        try:
            packer = codec.Packer()
            client = ws_client.connect(
                f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
            )
            # Receive metadata
            codec.unpackb(client.recv())

            # Send observation
            obs = {
                "prompt": "pick up the banana",
                "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
            }
            client.send(packer.pack(obs))
            response = codec.unpackb(client.recv())

            # VLM was called (compare-and-pick always generates a candidate)
            assert UppercaseVLM.call_count == 1

            # With low noise both probes are similar; the winner is either original
            # or rewritten — we just verify the pipeline ran and returned *some* prompt
            prompt_used = response["received_prompt"]
            assert prompt_used in ("pick up the banana", "PICK UP THE BANANA")

            client.close()
        finally:
            vla_server.shutdown()

    def test_adaptive_identity_vlm_skips_second_probe(self):
        """When VLM returns identical text, second probe is skipped."""
        global _MOCK_ACTION_NOISE
        _MOCK_ACTION_NOISE = 0.001
        vla_port = 19995
        proxy_port = 19996

        vla_server = _start_mock_vla(vla_port)
        IdentityVLM.call_count = 0

        config = ProxyConfig(
            vla_host="127.0.0.1",
            vla_port=vla_port,
            host="127.0.0.1",
            port=proxy_port,
            vlm=IdentityVLM(),
            rewrite_strategy="adaptive",
            adaptive_probe_n=5,
        )
        proxy = OrchestratorProxy(config)
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        time.sleep(0.5)

        import websockets.sync.client as ws_client

        try:
            packer = codec.Packer()
            client = ws_client.connect(
                f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
            )
            codec.unpackb(client.recv())

            obs = {
                "prompt": "pick up the banana",
                "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/joint_position": np.zeros(7, dtype=np.float32),
            }
            client.send(packer.pack(obs))
            response = codec.unpackb(client.recv())

            # VLM was called once
            assert IdentityVLM.call_count == 1
            # Original prompt kept (VLM returned same text, no second probe needed)
            assert response["received_prompt"] == "pick up the banana"

            client.close()
        finally:
            vla_server.shutdown()


# ---------------------------------------------------------------------------
#  SessionState adaptive fields test
# ---------------------------------------------------------------------------


class TestSessionStateAdaptiveFields:
    def test_defaults(self):
        s = SessionState()
        assert s.adaptive_decision is None
        assert s.probe_variance is None
        assert s.probe_result is None


# ---------------------------------------------------------------------------
#  Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestComputeVarianceMetrics,
        TestComputeTrajDisagreement,
        TestCompareAndPick,
        TestProbeResult,
        TestPolicyProber,
        TestProxyAdaptiveMode,
        TestSessionStateAdaptiveFields,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            name = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                print(f"  PASS: {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed! ✅")
    else:
        print(f"{failed} test(s) FAILED ❌")
        sys.exit(1)
