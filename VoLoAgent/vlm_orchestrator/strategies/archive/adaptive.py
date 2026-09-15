# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive compare-and-pick strategy.

Probes the VLA policy with both the original and a VLM-rewritten instruction,
then picks whichever produces lower trajectory disagreement.
"""

from __future__ import annotations

import logging

from ..base import OrchestrationStrategy, SessionState, StrategyContext

logger = logging.getLogger(__name__)


class AdaptiveStrategy(OrchestrationStrategy):
    """Compare-and-pick: probe original vs rewritten, pick lower disagreement."""

    def __init__(self, ctx: StrategyContext, *, probe_n: int = 30):
        super().__init__(ctx)
        self.probe_n = probe_n

    def process(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        current_prompt = self.ctx.get_prompt(obs)
        if current_prompt is None:
            return obs, state

        is_new = self._is_new_episode(obs, state)

        if is_new:
            return self._compare_and_pick(obs, state, current_prompt)

        if (
            state.rewritten_instruction is not None
            and state.rewritten_instruction != current_prompt
        ):
            obs = self.ctx.set_prompt(obs, state.rewritten_instruction)

        return obs, state

    # ------------------------------------------------------------------

    def _compare_and_pick(
        self,
        obs: dict,
        state: SessionState,
        current_prompt: str,
    ) -> tuple[dict, SessionState]:
        from vlm_orchestrator.signals.policy_prober import PolicyProber, compare_and_pick

        state.original_instruction = current_prompt
        state.rewritten_instruction = None

        image = self.ctx.get_vlm_image(obs)
        if image is None:
            state.adaptive_decision = "passthrough"
            state.rewritten_instruction = current_prompt
            return obs, state

        n = self.probe_n

        # Step 1: Probe original
        logger.info(
            f'Episode {state.episode_id}: COMPARE-AND-PICK probing '
            f'{n}x for original: "{current_prompt}"'
        )
        try:
            prober = PolicyProber(
                host=self.ctx.vla_host, port=self.ctx.vla_port,
            )
        except Exception as e:
            logger.error(f"Cannot connect prober: {e}; keeping original")
            state.adaptive_decision = "passthrough"
            state.rewritten_instruction = current_prompt
            return obs, state

        try:
            original_result = prober.probe(obs, n_samples=n)
            state.probe_result = original_result
            state.probe_variance = original_result.mean_action_var
        except Exception as e:
            logger.error(
                f"Original probe failed: {e}; falling back to VLM rewrite"
            )
            prober.close()
            state.adaptive_decision = "rewrite"
            return self._fallback_rewrite(obs, state)

        # Step 2: VLM rewrite
        try:
            rewritten, vlm_elapsed = self._rewrite_with_retry(
                current_prompt, image, obs,
            )
            logger.info(
                f'  VLM rewrite ({vlm_elapsed:.1f}s): "{rewritten}"'
            )
        except Exception as e:
            logger.warning(f"  VLM rewrite failed: {e}; keeping original")
            prober.close()
            state.adaptive_decision = "original"
            state.rewritten_instruction = current_prompt
            state.log({
                "type": "adaptive",
                "original": current_prompt,
                "rewritten": current_prompt,
                "adaptive_decision": "original",
                "vlm_latency_s": 0.0,
            })
            return obs, state

        # If still identical, skip second probe
        if rewritten == current_prompt:
            logger.info("  VLM returned identical instruction, keeping original")
            prober.close()
            state.adaptive_decision = "original"
            state.rewritten_instruction = current_prompt
            state.log({
                "type": "adaptive",
                "original": current_prompt,
                "rewritten": current_prompt,
                "adaptive_decision": "original",
                "vlm_latency_s": round(vlm_elapsed, 2),
                "original_probe": original_result.to_dict(),
            })
            return obs, state

        # Step 3: Probe rewritten
        logger.info(f'  Probing {n}x for rewritten: "{rewritten}"')
        try:
            rewritten_obs = self.ctx.set_prompt(obs, rewritten)
            rewritten_result = prober.probe(rewritten_obs, n_samples=n)
        except Exception as e:
            logger.error(f"Rewritten probe failed: {e}; keeping original")
            prober.close()
            state.adaptive_decision = "original"
            state.rewritten_instruction = current_prompt
            return obs, state
        finally:
            prober.close()

        # Step 4: Compare and pick
        cmp = compare_and_pick(original_result, rewritten_result)

        logger.info(
            f"  COMPARE: original_td={original_result.traj_disagreement:.4f}, "
            f"rewritten_td={rewritten_result.traj_disagreement:.4f} "
            f"-> winner={cmp.winner.upper()} "
            f"(improvement={cmp.improvement:+.4f})"
        )

        state.adaptive_decision = cmp.winner
        if cmp.winner == "rewritten":
            state.rewritten_instruction = rewritten
            obs = self.ctx.set_prompt(obs, rewritten)
        else:
            state.rewritten_instruction = current_prompt

        state.log({
            "type": "adaptive",
            "original": current_prompt,
            "rewritten": rewritten
            if cmp.winner == "rewritten"
            else current_prompt,
            "changed": cmp.winner == "rewritten",
            "adaptive_decision": cmp.winner,
            "vlm_latency_s": round(vlm_elapsed, 2),
            "improvement": round(cmp.improvement, 6),
            "original_probe": original_result.to_dict(),
            "rewritten_probe": rewritten_result.to_dict(),
        })

        return obs, state

    # ------------------------------------------------------------------

    def _fallback_rewrite(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        """VLM rewrite without comparison (when probe fails)."""
        current_prompt = self.ctx.get_prompt(obs)
        image = self.ctx.get_vlm_image(obs)

        try:
            rewritten, elapsed = self._rewrite_with_retry(
                current_prompt, image, obs,
            )
            if rewritten != current_prompt:
                logger.info(
                    f'  Fallback rewritten ({elapsed:.1f}s): "{rewritten}"'
                )
                state.rewritten_instruction = rewritten
                obs = self.ctx.set_prompt(obs, rewritten)
            else:
                logger.info(
                    f"  Fallback: VLM kept original ({elapsed:.1f}s)"
                )
                state.rewritten_instruction = current_prompt

            state.log({
                "type": "adaptive",
                "original": current_prompt,
                "rewritten": rewritten,
                "changed": rewritten != current_prompt,
                "adaptive_decision": "fallback_rewrite",
                "vlm_latency_s": round(elapsed, 2),
            })
        except Exception as e:
            logger.warning(f"  Fallback VLM rewrite also failed: {e}")
            state.rewritten_instruction = current_prompt

        return obs, state
