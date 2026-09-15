# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite strategy — VLM rewrites the instruction once at episode start."""

from __future__ import annotations

import logging

from ..base import OrchestrationStrategy, SessionState, StrategyContext

logger = logging.getLogger(__name__)


class RewriteStrategy(OrchestrationStrategy):
    """Rewrite the task instruction once using the VLM.

    ``mode`` controls *when* the rewrite happens:

    * ``"first_per_episode"`` — rewrite each time the prompt changes
      (i.e. every new episode).
    * ``"first_per_connection"`` — rewrite only on the very first
      observation after the client connects.
    """

    def __init__(
        self, ctx: StrategyContext, *, mode: str = "first_per_episode"
    ):
        super().__init__(ctx)
        if mode not in ("first_per_episode", "first_per_connection"):
            raise ValueError(f"Unknown rewrite mode: {mode!r}")
        self.mode = mode

    def process(
        self, obs: dict, state: SessionState
    ) -> tuple[dict, SessionState]:
        current_prompt = self.ctx.get_prompt(obs)
        if current_prompt is None:
            return obs, state

        is_new = self._is_new_episode(obs, state)

        should_rewrite = False
        if self.mode == "first_per_episode" and is_new:
            should_rewrite = True
        elif self.mode == "first_per_connection" and state.infer_count == 0:
            should_rewrite = True

        if is_new:
            state.original_instruction = current_prompt
            state.rewritten_instruction = None

        if should_rewrite:
            image = self.ctx.get_vlm_image(obs)
            if image is not None:
                logger.info(
                    f'Episode {state.episode_id}: VLM examining: '
                    f'"{current_prompt}"'
                )
                try:
                    rewritten, elapsed = self._rewrite_with_retry(
                        current_prompt, image, obs,
                    )
                    if rewritten != current_prompt:
                        logger.info(
                            f'  Rewritten ({elapsed:.1f}s): "{rewritten}"'
                        )
                        state.rewritten_instruction = rewritten
                        obs = self.ctx.set_prompt(obs, rewritten)
                    else:
                        logger.info(f"  Kept original ({elapsed:.1f}s)")
                        state.rewritten_instruction = current_prompt

                    state.log({
                        "type": "rewrite",
                        "original": current_prompt,
                        "rewritten": rewritten,
                        "changed": rewritten != current_prompt,
                        "vlm_latency_s": round(elapsed, 2),
                    })
                except Exception as e:
                    logger.warning(f"  VLM rewrite failed: {e}")
                    state.rewritten_instruction = current_prompt
            else:
                logger.warning(
                    f"No image at '{self.ctx.image_key}', skipping rewrite"
                )
        elif (
            state.rewritten_instruction is not None
            and state.rewritten_instruction != current_prompt
        ):
            # Ongoing episode: keep using the rewritten instruction
            obs = self.ctx.set_prompt(obs, state.rewritten_instruction)

        return obs, state
