# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM Interpretation Entropy metric.

Ask a VLM to generate K diverse interpretations of an instruction given
a scene image, cluster them by text similarity, and compute Shannon entropy.

High entropy → many distinct interpretations → ambiguous instruction.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from vlm_orchestrator.signals.base import Metric, MetricResult


# ── Shared utilities ────────────────────────────────────────────────────────

def cluster_by_text_similarity(
    interpretations: list[str], threshold: float = 0.5
) -> list[int]:
    """Cluster strings by Jaccard word-overlap similarity."""
    n = len(interpretations)
    word_sets = [set(interp.lower().split()) for interp in interpretations]
    labels = [-1] * n
    cluster_id = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = cluster_id
        for j in range(i + 1, n):
            if labels[j] >= 0:
                continue
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if intersection / max(union, 1) >= threshold:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def compute_entropy(labels: list[int]) -> float:
    """Shannon entropy (bits) of a label distribution."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    probs = [c / total for c in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))


def compute_normalized_entropy(labels: list[int]) -> float:
    """Entropy normalised by log₂(K), in [0, 1]."""
    k = len(labels)
    if k <= 1:
        return 0.0
    raw = compute_entropy(labels)
    return raw / np.log2(k)


INTERPRETATION_PROMPT = """\
You are analyzing a robot manipulation instruction given a scene image.

The robot arm is visible in the scene. The instruction given to the robot is:
"{instruction}"

Generate ONE specific, concrete interpretation of what this instruction means \
the robot should do, given the objects visible in the scene. Be specific about:
- Which exact object(s) to manipulate
- Where exactly to place them
- What order to do things in
- What "away" or other vague terms mean in this context

Respond with ONLY the interpretation as a single sentence, nothing else. \
Be creative and consider different valid interpretations - not just the most obvious one."""


# ── Metric class ────────────────────────────────────────────────────────────

class VLMEntropy(Metric):
    """VLM interpretation entropy metric."""

    name = "vlm_entropy"
    needs_vlm = True
    needs_obs = True  # needs a scene image

    def __init__(
        self,
        k_samples: int = 15,
        temperature: float = 1.0,
        vlm_model: str = "YOUR_VLM_MODEL",
        cluster_threshold: float = 0.5,
    ):
        self.k_samples = k_samples
        self.temperature = temperature
        self.vlm_model = vlm_model
        self.cluster_threshold = cluster_threshold
        self._client = None

    def setup(self, **kwargs):
        import os
        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://YOUR_VLM_ENDPOINT/v1",
            api_key=os.environ.get("VLM_API_KEY", ""),
        )

    def teardown(self):
        self._client = None

    def _get_image_b64(self, obs: dict | None, image: np.ndarray | None) -> str:
        """Extract and encode scene image."""
        import base64
        import io
        from PIL import Image

        img = image
        if img is None and obs is not None:
            img = obs.get("observation/exterior_image_1_left")
        if img is None:
            raise ValueError("VLMEntropy requires a scene image (obs= or image=)")

        pil = Image.fromarray(img.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def compute(self, instruction, *, obs=None, image=None, **kwargs) -> MetricResult:
        if self._client is None:
            self.setup()

        scene_b64 = self._get_image_b64(obs, image)

        # Sample K interpretations
        interpretations: list[str] = []
        for i in range(self.k_samples):
            try:
                from vlm_orchestrator.vlm import chat_create
                resp = chat_create(
                    self._client,
                    model=self.vlm_model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{scene_b64}"}},
                            {"type": "text",
                             "text": INTERPRETATION_PROMPT.format(instruction=instruction)},
                        ],
                    }],
                    max_tokens=200,
                    temperature=self.temperature,
                )
                interpretations.append(resp.choices[0].message.content.strip())
            except Exception as e:
                # Log and continue — partial data is fine
                import logging

                logging.getLogger(__name__).warning(f"VLM call {i} failed: {e}")
                time.sleep(2)

        if not interpretations:
            return MetricResult(name=self.name, score=0.0, detail={"error": "no interpretations"})

        labels = cluster_by_text_similarity(interpretations, self.cluster_threshold)
        n_clusters = len(set(labels))
        entropy = compute_entropy(labels)
        norm_entropy = compute_normalized_entropy(labels)

        return MetricResult(
            name=self.name,
            score=entropy,
            detail={
                "n_clusters": n_clusters,
                "normalized_entropy": norm_entropy,
                "k_samples": len(interpretations),
                "cluster_labels": labels,
                "interpretations": interpretations,
            },
        )
