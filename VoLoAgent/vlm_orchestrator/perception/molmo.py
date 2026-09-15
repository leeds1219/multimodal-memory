# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 pointing client — single-target API for the grasp / place tools.

Connects to an OpenAI-compatible server running **Molmo2-8B**
(default: ``allenai/Molmo2-8B`` at ``http://127.0.0.1:8122/v1``).
Sends the standard "Point at <target>" prompt and parses Molmo2's
``<points coords="...">`` XML response into a single normalised
``(x_norm, y_norm)`` pair in [0, 1].

Design notes:

- Molmo2 uses a 0-1000 normalised coordinate space; we convert to 0-1.
- **Molmo1 (Molmo-7B-D-0924) is intentionally not supported.**  In our
  comparison on Isaac-Sim renders Molmo1 was worse than GDinoV2 —
  pointing collapsed to a y≈0.67 cluster with no spatial
  discrimination.  See ``debug_perception/`` artefacts.  If a Molmo1
  response (``<point x= y=>``) is ever observed here, the parser will
  reject it.
- A "Point at" prompt typically returns 1–N points (Molmo2 points to
  every instance it finds).  We return the FIRST point and let the
  caller decide how to disambiguate via a more specific prompt.
- No silent fallback: if Molmo2's response can't be parsed, raise
  :class:`MolmoPointError` (caller decides whether to retry / abort).
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass

import numpy as np
import PIL.Image
import requests

logger = logging.getLogger(__name__)


# Defaults — overridable per-call.
DEFAULT_BASE_URL = "http://127.0.0.1:8122/v1"
DEFAULT_MODEL = "allenai/Molmo2-8B"
DEFAULT_TIMEOUT_S = 60.0


class MolmoPointError(RuntimeError):
    """Raised when Molmo's response can't be parsed into a point."""


@dataclass(frozen=True)
class MolmoPoint:
    """One Molmo pointing result, all coords in [0, 1]."""
    x_norm: float
    y_norm: float
    raw_text: str        # full model output (for debug logs)


def _image_to_data_url(image_rgb: np.ndarray) -> str:
    """Encode an RGB ndarray as a ``data:image/jpeg;base64,...`` URL."""
    pil = PIL.Image.fromarray(np.asarray(image_rgb).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_first_point(text: str) -> tuple[float, float] | None:
    """Parse the first ``<points coords>`` from Molmo2 output.

    Returns ``(x_norm, y_norm)`` in [0, 1], or ``None`` if no point
    found.  We only accept Molmo2's ``<points coords="...">`` format
    (0-1000 normalisation).  The Molmo1 ``<point x= y=>`` (0-100) and
    legacy ``<points x1= y1=>`` formats are intentionally rejected —
    Molmo1 is unusable on Isaac-Sim renders (see
    ``debug_perception/`` for the empirical comparison).
    """
    # Molmo2: <points coords="type obj_idx x y ..."> — triplets after type.
    coords_match = re.search(
        r'<points\s+coords\s*=\s*["\']([^"\']+)["\']',
        text, flags=re.IGNORECASE,
    )
    if coords_match:
        nums = [float(n) for n in coords_match.group(1).split()]
        # Skip first number (type indicator), then triplets (obj_idx, x, y).
        if len(nums) >= 4:
            x, y = nums[2], nums[3]
            return (x / 1000.0, y / 1000.0)

    return None


def point_at(
    image_rgb: np.ndarray,
    target_phrase: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    session: requests.Session | None = None,
) -> MolmoPoint:
    """Ask Molmo2 to point at ``target_phrase`` in ``image_rgb``.

    Returns the FIRST point Molmo emits, normalised to [0, 1].  Raises
    :class:`MolmoPointError` on transport failure, empty response, or
    unparseable text — no silent fallback to a default coordinate.

    The ``target_phrase`` can be a single noun phrase ("orange fruit")
    or a free-form spatial expression ("empty spot on the rack next to
    the bottles") — Molmo handles both.
    """
    if session is None:
        session = requests.Session()

    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Point at the destination: {target_phrase}"},
                {"type": "image_url",
                 "image_url": {"url": _image_to_data_url(image_rgb)}},
            ],
        }],
        "max_tokens": 256,
        "temperature": 0.0,
        "stop": ["<|endoftext|>"],
    }

    try:
        resp = session.post(
            chat_url, json=payload, headers=headers, timeout=timeout_s,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise MolmoPointError(
            f"Molmo HTTP call failed for {target_phrase!r} "
            f"(url={chat_url}): {e}"
        ) from e

    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, ValueError, TypeError) as e:
        raise MolmoPointError(
            f"Molmo response missing choices[0].message.content "
            f"for {target_phrase!r}: {e}; body={resp.text[:200]!r}"
        ) from e

    if not text.strip():
        raise MolmoPointError(
            f"Molmo returned empty response for {target_phrase!r}"
        )

    parsed = _parse_first_point(text)
    if parsed is None:
        raise MolmoPointError(
            f"Could not parse a point from Molmo output for "
            f"{target_phrase!r}: raw={text!r}"
        )

    x_norm, y_norm = parsed
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        raise MolmoPointError(
            f"Molmo point out of [0, 1] for {target_phrase!r}: "
            f"({x_norm}, {y_norm}); raw={text!r}"
        )

    logger.info(
        f"  Molmo pointed at {target_phrase!r}: "
        f"({x_norm:.3f}, {y_norm:.3f})"
    )
    return MolmoPoint(x_norm=x_norm, y_norm=y_norm, raw_text=text)
