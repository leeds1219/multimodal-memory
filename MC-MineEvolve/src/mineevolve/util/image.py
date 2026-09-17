"""POV / image utilities."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


def encode_pov_to_base64(pov: Any, format: str = "PNG") -> str:
    """Encode a HxWx3 uint8 numpy array as a base64 PNG string."""

    if not isinstance(pov, np.ndarray):
        return ""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return ""
    img = Image.fromarray(pov.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_pov(value: Any) -> np.ndarray | None:
    """Inverse of :func:`encode_pov_to_base64`, tolerant of the other wire formats.

    Accepts a base64 PNG/JPEG string, a nested list (``ndarray.tolist()``) or an
    ndarray and returns a HxWx3 uint8 array, or ``None`` if there is no frame.
    """

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.uint8, copy=False)
    if isinstance(value, str):
        if not value:
            return None
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8)
