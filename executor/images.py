"""Game frames as JPEG, and as an image part of an LLM message."""
from __future__ import annotations

import base64
import io


def jpeg(pov, size=(640, 360), q=80) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    im = Image.fromarray(pov)
    if im.size != size:
        im = im.resize(size)
    im.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def img(b: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
