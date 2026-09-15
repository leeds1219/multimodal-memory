# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for the place pipeline.

Placement reuses the grasp server's existing endpoints
(``/detect_and_segment`` for SAM3 / GDino+SAM2).  No new server surface
area — this module is a thin re-export so callers can build a client
from ``PLACE_SERVER_HOST`` / ``PLACE_SERVER_PORT`` env vars (which fall
through to ``GRASP_SERVER_*`` when unset).
"""

from __future__ import annotations

import os

from vlm_orchestrator.grasp.client import GraspClient

__all__ = ["PlaceClient", "default_place_url"]


def default_place_url() -> str:
    """Build the place server URL from env vars.

    ``PLACE_SERVER_HOST`` / ``PLACE_SERVER_PORT`` win when set; otherwise
    falls through to ``GRASP_SERVER_HOST`` / ``GRASP_SERVER_PORT``;
    final fallback is ``localhost:8003``.
    """
    host = (
        os.environ.get("PLACE_SERVER_HOST")
        or os.environ.get("GRASP_SERVER_HOST")
        or "localhost"
    )
    port = (
        os.environ.get("PLACE_SERVER_PORT")
        or os.environ.get("GRASP_SERVER_PORT")
        or "8003"
    )
    return f"http://{host}:{port}"


class PlaceClient(GraspClient):
    """HTTP client for the place pipeline.

    Inherits all endpoints from :class:`GraspClient`; only the default
    URL resolution differs (see :func:`default_place_url`).
    """

    def __init__(self, url: str | None = None, timeout: float | None = None):
        kwargs: dict = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        super().__init__(url=url or default_place_url(), **kwargs)
