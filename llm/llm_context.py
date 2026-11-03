"""The file the LLM call context is written to ($LLM_CTX_FILE; read by client.context() in every process of the
episode). Kept apart from client.py, which loads its config when imported."""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_ctx(ctx_file: str | os.PathLike, ctx: dict) -> None:
    tmp = Path(str(ctx_file) + ".tmp")
    tmp.write_text(json.dumps(dict(ctx)))
    os.replace(tmp, ctx_file)
