"""Domains: everything specific to one environment (world, System 1 agent, log reader); domains/base.py says what
each provides to the method. load() returns the domain object of domains/<name>/domain.py."""
from __future__ import annotations

import importlib
import os


def load(name: str | None = None):
    """The domain named `name` (default $MM_DOMAIN, else minecraft); its folders must be on the import path
    (paths.py adds those of $MM_DOMAIN)."""
    return importlib.import_module(f"domains.{name or os.environ.get('MM_DOMAIN', 'minecraft')}.domain").DOMAIN
