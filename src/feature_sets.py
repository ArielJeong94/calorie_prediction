
"""Convenience accessors for notebook-faithful feature sets."""
from __future__ import annotations

from .config import FEATURE_SETS, TARGET_COL, resolve_feature_set


def get_feature_set(name: str, include_target: bool = False) -> list[str]:
    cols = resolve_feature_set(name)
    if include_target and TARGET_COL not in cols:
        cols = cols + [TARGET_COL]
    return cols


def available_feature_sets() -> list[str]:
    return list(FEATURE_SETS.keys())
