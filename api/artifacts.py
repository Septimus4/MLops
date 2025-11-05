"""Helpers for loading static inference artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from .config import get_settings


class ArtifactLoadError(RuntimeError):
    """Raised when a required artifact cannot be loaded."""


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise ArtifactLoadError(f"Artifact not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ArtifactLoadError(f"Invalid JSON in artifact {path}: {exc}") from exc


@lru_cache(maxsize=1)
def get_feature_defaults() -> Dict[str, float]:
    """Median feature defaults used to backfill missing inputs."""

    settings = get_settings()
    path = Path(settings.feature_defaults_path)
    return {key: float(value) for key, value in _load_json(path).items()}


@lru_cache(maxsize=1)
def get_categorical_mappings() -> Dict[str, Dict[str, int]]:
    """Label-encoder mappings for categorical features."""

    settings = get_settings()
    path = Path(settings.categorical_mappings_path)
    raw = _load_json(path)
    return {column: {label: int(index) for label, index in mapping.items()} for column, mapping in raw.items()}


@lru_cache(maxsize=1)
def get_feature_list() -> List[str]:
    """Feature ordering required by the production model artifact."""

    settings = get_settings()
    path = Path(settings.feature_list_path)
    data = _load_json(path)
    if not isinstance(data, list):  # pragma: no cover - sanity check
        raise ArtifactLoadError(f"Feature list must be a list, got {type(data)!r}")
    return [str(item) for item in data]
