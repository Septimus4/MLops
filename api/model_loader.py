"""Utilities for loading the production scoring model."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import xgboost as xgb

from .artifacts import get_feature_list
from .config import get_settings


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing the loaded model artifact."""

    model_name: str
    model_version: str
    model_path: Path
    threshold: float | None
    raw_metadata: Dict[str, Any]

    @property
    def feature_count(self) -> int:
        return len(get_feature_list())


class ModelLoadError(RuntimeError):
    """Raised when the model artifact cannot be loaded."""


def _resolve_model_from_registry(settings) -> Tuple[Path, Dict[str, Any]]:
    registry_path = Path(settings.model_registry_path) / settings.registry_file
    if not registry_path.exists():
        raise ModelLoadError(f"Model registry file not found: {registry_path}")

    registry_payload = json.loads(registry_path.read_text())
    models = registry_payload.get(settings.model_name)
    if not models:
        raise ModelLoadError(f"Model {settings.model_name} not found in registry")

    if settings.model_version:
        candidate = next(
            (m for m in models if m.get("version") == settings.model_version), None
        )
        if not candidate:
            raise ModelLoadError(
                f"Version {settings.model_version} not available for model {settings.model_name}"
            )
    else:
        candidate = sorted(
            models, key=lambda item: item.get("registered_at", ""), reverse=True
        )[0]

    model_path = Path(candidate.get("model_path", "")).expanduser()
    if not model_path.is_absolute():
        candidate_path = (registry_path.parent / model_path).resolve()
        if candidate_path.exists():
            model_path = candidate_path
        else:
            # Handle registry entries that already include the registry directory
            project_relative = (
                Path(settings.model_registry_path).parent / model_path
            ).resolve()
            model_path = project_relative
    return model_path, candidate


def _load_booster(model_path: Path) -> xgb.Booster:
    if not model_path.exists():
        raise ModelLoadError(f"Model artifact not found at {model_path}")
    with model_path.open("rb") as fh:
        artifact = pickle.load(fh)
    if not isinstance(artifact, xgb.Booster):
        raise ModelLoadError(f"Expected xgb.Booster, received {type(artifact)!r}")
    return artifact


@lru_cache(maxsize=1)
def get_model() -> Tuple[xgb.Booster, ModelMetadata]:
    """Load and memoise the production model along with metadata."""

    settings = get_settings()

    if settings.model_path:
        model_path = Path(settings.model_path)
        raw_metadata: Dict[str, Any] = {
            "model_name": settings.model_name,
            "version": settings.model_version or model_path.stem,
        }
    else:
        model_path, raw_metadata = _resolve_model_from_registry(settings)

    booster = _load_booster(model_path)
    threshold = None
    metadata_info = raw_metadata.get("metadata") or {}
    if "threshold" in metadata_info:
        threshold = float(metadata_info["threshold"])

    metadata = ModelMetadata(
        model_name=raw_metadata.get("model_name", settings.model_name),
        model_version=raw_metadata.get("version", settings.model_version or "unknown"),
        model_path=model_path,
        threshold=threshold,
        raw_metadata=raw_metadata,
    )

    return booster, metadata
