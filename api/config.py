"""Application configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field, ConfigDict, field_validator


class Settings(BaseModel):
    """Runtime configuration derived from environment variables and defaults."""

    environment: str = Field(default=os.getenv("APP_ENV", "dev"))
    model_name: str = Field(default=os.getenv("MODEL_NAME", "home_credit_model"))
    model_version: str | None = Field(
        default=os.getenv("MODEL_VERSION", "v_20251106_103251")
    )
    model_path: str | None = Field(default=os.getenv("MODEL_PATH"))
    model_registry_path: str = Field(
        default=os.getenv("MODEL_REGISTRY_PATH", "model_registry")
    )
    registry_file: str = Field(default=os.getenv("REGISTRY_FILE", "registry.json"))

    artifacts_dir: str = Field(default=os.getenv("ARTIFACTS_DIR", "artifacts"))
    feature_defaults_path: str = Field(
        default=os.getenv("FEATURE_DEFAULTS_PATH", "artifacts/feature_defaults.json")
    )
    categorical_mappings_path: str = Field(
        default=os.getenv(
            "CATEGORICAL_MAPPINGS_PATH", "artifacts/categorical_mappings.json"
        )
    )
    feature_list_path: str = Field(
        default=os.getenv("FEATURE_LIST_PATH", "artifacts/feature_list.json")
    )

    log_dir: str = Field(default=os.getenv("LOG_DIR", "data/logs"))
    metrics_dir: str = Field(default=os.getenv("METRICS_DIR", "data/metrics"))
    reference_dir: str = Field(default=os.getenv("REFERENCE_DIR", "data/reference"))

    log_retention_days: int = Field(default=int(os.getenv("LOG_RETENTION_DAYS", "30")))
    monitor_window_minutes: int = Field(
        default=int(os.getenv("MONITOR_WINDOW_MINUTES", "1440"))
    )
    request_timeout_seconds: float = Field(
        default=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
    )

    default_threshold: float = Field(
        default=float(os.getenv("DEFAULT_THRESHOLD", "0.5"))
    )
    monitor_features: Sequence[str] = Field(
        default=(
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "CNT_CHILDREN",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "GOODS_CREDIT_RATIO",
            "AGE",
            "YEARS_EMPLOYED",
            "YEARS_REGISTRATION",
            "YEARS_ID_PUBLISH",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "CODE_GENDER",
            "NAME_CONTRACT_TYPE",
        )
    )
    monitor_min_sample_size: int = Field(
        default=int(os.getenv("MONITOR_MIN_SAMPLE_SIZE", "25"))
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("default_threshold")
    @classmethod
    def _threshold_bounds(cls, value: float) -> float:  # noqa: D401
        """Ensure the default threshold stays within [0, 1]."""

        if not 0.0 <= value <= 1.0:
            raise ValueError("DEFAULT_THRESHOLD must be within [0, 1]")
        return value

    def ensure_runtime_dirs(self) -> None:
        """Create directories that must exist at runtime, such as log storage."""

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metrics_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reference_dir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
