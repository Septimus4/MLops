"""Pydantic schemas for the scoring API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, model_validator


class InputPayload(BaseModel):
    """Inbound payload for scoring requests."""

    request_id: str = Field(default_factory=lambda: uuid4().hex, description="Idempotency token")
    applicant_id: Optional[int] = Field(default=None, description="Domain identifier of the applicant")
    features: Dict[str, Any] = Field(..., description="Key/value feature dictionary aligned with training schema")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _ensure_features_present(self) -> "InputPayload":  # noqa: D401
        """Ensure the features dictionary is not empty."""

        if not self.features:
            raise ValueError("features must contain at least one entry")
        return self


class PredictionResponse(BaseModel):
    """API response returned after scoring."""

    request_id: str = Field(..., description="Idempotency token echo")
    applicant_id: Optional[int] = Field(default=None)
    model_name: str
    model_version: str
    score: float = Field(..., ge=0.0, le=1.0)
    binary_decision: int = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0.0, le=1.0)
    inference_ms: int = Field(..., ge=0)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    request_id: str
    message: str


class MetricsResponse(BaseModel):
    """Aggregated operational metrics."""

    window_minutes: int
    request_count: int
    error_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    mean_score: float
    drift_alerts: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Payload returned by the health endpoint."""

    status: str
    model_name: str
    model_version: str
    model_path: str
    feature_count: int
    threshold: Optional[float]
    environment: str
