"""FastAPI application exposing the credit scoring model."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .inference import FeatureValidationError, predict_one
from .logging_utils import aggregate_metrics, log_prediction
from .model_loader import get_model
from .schemas import ErrorResponse, HealthResponse, InputPayload, MetricsResponse, PredictionResponse

logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title="Home Credit Scoring API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service readiness information."""

    model, metadata = get_model()
    return HealthResponse(
        status="ok",
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        model_path=str(metadata.model_path),
        feature_count=metadata.feature_count,
        threshold=metadata.threshold,
        environment=settings.environment,
    )


@app.post("/predict", response_model=PredictionResponse, responses={422: {"model": ErrorResponse}})
def predict(payload: InputPayload) -> PredictionResponse:
    """Score a single applicant."""

    booster, metadata = get_model()
    start = perf_counter()
    try:
        processed_features, score, monitor_features = predict_one(payload, booster)
    except FeatureValidationError as exc:
        logger.warning("Validation error for request %s: %s", payload.request_id, exc)
        log_prediction(
            payload=payload,
            response_body={
                "request_id": payload.request_id,
                "applicant_id": payload.applicant_id,
                "error": str(exc),
            },
            processed_features={},
            monitor_features={},
            status="error",
            latency_ms=float((perf_counter() - start) * 1000),
            metadata=metadata,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected inference failure")
        log_prediction(
            payload=payload,
            response_body={
                "request_id": payload.request_id,
                "applicant_id": payload.applicant_id,
                "error": "Unexpected inference failure",
            },
            processed_features={},
            monitor_features={},
            status="error",
            latency_ms=float((perf_counter() - start) * 1000),
            metadata=metadata,
        )
        raise HTTPException(status_code=500, detail="Unexpected inference failure") from exc

    elapsed_ms = int((perf_counter() - start) * 1000)
    threshold = metadata.threshold or settings.default_threshold
    binary = int(score >= threshold)

    response_dict: dict[str, Any] = {
        "request_id": payload.request_id,
        "applicant_id": payload.applicant_id,
        "model_name": metadata.model_name,
        "model_version": metadata.model_version,
        "score": score,
        "binary_decision": binary,
        "threshold": float(threshold),
        "inference_ms": elapsed_ms,
    }

    log_prediction(
        payload=payload,
        response_body=response_dict,
        processed_features=dict(processed_features),
        monitor_features=monitor_features,
        status="ok",
        latency_ms=float(elapsed_ms),
        metadata=metadata,
    )

    return PredictionResponse(**response_dict)


@app.get("/metrics", response_model=MetricsResponse)
def metrics(window_minutes: int | None = None) -> MetricsResponse:
    """Return aggregated operational metrics for the scoring service."""

    lookback = window_minutes or settings.monitor_window_minutes
    aggregated = aggregate_metrics(lookback)
    return MetricsResponse(**aggregated)
