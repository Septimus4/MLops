from __future__ import annotations

from pathlib import Path

from api.logging_utils import aggregate_metrics, log_prediction
from api.model_loader import get_model
from api.schemas import InputPayload


def test_log_prediction_creates_jsonl(tmp_path: Path) -> None:
    payload = InputPayload(
        features={
            "AMT_INCOME_TOTAL": 100000,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "F",
        }
    )
    response = {
        "request_id": payload.request_id,
        "applicant_id": None,
        "model_name": "home_credit_model",
        "model_version": "test",
        "score": 0.5,
        "binary_decision": 1,
        "threshold": 0.5,
        "inference_ms": 12,
    }

    _, metadata = get_model()
    log_prediction(
        payload=payload,
        response_body=response,
        processed_features={
            "AMT_INCOME_TOTAL": 100000.0,
            "NAME_CONTRACT_TYPE": 0.0,
            "CODE_GENDER": 0.0,
        },
        monitor_features={"AMT_INCOME_TOTAL": 100000.0},
        status="ok",
        latency_ms=12.0,
        metadata=metadata,
    )

    from api.config import get_settings

    log_dir = Path(get_settings().log_dir)
    files = list(log_dir.glob("*.jsonl"))
    assert files

    metrics = aggregate_metrics(60)
    assert metrics["request_count"] >= 1
