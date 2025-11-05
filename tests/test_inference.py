from __future__ import annotations

import pytest

from api.artifacts import get_feature_defaults
from api.inference import FeatureValidationError, predict_one
from api.model_loader import get_model
from api.schemas import InputPayload


@pytest.fixture
def sample_payload() -> InputPayload:
    defaults = get_feature_defaults()
    features = {
        "AMT_INCOME_TOTAL": defaults["AMT_INCOME_TOTAL"],
        "AMT_CREDIT": defaults["AMT_CREDIT"],
        "AMT_ANNUITY": defaults["AMT_ANNUITY"],
        "AMT_GOODS_PRICE": defaults["AMT_GOODS_PRICE"],
        "CNT_CHILDREN": defaults["CNT_CHILDREN"],
        "DAYS_BIRTH": defaults["DAYS_BIRTH"],
        "DAYS_EMPLOYED": defaults["DAYS_EMPLOYED"],
        "DAYS_REGISTRATION": defaults["DAYS_REGISTRATION"],
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
    }
    return InputPayload(features=features)


def test_predict_one_returns_probability(sample_payload: InputPayload) -> None:
    model, _ = get_model()
    processed, score, monitor = predict_one(sample_payload, model)
    assert 0.0 <= score <= 1.0
    assert len(processed) > 0
    assert monitor


def test_predict_one_invalid_categorical(sample_payload: InputPayload) -> None:
    sample_payload.features["CODE_GENDER"] = "Invalid"
    model, _ = get_model()
    with pytest.raises(FeatureValidationError):
        predict_one(sample_payload, model)
