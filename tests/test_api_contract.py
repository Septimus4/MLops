from __future__ import annotations

import pytest

from api.artifacts import get_feature_defaults


def _build_payload() -> dict:
    defaults = get_feature_defaults()
    return {
        "features": {
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
    }


@pytest.mark.anyio("asyncio")
async def test_health_endpoint(api_client) -> None:
    response = await api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert body["model_version"] == "v_20251106_103251"
    assert body["model_path"].endswith(
        "model_registry/home_credit_model_v_20251106_103251.pkl"
    )


@pytest.mark.anyio("asyncio")
async def test_predict_success(api_client) -> None:
    payload = _build_payload()
    response = await api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["binary_decision"] in (0, 1)
    assert body["binary_decision"] == int(body["score"] <= body["threshold"])


@pytest.mark.anyio("asyncio")
async def test_predict_validation_error(api_client) -> None:
    payload = _build_payload()
    payload["features"]["CODE_GENDER"] = "Invalid"
    response = await api_client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["request_id"]
    assert "Unknown category" in body["message"]
