from __future__ import annotations

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


def test_health_endpoint(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_predict_success(api_client) -> None:
    payload = _build_payload()
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["binary_decision"] in (0, 1)


def test_predict_validation_error(api_client) -> None:
    payload = _build_payload()
    payload["features"]["CODE_GENDER"] = "Invalid"
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422
