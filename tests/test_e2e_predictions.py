from __future__ import annotations

import pytest

from api.artifacts import get_feature_defaults


def _baseline_features() -> dict[str, float | int | str]:
    defaults = get_feature_defaults()
    return {
        "AMT_INCOME_TOTAL": defaults["AMT_INCOME_TOTAL"],
        "AMT_CREDIT": defaults["AMT_CREDIT"],
        "AMT_ANNUITY": defaults["AMT_ANNUITY"],
        "AMT_GOODS_PRICE": defaults["AMT_GOODS_PRICE"],
        "CNT_CHILDREN": int(defaults["CNT_CHILDREN"]),
        "DAYS_BIRTH": int(defaults["DAYS_BIRTH"]),
        "DAYS_EMPLOYED": int(defaults["DAYS_EMPLOYED"]),
        "DAYS_REGISTRATION": int(defaults["DAYS_REGISTRATION"]),
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "N",
    }


@pytest.mark.anyio("asyncio")
async def test_risky_credit_request_declines(api_client) -> None:
    payload = {"features": _baseline_features()}
    baseline_response = await api_client.post("/predict", json=payload)
    assert baseline_response.status_code == 200
    baseline_body = baseline_response.json()

    assert baseline_body["binary_decision"] == int(
        baseline_body["score"] <= baseline_body["threshold"]
    )
    assert baseline_body["binary_decision"] == 1  # healthy applicant should be approved

    risky_request = payload["features"].copy()
    risky_request["AMT_CREDIT"] = risky_request["AMT_CREDIT"] * 1.4

    risky_response = await api_client.post("/predict", json={"features": risky_request})
    assert risky_response.status_code == 200
    risky_body = risky_response.json()

    assert risky_body["score"] > baseline_body["score"]
    assert risky_body["binary_decision"] == int(
        risky_body["score"] <= risky_body["threshold"]
    )
    assert (
        risky_body["binary_decision"] == 0
    )  # elevated credit request should be declined
