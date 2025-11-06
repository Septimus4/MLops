"""Quick smoke test that sends a sample payload to the scoring API."""

from __future__ import annotations

import argparse
import json
from uuid import uuid4

import requests

from api.artifacts import get_feature_defaults
from api.schemas import InputPayload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a smoke-test request to the scoring API"
    )
    parser.add_argument(
        "--url", default="http://localhost:8080/predict", help="Prediction endpoint URL"
    )
    args = parser.parse_args()

    defaults = get_feature_defaults()
    sample_features = {
        "AMT_INCOME_TOTAL": defaults["AMT_INCOME_TOTAL"],
        "AMT_CREDIT": defaults["AMT_CREDIT"],
        "AMT_ANNUITY": defaults["AMT_ANNUITY"],
        "CNT_CHILDREN": defaults["CNT_CHILDREN"],
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "DAYS_BIRTH": defaults["DAYS_BIRTH"],
        "DAYS_EMPLOYED": defaults["DAYS_EMPLOYED"],
        "DAYS_REGISTRATION": defaults["DAYS_REGISTRATION"],
    }

    payload = InputPayload(request_id=uuid4().hex, features=sample_features)
    response = requests.post(args.url, json=json.loads(payload.model_dump_json()))
    response.raise_for_status()
    print("Response:", response.json())


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
