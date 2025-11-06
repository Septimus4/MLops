"""Model inference helpers."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .artifacts import get_categorical_mappings, get_feature_defaults, get_feature_list
from .config import get_settings
from .schemas import InputPayload


class FeatureValidationError(ValueError):
    """Raised when the incoming payload does not respect the training schema."""


def _coerce_numeric(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return 0.0
        try:
            return float(cleaned)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise FeatureValidationError(
                f"Cannot coerce value '{value}' to float"
            ) from exc
    raise FeatureValidationError(f"Unsupported value type: {type(value)!r}")


def _apply_categorical_mapping(
    feature: str, raw_value: object, mappings: Dict[str, Dict[str, int]]
) -> float:
    mapping = mappings.get(feature)
    if mapping is None:
        return _coerce_numeric(raw_value)

    if isinstance(raw_value, str):
        key = raw_value.strip()
        if key not in mapping:
            raise FeatureValidationError(
                f"Unknown category '{raw_value}' for feature '{feature}'"
            )
        return float(mapping[key])

    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        return float(raw_value)

    raise FeatureValidationError(
        f"Unsupported categorical value type for {feature}: {type(raw_value)!r}"
    )


def _recompute_derived(features: Dict[str, float]) -> None:
    income = features.get("AMT_INCOME_TOTAL", 0.0)
    credit = features.get("AMT_CREDIT", 0.0)
    annuity = features.get("AMT_ANNUITY", 0.0)
    goods = features.get("AMT_GOODS_PRICE", 0.0)

    if income:
        features["CREDIT_INCOME_RATIO"] = float(credit / income)
        features["ANNUITY_INCOME_RATIO"] = float(annuity / income)
    else:
        features["CREDIT_INCOME_RATIO"] = 0.0
        features["ANNUITY_INCOME_RATIO"] = 0.0

    if credit:
        features["GOODS_CREDIT_RATIO"] = float(goods / credit)
    else:
        features["GOODS_CREDIT_RATIO"] = 0.0

    # Derived year features from absolute day counters
    days_birth = features.get("DAYS_BIRTH")
    if days_birth is not None:
        features["AGE"] = float(abs(days_birth) / 365.0)

    days_employed = features.get("DAYS_EMPLOYED")
    if days_employed is not None:
        features["YEARS_EMPLOYED"] = float(abs(days_employed) / 365.0)

    days_registration = features.get("DAYS_REGISTRATION")
    if days_registration is not None:
        features["YEARS_REGISTRATION"] = float(abs(days_registration) / 365.0)

    days_id = features.get("DAYS_ID_PUBLISH")
    if days_id is not None:
        features["YEARS_ID_PUBLISH"] = float(abs(days_id) / 365.0)


def prepare_features(payload: InputPayload) -> OrderedDict[str, float]:
    """Merge payload features with defaults, applying categorical mappings."""

    defaults = get_feature_defaults()
    mappings = get_categorical_mappings()
    feature_order = get_feature_list()

    prepared: Dict[str, float] = {
        feature: float(defaults.get(feature, 0.0)) for feature in feature_order
    }

    for feature, raw_value in payload.features.items():
        if feature not in prepared:
            raise FeatureValidationError(
                f"Feature '{feature}' not recognised by the model schema"
            )
        prepared[feature] = _apply_categorical_mapping(feature, raw_value, mappings)

    _recompute_derived(prepared)

    # Replace NaNs with zeros to avoid xgboost errors.
    for key, value in prepared.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            prepared[key] = 0.0

    ordered = OrderedDict(
        (feature, float(prepared[feature])) for feature in feature_order
    )
    return ordered


def predict_one(
    payload: InputPayload, model: xgb.Booster
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """Generate a prediction and return processed feature values used for scoring."""

    ordered_features = prepare_features(payload)
    frame = pd.DataFrame([ordered_features])
    matrix = xgb.DMatrix(frame, feature_names=list(ordered_features.keys()))
    score = float(model.predict(matrix)[0])

    settings = get_settings()
    monitor_feats = {
        name: ordered_features[name]
        for name in settings.monitor_features
        if name in ordered_features
    }
    return ordered_features, score, monitor_feats
