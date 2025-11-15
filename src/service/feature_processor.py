"""
Feature processing module.
Handles conversion of raw input features to model-ready feature vectors.
"""

from typing import Any, Dict, Tuple

import numpy as np

from src.training.feature_config import FEATURE_DTYPES, FEATURE_DEFAULTS


def build_feature_vector(
    input_features: Dict[str, Any],
    feature_names: list[str],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build feature vector from input features.

    Args:
        input_features: Dictionary of input feature values
        feature_names: List of feature names in the correct order

    Returns:
        Tuple of (feature_vector, filled_features_dict)
        - feature_vector: numpy array ready for model prediction
        - filled_features_dict: dictionary with all features filled (including defaults)
    """
    filled_features = {}
    feature_vector = []

    for feature_name in feature_names:
        # Get value from input or use default
        if feature_name in input_features:
            raw_value = input_features[feature_name]
        else:
            raw_value = FEATURE_DEFAULTS.get(feature_name, 0.0)

        # Convert to appropriate type
        dtype = FEATURE_DTYPES.get(feature_name, "float")

        try:
            if dtype == "int":
                value = int(float(raw_value))
            elif dtype == "float":
                value = float(raw_value)
            elif dtype == "category":
                # For categories, try to convert to numeric
                # In real scenario, you'd have a mapping
                value = float(raw_value)
            else:
                value = float(raw_value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert feature '{feature_name}' value '{raw_value}' to {dtype}: {e}"
            )

        filled_features[feature_name] = float(value)
        feature_vector.append(float(value))

    return np.array(feature_vector), filled_features


def validate_features(input_features: Dict[str, Any]) -> None:
    """
    Validate input features.

    Args:
        input_features: Dictionary of input feature values

    Raises:
        ValueError: If features are invalid
    """
    if not isinstance(input_features, dict):
        raise ValueError("Features must be a dictionary")

    if not input_features:
        # Empty features are allowed - will use all defaults
        return

    # Check keys are known
    unknown_keys = [k for k in input_features.keys() if k not in FEATURE_DTYPES]
    if unknown_keys:
        raise ValueError(
            "Unknown feature keys provided: "
            + ", ".join(unknown_keys)
            + ". Allowed features: "
            + ", ".join(sorted(FEATURE_DTYPES.keys()))
        )

    # Check for obviously invalid values
    for key, value in input_features.items():
        if value is None:
            raise ValueError(f"Feature '{key}' cannot be None")

        # Try to convert to number to validate
        try:
            float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{key}' value '{value}' is not numeric")
