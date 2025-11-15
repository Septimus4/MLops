"""
Tests for feature processor module.
"""


import numpy as np
import pytest

from src.service.feature_processor import build_feature_vector, validate_features


def test_build_feature_vector_with_all_features():
    """Test building feature vector with all features provided."""
    feature_names = ["EXT_SOURCE_1", "EXT_SOURCE_2", "AMT_CREDIT"]
    input_features = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "AMT_CREDIT": 600000.0,
    }

    vector, filled = build_feature_vector(input_features, feature_names)

    assert isinstance(vector, np.ndarray)
    assert len(vector) == 3
    assert vector[0] == 0.5
    assert vector[1] == 0.6
    assert vector[2] == 600000.0

    assert len(filled) == 3
    assert filled["EXT_SOURCE_1"] == 0.5


def test_build_feature_vector_with_missing_features():
    """Test building feature vector with missing features (should use defaults)."""
    feature_names = ["EXT_SOURCE_1", "EXT_SOURCE_2", "AMT_CREDIT"]
    input_features = {
        "EXT_SOURCE_1": 0.5,
        # EXT_SOURCE_2 is missing
        # AMT_CREDIT is missing
    }

    vector, filled = build_feature_vector(input_features, feature_names)

    assert isinstance(vector, np.ndarray)
    assert len(vector) == 3
    assert vector[0] == 0.5
    # Other values should be defaults
    assert vector[1] > 0  # Should have a default value
    assert vector[2] > 0  # Should have a default value

    assert len(filled) == 3


def test_build_feature_vector_ordering():
    """Test that feature vector maintains correct ordering."""
    feature_names = ["A", "B", "C"]
    input_features = {
        "C": 3.0,
        "A": 1.0,
        "B": 2.0,
    }

    vector, filled = build_feature_vector(input_features, feature_names)

    # Should be ordered as A, B, C
    assert vector[0] == 1.0
    assert vector[1] == 2.0
    assert vector[2] == 3.0


def test_build_feature_vector_type_conversion():
    """Test type conversion in feature vector building."""
    feature_names = ["DAYS_BIRTH", "AMT_CREDIT"]
    input_features = {
        "DAYS_BIRTH": "-14000",  # String
        "AMT_CREDIT": "600000.5",  # String
    }

    vector, filled = build_feature_vector(input_features, feature_names)

    assert isinstance(vector, np.ndarray)
    assert vector[0] == -14000.0
    assert vector[1] == 600000.5


def test_build_feature_vector_invalid_value():
    """Test that invalid values raise ValueError."""
    feature_names = ["EXT_SOURCE_1"]
    input_features = {
        "EXT_SOURCE_1": "invalid",
    }

    with pytest.raises(ValueError, match="Cannot convert"):
        build_feature_vector(input_features, feature_names)


def test_validate_features_valid():
    """Test validation with valid features."""
    features = {
        "EXT_SOURCE_1": 0.5,
        "AMT_CREDIT": 600000,
    }

    # Should not raise
    validate_features(features)


def test_validate_features_empty():
    """Test validation with empty features (should be valid)."""
    features = {}

    # Should not raise
    validate_features(features)


def test_validate_features_not_dict():
    """Test validation with non-dict input."""
    features = ["not", "a", "dict"]

    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_features(features)


def test_validate_features_none_value():
    """Test validation with None value."""
    features = {
        "EXT_SOURCE_1": None,
    }

    with pytest.raises(ValueError, match="cannot be None"):
        validate_features(features)


def test_validate_features_non_numeric():
    """Test validation with non-numeric value."""
    features = {
        "EXT_SOURCE_1": "not_a_number",
    }

    with pytest.raises(ValueError, match="not numeric"):
        validate_features(features)
