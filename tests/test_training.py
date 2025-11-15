"""
Tests for training module.
"""

import os
import tempfile

import joblib
import pandas as pd
import pytest

from src.training.compute_baseline_stats import compute_baseline_statistics
from src.training.train_model import main as train_main


@pytest.fixture
def synthetic_data():
    """Create synthetic training data for testing."""
    # Create small synthetic dataset
    data = {
        "SK_ID_CURR": list(range(100)),
        "TARGET": [0] * 80 + [1] * 20,  # 20% default rate
        "EXT_SOURCE_1": [0.5 + i * 0.01 for i in range(100)],
        "EXT_SOURCE_2": [0.4 + i * 0.01 for i in range(100)],
        "AMT_CREDIT": [600000 + i * 1000 for i in range(100)],
        "AMT_ANNUITY": [27000 + i * 100 for i in range(100)],
        "DAYS_BIRTH": [-14000 - i * 10 for i in range(100)],
    }

    df = pd.DataFrame(data)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        df.to_csv(csv_path, index=False)

    yield csv_path

    # Cleanup
    if os.path.exists(csv_path):
        os.unlink(csv_path)


@pytest.fixture
def temp_model_path():
    """Create temporary path for model."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name

    yield model_path

    # Cleanup
    if os.path.exists(model_path):
        os.unlink(model_path)


@pytest.fixture
def temp_baseline_path():
    """Create temporary path for baseline stats."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        baseline_path = f.name

    yield baseline_path

    # Cleanup
    if os.path.exists(baseline_path):
        os.unlink(baseline_path)


def test_train_model_creates_artifact(synthetic_data, temp_model_path):
    """Test that training creates a model artifact."""
    artifact = train_main(data_path=synthetic_data, output_path=temp_model_path)

    # Check artifact structure
    assert "model" in artifact
    assert "feature_names" in artifact
    assert "model_version" in artifact

    # Check that model was saved
    assert os.path.exists(temp_model_path)

    # Load and verify
    loaded = joblib.load(temp_model_path)
    assert "model" in loaded
    assert len(loaded["feature_names"]) > 0


def test_train_model_can_predict(synthetic_data, temp_model_path):
    """Test that trained model can make predictions."""
    artifact = train_main(data_path=synthetic_data, output_path=temp_model_path)

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    # Create test input
    import numpy as np

    test_input = np.array([[0.5, 0.5, 600000, 27000, -14000]])

    # Should be able to predict
    if test_input.shape[1] != len(feature_names):
        # Adjust input to match number of features
        test_input = np.zeros((1, len(feature_names)))

    proba = model.predict_proba(test_input)

    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_compute_baseline_stats_creates_json(synthetic_data, temp_model_path, temp_baseline_path):
    """Test that baseline stats computation creates JSON file."""
    # First train model
    train_main(data_path=synthetic_data, output_path=temp_model_path)

    # Then compute baseline
    stats = compute_baseline_statistics(
        data_path=synthetic_data, model_path=temp_model_path, output_path=temp_baseline_path
    )

    # Check that file was created
    assert os.path.exists(temp_baseline_path)

    # Check structure
    assert isinstance(stats, dict)
    assert len(stats) > 0

    # Each feature should have mean and std
    for feature, stat in stats.items():
        assert "mean" in stat
        assert "std" in stat
        assert isinstance(stat["mean"], float)
        assert isinstance(stat["std"], float)


def test_compute_baseline_stats_values(synthetic_data, temp_model_path, temp_baseline_path):
    """Test that baseline stats have reasonable values."""
    # First train model
    train_main(data_path=synthetic_data, output_path=temp_model_path)

    # Then compute baseline
    stats = compute_baseline_statistics(
        data_path=synthetic_data, model_path=temp_model_path, output_path=temp_baseline_path
    )

    # Check that EXT_SOURCE_1 has expected mean (around 0.5 + 99*0.01/2 = 0.995)
    if "EXT_SOURCE_1" in stats:
        assert 0.9 < stats["EXT_SOURCE_1"]["mean"] < 1.1
        assert stats["EXT_SOURCE_1"]["std"] > 0


def test_train_model_missing_data_file(temp_model_path):
    """Test that training fails gracefully with missing data file."""
    with pytest.raises(SystemExit):
        train_main(data_path="/nonexistent/file.csv", output_path=temp_model_path)


def test_compute_baseline_missing_model(synthetic_data, temp_baseline_path):
    """Test that baseline computation fails gracefully with missing model."""
    with pytest.raises(SystemExit):
        compute_baseline_statistics(
            data_path=synthetic_data,
            model_path="/nonexistent/model.joblib",
            output_path=temp_baseline_path,
        )
