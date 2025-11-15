"""
Tests for FastAPI endpoints.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.service.main import app as _app_reference


@pytest.fixture
def mock_model():
    """Mock the model loader to avoid needing actual model file."""
    with patch("src.service.main.model_loader") as mock:
        mock.MODEL = MagicMock()
        mock.MODEL.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
        mock.FEATURE_NAMES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "AMT_CREDIT"]
        mock.MODEL_VERSION = "v1.0.0"
        mock.predict_proba_row = MagicMock(return_value=0.7)
        yield mock


@pytest.fixture
def mock_drift():
    """Mock the drift module to avoid needing baseline stats."""
    with patch("src.service.main.drift") as mock:
        mock.BASELINE_STATS = {"feature_1": {"mean": 100.0, "std": 10.0}}
        mock.compute_drift_metrics = MagicMock(return_value=[])
        mock.get_num_samples = MagicMock(return_value=0)
        yield mock


@pytest.fixture
def mock_db():
    """Mock the database module."""
    with patch("src.service.main.db") as mock:
        mock.init_db = MagicMock()
        mock.log_prediction = MagicMock()
        yield mock


@pytest.fixture
def client(mock_model, mock_drift, mock_db):
    """Create test client with mocked dependencies."""
    return TestClient(_app_reference)


def test_health_endpoint(client, mock_model):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert data["model_loaded"] is True
    assert data["model_version"] == "v1.0.0"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_predict_endpoint_success(client, mock_model, mock_db):
    """Test successful prediction."""
    request_data = {
        "features": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "AMT_CREDIT": 600000.0,
        }
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert "predicted_class" in data
    assert "model_version" in data
    assert "feature_values" in data

    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["predicted_class"] in [0, 1]
    assert data["model_version"] == "v1.0.0"

    # Verify logging was called
    mock_db.log_prediction.assert_called_once()


def test_predict_endpoint_with_missing_features(client, mock_model):
    """Test prediction with missing features (should use defaults)."""
    request_data = {
        "features": {
            "EXT_SOURCE_1": 0.5,
            # Other features missing
        }
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert len(data["feature_values"]) >= 1


def test_predict_endpoint_empty_features(client, mock_model):
    """Test prediction with empty features (should use all defaults)."""
    request_data = {"features": {}}

    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data


def test_predict_endpoint_invalid_feature_value(client, mock_model):
    """Test prediction with invalid feature value (expect validation error)."""
    request_data = {
        "features": {
            "EXT_SOURCE_1": "not_a_number",
        }
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code in (400, 422)


def test_predict_endpoint_none_feature_value(client, mock_model):
    """Test prediction with None feature value (should be rejected)."""
    request_data = {
        "features": {
            "EXT_SOURCE_1": None,
        }
    }

    response = client.post("/predict", json=request_data)

    assert response.status_code in (400, 422)


def test_drift_endpoint_success(client, mock_drift):
    """Test drift endpoint with default window."""
    mock_drift.compute_drift_metrics.return_value = [
        {
            "feature_name": "feature_1",
            "mean_train": 100.0,
            "mean_live": 110.0,
            "z_score": 1.0,
        }
    ]
    mock_drift.get_num_samples.return_value = 10

    response = client.get("/drift")

    assert response.status_code == 200
    data = response.json()

    assert "window_hours" in data
    assert "metrics" in data
    assert "num_samples" in data

    assert data["window_hours"] == 24
    assert data["num_samples"] == 10
    assert len(data["metrics"]) == 1

    metric = data["metrics"][0]
    assert metric["feature_name"] == "feature_1"
    assert metric["z_score"] == 1.0


def test_drift_endpoint_custom_window(client, mock_drift):
    """Test drift endpoint with custom window."""
    mock_drift.compute_drift_metrics.return_value = []
    mock_drift.get_num_samples.return_value = 0

    response = client.get("/drift?window_hours=48")

    assert response.status_code == 200
    data = response.json()

    assert data["window_hours"] == 48

    # Verify the function was called with correct window
    mock_drift.compute_drift_metrics.assert_called_with(window_hours=48)


def test_drift_endpoint_empty_metrics(client, mock_drift):
    """Test drift endpoint with no data."""
    mock_drift.compute_drift_metrics.return_value = []
    mock_drift.get_num_samples.return_value = 0

    response = client.get("/drift")

    assert response.status_code == 200
    data = response.json()

    assert data["metrics"] == []
    assert data["num_samples"] == 0


def test_drift_endpoint_invalid_window_too_small(client):
    """Test drift endpoint with invalid window (too small)."""
    response = client.get("/drift?window_hours=0")

    assert response.status_code == 400


def test_drift_endpoint_invalid_window_too_large(client):
    """Test drift endpoint with invalid window (too large)."""
    response = client.get("/drift?window_hours=200")

    assert response.status_code == 400
