"""Additional tests to raise coverage for error and edge paths."""

import importlib
import json
import pytest
from fastapi.testclient import TestClient

from src.service import model_loader, db, drift
from src.service.main import app
from src.service.feature_processor import validate_features


@pytest.fixture
def api_client():
    return TestClient(app)


def test_predict_model_not_loaded(monkeypatch, api_client):
    """Model None should yield 503."""
    original_model = model_loader.MODEL
    monkeypatch.setattr(model_loader, "MODEL", None)
    resp = api_client.post("/predict", json={"features": {"EXT_SOURCE_1": 0.5}})
    assert resp.status_code == 503
    # Restore
    monkeypatch.setattr(model_loader, "MODEL", original_model)


@pytest.mark.parametrize("window", [0, -1])
def test_drift_invalid_window_low(api_client, window):
    resp = api_client.get(f"/drift?window_hours={window}")
    assert resp.status_code == 400


def test_drift_invalid_window_high(api_client):
    resp = api_client.get("/drift?window_hours=169")
    assert resp.status_code == 400


def test_validate_features_unknown_key():
    with pytest.raises(ValueError):
        validate_features({"UNKNOWN_KEY": 1.0})


def test_model_loader_predict_runtime_error(monkeypatch):
    # Ensure model is None triggers RuntimeError in predict_proba_row
    original = model_loader.MODEL
    monkeypatch.setattr(model_loader, "MODEL", None)
    import numpy as np

    with pytest.raises(RuntimeError):
        model_loader.predict_proba_row(np.array([0.0, 0.0]))
    monkeypatch.setattr(model_loader, "MODEL", original)


def test_db_log_and_fetch(monkeypatch, tmp_path):
    test_db = tmp_path / "predictions.db"
    monkeypatch.setenv("DB_PATH", str(test_db))
    importlib.reload(db)
    db.log_prediction("vX", {"A": 1.0}, 0.2, 0)
    assert db.get_prediction_count() == 1
    fetched = db.fetch_predictions_since(1)
    assert fetched is not None
    assert len(fetched) == 1


def test_drift_compute_empty(monkeypatch, tmp_path):
    # Use empty DB; expect no metrics
    test_db = tmp_path / "predictions.db"
    monkeypatch.setenv("DB_PATH", str(test_db))
    importlib.reload(db)
    monkeypatch.setenv("BASELINE_PATH", str(tmp_path / "baseline.json"))
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps({"EXT_SOURCE_1": {"mean": 0.5, "std": 0.1}}))
    importlib.reload(drift)
    metrics = drift.compute_drift_metrics(window_hours=1)
    assert metrics == []
    assert drift.get_num_samples(1) == 0


def test_drift_compute_with_sample(monkeypatch, tmp_path):
    # Insert a prediction with feature EXT_SOURCE_1 and compute drift
    test_db = tmp_path / "predictions.db"
    monkeypatch.setenv("DB_PATH", str(test_db))
    importlib.reload(db)
    monkeypatch.setenv("BASELINE_PATH", str(tmp_path / "baseline.json"))
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps({"EXT_SOURCE_1": {"mean": 0.5, "std": 0.1}}))
    importlib.reload(drift)
    db.log_prediction("vX", {"EXT_SOURCE_1": 0.7}, 0.3, 0)
    metrics = drift.compute_drift_metrics(window_hours=1)
    assert metrics
    assert metrics[0]["feature_name"] == "EXT_SOURCE_1"
    assert drift.get_num_samples(1) == 1
