"""
Tests for drift detection module.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from src.service.drift import compute_drift_metrics


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    # Create schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            features_json TEXT NOT NULL,
            risk_score REAL NOT NULL,
            predicted_class INTEGER NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def baseline_stats():
    """Create baseline statistics for testing."""
    return {
        "feature_1": {"mean": 100.0, "std": 10.0},
        "feature_2": {"mean": 50.0, "std": 5.0},
        "feature_3": {"mean": 200.0, "std": 20.0},
    }


def test_compute_drift_metrics_empty_db(temp_db, baseline_stats):
    """Test drift computation with empty database."""
    # Override DB path
    import src.service.db as db_module

    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_db

    try:
        metrics = compute_drift_metrics(window_hours=24, baseline=baseline_stats)

        # Should return empty list
        assert metrics == []
    finally:
        db_module.DB_PATH = original_path


def test_compute_drift_metrics_with_data(temp_db, baseline_stats):
    """Test drift computation with some data."""
    # Insert test data
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert predictions with features matching baseline
    now = datetime.now(timezone.utc)

    for i in range(5):
        timestamp = (now - timedelta(hours=1)).isoformat()
        features = {
            "feature_1": 110.0 + i,  # Slightly higher than baseline mean
            "feature_2": 50.0,  # Same as baseline
            "feature_3": 220.0,  # Higher than baseline
        }

        cursor.execute(
            """
            INSERT INTO predictions (timestamp, model_version, features_json, risk_score, predicted_class)
            VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, "v1.0.0", json.dumps(features), 0.5, 0),
        )

    conn.commit()
    conn.close()

    # Override DB path
    import src.service.db as db_module

    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_db

    try:
        metrics = compute_drift_metrics(window_hours=24, baseline=baseline_stats)

        # Should have metrics for all features
        assert len(metrics) == 3

        # Check that metrics are sorted by z_score descending
        z_scores = [m["z_score"] for m in metrics]
        assert z_scores == sorted(z_scores, reverse=True)

        # Feature 3 should have highest drift (mean 220 vs 200, std 20)
        # z_score = |220 - 200| / 20 = 1.0
        feature_3_metric = next(m for m in metrics if m["feature_name"] == "feature_3")
        assert feature_3_metric["mean_train"] == 200.0
        assert feature_3_metric["mean_live"] == 220.0
        assert feature_3_metric["z_score"] == pytest.approx(1.0, abs=0.01)

        # Feature 2 should have no drift
        feature_2_metric = next(m for m in metrics if m["feature_name"] == "feature_2")
        assert feature_2_metric["z_score"] == pytest.approx(0.0, abs=0.01)

    finally:
        db_module.DB_PATH = original_path


def test_compute_drift_metrics_z_score_calculation(temp_db, baseline_stats):
    """Test that z-score is calculated correctly."""
    # Insert single prediction
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc)
    timestamp = (now - timedelta(minutes=30)).isoformat()

    # Live mean = 120, baseline mean = 100, baseline std = 10
    # Expected z-score = |120 - 100| / 10 = 2.0
    features = {
        "feature_1": 120.0,
        "feature_2": 50.0,
        "feature_3": 200.0,
    }

    cursor.execute(
        """
        INSERT INTO predictions (timestamp, model_version, features_json, risk_score, predicted_class)
        VALUES (?, ?, ?, ?, ?)
    """,
        (timestamp, "v1.0.0", json.dumps(features), 0.5, 0),
    )

    conn.commit()
    conn.close()

    # Override DB path
    import src.service.db as db_module

    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_db

    try:
        metrics = compute_drift_metrics(window_hours=24, baseline=baseline_stats)

        feature_1_metric = next(m for m in metrics if m["feature_name"] == "feature_1")

        assert feature_1_metric["mean_live"] == 120.0
        assert feature_1_metric["mean_train"] == 100.0
        assert feature_1_metric["z_score"] == pytest.approx(2.0, abs=0.01)

    finally:
        db_module.DB_PATH = original_path


def test_compute_drift_metrics_window_filtering(temp_db, baseline_stats):
    """Test that time window filtering works correctly."""
    # Insert predictions at different times
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc)

    # Old prediction (outside window)
    old_timestamp = (now - timedelta(hours=25)).isoformat()
    old_features = {"feature_1": 90.0, "feature_2": 40.0, "feature_3": 180.0}
    cursor.execute(
        """
        INSERT INTO predictions (timestamp, model_version, features_json, risk_score, predicted_class)
        VALUES (?, ?, ?, ?, ?)
    """,
        (old_timestamp, "v1.0.0", json.dumps(old_features), 0.5, 0),
    )

    # Recent prediction (inside window)
    recent_timestamp = (now - timedelta(hours=1)).isoformat()
    recent_features = {"feature_1": 110.0, "feature_2": 55.0, "feature_3": 220.0}
    cursor.execute(
        """
        INSERT INTO predictions (timestamp, model_version, features_json, risk_score, predicted_class)
        VALUES (?, ?, ?, ?, ?)
    """,
        (recent_timestamp, "v1.0.0", json.dumps(recent_features), 0.5, 0),
    )

    conn.commit()
    conn.close()

    # Override DB path
    import src.service.db as db_module

    original_path = db_module.DB_PATH
    db_module.DB_PATH = temp_db

    try:
        metrics = compute_drift_metrics(window_hours=24, baseline=baseline_stats)

        # Should only use recent prediction
        feature_1_metric = next(m for m in metrics if m["feature_name"] == "feature_1")
        assert feature_1_metric["mean_live"] == 110.0  # Not 100.0 (average of both)

    finally:
        db_module.DB_PATH = original_path
