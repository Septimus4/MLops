"""
Database module for logging predictions and drift metrics.
Uses SQLite for simplicity.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Database path
DB_PATH = os.environ.get(
    "DB_PATH", str(Path(__file__).parent.parent.parent / "data" / "artifacts" / "predictions.db")
)


def get_connection():
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Initialize database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create predictions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_version TEXT NOT NULL,
            features_json TEXT NOT NULL,
            risk_score REAL NOT NULL,
            predicted_class INTEGER NOT NULL
        )
    """
    )

    # Create index on timestamp for efficient queries
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON predictions(timestamp)
    """
    )

    # Create drift_metrics table (optional, for storing computed metrics)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            mean_train REAL NOT NULL,
            mean_live REAL NOT NULL,
            z_score REAL NOT NULL,
            window_hours INTEGER NOT NULL
        )
    """
    )

    conn.commit()
    conn.close()

    print(f"Database initialized at {DB_PATH}")


def log_prediction(
    model_version: str, features_dict: Dict[str, float], risk_score: float, predicted_class: int
):
    """
    Log a prediction to the database.

    Args:
        model_version: Version of the model used
        features_dict: Dictionary of feature values
        risk_score: Predicted risk score (probability)
        predicted_class: Predicted class (0 or 1)
    """
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now(timezone.utc).isoformat()
    features_json = json.dumps(features_dict)

    cursor.execute(
        """
        INSERT INTO predictions (timestamp, model_version, features_json, risk_score, predicted_class)
        VALUES (?, ?, ?, ?, ?)
    """,
        (timestamp, model_version, features_json, risk_score, predicted_class),
    )

    conn.commit()
    conn.close()


def fetch_predictions_since(hours: int) -> Optional[pd.DataFrame]:
    """
    Fetch predictions from the last N hours.

    Args:
        hours: Number of hours to look back

    Returns:
        DataFrame with predictions or None if no data
    """
    conn = get_connection()

    # Calculate cutoff time
    cutoff = datetime.now(timezone.utc).replace(microsecond=0)
    # Subtract hours (simple approach)
    from datetime import timedelta

    cutoff = cutoff - timedelta(hours=hours)
    cutoff_str = cutoff.isoformat()

    query = """
        SELECT timestamp, features_json, risk_score, predicted_class
        FROM predictions
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """

    try:
        df = pd.read_sql_query(query, conn, params=(cutoff_str,))
        conn.close()

        if df.empty:
            return None

        return df
    except Exception as e:
        conn.close()
        print(f"Error fetching predictions: {e}")
        return None


def get_prediction_count() -> int:
    """Get total number of predictions logged."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]

    conn.close()
    return count


# Initialize database on module import
try:
    init_db()
except Exception as e:
    print(f"Warning: Could not initialize database: {e}")
