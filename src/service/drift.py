"""
Drift detection module.
Computes feature drift using z-scores compared to baseline statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .db import fetch_predictions_since

# Path to baseline statistics
BASELINE_PATH = os.environ.get(
    "BASELINE_PATH",
    str(Path(__file__).parent.parent.parent / "data" / "artifacts" / "baseline_stats.json"),
)

# Global baseline stats
BASELINE_STATS = None


def load_baseline_stats() -> Dict[str, Dict[str, float]]:
    """Load baseline statistics from JSON file."""
    global BASELINE_STATS

    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(
            f"Baseline stats file not found at {BASELINE_PATH}. "
            "Please compute baseline statistics first using: python -m src.training.compute_baseline_stats"
        )

    with open(BASELINE_PATH, "r") as f:
        BASELINE_STATS = json.load(f)

    print(f"Loaded baseline statistics for {len(BASELINE_STATS)} features")
    return BASELINE_STATS


def compute_drift_metrics(
    window_hours: int = 24, baseline: Dict[str, Dict[str, float]] = None
) -> List[Dict]:
    """
    Compute drift metrics for features in the given time window.

    Args:
        window_hours: Time window in hours for computing live statistics
        baseline: Baseline statistics dict (if None, uses loaded baseline)

    Returns:
        List of drift metric dictionaries
    """
    if baseline is None:
        if BASELINE_STATS is None:
            load_baseline_stats()
        baseline = BASELINE_STATS

    # Fetch predictions from the window
    df = fetch_predictions_since(window_hours)

    if df is None or df.empty:
        return []

    # Extract features from JSON
    features_list = []
    for features_json in df["features_json"]:
        features_dict = json.loads(features_json)
        features_list.append(features_dict)

    features_df = pd.DataFrame(features_list)

    # Compute drift metrics for each feature
    metrics = []

    for feature_name, baseline_info in baseline.items():
        if feature_name not in features_df.columns:
            continue

        # Get live statistics
        live_values = features_df[feature_name].dropna()

        if len(live_values) == 0:
            continue

        live_mean = float(live_values.mean())
        mean_train = baseline_info["mean"]
        std_train = baseline_info["std"]

        # Compute z-score (avoid division by zero)
        if std_train > 0:
            z_score = abs(live_mean - mean_train) / std_train
        else:
            z_score = 0.0

        metrics.append(
            {
                "feature_name": feature_name,
                "mean_train": mean_train,
                "mean_live": live_mean,
                "z_score": z_score,
            }
        )

    # Sort by z_score descending
    metrics.sort(key=lambda x: x["z_score"], reverse=True)

    return metrics


def get_num_samples(window_hours: int) -> int:
    """
    Get number of prediction samples in the given time window.

    Args:
        window_hours: Time window in hours

    Returns:
        Number of samples
    """
    df = fetch_predictions_since(window_hours)

    if df is None or df.empty:
        return 0

    return len(df)


# Load baseline stats on module import
try:
    load_baseline_stats()
except Exception as e:
    print(f"Warning: Could not load baseline stats on import: {e}")
    print("Baseline stats will need to be loaded explicitly before computing drift.")
