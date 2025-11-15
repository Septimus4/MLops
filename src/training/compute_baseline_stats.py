"""
Compute baseline statistics for drift detection.
Calculates mean and standard deviation for each feature in the training data.
"""

import json
import os
import sys

import pandas as pd

from .feature_config import ID_COLUMN, TARGET_COLUMN


def compute_baseline_statistics(
    data_path: str = "data/raw/application_train.csv",
    model_path: str = "data/artifacts/home_credit_model.joblib",
    output_path: str = "data/artifacts/baseline_stats.json",
):
    """
    Compute baseline statistics from training data.

    Args:
        data_path: Path to training CSV file
        model_path: Path to trained model (to get feature names)
        output_path: Path to save baseline statistics JSON
    """
    print(f"Loading data from {data_path}...")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Load model to get feature names
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train_model.py")
        sys.exit(1)

    import joblib

    artifact = joblib.load(model_path)
    feature_names = artifact["feature_names"]

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    # Extract features (same processing as training)
    X = df.drop([TARGET_COLUMN, ID_COLUMN], axis=1, errors="ignore")

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X = X[numeric_cols]

    # Fill missing values with median
    X = X.fillna(X.median())

    # Compute statistics for each feature
    baseline_stats = {}

    for feature in feature_names:
        if feature in X.columns:
            values = X[feature]
            stats = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
            }
            baseline_stats[feature] = stats
            print(f"{feature}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        else:
            print(f"Warning: Feature {feature} not found in data")

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"\nBaseline statistics saved to {output_path}")
    print(f"Total features: {len(baseline_stats)}")

    return baseline_stats


def main():
    """Main entry point."""
    compute_baseline_statistics()


if __name__ == "__main__":
    main()
