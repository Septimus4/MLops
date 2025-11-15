"""
Training script for Home Credit Default Risk model.
Trains a LightGBM classifier and saves the model artifact.
"""

import os
import sys

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .feature_config import ID_COLUMN, TARGET_COLUMN


def main(
    data_path: str = "data/raw/application_train.csv",
    output_path: str = "data/artifacts/home_credit_model.joblib",
):
    """
    Train LightGBM model on Home Credit data.

    Args:
        data_path: Path to training CSV file
        output_path: Path to save model artifact
    """
    print(f"Loading data from {data_path}...")

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print(
            "Please download the Home Credit dataset from Kaggle and place application_train.csv in data/raw/"
        )
        sys.exit(1)

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Extract target and features
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in data")
        sys.exit(1)

    y = df[TARGET_COLUMN]
    X = df.drop([TARGET_COLUMN, ID_COLUMN], axis=1)

    # Keep only numeric columns for simplicity
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X = X[numeric_cols]

    print(f"Using {len(X.columns)} numeric features")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Fill missing values with median
    X = X.fillna(X.median())

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Train LightGBM model
    print("\nTraining LightGBM model...")
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)

    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Save model artifact
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": X.columns.tolist(),
        "model_version": "v1.0.0",
    }

    joblib.dump(artifact, output_path)
    print(f"\nModel saved to {output_path}")
    print(f"Features: {len(artifact['feature_names'])}")

    return artifact


if __name__ == "__main__":
    main()
