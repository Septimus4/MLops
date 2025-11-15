"""
Model loader module.
Loads the trained model artifact and exposes prediction functionality.
"""

import os
from pathlib import Path

import joblib
import numpy as np

# Path to model artifact
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent.parent.parent / "data" / "artifacts" / "home_credit_model.joblib"),
)

# Global variables for model artifact
MODEL_ARTIFACT = None
MODEL = None
FEATURE_NAMES = None
MODEL_VERSION = "unknown"


def load_model():
    """Load model artifact from disk."""
    global MODEL_ARTIFACT, MODEL, FEATURE_NAMES, MODEL_VERSION

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please train the model first using: python -m src.training.train_model"
        )

    print(f"Loading model from {MODEL_PATH}...")
    MODEL_ARTIFACT = joblib.load(MODEL_PATH)
    MODEL = MODEL_ARTIFACT["model"]
    FEATURE_NAMES = MODEL_ARTIFACT["feature_names"]
    MODEL_VERSION = MODEL_ARTIFACT.get("model_version", "unknown")

    print(f"Model loaded successfully (version: {MODEL_VERSION})")
    print(f"Features: {len(FEATURE_NAMES)}")

    return MODEL_ARTIFACT


def predict_proba_row(vector: np.ndarray) -> float:
    """
    Predict probability for a single row.

    Args:
        vector: Feature vector as numpy array

    Returns:
        Probability of positive class (default risk)
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Reshape if needed
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    # Get probability of positive class (index 1)
    proba = MODEL.predict_proba(vector)
    return float(proba[0, 1])


# Load model on module import
try:
    load_model()
except Exception as e:
    print(f"Warning: Could not load model on import: {e}")
    print("Model will need to be loaded explicitly before use.")
