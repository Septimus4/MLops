"""Generate reference statistics and defaults for inference and monitoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_prep.data_prep import DataPreprocessor


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inference artifacts from the training data"
    )
    parser.add_argument(
        "--data-dir",
        default="home-credit-default-risk-DATA",
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--sample-size", type=float, default=0.2, help="Fraction of the dataset to use"
    )
    parser.add_argument(
        "--artifacts-dir", default="artifacts", help="Directory for inference artifacts"
    )
    parser.add_argument(
        "--reference-dir", default="data/reference", help="Directory for reference data"
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    reference_dir = Path(args.reference_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = DataPreprocessor(data_dir=args.data_dir)
    preprocessor.load_data()
    X, y, feature_names = preprocessor.prepare_main_dataset(
        encoding_method="label", sample_size=args.sample_size
    )

    feature_defaults = {}
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            feature_defaults[col] = (
                float(series.median()) if not series.isna().all() else 0.0
            )
        else:
            feature_defaults[col] = series.mode(dropna=True).iloc[0]

    categorical_mappings = {}
    for col, encoder in preprocessor.encoders.items():
        if hasattr(encoder, "classes_"):
            categorical_mappings[col] = {
                str(cat): int(idx) for idx, cat in enumerate(encoder.classes_)
            }

    _write_json(artifacts_dir / "feature_defaults.json", feature_defaults)
    _write_json(artifacts_dir / "categorical_mappings.json", categorical_mappings)
    _write_json(artifacts_dir / "feature_list.json", feature_names)

    sample = X.head(500).copy()
    if y is not None:
        sample["TARGET"] = y.head(500).values
    sample.to_parquet(reference_dir / "reference_sample.parquet", index=False)

    stats = X.describe().transpose()
    stats["missing_fraction"] = X.isna().mean()
    stats.reset_index().rename(columns={"index": "feature"}).to_parquet(
        reference_dir / "feature_stats.parquet", index=False
    )

    print(f"Artifacts created in {artifacts_dir.resolve()}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
