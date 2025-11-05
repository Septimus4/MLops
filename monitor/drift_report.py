"""Generate Evidently data drift reports."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Sequence

import pandas as pd

try:  # pragma: no cover - optional dependency
    DataDriftPreset = getattr(import_module("evidently.metric_preset"), "DataDriftPreset")  # type: ignore[attr-defined]
    Report = getattr(import_module("evidently.report"), "Report")  # type: ignore[attr-defined]
except (ImportError, AttributeError) as exc:  # pragma: no cover - fail fast when missing
    raise RuntimeError("Evidently must be installed to compute drift reports") from exc

from api.config import get_settings


def make_drift_report(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    features: Sequence[str] | None = None,
    destination: str | Path | None = None,
) -> Path:
    settings = get_settings()
    if current.empty or reference.empty:
        raise ValueError("Both current and reference data must be provided for drift analysis")

    if features is None:
        features = [column for column in settings.monitor_features if column in current.columns and column in reference.columns]
    else:
        features = [column for column in features if column in current.columns and column in reference.columns]

    if not features:
        raise ValueError("No overlapping features available to compute drift")

    report = Report(metrics=[DataDriftPreset(columns=features)])
    report.run(reference_data=reference[features], current_data=current[features])

    output_dir = Path(destination or (Path(settings.metrics_dir) / "drift"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_drift_report.html"
    report.save_html(str(output_path))
    return output_path
