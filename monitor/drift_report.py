"""Generate Evidently data drift reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from api.config import get_settings


def _import_optional(candidates: Iterable[tuple[str, str]]):
    for module, attr in candidates:
        try:
            return getattr(import_module(module), attr)
        except (ImportError, AttributeError):
            continue
    return None


DataDriftPreset = _import_optional(
    (
        ("evidently.metric_preset", "DataDriftPreset"),
        ("evidently.presets", "DataDriftPreset"),
    )
)  # pragma: no cover - optional dependency
Report = _import_optional(
    (
        ("evidently.report", "Report"),
        ("evidently.core.report", "Report"),
    )
)  # pragma: no cover - optional dependency

AnyReport = object


def make_drift_report(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    features: Sequence[str] | None = None,
    destination: str | Path | None = None,
    min_rows: int = 25,
    min_feature_coverage: float = 0.5,
) -> Path:
    if DataDriftPreset is None or Report is None:
        raise RuntimeError("Evidently must be installed to compute drift reports")
    settings = get_settings()
    if current.empty or reference.empty:
        raise ValueError(
            "Both current and reference data must be provided for drift analysis"
        )

    min_rows = max(int(min_rows), 1)
    if len(current) < min_rows or len(reference) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} rows in both current and reference data to compute drift"
        )

    if features is None:
        features = [
            column
            for column in settings.monitor_features
            if column in current.columns and column in reference.columns
        ]
    else:
        features = [
            column
            for column in features
            if column in current.columns and column in reference.columns
        ]

    min_non_null = max(int(min_rows * min_feature_coverage), 1)
    sufficient_features: List[str] = []
    dropped_features: List[str] = []
    for column in features:
        current_non_null = current[column].notna().sum()
        reference_non_null = reference[column].notna().sum()
        if current_non_null < min_non_null or reference_non_null < min_non_null:
            dropped_features.append(column)
            continue
        sufficient_features.append(column)

    features = sufficient_features
    if not features:
        raise ValueError(
            "No overlapping features available to compute drift with sufficient coverage"
        )

    report = Report(metrics=[DataDriftPreset(columns=features)])
    run_result = report.run(
        reference_data=reference[features], current_data=current[features]
    )
    report_instance = run_result if run_result is not None else report

    output_dir = Path(destination or (Path(settings.metrics_dir) / "drift"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_drift_report.html"
    _save_html(report_instance, output_path)
    summary_path = output_dir / "latest_drift_report.json"
    drift_summary = _drift_summary(report_instance)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "drifted_columns": drift_summary["drifted_columns"],
        "metrics": drift_summary["metrics"],
        "dropped_features": dropped_features,
        "current_row_count": int(len(current)),
        "reference_row_count": int(len(reference)),
        "min_feature_non_null": min_non_null,
    }
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


def _save_html(report: AnyReport, destination: Path) -> None:
    save_html = getattr(report, "save_html", None)
    if callable(save_html):
        save_html(str(destination))
        return
    raise RuntimeError(
        "Installed Evidently version does not support saving reports to HTML"
    )


def _drift_summary(report: AnyReport) -> dict:
    """Extract a lightweight summary from an Evidently report."""

    try:
        report_dict = report.as_dict()
    except AttributeError:  # pragma: no cover - defensive guard
        report_dict = report.dict() if hasattr(report, "dict") else None
    if not isinstance(report_dict, dict):
        return {"drifted_columns": 0, "metrics": {}}

    metrics = report_dict.get("metrics", [])
    drifted_columns = _count_drifted_columns(metrics)

    return {"drifted_columns": drifted_columns, "metrics": metrics}


def _count_drifted_columns(metrics: Sequence[dict]) -> int:
    """Calculate the number of drifted columns from Evidently metrics output."""

    count_from_total = None
    count_fallback = 0

    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        result = metric.get("result")
        if isinstance(result, dict):
            column_metrics = result.get("column_metrics") or {}
            if isinstance(column_metrics, dict):
                for column_result in column_metrics.values():
                    if isinstance(column_result, dict) and column_result.get(
                        "drift_detected"
                    ):
                        count_fallback += 1
            continue

        metric_id = metric.get("metric_id")
        value = metric.get("value")
        if (
            isinstance(metric_id, str)
            and metric_id.startswith("DriftedColumnsCount")
            and isinstance(value, dict)
        ):
            maybe_count = value.get("count")
            if isinstance(maybe_count, (int, float)):
                count_from_total = int(maybe_count)

    if count_from_total is not None:
        return count_from_total
    return count_fallback
