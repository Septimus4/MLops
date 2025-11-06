"""Utilities for loading logs and reference data for monitoring."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from api.config import get_settings
from api.artifacts import get_feature_defaults


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_logs_df(log_dir: str | Path | None = None, max_days: int = 7) -> pd.DataFrame:
    settings = get_settings()
    base = Path(log_dir or settings.log_dir)
    if not base.exists():
        return pd.DataFrame()

    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=max_days)

    records: List[Dict[str, Any]] = []
    for log_file in sorted(base.glob("*.jsonl"), reverse=True):
        try:
            file_date = datetime.strptime(log_file.stem, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if file_date < cutoff:
            continue
        for record in _load_jsonl(log_file):
            timestamp_raw = record.get("timestamp")
            if isinstance(timestamp_raw, str):
                try:
                    record_time = datetime.fromisoformat(timestamp_raw)
                    if record_time.tzinfo is None:
                        record_time = record_time.replace(tzinfo=timezone.utc)
                except ValueError:
                    record_time = now
            else:
                record_time = now

            if record_time < cutoff:
                continue

            flattened = {
                "timestamp": record_time,
                "request_id": record.get("request_id"),
                "status": record.get("status"),
                "latency_ms": record.get("latency_ms"),
                "score": record.get("score"),
                "binary_decision": record.get("binary_decision"),
                "model_version": record.get("model_version"),
            }

            monitor_features = record.get("monitor_features")
            if isinstance(monitor_features, dict):
                flattened.update({k: monitor_features.get(k) for k in monitor_features})

            records.append(flattened)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df.sort_values("timestamp", inplace=True)
    df = df.copy()

    defaults = get_feature_defaults()
    fill_values = {
        column: float(defaults[column]) for column in defaults if column in df.columns
    }

    for column, value in fill_values.items():
        if column not in df.columns:
            continue
        series = df[column]
        if series.isna().all():
            df[column] = float(value)
        else:
            df[column] = series.fillna(value)

    return df


def load_reference_sample(
    path: str | Path | None = None, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    settings = get_settings()
    reference_path = Path(
        path or (Path(settings.reference_dir) / "reference_sample.parquet")
    )
    if not reference_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(reference_path)
    if columns:
        existing = [column for column in columns if column in df.columns]
        if existing:
            df = df[existing].copy()
        else:
            df = df.copy()
    else:
        df = df.copy()

    defaults = get_feature_defaults()
    fill_values = {
        column: float(defaults[column]) for column in defaults if column in df.columns
    }
    for column, value in fill_values.items():
        if column not in df.columns:
            continue
        series = df[column]
        if series.isna().all():
            df[column] = float(value)
        else:
            df[column] = series.fillna(value)

    return df


def load_reference_stats(path: str | Path | None = None) -> pd.DataFrame:
    settings = get_settings()
    stats_path = Path(path or (Path(settings.reference_dir) / "feature_stats.parquet"))
    if not stats_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(stats_path)
