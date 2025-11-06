"""Structured logging utilities for the scoring API."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

from .config import get_settings
from .model_loader import ModelMetadata
from .schemas import InputPayload


def _log_path(base: Path, timestamp: datetime) -> Path:
    return base / f"{timestamp:%Y-%m-%d}.jsonl"


def log_prediction(
    payload: InputPayload,
    response_body: Dict[str, Any],
    processed_features: Dict[str, float],
    monitor_features: Dict[str, float],
    status: str,
    latency_ms: float,
    metadata: ModelMetadata,
) -> None:
    """Persist a structured record of an inference request."""

    settings = get_settings()
    timestamp = datetime.now(tz=timezone.utc)
    base_dir = Path(settings.log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": timestamp.isoformat(),
        "request_id": payload.request_id,
        "applicant_id": payload.applicant_id,
        "status": status,
        "latency_ms": float(latency_ms),
        "model_name": metadata.model_name,
        "model_version": metadata.model_version,
        "score": response_body.get("score"),
        "binary_decision": response_body.get("binary_decision"),
        "threshold": response_body.get("threshold"),
        "input_features": payload.features,
        "processed_features": processed_features,
        "monitor_features": monitor_features,
        "error": response_body.get("error"),
    }

    log_file = _log_path(base_dir, timestamp)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _read_records(paths: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # pragma: no cover - defensive guard


def _list_candidate_files(log_dir: Path, since: datetime) -> List[Path]:
    candidates = []
    for path in sorted(log_dir.glob("*.jsonl")):
        try:
            date = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if date >= since.replace(hour=0, minute=0, second=0, microsecond=0):
            candidates.append(path)
    return candidates


def _latest_drift_alerts(metrics_dir: Path) -> int:
    summary_path = metrics_dir / "drift" / "latest_drift_report.json"
    if not summary_path.exists():
        return 0
    try:
        payload = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):  # pragma: no cover - defensive guard
        return 0
    alerts = payload.get("drifted_columns")
    if isinstance(alerts, int) and alerts >= 0:
        return alerts
    return 0


def aggregate_metrics(window_minutes: int) -> Dict[str, Any]:
    """Aggregate simple metrics over the specified lookback window."""

    settings = get_settings()
    log_dir = Path(settings.log_dir)
    metrics_dir = Path(settings.metrics_dir)
    if not log_dir.exists():
        return {
            "window_minutes": window_minutes,
            "request_count": 0,
            "error_count": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "mean_score": 0.0,
            "drift_alerts": _latest_drift_alerts(metrics_dir),
        }

    now = datetime.now(tz=timezone.utc)
    since = now - timedelta(minutes=window_minutes)
    files = _list_candidate_files(log_dir, since)

    latencies: List[float] = []
    scores: List[float] = []
    error_count = 0
    total = 0

    for record in _read_records(files):
        try:
            raw_timestamp = record.get("timestamp")
            if not isinstance(raw_timestamp, str):
                continue
            ts = datetime.fromisoformat(raw_timestamp)
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < since:
            continue
        total += 1
        if record.get("status") != "ok":
            error_count += 1
        latency = record.get("latency_ms")
        if latency is not None:
            latencies.append(float(latency))
        score = record.get("score")
        if score is not None:
            scores.append(float(score))

    if not total:
        avg_latency = 0.0
        p95_latency = 0.0
        mean_score = 0.0
    else:
        avg_latency = float(mean(latencies)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
        mean_score = float(mean(scores)) if scores else 0.0

    return {
        "window_minutes": window_minutes,
        "request_count": total,
        "error_count": error_count,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "mean_score": mean_score,
        "drift_alerts": _latest_drift_alerts(metrics_dir),
    }
