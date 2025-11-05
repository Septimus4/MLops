"""Metric aggregation helpers used by the monitoring dashboard."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def summarise_kpis(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "request_count": 0,
            "error_count": 0,
            "approval_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "mean_score": 0.0,
        }

    request_count = len(df)
    error_count = int((df["status"] != "ok").sum()) if "status" in df else 0
    approvals = int((df["binary_decision"] == 1).sum()) if "binary_decision" in df else 0
    approval_rate = approvals / request_count if request_count else 0.0
    latencies = df["latency_ms"].dropna() if "latency_ms" in df else pd.Series([], dtype=float)
    scores = df["score"].dropna() if "score" in df else pd.Series([], dtype=float)

    avg_latency = float(latencies.mean()) if not latencies.empty else 0.0
    p95_latency = float(latencies.quantile(0.95)) if not latencies.empty else 0.0
    mean_score = float(scores.mean()) if not scores.empty else 0.0

    return {
        "request_count": request_count,
        "error_count": error_count,
        "approval_rate": approval_rate,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "mean_score": mean_score,
    }


def latency_series(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    if df.empty or "timestamp" not in df:
        return pd.DataFrame(columns=["timestamp", "latency_ms"])
    resampled = (
        df.set_index("timestamp")["latency_ms"].dropna().resample(freq).mean().reset_index()
    )
    resampled.rename(columns={"latency_ms": "avg_latency_ms"}, inplace=True)
    return resampled


def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "score" not in df:
        return pd.DataFrame(columns=["bucket", "count"])
    buckets = pd.cut(df["score"], bins=10, labels=False, include_lowest=True)
    histogram = buckets.value_counts().sort_index().reset_index()
    histogram.columns = ["bucket", "count"]
    histogram["bucket"] = histogram["bucket"].astype(int)
    return histogram
