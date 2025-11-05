from __future__ import annotations

import pandas as pd

from monitor.metrics import latency_series, score_distribution, summarise_kpis


def test_kpi_summary_handles_data() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "status": ["ok", "error", "ok"],
            "latency_ms": [100, 200, 150],
            "score": [0.2, 0.8, 0.5],
            "binary_decision": [0, 1, 0],
        }
    )
    summary = summarise_kpis(df)
    assert summary["request_count"] == 3
    assert summary["error_count"] == 1
    assert 0 <= summary["approval_rate"] <= 1


def test_latency_series_generates_rows() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="min"),
            "latency_ms": [100] * 10,
        }
    )
    series = latency_series(df, freq="5min")
    assert not series.empty


def test_score_distribution_has_buckets() -> None:
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    distribution = score_distribution(df)
    assert distribution["count"].sum() == len(df)
